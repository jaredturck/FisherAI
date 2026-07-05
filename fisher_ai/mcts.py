import numpy as np
import torch

from fisher_ai import chess
from fisher_ai.encoding import encode_state, move_to_action

MAX_LEGAL_ACTIONS = 256
UNEXPANDED = -1
C_PUCT = 1.5
VIRTUAL_LOSS = 1.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25

PROMOTION_CODES = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
CODE_PROMOTIONS = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def pack_move(move):
    promotion = PROMOTION_CODES[move.promotion]
    return move.from_square | move.to_square << 6 | promotion << 12


def unpack_move(packed_move):
    packed_move = int(packed_move)
    from_square = packed_move & 63
    to_square = packed_move >> 6 & 63
    promotion = CODE_PROMOTIONS[packed_move >> 12]
    return chess.Move(from_square, to_square, promotion)


class TorchEvaluator:
    def __init__(self, model, device="cpu", inference_batch_size=512):
        self.model = model
        self.device = torch.device(device)
        self.inference_batch_size = inference_batch_size
        self.channels_last = self.device.type == "cuda"
        self.model.to(self.device)

        if self.channels_last:
            self.model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True
        self.model.eval()

    def evaluate_encoded(self, encoded_states, legal_actions):
        policies = []
        values = []

        for start in range(0, len(encoded_states), self.inference_batch_size):
            end = min(start + self.inference_batch_size, len(encoded_states))
            array = np.stack(encoded_states[start:end]).astype(
                np.float32,
                copy=False,
            )
            states = torch.from_numpy(array)
            if self.device.type == "cuda":
                states = states.pin_memory()
            states = states.to(self.device, non_blocking=True)
            if self.channels_last:
                states = states.contiguous(memory_format=torch.channels_last)

            chunk_actions = legal_actions[start:end]
            max_actions = max(len(actions) for actions in chunk_actions)
            padded_actions = np.zeros(
                (len(chunk_actions), max_actions),
                dtype=np.int64,
            )
            for index, actions in enumerate(chunk_actions):
                padded_actions[index, : len(actions)] = actions

            actions = torch.from_numpy(padded_actions)
            if self.device.type == "cuda":
                actions = actions.pin_memory()
            actions = actions.to(self.device, non_blocking=True)

            with torch.inference_mode():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    policy, value = self.model(states)
                    gathered = policy.gather(1, actions)

            gathered = gathered.float().cpu().numpy()
            policies.extend(
                gathered[index, : len(action_list)]
                for index, action_list in enumerate(chunk_actions)
            )
            values.append(value.float().cpu().numpy())

        return policies, np.concatenate(values)


class MCTSTree:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.actions = np.empty(self.capacity, dtype=np.uint16)
        self.moves = np.empty(self.capacity, dtype=np.uint16)
        self.priors = np.empty(self.capacity, dtype=np.float32)
        self.visit_counts = np.empty(self.capacity, dtype=np.uint32)
        self.value_sums = np.empty(self.capacity, dtype=np.float32)
        self.virtual_visits = np.empty(self.capacity, dtype=np.uint16)
        self.first_children = np.empty(self.capacity, dtype=np.int32)
        self.child_counts = np.empty(self.capacity, dtype=np.uint16)
        self.root_priors = np.empty(MAX_LEGAL_ACTIONS, dtype=np.float32)
        self.reset()

    @property
    def visit_count(self):
        return int(self.visit_counts[self.root_index])

    @property
    def virtual_visit_count(self):
        return int(self.virtual_visits[self.root_index])

    @property
    def expanded(self):
        return self.first_children[self.root_index] != UNEXPANDED

    def reset(self):
        self.root_index = 0
        self.next_free = 1
        self.root_prior_count = 0
        self.clear_records(0, 1)

    def clear_records(self, start, end):
        self.visit_counts[start:end] = 0
        self.value_sums[start:end] = 0.0
        self.virtual_visits[start:end] = 0
        self.first_children[start:end] = UNEXPANDED
        self.child_counts[start:end] = 0

    def prepare(self, required_records):
        if self.next_free + required_records > self.capacity:
            self.reset()

    def child_range(self, node_id):
        start = int(self.first_children[node_id])
        if start == UNEXPANDED:
            return 0, 0
        return start, start + int(self.child_counts[node_id])

    def allocate_children(self, node_id, actions, packed_moves, priors):
        child_count = len(actions)
        start = self.next_free
        end = start + child_count
        assert end <= self.capacity

        self.clear_records(start, end)
        self.actions[start:end] = actions
        self.moves[start:end] = packed_moves
        self.priors[start:end] = priors
        self.first_children[node_id] = start
        self.child_counts[node_id] = child_count
        self.next_free = end

    def move_for_action(self, action):
        start, end = self.child_range(self.root_index)
        actions = self.actions[start:end]
        index = int(np.searchsorted(actions, action))
        assert index < len(actions) and actions[index] == action
        return unpack_move(self.moves[start + index])

    def advance(self, action):
        start, end = self.child_range(self.root_index)
        actions = self.actions[start:end]
        index = int(np.searchsorted(actions, action))
        assert index < len(actions) and actions[index] == action
        self.root_index = start + index
        self.root_prior_count = 0
        return unpack_move(self.moves[self.root_index])


class MCTS:
    def __init__(
        self, evaluator, simulations=128, parallel_searches=8, seed=7
    ):
        self.evaluator = evaluator
        self.simulations = int(simulations)
        self.parallel_searches = int(parallel_searches)
        self.rng = np.random.default_rng(seed)
        required = (self.simulations + 1) * MAX_LEGAL_ACTIONS + 1
        self.tree_capacity = 1 << (required - 1).bit_length()
        self.score_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.float32)
        self.work_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.float32)
        self.denominator_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.float32)
        self.unavailable_buffer = np.empty(MAX_LEGAL_ACTIONS, dtype=np.bool_)

    def create_tree(self):
        return MCTSTree(self.tree_capacity)

    def run(self, states, roots=None, add_noise=False):
        if roots is None:
            roots = [self.create_tree() for _ in states]

        required_records = (self.simulations + 1) * MAX_LEGAL_ACTIONS + 1
        for root in roots:
            root.prepare(required_records)

        terminal_flags = [state.is_terminal() for state in states]
        target_visits = [root.visit_count + self.simulations for root in roots]

        unexpanded_trees = []
        unexpanded_nodes = []
        unexpanded_states = []
        unexpanded_encoded = []
        for root, state, terminal in zip(
            roots,
            states,
            terminal_flags,
            strict=True,
        ):
            if not root.expanded and not terminal:
                unexpanded_trees.append(root)
                unexpanded_nodes.append(root.root_index)
                unexpanded_states.append(state)
                unexpanded_encoded.append(encode_state(state))

        if unexpanded_trees:
            self.evaluate_and_expand(
                unexpanded_trees,
                unexpanded_nodes,
                unexpanded_states,
                unexpanded_encoded,
            )

        for root in roots:
            if root.expanded:
                self.set_root_priors(root, add_noise)

        while any(
            root.visit_count < target
            for root, target, terminal in zip(
                roots,
                target_visits,
                terminal_flags,
                strict=True,
            )
            if not terminal
        ):
            pending_trees = []
            pending_nodes = []
            pending_paths = []
            pending_states = []
            pending_encoded = []
            made_progress = False

            for root, state, target, terminal in zip(
                roots,
                states,
                target_visits,
                terminal_flags,
                strict=True,
            ):
                if terminal:
                    continue

                reservations = 0
                while (
                    root.visit_count + root.virtual_visit_count < target
                    and reservations < self.parallel_searches
                ):
                    leaf_id, path = self.select_leaf(root)
                    if leaf_id is None:
                        break

                    self.reserve(root, path)
                    made_progress = True
                    reservations += 1
                    leaf_state = self.build_state(state, root, path)

                    if leaf_state.is_terminal():
                        self.release(root, path)
                        self.backup(root, path, leaf_state.terminal_value())
                    else:
                        pending_trees.append(root)
                        pending_nodes.append(leaf_id)
                        pending_paths.append(path)
                        pending_states.append(leaf_state)
                        pending_encoded.append(encode_state(leaf_state))

            if pending_trees:
                values = self.evaluate_and_expand(
                    pending_trees,
                    pending_nodes,
                    pending_states,
                    pending_encoded,
                )
                for root, path, value in zip(
                    pending_trees,
                    pending_paths,
                    values,
                    strict=True,
                ):
                    self.release(root, path)
                    self.backup(root, path, float(value))

            if not made_progress:
                break

        return roots

    def select_leaf(self, tree):
        node_id = tree.root_index
        path = [node_id]

        while (
            tree.first_children[node_id] != UNEXPANDED
            and tree.child_counts[node_id]
        ):
            node_id = self.select_child(tree, node_id)
            if node_id is None:
                return None, None
            path.append(node_id)

        return node_id, path

    def select_child(self, tree, node_id):
        start, end = tree.child_range(node_id)
        child_count = end - start
        visits = tree.visit_counts[start:end]
        values = tree.value_sums[start:end]
        virtual_visits = tree.virtual_visits[start:end]
        priors = tree.priors[start:end]
        if node_id == tree.root_index and tree.root_prior_count == child_count:
            priors = tree.root_priors[:child_count]

        scores = self.score_buffer[:child_count]
        work = self.work_buffer[:child_count]
        denominators = self.denominator_buffer[:child_count]
        unavailable = self.unavailable_buffer[:child_count]

        scores.fill(0.0)
        np.divide(values, visits, out=scores, where=visits > 0)
        scores *= -1.0

        np.add(visits, virtual_visits, out=denominators, casting="unsafe")
        denominators += 1.0
        parent_visits = max(
            1,
            tree.visit_counts[node_id] + tree.virtual_visits[node_id],
        )
        np.multiply(
            priors,
            C_PUCT * np.sqrt(parent_visits),
            out=work,
        )
        work /= denominators
        scores += work

        np.multiply(
            virtual_visits,
            VIRTUAL_LOSS,
            out=work,
            casting="unsafe",
        )
        scores -= work
        np.logical_and(virtual_visits > 0, visits == 0, out=unavailable)
        scores[unavailable] = -np.inf

        index = int(np.argmax(scores))
        if not np.isfinite(scores[index]):
            return None
        return start + index

    @staticmethod
    def build_state(root_state, tree, path):
        state = root_state.copy()
        for node_id in path[1:]:
            state.push(unpack_move(tree.moves[node_id]))
        return state

    @staticmethod
    def legal_action_data(state):
        actions = []
        packed_moves = []
        for move in state.board.legal_moves:
            actions.append(move_to_action(move, state.board.turn))
            packed_moves.append(pack_move(move))

        actions = np.asarray(actions, dtype=np.int64)
        packed_moves = np.asarray(packed_moves, dtype=np.uint16)
        order = np.argsort(actions)
        return actions[order], packed_moves[order]

    def evaluate_and_expand(self, trees, node_ids, states, encoded_states):
        action_lists = []
        packed_move_lists = []
        for state in states:
            actions, packed_moves = self.legal_action_data(state)
            action_lists.append(actions)
            packed_move_lists.append(packed_moves)

        policy_output, values = self.evaluator.evaluate_encoded(
            encoded_states,
            action_lists,
        )

        for index, (tree, node_id) in enumerate(
            zip(trees, node_ids, strict=True)
        ):
            logits = policy_output[index]
            logits = logits - logits.max()
            priors = np.exp(logits)
            priors /= priors.sum()
            tree.allocate_children(
                node_id,
                action_lists[index],
                packed_move_lists[index],
                priors.astype(np.float32, copy=False),
            )

        return values

    def set_root_priors(self, tree, add_noise):
        start, end = tree.child_range(tree.root_index)
        child_count = end - start
        tree.root_prior_count = 0

        if not add_noise or child_count == 0:
            return

        tree.root_priors[:child_count] = tree.priors[start:end]
        noise = self.rng.dirichlet(np.full(child_count, DIRICHLET_ALPHA))
        tree.root_priors[:child_count] *= 1.0 - DIRICHLET_FRACTION
        tree.root_priors[:child_count] += DIRICHLET_FRACTION * noise
        tree.root_prior_count = child_count

    @staticmethod
    def reserve(tree, path):
        for node_id in path:
            tree.virtual_visits[node_id] += 1

    @staticmethod
    def release(tree, path):
        for node_id in path:
            tree.virtual_visits[node_id] -= 1

    @staticmethod
    def backup(tree, path, value):
        for node_id in reversed(path):
            tree.visit_counts[node_id] += 1
            tree.value_sums[node_id] += value
            value = -value

    @staticmethod
    def visit_counts(root):
        start, end = root.child_range(root.root_index)
        actions = root.actions[start:end].astype(np.int64, copy=True)
        counts = root.visit_counts[start:end].astype(np.float32, copy=True)
        return actions, counts

    def choose_action(self, root, temperature=1.0, greedy=False):
        actions, counts = self.visit_counts(root)
        assert counts.sum() > 0

        if greedy or temperature <= 0:
            return int(actions[np.argmax(counts)])

        weights = counts ** (1.0 / temperature)
        probabilities = weights / weights.sum()
        return int(self.rng.choice(actions, p=probabilities))
