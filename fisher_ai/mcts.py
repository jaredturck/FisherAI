import numpy as np
import torch

from fisher_ai import chess
from fisher_ai.encoding import INPUT_PLANES, encode_state, move_to_action

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

    def evaluate_encoded(self, encoded_states, legal_actions, legal_lengths):
        batch_size = len(encoded_states)
        policies = np.zeros(
            (batch_size, legal_actions.shape[1]),
            dtype=np.float32,
        )
        values = np.empty(batch_size, dtype=np.float32)

        for start in range(0, batch_size, self.inference_batch_size):
            end = min(start + self.inference_batch_size, batch_size)
            state_array = np.asarray(encoded_states[start:end])
            if self.device.type == "cpu":
                state_array = state_array.astype(np.float32, copy=False)

            states = torch.from_numpy(state_array)
            if self.device.type == "cuda":
                states = states.pin_memory()
            states = states.to(self.device, non_blocking=True)
            if self.channels_last:
                states = states.contiguous(memory_format=torch.channels_last)

            chunk_lengths = legal_lengths[start:end]
            max_actions = int(chunk_lengths.max(initial=0))
            action_array = legal_actions[start:end, :max_actions].astype(
                np.int64,
                copy=False,
            )
            actions = torch.from_numpy(action_array)
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

            policies[start:end, :max_actions] = gathered.float().cpu().numpy()
            values[start:end] = value.float().cpu().numpy()

        return policies, values


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
        self.terminal_values = np.empty(self.capacity, dtype=np.float32)
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
        self.state_cache = [None] * self.capacity
        self.clear_records(0, 1)

    def clear_records(self, start, end):
        self.visit_counts[start:end] = 0
        self.value_sums[start:end] = 0.0
        self.virtual_visits[start:end] = 0
        self.first_children[start:end] = UNEXPANDED
        self.child_counts[start:end] = 0
        self.terminal_values[start:end] = np.nan

    def prepare(self, required_records):
        if self.next_free + required_records > self.capacity:
            self.reset()

    def cache_root_state(self, state):
        self.state_cache[self.root_index] = state.copy()

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
        matches = np.flatnonzero(actions == action)
        assert len(matches)
        return unpack_move(self.moves[start + int(matches[0])])

    def advance(self, action):
        start, end = self.child_range(self.root_index)
        actions = self.actions[start:end]
        matches = np.flatnonzero(actions == action)
        assert len(matches)
        self.root_index = start + int(matches[0])
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
        self.maximum_path_length = self.simulations + 2
        self.buffer_tree_count = 0
        self.buffer_pending_count = 0

    def create_tree(self):
        return MCTSTree(self.tree_capacity)

    def ensure_buffers(self, tree_count):
        pending_count = max(1, tree_count * self.parallel_searches)
        if (
            tree_count <= self.buffer_tree_count
            and pending_count <= self.buffer_pending_count
        ):
            return

        self.buffer_tree_count = max(tree_count, self.buffer_tree_count)
        self.buffer_pending_count = max(
            pending_count,
            self.buffer_pending_count,
        )
        tree_count = self.buffer_tree_count
        pending_count = self.buffer_pending_count

        self.selection_paths = np.empty(
            (tree_count, self.maximum_path_length),
            dtype=np.int32,
        )
        self.selection_path_lengths = np.empty(tree_count, dtype=np.uint16)
        self.selection_nodes = np.empty(tree_count, dtype=np.int32)
        self.selection_active = np.empty(tree_count, dtype=np.bool_)

        self.pending_paths = np.empty(
            (pending_count, self.maximum_path_length),
            dtype=np.int32,
        )
        self.pending_path_lengths = np.empty(pending_count, dtype=np.uint16)
        self.encoded_states = np.empty(
            (pending_count, INPUT_PLANES, 8, 8),
            dtype=np.float16,
        )
        self.legal_actions = np.empty(
            (pending_count, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.packed_moves = np.empty(
            (pending_count, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.legal_lengths = np.empty(pending_count, dtype=np.uint16)

    def run(self, states, roots=None, add_noise=False):
        if roots is None:
            roots = [self.create_tree() for _ in states]

        self.ensure_buffers(len(states))
        required_records = (self.simulations + 1) * MAX_LEGAL_ACTIONS + 1
        target_visits = []
        terminal_flags = np.zeros(len(states), dtype=np.bool_)

        initial_trees = []
        initial_nodes = []
        initial_states = []
        initial_count = 0

        for index, (root, state) in enumerate(zip(roots, states, strict=True)):
            root.prepare(required_records)
            root.cache_root_state(state)
            target_visits.append(root.visit_count + self.simulations)

            if root.expanded:
                continue

            terminal_value = self.prepare_leaf(state, initial_count)
            if terminal_value is not None:
                root.terminal_values[root.root_index] = terminal_value
                terminal_flags[index] = True
                continue

            encode_state(state, output=self.encoded_states[initial_count])
            initial_trees.append(root)
            initial_nodes.append(root.root_index)
            initial_states.append(state)
            initial_count += 1

        if initial_count:
            self.evaluate_and_expand(
                initial_trees,
                initial_nodes,
                initial_states,
                initial_count,
            )

        for root in roots:
            if root.expanded:
                self.set_root_priors(root, add_noise)

        while True:
            active_indices = [
                index
                for index, (root, target, terminal) in enumerate(
                    zip(
                        roots,
                        target_visits,
                        terminal_flags,
                        strict=True,
                    )
                )
                if not terminal
                and root.visit_count + root.virtual_visit_count < target
            ]
            if not active_indices:
                break

            pending_trees = []
            pending_nodes = []
            pending_states = []
            pending_count = 0
            made_progress = False

            for _ in range(self.parallel_searches):
                wave_indices = [
                    index
                    for index in active_indices
                    if roots[index].visit_count
                    + roots[index].virtual_visit_count
                    < target_visits[index]
                ]
                if not wave_indices:
                    break

                wave_trees = [roots[index] for index in wave_indices]
                leaf_ids, path_lengths = self.select_leaves(wave_trees)

                for row, tree_index in enumerate(wave_indices):
                    leaf_id = int(leaf_ids[row])
                    if leaf_id < 0:
                        continue

                    root = roots[tree_index]
                    path_length = int(path_lengths[row])
                    path = self.selection_paths[row]
                    self.reserve(root, path, path_length)
                    made_progress = True

                    terminal_value = root.terminal_values[leaf_id]
                    if np.isfinite(terminal_value):
                        self.release(root, path, path_length)
                        self.backup(
                            root,
                            path,
                            path_length,
                            float(terminal_value),
                        )
                        continue

                    leaf_state = self.materialize_state(
                        root,
                        path,
                        path_length,
                    )
                    terminal_value = self.prepare_leaf(
                        leaf_state,
                        pending_count,
                    )
                    if terminal_value is not None:
                        root.terminal_values[leaf_id] = terminal_value
                        self.release(root, path, path_length)
                        self.backup(
                            root,
                            path,
                            path_length,
                            terminal_value,
                        )
                        continue

                    self.pending_paths[
                        pending_count,
                        :path_length,
                    ] = path[:path_length]
                    self.pending_path_lengths[pending_count] = path_length
                    encode_state(
                        leaf_state,
                        output=self.encoded_states[pending_count],
                    )
                    pending_trees.append(root)
                    pending_nodes.append(leaf_id)
                    pending_states.append(leaf_state)
                    pending_count += 1

            if pending_count:
                values = self.evaluate_and_expand(
                    pending_trees,
                    pending_nodes,
                    pending_states,
                    pending_count,
                )
                for index, (root, value) in enumerate(
                    zip(pending_trees, values, strict=True)
                ):
                    path_length = int(self.pending_path_lengths[index])
                    path = self.pending_paths[index]
                    self.release(root, path, path_length)
                    self.backup(
                        root,
                        path,
                        path_length,
                        float(value),
                    )

            if not made_progress:
                break

        return roots

    def select_leaves(self, trees):
        tree_count = len(trees)
        paths = self.selection_paths[:tree_count]
        path_lengths = self.selection_path_lengths[:tree_count]
        current_nodes = self.selection_nodes[:tree_count]
        active = self.selection_active[:tree_count]

        path_lengths.fill(1)
        active.fill(True)
        for row, tree in enumerate(trees):
            node_id = tree.root_index
            current_nodes[row] = node_id
            paths[row, 0] = node_id

        while True:
            advanced = False
            for row, tree in enumerate(trees):
                if not active[row]:
                    continue

                node_id = int(current_nodes[row])
                if (
                    tree.first_children[node_id] == UNEXPANDED
                    or not tree.child_counts[node_id]
                ):
                    active[row] = False
                    continue

                child_id = self.select_child(tree, node_id)
                if child_id < 0:
                    current_nodes[row] = -1
                    active[row] = False
                    continue

                length = int(path_lengths[row])
                if length >= self.maximum_path_length:
                    raise RuntimeError(
                        "MCTS path exceeded its allocated buffer"
                    )
                paths[row, length] = child_id
                path_lengths[row] = length + 1
                current_nodes[row] = child_id
                advanced = True

            if not advanced:
                break

        return current_nodes, path_lengths

    @staticmethod
    def select_child(tree, node_id):
        start, end = tree.child_range(node_id)
        child_count = end - start
        if node_id == tree.root_index and tree.root_prior_count == child_count:
            priors = tree.root_priors
        else:
            priors = tree.priors[start:end]

        parent_visits = max(
            1,
            int(tree.visit_counts[node_id])
            + int(tree.virtual_visits[node_id]),
        )
        exploration_scale = C_PUCT * parent_visits**0.5
        best_score = -np.inf
        best_child = -1

        for offset, child_id in enumerate(range(start, end)):
            visits = int(tree.visit_counts[child_id])
            virtual_visits = int(tree.virtual_visits[child_id])
            if virtual_visits and not visits:
                continue

            value = (
                -float(tree.value_sums[child_id]) / visits if visits else 0.0
            )
            exploration = (
                float(priors[offset])
                * exploration_scale
                / (visits + virtual_visits + 1)
            )
            score = value + exploration - virtual_visits * VIRTUAL_LOSS
            if score > best_score:
                best_score = score
                best_child = child_id

        return best_child

    def materialize_state(self, tree, path, path_length):
        leaf_id = int(path[path_length - 1])
        cached = tree.state_cache[leaf_id]
        if cached is not None:
            return cached

        ancestor_position = path_length - 2
        while ancestor_position >= 0:
            ancestor_id = int(path[ancestor_position])
            cached = tree.state_cache[ancestor_id]
            if cached is not None:
                break
            ancestor_position -= 1

        if cached is None:
            raise RuntimeError("MCTS path has no cached ancestor state")

        state = cached.copy()
        for position in range(ancestor_position + 1, path_length):
            node_id = int(path[position])
            state.push(unpack_move(tree.moves[node_id]))
        tree.state_cache[leaf_id] = state
        return state

    @staticmethod
    def fill_legal_action_data(state, actions, packed_moves):
        count = 0
        turn = state.board.turn
        for move in state.board.legal_moves:
            actions[count] = move_to_action(move, turn)
            packed_moves[count] = pack_move(move)
            count += 1

        actions[count:] = 0
        return count

    @staticmethod
    def legal_action_data(state):
        actions = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint16)
        packed_moves = np.empty(MAX_LEGAL_ACTIONS, dtype=np.uint16)
        count = MCTS.fill_legal_action_data(state, actions, packed_moves)
        return actions[:count].copy(), packed_moves[:count].copy()

    def prepare_leaf(self, state, row):
        if state.is_rule_draw():
            self.legal_lengths[row] = 0
            return 0.0

        count = self.fill_legal_action_data(
            state,
            self.legal_actions[row],
            self.packed_moves[row],
        )
        self.legal_lengths[row] = count
        if count:
            return None
        return -1.0 if state.board.is_check() else 0.0

    def evaluate_and_expand(self, trees, node_ids, states, count):
        policies, values = self.evaluator.evaluate_encoded(
            self.encoded_states[:count],
            self.legal_actions[:count],
            self.legal_lengths[:count],
        )

        for index, (tree, node_id, state) in enumerate(
            zip(trees, node_ids, states, strict=True)
        ):
            child_count = int(self.legal_lengths[index])
            logits = policies[index, :child_count]
            logits = logits - logits.max()
            priors = np.exp(logits)
            priors /= priors.sum()
            tree.allocate_children(
                node_id,
                self.legal_actions[index, :child_count],
                self.packed_moves[index, :child_count],
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
    def reserve(tree, path, path_length):
        for position in range(path_length):
            tree.virtual_visits[int(path[position])] += 1

    @staticmethod
    def release(tree, path, path_length):
        for position in range(path_length):
            tree.virtual_visits[int(path[position])] -= 1

    @staticmethod
    def backup(tree, path, path_length, value):
        for position in range(path_length - 1, -1, -1):
            node_id = int(path[position])
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
