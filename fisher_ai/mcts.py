import numpy as np
import torch

from fisher_ai import chess
from fisher_ai.encoding import (
    INPUT_PLANES,
    StateEncodingWorkspace,
    encode_states,
    moves_to_actions,
)
from fisher_ai.game import HISTORY_LENGTH, MAX_GAME_PLIES, GameState

MAX_LEGAL_ACTIONS = 256
UNEXPANDED = -1
NO_STATE = -1
C_PUCT = 1.5
VIRTUAL_LOSS = 1.0
DIRICHLET_ALPHA = 0.3
DIRICHLET_FRACTION = 0.25


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


class MCTSStatePool:
    def __init__(self, capacity=2048):
        self.capacity = int(capacity)
        self.count = 0
        self.root_slot = NO_STATE
        self.root_hashes = np.empty(MAX_GAME_PLIES + 1, dtype=np.uint64)
        self.root_hash_count = 0
        self.allocate(self.capacity)

    def allocate(self, capacity):
        old_count = getattr(self, "count", 0)
        old_arrays = {
            name: getattr(self, name)
            for name in (
                "pawns",
                "knights",
                "bishops",
                "rooks",
                "queens",
                "kings",
                "occupied_white",
                "occupied_black",
                "occupied",
                "turn",
                "castling_rights",
                "ep_square",
                "halfmove_clock",
                "fullmove_number",
                "history_bitboards",
                "history_repetitions",
                "history_length",
                "repetition_count",
                "position_hash",
                "parent_slot",
            )
            if hasattr(self, name)
        }

        self.capacity = int(capacity)
        self.pawns = np.empty(self.capacity, dtype=np.uint64)
        self.knights = np.empty(self.capacity, dtype=np.uint64)
        self.bishops = np.empty(self.capacity, dtype=np.uint64)
        self.rooks = np.empty(self.capacity, dtype=np.uint64)
        self.queens = np.empty(self.capacity, dtype=np.uint64)
        self.kings = np.empty(self.capacity, dtype=np.uint64)
        self.occupied_white = np.empty(self.capacity, dtype=np.uint64)
        self.occupied_black = np.empty(self.capacity, dtype=np.uint64)
        self.occupied = np.empty(self.capacity, dtype=np.uint64)
        self.turn = np.empty(self.capacity, dtype=np.bool_)
        self.castling_rights = np.empty(self.capacity, dtype=np.uint64)
        self.ep_square = np.empty(self.capacity, dtype=np.int8)
        self.halfmove_clock = np.empty(self.capacity, dtype=np.uint16)
        self.fullmove_number = np.empty(self.capacity, dtype=np.uint16)
        self.history_bitboards = np.empty(
            (self.capacity, HISTORY_LENGTH, 12),
            dtype=np.uint64,
        )
        self.history_repetitions = np.empty(
            (self.capacity, HISTORY_LENGTH),
            dtype=np.uint8,
        )
        self.history_length = np.empty(self.capacity, dtype=np.uint8)
        self.repetition_count = np.empty(self.capacity, dtype=np.uint8)
        self.position_hash = np.empty(self.capacity, dtype=np.uint64)
        self.parent_slot = np.empty(self.capacity, dtype=np.int32)

        for name, old_array in old_arrays.items():
            getattr(self, name)[:old_count] = old_array[:old_count]

    def reset(self):
        self.count = 0
        self.root_slot = NO_STATE
        self.root_hash_count = 0

    def grow(self):
        self.allocate(self.capacity * 2)

    def allocate_slot(self):
        if self.count >= self.capacity:
            self.grow()
        slot = self.count
        self.count += 1
        return slot

    def write_state(self, slot, state):
        board = state.board
        self.pawns[slot] = board.pawns
        self.knights[slot] = board.knights
        self.bishops[slot] = board.bishops
        self.rooks[slot] = board.rooks
        self.queens[slot] = board.queens
        self.kings[slot] = board.kings
        self.occupied_white[slot] = board.occupied_co[chess.WHITE]
        self.occupied_black[slot] = board.occupied_co[chess.BLACK]
        self.occupied[slot] = board.occupied
        self.turn[slot] = board.turn
        self.castling_rights[slot] = board.castling_rights
        self.ep_square[slot] = (
            -1 if board.ep_square is None else board.ep_square
        )
        self.halfmove_clock[slot] = board.halfmove_clock
        self.fullmove_number[slot] = board.fullmove_number
        self.history_bitboards[slot] = state.history_bitboards
        self.history_repetitions[slot] = state.history_repetitions
        self.history_length[slot] = state.history_length
        self.repetition_count[slot] = state.repetition_count

    def store_root(self, state, slot=None):
        if slot is None:
            slot = self.allocate_slot()
        self.write_state(slot, state)
        self.parent_slot[slot] = NO_STATE
        self.position_hash[slot] = state.position_hashes[
            state.position_hash_length - 1
        ]
        self.root_slot = slot
        self.root_hash_count = state.position_hash_length
        self.root_hashes[: self.root_hash_count] = state.position_hashes[
            : self.root_hash_count
        ]
        return slot

    def store_child(self, state, parent_slot, position_hash):
        slot = self.allocate_slot()
        self.write_state(slot, state)
        self.parent_slot[slot] = parent_slot
        self.position_hash[slot] = position_hash
        return slot

    def repetition_count_for(self, parent_slot, position_hash):
        count = int(
            np.count_nonzero(
                self.root_hashes[: self.root_hash_count] == position_hash
            )
        )
        slot = parent_slot
        while slot != NO_STATE and slot != self.root_slot:
            if self.position_hash[slot] == position_hash:
                count += 1
            slot = int(self.parent_slot[slot])
        return count

    def load(self, slot, state):
        board = state.board
        board.pawns = int(self.pawns[slot])
        board.knights = int(self.knights[slot])
        board.bishops = int(self.bishops[slot])
        board.rooks = int(self.rooks[slot])
        board.queens = int(self.queens[slot])
        board.kings = int(self.kings[slot])
        board.occupied_co[chess.WHITE] = int(self.occupied_white[slot])
        board.occupied_co[chess.BLACK] = int(self.occupied_black[slot])
        board.occupied = int(self.occupied[slot])
        board.turn = bool(self.turn[slot])
        board.castling_rights = int(self.castling_rights[slot])
        ep_square = int(self.ep_square[slot])
        board.ep_square = None if ep_square < 0 else ep_square
        board.halfmove_clock = int(self.halfmove_clock[slot])
        board.fullmove_number = int(self.fullmove_number[slot])
        state.history_bitboards[:] = self.history_bitboards[slot]
        state.history_repetitions[:] = self.history_repetitions[slot]
        state.history_length = int(self.history_length[slot])
        state.repetition_count = int(self.repetition_count[slot])
        return state


class MCTSTree:
    def __init__(self, capacity, state_capacity=256):
        self.capacity = int(capacity)
        self.actions = np.empty(self.capacity, dtype=np.uint16)
        self.moves = np.empty(self.capacity, dtype=np.uint16)
        self.priors = np.empty(self.capacity, dtype=np.float32)
        self.visit_counts = np.empty(self.capacity, dtype=np.uint32)
        self.mean_values = np.empty(self.capacity, dtype=np.float32)
        self.virtual_visits = np.empty(self.capacity, dtype=np.uint16)
        self.first_children = np.empty(self.capacity, dtype=np.int32)
        self.child_counts = np.empty(self.capacity, dtype=np.uint16)
        self.terminal_values = np.empty(self.capacity, dtype=np.float32)
        self.state_slots = np.empty(self.capacity, dtype=np.int32)
        self.root_priors = np.empty(MAX_LEGAL_ACTIONS, dtype=np.float32)
        self.state_pool = MCTSStatePool(state_capacity)
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
        self.state_pool.reset()
        self.clear_records(0, 1)

    def clear_records(self, start, end):
        self.visit_counts[start:end] = 0
        self.mean_values[start:end] = 0.0
        self.virtual_visits[start:end] = 0
        self.first_children[start:end] = UNEXPANDED
        self.child_counts[start:end] = 0
        self.terminal_values[start:end] = np.nan
        self.state_slots[start:end] = NO_STATE

    def prepare(self, required_records):
        if self.next_free + required_records > self.capacity:
            self.reset()

    def cache_root_state(self, state):
        slot = int(self.state_slots[self.root_index])
        slot = self.state_pool.store_root(
            state,
            slot=None if slot == NO_STATE else slot,
        )
        self.state_slots[self.root_index] = slot

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

    def node_for_action(self, action):
        start, end = self.child_range(self.root_index)
        matches = np.flatnonzero(self.actions[start:end] == action)
        assert len(matches)
        return start + int(matches[0])

    def move_for_action(self, action):
        return int(self.moves[self.node_for_action(action)])

    def advance(self, action):
        self.root_index = self.node_for_action(action)
        self.root_prior_count = 0
        return int(self.moves[self.root_index])


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
        self.state_pool_capacity = self.simulations * 16 + 64
        self.buffer_tree_count = 0
        self.buffer_pending_count = 0
        lookup_size = self.simulations + self.parallel_searches + 4
        self.exploration_lookup = C_PUCT * np.sqrt(
            np.arange(lookup_size, dtype=np.float32)
        )
        self.reciprocal_lookup = 1.0 / np.arange(
            1,
            lookup_size + 1,
            dtype=np.float32,
        )

    def create_tree(self):
        return MCTSTree(self.tree_capacity, self.state_pool_capacity)

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
        self.selection_active_rows = np.empty(tree_count, dtype=np.uint16)
        self.selection_child_counts = np.empty(tree_count, dtype=np.uint16)
        self.selection_starts = np.empty(tree_count, dtype=np.int32)
        self.selection_scores = np.empty(
            (tree_count, MAX_LEGAL_ACTIONS),
            dtype=np.float32,
        )
        self.selection_visits = np.empty(
            (tree_count, MAX_LEGAL_ACTIONS),
            dtype=np.uint32,
        )
        self.selection_virtual = np.empty(
            (tree_count, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.selection_means = np.empty(
            (tree_count, MAX_LEGAL_ACTIONS),
            dtype=np.float32,
        )
        self.selection_priors = np.empty(
            (tree_count, MAX_LEGAL_ACTIONS),
            dtype=np.float32,
        )
        self.selection_scales = np.empty(tree_count, dtype=np.float32)

        self.pending_paths = np.empty(
            (pending_count, self.maximum_path_length),
            dtype=np.int32,
        )
        self.pending_path_lengths = np.empty(pending_count, dtype=np.uint16)
        self.pending_tree_indices = np.empty(pending_count, dtype=np.uint16)
        self.pending_nodes = np.empty(pending_count, dtype=np.int32)
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
        self.state_workspaces = [GameState() for _ in range(pending_count)]
        self.encoding_workspace = StateEncodingWorkspace(pending_count)
        self.active_indices = np.empty(tree_count, dtype=np.uint16)
        self.wave_indices = np.empty(tree_count, dtype=np.uint16)
        self.target_visits = np.empty(tree_count, dtype=np.uint32)
        self.terminal_flags = np.empty(tree_count, dtype=np.bool_)

    def run(self, states, roots=None, add_noise=False):
        if roots is None:
            roots = [self.create_tree() for _ in states]

        tree_count = len(states)
        self.ensure_buffers(tree_count)
        required_records = (self.simulations + 1) * MAX_LEGAL_ACTIONS + 1
        self.terminal_flags[:tree_count] = False
        initial_count = 0

        for index, (root, state) in enumerate(zip(roots, states, strict=True)):
            root.prepare(required_records)
            root.cache_root_state(state)
            self.target_visits[index] = root.visit_count + self.simulations

            if root.expanded:
                continue

            terminal_value = self.prepare_leaf(state, initial_count)
            if terminal_value is not None:
                root.terminal_values[root.root_index] = terminal_value
                self.terminal_flags[index] = True
                continue

            self.pending_tree_indices[initial_count] = index
            self.pending_nodes[initial_count] = root.root_index
            self.state_workspaces[initial_count].copy_from(state)
            initial_count += 1

        if initial_count:
            encode_states(
                self.state_workspaces[:initial_count],
                output=self.encoded_states[:initial_count],
                workspace=self.encoding_workspace,
            )
            self.evaluate_and_expand(roots, initial_count)

        for root in roots:
            if root.expanded:
                self.set_root_priors(root, add_noise)

        while True:
            active_count = self.collect_active_indices(roots, tree_count)
            if not active_count:
                break

            pending_count = 0
            made_progress = False

            for _ in range(self.parallel_searches):
                wave_count = self.collect_wave_indices(
                    roots,
                    active_count,
                )
                if not wave_count:
                    break

                leaf_ids, path_lengths = self.select_leaves(
                    roots,
                    self.wave_indices,
                    wave_count,
                )

                for row in range(wave_count):
                    tree_index = int(self.wave_indices[row])
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

                    workspace = self.state_workspaces[pending_count]
                    leaf_state = self.materialize_state(
                        root,
                        path,
                        path_length,
                        workspace,
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
                    self.pending_tree_indices[pending_count] = tree_index
                    self.pending_nodes[pending_count] = leaf_id
                    pending_count += 1

            if pending_count:
                encode_states(
                    self.state_workspaces[:pending_count],
                    output=self.encoded_states[:pending_count],
                    workspace=self.encoding_workspace,
                )
                values = self.evaluate_and_expand(roots, pending_count)
                for index in range(pending_count):
                    root = roots[int(self.pending_tree_indices[index])]
                    path_length = int(self.pending_path_lengths[index])
                    path = self.pending_paths[index]
                    self.release(root, path, path_length)
                    self.backup(
                        root,
                        path,
                        path_length,
                        float(values[index]),
                    )

            if not made_progress:
                break

        return roots

    def collect_active_indices(self, roots, tree_count):
        count = 0
        for index in range(tree_count):
            root = roots[index]
            if self.terminal_flags[index]:
                continue
            if (
                root.visit_count + root.virtual_visit_count
                >= self.target_visits[index]
            ):
                continue
            self.active_indices[count] = index
            count += 1
        return count

    def collect_wave_indices(self, roots, active_count):
        count = 0
        for offset in range(active_count):
            index = int(self.active_indices[offset])
            root = roots[index]
            if (
                root.visit_count + root.virtual_visit_count
                >= self.target_visits[index]
            ):
                continue
            self.wave_indices[count] = index
            count += 1
        return count

    def select_leaves(self, roots, tree_indices, tree_count):
        paths = self.selection_paths[:tree_count]
        path_lengths = self.selection_path_lengths[:tree_count]
        current_nodes = self.selection_nodes[:tree_count]
        active = self.selection_active[:tree_count]

        path_lengths.fill(1)
        active.fill(True)
        for row in range(tree_count):
            tree = roots[int(tree_indices[row])]
            node_id = tree.root_index
            current_nodes[row] = node_id
            paths[row, 0] = node_id

        while True:
            active_row_count = 0
            for row in range(tree_count):
                if not active[row]:
                    continue
                tree = roots[int(tree_indices[row])]
                node_id = int(current_nodes[row])
                if (
                    tree.first_children[node_id] == UNEXPANDED
                    or not tree.child_counts[node_id]
                ):
                    active[row] = False
                    continue
                self.selection_active_rows[active_row_count] = row
                active_row_count += 1

            if not active_row_count:
                break

            active_rows = self.selection_active_rows[:active_row_count]
            selected = self.select_children_batch(
                roots,
                tree_indices,
                current_nodes,
                active_rows,
                active_row_count,
            )
            advanced = False
            for local_index in range(active_row_count):
                row = int(active_rows[local_index])
                child_id = int(selected[local_index])
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

    def select_children_batch(
        self,
        roots,
        tree_indices,
        current_nodes,
        active_rows,
        row_count,
    ):
        scores = self.selection_scores[:row_count]
        visits = self.selection_visits[:row_count]
        virtual = self.selection_virtual[:row_count]
        means = self.selection_means[:row_count]
        priors = self.selection_priors[:row_count]
        scores.fill(-np.inf)

        for output_row, source_row in enumerate(active_rows):
            tree = roots[int(tree_indices[source_row])]
            node_id = int(current_nodes[source_row])
            start, end = tree.child_range(node_id)
            child_count = end - start
            self.selection_starts[output_row] = start
            self.selection_child_counts[output_row] = child_count
            visits[output_row, :child_count] = tree.visit_counts[start:end]
            virtual[output_row, :child_count] = tree.virtual_visits[start:end]
            means[output_row, :child_count] = tree.mean_values[start:end]
            if (
                node_id == tree.root_index
                and tree.root_prior_count == child_count
            ):
                priors[output_row, :child_count] = tree.root_priors[
                    :child_count
                ]
            else:
                priors[output_row, :child_count] = tree.priors[start:end]

            parent_visits = max(
                1,
                int(tree.visit_counts[node_id])
                + int(tree.virtual_visits[node_id]),
            )
            if parent_visits < len(self.exploration_lookup):
                self.selection_scales[output_row] = self.exploration_lookup[
                    parent_visits
                ]
            else:
                self.selection_scales[output_row] = C_PUCT * parent_visits**0.5

        max_children = int(
            self.selection_child_counts[:row_count].max(initial=0)
        )
        current_visits = visits[:, :max_children]
        current_virtual = virtual[:, :max_children]
        denominators = current_visits + current_virtual + 1
        current_scores = scores[:, :max_children]
        current_scores[:] = -means[:, :max_children]
        current_scores += (
            priors[:, :max_children]
            * self.selection_scales[:row_count, None]
            / denominators
        )
        current_scores -= current_virtual * VIRTUAL_LOSS
        current_scores[(current_virtual > 0) & (current_visits == 0)] = -np.inf

        columns = np.arange(max_children)
        current_scores[
            columns[None, :] >= self.selection_child_counts[:row_count, None]
        ] = -np.inf
        offsets = np.argmax(current_scores, axis=1)
        finite = np.isfinite(current_scores[np.arange(row_count), offsets])
        selected = self.selection_starts[:row_count] + offsets
        selected[~finite] = -1
        return selected

    def materialize_state(self, tree, path, path_length, output):
        leaf_id = int(path[path_length - 1])
        slot = int(tree.state_slots[leaf_id])
        if slot != NO_STATE:
            return tree.state_pool.load(slot, output)

        ancestor_position = path_length - 2
        while ancestor_position >= 0:
            ancestor_id = int(path[ancestor_position])
            slot = int(tree.state_slots[ancestor_id])
            if slot != NO_STATE:
                break
            ancestor_position -= 1

        if slot == NO_STATE:
            raise RuntimeError("MCTS path has no cached ancestor state")

        tree.state_pool.load(slot, output)
        parent_slot = slot
        for position in range(ancestor_position + 1, path_length):
            node_id = int(path[position])
            output.board.push(int(tree.moves[node_id]))
            position_hash = output.board.position_hash()
            repetition_count = tree.state_pool.repetition_count_for(
                parent_slot,
                position_hash,
            )
            output.repetition_count = repetition_count
            output._append_snapshot(repetition_count)
            parent_slot = tree.state_pool.store_child(
                output,
                parent_slot,
                position_hash,
            )
            tree.state_slots[node_id] = parent_slot

        return output

    @staticmethod
    def fill_legal_action_data(state, actions, packed_moves):
        count = state.board.fill_legal_moves(packed_moves)
        moves_to_actions(
            packed_moves,
            state.board.turn,
            output=actions,
            count=count,
        )
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

    def evaluate_and_expand(self, roots, count):
        policies, values = self.evaluator.evaluate_encoded(
            self.encoded_states[:count],
            self.legal_actions[:count],
            self.legal_lengths[:count],
        )

        for index in range(count):
            tree = roots[int(self.pending_tree_indices[index])]
            node_id = int(self.pending_nodes[index])
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
        tree.virtual_visits[path[:path_length]] += 1

    @staticmethod
    def release(tree, path, path_length):
        tree.virtual_visits[path[:path_length]] -= 1

    @staticmethod
    def backup(tree, path, path_length, value):
        for position in range(path_length - 1, -1, -1):
            node_id = int(path[position])
            visits = int(tree.visit_counts[node_id])
            tree.visit_counts[node_id] = visits + 1
            mean = float(tree.mean_values[node_id])
            tree.mean_values[node_id] = mean + (value - mean) / (visits + 1)
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
