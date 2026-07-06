import numpy as np

from fisher_ai.encoding import INPUT_PLANES, encode_window_batch
from fisher_ai.mcts import MAX_LEGAL_ACTIONS


class GameRecord:
    def __init__(
        self,
        snapshot_bitboards,
        snapshot_repetitions,
        current_colors,
        plies,
        castling_masks,
        halfmove_clocks,
        legal_lengths,
        legal_actions,
        visit_counts,
        values,
    ):
        self.snapshot_bitboards = np.asarray(
            snapshot_bitboards,
            dtype=np.uint64,
        )
        self.snapshot_repetitions = np.asarray(
            snapshot_repetitions,
            dtype=np.uint8,
        )
        self.current_colors = np.asarray(current_colors, dtype=np.bool_)
        self.plies = np.asarray(plies, dtype=np.uint16)
        self.castling_masks = np.asarray(castling_masks, dtype=np.uint8)
        self.halfmove_clocks = np.asarray(
            halfmove_clocks,
            dtype=np.uint8,
        )
        self.legal_lengths = np.asarray(legal_lengths, dtype=np.uint16)
        self.legal_actions = np.asarray(legal_actions, dtype=np.uint16)
        self.visit_counts = np.asarray(visit_counts, dtype=np.uint16)
        self.values = np.asarray(values, dtype=np.float32)

    @property
    def position_count(self):
        return len(self.values)


class InMemoryWindow:
    def __init__(self, target_positions):
        self.target_positions = int(target_positions)
        self.capacity = max(1024, self.target_positions)
        self.position_count = 0
        self.game_count = 0
        self.allocate(self.capacity)

    @property
    def full(self):
        return self.position_count >= self.target_positions

    def allocate(self, capacity):
        old_count = getattr(self, "position_count", 0)
        old_arrays = {
            name: getattr(self, name)
            for name in (
                "snapshot_bitboards",
                "snapshot_repetitions",
                "game_starts",
                "current_colors",
                "plies",
                "castling_masks",
                "halfmove_clocks",
                "legal_lengths",
                "legal_actions",
                "visit_counts",
                "values",
            )
            if hasattr(self, name)
        }

        self.capacity = int(capacity)
        self.snapshot_bitboards = np.empty(
            (self.capacity, 12),
            dtype=np.uint64,
        )
        self.snapshot_repetitions = np.empty(
            self.capacity,
            dtype=np.uint8,
        )
        self.game_starts = np.empty(self.capacity, dtype=np.int64)
        self.current_colors = np.empty(self.capacity, dtype=np.bool_)
        self.plies = np.empty(self.capacity, dtype=np.uint16)
        self.castling_masks = np.empty(self.capacity, dtype=np.uint8)
        self.halfmove_clocks = np.empty(self.capacity, dtype=np.uint8)
        self.legal_lengths = np.empty(self.capacity, dtype=np.uint16)
        self.legal_actions = np.zeros(
            (self.capacity, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.visit_counts = np.zeros(
            (self.capacity, MAX_LEGAL_ACTIONS),
            dtype=np.uint16,
        )
        self.values = np.empty(self.capacity, dtype=np.float32)

        for name, old_array in old_arrays.items():
            getattr(self, name)[:old_count] = old_array[:old_count]

    def ensure_capacity(self, required):
        if required <= self.capacity:
            return
        self.allocate(max(required, self.capacity * 2))

    def add_game(self, game):
        self.add_arrays(
            game.snapshot_bitboards,
            game.snapshot_repetitions,
            game.current_colors,
            game.plies,
            game.castling_masks,
            game.halfmove_clocks,
            game.legal_lengths,
            game.legal_actions,
            game.visit_counts,
            game.values,
        )

    def add_arrays(
        self,
        snapshot_bitboards,
        snapshot_repetitions,
        current_colors,
        plies,
        castling_masks,
        halfmove_clocks,
        legal_lengths,
        legal_actions,
        visit_counts,
        values,
    ):
        count = len(values)
        if not count:
            return

        start = self.position_count
        end = start + count
        self.ensure_capacity(end)
        self.snapshot_bitboards[start:end] = snapshot_bitboards
        self.snapshot_repetitions[start:end] = snapshot_repetitions
        self.game_starts[start:end] = start
        self.current_colors[start:end] = current_colors
        self.plies[start:end] = plies
        self.castling_masks[start:end] = castling_masks
        self.halfmove_clocks[start:end] = halfmove_clocks
        self.legal_lengths[start:end] = legal_lengths
        self.legal_actions[start:end] = legal_actions
        self.visit_counts[start:end] = visit_counts
        self.values[start:end] = values
        self.position_count = end
        self.game_count += 1

    def shuffled_indices(self, rng):
        indices = np.arange(self.position_count, dtype=np.int64)
        rng.shuffle(indices)
        return indices

    def materialize_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        batch_size = len(indices)
        states = np.empty(
            (batch_size, INPUT_PLANES, 8, 8),
            dtype=np.float16,
        )
        encode_window_batch(
            self.snapshot_bitboards,
            self.snapshot_repetitions,
            self.game_starts,
            indices,
            self.current_colors,
            self.plies,
            self.castling_masks,
            self.halfmove_clocks,
            output=states,
        )

        lengths = self.legal_lengths[indices]
        max_legal_moves = int(lengths.max(initial=0))
        legal_actions = self.legal_actions[
            indices,
            :max_legal_moves,
        ].astype(np.int64, copy=True)
        visit_counts = self.visit_counts[
            indices,
            :max_legal_moves,
        ].astype(np.float32, copy=True)
        legal_mask = np.arange(max_legal_moves)[None, :] < lengths[:, None]
        values = self.values[indices].copy()
        return states, legal_actions, visit_counts, legal_mask, values
