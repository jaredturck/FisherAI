"""Store generated self-play positions in contiguous arrays."""

import numpy as np

from fisher_ai.encoding import INPUT_PLANES, encode_window_batch
from fisher_ai.mcts import MAX_LEGAL_ACTIONS


class GameRecord:
    """Store one completed game as contiguous training arrays."""

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
        """Return the number of stored positions in the game."""
        return len(self.values)


class InMemoryWindow:
    """Accumulate completed games in contiguous structure-of-arrays form."""

    ARRAY_NAMES = (
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

    def __init__(self, target_positions, initial_capacity=None):
        self.target_positions = int(target_positions)
        if initial_capacity is None:
            initial_capacity = self.target_positions
        self.capacity = max(1024, int(initial_capacity))
        self.position_count = 0
        self.game_count = 0
        self.game_ends = []
        self.allocate(self.capacity)

    @property
    def full(self):
        """Report whether the target window size has been reached."""
        return self.position_count >= self.target_positions

    def allocate(self, capacity):
        """Allocate contiguous arrays for the requested capacity."""
        old_count = getattr(self, "position_count", 0)
        old_arrays = {
            name: getattr(self, name)
            for name in self.ARRAY_NAMES
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
        """Grow window storage to fit a required position count."""
        if required <= self.capacity:
            return
        self.allocate(max(required, self.capacity * 2))

    def add_game(self, game):
        """Append every position from a completed game record."""
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
        """Append one complete game from contiguous training arrays."""
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
        self.game_ends.append(end)

    def game_ranges(self):
        """Yield the contiguous position range for every stored game."""
        start = 0
        for end in self.game_ends:
            yield start, end
            start = end

    def shuffled_indices(self, rng):
        """Return a shuffled index for every stored position."""
        indices = np.arange(self.position_count, dtype=np.int64)
        rng.shuffle(indices)
        return indices

    def materialize_batch(self, indices):
        """Gather and encode one training batch by position indices."""
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


class ReplayWindow(InMemoryWindow):
    """Retain a fixed-capacity FIFO history of complete self-play games."""

    def __init__(self, max_positions):
        self.max_positions = int(max_positions)
        self.game_iterations = []
        super().__init__(self.max_positions, initial_capacity=1024)

    @property
    def oldest_iteration(self):
        """Return the oldest retained source iteration when available."""
        return self.game_iterations[0] if self.game_iterations else None

    def ensure_capacity(self, required):
        """Grow replay storage without exceeding its configured maximum."""
        if required <= self.capacity:
            return
        self.allocate(
            min(self.max_positions, max(required, self.capacity * 2))
        )

    def _drop_prefix(self, game_count):
        """Evict the requested number of oldest complete games."""
        if game_count <= 0:
            return

        drop_positions = self.game_ends[game_count - 1]
        remaining = self.position_count - drop_positions
        for name in self.ARRAY_NAMES:
            array = getattr(self, name)
            array[:remaining] = array[drop_positions : self.position_count]

        if remaining:
            self.game_starts[:remaining] -= drop_positions
        self.position_count = remaining
        self.game_count -= game_count
        self.game_ends = [
            end - drop_positions for end in self.game_ends[game_count:]
        ]
        self.game_iterations = self.game_iterations[game_count:]

    def _make_room(self, incoming_positions):
        """Evict oldest games until an incoming window fits."""
        allowed_positions = self.max_positions - incoming_positions
        game_count = 0
        while (
            game_count < self.game_count
            and self.position_count - self.game_ends[game_count]
            > allowed_positions
        ):
            game_count += 1
        if self.position_count > allowed_positions:
            game_count += 1
        self._drop_prefix(min(game_count, self.game_count))

    def append_window(self, window, iteration):
        """Append every complete game and evict the oldest games first."""
        ranges = list(window.game_ranges())
        incoming_positions = sum(end - start for start, end in ranges)
        if incoming_positions > self.max_positions:
            while ranges and incoming_positions > self.max_positions:
                start, end = ranges.pop(0)
                incoming_positions -= end - start

        self._make_room(incoming_positions)
        for start, end in ranges:
            self.add_arrays(
                window.snapshot_bitboards[start:end],
                window.snapshot_repetitions[start:end],
                window.current_colors[start:end],
                window.plies[start:end],
                window.castling_masks[start:end],
                window.halfmove_clocks[start:end],
                window.legal_lengths[start:end],
                window.legal_actions[start:end],
                window.visit_counts[start:end],
                window.values[start:end],
            )
            self.game_iterations.append(int(iteration))

    def sample_indices(self, count, rng):
        """Sample replay positions uniformly without replacement."""
        count = min(int(count), self.position_count)
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        return rng.choice(self.position_count, size=count, replace=False)


def materialize_mixed_batch(
    fresh_window,
    replay_window,
    source_flags,
    position_indices,
):
    """Materialize one shuffled batch drawn from fresh and replay data."""
    source_flags = np.asarray(source_flags, dtype=np.bool_)
    position_indices = np.asarray(position_indices, dtype=np.int64)
    if not source_flags.any():
        return fresh_window.materialize_batch(position_indices)
    if source_flags.all():
        return replay_window.materialize_batch(position_indices)

    batch_size = len(position_indices)
    fresh_rows = np.flatnonzero(~source_flags)
    replay_rows = np.flatnonzero(source_flags)
    fresh_batch = fresh_window.materialize_batch(position_indices[fresh_rows])
    replay_batch = replay_window.materialize_batch(
        position_indices[replay_rows]
    )
    max_legal_moves = max(fresh_batch[1].shape[1], replay_batch[1].shape[1])

    states = np.empty(
        (batch_size, INPUT_PLANES, 8, 8),
        dtype=np.float16,
    )
    legal_actions = np.zeros(
        (batch_size, max_legal_moves),
        dtype=np.int64,
    )
    visit_counts = np.zeros(
        (batch_size, max_legal_moves),
        dtype=np.float32,
    )
    legal_mask = np.zeros(
        (batch_size, max_legal_moves),
        dtype=np.bool_,
    )
    values = np.empty(batch_size, dtype=np.float32)

    for rows, batch in (
        (fresh_rows, fresh_batch),
        (replay_rows, replay_batch),
    ):
        width = batch[1].shape[1]
        states[rows] = batch[0]
        legal_actions[rows, :width] = batch[1]
        visit_counts[rows, :width] = batch[2]
        legal_mask[rows, :width] = batch[3]
        values[rows] = batch[4]

    return states, legal_actions, visit_counts, legal_mask, values
