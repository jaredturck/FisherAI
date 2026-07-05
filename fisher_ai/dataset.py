from bisect import bisect_right

import numpy as np

from fisher_ai.encoding import INPUT_PLANES, encode_history


class PositionTarget:
    def __init__(
        self,
        current_color,
        ply,
        castling_mask,
        halfmove_clock,
        legal_actions,
        visit_counts,
        value=0.0,
    ):
        self.current_color = bool(current_color)
        self.ply = int(ply)
        self.castling_mask = int(castling_mask)
        self.halfmove_clock = int(halfmove_clock)
        self.legal_actions = np.asarray(legal_actions, dtype=np.uint16)
        self.visit_counts = np.asarray(visit_counts, dtype=np.uint16)
        self.value = float(value)


class GameRecord:
    def __init__(self, snapshots, samples):
        self.snapshots = snapshots
        self.samples = samples

    def materialize_state(self, sample_index, output=None):
        if output is None:
            output = np.empty((INPUT_PLANES, 8, 8), dtype=np.float16)

        sample = self.samples[sample_index]
        history_start = max(0, sample_index - 7)
        snapshots = self.snapshots[history_start : sample_index + 1]
        return encode_history(
            snapshots,
            sample.current_color,
            sample.ply,
            sample.castling_mask,
            sample.halfmove_clock,
            output=output,
        )


class InMemoryWindow:
    def __init__(self, target_positions):
        self.target_positions = int(target_positions)
        self.games = []
        self.position_count = 0
        self.cumulative_positions = []

    @property
    def full(self):
        return self.position_count >= self.target_positions

    def add_game(self, game):
        if not game.samples:
            return

        self.games.append(game)
        self.position_count += len(game.samples)
        self.cumulative_positions.append(self.position_count)

    def locate(self, position_index):
        game_index = bisect_right(
            self.cumulative_positions, int(position_index)
        )
        previous = (
            0 if game_index == 0 else self.cumulative_positions[game_index - 1]
        )
        return self.games[game_index], int(position_index - previous)

    def shuffled_indices(self, rng):
        indices = np.arange(self.position_count, dtype=np.int64)
        rng.shuffle(indices)
        return indices

    def materialize_batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        batch_size = len(indices)
        states = np.empty((batch_size, INPUT_PLANES, 8, 8), dtype=np.float16)
        samples = []

        for output_index, position_index in enumerate(indices):
            game, sample_index = self.locate(position_index)
            game.materialize_state(sample_index, output=states[output_index])
            samples.append(game.samples[sample_index])

        max_legal_moves = max(len(sample.legal_actions) for sample in samples)
        legal_actions = np.zeros((batch_size, max_legal_moves), dtype=np.int64)
        visit_counts = np.zeros(
            (batch_size, max_legal_moves), dtype=np.float32
        )
        legal_mask = np.zeros((batch_size, max_legal_moves), dtype=np.bool_)
        values = np.empty(batch_size, dtype=np.float32)

        for index, sample in enumerate(samples):
            length = len(sample.legal_actions)
            legal_actions[index, :length] = sample.legal_actions
            visit_counts[index, :length] = sample.visit_counts
            legal_mask[index, :length] = True
            values[index] = sample.value

        return states, legal_actions, visit_counts, legal_mask, values
