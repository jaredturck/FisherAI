import numpy as np

from fisher_ai.encoding import INPUT_PLANES, encode_history


class PositionTarget:
    def __init__(
        self,
        snapshot_index,
        current_color,
        ply,
        castling_mask,
        halfmove_clock,
        legal_actions,
        visit_counts,
        value=0.0,
        policy_weight=1.0,
    ):
        self.snapshot_index = int(snapshot_index)
        self.current_color = bool(current_color)
        self.ply = int(ply)
        self.castling_mask = int(castling_mask)
        self.halfmove_clock = int(halfmove_clock)
        self.legal_actions = np.asarray(legal_actions, dtype=np.uint16)
        self.visit_counts = np.asarray(visit_counts, dtype=np.uint16)
        self.value = float(value)
        self.policy_weight = float(policy_weight)


class GameRecord:
    def __init__(
        self,
        snapshots=None,
        samples=None,
        moves=None,
        result=0,
        checkpoint_step=0,
        max_game_plies=320,
    ):
        self.snapshots = snapshots or []
        self.samples = samples or []
        self.moves = moves or []
        self.result = int(result)
        self.checkpoint_step = int(checkpoint_step)
        self.max_game_plies = int(max_game_plies)

    def trim(self, position_count):
        position_count = max(0, min(int(position_count), len(self.samples)))
        samples = self.samples[:position_count]
        snapshot_count = samples[-1].snapshot_index + 1 if samples else 0
        return GameRecord(
            snapshots=self.snapshots[:snapshot_count],
            samples=samples,
            moves=self.moves,
            result=self.result,
            checkpoint_step=self.checkpoint_step,
            max_game_plies=self.max_game_plies,
        )

    def materialize_state(self, sample_index, output=None):
        if output is None:
            output = np.empty((INPUT_PLANES, 8, 8), dtype=np.float16)
        sample = self.samples[sample_index]
        history_start = max(0, sample.snapshot_index - 7)
        snapshots = self.snapshots[history_start : sample.snapshot_index + 1]
        return encode_history(
            snapshots,
            sample.current_color,
            sample.ply,
            self.max_game_plies,
            sample.castling_mask,
            sample.halfmove_clock,
            output=output,
        )

    @property
    def memory_bytes(self):
        snapshot_bytes = sum(snapshot.bitboards.nbytes for snapshot in self.snapshots)
        target_bytes = 0
        for sample in self.samples:
            target_bytes += sample.legal_actions.nbytes + sample.visit_counts.nbytes + 24
        return snapshot_bytes + target_bytes


class InMemoryWindow:
    def __init__(self, target_positions):
        self.target_positions = int(target_positions)
        self.games = []
        self.position_count = 0
        self.cumulative_positions = np.asarray([], dtype=np.int64)

    @property
    def full(self):
        return self.position_count >= self.target_positions

    @property
    def game_count(self):
        return len(self.games)

    @property
    def memory_bytes(self):
        return sum(game.memory_bytes for game in self.games)

    def add_game(self, game):
        if not game.samples:
            return 0

        self.games.append(game)
        self.position_count += len(game.samples)
        self.cumulative_positions = np.append(
            self.cumulative_positions,
            self.position_count,
        )
        return len(game.samples)

    def locate(self, position_index):
        game_index = int(np.searchsorted(self.cumulative_positions, position_index, side="right"))
        previous = 0 if game_index == 0 else int(self.cumulative_positions[game_index - 1])
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
        visit_counts = np.zeros((batch_size, max_legal_moves), dtype=np.float32)
        legal_mask = np.zeros((batch_size, max_legal_moves), dtype=np.bool_)
        values = np.empty(batch_size, dtype=np.float32)
        policy_weights = np.empty(batch_size, dtype=np.float32)

        for index, sample in enumerate(samples):
            length = len(sample.legal_actions)
            legal_actions[index, :length] = sample.legal_actions
            visit_counts[index, :length] = sample.visit_counts
            legal_mask[index, :length] = True
            values[index] = sample.value
            policy_weights[index] = sample.policy_weight

        return (
            states,
            legal_actions,
            visit_counts,
            legal_mask,
            values,
            policy_weights,
        )
