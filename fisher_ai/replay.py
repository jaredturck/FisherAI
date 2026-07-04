import math
import pickle
from pathlib import Path

import lmdb
import numpy as np
import zstandard as zstd

INDEX_KEY = b"meta:index"
NEXT_ID_KEY = b"meta:next_id"
GENERATION_KEY = b"meta:generation"
TOTAL_POSITIONS_KEY = b"meta:total_positions"
SCHEMA_KEY = b"meta:schema"
SCHEMA_VERSION = 2
BINARY_PLANE_INDICES = list(range(113)) + list(range(114, 118))


class TrainingSample:
    def __init__(
        self,
        state,
        legal_actions,
        visit_counts,
        value=0.0,
        policy_weight=1.0,
    ):
        self.state = np.asarray(state, dtype=np.float16)
        self.legal_actions = np.asarray(legal_actions, dtype=np.uint16)
        self.visit_counts = np.asarray(visit_counts, dtype=np.uint16)
        self.value = float(value)
        self.policy_weight = float(policy_weight)

    def to_dict(self):
        binary_planes = self.state[BINARY_PLANE_INDICES] > 0.5
        packed_state = np.packbits(binary_planes.reshape(-1))
        scalar_planes = np.asarray(
            [self.state[113, 0, 0], self.state[118, 0, 0]],
            dtype=np.float16,
        )
        return {
            "packed_state": packed_state,
            "scalar_planes": scalar_planes,
            "legal_actions": self.legal_actions,
            "visit_counts": self.visit_counts,
            "value": self.value,
            "policy_weight": self.policy_weight,
        }

    @classmethod
    def from_dict(cls, data):
        bit_count = len(BINARY_PLANE_INDICES) * 64
        binary = np.unpackbits(data["packed_state"], count=bit_count)
        binary = binary.reshape(len(BINARY_PLANE_INDICES), 8, 8)
        state = np.zeros((119, 8, 8), dtype=np.float16)
        state[BINARY_PLANE_INDICES] = binary
        state[113].fill(data["scalar_planes"][0])
        state[118].fill(data["scalar_planes"][1])
        return cls(
            state,
            data["legal_actions"],
            data["visit_counts"],
            data["value"],
            policy_weight=data.get("policy_weight", 1.0),
        )


class GameRecord:
    def __init__(self, samples=None, moves=None, result=0, checkpoint_step=0):
        self.samples = samples or []
        self.moves = moves or []
        self.result = int(result)
        self.checkpoint_step = int(checkpoint_step)

    def to_dict(self):
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "moves": self.moves,
            "result": self.result,
            "checkpoint_step": self.checkpoint_step,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            samples=[TrainingSample.from_dict(sample) for sample in data["samples"]],
            moves=data.get("moves", []),
            result=data.get("result", 0),
            checkpoint_step=data.get("checkpoint_step", 0),
        )


class ReplayBuffer:
    def __init__(self, path, max_positions=2000000, map_size=1 << 40):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_positions = max_positions
        self.env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=True,
            create=True,
            lock=True,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        self.compressor = zstd.ZstdCompressor(level=1)
        self.decompressor = zstd.ZstdDecompressor()
        self.index = []
        self.cumulative_positions = np.asarray([], dtype=np.int64)
        self.total_positions_added = 0
        self.generation = -1
        self.ensure_schema()
        self.refresh(force=True)

    def ensure_schema(self):
        with self.env.begin(write=True) as transaction:
            raw_schema = transaction.get(SCHEMA_KEY)
            raw_index = transaction.get(INDEX_KEY)

            if raw_schema is None and raw_index is None:
                transaction.put(SCHEMA_KEY, str(SCHEMA_VERSION).encode())
                return

            schema = int(raw_schema.decode()) if raw_schema else 1
            if schema != SCHEMA_VERSION:
                raise RuntimeError(
                    "Replay schema is incompatible. Remove the existing replay LMDB "
                    "before using this Fisher AI version."
                )

    def set_index(self, index):
        self.index = index
        counts = np.asarray([position_count for _, position_count in index], dtype=np.int64)
        self.cumulative_positions = np.cumsum(counts)

    def refresh(self, force=False):
        with self.env.begin() as transaction:
            raw_generation = transaction.get(GENERATION_KEY)
            generation = int(raw_generation.decode()) if raw_generation else 0
            if generation == self.generation and not force:
                return
            raw_index = transaction.get(INDEX_KEY)
            raw_total_positions = transaction.get(TOTAL_POSITIONS_KEY)

        self.set_index(pickle.loads(raw_index) if raw_index else [])
        self.total_positions_added = (
            int(raw_total_positions.decode()) if raw_total_positions else self.position_count
        )
        self.generation = generation

    @property
    def game_count(self):
        return len(self.index)

    @property
    def position_count(self):
        if len(self.cumulative_positions) == 0:
            return 0
        return int(self.cumulative_positions[-1])

    def encode_game(self, game):
        payload = pickle.dumps(game.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
        return self.compressor.compress(payload)

    def add_game(self, game):
        return self.add_games([game])[0]

    def add_games(self, games):
        payloads = [self.encode_game(game) for game in games]
        new_ids = []

        with self.env.begin(write=True) as transaction:
            raw_index = transaction.get(INDEX_KEY)
            index = pickle.loads(raw_index) if raw_index else []
            raw_next_id = transaction.get(NEXT_ID_KEY)
            next_id = int(raw_next_id.decode()) if raw_next_id else 0
            raw_generation = transaction.get(GENERATION_KEY)
            generation = int(raw_generation.decode()) if raw_generation else 0
            raw_total_positions = transaction.get(TOTAL_POSITIONS_KEY)
            total_positions_added = (
                int(raw_total_positions.decode()) if raw_total_positions else 0
            )
            retained_positions = sum(position_count for _, position_count in index)

            for game, payload in zip(games, payloads, strict=True):
                game_id = next_id
                position_count = len(game.samples)
                key = f"game:{game_id:020d}".encode()
                transaction.put(key, payload)
                index.append((game_id, position_count))
                new_ids.append(game_id)
                next_id += 1
                retained_positions += position_count
                total_positions_added += position_count

            while retained_positions > self.max_positions and len(index) > 1:
                old_id, old_positions = index.pop(0)
                transaction.delete(f"game:{old_id:020d}".encode())
                retained_positions -= old_positions

            generation += 1
            transaction.put(INDEX_KEY, pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
            transaction.put(NEXT_ID_KEY, str(next_id).encode())
            transaction.put(GENERATION_KEY, str(generation).encode())
            transaction.put(TOTAL_POSITIONS_KEY, str(total_positions_added).encode())

        self.set_index(index)
        self.total_positions_added = total_positions_added
        self.generation = generation
        return new_ids

    def get_game(self, game_id, transaction=None):
        key = f"game:{game_id:020d}".encode()
        if transaction is None:
            with self.env.begin() as read_transaction:
                payload = read_transaction.get(key)
        else:
            payload = transaction.get(key)

        assert payload is not None
        data = pickle.loads(self.decompressor.decompress(payload))
        return GameRecord.from_dict(data)

    def sample(self, batch_size, rng=None, positions_per_game=8):
        self.refresh()
        assert self.index
        rng = rng or np.random.default_rng()
        positions_per_game = max(1, positions_per_game)
        group_count = math.ceil(batch_size / positions_per_game)
        counts = np.asarray([position_count for _, position_count in self.index])
        probabilities = counts / counts.sum()
        selected_games = rng.choice(len(self.index), size=group_count, p=probabilities)
        requests = {}
        remaining = batch_size

        for game_index in selected_games:
            game_id, position_count = self.index[int(game_index)]
            sample_count = min(positions_per_game, remaining)
            positions = rng.integers(0, position_count, size=sample_count)
            requests.setdefault(game_id, []).extend(int(position) for position in positions)
            remaining -= sample_count
            if remaining == 0:
                break

        samples = []
        with self.env.begin() as transaction:
            for game_id, position_indices in requests.items():
                game = self.get_game(game_id, transaction=transaction)
                samples.extend(game.samples[position_index] for position_index in position_indices)

        rng.shuffle(samples)
        return samples

    def close(self):
        self.env.close()
