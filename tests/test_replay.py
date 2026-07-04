import numpy as np

from fisher_ai.replay import GameRecord, ReplayBuffer, TrainingSample


def make_game(value, positions=3, policy_weight=1.0):
    samples = []
    for index in range(positions):
        state = np.zeros((119, 8, 8), dtype=np.float16)
        state[index, index, index] = 1
        state[112].fill(index % 2)
        state[113].fill(index / 10)
        state[114].fill(1)
        state[118].fill(index / 20)
        samples.append(
            TrainingSample(
                state,
                [1, 2],
                [3, 1],
                value=value,
                policy_weight=policy_weight,
            )
        )
    return GameRecord(samples=samples, moves=["e2e4"], result=value)


def test_replay_round_trip_and_grouped_sampling(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay.lmdb", max_positions=10)
    first_id = replay.add_game(make_game(1, policy_weight=0))
    replay.add_game(make_game(-1, positions=2))

    loaded = replay.get_game(first_id)
    sampled = replay.sample(12, rng=np.random.default_rng(1), positions_per_game=4)

    assert loaded.result == 1
    assert len(loaded.samples) == 3
    assert loaded.samples[0].policy_weight == 0
    assert np.array_equal(loaded.samples[2].state, make_game(1).samples[2].state)
    assert len(sampled) == 12
    assert replay.game_count == 2
    assert replay.position_count == 5
    assert replay.total_positions_added == 5
    replay.close()


def test_replay_discards_old_games_by_position_count(tmp_path):
    replay = ReplayBuffer(tmp_path / "replay.lmdb", max_positions=5)
    replay.add_game(make_game(1, positions=3))
    replay.add_game(make_game(0, positions=3))
    replay.add_game(make_game(-1, positions=2))

    assert replay.game_count == 2
    assert replay.position_count == 5
    assert replay.total_positions_added == 8
    replay.close()
