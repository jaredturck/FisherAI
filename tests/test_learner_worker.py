from fisher_ai import learner_worker


def test_checkpoint_is_due_only_after_elapsed_interval_and_new_steps(monkeypatch):
    monkeypatch.setattr(learner_worker, "CHECKPOINT_INTERVAL_SECONDS", 600.0)

    assert not learner_worker.checkpoint_due(100.0, 699.9, 10, 11)
    assert learner_worker.checkpoint_due(100.0, 700.0, 10, 11)
    assert not learner_worker.checkpoint_due(100.0, 700.0, 10, 10)


def test_eta_uses_wall_clock_session_progress():
    eta = learner_worker.estimate_remaining_time(
        start_step=100,
        current_step=200,
        target_step=500,
        elapsed_seconds=60.0,
    )

    assert eta == 180.0
    assert learner_worker.estimate_remaining_time(100, 500, 500, 60.0) == 0.0
