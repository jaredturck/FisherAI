import csv

from fisher_ai.benchmark import benchmark_profiles, write_benchmark_reports
from fisher_ai.distributed import BATCHED_SCHEDULER, THREADED_SCHEDULER


def make_result(
    configuration_id,
    moves_per_second,
    run_stage="sweep",
    scheduler=BATCHED_SCHEDULER,
    actor_count=24,
    games_per_actor=6,
    pending_leaves=8,
    slots=1,
    target_batch=512,
    maximum_batch=1024,
    batch_wait_ms=2.0,
):
    return {
        "configuration_id": configuration_id,
        "run_stage": run_stage,
        "confirmation_rank": 1 if run_stage == "confirmation" else "",
        "scheduler": scheduler,
        "actor_count": actor_count,
        "games_per_actor": games_per_actor,
        "pending_leaves": pending_leaves,
        "max_inflight_requests_per_actor": slots,
        "target_batch": target_batch,
        "maximum_batch": maximum_batch,
        "batch_wait_ms": batch_wait_ms,
        "request_capacity": games_per_actor * pending_leaves,
        "elapsed_seconds": 15.0,
        "moves_per_second": moves_per_second,
        "evaluations_per_second": 8000.0,
        "positions_per_second": 20.0,
        "games_per_hour": 500.0,
        "completed_games": 2,
        "completed_positions": 300,
        "evaluations": 120000,
        "inference_requests": 1000,
        "average_actor_request": 48.0,
        "average_gpu_batch": 400.0,
        "median_gpu_batch": 420.0,
        "p95_gpu_batch": 700.0,
        "maximum_gpu_batch": 900,
        "average_queue_depth": 10.0,
        "maximum_queue_depth": 30,
        "average_replay_queue_depth": 0.0,
        "cpu_utilization": 80.0,
        "actor_work_percent": 72.0,
        "inference_wait_percent": 24.0,
        "queue_wait_percent": 1.0,
        "replay_wait_percent": 0.0,
        "outstanding_requests": 12,
        "blocked_slot_waits": 0,
    }


def test_benchmark_profiles_compare_schedulers_and_batched_candidates():
    profiles = benchmark_profiles()
    assert len(profiles) == 15
    assert profiles[0].profile_id == "baseline"
    assert profiles[0].scheduler == BATCHED_SCHEDULER
    assert len({profile.profile_id for profile in profiles}) == len(profiles)

    settings = {
        (
            profile.scheduler,
            profile.actor_count,
            profile.games_per_actor,
            profile.pending_leaves,
            profile.max_inflight_requests_per_actor,
            profile.target_batch,
            profile.maximum_batch,
            profile.batch_wait_ms,
        )
        for profile in profiles
    }
    assert len(settings) == len(profiles)
    assert {profile.scheduler for profile in profiles} == {
        BATCHED_SCHEDULER,
        THREADED_SCHEDULER,
    }

    batched = {
        (profile.games_per_actor, profile.pending_leaves)
        for profile in profiles
        if profile.scheduler == BATCHED_SCHEDULER
        and profile.target_batch == 512
        and profile.maximum_batch == 1024
        and profile.batch_wait_ms == 2.0
        and profile.actor_count == 24
    }
    assert batched >= {
        (4, 8),
        (4, 16),
        (4, 24),
        (4, 32),
        (6, 8),
        (6, 16),
        (6, 24),
        (6, 32),
        (8, 16),
        (8, 24),
    }
    assert all(
        profile.max_inflight_requests_per_actor == 1
        for profile in profiles
        if profile.scheduler == BATCHED_SCHEDULER
    )
    assert all(
        profile.max_inflight_requests_per_actor >= profile.games_per_actor
        for profile in profiles
        if profile.scheduler == THREADED_SCHEDULER
    )


def test_benchmark_profiles_follow_custom_base_actor_count():
    profiles = benchmark_profiles(actor_count=20)
    actor_counts = {profile.actor_count for profile in profiles}
    assert actor_counts >= {18, 20}
    assert any(profile.profile_id == "batched_18actors_6x24" for profile in profiles)
    profile = next(
        profile
        for profile in profiles
        if profile.profile_id == "batched_18actors_6x24"
    )
    assert profile.actor_count == 18


def test_benchmark_reports_separate_sweep_and_confirmation(tmp_path):
    results = [
        make_result("baseline", 100.0),
        make_result(
            "threaded_current_4x24",
            80.0,
            scheduler=THREADED_SCHEDULER,
            games_per_actor=4,
            pending_leaves=24,
            slots=8,
        ),
        make_result(
            "baseline",
            108.0,
            run_stage="confirmation",
        ),
        make_result(
            "threaded_current_4x24",
            82.0,
            run_stage="confirmation",
            scheduler=THREADED_SCHEDULER,
            games_per_actor=4,
            pending_leaves=24,
            slots=8,
        ),
    ]
    metadata = {
        "generated_at": "2026-07-05T12:00:00",
        "actor_count": 24,
        "sweep_profile_count": 2,
        "warmup_seconds": 5.0,
        "measure_seconds": 15.0,
        "confirmation_seconds": 30.0,
        "confirmation_run_count": 2,
        "devices": ["cuda:0", "cuda:1"],
    }

    csv_path, markdown_path = write_benchmark_reports(results, tmp_path, metadata)

    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert rows[2]["run_stage"] == "confirmation"
    assert rows[0]["scheduler"] == BATCHED_SCHEDULER
    assert rows[0]["request_capacity"] == "48"

    summary = markdown_path.read_text()
    assert "Initial sweep results" in summary
    assert "Confirmed top configurations" in summary
    assert "threaded regression" in summary
    assert "The highest-throughput confirmed result was `baseline`" in summary
    assert '"self_play_scheduler": "batched"' in summary
    assert '"actor_processes": 24' in summary
    assert '"inference_request_batch_size": 0' in summary
    assert "does not modify" in summary
