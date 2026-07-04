import csv

from fisher_ai.benchmark import benchmark_profiles, write_benchmark_reports


def make_result(
    configuration_id,
    moves_per_second,
    run_stage="sweep",
    actor_count=24,
    games_per_actor=6,
    pending_leaves=24,
    slots=8,
    target_batch=512,
    maximum_batch=1024,
    batch_wait_ms=2.0,
):
    return {
        "configuration_id": configuration_id,
        "run_stage": run_stage,
        "confirmation_rank": 1 if run_stage == "confirmation" else "",
        "actor_count": actor_count,
        "games_per_actor": games_per_actor,
        "pending_leaves": pending_leaves,
        "max_inflight_requests_per_actor": slots,
        "target_batch": target_batch,
        "maximum_batch": maximum_batch,
        "batch_wait_ms": batch_wait_ms,
        "elapsed_seconds": 15.0,
        "moves_per_second": moves_per_second,
        "evaluations_per_second": 8000.0,
        "positions_per_second": 20.0,
        "games_per_hour": 500.0,
        "completed_games": 2,
        "completed_positions": 300,
        "evaluations": 120000,
        "average_gpu_batch": 400.0,
        "median_gpu_batch": 420.0,
        "p95_gpu_batch": 700.0,
        "maximum_gpu_batch": 900,
        "average_queue_depth": 10.0,
        "maximum_queue_depth": 30,
        "average_replay_queue_depth": 0.0,
        "cpu_utilization": 95.0,
        "outstanding_requests": 12,
        "blocked_slot_waits": 0,
    }


def test_benchmark_profiles_cover_final_unique_sweep():
    profiles = benchmark_profiles()
    assert len(profiles) == 26
    assert profiles[0].profile_id == "baseline"
    assert len({profile.profile_id for profile in profiles}) == len(profiles)

    settings = {
        (
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
    assert {profile.games_per_actor for profile in profiles} >= {4, 6, 8, 10, 12, 14}
    assert {profile.pending_leaves for profile in profiles} >= {8, 16, 24, 32}
    assert {profile.target_batch for profile in profiles} >= {256, 384, 512, 768, 1024}
    assert {profile.batch_wait_ms for profile in profiles} >= {0.5, 1.0, 2.0, 4.0}
    assert {profile.actor_count for profile in profiles} >= {24, 28, 32}
    assert all(
        profile.max_inflight_requests_per_actor >= profile.games_per_actor
        for profile in profiles
    )

    combined = {
        (profile.games_per_actor, profile.pending_leaves)
        for profile in profiles
        if profile.target_batch == 512
        and profile.maximum_batch == 1024
        and profile.batch_wait_ms == 2.0
    }
    assert combined >= {
        (4, 24),
        (6, 24),
        (6, 32),
        (8, 24),
        (8, 32),
        (10, 24),
        (12, 24),
        (14, 24),
    }


def test_benchmark_profiles_follow_custom_base_actor_count():
    profiles = benchmark_profiles(actor_count=20)
    actor_counts = {profile.actor_count for profile in profiles}
    assert actor_counts >= {20, 24, 28}
    assert any(profile.profile_id == "actors_24" for profile in profiles)
    assert any(profile.profile_id == "actors_28" for profile in profiles)


def test_benchmark_reports_separate_sweep_and_confirmation(tmp_path):
    results = [
        make_result(
            "baseline",
            100.0,
            games_per_actor=10,
            pending_leaves=16,
            slots=10,
        ),
        make_result("recommended_6x24", 110.0),
        make_result(
            "baseline",
            108.0,
            run_stage="confirmation",
            games_per_actor=10,
            pending_leaves=16,
            slots=10,
        ),
        make_result(
            "recommended_6x24",
            105.0,
            run_stage="confirmation",
        ),
    ]
    metadata = {
        "generated_at": "2026-07-04T12:00:00",
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
    assert rows[0]["actor_count"] == "24"
    assert rows[0]["max_inflight_requests_per_actor"] == "10"

    summary = markdown_path.read_text()
    assert "Initial sweep results" in summary
    assert "Confirmed top configurations" in summary
    assert "The highest-throughput confirmed result was `baseline`" in summary
    assert '"actor_processes": 24' in summary
    assert '"max_inflight_requests_per_actor": 10' in summary
    assert "does not modify" in summary
