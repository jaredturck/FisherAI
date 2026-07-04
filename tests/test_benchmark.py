import csv

from fisher_ai.benchmark import benchmark_profiles, write_benchmark_reports


def test_benchmark_profiles_cover_expected_sweeps():
    profiles = benchmark_profiles()
    assert len(profiles) == 14
    assert profiles[0].profile_id == "baseline"
    assert {profile.games_per_actor for profile in profiles} >= {6, 8, 10, 12, 14}
    assert {profile.pending_leaves for profile in profiles} >= {8, 16, 24, 32}
    assert {profile.target_batch for profile in profiles} >= {256, 512, 768, 1024}


def test_benchmark_reports_write_csv_and_markdown(tmp_path):
    results = [
        {
            "configuration_id": "baseline",
            "games_per_actor": 10,
            "pending_leaves": 16,
            "target_batch": 512,
            "maximum_batch": 1024,
            "batch_wait_ms": 2.0,
            "moves_per_second": 100.0,
            "evaluations_per_second": 8000.0,
            "positions_per_second": 20.0,
            "games_per_hour": 500.0,
            "average_gpu_batch": 400.0,
            "median_gpu_batch": 420.0,
            "p95_gpu_batch": 700.0,
            "cpu_utilization": 95.0,
        },
        {
            "configuration_id": "games_14",
            "games_per_actor": 14,
            "pending_leaves": 16,
            "target_batch": 512,
            "maximum_batch": 1024,
            "batch_wait_ms": 2.0,
            "moves_per_second": 110.0,
            "evaluations_per_second": 8500.0,
            "positions_per_second": 22.0,
            "games_per_hour": 550.0,
            "average_gpu_batch": 430.0,
            "median_gpu_batch": 450.0,
            "p95_gpu_batch": 750.0,
            "cpu_utilization": 98.0,
        },
    ]
    metadata = {
        "generated_at": "2026-07-04T12:00:00",
        "actor_count": 24,
        "warmup_seconds": 5.0,
        "measure_seconds": 15.0,
        "devices": ["cuda:0", "cuda:1"],
    }

    csv_path, markdown_path = write_benchmark_reports(results, tmp_path, metadata)

    with csv_path.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[1]["configuration_id"] == "games_14"

    summary = markdown_path.read_text()
    assert "games_14" in summary
    assert "Suggested manual settings" in summary
    assert "does not modify" in summary
