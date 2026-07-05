import csv

from fisher_ai.benchmark import write_reports


def test_benchmark_overwrites_current_generation_and_training_reports(
    tmp_path,
):
    generation = {
        "elapsed_seconds": 10.0,
        "positions_per_second": 500.0,
        "evaluations_per_second": 8000.0,
        "average_inference_batch": 400.0,
        "max_batch": 700,
        "games": 25,
    }
    training = {
        "elapsed_seconds": 2.0,
        "positions": 15000,
        "positions_per_second": 7500.0,
        "optimizer_steps": 9,
        "epochs": 3,
    }

    csv_path, markdown_path = write_reports(
        tmp_path,
        5000,
        generation,
        training,
    )
    write_reports(tmp_path, 5000, generation, training)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "benchmark_results.csv",
        "benchmark_summary.md",
    ]
    with csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert [row["phase"] for row in rows] == ["generation", "training"]
    assert rows[1]["positions"] == "15000"
    summary = markdown_path.read_text()
    assert "Training epochs: 3" in summary
    assert "Games completed: 25" in summary
    assert "Peak compact window memory" not in summary
