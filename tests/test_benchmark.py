import csv

from fisher_ai.benchmark import write_reports


def test_benchmark_reports_generation_and_training_phases(tmp_path):
    generation = {
        "elapsed_seconds": 10.0,
        "positions_per_second": 500.0,
        "evaluations_per_second": 8000.0,
        "average_inference_batch": 400.0,
        "max_batch": 700,
        "memory_bytes": 1024 * 1024,
    }
    training = {
        "elapsed_seconds": 2.0,
        "positions_per_second": 2500.0,
        "optimizer_steps": 3,
        "loss": 1.25,
    }

    csv_path, markdown_path = write_reports(
        tmp_path,
        5000,
        generation,
        training,
    )

    with csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert [row["phase"] for row in rows] == ["generation", "training"]
    assert rows[0]["positions"] == "5000"
    assert rows[1]["optimizer_steps"] == "3"

    summary = markdown_path.read_text()
    assert "Fisher AI Phased Benchmark" in summary
    assert "Generation" in summary
    assert "Training" in summary
    assert "Peak compact window memory" in summary
