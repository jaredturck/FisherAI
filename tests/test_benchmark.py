import csv

from fisher_ai.benchmark import run_training_benchmark, write_reports
from fisher_ai.benchmark_metrics import distribution_row, percentile
from fisher_ai.trainer import AlphaZeroTrainer
from tests.test_trainer import TinyPolicyValueModel, make_window


def benchmark_inputs():
    generation = {
        "elapsed_seconds": 10.0,
        "positions_per_second": 500.0,
        "evaluations_per_second": 8000.0,
        "average_inference_batch": 400.0,
        "max_batch": 700,
        "games": 25,
        "evaluations": 80000,
        "batches": 200,
    }
    training = {
        "elapsed_seconds": 2.0,
        "positions": 15000,
        "positions_per_second": 7500.0,
        "optimizer_steps": 9,
        "epochs": 3,
        "loss": 2.0,
        "policy_loss": 1.5,
        "value_loss": 0.5,
        "learning_rate": 0.05,
        "fresh_positions_per_epoch": 5000,
        "replay_positions_per_epoch": 0,
    }
    return generation, training


def test_percentile_interpolates_sorted_values():
    assert percentile([4, 1, 3, 2], 50) == 2.5
    assert percentile([9], 95) == 9
    assert percentile([], 50) == ""


def test_benchmark_writes_long_format_metrics_and_summary(tmp_path):
    generation, training = benchmark_inputs()
    extra_rows = [
        distribution_row(
            "blocking",
            "generation",
            "game_queue",
            "wait_seconds",
            [0.1, 0.2, 0.3],
            "seconds",
            share_base=10.0,
        )
    ]

    csv_path, markdown_path = write_reports(
        tmp_path,
        5000,
        generation,
        training,
        extra_rows,
    )
    write_reports(
        tmp_path,
        5000,
        generation,
        training,
        extra_rows,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "benchmark_results.csv",
        "benchmark_summary.md",
    ]
    with csv_path.open() as file:
        rows = list(csv.DictReader(file))

    assert len(rows) > 20
    assert rows[0]["category"] == "phase"
    assert rows[0]["component"] == "generation"
    blocking = next(
        row
        for row in rows
        if row["category"] == "blocking" and row["component"] == "game_queue"
    )
    assert blocking["count"] == "3"
    assert blocking["p95"] != ""

    summary = markdown_path.read_text()
    assert "CSV metric rows:" in summary
    assert "Blocking and waiting" in summary
    assert "Training epochs: 3" in summary
    assert "Games completed: 25" in summary
    assert "Production training modules are not instrumented" in summary


def test_training_benchmark_wraps_the_production_training_path():
    trainer = AlphaZeroTrainer(
        TinyPolicyValueModel(),
        batch_size=3,
        device="cpu",
    )
    original_train_batch = trainer.train_batch

    metrics, rows = run_training_benchmark(trainer, make_window(7))

    materialization = next(
        row for row in rows if row["component"] == "batch_materialization"
    )
    optimizer = next(
        row for row in rows if row["component"] == "optimizer_batch"
    )
    assert metrics["optimizer_steps"] == 9
    assert materialization["count"] == 9
    assert optimizer["count"] == 9
    assert trainer.train_batch == original_train_batch
