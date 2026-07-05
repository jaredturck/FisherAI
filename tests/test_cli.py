import argparse
import sys
from types import ModuleType

import pytest

from fisher_ai import cli


def test_cli_only_exposes_required_commands():
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {"train", "benchmark", "gui"}
    assert parser.parse_args(["train"]).config == "fisher_config.json"
    assert parser.parse_args(["train"]).iterations is None
    assert parser.parse_args(["benchmark"]).positions is None

    for command in ("workstation", "evaluate", "uci", "learn"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_train_runs_pipeline(monkeypatch):
    calls = {}

    class Pipeline:
        def __init__(self, config_path):
            calls["config_path"] = config_path

        def run(self, iterations=None):
            calls["iterations"] = iterations

    monkeypatch.setattr(cli, "TrainingPipeline", Pipeline)
    cli.command_train(argparse.Namespace(config="custom.json", iterations=3))

    assert calls == {"config_path": "custom.json", "iterations": 3}


def test_benchmark_command_reports_both_phases(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_benchmark",
        lambda **kwargs: (
            {
                "generation": {"positions_per_second": 12.5},
                "training": {"positions_per_second": 25.0},
            },
            "results.csv",
            "summary.md",
        ),
    )

    cli.command_benchmark(
        argparse.Namespace(config="custom.json", positions=100)
    )
    output = capsys.readouterr().out

    assert "Generation: 12.50 positions/s" in output
    assert "Training: 25.00 positions/s" in output


def test_gui_command_launches_gui(monkeypatch):
    launched = []
    gui_module = ModuleType("gui.main")
    gui_module.main = lambda: launched.append(True)
    monkeypatch.setitem(sys.modules, "gui.main", gui_module)

    cli.command_gui(argparse.Namespace())

    assert launched == [True]
