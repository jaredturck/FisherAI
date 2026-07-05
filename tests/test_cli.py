import argparse
import sys
from types import ModuleType

import pytest

from fisher_ai import cli


def test_cli_only_exposes_required_commands():
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )

    assert set(subparsers.choices) == {"workstation", "benchmark", "gui"}
    assert parser.prog == "python -m fisher_ai"
    assert parser.parse_args(["workstation"]).config == "fisher_config.json"
    assert parser.parse_args(["benchmark"]).config == "fisher_config.json"
    assert parser.parse_args(["gui"]).handler is cli.command_gui

    for command in ("init", "self-play", "learn", "train", "evaluate", "uci"):
        with pytest.raises(SystemExit):
            parser.parse_args([command])


def test_workstation_uses_configured_pool_and_internal_learner(monkeypatch):
    calls = {}

    class Manager:
        def latest_path(self):
            return "checkpoint.pt"

    class Pool:
        def __init__(self, **kwargs):
            calls["pool_kwargs"] = kwargs

        def start(self):
            calls["started"] = True

        def monitor(self, external_processes):
            calls["monitored"] = external_processes

        def stop(self):
            calls["stopped"] = True

    class Process:
        returncode = None

        def poll(self):
            return 0

    process = Process()

    monkeypatch.setattr(cli, "load_config", lambda path: object())
    monkeypatch.setattr(cli, "build_model", lambda config: object())
    monkeypatch.setattr(cli, "build_checkpoint_manager", lambda config: Manager())
    monkeypatch.setattr(cli, "DistributedSelfPlayPool", Pool)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda command: calls.update(command=command) or process,
    )

    args = argparse.Namespace(config="custom.json")
    cli.command_workstation(args)

    assert calls["pool_kwargs"] == {"config_path": "custom.json"}
    assert calls["command"] == [
        sys.executable,
        "-m",
        "fisher_ai.learner_worker",
        "custom.json",
    ]
    assert calls["monitored"] == [process]
    assert calls["started"]
    assert calls["stopped"]


def test_gui_command_launches_gui(monkeypatch):
    launched = []
    gui_module = ModuleType("gui.main")
    gui_module.main = lambda: launched.append(True)
    monkeypatch.setitem(sys.modules, "gui.main", gui_module)

    cli.command_gui(argparse.Namespace())

    assert launched == [True]
