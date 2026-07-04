import json
import os
from pathlib import Path

import torch

from fisher_ai.config import config_to_dict


class CheckpointManager:
    def __init__(self, directory, keep_recent=5, milestone_interval=10000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.latest_file = self.directory / "latest.json"
        self.keep_recent = keep_recent
        self.milestone_interval = milestone_interval

    def checkpoint_path(self, step):
        return self.directory / f"fisher_ai_{step:09d}.pt"

    def save(self, model, config, step, optimizer=None, scaler=None, extra=None):
        path = self.checkpoint_path(step)
        pending_path = path.with_suffix(".pending.pt")
        payload = {
            "model": model.state_dict(),
            "config": config_to_dict(config),
            "step": int(step),
            "extra": extra or {},
        }

        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()

        torch.save(payload, pending_path)
        os.replace(pending_path, path)
        latest = {"path": path.name, "step": int(step)}
        pending_file = self.directory / "latest.pending"
        pending_file.write_text(json.dumps(latest, indent=2) + "\n")
        os.replace(pending_file, self.latest_file)
        self.prune()
        return path

    def checkpoint_step(self, path):
        return int(path.stem.rsplit("_", 1)[1])

    def prune(self):
        candidates = sorted(self.directory.glob("fisher_ai_*.pt"))
        if len(candidates) <= self.keep_recent:
            return

        recent = set(candidates[-self.keep_recent :])
        for path in candidates:
            step = self.checkpoint_step(path)
            milestone = step == 0 or (
                self.milestone_interval > 0 and step % self.milestone_interval == 0
            )
            if path not in recent and not milestone:
                path.unlink()

    def latest_path(self):
        if self.latest_file.exists():
            data = json.loads(self.latest_file.read_text())
            path = self.directory / data["path"]
            if path.exists():
                return path

        candidates = sorted(self.directory.glob("fisher_ai_*.pt"))
        return candidates[-1] if candidates else None

    def load(self, model, path=None, optimizer=None, scaler=None, device="cpu"):
        path = Path(path) if path else self.latest_path()
        if path is None:
            return 0, {}

        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])

        return int(payload.get("step", 0)), payload.get("extra", {})
