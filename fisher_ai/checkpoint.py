"""Save and restore the single FisherAI training checkpoint."""

import os
from pathlib import Path

import torch


class CheckpointManager:
    """Manage atomic saves and restores of the latest checkpoint."""

    def __init__(self, directory="checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "latest.pt"
        self.pending_path = self.directory / "latest.pending.pt"

    def latest_path(self):
        """Return the current checkpoint path when it exists."""
        return self.path if self.path.exists() else None

    def ensure(self, model):
        """Create an initial checkpoint when none exists."""
        if not self.path.exists():
            self.save(model, 0, cumulative_fresh_positions=0)
        return self.path

    def save(
        self,
        model,
        step,
        optimizer=None,
        scaler=None,
        cumulative_fresh_positions=0,
    ):
        """Atomically save model, optimizer, and schedule state."""
        payload = {
            "model": model.state_dict(),
            "step": int(step),
            "cumulative_fresh_positions": int(cumulative_fresh_positions),
        }
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if scaler is not None:
            payload["scaler"] = scaler.state_dict()

        torch.save(payload, self.pending_path)
        os.replace(self.pending_path, self.path)
        return self.path

    def cumulative_fresh_positions(self, path=None):
        """Read the persisted fresh-position schedule counter."""
        path = Path(path) if path else self.latest_path()
        if path is None:
            return 0
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("cumulative_fresh_positions", 0))

    def load(
        self, model, path=None, optimizer=None, scaler=None, device="cpu"
    ):
        """Restore model and optional optimizer state from a checkpoint."""
        path = Path(path) if path else self.latest_path()
        if path is None:
            return 0

        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        return int(payload.get("step", 0))
