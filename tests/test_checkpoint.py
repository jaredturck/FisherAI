import json

import torch.nn as nn

from fisher_ai.checkpoint import CheckpointManager
from fisher_ai.config import FisherConfig


def test_checkpoint_manager_keeps_only_newest_files(tmp_path):
    manager = CheckpointManager(tmp_path, keep_recent=3)
    model = nn.Linear(1, 1)
    config = FisherConfig()

    for step in range(5):
        manager.save(model, config, step)

    checkpoints = sorted(path.name for path in tmp_path.glob("fisher_ai_*.pt"))
    latest = json.loads((tmp_path / "latest.json").read_text())

    assert checkpoints == [
        "fisher_ai_000000002.pt",
        "fisher_ai_000000003.pt",
        "fisher_ai_000000004.pt",
    ]
    assert latest == {"path": "fisher_ai_000000004.pt", "step": 4}
