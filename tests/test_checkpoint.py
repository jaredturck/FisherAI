import torch
import torch.nn as nn

from fisher_ai.checkpoint import CheckpointManager


def test_checkpoint_manager_overwrites_one_atomic_checkpoint(tmp_path):
    manager = CheckpointManager(tmp_path)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    manager.save(model, 4, optimizer=optimizer)
    manager.save(
        model,
        5,
        optimizer=optimizer,
        cumulative_fresh_positions=12345,
    )

    assert manager.latest_path() == tmp_path / "latest.pt"
    assert not (tmp_path / "latest.pending.pt").exists()
    assert list(tmp_path.glob("*.pt")) == [tmp_path / "latest.pt"]

    payload = torch.load(
        manager.latest_path(),
        map_location="cpu",
        weights_only=False,
    )
    assert set(payload) == {
        "model",
        "optimizer",
        "step",
        "cumulative_fresh_positions",
    }
    assert payload["step"] == 5
    assert payload["cumulative_fresh_positions"] == 12345
    assert manager.cumulative_fresh_positions() == 12345
