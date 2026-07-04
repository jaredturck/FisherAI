import torch

from fisher_ai.config import FisherConfig, NetworkConfig
from fisher_ai.network import FisherNetwork


def test_network_output_shapes_and_value_range():
    config = NetworkConfig(
        channels=16,
        residual_blocks=2,
        squeeze_excitation_channels=4,
        policy_channels=16,
        value_channels=2,
        value_hidden=32,
    )
    model = FisherNetwork(config)
    states = torch.randn(4, 119, 8, 8)

    policy, value = model(states)

    assert policy.shape == (4, 4672)
    assert value.shape == (4,)
    assert torch.all(value >= -1)
    assert torch.all(value <= 1)


def test_default_workstation_model_size_is_stable():
    model = FisherNetwork(FisherConfig().network)

    assert model.parameter_count() == 3310234
