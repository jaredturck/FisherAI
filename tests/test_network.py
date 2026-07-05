import torch

from fisher_ai.network import FisherNetwork


def test_network_output_shapes_value_range_and_size():
    model = FisherNetwork()
    states = torch.randn(2, 119, 8, 8)

    policy, value = model(states)

    assert policy.shape == (2, 4672)
    assert value.shape == (2,)
    assert torch.all(value >= -1)
    assert torch.all(value <= 1)
    assert (
        sum(parameter.numel() for parameter in model.parameters()) == 3310234
    )
