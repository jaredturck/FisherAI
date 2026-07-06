"""Define the FisherAI policy and value neural network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_PLANES = 119
CHANNELS = 128
RESIDUAL_BLOCKS = 10
SQUEEZE_EXCITATION_CHANNELS = 32
POLICY_CHANNELS = 128
VALUE_CHANNELS = 8
VALUE_HIDDEN = 128


class SqueezeExcitation(nn.Module):
    """Reweight residual channels with learned global context."""

    def __init__(self, channels, hidden_channels):
        super().__init__()
        hidden_channels = min(channels, hidden_channels)
        self.linear1 = nn.Linear(channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, channels * 2)

    def forward(self, x):
        """Apply learned channel-wise excitation to a feature map."""
        pooled = x.mean(dim=(2, 3))
        scale_bias = self.linear2(F.relu(self.linear1(pooled)))
        scale, bias = scale_bias.chunk(2, dim=1)
        scale = (2.0 * torch.sigmoid(scale)).unsqueeze(2).unsqueeze(3)
        bias = bias.unsqueeze(2).unsqueeze(3)
        return x * scale + bias


class ResidualBlock(nn.Module):
    """Apply one residual convolutional block with channel attention."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.squeeze_excitation = SqueezeExcitation(
            channels,
            SQUEEZE_EXCITATION_CHANNELS,
        )

    def forward(self, x):
        """Apply the residual block while preserving its shortcut."""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.squeeze_excitation(x)
        return F.relu(x + residual)


class FisherNetwork(nn.Module):
    """Predict policy logits and position value from encoded states."""

    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(
            INPUT_PLANES,
            CHANNELS,
            3,
            padding=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(CHANNELS)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(CHANNELS) for _ in range(RESIDUAL_BLOCKS)]
        )

        self.policy_conv1 = nn.Conv2d(
            CHANNELS,
            POLICY_CHANNELS,
            1,
            bias=False,
        )
        self.policy_bn = nn.BatchNorm2d(POLICY_CHANNELS)
        self.policy_conv2 = nn.Conv2d(POLICY_CHANNELS, 73, 1)

        self.value_conv = nn.Conv2d(
            CHANNELS,
            VALUE_CHANNELS,
            1,
            bias=False,
        )
        self.value_bn = nn.BatchNorm2d(VALUE_CHANNELS)
        self.value_linear1 = nn.Linear(VALUE_CHANNELS * 8 * 8, VALUE_HIDDEN)
        self.value_linear2 = nn.Linear(VALUE_HIDDEN, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize convolutional and linear network parameters."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for block in self.residual_tower:
            nn.init.zeros_(block.bn2.weight)
            nn.init.zeros_(block.squeeze_excitation.linear2.weight)
            nn.init.zeros_(block.squeeze_excitation.linear2.bias)

        nn.init.normal_(self.policy_conv2.weight, std=0.01)
        nn.init.zeros_(self.policy_conv2.bias)
        nn.init.normal_(self.value_linear2.weight, std=0.01)
        nn.init.zeros_(self.value_linear2.bias)

    def forward(self, x):
        """Return policy logits and value predictions for a state batch."""
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.residual_tower(x)

        policy = F.relu(self.policy_bn(self.policy_conv1(x)))
        policy = self.policy_conv2(policy).flatten(1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_linear1(value))
        value = torch.tanh(self.value_linear2(value)).squeeze(1)
        return policy, value
