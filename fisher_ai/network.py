import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        hidden_channels = min(channels, hidden_channels)
        self.linear1 = nn.Linear(channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, channels * 2)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        scale_bias = self.linear2(F.relu(self.linear1(pooled)))
        scale, bias = scale_bias.chunk(2, dim=1)
        scale = (2.0 * torch.sigmoid(scale)).unsqueeze(2).unsqueeze(3)
        bias = bias.unsqueeze(2).unsqueeze(3)
        return x * scale + bias


class ResidualBlock(nn.Module):
    def __init__(self, channels, squeeze_excitation_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.squeeze_excitation = SqueezeExcitation(
            channels,
            squeeze_excitation_channels,
        )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.squeeze_excitation(x)
        return F.relu(x + residual)


class FisherNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config.channels

        self.stem_conv = nn.Conv2d(config.input_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.residual_tower = nn.Sequential(
            *[
                ResidualBlock(channels, config.squeeze_excitation_channels)
                for _ in range(config.residual_blocks)
            ]
        )

        self.policy_conv1 = nn.Conv2d(channels, config.policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_conv2 = nn.Conv2d(config.policy_channels, 73, 1)

        self.value_conv = nn.Conv2d(channels, config.value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_linear1 = nn.Linear(config.value_channels * 8 * 8, config.value_hidden)
        self.value_linear2 = nn.Linear(config.value_hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):
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
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = self.residual_tower(x)

        policy = F.relu(self.policy_bn(self.policy_conv1(x)))
        policy = self.policy_conv2(policy).flatten(1)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = F.relu(self.value_linear1(value))
        value = torch.tanh(self.value_linear2(value)).squeeze(1)

        return policy, value

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())
