"""Cycle013 corrected zero-residual adapter (Up=0, alpha=1)."""
import torch
from torch import nn


class MemoryDualScaleResidualAdapter(nn.Module):
    def __init__(self, channels=256, bottleneck=16):
        super().__init__()
        self.down = nn.Conv2d(2 * channels + 1, bottleneck, 1)
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.alpha = nn.Parameter(torch.ones(()))

    def forward(self, global_features, local_features_mapped, gate):
        residual = self.up(torch.nn.functional.gelu(
            self.down(torch.cat((global_features, local_features_mapped, gate), dim=1))))
        return global_features + self.alpha * gate * residual
