"""Small offline, class-independent coordinate residual refiner.

The RGB encoder is frozen and outside this module. B2 and B3 instantiate the
same parameters; B2 substitutes zeros for its 576-D per-hand RGB features.
No class identifier, target word, reference gesture, or hidden coordinate is
an argument. Outputs are estimates, never promoted to observed landmarks.
"""
from __future__ import annotations

import torch
from torch import nn


POSE_DIM = 575


class TemporalRefiner(nn.Module):
    def __init__(self, *, rgb_dim=576, pose_projection=64, rgb_projection=32,
                 hidden=48, use_rgb=True):
        super().__init__()
        self.use_rgb = bool(use_rgb)
        self.rgb_dim = int(rgb_dim)
        self.pose = nn.Sequential(nn.Linear(POSE_DIM, pose_projection), nn.ReLU())
        self.rgb = nn.Sequential(nn.Linear(rgb_dim, rgb_projection), nn.ReLU())
        self.temporal = nn.GRU(pose_projection + rgb_projection, hidden,
                               batch_first=True, bidirectional=True)
        self.residual = nn.Linear(2 * hidden, 42)
        # Both conditions start at the same transparent interpolation control.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, pose, rgb, interpolation):
        if pose.ndim != 3 or pose.shape[-1] != POSE_DIM:
            raise ValueError("Expected [batch, time, 575] permitted pose features")
        if rgb.shape[:2] != pose.shape[:2] or rgb.shape[-1] != self.rgb_dim:
            raise ValueError("RGB feature clock/shape mismatch")
        if interpolation.shape != (*pose.shape[:2], 42):
            raise ValueError("Expected [batch, time, 42] interpolation")
        if not self.use_rgb:
            rgb = torch.zeros_like(rgb)
        features = torch.cat((self.pose(pose), self.rgb(rgb)), dim=-1)
        sequence, _ = self.temporal(features)
        return interpolation + self.residual(sequence)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
