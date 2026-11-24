"""Small non-causal temporal convolution classifier, per model_contract.json.

Five class logits. They are word identity only - never a correctness verdict and never a quality
score. Padding is masked at every stage so a longer batch mate cannot change another clip's
output; tests/test_pilot.py checks that property rather than trusting it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import CLASSES, TOTAL_CHANNELS


class ResidualBlock(nn.Module):
    """Dilated, non-causal, symmetric padding so the block keeps the time length."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        # zero the padded positions before every convolution, so padding never leaks inward
        y = self.drop(self.act(self.norm1(self.conv1(h * mask)))) * mask
        y = self.drop(self.act(self.norm2(self.conv2(y * mask)))) * mask
        return (h + y) * mask


class SignTCN(nn.Module):
    def __init__(self, in_channels: int = TOTAL_CHANNELS, width: int = 32,
                 dilations=(1, 2, 4, 8), kernel_size: int = 3, dropout: float = 0.1,
                 n_classes: int = len(CLASSES)):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [ResidualBlock(width, d, kernel_size, dropout) for d in dilations])
        self.head = nn.Linear(width, n_classes)

    def forward(self, x, pad_mask):
        """x [B,T,C], pad_mask [B,T] with 1 on real frames."""
        h = x.transpose(1, 2)                       # [B,C,T]
        m = pad_mask.unsqueeze(1)                   # [B,1,T]
        h = self.proj(h * m) * m
        for blk in self.blocks:
            h = blk(h, m)
        # masked mean over time: padded positions contribute nothing to value or divisor
        denom = pad_mask.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        pooled = (h * m).sum(dim=2) / denom
        return self.head(pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
