"""Candidate TCN whose normalisation statistics exclude padded frames.

Why this exists
  src/pilot/model.py masks padding around every convolution and pools with a masked mean, and
  tests/test_pilot.py checks that a longer batch mate cannot change another clip's output. That
  check runs in eval mode, where BatchNorm uses its stored running statistics, so it passes while
  the defect is live.

  Independent audit, dropout set to 0, same weights, optimizer step 0, one batch [B=2, T=31, C=280]
  compared against the same valid input with 62 masked-zero frames appended:
      eval-mode  max output difference   4.77e-07
      train-mode max output difference   7.3410
      BatchNorm running_mean difference  0.12037
  In train mode nn.BatchNorm1d reduces over (batch, time) including the padded columns, so the
  batch mean and variance - and the running statistics written from them - depend on how much
  padding happened to be in the batch.

  This is a structural counterexample with dropout disabled. It is NOT a claim that this caused the
  알다 transfer failure, and it is not a new performance test.

What this module changes, and what it does not
  Same blocks, width, kernel, dilations, activation, residual structure and parameter count as
  model.SignTCN. Only the normalisation reduction changes: MaskedBatchNorm1d computes the channel
  mean and variance over positions where pad_mask == 1, across batch and time together - the same
  axes nn.BatchNorm1d uses, minus the padding. Inference uses the stored running statistics, as
  before.

  Parameter and buffer names match model.SignTCN, so a checkpoint saved from that class loads here.
  That is a COMPATIBILITY property only: weights carry over, and the running statistics carried
  over with them were still estimated with padding mixed in. Loading does not correct them. Only
  retraining, or re-estimating the statistics, would - and neither is done here.

Degenerate batches
  valid positions == 0  the block cannot form a statistic. Output is zeroed at every position and
                        running statistics are left untouched, so an all-padding batch never writes
                        a value and never produces NaN.
  valid positions == 1  mean is that single value and the biased variance is 0, so the normalised
                        output is 0 there. running_mean is updated; running_var is NOT, because the
                        unbiased estimate needs at least two samples. Both cases are reported by
                        `last_valid_count` for tests to read.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .data import CLASSES, TOTAL_CHANNELS

MODEL_TYPE = "wis_masked_tcn"
MODEL_VERSION = "masked_v1"


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d over [B,C,T] that reduces only where mask == 1.

    Parameter and buffer names match nn.BatchNorm1d so state dicts interchange.
    """

    def __init__(self, channels: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.channels, self.eps, self.momentum = channels, eps, momentum
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        self.last_valid_count = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B,C,T]; mask [B,1,T] with 1 on real frames."""
        m = mask.to(x.dtype)
        n = m.sum()
        self.last_valid_count = int(n.item())
        if self.training:
            if self.last_valid_count == 0:
                # nothing was observed: write no statistic, emit no value
                return torch.zeros_like(x)
            mean = (x * m).sum(dim=(0, 2)) / n
            centred = (x - mean.view(1, -1, 1)) * m
            var = (centred * centred).sum(dim=(0, 2)) / n          # biased, as BatchNorm normalises
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                if self.last_valid_count > 1:
                    unbiased = var.detach() * n / (n - 1)
                    self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased)
                # with a single valid position the unbiased variance is undefined; leave it
        else:
            mean, var = self.running_mean, self.running_var
        y = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + self.eps)
        y = y * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return y * m           # padded positions carry no value out of the layer


class MaskedResidualBlock(nn.Module):
    """Same shape as model.ResidualBlock; only the normalisation is mask aware."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = MaskedBatchNorm1d(channels)
        self.norm2 = MaskedBatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h, mask):
        y = self.drop(self.act(self.norm1(self.conv1(h * mask), mask))) * mask
        y = self.drop(self.act(self.norm2(self.conv2(y * mask), mask))) * mask
        return (h + y) * mask


class MaskedSignTCN(nn.Module):
    """model.SignTCN with masked normalisation. A retraining candidate, not the web model."""

    model_type = MODEL_TYPE
    model_version = MODEL_VERSION

    def __init__(self, in_channels: int = TOTAL_CHANNELS, width: int = 32,
                 dilations=(1, 2, 4, 8), kernel_size: int = 3, dropout: float = 0.1,
                 n_classes: int = len(CLASSES)):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, width, kernel_size=1)
        self.blocks = nn.ModuleList(
            [MaskedResidualBlock(width, d, kernel_size, dropout) for d in dilations])
        self.head = nn.Linear(width, n_classes)

    def forward(self, x, pad_mask):
        """x [B,T,C], pad_mask [B,T] with 1 on real frames."""
        h = x.transpose(1, 2)
        m = pad_mask.unsqueeze(1)
        h = self.proj(h * m) * m
        for blk in self.blocks:
            h = blk(h, m)
        denom = pad_mask.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        pooled = (h * m).sum(dim=2) / denom
        return self.head(pooled)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def identity(self) -> dict:
        return {"model_type": self.model_type, "model_version": self.model_version,
                "parameter_count": self.parameter_count(),
                "normalisation": "MaskedBatchNorm1d, statistics over pad_mask==1 only",
                "running_statistics_of_a_loaded_checkpoint": "NOT corrected by loading"}
