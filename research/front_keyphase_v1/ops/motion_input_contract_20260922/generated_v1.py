"""Derived motion-input feature path for CPU float32 PyTorch sequence models.

Public API
----------
motion_features(x, pad_mask)           -> torch.Tensor [B, T, 3P]
fit_velocity_scale(sequences, labels)  -> dict
MotionAugmentedTCN(parent, velocity_scale, enabled=True)

Layout contract
---------------
x : float32 [B, T, 3P+1] on CPU, all finite.
    channels [0, 2P)     : P normalized relative XY pairs, INTERLEAVED, i.e.
                           pair p occupies channels (2p, 2p+1).
    channels [2P, 3P)    : binary observation mask, one per pair.
    channel  3P          : dt in seconds measured from the previous row.
pad_mask : float32 [B, T] on CPU, binary (1 = real frame, 0 = padding).

dt may be zero or negative; such rows are simply invalid FOR DERIVATIVES and are
never repaired.  Coordinates whose observation mask is 0 may hold arbitrary
finite values and are zeroed before any subtraction, never interpolated,
clipped, or bridged.  No future row is ever read.

This is a derived-feature path only: the velocity channels are an exact finite
difference of coordinates the caller already supplied.  It adds no new observed
information and carries no performance claim.  The first 2P outputs are the
finite-interval average rate of change of NORMALIZED 2D positions -- not a
physical 3D speed, and a large positive dt yields an interval average, not
evidence of continuously observed motion between the two rows.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["motion_features", "fit_velocity_scale", "MotionAugmentedTCN"]


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _infer_pairs(num_channels):
    if num_channels < 4 or (num_channels - 1) % 3 != 0:
        raise ValueError(
            "channel count must be 3P+1 for a positive integer P, got %d" % num_channels
        )
    return (num_channels - 1) // 3


def _require_binary(t, name):
    if not torch.isfinite(t).all():
        raise ValueError("%s contains non-finite values" % name)
    if not torch.all((t == 0) | (t == 1)):
        raise ValueError("%s must be binary (0 or 1)" % name)


def _validate_sequence_inputs(x, pad_mask):
    """Full contract check for (x, pad_mask); returns P."""
    if not isinstance(x, torch.Tensor) or not isinstance(pad_mask, torch.Tensor):
        raise TypeError("x and pad_mask must be torch.Tensor")
    if x.device.type != "cpu" or pad_mask.device.type != "cpu":
        raise ValueError("x and pad_mask must be CPU tensors")
    if x.dtype != torch.float32 or pad_mask.dtype != torch.float32:
        raise TypeError("x and pad_mask must be float32 (no conversion is performed)")
    if x.dim() != 3:
        raise ValueError("x must have shape [B, T, 3P+1], got %r" % (tuple(x.shape),))
    if pad_mask.dim() != 2:
        raise ValueError("pad_mask must have shape [B, T], got %r" % (tuple(pad_mask.shape),))
    b, t, c = x.shape
    if pad_mask.shape[0] != b or pad_mask.shape[1] != t:
        raise ValueError("pad_mask shape %r does not match x [B, T] = %r"
                         % (tuple(pad_mask.shape), (b, t)))
    if b < 1 or t < 1:
        raise ValueError("B and T must both be >= 1, got B=%d T=%d" % (b, t))
    p = _infer_pairs(c)
    if not torch.isfinite(x).all():
        raise ValueError("x contains non-finite values")
    _require_binary(pad_mask, "pad_mask")
    _require_binary(x[:, :, 2 * p:3 * p], "observation mask channels of x")
    return p


# --------------------------------------------------------------------------- #
# 1) motion features
# --------------------------------------------------------------------------- #
def _motion_features(x, pad_mask, p):
    b, t, _ = x.shape
    first = x.new_zeros((b, 1, 3 * p))
    if t == 1:
        return first

    xy = x[:, :, :2 * p]
    obs = x[:, :, 2 * p:3 * p]
    dt_cur = x[:, 1:, 3 * p:3 * p + 1]                      # [B, T-1, 1]

    valid = (obs[:, 1:, :] * obs[:, :-1, :]
             * pad_mask[:, 1:].unsqueeze(-1)
             * pad_mask[:, :-1].unsqueeze(-1)
             * (dt_cur > 0).to(x.dtype))                    # [B, T-1, P]
    valid_xy = torch.repeat_interleave(valid, 2, dim=2)     # [B, T-1, 2P]

    # Zero the coordinates first so absent/arbitrary values never enter the
    # subtraction, and replace non-positive dt so no division by zero occurs.
    cur = xy[:, 1:, :] * valid_xy
    prev = xy[:, :-1, :] * valid_xy
    safe_dt = torch.where(dt_cur > 0, dt_cur, torch.ones_like(dt_cur))
    v = ((cur - prev) / safe_dt) * valid_xy

    out = torch.cat([first, torch.cat([v, valid], dim=2)], dim=1)
    if not torch.isfinite(out).all():
        raise ValueError(
            "derived motion features are non-finite (e.g. dt underflow); "
            "rejecting instead of clipping"
        )
    return out


def motion_features(x, pad_mask):
    """Finite-difference motion features, shape [B, T, 3P].

    Channels [0, 2P) hold v_t = (xy_t - xy_{t-1}) / dt_t per XY pair, emitted
    only when t > 0, pad_mask at t and t-1 are 1, the pair's observation mask is
    1 at BOTH rows, and dt_t > 0; otherwise both coordinates of the pair are 0.
    Channels [2P, 3P) hold that per-pair derivative-valid mask as float32.
    Row 0 is always all zeros.  Inputs are never mutated.
    """
    p = _validate_sequence_inputs(x, pad_mask)
    return _motion_features(x, pad_mask, p)


# --------------------------------------------------------------------------- #
# 2) velocity scale fitting
# --------------------------------------------------------------------------- #
def _validate_labels(labels, n):
    if isinstance(labels, np.ndarray):
        if labels.dtype != np.int64:
            raise TypeError("labels array must have dtype int64")
        if labels.ndim != 1:
            raise ValueError("labels must be 1-D")
        arr = np.array(labels, copy=True)
    elif isinstance(labels, (list, tuple)):
        for v in labels:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise TypeError("labels must be integers")
        arr = np.array(labels, dtype=np.int64)
    else:
        raise TypeError("labels must be an int64 ndarray or a sequence of ints")
    if arr.shape[0] != n:
        raise ValueError("labels length %d does not match sequences length %d"
                         % (arr.shape[0], n))
    if arr.size == 0 or np.any(arr < 0):
        raise ValueError("labels must be non-empty and non-negative")
    return arr


def fit_velocity_scale(sequences, labels):
    """Fit per-coordinate-channel velocity scales on caller-supplied TRAINING data.

    Each clip contributes, per coordinate channel, mean(v^2) over that channel's
    valid derivative pairs; clips with no valid pair for a channel contribute no
    weight.  Clip weight = 1 / (num_distinct_labels * count_of_that_label).
    scale = sqrt(sum(w * clip_mean) / sum(w)), computed in float64.  Channels
    with a zero ratio or no observation get scale 1.  No epsilon, no threshold,
    no centering, no clipping, no label-specific branch, no development data.
    """
    if not isinstance(sequences, (list, tuple)) or len(sequences) == 0:
        raise ValueError("sequences must be a non-empty list of float32 [T, 3P+1] arrays")
    lab = _validate_labels(labels, len(sequences))

    if not isinstance(sequences[0], np.ndarray) or sequences[0].ndim != 2:
        raise TypeError("each sequence must be a 2-D numpy array [T, 3P+1]")
    channels = int(sequences[0].shape[1])
    p = _infer_pairs(channels)
    two_p = 2 * p

    classes, counts = np.unique(lab, return_counts=True)
    class_count = int(classes.shape[0])
    per_class = {int(k): int(c) for k, c in zip(classes.tolist(), counts.tolist())}

    num = np.zeros(two_p, dtype=np.float64)
    den = np.zeros(two_p, dtype=np.float64)

    for i, seq in enumerate(sequences):
        if not isinstance(seq, np.ndarray):
            raise TypeError("sequence %d is not a numpy array" % i)
        if seq.ndim != 2 or int(seq.shape[1]) != channels:
            raise ValueError("sequence %d must have shape [T, %d]" % (i, channels))
        if seq.dtype != np.float32:
            raise TypeError("sequence %d must have dtype float32 (no conversion)" % i)
        if int(seq.shape[0]) < 1:
            raise ValueError("sequence %d must have T >= 1" % i)

        x = torch.from_numpy(np.array(seq, copy=True)).unsqueeze(0)
        pad = torch.ones((1, x.shape[1]), dtype=torch.float32)
        feats = motion_features(x, pad)[0]

        v = feats[:, :two_p].to(torch.float64).numpy()
        valid_xy = np.repeat(feats[:, two_p:].numpy(), 2, axis=1).astype(np.float64)

        n_valid = valid_xy.sum(axis=0)
        observed = n_valid > 0.0
        if not np.any(observed):
            continue
        sq = (v * v).sum(axis=0)
        w = 1.0 / (float(class_count) * float(per_class[int(lab[i])]))
        num[observed] += w * (sq[observed] / n_valid[observed])
        den[observed] += w

    ratio = np.zeros(two_p, dtype=np.float64)
    has_weight = den > 0.0
    ratio[has_weight] = num[has_weight] / den[has_weight]

    scale64 = np.ones(two_p, dtype=np.float64)
    positive = ratio > 0.0
    scale64[positive] = np.sqrt(ratio[positive])

    scale32 = scale64.astype(np.float32)
    if not np.all(np.isfinite(scale32)) or not np.all(scale32 > 0.0):
        bad = [int(i) for i in np.nonzero(~(np.isfinite(scale32) & (scale32 > 0.0)))[0]]
        raise ValueError(
            "fitted positive scale is not representable as a positive finite "
            "float32 for channels %r" % bad
        )

    return {
        "scale": scale32,
        # per-channel denominator: sum of weights over clips observing it
        "valid_clip_weight": den,
        "zero_or_unobserved_channels": [int(i) for i in np.nonzero(~positive)[0]],
        "clip_count": int(len(sequences)),
        "class_count": class_count,
    }


# --------------------------------------------------------------------------- #
# 3) wrapper module
# --------------------------------------------------------------------------- #
def _as_scale_tensor(velocity_scale):
    if isinstance(velocity_scale, torch.Tensor):
        if velocity_scale.device.type != "cpu":
            raise ValueError("velocity_scale must be on CPU")
        if velocity_scale.dtype != torch.float32:
            raise TypeError("velocity_scale must be float32 (no conversion)")
        t = velocity_scale.detach().clone()
    elif isinstance(velocity_scale, np.ndarray):
        if velocity_scale.dtype != np.float32:
            raise TypeError("velocity_scale must be float32 (no conversion)")
        t = torch.from_numpy(np.array(velocity_scale, copy=True))
    else:
        raise TypeError("velocity_scale must be a float32 ndarray or torch.Tensor")
    if t.dim() != 1 or t.numel() == 0 or t.numel() % 2 != 0:
        raise ValueError("velocity_scale must be 1-D of even length 2P")
    if not torch.isfinite(t).all() or not torch.all(t > 0):
        raise ValueError("velocity_scale entries must be positive and finite")
    return t.contiguous()


def _validate_parent_tensors(module):
    for name, prm in module.named_parameters():
        if prm.device.type != "cpu":
            raise ValueError("parent parameter %s is not on CPU" % name)
        if prm.dtype != torch.float32:
            raise TypeError("parent parameter %s is not float32" % name)
        if not torch.isfinite(prm.detach()).all():
            raise ValueError("parent parameter %s is non-finite" % name)
    for name, buf in module.named_buffers():
        if buf.device.type != "cpu":
            raise ValueError("parent buffer %s is not on CPU" % name)
        if buf.is_floating_point():
            if buf.dtype != torch.float32:
                raise TypeError("parent floating buffer %s is not float32" % name)
            if not torch.isfinite(buf).all():
                raise ValueError("parent buffer %s is non-finite" % name)
        elif buf.is_complex():
            raise TypeError("parent buffer %s has an unsupported complex dtype" % name)


class MotionAugmentedTCN(nn.Module):
    """Adds a zero-initialized, bias-free motion residual to a frozen parent.

    The parent is deep-copied; the caller's module is never mutated.  All copied
    parameters are frozen except ``base.head.*``; ``velocity_weight`` is the only
    new trainable tensor and is created with ``torch.zeros`` so construction
    consumes no RNG.  Because the residual starts at exactly zero, outputs are
    identical to the parent's for any valid input until it is trained.
    """

    def __init__(self, parent, velocity_scale, enabled=True):
        super().__init__()
        if not isinstance(parent, nn.Module):
            raise TypeError("parent must be an nn.Module")
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be an explicit bool")

        scale = _as_scale_tensor(velocity_scale)
        pairs = scale.numel() // 2
        in_channels = 3 * pairs + 1

        proj = getattr(parent, "proj", None)
        blocks = getattr(parent, "blocks", None)
        head = getattr(parent, "head", None)
        if not isinstance(proj, nn.Conv1d):
            raise TypeError("parent.proj must be an nn.Conv1d")
        if not isinstance(blocks, nn.ModuleList) or len(blocks) == 0:
            raise TypeError("parent.blocks must be a non-empty nn.ModuleList")
        if not isinstance(head, nn.Linear):
            raise TypeError("parent.head must be an nn.Linear")
        if proj.in_channels != in_channels:
            raise ValueError("parent.proj.in_channels=%d does not match 3P+1=%d implied "
                             "by velocity_scale" % (proj.in_channels, in_channels))
        if proj.out_channels != head.in_features:
            raise ValueError("parent.proj.out_channels=%d must equal parent.head.in_features=%d"
                             % (proj.out_channels, head.in_features))
        _validate_parent_tensors(parent)

        self.num_pairs = int(pairs)
        self.width = int(proj.out_channels)
        self.enabled = bool(enabled)

        self.base = copy.deepcopy(parent)
        self.base.eval()
        for prm in self.base.parameters():
            prm.requires_grad_(False)
        for prm in self.base.head.parameters():
            prm.requires_grad_(True)

        self.register_buffer("velocity_scale", scale)
        self.velocity_weight = nn.Parameter(
            torch.zeros(self.width, 3 * self.num_pairs, 1, dtype=torch.float32)
        )

    def train(self, mode=True):
        """Keep every base submodule in eval mode (dropout off, stats frozen)."""
        if not isinstance(mode, bool):
            raise TypeError("mode must be a bool")
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, pad_mask):
        pairs = _validate_sequence_inputs(x, pad_mask)
        if pairs != self.num_pairs:
            raise ValueError("x implies P=%d but this module was built for P=%d"
                             % (pairs, self.num_pairs))
        two_p = 2 * self.num_pairs
        seq_len = x.shape[1]
        m = pad_mask.unsqueeze(1)                                   # [B, 1, T]

        h_base = self.base.proj(x.transpose(1, 2) * m) * m
        if h_base.shape[-1] != seq_len:
            raise ValueError("parent.proj must preserve the time dimension "
                             "(got %d, expected %d)" % (h_base.shape[-1], seq_len))

        motion = _motion_features(x, pad_mask, self.num_pairs)      # [B, T, 3P]
        if self.enabled:
            motion_in = torch.cat(
                [motion[:, :, :two_p] / self.velocity_scale, motion[:, :, two_p:]], dim=2
            )
            if not torch.isfinite(motion_in).all():
                raise ValueError("scaled motion input is non-finite; rejecting "
                                 "instead of clipping")
        else:
            motion_in = torch.zeros_like(motion)

        h = h_base + F.conv1d(motion_in.transpose(1, 2), self.velocity_weight) * m

        for block in self.base.blocks:
            h = block(h, m)

        pooled = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.base.head(pooled)
