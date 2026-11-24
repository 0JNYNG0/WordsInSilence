"""Paired position/motion residual adaptation on a frozen TCN parent.

All motion logic (validation, rate features, the augmented wrapper) is reused
unchanged from the locally supplied, already-tested ``motion_base`` module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from motion_base import MotionAugmentedTCN, _validate_sequence_inputs

__all__ = ["PairedInputTCN", "position_features", "fit_position_scale", "run_pair"]


# --------------------------------------------------------------------------- #
# 1. position features
# --------------------------------------------------------------------------- #
def position_features(x, pad_mask):
    """[B,T,3P+1] -> [B,T,3P]: gated current XY followed by their validity mask.

    A coordinate is kept only when its current joint observation mask and the
    frame mask are both 1; otherwise it is zero.  No dt use, no future context,
    no scaling / interpolation / clipping, no mutation of the inputs.
    """
    P = _validate_sequence_inputs(x, pad_mask)
    frame = pad_mask.to(torch.float32).unsqueeze(-1)                  # [B,T,1]
    valid = x[..., 2 * P:3 * P].to(torch.float32) * frame             # [B,T,P]
    values = x[..., :2 * P].to(torch.float32) * torch.repeat_interleave(valid, 2, dim=-1)
    return torch.cat([values, valid], dim=-1).contiguous()


# --------------------------------------------------------------------------- #
# shared validation helpers
# --------------------------------------------------------------------------- #
def _check_clip_list(sequences, name, channels=None):
    if not isinstance(sequences, (list, tuple)) or len(sequences) == 0:
        raise ValueError(f"{name} must be a non-empty list of clip arrays")
    for i, s in enumerate(sequences):
        if not isinstance(s, np.ndarray) or s.dtype != np.float32:
            raise TypeError(f"{name}[{i}] must be a float32 numpy array")
        if s.ndim != 2 or s.shape[0] < 1:
            raise ValueError(f"{name}[{i}] must have shape [T>=1, 3P+1]")
        if channels is None:
            channels = int(s.shape[1])
            if channels < 4 or (channels - 1) % 3 != 0:
                raise ValueError(f"{name}[{i}] must have 3P+1 channels with P>=1")
        elif int(s.shape[1]) != channels:
            raise ValueError(f"{name}[{i}] has an inconsistent channel count")
        if not np.isfinite(s).all():
            raise ValueError(f"{name}[{i}] contains non-finite values")
        p = (channels - 1) // 3
        obs = s[:, 2 * p:3 * p]
        if not np.all((obs == 0.0) | (obs == 1.0)):
            raise ValueError(f"{name}[{i}] observation mask must be binary")
    return channels


def _check_labels(labels, count, name, num_classes=None):
    if not isinstance(labels, np.ndarray) or labels.dtype != np.int64:
        raise TypeError(f"{name} must be an int64 numpy array")
    if labels.ndim != 1 or int(labels.shape[0]) != count:
        raise ValueError(f"{name} must be 1D with length {count}")
    if int(labels.min()) < 0:
        raise ValueError(f"{name} must be non-negative")
    if num_classes is not None and int(labels.max()) >= num_classes:
        raise ValueError(f"{name} contains a label outside [0, {num_classes})")
    return labels


def _check_schedule(indices, name, limit):
    if not isinstance(indices, np.ndarray) or indices.dtype != np.int64:
        raise TypeError(f"{name} must be an int64 numpy array")
    if indices.ndim != 2 or indices.size == 0:
        raise ValueError(f"{name} must be a non-empty [steps, half] array")
    if int(indices.min()) < 0 or int(indices.max()) >= limit:
        raise ValueError(f"{name} contains out-of-range clip indices")
    return indices


def _as_scale_tensor(scale, length, name):
    if isinstance(scale, torch.Tensor):
        if scale.device.type != "cpu" or scale.dtype != torch.float32:
            raise TypeError(f"{name} must be a CPU float32 tensor")
        out = scale.detach().clone()
    else:
        arr = np.asarray(scale)
        if arr.dtype != np.float32:
            raise TypeError(f"{name} must be float32")
        out = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
    if out.ndim != 1 or int(out.shape[0]) != length:
        raise ValueError(f"{name} must be 1D with length {length}")
    if not bool(torch.isfinite(out).all()) or not bool((out > 0).all()):
        raise ValueError(f"{name} must be finite and strictly positive")
    return out


def _check_weight(value, name, allow_zero):
    v = float(value)
    if not np.isfinite(v) or (v <= 0.0 if not allow_zero else v < 0.0):
        raise ValueError(f"{name} must be finite and "
                         f"{'non-negative' if allow_zero else 'positive'}")
    return v


# --------------------------------------------------------------------------- #
# 2. class-balanced clip-RMS position scale
# --------------------------------------------------------------------------- #
def fit_position_scale(sequences, labels):
    """Class-balanced clip-RMS scale over current XY observations (training data only)."""
    channels = _check_clip_list(sequences, "sequences")
    P = (channels - 1) // 3
    labels = _check_labels(labels, len(sequences), "labels")

    classes, counts = np.unique(labels, return_counts=True)
    class_count = int(classes.size)
    per_class = {int(c): 1.0 / (class_count * int(k)) for c, k in zip(classes, counts)}

    numerator = np.zeros(2 * P, dtype=np.float64)
    denominator = np.zeros(2 * P, dtype=np.float64)
    for clip, label in zip(sequences, labels):
        s = clip.astype(np.float64)
        valid = np.repeat(s[:, 2 * P:3 * P], 2, axis=1)               # [T,2P]
        observations = valid.sum(axis=0)
        observed = observations > 0.0
        mean_square = np.zeros(2 * P, dtype=np.float64)
        squared = ((s[:, :2 * P] * valid) ** 2).sum(axis=0)
        mean_square[observed] = squared[observed] / observations[observed]
        weight = per_class[int(label)]
        numerator[observed] += weight * mean_square[observed]
        denominator[observed] += weight

    scale = np.ones(2 * P, dtype=np.float64)
    usable = (denominator > 0.0) & (numerator > 0.0)
    scale[usable] = np.sqrt(numerator[usable] / denominator[usable])
    if not np.all(np.isfinite(scale)) or not np.all(scale > 0.0):
        raise ValueError("fitted position scale is not finite and positive")
    scale32 = scale.astype(np.float32)
    if not np.all(np.isfinite(scale32)) or not np.all(scale32 > 0.0):
        raise ValueError("fitted position scale is not representable in float32")

    return {
        "scale": scale32,
        "valid_clip_weight": denominator,
        "zero_or_unobserved_channels": [int(i) for i in np.flatnonzero(~usable)],
        "clip_count": int(len(sequences)),
        "class_count": class_count,
    }


# --------------------------------------------------------------------------- #
# 3. paired wrapper
# --------------------------------------------------------------------------- #
class PairedInputTCN(MotionAugmentedTCN):
    """MotionAugmentedTCN whose zero-initialised residual reads motion *or* position.

    The ``velocity_scale`` buffer keeps its name for checkpoint compatibility but
    holds the scale of whichever feature was selected.
    """

    def __init__(self, parent, scale, feature_mode):
        if feature_mode not in ("position", "motion"):
            raise ValueError("feature_mode must be 'position' or 'motion'")
        super().__init__(parent, scale, enabled=True)
        self.feature_mode = feature_mode

    def forward(self, x, pad_mask):
        if self.feature_mode == "motion":
            return super().forward(x, pad_mask)

        features = position_features(x, pad_mask)
        P = features.shape[-1] // 3
        mask = pad_mask.to(torch.float32).unsqueeze(1)                # [B,1,T]

        residual_input = torch.cat(
            [features[..., :2 * P] / self.velocity_scale, features[..., 2 * P:]], dim=-1
        ).transpose(1, 2)

        h = self.base.proj((x * mask.transpose(1, 2)).transpose(1, 2)) * mask
        h = h + F.conv1d(residual_input, self.velocity_weight) * mask
        for block in self.base.blocks:
            h = block(h, mask)
        pooled = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.base.head(pooled)


# --------------------------------------------------------------------------- #
# 4. paired run
# --------------------------------------------------------------------------- #
def _pad_batch(clips):
    length = max(int(c.shape[0]) for c in clips)
    x = np.zeros((len(clips), length, int(clips[0].shape[1])), dtype=np.float32)
    m = np.zeros((len(clips), length), dtype=np.float32)
    for i, clip in enumerate(clips):
        t = int(clip.shape[0])
        x[i, :t] = clip
        m[i, :t] = 1.0
    return torch.from_numpy(x), torch.from_numpy(m)


def _train_arm(arm, feature_mode, parent, scale, base_sequences, base_y,
               context_sequences, context_y, teacher, base_indices, context_indices,
               learning_rate, weight_decay, anchor_coefficient, output_coefficient,
               progress):
    model = PairedInputTCN(parent, scale, feature_mode)
    model.train()                                    # override keeps base in eval

    names = [n for n, _ in model.named_parameters()]
    trainable_names = [n for n in names if n == "velocity_weight" or n.startswith("base.head.")]
    if "velocity_weight" not in trainable_names or "base.head.weight" not in trainable_names:
        raise ValueError("wrapper does not expose the expected trainable parameters")
    params = dict(model.named_parameters())
    for name, param in params.items():
        param.requires_grad_(name in trainable_names)
    if any(module.training for module in model.base.modules()):
        raise RuntimeError("base modules must remain in eval mode")

    initial = {n: params[n].detach().clone() for n in trainable_names}
    frozen_reference = {n: p.detach().clone() for n, p in params.items()
                        if n not in trainable_names}
    buffer_reference = {n: b.detach().clone() for n, b in model.named_buffers()}

    optimizer = torch.optim.AdamW([params[n] for n in trainable_names],
                                  lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=weight_decay)

    steps, half = int(base_indices.shape[0]), int(base_indices.shape[1])
    losses = np.zeros((steps, 4), dtype=np.float64)
    for step in range(steps):
        base_rows, context_rows = base_indices[step], context_indices[step]
        clips = ([base_sequences[int(j)] for j in base_rows]
                 + [context_sequences[int(j)] for j in context_rows])
        x, pad_mask = _pad_batch(clips)
        targets = torch.from_numpy(np.concatenate([base_y[base_rows], context_y[context_rows]]))

        logits = model(x, pad_mask)
        cross_entropy = F.cross_entropy(logits, targets)

        anchor = logits.new_zeros(())
        for name in trainable_names:
            anchor = anchor + ((params[name] - initial[name]) ** 2).sum()
        anchor = 0.5 * anchor_coefficient * anchor

        delta = logits[:half].double() - teacher[torch.from_numpy(np.array(base_rows, copy=True))]
        output = 0.5 * output_coefficient * ((delta - delta.mean(dim=1, keepdim=True)) ** 2).mean()

        total = cross_entropy + anchor + output
        row = np.array([cross_entropy.detach().item(), anchor.detach().item(),
                        output.detach().item(), total.detach().item()], dtype=np.float64)
        losses[step] = row

        optimizer.zero_grad(set_to_none=True)
        if not np.isfinite(row).all():
            raise FloatingPointError(f"non-finite loss at step {step} of arm '{arm}'")
        total.backward()
        for name in trainable_names:
            grad = params[name].grad
            if grad is not None and not bool(torch.isfinite(grad).all()):
                raise FloatingPointError(
                    f"non-finite gradient for {name} at step {step} of arm '{arm}'")
        optimizer.step()

        if progress is not None and ((step + 1) % 100 == 0 or step + 1 == steps):
            progress(arm, step + 1, steps)

    frozen_unchanged = all(torch.equal(p.detach(), frozen_reference[n])
                           for n, p in model.named_parameters() if n in frozen_reference)
    buffers_unchanged = all(torch.equal(b.detach(), buffer_reference[n])
                            for n, b in model.named_buffers())
    assert frozen_unchanged, f"arm '{arm}' modified frozen parameters"
    assert buffers_unchanged, f"arm '{arm}' modified buffers"
    assert not any(module.training for module in model.base.modules())

    return {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "losses": losses,
        "trainable_names": list(trainable_names),
        "updates": steps,
        "feature_mode": feature_mode,
        "buffers_unchanged": bool(buffers_unchanged),
        "frozen_parameters_unchanged": bool(frozen_unchanged),
    }


def run_pair(parent, base_sequences, base_y, context_sequences, context_y,
             teacher_base_logits, base_indices, context_indices, position_scale,
             velocity_scale, *, learning_rate=0.001, weight_decay=0.001,
             anchor_coefficient=0.01, output_coefficient=1.0, progress=None):
    """Train the position (control) and motion (treatment) arms independently."""
    if not isinstance(parent, torch.nn.Module):
        raise TypeError("parent must be a torch.nn.Module")

    channels = _check_clip_list(base_sequences, "base_sequences")
    _check_clip_list(context_sequences, "context_sequences", channels)
    P = (channels - 1) // 3

    if not isinstance(teacher_base_logits, np.ndarray) or \
            teacher_base_logits.dtype not in (np.float32, np.float64):
        raise TypeError("teacher_base_logits must be a float32/float64 numpy array")
    if teacher_base_logits.ndim != 2 or \
            int(teacher_base_logits.shape[0]) != len(base_sequences) or \
            int(teacher_base_logits.shape[1]) < 1:
        raise ValueError("teacher_base_logits must have shape [len(base_sequences), C]")
    if not np.isfinite(teacher_base_logits).all():
        raise ValueError("teacher_base_logits contains non-finite values")
    num_classes = int(teacher_base_logits.shape[1])

    base_y = _check_labels(base_y, len(base_sequences), "base_y", num_classes)
    context_y = _check_labels(context_y, len(context_sequences), "context_y", num_classes)

    base_indices = _check_schedule(base_indices, "base_indices", len(base_sequences))
    context_indices = _check_schedule(context_indices, "context_indices", len(context_sequences))
    if base_indices.shape != context_indices.shape:
        raise ValueError("base_indices and context_indices must have the same shape")

    learning_rate = _check_weight(learning_rate, "learning_rate", allow_zero=False)
    weight_decay = _check_weight(weight_decay, "weight_decay", allow_zero=True)
    anchor_coefficient = _check_weight(anchor_coefficient, "anchor_coefficient", allow_zero=True)
    output_coefficient = _check_weight(output_coefficient, "output_coefficient", allow_zero=True)
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    position = _as_scale_tensor(position_scale, 2 * P, "position_scale")
    velocity = _as_scale_tensor(velocity_scale, 2 * P, "velocity_scale")
    teacher = torch.tensor(np.asarray(teacher_base_logits, dtype=np.float64),
                           dtype=torch.float64)

    parent_reference = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}

    arms = {}
    for arm, mode, scale in (("control", "position", position),
                             ("treatment", "motion", velocity)):
        arms[arm] = _train_arm(arm, mode, parent, scale, base_sequences, base_y,
                               context_sequences, context_y, teacher, base_indices,
                               context_indices, learning_rate, weight_decay,
                               anchor_coefficient, output_coefficient, progress)

    parent_after = parent.state_dict()
    assert set(parent_after) == set(parent_reference) and all(
        torch.equal(parent_after[k].detach().cpu(), v) for k, v in parent_reference.items()
    ), "caller parent state changed during run_pair"

    settings = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "anchor_coefficient": anchor_coefficient,
        "output_coefficient": output_coefficient,
        "optimizer": "AdamW",
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "steps": int(base_indices.shape[0]),
        "base_half": int(base_indices.shape[1]),
        "context_half": int(context_indices.shape[1]),
        "batch_size": int(base_indices.shape[1] + context_indices.shape[1]),
        "channels": int(channels),
        "joint_count": int(P),
        "class_count": num_classes,
        "control_feature_mode": "position",
        "treatment_feature_mode": "motion",
        "position_scale": position.numpy().copy(),
        "velocity_scale": velocity.numpy().copy(),
    }

    return {
        "control": arms["control"],
        "treatment": arms["treatment"],
        "settings": settings,
        "schedule": {"base_indices": base_indices.copy(),
                     "context_indices": context_indices.copy()},
    }
