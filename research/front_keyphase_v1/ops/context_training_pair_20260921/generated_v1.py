"""Generic paired linear-head research helper.

Trains two independent linear classifier heads (a *control* arm and a
*treatment* arm) on top of already frozen feature vectors.  Both arms share a
single precomputed sampling schedule and start from exact copies of the same
parent head, so the only difference between them is which paired feature
matrix supplies the second half of every batch.

The module builds no encoder, touches no files, and uses only numpy, torch and
the standard library.  Global numpy / torch RNG state and global torch thread
settings are never modified.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["run_pair"]

_REAL_FEATURE_KINDS = frozenset("fiu")
_INTEGER_KINDS = frozenset("iu")
_BETAS = (0.9, 0.999)
_EPS = 1e-8


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _reject_bool_scalar(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be a boolean")


def _check_positive_int(value: Any, name: str) -> int:
    _reject_bool_scalar(value, name)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def _check_nonnegative_int(value: Any, name: str) -> int:
    _reject_bool_scalar(value, name)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative, got {out}")
    return out


def _check_float(value: Any, name: str, *, strictly_positive: bool) -> float:
    _reject_bool_scalar(value, name)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a real scalar, got {type(value).__name__}")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    if strictly_positive:
        if out <= 0.0:
            raise ValueError(f"{name} must be strictly positive, got {out!r}")
    elif out < 0.0:
        raise ValueError(f"{name} must be non-negative, got {out!r}")
    return out


def _as_float32_array(value: Any, name: str, ndim: int) -> np.ndarray:
    """Copy `value` into a finite float32 array of exactly `ndim` dimensions."""
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must not be a boolean array")
    if kind == "c":
        raise TypeError(f"{name} must not be a complex array")
    if kind not in _REAL_FEATURE_KINDS:
        raise TypeError(f"{name} must be a real numeric array, got dtype {array.dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {array.shape}")
    out = np.array(array, dtype=np.float32, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    if out.size == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _as_label_array(value: Any, name: str) -> np.ndarray:
    """Copy `value` into a 1-D int64 array of integer class IDs."""
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must be integer class IDs, not booleans")
    if kind == "c":
        raise TypeError(f"{name} must not be a complex array")
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {array.shape}")
    if kind in _INTEGER_KINDS:
        out = np.array(array, dtype=np.int64, copy=True)
    elif kind == "f":
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        rounded = np.rint(array)
        if not np.array_equal(rounded, array):
            raise ValueError(f"{name} must contain integer-valued labels only")
        out = rounded.astype(np.int64, copy=False)
        out = np.array(out, dtype=np.int64, copy=True)
    else:
        raise TypeError(f"{name} must be an integer label array, got dtype {array.dtype}")
    if out.size == 0:
        raise ValueError(f"{name} must not be empty")
    return out


def _check_label_range(labels: np.ndarray, name: str, num_classes: int) -> None:
    if int(labels.min()) < 0 or int(labels.max()) >= num_classes:
        raise ValueError(
            f"{name} must lie in [0, {num_classes}), got range "
            f"[{int(labels.min())}, {int(labels.max())}]"
        )


def _rows_by_class(labels: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return sorted unique labels and, per label, the rows carrying it."""
    classes = np.unique(labels)
    rows = [np.flatnonzero(labels == int(cls)) for cls in classes]
    return classes, rows


# ---------------------------------------------------------------------------
# sampling schedule
# ---------------------------------------------------------------------------


def _build_schedule(
    base_labels: np.ndarray,
    paired_labels: np.ndarray,
    steps: int,
    half_batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute both index arrays with a local RNG (globals untouched)."""
    rng = np.random.default_rng(seed)

    _, base_rows = _rows_by_class(base_labels)
    _, paired_rows = _rows_by_class(paired_labels)
    num_base_classes = len(base_rows)
    num_paired_classes = len(paired_rows)

    base_indices = np.empty((steps, half_batch_size), dtype=np.int64)
    pair_indices = np.empty((steps, half_batch_size), dtype=np.int64)

    for step in range(steps):
        # base half first, one independent class draw then one row draw per slot
        for slot in range(half_batch_size):
            rows = base_rows[int(rng.integers(num_base_classes))]
            base_indices[step, slot] = rows[int(rng.integers(rows.size))]
        # then the paired half, drawn the same way
        for slot in range(half_batch_size):
            rows = paired_rows[int(rng.integers(num_paired_classes))]
            pair_indices[step, slot] = rows[int(rng.integers(rows.size))]

    return base_indices, pair_indices


# ---------------------------------------------------------------------------
# shared arm trainer
# ---------------------------------------------------------------------------


def _train_arm(
    arm_name: str,
    base_features: torch.Tensor,
    second_features: torch.Tensor,
    base_labels: torch.Tensor,
    paired_labels: torch.Tensor,
    base_index_t: torch.Tensor,
    pair_index_t: torch.Tensor,
    parent_weight: torch.Tensor,
    parent_bias: torch.Tensor,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    anchor_coefficient: float,
) -> Dict[str, Any]:
    """Train one head; identical code path for both arms to avoid drift."""
    weight = parent_weight.clone().detach().requires_grad_(True)
    bias = parent_bias.clone().detach().requires_grad_(True)
    anchor_weight = parent_weight.clone().detach()
    anchor_bias = parent_bias.clone().detach()

    optimizer = torch.optim.AdamW(
        [weight, bias],
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=_BETAS,
        eps=_EPS,
    )

    pre_update_losses = np.empty((steps, 3), dtype=np.float64)

    for step in range(steps):
        base_rows = base_index_t[step]
        pair_rows = pair_index_t[step]
        batch_x = torch.cat((base_features[base_rows], second_features[pair_rows]), dim=0)
        batch_y = torch.cat((base_labels[base_rows], paired_labels[pair_rows]), dim=0)

        logits = F.linear(batch_x, weight, bias)
        cross_entropy = F.cross_entropy(logits, batch_y)
        anchor_loss = 0.5 * anchor_coefficient * (
            ((weight - anchor_weight) ** 2).sum() + ((bias - anchor_bias) ** 2).sum()
        )
        total_loss = cross_entropy + anchor_loss

        # recorded strictly before optimizer.step()
        pre_update_losses[step, 0] = float(cross_entropy.detach())
        pre_update_losses[step, 1] = float(anchor_loss.detach())
        pre_update_losses[step, 2] = float(total_loss.detach())
        if not np.isfinite(pre_update_losses[step]).all():
            raise RuntimeError(
                f"{arm_name} arm produced a non-finite loss at step {step}: "
                f"{pre_update_losses[step].tolist()}"
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for tensor, label in ((weight, "weight"), (bias, "bias")):
            grad = tensor.grad
            if grad is None:
                raise RuntimeError(
                    f"{arm_name} arm produced no gradient for {label} at step {step}"
                )
            if not bool(torch.isfinite(grad).all()):
                raise RuntimeError(
                    f"{arm_name} arm produced a non-finite {label} gradient at step {step}"
                )

        optimizer.step()

        with torch.no_grad():
            if not (bool(torch.isfinite(weight).all()) and bool(torch.isfinite(bias).all())):
                raise RuntimeError(
                    f"{arm_name} arm produced non-finite parameters after step {step}"
                )

    with torch.no_grad():
        final_weight = weight.detach().cpu().numpy().astype(np.float32, copy=True)
        final_bias = bias.detach().cpu().numpy().astype(np.float32, copy=True)
        delta_weight = (
            weight.detach().cpu().numpy().astype(np.float64)
            - anchor_weight.cpu().numpy().astype(np.float64)
        )
        delta_bias = (
            bias.detach().cpu().numpy().astype(np.float64)
            - anchor_bias.cpu().numpy().astype(np.float64)
        )
    final_anchor_distance_squared = float(
        np.square(delta_weight).sum() + np.square(delta_bias).sum()
    )
    if not np.isfinite(final_anchor_distance_squared):
        raise RuntimeError(
            f"{arm_name} arm produced a non-finite final anchor distance"
        )

    return {
        "weight": final_weight,
        "bias": final_bias,
        "pre_update_losses": pre_update_losses,
        "final_anchor_distance_squared": final_anchor_distance_squared,
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_pair(
    base_x: Any,
    base_y: Any,
    paired_annotation_x: Any,
    paired_context_x: Any,
    paired_y: Any,
    parent_weight: Any,
    parent_bias: Any,
    *,
    steps: int = 660,
    batch_size: int = 32,
    seed: int = 20260919,
    learning_rate: float = 0.001,
    weight_decay: float = 0.001,
    anchor_coefficient: float = 0.01,
) -> Dict[str, Any]:
    """Train a control and a treatment linear head over frozen features.

    The control arm builds each batch from ``base_x`` rows plus
    ``paired_annotation_x`` rows; the treatment arm substitutes
    ``paired_context_x`` for that second half.  Both arms reuse the same
    precomputed index schedule, the same labels, and the same parent
    initialisation, so any difference between them comes from the paired
    feature matrix alone.

    Inputs are never mutated; every array is copied before use.
    """
    steps = _check_positive_int(steps, "steps")
    batch_size = _check_positive_int(batch_size, "batch_size")
    if batch_size % 2 != 0:
        raise ValueError(f"batch_size must be even, got {batch_size}")
    seed = _check_nonnegative_int(seed, "seed")
    learning_rate = _check_float(learning_rate, "learning_rate", strictly_positive=True)
    weight_decay = _check_float(weight_decay, "weight_decay", strictly_positive=False)
    anchor_coefficient = _check_float(
        anchor_coefficient, "anchor_coefficient", strictly_positive=False
    )
    half_batch_size = batch_size // 2

    base_features = _as_float32_array(base_x, "base_x", 2)
    annotation_features = _as_float32_array(paired_annotation_x, "paired_annotation_x", 2)
    context_features = _as_float32_array(paired_context_x, "paired_context_x", 2)
    weight_parent = _as_float32_array(parent_weight, "parent_weight", 2)
    bias_parent = _as_float32_array(parent_bias, "parent_bias", 1)
    base_labels = _as_label_array(base_y, "base_y")
    paired_labels = _as_label_array(paired_y, "paired_y")

    num_classes, feature_dim = weight_parent.shape
    if num_classes < 2:
        raise ValueError(f"parent_weight must define at least 2 classes, got {num_classes}")
    if feature_dim < 1:
        raise ValueError(f"parent_weight must define at least 1 feature, got {feature_dim}")
    if bias_parent.shape[0] != num_classes:
        raise ValueError(
            f"parent_bias must have length {num_classes}, got {bias_parent.shape[0]}"
        )

    num_base_rows = base_features.shape[0]
    num_paired_rows = annotation_features.shape[0]
    if num_base_rows < 1:
        raise ValueError("base_x must contain at least one row")
    if num_paired_rows < 1:
        raise ValueError("paired_annotation_x must contain at least one row")
    if base_features.shape[1] != feature_dim:
        raise ValueError(
            f"base_x feature dimension {base_features.shape[1]} does not match "
            f"parent_weight dimension {feature_dim}"
        )
    if annotation_features.shape != (num_paired_rows, feature_dim):
        raise ValueError(
            f"paired_annotation_x must have shape ({num_paired_rows}, {feature_dim}), "
            f"got {annotation_features.shape}"
        )
    if context_features.shape != annotation_features.shape:
        raise ValueError(
            "paired_annotation_x and paired_context_x must have the same shape, got "
            f"{annotation_features.shape} and {context_features.shape}"
        )
    if base_labels.shape[0] != num_base_rows:
        raise ValueError(
            f"base_y must have length {num_base_rows}, got {base_labels.shape[0]}"
        )
    if paired_labels.shape[0] != num_paired_rows:
        raise ValueError(
            f"paired_y must have length {num_paired_rows}, got {paired_labels.shape[0]}"
        )

    _check_label_range(base_labels, "base_y", num_classes)
    _check_label_range(paired_labels, "paired_y", num_classes)
    if not np.array_equal(np.unique(base_labels), np.arange(num_classes, dtype=np.int64)):
        raise ValueError(
            f"base_y must contain every class in [0, {num_classes}); missing "
            f"{sorted(set(range(num_classes)) - set(int(c) for c in np.unique(base_labels)))}"
        )

    base_indices, pair_indices = _build_schedule(
        base_labels, paired_labels, steps, half_batch_size, seed
    )

    base_features_t = torch.tensor(base_features, dtype=torch.float32)
    annotation_features_t = torch.tensor(annotation_features, dtype=torch.float32)
    context_features_t = torch.tensor(context_features, dtype=torch.float32)
    base_labels_t = torch.tensor(base_labels, dtype=torch.long)
    paired_labels_t = torch.tensor(paired_labels, dtype=torch.long)
    parent_weight_t = torch.tensor(weight_parent, dtype=torch.float32)
    parent_bias_t = torch.tensor(bias_parent, dtype=torch.float32)
    base_index_t = torch.tensor(base_indices, dtype=torch.long)
    pair_index_t = torch.tensor(pair_indices, dtype=torch.long)

    arms: Dict[str, Any] = {}
    for arm_name, second_features_t in (
        ("control", annotation_features_t),
        ("treatment", context_features_t),
    ):
        arms[arm_name] = _train_arm(
            arm_name=arm_name,
            base_features=base_features_t,
            second_features=second_features_t,
            base_labels=base_labels_t,
            paired_labels=paired_labels_t,
            base_index_t=base_index_t,
            pair_index_t=pair_index_t,
            parent_weight=parent_weight_t,
            parent_bias=parent_bias_t,
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            anchor_coefficient=anchor_coefficient,
        )

    settings: Dict[str, Any] = {
        "steps": steps,
        "batch_size": batch_size,
        "half_batch_size": half_batch_size,
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "anchor_coefficient": anchor_coefficient,
        "optimizer": "AdamW",
        "betas": _BETAS,
        "eps": _EPS,
        "num_classes": int(num_classes),
        "feature_dim": int(feature_dim),
        "num_base_rows": int(num_base_rows),
        "num_paired_rows": int(num_paired_rows),
        "base_classes": np.unique(base_labels).astype(np.int64, copy=True),
        "paired_classes": np.unique(paired_labels).astype(np.int64, copy=True),
        "device": "cpu",
        "dtype": "float32",
        "loss_columns": ("cross_entropy", "anchor_loss", "total_loss"),
        "control_second_half": "paired_annotation_x",
        "treatment_second_half": "paired_context_x",
    }

    return {
        "control": arms["control"],
        "treatment": arms["treatment"],
        "schedule": {
            "base_indices": base_indices.astype(np.int64, copy=True),
            "pair_indices": pair_indices.astype(np.int64, copy=True),
        },
        "settings": settings,
    }
