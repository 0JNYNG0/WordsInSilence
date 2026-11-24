"""Train exactly one affine head on frozen CPU features with a caller-supplied schedule.

This module is deliberately generic: it knows nothing about any particular dataset,
model, or task.  It receives frozen float32 features, integer labels, a parent
(affine) head, a fully materialised integer sampling schedule, and a caller-computed
float64 Gram matrix, and it runs a deterministic AdamW optimisation of a single
``[C, D]`` weight / ``[C]`` bias pair.

Contract highlights
-------------------
* No randomness of any kind is used or touched: no layer init, no ``manual_seed``,
  no dropout, no augmentation, no shuffling.  The batch composition for every step
  comes only from ``base_indices`` / ``pair_indices``.
* No thread, device, or dtype global settings are modified.  Everything runs on CPU;
  parameters are float32, the output-protection term is float64.
* Inputs are never mutated.  Every array is validated and copied by the shared
  validation helpers (or by the local schedule / Gram validators) before any tensor
  or optimizer is created.
* ``gram`` is used exactly as supplied.  It is never recomputed, symmetrised, or
  clipped; violations of the documented structural requirements are rejected.
  Proving that ``gram`` really is the class-balanced second moment of the augmented
  original base features ``[x, 1]`` is the caller's responsibility.
* ``output_coefficient=0.0`` is supported so callers can test the head in isolation
  (the raw output loss is still reported, it just does not enter the total).

Loss at every step (recorded *before* ``optimizer.step``)::

    CE      = cross_entropy(linear(batch, W, b), labels, reduction='mean')   # float32
    Anchor  = 0.5 * anchor_coefficient * (||W - Wp||^2 + ||b - bp||^2)       # float32
    D_raw   = [W - Wp, (b - bp)[:, None]]                                    # float64 [C, D+1]
    D_cent  = D_raw - D_raw.mean(dim=0, keepdim=True)                        # centred over classes
    Raw     = 0.5 / C * ((D_cent @ gram) * D_cent).sum()                     # float64
    Total   = CE + Anchor + output_coefficient * Raw                         # float64

The float64 output penalty is part of the autograd graph (``D`` is never detached),
so its gradient flows back into the float32 leaves ``W`` and ``b``.  ``gram`` and the
parent copies carry no gradient.

Raised errors
-------------
``TypeError`` for wrong kinds/dtypes, ``ValueError`` for shape/range/structure
violations, ``OverflowError`` for unsigned integer schedules outside the int64 range,
and ``FloatingPointError`` if a loss, gradient, or parameter becomes non-finite
during training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from generated_v2 import (
    _as_float32_array,
    _as_label_array,
    _check_float,
    _check_label_range,
    _check_positive_int,
)

__all__ = ["train_head"]


_INT64_MAX = 2**63 - 1

_GRAM_SYMMETRY_TOLERANCE = 1e-10
_GRAM_EIGENVALUE_TOLERANCE = -1e-10

_ADAMW_BETAS = (0.9, 0.999)
_ADAMW_EPS = 1e-8

_LOSS_COLUMNS = (
    "cross_entropy",
    "weighted_parameter_anchor",
    "weighted_output_penalty",
    "total",
)


# ---------------------------------------------------------------------------
# Local validation (the imported helpers cover features, labels, and scalars).
# ---------------------------------------------------------------------------


def _as_index_schedule(value, name, *, steps, half_batch, limit, limit_name):
    """Validate an integer sampling schedule and return an int64 copy.

    Requirements: integer dtype (signed or unsigned, never bool/complex/float/object),
    shape exactly ``(steps, half_batch)``, and every entry in ``[0, limit)``.
    """
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must have an integer dtype, not boolean")
    if kind == "c":
        raise TypeError(f"{name} must have an integer dtype, not complex")
    if kind not in ("i", "u"):
        raise TypeError(
            f"{name} must have an integer dtype (signed or unsigned), got dtype {array.dtype!r}"
        )
    if kind == "u" and array.size and int(array.max()) > _INT64_MAX:
        raise OverflowError(
            f"{name} contains unsigned values larger than the int64 maximum and cannot "
            f"be represented losslessly"
        )

    schedule = array.astype(np.int64, copy=True)
    if schedule.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (steps, batch_size // 2), got "
            f"{schedule.ndim} dimension(s)"
        )
    if schedule.shape != (steps, half_batch):
        raise ValueError(
            f"{name} must have shape {(steps, half_batch)} (steps, batch_size // 2), "
            f"got {schedule.shape}"
        )
    if schedule.size == 0:
        raise ValueError(f"{name} must be non-empty")

    minimum = int(schedule.min())
    maximum = int(schedule.max())
    if minimum < 0 or maximum >= limit:
        raise ValueError(
            f"{name} entries must lie in [0, {limit}) ({limit_name}); observed range "
            f"[{minimum}, {maximum}]"
        )
    return schedule


def _as_gram_matrix(value, name, expected_dim):
    """Validate the caller-supplied Gram matrix and return a float64 copy.

    Checks real numeric dtype, ``(expected_dim, expected_dim)`` shape, finiteness,
    symmetry within ``1e-10`` (max absolute deviation), and PSD-ness within
    ``-1e-10`` (minimum eigenvalue).  Nothing is symmetrised or clipped: structural
    violations are errors, never silently repaired.
    """
    array = np.asarray(value)
    kind = array.dtype.kind
    if kind == "b":
        raise TypeError(f"{name} must be a real numeric matrix, not boolean")
    if kind == "c":
        raise TypeError(f"{name} must be a real numeric matrix, not complex")
    if kind not in ("i", "u", "f"):
        raise TypeError(
            f"{name} must be a real numeric matrix, got dtype {array.dtype!r}"
        )

    gram = array.astype(np.float64, copy=True)
    if gram.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (D + 1, D + 1), got {gram.ndim} dimension(s)"
        )
    if gram.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"{name} must have shape {(expected_dim, expected_dim)} (D + 1, D + 1), "
            f"got {gram.shape}"
        )
    if not np.isfinite(gram).all():
        raise ValueError(f"{name} must contain only finite values")

    asymmetry = float(np.max(np.abs(gram - gram.T)))
    if asymmetry > _GRAM_SYMMETRY_TOLERANCE:
        raise ValueError(
            f"{name} must be symmetric: maximum absolute deviation {asymmetry!r} exceeds "
            f"tolerance {_GRAM_SYMMETRY_TOLERANCE!r} (no silent symmetrisation is performed)"
        )

    min_eigenvalue = float(np.linalg.eigvalsh(gram).min())
    if min_eigenvalue < _GRAM_EIGENVALUE_TOLERANCE:
        raise ValueError(
            f"{name} must be positive semi-definite: minimum eigenvalue {min_eigenvalue!r} "
            f"is below tolerance {_GRAM_EIGENVALUE_TOLERANCE!r} (no silent clipping is performed)"
        )
    return gram, asymmetry, min_eigenvalue


def _require_finite(tensor, label, step):
    if not bool(torch.isfinite(tensor.detach()).all()):
        raise FloatingPointError(f"{label} became non-finite at step {step}")


# ---------------------------------------------------------------------------
# Shared loss pieces (used both inside the loop and for the final report).
# ---------------------------------------------------------------------------


def _anchor_distance_squared(weight, bias, parent_weight, parent_bias):
    """``||W - Wp||_F^2 + ||b - bp||^2`` in the dtype of the given tensors."""
    return (weight - parent_weight).square().sum() + (bias - parent_bias).square().sum()


def _output_loss_raw_float64(weight, bias, parent_weight, parent_bias, gram, num_classes):
    """Raw (unweighted) output-protection loss, computed entirely in float64.

    ``weight``/``bias`` may be float32 leaves that require grad: they are cast to
    float64 *before* the subtraction and the result is never detached, so autograd
    carries the gradient back into the float32 parameters.
    """
    weight_delta = weight.to(torch.float64) - parent_weight.to(torch.float64)
    bias_delta = bias.to(torch.float64) - parent_bias.to(torch.float64)
    delta_raw = torch.cat((weight_delta, bias_delta.unsqueeze(1)), dim=1)
    delta_centered = delta_raw - delta_raw.mean(dim=0, keepdim=True)
    return (0.5 / float(num_classes)) * ((delta_centered @ gram) * delta_centered).sum()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_head(
    base_x,
    base_y,
    context_x,
    context_y,
    parent_weight,
    parent_bias,
    base_indices,
    pair_indices,
    gram,
    *,
    steps=660,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=0.001,
    anchor_coefficient=0.01,
    output_coefficient=1.0,
):
    """Train one affine head on frozen CPU features using a provided schedule.

    Parameters
    ----------
    base_x, base_y : ``[N, D]`` float-convertible features and ``[N]`` integer labels.
    context_x, context_y : ``[P, D]`` features and ``[P]`` integer labels.
    parent_weight, parent_bias : ``[C, D]`` and ``[C]`` parent head (``C >= 2``).
        The trained parameters start as exact copies of these arrays.
    base_indices, pair_indices : integer schedules of shape
        ``(steps, batch_size // 2)`` indexing ``base_x`` and ``context_x``.
    gram : symmetric PSD float64 ``[D + 1, D + 1]`` matrix supplied by the caller.
    steps, batch_size : positive ints; ``batch_size`` must be even.
    learning_rate : float > 0.
    weight_decay, anchor_coefficient, output_coefficient : floats >= 0.

    Returns
    -------
    dict with keys ``weight``, ``bias`` (final float32 numpy copies),
    ``pre_update_losses`` (float64 ``[steps, 4]``),
    ``final_anchor_distance_squared`` and ``final_output_loss_raw`` (floats computed
    in float64 from the final parameters), ``schedule`` (int64 copies of both index
    arrays), and ``settings`` (plain metadata).
    """
    # ---- array inputs ----------------------------------------------------
    base_features = _as_float32_array(base_x, "base_x", 2)
    base_labels = _as_label_array(base_y, "base_y")
    context_features = _as_float32_array(context_x, "context_x", 2)
    context_labels = _as_label_array(context_y, "context_y")
    parent_weight_array = _as_float32_array(parent_weight, "parent_weight", 2)
    parent_bias_array = _as_float32_array(parent_bias, "parent_bias", 1)

    base_count, feature_dim = base_features.shape
    context_count, context_dim = context_features.shape
    num_classes, weight_dim = parent_weight_array.shape

    if base_labels.shape[0] != base_count:
        raise ValueError(
            f"base_y must have one label per base_x row: got {base_labels.shape[0]} "
            f"labels for {base_count} rows"
        )
    if context_labels.shape[0] != context_count:
        raise ValueError(
            f"context_y must have one label per context_x row: got "
            f"{context_labels.shape[0]} labels for {context_count} rows"
        )
    if context_dim != feature_dim:
        raise ValueError(
            f"context_x feature dimension {context_dim} does not match base_x "
            f"feature dimension {feature_dim}"
        )
    if weight_dim != feature_dim:
        raise ValueError(
            f"parent_weight feature dimension {weight_dim} does not match base_x "
            f"feature dimension {feature_dim}"
        )
    if parent_bias_array.shape[0] != num_classes:
        raise ValueError(
            f"parent_bias must have one entry per class: got "
            f"{parent_bias_array.shape[0]} for {num_classes} classes"
        )
    if num_classes < 2:
        raise ValueError(f"parent_weight must define at least 2 classes, got {num_classes}")

    _check_label_range(base_labels, "base_y", num_classes)
    _check_label_range(context_labels, "context_y", num_classes)

    present = np.unique(base_labels)
    if present.size != num_classes:
        missing = sorted(set(range(num_classes)) - set(int(v) for v in present.tolist()))
        raise ValueError(
            f"base_y must contain every class in [0, {num_classes}); missing {missing}"
        )

    # ---- scalar settings -------------------------------------------------
    _check_positive_int(steps, "steps")
    _check_positive_int(batch_size, "batch_size")
    steps = int(steps)
    batch_size = int(batch_size)
    if batch_size % 2 != 0:
        raise ValueError(
            f"batch_size must be even so each step takes batch_size // 2 base rows and "
            f"batch_size // 2 context rows, got {batch_size}"
        )
    half_batch = batch_size // 2

    _check_float(learning_rate, "learning_rate", strictly_positive=True)
    _check_float(weight_decay, "weight_decay", strictly_positive=False)
    _check_float(anchor_coefficient, "anchor_coefficient", strictly_positive=False)
    _check_float(output_coefficient, "output_coefficient", strictly_positive=False)
    learning_rate = float(learning_rate)
    weight_decay = float(weight_decay)
    anchor_coefficient = float(anchor_coefficient)
    output_coefficient = float(output_coefficient)

    # ---- schedules and Gram ---------------------------------------------
    base_schedule = _as_index_schedule(
        base_indices,
        "base_indices",
        steps=steps,
        half_batch=half_batch,
        limit=base_count,
        limit_name="base_x rows",
    )
    pair_schedule = _as_index_schedule(
        pair_indices,
        "pair_indices",
        steps=steps,
        half_batch=half_batch,
        limit=context_count,
        limit_name="context_x rows",
    )
    gram_matrix, gram_asymmetry, gram_min_eigenvalue = _as_gram_matrix(
        gram, "gram", feature_dim + 1
    )

    # ---- tensors (all validation is complete before this point) ----------
    base_features_t = torch.from_numpy(base_features)
    base_labels_t = torch.from_numpy(base_labels)
    context_features_t = torch.from_numpy(context_features)
    context_labels_t = torch.from_numpy(context_labels)
    base_schedule_t = torch.from_numpy(base_schedule)
    pair_schedule_t = torch.from_numpy(pair_schedule)
    gram_t = torch.from_numpy(gram_matrix)

    # Frozen parent copies: detached leaves, never written to, no gradient.
    frozen_weight = torch.tensor(parent_weight_array, dtype=torch.float32)
    frozen_bias = torch.tensor(parent_bias_array, dtype=torch.float32)

    # Trainable leaves: exact copies of the parent arrays, no random init.
    weight = torch.tensor(parent_weight_array, dtype=torch.float32, requires_grad=True)
    bias = torch.tensor(parent_bias_array, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.AdamW(
        [weight, bias],
        lr=learning_rate,
        betas=_ADAMW_BETAS,
        eps=_ADAMW_EPS,
        weight_decay=weight_decay,
    )

    pre_update_losses = np.empty((steps, 4), dtype=np.float64)

    # ---- training loop: exactly `steps` optimizer steps ------------------
    with torch.enable_grad():
        for step in range(steps):
            base_rows = base_schedule_t[step]
            pair_rows = pair_schedule_t[step]
            batch_features = torch.cat(
                (
                    base_features_t.index_select(0, base_rows),
                    context_features_t.index_select(0, pair_rows),
                ),
                dim=0,
            )
            batch_labels = torch.cat(
                (
                    base_labels_t.index_select(0, base_rows),
                    context_labels_t.index_select(0, pair_rows),
                ),
                dim=0,
            )

            logits = F.linear(batch_features, weight, bias)
            cross_entropy = F.cross_entropy(logits, batch_labels, reduction="mean")

            anchor = 0.5 * anchor_coefficient * _anchor_distance_squared(
                weight, bias, frozen_weight, frozen_bias
            )
            output_raw = _output_loss_raw_float64(
                weight, bias, frozen_weight, frozen_bias, gram_t, num_classes
            )
            output_penalty = output_coefficient * output_raw
            total = cross_entropy + anchor + output_penalty

            _require_finite(cross_entropy, "cross_entropy loss", step)
            _require_finite(anchor, "weighted_parameter_anchor loss", step)
            _require_finite(output_penalty, "weighted_output_penalty loss", step)
            _require_finite(total, "total loss", step)

            pre_update_losses[step, 0] = float(cross_entropy.detach())
            pre_update_losses[step, 1] = float(anchor.detach())
            pre_update_losses[step, 2] = float(output_penalty.detach())
            pre_update_losses[step, 3] = float(total.detach())

            optimizer.zero_grad(set_to_none=True)
            total.backward()

            for label, parameter in (("weight", weight), ("bias", bias)):
                gradient = parameter.grad
                if gradient is None:
                    raise FloatingPointError(
                        f"{label} received no gradient at step {step}"
                    )
                if not bool(torch.isfinite(gradient).all()):
                    raise FloatingPointError(
                        f"{label} gradient became non-finite at step {step}"
                    )

            optimizer.step()

            if not bool(torch.isfinite(weight.detach()).all()):
                raise FloatingPointError(f"weight became non-finite after step {step}")
            if not bool(torch.isfinite(bias.detach()).all()):
                raise FloatingPointError(f"bias became non-finite after step {step}")

    # ---- final float64 diagnostics --------------------------------------
    with torch.no_grad():
        final_weight_t = weight.detach()
        final_bias_t = bias.detach()
        final_anchor_distance_squared = float(
            _anchor_distance_squared(
                final_weight_t.to(torch.float64),
                final_bias_t.to(torch.float64),
                frozen_weight.to(torch.float64),
                frozen_bias.to(torch.float64),
            )
        )
        final_output_loss_raw = float(
            _output_loss_raw_float64(
                final_weight_t, final_bias_t, frozen_weight, frozen_bias, gram_t, num_classes
            )
        )

    if not np.isfinite(final_anchor_distance_squared):
        raise FloatingPointError("final_anchor_distance_squared is non-finite")
    if not np.isfinite(final_output_loss_raw):
        raise FloatingPointError("final_output_loss_raw is non-finite")

    settings = {
        "task": "single affine head on frozen CPU features with a provided schedule",
        "device": "cpu",
        "steps": steps,
        "optimizer_steps_taken": steps,
        "batch_size": batch_size,
        "half_batch_size": half_batch,
        "batch_composition": "base_x[base_indices[t]] rows then context_x[pair_indices[t]] rows",
        "num_classes": int(num_classes),
        "feature_dim": int(feature_dim),
        "base_sample_count": int(base_count),
        "context_sample_count": int(context_count),
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "anchor_coefficient": anchor_coefficient,
        "output_coefficient": output_coefficient,
        "optimizer": "torch.optim.AdamW",
        "optimizer_betas": list(_ADAMW_BETAS),
        "optimizer_eps": _ADAMW_EPS,
        "optimizer_parameters": ["weight", "bias"],
        "gradient_reset": "zero_grad(set_to_none=True)",
        "loss_columns": list(_LOSS_COLUMNS),
        "loss_record_dtype": "float64",
        "loss_record_timing": "recorded before optimizer.step",
        "cross_entropy": "F.cross_entropy(F.linear(batch, W, b), labels, reduction='mean')",
        "cross_entropy_dtype": "float32",
        "anchor_formula": "0.5 * anchor_coefficient * (sum((W - Wp)^2) + sum((b - bp)^2))",
        "anchor_dtype": "float32",
        "output_penalty_formula": (
            "output_coefficient * 0.5 / C * sum((D_centered @ gram) * D_centered)"
        ),
        "output_penalty_dtype": "float64",
        "output_penalty_delta": "D_raw = [W - Wp, (b - bp)[:, None]] cast to float64 before subtraction",
        "output_penalty_centering": "D_raw minus its mean over the C output classes",
        "output_penalty_differentiable": True,
        "total_loss_formula": "cross_entropy + weighted_parameter_anchor + weighted_output_penalty",
        "parameter_dtype": "float32",
        "feature_dtype": "float32",
        "label_dtype": "int64",
        "schedule_dtype": "int64",
        "gram_dtype": "float64",
        "gram_shape": [int(feature_dim + 1), int(feature_dim + 1)],
        "gram_source": "caller-supplied class-balanced second moment of augmented original base features [x, 1]",
        "gram_symmetry_max_abs_deviation": gram_asymmetry,
        "gram_symmetry_tolerance": _GRAM_SYMMETRY_TOLERANCE,
        "gram_min_eigenvalue": gram_min_eigenvalue,
        "gram_eigenvalue_tolerance": _GRAM_EIGENVALUE_TOLERANCE,
        "gram_recomputed_or_modified": False,
        "parameter_init": "exact copy of parent_weight / parent_bias (no random init)",
        "parent_copies": "detached float32 leaves, requires_grad=False, never mutated",
        "uses_random_state": False,
        "modifies_global_state": False,
        "mutates_inputs": False,
        "uses_scheduler_or_gradient_clipping": False,
        "uses_early_stopping": False,
        "per_class_special_casing": False,
        "final_output_loss_raw_is_unweighted": True,
    }

    return {
        "weight": final_weight_t.numpy().copy(),
        "bias": final_bias_t.numpy().copy(),
        "pre_update_losses": pre_update_losses,
        "final_anchor_distance_squared": final_anchor_distance_squared,
        "final_output_loss_raw": final_output_loss_raw,
        "schedule": {
            "base_indices": base_schedule.copy(),
            "pair_indices": pair_schedule.copy(),
        },
        "settings": settings,
    }
