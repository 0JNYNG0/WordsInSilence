"""Prefix extraction, prefix prediction, and paired control/treatment fine-tuning
for masked 1-D encoder models (proj -> blocks -> masked mean -> head).

The architecture itself is supplied by the caller and is never modified here:
this module only re-uses `model.proj`, `model.blocks[i]` and `model.head` with
the masking convention h[B, width, T], m[B, 1, T].
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["extract_prefix", "predict_prefix", "run_pair"]


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
def _require(cond: bool, msg: str, err=ValueError) -> None:
    if not cond:
        raise err(msg)


def _inspect_model(model):
    """Validate the model contract; return (width, in_channels, num_classes)."""
    _require(isinstance(model, nn.Module), "model must be an nn.Module", TypeError)
    _require(isinstance(getattr(model, "proj", None), nn.Conv1d), "model.proj must be nn.Conv1d", TypeError)
    _require(isinstance(getattr(model, "blocks", None), nn.ModuleList), "model.blocks must be nn.ModuleList", TypeError)
    _require(len(model.blocks) > 0, "model.blocks must be non-empty")
    _require(isinstance(getattr(model, "head", None), nn.Linear), "model.head must be nn.Linear", TypeError)
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        _require(t.device.type == "cpu", f"tensor '{name}' must live on CPU")
        _require(not t.is_floating_point() or t.dtype == torch.float32, f"float tensor '{name}' must be float32")
    width = int(model.proj.out_channels)
    _require(int(model.head.in_features) == width, "head.in_features must equal proj.out_channels")
    return width, int(model.proj.in_channels), int(model.head.out_features)


def _check_matrix(a, cols: int, label: str) -> np.ndarray:
    _require(isinstance(a, np.ndarray), f"{label} must be a numpy array", TypeError)
    _require(a.dtype == np.float32, f"{label} must be float32", TypeError)
    _require(a.ndim == 2, f"{label} must be 2-D [T, C]")
    _require(a.shape[0] >= 1 and a.shape[1] == cols, f"{label} must have shape [T>=1, {cols}], got {tuple(a.shape)}")
    _require(bool(np.isfinite(a).all()), f"{label} contains non-finite values")
    return np.ascontiguousarray(a)


def _check_matrix_list(items, cols: int, label: str) -> List[np.ndarray]:
    _require(isinstance(items, (list, tuple)), f"{label} must be a list or tuple", TypeError)
    _require(len(items) > 0, f"{label} must be non-empty")
    return [_check_matrix(a, cols, f"{label}[{i}]") for i, a in enumerate(items)]


def _check_labels(y, n: int, num_classes: int, label: str) -> np.ndarray:
    _require(isinstance(y, np.ndarray), f"{label} must be a numpy array", TypeError)
    _require(y.dtype == np.int64, f"{label} must be int64", TypeError)
    _require(y.ndim == 1 and y.shape[0] == n, f"{label} must be 1-D with length {n}")
    _require(int(y.min()) >= 0 and int(y.max()) < num_classes, f"{label} holds out-of-range class ids")
    return y


def _check_schedule(s, n: int, label: str) -> np.ndarray:
    _require(isinstance(s, np.ndarray), f"{label} must be a numpy array", TypeError)
    _require(s.dtype == np.int64, f"{label} must be int64", TypeError)
    _require(s.ndim == 2, f"{label} must be 2-D [steps, half_batch]")
    _require(s.shape[0] >= 1 and s.shape[1] >= 1, f"{label} must be non-empty in both dimensions")
    _require(int(s.min()) >= 0 and int(s.max()) < n, f"{label} holds out-of-range indices")
    return s


def _check_teacher(t, n: int, c: int) -> torch.Tensor:
    _require(isinstance(t, np.ndarray), "teacher_base_logits must be a numpy array", TypeError)
    _require(t.dtype in (np.float64, np.float32), "teacher_base_logits must be float64 or float32", TypeError)
    _require(t.ndim == 2 and t.shape == (n, c), f"teacher_base_logits must have shape [{n}, {c}]")
    _require(bool(np.isfinite(t).all()), "teacher_base_logits contains non-finite values")
    return torch.from_numpy(np.ascontiguousarray(t, dtype=np.float64)).clone()


def _check_scalar(v, label: str, positive: bool = False) -> float:
    _require(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool),
             f"{label} must be a real number", TypeError)
    v = float(v)
    _require(math.isfinite(v), f"{label} must be finite")
    _require(v > 0.0 if positive else v >= 0.0, f"{label} must be {'positive' if positive else 'non-negative'}")
    return v


# --------------------------------------------------------------------------- #
# tensor helpers
# --------------------------------------------------------------------------- #
def _pad_batch(arrays: Sequence[np.ndarray]):
    """Zero-pad [T, width] clips into (h[B, width, Tmax], m[B, 1, Tmax])."""
    b = len(arrays)
    t_max = max(int(a.shape[0]) for a in arrays)
    width = int(arrays[0].shape[1])
    h = np.zeros((b, width, t_max), dtype=np.float32)
    m = np.zeros((b, 1, t_max), dtype=np.float32)
    for i, a in enumerate(arrays):
        t = int(a.shape[0])
        h[i, :, :t] = a.T
        m[i, 0, :t] = 1.0
    return torch.from_numpy(h), torch.from_numpy(m)


def _masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (h * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)


def _snapshot(module: nn.Module):
    return ({n: p.detach().clone() for n, p in module.named_parameters()},
            {n: b.detach().clone() for n, b in module.named_buffers()})


def _params_match(module, ref, skip=()) -> bool:
    return all(torch.equal(p.detach(), ref[n]) for n, p in module.named_parameters() if n not in skip)


def _buffers_match(module, ref) -> bool:
    return all(torch.equal(b.detach(), ref[n]) for n, b in module.named_buffers())


# --------------------------------------------------------------------------- #
# 1) prefix extraction
# --------------------------------------------------------------------------- #
def extract_prefix(model: nn.Module, sequences: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Run proj and every block except the last, one unpadded clip at a time."""
    _, in_channels, _ = _inspect_model(model)
    clips = _check_matrix_list(sequences, in_channels, "sequences")
    work = copy.deepcopy(model).eval()
    n_blocks = len(work.blocks)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for clip in clips:
            t = int(clip.shape[0])
            x = torch.from_numpy(clip).clone().unsqueeze(0)          # [1, T, C]
            m = torch.ones(1, 1, t, dtype=torch.float32)
            h = work.proj(x.transpose(1, 2) * m) * m
            for i in range(n_blocks - 1):
                h = work.blocks[i](h, m)
            prefix = h.squeeze(0).transpose(0, 1).contiguous()        # [T, width]
            out.append(np.array(prefix.numpy(), dtype=np.float32, copy=True))
    return out


# --------------------------------------------------------------------------- #
# 2) prefix prediction
# --------------------------------------------------------------------------- #
def predict_prefix(model: nn.Module, prefixes: Sequence[np.ndarray], batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Last block -> masked temporal mean -> head, over zero-padded prefix batches."""
    width, _, _ = _inspect_model(model)
    clips = _check_matrix_list(prefixes, width, "prefixes")
    _require(isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, bool),
             "batch_size must be an integer", TypeError)
    batch_size = int(batch_size)
    _require(batch_size >= 1, "batch_size must be >= 1")
    work = copy.deepcopy(model).eval()
    block = work.blocks[len(work.blocks) - 1]
    pooled_chunks, logit_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(clips), batch_size):
            h, m = _pad_batch(clips[start:start + batch_size])
            pooled = _masked_mean(block(h, m), m)
            logits = work.head(pooled)
            pooled_chunks.append(np.array(pooled.numpy(), dtype=np.float32, copy=True))
            logit_chunks.append(np.array(logits.numpy(), dtype=np.float32, copy=True))
    return {"pooled": np.concatenate(pooled_chunks, axis=0).astype(np.float32, copy=False),
            "logits": np.concatenate(logit_chunks, axis=0).astype(np.float32, copy=False)}


# --------------------------------------------------------------------------- #
# 3) paired control / treatment fine-tuning
# --------------------------------------------------------------------------- #
def run_pair(model: nn.Module,
             base_prefixes: Sequence[np.ndarray],
             base_y: np.ndarray,
             context_prefixes: Sequence[np.ndarray],
             context_y: np.ndarray,
             teacher_base_logits: np.ndarray,
             base_indices: np.ndarray,
             context_indices: np.ndarray,
             *,
             learning_rate: float = 1e-3,
             weight_decay: float = 1e-3,
             anchor_coefficient: float = 1e-2,
             output_coefficient: float = 1.0,
             progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, object]:
    width, _, num_classes = _inspect_model(model)
    base = _check_matrix_list(base_prefixes, width, "base_prefixes")
    ctx = _check_matrix_list(context_prefixes, width, "context_prefixes")
    by = _check_labels(base_y, len(base), num_classes, "base_y")
    cy = _check_labels(context_y, len(ctx), num_classes, "context_y")
    teacher = _check_teacher(teacher_base_logits, len(base), num_classes)
    bidx = _check_schedule(base_indices, len(base), "base_indices")
    cidx = _check_schedule(context_indices, len(ctx), "context_indices")
    _require(bidx.shape == cidx.shape, "base_indices and context_indices must share the same shape")
    lr = _check_scalar(learning_rate, "learning_rate", positive=True)
    wd = _check_scalar(weight_decay, "weight_decay")
    anchor_c = _check_scalar(anchor_coefficient, "anchor_coefficient")
    output_c = _check_scalar(output_coefficient, "output_coefficient")
    _require(progress is None or callable(progress), "progress must be callable or None", TypeError)

    steps, half = int(bidx.shape[0]), int(bidx.shape[1])
    last_index = len(model.blocks) - 1
    head_prefix, block_prefix = "head.", f"blocks.{last_index}."
    source_p, source_b = _snapshot(model)

    def _run_arm(arm: str, trainable_prefixes) -> Dict[str, object]:
        work = copy.deepcopy(model)
        work.eval()                                   # eval for the whole run, incl. autograd
        for p in work.parameters():
            p.requires_grad_(False)
        names, params = [], []
        for n, p in work.named_parameters():
            if any(n.startswith(pfx) for pfx in trainable_prefixes):
                p.requires_grad_(True)
                names.append(n)
                params.append(p)
        _require(len(params) > 0, f"{arm}: no trainable parameters matched {trainable_prefixes}")
        init_p, init_b = _snapshot(work)
        anchors = [init_p[n] for n in names]
        frozen_names = [n for n, _ in work.named_parameters() if n not in set(names)]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
        losses = np.zeros((steps, 4), dtype=np.float64)
        block = work.blocks[last_index]

        with torch.enable_grad():
            for t in range(steps):
                brow, crow = bidx[t], cidx[t]
                clips = [base[int(i)] for i in brow] + [ctx[int(j)] for j in crow]
                labels = torch.from_numpy(np.concatenate((by[brow], cy[crow])))
                h, m = _pad_batch(clips)
                logits = work.head(_masked_mean(block(h, m), m))

                ce = F.cross_entropy(logits, labels)
                sq = torch.zeros((), dtype=torch.float32)
                for p, p0 in zip(params, anchors):
                    sq = sq + (p - p0).pow(2).sum()
                anchor = 0.5 * anchor_c * sq
                rows = torch.from_numpy(np.ascontiguousarray(brow))
                delta = logits[:half].to(torch.float64) - teacher.index_select(0, rows)
                delta = delta - delta.mean(dim=1, keepdim=True)
                output = 0.5 * output_c * delta.pow(2).mean()
                total = ce.to(torch.float64) + anchor.to(torch.float64) + output

                for label, value in (("cross_entropy", ce), ("anchor", anchor),
                                     ("output", output), ("total", total)):
                    _require(bool(torch.isfinite(value)), f"{arm} step {t}: non-finite {label} loss", RuntimeError)
                losses[t] = (float(ce), float(anchor), float(output), float(total))

                opt.zero_grad(set_to_none=True)
                total.backward()
                for n, p in zip(names, params):
                    if p.grad is not None:
                        _require(bool(torch.isfinite(p.grad).all()),
                                 f"{arm} step {t}: non-finite gradient for '{n}'", RuntimeError)
                opt.step()

                done = t + 1
                if progress is not None and (done % 100 == 0 or done == steps):
                    progress(arm, done, steps)

        frozen_ok = _params_match(work, init_p, skip=set(names))
        buffers_ok = _buffers_match(work, init_b)
        _require(frozen_ok, f"{arm}: frozen parameters were modified", RuntimeError)
        _require(buffers_ok, f"{arm}: buffers were modified", RuntimeError)
        return {"state_dict": {k: v.detach().cpu().clone() for k, v in work.state_dict().items()},
                "losses": losses,
                "trainable_names": list(names),
                "frozen_names": frozen_names,
                "updates": steps,
                "frozen_parameters_unchanged": bool(frozen_ok),
                "buffers_unchanged": bool(buffers_ok)}

    control = _run_arm("control", (head_prefix,))
    treatment = _run_arm("treatment", (head_prefix, block_prefix))

    _require(_params_match(model, source_p) and _buffers_match(model, source_b),
             "the input model was modified during run_pair", RuntimeError)

    settings = {
        "objective": ("cross_entropy(mean over full batch) "
                      "+ 0.5*anchor_coefficient*sum((theta-theta0)^2 over trainable params) "
                      "+ 0.5*output_coefficient*mean(center_over_classes(student-teacher)^2 on base half)"),
        "output_penalty": {"scope": "base_half_only", "centered_over_classes": True,
                           "dtype": "float64", "teacher": "fixed, never updated",
                           "differentiable_through": "student"},
        "optimizer": {"name": "AdamW", "learning_rate": lr, "weight_decay": wd,
                      "betas": (0.9, 0.999), "eps": 1e-8, "grad_clipping": None,
                      "scheduler": None, "early_stopping": False},
        "anchor_coefficient": anchor_c,
        "output_coefficient": output_c,
        "steps": steps,
        "half_batch": half,
        "batch_size": 2 * half,
        "width": width,
        "num_classes": num_classes,
        "last_block_index": last_index,
        "trainable_scopes": {"control": [head_prefix], "treatment": [head_prefix, block_prefix]},
        "loss_columns": ["ce", "weighted_anchor", "weighted_output", "total"],
        "losses_recorded": "pre-update",
        "module_mode": "eval (dropout off, normalization running stats fixed)",
        "device": "cpu",
        "dtype": "float32",
        "deterministic": True,
    }
    return {"control": control, "treatment": treatment, "settings": settings,
            "schedule": {"base_indices": bidx.copy(), "context_indices": cidx.copy()}}
