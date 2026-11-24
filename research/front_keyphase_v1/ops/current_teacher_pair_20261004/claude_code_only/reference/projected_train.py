"""Head-only learner with split old/extra gradients and an optional one-sided projection.

Forward and loss arithmetic are copied from train_addition.train_arm
(ops/target_input_adaptation_20260928). The only change: instead of total.backward(),
    g_old   = grad(CE32 + weighted anchor + weighted output)      (old objective, saved before extra)
    g_extra = grad(extra_weight * CE(extra rows))
are taken separately; with project_extra=True, g_extra loses its component along g_old when
<g_old, g_extra> < 0 (head weight+bias as one vector, float64). AdamW receives g_old + g_extra'.
The matched control uses the same split path with the projection disabled.

Old modules are loaded from explicit FRONT paths under private names; nothing old is modified.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
FRONT = Path(__file__).resolve().parents[2]
OLD_OPS = FRONT / "ops" / "target_input_adaptation_20260928"
OLD_EXPERIMENT = OLD_OPS / "experiment.py"
OLD_TRAIN = OLD_OPS / "train_addition.py"
OLD_GENERATED = FRONT / "ops" / "partial_encoder_pair_20260922" / "generated_v1.py"
_E_NAME = "gpp_old_tia_experiment"
HEAD = "head."
SCRIPTS = ("projected_train.py", "run_pair.py", "test_projection.py")
LOSS_COLUMNS = ["old_ce", "extra_ce", "weighted_extra", "weighted_anchor", "weighted_output", "total"]
DIAG_COLUMNS = ["old_norm", "new_norm", "dot_before", "dot_after", "removed_norm", "projected",
                "old_dot_actual_delta"]


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_experiment():
    """Old experiment.py (exposes E.T, E.G, load_models, load_prior_cache, load_new, ...)."""
    if os.environ.get("TIA_PRIOR_GENERATED"):
        raise RuntimeError("TIA_PRIOR_GENERATED override is not allowed")
    mod = sys.modules.get(_E_NAME)
    if mod is None:
        if not OLD_EXPERIMENT.is_file():
            raise ImportError(f"old experiment.py not found at {OLD_EXPERIMENT}")
        spec = importlib.util.spec_from_file_location(_E_NAME, OLD_EXPERIMENT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[_E_NAME] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(_E_NAME, None)
            raise
    bound = {"experiment": (Path(mod.__file__).resolve(), OLD_EXPERIMENT.resolve()),
             "train_addition": (Path(mod.T.__file__).resolve(), OLD_TRAIN.resolve()),
             "generated_v1": (Path(mod.G.__file__).resolve(), OLD_GENERATED.resolve()),
             "prior_generated": (Path(mod.T.PRIOR_GENERATED).resolve(), OLD_GENERATED.resolve())}
    bad = {k: str(a) for k, (a, b) in bound.items() if a != b}
    if bad:
        raise ImportError(f"old modules bound to unexpected files: {bad}")
    if list(mod.T.LOSS_COLUMNS) != LOSS_COLUMNS:
        raise ImportError("old LOSS_COLUMNS differ")
    return mod


def code_hashes() -> Dict[str, str]:
    out = {n: _sha(HERE / n) for n in SCRIPTS}
    out.update({"old/experiment.py": _sha(OLD_EXPERIMENT), "old/train_addition.py": _sha(OLD_TRAIN),
                "old/generated_v1.py": _sha(OLD_GENERATED)})
    return out


# --------------------------------------------------------------------------- #
# projection
# --------------------------------------------------------------------------- #
def _flat64(gs):
    return torch.cat([g.detach().reshape(-1).to(torch.float64) for g in gs])


def project_extra_gradients(old_grads: Sequence[torch.Tensor], extra_grads: Sequence[torch.Tensor],
                            enabled: bool):
    """One-sided projection of the extra gradient against the old-objective gradient.

    All tensors form ONE global vector (e.g. head.weight and head.bias together); dot and squared
    norm are float64. If enabled and dot < 0 and |g_old|^2 > 0:
        g_extra' = g_extra - dot / |g_old|^2 * g_old
    otherwise g_extra' = g_extra. Results are new tensors in each extra tensor's dtype; inputs are
    never modified. Diagnostics are computed on the returned (dtype round-tripped) tensors, so
    dot_after is ~0 only up to that rounding. Non-finite input, float64 norm/dot overflow, projection,
    output or diagnostic raises RuntimeError (both enabled and disabled).
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be bool")
    old, ext = list(old_grads), list(extra_grads)
    if not old or len(old) != len(ext):
        raise ValueError("old/extra gradient lists must be non-empty and equally long")
    for a, b in zip(old, ext):
        if a.shape != b.shape:
            raise ValueError(f"gradient shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    o64, e64 = _flat64(old), _flat64(ext)
    if not bool(torch.isfinite(o64).all()) or not bool(torch.isfinite(e64).all()):
        raise RuntimeError("non-finite gradient before projection")
    n = torch.dot(o64, o64)
    d = torch.dot(o64, e64)
    if not bool(torch.isfinite(n)) or not bool(torch.isfinite(d)):
        raise RuntimeError("non-finite gradient norm/dot (float64 overflow)")
    project = enabled and bool(d < 0) and bool(n > 0)
    if project:
        coef = d / n
        a64 = e64 - coef * o64
        if not bool(torch.isfinite(coef)) or not bool(torch.isfinite(a64).all()):
            raise RuntimeError("non-finite projection")
        adjusted, off = [], 0
        for e in ext:
            k = e.numel()
            adjusted.append(a64[off:off + k].reshape(e.shape).to(e.dtype).clone())
            off += k
    else:
        adjusted = [e.detach().clone() for e in ext]
    r64 = _flat64(adjusted)
    if not bool(torch.isfinite(r64).all()):
        raise RuntimeError("non-finite adjusted gradient")
    info = {"old_norm": float(torch.sqrt(n)), "new_norm": float(torch.linalg.vector_norm(e64)),
            "dot_before": float(d), "dot_after": float(torch.dot(o64, r64)),
            "removed_norm": float(torch.linalg.vector_norm(e64 - r64)), "projected": float(project)}
    bad = [k for k, v in info.items() if not math.isfinite(v)]
    if bad:
        raise RuntimeError(f"non-finite projection diagnostics {bad}")
    return adjusted, info


# --------------------------------------------------------------------------- #
# one arm (train_addition.train_arm + split gradients)
# --------------------------------------------------------------------------- #
def train_arm(model: nn.Module,
              base_prefixes: Sequence[np.ndarray], base_y: np.ndarray,
              context_prefixes: Sequence[np.ndarray], context_y: np.ndarray,
              teacher_base_logits: np.ndarray,
              base_indices: np.ndarray, context_indices: np.ndarray,
              *,
              extra_prefixes: Optional[Sequence[np.ndarray]] = None,
              extra_y: Optional[np.ndarray] = None,
              extra_indices: Optional[np.ndarray] = None,
              extra_weight: float = 0.0,
              learning_rate: float = 1e-3, weight_decay: float = 1e-3,
              anchor_coefficient: float = 1e-2, output_coefficient: float = 1.0,
              arm: str = "control",
              progress: Optional[Callable[[str, int, int], None]] = None,
              project_extra: bool = False,
              G=None) -> Dict[str, object]:
    G = load_experiment().G if G is None else G
    _require = G._require
    width, _, num_classes = G._inspect_model(model)
    base = G._check_matrix_list(base_prefixes, width, "base_prefixes")
    ctx = G._check_matrix_list(context_prefixes, width, "context_prefixes")
    by = G._check_labels(base_y, len(base), num_classes, "base_y")
    cy = G._check_labels(context_y, len(ctx), num_classes, "context_y")
    teacher = G._check_teacher(teacher_base_logits, len(base), num_classes)
    bidx = G._check_schedule(base_indices, len(base), "base_indices")
    cidx = G._check_schedule(context_indices, len(ctx), "context_indices")
    _require(bidx.shape == cidx.shape, "base_indices and context_indices must share the same shape")
    has_extra = extra_indices is not None
    if has_extra:
        ext = G._check_matrix_list(extra_prefixes, width, "extra_prefixes")
        ey = G._check_labels(extra_y, len(ext), num_classes, "extra_y")
        eidx = G._check_schedule(extra_indices, len(ext), "extra_indices")
        _require(eidx.shape[0] == bidx.shape[0], "extra_indices must have one row per step")
    else:
        _require(extra_prefixes is None and extra_y is None, "extra data given without extra_indices")
    w = G._check_scalar(extra_weight, "extra_weight")
    _require(has_extra or w == 0.0, "extra_weight > 0 requires extra data")
    lr = G._check_scalar(learning_rate, "learning_rate", positive=True)
    wd = G._check_scalar(weight_decay, "weight_decay")
    anchor_c = G._check_scalar(anchor_coefficient, "anchor_coefficient")
    output_c = G._check_scalar(output_coefficient, "output_coefficient")
    _require(progress is None or callable(progress), "progress must be callable or None", TypeError)
    _require(isinstance(project_extra, bool), "project_extra must be bool", TypeError)

    steps, half = int(bidx.shape[0]), int(bidx.shape[1])
    last_index = len(model.blocks) - 1
    source_p, source_b = G._snapshot(model)

    work = copy.deepcopy(model)
    work.eval()
    for p in work.parameters():
        p.requires_grad_(False)
    names, params = [], []
    for n, p in work.named_parameters():
        if n.startswith(HEAD):
            p.requires_grad_(True)
            names.append(n)
            params.append(p)
    _require(len(params) > 0, f"{arm}: no head parameters")
    init_p, init_b = G._snapshot(work)
    anchors = [init_p[n] for n in names]
    frozen_names = [n for n, _ in work.named_parameters() if n not in set(names)]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    losses = np.full((steps, len(LOSS_COLUMNS)), np.nan, dtype=np.float64)
    diags = np.full((steps, len(DIAG_COLUMNS)), np.nan, dtype=np.float64)
    block = work.blocks[last_index]

    with torch.enable_grad():
        for t in range(steps):
            # ---- old 32-row batch: verbatim train_addition arithmetic
            brow, crow = bidx[t], cidx[t]
            clips = [base[int(i)] for i in brow] + [ctx[int(j)] for j in crow]
            labels = torch.from_numpy(np.concatenate((by[brow], cy[crow])))
            h, m = G._pad_batch(clips)
            logits = work.head(G._masked_mean(block(h, m), m))

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
            old_objective = total                      # saved BEFORE the extra term is added
            checks = [("cross_entropy", ce), ("anchor", anchor), ("output", output)]

            # ---- added rows: separate forward, separate padding
            if has_extra:
                erow = eidx[t]
                eh, em = G._pad_batch([ext[int(i)] for i in erow])
                elogits = work.head(G._masked_mean(block(eh, em), em))
                extra_ce = F.cross_entropy(elogits, torch.from_numpy(np.ascontiguousarray(ey[erow])))
                weighted_extra = extra_ce.to(torch.float64) * w
                total = total + weighted_extra
                checks += [("extra_cross_entropy", extra_ce), ("weighted_extra", weighted_extra)]
            checks.append(("total", total))
            for label, value in checks:
                _require(bool(torch.isfinite(value)), f"{arm} step {t}: non-finite {label} loss", RuntimeError)
            losses[t, [0, 3, 4, 5]] = (float(ce), float(anchor), float(output), float(total))
            if has_extra:
                losses[t, 1:3] = (float(extra_ce), float(weighted_extra))

            # ---- split gradients (replaces total.backward())
            g_old = torch.autograd.grad(old_objective, params)
            if has_extra:
                g_extra = torch.autograd.grad(weighted_extra, params)
            else:
                g_extra = tuple(torch.zeros_like(p) for p in params)
            for tag, gs in (("old", g_old), ("extra", g_extra)):
                for n, g in zip(names, gs):
                    _require(bool(torch.isfinite(g).all()),
                             f"{arm} step {t}: non-finite {tag} gradient for '{n}'", RuntimeError)
            try:
                adjusted, info = project_extra_gradients(g_old, g_extra, project_extra)
            except RuntimeError as e:
                raise RuntimeError(f"{arm} step {t}: {e}") from None
            _require(all(math.isfinite(v) for v in info.values()),
                     f"{arm} step {t}: non-finite projection diagnostic", RuntimeError)
            final = [go.to(p.dtype) + ga.to(p.dtype) for go, ga, p in zip(g_old, adjusted, params)]
            for n, g in zip(names, final):
                _require(bool(torch.isfinite(g).all()),
                         f"{arm} step {t}: non-finite final gradient for '{n}'", RuntimeError)

            before = [p.detach().clone() for p in params]
            opt.zero_grad(set_to_none=True)
            for p, g in zip(params, final):
                p.grad = g
            opt.step()
            moved = _flat64([p.detach() - b for p, b in zip(params, before)])
            info["old_dot_actual_delta"] = float(torch.dot(_flat64(g_old), moved))
            _require(math.isfinite(info["old_dot_actual_delta"]),
                     f"{arm} step {t}: non-finite old_dot_actual_delta", RuntimeError)
            diags[t] = [info[c] for c in DIAG_COLUMNS]

            done = t + 1
            if progress is not None and (done % 100 == 0 or done == steps):
                progress(arm, done, steps)

    frozen_ok = G._params_match(work, init_p, skip=set(names))
    buffers_ok = G._buffers_match(work, init_b)
    _require(frozen_ok, f"{arm}: frozen parameters were modified", RuntimeError)
    _require(buffers_ok, f"{arm}: buffers were modified", RuntimeError)
    _require(G._params_match(model, source_p) and G._buffers_match(model, source_b),
             f"{arm}: the input model was modified", RuntimeError)
    return {"state_dict": {k: v.detach().cpu().clone() for k, v in work.state_dict().items()},
            "losses": losses,
            "loss_columns": list(LOSS_COLUMNS),
            "diagnostics": diags,
            "diagnostic_columns": list(DIAG_COLUMNS),
            "project_extra": project_extra,
            "trainable_names": list(names),
            "frozen_names": frozen_names,
            "updates": steps,
            "extra_weight": w,
            "base_indices": bidx.copy(), "context_indices": cidx.copy(),
            "extra_indices": eidx.copy() if has_extra else None,
            "frozen_parameters_unchanged": bool(frozen_ok),
            "buffers_unchanged": bool(buffers_ok)}
