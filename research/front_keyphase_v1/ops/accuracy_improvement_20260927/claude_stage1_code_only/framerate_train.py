"""Stage 1: paired head-only fine-tuning with one fixed per-draw input replacement.

The control arm is the head-only ("control") arm of the prior partial-encoder
`generated_v1.run_pair`, with the same arithmetic: same deepcopy/eval/requires_grad
set-up, AdamW settings, batch assembly (16 base draws then 16 context draws, zero padded),
frozen last block -> masked mean -> head, and the same loss
    CE(32) + 0.5*anchor*sum((theta-theta0)^2) + 0.5*output*mean(centred(student-teacher)^2 on 16 base).
The candidate arm differs in exactly one place: for a base draw (t, k) with coins[t, k] True
AND eligible[base_indices[t, k]] True, the clip fed to the frozen last block is that row's
decimated prefix instead of its original prefix. Labels, teacher rows (fixed parent logits
of the ORIGINAL input), context draws, schedule, optimiser and loss are identical. The
control arm receives the same coin array and ignores it.

Both arms train `head.*` only. No interpolation, no extra loss, no coefficient search.
The prior module's validation / padding / snapshot helpers are reused unchanged; that
module is loaded from its explicit path under a private module name (no sys.path change).
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["G", "PRIOR_GENERATED", "COIN_SEED", "COIN_SHAPE", "make_coins", "replacement_mask",
           "select_batch", "run_framerate_pair", "prior_module_sha256"]

PRIOR_GENERATED = Path(__file__).resolve().parents[1] / "partial_encoder_pair_20260922" / "generated_v1.py"
_PRIOR_NAME = "fr_prior_generated_v1"
COIN_SEED = 20260927
COIN_SHAPE = (660, 16)
HEAD_PREFIX = "head."


def _load_prior():
    mod = sys.modules.get(_PRIOR_NAME)
    if mod is not None:
        if Path(mod.__file__).resolve() != PRIOR_GENERATED:
            raise ImportError(f"{_PRIOR_NAME} already bound to {mod.__file__}")
        return mod
    if not PRIOR_GENERATED.is_file():
        raise ImportError(f"prior generated_v1.py not found at {PRIOR_GENERATED}")
    spec = importlib.util.spec_from_file_location(_PRIOR_NAME, PRIOR_GENERATED)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PRIOR_NAME] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(_PRIOR_NAME, None)
        raise
    return mod


G = _load_prior()
_require = G._require


def prior_module_sha256() -> str:
    return hashlib.sha256(PRIOR_GENERATED.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# coins, eligibility, batch selection
# --------------------------------------------------------------------------- #
def make_coins(seed: int = COIN_SEED, shape=COIN_SHAPE) -> np.ndarray:
    """numpy.random.default_rng(seed).random(shape) < 0.5 -> bool [steps, half]."""
    _require(type(seed) is int, "seed must be an int", TypeError)
    _require(isinstance(shape, tuple) and len(shape) == 2
             and all(type(s) is int and s >= 1 for s in shape), "shape must be a tuple of 2 positive ints")
    return np.random.default_rng(seed).random(shape) < 0.5


def _check_eligible(e, n: int) -> np.ndarray:
    _require(isinstance(e, np.ndarray), "eligible must be a numpy array", TypeError)
    _require(e.dtype == np.bool_, "eligible must be bool", TypeError)
    _require(e.ndim == 1 and e.shape[0] == n, f"eligible must be 1-D with length {n}")
    return e


def _check_coins(c, shape) -> np.ndarray:
    _require(isinstance(c, np.ndarray), "coins must be a numpy array", TypeError)
    _require(c.dtype == np.bool_, "coins must be bool", TypeError)
    _require(c.shape == tuple(shape), f"coins must have shape {tuple(shape)}, got {c.shape}")
    return c


def _check_decimated(dec, eligible: np.ndarray, width: int) -> List[Optional[np.ndarray]]:
    """Entry i is a [T>=1, width] float32 prefix iff row i is eligible, else None."""
    _require(isinstance(dec, (list, tuple)), "decimated_base_prefixes must be a list or tuple", TypeError)
    _require(len(dec) == eligible.shape[0], "decimated_base_prefixes must have one entry per base row")
    out: List[Optional[np.ndarray]] = []
    for i, a in enumerate(dec):
        if bool(eligible[i]):
            _require(a is not None, f"eligible base row {i} has no decimated prefix")
            out.append(G._check_matrix(a, width, f"decimated_base_prefixes[{i}]"))
        else:
            _require(a is None, f"non-eligible base row {i} must not carry a decimated prefix")
            out.append(None)
    return out


def replacement_mask(base_indices: np.ndarray, coins: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """[steps, half] bool: the coin is 1 AND the drawn base row is eligible."""
    return np.logical_and(coins, eligible[base_indices])


def select_batch(step: int, base_indices, context_indices, base, context, decimated,
                 base_y, context_y, replace_row):
    """Clips in the prior order: 16 base draws (a replaced draw uses decimated[i]), then 16
    context draws (never replaced). Labels follow the same order. Returns (clips, labels, brow)."""
    brow, crow = base_indices[step], context_indices[step]
    _require(isinstance(replace_row, np.ndarray) and replace_row.dtype == np.bool_
             and replace_row.shape == brow.shape, "replace_row must be bool with the base-row shape")
    clips = []
    for i, r in zip(brow, replace_row):
        if bool(r):
            _require(decimated[int(i)] is not None, f"step {step}: replacement requested for row {int(i)} without a prefix")
            clips.append(decimated[int(i)])
        else:
            clips.append(base[int(i)])
    clips += [context[int(j)] for j in crow]
    labels = np.concatenate((base_y[brow], context_y[crow]))
    return clips, labels, brow


# --------------------------------------------------------------------------- #
# paired run
# --------------------------------------------------------------------------- #
def run_framerate_pair(model: nn.Module,
                       base_prefixes: Sequence[np.ndarray],
                       base_y: np.ndarray,
                       context_prefixes: Sequence[np.ndarray],
                       context_y: np.ndarray,
                       teacher_base_logits: np.ndarray,
                       base_indices: np.ndarray,
                       context_indices: np.ndarray,
                       decimated_base_prefixes: Sequence[Optional[np.ndarray]],
                       eligible: np.ndarray,
                       coins: np.ndarray,
                       *,
                       learning_rate: float = 1e-3,
                       weight_decay: float = 1e-3,
                       anchor_coefficient: float = 1e-2,
                       output_coefficient: float = 1.0,
                       progress: Optional[Callable[[str, int, int], None]] = None) -> Dict[str, object]:
    width, _, num_classes = G._inspect_model(model)
    base = G._check_matrix_list(base_prefixes, width, "base_prefixes")
    ctx = G._check_matrix_list(context_prefixes, width, "context_prefixes")
    by = G._check_labels(base_y, len(base), num_classes, "base_y")
    cy = G._check_labels(context_y, len(ctx), num_classes, "context_y")
    teacher = G._check_teacher(teacher_base_logits, len(base), num_classes)
    bidx = G._check_schedule(base_indices, len(base), "base_indices")
    cidx = G._check_schedule(context_indices, len(ctx), "context_indices")
    _require(bidx.shape == cidx.shape, "base_indices and context_indices must share the same shape")
    elig = _check_eligible(eligible, len(base))
    coin = _check_coins(coins, bidx.shape)
    dec = _check_decimated(decimated_base_prefixes, elig, width)
    lr = G._check_scalar(learning_rate, "learning_rate", positive=True)
    wd = G._check_scalar(weight_decay, "weight_decay")
    anchor_c = G._check_scalar(anchor_coefficient, "anchor_coefficient")
    output_c = G._check_scalar(output_coefficient, "output_coefficient")
    _require(progress is None or callable(progress), "progress must be callable or None", TypeError)

    steps, half = int(bidx.shape[0]), int(bidx.shape[1])
    last_index = len(model.blocks) - 1
    expected = replacement_mask(bidx, coin, elig)
    source_p, source_b = G._snapshot(model)

    def _run_arm(arm: str, replace: bool) -> Dict[str, object]:
        work = copy.deepcopy(model)
        work.eval()                                   # eval for the whole run, incl. autograd
        for p in work.parameters():
            p.requires_grad_(False)
        names, params = [], []
        for n, p in work.named_parameters():
            if n.startswith(HEAD_PREFIX):
                p.requires_grad_(True)
                names.append(n)
                params.append(p)
        _require(len(params) > 0, f"{arm}: no head parameters")
        _require(all(n.startswith(HEAD_PREFIX) for n in names), f"{arm}: non-head trainable", RuntimeError)
        init_p, init_b = G._snapshot(work)
        anchors = [init_p[n] for n in names]
        frozen_names = [n for n, _ in work.named_parameters() if n not in set(names)]
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
        losses = np.zeros((steps, 4), dtype=np.float64)
        replaced = np.zeros((steps, half), dtype=bool)
        no_replace = np.zeros(half, dtype=bool)
        block = work.blocks[last_index]

        with torch.enable_grad():
            for t in range(steps):
                row_mask = expected[t] if replace else no_replace
                clips, labels_np, brow = select_batch(t, bidx, cidx, base, ctx, dec, by, cy, row_mask)
                replaced[t] = row_mask
                labels = torch.from_numpy(labels_np)
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

        frozen_ok = G._params_match(work, init_p, skip=set(names))
        buffers_ok = G._buffers_match(work, init_b)
        _require(frozen_ok, f"{arm}: frozen parameters were modified", RuntimeError)
        _require(buffers_ok, f"{arm}: buffers were modified", RuntimeError)
        want = expected if replace else np.zeros_like(expected)
        _require(np.array_equal(replaced, want), f"{arm}: actual replacements differ from the coin rule", RuntimeError)
        return {"state_dict": {k: v.detach().cpu().clone() for k, v in work.state_dict().items()},
                "losses": losses,
                "replaced": replaced,
                "replacements": int(replaced.sum()),
                "replaced_distinct_rows": int(np.unique(bidx[replaced]).size),
                "trainable_names": list(names),
                "frozen_names": frozen_names,
                "updates": steps,
                "frozen_parameters_unchanged": bool(frozen_ok),
                "buffers_unchanged": bool(buffers_ok)}

    control = _run_arm("control", False)
    candidate = _run_arm("candidate", True)

    _require(G._params_match(model, source_p) and G._buffers_match(model, source_b),
             "the input model was modified during run_framerate_pair", RuntimeError)

    settings = {
        "objective": ("cross_entropy(mean over full batch) "
                      "+ 0.5*anchor_coefficient*sum((theta-theta0)^2 over trainable params) "
                      "+ 0.5*output_coefficient*mean(center_over_classes(student-teacher)^2 on base half)"),
        "output_penalty": {"scope": "base_half_only, replaced draws included", "centered_over_classes": True,
                           "dtype": "float64", "teacher": "fixed parent logits of the ORIGINAL input, never updated",
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
        "trainable_scopes": {"control": [HEAD_PREFIX], "candidate": [HEAD_PREFIX]},
        "replacement": {"control": "none (coins received and ignored)",
                        "candidate": "base draw (t,k) uses decimated prefix iff coins[t,k] and eligible[row]",
                        "context_draws": "never replaced",
                        "coins_true": int(coin.sum()), "eligible_rows": int(elig.sum()),
                        "expected_candidate_replacements": int(expected.sum())},
        "loss_columns": ["ce", "weighted_anchor", "weighted_output", "total"],
        "losses_recorded": "pre-update",
        "module_mode": "eval (dropout off, normalization running stats fixed)",
        "device": "cpu",
        "dtype": "float32",
        "deterministic": True,
    }
    return {"control": control, "candidate": candidate, "settings": settings,
            "schedule": {"base_indices": bidx.copy(), "context_indices": cidx.copy(), "coins": coin.copy()}}
