"""Head-only learner for the target-input adaptation development comparison.

control   : exactly the prior generated_v1 `run_pair` control arm (same deepcopy / eval /
            requires_grad set-up, AdamW, 16 base + 16 pair_context draws, zero padding,
            frozen last block -> masked mean -> head, and loss
            CE(32) + 0.5*0.01*sum((theta-theta0)^2) + 0.5*1.0*mean(centred(student-teacher)^2 on 16 base)).
candidate : the same, plus  extra_weight * CE(4 added inputs)  computed in a SEPARATE forward
            (so the old 32-row batch, its padding and arithmetic are untouched). The added inputs
            are two rows per added class per step from a deterministic cyclic schedule over that
            fold's TRAINING rows only. No teacher penalty on added rows, no augmentation.

The prior module is loaded from its explicit path under a private module name and its helpers
are used unchanged. No sys.path modification here.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
# Runtime default: FRONT/ops/partial_encoder_pair_20260922/generated_v1.py. The env override exists
# only for an initial synthetic test run; experiment.py refuses to train/evaluate with an override.
PRIOR_GENERATED = Path(os.environ.get("TIA_PRIOR_GENERATED")
                       or HERE.parent / "partial_encoder_pair_20260922" / "generated_v1.py").resolve()
_PRIOR_NAME = "tia_prior_generated_v1"
HEAD = "head."
FOLD_SALT = "wis-20260928-fold:"
N_FOLDS = 3
LOSS_COLUMNS = ["old_ce", "extra_ce", "weighted_extra", "weighted_anchor", "weighted_output", "total"]

TRAIN_KW = dict(learning_rate=1e-3, weight_decay=1e-3, anchor_coefficient=1e-2, output_coefficient=1.0)
RECIPE = {
    "trainable": "head.* only", "module_mode": "eval", "device": "cpu", "cpu_threads": 2,
    "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-3, "betas": [0.9, 0.999],
                  "eps": 1e-8, "grad_clipping": None, "scheduler": None, "early_stopping": False},
    "steps": 660, "anchor_coefficient": 1e-2, "output_coefficient": 1.0,
    "old_objective": "identical to generated_v1 control: CE over 32 + 0.5*0.01*head anchor "
                     "+ 0.5*1.0*mean centred logit residual to fixed teacher (float64) on base 16",
    "old_exposures": "train_prefixes.npz base_indices/context_indices [660,16], identical in every arm",
    "extra_weight": 0.125,
    "extra_weight_note": "4/32 exposure weight; not fitted, not claimed optimal",
    "extra_per_class": 2,
    "extra_schedule": "per added class (class-index order): fold training rows sorted by id; "
                      "step t uses rows (2t mod n, (2t+1) mod n)",
    "extra_teacher_penalty": None, "augmentation": None,
    "fold_rule": "bucket = filename prefix before first '_'; buckets sorted by "
                 "sha256('wis-20260928-fold:'+bucket).hexdigest(); fold = sorted index % 3",
    "n_folds": 3,
    "arms": ["control", "candidate_fold0", "candidate_fold1", "candidate_fold2"],
    "loss_columns": LOSS_COLUMNS,
    "losses_recorded": "pre-update",
}


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


def prior_sha256() -> str:
    return hashlib.sha256(PRIOR_GENERATED.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# folds and schedules
# --------------------------------------------------------------------------- #
def bucket_of(member: str) -> str:
    """'16/00_16.MP4' -> '00'. A bookkeeping bucket, NOT a confirmed signer identity."""
    name = str(member).replace("\\", "/").rsplit("/", 1)[-1]
    _require("_" in name and name.split("_", 1)[0] != "", f"member {member!r} has no filename prefix")
    return name.split("_", 1)[0]


def assign_folds(members: Sequence[str], n_folds: int = N_FOLDS):
    """Returns (bucket per row, fold per row int64, bucket order)."""
    buckets = [bucket_of(m) for m in members]
    order = sorted(set(buckets), key=lambda b: (hashlib.sha256((FOLD_SALT + b).encode("utf-8")).hexdigest(), b))
    fold_of = {b: i % n_folds for i, b in enumerate(order)}
    return buckets, np.array([fold_of[b] for b in buckets], dtype=np.int64), order


def check_split(ids, y, folds, buckets, hashes: Dict[str, Sequence[str]], n_folds: int = N_FOLDS,
                min_train: int = 2, min_eval: int = 1) -> Dict[str, object]:
    """Reject leakage: ids unique, buckets never span folds, identical hashes never span folds,
    each class has >= min_train training rows and >= min_eval evaluation rows in every fold."""
    y, folds = np.asarray(y), np.asarray(folds)
    _require(len(set(ids)) == len(ids) == len(y) == len(folds) == len(buckets), "ids/labels/folds misaligned")
    classes = sorted(set(y.tolist()))
    per_fold = {}
    for f in range(n_folds):
        ev = folds == f
        _require(bool(ev.any()), f"fold {f} is empty")
        tb = {buckets[i] for i in np.flatnonzero(~ev)}
        eb = {buckets[i] for i in np.flatnonzero(ev)}
        _require(not (tb & eb), f"fold {f}: bucket in both train and eval: {sorted(tb & eb)}")
        counts = {}
        for c in classes:
            ntr, nev = int(((~ev) & (y == c)).sum()), int((ev & (y == c)).sum())
            _require(ntr >= min_train, f"fold {f}: class {c} has {ntr} < {min_train} training rows")
            _require(nev >= min_eval, f"fold {f}: class {c} has {nev} < {min_eval} evaluation rows")
            counts[int(c)] = {"train": ntr, "eval": nev}
        per_fold[f] = {"eval_buckets": sorted(eb), "class_counts": counts}
    duplicates = {}
    for name, hs in hashes.items():
        _require(len(hs) == len(ids), f"{name}: one hash per row required")
        groups: Dict[str, List[int]] = {}
        for i, h in enumerate(hs):
            groups.setdefault(h, []).append(i)
        for h, rows in groups.items():
            if len(rows) > 1:
                _require(len({int(folds[i]) for i in rows}) == 1,
                         f"{name}: identical hash {h} crosses folds ({[ids[i] for i in rows]})")
                duplicates.setdefault(name, []).append([ids[i] for i in rows])
    return {"per_fold": per_fold, "same_fold_duplicates": duplicates}


def fold_training_rows(ids, y, folds, fold: int, class_order: Sequence[int]) -> List[List[int]]:
    """Global row indices NOT in `fold`, grouped by class (class_order), each group sorted by id."""
    y, folds = np.asarray(y), np.asarray(folds)
    return [sorted((i for i in range(len(ids)) if int(folds[i]) != fold and int(y[i]) == int(c)),
                   key=lambda i: ids[i]) for c in class_order]


def cyclic_schedule(group_sizes: Sequence[int], steps: int, per_class: int = 2) -> np.ndarray:
    """[steps, per_class*len(groups)] indices into the concatenated groups.
    Group g at step t uses offset_g + (per_class*t + k) mod n_g, k = 0..per_class-1."""
    out = np.zeros((steps, per_class * len(group_sizes)), dtype=np.int64)
    offset = 0
    for g, n in enumerate(group_sizes):
        _require(n >= per_class, f"group {g} has {n} < {per_class} rows")
        for k in range(per_class):
            out[:, g * per_class + k] = offset + (per_class * np.arange(steps, dtype=np.int64) + k) % n
        offset += n
    return out


# --------------------------------------------------------------------------- #
# one arm
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
    block = work.blocks[last_index]

    with torch.enable_grad():
        for t in range(steps):
            # ---- old 32-row batch: verbatim generated_v1 control arithmetic
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
    _require(G._params_match(model, source_p) and G._buffers_match(model, source_b),
             f"{arm}: the input model was modified", RuntimeError)
    return {"state_dict": {k: v.detach().cpu().clone() for k, v in work.state_dict().items()},
            "losses": losses,
            "trainable_names": list(names),
            "frozen_names": frozen_names,
            "updates": steps,
            "extra_weight": w,
            "base_indices": bidx.copy(), "context_indices": cidx.copy(),
            "extra_indices": eidx.copy() if has_extra else None,
            "frozen_parameters_unchanged": bool(frozen_ok),
            "buffers_unchanged": bool(buffers_ok)}
