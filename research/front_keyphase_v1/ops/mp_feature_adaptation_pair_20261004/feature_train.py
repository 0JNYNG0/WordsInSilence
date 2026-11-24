"""Head-only or last-block+head learner for the MP feature-adaptation pair (FROZEN_PLAN.md).

Verbatim copy of train_addition.train_arm arithmetic (deepcopy, eval mode, AdamW, old 16+16 batch,
anchor, centred float64 output penalty on base 16, separate extra forward, ONE total.backward()).
The only addition is `train_last_block`:

  False : trainable = head.*                       (identical to train_addition.train_arm)
  True  : trainable = blocks.{last}.* + head.*     (earlier blocks/proj frozen)

The whole work model stays in eval mode, so every buffer (normalisation statistics) is frozen.
The anchor sums (theta - theta_parent)^2 over ALL trainable parameters of the arm.
No projection, no split gradients, no new loss. The prior generated_v1 module is passed in as `G`
(this file loads nothing from disk).
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HEAD = "head."
LOSS_COLUMNS = ["old_ce", "extra_ce", "weighted_extra", "weighted_anchor", "weighted_output", "total"]


def last_block_prefix(model: nn.Module) -> str:
    return f"blocks.{len(model.blocks) - 1}."


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
              train_last_block: bool = False,
              G=None) -> Dict[str, object]:
    if G is None or not all(hasattr(G, a) for a in ("_pad_batch", "_masked_mean", "_inspect_model", "_snapshot")):
        raise TypeError("G (prior generated_v1 module) is required")
    _require = G._require
    _require(isinstance(train_last_block, bool), "train_last_block must be bool", TypeError)
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
    prefixes = (HEAD, last_block_prefix(model)) if train_last_block else (HEAD,)
    source_p, source_b = G._snapshot(model)

    work = copy.deepcopy(model)
    work.eval()
    for p in work.parameters():
        p.requires_grad_(False)
    names, params = [], []
    for n, p in work.named_parameters():
        if n.startswith(prefixes):
            p.requires_grad_(True)
            names.append(n)
            params.append(p)
    _require(any(n.startswith(HEAD) for n in names), f"{arm}: no head parameters")
    _require(not train_last_block or any(n.startswith(prefixes[1]) for n in names),
             f"{arm}: no last-block parameters")
    init_p, init_b = G._snapshot(work)
    anchors = [init_p[n] for n in names]
    frozen_names = [n for n, _ in work.named_parameters() if n not in set(names)]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    losses = np.full((steps, len(LOSS_COLUMNS)), np.nan, dtype=np.float64)
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
            "trainable_prefixes": list(prefixes),
            "train_last_block": train_last_block,
            "updates": steps,
            "extra_weight": w,
            "base_indices": bidx.copy(), "context_indices": cidx.copy(),
            "extra_indices": eidx.copy() if has_extra else None,
            "frozen_parameters_unchanged": bool(frozen_ok),
            "buffers_unchanged": bool(buffers_ok)}
