"""Minimum adapter: add a masked handshape auxiliary loss to the approved blocks.3+head routine.

The word path is unchanged from ops/partial_encoder_pair_20260922/generated_v1.py `_train_arm`:
    pooled = masked_mean(blocks[last](h, m), m)
    word_logits = head(pooled)
The only addition is a linear auxiliary head on the SAME pooled tensor, so its gradient reaches
blocks[last] - the shared features the word head uses. This is not a probe on frozen features.

Choices fixed before any run:
  * auxiliary head initialised to ZEROS, so no RNG is consumed and step 0 predicts uniformly.
  * the auxiliary term is the mean cross entropy over the LABELLED rows of the batch only.
    A batch with no labelled row contributes exactly 0.0.
  * the anchor term covers only parameters that exist in the parent. The auxiliary head has no
    parent counterpart and is excluded from the anchor; it does receive the optimizer's weight decay.
  * the output-preservation term is unchanged and applies to the word logits on the base half.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["AuxWordModel", "train_arm"]


def _require(c, m, e=ValueError):
    if not c:
        raise e(m)


def _pad_batch(arrays):
    b = len(arrays)
    t = max(int(a.shape[0]) for a in arrays)
    w = int(arrays[0].shape[1])
    h = np.zeros((b, w, t), dtype=np.float32)
    m = np.zeros((b, 1, t), dtype=np.float32)
    for i, a in enumerate(arrays):
        n = int(a.shape[0])
        h[i, :, :n] = a.T
        m[i, 0, :n] = 1.0
    return torch.from_numpy(h), torch.from_numpy(m)


def _masked_mean(h, m):
    return (h * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)


class AuxWordModel(nn.Module):
    """Frozen prefix -> last block -> masked mean -> (word head, auxiliary shape head)."""

    def __init__(self, parent, n_shape_classes, with_aux):
        super().__init__()
        _require(isinstance(parent.head, nn.Linear), "parent.head must be Linear", TypeError)
        self.base = copy.deepcopy(parent)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.last = len(self.base.blocks) - 1
        for n, p in self.base.named_parameters():
            if n.startswith("head.") or n.startswith(f"blocks.{self.last}."):
                p.requires_grad_(True)
        self.with_aux = bool(with_aux)
        width = int(self.base.head.in_features)
        self.shape_head = nn.Linear(width, int(n_shape_classes))
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.shape_head.bias)
        if not self.with_aux:
            for p in self.shape_head.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, h, m):
        pooled = _masked_mean(self.base.blocks[self.last](h, m), m)
        return self.base.head(pooled), self.shape_head(pooled), pooled


def train_arm(parent, base_prefixes, base_y, base_shape_y,
              ctx_prefixes, ctx_y, ctx_shape_y, teacher_base_logits,
              base_indices, context_indices, *, n_shape_classes, aux_weight,
              learning_rate=1e-3, weight_decay=1e-3, anchor_coefficient=1e-2,
              output_coefficient=1.0, with_aux=True, progress=None):
    model = AuxWordModel(parent, n_shape_classes, with_aux)
    model.eval()
    names, params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)
    expected = {f"base.{k}" for k, p in model.base.named_parameters()
                if k.startswith("head.") or k.startswith(f"blocks.{model.last}.")}
    if with_aux:
        expected |= {"shape_head.weight", "shape_head.bias"}
    _require(set(names) == expected, f"unexpected trainable set: {sorted(set(names) ^ expected)}")

    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    bufs = {n: b.detach().clone() for n, b in model.named_buffers()}
    # anchor covers ONLY parameters that exist in the parent
    anchored = [(n, p) for n, p in zip(names, params) if n.startswith("base.")]
    opt = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=weight_decay)
    steps, half = int(base_indices.shape[0]), int(base_indices.shape[1])
    losses = np.zeros((steps, 5), dtype=np.float64)
    aux_rows = np.zeros(steps, dtype=np.int64)
    teacher = torch.from_numpy(np.ascontiguousarray(teacher_base_logits, dtype=np.float64))

    with torch.enable_grad():
        for t in range(steps):
            br, cr = base_indices[t], context_indices[t]
            clips = [base_prefixes[int(i)] for i in br] + [ctx_prefixes[int(j)] for j in cr]
            y = torch.from_numpy(np.concatenate((base_y[br], ctx_y[cr])))
            sy = torch.from_numpy(np.concatenate((base_shape_y[br], ctx_shape_y[cr])))
            h, m = _pad_batch(clips)
            word_logits, shape_logits, _ = model(h, m)

            ce = F.cross_entropy(word_logits, y)          # every row, labelled or not
            sq = torch.zeros((), dtype=torch.float32)
            for n, p in anchored:
                sq = sq + (p - init[n]).pow(2).sum()
            anchor = 0.5 * anchor_coefficient * sq
            rows = torch.from_numpy(np.ascontiguousarray(br))
            delta = word_logits[:half].to(torch.float64) - teacher.index_select(0, rows)
            delta = delta - delta.mean(dim=1, keepdim=True)
            output = 0.5 * output_coefficient * delta.pow(2).mean()

            keep = sy >= 0
            n_keep = int(keep.sum())
            aux_rows[t] = n_keep
            if with_aux and n_keep > 0:
                aux = F.cross_entropy(shape_logits[keep], sy[keep])   # mean over labelled rows only
            else:
                aux = torch.zeros((), dtype=torch.float32)

            total = (ce.to(torch.float64) + anchor.to(torch.float64) + output
                     + float(aux_weight) * aux.to(torch.float64))
            for lbl, v in (("ce", ce), ("anchor", anchor), ("output", output),
                           ("aux", aux), ("total", total)):
                _require(bool(torch.isfinite(v)), f"step {t}: non-finite {lbl}", RuntimeError)
            losses[t] = (float(ce), float(anchor), float(output), float(aux), float(total))

            opt.zero_grad(set_to_none=True)
            total.backward()
            for n, p in zip(names, params):
                if p.grad is not None:
                    _require(bool(torch.isfinite(p.grad).all()),
                             f"step {t}: non-finite grad {n}", RuntimeError)
            opt.step()
            if progress is not None and ((t + 1) % 100 == 0 or t + 1 == steps):
                progress(t + 1, steps)

    frozen_ok = all(torch.equal(p.detach(), init[n]) for n, p in model.named_parameters()
                    if n not in set(names))
    bufs_ok = all(torch.equal(b.detach(), bufs[n]) for n, b in model.named_buffers())
    _require(frozen_ok, "frozen parameters changed", RuntimeError)
    _require(bufs_ok, "buffers changed", RuntimeError)
    return {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "losses": losses, "aux_rows_per_step": aux_rows,
            "trainable_names": sorted(names), "updates": steps,
            "frozen_parameters_unchanged": bool(frozen_ok), "buffers_unchanged": bool(bufs_ok),
            "loss_columns": ["ce", "weighted_anchor", "weighted_output", "aux_raw", "total"],
            "aux_weight": float(aux_weight),
            "anchor_scope": "parent parameters only; the auxiliary head is excluded from the anchor",
            "aux_reduction": "mean over labelled rows of the batch; 0.0 when a batch has none",
            "aux_init": "zeros; no RNG consumed"}
