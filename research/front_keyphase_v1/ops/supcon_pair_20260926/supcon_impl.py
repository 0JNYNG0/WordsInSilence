"""blocks.3+head routine with ONE optional addition: a supervised contrastive term on the pooled tensor.

Word path identical to ops/handshape_aux_pair_20260925/generated_v1.py with with_aux=False:
    pooled = masked_mean(blocks[last](h, m), m);  word_logits = head(pooled)
    total  = CE + 0.5*anchor*sum((theta-theta0)^2) + 0.5*output*mean(centered(student-teacher)^2 on base half)
Candidate adds  lambda * SupCon(normalize(pooled)). With lambda == 0 the term is NOT placed in the
graph (only logged under no_grad), so the control run is the plain routine.

Pair rules (fixed in comparison_policy.json):
  eligible  = rows with a CONFIRMED signer id; NIKL rows (signer unverified) are outside the term
  positive  = same word, different signer, different recording
  excluded from numerator AND denominator: same word with same signer, or same recording
  negative  = eligible row of a different word
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["WordModel", "pair_masks", "supcon", "train_arm", "_pad_batch"]


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


class WordModel(nn.Module):
    """Frozen prefix -> last block -> masked mean -> word head. Only blocks.last and head train."""

    def __init__(self, parent):
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

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, h, m):
        pooled = _masked_mean(self.base.blocks[self.last](h, m), m)
        return self.base.head(pooled), pooled


def pair_masks(labels, signers, recordings):
    """-> (eligible[B], positive[B,B], denominator[B,B]) as bool numpy arrays.
    signers: int >= 0 for a confirmed signer, -1 for unverified (outside the term)."""
    labels, signers = np.asarray(labels), np.asarray(signers)
    recordings = np.asarray(recordings)
    b = len(labels)
    elig = signers >= 0
    both = elig[:, None] & elig[None, :]
    not_self = ~np.eye(b, dtype=bool)
    same_word = labels[:, None] == labels[None, :]
    same_signer = signers[:, None] == signers[None, :]
    same_rec = recordings[:, None] == recordings[None, :]
    pos = both & not_self & same_word & ~same_signer & ~same_rec
    excluded = same_word & (same_signer | same_rec)          # never pushed apart, never pulled
    den = both & not_self & ~(excluded & ~pos)
    return elig, pos, den


def supcon(pooled, pos, den, tau):
    """mean over anchors with >= 1 positive of  -mean_p log softmax_den(z_a . z_p / tau)."""
    pos_t = torch.from_numpy(pos)
    den_t = torch.from_numpy(den)
    anchors = pos_t.any(dim=1)
    n_anchor = int(anchors.sum())
    if n_anchor == 0:
        return torch.zeros((), dtype=torch.float32), 0
    z = F.normalize(pooled, dim=1)
    sim = (z @ z.t()) / float(tau)
    neg_inf = torch.finfo(sim.dtype).min
    logits = sim.masked_fill(~den_t, neg_inf)
    log_den = torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = sim - log_den
    per_anchor = -(log_prob * pos_t).sum(dim=1) / pos_t.sum(dim=1).clamp_min(1)
    return per_anchor[anchors].mean(), n_anchor


def train_arm(parent, base_prefixes, base_y, base_signer, base_rec,
              ctx_prefixes, ctx_y, ctx_signer, ctx_rec, teacher_base_logits,
              base_indices, context_indices, *, lam, tau,
              learning_rate=1e-3, weight_decay=1e-3, anchor_coefficient=1e-2,
              output_coefficient=1.0, progress=None):
    model = WordModel(parent)
    model.eval()
    names, params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)
    expected = {f"base.{k}" for k, p in model.base.named_parameters()
                if k.startswith("head.") or k.startswith(f"blocks.{model.last}.")}
    _require(set(names) == expected, f"unexpected trainable set: {sorted(set(names) ^ expected)}")

    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    bufs = {n: b.detach().clone() for n, b in model.named_buffers()}
    anchored = list(zip(names, params))
    opt = torch.optim.AdamW(params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=weight_decay)
    steps, half = int(base_indices.shape[0]), int(base_indices.shape[1])
    losses = np.zeros((steps, 5), dtype=np.float64)
    anchors_per_step = np.zeros(steps, dtype=np.int64)
    teacher = torch.from_numpy(np.ascontiguousarray(teacher_base_logits, dtype=np.float64))
    use_term = float(lam) != 0.0

    with torch.enable_grad():
        for t in range(steps):
            br, cr = base_indices[t], context_indices[t]
            clips = [base_prefixes[int(i)] for i in br] + [ctx_prefixes[int(j)] for j in cr]
            y_np = np.concatenate((base_y[br], ctx_y[cr]))
            y = torch.from_numpy(y_np)
            h, m = _pad_batch(clips)
            word_logits, pooled = model(h, m)

            ce = F.cross_entropy(word_logits, y)
            sq = torch.zeros((), dtype=torch.float32)
            for n, p in anchored:
                sq = sq + (p - init[n]).pow(2).sum()
            anchor = 0.5 * anchor_coefficient * sq
            rows = torch.from_numpy(np.ascontiguousarray(br))
            delta = word_logits[:half].to(torch.float64) - teacher.index_select(0, rows)
            delta = delta - delta.mean(dim=1, keepdim=True)
            output = 0.5 * output_coefficient * delta.pow(2).mean()

            _, pos, den = pair_masks(y_np, np.concatenate((base_signer[br], ctx_signer[cr])),
                                     np.concatenate((base_rec[br], ctx_rec[cr])))
            if use_term:
                sc, n_anchor = supcon(pooled, pos, den, tau)
            else:
                with torch.no_grad():
                    sc, n_anchor = supcon(pooled.detach(), pos, den, tau)
            anchors_per_step[t] = n_anchor

            total = ce.to(torch.float64) + anchor.to(torch.float64) + output
            if use_term:
                total = total + float(lam) * sc.to(torch.float64)
            for lbl, v in (("ce", ce), ("anchor", anchor), ("output", output),
                           ("supcon", sc), ("total", total)):
                _require(bool(torch.isfinite(v)), f"step {t}: non-finite {lbl}", RuntimeError)
            losses[t] = (float(ce), float(anchor), float(output), float(sc), float(total))

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
            "losses": losses, "anchors_per_step": anchors_per_step,
            "trainable_names": sorted(names), "updates": steps,
            "frozen_parameters_unchanged": bool(frozen_ok), "buffers_unchanged": bool(bufs_ok),
            "loss_columns": ["ce", "weighted_anchor", "weighted_output", "supcon_raw", "total"],
            "lam": float(lam), "tau": float(tau), "term_in_graph": use_term}
