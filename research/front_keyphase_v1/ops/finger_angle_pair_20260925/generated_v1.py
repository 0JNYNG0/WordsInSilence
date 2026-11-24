"""Width-generic extra-input residual on a frozen TCN parent, and a matched pair trainer.

Same construction the approved motion path uses, with the extra-feature width freed from the
3P+1 coordinate layout so a 76-channel geometry block fits:

  h = parent.proj(x) + conv1d(extra / scale, W)        W initialised to ZERO

Because W starts at exactly zero the module reproduces the parent's output bit for bit before
any update, so both arms start from the same point. Everything in the parent is frozen except
head.*; the base stays in eval mode for the whole run (dropout off, normalisation stats fixed).

Only the VALUE half of the extra block is scaled; the validity half is left as 0/1.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ExtraInputTCN", "fit_channel_scale", "run_pair"]


def _require(cond, msg, err=ValueError):
    if not cond:
        raise err(msg)


def _check_clips(seqs, channels, name):
    _require(isinstance(seqs, (list, tuple)) and len(seqs) > 0, f"{name} must be a non-empty list")
    out = []
    for i, a in enumerate(seqs):
        _require(isinstance(a, np.ndarray), f"{name}[{i}] must be ndarray", TypeError)
        _require(a.dtype == np.float32, f"{name}[{i}] must be float32", TypeError)
        _require(a.ndim == 2 and a.shape[1] == channels,
                 f"{name}[{i}] must be [T,{channels}], got {a.shape}")
        _require(a.shape[0] >= 1, f"{name}[{i}] must have at least one frame")
        _require(bool(np.isfinite(a).all()), f"{name}[{i}] holds a non-finite value")
        out.append(np.ascontiguousarray(a))
    return out


def fit_channel_scale(extra_seqs, labels, n_values):
    """Per-channel RMS over TRAIN clips only, class balanced. Mirrors the approved position fit.

    A channel contributes a clip only where that clip actually observed it. A channel never
    observed anywhere keeps scale 1.0. No centring, no outlier removal, no result-driven tuning.
    """
    seqs = _check_clips(extra_seqs, extra_seqs[0].shape[1], "extra_seqs")
    width = seqs[0].shape[1]
    _require(0 < n_values < width, "n_values must be inside the block width")
    labels = np.asarray(labels)
    _require(labels.shape == (len(seqs),), "labels must be one per clip")
    means, seen = [], []
    for a in seqs:
        mask = a[:, n_values:] > 0.5                       # [T, n_values]
        count = mask.sum(0)
        seen.append(count > 0)
        sq = np.where(mask, a[:, :n_values].astype(np.float64) ** 2, 0.0).sum(0)
        means.append(np.divide(sq, count, out=np.zeros(n_values), where=count > 0))
    uniq, counts = np.unique(labels, return_counts=True)
    w = np.array([1.0 / (len(uniq) * counts[np.flatnonzero(uniq == y)[0]]) for y in labels])
    num = (np.asarray(means) * w[:, None]).sum(0)
    den = (np.asarray(seen) * w[:, None]).sum(0)
    scale = np.ones(n_values, dtype=np.float64)
    good = (num > 0) & (den > 0)
    scale[good] = np.sqrt(num[good] / den[good])
    return {"scale": scale.astype(np.float32), "valid_clip_weight": den,
            "clip_count": len(seqs), "n_values": int(n_values), "width": int(width),
            "never_observed_channels": int((~good).sum())}


class ExtraInputTCN(nn.Module):
    def __init__(self, parent, scale, n_values, width):
        super().__init__()
        _require(isinstance(parent, nn.Module), "parent must be nn.Module", TypeError)
        proj, blocks, head = parent.proj, parent.blocks, parent.head
        _require(isinstance(proj, nn.Conv1d), "parent.proj must be Conv1d", TypeError)
        _require(isinstance(blocks, nn.ModuleList) and len(blocks) > 0, "parent.blocks empty", TypeError)
        _require(isinstance(head, nn.Linear), "parent.head must be Linear", TypeError)
        _require(proj.out_channels == head.in_features, "proj.out_channels must equal head.in_features")
        for n, t in list(parent.named_parameters()) + list(parent.named_buffers()):
            _require(t.device.type == "cpu", f"{n} must be on cpu")
            _require(not t.is_floating_point() or t.dtype == torch.float32, f"{n} must be float32")
        s = torch.as_tensor(np.asarray(scale, dtype=np.float32)).clone()
        _require(s.ndim == 1 and s.numel() == n_values, "scale must be one value per value-channel")
        _require(bool(torch.isfinite(s).all()) and bool((s > 0).all()), "scale must be finite and positive")
        self.n_values, self.extra_width = int(n_values), int(width)
        self.hidden = int(proj.out_channels)
        self.base = copy.deepcopy(parent)
        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.base.head.parameters():
            p.requires_grad_(True)
        self.register_buffer("extra_scale", s)
        self.extra_weight = nn.Parameter(torch.zeros(self.hidden, self.extra_width, 1,
                                                     dtype=torch.float32))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, extra, pad_mask):
        _require(x.ndim == 3 and extra.ndim == 3 and pad_mask.ndim == 2, "bad input rank")
        _require(x.shape[0] == extra.shape[0] == pad_mask.shape[0], "batch mismatch")
        _require(x.shape[1] == extra.shape[1] == pad_mask.shape[1], "time mismatch")
        _require(extra.shape[2] == self.extra_width, "extra width mismatch")
        m = pad_mask.unsqueeze(1)
        h = self.base.proj(x.transpose(1, 2) * m) * m
        vals = extra[:, :, :self.n_values] / self.extra_scale
        feat = torch.cat([vals, extra[:, :, self.n_values:]], dim=2)
        _require(bool(torch.isfinite(feat).all()), "scaled extra input is non-finite", RuntimeError)
        h = h + F.conv1d(feat.transpose(1, 2), self.extra_weight) * m
        for b in self.base.blocks:
            h = b(h, m)
        pooled = (h * m).sum(dim=2) / m.sum(dim=2).clamp(min=1.0)
        return self.base.head(pooled)


def _pad(xs, es):
    b = len(xs)
    t = max(a.shape[0] for a in xs)
    x = np.zeros((b, t, xs[0].shape[1]), dtype=np.float32)
    e = np.zeros((b, t, es[0].shape[1]), dtype=np.float32)
    m = np.zeros((b, t), dtype=np.float32)
    for i, (a, g) in enumerate(zip(xs, es)):
        n = a.shape[0]
        x[i, :n] = a
        e[i, :n] = g
        m[i, :n] = 1.0
    return torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(m)


def _train_arm(arm, parent, scale, base_x, base_e, base_y, ctx_x, ctx_e, ctx_y,
               teacher, bidx, cidx, lr, wd, anchor_c, output_c, n_values, width, progress):
    model = ExtraInputTCN(parent, scale, n_values, width)
    model.eval()
    names, params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            names.append(n)
            params.append(p)
    _require(set(names) == {"extra_weight", "base.head.weight", "base.head.bias"},
             f"{arm}: unexpected trainable set {sorted(names)}")
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    bufs = {n: b.detach().clone() for n, b in model.named_buffers()}
    anchors = [init[n] for n in names]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=wd)
    steps, half = int(bidx.shape[0]), int(bidx.shape[1])
    losses = np.zeros((steps, 4), dtype=np.float64)
    tt = torch.from_numpy(np.ascontiguousarray(teacher, dtype=np.float64))
    with torch.enable_grad():
        for t in range(steps):
            br, cr = bidx[t], cidx[t]
            xs = [base_x[int(i)] for i in br] + [ctx_x[int(j)] for j in cr]
            es = [base_e[int(i)] for i in br] + [ctx_e[int(j)] for j in cr]
            y = torch.from_numpy(np.concatenate((base_y[br], ctx_y[cr])))
            x, e, m = _pad(xs, es)
            logits = model(x, e, m)
            ce = F.cross_entropy(logits, y)
            sq = torch.zeros((), dtype=torch.float32)
            for p, p0 in zip(params, anchors):
                sq = sq + (p - p0).pow(2).sum()
            anchor = 0.5 * anchor_c * sq
            d = logits[:half].to(torch.float64) - tt.index_select(0, torch.from_numpy(np.ascontiguousarray(br)))
            d = d - d.mean(dim=1, keepdim=True)
            out = 0.5 * output_c * d.pow(2).mean()
            total = ce.to(torch.float64) + anchor.to(torch.float64) + out
            for lbl, v in (("ce", ce), ("anchor", anchor), ("output", out), ("total", total)):
                _require(bool(torch.isfinite(v)), f"{arm} step {t}: non-finite {lbl}", RuntimeError)
            losses[t] = (float(ce), float(anchor), float(out), float(total))
            opt.zero_grad(set_to_none=True)
            total.backward()
            for n, p in zip(names, params):
                if p.grad is not None:
                    _require(bool(torch.isfinite(p.grad).all()),
                             f"{arm} step {t}: non-finite grad {n}", RuntimeError)
            opt.step()
            if progress and ((t + 1) % 100 == 0 or t + 1 == steps):
                progress(arm, t + 1, steps)
    frozen_ok = all(torch.equal(p.detach(), init[n]) for n, p in model.named_parameters()
                    if n not in set(names))
    bufs_ok = all(torch.equal(b.detach(), bufs[n]) for n, b in model.named_buffers())
    _require(frozen_ok, f"{arm}: frozen parameters changed", RuntimeError)
    _require(bufs_ok, f"{arm}: buffers changed", RuntimeError)
    return {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "losses": losses, "trainable_names": sorted(names), "updates": steps,
            "frozen_parameters_unchanged": bool(frozen_ok), "buffers_unchanged": bool(bufs_ok),
            "extra_weight_nonzero": bool(torch.count_nonzero(
                model.state_dict()["extra_weight"]).item())}


def run_pair(parent, base_x, base_y, ctx_x, ctx_y, teacher, bidx, cidx,
             control_extra, treatment_extra, control_scale, treatment_scale,
             n_values, width, *, learning_rate=1e-3, weight_decay=1e-3,
             anchor_coefficient=1e-2, output_coefficient=1.0, progress=None):
    """Two arms, identical in everything but which extra block feeds the added projection."""
    src_p = {n: p.detach().clone() for n, p in parent.named_parameters()}
    src_b = {n: b.detach().clone() for n, b in parent.named_buffers()}
    arms = {}
    for arm, extra, scale in (("control", control_extra, control_scale),
                              ("treatment", treatment_extra, treatment_scale)):
        arms[arm] = _train_arm(arm, parent, scale,
                               base_x, extra["base"], base_y, ctx_x, extra["context"], ctx_y,
                               teacher, bidx, cidx, learning_rate, weight_decay,
                               anchor_coefficient, output_coefficient, n_values, width, progress)
    _require(all(torch.equal(p.detach(), src_p[n]) for n, p in parent.named_parameters())
             and all(torch.equal(b.detach(), src_b[n]) for n, b in parent.named_buffers()),
             "the parent model was modified", RuntimeError)
    return {"control": arms["control"], "treatment": arms["treatment"],
            "settings": {"optimizer": {"name": "AdamW", "learning_rate": learning_rate,
                                       "weight_decay": weight_decay, "betas": [0.9, 0.999],
                                       "eps": 1e-8, "scheduler": None, "grad_clipping": None,
                                       "early_stopping": False},
                         "anchor_coefficient": anchor_coefficient,
                         "output_coefficient": output_coefficient,
                         "steps": int(bidx.shape[0]), "half_batch": int(bidx.shape[1]),
                         "batch_size": 2 * int(bidx.shape[1]),
                         "extra_width": int(width), "extra_value_channels": int(n_values),
                         "control_feature": "chord", "treatment_feature": "angle",
                         "trainable": ["extra_weight", "base.head.weight", "base.head.bias"],
                         "encoder_and_buffers": "frozen",
                         "module_mode": "eval (dropout off, normalisation stats fixed)",
                         "device": "cpu", "dtype": "float32", "deterministic": True}}
