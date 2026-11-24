"""Unsupported-frame factorial: pure NumPy/torch helpers (no data, no model file, no camera).

Question: when the frozen time-grid restorer adds fully unsupported interior frames, does the
regression come from temporal feature processing (blocks) and/or the final pooling?

  support_mask(restored)   1 everywhere except GENERATED grid frames (exact_source_index < 0)
                           that carry no observed and no estimated joint. A retained source frame
                           is never excluded, even when absent; a partly observed/estimated frame
                           is never excluded. No label, time or gap threshold enters.
  diagnostic_forward       the masked TCN forward written out once, with the block mask and the
                           pool mask chosen independently per arm. Time indices are NOT compressed.

    arm               block mask  pool mask
    grid_baseline     ones        ones       == model.forward(x, ones)
    pool_only         ones        support    same hidden sequence as baseline, new mean
    blocks_only       support     ones       same hidden sequence as both, mean over all T
    blocks_and_pool   support     support    == model.forward(x, support)  (primary)

'padding mask' originally meant batch membership; using it for internal missing observations is
a new semantic use the model was never trained with. This is a diagnostic, never a deployment
candidate.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

ARMS = ("grid_baseline", "pool_only", "blocks_only", "blocks_and_pool")
PRIMARY = "blocks_and_pool"
BLOCK_SUPPORT = frozenset({"blocks_only", "blocks_and_pool"})
POOL_SUPPORT = frozenset({"pool_only", "blocks_and_pool"})
RESTORED_KEYS = ("exact_source_index", "present", "observed", "estimated")


class MaskError(ValueError):
    pass


def _binary(a, what) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return a.astype(bool)
    if (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)) \
            and np.all((a == 0) | (a == 1)):
        return a == 1
    raise MaskError(f"{what} must be bool or exactly 0/1")


def support_mask(restored) -> np.ndarray:
    """float32[T]: NOT (generated frame AND no joint present). Inputs are only read."""
    if not isinstance(restored, dict) or any(k not in restored for k in RESTORED_KEYS):
        raise MaskError(f"restored must be a dict with {RESTORED_KEYS}")
    src = np.asarray(restored["exact_source_index"])
    if src.ndim != 1 or len(src) < 1 or src.dtype == np.bool_ or not np.issubdtype(src.dtype, np.integer):
        raise MaskError(f"exact_source_index must be integer [T>=1], got {src.dtype} {src.shape}")
    if (src < -1).any():
        raise MaskError("exact_source_index below -1")
    t = len(src)
    pr = np.asarray(restored["present"])
    if pr.ndim != 2 or pr.shape[0] != t or pr.shape[1] < 1:
        raise MaskError(f"present must be [{t},J], got {pr.shape}")
    pb = _binary(pr, "present")
    obs, est = _binary(restored["observed"], "observed"), _binary(restored["estimated"], "estimated")
    if obs.shape != pb.shape or est.shape != pb.shape:
        raise MaskError("observed/estimated shape differs from present")
    if (obs & est).any() or not np.array_equal(obs | est, pb):
        raise MaskError("observed/estimated are not a disjoint cover of present")
    exact = src >= 0
    if est[exact].any() or obs[~exact].any():
        raise MaskError("provenance inconsistent with exact_source_index")
    keep = ~(~exact & ~pb.any(axis=1))
    if not keep.any():
        raise MaskError("no supported frame")
    return keep.astype(np.float32)


def excluded_indices(support) -> list:
    return np.flatnonzero(np.asarray(support) == 0).tolist()


def check_excluded_input(x, support, dt_channel=-1) -> int:
    """Excluded frames must be all-zero in every built channel except dt. -> excluded count."""
    x, s = np.asarray(x), np.asarray(support)
    if x.ndim != 2 or s.shape != (x.shape[0],):
        raise MaskError(f"x {x.shape} / support {s.shape} inconsistent")
    if s[0] != 1 or s[-1] != 1:
        raise MaskError("support endpoints must be 1")
    keep = np.ones(x.shape[1], bool)
    keep[dt_channel] = False
    ex = s == 0
    if x[ex][:, keep].any():
        raise MaskError("an excluded frame carries a non-zero non-dt channel")
    return int(ex.sum())


def reuse_allowed(support, input_sha, prior_input_sha) -> bool:
    """Cached grid logits may stand for every arm only when all masks are ones and x is identical."""
    s = np.asarray(support)
    return bool(s.size and np.all(s == 1) and input_sha == prior_input_sha)


# --- model side ---------------------------------------------------------------------
def require_eval(model):
    bad = [n or "<root>" for n, m in model.named_modules() if m.training]
    if bad:
        raise MaskError(f"modules in training mode: {bad[:10]}")


def masks_for(arm, support):
    if arm not in ARMS:
        raise MaskError(f"arm must be one of {ARMS}, got {arm!r}")
    ones = torch.ones_like(support)
    return (support if arm in BLOCK_SUPPORT else ones), (support if arm in POOL_SUPPORT else ones)


def diagnostic_forward(model, x, support, arm):
    """x float32 [B,T,C], support float32 [B,T] binary, every row >= 1 -> logits [B,K]."""
    require_eval(model)
    if not (isinstance(x, torch.Tensor) and isinstance(support, torch.Tensor)):
        raise MaskError("x and support must be tensors")
    if x.dtype != torch.float32 or support.dtype != torch.float32:
        raise MaskError(f"x/support must be float32, got {x.dtype}/{support.dtype}")
    if x.dim() != 3 or support.dim() != 2 or tuple(x.shape[:2]) != tuple(support.shape):
        raise MaskError(f"shape mismatch x{tuple(x.shape)} support{tuple(support.shape)}")
    if not (torch.isfinite(x).all() and torch.isfinite(support).all()):
        raise MaskError("non-finite x/support")
    if not ((support == 0) | (support == 1)).all():
        raise MaskError("support must be exactly 0/1")
    if not (support.sum(dim=1) > 0).all():
        raise MaskError("a row has no supported frame")
    block_mask, pool_mask = masks_for(arm, support)
    with torch.inference_mode():
        # same expressions and order as MaskedSignTCN.forward
        h = x.transpose(1, 2)
        m = block_mask.unsqueeze(1)
        h = model.proj(h * m) * m
        for blk in model.blocks:
            h = blk(h, m)
        p = pool_mask.unsqueeze(1)
        denom = pool_mask.sum(dim=1).clamp(min=1.0).unsqueeze(1)
        pooled = (h * p).sum(dim=2) / denom
        return model.head(pooled)


def state_digest(model) -> dict:
    """Per-tensor sha256 of state_dict (parameters and persistent buffers) + combined digest."""
    per = {}
    for k, v in model.state_dict().items():
        t = v.detach().cpu().contiguous()
        per[k] = hashlib.sha256(f"{k}|{tuple(t.shape)}|{t.dtype}|".encode() + t.numpy().tobytes()).hexdigest()
    return {"tensors": per, "combined": hashlib.sha256("|".join(f"{k}={d}" for k, d in per.items()).encode()).hexdigest()}


# --- tallies (rows: {'id','signer_id','true','arms':{arm:{'pred','correct','rank','margin'}}}) ------
def breakdown(rows, key, arms) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r[key]), {"n": 0, **{a: 0 for a in arms}})
        g["n"] += 1
        for a in arms:
            g[a] += int(r["arms"][a]["correct"])
    return dict(sorted(out.items()))


def recoveries(rows, src_ok="prior_corrupted", src_bad="grid_baseline", arms=ARMS[1:]) -> dict:
    """Rows correct in src_ok but wrong in src_bad; how many each new arm gets right."""
    sub = [r for r in rows if r["arms"][src_ok]["correct"] and not r["arms"][src_bad]["correct"]]
    return {"definition": f"{src_ok} correct and {src_bad} wrong", "n": len(sub),
            **{f"correct_in_{a}": sum(int(r["arms"][a]["correct"]) for r in sub) for a in arms},
            "rows": [{"id": r["id"], "true": r["true"], f"{src_bad}_pred": r["arms"][src_bad]["pred"],
                      **{a: {k: r["arms"][a][k] for k in ("pred", "correct", "rank")} for a in arms}}
                     for r in sub]}


def primary_effect(rows, base="grid_baseline", arm=PRIMARY) -> dict:
    """Net correct change base -> arm on one group/case, with every id that moved or stayed wrong."""
    nb = sum(int(r["arms"][base]["correct"]) for r in rows)
    na = sum(int(r["arms"][arm]["correct"]) for r in rows)
    sig = breakdown(rows, "signer_id", (base, arm))
    return {"base": base, "arm": arm, "n": len(rows), "base_correct": nb, "arm_correct": na, "net": na - nb,
            "newly_correct": [r["id"] for r in rows if r["arms"][arm]["correct"] and not r["arms"][base]["correct"]],
            "newly_wrong": [r["id"] for r in rows if r["arms"][base]["correct"] and not r["arms"][arm]["correct"]],
            "remaining_errors": [r["id"] for r in rows if not r["arms"][arm]["correct"]],
            "per_signer": {s: {base: v[base], arm: v[arm], "net": v[arm] - v[base]} for s, v in sig.items()}}


def contribution(effects_by_group) -> dict:
    """True only if the primary net vs grid_baseline is > 0 in EVERY group (burst_drop)."""
    nets = {g: e["net"] for g, e in effects_by_group.items()}
    pos = bool(nets) and all(v > 0 for v in nets.values())
    return {"net_by_group": nets, "diagnostic_contribution": pos,
            "direction_by_group": {g: "positive" if v > 0 else ("neutral" if v == 0 else "harm")
                                   for g, v in nets.items()}}
