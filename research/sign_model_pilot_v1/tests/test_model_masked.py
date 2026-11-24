"""Checks the masked-normalisation candidate, anchored to the defect it is meant to fix.

Check 0 reproduces the original model's train-mode padding sensitivity first, so the rest is tied
to a real failure rather than asserted in the abstract. Dropout is 0 everywhere, optimizer step
count is 0, and both models start from identical weights; the only difference between the two
forward passes is masked-zero padding appended to the batch.

Run: py -3.13 tests/test_model_masked.py
"""
import copy
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.model import SignTCN  # noqa: E402
from pilot.model_masked_v1 import MaskedBatchNorm1d, MaskedSignTCN  # noqa: E402

CHECKS = []
MEASURED = {}
B, T, PAD = 2, 31, 62


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def batch(seed=0, t=T, pad=0, scale=1.0):
    """Same valid content every time; `pad` masked-zero frames appended."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, t, D.TOTAL_CHANNELS, generator=g) * scale
    m = torch.ones(B, t)
    if pad:
        x = torch.cat([x, torch.zeros(B, pad, D.TOTAL_CHANNELS)], dim=1)
        m = torch.cat([m, torch.zeros(B, pad)], dim=1)
    return x, m


def fresh(cls, seed=7, dropout=0.0):
    torch.manual_seed(seed)
    net = cls(dropout=dropout)
    net.train()
    return net


def run_pair(cls, need_grad=False):
    """Same weights, same valid input, with and without padding -> (out_a, out_b, nets)."""
    a, b = fresh(cls), fresh(cls)
    b.load_state_dict(copy.deepcopy(a.state_dict()))
    xa, ma = batch(pad=0)
    xb, mb = batch(pad=PAD)
    if need_grad:
        xa.requires_grad_(True)
        xb.requires_grad_(True)
    oa, ob = a(xa, ma), b(xb, mb)
    return oa, ob, a, b, xa, xb


@check("0. the original model IS padding sensitive in train mode (the defect being fixed)")
def _original_defect():
    oa, ob, a, b, _, _ = run_pair(SignTCN)
    out_diff = float((oa - ob).abs().max().detach())
    rm_diff = max(float((pa - pb).abs().max())
                  for (na, pa), (nb, pb) in zip(a.named_buffers(), b.named_buffers())
                  if "running_mean" in na)
    ea, eb = fresh(SignTCN), fresh(SignTCN)
    eb.load_state_dict(copy.deepcopy(ea.state_dict()))
    ea.eval(); eb.eval()
    with torch.no_grad():
        ed = float((ea(*batch(pad=0)) - eb(*batch(pad=PAD))).abs().max())
    MEASURED["original_model"] = {"train_max_output_diff": round(out_diff, 4),
                                  "train_max_running_mean_diff": round(rm_diff, 5),
                                  "eval_max_output_diff": ed}
    assert out_diff > 1.0, f"expected the reported train-mode sensitivity, got {out_diff}"
    assert rm_diff > 0.01, f"expected running_mean to differ, got {rm_diff}"
    assert ed < 1e-5, f"eval mode was supposed to look clean, got {ed}"


@check("1. masked model: padding changes neither train output, nor gradients, nor running stats")
def _masked_train_invariant():
    oa, ob, a, b, xa, xb = run_pair(MaskedSignTCN, need_grad=True)
    out_diff = float((oa - ob).abs().max().detach())
    oa.sum().backward()
    ob.sum().backward()
    grad_diff = float((xa.grad[:, :T] - xb.grad[:, :T]).abs().max())
    pad_grad = float(xb.grad[:, T:].abs().max())
    buf_diff = max(float((pa - pb).abs().max())
                   for (_, pa), (_, pb) in zip(a.named_buffers(), b.named_buffers())
                   if pa.dtype.is_floating_point)
    MEASURED["masked_model"] = {"train_max_output_diff": out_diff,
                                "valid_input_grad_max_diff": grad_diff,
                                "grad_leaking_into_padding": pad_grad,
                                "max_running_stat_diff": buf_diff}
    assert out_diff < 1e-5, out_diff
    assert grad_diff < 1e-5, grad_diff
    assert pad_grad < 1e-6, f"padding received gradient: {pad_grad}"
    assert buf_diff < 1e-6, buf_diff


@check("2. masked model: eval output is unaffected by padding length")
def _masked_eval_invariant():
    net = fresh(MaskedSignTCN)
    net.eval()
    with torch.no_grad():
        base = net(*batch(pad=0))
        for pad in (1, 17, PAD, 200):
            assert float((base - net(*batch(pad=pad))).abs().max()) < 1e-5, pad


@check("3. changing content inside the valid mask does change the output")
def _not_ignoring_input():
    net = fresh(MaskedSignTCN)
    net.eval()
    with torch.no_grad():
        a = net(*batch(seed=0, pad=PAD))
        b = net(*batch(seed=1, pad=PAD))
        c = net(*batch(seed=0, pad=PAD, scale=1.6))
    assert float((a - b).abs().max()) > 1e-3, "different valid content gave the same output"
    assert float((a - c).abs().max()) > 1e-3, "scaled valid content gave the same output"


@check("4. the masked statistics equal an independent mean/var over valid positions only")
def _statistics_match():
    torch.manual_seed(3)
    ch = 5
    norm = MaskedBatchNorm1d(ch)
    norm.train()
    x = torch.randn(4, ch, 20)
    m = torch.ones(4, 1, 20)
    m[:, :, 13:] = 0.0
    x = x * m                                   # padded columns are literal zeros, as in the model
    y = norm(x, m)

    sel = x[:, :, :13].reshape(4, ch, -1).transpose(0, 1).reshape(ch, -1)
    mean = sel.mean(dim=1)
    var = sel.var(dim=1, unbiased=False)
    expect = (x - mean.view(1, -1, 1)) / torch.sqrt(var.view(1, -1, 1) + norm.eps)
    expect = (expect * norm.weight.view(1, -1, 1) + norm.bias.view(1, -1, 1)) * m
    gap = float((y - expect).abs().max().detach())
    assert gap < 1e-5, gap

    # the running mean must follow the valid-only mean, not one diluted by 35 percent zeros
    assert float((norm.running_mean - 0.1 * mean).abs().max()) < 1e-6
    naive = (x.transpose(0, 1).reshape(ch, -1)).mean(dim=1)
    MEASURED["statistics"] = {
        "valid_positions": int(m.sum()), "padded_positions": int((1 - m).sum()),
        "max_gap_between_valid_only_and_all_position_mean": round(
            float((mean - naive).abs().max()), 5)}
    assert float((mean - naive).abs().max()) > 1e-3, "the test batch cannot tell the two apart"


@check("5. degenerate batches produce no NaN and write statistics only when they can")
def _degenerate():
    torch.manual_seed(5)
    norm = MaskedBatchNorm1d(3)
    norm.train()
    x = torch.randn(2, 3, 6)
    zero = torch.zeros(2, 1, 6)
    before = (norm.running_mean.clone(), norm.running_var.clone())
    y = norm(x * zero, zero)
    assert torch.isfinite(y).all() and float(y.abs().max()) == 0.0
    assert torch.equal(norm.running_mean, before[0]) and torch.equal(norm.running_var, before[1])
    assert norm.last_valid_count == 0

    one = torch.zeros(2, 1, 6)
    one[0, 0, 0] = 1.0
    var_before = norm.running_var.clone()
    y = norm(x * one, one)
    assert torch.isfinite(y).all(), "a single valid position produced NaN"
    assert norm.last_valid_count == 1
    assert torch.equal(norm.running_var, var_before), "unbiased variance written from one sample"
    assert not torch.equal(norm.running_mean, before[0]), "the single sample was ignored entirely"


@check("6. state dict interchanges with model.SignTCN, and identity/size are reported")
def _interchange_and_identity():
    torch.manual_seed(7)
    plain = SignTCN(dropout=0.0)
    torch.manual_seed(7)
    masked = MaskedSignTCN(dropout=0.0)
    assert plain.parameter_count() == masked.parameter_count(), (plain.parameter_count(),
                                                                 masked.parameter_count())
    assert set(plain.state_dict()) == set(masked.state_dict()), "state dict keys diverge"
    masked.load_state_dict(plain.state_dict())
    plain.eval(); masked.eval()
    with torch.no_grad():
        x, m = batch(pad=PAD)
        assert float((plain(x, m) - masked(x, m)).abs().max()) < 1e-5, \
            "eval compatibility with the existing checkpoint format is broken"
    p = os.path.join(ROOT, "runs", "_masked_roundtrip.pt")
    torch.save({"state_dict": masked.state_dict(), "model_type": masked.model_type,
                "model_version": masked.model_version}, p)
    ck = torch.load(p, weights_only=True)
    again = MaskedSignTCN(dropout=0.0)
    again.load_state_dict(ck["state_dict"])
    again.eval()
    with torch.no_grad():
        assert float((masked(x, m) - again(x, m)).abs().max()) == 0.0
    os.remove(p)
    MEASURED["identity"] = {**masked.identity(),
                            "state_dict_keys_match_SignTCN": True,
                            "eval_equality_with_loaded_SignTCN_weights": "compatibility only; the "
                            "loaded running statistics were still estimated with padding mixed in "
                            "and are NOT corrected by loading"}


@check("7. the frozen files stay byte identical")
def _frozen():
    frozen = {
        "src/pilot/model.py": "01138e88f74aa16f7cf0e2a3c84c7488a7dbdd8529dbbe205ea39d95254741b8",
        "src/pilot/data.py": "5daa52beee310572c4298c74702fdabf08f0931ce38cff9f20201b341f1aa81d",
        "src/pilot/axes.py": "e3c00071c56b89119e6735e2a2818a20d43fb38915ae40681e81991f0d645aa0",
        "runs/expanded_v1/model.pt":
            "6dd24017631dab6dd32c62f38cd8020d6aabd973d1a2914efd705eb6983b1bfe",
    }
    got = {}
    for rel, want in frozen.items():
        h = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        got[rel] = h == want
        assert h == want, f"{rel} changed"
    MEASURED["hash_verified_unchanged"] = got
    # the pilot checkpoint has no recorded prior hash here, so only its presence is checked.
    # Reporting that as "unchanged" would overstate it.
    first = os.path.join(ROOT, "runs", "first_run", "model.pt")
    MEASURED["existence_only_not_hash_verified"] = {
        "runs/first_run/model.pt": os.path.exists(first)}
    assert os.path.exists(first), "the pilot checkpoint is missing"


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=float))
    with open(os.path.join(ROOT, "evidence", "autonomous_02_checks.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump({"passed": len(CHECKS) - failed, "total": len(CHECKS), "measured": MEASURED},
                  fh, ensure_ascii=False, indent=2, default=float)
        fh.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
