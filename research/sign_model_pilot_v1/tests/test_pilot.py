"""Self-checks required by the instruction, run before trusting any number.

Covers: translation/scale normalisation, left/right and finger order, leakage and NaN from
missing data, padding length independence, train/validation signer and sequence overlap, and
saved-model reload identity.

Run: py -3.13 tests/test_pilot.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch  # noqa: E402

from pilot import data as D  # noqa: E402
from pilot.model import SignTCN  # noqa: E402

PREP = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
        "WordsInSilence_학습준비_20260912")
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


_cache = {}


def sample(split="train", n=6):
    key = (split, n)
    if key not in _cache:
        items = [x for x in D.load_manifest(PREP)
                 if x["source_only_pilot_eligible"] and x["split_role"] == split][:n]
        _cache[key] = [D.load_clip(PREP, it) for it in items]
    return _cache[key]


def raw(item):
    z = np.load(os.path.join(PREP, "pilot_npz", os.path.basename(item["npz"])))
    return z["xy_native"].copy(), z["coordinate_present"].copy(), z["timestamp_nominal_us"].copy()


@check("channel layout matches the contract exactly")
def _layout():
    assert D.TOTAL_CHANNELS == 280, D.TOTAL_CHANNELS
    assert D.CH_BODY_XY == 102 and D.CH_HAND_XY == 84
    assert D.CH_BODY_VALID == 51 and D.CH_HAND_VALID == 42
    c = sample()[0]
    assert c["x"].shape[1] == 280, c["x"].shape


@check("translating the whole clip leaves the features unchanged")
def _translation():
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]][:3]
    for it in items:
        xy, pr, ts = raw(it)
        a = D.build_channels(xy, pr, ts)["x"]
        moved = xy.copy()
        moved[pr] += np.array([137.0, -211.0], dtype=np.float32)   # only real points move
        b = D.build_channels(moved, pr, ts)["x"]
        assert np.allclose(a, b, atol=1e-4), (it["sample_id"], float(np.abs(a - b).max()))


@check("scaling the whole clip leaves the features unchanged")
def _scale():
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]][:3]
    for it in items:
        xy, pr, ts = raw(it)
        a = D.build_channels(xy, pr, ts)["x"]
        scaled = xy.copy()
        scaled[pr] *= 1.7
        b = D.build_channels(scaled, pr, ts)["x"]
        assert np.allclose(a, b, atol=1e-4), (it["sample_id"], float(np.abs(a - b).max()))


@check("hand slices are left then right, verified against the body wrists")
def _left_right():
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]][:20]
    L_WRIST_B, R_WRIST_B = 7, 4      # BODY_25 native indices
    # per-hand denominators; a frame with only one hand detected must not count against the other
    hit = {0: 0, 1: 0}
    seen = {0: 0, 1: 0}
    for it in items:
        xy, pr, _ = raw(it)
        for t in range(len(xy)):
            if not (pr[t, L_WRIST_B] and pr[t, R_WRIST_B]):
                continue
            lw, rw = xy[t, L_WRIST_B], xy[t, R_WRIST_B]
            for hi, sl in enumerate(D.NATIVE_HAND_SLICES):
                if not pr[t, sl[0]]:
                    continue
                h = xy[t, sl[0]]
                own, other = (lw, rw) if hi == 0 else (rw, lw)
                seen[hi] += 1
                hit[hi] += np.linalg.norm(h - own) < np.linalg.norm(h - other)
    assert min(seen.values()) > 100, seen
    for hi, name in ((0, "left"), (1, "right")):
        r = hit[hi] / seen[hi]
        assert r > 0.95, f"{name} slice matched its own wrist only {hit[hi]}/{seen[hi]}"


@check("finger order is preserved: wrist 0, then thumb, index, middle, ring, pinky")
def _finger_order():
    """Each MCP must sit further from the wrist than the wrist itself, and the chain must be
    monotone outward on a spread hand - checked as tip further than MCP for most frames."""
    ok = bad = 0
    for c in sample(n=8):
        for hand in range(2):
            base = D.CH_BODY_XY + hand * D.N_HAND * 2
            xy = c["x"][:, base:base + D.N_HAND * 2].reshape(len(c["x"]), D.N_HAND, 2)
            vb = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS + hand * D.N_HAND
            valid = c["x"][:, vb:vb + D.N_HAND]
            for t in range(len(xy)):
                if valid[t].sum() < D.N_HAND:
                    continue
                assert np.allclose(xy[t, 0], 0.0, atol=1e-5), "wrist must be the local origin"
                for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
                    if np.linalg.norm(xy[t, tip]) >= np.linalg.norm(xy[t, mcp]):
                        ok += 1
                    else:
                        bad += 1
    assert ok + bad > 200, ok + bad
    assert ok / (ok + bad) > 0.6, f"fingertip beyond MCP only {ok}/{ok + bad} - order suspect"


@check("missing coordinates produce no NaN and leak nothing into the values")
def _missing():
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]][:5]
    for it in items:
        xy, pr, ts = raw(it)
        out = D.build_channels(xy, pr, ts)
        assert np.isfinite(out["x"]).all(), it["sample_id"]
        # every coordinate whose validity flag is 0 must be exactly zero, and changing the
        # underlying garbage must not change the output
        vb = D.CH_BODY_XY + D.CH_HAND_XY
        body_valid = out["x"][:, vb:vb + D.N_POINTS]
        body_xy = out["x"][:, :D.CH_BODY_XY].reshape(len(out["x"]), D.N_POINTS, 2)
        assert np.abs(body_xy[body_valid == 0]).max(initial=0.0) == 0.0, it["sample_id"]
        noisy = xy.copy()
        noisy[~pr] = 9999.0
        b = D.build_channels(noisy, pr, ts)["x"]
        assert np.allclose(out["x"], b), f"{it['sample_id']}: invalid points influenced the output"


@check("trailing padding does not change a clip's output")
def _padding():
    torch.manual_seed(0)
    model = SignTCN()
    model.eval()
    clips = sample(n=4)
    with torch.no_grad():
        x1, p1, _ = D.collate([clips[0]])
        alone = model(torch.from_numpy(x1), torch.from_numpy(p1))
        # same clip batched with a much longer one, so it receives real padding
        longest = max(clips, key=lambda c: c["frames"])
        x2, p2, _ = D.collate([clips[0], longest])
        padded = model(torch.from_numpy(x2), torch.from_numpy(p2))[0:1]
    assert torch.allclose(alone, padded, atol=1e-5), \
        f"padding changed the logits by {float((alone - padded).abs().max())}"


@check("train and validation share no signer and no source sequence")
def _split():
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]]
    tr = {x["signer_id"] for x in items if x["split_role"] == "train"}
    va = {x["signer_id"] for x in items if x["split_role"] == "validation"}
    te = {x["signer_id"] for x in items if x["split_role"] == "reserved_test"}
    assert not (tr & va) and not (tr & te) and not (va & te), (tr, va, te)
    seq_tr = {x["original_sequence_id"] for x in items if x["split_role"] == "train"}
    seq_va = {x["original_sequence_id"] for x in items if x["split_role"] == "validation"}
    assert not (seq_tr & seq_va), sorted(seq_tr & seq_va)[:5]
    assert sum(1 for x in items if x["split_role"] == "train") == 98
    assert sum(1 for x in items if x["split_role"] == "validation") == 32
    assert sum(1 for x in items if x["split_role"] == "reserved_test") == 33


@check("the model stays under the contract's 100k parameters")
def _size():
    n = SignTCN().parameter_count()
    assert n < 100000, n


@check("a saved model reloads to identical outputs")
def _reload():
    torch.manual_seed(1)
    model = SignTCN()
    model.eval()
    c = sample(n=1)
    x, pad, _ = D.collate(c)
    with torch.no_grad():
        before = model(torch.from_numpy(x), torch.from_numpy(pad))
    tmp = os.path.join(HERE, "_reload_check.pt")
    torch.save(model.state_dict(), tmp)
    again = SignTCN()
    again.load_state_dict(torch.load(tmp, weights_only=True))
    again.eval()
    with torch.no_grad():
        after = again(torch.from_numpy(x), torch.from_numpy(pad))
    os.remove(tmp)
    assert torch.allclose(before, after, atol=1e-6), float((before - after).abs().max())


@check("the source confidence column is never used as observation quality")
def _confidence_unused():
    src = open(os.path.join(ROOT, "src", "pilot", "data.py"), encoding="utf-8").read()
    assert "confidence_native" not in src.split('"""')[2], \
        "confidence_native must not be read in the code body"
    # the point of this check: confidence is 1 even where a coordinate is missing, so the two
    # must never be conflated. Absences are sparse (about 6k of 355k samples), so scan the pack
    # rather than one clip.
    items = [x for x in D.load_manifest(PREP) if x["source_only_pilot_eligible"]]
    conf_all_one = True
    absent = present_total = 0
    for it in items:
        z = np.load(os.path.join(PREP, "pilot_npz", os.path.basename(it["npz"])))
        conf_all_one &= bool((z["confidence_native"] == 1).all())
        absent += int((~z["coordinate_present"]).sum())
        present_total += z["coordinate_present"].size
    assert conf_all_one, "source confidence was expected to be 1 everywhere"
    assert absent > 0, "expected some absent coordinates despite confidence 1"
    print(f"      (confidence 1 everywhere; {absent} of {present_total} coordinates absent)")


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
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
