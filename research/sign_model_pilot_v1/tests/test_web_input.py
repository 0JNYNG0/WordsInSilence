"""Checks the MediaPipe -> common input mapping before any transfer number is trusted.

A wrong mapping would make the cross-extractor result meaningless, so left/right, finger order and
the normalisation invariances are verified on the web side too, not just on the source side.

Run: py -3.13 tests/test_web_input.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import data as D  # noqa: E402
from pilot import web_input as W  # noqa: E402

WEB = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs", "ses_cbd462050a95"))
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def npz_paths(n=6):
    return sorted(glob.glob(os.path.join(WEB, "trials", "*.npz")))[:n]


@check("web clips convert to exactly the contract channel layout")
def _layout():
    for p in npz_paths(3):
        c = W.load_web_clip(p)
        assert c["x"].shape[1] == D.TOTAL_CHANNELS, c["x"].shape
        assert np.isfinite(c["x"]).all()
        assert c["frames"] > 0


@check("the nine common body slots come from the MediaPipe indices the contract names")
def _body_indices():
    assert W.MEDIAPIPE_POSE33_INDICES == (0, 11, 12, 13, 14, 15, 16, 23, 24)
    # and they land in the BODY_25 slots the source pack uses for the same joints
    p = npz_paths(1)[0]
    with np.load(p) as z:
        xy_px, usable = z["xy_px"].astype(np.float32), z["usable"].astype(bool)
    xy, present = W.to_native_like(xy_px, usable)
    for dst, src in zip((0, 5, 2, 6, 3, 7, 4, 12, 9), W.MEDIAPIPE_POSE33_INDICES):
        m = usable[:, src]
        assert np.allclose(xy[m, dst], xy_px[m, src]), (dst, src)
        assert (present[:, dst] == m).all(), (dst, src)


@check("left and right hands stay on the correct sides after mapping")
def _left_right():
    # each hand gets its own denominator: a frame where only one hand was detected must not be
    # counted against the other. Getting this wrong made an earlier version of this check read
    # 79% when the mapping was in fact 99% correct.
    hit = {25: 0, 46: 0}
    seen = {25: 0, 46: 0}
    for p in npz_paths(20):
        with np.load(p) as z:
            xy_px, usable = z["xy_px"].astype(np.float32), z["usable"].astype(bool)
        xy, present = W.to_native_like(xy_px, usable)
        for t in range(len(xy)):
            if not (present[t, 7] and present[t, 4]):     # BODY_25 LWrist / RWrist slots
                continue
            lw, rw = xy[t, 7], xy[t, 4]
            for base, own in ((25, lw), (46, rw)):
                if not present[t, base]:
                    continue
                other = rw if base == 25 else lw
                seen[base] += 1
                hit[base] += np.linalg.norm(xy[t, base] - own) < np.linalg.norm(xy[t, base] - other)
    assert min(seen.values()) > 100, seen
    for base, name in ((25, "left"), (46, "right")):
        r = hit[base] / seen[base]
        assert r > 0.9, f"{name} hand matched its own wrist only {hit[base]}/{seen[base]}"


@check("the web wrist is the local origin and fingertips sit beyond their MCP")
def _finger_order():
    ok = bad = 0
    for p in npz_paths(6):
        c = W.load_web_clip(p)
        for hand in range(2):
            base = D.CH_BODY_XY + hand * D.N_HAND * 2
            xy = c["x"][:, base:base + D.N_HAND * 2].reshape(c["frames"], D.N_HAND, 2)
            vb = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS + hand * D.N_HAND
            valid = c["x"][:, vb:vb + D.N_HAND]
            for t in range(c["frames"]):
                if valid[t].sum() < D.N_HAND:
                    continue
                assert np.allclose(xy[t, 0], 0.0, atol=1e-5)
                for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
                    ok += np.linalg.norm(xy[t, tip]) >= np.linalg.norm(xy[t, mcp])
                    bad += np.linalg.norm(xy[t, tip]) < np.linalg.norm(xy[t, mcp])
    assert ok + bad > 200, ok + bad
    assert ok / (ok + bad) > 0.6, f"fingertip beyond MCP only {ok}/{ok + bad}"


@check("web features are invariant to translation and scale, like the source features")
def _invariance():
    for p in npz_paths(3):
        with np.load(p) as z:
            xy_px, usable, ts = (z["xy_px"].astype(np.float32), z["usable"].astype(bool),
                                 z["timestamp_us"].astype(np.int64))
        xy, present = W.to_native_like(xy_px, usable)
        a = D.build_channels(xy, present, ts)["x"]
        moved = xy.copy()
        moved[present] += np.array([91.0, -55.0], dtype=np.float32)
        assert np.allclose(a, D.build_channels(moved, present, ts)["x"], atol=1e-4)
        scaled = xy.copy()
        scaled[present] *= 0.6
        assert np.allclose(a, D.build_channels(scaled, present, ts)["x"], atol=1e-4)


@check("a lost hand stays lost: the mask reports it instead of inventing a hold")
def _missing_hand_honest():
    found = False
    for p in npz_paths(20):
        with np.load(p) as z:
            xy_px, usable = z["xy_px"].astype(np.float32), z["usable"].astype(bool)
        xy, present = W.to_native_like(xy_px, usable)
        lost = ~present[:, 25]
        if lost.any():
            found = True
            assert (xy[lost, 25] == 0).all(), "absent wrist was given a position"
    assert found, "expected at least one frame with a missing hand in this session"


@check("reading the web session does not modify it")
def _read_only():
    before = {p: os.path.getmtime(p) for p in npz_paths(5)}
    for p in before:
        W.load_web_clip(p)
    for p, t in before.items():
        assert os.path.getmtime(p) == t, p


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
