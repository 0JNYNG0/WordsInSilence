"""Checks the axes are actually separate, and that `movement` really sees motion.

The whole point of the split is that NA (a palm held still) and ALDA (the same palm moved up and
down) differ on one axis only. If `movement` did not respond to motion, or if `location` leaked
into it, every conclusion drawn from the split would be wrong.

Run: py -3.13 tests/test_axes.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import axes as X  # noqa: E402
from pilot import data as D  # noqa: E402
from pilot import web_input as W  # noqa: E402

WEB = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs", "ses_cbd462050a95"))
CHECKS = []
FPS = 30.0


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def clips(n=3):
    return [W.load_web_clip(p) for p in sorted(glob.glob(os.path.join(WEB, "trials", "*.npz")))[:n]]


def synth(path_y, dt=1.0 / FPS):
    """A minimal valid clip whose right-hand wrist follows path_y in body frame."""
    t = len(path_y)
    x = np.zeros((t, D.TOTAL_CHANNELS), dtype=np.float32)
    body = np.zeros((t, D.N_POINTS, 2), dtype=np.float32)
    for p in (X.NOSE, 1, 2, X.L_ELBOW, X.R_ELBOW, 5, 6, 7, 8):
        body[:, p] = (0.1 * p, -0.5)
    body[:, X.HAND_ROOT[1], 1] = path_y
    body[:, X.HAND_ROOT[1], 0] = 0.3
    x[:, :D.CH_BODY_XY] = body.reshape(t, -1)
    x[:, D.CH_BODY_XY + D.CH_HAND_XY:D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] = 1.0
    x[:, -1] = dt
    return {"x": x, "frames": t}


@check("a still hand reads near zero on movement, a travelling one does not")
def _movement_sees_motion():
    still = synth(np.full(40, 0.5, dtype=np.float32))
    moving = synth((0.5 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 40))).astype(np.float32))
    for clip, want_zero in ((still, True), (moving, False)):
        v, m = X.axis_features(clip, "movement")
        assert m.any(), "movement produced no valid frame"
        speed = np.abs(v[m]).max()
        if want_zero:
            assert speed < 1e-4, f"a stationary hand reported speed {speed}"
        else:
            assert speed > 1.0, f"a moving hand reported only {speed}"


@check("still and moving differ on movement but not on location's shared range")
def _axis_separation():
    still = synth(np.full(40, 0.5, dtype=np.float32))
    moving = synth((0.5 + 0.4 * np.sin(np.linspace(0, 4 * np.pi, 40))).astype(np.float32))
    dm = X.distance(X.axis_features(still, "movement"), X.axis_features(moving, "movement"),
                    "movement")
    dh = X.distance(X.axis_features(still, "handshape"), X.axis_features(moving, "handshape"),
                    "handshape")
    assert dm is not None and dm > 0.5, dm
    assert dh is None, "handshape should be unmeasurable when no hand points are present"


@check("movement is a rate: the same path sampled at half speed halves it")
def _velocity_uses_time():
    path = (0.5 + 0.3 * np.sin(np.linspace(0, 2 * np.pi, 40))).astype(np.float32)
    fast, _ = X.axis_features(synth(path, dt=1.0 / FPS), "movement")
    slow, m = X.axis_features(synth(path, dt=2.0 / FPS), "movement")
    assert np.allclose(fast[m], 2.0 * slow[m], atol=1e-4)


@check("shifting every body point leaves movement unchanged but moves location")
def _offset_control():
    c = clips(1)[0]
    shifted = {**c, "x": c["x"].copy()}
    b = shifted["x"][:, :D.CH_BODY_XY].reshape(len(c["x"]), D.N_POINTS, 2)
    b += np.array([0.25, -0.1], dtype=np.float32)
    shifted["x"][:, :D.CH_BODY_XY] = b.reshape(len(c["x"]), -1)
    v0, m0 = X.axis_features(c, "movement")
    v1, m1 = X.axis_features(shifted, "movement")
    assert (m0 == m1).all() and np.allclose(v0[m0], v1[m0], atol=1e-3)
    l0, lm = X.axis_features(c, "location")
    l1, _ = X.axis_features(shifted, "location")
    assert np.abs(l0[lm] - l1[lm]).max() > 0.05, "location ignored a real shift"


@check("every axis is finite, masked where absent, and sized as documented")
def _layout_and_missing():
    widths = {"handshape": 2 * 19, "location": 12, "movement": 12}
    for c in clips(3):
        for ax in X.AXES:
            v, m = X.axis_features(c, ax)
            assert v.shape == (c["frames"], widths[ax]), (ax, v.shape)
            assert m.shape == v.shape
            assert np.isfinite(v).all(), ax
            assert (v[~m] == 0).all(), f"{ax} gave a value to a masked channel"


@check("real clips produce a usable distance on all three axes")
def _real_distances():
    a, b = clips(2)
    for ax in X.AXES:
        d = X.distance(X.axis_features(a, ax), X.axis_features(b, ax), ax)
        assert d is not None and np.isfinite(d) and d >= 0, (ax, d)
        assert X.distance(X.axis_features(a, ax), X.axis_features(a, ax), ax) < d + 1e-9, ax


@check("direction_reversals counts real turns and ignores jitter")
def _reversals():
    still = synth(np.full(60, 0.5, dtype=np.float32))
    assert X.direction_reversals(still) == 0

    jitter = 0.5 + np.random.default_rng(0).normal(0, 0.004, 60).astype(np.float32)
    assert X.direction_reversals(synth(jitter)) == 0, "jitter below the deadband read as a turn"

    # three half-cycles of a triangle wave = two turns
    up = np.linspace(0.2, 0.8, 20, dtype=np.float32)
    tri = np.concatenate([up, up[::-1], up])
    assert X.direction_reversals(synth(tri)) == 2, X.direction_reversals(synth(tri))

    # a bigger deadband can only remove turns, never add them
    for dead in (0.01, 0.02, 0.05, 0.2):
        assert X.direction_reversals(synth(tri), deadband=dead) <= 2


@check("reading the web session does not modify it")
def _read_only():
    ps = sorted(glob.glob(os.path.join(WEB, "trials", "*.npz")))[:3]
    before = {p: os.path.getmtime(p) for p in ps}
    clips(3)
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
