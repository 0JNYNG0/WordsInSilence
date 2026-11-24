"""Checks the claim the whole experiment rests on: these features survive a hand rotation.

If flexion angles are not actually rotation invariant, every number produced from them is
meaningless, so that is verified directly rather than assumed.

Run: py -3.13 tests/test_angles.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import angles as A  # noqa: E402
from pilot import data as D  # noqa: E402
from pilot import web_input as W  # noqa: E402

WEB = os.path.normpath(os.path.join(ROOT, "..", "front_keyphase_v1", "runs", "ses_cbd462050a95"))
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def clips(n=4):
    return [W.load_web_clip(p) for p in sorted(glob.glob(os.path.join(WEB, "trials", "*.npz")))[:n]]


def hand_of(c, h=1):
    t = len(c["x"])
    vb = D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS
    pts = c["x"][:, D.CH_BODY_XY + h * 42:D.CH_BODY_XY + (h + 1) * 42].reshape(t, 21, 2)
    return pts, c["x"][:, vb + h * 21:vb + (h + 1) * 21] > 0.5


def rotate(pts, deg):
    r = np.deg2rad(deg)
    m = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]], dtype=np.float32)
    return pts @ m.T


@check("rotating the hand leaves every flexion angle unchanged")
def _rotation_invariant():
    for c in clips():
        pts, v = hand_of(c)
        if not v.any():
            continue
        base, ok = A.hand_angles(pts, v)
        for deg in (17.0, 90.0, 180.0, -63.5):
            rot, ok2 = A.hand_angles(rotate(pts, deg), v)
            assert (ok == ok2).all(), deg
            assert np.allclose(base[ok], rot[ok], atol=2e-3), (deg, np.abs(base[ok] - rot[ok]).max())


@check("the wrist-relative coordinates it replaces are NOT rotation invariant")
def _xy_is_not():
    # the control: if the old features were already invariant there would be nothing to fix
    c = clips(1)[0]
    pts, v = hand_of(c)
    assert v.any()
    moved = np.abs(rotate(pts, 90.0)[v] - pts[v]).max()
    assert moved > 0.1, f"rotation barely moved the coordinates ({moved})"


@check("scaling and translating the hand leaves the angles unchanged")
def _scale_translate():
    for c in clips(2):
        pts, v = hand_of(c)
        if not v.any():
            continue
        base, ok = A.hand_angles(pts, v)
        for mod in (pts * 3.7, pts + np.array([5.0, -2.0], dtype=np.float32)):
            out, ok2 = A.hand_angles(mod, v)
            assert (ok == ok2).all()
            assert np.allclose(base[ok], out[ok], atol=2e-3)


@check("a curled finger reads as a smaller angle than a straight one")
def _bend_direction():
    straight = np.zeros((1, 21, 2), dtype=np.float32)
    curled = np.zeros((1, 21, 2), dtype=np.float32)
    for k, j in enumerate((5, 6, 7, 8)):                 # index finger, wrist at origin
        straight[0, j] = (0.0, 1.0 + k)
    curled[0, 5], curled[0, 6], curled[0, 7], curled[0, 8] = ((0, 1), (0, 2), (0.7, 2.2), (0.7, 1.4))
    v = np.zeros((1, 21), dtype=bool)
    v[0, [5, 6, 7, 8]] = True
    a_s, ok_s = A.hand_angles(straight, v)
    a_c, ok_c = A.hand_angles(curled, v)
    # index PIP is the 5th angle overall (thumb takes 0-2, index 3-5); index PIP = chain(5,6,7)
    assert ok_s[0, 4] and ok_c[0, 4]
    assert a_s[0, 4] > 0.95, a_s[0, 4]                   # straight -> near 1.0 (180 degrees)
    assert a_c[0, 4] < a_s[0, 4] - 0.2, (a_c[0, 4], a_s[0, 4])


@check("a missing point invalidates only the angles that use it, and leaks no NaN")
def _missing_points():
    c = clips(1)[0]
    pts, v = hand_of(c)
    assert v.any()
    full, ok_full = A.hand_angles(pts, v)
    v2 = v.copy()
    v2[:, 8] = False                                     # index fingertip gone
    out, ok2 = A.hand_angles(pts, v2)
    assert np.isfinite(out).all()
    assert (out[~ok2] == 0).all(), "an invalid angle was given a value"
    lost = ok_full & ~ok2
    assert lost.any(), "dropping the index tip invalidated nothing"
    kept = ok_full & ok2
    assert np.allclose(full[kept], out[kept], atol=1e-6), "an unrelated angle changed"


@check("the variant builder keeps the body block and swaps only the hand channels")
def _variant_layout():
    c = clips(1)[0]
    xy, _ = A.variant(c, hand_xy=True, use_angles=False, orientation=False)
    ang, m = A.variant(c, hand_xy=False, use_angles=True, orientation=True)
    assert xy.shape[1] == D.CH_BODY_XY + D.CH_HAND_XY, xy.shape
    assert ang.shape[1] == D.CH_BODY_XY + 2 * (A.N_ANGLES + 2), ang.shape
    assert np.array_equal(xy[:, :D.CH_BODY_XY], ang[:, :D.CH_BODY_XY])
    assert m.shape == ang.shape and np.isfinite(ang).all()


@check("reading the web session does not modify it")
def _read_only():
    ps = sorted(glob.glob(os.path.join(WEB, "trials", "*.npz")))[:4]
    before = {p: os.path.getmtime(p) for p in ps}
    clips(4)
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
