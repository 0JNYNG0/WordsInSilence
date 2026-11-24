"""R11: condition C of the functional_08 diagnostic, after the independent counterexample.

The first version gated the whole resampling on runs of HAND observation and interpolated even at
times that landed exactly on an original frame. Against a 7-frame input already at 30 Hz with the
hand missing on one frame, it dropped an observed hand boundary AND erased the body wherever the
hand was absent (time_resample_counterexample.json). These checks reproduce that shape first and
then require the corrected behaviour.

Synthetic arrays only. No model, no camera, no training, no web file.

Run: py -3.13 tests/test_r11_time_resample.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import tools_diag_time_conditions as D  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic native arrays; not a measurement of anyone"}
NL = chr(10)
N_POINTS = 67
HAND = D.HAND_SLOTS            # slice(25, 67)
BODY = slice(0, 25)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def at_30hz(n=7, rounding="rint"):
    """n frames already on the training grid, body always seen, both hands seen.

    The source's integer microseconds may be rounded or truncated; both are legitimate and the
    two differ by under a microsecond at the end of a clip. That difference used to cost the last
    original sample, so both conventions are exercised.
    """
    k = np.arange(n) * D.TRAIN_STEP_US
    ts = (np.rint(k) if rounding == "rint" else np.floor(k)).astype(np.int64)
    xy = np.zeros((n, N_POINTS, 2), dtype=np.float32)
    for t in range(n):
        xy[t, :, 0] = np.arange(N_POINTS) + t          # distinct, so a copy is provable
        xy[t, :, 1] = t
    present = np.ones((n, N_POINTS), dtype=bool)
    return xy, present, ts


@check("an input already at 30 Hz comes back unchanged, frame for frame")
def _identity():
    xy, pr, ts = at_30hz()
    a, p, t = D.condition_c(xy, pr, ts)
    assert len(a) == len(xy), (len(a), len(xy))
    assert np.array_equal(p, pr), "a mask changed on an identity resample"
    assert np.allclose(a, xy), "a coordinate moved on an identity resample"
    assert np.all(np.diff(t) > 0)
    MEASURED["identity"] = {"frames_in": int(len(xy)), "frames_out": int(len(a))}


@check("the last original sample survives both integer-microsecond conventions")
def _end_boundary():
    seen = {}
    for rounding in ("rint", "trunc"):
        for n in (6, 7, 32):
            xy, pr, ts = at_30hz(n, rounding)
            a, p, t = D.condition_c(xy, pr, ts)
            assert len(a) == n, f"{rounding} n={n} produced {len(a)} frames"
            assert np.allclose(a[-1], xy[-1]), f"{rounding} n={n} lost the last original sample"
            assert np.array_equal(p, pr)
            # nothing past the original end is emitted
            assert int(t[-1]) == int(ts[-1] - ts[0]), (rounding, n, int(t[-1]))
            seen[f"{rounding}_{n}"] = int(len(a))
    MEASURED["end_boundary"] = seen


@check("a missing hand never deletes the body, and the observed hand boundary survives")
def _hand_gap_keeps_body():
    xy, pr, ts = at_30hz(7)
    pr[3, HAND] = False                                # only the hand, only on frame 3
    a, p, t = D.condition_c(xy, pr, ts)
    body = p[:, BODY].all(axis=1).tolist()
    hand = p[:, HAND].any(axis=1).tolist()
    assert body == [True] * 7, f"the body was erased where the hand was missing: {body}"
    assert hand == [True, True, True, False, True, True, True], f"hand boundary lost: {hand}"
    assert np.allclose(a[2], xy[2]) and np.allclose(a[4], xy[4]), "a boundary frame was altered"
    MEASURED["hand_gap"] = {"body_present": body, "hand_present": hand}


@check("between two original frames a point exists only if both sides saw it")
def _pointwise_gap():
    n = 3
    ts = np.array([0, 100_000, 200_000], dtype=np.int64)   # 10 Hz: targets fall between frames
    xy = np.zeros((n, N_POINTS, 2), dtype=np.float32)
    xy[:, :, 0] = np.arange(N_POINTS)
    xy[1, :, 0] += 10.0
    xy[2, :, 0] += 20.0
    pr = np.ones((n, N_POINTS), dtype=bool)
    pr[1, 30] = False                                      # one point missing on the middle frame
    a, p, t = D.condition_c(xy, pr, ts)
    exact = [k for k in range(len(t))
             if any(abs(int(ts[i]) - int(ts[0] + round(k * D.TRAIN_STEP_US))) <= D.EXACT_TOL_US
                    for i in range(n))]
    between = [k for k in range(len(p)) if k not in exact]
    assert between, "the fixture produced no interpolated target"
    touching = [k for k in between if k > 0 and k < len(p) - 1]
    assert any(not p[k, 30] for k in touching), "a point missing on one side was invented"
    assert all(p[k, 29] for k in touching), "a fully observed point was dropped"
    MEASURED["pointwise_gap"] = {"out_frames": int(len(p)),
                                 "point30_observed": int(p[:, 30].sum()),
                                 "point29_observed": int(p[:, 29].sum())}


@check("no target is made past the original end, and an isolated exact sample survives")
def _bounds_and_isolated():
    xy, pr, ts = at_30hz(5)
    pr[:, HAND] = False
    pr[2, HAND] = True                                  # one isolated observed hand frame
    a, p, t = D.condition_c(xy, pr, ts)
    assert len(a) <= len(xy), "a target was made past the original end"
    assert p[2, HAND].any(), "an isolated exact observation was dropped"
    assert not p[1, HAND].any() and not p[3, HAND].any()
    # the last output sample may not sit beyond the last original timestamp
    span_us = int(ts[-1] - ts[0])
    assert int(t[-1]) <= span_us
    MEASURED["bounds"] = {"frames_in": int(len(xy)), "frames_out": int(len(a)),
                          "isolated_hand_kept": True}


@check("the caller's arrays are not modified")
def _input_unchanged():
    xy, pr, ts = at_30hz(6)
    pr[4, HAND] = False
    xy0, pr0, ts0 = xy.copy(), pr.copy(), ts.copy()
    D.condition_c(xy, pr, ts)
    assert np.array_equal(xy, xy0) and np.array_equal(pr, pr0) and np.array_equal(ts, ts0)


@check("the exact-copy tolerance is stated and is one microsecond")
def _tolerance_stated():
    assert D.EXACT_TOL_US == 1
    doc = " ".join(D.condition_c.__doc__.split())      # the contract wraps across lines
    assert "EXACT_TOL_US = 1 us" in doc and "VERBATIM" in doc
    assert "a missing hand never deletes the body" in doc
    assert "No target is generated beyond the original last timestamp" in doc
    assert "SNAPPED to the original end" in doc
    assert "actual target times minus the start" in doc


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
