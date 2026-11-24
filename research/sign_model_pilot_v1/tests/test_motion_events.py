"""Regression checks for movement_events (v2), against the failures external review found in v1.

The first check REPRODUCES the documented v1 counterexample rather than asserting it away: the same
path returns 3 turns at 1s/30fps and 0 at both 4s/30fps and 1s/120fps. Every later check then
requires v2 not to do that. A fix that is not anchored to the failure it fixes is not a fix.

The remaining checks are the ones the review asked for: same path slow and fast, a frame-rate
change, real stillness with small jitter, a genuine change in repetition count, a long observation
gap, and not enough observation at all.

Run: py -3.13 tests/test_motion_events.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import axes as X  # noqa: E402
from pilot import data as D  # noqa: E402

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def path(waypoints, seconds, fps):
    """Waypoints walked at constant speed, sampled at fps for `seconds`."""
    n = max(2, int(round(seconds * fps)))
    return np.interp(np.linspace(0, len(waypoints) - 1, n),
                     np.arange(len(waypoints)), np.asarray(waypoints, dtype=np.float64))


def synth(y_right, fps=30.0, y_left=None, right_valid=None, left_valid=None):
    """A clip whose wrists follow the given vertical paths, with explicit per-hand validity."""
    t = len(y_right)
    x = np.zeros((t, D.TOTAL_CHANNELS), dtype=np.float32)
    body = np.zeros((t, D.N_POINTS, 2), dtype=np.float32)
    bval = np.zeros((t, D.N_POINTS), dtype=np.float32)
    for p in range(D.N_BODY):
        body[:, p] = (0.1 * p, -0.5)
        bval[:, p] = 1.0
    lroot, rroot = X.HAND_ROOT
    body[:, rroot, 0] = 0.3
    body[:, rroot, 1] = y_right
    bval[:, rroot] = 1.0 if right_valid is None else right_valid
    if y_left is not None:
        body[:, lroot, 0] = -0.3
        body[:, lroot, 1] = y_left
        bval[:, lroot] = 1.0 if left_valid is None else left_valid
    x[:, :D.CH_BODY_XY] = body.reshape(t, -1)
    x[:, D.CH_BODY_XY + D.CH_HAND_XY:D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] = bval
    x[1:, -1] = 1.0 / fps
    return {"x": x, "frames": t}


WAY = [0.0, 0.2, 0.0, 0.2, 0.0]      # up, down, up, down -> 3 direction changes


def turns(clip, **kw):
    return X.movement_events(clip, **kw)["right"]["turns"]


@check("v1's documented failure is reproduced, not assumed away")
def _reproduce_v1():
    got = {name: X.direction_reversals(synth(path(WAY, sec, fps), fps=fps))
           for name, sec, fps in (("1s_30fps", 1.0, 30.0), ("4s_30fps", 4.0, 30.0),
                                  ("1s_120fps", 1.0, 120.0))}
    assert got == {"1s_30fps": 3, "4s_30fps": 0, "1s_120fps": 0}, got


@check("v2 returns the same count on the same path walked slowly or quickly")
def _speed_invariant():
    got = [turns(synth(path(WAY, sec, 30.0), fps=30.0)) for sec in (0.5, 1.0, 2.0, 4.0, 8.0)]
    assert got == [3, 3, 3, 3, 3], got


@check("v2 returns the same count at any frame rate")
def _framerate_invariant():
    got = [turns(synth(path(WAY, 1.0, fps), fps=fps)) for fps in (15.0, 30.0, 60.0, 120.0, 240.0)]
    assert got == [3, 3, 3, 3, 3], got


@check("a genuinely still hand with jitter reads OK and zero, not insufficient")
def _still_is_zero_not_unknown():
    rng = np.random.default_rng(7)
    for sd in (0.002, 0.005, 0.01):
        y = 0.4 + rng.normal(0, sd, 90)
        rec = X.movement_events(synth(y.astype(np.float64)))["right"]
        assert rec["status"] == "OK", rec
        assert rec["turns"] == 0, (sd, rec["turns"])


@check("more real repetitions give proportionally more turns")
def _repetition_count():
    for reps in (1, 2, 3, 4):
        way = [0.0] + [0.2, 0.0] * reps
        got = turns(synth(path(way, 2.0, 30.0)))
        assert got == 2 * reps - 1, (reps, got)


@check("a long observation gap splits the series instead of joining across it")
def _gap_not_joined():
    y = np.concatenate([path([0.0, 0.2, 0.0], 1.0, 30.0), path([0.3, 0.1, 0.3], 1.0, 30.0)])
    valid = np.ones(len(y), dtype=np.float32)
    valid[30:45] = 0.0                                   # half a second unobserved
    rec = X.movement_events(synth(y, right_valid=valid))["right"]
    assert rec["status"] == "OK"
    assert rec["segments"] == 2, rec["segments"]
    assert rec["longest_gap_seconds"] >= 0.4, rec["longest_gap_seconds"]
    joined = X.movement_events(synth(y))["right"]        # same path, nothing missing
    assert rec["turns"] <= joined["turns"], (rec["turns"], joined["turns"])


@check("too little observation returns INSUFFICIENT with turns None, never a still zero")
def _insufficient_is_not_still():
    y = path(WAY, 2.0, 30.0)
    valid = np.zeros(len(y), dtype=np.float32)
    valid[:6] = 1.0
    rec = X.movement_events(synth(y, right_valid=valid))["right"]
    assert rec["status"] == "INSUFFICIENT_OBSERVATION", rec["status"]
    assert rec["turns"] is None, "an unobserved hand was reported as a still hand"
    assert rec["reason"], "no reason given for the insufficiency"


@check("a missing hand is reported missing, not rewritten as a resting hand")
def _missing_hand_honest():
    y = path(WAY, 2.0, 30.0)
    out = X.movement_events(synth(y, y_left=np.full(len(y), 0.4),
                                  left_valid=np.zeros(len(y), dtype=np.float32)))
    assert out["right"]["status"] == "OK" and out["right"]["turns"] == 3
    assert out["left"]["status"] == "INSUFFICIENT_OBSERVATION"
    assert out["left"]["turns"] is None
    assert out["left"]["observed_frames"] == 0


@check("each hand keeps its own count; neither is silently replaced by the other")
def _per_hand_preserved():
    y_r = path(WAY, 2.0, 30.0)
    y_l = np.full(len(y_r), 0.4)                         # left held still
    out = X.movement_events(synth(y_r, y_left=y_l))
    assert out["right"]["turns"] == 3 and out["left"]["turns"] == 0, out
    assert out["right"]["observed_fraction"] == 1.0 and out["left"]["observed_fraction"] == 1.0


@check("the count needs no speed estimate: it is unchanged by an irregular clock")
def _no_speed_estimate():
    y = path(WAY, 2.0, 30.0)
    base = synth(y)
    jittery = synth(y)
    rng = np.random.default_rng(3)
    jittery["x"][1:, -1] = (1.0 / 30.0) * rng.uniform(0.3, 3.0, len(y) - 1)
    a = X.movement_events(base)["right"]
    b = X.movement_events(jittery)["right"]
    assert a["turns"] == b["turns"] == 3, (a["turns"], b["turns"])
    assert a["observed_seconds"] != b["observed_seconds"], "the clock should still be reported"


@check("the amplitude threshold behaves monotonically and is not word-specific")
def _threshold_monotonic():
    y = path(WAY, 2.0, 30.0)                             # excursion 0.2
    got = [turns(synth(y), turn_amplitude=a) for a in (0.02, 0.05, 0.1, 0.15, 0.25)]
    assert got[:3] == [3, 3, 3], got
    assert all(got[i] >= got[i + 1] for i in range(len(got) - 1)), got
    assert got[-1] == 0, "a threshold above the excursion should find no turn"


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
