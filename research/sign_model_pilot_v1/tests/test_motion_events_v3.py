"""Anchors v3 to the two v2 failures external review reproduced, then re-runs the v2 regressions.

Each of the first two checks REPRODUCES the wrong v2 output before requiring the right v3 one, so
the fix is tied to the defect it fixes rather than asserted in the abstract. v1 and v2 are imported
unchanged; nothing in axes.py is modified.

Channel layout used to build the synthetic clips, matching the review's own construction:
  right-hand body-relative y = channel 61   (body point 30 = 9 body + 21 left-hand, times 2, +1)
  body validity block starts at channel 186, so the right wrist's flag is 186 + 30
  delta_time = channel 279

Run: py -3.13 tests/test_motion_events_v3.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from pilot import axes as V12  # noqa: E402  (v1 direction_reversals, v2 movement_events)
from pilot import data as D  # noqa: E402
from pilot.motion_events_v3 import movement_events_v3  # noqa: E402

RIGHT_Y = 61
VALID_BASE = 186
RIGHT_ROOT = 30
DT = 279
WAY = [0.0, 0.2, 0.0, 0.2, 0.0]
CHECKS = []
MEASURED = {}


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def path(waypoints, n):
    return np.interp(np.linspace(0, len(waypoints) - 1, n),
                     np.arange(len(waypoints)), np.asarray(waypoints, dtype=np.float64))


def clip(y, dt, right_valid=None):
    t = len(y)
    x = np.zeros((t, D.TOTAL_CHANNELS), dtype=np.float32)
    x[:, RIGHT_Y] = y
    x[:, VALID_BASE:VALID_BASE + D.N_POINTS] = 1.0
    if right_valid is not None:
        x[:, VALID_BASE + RIGHT_ROOT] = np.asarray(right_valid, dtype=np.float32)
    x[1:, DT] = dt
    return {"x": x, "frames": t}


@check("counterexample 1: v2 bridged two holes and reported it had not; v3 keeps three runs")
def _gap_bridging():
    valid = np.ones(31)
    valid[[10, 11, 21, 22]] = 0.0
    c = clip(path(WAY, 31), 1.0 / 30.0, valid)

    old = V12.movement_events(c)["right"]
    assert old["segments"] == 1, f"v2 no longer reproduces the reported segments=1: {old}"
    assert old["turns"] == 3 and old["turns_span_gaps"] is False, old

    new = movement_events_v3(c)["right"]
    assert new["status"] == "OK", new
    assert new["runs_total"] == 3, f"expected three observed stretches, got {new['runs_total']}"
    assert new["gap_count"] == 2, new["gap_count"]
    assert new["completeness"] == "PARTIAL", new["completeness"]
    assert new["observed_frames"] == 27 and new["total_frames"] == 31
    assert new["observed_seconds"] < new["span_seconds"], (new["observed_seconds"],
                                                           new["span_seconds"])
    MEASURED["counterexample_1"] = {"v2": {k: old[k] for k in ("segments", "turns",
                                                               "turns_span_gaps")},
                                    "v3": {k: new[k] for k in ("status", "runs_total", "turns",
                                                               "completeness", "observed_seconds",
                                                               "span_seconds")}}


@check("counterexample 2: v2 called single-frame runs a still hand; v3 calls it uncountable")
def _single_frame_runs():
    valid = np.zeros(17)
    valid[::2] = 1.0
    c = clip(path(WAY, 17), 0.2, valid)

    old = V12.movement_events(c)["right"]
    assert old["status"] == "OK" and old["turns"] == 0, old
    assert old["segments"] == 9, old["segments"]
    assert abs(old["observed_seconds"] - 3.2) < 1e-6, old["observed_seconds"]

    new = movement_events_v3(c)["right"]
    assert new["status"] == "INSUFFICIENT_OBSERVATION", new
    assert new["turns"] is None, "an uncountable hand was reported as a still hand"
    assert new["longest_run_frames"] == 1, new["longest_run_frames"]
    assert new["reason"], "no reason recorded"
    MEASURED["counterexample_2"] = {"v2": {k: old[k] for k in ("status", "turns", "segments",
                                                               "observed_seconds")},
                                    "v3": {k: new[k] for k in ("status", "turns",
                                                               "longest_run_frames", "reason")}}


@check("a fully observed still hand with jitter is OK, COMPLETE and zero, not uncountable")
def _still_is_zero():
    rng = np.random.default_rng(7)
    for sd in (0.002, 0.005, 0.01):
        r = movement_events_v3(clip(0.4 + rng.normal(0, sd, 90), 1.0 / 30.0))["right"]
        assert r["status"] == "OK" and r["completeness"] == "COMPLETE", (sd, r["status"])
        assert r["turns"] == 0, (sd, r["turns"])


@check("the same path walked slowly or quickly gives the same count")
def _speed_invariant():
    got = [movement_events_v3(clip(path(WAY, n), 1.0 / 30.0))["right"]["turns"]
           for n in (15, 30, 60, 120, 240)]
    assert got == [3] * 5, got


@check("the same path at different frame rates gives the same count, above the measured floor")
def _framerate_invariant():
    got = {fps: movement_events_v3(clip(path(WAY, max(2, int(fps))), 1.0 / fps))["right"]["turns"]
           for fps in (4, 6, 8, 10, 15, 30, 60, 120, 240)}
    floor = min(f for f in sorted(got) if all(got[g] == 3 for g in sorted(got) if g >= f))
    MEASURED["sampling_floor"] = {"turns_by_frames_sampled": {str(k): v for k, v in got.items()},
                                  "lowest_rate_holding_3_turns": floor,
                                  "meaning": "the count is stable above this sampling density for "
                                             "this synthetic 2-cycle path; it is NOT a guarantee "
                                             "for every path or every occlusion pattern"}
    assert got[30] == got[120] == got[240] == 3, got
    assert floor <= 15, f"the count only stabilised at {floor} samples, which is too coarse"


@check("more real repetitions give proportionally more turns")
def _repetition_count():
    for reps in (1, 2, 3, 4):
        way = [0.0] + [0.2, 0.0] * reps
        got = movement_events_v3(clip(path(way, 60), 1.0 / 30.0))["right"]["turns"]
        assert got == 2 * reps - 1, (reps, got)


@check("an irregular clock splits a run when two observed frames are far apart in time")
def _time_gap_still_breaks():
    y = path(WAY, 40)
    c = clip(y, 1.0 / 30.0)
    c["x"][20, DT] = 0.9                       # one long inter-frame interval, nothing missing
    r = movement_events_v3(c)["right"]
    assert r["runs_total"] == 2, r["runs_total"]
    assert r["completeness"] == "PARTIAL", r["completeness"]
    assert r["observed_frames"] == 40, "no frame was missing, so all 40 stay observed"


@check("a hand never observed returns INSUFFICIENT, and the other hand is unaffected")
def _missing_hand():
    c = clip(path(WAY, 40), 1.0 / 30.0)
    c["x"][:, VALID_BASE + D.N_BODY] = 0.0     # left hand root flag off
    out = movement_events_v3(c)
    assert out["right"]["status"] == "OK" and out["right"]["turns"] == 3
    assert out["left"]["status"] == "INSUFFICIENT_OBSERVATION"
    assert out["left"]["turns"] is None and out["left"]["observed_frames"] == 0


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
    if MEASURED:
        import json
        print(json.dumps(MEASURED, ensure_ascii=False, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
