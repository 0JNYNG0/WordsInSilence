"""Movement events v3: runs are built from the observed frames themselves, not from a time rule.

Version chain, all three kept so every submitted result stays reproducible:
  v1  axes.direction_reversals   a per-frame deadband. External review showed it is a speed test:
                                 the same path returns 3 turns at 1s/30fps and 0 at 4s/30fps.
  v2  axes.movement_events       amplitude hysteresis, which fixed speed and sampling dependence,
                                 but deleted missing frames and then re-joined the neighbours
                                 whenever the TIME across the hole was under 0.25 s.
  v3  this module                runs break at every explicitly missing frame as well as at a long
                                 time gap, and the report says what was actually observed.

The two failures v2 was shown to have, both reproduced in tests/test_motion_events_v3.py before
anything here is trusted:

  1. 31 frames at 30 fps with the right hand invalid at frames 10, 11, 21 and 22 is three separate
     observed stretches. v2 reported segments=1, turns=3 and turns_span_gaps=false - it had bridged
     two holes and then declared that it had not.
  2. 17 frames at dt=0.2 s with the right hand valid only on even indices has no unbroken stretch
     longer than a single frame. v2 reported status=OK, turns=0, segments=9 and
     observed_seconds=3.2, which reads as a well-observed still hand. Nothing could have been
     counted, and 3.2 s was the first-to-last span, not time actually observed.

What v3 promises, and what it does not:
  * a turn is counted only inside one unbroken run of observed frames. Nothing is bridged, so a
    reported count never depends on an invented sample.
  * `completeness` says whether the clip was seen whole. When it is PARTIAL the count is neither an
    upper nor a lower bound: a gap can hide a turn, and can also split one movement into two runs.
  * "observed" durations are summed over runs. The first-to-last distance is reported separately as
    `span_seconds` and is never called observation.
  * a hand that was seen and held still returns OK with 0 turns. A hand that could not be counted
    returns INSUFFICIENT_OBSERVATION with turns None. These are never the same value.
  * NO claim of invariance at every capture rate or under every occlusion. The count is invariant to
    speed and to sampling rate only while each half-cycle of the movement keeps enough samples
    inside one run; tests/test_motion_events_v3.py measures where that stops holding on a synthetic
    path and the measured floor is recorded in the result file rather than asserted in prose.
"""
from __future__ import annotations

import numpy as np

from . import data as D

TURN_AMPLITUDE = 0.05          # retracement in shoulder widths that confirms a direction change
MAX_GAP_SECONDS = 0.25         # observed frames further apart than this also break a run
MIN_RUN_FRAMES = 3             # a reversal cannot be seen in fewer than three consecutive samples
MIN_OBSERVED_FRAMES = 8
MIN_OBSERVED_FRACTION = 0.5
# the nine common body points are laid out by data.COMMON_BODY_ORDER, then the two hands
HAND_ROOT = (D.N_BODY, D.N_BODY + D.N_HAND)
EPS = 1e-6


def _turns(y: np.ndarray, amp: float) -> int:
    """Direction changes confirmed by retracement amplitude, within one unbroken run."""
    if len(y) < 2:
        return 0
    turns, direction = 0, 0
    lo = hi = ext = float(y[0])
    for v in map(float, y[1:]):
        if direction == 0:
            lo, hi = min(lo, v), max(hi, v)
            if hi - v >= amp:
                direction, ext = -1, v
            elif v - lo >= amp:
                direction, ext = 1, v
        elif direction == 1:
            if v > ext:
                ext = v
            elif ext - v >= amp:
                turns, direction, ext = turns + 1, -1, v
        else:
            if v < ext:
                ext = v
            elif v - ext >= amp:
                turns, direction, ext = turns + 1, 1, v
    return turns


def observed_runs(valid: np.ndarray, clock: np.ndarray, max_gap: float):
    """Indices of each unbroken observed stretch.

    A run ends at a missing frame, and also when two observed frames are further apart in time than
    max_gap - an irregular clock can separate two frames that are adjacent by index.
    """
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero((np.diff(idx) != 1) | (np.diff(clock[idx]) > max_gap)) + 1
    return [r for r in np.split(idx, breaks) if r.size]


def movement_events_v3(clip: dict, turn_amplitude: float = None, max_gap_seconds: float = None,
                       axis_index: int = 1) -> dict:
    """Per-hand reversal count over observed runs only, with an explicit observation status."""
    amp = TURN_AMPLITUDE if turn_amplitude is None else turn_amplitude
    gap = MAX_GAP_SECONDS if max_gap_seconds is None else max_gap_seconds
    x = clip["x"]
    t = len(x)
    body = x[:, :D.CH_BODY_XY].reshape(t, D.N_POINTS, 2)
    bval = x[:, D.CH_BODY_XY + D.CH_HAND_XY:
             D.CH_BODY_XY + D.CH_HAND_XY + D.N_POINTS] > 0.5
    clock = np.cumsum(x[:, -1].astype(np.float64))

    out = {"version": "v3", "total_frames": t,
           "settings": {"turn_amplitude": amp, "max_gap_seconds": gap,
                        "min_run_frames": MIN_RUN_FRAMES,
                        "min_observed_frames": MIN_OBSERVED_FRAMES,
                        "min_observed_fraction": MIN_OBSERVED_FRACTION,
                        "axis": "vertical" if axis_index == 1 else "horizontal",
                        "runs_break_at": "any missing frame, or a time gap over max_gap_seconds",
                        "bridging": "none"}}
    for name, root in zip(("left", "right"), HAND_ROOT):
        valid = bval[:, root]
        runs = observed_runs(valid, clock, gap)
        usable = [r for r in runs if r.size >= MIN_RUN_FRAMES]
        observed = int(valid.sum())
        usable_frames = int(sum(r.size for r in usable))
        fraction = round(observed / t, 4) if t else 0.0
        rec = {
            "observed_frames": observed, "total_frames": t, "observed_fraction": fraction,
            "runs_total": len(runs), "runs_usable": len(usable),
            "run_lengths": [int(r.size) for r in runs][:32],
            "longest_run_frames": int(max((r.size for r in runs), default=0)),
            "usable_frames": usable_frames,
            "gap_count": max(len(runs) - 1, 0),
        }
        if runs:
            idx = np.flatnonzero(valid)
            rec["span_seconds"] = round(float(clock[idx[-1]] - clock[idx[0]]), 4)
            rec["observed_seconds"] = round(
                float(sum(clock[r[-1]] - clock[r[0]] for r in runs)), 4)
            rec["longest_gap_seconds"] = round(float(max(
                (clock[b[0]] - clock[a[-1]] for a, b in zip(runs, runs[1:])), default=0.0)), 4)
            rec["span_is_not_observation"] = True
        else:
            rec.update(span_seconds=0.0, observed_seconds=0.0, longest_gap_seconds=0.0,
                       span_is_not_observation=True)

        reasons = []
        if observed < MIN_OBSERVED_FRAMES:
            reasons.append(f"only {observed} observed frames, need {MIN_OBSERVED_FRAMES}")
        if fraction < MIN_OBSERVED_FRACTION:
            reasons.append(f"observed fraction {fraction:.2f}, need {MIN_OBSERVED_FRACTION:.2f}")
        if not usable:
            reasons.append(f"no unbroken run reaches {MIN_RUN_FRAMES} frames")
        elif usable_frames < MIN_OBSERVED_FRAMES:
            reasons.append(f"only {usable_frames} frames sit in countable runs, "
                           f"need {MIN_OBSERVED_FRAMES}")
        if reasons:
            rec.update(status="INSUFFICIENT_OBSERVATION", turns=None, completeness="UNCOUNTABLE",
                       reason="; ".join(reasons),
                       distinguishes_still_from_unseen="turns is None, never 0, when nothing "
                                                       "could be counted")
        else:
            complete = len(runs) == 1 and observed == t
            y = body[:, root, axis_index]
            rec.update(
                status="OK",
                turns=int(sum(_turns(y[r], amp) for r in usable)),
                completeness="COMPLETE" if complete else "PARTIAL",
                turns_scope=("the whole clip was observed in one run" if complete else
                             "counted inside observed runs only; a gap can hide a turn and can "
                             "also split one movement across two runs, so this is neither an "
                             "upper nor a lower bound on the clip's true count"),
                excursion_within_runs=round(float(max(
                    (y[r].max() - y[r].min() for r in usable), default=0.0)), 4),
                reason=None)
        out[name] = rec
    return out
