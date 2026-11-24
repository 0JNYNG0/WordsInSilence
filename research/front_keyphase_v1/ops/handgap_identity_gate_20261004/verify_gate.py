"""Independent verifier derived from FROZEN_PLAN.json, not candidate source.

Default CLI runs synthetic fixtures only. No model, video, detector, or service
imports. Real-input/artifact adapters may be added after the runner schema is
declared. Decisions enumerate ONLY the original one-frame restore candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SIDES = ("left", "right")
START = {"left": 25, "right": 46}
WRIST = {"left": 7, "right": 4}
PALM = (0, 5, 9, 13, 17)
MAX_SPAN_US = 66668


def _validated(xy, present, ts):
    xy, present, ts = map(np.asarray, (xy, present, ts))
    if xy.ndim != 3 or xy.shape[0] < 1 or xy.shape[1:] != (67, 2):
        raise ValueError("xy must have shape [T>=1,67,2]")
    if not np.issubdtype(xy.dtype, np.floating):
        raise ValueError("xy must be floating")
    if present.shape != xy.shape[:2]:
        raise ValueError("present shape mismatch")
    if present.dtype.kind not in "biuf" or not np.all((present == 0) | (present == 1)):
        raise ValueError("present must be boolean or exactly 0/1")
    pb = present.astype(bool)
    if ts.shape != (len(xy),) or ts.dtype.kind not in "iu":
        raise ValueError("ts must be integer microseconds, shape [T]")
    times = [int(t) for t in ts]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("timestamps must strictly increase")
    if not np.isfinite(xy[pb]).all():
        raise ValueError("nonfinite present coordinate")
    return xy, present, ts, pb, times


def _hand(side):
    return slice(START[side], START[side] + 21)


def _palm(xy, pb, row, side):
    indices = [START[side] + local for local in PALM]
    if not bool(pb[row, indices].all()):
        return None
    values = xy[row, indices].astype(np.float64)
    return values if np.isfinite(values).all() else None


def _distance(a, b):
    # Mean of corresponding-point Euclidean distances, not centroid distance.
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.sum(d * d, axis=-1)).mean())


def reference_decisions(xy, present, ts):
    """Return [{frame, side, accepted, reasons}] for original eligible gaps.

    Every evidence access is to the received arrays. No synthesized point or
    decision from another event can influence another decision.
    """
    xy, _, _, pb, times = _validated(xy, present, ts)
    decisions = []
    for row in range(1, len(xy) - 1):
        for side in SIDES:
            target = _hand(side)
            if pb[row, target].any():
                continue
            if not (pb[row - 1, target].all() and pb[row + 1, target].all()):
                continue
            if times[row + 1] - times[row - 1] > MAX_SPAN_US:
                continue
            opposite = "right" if side == "left" else "left"
            other = _hand(opposite)
            reasons = []
            palms = {end: _palm(xy, pb, end, side) for end in (row - 1, row + 1)}

            for end in (row - 1, row + 1):
                own_pose, other_pose = WRIST[side], WRIST[opposite]
                if not (pb[end, own_pose] and pb[end, other_pose]):
                    reasons.append(f"pose_wrist_evidence_missing:{end}")
                    continue
                target_wrist = xy[end, START[side]].astype(np.float64)
                own_distance = _distance(target_wrist, xy[end, own_pose])
                other_distance = _distance(target_wrist, xy[end, other_pose])
                if not own_distance < other_distance:
                    reasons.append(f"pose_identity_not_strict:{end}")

            opposite_palms = {}
            for end in (row - 1, row + 1):
                opposite_palms[end] = _palm(xy, pb, end, opposite)
                if pb[end, other].any() and opposite_palms[end] is None:
                    reasons.append(f"opposite_endpoint_palm_incomplete:{end}")

            target_link = _distance(palms[row - 1], palms[row + 1])
            # Forward: before target must prefer after target to after opposite.
            if opposite_palms[row + 1] is not None:
                competing = _distance(palms[row - 1], opposite_palms[row + 1])
                if not target_link < competing:
                    reasons.append("forward_target_not_strict_nearest")
            # Reverse: after target must prefer before target to before opposite.
            if opposite_palms[row - 1] is not None:
                competing = _distance(palms[row + 1], opposite_palms[row - 1])
                if not target_link < competing:
                    reasons.append("reverse_target_not_strict_nearest")

            if pb[row, other].any():
                middle_other = _palm(xy, pb, row, opposite)
                if middle_other is None:
                    reasons.append("middle_opposite_palm_incomplete")
                if any(opposite_palms[end] is None for end in (row - 1, row + 1)):
                    reasons.append("middle_witness_endpoint_missing")
                if middle_other is not None:
                    for end in (row - 1, row + 1):
                        endpoint_other = opposite_palms[end]
                        if endpoint_other is not None and not (
                            _distance(middle_other, endpoint_other)
                            < _distance(middle_other, palms[end])
                        ):
                            reasons.append(f"middle_witness_not_strict:{end}")
            decisions.append({"frame": row, "side": side,
                              "accepted": not reasons, "reasons": reasons})
    return decisions


def reference_restore(xy, present, ts):
    """Independent timestamp-weighted fill at accepted original-only events."""
    xy, present, ts, pb, times = _validated(xy, present, ts)
    out_xy, out_present = xy.copy(), present.copy()
    estimated = np.zeros(pb.shape, dtype=bool)
    decisions = reference_decisions(xy, present, ts)
    for event in decisions:
        if not event["accepted"]:
            continue
        row, hand = event["frame"], _hand(event["side"])
        w = (times[row] - times[row - 1]) / (times[row + 1] - times[row - 1])
        a, b = xy[row - 1, hand].astype(np.float64), xy[row + 1, hand].astype(np.float64)
        out_xy[row, hand] = ((1.0 - w) * a + w * b).astype(xy.dtype)
        out_present[row, hand] = 1
        estimated[row, hand] = True
    return {"xy": out_xy, "present": out_present, "ts": ts.copy(),
            "observed": pb.copy(), "estimated": estimated, "decisions": decisions}


def _byte_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_decisions(expected, actual):
    """Ignore ordering/reason spelling; reject duplicates or missing events."""
    def keyed(rows):
        out = {}
        for row in rows:
            key = (int(row["frame"]), str(row["side"]))
            if key in out:
                raise AssertionError(f"duplicate decision: {key}")
            if not isinstance(row["accepted"], (bool, np.bool_)):
                raise AssertionError(f"nonboolean accepted: {key}")
            out[key] = bool(row["accepted"])
        return out
    a, b = keyed(expected), keyed(actual)
    if a != b:
        keys = sorted(set(a) | set(b))
        mismatches = [(k, a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)]
        raise AssertionError(f"decision mismatch: {mismatches[:12]}")
    return len(a)


def verify_candidate_output(xy, present, ts, candidate):
    """Validate full output and original preservation; no candidate import."""
    expected = reference_restore(xy, present, ts)
    for field in ("xy", "present", "ts", "estimated"):
        if field not in candidate or not _byte_equal(expected[field], candidate[field]):
            raise AssertionError(f"candidate mismatch: {field}")
    if "observed" in candidate and not _byte_equal(expected["observed"], candidate["observed"]):
        raise AssertionError("candidate changed observed flags")
    unchanged = ~expected["estimated"]
    if not _byte_equal(np.asarray(candidate["xy"])[unchanged], np.asarray(xy)[unchanged]):
        raise AssertionError("received coordinates outside accepted gaps changed")
    if not _byte_equal(np.asarray(candidate["present"])[unchanged], np.asarray(present)[unchanged]):
        raise AssertionError("received presence outside accepted gaps changed")
    if "decisions" in candidate:
        compare_decisions(expected["decisions"], candidate["decisions"])
    return {"eligible": len(expected["decisions"]),
            "accepted": sum(e["accepted"] for e in expected["decisions"]),
            "estimated_joints": int(expected["estimated"].sum())}


def _fixture():
    xy = np.full((3, 67, 2), np.nan, dtype=np.float32)
    present = np.zeros((3, 67), dtype=bool)
    ts = np.array([0, 30000, 60000], dtype=np.int64)
    for row in range(3):
        for side, offset in (("left", 0.0), ("right", 20.0)):
            xy[row, WRIST[side]] = [offset, 0.0]
            present[row, WRIST[side]] = True
            for local in range(21):
                xy[row, START[side] + local] = [offset + local * 0.01, local * 0.02]
                present[row, START[side] + local] = True
    present[1, _hand("left")] = False
    xy[1, _hand("left")] = np.nan
    return xy, present, ts


def run_synthetic():
    """Small explicit contract fixtures. No real input or optional imports."""
    checks = []

    def decision(name, arrays, expected):
        xy, pr, ts = arrays
        snapshots = [a.copy() for a in arrays]
        actual = reference_decisions(xy, pr, ts)
        want = [] if expected is None else [(1, "left", expected)]
        got = [(r["frame"], r["side"], r["accepted"]) for r in actual]
        assert got == want, (name, got, want)
        output = reference_restore(xy, pr, ts)
        verify_candidate_output(xy, pr, ts, output)
        assert all(_byte_equal(a, b) for a, b in zip(arrays, snapshots)), name
        checks.append({"name": name, "pass": True})

    decision("complete_evidence_accept", _fixture(), True)
    x, p, t = _fixture(); p[:, _hand("right")] = False; x[:, _hand("right")] = np.nan
    decision("opposite_wholly_absent_no_competitor", (x, p, t), True)
    x, p, t = _fixture(); p[0, WRIST["right"]] = False
    decision("endpoint_pose_missing_reject", (x, p, t), False)
    x, p, t = _fixture(); x[0, WRIST["left"]] = [-1, 0]; x[0, WRIST["right"]] = [1, 0]
    decision("pose_distance_tie_reject", (x, p, t), False)
    x, p, t = _fixture(); p[2, START["right"] + 5] = False
    decision("opposite_endpoint_partial_palm_reject", (x, p, t), False)
    x, p, t = _fixture(); x[2, _hand("right")] = x[2, _hand("left")]
    decision("forward_mutual_tie_reject", (x, p, t), False)
    x, p, t = _fixture(); x[0, _hand("right")] = x[0, _hand("left")]
    decision("reverse_mutual_tie_reject", (x, p, t), False)
    x, p, t = _fixture(); p[1, START["right"] + 9] = False
    decision("middle_opposite_partial_palm_reject", (x, p, t), False)
    x, p, t = _fixture(); p[0, _hand("right")] = False; x[0, _hand("right")] = np.nan
    decision("middle_witness_needs_both_endpoints", (x, p, t), False)
    x, p, t = _fixture(); x[1, _hand("right")] = x[0, _hand("left")]
    decision("middle_opposite_follows_target_reject", (x, p, t), False)
    x, p, t = _fixture(); p[1, START["left"]] = True; x[1, START["left"]] = [0, 0]
    decision("partial_target_is_not_eligible", (x, p, t), None)
    x, p, t = _fixture(); p[0, START["left"] + 20] = False
    decision("partial_target_endpoint_not_eligible", (x, p, t), None)
    x, p, t = _fixture(); t[-1] = 66668
    decision("maximum_span_inclusive", (x, p, t), True)
    x, p, t = _fixture(); t[-1] = 66669
    decision("over_maximum_span_not_eligible", (x, p, t), None)

    x, p, t = _fixture(); t[1] = 10000; x[2, _hand("left")] += [3, 0]
    out = reference_restore(x, p, t)
    expected = ((5.0 / 6.0) * x[0, _hand("left")].astype(np.float64)
                + (1.0 / 6.0) * x[2, _hand("left")].astype(np.float64)).astype(x.dtype)
    assert _byte_equal(out["xy"][1, _hand("left")], expected)
    verify_candidate_output(x, p, t, out)
    checks.append({"name": "timestamp_weighted_not_midpoint", "pass": True})

    x, p, _ = _fixture()
    x = np.concatenate((x[:1], x[1:2], x[1:2], x[2:]), axis=0)
    p = np.concatenate((p[:1], p[1:2], p[1:2], p[2:]), axis=0)
    t = np.array([0, 20000, 40000, 60000], dtype=np.int64)
    assert reference_decisions(x, p, t) == []
    out = reference_restore(x, p, t)
    assert _byte_equal(out["xy"], x) and not out["estimated"].any()
    checks.append({"name": "consecutive_gaps_never_chain_estimates", "pass": True})

    x, p, t = _fixture(); out = reference_restore(x, p, t)
    bad = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in out.items()}
    bad["xy"][0, WRIST["left"], 0] = 999
    try:
        verify_candidate_output(x, p, t, bad)
    except AssertionError:
        checks.append({"name": "verifier_catches_observation_mutation", "pass": True})
    else:
        raise AssertionError("verifier failed to catch coordinate mutation")
    for name, mutate in (
        ("nonfinite_present_rejected", lambda x, p, t: x.__setitem__((0, 7, 0), np.nan)),
        ("nonincreasing_time_rejected", lambda x, p, t: t.__setitem__(1, 0)),
    ):
        x, p, t = _fixture(); mutate(x, p, t)
        try:
            reference_decisions(x, p, t)
        except ValueError:
            checks.append({"name": name, "pass": True})
        else:
            raise AssertionError(name)
    return {"schema": "wis_handgap_independent_synthetic_v1", "pass": True,
            "count": len(checks), "checks": checks,
            "actual_media_calls": 0, "model_forwards": 0, "real_inputs_loaded": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Run the default pure synthetic checks")
    parser.add_argument("--output", type=Path, help="Optional JSON receipt path")
    args = parser.parse_args()
    result = run_synthetic()
    result["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
