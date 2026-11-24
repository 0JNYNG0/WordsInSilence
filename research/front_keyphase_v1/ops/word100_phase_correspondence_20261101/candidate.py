"""Observed relation-transition constraints around an unchanged C13 objective.

No words, phase labels, source-index truth, inference or coordinate edits.
Forced LCS pairs are combinatorially forced among OBSERVED events only;
they are not verified action occurrences, especially across missing evidence.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit

_SOURCE = Path(__file__).resolve().parent.parent / "word100_relational_sequence_compare_20261031/relational.py"
_SPEC = importlib.util.spec_from_file_location("_wis_phase_relational_readonly", _SOURCE)
_REL = importlib.util.module_from_spec(_SPEC)
_BYTECODE = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    _SPEC.loader.exec_module(_REL)
finally:
    sys.dont_write_bytecode = _BYTECODE

BAND = .25  # Exact stored C_relational_path search convention.
NUMERIC_EPS = 1e-9  # Ambiguity diagnostic only; never picks a higher cost.
ALLOWED_KEYS = {"x", "mask", "frame_id", "timestamp_us", "names"}


def _input(a):
    if set(a) - ALLOWED_KEYS:
        raise ValueError("Only measured arrays and their axes/names are accepted")
    if not {"x", "mask", "frame_id", "timestamp_us"}.issubset(a):
        raise ValueError("Missing measured arrays or original time axes")
    x = np.asarray(a["x"], dtype=np.float64)
    mask = np.asarray(a["mask"])
    fid, pts = np.asarray(a["frame_id"]), np.asarray(a["timestamp_us"])
    if x.ndim != 2 or x.shape[1] != 69 or not len(x):
        raise ValueError("The original 69-feature layout is required")
    if mask.shape != x.shape or mask.dtype != np.bool_:
        raise ValueError("Explicit same-shape boolean masks are required")
    if fid.shape != (len(x),) or pts.shape != fid.shape or not np.issubdtype(fid.dtype, np.integer):
        raise ValueError("Frame/PTS dimensions or frame-ID dtype invalid")
    if not np.isfinite(pts).all() or not np.all(np.diff(pts) > 0) or not np.all(np.diff(fid) > 0):
        raise ValueError("Frame IDs and PTS must increase strictly")
    if "names" in a and (len(a["names"]) != 69 or len(set(a["names"])) != 69):
        raise ValueError("Invalid optional 69-feature names")
    # Nonfinite unsupported coordinates remain unknown; no value is filled.
    return dict(x=x, mask=mask & np.isfinite(x), frame_id=fid, timestamp_us=pts)


def _path(path, n, m):
    p = np.asarray(path)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("An integer full-scope baseline path is required")
    if not np.array_equal(p[0], [0, 0]) or not np.array_equal(p[-1], [n-1, m-1]):
        raise ValueError("Baseline endpoints do not preserve full scope")
    moves = np.diff(p, axis=0)
    if np.any(moves < 0) or np.any(moves > 1) or np.any(moves.sum(1) == 0):
        raise ValueError("Baseline must use the original three monotone moves")
    if set(p[:, 0]) != set(range(n)) or set(p[:, 1]) != set(range(m)):
        raise ValueError("Baseline loses source frames")
    return p.astype(np.int64, copy=True)


def _interval(a, lo, hi):
    return dict(index_range=[int(lo), int(hi)],
                frame_range=[int(a["frame_id"][lo]), int(a["frame_id"][hi])],
                timestamp_us_range=[int(a["timestamp_us"][lo]), int(a["timestamp_us"][hi])])


def events(a):
    """Strict signs of saved R-minus-L wrist relation, no threshold or smoothing.

    Both anatomical wrist XY observations and the saved relation component
    must exist. Exact-zero plateaus are bracketed, not called a semantic hold.
    An interval is between samples; it does not reconstruct crossing timing.
    """
    a = _input(a)
    x, mask, ids = a["x"], a["mask"], a["frame_id"]
    out, support = [], {}
    joint = mask[:, [0, 1, 28, 29]].all(1)
    frame_gaps = [dict(before_index=i-1, after_index=i,
                       before_frame_id=int(ids[i-1]), after_frame_id=int(ids[i]))
                  for i in range(1, len(ids)) if ids[i] != ids[i-1]+1]
    for axis, col in (("x", 56), ("y", 57)):
        valid = joint & mask[:, col]
        support[axis] = dict(possible=len(x), observed=int(valid.sum()),
                             unobserved=int((~valid).sum()),
                             missing_indices=np.flatnonzero(~valid).tolist(),
                             missing_frame_ids=ids[~valid].tolist())
        last = None
        run = -1
        previous_valid = False
        for i in range(len(x)):
            contiguous = i == 0 or ids[i] == ids[i-1]+1
            if not valid[i]:
                last = None
                previous_valid = False
                continue
            if not previous_valid or not contiguous:
                run += 1
                last = None
            previous_valid = True
            sign = int(np.sign(x[i, col]))
            if sign == 0:
                continue
            if last is not None:
                j, before = last
                if sign != before:
                    e = dict(axis=axis, from_sign=before, to_sign=sign,
                             type=[axis, before, sign], observation_run=run,
                             zero_plateau_samples=i-j-1,
                             overlapping_other_axis=False,
                             semantic_phase_label=None)
                    e.update(_interval(a, j, i))
                    out.append(e)
            last = (i, sign)
    # Different axes can cross within overlapping sample intervals. There is
    # no observed total order between those events: exclude them as anchors.
    for i, e in enumerate(out):
        for f in out[i+1:]:
            if e["axis"] != f["axis"] and max(e["index_range"][0], f["index_range"][0]) <= min(e["index_range"][1], f["index_range"][1]):
                e["overlapping_other_axis"] = True
                f["overlapping_other_axis"] = True
    out.sort(key=lambda e: (e["index_range"][0], e["index_range"][1], e["axis"]))
    for i, e in enumerate(out):
        e["event_id"] = i
        e["anchor_eligible"] = not e["overlapping_other_axis"]
    return dict(events=out, support=support, frame_gaps=frame_gaps,
                excluded_overlap_event_ids=[e["event_id"] for e in out if not e["anchor_eligible"]],
                possible_relation_samples=2*len(x),
                observed_relation_samples=sum(v["observed"] for v in support.values()),
                relation_definition="saved R_hand_wrist minus L_hand_wrist; image-plane coordinates",
                exact_zero_is_semantic_hold=False, gap_interpolation=False,
                jitter_filtered=False, continuous_crossing_time_reconstructed=False)


def match_events(q, r):
    """All possible rank-pairs of maximum-cardinality ordered event matches.

    A rank containing one possible pair is common to every optimal matching.
    Skipped events remain explicit; missing intervals may hide unknown events.
    No minimum number of required repetitions is assumed.
    """
    a = [e for e in q["events"] if e["anchor_eligible"]]
    b = [e for e in r["events"] if e["anchor_eligible"]]
    n, m = len(a), len(b)
    pre = np.zeros((n+1, m+1), dtype=np.int64)
    suf = np.zeros_like(pre)
    for i in range(n):
        for j in range(m):
            pre[i+1, j+1] = max(pre[i, j+1], pre[i+1, j],
                                pre[i, j]+(a[i]["type"] == b[j]["type"]))
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            suf[i, j] = max(suf[i+1, j], suf[i, j+1],
                            suf[i+1, j+1]+(a[i]["type"] == b[j]["type"]))
    length = int(pre[n, m])
    ranks = [[] for _ in range(length)]
    for i in range(n):
        for j in range(m):
            if a[i]["type"] == b[j]["type"] and pre[i, j]+1+suf[i+1, j+1] == length:
                rank = int(pre[i, j])
                ranks[rank].append(dict(query_event_id=a[i]["event_id"],
                                        reference_event_id=b[j]["event_id"]))
    forced = [dict(rank=k+1, **pairs[0]) for k, pairs in enumerate(ranks) if len(pairs) == 1]
    possible_q = {p["query_event_id"] for pairs in ranks for p in pairs}
    possible_r = {p["reference_event_id"] for pairs in ranks for p in pairs}
    ambiguous = any(len(pairs) > 1 for pairs in ranks)
    return dict(maximum_observed_event_matches=length, possible_pairs_by_rank=ranks,
                forced_pairs=forced, combinatorially_unique=not ambiguous,
                optimal_matching_count_capped=2 if ambiguous else 1,
                query_events_in_no_maximum_match=[e["event_id"] for e in a if e["event_id"] not in possible_q],
                reference_events_in_no_maximum_match=[e["event_id"] for e in b if e["event_id"] not in possible_r],
                unmatched_count_per_maximum_match=dict(query=n-length, reference=m-length),
                matching_is_not_physical_occurrence_truth=True)


def _anchors(q, r, matching):
    qs = {e["event_id"]: e for e in q["events"]}
    rs = {e["event_id"]: e for e in r["events"]}
    out = []
    for pair in matching["forced_pairs"]:
        a, b = qs[pair["query_event_id"]], rs[pair["reference_event_id"]]
        out.append(dict(**pair, type=a["type"], query_interval=a["index_range"],
                        reference_interval=b["index_range"],
                        query_frame_range=a["frame_range"], reference_frame_range=b["frame_range"],
                        query_timestamp_us_range=a["timestamp_us_range"],
                        reference_timestamp_us_range=b["timestamp_us_range"],
                        physical_occurrence_verified=False))
    return out


def _passes(path, gates):
    return all(np.any((path[:, 0] >= a) & (path[:, 0] <= b) &
                         (path[:, 1] >= c) & (path[:, 1] <= d)) for a, b, c, d in gates)


@njit(cache=False)
def _gated_dp(cost, gates):
    # C13's original three moves and unweighted cell sum, with a gate-count
    # state. A path must visit each full interval rectangle, not its midpoint.
    n, m = cost.shape
    count = len(gates)
    dist = np.full((n, m, count+1), np.inf)
    back_move = np.full((n, m, count+1), -1, np.int8)
    back_gate = np.full((n, m, count+1), -1, np.int32)
    ways = np.zeros((n, m, count+1), np.int8)
    near = np.zeros((n, m, count+1), np.bool_)
    for i in range(n):
        for j in range(m):
            if abs(i/max(n-1, 1)-j/max(m-1, 1)) > BAND:
                continue
            for oldk in range(count+1):
                k = oldk
                if k < count:
                    a, b, c, d = gates[k]
                    if i > b or j > d:
                        continue
                    if a <= i <= b and c <= j <= d:
                        k += 1
                if i == 0 and j == 0:
                    if oldk == 0:
                        dist[i, j, k] = cost[i, j]
                        ways[i, j, k] = 1
                    continue
                for move in range(3):  # Original diagonal/query/reference order.
                    pi = i - (move != 2)
                    pj = j - (move != 1)
                    if pi < 0 or pj < 0 or not np.isfinite(dist[pi, pj, oldk]):
                        continue
                    value = dist[pi, pj, oldk] + cost[i, j]
                    best = dist[i, j, k]
                    if value < best:
                        numeric = np.isfinite(best) and best-value <= NUMERIC_EPS
                        dist[i, j, k] = value
                        back_move[i, j, k] = move
                        back_gate[i, j, k] = oldk
                        ways[i, j, k] = ways[pi, pj, oldk]
                        near[i, j, k] = near[pi, pj, oldk] or numeric
                    elif value == best:
                        ways[i, j, k] = min(2, ways[i, j, k]+ways[pi, pj, oldk])
                        near[i, j, k] = near[i, j, k] or near[pi, pj, oldk]
                        # Preserve diagonal/query/reference precedence even
                        # when equal-cost paths enter from different gate states.
                        old_move = back_move[i, j, k]
                        if move < old_move or (move == old_move and oldk < back_gate[i, j, k]):
                            back_move[i, j, k] = move
                            back_gate[i, j, k] = oldk
                    elif value-best <= NUMERIC_EPS:
                        near[i, j, k] = True
    return dist, back_move, back_gate, ways, near


def solve(cost, gates):
    """Constrained numeric C13 path; endpoint and all-source-frame scope intact."""
    cost = np.asarray(cost, dtype=np.float64)
    gates = np.asarray(gates, dtype=np.int64).reshape(-1, 4)
    if cost.ndim != 2 or not cost.size or not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Finite nonnegative cost matrix required")
    n, m = cost.shape
    for k, (a, b, c, d) in enumerate(gates):
        if not (0 <= a <= b < n and 0 <= c <= d < m):
            raise ValueError("Gate lies outside source time axes")
        if k and (gates[k-1, 1] >= a or gates[k-1, 3] >= c):
            raise ValueError("Gate intervals must be disjoint and strictly ordered")
    dp, move, previous, ways, near = _gated_dp(cost, gates)
    i, j, k = n-1, m-1, len(gates)
    if not np.isfinite(dp[i, j, k]):
        return dict(path=None, status="NO_FEASIBLE_CONSTRAINED_C13_PATH")
    total = float(dp[i, j, k])
    number, numeric = int(ways[i, j, k]), bool(near[i, j, k])
    path = []
    while True:
        path.append([i, j])
        if i == 0 and j == 0:
            break
        step, oldk = int(move[i, j, k]), int(previous[i, j, k])
        if step < 0 or oldk < 0:
            raise RuntimeError("Broken constrained path")
        i -= step != 2
        j -= step != 1
        k = oldk
    path.reverse()
    checked = _path(path, n, m)
    if not _passes(checked, gates):
        raise RuntimeError("A gate was not visited")
    return dict(path=path, status="CONSTRAINED_C13_NUMERIC_PATH",
                objective_sum=total, objective_is_accuracy=False,
                exact_path_count_capped=number, numeric_near_tie=numeric,
                representative_is_physical_timing_truth=False)


def align(query, reference, baseline_path):
    start = time.perf_counter()
    q, r = _input(query), _input(reference)
    baseline = _path(baseline_path, len(q["x"]), len(r["x"]))
    qe, re = events(q), events(r)
    matching = match_events(qe, re)
    anchors = _anchors(qe, re, matching)
    gates = np.asarray([a["query_interval"]+a["reference_interval"] for a in anchors], dtype=np.int64).reshape(-1, 4)
    gaps = bool(qe["frame_gaps"] or re["frame_gaps"] or
                qe["observed_relation_samples"] < qe["possible_relation_samples"] or
                re["observed_relation_samples"] < re["possible_relation_samples"])
    output = dict(path=baseline.tolist(), events=dict(query=qe, reference=re),
                  event_matching=matching, anchors=anchors, fallback=True,
                  alignment_changed=False, occurrence_unresolved=True,
                  unknown_intervals_can_hide_occurrences=gaps,
                  physical_occurrence_verified=False, alignment_verified=False,
                  semantic_order_verified=False, whole_sign_verdict=None,
                  correction_advice=None, source_values_changed=False,
                  new_inference=0, new_training=0,
                  query_frame_ids=q["frame_id"].tolist(), reference_frame_ids=r["frame_id"].tolist(),
                  query_timestamp_us=q["timestamp_us"].tolist(), reference_timestamp_us=r["timestamp_us"].tolist(),
                  baseline_role="SUPPLIED_SAVED_C_RELATIONAL_13_GROUP_PATH",
                  cost_role="UNCHANGED_C13_REFERENCE_MASK_COST_AND_EQUAL_GROUP_WEIGHTS",
                  source_scope_preserved=True, all_feature_masks_preserved=True)
    if not anchors:
        output["status"] = "FALLBACK_NO_FORCED_OBSERVED_TRANSITION_ANCHOR"
    elif any(gates[k-1, 1] >= gates[k, 0] or gates[k-1, 3] >= gates[k, 2] for k in range(1, len(gates))):
        output["status"] = "FALLBACK_OVERLAPPING_OR_CONFLICTING_ANCHOR_INTERVALS"
    elif _passes(baseline, gates):
        output.update(status="BASELINE_ALREADY_PASSES_OBSERVED_ANCHORS", fallback=False)
    else:
        # Importing geometry does not call old compare/DTW or old classifiers.
        cost = _REL.alignment_cost(_REL.features(q), _REL.features(r))
        result = solve(cost, gates)
        output["candidate_solver"] = {k: v for k, v in result.items() if k != "path"}
        if result["path"] is None:
            output["status"] = "FALLBACK_NO_FEASIBLE_CONSTRAINED_C13_PATH"
        else:
            output.update(path=result["path"], status="CANDIDATE_CONSTRAINED_BY_OBSERVED_TRANSITIONS",
                          fallback=False, alignment_changed=result["path"] != baseline.tolist())
    # Output paths are candidates only. Other raw groups and both-axis feature
    # opportunities belong to the unchanged readout adapter, not this gate.
    _path(output["path"], len(q["x"]), len(r["x"]))
    output["baseline_exact_when_fallback"] = not output["fallback"] or output["path"] == baseline.tolist()
    output["seconds"] = time.perf_counter()-start
    return output
