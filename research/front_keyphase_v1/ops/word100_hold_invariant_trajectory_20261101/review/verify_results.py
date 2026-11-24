"""Independent fixed-trajectory result audit; run only when root authorizes.

Never imports producer/common/legacy modules and never reruns a legacy DTW.
Uses raw saved coordinates, independent mask arithmetic, and graph reachability
at two cost thresholds to certify each stored candidate bottleneck.
"""
from pathlib import Path
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import ast
import gzip
import hashlib
import json
import math
import statistics
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
U, OPS = HERE.parent, HERE.parent.parent
OUTPUT = HERE / "RESULT_VERIFICATION.json"
COMPONENTS = ["L_wrist", "R_wrist", "hands_delta", "L_to_R_forearm", "R_to_L_forearm"]
checks, failures, pins = 0, [], {}
cache, geometry_cache = {}, {}
verified_rows, candidate_counts = [], Counter()
technical_checks, technical_failures = Counter(), []
gap_counts, denominator_counts = Counter(), Counter()
path_origins = Counter()


def read(p):
    p = Path(p); raw = p.read_bytes()
    return json.loads(gzip.decompress(raw) if p.suffix == ".gz" else raw.decode("utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(ok, label):
    global checks
    checks += 1
    if not ok:
        failures.append(label)


def require(ok, label):
    check(ok, label)
    if not ok:
        raise AssertionError(label)


def pin(p, expected=None):
    p = str(Path(p).resolve()); h = sha(p)
    check(expected is None or expected == h, "SHA " + p); pins[p] = h
    return h


def same(saved, expected, label, tol=1e-11):
    if isinstance(expected, dict):
        check(set(saved) == set(expected), label + " keys")
        for k in set(saved) & set(expected):
            same(saved[k], expected[k], label + "/" + str(k), tol)
    elif isinstance(expected, (list, tuple)):
        check(isinstance(saved, list) and len(saved) == len(expected), label + " length")
        if isinstance(saved, list):
            for a, b in zip(saved, expected):
                same(a, b, label, tol)
    elif isinstance(expected, (float, np.floating)):
        check(saved is not None and math.isclose(saved, float(expected), rel_tol=1e-12, abs_tol=tol), label)
    else:
        check(saved == expected, label)


def array(p):
    p = str(p)
    if p not in cache:
        with np.load(p, allow_pickle=False) as z:
            cache[p] = {k: z[k].copy() for k in ("x", "mask", "frame_id", "timestamp_us")}
    return cache[p]


def geometry(p, names):
    p = str(p)
    if p in geometry_cache:
        return geometry_cache[p]
    a = array(p); x = np.asarray(a["x"], np.float64)
    mask = a["mask"] & np.isfinite(x)
    def cols(columns):
        ids = [names.index(k) for k in columns]
        return np.where(mask[:, ids], x[:, ids], 0), mask[:, ids]
    result = {}
    for side, other in (("L", "R"), ("R", "L")):
        value, known = cols([side + "_hand_wrist_" + k for k in ("x", "y")])
        result[side + "_wrist"] = dict(values=value, mask=known)
        wrist_ids = [names.index(side + "_hand_wrist_" + k) for k in ("x", "y")]
        pw_ids = [names.index(other + "_pose_wrist_" + k) for k in ("x", "y")]
        el_ids = [names.index(other + "_elbow_" + k) for k in ("x", "y")]
        direction = x[:, pw_ids] - x[:, el_ids]
        offset = x[:, wrist_ids] - x[:, el_ids]
        norm2 = np.sum(direction * direction, axis=1)
        good = mask[:, wrist_ids].all(1) & mask[:, pw_ids].all(1) & mask[:, el_ids].all(1) & (norm2 > 1e-8)
        value = np.zeros((len(x), 2))
        value[good, 0] = np.sum(offset[good] * direction[good], axis=1) / norm2[good]
        value[good, 1] = (direction[good, 0] * offset[good, 1] - direction[good, 1] * offset[good, 0]) / norm2[good]
        result[side + "_to_" + other + "_forearm"] = dict(values=value, mask=np.repeat(good[:, None], 2, axis=1))
    value, known = cols(["hands_wrist_delta_x", "hands_wrist_delta_y"])
    result["hands_delta"] = dict(values=value, mask=known)
    for component in result.values():
        component.update(frame_id=a["frame_id"], timestamp_us=a["timestamp_us"])
    geometry_cache[p] = result
    return result


def slice_axis(a, lo, hi):
    return {k: v[lo:hi + 1] for k, v in a.items()}


def fixed_crop(full_path, parent):
    path = np.asarray(full_path, dtype=int)
    lo, hi = parent["core"]
    indices = np.flatnonzero((path[:, 1] >= lo) & (path[:, 1] <= hi))
    require(len(indices) > 0 and np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)),
            "Existing parent slice contiguous")
    segment = path[indices]
    qlo, qhi = int(segment[:, 0].min()), int(segment[:, 0].max())
    require(set(segment[:, 0]) == set(range(qlo, qhi + 1)) and set(segment[:, 1]) == set(range(lo, hi + 1)),
            "Parent crop preserves every original axis point")
    return segment, segment - [qlo, lo], [qlo, qhi], [lo, hi]


def valid_path(path, n, m, label):
    p = np.asarray(path, dtype=int)
    require(p.ndim == 2 and p.shape[1] == 2 and len(p) > 0, label + " shape")
    check(p[0].tolist() == [0, 0] and p[-1].tolist() == [n - 1, m - 1], label + " endpoints")
    check(set(p[:, 0]) == set(range(n)) and set(p[:, 1]) == set(range(m)), label + " both axes coverage")
    check(all(tuple(v) in {(1, 0), (0, 1), (1, 1)} for v in np.diff(p, axis=0)), label + " monotone unit steps")
    return p


def reachable(costs, threshold):
    """Graph search with successor expansion, not the producer min-max recurrence."""
    allowed = costs <= threshold
    n, m = allowed.shape
    if not allowed[0, 0] or not allowed[-1, -1]:
        return False
    todo, seen = deque([(0, 0)]), {(0, 0)}
    while todo:
        i, j = todo.popleft()
        if (i, j) == (n - 1, m - 1):
            return True
        for a, b in ((i + 1, j), (i, j + 1), (i + 1, j + 1)):
            if a < n and b < m and allowed[a, b] and (a, b) not in seen:
                seen.add((a, b)); todo.append((a, b))
    return False


def capped_paths_at(costs, threshold):
    """Count admitted graph paths by forward antidiagonal propagation, cap2."""
    n, m = costs.shape
    counts = np.zeros((n, m), dtype=np.int64)
    if costs[0, 0] <= threshold:
        counts[0, 0] = 1
    for diagonal in range(n + m - 1):
        for i in range(max(0, diagonal - m + 1), min(n - 1, diagonal) + 1):
            j = diagonal - i
            for a, b in ((i + 1, j), (i, j + 1), (i + 1, j + 1)):
                if a < n and b < m and costs[a, b] <= threshold:
                    counts[a, b] = min(2, counts[a, b] + counts[i, j])
    return int(counts[-1, -1])


def stats(indices, costs, shared, dims):
    indices = np.asarray(indices, dtype=int)
    counts = shared[indices]
    values = costs[indices][counts > 0]
    complete = costs[indices][counts == dims]
    return dict(link_count=len(indices), assessable_link_count=len(values), complete_link_count=len(complete),
                partial_link_count=int(((counts > 0) & (counts < dims)).sum()),
                unsupported_link_count=int((counts == 0).sum()),
                mean=float(np.mean(values)) if len(values) else None, max=float(np.max(values)) if len(values) else None,
                complete_link_mean=float(np.mean(complete)) if len(complete) else None,
                complete_link_max=float(np.max(complete)) if len(complete) else None,
                partial_coordinate_values_are_conditional_lower_bounds=True)


def runs(flags):
    edges = np.diff(np.r_[0, np.asarray(flags, int), 0])
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def range_info(a, start, end):
    return dict(start=start, end=end, frame_count=end - start + 1,
                frame_range=[int(a["frame_id"][start]), int(a["frame_id"][end])],
                pts_range=[int(a["timestamp_us"][start]), int(a["timestamp_us"][end])])


def verify_scope(saved, a, axis_indices, costs, shared, label):
    n, dims = a["mask"].shape
    known = a["mask"]; full = known.all(1)
    expected = dict(frame_id=a["frame_id"].tolist(), timestamp_us=a["timestamp_us"].tolist(),
        observed_mask=known.tolist(), possible_frames=n, possible_cells=n * dims, observed_cells=int(known.sum()),
        missing_cells=int((~known).sum()), observed_dimension_count=known.sum(1).tolist(),
        fully_observed_frames=int(full.sum()), incomplete_frames=int((~full).sum()),
        masked_nonfinite_cells=int((~known & ~np.isfinite(a["values"])).sum()),
        intervals_are_inclusive_original_axis_indices=True, observed_blocks_are_not_joined=True, all_opportunities_retained=True)
    for k, v in expected.items():
        same(saved[k], v, label + "/" + k)
    for name, flag, observed in (("observed_blocks", full, True), ("missing_intervals", ~full, False)):
        expected_blocks = []
        for start, end in runs(flag):
            ids = np.flatnonzero((axis_indices >= start) & (axis_indices <= end))
            expected_blocks.append(dict(**range_info(a, start, end), possible_cells=(end - start + 1) * dims,
                observed_cells=int(known[start:end + 1].sum()), has_at_least_two_frames=end > start,
                has_two_distinct_observed_positions=bool(np.any(a["values"][start:end + 1] != a["values"][start])) if observed else None,
                baseline=stats(ids, costs, shared, dims)))
        same(saved[name], expected_blocks, label + "/" + name)
        gap_counts[name] += len(expected_blocks)
    expected_missing = [dict(dimension=d, intervals=[range_info(a, lo, hi) for lo, hi in runs(~known[:, d])]) for d in range(dims)]
    same(saved["missing_by_dimension"], expected_missing, label + "/each missing coordinate")
    denominator_counts["cells"] += n * dims
    denominator_counts["observed_cells"] += int(known.sum())
    denominator_counts["missing_cells"] += int((~known).sum())
    denominator_counts["frames"] += n


def verify_baseline(out, q, r, path, label):
    baseline = out["baseline"]; n, dims = q["mask"].shape; m = len(r["mask"])
    p = valid_path(path, n, m, label + " saved baseline")
    qi, ri = p.T
    shared_mask = q["mask"][qi] & r["mask"][ri]
    shared = shared_mask.sum(1)
    delta = np.abs(q["values"][qi] - r["values"][ri])
    costs = np.where(shared_mask, delta, -np.inf).max(1)
    same(baseline["path"], path, label + " path copied")
    same(baseline["point_costs"], [float(c) if s else None for c, s in zip(costs, shared)], label + " path coordinate residual")
    same(baseline["shared_dimension_counts"], shared.tolist(), label + " shared coordinate counts")
    expected = stats(np.arange(len(p)), costs, shared, dims)
    for k, v in expected.items():
        same(baseline[k], v, label + " baseline/" + k)
    same(baseline["possible_link_dimension_opportunities"], len(p) * dims, label + " baseline link denominator")
    same(baseline["shared_link_dimension_opportunities"], int(shared.sum()), label + " baseline link support")
    for axis, source, indices, other_indices, opposite in (
        ("query", q, qi, ri, r), ("reference", r, ri, qi, q)):
        size = len(source["mask"])
        matched = np.zeros((size, dims), dtype=int)
        missing = np.zeros((size, dims), dtype=int)
        np.add.at(matched, indices, shared_mask)
        np.add.at(missing, indices, source["mask"][indices] & ~opposite["mask"][other_indices])
        expected_axis = dict(matched_support=(matched > 0).tolist(), matched_link_count=matched.tolist(),
            missing_counterpart_link_count=missing.tolist(), possible_frames=size, possible_cells=size * dims,
            own_observed_cells=int(source["mask"].sum()), matched_cells=int((matched > 0).sum()), unaligned_indices=[])
        frame_stats = [stats(np.flatnonzero(indices == i), costs, shared, dims) for i in range(size)]
        for field in ("mean", "max", "complete_link_mean", "complete_link_max", "assessable_link_count",
                      "complete_link_count", "partial_link_count", "unsupported_link_count"):
            expected_axis[field] = [x[field] for x in frame_stats]
        expected_axis["aligned_link_count"] = [x["link_count"] for x in frame_stats]
        same(baseline[axis], expected_axis, label + " baseline/" + axis)
        verify_scope(out[axis], source, indices, costs, shared, label + "/" + axis)
    check(baseline["full_scope"] is True and baseline["status"] == "SUPPLIED_FULL_SCOPE" and
          baseline["path_unchanged"] is True and baseline["is_previous_model_score"] is False and
          baseline["whole_vector_distance_known_on_every_link"] == bool(np.all(shared == dims)),
          label + " complete baseline semantics")
    return expected


def verify_candidate(candidate, q, r, label):
    complete = bool(len(q["mask"]) and len(r["mask"]) and q["mask"].all() and r["mask"].all())
    check(candidate["complete"] == complete, label + " full masks govern comparability")
    check(candidate["educational_error"] is None and candidate["alignment_verified"] is False and
          candidate["stage_order_verified"] is False and candidate["original_global_path_replaced"] is False,
          label + " no educational/stage claim")
    if not complete:
        check(candidate["distance"] is None and candidate["path"] == [] and candidate["path_point_costs"] == [] and
              candidate["bottleneck_witnesses"] == [] and candidate["optimal_path_ambiguous"] is None and
              candidate["optimal_path_count_capped"] == 0 and candidate["status"] == "INCOMPLETE_OBSERVATION",
              label + " any gap means null, never agreement")
        candidate_counts["incomplete"] += 1
        return False
    costs = np.max(np.abs(q["values"][:, None] - r["values"][None]), axis=2)
    path = valid_path(candidate["path"], len(q["mask"]), len(r["mask"]), label + " candidate")
    values = costs[path[:, 0], path[:, 1]]
    actual = float(values.max())
    same(candidate["path_point_costs"], values.tolist(), label + " candidate witness costs")
    same(candidate["distance"], actual, label + " attained bottleneck")
    previous = np.unique(costs[costs < actual])
    check(reachable(costs, actual), label + " optimum threshold reachable")
    check(not len(previous) or not reachable(costs, float(previous[-1])), label + " next smaller cost unreachable")
    count = capped_paths_at(costs, actual)
    check(candidate["optimal_path_count_capped"] == count and candidate["optimal_path_ambiguous"] == (count > 1),
          label + " all threshold-admissible optimal path ambiguity")
    expected_witnesses = []
    for (i, j), cost in zip(path.tolist(), values):
        if cost == actual:
            delta = np.abs(q["values"][i] - r["values"][j])
            expected_witnesses.append(dict(query_index=i, reference_index=j,
                query_frame_id=int(q["frame_id"][i]), reference_frame_id=int(r["frame_id"][j]),
                query_timestamp_us=int(q["timestamp_us"][i]), reference_timestamp_us=int(r["timestamp_us"][j]),
                point_cost=actual, dimensions=np.flatnonzero(delta == actual).tolist()))
    same(candidate["bottleneck_witnesses"], expected_witnesses, label + " full native bottleneck witnesses")
    check(candidate["all_input_frames_visited"] and candidate["endpoint_constrained"] and
          candidate["status"] == "COMPLETE_OBSERVED_GEOMETRY_ONLY", label + " candidate full scope")
    candidate_counts["complete"] += 1
    candidate_counts["ambiguous"] += count > 1
    return True


def expected_edit(control, q, reference, qr, rr):
    metadata = control["metadata"]; n, bn = len(q["x"]), len(reference["x"])
    mapping = np.asarray(metadata.get("original_indices", list(range(n))), dtype=int)
    require(len(mapping) == n and mapping.min() >= 0 and mapping.max() < bn, "Control source index map")
    original, mask = reference["x"][mapping], reference["mask"][mapping]
    changed_value = np.any((q["x"] != original) & q["mask"] & mask, axis=1)
    changed_mask = np.any(q["mask"] != mask, axis=1)
    changed = set(metadata.get("changed_query_indices", [])) | set(metadata.get("changed_frames", []))
    changed |= set(np.flatnonzero(changed_value | changed_mask).tolist())
    if control["control"] in ("swap_adjacent_core_blocks", "reverse_first_parent"):
        changed |= set(np.flatnonzero(mapping != np.arange(n)).tolist())
    removed = set(metadata.get("omitted_original_frames", [])) | set(metadata.get("removed_original_frames", []))
    removed |= set(range(bn)) - set(mapping.tolist())
    inside = {i for i in changed if qr[0] <= i <= qr[1]}
    relevant = {i for i in changed if rr[0] <= mapping[i] <= rr[1]}
    return dict(original_indices=mapping[qr[0]:qr[1] + 1].tolist(), changed_query_indices=sorted(changed),
        changed_query_inside_scope=sorted(inside), changed_query_outside_scope=sorted(changed - inside),
        relevant_changed_query_indices=sorted(relevant), relevant_changed_outside_scope=sorted(relevant - inside),
        intended_changed_reference_frames=sorted({int(mapping[i]) for i in relevant}),
        removed_original_frames_in_reference_scope=sorted(i for i in removed if rr[0] <= i <= rr[1]),
        source_point_values_changed=int(changed_value.sum()), source_masks_changed=int(changed_mask.sum()),
        original_index_is_evaluation_only=True, actual_linguistic_error=None)


def technical(key, ok, item):
    technical_checks[key] += 1
    if not ok:
        technical_failures.append(dict(check=key, id=item))


def describe(values):
    return dict(n=len(values), minimum=min(values) if values else None, median=float(statistics.median(values)) if values else None,
                mean=float(statistics.fmean(values)) if values else None, maximum=max(values) if values else None)


def verify_fresh_control_inputs(tasks):
    """Check all64 new serialized transforms against their frozen source arrays."""
    names = ("hold_insert5", "reverse_first_parent", "small_common_coordinate_drift", "right_middle_gap5")
    right_missing = [28, 29, 34, 35, 36, 37] + list(range(39, 59)) + [60, 65, 66, 67, 68]
    counts = Counter()
    for c in tasks["controls"]:
        kind = c["control"]
        if kind not in names:
            continue
        counts[kind] += 1
        base = array(tasks["rows"][c["alias"]]["input"]); actual = array(c["array"])
        n = len(base["x"]); lo, hi = c["parents"][0]["core"]; midpoint = (lo + hi) // 2
        ids = list(range(n))
        if kind == "hold_insert5":
            ids = ids[:midpoint + 1] + [midpoint] * 5 + ids[midpoint + 1:]
        elif kind == "reverse_first_parent":
            ids = ids[:lo] + ids[lo:hi + 1][::-1] + ids[hi + 1:]
        same(c["metadata"]["original_indices"], ids, "Frozen transform source map " + c["id"])
        expected = {k: v[ids].copy() for k, v in base.items()}
        if kind in ("hold_insert5", "reverse_first_parent"):
            expected["frame_id"] = np.arange(len(ids), dtype=np.int64)
            expected["timestamp_us"] = np.arange(len(ids), dtype=np.int64) * 33333
        elif kind == "small_common_coordinate_drift":
            feature_names = tasks["rows"][c["alias"]]["feature_names"]
            jitter = .002 * np.sin(np.arange(n) * 2 * np.pi / 8)
            for side in ("L", "R"):
                for part in ("hand_wrist", "pose_wrist", "elbow", "index_tip", "palm_center"):
                    for axis in ("x", "y"):
                        col = feature_names.index(side + "_" + part + "_" + axis)
                        expected["x"][:, col] += np.where(expected["mask"][:, col], jitter, 0)
        elif kind == "right_middle_gap5":
            selected = np.arange(max(0, midpoint - 2), min(n, midpoint + 3))
            expected["mask"][np.ix_(selected, right_missing)] = False
        for field in expected:
            check(np.array_equal(expected[field], actual[field], equal_nan=True),
                  "Exact new control transform " + c["id"] + "/" + field)
    check(counts == {name: 16 for name in names}, "All4 fixed new controls for every16 public clip")


def verify_record(rec, expected, tasks, frozen_hash):
    pin(rec["path"], rec["sha256"]); d = read(rec["path"])
    kind, task, parent, component = expected
    qa = task["query"] if kind == "natural" else task["alias"]
    ra = task["reference"] if kind == "natural" else qa
    qsource = tasks["rows"][qa]["input"] if kind == "natural" else task["array"]
    rsource = tasks["rows"][ra]["input"]
    source_baseline = task["baseline"] if task["baseline"] else str(U / "new_control_baselines" / (task["id"] + ".json"))
    full = read(source_baseline)["path"]
    segment, local, qr, rr = fixed_crop(full, parent)
    check(all(d[k] == v for k, v in dict(kind=kind, query=qa, reference=ra, role=task["role"], word_id=task["word_id"],
          parent=parent, component=component, source_query=qsource, source_reference=rsource,
          baseline_source=source_baseline, frozen_sha256=frozen_hash).items()), d["id"] + " source identity")
    check(d["query_range"] == qr and d["reference_range"] == rr and d["baseline_full_path"] == segment.tolist(),
          d["id"] + " exact fixed crop")
    q = slice_axis(geometry(qsource, tasks["rows"][qa]["feature_names"])[component], *qr)
    r = slice_axis(geometry(rsource, tasks["rows"][ra]["feature_names"])[component], *rr)
    for axis, values in (("query", q), ("reference", r)):
        check(np.array_equal(d[axis + "_plot"]["mask"], values["mask"]), d["id"] + " exact fixed geometry mask")
        check(np.allclose(d[axis + "_plot"]["values"], values["values"], atol=1e-13, rtol=1e-13), d["id"] + " raw-column geometry")
    comparison, candidate, baseline = d["comparison"], d["comparison"]["candidate"], d["comparison"]["baseline"]
    check(comparison["semantic_verdict"] is None and comparison["whole_sign_verdict"] is None and
          comparison["correction_advice"] is None and comparison["alignment_verified"] is False and
          comparison["missing_values_imputed"] is False and comparison["input_values_and_masks_unchanged"] is True,
          d["id"] + " preserved unknown/nonsemantic output")
    old = verify_baseline(comparison, q, r, local.tolist(), d["id"])
    complete = verify_candidate(candidate, q, r, d["id"])
    path_origins[d["baseline_origin"]] += 1
    technical("full_scope", baseline["full_scope"], d["id"])
    technical("complete_exact_masks", candidate["complete"] == complete, d["id"])
    technical("missing_null_not_agreement", complete or (candidate["distance"] is None and not candidate["path"]), d["id"])
    eps = tasks["numerical_tolerance"]
    if complete:
        technical("minmax_le_saved_path_max", candidate["distance"] <= old["complete_link_max"] + eps, d["id"])
    support = {axis: dict(possible_cells=a["mask"].size, observed_cells=int(a["mask"].sum()),
                         possible_frames=len(a["mask"]), fully_observed_frames=int(a["mask"].all(1).sum()))
               for axis, a in (("query", q), ("reference", r))}
    row = dict(id=d["id"], kind=kind, query=qa, reference=ra, word_id=task["word_id"], role=task["role"],
        parent_id=parent["segment_id"], component=component, control=d.get("control"),
        candidate_distance=candidate["distance"], baseline_mean=old["mean"], baseline_max=old["max"],
        baseline_complete_max=old["complete_link_max"], complete=complete,
        optimal_path_ambiguous=candidate["optimal_path_ambiguous"], support=support, seconds=d["elapsed_seconds"],
        result=rec["path"], semantic_verdict=None, normal_speed_or_hold_error=None)
    if kind == "natural":
        check(d["baseline_origin"] == "saved_global" and d["seed_aliases"] == [42, 43, 44] and
              d["source_readouts"] == task["source_readouts"] and d["original_global_path_replaced"] is False,
              d["id"] + " original36 paths and108 seed aliases retained")
    else:
        check(d["baseline_origin"] == task["baseline_origin"] and d["seed_aliases"] == [] and
              d["technical_control_not_real_ksl_error"] is True, d["id"] + " technical control provenance")
        edit = expected_edit(task, array(qsource), array(rsource), qr, rr)
        same(d["edit"], edit, d["id"] + " full edit ledger")
        mapping = edit["original_indices"]
        collapsed = [v for i, v in enumerate(mapping) if i == 0 or v != mapping[i - 1]]
        same_scope = collapsed == list(range(rr[0], rr[1] + 1))
        row.update(exact_source_scope_with_only_duplicates=same_scope,
            edited_query_outside_parent=len(edit["relevant_changed_outside_scope"]),
            intended_edit_in_parent=bool(edit["intended_changed_reference_frames"] or edit["removed_original_frames_in_reference_scope"]),
            removed_original_frames=len(edit["removed_original_frames_in_reference_scope"]),
            change_scope_preserved=not edit["relevant_changed_outside_scope"])
        if complete and same_scope:
            bound = float(np.abs(q["values"] - r["values"][np.asarray(mapping) - rr[0]]).max())
            row["actual_same_index_coordinate_perturbation_bound"] = bound
            if task["control"] in ("identity", "speed_075", "hold_insert5"):
                technical("pure_timing_exact_values", bound == 0, d["id"])
                technical("pure_timing_distance_zero", candidate["distance"] <= eps, d["id"])
            if task["control"] == "small_common_coordinate_drift":
                technical("jitter_extracted_geometry_bound", candidate["distance"] <= bound + eps, d["id"])
        if complete:
            row["positive_geometry_distance"] = candidate["distance"] > eps
    verified_rows.append(row)


def verify_aggregates(evaluation, manifest, tasks):
    rows = verified_rows; groups = defaultdict(list)
    for row in rows:
        groups[row["kind"], row["role"], row["component"], row["control"]].append(row)
    summaries = []
    for (kind, role, component, control), rr in sorted(groups.items(), key=lambda x: str(x[0])):
        cc = [r for r in rr if r["complete"]]
        summaries.append(dict(kind=kind, role=role, component=component, control=control,
            actual_records=len(rr), complete=len(cc), unknown=len(rr) - len(cc),
            old_mean_on_complete=describe([x["baseline_mean"] for x in cc]),
            old_max_on_complete=describe([x["baseline_max"] for x in cc]),
            new_bottleneck_on_complete=describe([x["candidate_distance"] for x in cc]),
            positive_geometry=sum(x.get("positive_geometry_distance", False) for x in cc),
            exact_source_scope=sum(x.get("exact_source_scope_with_only_duplicates", False) for x in rr),
            exact_source_scope_complete=sum(x.get("exact_source_scope_with_only_duplicates", False) for x in cc),
            changed_parent_scope_incomplete=sum(x.get("edited_query_outside_parent", 0) > 0 for x in rr),
            timing_ms=describe([x["seconds"] * 1000 for x in rr])))
    same(evaluation["summary"], summaries, "All grouped summaries")
    eps = tasks["numerical_tolerance"]; controls = []
    for name in sorted({r["control"] for r in rows if r["kind"] == "control"}):
        rr = [x for x in rows if x["control"] == name]; cc = [x for x in rr if x["complete"]]
        missing = [x for x in manifest["missing_component_opportunities"] if x["control"] == name]
        eligible = [x for x in cc if x["exact_source_scope_with_only_duplicates"]]
        controls.append(dict(control=name, component_denominator=len(rr) + len(missing), computed=len(rr), missing_control=len(missing),
            complete=len(cc), unknown=len(rr) - len(cc), positive_distance=sum(x["candidate_distance"] > eps for x in cc),
            zero_distance=sum(x["candidate_distance"] <= eps for x in cc), exact_source_scope_complete=len(eligible),
            exact_source_scope_zero=sum(x["candidate_distance"] <= eps for x in eligible),
            crop_changed_scope=sum(not x["exact_source_scope_with_only_duplicates"] for x in rr),
            relevant_edit_outside_crop=sum(x["edited_query_outside_parent"] > 0 for x in rr),
            new_bottleneck=describe([x["candidate_distance"] for x in cc]),
            old_mean=describe([x["baseline_mean"] for x in cc]), old_max=describe([x["baseline_max"] for x in cc])))
    same(evaluation["controls"], controls, "All control denominator and result summaries")
    natural = [r for r in rows if r["kind"] == "natural"]; complete = [r for r in natural if r["complete"]]
    expected = dict(pairs=36, public_clips=16, public_words=4, parent_instances=117, components=585,
        complete=len(complete), unknown=585 - len(complete),
        per_role={role: dict(opportunities=sum(x["role"] == role for x in natural),
                            complete=sum(x["role"] == role for x in complete)) for role in ("fit", "development")},
        mean_summary=describe([x["baseline_mean"] for x in complete]), old_max_summary=describe([x["baseline_max"] for x in complete]),
        new_summary=describe([x["candidate_distance"] for x in complete]))
    same(evaluation["natural"], expected, "All natural opportunities and geometry summaries")
    same(evaluation["checks"], dict(technical_checks), "Original technical gate denominators")
    same(evaluation["technical_failures"], technical_failures, "Original technical gate failures")
    check(evaluation["technical_failure_count"] == len(technical_failures), "Technical failure count")
    same(evaluation["rows"], rows, "Every2825 evaluation row")
    check(sum(x["component_denominator"] for x in controls) == 2250 and
          sum(x["computed"] for x in controls) == 2240 and sum(x["missing_control"] for x in controls) == 10,
          "Controls2250=2240computed+10notavailable, no denominator shrink")
    decision = evaluation["decision"]
    check(decision["technical_checks_pass"] == (not technical_failures) and decision["original_global_paths_changed"] is False and
          decision["production_adopted"] is False and decision["educational_accuracy"] is None and
          decision["recognition_accuracy"] is None and decision["old_stage_score_recomputed"] is False and
          decision["all_words_generalization_proven"] is False and decision["no_new_training"] is True,
          "Technical properties not performance/adoption/education claims")
    check(any("mathematical optimization, not correspondence accuracy" in x for x in evaluation["limits"]) and
          any("not before/after accuracy" in x for x in evaluation["limits"]), "Mean/max/optimized geometry interpretation")


def main():
    for p in ("FROZEN.json", "RESULT_MANIFEST.json", "EVALUATION.json"):
        require((U / p).exists(), "Completed output required before audit: " + p)
    frozen = read(U / "FROZEN.json"); frozen_hash = pin(U / "FROZEN.json")
    for section in ("pins", "protected"):
        for p, h in frozen[section].items():
            pin(p, h)
    if (U / "FIRST_PREP_FAILURE.json").exists():
        pin(U / "FIRST_PREP_FAILURE.json")
    tasks, manifest, evaluation = read(U / "TASKS.json"), read(U / "RESULT_MANIFEST.json"), read(U / "EVALUATION.json")
    pin(U / "RESULT_MANIFEST.json"); pin(U / "EVALUATION.json")
    verify_fresh_control_inputs(tasks)
    implementation = ast.parse((U / "trajectory.py").read_text(encoding="utf-8-sig"))
    imports = {n.name for node in ast.walk(implementation) if isinstance(node, ast.Import) for n in node.names}
    check(imports == {"numpy"} and not any(isinstance(node, ast.ImportFrom) for node in ast.walk(implementation)),
          "Pure trajectory module has no model/data/label/runtime imports")
    driver = (U / "run.py").read_text(encoding="utf-8-sig")
    check("trajectory.compare(q,r,local.tolist())" in driver and
          "legacy.dtw(legacy.alignment_cost(legacy.features(qraw),legacy.features(raw[qa])))" in driver and
          "original_indices_evaluation_only" not in driver,
          "Candidate receives only fixed geometry/path; fresh control baseline only receives raw observed features")
    check(tasks["component_names"] == COMPONENTS and tasks["geometry_independent_of_saved_seeds"] is True and
          tasks["query_truth_runtime"] is False, "Frozen component/seed/query truth contract")
    expected = {}
    for event in tasks["natural"]:
        for c in COMPONENTS:
            expected["pair_" + event["query"] + "_" + event["reference"] + "_" + event["parent"]["segment_id"] + "__" + c] = (
                "natural", event, event["parent"], c)
    for control in tasks["controls"]:
        for parent in control["parents"]:
            for c in COMPONENTS:
                expected["control_" + control["id"] + "_" + parent["segment_id"] + "__" + c] = ("control", control, parent, c)
    require(len(expected) == len(manifest["results"]) == 2825 and
            {r["id"] for r in manifest["results"]} == set(expected), "Exact2825 expected output inventory")
    require(len(manifest["new_control_baselines"]) == 64 and sum(c["baseline"] is None for c in tasks["controls"]) == 64,
            "Exactly64 new baselines only for new technical inputs")
    for b in manifest["new_control_baselines"]:
        pin(b["path"], b["sha256"]); saved = read(b["path"])
        require(saved["new_input_only"] and saved["true_index_used"] is False and saved["frozen_sha256"] == frozen_hash,
                "New baseline source-index/metadata isolation")
        control = next(c for c in tasks["controls"] if c["id"] == Path(b["path"]).stem)
        check(control["baseline"] is None and saved["source_query"] == control["array"] and
              saved["source_reference"] == tasks["rows"][control["alias"]]["input"], "New baseline exact technical-source identity")
        pin(saved["algorithm"], saved["algorithm_sha256"])
        valid_path(saved["path"], len(array(saved["source_query"])["x"]), len(array(saved["source_reference"])["x"]), "New fixed REL control baseline")
    missing = [dict(alias=c["alias"], role=c["role"], control=c["control"], parent_id=p["segment_id"], component=g,
                    status="NO_SAVED_CONTROL") for c in tasks["missing_controls"] for p in c["parents"] for g in COMPONENTS]
    same(manifest["missing_component_opportunities"], missing, "All10 unavailable control opportunities")
    same(evaluation["missing"], missing, "Unavailable opportunities retained in evaluation")
    for i, record in enumerate(manifest["results"]):
        verify_record(record, expected[record["id"]], tasks, frozen_hash)
        if (i + 1) % 250 == 0:
            print(f"Verified stored geometry {i + 1}/2825", flush=True)
    verify_aggregates(evaluation, manifest, tasks)
    check(manifest["new_model_inference"] == 0 and manifest["new_training"] == 0 and
          manifest["old_global_paths_replaced"] == 0, "No old path/model/training execution")
    for p, h in list(pins.items()):
        check(sha(p) == h, "Input/output unchanged throughout verification " + p)


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Preserve first verification result; no overwrite.")
    error = None
    try:
        main()
    except Exception:
        error = traceback.format_exc(); failures.append("Verification exception; see exception")
    report = dict(schema="wis_independent_hold_invariant_trajectory_verification_v1",
        status="PASS" if not failures else "FAIL", at_utc=datetime.now(timezone.utc).isoformat(),
        reviewer="Codex semantic_data_audit", checks=checks, failure_count=len(failures), failures=failures[:100],
        exception=error, verified_geometry_records=len(verified_rows), candidate_status_counts=dict(candidate_counts),
        baseline_origin_component_counts=dict(path_origins), axis_denominators=dict(denominator_counts),
        observed_and_missing_interval_counts=dict(gap_counts), technical_gate_checks=dict(technical_checks),
        technical_gate_failures=technical_failures,
        method="Independent raw named-column geometry, source-slice/mask arithmetic, BFS threshold reachability plus antidiagonal admissible-path counts; no producer or legacy alignment module imported.",
        limitations=["Candidate <= fixed-path maximum is an optimization property, not improved alignment accuracy.",
                     "Parent scopes inherit old baseline ambiguity; edits outside a crop stay in the ledger.",
                     "Hold/speed075 invariance is tested only for full coordinates and exact source coverage; speed125 drops samples.",
                     "Unknown coordinates keep whole-curve distance null. Partial baseline values are conditional lower bounds.",
                     "Historical public data and synthetic feature edits do not establish educational accuracy or whole-sign correctness."],
        actions=dict(model_inference=0, training=0, legacy_path_recalculation=0, old_score_recalculation=0,
                     candidate_minmax_rerun=0, independent_threshold_graph_checks=True, raw_video_decode=0,
                     private_or_mentor_access=0, old_files_modified=0),
        pins=pins, script_sha256=sha(__file__))
    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=report["status"], checks=checks, failures=len(failures), records=len(verified_rows),
                         output=str(OUTPUT), sha256=sha(OUTPUT))))
    if failures:
        raise SystemExit(1)
