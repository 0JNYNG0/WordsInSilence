"""Pure fixed-coordinate trajectory evidence; no data, models, or file access.

Candidate: endpoint-constrained discrete Frechet, using unweighted, unclipped
L-infinity point distance and monotone diagonal/horizontal/vertical steps.
Consecutive duplicate points do not change its distance. Deleting other samples
can change a discrete distance without changing the underlying physical curve.

Missing coordinates invalidate the whole candidate distance. Observed blocks
are described separately, never joined or independently optimized. The supplied
baseline path is only read: partial-coordinate residuals are conditional lower
bounds, not complete-vector distances, agreement, or educational error scores.
"""
import numpy as np


def _input(a):
    values = np.asarray(a["values"], dtype=np.float64)
    mask = np.asarray(a["mask"])
    if values.ndim != 2 or values.shape[1] < 1 or mask.shape != values.shape:
        raise ValueError("values and mask must share an N x D shape, D >= 1")
    if mask.dtype.kind != "b":
        raise ValueError("mask must contain booleans, not numeric confidence")
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed coordinates must be finite")
    n = len(values)
    frames, pts = np.asarray(a["frame_id"]), np.asarray(a["timestamp_us"])
    for key, array in (("frame_id", frames), ("timestamp_us", pts)):
        if array.shape != (n,) or array.dtype.kind not in "iu":
            raise ValueError(key + " must be an integer vector on the original axis")
        if n > 1 and not np.all(array[1:] > array[:-1]):
            raise ValueError(key + " must strictly increase; inputs are never reordered")
    return dict(values=values, mask=mask, frame_id=frames, timestamp_us=pts)


def _path(path, n, m):
    if path is None:
        return None
    out, previous = [], None
    for pair in path:
        if len(pair) != 2 or any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in pair):
            raise ValueError("Baseline path must contain integer index pairs")
        i, j = int(pair[0]), int(pair[1])
        if not (0 <= i < n and 0 <= j < m):
            raise ValueError("Baseline index outside original axis")
        if previous is not None and (i < previous[0] or j < previous[1] or [i, j] == previous):
            raise ValueError("Baseline path must be monotone and have no duplicate pairs")
        out.append([i, j]); previous = [i, j]
    return out


def solve(point_costs):
    """Exact min-max DP. All globally optimal paths count before tie selection.

The representative uses fixed diagonal/up/left priority for exact ties. The
ambiguity count (capped at two) uses all monotone paths below the final optimal
bottleneck, including nonoptimal prefixes that share the final bottleneck.
"""
    c = np.asarray(point_costs, dtype=np.float64)
    if c.ndim != 2 or not np.isfinite(c).all() or np.any(c < 0):
        raise ValueError("Point costs must be a finite nonnegative matrix")
    n, m = c.shape
    if not n or not m:
        return dict(distance=None, path=[], path_point_costs=[], optimal_path_ambiguous=None,
                    optimal_path_count_capped=0, complete=False, status="EMPTY_INPUT")
    dp = np.full((n, m), np.inf)
    parent = np.full((n, m), -1, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                dp[i, j] = c[i, j]
                continue
            choices = []
            if i and j:
                choices.append((max(c[i, j], dp[i - 1, j - 1]), 0))
            if i:
                choices.append((max(c[i, j], dp[i - 1, j]), 1))
            if j:
                choices.append((max(c[i, j], dp[i, j - 1]), 2))
            best, direction = min(choices, key=lambda value: value[0])
            dp[i, j], parent[i, j] = best, direction
    distance = float(dp[-1, -1])
    counts = np.zeros((n, m), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            if c[i, j] > distance:
                continue
            if not i and not j:
                counts[i, j] = 1
            else:
                counts[i, j] = min(2, (int(counts[i - 1, j - 1]) if i and j else 0)
                    + (int(counts[i - 1, j]) if i else 0) + (int(counts[i, j - 1]) if j else 0))
    path, i, j = [], n - 1, m - 1
    while True:
        path.append([i, j])
        if not i and not j:
            break
        direction = parent[i, j]
        if direction == 0:
            i -= 1; j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    if {i for i, j in path} != set(range(n)) or {j for i, j in path} != set(range(m)):
        raise AssertionError("Candidate path omitted original frames")
    return dict(distance=distance, path=path, path_point_costs=[float(c[i, j]) for i, j in path],
                complete=True, status="COMPLETE_OBSERVED_GEOMETRY_ONLY",
                optimal_path_count_capped=int(counts[-1, -1]), optimal_path_ambiguous=bool(counts[-1, -1] > 1),
                tie_representative="EXACT_MINMAX_THEN_FIXED_DIAGONAL_UP_LEFT_PRIORITY",
                all_input_frames_visited=True, endpoint_constrained=True)


def _stats(indices, costs, shared_counts, d):
    values = [costs[k] for k in indices if costs[k] is not None]
    complete_values = [costs[k] for k in indices if shared_counts[k] == d]
    complete = len(complete_values)
    partial = sum(0 < shared_counts[k] < d for k in indices)
    return dict(link_count=len(indices), assessable_link_count=len(values), complete_link_count=complete,
                partial_link_count=partial, unsupported_link_count=len(indices) - len(values),
                mean=float(np.mean(values)) if values else None, max=max(values) if values else None,
                complete_link_mean=float(np.mean(complete_values)) if complete_values else None,
                complete_link_max=max(complete_values) if complete_values else None,
                partial_coordinate_values_are_conditional_lower_bounds=True)


def _baseline(q, r, path):
    n, d = q["values"].shape
    m = len(r["values"])
    links = path or []
    costs, shared_counts = [], []
    matched = [np.zeros((n, d), dtype=np.int64), np.zeros((m, d), dtype=np.int64)]
    missing = [np.zeros((n, d), dtype=np.int64), np.zeros((m, d), dtype=np.int64)]
    adjacency = [[[] for _ in range(n)], [[] for _ in range(m)]]
    for k, (i, j) in enumerate(links):
        shared = q["mask"][i] & r["mask"][j]
        count = int(shared.sum())
        value = float(np.max(np.abs(q["values"][i, shared] - r["values"][j, shared]))) if count else None
        costs.append(value); shared_counts.append(count)
        matched[0][i] += shared; matched[1][j] += shared
        missing[0][i] += q["mask"][i] & ~r["mask"][j]
        missing[1][j] += r["mask"][j] & ~q["mask"][i]
        adjacency[0][i].append(k); adjacency[1][j].append(k)
    out = dict(path=links, point_costs=costs, shared_dimension_counts=shared_counts,
               possible_link_dimension_opportunities=len(links) * d,
               shared_link_dimension_opportunities=sum(shared_counts),
               **_stats(list(range(len(links))), costs, shared_counts, d))
    for axis, source, side in (("query", q, 0), ("reference", r, 1)):
        per_frame = [_stats(ids, costs, shared_counts, d) for ids in adjacency[side]]
        out[axis] = dict(mean=[x["mean"] for x in per_frame], max=[x["max"] for x in per_frame],
                         complete_link_mean=[x["complete_link_mean"] for x in per_frame],
                         complete_link_max=[x["complete_link_max"] for x in per_frame],
                         aligned_link_count=[len(x) for x in adjacency[side]],
                         assessable_link_count=[x["assessable_link_count"] for x in per_frame],
                         complete_link_count=[x["complete_link_count"] for x in per_frame],
                         partial_link_count=[x["partial_link_count"] for x in per_frame],
                         unsupported_link_count=[x["unsupported_link_count"] for x in per_frame],
                         matched_support=(matched[side] > 0).tolist(),
                         matched_link_count=matched[side].tolist(),
                         missing_counterpart_link_count=missing[side].tolist(),
                         possible_frames=len(source["values"]), possible_cells=source["mask"].size,
                         own_observed_cells=int(source["mask"].sum()),
                         matched_cells=int((matched[side] > 0).sum()),
                         unaligned_indices=[i for i, x in enumerate(adjacency[side]) if not x])
    full = bool(n and m and links and links[0] == [0, 0] and links[-1] == [n - 1, m - 1]
                and all(adjacency[0]) and all(adjacency[1]))
    out.update(status="NOT_SUPPLIED" if path is None else "SUPPLIED_FULL_SCOPE" if full else "SUPPLIED_PARTIAL_OR_EMPTY_SCOPE",
               full_scope=full, mean_weighting="UNWEIGHTED_OVER_SUPPLIED_LINKS_WITH_ANY_SHARED_COORDINATE",
               source_kind="CALLER_SUPPLIED_PATH_NOT_RECOMPUTED_OR_CERTIFIED_AUTOMATIC",
               is_previous_model_score=False, path_unchanged=True,
               whole_vector_distance_known_on_every_link=bool(full and all(c == d for c in shared_counts)))
    return out, adjacency


def _runs(flags):
    runs, start = [], None
    for i in range(len(flags) + 1):
        active = i < len(flags) and bool(flags[i])
        if active and start is None:
            start = i
        elif not active and start is not None:
            runs.append((start, i - 1)); start = None
    return runs


def _range(a, start, end):
    return dict(start=start, end=end, frame_count=end - start + 1,
                frame_range=[int(a["frame_id"][start]), int(a["frame_id"][end])],
                pts_range=[int(a["timestamp_us"][start]), int(a["timestamp_us"][end])])


def _scope(a, adjacency, baseline):
    n, d = a["values"].shape
    full = np.all(a["mask"], axis=1)
    def intervals(flags, observed_block):
        out = []
        for start, end in _runs(flags):
            ids = [k for frame in adjacency[start:end + 1] for k in frame]
            out.append(dict(**_range(a, start, end),
                            possible_cells=(end - start + 1) * d,
                            observed_cells=int(a["mask"][start:end + 1].sum()),
                            has_at_least_two_frames=end > start,
                            has_two_distinct_observed_positions=bool(np.any(a["values"][start:end + 1] != a["values"][start])) if observed_block else None,
                            baseline=_stats(ids, baseline["point_costs"], baseline["shared_dimension_counts"], d)))
        return out
    return dict(frame_id=a["frame_id"].tolist(), timestamp_us=a["timestamp_us"].tolist(),
                observed_mask=a["mask"].tolist(), possible_frames=n, possible_cells=n * d,
                observed_cells=int(a["mask"].sum()), missing_cells=int((~a["mask"]).sum()),
                observed_dimension_count=a["mask"].sum(axis=1).tolist(),
                fully_observed_frames=int(full.sum()), incomplete_frames=int((~full).sum()),
                observed_blocks=intervals(full, True), missing_intervals=intervals(~full, False),
                missing_by_dimension=[dict(dimension=k, intervals=[_range(a, start, end)
                                       for start, end in _runs(~a["mask"][:, k])]) for k in range(d)],
                masked_nonfinite_cells=int((~a["mask"] & ~np.isfinite(a["values"])).sum()),
                intervals_are_inclusive_original_axis_indices=True,
                observed_blocks_are_not_joined=True, all_opportunities_retained=True)


def compare(q, r, path=None):
    """Return independent candidate geometry and unchanged-path partial evidence.

Only values/mask/frame_id/timestamp_us are read. Dimension identity is supplied
by the caller and positional; no word, phase, source-index, hand reassignment,
or query annotation controls the candidate's correspondence search.
"""
    query, reference = _input(q), _input(r)
    n, d = query["values"].shape
    m, rd = reference["values"].shape
    if d != rd:
        raise ValueError("Query and reference coordinate dimensions differ")
    original_path = _path(path, n, m)
    baseline, adjacency = _baseline(query, reference, original_path)
    complete = bool(n and m and query["mask"].all() and reference["mask"].all())
    if complete:
        point_costs = np.max(np.abs(query["values"][:, None, :] - reference["values"][None, :, :]), axis=2)
        candidate = solve(point_costs)
        witnesses = []
        for k, (i, j) in enumerate(candidate["path"]):
            if candidate["path_point_costs"][k] != candidate["distance"]:
                continue
            delta = np.abs(query["values"][i] - reference["values"][j])
            witnesses.append(dict(query_index=i, reference_index=j,
                query_frame_id=int(query["frame_id"][i]), reference_frame_id=int(reference["frame_id"][j]),
                query_timestamp_us=int(query["timestamp_us"][i]), reference_timestamp_us=int(reference["timestamp_us"][j]),
                point_cost=float(candidate["distance"]), dimensions=np.flatnonzero(delta == candidate["distance"]).tolist()))
        candidate["bottleneck_witnesses"] = witnesses
    else:
        candidate = dict(distance=None, path=[], path_point_costs=[], bottleneck_witnesses=[],
                         complete=False, status="EMPTY_INPUT" if not n or not m else "INCOMPLETE_OBSERVATION",
                         optimal_path_ambiguous=None, optimal_path_count_capped=0,
                         all_input_frames_visited=False, endpoint_constrained=True)
    candidate.update(metric="UNWEIGHTED_UNCLIPPED_L_INFINITY_DISCRETE_FRECHET",
                     has_at_least_two_frames_each_axis=bool(n >= 2 and m >= 2),
                     query_has_two_distinct_positions=bool(np.any(query["values"] != query["values"][0])) if complete else None,
                     reference_has_two_distinct_positions=bool(np.any(reference["values"] != reference["values"][0])) if complete else None,
                     bottleneck_witness_scope="REPRESENTATIVE_OPTIMAL_PATH_NOT_UNIQUE_CAUSE",
                     original_global_path_replaced=False, educational_error=None,
                     alignment_verified=False, stage_order_verified=False)
    return dict(schema="wis_hold_invariant_trajectory_v1", dimensions=d,
                query=_scope(query, adjacency[0], baseline), reference=_scope(reference, adjacency[1], baseline),
                candidate=candidate, baseline=baseline, semantic_verdict=None,
                whole_sign_verdict=None, correction_advice=None, alignment_verified=False,
                missing_values_imputed=False, input_values_and_masks_unchanged=True,
                coordinate_identity="FIXED_INPUT_COLUMNS_NO_LEFT_RIGHT_REASSIGNMENT",
                identifiability_limits=[
                    "Consecutive identical point duplication is invariant; arbitrary resampling/deletion is not guaranteed invariant.",
                    "Straight-path frame deletion can be indistinguishable from speed change; a discrete sampling difference is not a linguistic omission.",
                    "Symmetric reversal, repeated visits, self-crossings, and hidden motion can remain unidentifiable.",
                    "Small geometry difference is not correct sign; large geometry difference is not automatic educational error."])
