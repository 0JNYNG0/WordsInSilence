"""Read-only context for an existing path and its full two-axis evidence.

No alignment, numerical residual, mask, stage inference, or semantic verdict is
computed here. JSON pointers address the supplied, unchanged evidence. Context
episodes compress consecutive non-DIRECT cells of one exact feature. Reference
availability runs compress identical support lists, never values or residuals.
"""
from collections import Counter
import math


AXES = ("query", "reference")
MATRICES = ("own_observed", "matched_support", "mixed_counterpart_missing",
            "mean_residual", "max_residual", "cell_status", "cell_reason")


def _pointer(group, axis, field):
    escaped = str(group).replace("~", "~0").replace("/", "~1")
    return "/groups/" + escaped + "/" + axis + "/" + field


def _flags():
    return dict(verdict=None, semantic_verdict=None, whole_sign_verdict=None,
                correspondence_correct=None, alignment_verified=False,
                correction_advice=None, missing_values_imputed=False,
                raw_evidence_unchanged=True)


def _check_axis(a):
    n, d = a["possible_shape"]
    if not isinstance(n, int) or not isinstance(d, int) or n < 1 or d < 1:
        raise ValueError("Invalid possible_shape")
    if len(a["dimension_labels"]) != d or len(set(a["dimension_labels"])) != d:
        raise ValueError("Invalid dimension labels")
    for key in ("frame_id", "timestamp_us"):
        values = a[key]
        if len(values) != n or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                   or not math.isfinite(v) for v in values):
            raise ValueError("Invalid original " + key)
        if any(x >= y for x, y in zip(values, values[1:])):
            raise ValueError("Original frame IDs/PTS must increase; never reorder")
    for key in MATRICES:
        if len(a[key]) != n or any(len(row) != d for row in a[key]):
            raise ValueError("Invalid matrix shape: " + key)
    for i in range(n):
        for j in range(d):
            own, matched, mixed = (a[k][i][j] for k in MATRICES[:3])
            if any(type(v) is not bool for v in (own, matched, mixed)):
                raise ValueError("Observation masks must be booleans")
            if (matched and not own) or (mixed and not matched):
                raise ValueError("Inconsistent observation masks")
            for key in ("mean_residual", "max_residual"):
                value = a[key][i][j]
                if (not matched and value is not None) or (matched and
                    (not isinstance(value, (int, float)) or not math.isfinite(value))):
                    raise ValueError("Residual availability differs from shared support")
    return n, d


def _check_evidence(evidence):
    groups = evidence["groups"]
    if not groups:
        raise ValueError("No evidence groups")
    original_axes = {}
    for group in groups.values():
        for axis in AXES:
            _check_axis(group[axis])
            signature = (group[axis]["frame_id"], group[axis]["timestamp_us"])
            if axis in original_axes and signature != original_axes[axis]:
                raise ValueError("Groups disagree on original frame axes")
            original_axes[axis] = signature
        if group["query"]["dimension_labels"] != group["reference"]["dimension_labels"]:
            raise ValueError("Query/reference feature identities differ")
    return groups


def _links(path, n, m):
    qlinks, rlinks = [[] for _ in range(n)], [[] for _ in range(m)]
    previous = None
    for pair in path:
        if (len(pair) != 2 or any(type(v) is not int for v in pair)
                or not (0 <= pair[0] < n and 0 <= pair[1] < m)):
            raise ValueError("Invalid original path index")
        i, j = pair
        if previous is not None:
            di, dj = i - previous[0], j - previous[1]
            if di < 0 or dj < 0 or di + dj == 0:
                raise ValueError("Original path is nonmonotone or duplicated")
        qlinks[i].append(j)
        rlinks[j].append(i)
        previous = pair
    if path and (any(not x for x in qlinks) or any(not x for x in rlinks)):
        raise ValueError("Original path omits original axis frames")
    return {"query": qlinks, "reference": rlinks}


def _state(a, i, d):
    if a["matched_support"][i][d]:
        return "PARTIAL_SHARED" if a["mixed_counterpart_missing"][i][d] else "DIRECT_COMPLETE"
    return "NO_SHARED"


def _reference_catalog(clip, frame_ids):
    if clip.get("role") != "fit":
        raise ValueError("Only the current FIT reference clip may supply stage context")
    alias = clip.get("alias")
    if not alias:
        raise ValueError("Reference occurrence identity requires reference alias")
    catalog = {}
    for region in clip["regions"]:
        if region.get("region_type") != "observed_physical_core":
            continue
        rid, core = region["id"], region["core"]
        if rid in catalog or len(core) != 2 or core[0] > core[1]:
            raise ValueError("Invalid or duplicate FIT occurrence")
        catalog[rid] = {k: region[k] for k in ("id", "kind", "description_ko", "source_segment_id", "source_kind", "motion_subphase") if k in region}
        catalog[rid].update(reference_id=alias, core=list(core), occurrence_key=[alias, rid])
    membership = {fid: [rid for rid, region in catalog.items()
                         if region["core"][0] <= fid <= region["core"][1]]
                  for fid in frame_ids}
    return catalog, membership


def _scope(frame_ids, membership):
    ids = set()
    unknown, overlapping = [], []
    for fid in frame_ids:
        found = membership[fid]
        ids.update(found)
        if not found:
            unknown.append(fid)
        elif len(found) > 1:
            overlapping.append(fid)
    return dict(occurrence_ids=sorted(ids), unknown_reference_frames=unknown,
                overlapping_reference_frames=overlapping,
                certain_singleton=len(ids) == 1 and not unknown and not overlapping)


def _reference_context(before, after):
    if before is None or after is None:
        return "INSUFFICIENT_ANCHORS"
    b, a = before["reference_scope"], after["reference_scope"]
    if not b["certain_singleton"] or not a["certain_singleton"]:
        return "MIXED_OR_UNCERTAIN_REFERENCE_SCOPE"
    return ("SAME_REFERENCE_OCCURRENCE_BRACKET" if b["occurrence_ids"] == a["occurrence_ids"]
            else "CROSS_REFERENCE_OCCURRENCES")


def _anchor(a, other, links, i, d, axis, membership):
    if i is None:
        return None
    shared = [j for j in links[i] if a["own_observed"][i][d] and other["own_observed"][j][d]]
    if not shared or len(shared) != len(links[i]):
        raise ValueError("Context anchor is not DIRECT_COMPLETE")
    other_frames = [other["frame_id"][j] for j in shared]
    reference_frames = other_frames if axis == "query" else [a["frame_id"][i]]
    return dict(index=i, frame_id=a["frame_id"][i], timestamp_us=a["timestamp_us"][i],
                observation_state="DIRECT_COMPLETE", shared_counterpart_indices=shared,
                shared_counterpart_frame_ids=other_frames,
                shared_counterpart_timestamp_us=[other["timestamp_us"][j] for j in shared],
                aligned_counterpart_count=len(links[i]), shared_counterpart_count=len(shared),
                missing_counterpart_count=0, reference_scope=_scope(reference_frames, membership))


def _temporal(before, after):
    if before is None and after is None:
        return "NO_ANCHORS"
    if before is None or after is None:
        return "OPEN_BOUNDARY"
    b, a = before["shared_counterpart_indices"], after["shared_counterpart_indices"]
    if max(b) < min(a):
        return "BOUNDED_BY_DISTINCT_OBSERVATIONS"
    if min(b) > max(a):
        raise ValueError("Nonmonotone counterpart anchors")
    return "MAPPING_COMPRESSED_OR_OVERLAPPING"


def _axis_context(group, axis, a, other, links, membership):
    n, d = a["possible_shape"]
    episodes, total_states, total_reasons, total_links = [], Counter(), Counter(), Counter()
    for dim in range(d):
        states = [_state(a, i, dim) for i in range(n)]
        totals_by_cell = []
        for i in range(n):
            own = a["own_observed"][i][dim]
            observed = sum(other["own_observed"][j][dim] for j in links[i])
            shared = observed if own else 0
            missing = len(links[i]) - observed
            if (a["matched_support"][i][dim] != (shared > 0) or
                    a["mixed_counterpart_missing"][i][dim] != (shared > 0 and missing > 0)):
                raise ValueError("Stored support disagrees with original path and masks")
            counts = dict(aligned_links=len(links[i]), counterpart_observed_links=observed,
                          shared_links=shared, own_observed_counterpart_missing_links=missing if own else 0,
                          own_missing_counterpart_observed_links=observed if not own else 0,
                          both_missing_links=missing if not own else 0)
            totals_by_cell.append(counts)
            total_links.update(counts)
            total_states[states[i]] += 1
            total_reasons[a["cell_reason"][i][dim]] += 1
        i = 0
        while i < n:
            if states[i] == "DIRECT_COMPLETE":
                i += 1
                continue
            start = i
            while i + 1 < n and states[i + 1] != "DIRECT_COMPLETE":
                i += 1
            end = i
            before = _anchor(a, other, links, start - 1 if start else None, dim, axis, membership)
            after = _anchor(a, other, links, end + 1 if end + 1 < n else None, dim, axis, membership)
            link_counts = Counter()
            for cell in totals_by_cell[start:end + 1]:
                link_counts.update(cell)
            episodes.append(dict(dimension=dim, dimension_label=a["dimension_labels"][dim],
                start=start, end=end, cell_count=end - start + 1,
                frame_range=[a["frame_id"][start], a["frame_id"][end]],
                pts_range=[a["timestamp_us"][start], a["timestamp_us"][end]],
                anchor_before=before, anchor_after=after,
                temporal_context=_temporal(before, after),
                reference_context=_reference_context(before, after),
                status_counts=dict(Counter(states[start:end + 1])),
                missing_reason_counts=dict(Counter(a["cell_reason"][k][dim] for k in range(start, end + 1))),
                source_cell_status_counts=dict(Counter(a["cell_status"][k][dim] for k in range(start, end + 1))),
                link_provenance_counts=dict(link_counts)))
            i += 1
    direct = total_states["DIRECT_COMPLETE"]
    needed = n * d - direct
    if sum(e["cell_count"] for e in episodes) != needed:
        raise AssertionError("Context compression lost source opportunities")
    return dict(possible_shape=[n, d], possible=n * d, direct_complete_cells=direct,
                needs_context_cells=needed, status_counts=dict(total_states),
                missing_reason_counts=dict(total_reasons), link_provenance_counts=dict(total_links),
                episodes=episodes, ranges_are_inclusive_original_axis_indices=True,
                all_opportunities_retained=True, direct_cells_retained_in_raw_evidence=True,
                source_fields={key: _pointer(group, axis, key) for key in
                               MATRICES + ("frame_id", "timestamp_us", "dimension_labels", "counts")},
                **_flags())


def annotate(evidence, path, reference_clip):
    """Attach compressed observational context; never alter the input objects."""
    groups = _check_evidence(evidence)
    first = next(iter(groups.values()))
    links = _links(path, len(first["query"]["frame_id"]), len(first["reference"]["frame_id"]))
    catalog, membership = _reference_catalog(reference_clip, first["reference"]["frame_id"])
    out = {}
    for group, axes in groups.items():
        out[group] = {axis: _axis_context(group, axis, axes[axis], axes[AXES[1 - AXES.index(axis)]], links[axis], membership)
                      for axis in AXES}
    return dict(schema="wis_fixed_path_context_v1", groups=out,
                reference_id=reference_clip["alias"], reference_occurrences=catalog,
                possible=sum(a[axis]["possible"] for a in out.values() for axis in AXES),
                anchor_eligibility="DIRECT_COMPLETE_ONLY", source_pointer_root="supplied_evidence",
                context_interpolates_values=False, reference_scope_is_query_stage_claim=False,
                observation_state_is_alignment_correctness=False, path_unchanged=True,
                **_flags())


def availability(records):
    """List every supplied reference's query support, without combining values."""
    if not records:
        raise ValueError("Reference records are required")
    records = sorted(records, key=lambda rec: rec["reference_id"])
    ids = [r["reference_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate reference record")
    base = records[0]
    for rec in records:
        if rec["query_id"] != base["query_id"] or rec["seed"] != base["seed"]:
            raise ValueError("INPUT_VERSION_MISMATCH: query or seed")
        _check_evidence(rec["evidence"])
        if set(rec["evidence"]["groups"]) != set(base["evidence"]["groups"]):
            raise ValueError("INPUT_VERSION_MISMATCH: feature groups")
    outputs = {}
    per_ref = {r["reference_id"]: dict(query_possible=0, reference_possible=0,
                                     direct_complete=0, partial_shared=0, unsupported=0)
               for r in records}
    for group, axes in base["evidence"]["groups"].items():
        q = axes["query"]
        n, d = q["possible_shape"]
        by_ref = []
        for rec in records:
            current = rec["evidence"]["groups"][group]["query"]
            for field in ("possible_shape", "frame_id", "timestamp_us", "dimension_labels", "own_observed"):
                if current[field] != q[field]:
                    raise ValueError("INPUT_VERSION_MISMATCH: query " + field)
            by_ref.append(current)
            counts = per_ref[rec["reference_id"]]
            counts["query_possible"] += n * d
            rn, rd = rec["evidence"]["groups"][group]["reference"]["possible_shape"]
            counts["reference_possible"] += rn * rd
        runs, counts = [], Counter()
        for dim in range(d):
            previous, start = None, 0
            for i in range(n + 1):
                signature = None
                if i < n:
                    direct, partial, unsupported = [], [], []
                    for rid, a in zip(ids, by_ref):
                        state = _state(a, i, dim)
                        if state == "DIRECT_COMPLETE":
                            direct.append(rid); key = "direct_complete"
                        elif state == "PARTIAL_SHARED":
                            partial.append(rid); key = "partial_shared"
                        else:
                            unsupported.append((rid, a["cell_reason"][i][dim])); key = "unsupported"
                        per_ref[rid][key] += 1
                    signature = (tuple(direct), tuple(partial), tuple(unsupported))
                    counts["possible"] += 1
                    counts["own_observed" if q["own_observed"][i][dim] else "own_unobserved"] += 1
                    counts["any_direct" if direct else "partial_only" if partial else "all_unsupported"] += 1
                    if direct and unsupported:
                        counts["covered_despite_other_reference_missing"] += 1
                if i and signature != previous:
                    direct, partial, unsupported = previous
                    runs.append(dict(dimension=dim, dimension_label=q["dimension_labels"][dim],
                        start=start, end=i - 1, cell_count=i - start,
                        direct_refs=list(direct), partial_refs=list(partial),
                        unsupported_refs=[dict(reference_id=rid, reason=reason) for rid, reason in unsupported],
                        covered_despite_other_reference_missing=bool(direct and unsupported)))
                    start = i
                previous = signature
        if sum(run["cell_count"] for run in runs) != n * d:
            raise AssertionError("Availability compression lost query opportunities")
        outputs[group] = dict(query_shape=[n, d], possible=n * d, runs=runs, counts=dict(counts),
                              source_fields={key: _pointer(group, "query", key) for key in
                                             ("frame_id", "timestamp_us", "dimension_labels", "own_observed")})
    for counts in per_ref.values():
        if counts["direct_complete"] + counts["partial_shared"] + counts["unsupported"] != counts["query_possible"]:
            raise AssertionError("Reference support partition lost opportunities")
    return dict(schema="wis_reference_availability_v1", query_id=base["query_id"], seed=base["seed"],
                reference_ids=ids, groups=outputs, per_reference_counts=per_ref,
                unique_query_possible=sum(group["possible"] for group in outputs.values()),
                source_records=[dict(reference_id=r["reference_id"], source_id=r["source_id"],
                                     annotation_id=r.get("annotation_id")) for r in records],
                reference_list_scope="ALL_SUPPLIED_RECORDS_CALLER_RETAINS_EXPECTED_REFERENCE_MANIFEST",
                coverage_only_not_value_agreement=True, residuals_aggregated=False,
                reference_winner=None, original_unsupported_evidence_retained=True,
                **_flags())
