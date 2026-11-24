"""Independent saved-output verification. No candidate/model/DP/scoring imports.

Run after root authorizes completed output verification. Original residuals are
checked by immutable source hashes/digests/pointers, not recomputed. Support and
context are independently reconstructed with path incidence-matrix arithmetic.
First result uses exclusive creation and cannot be silently overwritten.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import ast
import gzip
import hashlib
import json
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
OPS = UNIT.parent
OUTPUT = HERE / "RESULT_VERIFICATION.json"
FLAGS = dict(verdict=None, semantic_verdict=None, whole_sign_verdict=None,
             correspondence_correct=None, alignment_verified=False,
             correction_advice=None, missing_values_imputed=False,
             raw_evidence_unchanged=True)
LINK_NAMES = ("aligned_links", "counterpart_observed_links", "shared_links",
              "own_observed_counterpart_missing_links", "own_missing_counterpart_observed_links",
              "both_missing_links")
checks, failures, audit_pins = 0, [], {}
totals, temporal_totals, reference_totals, old_counts, availability_totals = (Counter() for _ in range(5))
axis_totals, query_cache = defaultdict(Counter), defaultdict(dict)
context_rows, availability_rows, annotation_paths = {}, {}, {}


def read(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw.decode("utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def check(condition, message):
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def require(condition, message):
    check(condition, message)
    if not condition:
        raise AssertionError(message)


def pin(path, expected=None):
    path = str(Path(path).resolve())
    actual = sha(path)
    if expected is not None:
        check(actual == expected, "SHA: " + path)
    audit_pins[path] = actual
    return actual


def counter_equal(saved, expected, label):
    check({k: v for k, v in saved.items() if v != 0} ==
          {str(k): int(v) for k, v in expected.items() if v != 0}, label)


def flags(value, label):
    check(all(k in value and value[k] == v for k, v in FLAGS.items()), label)


def pointer(group, axis, field):
    return "/groups/" + group.replace("~", "~0").replace("/", "~1") + "/" + axis + "/" + field


def catalog_for(reference):
    require(reference["role"] == "fit", "Only FIT reference supplies occurrence context")
    catalog = {}
    for region in reference["regions"]:
        if region["region_type"] != "observed_physical_core":
            continue
        rid = region["id"]
        require(rid not in catalog, "Unique reference occurrence ID")
        catalog[rid] = {k: region[k] for k in
                       ("id", "kind", "description_ko", "source_segment_id", "source_kind", "motion_subphase") if k in region}
        catalog[rid].update(reference_id=reference["alias"], core=region["core"],
                            occurrence_key=[reference["alias"], rid])
    return catalog


def occurrence_scope(frame_ids, catalog):
    matches = [[rid for rid, r in catalog.items() if r["core"][0] <= f <= r["core"][1]] for f in frame_ids]
    ids = sorted({rid for membership in matches for rid in membership})
    unknown = [f for f, membership in zip(frame_ids, matches) if not membership]
    overlap = [f for f, membership in zip(frame_ids, matches) if len(membership) > 1]
    return dict(occurrence_ids=ids, unknown_reference_frames=unknown,
                overlapping_reference_frames=overlap,
                certain_singleton=len(ids) == 1 and not unknown and not overlap)


def expected_anchor(index, own, other, incidence, axis, catalog):
    if index is None:
        return None
    indices = np.flatnonzero(incidence[index]).tolist()
    frames = [other["frame_id"][j] for j in indices]
    ref_frames = frames if axis == "query" else [own["frame_id"][index]]
    return dict(index=index, frame_id=own["frame_id"][index], timestamp_us=own["timestamp_us"][index],
                observation_state="DIRECT_COMPLETE", shared_counterpart_indices=indices,
                shared_counterpart_frame_ids=frames,
                shared_counterpart_timestamp_us=[other["timestamp_us"][j] for j in indices],
                aligned_counterpart_count=len(indices), shared_counterpart_count=len(indices),
                missing_counterpart_count=0, reference_scope=occurrence_scope(ref_frames, catalog))


def verify_axis(saved, original, other, incidence, group, axis, catalog, label):
    n, dims = original["possible_shape"]
    own = np.asarray(original["own_observed"], dtype=bool)
    counterpart = np.asarray(other["own_observed"], dtype=np.int64)
    links = incidence.sum(axis=1)[:, None]
    observed = incidence @ counterpart
    absent = links - observed
    shared = own & (observed > 0)
    partial = shared & (absent > 0)
    direct = own & (absent == 0) & (links > 0)
    state = np.where(direct, "DIRECT_COMPLETE", np.where(partial, "PARTIAL_SHARED", "NO_SHARED"))
    provenance = np.stack((np.broadcast_to(links, own.shape), observed,
                           np.where(own, observed, 0), np.where(own, absent, 0),
                           np.where(own, 0, observed), np.where(own, 0, absent)), axis=-1)
    require(own.shape == (n, dims), label + ": mask shape")
    check(np.array_equal(shared, original["matched_support"]), label + ": raw path/mask shared support")
    check(np.array_equal(partial, original["mixed_counterpart_missing"]), label + ": raw path/mask partial support")
    direct_count = int(direct.sum())
    base_count = dict(possible=n * dims, direct_complete_cells=direct_count, needs_context_cells=n * dims - direct_count)
    check(saved["possible_shape"] == [n, dims] and all(saved[k] == v for k, v in base_count.items()),
          label + ": opportunity partition")
    counter_equal(saved["status_counts"], Counter(state.ravel()), label + ": all status counts")
    reasons = np.asarray(original["cell_reason"], dtype=object)
    statuses = np.asarray(original["cell_status"], dtype=object)
    counter_equal(saved["missing_reason_counts"], Counter(reasons.ravel()), label + ": all reasons")
    counter_equal(saved["link_provenance_counts"], dict(zip(LINK_NAMES, provenance.sum(axis=(0, 1)))),
                  label + ": all link provenance")
    for field, value in saved["source_fields"].items():
        check(field in original and value == pointer(group, axis, field), label + ": original pointer " + field)
    require({"mean_residual", "max_residual", "own_observed", "matched_support", "mixed_counterpart_missing",
             "frame_id", "timestamp_us", "dimension_labels", "counts", "cell_status", "cell_reason"}.issubset(saved["source_fields"]),
            label + ": complete direct-evidence pointers")
    flags(saved, label + ": nonsemantic flags")
    check(saved["ranges_are_inclusive_original_axis_indices"] and saved["all_opportunities_retained"] and
          saved["direct_cells_retained_in_raw_evidence"], label + ": immutable axis semantics")
    expected_ranges = []
    for dim in range(dims):
        transition = np.diff(np.r_[0, (~direct[:, dim]).astype(int), 0])
        expected_ranges.extend((dim, int(start), int(end) - 1) for start, end in
                               zip(np.flatnonzero(transition == 1), np.flatnonzero(transition == -1)))
    require([(e["dimension"], e["start"], e["end"]) for e in saved["episodes"]] == expected_ranges,
            label + ": exact maximal non-DIRECT runs")
    for episode, (dim, start, end) in zip(saved["episodes"], expected_ranges):
        loc = label + f": dimension {dim} frames {start}:{end}"
        check(episode["dimension_label"] == original["dimension_labels"][dim] and
              episode["cell_count"] == end - start + 1 and
              episode["frame_range"] == [original["frame_id"][start], original["frame_id"][end]] and
              episode["pts_range"] == [original["timestamp_us"][start], original["timestamp_us"][end]],
              loc + ": native cell identity/range")
        before_indices = np.flatnonzero(direct[:start, dim])
        after_indices = np.flatnonzero(direct[end + 1:, dim]) + end + 1
        before = expected_anchor(int(before_indices[-1]) if len(before_indices) else None,
                                 original, other, incidence, axis, catalog)
        after = expected_anchor(int(after_indices[0]) if len(after_indices) else None,
                                original, other, incidence, axis, catalog)
        check(episode["anchor_before"] == before and episode["anchor_after"] == after,
              loc + ": nearest strict DIRECT anchors, every counterpart and occurrence")
        if before is None and after is None:
            temporal = "NO_ANCHORS"
        elif before is None or after is None:
            temporal = "OPEN_BOUNDARY"
        elif max(before["shared_counterpart_indices"]) < min(after["shared_counterpart_indices"]):
            temporal = "BOUNDED_BY_DISTINCT_OBSERVATIONS"
        else:
            temporal = "MAPPING_COMPRESSED_OR_OVERLAPPING"
        if before is None or after is None:
            reference = "INSUFFICIENT_ANCHORS"
        elif not before["reference_scope"]["certain_singleton"] or not after["reference_scope"]["certain_singleton"]:
            reference = "MIXED_OR_UNCERTAIN_REFERENCE_SCOPE"
        elif before["reference_scope"]["occurrence_ids"] == after["reference_scope"]["occurrence_ids"]:
            reference = "SAME_REFERENCE_OCCURRENCE_BRACKET"
        else:
            reference = "CROSS_REFERENCE_OCCURRENCES"
        check(episode["temporal_context"] == temporal and episode["reference_context"] == reference,
              loc + ": bracket versus occurrence ID, never kind-equivalence")
        counter_equal(episode["status_counts"], Counter(state[start:end + 1, dim]), loc + ": states")
        counter_equal(episode["missing_reason_counts"], Counter(reasons[start:end + 1, dim]), loc + ": reasons")
        counter_equal(episode["source_cell_status_counts"], Counter(statuses[start:end + 1, dim]), loc + ": original statuses")
        counter_equal(episode["link_provenance_counts"], dict(zip(LINK_NAMES, provenance[start:end + 1, dim].sum(axis=0))),
                      loc + ": provenance partition")
        temporal_totals[temporal] += end - start + 1
        reference_totals[reference] += end - start + 1
    check(sum(e["cell_count"] for e in saved["episodes"]) + direct_count == n * dims,
          label + ": full denominator after compression")
    totals.update(base_count)
    totals["episodes"] += len(expected_ranges)
    axis_totals[axis].update(base_count)
    old_counts.update(dict(possible=n * dims, shared=int(shared.sum()), partial=int(partial.sum()),
                           own_unobserved=int((~own).sum()), own_observed_without_shared=int((own & ~shared).sum())))
    return dict(shape=[n, dims], frame_id=original["frame_id"], timestamp_us=original["timestamp_us"],
                dimension_labels=original["dimension_labels"], own=own, state=state, reasons=reasons,
                reference_possible=len(other["frame_id"]) * dims), base_count


def verify_context(task, item, refs, frozen_sha):
    label = f"{task['query']}/{task['reference']}/{task['seed']}"
    pin(item["path"], item["sha256"])
    out = read(item["path"])
    pin(task["path"], task["sha256"])
    source = read(task["path"])
    pin(task["source"], task["source_sha256"])
    old = read(task["source"])
    check(out["source"] == task["path"] and item["source"] == task["path"] and
          out["source_sha256"] == task["sha256"], label + ": original readout SHA/pointer")
    check(all(out[k] == task[k] and source[k] == task[k] for k in ("query", "reference", "seed")),
          label + ": source identity")
    check(out["original_evidence_digest"] == digest(source) and out["original_path_exact"] is True,
          label + ": exact whole evidence digest, including every residual/null/mask")
    check(source["path"] == old["path"] and source["source"] == task["source"] and
          source["source_sha256"] == task["source_sha256"], label + ": old global path exact")
    check(out["frozen_sha256"] == frozen_sha and out["query_truth_used"] is False,
          label + ": frozen execution without query truth")
    original, annotation = source["evidence"], out["annotation"]
    flags(annotation, label + ": not a verdict")
    check(annotation["reference_scope_is_query_stage_claim"] is False and
          annotation["observation_state_is_alignment_correctness"] is False and
          annotation["context_interpolates_values"] is False and annotation["path_unchanged"] is True and
          annotation["anchor_eligibility"] == "DIRECT_COMPLETE_ONLY", label + ": claim boundaries")
    require(set(annotation["groups"]) == set(original["groups"]) and len(annotation["groups"]) == 16,
            label + ": complete source groups")
    catalog = catalog_for(refs[task["reference"]])
    check(annotation["reference_occurrences"] == catalog and annotation["reference_id"] == task["reference"],
          label + ": exact FIT core catalog, no query catalog or widened outer")
    n, m = len(original["query_frame_ids"]), len(original["reference_frame_ids"])
    path = np.asarray(source["path"], dtype=np.int64)
    require(path.ndim == 2 and path.shape[1] == 2, label + ": fixed path shape")
    incidence = np.zeros((n, m), dtype=np.int64)
    incidence[path[:, 0], path[:, 1]] = 1
    require(int(incidence.sum()) == len(path) and np.all(incidence.sum(axis=0)) and
            np.all(incidence.sum(axis=1)), label + ": all original frames, no duplicated pair")
    check(path[0].tolist() == [0, 0] and path[-1].tolist() == [n - 1, m - 1] and
          np.all(np.diff(path, axis=0) >= 0), label + ": monotone full endpoints")
    row_count, qgroups = Counter(), {}
    for group, axes in original["groups"].items():
        for axis, other_axis, matrix in (("query", "reference", incidence), ("reference", "query", incidence.T)):
            cache, counts = verify_axis(annotation["groups"][group][axis], axes[axis], axes[other_axis],
                                         matrix, group, axis, catalog, label + "/" + group + "/" + axis)
            row_count.update(counts)
            if axis == "query":
                qgroups[group] = cache
    check(annotation["possible"] == row_count["possible"], label + ": annotation total")
    query_cache[task["query"], task["seed"]][task["reference"]] = qgroups
    context_rows[task["query"], task["reference"], task["seed"]] = dict(row_count)
    annotation_paths[task["query"], task["reference"], task["seed"]] = item["path"]


def verify_availability(key, tasks, item, frozen_sha):
    query, seed = key
    label = f"availability {query}/{seed}"
    pin(item["path"], item["sha256"])
    out = read(item["path"])
    saved = out["availability"]
    source_by_ref = {r["reference"]: r for r in tasks}
    ids = sorted(source_by_ref)
    check(out["query"] == query and out["seed"] == seed and saved["query_id"] == query and saved["seed"] == seed,
          label + ": identity")
    check(out["frozen_sha256"] == frozen_sha and out["source_records"] == [r["path"] for r in tasks],
          label + ": exact complete source inventory/frozen identity")
    check(saved["reference_ids"] == ids and len(ids) in (2, 3), label + ": all expected distinct references")
    check(saved["source_records"] == [dict(reference_id=rid, source_id=source_by_ref[rid]["path"],
          annotation_id=annotation_paths[query, rid, seed]) for rid in ids], label + ": every source retained")
    flags(saved, label + ": no whole-sign verdict")
    check(saved["coverage_only_not_value_agreement"] is True and saved["residuals_aggregated"] is False and
          saved["reference_winner"] is None and saved["original_unsupported_evidence_retained"] is True,
          label + ": no pooling/selection/missing recovery")
    by_ref, row = query_cache[key], Counter()
    base = by_ref[ids[0]]
    require(set(saved["groups"]) == set(base), label + ": groups complete")
    per_reference = {rid: Counter() for rid in ids}
    for group, first in base.items():
        g = saved["groups"][group]
        n, dims = first["shape"]
        arrays = [by_ref[rid][group] for rid in ids]
        for rid, current in zip(ids, arrays):
            check(all(current[f] == first[f] for f in ("shape", "frame_id", "timestamp_us", "dimension_labels")) and
                  np.array_equal(current["own"], first["own"]), label + "/" + group + ": query identity/mask " + rid)
            states = current["state"]
            per_reference[rid].update(dict(query_possible=n * dims, reference_possible=current["reference_possible"],
                direct_complete=int((states == "DIRECT_COMPLETE").sum()), partial_shared=int((states == "PARTIAL_SHARED").sum()),
                unsupported=int((states == "NO_SHARED").sum())))
        direct = np.stack([a["state"] == "DIRECT_COMPLETE" for a in arrays])
        partial = np.stack([a["state"] == "PARTIAL_SHARED" for a in arrays])
        unsupported = ~(direct | partial)
        any_direct = direct.any(axis=0)
        partial_only = ~any_direct & partial.any(axis=0)
        none = ~(any_direct | partial_only)
        covered_elsewhere = any_direct & unsupported.any(axis=0)
        expected_counts = dict(possible=n * dims, own_observed=int(first["own"].sum()), own_unobserved=int((~first["own"]).sum()),
            any_direct=int(any_direct.sum()), partial_only=int(partial_only.sum()), all_unsupported=int(none.sum()),
            covered_despite_other_reference_missing=int(covered_elsewhere.sum()))
        counter_equal(g["counts"], expected_counts, label + "/" + group + ": unique-query counts")
        check(g["query_shape"] == [n, dims] and g["possible"] == n * dims, label + "/" + group + ": fixed denominator")
        for field, value in g["source_fields"].items():
            check(field in ("frame_id", "timestamp_us", "dimension_labels", "own_observed") and
                  value == pointer(group, "query", field), label + "/" + group + ": query identity pointer")
        by_dimension = defaultdict(list)
        for run in g["runs"]:
            by_dimension[run["dimension"]].append(run)
        require(set(by_dimension) == set(range(dims)), label + "/" + group + ": every dimension")
        for dim, runs in by_dimension.items():
            require(runs[0]["start"] == 0 and runs[-1]["end"] == n - 1 and
                    all(left["end"] + 1 == right["start"] for left, right in zip(runs, runs[1:])),
                    label + f"/{group}/{dim}: contiguous full frame partition")
            previous_signature = None
            for run in runs:
                start, end = run["start"], run["end"]
                require(0 <= start <= end < n, label + "/" + group + ": valid bounds")
                check(run["cell_count"] == end - start + 1 and run["dimension_label"] == first["dimension_labels"][dim],
                      label + "/" + group + ": native dimension/count")
                direct_ids = [rid for rid, a in zip(ids, arrays) if a["state"][start, dim] == "DIRECT_COMPLETE"]
                partial_ids = [rid for rid, a in zip(ids, arrays) if a["state"][start, dim] == "PARTIAL_SHARED"]
                missing_ids = [dict(reference_id=rid, reason=a["reasons"][start, dim]) for rid, a in zip(ids, arrays)
                               if a["state"][start, dim] == "NO_SHARED"]
                signature = (tuple(direct_ids), tuple(partial_ids), tuple((x["reference_id"], x["reason"]) for x in missing_ids))
                check(run["direct_refs"] == direct_ids and run["partial_refs"] == partial_ids and
                      run["unsupported_refs"] == missing_ids and
                      run["covered_despite_other_reference_missing"] == bool(direct_ids and missing_ids),
                      label + "/" + group + ": all reference statuses/reasons")
                for rid, current in zip(ids, arrays):
                    expected_state = current["state"][start, dim]
                    check(np.all(current["state"][start:end + 1, dim] == expected_state) and
                          (expected_state != "NO_SHARED" or np.all(current["reasons"][start:end + 1, dim] == current["reasons"][start, dim])),
                          label + "/" + group + ": every compressed cell exact " + rid)
                check(signature != previous_signature, label + "/" + group + ": maximal runs")
                previous_signature = signature
        check(sum(r["cell_count"] for r in g["runs"]) == n * dims, label + "/" + group + ": no loss")
        row.update(dict(possible=n * dims, at_least_one_direct_reference=expected_counts["any_direct"],
                        partial_references_only=expected_counts["partial_only"], no_shared_reference=expected_counts["all_unsupported"],
                        direct_reference_with_other_unsupported=expected_counts["covered_despite_other_reference_missing"]))
    for rid in ids:
        counter_equal(saved["per_reference_counts"][rid], per_reference[rid], label + ": per-reference denominator " + rid)
    check(saved["unique_query_possible"] == row["possible"], label + ": unique query total")
    availability_totals.update(row)
    availability_rows[key] = dict(row)


def static_scope_review():
    tree = ast.parse((UNIT / "context_evidence.py").read_text(encoding="utf-8-sig"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    check(imports <= {"collections", "math"}, "Context module imports only local arithmetic utilities")
    check(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
                  n.func.id in {"open", "eval", "exec", "compile", "__import__"} for n in ast.walk(tree)),
          "Context module has no dynamic code/file/model dispatch")
    driver = (UNIT / "run.py").read_text(encoding="utf-8-sig")
    check("context.annotate(source['evidence'],source['path'],reference)" in driver and
          "reference=copy.deepcopy(refs[rec['reference']])" in driver and
          "reference['role']=='fit'" in driver and "refs[query]" not in driver and "refs[rec['query']]" not in driver,
          "Driver supplies saved evidence/path and only FIT reference metadata, not query truth")
    check("context.availability(availability_inputs)" in driver, "Saved support-listing API only")


def main():
    for name in ("FROZEN.json", "TASKS.json", "RESULT_MANIFEST.json", "EVALUATION.json"):
        require((UNIT / name).is_file(), "Completed output required: " + name)
    frozen = read(UNIT / "FROZEN.json")
    frozen_sha = pin(UNIT / "FROZEN.json")
    for section in ("pins", "protected"):
        for path, expected in frozen[section].items():
            pin(path, expected)
    plan = read(HERE / "EVALUATION_PLAN.json")
    for section in ("pins", "source_protected"):
        for path, expected in plan[section].items():
            pin(path, expected)
    tasks, manifest = read(UNIT / "TASKS.json"), read(UNIT / "RESULT_MANIFEST.json")
    pin(UNIT / "RESULT_MANIFEST.json")
    pin(UNIT / "EVALUATION.json")
    check(all(manifest[k] == 0 for k in ("new_alignment", "new_scores", "new_model_inference", "new_training")),
          "No new paths/model/training/performance scores")
    require(len(tasks["records"]) == len(manifest["contexts"]) == 108 and len(manifest["availability"]) == 48,
            "Complete108 contexts/48 availability outputs")
    key = lambda r: (r["query"], r["reference"], r["seed"])
    expected = {key(r): r for r in tasks["records"]}
    actual = {key(r): r for r in manifest["contexts"]}
    require(len(expected) == len(actual) == 108 and set(expected) == set(actual), "Exact context inventory")
    grouped = defaultdict(list)
    for r in tasks["records"]:
        grouped[r["query"], r["seed"]].append(r)
    av_actual = {(r["query"], r["seed"]): r for r in manifest["availability"]}
    require(len(grouped) == len(av_actual) == 48 and set(grouped) == set(av_actual), "Exact availability inventory")
    check(Counter(len(v) for v in grouped.values()) == {2: 36, 3: 12}, "All expected2/3 references retained")
    refs_path = OPS / "word100_target_known_feedback_20261101/REFERENCE_PHASES.json"
    refs = read(refs_path)["clips"]
    pin(refs_path)
    static_scope_review()
    for index, (task_key, task) in enumerate(expected.items()):
        verify_context(task, actual[task_key], refs, frozen_sha)
        if (index + 1) % 18 == 0:
            print(f"Stored contexts verified {index + 1}/108", flush=True)
    require(len(query_cache) == 48, "All query/seed sources present")
    for group_key, records in grouped.items():
        verify_availability(group_key, records, av_actual[group_key], frozen_sha)
    counter_equal(old_counts, dict(possible=1926120, shared=736800, partial=3963,
                  own_unobserved=1127556, own_observed_without_shared=61764), "Original full opportunity partition")
    check(totals["possible"] == 1926120 and totals["direct_complete_cells"] == 732837 and
          totals["needs_context_cells"] == 1193283, "Non-DIRECT includes3963 partial cells without promotion")
    check(availability_totals["possible"] == 437640 and
          sum(availability_totals[k] for k in ("at_least_one_direct_reference", "partial_references_only", "no_shared_reference")) == 437640,
          "Unique-query denominator separate from original pairwise two-axis denominator")
    evaluation = read(UNIT / "EVALUATION.json")
    counter_equal(evaluation["total"], totals, "Published context aggregate")
    counter_equal(evaluation["temporal_context_cells"], temporal_totals, "Published temporal aggregate")
    counter_equal(evaluation["reference_context_cells"], reference_totals, "Published occurrence aggregate")
    counter_equal(evaluation["availability"], availability_totals, "Published availability aggregate")
    for axis in ("query", "reference"):
        counter_equal(evaluation["axis_counts"][axis], axis_totals[axis], "Published " + axis + " denominator")
    require(len(evaluation["contexts"]) == 108 and len(evaluation["availability_rows"]) == 48, "All published rows")
    check({(r["query"], r["reference"], r["seed"]): r["counts"] for r in evaluation["contexts"]} == context_rows,
          "All108 published context counts")
    check({(r["query"], r["seed"]): Counter(r["counts"]) for r in evaluation["availability_rows"]} ==
          {k: Counter(v) for k, v in availability_rows.items()}, "All48 published availability counts")
    decision = evaluation["decision"]
    check(decision["educational_correctness"] is None and decision["alignment_improved"] is False and
          decision["production_changed"] is False and all(decision[k] is False for k in
          ("new_alignment", "new_scores", "new_inference", "new_training")), "No performance/education/production adoption claim")
    check(all(decision[k] is True for k in ("raw_evidence_path_scope_unchanged", "no_interpolation", "no_query_truth_used",
                                           "observational_context_added")), "Additive-only decision")
    limits = " ".join(evaluation["limits"])
    check("not correct same-stage alignment" in limits and "not mandatory linguistic truth" in limits and
          "not annotated real educational accuracy" in limits, "DIRECT/BOUNDED/natural-variation limits")
    for path, expected_sha in list(audit_pins.items()):
        check(sha(path) == expected_sha, "Unchanged during independent verification: " + path)


if __name__ == "__main__":
    if OUTPUT.exists():
        raise SystemExit("Preserve first verification result; do not overwrite it.")
    error = None
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        failures.append("Verification exception; see exception field")
    report = dict(schema="wis_independent_saved_context_verification_v1", status="PASS" if not failures else "FAIL",
                  reviewer="Codex semantic_data_audit", at_utc=datetime.now(timezone.utc).isoformat(),
                  checks=checks, failure_count=len(failures), failures=failures[:100], exception=error,
                  source_scope=dict(context_records=108, unique_query_seed_groups=48, original_global_paths=36,
                                    original_axis_opportunities=1926120, unique_query_opportunities=437640),
                  original_counts=dict(old_counts), context_counts=dict(totals),
                  axis_counts={k: dict(v) for k, v in axis_totals.items()},
                  temporal_context_cells=dict(temporal_totals), reference_context_cells=dict(reference_totals),
                  availability_counts=dict(availability_totals),
                  method="Independent incidence-matrix mask arithmetic and saved run/anchor/source-pointer checks. No producer module imported or executed.",
                  limits=["Residual values are preserved by exact source SHA/digest and pointers, not recomputed.",
                          "DIRECT means shared observation, never same stage or correctness; BOUNDED does not determine hidden interiors.",
                          "FIT occurrences are draft physical-core metadata, not query or mandatory linguistic truth.",
                          "Natural crossing/lowering educational accuracy was not tested; only synthetic properties/source preservation."],
                  actions=dict(new_alignment=0, model_inference=0, training=0, performance_scores=0,
                               raw_media_access=0, private_or_mentor_access=0, old_artifact_changes=0),
                  pins=audit_pins, script_sha256=sha(__file__))
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=report["status"], checks=checks, failure_count=len(failures),
                         output=str(OUTPUT), sha256=sha(OUTPUT))))
    if failures:
        raise SystemExit(1)
