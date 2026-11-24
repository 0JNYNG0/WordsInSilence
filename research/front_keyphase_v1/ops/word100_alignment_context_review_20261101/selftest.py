"""Small synthetic fixtures only; no datasets, alignment, models, or file writes.

Root executes run() and stores the returned JSON. CLI prints that same result.
"""
import copy
import json
import context_evidence as C


def fixture(q, r, qm=None, rm=None, path=None, qpts=None, rpts=None,
            qframes=None, rframes=None, labels=None, group="R_shape"):
    """Independent small raw-evidence builder, not an alignment solver."""
    q = [[x] if isinstance(x, (int, float)) else list(x) for x in q]
    r = [[x] if isinstance(x, (int, float)) else list(x) for x in r]
    n, m, d = len(q), len(r), len(q[0])
    def masks(v, size):
        return [[True] * d for _ in range(size)] if v is None else [
            [x] * d if type(x) is bool else list(x) for x in v]
    qm, rm = masks(qm, n), masks(rm, m)
    path = [[i, i] for i in range(n)] if path is None else path
    labels = labels or ["feature_" + str(i) for i in range(d)]
    def axis(values, own, other_values, other, pairs, frames, pts):
        size = len(values)
        ans = dict(possible_shape=[size, d], dimension_labels=labels[:],
                   frame_id=list(range(size)) if frames is None else frames[:],
                   timestamp_us=[i * 10000 for i in range(size)] if pts is None else pts[:],
                   own_observed=copy.deepcopy(own), matched_support=[], mixed_counterpart_missing=[],
                   mean_residual=[], max_residual=[], cell_status=[], cell_reason=[],
                   counts={"possible": size * d})
        for i in range(size):
            linked = [j for oi, j in pairs if oi == i]
            rows = {k: [] for k in C.MATRICES if k != "own_observed"}
            for dim in range(d):
                shared = [min(abs(values[i][dim] - other_values[j][dim]), 1)
                          for j in linked if own[i][dim] and other[j][dim]]
                mixed = bool(shared) and any(not other[j][dim] for j in linked)
                rows["matched_support"].append(bool(shared))
                rows["mixed_counterpart_missing"].append(mixed)
                rows["mean_residual"].append(sum(shared) / len(shared) if shared else None)
                rows["max_residual"].append(max(shared) if shared else None)
                rows["cell_status"].append("PARTIAL_SHARED_SUPPORT" if mixed else "COMPARED" if shared else "NOT_COMPARABLE")
                rows["cell_reason"].append("SHARED_AND_MISSING_COUNTERPART_LINKS" if mixed else
                    "SHARED_OBSERVATION" if shared else "OWN_UNOBSERVED" if not own[i][dim] else
                    "COUNTERPART_UNOBSERVED" if linked else "NO_ALIGNED_COUNTERPART")
            for key, row in rows.items():
                ans[key].append(row)
        return ans
    evidence = {"groups": {group: {
        "query": axis(q, qm, r, rm, path, qframes, qpts),
        "reference": axis(r, rm, q, qm, [[j, i] for i, j in path], rframes, rpts)}}}
    return evidence, copy.deepcopy(path)


def clip(regions=None, alias="FIT_A"):
    return dict(alias=alias, role="fit", regions=regions or [
        dict(id="occurrence_A", region_type="observed_physical_core", kind="same_kind",
             core=[0, 100], outer=[0, 100], description_ko="합성 참고 구간")])


def record(evidence, reference_id="FIT_A", query_id="QUERY", seed=42):
    return dict(reference_id=reference_id, evidence=evidence, source_id="synthetic:" + reference_id,
                annotation_id="synthetic-context:" + reference_id, query_id=query_id, seed=seed)


def run():
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)
    def rejects(name, fn):
        try:
            fn()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    def context(e, p, c=None):
        before = copy.deepcopy((e, p, c))
        out = C.annotate(e, p, clip() if c is None else c)
        check("input_evidence_path_reference_unchanged_" + str(len(checks)), before == (e, p, c))
        check("no_values_filled_or_sign_verdict_" + str(len(checks)), out["verdict"] is None and
              out["correction_advice"] is None and not out["alignment_verified"] and not out["missing_values_imputed"])
        for axes in out["groups"].values():
            for a in axes.values():
                check("all_original_axis_opportunities_" + str(len(checks)), a["possible"] ==
                      a["direct_complete_cells"] + sum(x["cell_count"] for x in a["episodes"]))
        return out
    # Matching endpoints do not recover an unobserved intermediate change.
    e, p = fixture([0, 1, 0, 0], [0, 0, 0, 0], rm=[True, False, True, True])
    out = context(e, p)
    episode = out["groups"]["R_shape"]["query"]["episodes"][0]
    check("hidden_change_remains_null", e["groups"]["R_shape"]["query"]["mean_residual"][1][0] is None)
    check("bounded_is_only_context", episode["temporal_context"] == "BOUNDED_BY_DISTINCT_OBSERVATIONS"
          and episode["reference_context"] == "SAME_REFERENCE_OCCURRENCE_BRACKET"
          and episode["missing_reason_counts"] == {"COUNTERPART_UNOBSERVED": 1})
    check("original_residual_pointer", out["groups"]["R_shape"]["query"]["source_fields"]["mean_residual"] ==
          "/groups/R_shape/query/mean_residual")
    # Partial evidence is kept, but is not a complete context anchor.
    e, p = fixture([0, .5, 0, 0], [0, 0, 0, 0, 0], qm=[True, True, False, True],
                   rm=[True, True, False, True, True], path=[[0, 0], [1, 1], [1, 2], [2, 3], [3, 4]])
    out = context(e, p)
    episode = out["groups"]["R_shape"]["query"]["episodes"][0]
    check("partial_residual_preserved", e["groups"]["R_shape"]["query"]["mean_residual"][1][0] == .5)
    check("partial_not_promoted_to_anchor", episode["start"] == 1 and episode["end"] == 2
          and episode["anchor_before"]["index"] == 0 and episode["anchor_after"]["index"] == 3
          and episode["status_counts"] == {"PARTIAL_SHARED": 1, "NO_SHARED": 1})
    check("own_missing_counterpart_observed_not_both_unknown", episode["link_provenance_counts"][
          "own_missing_counterpart_observed_links"] == 1 and episode["link_provenance_counts"]["both_missing_links"] == 0)
    # Open normal lowering has no error or correction inference.
    e, p = fixture([0, .5, 1], [0, .5, 1], qm=[True, True, False])
    out = context(e, p)
    check("open_hand_lowering_only_open_context", out["groups"]["R_shape"]["query"]["episodes"][0]["temporal_context"] == "OPEN_BOUNDARY"
          and out["correction_advice"] is None)
    # Independent L/R source features must never be reassigned at a crossing.
    e, p = fixture([[0, 1], [1, 0]], [[1, 0], [0, 1]], labels=["L_axis_x", "R_axis_x"], group="crossing")
    out = context(e, p)
    check("crossing_preserves_side_and_difference", e["groups"]["crossing"]["query"]["max_residual"] == [[1, 1], [1, 1]]
          and out["groups"]["crossing"]["query"]["direct_complete_cells"] == 4)
    # Fixed normal time warp uses actual PTS, without time/value interpolation.
    e, p = fixture([0, 0, 1, 1], [0, 1], path=[[0, 0], [1, 0], [2, 1], [3, 1]],
                   qpts=[0, 12000, 80000, 150000], rpts=[0, 50000])
    out = context(e, p)
    check("timewarp_keeps_direct_values_and_all_axes", out["groups"]["R_shape"]["query"]["direct_complete_cells"] == 4
          and out["groups"]["R_shape"]["reference"]["direct_complete_cells"] == 2
          and e["groups"]["R_shape"]["query"]["mean_residual"] == [[0], [0], [0], [0]])
    # Identity is the child occurrence ID, not its kind.
    c = clip([dict(id="a", region_type="observed_physical_core", kind="same", core=[10, 10], outer=[9, 11]),
              dict(id="b", region_type="observed_physical_core", kind="same", core=[12, 12], outer=[11, 13])])
    e, p = fixture([0, 0, 0], [0, 0, 0], qm=[True, False, True], rframes=[10, 11, 12])
    out = context(e, p, c)
    check("same_kind_distinct_ids_not_merged", out["groups"]["R_shape"]["query"]["episodes"][0]["reference_context"] == "CROSS_REFERENCE_OCCURRENCES")
    # Outer brackets do not widen the certain observed core.
    c["regions"][0]["core"] = [9, 9]
    out = context(e, p, c)
    check("outer_only_reference_is_uncertain", out["groups"]["R_shape"]["query"]["episodes"][0]["reference_context"] == "MIXED_OR_UNCERTAIN_REFERENCE_SCOPE")
    # Complete anchor counterpart sets can overlap even with different minima.
    e, p = fixture([0, 0, 0], [0, 0, 0], qm=[True, False, True],
                   path=[[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]])
    out = context(e, p)
    episode = out["groups"]["R_shape"]["query"]["episodes"][0]
    check("overlap_is_not_bounded", episode["temporal_context"] == "MAPPING_COMPRESSED_OR_OVERLAPPING"
          and episode["anchor_before"]["shared_counterpart_indices"] == [0, 1]
          and episode["anchor_after"]["shared_counterpart_indices"] == [1, 2])
    e, p = fixture([0, 1, 0], [0], qm=[True, False, True], path=[[0, 0], [1, 0], [2, 0]])
    out = context(e, p)
    check("same_counterpart_frame_is_compressed", out["groups"]["R_shape"]["query"]["episodes"][0]["temporal_context"] == "MAPPING_COMPRESSED_OR_OVERLAPPING")
    e, p = fixture([0, 1], [0, 1], qm=[False, False], rm=[False, True])
    out = context(e, p)
    q = out["groups"]["R_shape"]["query"]
    check("no_anchors_no_recovery", q["episodes"][0]["temporal_context"] == "NO_ANCHORS"
          and q["link_provenance_counts"]["both_missing_links"] == 1
          and q["link_provenance_counts"]["own_missing_counterpart_observed_links"] == 1)
    # Availability exposes another direct reference without erasing unknowns.
    missing, _ = fixture([0, 1, 0], [0, 0, 0], rm=[True, False, True])
    direct, _ = fixture([0, 1, 0], [0, 0, 0])
    records = [record(missing, "FIT_A"), record(direct, "FIT_B")]
    saved = copy.deepcopy(records)
    av = C.availability(records)
    run1 = next(x for x in av["groups"]["R_shape"]["runs"] if x["start"] == 1)
    check("availability_does_not_mutate_or_impute", saved == records and missing["groups"]["R_shape"]["query"]["mean_residual"][1][0] is None)
    check("other_reference_local_support_listed", run1["direct_refs"] == ["FIT_B"] and run1["unsupported_refs"] ==
          [{"reference_id": "FIT_A", "reason": "COUNTERPART_UNOBSERVED"}]
          and run1["covered_despite_other_reference_missing"])
    check("unique_and_per_reference_denominators_distinct", av["unique_query_possible"] == 3 and
          sum(x["query_possible"] for x in av["per_reference_counts"].values()) == 6)
    check("available_large_difference_not_overridden", direct["groups"]["R_shape"]["query"]["max_residual"][1][0] == 1
          and av["reference_winner"] is None and av["verdict"] is None)
    bad = copy.deepcopy(records)
    bad[1]["evidence"]["groups"]["R_shape"]["query"]["own_observed"][1][0] = False
    rejects("query_own_mask_mismatch_rejected", lambda: C.availability(bad))
    for field in ("frame_id", "timestamp_us", "dimension_labels"):
        bad = copy.deepcopy(records)
        a = bad[1]["evidence"]["groups"]["R_shape"]["query"]
        a[field] = [v + 10 for v in a[field]] if field != "dimension_labels" else ["different_feature"]
        rejects("query_identity_mismatch_" + field, lambda: C.availability(bad))
    rejects("different_query_rejected", lambda: C.availability([record(missing), record(direct, "FIT_B", "OTHER_QUERY")]))
    rejects("different_seed_rejected", lambda: C.availability([record(missing), record(direct, "FIT_B", seed=43)]))
    c = clip(); c["role"] = "dev"
    rejects("non_fit_reference_scope_rejected", lambda: C.annotate(missing, [[i, i] for i in range(3)], c))
    return dict(status="PASS_SYNTHETIC_CONTEXT_ONLY", checks=len(checks), results=checks,
                public_data_executed=False, alignment_executed=False, inference_executed=False,
                semantic_accuracy_claim=None,
                limitations=["Context preserves unknown; it does not resolve correspondence or hidden changes.",
                             "Availability does not combine values or select a correct reference."])


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
