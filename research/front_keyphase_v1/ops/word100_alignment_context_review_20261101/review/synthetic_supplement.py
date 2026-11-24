"""Contract counterexamples using synthetic fixtures only; no public data.

The fixture builder is reused for serialization, but expected properties below
are independently specified. This never invokes the producer selftest.run().
"""
from pathlib import Path
from datetime import datetime, timezone
import copy
import hashlib
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import context_evidence as candidate
from selftest import fixture, clip, record


def run():
    results = []

    def check(case, condition, expectation):
        results.append(dict(case=case, passed=bool(condition), expectation=expectation))

    def rejects(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    def annotate(e, p):
        before = copy.deepcopy((e, p))
        out = candidate.annotate(e, p, clip())
        assert (e, p) == before
        assert out["semantic_verdict"] is None and not out["alignment_verified"]
        return out

    a, p = fixture([0, .8, 0, 0], [0, 0, 0, 0], rm=[True, False, False, True])
    b, _ = fixture([0, .8, 0, 0], [0, 0, 0, 0])
    records = [record(a, "FIT_A"), record(b, "FIT_B")]
    before = copy.deepcopy(records)
    x = candidate.availability(records)
    check("reference_order_permutation", x == candidate.availability(records[::-1]) and records == before,
          "Reference lists are order-independent; supplied source evidence is unchanged.")
    check("duplicate_reference", rejects(lambda: candidate.availability([records[0], records[0]])),
          "Duplicate references are rejected, not counted twice.")
    unknown, _ = fixture([0, .8, 0, 0], [0, 0, 0, 0], qm=[False] * 4)
    check("valid_query_mask_mismatch", rejects(lambda: candidate.availability([records[0], record(unknown, "FIT_B")])),
          "Even individually consistent inputs cannot be grouped when query own masks differ.")
    no_ref, _ = fixture([0, 1, 0], [0, 0, 0], rm=[False] * 3)
    x = candidate.availability([record(no_ref), record(no_ref, "FIT_B")])
    runs = x["groups"]["R_shape"]["runs"]
    check("all_references_unsupported", len(runs) == 1 and runs[0]["cell_count"] == 3 and
          not runs[0]["direct_refs"] and not runs[0]["partial_refs"] and len(runs[0]["unsupported_refs"]) == 2,
          "All three query cells remain unsupported; no available-evidence claim.")
    x = candidate.availability([record(unknown), record(unknown, "FIT_B")])
    check("query_unobserved_with_observed_references", x["unique_query_possible"] == 4 and
          all(not r["direct_refs"] and not r["partial_refs"] for r in x["groups"]["R_shape"]["runs"]),
          "Reference observations cannot supply a missing query observation.")
    partial, _ = fixture([1], [0, 0], rm=[True, False], path=[[0, 0], [0, 1]])
    none, _ = fixture([1], [0, 0], rm=[False, False], path=[[0, 0], [0, 1]])
    x = candidate.availability([record(partial), record(none, "FIT_B")])
    r = x["groups"]["R_shape"]["runs"][0]
    check("partial_only_reference", r["partial_refs"] == ["FIT_A"] and not r["direct_refs"] and
          not r["covered_despite_other_reference_missing"] and partial["groups"]["R_shape"]["query"]["mean_residual"] == [[1.0]],
          "Partial support stays partial; its nonzero observed residual is retained.")
    x = annotate(a, p)["groups"]["R_shape"]["query"]
    r = x["episodes"][0]
    check("two_cell_bounded_gap", r["start"] == 1 and r["end"] == 2 and r["cell_count"] == 2 and
          r["anchor_before"]["index"] == 0 and r["anchor_after"]["index"] == 3 and
          a["groups"]["R_shape"]["query"]["mean_residual"][1:3] == [[None], [None]],
          "Both missing cells remain null between strict nearest direct anchors.")
    hidden1, hp = fixture([0, 0, 0, 0], [0, 0, 0, 0], rm=[True, False, False, True])
    hidden2, _ = fixture([0, 0, 0, 0], [0, 999, -999, 0], rm=[True, False, False, True])
    check("equal_ends_hidden_change_invariance", hidden1 == hidden2 and annotate(hidden1, hp) == annotate(hidden2, hp),
          "Masked counterfactual changes do not produce an inferred hidden value or normality claim.")
    peak, pp = fixture([0, .9, 0, 0], [0, 0, 0, 0], rm=[True, True, False, True])
    x = annotate(peak, pp)["groups"]["R_shape"]["query"]
    check("visible_peak_next_to_gap", peak["groups"]["R_shape"]["query"]["max_residual"][1] == [.9] and
          x["episodes"][0]["anchor_before"]["index"] == 1 and x["episodes"][0]["start"] == 2,
          "A nonzero measured peak remains direct, is not absorbed into a gap, and is not called correct by DIRECT.")
    left, cp = fixture([0, 1, 2], [2, 1, 0], qm=[True, False, True], group="L_axis")
    right, _ = fixture([2, 1, 0], [0, 1, 2], group="R_axis")
    crossing = {"groups": {**left["groups"], **right["groups"]}}
    x = annotate(crossing, cp)
    check("crossing_with_one_side_gap", set(x["groups"]) == {"L_axis", "R_axis"} and
          len(x["groups"]["L_axis"]["query"]["episodes"]) == 1 and
          not x["groups"]["R_axis"]["query"]["episodes"] and
          crossing["groups"]["R_axis"]["query"]["max_residual"] == [[1], [0], [1]],
          "Image crossing does not swap anatomical source groups or hide the opposite-side differences.")
    timed, tp = fixture([0, 0, 0], [0, 0, 0], qm=[True, False, True],
                        qpts=[10, 23, 1200], rpts=[5, 111, 901], qframes=[8, 10, 19], rframes=[3, 7, 14])
    x = annotate(timed, tp)["groups"]["R_shape"]["query"]["episodes"][0]
    check("native_nonuniform_time_anchors", x["pts_range"] == [23, 23] and
          x["anchor_before"]["timestamp_us"] == 10 and x["anchor_after"]["timestamp_us"] == 1200 and
          x["anchor_after"]["shared_counterpart_timestamp_us"] == [901] and
          x["anchor_after"]["shared_counterpart_frame_ids"] == [14],
          "Anchors use original PTS and source frame IDs, with no uniform-time replacement.")
    swapped = {"groups": {g: {"query": copy.deepcopy(v["reference"]), "reference": copy.deepcopy(v["query"])}
                          for g, v in timed["groups"].items()}}
    normal = annotate(timed, tp)["groups"]["R_shape"]
    reverse = annotate(swapped, [[j, i] for i, j in tp])["groups"]["R_shape"]
    fields = ("possible", "direct_complete_cells", "needs_context_cells", "status_counts", "missing_reason_counts")
    check("axis_reversal", all(normal[old][k] == reverse[new][k] for old, new in (("query", "reference"), ("reference", "query"))
                               for k in fields),
          "Both original-axis opportunity and missingness inventories swap with their sources.")
    injected = copy.deepcopy(timed)
    injected["query_ground_truth"] = {"stage": "deliberately incorrect", "normal": True, "word": "arbitrary"}
    check("query_truth_extras_invariance", annotate(timed, tp) == annotate(injected, tp),
          "Extra query truth labels cannot change context or authorize hidden normality.")
    bad = copy.deepcopy(timed)
    bad["groups"]["R_shape"]["query"]["timestamp_us"] = [10, 1200, 23]
    check("unordered_source_time", rejects(lambda: annotate(bad, tp)),
          "Unordered original timestamps are rejected rather than silently sorted.")
    return results


if __name__ == "__main__":
    destination = HERE / "SYNTHETIC_SUPPLEMENT.json"
    if destination.exists():
        raise SystemExit("Preserve the first result; choose a separately reviewed follow-up instead.")
    results = run()
    report = dict(schema="wis_independent_context_synthetic_supplement_v1",
                  status="PASS" if all(r["passed"] for r in results) else "FAIL",
                  at_utc=datetime.now(timezone.utc).isoformat(), reviewer="Codex semantic_data_audit",
                  cases=len(results), results=results,
                  incomplete_inventory="Checked against TASKS in the independent stored-result verifier; availability has no expected-manifest argument.",
                  claim_limit="Synthetic structural properties only; no natural crossing/lowering educational accuracy was evaluated.",
                  actions=dict(public_output_verification=0, public_data_access=0, new_alignment=0,
                               model_inference=0, training=0, performance_scoring=0),
                  pins={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (HERE.parent / "context_evidence.py", HERE.parent / "selftest.py", Path(__file__))})
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=report["status"], cases=report["cases"], output=str(destination),
                         sha256=hashlib.sha256(destination.read_bytes()).hexdigest())))
    if report["status"] != "PASS":
        raise SystemExit(1)
