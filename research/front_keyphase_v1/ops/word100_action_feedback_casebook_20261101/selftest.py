"""Synthetic stored-record contract tests; no old producer/DTW/model is called."""
import sys
sys.dont_write_bytecode = True
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import case_evidence as module


def fixture(n=3, m=4, path=None, unknown=(), value=None):
    """Construct a small saved-readout fixture, not a physical motion simulation."""
    if path is None:
        path = [[0, 0], [1, 1], [1, 2], [2, 3]] if (n, m) == (3, 4) else [[i, i] for i in range(min(n, m))]
    unknown = set(unknown)
    value = value or (lambda group, k, d: 10.0 if k == 2 else float(d % 3))
    frames = dict(query=[101 + 4*i for i in range(n)], reference=[201 + 3*i for i in range(m)])
    pts = dict(query=[1000000 + 45000*i for i in range(n)], reference=[2000000 + 51000*i for i in range(m)])
    groups = {}
    for name, d in module.GROUP_DIMENSIONS.items():
        labels = [name + ":" + str(j) for j in range(d)]
        own = {axis: [[(name, axis, i, j) not in unknown for j in range(d)] for i in range(length)]
               for axis, length in (("query", n), ("reference", m))}
        shared = [[own["query"][qi][j] and own["reference"][ri][j] for j in range(d)] for qi, ri in path]
        delta = [[float(value(name, k, j)) if shared[k][j] else None for j in range(d)] for k in range(len(path))]
        group = dict(definition="Synthetic saved-readout fixture", source_columns=list(range(d)),
                     source_feature_names=labels, units="degrees", circular=not name.endswith("angles14"),
                     links=dict(shared_support=shared, signed_difference_deg=deepcopy(delta),
                                absolute_difference_deg=delta,
                                antipodal_signed_direction_ambiguous=[[False]*d for _ in path],
                                signed_difference_convention="query_minus_reference"))
        for axis, length, column in (("query", n, 0), ("reference", m, 1)):
            means, maxima, supported, mixed, status, reason = [], [], [], [], [], []
            aligned, matched, missing, counterpart_count = [], [], [], []
            for i in range(length):
                links = [k for k, pair in enumerate(path) if pair[column] == i]
                aligned.append(len(links))
                rows = [[], [], [], [], [], [], [], [], []]
                for j in range(d):
                    values = [delta[k][j] for k in links if shared[k][j]]
                    other = "reference" if axis == "query" else "query"
                    absent = sum(not own[other][path[k][1-column]][j] for k in links)
                    support = bool(values)
                    partial = support and bool(absent)
                    rows[0].append(sum(values)/len(values) if values else None)
                    rows[1].append(max(values) if values else None)
                    rows[2].append(support); rows[3].append(partial)
                    rows[4].append("PARTIAL_SHARED_SUPPORT" if partial else ("COMPARED" if support else "NOT_COMPARABLE"))
                    rows[5].append("SHARED_AND_MISSING_COUNTERPART_LINKS" if partial else
                                   ("SHARED_OBSERVATION" if support else ("OWN_UNOBSERVED" if not own[axis][i][j] else
                                    ("COUNTERPART_UNOBSERVED" if links else "NO_ALIGNED_COUNTERPART"))))
                    rows[6].append(len(values)); rows[7].append(absent); rows[8].append(len(links)-absent)
                for target, row in zip((means, maxima, supported, mixed, status, reason, matched, missing, counterpart_count), rows):
                    target.append(row)
            counts = dict(possible=length*d, own_observed=sum(sum(row) for row in own[axis]),
                          own_unobserved=sum(not v for row in own[axis] for v in row),
                          matched_support=sum(sum(row) for row in supported),
                          direct_complete=sum(s and not t for sr, tr in zip(supported, mixed) for s, t in zip(sr, tr)),
                          partial_shared=sum(sum(row) for row in mixed),
                          not_comparable=sum(not v for row in supported for v in row),
                          observed_without_shared_counterpart=sum(o and not s for ar, sr in zip(own[axis], supported) for o, s in zip(ar, sr)))
            own_reason = [["OBSERVED" if v else "SOURCE_UNOBSERVED" for v in row] for row in own[axis]]
            group[axis] = dict(possible_shape=[length, d], dimension_labels=labels,
                frame_id=frames[axis], timestamp_us=pts[axis], values_degrees=[[20.0 if o else None for o in row] for row in own[axis]],
                own_observed=own[axis], own_reason=own_reason, original_source_mask=deepcopy(own[axis]),
                matched_support=supported, mixed_counterpart_missing=mixed, mean_abs_deg=means, max_abs_deg=maxima,
                aligned_link_count=aligned, matched_link_count=matched, missing_counterpart_link_count=missing,
                counterpart_observed_link_count=counterpart_count, cell_status=status, cell_reason=reason,
                counts=counts, own_reason_counts=dict(Counter(v for row in own_reason for v in row)),
                unaligned_frame_indices=[i for i, a in enumerate(aligned) if not a], diagnostics={})
        summaries = []
        for j in range(d):
            values = [row[j] for row in delta if row[j] is not None]
            maximum = max(values) if values else None
            summaries.append(dict(dimension=j, label=labels[j], possible_links=len(path), shared_links=len(values),
                                  mean_abs_deg=sum(values)/len(values) if values else None, max_abs_deg=maximum,
                                  peak_link_indices=[k for k in range(len(path)) if shared[k][j] and delta[k][j] == maximum]))
        group["summary_by_dimension"] = summaries
        groups[name] = group
    return dict(id="SYNTHETIC_FIXTURE", kind="natural", query="SYNTHETIC_Q", reference="SYNTHETIC_R", parent_id="SYNTHETIC_PARENT",
                query_range=[0, max(0, n-1)], reference_range=[0, max(0, m-1)], technical_edit=None,
                evidence=dict(schema="wis_independent_hand_evidence_v1", groups=groups, path=path,
                              path_full_scope={i for i, _ in path} == set(range(n)) and {j for _, j in path} == set(range(m)),
                              limits=["Synthetic saved-readout fixture only"]))


def provenance():
    return dict(path="SYNTHETIC_ONLY.json", sha256="a"*64, json_pointer="/fixture")


def run():
    checks, tests = 0, []
    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(label)
    def reject(call, label):
        try:
            call()
        except ValueError:
            check(True, label)
            return
        check(False, label)
    name = "L_finger_angles14"
    record = fixture(unknown={(name, "reference", 2, 0), (name, "query", 2, 1)})
    before = deepcopy(record)
    out = module.summarize(record)
    check(record == before, "Summary must not mutate source")
    check(out["visible_difference_review"]["status"] == "NOT_REVIEWED", "No invented RGB review")
    for group_name, original in record["evidence"]["groups"].items():
        result = out["groups"][group_name]
        for axis in ("query", "reference"):
            check(result[axis]["counts"] == original[axis]["counts"], "Both-axis denominators preserved")
            for dim in result[axis]["dimensions"]:
                buckets = dim["frame_indices_by_support"]
                check(sorted(buckets["direct_complete"]+buckets["partial_shared"]+buckets["not_comparable"]) == list(range(dim["possible"])), "Every original frame opportunity retained")
        for saved, summary in zip(original["summary_by_dimension"], result["summary_by_dimension"]):
            check(all(summary[key] == value for key, value in saved.items()), "Stored mean/max and all peaks unchanged")
            if saved["peak_link_indices"]:
                peak = summary["first_peak"]
                qi, ri = record["evidence"]["path"][saved["peak_link_indices"][0]]
                check(peak["query"]["frame_id"] == original["query"]["frame_id"][qi] and
                      peak["reference"]["timestamp_us"] == original["reference"]["timestamp_us"][ri], "Original peak frame and PTS preserved")
    partial = out["groups"][name]["query"]["dimensions"][0]
    check(partial["frame_indices_by_support"]["partial_shared"] == [1], "Partial support retained")
    check(out["groups"][name]["query"]["dimensions"][1]["frame_indices_by_support"]["own_unobserved"] == [2], "Single missing angle remains missing")
    check(out["groups"][name]["query"]["dimensions"][2]["counts"]["direct_complete"] == 3, "Other angles remain independently available")
    tests.append("All six groups, both axes, fourteen independent angles, original peak times, partial and missing opportunities")

    meta = deepcopy(record)
    meta.update(position_sources=[{"distance":0}], shape_sources=[{"open_probability":0,"index_probability":1}],
                word_hint="TARGET_IS_CORRECT", arbitrary_stage_truth="GOOD")
    check(module.summarize(meta) == out, "Position zero, posterior, word and stage metadata cannot suppress other features")
    desc = dict(source=provenance(), cards=[dict(source_status="unverified_edge_on", source_links=[{"status":"visible_proximity_only"}],
                                              text="REFERENCE EXPECTATION ONLY")],
                legacy_decisions={"old_location":{"saved_range":[0,1],"saved_decision":"within_small_reference_range","feedback":{"educational_correct":None}}})
    corr = dict(source=provenance(), details={"status":"ambiguous_occurrence","flags":["order_unresolved"]},
                actual_stage_correctness_verified=False)
    joined = module.summarize(record, desc, corr)
    check(joined["groups"] == out["groups"], "Description and correspondence uncertainty do not gate numeric support")
    check(joined["description_support"]["provided"] == desc and joined["correspondence_support"]["provided"] == corr, "Source qualifiers and old decisions preserved separately")
    check(joined["description_support"]["source_status_counts"] == {"unverified_edge_on":1}, "Uncertain source status not promoted")
    check(joined["linguistic_importance"]["status"] == "UNESTABLISHED" and joined["semantic_verdict"] is None and
          joined["whole_sign_verdict"] is None and joined["correction_advice"] is None, "No educational or important-feature verdict")
    tests.append("Position/posterior/hint independence, legacy decision separation and correspondence ambiguity")

    visual = dict(status="REVIEWED", source=provenance(), windows=[{"frame_range":[105,109]}],
                  observations=[{"source":provenance(),"review":{"visible_difference":"scoped observation only"}}])
    frozen_inputs = deepcopy((record, desc, corr, visual))
    reviewed = module.summarize(record, desc, corr, visual)
    check(reviewed["visible_difference_review"]["provided"] == visual and reviewed["groups"] == out["groups"], "Scoped visual review is not numerical feedback")
    check((record,desc,corr,visual) == frozen_inputs, "No input wrapper mutation")
    reviewed["description_support"]["provided"]["cards"].clear()
    check(desc["cards"] == frozen_inputs[1]["cards"], "Output mutations do not alter source objects")
    reject(lambda: module.summarize(record, visual_review={"status":"REVIEWED","windows":[{}],"observations":[{}]}), "Unsourced visual claim rejected")
    reject(lambda: module.summarize(record, visual_review=dict(status="REVIEWED",source=provenance(),windows=[],observations=[])), "Unscoped empty review rejected")
    tests.append("NOT_REVIEWED default, sourced window-only RGB review, deep-copy immutability")

    technical = deepcopy(record)
    technical.update(kind="technical", technical_edit={"condition":"short_axis_change","feature_space_only_not_physical_video":True})
    synth = module.summarize(technical)
    check(synth["synthetic"] and synth["evidence_role"] == "SYNTHETIC_FEATURE_EDIT", "Technical edits explicitly synthetic")
    reject(lambda: module.summarize(technical, visual_review=visual), "Feature edit is not a performed RGB video")
    tests.append("Synthetic feature edits cannot acquire real performed-video review")

    zero = module.summarize(fixture(value=lambda g,k,d:0))
    check(all(s["peak_link_indices"] == [0,1,2,3] for g in zero["groups"].values() for s in g["summary_by_dimension"]), "Every zero-valued tied peak retained")
    check(zero["semantic_verdict"] is None and not zero["measurement_difference"]["zero_is_correct_performance"], "Zero is not correct performance")
    unknown = {(g,a,i,j) for g,d in module.GROUP_DIMENSIONS.items() for a,l in (("query",3),("reference",4)) for i in range(l) for j in range(d)}
    no = module.summarize(fixture(unknown=unknown))
    check(all(s["mean_abs_deg"] is None and s["max_abs_deg"] is None and s["first_peak"] is None for g in no["groups"].values() for s in g["summary_by_dimension"]), "No evidence stays null")
    check(sum(g["query"]["counts"]["possible"]+g["reference"]["counts"]["possible"] for g in no["groups"].values()) == 224, "All-unknown denominator retained")
    empty = module.summarize(fixture(0,0,[]))
    check(all(a["counts"]["possible"] == 0 for g in empty["groups"].values() for a in (g["query"],g["reference"])), "Empty source handled without invented opportunities")
    short = module.summarize(fixture(5,5,value=lambda g,k,d:18 if g == name and d == 2 and k == 2 else 0))
    short_stat = short["groups"][name]["summary_by_dimension"][2]
    check(short_stat["max_abs_deg"] == 18 and short_stat["mean_abs_deg"] == 3.6 and short_stat["first_peak"]["query"]["frame_id"] == 109, "Short supported change remains despite diluted mean")
    check(short["groups"]["R_finger_angles14"]["summary_by_dimension"][2]["max_abs_deg"] == 0, "Left/right identity never reassigned")
    tests.append("Null versus zero, complete tied peaks, empty denominators, short-change maxima and stable hand identity")

    bad = deepcopy(record); bad["evidence"]["groups"][name]["query"]["counts"]["possible"] -= 1
    reject(lambda: module.summarize(bad), "Shrunk denominator rejected")
    bad = fixture(unknown=unknown); bad["evidence"]["groups"][name]["query"]["mean_abs_deg"][0][0] = 0
    reject(lambda: module.summarize(bad), "Unsupported zero rejected")
    bad = fixture(value=lambda g,k,d:0); bad["evidence"]["groups"][name]["summary_by_dimension"][0]["peak_link_indices"] = [0]
    reject(lambda: module.summarize(bad), "Dropped tied peaks rejected")
    bad = deepcopy(record)
    bad["evidence"]["groups"][name]["query"]["frame_id"] = list(bad["evidence"]["groups"][name]["query"]["frame_id"])
    bad["evidence"]["groups"][name]["query"]["frame_id"][1] += 1
    reject(lambda: module.summarize(bad), "Cross-group original time mismatch rejected")
    tests.append("Malformed support, dropped peaks and changed source axes fail explicitly")
    return dict(status="PASS_SYNTHETIC_SAVED_SUMMARY_CONTRACT", checks=checks, tests=tests,
                module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                selftest_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                source_kind="Hand-constructed saved-readout fixtures only", public_records_processed=0,
                old_compare_calls=0, new_dtw=0, new_inference=0, educational_accuracy=None,
                prior_local_attempt=[{"status":"FAIL_TEST_FIXTURE", "test":"Cross-group original time mismatch rejected",
                    "cause":"Fixture frame lists were shared across groups; mutation changed every group together.",
                    "correction":"Copy the selected group's frame list before perturbing it; production code was unchanged."}])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run()
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
