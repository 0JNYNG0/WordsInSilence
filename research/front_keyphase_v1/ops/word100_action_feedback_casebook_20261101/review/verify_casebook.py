"""Independent saved-output verification; never imports a producer or recomputes geometry.

Usage: python -X utf8 review/verify_casebook.py [--selftest]
Result writes are exclusive. Preserve a first failure; use --output for a reviewed correction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True
UNIT = Path(__file__).resolve().parents[1]
OPS = UNIT.parent
PREVIOUS = OPS / "word100_independent_hand_evidence_20261101"
LEGACY = OPS / "word100_description_evidence_feedback_20261101"
EXPECTED = {
    "P036": {"references": ["P033", "P034", "P035"], "parents": 9, "linked": 2, "clauses": 4},
    "P048": {"references": ["P045", "P046", "P047"], "parents": 6, "linked": 2, "clauses": 4},
    "P066": {"references": ["P063", "P064", "P065"], "parents": 15, "linked": 5, "clauses": 10},
    "P072": {"references": ["P069", "P070", "P071"], "parents": 9, "linked": 3, "clauses": 12},
}


def read(path):
    p = Path(path)
    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pointer(document, path):
    if path == "":
        return document
    if not path.startswith("/"):
        raise ValueError("JSON pointer must be empty or begin with slash")
    value = document
    for raw in path[1:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


class Audit:
    def __init__(self):
        self.checks = 0
        self.failures = []
        self.verified_hashes = {}
        self.counts = Counter()

    def check(self, condition, label, detail=None):
        self.checks += 1
        if not condition:
            self.failures.append({"check": label, "detail": detail})

    def equal(self, actual, expected, label):
        self.check(actual == expected, label, None if actual == expected else {"actual": actual, "expected": expected})

    def number(self, actual, expected, label):
        if expected is None or actual is None:
            self.equal(actual, expected, label)
        else:
            ok = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10)
            self.check(ok, label, None if ok else {"actual": actual, "expected": expected})

    def artifact(self, item, label):
        path = str(Path(item["path"]).resolve())
        if path not in self.verified_hashes:
            self.verified_hashes[path] = sha(path)
        self.equal(self.verified_hashes[path], item["sha256"], label + ":sha256")
        return read(path)


def verify_summary(a, source, summary, label):
    """Verify summary fields against saved arrays, not a second geometry calculation."""
    evidence = source["evidence"]
    a.equal(summary["schema"], "wis_action_case_evidence_v1", label + ":summary schema")
    a.equal(summary["record_id"], source["id"], label + ":summary ID")
    a.equal(summary["source_record_pointer"], "", label + ":record pointer")
    a.equal(summary["identity"], {k: source.get(k) for k in ("kind", "query", "reference", "parent_id", "query_range", "reference_range")}, label + ":summary identity")
    a.equal(summary["synthetic"], source["kind"] == "technical", label + ":evidence role synthetic")
    a.equal(summary["technical_edit"], source.get("technical_edit"), label + ":exact technical edit metadata")
    a.equal(summary["source_path"], evidence["path"], label + ":unchanged local path")
    a.equal(set(summary["groups"]), set(evidence["groups"]), label + ":six groups retained")
    for field in ("semantic_verdict", "whole_sign_verdict", "correction_advice"):
        a.equal(summary[field], None, label + ":null " + field)
    for field in ("missing_values_imputed", "new_comparison", "new_alignment", "new_inference"):
        a.equal(summary[field], False, label + ":false " + field)
    for field in ("input_records_unchanged", "all_groups_and_dimensions_retained"):
        a.equal(summary[field], True, label + ":true " + field)
    a.equal(summary["measurement_difference"]["status"], "SAVED_NUMERIC_DIFFERENCES_ONLY", label + ":numeric-only status")
    for field in ("scalar_difference", "educational_error"):
        a.equal(summary["measurement_difference"][field], None, label + ":no pooled/correctness " + field)
    for field in ("zero_is_correct_performance", "position_or_posterior_used_for_gating", "threshold_applied"):
        a.equal(summary["measurement_difference"][field], False, label + ":no promotion " + field)
    a.equal(summary["linguistic_importance"], {"status": "UNESTABLISHED", "important_difference": None, "allowable_difference": None, "required_feature": None, "educational_error": None}, label + ":linguistic limits")
    a.equal(summary["correspondence_support"]["status"], "SAVED_PATH_NOT_SEMANTICALLY_VERIFIED", label + ":correspondence caveat")
    for field in ("alignment_verified", "observed_action_order_verified"):
        a.equal(summary["correspondence_support"][field], False, label + ":not verified " + field)
    a.equal(summary["correspondence_support"]["path_full_scope"], evidence["path_full_scope"], label + ":path scope flag")
    for field in ("inferred_from_measurements", "review_applies_to_whole_parent", "review_is_training_truth"):
        a.equal(summary["visible_difference_review"][field], False, label + ":visual boundary " + field)
    for field in ("full_span_semantic_verified", "legacy_decisions_applied_to_new_measurements"):
        a.equal(summary["description_support"][field], False, label + ":description boundary " + field)
    a.equal(summary["description_support"]["source_links_and_qualifiers_preserved"], True, label + ":description qualifiers flag")
    for name, group in evidence["groups"].items():
        out = summary["groups"][name]
        prefix = label + ":" + name
        gp = "/evidence/groups/" + name
        a.equal(out["source_pointer"], gp, prefix + ":group pointer")
        for field in ("definition", "source_columns", "source_feature_names", "units", "circular"):
            a.equal(out[field], group[field], prefix + ":retained " + field)
        a.equal(out["scalar_agreement"], None, prefix + ":no scalar agreement")
        a.equal(pointer(source, out["signed_and_absolute_link_values_pointer"]), group["links"], prefix + ":signed and absolute link provenance")
        for axis in ("query", "reference"):
            old = group[axis]
            new = out[axis]
            ap = gp + "/" + axis
            n, d = old["possible_shape"]
            axlabel = prefix + ":" + axis
            a.equal(summary["source_axes"][axis], {"frame_id": old["frame_id"], "timestamp_us": old["timestamp_us"]}, axlabel + ":source frame/PTS")
            a.equal(new["source_pointer"], ap, axlabel + ":axis pointer")
            a.equal(new["possible_shape"], [n, d], axlabel + ":fixed shape")
            a.equal(new["dimension_labels"], old["dimension_labels"], axlabel + ":dimension identity")
            a.equal(pointer(source, new["frame_id_pointer"]), old["frame_id"], axlabel + ":frame pointer")
            a.equal(pointer(source, new["timestamp_us_pointer"]), old["timestamp_us"], axlabel + ":PTS pointer")
            a.equal(new["unaligned_frame_indices"], old["unaligned_frame_indices"], axlabel + ":unaligned opportunities")
            a.equal(new["all_opportunities_retained"], True, axlabel + ":opportunities flag")
            a.equal(new["numeric_values_recomputed"], False, axlabel + ":saved values only")
            for field, ref in new["raw_fields"].items():
                a.equal(ref, ap + "/" + field, axlabel + ":raw pointer " + field)
                a.equal(pointer(source, ref), old[field], axlabel + ":source resolves " + field)
            required_fields = {"mean_abs_deg", "max_abs_deg", "cell_status", "cell_reason", "own_reason", "own_observed", "matched_support", "mixed_counterpart_missing", "values_degrees", "aligned_link_count", "matched_link_count", "missing_counterpart_link_count", "counterpart_observed_link_count", "original_source_mask", "diagnostics"}
            a.equal(set(new["raw_fields"]), required_fields, axlabel + ":all raw independent evidence exposed")
            own = [v for row in old["own_observed"] for v in row]
            shared = [v for row in old["matched_support"] for v in row]
            mixed = [v for row in old["mixed_counterpart_missing"] for v in row]
            counts = {"possible": n*d, "own_observed": sum(own), "own_unobserved": n*d-sum(own), "matched_support": sum(shared),
                      "direct_complete": sum(s and not m for s,m in zip(shared,mixed)), "partial_shared": sum(mixed),
                      "not_comparable": n*d-sum(shared), "observed_without_shared_counterpart": sum(o and not s for o,s in zip(own,shared))}
            a.equal(new["counts"], counts, axlabel + ":independent support counts")
            a.equal(new["counts"], old["counts"], axlabel + ":saved denominator unchanged")
            a.equal(new["own_reason_counts"], dict(Counter(v for row in old["own_reason"] for v in row)), axlabel + ":own reasons")
            a.equal(new["cell_reason_counts"], dict(Counter(v for row in old["cell_reason"] for v in row)), axlabel + ":cell reasons")
            a.equal(summary["observation_support"]["groups"][name][axis], counts, axlabel + ":independent support channel")
            a.equal(len(new["dimensions"]), d, axlabel + ":all dimensions")
            for j, dim in enumerate(new["dimensions"]):
                indices = {"direct_complete": [], "partial_shared": [], "not_comparable": [], "own_unobserved": []}
                for i in range(n):
                    own_ij, shared_ij, mixed_ij = old["own_observed"][i][j], old["matched_support"][i][j], old["mixed_counterpart_missing"][i][j]
                    category = "partial_shared" if mixed_ij else "direct_complete" if shared_ij else "not_comparable"
                    indices[category].append(i)
                    if not own_ij:
                        indices["own_unobserved"].append(i)
                    a.check((old["mean_abs_deg"][i][j] is not None) == shared_ij and (old["max_abs_deg"][i][j] is not None) == shared_ij, axlabel + f":source null retained [{i},{j}]")
                a.equal(dim["dimension"], j, axlabel + f":dimension {j}")
                a.equal(dim["label"], old["dimension_labels"][j], axlabel + f":label {j}")
                a.equal(dim["possible"], n, axlabel + f":fixed per-dimension frames {j}")
                a.equal(dim["counts"], {k:len(v) for k,v in indices.items()}, axlabel + f":dimension counts {j}")
                a.equal(dim["frame_indices_by_support"], indices, axlabel + f":exact missing/partial frame indices {j}")
                a.equal(dim["raw_cell_pointer_template"], ap + "/{field}/{frame_index}/" + str(j), axlabel + f":raw cell pointer {j}")
            for key, value in counts.items():
                a.counts[source["kind"] + "_" + axis + "_" + key] += value
        a.equal(len(out["summary_by_dimension"]), len(group["summary_by_dimension"]), prefix + ":all per-dimension measurements")
        empty_dimensions = []
        for j, new in enumerate(out["summary_by_dimension"]):
            saved = group["summary_by_dimension"][j]
            dimlabel = prefix + ":dimension" + str(j)
            for field, value in saved.items():
                a.equal(new[field], value, dimlabel + ":exact saved " + field)
            values = [row[j] for row in group["links"]["absolute_difference_deg"] if row[j] is not None]
            a.equal(new["possible_links"], len(evidence["path"]), dimlabel + ":fixed path opportunities")
            a.equal(new["shared_links"], len(values), dimlabel + ":shared link count")
            a.number(new["mean_abs_deg"], sum(values)/len(values) if values else None, dimlabel + ":mean of saved links")
            a.number(new["max_abs_deg"], max(values) if values else None, dimlabel + ":maximum of saved links")
            peaks = [i for i,row in enumerate(group["links"]["absolute_difference_deg"]) if row[j] is not None and row[j] == max(values)] if values else []
            a.equal(new["peak_link_indices"], peaks, dimlabel + ":every tied peak")
            a.equal(pointer(source, new["source_pointer"]), saved, dimlabel + ":summary source pointer")
            a.equal(new["measurement_status"], "SHARED_MEASUREMENTS" if values else "NO_SHARED_OBSERVATION", dimlabel + ":no unknown-to-agreement")
            a.equal(new["all_tied_peaks_retained"], True, dimlabel + ":ties flag")
            a.equal(new["numerical_difference_is_educational_error"], False, dimlabel + ":difference is not error")
            a.equal(new["peak_source_mapping"], {"all_peak_indices_field":"peak_link_indices", "summary_path_pointer":"/source_path", "summary_axes_pointer":"/source_axes", "original_path_pointer":"/evidence/path", "dimension":j, "original_absolute_difference_pointer":gp+"/links/absolute_difference_deg"}, dimlabel + ":all peaks mapping")
            if peaks:
                k = peaks[0]
                qi, ri = evidence["path"][k]
                expected_peak = {"link_index": k}
                for axis, index in (("query",qi),("reference",ri)):
                    expected_peak[axis] = {"frame_index":index, "frame_id":group[axis]["frame_id"][index], "timestamp_us":group[axis]["timestamp_us"][index]}
                a.equal(new["first_peak"], expected_peak, dimlabel + ":first peak original time")
            else:
                empty_dimensions.append(j)
                a.equal(new["first_peak"], None, dimlabel + ":unknown has no peak")
            a.counts[source["kind"] + "_dimension_opportunities"] += 1
            a.counts[source["kind"] + "_supported_dimensions"] += bool(values)
            a.counts[source["kind"] + "_positive_numeric_dimensions_not_errors"] += bool(values) and max(values) > 0
        a.equal(summary["measurement_difference"]["groups"][name]["no_shared_dimensions"], empty_dimensions, prefix + ":independent missing dimensions")


def verify_visual_scope(a, selection, rgb, visual):
    selected = {w["window_id"]:w for w in selection["windows"]}
    rgb_windows = {w["window_id"]:w for w in rgb["windows"]}
    frames = {(f["alias"],f["frame_id"]):f for f in rgb["full_frames"]}
    pages = {p["file"]:p for p in rgb["pages"]}
    a.equal({w["window_id"] for w in visual["windows"]}, set(selected), "exact prespecified visual windows")
    a.equal(visual["provenance"]["selection_sha256"], sha(UNIT/"review/WINDOW_SELECTION.json"), "visual selection pin")
    a.equal(visual["provenance"]["rgb_manifest_sha256"], sha(UNIT/"review/RGB_MANIFEST.json"), "visual RGB manifest pin")
    a.equal(visual["provenance"]["selection_based_on_new_residual_scores"], False, "no visual score selection")
    a.equal(set(visual["provenance"]["all_contact_pages_actually_viewed"]), set(pages), "all contact pages recorded viewed")
    unique_frames = set()
    for w in visual["windows"]:
        wid = w["window_id"]
        original = selected[wid]
        for k in ("query","reference","parent_id","categories","word_id"):
            a.equal(w[k], original[k], wid+":exact selected "+k)
        for k in ("educational_verdict","actual_contact_verdict","actual_3d_rotation_verdict","anatomical_wrist_bend_verdict"):
            a.equal(w[k], None, wid+":no promotion "+k)
        a.equal(w["source_evidence"]["saved_core_correspondence"], original["saved_core_correspondence"], wid+":saved core mapping")
        for axis in ("query_axis","reference_axis"):
            old = original[axis]
            seen = w[axis]
            prefix = wid+":"+axis
            for k in ("alias","clip_id","source","source_sha256","input","input_sha256","core_frame_range_inclusive","context_frame_range_inclusive"):
                a.equal(seen[k],old[k],prefix+":original "+k)
            a.equal(seen["reviewed_frame_ids"], old["context_frame_ids"], prefix+":reviewed exact selected frame IDs")
            a.equal(seen["reviewed_timestamp_us"], old["context_timestamp_us"], prefix+":reviewed exact native PTS")
            a.equal(seen["full_rgb_frames"],rgb_windows[wid]["axes"][axis],prefix+":exact manifest images")
            a.equal([f["frame_id"] for f in seen["full_rgb_frames"]],seen["reviewed_frame_ids"],prefix+":manifest frame coverage")
            a.equal([f["timestamp_us"] for f in seen["full_rgb_frames"]],seen["reviewed_timestamp_us"],prefix+":manifest PTS coverage")
            covered_by_pages = set()
            for page in seen["contact_pages"]:
                a.equal(page,pages[page["file"]],prefix+":contact page original record")
                a.equal(page["alias"],seen["alias"],prefix+":contact page identity")
                a.equal(sha(page["file"]),page["sha256"],prefix+":contact page hash")
                covered_by_pages.update(page["frame_ids"])
            a.check(set(seen["reviewed_frame_ids"]) <= covered_by_pages,prefix+":every reviewed frame displayed")
            a.check(set(seen["separately_opened_full_rgb_frame_ids"]) <= set(seen["reviewed_frame_ids"]),prefix+":separate full-frame opens within selection")
            for f in seen["full_rgb_frames"]:
                key=(f["alias"],f["frame_id"])
                a.equal(f,frames[key],prefix+":full image provenance "+str(f["frame_id"]))
                a.equal(f["source"],old["source"],prefix+":full image original video")
                a.equal(f["source_sha256"],old["source_sha256"],prefix+":video hash")
                a.equal(f["timestamp_us"],f["decoded_timestamp_us"],prefix+":native decode timestamp")
                if key not in unique_frames:
                    a.equal(sha(f["image"]),f["image_sha256"],prefix+":full RGB image hash")
                unique_frames.add(key)
            a.counts["visual_frame_opportunities"] += len(seen["reviewed_frame_ids"])
    a.equal(len(unique_frames),303,"fixed unique RGB frames")
    a.equal(a.counts["visual_frame_opportunities"],361,"fixed overlapping-window RGB opportunities")


def verify():
    a = Audit()
    inputs = read(UNIT / "INPUTS.json")
    manifest = read(UNIT / "RESULT_MANIFEST.json")
    casebook = read(UNIT / "CASEBOOK.json")
    old_inputs = read(PREVIOUS / "INPUTS.json")
    old_manifest = read(PREVIOUS / "RESULT_MANIFEST.json")
    old_report = read(LEGACY / "REPORT_DATA.json")
    frozen = read(UNIT / "FROZEN.json")
    window_selection = read(UNIT / "review/WINDOW_SELECTION.json")
    visual_review = read(UNIT / "review/VISUAL_REVIEW.json")
    verify_visual_scope(a,window_selection,read(UNIT/"review/RGB_MANIFEST.json"),visual_review)
    old_records = {r["id"]: r for r in old_manifest["results"]}
    wanted_natural = [r for r in old_inputs["natural"] if r["query"] in EXPECTED]
    natural_by_id = {r["id"]: r for r in wanted_natural}
    tasks = inputs["natural"] + inputs["technical"]
    task_by_id = {r["id"]: r for r in tasks}
    a.equal(len(tasks), 849, "total fixed summaries")
    a.equal(len(task_by_id), 849, "unique summary IDs")
    a.equal({t["id"] for t in inputs["natural"]}, set(natural_by_id), "exact selected 39 natural IDs")
    a.equal({t["id"] for t in inputs["technical"]}, {r["id"] for r in old_inputs["technical"]}, "exact saved 810 technical IDs")
    a.equal(len(manifest["results"]), 849, "manifest complete")
    a.equal({r["id"] for r in manifest["results"]}, set(task_by_id), "manifest IDs exact")
    a.equal(len({r["id"] for r in manifest["results"]}), 849, "manifest IDs unique")
    a.equal(len(casebook["cases"]), 4, "casebook four cases")
    a.equal({c["query"] for c in casebook["cases"]}, set(EXPECTED), "casebook exact query IDs")
    a.equal(casebook["educational_accuracy"], None, "no educational accuracy")
    a.equal(casebook["historical_development_not_new_generalization"], True, "development role preserved")
    for field in ("new_model_inference", "new_training", "new_dtw", "new_coordinate_comparisons"):
        a.equal(manifest[field], 0, "manifest no " + field)
    a.equal(manifest["end_to_end_user_latency"], None, "summary timing not end-to-end")
    rows_by_id = {}
    source_links = Counter()
    for link in manifest["results"]:
        ident = link["id"]
        t = task_by_id[ident]
        output = a.artifact(link, ident + ":output")
        rows_by_id[ident] = output
        a.equal(output["source"], old_records[ident], ident + ":exact source reference")
        source = a.artifact(output["source"], ident + ":source")
        for field in ("id", "kind"):
            a.equal(output[field], source[field], ident + ":" + field)
        a.equal(output["frozen_sha256"], sha(UNIT / "FROZEN.json"), ident + ":freeze pin")
        for field in ("position_sources", "shape_sources", "shape_guide_link"):
            a.equal(output[field], source.get(field, [] if field != "shape_guide_link" else None), ident + ":" + field)
        for field in ("new_inference", "new_dtw", "new_coordinate_comparison", "training_target"):
            a.equal(output[field], False, ident + ":" + field)
        a.equal(output["educational_verdict"], None, ident + ":no educational verdict")
        verify_summary(a, source, output["summary"], ident)
        a.counts[t["kind"] + "_summaries"] += 1
        if t["kind"] == "natural":
            original = natural_by_id[ident]
            for field in ("query", "reference", "query_range", "reference_range"):
                a.equal(output[field], original[field], ident + ":preserved " + field)
            a.equal(output["parent_id"], original["parent"]["segment_id"], ident + ":parent ID")
            a.equal(output["legacy_source"], t["legacy"], ident + ":legacy pointer")
            a.equal(sha(t["legacy"]["path"]), t["legacy"]["sha256"], ident + ":legacy source hash")
            stage = pointer(old_report, t["legacy"]["json_pointer"])
            a.equal(stage["id"], output["parent_id"], ident + ":legacy stage identity")
            a.equal(stage["path"], original["baseline_full_path"], ident + ":unchanged original path")
            a.equal(len(source["position_sources"]), 5, ident + ":five position sources")
            a.equal(len(source["shape_sources"]), 3, ident + ":all three saved shape seeds")
            a.equal(stage["source_cards"], old_report["cases"][t["legacy"]["case_index"]]["stages"][t["legacy"]["stage_index"]]["source_cards"], ident + ":exact cards source")
            summary = output["summary"]
            desc = summary["description_support"]
            provided = desc["provided"]
            a.equal(provided["source"], t["legacy"], ident + ":description provenance")
            a.equal(provided["cards"], stage["source_cards"], ident + ":every source qualifier preserved")
            a.equal(provided["physical_description"], stage["reference_rgb_description"], ident + ":physical description unchanged")
            a.equal(provided["old_trace_pointer"], t["legacy"]["json_pointer"]+"/groups", ident + ":old trace pointer")
            a.equal(provided["legacy_decisions"], {k:{f:v.get(f) for f in ("saved_metric","saved_decision","saved_range","feedback")} for k,v in stage["groups"].items()}, ident + ":legacy decisions preserved separately")
            a.equal(desc["status"], "REFERENCE_SOURCE_CONTEXT" if stage["source_cards"] else "PHYSICAL_CONTEXT_ONLY", ident + ":source applicability")
            a.equal(desc["cards_count"], len(stage["source_cards"]), ident + ":source card count")
            sc = dict(Counter(c.get("source_status","NOT_SPECIFIED") for c in stage["source_cards"]))
            a.equal(desc["source_status_counts"], sc, ident + ":support qualifiers")
            source_links.update(sc)
            for card in provided["cards"]:
                a.equal(card["reference_segment_id"], output["parent_id"], ident + ":card exact occurrence")
                for field in ("full_span_semantic_verified", "named_shape_verified", "source_action_verified", "training_target"):
                    a.equal(card[field], False, ident + ":unpromoted card " + field)
            cor = summary["correspondence_support"]["provided"]
            a.equal(cor["source"], t["legacy"], ident + ":correspondence provenance")
            a.equal(cor["details"], stage["alignment"], ident + ":correspondence uncertainty exact")
            a.equal(cor["saved_path"], original["baseline_full_path"], ident + ":correspondence source path")
            a.equal(cor["actual_stage_correctness_verified"], False, ident + ":no stage truth promotion")
            a.equal(cor["new_path_computed"], False, ident + ":no replacement path")
            selected = [(i,w) for i,w in enumerate(visual_review["windows"]) if (w["query"],w["reference"],w["parent_id"]) == (t["query"],t["reference"],t["parent_id"])]
            vr = summary["visible_difference_review"]
            a.equal(vr["status"], "REVIEWED_SCOPED_OBSERVATIONS" if selected else "NOT_REVIEWED", ident + ":exact reviewed scope")
            if selected:
                a.equal(vr["provided"]["windows"], [w for _,w in selected], ident + ":original scoped observations")
                a.equal(len(vr["provided"]["observations"]), len(selected), ident + ":one source per actual window")
                for item,(index,window) in zip(vr["provided"]["observations"],selected):
                    a.equal(item["review"], window, ident + ":unchanged visual observation")
                    a.equal(item["source"], {"path":str(UNIT/"review/VISUAL_REVIEW.json"),"sha256":sha(UNIT/"review/VISUAL_REVIEW.json"),"json_pointer":"/windows/"+str(index)}, ident + ":visual exact source pointer")
                a.equal(vr["provided"]["observation_not_training_truth"], True, ident + ":visual not target")
                a.equal(vr["provided"]["scope_is_only_named_pair_parent_and_observed_frames"], True, ident + ":visual narrow scope")
            else:
                a.equal(vr["provided"], None, ident + ":unreviewed not filled")
            a.counts["visual_reviewed_parent_rows"] += bool(selected)
            a.counts["linked_parents" if stage["source_cards"] else "physical_only_parents"] += 1
            a.counts["clause_links"] += len(stage["source_cards"])
            a.counts["position_source_links"] += len(source["position_sources"])
            a.counts["shape_source_links"] += len(source["shape_sources"])
        else:
            a.equal(source["shape_sources"], [], ident + ":technical no reused model probabilities")
            for field in ("description_support", "correspondence_support", "visible_difference_review"):
                a.equal(output["summary"][field]["provided"], None, ident + ":technical no performed-video or guide " + field)
            a.equal(output["summary"]["visible_difference_review"]["status"], "NOT_REVIEWED", ident + ":technical not RGB reviewed")
            a.counts["technical_condition_" + t["condition"]] += 1
    for case in casebook["cases"]:
        q = case["query"]
        expected = EXPECTED[q]
        a.equal(case["references"], expected["references"], q + ":three separate references")
        a.equal(case["all_three_references_retained"], True, q + ":reference retention flag")
        a.equal(len(case["parents"]), expected["parents"], q + ":all parent opportunities")
        a.equal({r["id"] for r in case["parents"]}, {t["id"] for t in inputs["natural"] if t["query"] == q}, q + ":exact case rows")
        a.equal(sum(bool(t["source_card_count"]) for t in inputs["natural"] if t["query"] == q), expected["linked"], q + ":linked parent count")
        a.equal(sum(t["source_card_count"] for t in inputs["natural"] if t["query"] == q), expected["clauses"], q + ":clause count")
    for field, count in {"natural_summaries": 39, "technical_summaries": 810, "linked_parents": 12, "physical_only_parents": 27, "clause_links": 30, "position_source_links": 195, "shape_source_links": 117}.items():
        a.equal(a.counts[field], count, "fixed total " + field)
    a.equal(dict(source_links), {"visible_partial_match":24,"visible_proximity_only":3,"unverified_edge_on":3}, "source-link support 24/3/3 retained")
    a.check(len(window_selection["windows"]) <= 12, "prespecified review-window cap")
    a.equal(window_selection["residual_or_probability_summary_consulted_for_selection"], False, "no score-driven window selection")
    a.equal({w["window_id"] for w in visual_review["windows"]}, {w["window_id"] for w in window_selection["windows"]}, "review exactly selected windows")
    for path, expected_hash in frozen["code_and_review_pins"].items():
        a.equal(sha(path), expected_hash, "frozen implementation/review:" + Path(path).name)
    return a


def selftest():
    a = Audit()
    sample = {"a/b": [{"~": 2}], "empty": None}
    a.equal(pointer(sample, "/a~1b/0/~0"), 2, "escaped JSON pointer")
    a.equal(pointer(sample, ""), sample, "root JSON pointer")
    a.equal(pointer(sample, "/empty"), None, "null source remains null")
    # The accumulator must distinguish unknown from numerical zero and retain mismatches.
    negative = Audit()
    negative.number(0, None, "unknown incorrectly becomes zero")
    negative.number(None, 0, "known zero incorrectly discarded")
    negative.equal([1, 2], [1, 2, 3], "denominator shrinks")
    negative.equal(["P033"], ["P033", "P034", "P035"], "reference chosen")
    a.equal(len(negative.failures), 4, "adverse cases rejected")
    return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or UNIT / "review" / ("VERIFIER_SELFTEST.json" if args.selftest else "RESULT_VERIFICATION.json")
    if output.exists():
        raise FileExistsError(output)
    started = datetime.now(timezone.utc).isoformat()
    try:
        a = selftest() if args.selftest else verify()
    except Exception:
        a = Audit()
        a.check(False, "verifier protocol or input exception", traceback.format_exc())
    report = {"status": "PASS" if not a.failures else "FAIL", "checks": a.checks, "failure_count": len(a.failures), "failures": a.failures,
              "counts": dict(a.counts), "verified_artifact_hashes": a.verified_hashes,
              "verifier_sha256": sha(__file__), "started_at": started,
              "method": "Independent saved JSON/source-pointer and summary accounting, no producer imported or executed",
              "old_compare_reruns": 0, "new_alignment": 0, "new_inference": 0, "new_training": 0,
              "independent_human_KSL_truth": False, "educational_accuracy": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False, default=lambda value: sorted(value) if isinstance(value,set) else repr(value))
        stream.write("\n")
    print(json.dumps({"status": report["status"], "checks": a.checks, "failures": len(a.failures), "output": str(output)}, ensure_ascii=False))
    return 0 if not a.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
