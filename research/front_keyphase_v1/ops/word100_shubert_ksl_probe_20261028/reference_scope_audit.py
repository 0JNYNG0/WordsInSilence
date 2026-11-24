"""Supplemental reference-oriented coverage audit; existing results stay fixed.

Uses saved DTW paths only. Every reference anchor stays in the raw denominator.
Where reference consensus AND current evidence exist, its expected comparison
cell stays in the denominator even if the mapped query is unknown or missing.
No inference, training, new thresholds, model choice, or changed main results.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np

U = Path(__file__).resolve().parent
P = U.parent / "word100_shubert_feasibility_20261028"
METHODS = ("geometry", "shubert", "time_fraction")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(rows):
    result = {}
    for method in METHODS:
        items = [r for r in rows if r["method"] == method]
        counts = Counter(r["outcome"] for r in items)
        expected = sum(r["reference_expected"] for r in items)
        known = counts["agree"] + counts["different"]
        assert expected == known + counts["query_unknown"]
        assert len(items) == expected + counts["reference_unassessable"]
        result[method] = {
            "all_reference_anchor_feature_slots": len(items),
            "expected_reference_cells": expected,
            "reference_unassessable": counts["reference_unassessable"],
            "agree": counts["agree"], "different": counts["different"],
            "query_unknown_in_fixed_expected_scope": counts["query_unknown"],
            "known_query_cells": known,
            "known_query_coverage_of_expected": known / expected if expected else None,
            "agreement_over_fixed_expected_scope": counts["agree"] / expected if expected else None,
            "agreement_only_where_query_known": counts["agree"] / known if known else None,
        }
    # These denominators depend solely on the reference, never the method or
    # whether the selected query observation exists.
    assert len({x["all_reference_anchor_feature_slots"] for x in result.values()}) == 1
    assert len({x["expected_reference_cells"] for x in result.values()}) == 1
    assert len({x["reference_unassessable"] for x in result.values()}) == 1
    return result


def main():
    protected_names = ["FROZEN.json", "CORRESPONDENCE.json", "EVALUATION.json", "CONSENSUS.json", "RUBRIC.json", "SAMPLES.json"]
    before = {name: sha(U / name) for name in protected_names}
    frozen, correspondence = read(U / "FROZEN.json"), read(U / "CORRESPONDENCE.json")
    consensus_doc, sample_doc = read(U / "CONSENSUS.json"), read(U / "SAMPLES.json")
    values, fields = consensus_doc["values"], read(U / "RUBRIC.json")["fields"]
    samples = {r["sample_id"]: r for r in sample_doc["samples"]}
    meta = {r["sample_id"]: r for r in correspondence["sample_metadata"]}
    clip = {r["alias"]: r for r in frozen["rows"]}
    assert correspondence["frozen_sha256"] == before["FROZEN.json"]
    current = {}
    provenance_pins = {}
    for alias, row in clip.items():
        path = P / "inputs" / alias / "provenance.npz"
        provenance_pins[alias] = sha(path)
        assert provenance_pins[alias] == correspondence["input_pins"][alias]["provenance"]
        with np.load(path, allow_pickle=False) as z:
            current[alias] = np.array(z["source_kind"]) == 1

    def evidence(sid, field):
        sample = samples[sid]
        slots = [1] if field.startswith("R_") else ([0] if field.startswith("L_") else [0, 1])
        indices = sample["context_indices"] if field in fields[7:] else [sample["frame_index"]]
        return all(bool(current[sample["alias"]][frame, slot]) for frame in indices for slot in slots)

    rows, mapping_rows = [], []
    for pair in correspondence["stage_pairs"]:
        qa, ra, method = pair["query"], pair["reference"], pair["method"]
        assert method in METHODS
        assert clip[qa]["recorded_person_id"] == 3
        assert clip[ra]["recorded_person_id"] in (1, 2)
        assert clip[qa]["word_id"] == clip[ra]["word_id"]
        assert len(pair["reference_sample_ids"]) == len(pair["query_sample_ids"]) == 9
        assert {j for i, j in pair["path"]} == set(range(9))
        for j, rid in enumerate(pair["reference_sample_ids"]):
            matched_queries = sorted(i for i, ref_index in pair["path"] if ref_index == j)
            assert matched_queries
            qi = int(np.median(matched_queries))
            assert qi in matched_queries
            sid = pair["query_sample_ids"][qi]
            mapping_rows.append({"query": qa, "reference": ra, "method": method,
                                 "reference_anchor": j, "reference_sample_id": rid,
                                 "query_anchor": qi, "query_sample_id": sid,
                                 "path_query_anchors_for_reference": matched_queries,
                                 "reference_annotation_active": meta[rid]["annotation_active"]})
            for fi, field in enumerate(fields):
                reference_value, query_value = values[rid][fi], values[sid][fi]
                ref_current, query_current = evidence(rid, field), evidence(sid, field)
                expected = reference_value is not None and ref_current
                if not expected:
                    outcome = "reference_unassessable"
                elif query_value is None or not query_current:
                    outcome = "query_unknown"
                else:
                    outcome = "agree" if reference_value == query_value else "different"
                rows.append({"query": qa, "reference": ra, "method": method,
                             "reference_sample_id": rid, "query_sample_id": sid,
                             "field": field, "group": "motion" if fi >= 7 else "static",
                             "reference_annotation_active": meta[rid]["annotation_active"],
                             "reference_value": reference_value, "query_value": query_value,
                             "reference_current": ref_current, "query_current": query_current,
                             "reference_expected": expected, "outcome": outcome})

    assert len(mapping_rows) == 8 * 3 * 9
    assert len(rows) == 8 * 3 * 9 * 10
    expected_key_sets = {}
    all_key_sets = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        key = lambda r: (r["query"], r["reference"], r["reference_sample_id"], r["field"])
        all_key_sets[method] = {key(r) for r in subset}
        expected_key_sets[method] = {key(r) for r in subset if r["reference_expected"]}
        assert len(subset) == len(all_key_sets[method]) == 720
    assert all(all_key_sets[m] == all_key_sets["geometry"] for m in METHODS)
    assert all(expected_key_sets[m] == expected_key_sets["geometry"] for m in METHODS)
    active = [r for r in rows if r["reference_annotation_active"]]
    overall, active_summary = summarize(rows), summarize(active)
    # Supplemental contract control: marking all query evidence unknown must
    # change outcomes to unknown, never erase expected reference requirements.
    unknown_query_rows = [{**r, "query_value": None, "query_current": False,
                           "outcome": "query_unknown" if r["reference_expected"] else "reference_unassessable"} for r in rows]
    all_unknown = summarize(unknown_query_rows)
    for method in METHODS:
        assert all_unknown[method]["expected_reference_cells"] == overall[method]["expected_reference_cells"]
        assert all_unknown[method]["query_unknown_in_fixed_expected_scope"] == overall[method]["expected_reference_cells"]
        assert all_unknown[method]["agree"] == all_unknown[method]["different"] == 0
    after = {name: sha(U / name) for name in protected_names}
    assert before == after
    result = {
        "status": "PASS_SUPPLEMENTAL_REFERENCE_SCOPE_ARITHMETIC",
        "purpose": "Coverage supplement, not replacement or selection among existing primary evaluation results.",
        "scope": "All 9 reference anchors per same-word pair, mapped query median from existing frozen paths. Same two blind AI drafts only.",
        "definition": "Reference nonnull consensus plus current observation defines expected cells. Query missing/unknown remains query_unknown inside this fixed denominator.",
        "existing_primary_results_unchanged": True, "protected_pins": before,
        "provenance_pins": provenance_pins, "script_sha256": sha(__file__),
        "same_word_pairs": 8, "methods": list(METHODS),
        "anchors_per_reference": 9, "reference_anchors_per_method": 72,
        "raw_reference_feature_slots_per_method": 720,
        "same_expected_cell_keys_across_methods": True,
        "all_reference_slots_retained": True,
        "all_reference_anchors": overall, "reference_annotation_active": active_summary,
        "reference_active_static": summarize([r for r in active if r["group"] == "static"]),
        "reference_active_motion": summarize([r for r in active if r["group"] == "motion"]),
        "by_pair": [{"query": qa, "reference": ra,
                     "all": summarize([r for r in rows if r["query"] == qa and r["reference"] == ra]),
                     "reference_active": summarize([r for r in active if r["query"] == qa and r["reference"] == ra])}
                    for qa, ra in sorted({(r["query"], r["reference"]) for r in rows})],
        "all_query_unknown_control": {"pass": True, "summaries": all_unknown,
                                      "scope": "Arithmetic mask control, not inference or a new video experiment."},
        "mappings": mapping_rows, "records": rows,
        "training_runs": 0, "model_forward_calls": 0, "new_rule_tuning": False,
        "primary_result_replacement": False, "candidate_selection": False,
        "limitations": ["This is a supplemental change of reporting direction, declared after the primary evaluation; do not choose favorable numbers across reports.",
                        "Every reference anchor is counted but this does not recover unsampled frames or verify true linguistic phase boundaries.",
                        "A query anchor may be reused for multiple reference anchors; reuse itself is not proof of omission or correctness.",
                        "AI draft consensus and detector-current masks are not independent human semantic truth.",
                        "No new face-exclusion model inference is part of this audit."]}
    (U / "REFERENCE_SCOPE_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "all_reference_anchors": overall,
                      "reference_annotation_active": active_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
