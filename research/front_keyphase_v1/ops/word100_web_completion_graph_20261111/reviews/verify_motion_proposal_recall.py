"""Independent set/index arithmetic over saved outputs; no model imports/forward."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent.parent
P = U / "boundary_candidate_probe"
TARGET = U / "reviews/sequence_target/MOTION_EXTENTS.json"
AUDIT = P / "SOURCE_RECALL_AUDIT.json"
RESULTS = P / "RESULTS.json"
FAMILIES = U.parent / "word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support_at(bp, frame, k, channel, mapping, truth_source):
    """Raw saved-probability support, not a calibrated uncertainty interval.

    0.5 is the unchanged candidate_spans threshold, not an audit-selected value.
    No relation-validity mask is loaded/reconstructed by this verifier.
    """
    score = bp[:, k, channel]
    if score[frame] < .5:
        return {"point_probability": float(score[frame]), "raw_threshold_component": None}
    a = b = frame
    while a > 0 and score[a - 1] >= .5:
        a -= 1
    while b + 1 < len(score) and score[b + 1] >= .5:
        b += 1
    peak = a + int(np.argmax(score[a:b + 1]))
    ids = list(range(a, b + 1))
    source_ids = [mapping[q] for q in ids]
    return {
        "point_probability": float(score[frame]),
        "raw_threshold_component": [a, b],
        "component_query_indices": ids,
        "component_source_frame_ids": source_ids,
        "raw_component_argmax_query_index": peak,
        "raw_component_argmax_source_frame_id": mapping[peak],
        "candidate_point_equals_raw_component_argmax": frame == peak,
        "reviewed_band_source_intersection": sorted(set(source_ids) & truth_source),
        "component_crosses_source_discontinuity": any(
            mapping[q] != mapping[q - 1] + 1 for q in ids[1:]
        ),
    }


before = {str(p): sha(p) for p in [TARGET, AUDIT, RESULTS, P / "FROZEN.json", FAMILIES]}
truth = read(TARGET)
parent = read(AUDIT)
assert parent["target_sha256"] == before[str(TARGET)]
parent_rows = {(r["clip_id"], r["kind"], r["motion"]): r for r in parent["rows"]}
descriptors = {r["id"]: r for r in read(FAMILIES)["words"]["WORD0739"]}
axis_for = {"B": descriptors["close_chest"]["axis"], "C": descriptors["raise_forearm"]["axis"]}
assert axis_for == {"B": 0, "C": 1}
allowed = {(r["clip_id"], k) for r in truth["rows"] for k in ("full_normal", "dynamic_complete_blocks")}
all_saved = read(RESULTS)
saved = {(r["clip_id"], r["kind"]): r for r in all_saved if (r["clip_id"], r["kind"]) in allowed}
assert len(saved) == 6
rows, bp_checks = [], []
quality_vectors_checked = 0
for target in truth["rows"]:
    for kind in ("full_normal", "dynamic_complete_blocks"):
        prediction = saved[target["clip_id"], kind]
        mapping = (list(range(target["frame_count"])) if kind == "full_normal"
                   else target["query_to_source_frame_ids"])
        inverse = {s: q for q, s in enumerate(mapping)}
        assert sorted(mapping) == list(range(target["frame_count"]))
        seams = [q for q in range(1, len(mapping)) if mapping[q] != mapping[q - 1] + 1]
        recorded_seams = [] if kind == "full_normal" else [j["query_right"] for j in target["splice_joins"] if j["synthetic_discontinuity"]]
        assert seams == recorded_seams
        bp_path = Path(prediction["boundary_probability"])
        bp_hash = sha(bp_path)
        assert bp_hash == prediction["boundary_sha256"]
        before[str(bp_path)] = bp_hash
        with np.load(bp_path, allow_pickle=False) as z:
            bp = z["boundary"].copy()
        assert bp.shape == (len(mapping), 10, 3)
        assert np.isfinite(bp).all() and (bp >= 0).all() and (bp <= 1).all()
        bp_checks.append({"clip_id": target["clip_id"], "kind": kind, "path": str(bp_path),
                          "sha256": bp_hash, "shape": list(bp.shape), "source_index_roundtrip": True,
                          "seams_recomputed_from_source_adjacency": seams})
        for candidate in prediction["candidates"]:
            p = np.asarray(candidate["quality"])
            assert p.shape == (4,) and np.isfinite(p).all()
            assert np.isclose(p.sum(), 1, atol=1e-6)
            assert int(p.argmax()) == candidate["quality_class"]
            assert float(p.max()) == candidate["quality_confidence"]
            assert 0 <= candidate["a"] <= candidate["b"] < len(mapping)
            quality_vectors_checked += 1
        for name in ("B", "C"):
            motion = target["motions"][name]
            onset = set(motion["onset_uncertainty"]["source_frame_ids"])
            offset = set(motion["offset_uncertainty"]["source_frame_ids"])
            witness = set(motion["definite_moving_witness"]["source_frame_ids"])
            hold = {s for h in motion["post_motion_hold_ranges"] for s in h["source_frame_ids"]}
            matched, comparison_rows, crossings = [], [], []
            for c in prediction["candidates"]:
                if c["axis"] != axis_for[name] or c["moving"] != "R":
                    continue
                a, b = c["a"], c["b"]
                source_ids = mapping[a:b + 1]
                contains = witness.issubset(source_ids)
                crossed = [q for q in seams if a < q <= b]
                if contains and crossed:
                    crossings.append({"query_span": [a, b], "crossed_seams": crossed})
                if not contains or crossed:
                    continue
                on_ok, off_ok = mapping[a] in onset, mapping[b] in offset
                match = on_ok and off_ok
                # This compact object is recomputed, then compared exactly to root's audit.
                comparison_rows.append({"span": [a, b], "source": c["source"], "witness_covered": True,
                                        "onset_in_reviewed_band": on_ok, "offset_in_reviewed_band": off_ok,
                                        "boundary_extent_match": match,
                                        "quality_class": c["quality_class"], "quality_confidence": c["quality_confidence"]})
                matched.append({
                    "query_span": [a, b], "source_span": [mapping[a], mapping[b]],
                    "source_frame_ids": source_ids, "generator_source": c["source"],
                    "moving_witness_covered": True, "synthetic_seams_crossed": [],
                    "onset_point_in_reviewed_band": on_ok, "offset_point_in_reviewed_band": off_ok,
                    "both_endpoint_points_in_reviewed_bands": match,
                    "boundary_pair_proposal": c["boundary_pair_proposal"],
                    "coverage": c["coverage"], "quality": c["quality"],
                    "quality_class": c["quality_class"], "quality_confidence": c["quality_confidence"],
                    "physical_complete": c["physical_complete"], "semantic_complete": c["semantic_complete"],
                    "terminal_hold_source_overlap": sorted(set(source_ids) & hold),
                    "onset_probability_support": support_at(bp, a, axis_for[name] * 2, 1, mapping, onset),
                    "offset_probability_support": support_at(bp, b, axis_for[name] * 2, 2, mapping, offset),
                })
            reported = parent_rows[target["clip_id"], kind, name]
            assert comparison_rows == reported["candidates"]
            assert bool(matched) == reported["witnessed_motion_proposal_exists"]
            both = any(c["both_endpoint_points_in_reviewed_bands"] for c in matched)
            assert both == reported["boundary_extent_match_exists"]
            rows.append({"clip_id": target["clip_id"], "kind": kind, "motion": name,
                         "axis": axis_for[name], "moving": "R",
                         "onset_source_band": motion["onset_uncertainty"]["source_span_inclusive"],
                         "offset_source_band": motion["offset_uncertainty"]["source_span_inclusive"],
                         "witness_source_frame_ids": sorted(witness),
                         "witness_query_indices": [inverse[s] for s in sorted(witness)],
                         "terminal_hold_source_ranges": [h["source_span_inclusive"] for h in motion["post_motion_hold_ranges"]],
                         "witness_without_splice_exists": bool(matched), "both_endpoint_bands_match_exists": both,
                         "matched_candidates": matched, "witness_containing_splice_crossers_excluded": crossings,
                         "root_row_arithmetic_equal": True})

counts = {}
for kind in ("full_normal", "dynamic_complete_blocks"):
    subset = [r for r in rows if r["kind"] == kind]
    counts[kind] = {"motions": len(subset), "witnessed_motion": sum(r["witness_without_splice_exists"] for r in subset),
                    "reviewed_extent": sum(r["both_endpoint_bands_match_exists"] for r in subset)}
assert counts == parent["counts"]
misses = [r for r in rows if not r["both_endpoint_bands_match_exists"]]
assert {(r["clip_id"], r["motion"]) for r in misses} == {("NIA_SL_WORD0739_REAL11_F", "C")}
fail_details = []
for r in misses:
    candidates = [c for c in r["matched_candidates"] if c["onset_point_in_reviewed_band"]]
    assert len(candidates) == 1
    c = candidates[0]
    assert c["source_span"] == [82, 87]
    fail_details.append({"kind": r["kind"], "query_span": c["query_span"], "source_span": c["source_span"],
                         "reviewed_motion_offset_source_band": r["offset_source_band"],
                         "endpoint_before_earliest_reviewed_offset_frames": 1,
                         "endpoint_before_earliest_reviewed_offset_ms": 1000 / 30,
                         "offset_support": c["offset_probability_support"],
                         "post_motion_hold_source_ranges": r["terminal_hold_source_ranges"],
                         "candidate_hold_overlap": c["terminal_hold_source_overlap"],
                         "quality_class": c["quality_class"], "quality_confidence": c["quality_confidence"]})
assert all(sha(Path(p)) == h for p, h in before.items())
out = {
    "status": "PASS_REPORTED_ARITHMETIC_WITH_REPRESENTATIVE_ENDPOINT_LIMITATION",
    "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "reviewer": "Codex sequence_target_review agent; independent arithmetic implementation from root, same earlier RGB annotator; not independent human gold",
    "scope": {"saved_output_arithmetic_only": True, "new_inference": False, "model_loaded": False,
              "training": False, "source_RGB_or_video_reopened": False, "targets_changed": False,
              "thresholds_changed": False, "runtime_changed": False,
              "prediction_rows_checked": 6, "motions_checked": 12,
              "results_container_rows": len(all_saved),
              "other_prediction_rows_used_in_calculation": False,
              "quality_vectors_arithmetically_checked": quality_vectors_checked},
    "input_sha256": before,
    "counts_recomputed": counts,
    "root_audit_all_rows_and_counts_equal": True,
    "saved_boundary_checks": bp_checks,
    "why_five_of_six": {
        "common_exception": "WORD0739 REAL11 C only, in both full_normal and dynamic_complete_blocks",
        "diagnosis": "All direct moving witnesses are contained without a splice. The onset point is in source80..82; the generated offset representative is source87, one frame before the unchanged reviewed offset uncertainty band88..90. The broad saved offset support overlaps that band. Thus 5/6 is strict representative-point membership, not absence of an offset response, physical failure, or failure to include the prolonged terminal hold.",
        "normal_raw_offset_support": "query/source84..91, maximum87; intersects all reviewed88..90",
        "permuted_raw_offset_support": "query71..76 = source84..89, maximumquery74/source87; intersects reviewed88..89, not90",
        "quality_is_separate": "The onset-matched candidates have learned quality_class2 with approximately0.999996(normal) and0.999896(permuted). This is not observed boundary correctness, not calibrated certainty, and not physical/semantic completion; both explicit completion fields remain null.",
        "hold_is_separate": "REAL11 C conservative hold is source91..102 (permutedquery78..89). No entire-hold coverage requirement was imposed. The 87 endpoint precedes the motion-offset band itself, so the one-frame membership miss cannot be attributed to failing to cover the later long hold.",
        "no_relabel_or_reselection": "No source bands/candidates/thresholds were modified. This is diagnostic reporting, not a new selector or post-hoc pass criterion.",
        "details": fail_details,
    },
    "limits": [
        "A raw bp>=0.5 connected component is model threshold support, not a calibrated confidence/uncertainty interval and not a source RGB boundary label.",
        "Raw boundary components here omit the relation-validity mask used during generation. Stored candidate coverage is preserved; candidate generation or model inference was not rerun. For both exceptional onset-matched candidates the saved endpoint is also the raw component argmax.",
        "Witness inclusion only proves availability of candidates containing these reviewed samples, not all action frames or correct action association. Wider full-normal candidates can encompass multiple actions.",
        "Point-in-band counts are not whole-sign, natural-error, independent-human, or final held-out performance. Repeated public DEV diagnostics must not train/tune/select candidates.",
        "The strict index criterion keeps mapped disconnected source bands as exact sets; it never envelopes REAL11/13 B offset across a synthetic seam.",
    ],
    "rows": rows,
}
destination = U / "reviews/motion_proposal_recall_verification.json"
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"path": str(destination), "sha256": sha(destination), "counts": counts, "quality_vectors_checked": quality_vectors_checked}))
