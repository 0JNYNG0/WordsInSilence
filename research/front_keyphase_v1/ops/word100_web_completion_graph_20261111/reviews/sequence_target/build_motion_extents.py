"""Curate existing RGB-only DEV motion/hold evidence; do not load model outputs."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGETS = ROOT / "TARGETS.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inclusive(span):
    return list(range(span[0], span[1] + 1))


def runs(values):
    """Preserve source traversal order; do not envelope disconnected query runs."""
    out = []
    for value in values:
        if out and value == out[-1][1] + 1:
            out[-1][1] = value
        else:
            out.append([value, value])
    return out


targets_sha = sha(TARGETS)
targets = json.loads(TARGETS.read_text(encoding="utf-8"))
# These witnesses were re-viewed on existing dense_03..05 sheets only.
# They are multi-frame change evidence, not framewise velocity labels.
WITNESSES = {
    11: {"B": [71, 73, 76], "C": [83, 85, 87]},
    12: {"B": [63, 65, 67], "C": [78, 80, 81]},
    13: {"B": [81, 84, 87], "C": [95, 97, 98]},
}
rows = []
for row in targets["rows"]:
    if row["word_id"] != "WORD0739" or row["person"] not in WITNESSES:
        continue
    candidate = row["candidate"]
    forward = candidate["query_to_source_frame_ids"]
    inverse = candidate["source_to_query_frame_ids"]
    assert sorted(forward) == list(range(row["frame_count"]))
    assert all(forward[inverse[s]] == s for s in range(row["frame_count"]))
    boundaries = row["boundary_uncertainty_frames"]
    reviewed_now = []
    panels_now = []
    for panel in row["panels"]:
        if any(panel["path"].endswith(f"_dense_{n:02}.jpg") for n in (3, 4, 5)):
            assert sha(Path(panel["path"])) == panel["sha256"]
            panels_now.append(panel)
            reviewed_now.extend(panel["source_frame_ids"])
    assert len(panels_now) == 3
    reviewed_now = sorted(set(reviewed_now))

    def frame_set(ids):
        ids = list(ids)
        query = [inverse[s] for s in ids]
        assert all(forward[q] == s for s, q in zip(ids, query))
        return {
            "source_frame_ids": ids,
            "source_timestamp_us": [round(s * 1_000_000 / row["fps"]) for s in ids],
            "normal_query_indices": ids,
            "permuted_query_indices_in_source_order": query,
            "permuted_query_runs_in_source_order": runs(query),
            "permuted_query_timestamp_us_in_source_order": [
                round(q * 1_000_000 / row["fps"]) for q in query
            ],
        }

    def band(span):
        return {"source_span_inclusive": span, **frame_set(inclusive(span))}

    motions = {}
    for name, onset_key, offset_key in (
        ("B", "close_and_lower_onset", "lowering_end"),
        ("C", "raising_onset", "raising_arrival"),
    ):
        onset, offset = boundaries[onset_key], boundaries[offset_key]
        witness = WITNESSES[row["person"]][name]
        assert all(onset[1] < s < offset[0] for s in witness)
        assert set(witness).issubset(reviewed_now)
        block = candidate["blocks"][name]
        assert set(witness).issubset(block["source_frame_ids"])
        hold_observation = next(
            o for o in row["observations"] if o["role"] == "main_C_terminal_hold"
        )
        hold = [
            s for s in hold_observation["source_frame_ids"]
            if offset[1] < s < boundaries["return_onset"][0]
        ] if name == "C" else []
        if hold:
            assert hold == inclusive([hold[0], hold[-1]])
            assert set(hold).issubset(reviewed_now)
        observation_role = "main_B_close_and_lower" if name == "B" else "main_C_forearm_reorientation"
        observation = next(o for o in row["observations"] if o["role"] == observation_role)
        motions[name] = {
            "descriptor": "close_chest_dynamic" if name == "B" else "raise_forearm_dynamic",
            "block": block,
            "onset_uncertainty": band(onset),
            "offset_uncertainty": band(offset),
            "boundary_basis": {
                "onset_targets_key": onset_key,
                "offset_targets_key": offset_key,
                "unchanged_from_existing_targets": True,
                "inclusive_bands_are_not_exact_boundary_labels": True,
            },
            "possible_motion_envelope": band([onset[0], offset[1]]),
            "definite_moving_witness": {
                **frame_set(witness),
                "interpretation": "Visible change across the listed original, internally time-ordered frames; not instantaneous per-frame velocity or complete-action gold.",
                "observation": (
                    "Closed/closing hand travels lower from face toward chest height; forearm becomes more horizontal. Exact finger articulation/contact remains unknown."
                    if name == "B" else
                    "Forearm changes from a more horizontal orientation toward upright, with elbow/hand displacement. Wrist translation alone is not the defining evidence."
                ),
            },
            "post_motion_hold_ranges": [band([hold[0], hold[-1]])] if hold else [],
            "post_motion_hold_status": (
                "OBSERVED_STABLE_DESCRIPTOR_POSE_OUTSIDE_RECORDED_BOUNDARY_BANDS"
                if hold else "UNRESOLVED_NO_SEPARATE_HOLD_OUTSIDE_RECORDED_BOUNDARY_BANDS"
            ),
            "post_motion_hold_basis": (
                "Existing main_C_terminal_hold observation, restricted strictly after raising_arrival upper bound and before return_onset lower bound. Re-viewed on existing sheets. Stable descriptor pose does not assert zero physical motion."
                if hold else
                "Terminal near-horizontal posture is visible, but the recorded B offset and C onset bands leave no separately established B hold. Do not infer absence of all dwell or invent a precise hold duration."
            ),
            "inherited_RGB_observation": observation,
            "inherited_C_hold_observation": hold_observation if name == "C" else None,
            "onset_and_offset_bands_within_source_block": set(inclusive(onset) + inclusive(offset)).issubset(block["source_frame_ids"]),
            "exact_motion_start_frame": None,
            "exact_motion_end_frame": None,
            "whole_sign_completion_label": None,
        }
    rows.append({
        "clip_id": row["clip_id"], "word_id": row["word_id"], "role": "development_diagnostic_only",
        "source": row["source"], "source_sha256": row["source_sha256"],
        "source_hash_basis": "Copied from existing TARGETS; no source-video bytes reopened for this task.",
        "frame_origin": 0, "frame_count": row["frame_count"], "fps": row["fps"],
        "candidate_id": candidate["id"],
        "query_to_source_frame_ids": forward, "source_to_query_frame_ids": inverse,
        "splice_joins": candidate["splice_joins"],
        "existing_annotations_found": True,
        "new_exact_boundary_labels_created": False,
        "all_frames_previously_reviewed": row["all_source_frames_reviewed"],
        "previously_reviewed_source_frame_ids": row["reviewed_frame_ids"],
        "source_frame_ids_reviewed_again_for_this_task": reviewed_now,
        "existing_contact_sheets_reviewed_again": panels_now,
        "motions": motions,
        "global_return_onset_uncertainty": band(boundaries["return_onset"]),
        "global_return_source_block": candidate["blocks"]["suffix"],
        "ambiguities": [
            "Natural B offset uncertainty reaches the C block for REAL11/13. Preserve the mapped disconnected query runs; never replace them with their min/max envelope or truncate to the B block.",
            "A source block boundary is a mechanical cut, not an exact physiological motion boundary.",
            "C's terminal pose also appears at the beginning of the return; neither static pose nor the entire hold duration is a new dynamic occurrence.",
            "Across-splice position differences are not performed-action evidence. B-to-suffix jump can resemble a raise to an unmasked model.",
            "Occluded finger geometry, contact, exact angle, and educational/whole-sign correctness are not resolved by these contact sheets.",
        ],
    })

assert len(rows) == 3
assert sha(TARGETS) == targets_sha
result = {
    "task": "public_DEV_obedience_motion_extents_separated_from_terminal_holds",
    "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "author": "Codex sequence_target_review agent",
    "status": "EXISTING_BOUNDARY_BANDS_PRESERVED_HOLDS_SEPARATED",
    "targets_path": str(TARGETS), "targets_sha256": targets_sha,
    "scope": {"new_model_or_quality_outputs_read": False, "new_inference": False,
              "training_or_tuning_use": False, "independent_human_gold": False,
              "source_video_reopened": False, "new_RGB_extraction": False,
              "existing_source_contact_sheets_reviewed": 9,
              "source_contact_sheet_frames_reviewed_again": sum(len(r["source_frame_ids_reviewed_again_for_this_task"]) for r in rows)},
    "annotation_contract": {
        "frame_numbering": "Zero-based source-video frame IDs. All source spans and uncertainty bands are inclusive.",
        "timestamps": "round(frame_id * 1000000 / fps); source and synthetic-query time are separate coordinates.",
        "boundary_truth": "Inherited inclusive uncertainty bands, not exact boundary labels. No band was narrowed, shifted, or clipped to a synthetic block.",
        "moving_witness_truth": "Direct multi-frame RGB change evidence inside the inherited bands' strict interior; insufficient alone to prove the full action or sign complete.",
        "hold_truth": "Conservative subset of the existing C terminal-hold observation outside arrival/return uncertainty; no requirement to cover every held frame.",
        "proposal_audit": "Report motion witness coverage, relation to onset/offset uncertainty, and hold-only overlap separately. A proposal containing only terminal hold is not motion evidence. Requiring the complete C block including all terminal hold is not a valid motion-extent criterion. This file supplies no binary accuracy threshold or semantic-completion label.",
        "permutation_mapping": "Use explicit source-to-query correspondence. Disconnected mapped boundary bands must remain separate runs, not one query interval.",
        "allowed_use": "Repeated public DEV diagnostic interpretation only; no training, tuning, candidate selection, or independent human-gold claim.",
    },
    "checks": {"source_index_roundtrip": True, "witnesses_inside_original_blocks": True,
               "witnesses_outside_inclusive_uncertainty_bands": True,
               "contact_sheet_hashes_match_existing_targets": True,
               "targets_bytes_unchanged": True},
    "rows": rows,
}
out = ROOT / "MOTION_EXTENTS.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"path": str(out), "sha256": sha(out), "clips": len(rows), "reviewed_again_frames": result["scope"]["source_contact_sheet_frames_reviewed_again"]}))
