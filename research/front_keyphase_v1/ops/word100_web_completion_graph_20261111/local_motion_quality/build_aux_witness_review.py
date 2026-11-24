"""Record FIT RGB witnesses only. No synthesis, model execution, or admissions."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path


U = Path(__file__).resolve().parent.parent
PROJECT = U.parents[3]
OB = U / "reviews/obedience_occurrence_labels.json"
EG = U / "reviews/egypt_occurrence_labels.json"
FIT = U.parent / "word100_autonomous_localization_20261111/FIT_LABELS.json"
EG_FRAMES = U / "reviews/egypt_fit_shape/FRAMES.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT / p


inputs = {str(p): sha(p) for p in (OB, EG, FIT, EG_FRAMES)}
ob = {r["clip_id"]: r for r in read(OB)["clips"]}
eg = {r["clip_id"]: r for r in read(EG)["rows"]}
fit = read(FIT)
relations = [r for r in fit["rows"] + fit["additional"] if isinstance(r, dict)]
eg_images = read(EG_FRAMES)["rows"]
assert {r["record_id"] for r in eg_images} == {1, 4, 7}

# Explicit observations made on the existing sheets during this review.
# Endpoints bracket an interior change, not the whole action's onset/offset.
SPECS = [
    ("WORD0739", 1, "close_chest", 0, 62, 68, [64, 66], [60, 62], [68, 70, 72]),
    ("WORD0739", 1, "raise_forearm", 1, 74, 78, [76], [72, 74], [78, 80, 82]),
    ("WORD0739", 4, "close_chest", 0, 90, 96, [92, 94], [86, 88, 90], [96, 98, 100]),
    ("WORD0739", 4, "raise_forearm", 1, 104, 110, [106, 108], [102, 104], [110, 112, 114]),
    ("WORD0739", 7, "close_chest", 0, 54, 60, [56, 58], [50, 52, 54], [60, 62, 64]),
    ("WORD0739", 7, "raise_forearm", 1, 66, 70, [68], [64, 66], [70, 72, 74]),
    ("WORD2523", 1, "spread_hands", 3, 39, 43, [41], [37, 39], [43, 45, 49]),
    ("WORD2523", 4, "spread_hands", 3, 87, 97, [90, 94], [83, 87], [97, 101, 104]),
    ("WORD2523", 7, "spread_hands", 3, 45, 50, [48], [42, 45], [50, 53, 56]),
]
all_images = {}
rows = []
for word, person, clause, axis, left, right, middle, before, after in SPECS:
    clip = f"NIA_SL_{word}_REAL{person:02}_F"
    source = ob[clip] if word == "WORD0739" else eg[clip]
    assert (source.get("split") == "FIT" if word == "WORD0739" else source["role"] == "fit")
    observed = sorted(set(before + middle + after + [left, right]))
    rgb = []
    if word == "WORD0739":
        occurrence = next(o for o in source["occurrences"] if o.get("guide_clause") == clause)
        reviewed = set()
        for panel in source["review_images"]:
            if "_dense" not in panel["path"]:
                continue
            p = absolute(panel["path"])
            assert sha(p) == panel["sha256"]
            item = {"path": str(p), "sha256": panel["sha256"], "source_frame_ids": panel["frames"],
                    "reviewed_now": True, "kind": "existing_RGB_contact_sheet"}
            all_images[str(p)] = item
            rgb.append(item)
            reviewed.update(panel["frames"])
        assert set(observed).issubset(reviewed)
        explicitly_observed = {s for o in source["occurrences"] for a in o["direct_observations"] for s in a["frames"]}
        assert set(observed).issubset(explicitly_observed)
        boundaries = {k: occurrence[k] for k in ("start_boundary_uncertainty_frames", "end_boundary_uncertainty_frames", "movement_end_boundary_uncertainty_frames") if k in occurrence}
        evidence = occurrence["direct_observations"]
        occurrence_id = occurrence["occurrence_id"]
        native_images = []
    else:
        occurrence = next(o for o in source["episodes"] if o.get("clause_id") == clause)
        occurrence_id = occurrence["occurrence_id"]
        seq = U / "reviews/egypt_fit_shape" / f"REAL{person:02}_sequence.png"
        all_native = [r for r in eg_images if r["clip_id"] == clip and r["image"].endswith("_hands.png")]
        reviewed = sorted({r["frame_id"] for r in all_native})
        assert set(observed).issubset(reviewed)
        item = {"path": str(seq), "sha256": sha(seq), "source_frame_ids": reviewed,
                "reviewed_now": True, "kind": "existing_RGB_hand_sequence_sheet",
                "hash_basis": "Pinned on this review; underlying existing native frame hashes crosschecked with FRAMES.json"}
        all_images[str(seq)] = item
        rgb.append(item)
        native_images = []
        for r in all_native:
            if r["frame_id"] not in observed:
                continue
            assert r["source_sha256"] == source["source_sha256"]
            assert sha(Path(r["image"])) == r["image_sha256"]
            native_images.append({"path": r["image"], "sha256": r["image_sha256"],
                                  "source_frame_id": r["frame_id"], "viewed_as_tile_in": str(seq),
                                  "native_file_reopened_for_display": False})
        boundaries = {k: occurrence[k] for k in ("onset_frame_range", "offset_frame_range", "shape_change_end_frame_range", "separation_settling_end_frame_range") if k in occurrence}
        evidence = {"visible_features": occurrence["visible_features"], "confirmed_core_frames": occurrence["confirmed_core_frames"]}

    fits = [r for r in relations if r["clip_id"] == clip and r["axis"] == axis and r["moving"] == "R" and r["anchor"] == "L"]
    assert len(fits) <= 1
    relation = fits[0] if fits else None
    if relation:
        assert relation["role"] == "fit" and relation["source_sha256"] == source["source_sha256"]
        assert len(relation["events"]) == 1
        event_index = 0
        event = relation["events"][0]
        interior = [event["onset"][1], event["offset"][0]]
        compatible = interior[0] <= left < right <= interior[1]
    else:
        event_index = event = interior = None
        compatible = False
    assert left < right and middle and all(left < s < right for s in middle)
    proposed_removed = list(range(left + 1, right))
    assert set(middle).issubset(proposed_removed)
    middle_band_membership = {
        str(s): [key for key, span in boundaries.items() if span[0] <= s <= span[1]]
        for s in middle
    }
    if clause == "close_chest":
        before_text = "Right hand is earlier/higher in the same face-to-chest movement, with closing in progress; this is not an exact onset claim."
        middle_text = "Across the directly viewed middle samples, the right hand descends and the forearm becomes more horizontal. The sequence shows progress, not a hold-only sample."
        after_text = "Later samples are lower/closer to the chest-height near-horizontal posture; boundary-band samples are not exact motion-end labels."
        unknown = ["Individual curled fingers and face contact can be self-occluded; exact articulation/contact unknown.", "Sparse every-two-frame sheets do not establish continuous velocity or the unviewed intervening frames."]
    elif clause == "raise_forearm":
        before_text = "Closed right hand/forearm is earlier and more horizontal in the same reorientation."
        middle_text = "Intermediate forearm orientation differs visibly from both earlier and later samples. Elbow/forearm change is the evidence; wrist displacement alone is not required."
        after_text = "Forearm is more upright and later held; the held samples are context, not additional motion or a complete-action label."
        unknown = ["Closed hand hides individual finger joints; exact finger articulation and three-dimensional forearm angle unknown.", "Middle witness can be only one sampled intermediate frame; it proves a visible sequence state, not continuous per-frame motion."]
    else:
        before_text = "Both hands are closer in the central roof configuration, with wider projected thumb/finger-bundle opening."
        middle_text = "Both hands separate/lower and the projected thumb aperture narrows across the viewed sequence. This is not evidence for a fist or extension of every individual finger."
        after_text = "Later samples show greater hand separation and narrower projected aperture; the later persistent configuration is a hold, not further motion."
        unknown = ["Finger bundles overlap; exact individual joint flexion, palm depth/orientation, fingertip contact and any fully hidden finger are unknown.", "The hand crop supports projected separation/aperture change, not whole-body or whole-sign completion."]
    rows.append({
        "clip_id": clip, "word_id": word, "person": person, "split": "FIT",
        "source": source["source"], "source_sha256": source["source_sha256"],
        "source_hash_basis": "Existing public occurrence metadata; matched existing FIT relation hash where present. Video bytes were not reopened.",
        "frame_origin": 0, "fps": source["fps"], "source_indices_provenance_only": True,
        "occurrence_id": occurrence_id, "guide_clause": clause,
        "axis": axis, "moving": "R", "anchor": "L", "event_index": event_index,
        "existing_fit_event": event, "existing_fit_interior_core": interior,
        "existing_fit_relation_found": relation is not None,
        "left_source_frame": left, "right_source_frame": right,
        "observed_removed_frames": middle,
        "before_source_frames": before, "middle_source_frames": middle, "after_source_frames": after,
        "all_source_frames_directly_reviewed_for_this_row": observed,
        "before_middle_after": [
            {"phase": "before", "source_frame_ids": before, "observation": before_text},
            {"phase": "middle", "source_frame_ids": middle, "observation": middle_text},
            {"phase": "after", "source_frame_ids": after, "observation": after_text},
        ],
        "reviewed_mid_motion": True,
        "reviewed_mid_motion_meaning": "Visible temporal progression across sampled RGB states within the same occurrence; not a dense velocity, phase, correctness, or completion label.",
        "rgb_evidence": {"path": rgb[0]["path"], "sha256": rgb[0]["sha256"],
                         "additional_contact_sheets": rgb[1:]},
        "rgb_evidence_all": rgb, "native_frame_hash_references": native_images,
        "inherited_boundary_bands": boundaries, "middle_witness_boundary_membership": middle_band_membership,
        "inherited_occurrence_evidence": evidence,
        "proposed_source_edit": {
            "operation": "If separately admitted, omit the open interval (left_source_frame,right_source_frame) from raw input; preserve source order of retained frames.",
            "would_remove_source_frame_ids": proposed_removed,
            "reviewed_mid_motion_frames_that_would_be_removed": middle,
            "removed_frames_without_new_motion_label": [s for s in proposed_removed if s not in middle],
            "generated": False, "generated_sample_source_index_map_verified": False,
            "proof_limit": "This definition identifies a source edit if executed and its actual index map is checked. It does not prove an observed error sign, natural omission, missing completion, or distinguishability from fast normal motion.",
        },
        "fits_conservative_existing_event_interior": compatible,
        "mapping_deferral": ("MISSING_EXISTING_FIT_RELATION_NO_EVENT_LABEL_INVENTED" if relation is None else
                             "NO_REVIEWED_BEFORE_MIDDLE_AFTER_TRIPLE_INSIDE_EXISTING_CORE_46_50" if not compatible else None),
        "observational_identifiability": "unknown",
        "could_look_like_fast_normal": True,
        "fast_normal_reason": "An omitted short interior span can look like faster motion or temporal undersampling. No matched natural-fast control or blinded visual discrimination of a generated sample was reviewed.",
        "hand_regions_visible_in_listed_samples": True,
        "occlusion_and_sampling_unknowns": unknown,
        "semantic_complete": None, "physical_complete": None, "wrong_sign_label": None,
        "dense_motion_labels_created": False, "main_quality_target_from_edit": None,
        "admission": "not_granted_observational_identifiability_unknown",
        "aux_mask_for_current_comparison": 0,
    })

# One context sheet was also directly displayed. It does not supply a missing
# mid-action frame and is not used to widen any witness or boundary annotation.
context_sheet = U / "reviews/egypt_occurrence/REAL01_boundary_sheet.png"
all_images[str(context_sheet)] = {
    "path": str(context_sheet), "sha256": sha(context_sheet),
    "source_frame_ids": [23, 25, 27, 29, 31, 33, 53, 55, 57, 59, 61, 63, 65, 67],
    "reviewed_now": True, "kind": "existing_context_only_RGB_contact_sheet",
    "used_to_add_motion_witness": False,
}
assert len(rows) == 9
assert len({r["clip_id"] for r in rows}) == 6
assert sum(r["fits_conservative_existing_event_interior"] for r in rows) == 5
assert sum(not r["existing_fit_relation_found"] for r in rows) == 3
assert all(sha(Path(p)) == h for p, h in inputs.items())
assert all(sha(Path(p)) == r["sha256"] for p, r in all_images.items())
out = {
    "status": "FIT_WITNESSES_REVIEWED_AUX_ADMISSION_WITHHELD",
    "created_at": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "reviewer": "Codex sequence_target_review agent; direct existing-image review, not independent human/KSL expert gold",
    "scope": {"clips": 6, "dynamic_occurrence_rows": 9, "existing_image_sheets_reviewed": len(all_images),
              "new_model_or_DEV_outputs_read_for_this_task": False, "private_or_mentor_data": False,
              "source_video_reopened": False, "new_RGB_extraction": False,
              "synthesis_generated": False, "training_ran": False, "AUX_ADMISSIONS_created": False,
              "new_dense_labels": False, "existing_source_labels_modified": False},
    "input_label_sha256": inputs,
    "reviewed_images": list(all_images.values()),
    "summary": {
        "observed_dynamic_sequences": 9, "compatible_existing_FIT_interior_windows": 5,
        "missing_existing_fit_relation_rows": 3, "insufficient_sampled_triple_inside_existing_core_rows": 1,
        "source_deletions_established_observationally_identifiable": 0,
        "auxiliary_admissions": 0, "current_comparison_aux_mask": 0,
        "basis_for_aux_mask_zero": "All proposed edits have unknown observational identifiability against fast normal motion. Root explicitly agreed to withhold admissions and proceed with the existing extent B/C comparison at aux_mask0. Lack of auxiliary labels is not filled by guessed motion/incorrect-sign/completion targets.",
    },
    "policy": {
        "frame_semantics": "All IDs are zero-based original source frames, not latent indices. before/after bracket a witnessed interior change and do not claim exact action onset/completion.",
        "sampled_only": "Only the listed actually viewed samples support observations. Source boundary ranges are uncertainty brackets, not dense labels; no unseen intermediate frame is promoted to moving gold.",
        "source_edit_vs_observation": "Knowing an index sequence was edited is provenance. It does not establish that video-only input can discriminate the edit from natural fast motion or low-frame-rate observation.",
        "no_automatic_admission": "RGB review is not AUX_ADMISSIONS. These rows withhold auxiliary labels, main quality negative labels, extent targets, semantic labels, and completion labels from the proposed edits.",
        "no_static_formation_inference": "Obedience open-front hold and Egypt central-roof posture are not added as dynamic witnesses. Their exact formation is entangled with preparation or static presentation; seeing the held posture alone does not establish an interior dynamic deletion target.",
        "current_experiment": "Use aux_mask0 for the current extent B/C comparison. Do not infer extra labels or adjust boundaries to fill the missing auxiliary supervision.",
    },
    "rows": rows,
}
destination = U / "local_motion_quality/AUX_WITNESS_REVIEW.json"
destination.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"path": str(destination), "sha256": sha(destination), "rows": len(rows), "compatible_windows": 5, "aux_admissions": 0}))
