"""Prepare source-backed feature cards and explicit evidence gates, without inference."""
from pathlib import Path
from collections import Counter
import datetime
import hashlib
import json

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[3]
OPS = UNIT.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    path = UNIT / name
    if path.exists():
        raise FileExistsError(f"Preserve previous result: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    profile_path = OPS / "word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json"
    sequence_path = OPS / "word100_shape_sequence_bundle_20261027/BUNDLE.json"
    profiles, bundle = read(profile_path), read(sequence_path)
    protected = read(OPS / "word100_face_relation_alignment_20261027/FROZEN.json")["protected"]
    assert all(digest(Path(p)) == h for p, h in protected.items()), "PROTECTED_FILE_CHANGED"
    cards = []
    families = Counter()
    for profile in profiles["profiles"]:
        entries = []
        for entry in profile["candidate_entries"]:
            claims = []
            for claim in entry.get("claims", []):
                families[claim["family"]] += 1
                claims.append({
                    "family": claim["family"], "source_text": claim.get("text"),
                    "source_spans": claim.get("spans", []), "source": claim["source"],
                    "meaning": "source statement, not a frame label or universal acceptance rule",
                    "required_for_this_video": None, "accepted_tolerance": None,
                    "linked_video_interval": None, "physical_label_verified": False,
                    "semantic_rule_verified": False,
                })
            entries.append({
                "source_id": entry["source_id"], "entry_id": entry["entry_id"],
                "entry_label": entry["entry_label"], "source_url": entry["url"],
                "description": entry.get("description"), "feature_claims": claims,
                "video_variant_verified": False,
            })
        cards.append({"word_id": profile["word_id"], "display_label": profile["display_label"],
                      "source_variants": entries, "educational_verdict_enabled": False,
                      "search_absence_is_not_dictionary_absence": True})
    write("SOURCE_FEATURE_CARDS.json", {
        "status": "SOURCE_ORGANIZATION_ONLY_NOT_VERIFIED_KSL_RULES",
        "profiles": cards, "counts": {"profiles": len(cards),
            "with_sources": sum(bool(x["source_variants"]) for x in cards),
            "source_entries": sum(len(x["source_variants"]) for x in cards),
            "feature_claims": sum(families.values()), "by_family": dict(families)},
        "source_sha256": digest(profile_path),
    })
    contract = {
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
        "objective": "Learn word-independent visible physical hand features, preserve order and uncertainty, then validate reference feedback.",
        "first_physical_targets": ["index_only", "open_four", "fist_if_independent_label_available"],
        "no_inference_of_fist_from_two_zeros": True,
        "source_statements_are_not_frame_ground_truth": True,
        "word_hint_for_observation_model": False,
        "TCN": {"frozen": ["C", "T3-F"], "use": "parallel whole-word evidence only",
                "classification_score_is_not_execution_correctness": True},
        "conditions": [
            {"id": "J_T", "inputs": "joints + real timestamps + observation masks + temporal head"},
            {"id": "R_T", "inputs": "RGB hand features + real timestamps + temporal head"},
            {"id": "RJ_F", "inputs": "RGB + joints + masks, current frame only"},
            {"id": "RJ_T", "inputs": "RGB + joints + masks + real timestamps + temporal head"},
        ],
        "model_screen": {"release_date_is_not_selection_metric": True,
            "inspect_recent_primary_sources": True,
            "criteria": ["valid labels", "per-feature precision/recall", "unknown/coverage",
                         "short-transition preservation", "signer-held-out transfer", "CPU p50/p95 latency",
                         "parameter/model memory", "source/weight terms", "reproducible local loading"],
            "no_final_test_tuning": True, "no_operating_default_replacement": True},
        "data": {"exclude": ["user recordings", "mentor recordings", "prior additional199/200 final evaluations"],
            "existing_ai_drafts": {"frames": bundle["frames"], "feature_values": bundle["counts"]["feature_samples"],
                "independent_truth": False, "training_ready": False,
                "use": "input contract and descriptive transfer diagnostics only"},
            "splitting": "group by recorded person and original source before samples; keep continuous windows intact",
            "automated_landmarks": "possible input/weak teacher, never independent physical truth",
            "public_gesture_class": "possible source-domain target, not KSL grammatical correctness",
            "synthetic_missing": "technical robustness test, not actual occlusion recovery accuracy"},
        "gates": {
            "physical_training": ["source and annotation provenance", "independent physical labels for supervised targets",
                "positive and negative examples", "no person/source overlap", "frozen evaluation and selection protocol"],
            "temporal_comparison": ["real ordered clips and timestamps", "same input spans for four conditions",
                "independently supported visible states and changes", "do not copy still images to claim real motion validation"],
            "dtw_feedback": ["physical feature validation", "same-phase alignment evidence",
                "keep reference-required denominator during missing input", "multiple reference variants kept separate",
                "normal variation and verified errors evaluated separately", "no inferred contact from 2D overlap alone"],
            "web_candidate": ["offline comparison passes", "normal/crossed-hand regressions reported",
                "observed/inferred/unknown displayed separately", "no correctness claim outside validated scope"],
        },
        "protected": protected,
        "inputs": {str(profile_path): digest(profile_path), str(sequence_path): digest(sequence_path)},
        "training_started": False,
    }
    write("FROZEN_PREPARATION.json", contract)
    queue = []
    for seq in bundle["sequences"]:
        queue.append({"case_id": seq["case_id"], "clip_id": seq["clip_id"],
            "recorded_person_id": seq["recorded_person_id"], "role": seq["role"], "side": seq["side"],
            "window_only": True, "frames": [{"frame_id": f["frame_id"], "pts_us": f["pts_us"],
                "image": f["full_image"], "focus_image": f["focus_image"],
                "physical_labels": {"index_only": None, "open_four": None, "fist": None},
                "reviewer": None, "label_provenance": "UNREVIEWED"} for f in seq["frames"]]})
    write("INDEPENDENT_LABEL_TEMPLATE.json", {"purpose": "blank physical-feature review, no AI answers copied",
        "source": str(sequence_path), "source_sha256": digest(sequence_path), "windows": queue})
    print(json.dumps({"profiles": len(cards), "claims": sum(families.values()),
        "blank_review_windows": len(queue), "protected_hashes": len(protected),
        "training_started": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
