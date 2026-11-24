"""Read-only saved-evidence verification; no model loading, inference or input edits."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
LAB = ROOT / "research/front_keyphase_v1"
OPS = LAB / "ops"
OUT = Path(__file__).resolve().parent


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {
        "boundary": OPS / "observation_preserving_repair_20261003/BOUNDARY_AUDIT.json",
        "metadata": OPS / "context_training_pair_20260921/feature_metadata.json",
        "manifest": ROOT / "research/sign_model_pilot_v1/expand/manifest.json",
        "source_annotations": OPS / "native_context_readiness_20260921/source_annotations.json",
        "scores": OPS / "observation_preserving_repair_20261003/result/SCORES_native.npz",
        "historical_eval": LAB / "runs/partial_encoder_pair_v1/evaluation.json",
    }
    errors = read_json(paths["boundary"])["native_clean_errors"]
    error_ids = {e["id"] for e in errors}
    labels = {e["true"] for e in errors}
    meta = read_json(paths["metadata"])
    manifest = {r["sample_id"]: r for r in read_json(paths["manifest"])}
    source = read_json(paths["source_annotations"])
    rows = []
    temporal = []
    source_hashes = {}
    with np.load(paths["scores"], allow_pickle=False) as scores:
        score_ids = scores["ids"].tolist()
        classes = scores["classes"].tolist()
        cases = scores["cases"].tolist()
        arms = scores["arms"].tolist()
        for r in meta["annotation_dev"]:
            if r["label"] not in labels:
                continue
            i = score_ids.index(r["id"])
            m = manifest[r["id"]]
            a = source[m["clip_id"]]["document"]["data"][m["annotation_index"]]
            p = Path(r["path"])
            source_hashes[str(p)] = sha(p)
            with np.load(p, allow_pickle=False) as z:
                fr = z["frame_index"]
                ts = z["timestamp_nominal_us"]
                pr = z["coordinate_present"]
                row = {
                    "id": r["id"], "label": r["label"], "signer": r["signer_id"],
                    "source_path": str(p), "is_baseline_error": r["id"] in error_ids,
                    "correct": int(scores["logits"][i, 0, 0].argmax()) == int(scores["true_indices"][i]),
                    "frames": len(fr), "frame_first": int(fr[0]), "frame_last": int(fr[-1]),
                    "start_s": m["start_s"], "end_s": m["end_s"],
                    "all_67_points_present": bool(pr.all()),
                    "all_hand_points_present": bool(pr[:, 25:].all()),
                    "frame_step_one": bool(np.all(np.diff(fr) == 1)),
                    "nominal_dt_us_values": np.unique(np.diff(ts)).tolist(),
                    "sha_matches_metadata_and_manifest": source_hashes[str(p)] == r["sha256"] == m["npz_sha256"],
                    "label_and_window_match_source_annotation": m["word_label"] == a["attributes"][0]["name"] and m["start_s"] == a["start"] and m["end_s"] == a["end"],
                    "frames_inside_nominal_annotation_window": bool(np.all(fr / 30 >= m["start_s"]) and np.all(fr / 30 <= m["end_s"])),
                    "timestamps_match_nominal_frame_time": bool(np.array_equal(ts, np.round(fr / 30 * 1e6).astype("int64"))),
                    "source_schema": m["source_schema"], "timebase": m["timebase"],
                    "semantic_variant_verified": m["semantic_variant_verified"],
                }
            rows.append(row)
            if r["id"] in error_ids:
                temporal.append({"id": r["id"], "true": r["label"], "predictions": {
                    case: {arm: classes[int(scores["logits"][i, ci, ai].argmax())] for ai, arm in enumerate(arms)}
                    for ci, case in enumerate(cases)
                }})
    label_summary = []
    for label in sorted(labels):
        rr = [r for r in rows if r["label"] == label]
        tr = [r for r in meta["base"] if r["label"] == label]
        label_summary.append({
            "label": label, "development_correct": sum(r["correct"] for r in rr),
            "development_errors": sum(not r["correct"] for r in rr),
            "correct_frame_counts": [r["frames"] for r in rr if r["correct"]],
            "error_frame_counts": [r["frames"] for r in rr if not r["correct"]],
            "base_train_rows": len(tr), "known_train_signers": sorted({r["signer_id"] for r in tr if "signer_id" in r}),
            "train_rows_without_signer_id": sum("signer_id" not in r for r in tr),
        })
    old = read_json(paths["historical_eval"])
    historical = [r for r in old["rows"] if r["id"] in error_ids or r["id"] == "NIA_SL_SEN0355_REAL11_F"]
    checks = ["all_67_points_present", "all_hand_points_present", "frame_step_one", "sha_matches_metadata_and_manifest", "label_and_window_match_source_annotation", "frames_inside_nominal_annotation_window", "timestamps_match_nominal_frame_time"]
    result = {
        "nature": "Independent saved-evidence check; existing public development inputs only",
        "inference": 0, "training": 0, "user_data": 0, "reserved_arrays": 0, "input_changes": 0,
        "summary": {"rows": len(rows), "correct_controls": sum(r["correct"] for r in rows), "baseline_errors": sum(not r["correct"] for r in rows), "label_count": len(labels), "all_checks": {k: all(r[k] for r in rows) for k in checks}},
        "rows": rows, "per_label": label_summary, "error_temporal_predictions": temporal,
        "historical_832_rows": [{"group": r["group"], "id": r["id"], "true": r["true"], "full93": r["models"]["control"]["full93"]} for r in historical],
        "input_hashes": {k: {"path": str(p), "sha256": sha(p)} for k, p in paths.items()},
        "public_npz_hashes": source_hashes,
        "limits": ["Mask true denotes stored presence, not verified detection accuracy or absence of occlusion.", "Source annotation agreement does not establish semantic variant or true video boundary accuracy.", "All timestamps are nominal frame/30, not measured sensor PTS.", "Same-label controls and errors are already examined development data, not an independent validation set.", "Saved model predictions are reused. No original video inspected or tracking identity ground truth inferred."],
    }
    assert len(rows) == 56 and sum(r["correct"] for r in rows) == 48
    assert all(result["summary"]["all_checks"].values())
    (OUT / "SAVED_EVIDENCE_CHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
