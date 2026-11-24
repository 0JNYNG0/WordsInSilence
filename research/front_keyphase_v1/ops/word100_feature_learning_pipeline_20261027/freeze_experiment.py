"""Freeze an external manual-label feasibility experiment before model scores exist."""
from pathlib import Path
from collections import Counter
import hashlib
import json

U = Path(__file__).resolve().parent
def read(p): return json.loads(p.read_text(encoding="utf-8-sig"))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out = U / "EXPERIMENT.json"
    assert not out.exists()
    source = U / "data/DEEPHAND_INVENTORY.json"
    joinpath = U / "research/DEEPHAND_SIGNER_JOIN.json"
    d, j = read(source), read(joinpath)
    assert digest(source) == j["inventory_sha256"] and j["joined"] == 3359
    signer = {r["image_id"]: r["recorded_signer_id"] for r in j["rows"]}
    rows = sorted(d["rows"], key=lambda r: r["id"])
    classes = sorted({r["label"] for r in rows})
    for r in rows:
        r["signer"] = signer[r["id"]]
        n = int(r["signer"][-2:])
        r["role"] = "fit" if n <= 5 else "validation" if n <= 7 else "test"
        r["target"] = classes.index(r["label"])
    idx = {(r["clip"], r["original_frame_index"]): i for i, r in enumerate(rows)}
    assert len(idx) == len(rows)
    for r in rows:
        r["context_indices"] = [idx.get((r["clip"], r["original_frame_index"] + off), -1)
                                for off in (-2, -1, 0, 1, 2)]
        assert all(rows[i]["role"] == r["role"] for i in r["context_indices"] if i >= 0)
    summaries = {}
    for role in ("fit", "validation", "test"):
        rr = [r for r in rows if r["role"] == role]
        summaries[role] = {"images": len(rr), "signers": sorted({r["signer"] for r in rr}),
             "clips": len({r["clip"] for r in rr}), "class_counts": dict(Counter(r["label"] for r in rr)),
             "with_context": sum(sum(i >= 0 for i in r["context_indices"]) > 1 for r in rr),
             "near_different_label": sum(any(i >= 0 and rows[i]["label"] != r["label"]
                                         for i in r["context_indices"]) for r in rr)}
    assert not d["duplicate_pixels"]
    for key in ("signer", "clip", "pixel_sha256"):
        groups = [{r[key] for r in rows if r["role"] == role} for role in summaries]
        assert all(not groups[a] & groups[b] for a, b in [(0,1),(0,2),(1,2)])
    config = {
      "status": "FROZEN_BEFORE_FEATURE_ACCURACY_OR_TRAINING",
      "scope": "Research-only external physical handshape feasibility, not KSL accuracy or educational correctness",
      "source": "Deep Hand manually annotated PHOENIX2014 MS dev images repurposed; not official benchmark split",
      "inventory_sha256": digest(source), "join_sha256": digest(joinpath),
      "preparation_sha256": digest(U / "FROZEN_PREPARATION.json"),
      "classes": classes, "rows": rows, "summary": summaries,
      "split_rule": "Official signer01-05 fit,06-07 validation,08-09 test; source frames never cross roles",
      "seeds": [27101, 27102, 27103],
      "encoder_candidates": ["mobilenet_v3_small", "mobilenetv4_conv_small", "tinynext_t"],
      "encoder_selection": "Mean validation macro-F1 on supported validation classes; within 1 percentage point choose lower CPU p95; never select from test",
      "input_geometry": "Entire supplied hand crop letterboxed to224x224 with RGB128 background, no finger-removing center crop; model-specific official numeric normalization",
      "encoder_frozen": True, "encoder_threads": 4, "encoder_batch": 32,
      "probe": {"type": "L2-normalized frozen embedding to45-class linear head", "epochs": 80,
                "batch": 64, "lr": 0.003, "weight_decay": 0.01, "optimizer": "AdamW",
                "selection_epoch": "last fixed epoch; no best-epoch test lookup"},
      "modalities": ["J_T", "R_T", "RJ_F", "RJ_T"],
      "temporal": {"offsets_original_frames": [-2,-1,0,1,2], "source_fps": 25,
                   "relative_seconds": [-.08,-.04,0,.04,.08],
                   "missing_neighbors": "zero features plus explicit availability mask; no interpolation/duplicate stills",
                   "target": "Only author-labelled center frame; neighboring labels never model inputs",
                   "architecture": "per-slot projection128, two temporal Conv1d kernels3, center logits45; RJ_F same projection with current-frame MLP",
                   "epochs": 60, "batch": 64, "lr": 0.001, "weight_decay": 0.01,
                   "event_limit": "Sparse adjacent images do not establish complete motion or semantic stage boundaries"},
      "joints": {"input": "Current crop MediaPipe IMAGE detection, wrist-relative normalized by max wrist-to-MCP distance, observed mask; all missing cases remain denominator",
                 "truth": "Only manual source class is a target; automatic coordinates are inputs only"},
      "metrics": ["top1/denominator", "macro-F1 present classes", "per-class support/precision/recall",
                  "per-signer accuracy", "near different-label frame accuracy", "missing detector subgroup",
                  "CPU inference p50/p95; encoder-only, not full webcam FPS"],
      "test_policy": "No test predictions until all heads/conditions and encoder choice are frozen; each checkpoint tested once; no test-driven repair/tuning",
      "ksl_transfer": "Read-only public36-window descriptive diagnostics only, draft agreement not independent correctness; no user recordings",
      "feedback_and_web_gate": "No new educational correction or operating replacement unless independent KSL variation/error and visibility evidence passes",
      "unsupported_class_policy": "Keep all45 labels and all3359 rows; report missing fit support separately, never silently remove",
      "license": "CC BY-NC-SA3.0 research-only isolated artifacts; not shipping asset",
      "production_changed": False, "user_mentor_data_used": False,
    }
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=False))

if __name__ == "__main__": main()
