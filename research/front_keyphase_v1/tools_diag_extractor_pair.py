"""functional_16: the fixed TCN on the SAME public clip, through two extractors.

Contract fixed before this ran: design/functional_16_extractor_pair.md - the seven clip ids, the
label windows, the two frame-correspondence assumptions, the checkpoint hash and the candidate
lists are all written there first.

The question is narrow: does the pinned model identify the same recorded movement differently when
it arrives through the stored OpenPose keypoints versus a MediaPipe extraction of the same video?
These clips were in the model's own training material, so nothing here is an accuracy or a
generalisation result, and nothing about a user's performance follows from it.

What it does: reuses the three 나 MediaPipe caches from functional_13, extracts the four remaining
videos ONCE each into a separate diagnostic cache, and runs the pinned checkpoint blind over both
sides of every pair with the SAME feature builder the training used.

What it never does: training, a threshold search, a per-word rule, a model or web change, a camera,
a server restart, or re-inference of the user's ten attempts, the private recordings or the
reserved evaluation set. No coordinate array, video frame or personal data is printed.

Run: py -3.13 tools_diag_extractor_pair.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

from wis_lab.capture.landmarker import Landmarker  # noqa: E402
from wis_lab.capture.video_source import extract  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402
from wis_lab.service import neural  # noqa: E402

PILOT = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1")
EXTERNAL = os.path.join(ROOT, "evidence", "external")
CACHE_13 = os.path.join(ROOT, "runs", "cache_functional_13")      # the three 나, reused
CACHE_16 = os.path.join(ROOT, "runs", "cache_functional_16")      # the four new, one pass each
CHECKPOINT_SHA256 = "56a17a4d0a4795a2463add5572acc0d985c3cb7a1cbaee2853698f106d5b34b5"
CLASSES_SHA256 = "a4c7c31669c7ec8cb10bb1c240770e0eb22726133a7c2deb93110d2e11e696b8"

# fixed in the design document before this ran
SAMPLES = [
    ("NIA_SL_SEN0019_REAL01_F_a00_011", "NA", CACHE_13),
    ("NIA_SL_SEN0020_REAL01_F_a00_011", "NA", CACHE_13),
    ("NIA_SL_SEN0199_REAL01_F_a02_011", "NA", CACHE_13),
    ("NIA_SL_SEN0355_REAL01_F_a00_002", "GAMSAHAMNIDA", CACHE_16),
    ("NIA_SL_SEN0195_REAL01_F_a00_023", "MOREUDA", CACHE_16),
    ("NIA_SL_SEN0196_REAL01_F_a00_046", "ALDA", CACHE_16),
    ("NIA_SL_SEN0189_REAL01_F_a00_063", "JOTA", CACHE_16),
]
ALIGNMENTS = {"start_aligned": 0, "shifted_minus_2": -2}
HAND_SLOTS = slice(25, 67)
L_SHOULDER_SLOT, R_SHOULDER_SLOT = 5, 2          # BODY_25 slots for the common shoulders


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def identify(nat, present, ts, model, classes, true_label):
    """One blind forward pass. The answer key is never an input."""
    import torch
    from pilot.data import build_channels, collate

    if len(nat) < neural.MIN_FRAMES:
        return {"status": "TOO_FEW_FRAMES", "frames": int(len(nat))}
    if not np.all(np.diff(ts) > 0):
        return {"status": "NON_MONOTONIC_TIMESTAMPS"}
    clip = build_channels(nat, present, ts)
    x, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    order = np.argsort(-logit)
    ranked = [classes[int(i)] for i in order]
    idx = {classes[i]: i for i in range(len(classes))}
    restricted = sorted(((w, lab, float(logit[idx[lab]]))
                         for w, lab in neural.PROJECT_OVERLAP.items() if lab in idx),
                        key=lambda r: -r[2])
    observed = int(present[:, HAND_SLOTS].any(axis=1).sum())
    return {
        "status": "OK",
        "frames": int(clip["frames"]),
        "span_s": round(float(ts[-1] - ts[0]) / 1e6, 3),
        "hand_observed_frames": observed,
        "hand_observed_rate": round(observed / len(nat), 4),
        "top1_all80": ranked[0],
        "true_rank_all80": ranked.index(true_label) + 1 if true_label in ranked else None,
        "top1_project5": restricted[0][0] if restricted else None,
        "true_rank_project5": next((i + 1 for i, r in enumerate(restricted)
                                    if r[1] == true_label), None),
    }


def integrity(nat, present, ts, window) -> dict:
    """The minimal guards: nothing fabricated, nothing mirrored, nothing out of the window."""
    ok = present[:, L_SHOULDER_SLOT] & present[:, R_SHOULDER_SLOT]
    shoulder = (float((nat[ok, L_SHOULDER_SLOT, 0] > nat[ok, R_SHOULDER_SLOT, 0]).mean())
                if ok.any() else None)
    finite = np.isfinite(nat).all(axis=2)
    return {
        "frames": int(len(nat)),
        "window_frames": window,
        "frames_match_window": int(len(nat)) == window,
        "timestamps_monotonic": bool(np.all(np.diff(ts) > 0)),
        "left_shoulder_right_of_right_fraction": round(shoulder, 4)
        if shoulder is not None else None,
        "present_without_finite_coordinate": int((present & ~finite).sum()),
        "hand_points_present": int(present[:, HAND_SLOTS].sum()),
    }


def main() -> int:
    started = time.time()
    c = neural._load()                                # noqa: SLF001 - the pinned checkpoint
    if c["status"] != "READY":
        print("model not ready:", c.get("detail"))
        return 1
    model, classes = c["model"], c["classes"]
    got = hashlib.sha256(("|".join(classes)).encode("utf-8")).hexdigest()
    if got != CLASSES_SHA256 or neural.CHECKPOINT_SHA256 != CHECKPOINT_SHA256:
        print("model identity changed; refusing to run")
        return 1

    manifest = {m["sample_id"]: m for m in load_json(
        os.path.join(PILOT, "expand", "manifest.json"))}
    acq = {os.path.splitext(os.path.basename(s["source_path"]))[0]: s
           for s in load_json(os.path.join(EXTERNAL, "four_word_source_acquisition.json"))["selected"]}
    with open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8") as fh:
        contract = json.load(fh)
    os.makedirs(CACHE_16, exist_ok=True)

    from pilot.web_input import to_native_like

    rows, extractions = [], []
    for sample_id, word_id, cache in SAMPLES:
        m = manifest[sample_id]
        clip_id = m["clip_id"]
        label = m["word_label"]
        lo, hi = int(m["first_frame"]), int(m["last_frame"])
        window = int(m["frame_count"])

        npz = os.path.join(cache, clip_id + ".npz")
        if os.path.exists(npz):
            obs = Observation.load(npz)
            how = "reused_functional_13_cache" if cache == CACHE_13 else "reused_this_cache"
        else:
            src = acq[clip_id]["source_path"]
            with Landmarker(contract) as lm:
                obs = extract(src, lm, clip_id)
            obs.save(npz)
            how = "extracted_once"
            extractions.append(clip_id)
        mp_xy, mp_pr = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                      np.asarray(obs.usable, dtype=bool))
        mp_ts = np.asarray(obs.timestamp_us, dtype=np.int64)

        op = np.load(os.path.join(PILOT, "expand", "npz", os.path.basename(m["npz"])))
        op_xy, op_pr, op_ts = (op["xy_native"], op["coordinate_present"],
                               op["timestamp_nominal_us"])

        row = {
            "sample_id": sample_id, "clip_id": clip_id, "word_id": word_id,
            "true_label": label,
            "label_window_s": [round(m["start_s"], 3), round(m["end_s"], 3)],
            "openpose_frames": [lo, hi, window],
            "mediapipe_cache": os.path.relpath(npz, ROOT),
            "mediapipe_cache_sha256": sha256_file(npz),
            "mediapipe_extraction": how,
            "openpose": dict(identify(op_xy, op_pr, op_ts, model, classes, label),
                             integrity=integrity(op_xy, op_pr, op_ts, window)),
            "mediapipe_by_alignment": {},
        }
        for name, shift in ALIGNMENTS.items():
            a, b = max(lo + shift, 0), min(hi + shift, len(mp_xy) - 1)
            if b < a:
                row["mediapipe_by_alignment"][name] = {"status": "WINDOW_OUTSIDE_VIDEO"}
                continue
            sub_xy, sub_pr, sub_ts = mp_xy[a:b + 1], mp_pr[a:b + 1], mp_ts[a:b + 1]
            row["mediapipe_by_alignment"][name] = dict(
                identify(sub_xy, sub_pr, sub_ts, model, classes, label),
                rows=[a, b], integrity=integrity(sub_xy, sub_pr, sub_ts, window))
        rows.append(row)

    def tally(side, align=None):
        n = a80 = a5 = 0
        for r in rows:
            e = r["openpose"] if side == "openpose" else r["mediapipe_by_alignment"][align]
            if e.get("status") != "OK":
                continue
            n += 1
            a80 += int(e["top1_all80"] == r["true_label"])
            a5 += int(e["top1_project5"] == r["word_id"])
        return {"evaluated": n, "top1_all80_equals_label": a80,
                "top1_project5_equals_word": a5, "of": len(rows)}

    changed = {}
    for name in ALIGNMENTS:
        changed[name] = [
            {"clip_id": r["clip_id"], "true_label_word_id": r["word_id"],
             "openpose_all80": r["openpose"].get("top1_all80"),
             "mediapipe_all80": r["mediapipe_by_alignment"][name].get("top1_all80"),
             "openpose_project5": r["openpose"].get("top1_project5"),
             "mediapipe_project5": r["mediapipe_by_alignment"][name].get("top1_project5")}
            for r in rows
            if r["openpose"].get("top1_all80")
            != r["mediapipe_by_alignment"][name].get("top1_all80")
            or r["openpose"].get("top1_project5")
            != r["mediapipe_by_alignment"][name].get("top1_project5")]

    out = {
        "task_id": "functional_16_extractor_pair",
        "design": "design/functional_16_extractor_pair.md",
        "question": "does the pinned TCN identify the SAME recorded movement differently through "
                    "the stored OpenPose keypoints and a MediaPipe extraction of the same video?",
        "is_not": ["an accuracy result", "a generalisation result", "an assessment of any user's "
                   "performance", "a training or promotion decision"],
        "these_clips_were_in_training": True,
        "model": {"checkpoint_sha256": CHECKPOINT_SHA256, "classes": len(classes),
                  "classes_sha256": CLASSES_SHA256,
                  "answer_key_given_to_the_model": False,
                  "feature_builder": "pilot.data.build_channels, the same one training used"},
        "alignment_assumptions": {"both_reported": True, "rule": ALIGNMENTS,
                                  "note": "neither is chosen for looking better"},
        "extraction": {"new_extractions": extractions, "passes_per_video": 1,
                       "reused_from_functional_13": 3,
                       "diagnostic_cache": os.path.relpath(CACHE_16, ROOT),
                       "production_manifest_written": False},
        "tallies": {"openpose": tally("openpose"),
                    **{f"mediapipe_{k}": tally("mediapipe", k) for k in ALIGNMENTS}},
        "label_changed_vs_openpose": changed,
        "rows": rows,
        "no_training": True, "no_threshold_search": True, "no_per_word_rule": True,
        "no_model_change": True, "no_web_change": True, "camera_opened": False,
        "server_restarted": False, "user_attempts_re_inferred": False,
        "raw_arrays_emitted": False,
        "elapsed_s": round(time.time() - started, 1),
    }
    path = os.path.join(ROOT, "evidence", "functional_16_extractor_pair.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"tallies": out["tallies"],
                      "changed_counts": {k: len(v) for k, v in changed.items()},
                      "new_extractions": extractions,
                      "written": os.path.relpath(path, ROOT)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
