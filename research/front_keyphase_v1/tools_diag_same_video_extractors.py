"""functional_13: the SAME three videos through OpenPose (stored) and MediaPipe (re-extracted).

Every earlier comparison put a studio OpenPose clip against a MediaPipe clip of a different
recording, so extractor and performer were never apart. These three source videos now exist
locally, and the training material for the same label window was built from them, so the two
extractors can finally be read on identical footage.

What this does, once:
  * re-extracts each of the three videos with the project's own capture.video_source.extract and
    Landmarker, one pass each, into a SEPARATE diagnostic cache;
  * reads the OpenPose side from the training NPZ that was built for the same label window;
  * compares, over that window: right-hand PIP angles, per-hand observation rate, and the
    normalised 51-point input;
  * repeats the angle summary against NIKL_NA_01 to see whether the earlier difference survives.

What it never does: cmd_extract, any production manifest write, model inference, training, a web
change, a new recording, a download, or a second extraction pass. No raw coordinate array, video
or model leaves this process; the report carries aggregates, hashes and code paths only.

A coordinate difference here is NOT evidence about landmark accuracy and NOT evidence about which
hand shape is the correct sign. Projection and detection differences are reported apart from
anything that could be a real difference in what the hands did.

Run: py -3.13 tools_diag_same_video_extractors.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from statistics import median

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

from wis_lab.capture.landmarker import Landmarker  # noqa: E402
from wis_lab.capture.video_source import extract, probe  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402

PILOT = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1")
RECEIPT = ("C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff/"
           "WordsInSilence_기능완성_20260913/source_video_acquisition.json")
CACHE = os.path.join(ROOT, "runs", "cache_functional_13")     # diagnostic only, not the web cache
PIP = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}
HANDS = {"L": 25, "R": 46}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def pip_angles(xy, present, hand="R"):
    """2D interior angle at each PIP, only where MCP/PIP/DIP are all observed."""
    base = HANDS[hand]
    out = {}
    for f, p in PIP.items():
        a, b, c = base + p - 1, base + p, base + p + 1
        ok = present[:, a] & present[:, b] & present[:, c]
        if not ok.any():
            out[f] = None
            continue
        v1 = xy[ok, a] - xy[ok, b]
        v2 = xy[ok, c] - xy[ok, b]
        n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
        good = (n1 > 1e-6) & (n2 > 1e-6)
        if not good.any():
            out[f] = None
            continue
        cos = np.clip((v1[good] * v2[good]).sum(1) / (n1[good] * n2[good]), -1.0, 1.0)
        out[f] = round(float(np.median(np.degrees(np.arccos(cos)))), 1)
    return out


def observation_rate(present):
    n = len(present)
    return {h: {"frames": n,
                "any_point": int(present[:, b:b + 21].any(axis=1).sum()),
                "wrist": int(present[:, b].sum()),
                "rate_any": round(float(present[:, b:b + 21].any(axis=1).mean()), 4)}
            for h, b in HANDS.items()}


def common_input(xy, present, ts):
    from pilot.data import build_channels
    return build_channels(xy, present, ts)


def body_rel_difference(a_clip, b_clip):
    """Per-point distance between the two normalised 51-point inputs, where both are valid.

    Both sides are already divided by their own shoulder width, so this is in body units and is
    free of image scale. It is a DIFFERENCE, not an accuracy measure for either extractor.
    """
    from pilot.data import CH_BODY_XY, CH_HAND_XY, N_POINTS

    n = min(len(a_clip["x"]), len(b_clip["x"]))
    if n == 0:
        return None
    xa = a_clip["x"][:n, :CH_BODY_XY].reshape(n, N_POINTS, 2)
    xb = b_clip["x"][:n, :CH_BODY_XY].reshape(n, N_POINTS, 2)
    va = a_clip["x"][:n, CH_BODY_XY + CH_HAND_XY:CH_BODY_XY + CH_HAND_XY + N_POINTS] > 0.5
    vb = b_clip["x"][:n, CH_BODY_XY + CH_HAND_XY:CH_BODY_XY + CH_HAND_XY + N_POINTS] > 0.5
    both = va & vb
    d = np.linalg.norm(xa - xb, axis=2)
    per_point = []
    for p in range(N_POINTS):
        sel = both[:, p]
        per_point.append(round(float(np.median(d[sel, p])), 4) if sel.any() else None)
    measured = [v for v in per_point if v is not None]
    return {
        "frames_compared": int(n),
        "points_with_any_shared_frame": len(measured),
        "median_over_points": round(float(median(measured)), 4) if measured else None,
        "max_over_points": round(max(measured), 4) if measured else None,
        "body_point_medians": {str(i): per_point[i] for i in range(9)},
        "validity_agreement": round(float((va == vb).mean()), 4),
        "unit": "shoulder widths; both sides normalised by their own body scale",
        "meaning": "a difference between two extractors on the same footage. NOT an accuracy "
                   "figure for either, and not evidence about the sign itself.",
    }


def main() -> int:
    with open(RECEIPT, encoding="utf-8") as fh:
        receipt = json.load(fh)
    with open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8") as fh:
        contract = json.load(fh)
    with open(os.path.join(PILOT, "expand", "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    os.makedirs(CACHE, exist_ok=True)

    rows, extractor_id = [], None
    for vid in receipt["videos"]:
        path = vid["path"]
        clip_id = os.path.splitext(os.path.basename(path))[0]
        sample = next(m for m in manifest
                      if m["clip_id"] == clip_id and m["word_label"] == "나")
        meta = probe(path)
        npz = os.path.join(CACHE, clip_id + ".npz")
        if os.path.exists(npz):
            obs = Observation.load(npz)                    # one extraction pass, ever
            extraction = "cached_from_this_diagnostic"
        else:
            with Landmarker(contract) as lm:
                extractor_id = lm.extractor_id()
                obs = extract(path, lm, clip_id)
            obs.save(npz)
            extraction = "extracted_once"

        from pilot.web_input import to_native_like
        mp_xy, mp_present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                           np.asarray(obs.usable, dtype=bool))
        op = np.load(os.path.join(PILOT, "expand", "npz", os.path.basename(sample["npz"])))
        op_xy, op_present = op["xy_native"], op["coordinate_present"]
        lo, hi = int(sample["first_frame"]), int(sample["last_frame"])

        # The OpenPose index space is 2 frames longer than the decoded video on all three clips,
        # so which decoded frame index 0 is cannot be assumed. Both alignments are reported.
        windows = {"start_aligned": (lo, hi), "shifted_minus_2": (lo - 2, hi - 2)}
        per_alignment = {}
        for name, (a, b) in windows.items():
            a2, b2 = max(a, 0), min(b, len(mp_xy) - 1)
            if b2 < a2:
                per_alignment[name] = {"status": "WINDOW_OUTSIDE_VIDEO"}
                continue
            sub_xy, sub_pr = mp_xy[a2:b2 + 1], mp_present[a2:b2 + 1]
            ts = np.asarray(obs.timestamp_us, dtype=np.int64)[a2:b2 + 1]
            entry = {"mediapipe_rows": [a2, b2],
                     "right_pip_angles": pip_angles(sub_xy, sub_pr),
                     "observation": observation_rate(sub_pr)}
            try:
                mp_clip = common_input(sub_xy, sub_pr, ts)
                op_clip = common_input(op_xy, op_present,
                                       op["timestamp_nominal_us"])
                entry["normalised_input_difference"] = body_rel_difference(op_clip, mp_clip)
            except Exception as exc:                       # noqa: BLE001 - recorded, not hidden
                entry["normalised_input_difference"] = {"status": f"{type(exc).__name__}: {exc}"}
            per_alignment[name] = entry

        rows.append({
            "clip_id": clip_id,
            "video_sha256": vid["sha256"], "video_bytes": vid["bytes"],
            "video_crc32": vid["crc32"], "crc_check": vid["crc_check"],
            "decoded": {"frames": meta["frame_count"],
                        "first_pts_s": vid["decode"]["first_pts_s"],
                        "width": vid["decode"]["width"], "height": vid["decode"]["height"]},
            "openpose_json_frames": None,
            "training_window": {"first_frame": lo, "last_frame": hi,
                                "frames": int(sample["frame_count"]),
                                "label_start_s": sample["start_s"], "label_end_s": sample["end_s"]},
            "mediapipe_extraction": extraction,
            "mediapipe_cache": os.path.relpath(npz, ROOT),
            "mediapipe_observation_sha256": sha256_file(npz),
            "openpose_window": {"right_pip_angles": pip_angles(op_xy, op_present),
                                "observation": observation_rate(op_present)},
            "mediapipe_by_alignment": per_alignment,
        })

    # does the NIKL difference survive re-extraction of real source footage?
    with open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8") as fh:
        nikl_meta = next(c for c in json.load(fh)["clips"] if c["word_id"] == "NA")
    nikl_obs = Observation.load(os.path.join(ROOT, nikl_meta["observation_npz"]))
    from pilot.web_input import to_native_like as tn
    nikl_xy, nikl_pr = tn(np.asarray(nikl_obs.xy_px, dtype=np.float32),
                          np.asarray(nikl_obs.usable, dtype=bool))
    nikl_angles = pip_angles(nikl_xy, nikl_pr)

    out = {
        "task_id": "functional_13_same_video_extractors",
        "question": "on the SAME footage, how far apart are the stored OpenPose observation and a "
                    "fresh MediaPipe observation, and does the earlier NIKL difference survive?",
        "is_not": ["a landmark accuracy measurement", "evidence about which hand shape is the "
                   "correct sign", "a recognition or performance result"],
        "source_receipt": RECEIPT,
        "extractor_id": extractor_id,
        "diagnostic_cache": os.path.relpath(CACHE, ROOT),
        "production_manifest_written": False,
        "cmd_extract_used": False,
        "model_inference": False, "training": False, "new_recording": False, "download": False,
        "clips": rows,
        "nikl_reference_right_pip_angles": nikl_angles,
        "nikl_clip_id": nikl_meta["clip_id"],
    }
    path = os.path.join(ROOT, "evidence", "functional_13_same_video_extractors.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"clips": [{"clip_id": r["clip_id"],
                                 "extraction": r["mediapipe_extraction"],
                                 "openpose": r["openpose_window"]["right_pip_angles"],
                                 "mediapipe_start_aligned":
                                     r["mediapipe_by_alignment"]["start_aligned"].get(
                                         "right_pip_angles"),
                                 "mediapipe_shifted":
                                     r["mediapipe_by_alignment"]["shifted_minus_2"].get(
                                         "right_pip_angles")}
                                for r in rows],
                      "nikl": nikl_angles, "written": os.path.relpath(path, ROOT)},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
