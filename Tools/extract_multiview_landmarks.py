"""
Extract MediaPipe landmarks from 3-view sign language videos.
Outputs raw landmark JSON per view (same format as s04_out_v3a).

Usage:
  python extract_multiview_landmarks.py --sign-id SIGN001_THANKS
  python extract_multiview_landmarks.py --sign-id SIGN003_LOVE
"""

import argparse
import cv2
import hashlib
import json
import math
import os
import sys
import time

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

VIEWS = ["front", "left45", "right45"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def lm_to_list(lm):
    return [lm.x, lm.y, lm.z]


def lm_to_list_vis(lm):
    vis = getattr(lm, "visibility", None)
    return [lm.x, lm.y, lm.z, float(vis) if vis is not None else None]


def extract_view(view_name, video_path):
    """Full VIDEO-mode extraction: Pose + Hand, normalized + world."""
    actual_sha = sha256_file(video_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    frames = []
    fi = 0
    t0 = time.time()
    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            source_pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            mp_ts = int(fi * 1000 / fps)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hr = hand_lm.detect_for_video(mp_img, mp_ts)
            pr = pose_lm.detect_for_video(mp_img, mp_ts)

            frame_rec = {
                "fi": fi,
                "source_pts_ms": round(source_pts_ms, 3),
                "mp_ts_ms": mp_ts,
            }

            if pr and pr.pose_landmarks:
                lms = pr.pose_landmarks[0]
                frame_rec["pose_norm"] = [lm_to_list_vis(lm) for lm in lms]
                if pr.pose_world_landmarks:
                    frame_rec["pose_world"] = [lm_to_list_vis(lm)
                                               for lm in pr.pose_world_landmarks[0]]
                else:
                    frame_rec["pose_world"] = None
            else:
                frame_rec["pose_norm"] = None
                frame_rec["pose_world"] = None

            hands_list = []
            if hr and hr.hand_landmarks:
                for i, lms in enumerate(hr.hand_landmarks):
                    cat = hr.handedness[i][0].category_name
                    score = hr.handedness[i][0].score
                    h_norm = [lm_to_list(lm) for lm in lms]
                    h_world = None
                    if hr.hand_world_landmarks and i < len(hr.hand_world_landmarks):
                        h_world = [lm_to_list(lm)
                                   for lm in hr.hand_world_landmarks[i]]
                    hands_list.append({
                        "handedness": cat,
                        "score": round(float(score), 4),
                        "norm": h_norm,
                        "world": h_world,
                    })
            frame_rec["hands"] = hands_list if hands_list else None

            frames.append(frame_rec)
            fi += 1
            if fi % 200 == 0:
                elapsed = time.time() - t0
                eta = elapsed / fi * (total - fi)
                print(f"  {view_name}: {fi}/{total} ({fi*100//total}%) ETA {eta:.0f}s")

    cap.release()
    hand_count = sum(1 for f in frames if f["hands"])
    pose_count = sum(1 for f in frames if f["pose_norm"])
    elapsed = time.time() - t0
    print(f"  {view_name} done: {len(frames)} frames, "
          f"hand={hand_count} pose={pose_count} ({elapsed:.1f}s)")

    return {
        "view": view_name,
        "video_path": video_path,
        "sha256_actual": actual_sha,
        "resolution": [w, h],
        "total_frames_decoded": len(frames),
        "hand_detected_frames": hand_count,
        "pose_detected_frames": pose_count,
        "extraction_models": {
            "hand": os.path.basename(HAND_MODEL),
            "pose": os.path.basename(POSE_MODEL),
            "mode": "VIDEO",
            "num_hands": 2,
        },
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sign-id", required=True, help="e.g. SIGN001_THANKS")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip extraction, reuse cached JSONs")
    args = parser.parse_args()

    sign_id = args.sign_id
    out_dir = os.path.join(SCRIPT_DIR, "..", "_multiview_rebuild_v3",
                           f"pipeline_{sign_id}")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multiview Landmark Extraction: {sign_id}")
    print("=" * 60)

    video_files = {}
    for v in VIEWS:
        vf = os.path.join(DATASET_DIR, f"{sign_id}_P01_S01_B01_{v}.mp4")
        if not os.path.exists(vf):
            print(f"ERROR: {vf} not found!")
            sys.exit(1)
        video_files[v] = vf
        cap = cv2.VideoCapture(vf)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {v}: {w}x{h}, {n} frames")

    for v in VIEWS:
        out_path = os.path.join(out_dir, f"landmarks_raw_{v}.json")
        if args.skip_extract and os.path.exists(out_path):
            print(f"\n  {v}: cached, skipping")
            continue

        print(f"\nExtracting {v}...")
        data = extract_view(v, video_files[v])

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved: {out_path} ({size_mb:.1f}MB)")

    print("\n" + "=" * 60)
    print(f"Extraction complete: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
