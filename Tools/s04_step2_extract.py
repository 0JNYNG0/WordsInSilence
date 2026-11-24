"""
Tools/s04_step2_extract.py -- Phase 1 v3a Step 2
Raw landmark extraction + clap sync detection + review visuals.

Reads video paths from s04_out_v3a/manifest.json.
Outputs to s04_out_v3a/.

Usage:
  python Tools/s04_step2_extract.py
  python Tools/s04_step2_extract.py --skip-extract   (reuse cached landmark JSONs)
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

# ---- paths ----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "s04_out_v3a")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

VIEW_ORDER = ["front", "left45", "right45"]

# ---- helpers ---------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def lm_to_list(lm):
    """Landmark -> [x, y, z]."""
    return [lm.x, lm.y, lm.z]


def lm_to_list_vis(lm):
    """Landmark -> [x, y, z, visibility]."""
    vis = getattr(lm, "visibility", None)
    return [lm.x, lm.y, lm.z, float(vis) if vis is not None else None]


def dist2d(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

# ---- 1. extraction ---------------------------------------------------------

def extract_view(view_name, video_path, expected_sha):
    """Full VIDEO-mode extraction: Pose + Hand, normalized + world.

    Returns dict ready for JSON serialisation.
    No zero-filling -- missing detections stored as null.
    """
    # verify hash
    actual_sha = sha256_file(video_path)
    sha_ok = actual_sha == expected_sha
    if not sha_ok:
        print(f"  [WARN] SHA-256 mismatch for {view_name}!")
        print(f"    expected: {expected_sha}")
        print(f"    actual  : {actual_sha}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            source_pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            mp_ts = int(fi * 1000 / fps)  # monotonic int for MediaPipe

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hr = hand_lm.detect_for_video(mp_img, mp_ts)
            pr = pose_lm.detect_for_video(mp_img, mp_ts)

            frame_rec = {
                "fi": fi,
                "source_pts_ms": round(source_pts_ms, 3),
                "mp_ts_ms": mp_ts,
            }

            # -- pose --
            if pr and pr.pose_landmarks:
                lms = pr.pose_landmarks[0]
                frame_rec["pose_norm"] = [lm_to_list_vis(lm) for lm in lms]
                if pr.pose_world_landmarks:
                    frame_rec["pose_world"] = [lm_to_list_vis(lm) for lm in
                                               pr.pose_world_landmarks[0]]
                else:
                    frame_rec["pose_world"] = None
            else:
                frame_rec["pose_norm"] = None
                frame_rec["pose_world"] = None

            # -- hands --
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
            if fi % 100 == 0:
                print(f"  {view_name}: {fi}/{total} ({fi*100//total}%)",
                      file=sys.stderr)

    cap.release()
    hand_count = sum(1 for f in frames if f["hands"])
    pose_count = sum(1 for f in frames if f["pose_norm"])
    print(f"  {view_name} done: {len(frames)} frames, "
          f"hand={hand_count} pose={pose_count}", file=sys.stderr)

    return {
        "view": view_name,
        "video_path": video_path,
        "sha256_expected": expected_sha,
        "sha256_actual": actual_sha,
        "sha256_match": sha_ok,
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


# ---- 2. clap detection ----------------------------------------------------

def detect_clap_candidates(landmark_data):
    """Detect clap candidates per view using wrist-distance heuristic.

    Returns list of candidate dicts sorted by distance (ascending).
    Uses source_pts_ms for timing -- not frame index or FPS.
    """
    frames = landmark_data["frames"]
    view = landmark_data["view"]

    # build wrist-distance time series
    wrist_dists = []  # (fi, source_pts_ms, dist)
    for f in frames:
        if not f["hands"] or len(f["hands"]) < 2:
            continue
        # wrist = landmark 0 of each hand (normalized coords)
        h0 = f["hands"][0]["norm"][0]
        h1 = f["hands"][1]["norm"][0]
        d = dist2d(h0[0], h0[1], h1[0], h1[1])
        wrist_dists.append((f["fi"], f["source_pts_ms"], d))

    if not wrist_dists:
        return {"view": view, "candidates": [], "error": "no two-hand frames"}

    dists_arr = np.array([wd[2] for wd in wrist_dists])

    # Find local minima in first 30% of video (clap should be early)
    cutoff = int(len(dists_arr) * 0.30)
    if cutoff < 5:
        cutoff = len(dists_arr)

    # Smoothing (small window to avoid shifting peaks)
    kernel = np.ones(7) / 7
    smooth = np.convolve(dists_arr[:cutoff], kernel, mode="same")

    # Find local minima: lower than neighbours by at least min_dist frames apart
    from scipy.signal import find_peaks
    # invert to find minima as peaks
    peaks, props = find_peaks(-smooth, prominence=0.01, distance=15)

    candidates = []
    for pk in peaks:
        fi_orig = wrist_dists[pk][0]
        pts_ms  = wrist_dists[pk][1]
        dist_val = float(dists_arr[pk])
        prom     = float(props["prominences"][list(peaks).index(pk)])
        candidates.append({
            "frame_index": int(fi_orig),
            "source_pts_ms": float(pts_ms),
            "wrist_distance_norm": round(dist_val, 5),
            "prominence": round(prom, 5),
        })

    # sort by distance (most likely clap = smallest distance)
    candidates.sort(key=lambda c: c["wrist_distance_norm"])

    # also compute global minimum in first 30%
    global_min_idx = int(np.argmin(dists_arr[:cutoff]))
    global_min = {
        "frame_index": int(wrist_dists[global_min_idx][0]),
        "source_pts_ms": float(wrist_dists[global_min_idx][1]),
        "wrist_distance_norm": round(float(dists_arr[global_min_idx]), 5),
    }

    return {
        "view": view,
        "two_hand_frames_in_first_30pct": cutoff,
        "global_minimum": global_min,
        "local_minima_candidates": candidates[:10],  # top 10
    }


# ---- 3. review visuals ----------------------------------------------------

def extract_frame_image(video_path, frame_idx):
    """Read a single frame from video by index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    return bgr if ret else None


def make_contact_sheet(manifest_videos, clap_info, out_path, radius=5):
    """Grid of frames [clap-5 .. clap+5] for each view."""
    n_cols = 2 * radius + 1  # 11 frames
    n_rows = len(VIEW_ORDER)
    thumb_w, thumb_h = 256, 192

    canvas = np.zeros((n_rows * (thumb_h + 30), n_cols * thumb_w, 3), np.uint8)

    for row, view in enumerate(VIEW_ORDER):
        vid_path = manifest_videos[view]["absolute_path"]
        clap_fi  = clap_info[view]["best_candidate"]["frame_index"]
        total    = manifest_videos[view]["frame_count"]

        for col in range(n_cols):
            fi = clap_fi - radius + col
            fi = max(0, min(total - 1, fi))
            bgr = extract_frame_image(vid_path, fi)
            if bgr is None:
                continue
            thumb = cv2.resize(bgr, (thumb_w, thumb_h))

            # highlight centre column (the clap frame)
            if col == radius:
                cv2.rectangle(thumb, (0, 0), (thumb_w - 1, thumb_h - 1),
                              (0, 255, 0), 3)

            y0 = row * (thumb_h + 30)
            canvas[y0:y0 + thumb_h, col * thumb_w:(col + 1) * thumb_w] = thumb

            # label
            label = f"{view} fi={fi}"
            cv2.putText(canvas, label,
                        (col * thumb_w + 4, y0 + thumb_h + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.imwrite(out_path, canvas)
    print(f"  contact sheet: {out_path} ({os.path.getsize(out_path)//1024}KB)",
          file=sys.stderr)


def make_sync_review_mp4(manifest_videos, clap_info, out_path,
                          window_frames=60, fps_out=15):
    """Side-by-side MP4 showing +-window_frames around each view's clap."""
    thumb_w, thumb_h = 426, 320
    canvas_w = thumb_w * 3
    canvas_h = thumb_h + 40  # room for label bar

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_out,
                             (canvas_w, canvas_h))

    caps = {}
    clap_fis = {}
    totals = {}
    for view in VIEW_ORDER:
        vid_path = manifest_videos[view]["absolute_path"]
        caps[view] = cv2.VideoCapture(vid_path)
        clap_fis[view] = clap_info[view]["best_candidate"]["frame_index"]
        totals[view] = manifest_videos[view]["frame_count"]

    for offset in range(-window_frames, window_frames + 1):
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        for col, view in enumerate(VIEW_ORDER):
            fi = clap_fis[view] + offset
            fi = max(0, min(totals[view] - 1, fi))
            caps[view].set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, bgr = caps[view].read()
            if not ret:
                continue
            thumb = cv2.resize(bgr, (thumb_w, thumb_h))
            x0 = col * thumb_w
            canvas[:thumb_h, x0:x0 + thumb_w] = thumb

            # clap marker
            if offset == 0:
                cv2.rectangle(canvas, (x0, 0), (x0 + thumb_w - 1, thumb_h - 1),
                              (0, 255, 0), 3)
                cv2.putText(canvas, "CLAP", (x0 + 10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            label = f"{view} fi={fi} off={offset:+d}"
            cv2.putText(canvas, label,
                        (x0 + 4, thumb_h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        writer.write(canvas)

    for c in caps.values():
        c.release()
    writer.release()
    print(f"  sync review mp4: {out_path} ({os.path.getsize(out_path)//1024}KB)",
          file=sys.stderr)


# ---- 4. sync candidates report --------------------------------------------

def build_sync_report(clap_results, manifest_videos):
    """Build sync_candidates.json from per-view clap results."""
    front_pts = None
    per_view = {}

    for view in VIEW_ORDER:
        cr = clap_results[view]
        best = cr["global_minimum"]  # simplest robust pick
        per_view[view] = {
            "best_candidate": best,
            "all_local_minima": cr["local_minima_candidates"],
            "two_hand_frames_searched": cr.get("two_hand_frames_in_first_30pct", 0),
        }
        if view == "front":
            front_pts = best["source_pts_ms"]

    # compute lags relative to front (using PTS, not frame index)
    for view in VIEW_ORDER:
        best_pts = per_view[view]["best_candidate"]["source_pts_ms"]
        lag_ms = best_pts - front_pts if front_pts is not None else None
        per_view[view]["lag_from_front_ms"] = round(lag_ms, 3) if lag_ms is not None else None

    # drift analysis: compare FPS-implied timing vs PTS timing
    drift_notes = []
    for view in VIEW_ORDER:
        fi = per_view[view]["best_candidate"]["frame_index"]
        pts = per_view[view]["best_candidate"]["source_pts_ms"]
        fps = manifest_videos[view]["fps_reported"]
        fps_implied_ms = fi * 1000.0 / fps
        residual = abs(pts - fps_implied_ms)
        per_view[view]["fps_implied_ms"] = round(fps_implied_ms, 3)
        per_view[view]["pts_vs_fps_residual_ms"] = round(residual, 3)
        if residual > 100:
            drift_notes.append(
                f"{view}: PTS={pts:.1f}ms vs FPS-implied={fps_implied_ms:.1f}ms "
                f"(residual {residual:.1f}ms) -- possible timestamp drift")

    return {
        "method": "wrist_distance_minimum_in_first_30pct",
        "audio_available": False,
        "audio_note": "No audio tracks in any source video (confirmed Step 1).",
        "reference_view": "front",
        "views": per_view,
        "drift_notes": drift_notes if drift_notes else ["no significant drift detected"],
        "status": "PENDING_HUMAN_REVIEW",
        "human_review_instructions": [
            "1. Open s04_out_v3a/sync_contact_sheet.png and verify the green-bordered frame in each row shows the same physical moment (hands touching).",
            "2. Watch s04_out_v3a/sync_review.mp4 and confirm the green CLAP marker aligns across all 3 views.",
            "3. If misaligned, note the correct frame_index per view and provide as manual override.",
        ],
    }


# ---- numpy encoder --------------------------------------------------------

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="Reuse cached landmark JSONs")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # load manifest
    manifest_path = os.path.join(OUT_DIR, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_videos = manifest["videos"]

    # ---- extraction --------------------------------------------------------
    landmark_data = {}
    for view in VIEW_ORDER:
        vinfo = manifest_videos[view]
        cache_path = os.path.join(OUT_DIR, f"landmarks_raw_{view}.json")

        if args.skip_extract and os.path.exists(cache_path):
            print(f"[cache] {view}: loading {cache_path}", file=sys.stderr)
            with open(cache_path, encoding="utf-8") as f:
                landmark_data[view] = json.load(f)
            print(f"  {view}: {len(landmark_data[view]['frames'])} frames",
                  file=sys.stderr)
            continue

        print(f"\n[extract] {view}: {vinfo['filename']}", file=sys.stderr)
        data = extract_view(view, vinfo["absolute_path"], vinfo["sha256"])
        landmark_data[view] = data

        # save (compact JSON -- no indent to reduce size)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, cls=NpEncoder)
        sz = os.path.getsize(cache_path)
        print(f"  saved: {cache_path} ({sz // 1024}KB)", file=sys.stderr)

    # ---- clap detection ----------------------------------------------------
    print("\n[clap detection]", file=sys.stderr)
    clap_results = {}
    for view in VIEW_ORDER:
        clap_results[view] = detect_clap_candidates(landmark_data[view])
        gm = clap_results[view].get("global_minimum", {})
        print(f"  {view}: best fi={gm.get('frame_index','?')} "
              f"pts={gm.get('source_pts_ms','?')}ms "
              f"dist={gm.get('wrist_distance_norm','?')}",
              file=sys.stderr)

    # ---- sync report -------------------------------------------------------
    sync_report = build_sync_report(clap_results, manifest_videos)

    sync_path = os.path.join(OUT_DIR, "sync_candidates.json")
    with open(sync_path, "w", encoding="utf-8") as f:
        json.dump(sync_report, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    print(f"\n[saved] sync_candidates.json ({os.path.getsize(sync_path)} bytes)",
          file=sys.stderr)

    # ---- review visuals ----------------------------------------------------
    print("\n[visuals]", file=sys.stderr)

    clap_info = sync_report["views"]

    sheet_path = os.path.join(OUT_DIR, "sync_contact_sheet.png")
    make_contact_sheet(manifest_videos, clap_info, sheet_path, radius=5)

    mp4_path = os.path.join(OUT_DIR, "sync_review.mp4")
    make_sync_review_mp4(manifest_videos, clap_info, mp4_path,
                          window_frames=60, fps_out=15)

    # ---- summary -----------------------------------------------------------
    print("\n=== Step 2 complete ===", file=sys.stderr)
    for view in VIEW_ORDER:
        bc = sync_report["views"][view]["best_candidate"]
        lag = sync_report["views"][view]["lag_from_front_ms"]
        print(f"  {view:8s}: fi={bc['frame_index']:4d}  "
              f"pts={bc['source_pts_ms']:9.1f}ms  "
              f"lag={lag:+8.1f}ms" if lag is not None else
              f"  {view:8s}: fi={bc['frame_index']:4d}  "
              f"pts={bc['source_pts_ms']:9.1f}ms  lag=ref",
              file=sys.stderr)
    print(f"\nstatus: {sync_report['status']}", file=sys.stderr)


if __name__ == "__main__":
    main()
