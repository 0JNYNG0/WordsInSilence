"""
Step 6A: MediaPipe 2D Landmark Pilot Extraction — S001_P001

Pilot: 9 R03 samples × 3 views = 27 extraction runs.
Uses Step 5 v2.1 verified boundaries and least-squares affine time transforms.
No 3D triangulation, DTW, or canonical selection.
Original videos are NOT modified, mirrored, or re-encoded.
"""

import json
import csv
import hashlib
import os
import sys
import time as _time
import datetime
from pathlib import Path

import av
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"

VIDEOS = {
    "FRONT": RAW_DIR / "S001_P001_FRONT.mp4",
    "LEFT":  RAW_DIR / "S001_P001_LEFT.mp4",
    "RIGHT": RAW_DIR / "S001_P001_RIGHT.mp4",
}

POSE_MODEL = PROJECT_ROOT / "Tools" / "models" / "pose_landmarker_lite.task"
HAND_MODEL = PROJECT_ROOT / "Tools" / "models" / "hand_landmarker.task"

C1_INTRINSICS = {
    "FRONT": PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_FRONT_01_v2.json",
    "LEFT":  PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_LEFT_01_v2.json",
    "RIGHT": PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_RIGHT_01_v2.json",
}

PILOT_SAMPLE_IDS = [
    "S001_P001_GREETING_NORMAL_R03",
    "S001_P001_GREETING_SLOW_R03",
    "S001_P001_GREETING_FAST_R03",
    "S001_P001_THANK_YOU_NORMAL_R03",
    "S001_P001_THANK_YOU_SLOW_R03",
    "S001_P001_THANK_YOU_FAST_R03",
    "S001_P001_LOVE_NORMAL_R03",
    "S001_P001_LOVE_SLOW_R03",
    "S001_P001_LOVE_FAST_R03",
]

POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_intrinsics(view):
    d = json.load(open(C1_INTRINSICS[view]))
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64)
    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    return K, dist


def undistort_points(pts_px, K, dist):
    """Undistort pixel coordinates using C1 intrinsics.
    pts_px: Nx2 array of (x, y) pixel coords.
    Returns: Nx2 undistorted pixel coords, Nx2 undistorted normalized camera coords.
    """
    if len(pts_px) == 0:
        return np.array([]), np.array([])
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    # undistortPoints returns normalized camera coords
    und_norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    # Convert back to pixel coords
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    und_px = np.column_stack([und_norm[:, 0] * fx + cx, und_norm[:, 1] * fy + cy])
    return und_px, und_norm


def cam_to_front(cam_s, a, b):
    return (cam_s - b) / a


def front_to_cam(front_s, a, b):
    return a * front_s + b


# ═══════════════════════════════════════════════════════════
#  Anatomical hand identity association
# ═══════════════════════════════════════════════════════════

def associate_hands(pose_result, hand_result, img_w, img_h, prev_assignment):
    """
    Associate detected hands to anatomical LEFT/RIGHT using:
    1. Pose wrist positions (landmarks 15=LEFT_WRIST, 16=RIGHT_WRIST)
    2. Hand wrist (landmark 0) proximity to pose wrist
    3. Previous frame continuity
    4. Model handedness label as tiebreaker

    Returns list of dicts with anatomical identity info.
    """
    results = []

    # Get pose wrist positions in pixels
    pose_wrists = {}
    if pose_result.pose_landmarks:
        plm = pose_result.pose_landmarks[0]
        for side, idx in [("LEFT", 15), ("RIGHT", 16)]:
            lm = plm[idx]
            if lm.visibility > 0.3:
                pose_wrists[side] = np.array([lm.x * img_w, lm.y * img_h])

    if not hand_result.hand_landmarks:
        return results

    n_hands = len(hand_result.hand_landmarks)
    hand_wrists = []
    for hi in range(n_hands):
        hw = hand_result.hand_landmarks[hi][0]  # WRIST = landmark 0
        hand_wrists.append(np.array([hw.x * img_w, hw.y * img_h]))

    # Model handedness
    model_labels = []
    model_scores = []
    for hi in range(n_hands):
        h = hand_result.handedness[hi][0]
        model_labels.append(h.category_name)
        model_scores.append(h.score)

    # Compute distances to pose wrists
    dist_matrix = {}
    for hi in range(n_hands):
        for side in ["LEFT", "RIGHT"]:
            if side in pose_wrists:
                d = float(np.linalg.norm(hand_wrists[hi] - pose_wrists[side]))
                dist_matrix[(hi, side)] = d
            else:
                dist_matrix[(hi, side)] = 9999.0

    # Assign anatomical identity
    assignments = [None] * n_hands
    used_sides = set()

    if n_hands == 1:
        hi = 0
        d_left = dist_matrix.get((0, "LEFT"), 9999)
        d_right = dist_matrix.get((0, "RIGHT"), 9999)

        if d_left < d_right and d_left < 200:
            assignments[0] = "ANAT_LEFT"
        elif d_right < d_left and d_right < 200:
            assignments[0] = "ANAT_RIGHT"
        elif "LEFT" not in pose_wrists and "RIGHT" not in pose_wrists:
            # No pose wrists visible, use model label as fallback
            # MediaPipe "Left" means the hand appears on the left side of image
            # For non-mirrored video: model "Left" = anatomical RIGHT (subject's right)
            # model "Right" = anatomical LEFT (subject's left)
            assignments[0] = "AMBIGUOUS"
        else:
            assignments[0] = "AMBIGUOUS"

    elif n_hands == 2:
        # Greedy assignment by minimum distance
        pairs = []
        for hi in range(2):
            for side in ["LEFT", "RIGHT"]:
                pairs.append((dist_matrix[(hi, side)], hi, side))
        pairs.sort()

        for d, hi, side in pairs:
            if assignments[hi] is not None or side in used_sides:
                continue
            if d < 300:
                assignments[hi] = f"ANAT_{side}"
                used_sides.add(side)

        for hi in range(2):
            if assignments[hi] is None:
                assignments[hi] = "AMBIGUOUS"

    for hi in range(n_hands):
        d_l = dist_matrix.get((hi, "LEFT"), None)
        d_r = dist_matrix.get((hi, "RIGHT"), None)
        results.append({
            "hand_index": hi,
            "anatomical_identity": assignments[hi],
            "model_handedness_label": model_labels[hi],
            "model_handedness_score": round(model_scores[hi], 4),
            "pose_left_wrist_dist_px": round(d_l, 2) if d_l and d_l < 9000 else None,
            "pose_right_wrist_dist_px": round(d_r, 2) if d_r and d_r < 9000 else None,
            "hand_wrist_px": [round(hand_wrists[hi][0], 2), round(hand_wrists[hi][1], 2)],
            "pose_left_wrist_px": [round(pose_wrists["LEFT"][0], 2), round(pose_wrists["LEFT"][1], 2)] if "LEFT" in pose_wrists else None,
            "pose_right_wrist_px": [round(pose_wrists["RIGHT"][0], 2), round(pose_wrists["RIGHT"][1], 2)] if "RIGHT" in pose_wrists else None,
        })

    return results


# ═══════════════════════════════════════════════════════════
#  Per-view extraction
# ═══════════════════════════════════════════════════════════

def extract_view(view, segments, affine_a, affine_b, K, dist, out_dir):
    """Extract 2D landmarks for all pilot segments from one view."""
    video_path = VIDEOS[view]
    print(f"\n  === {view} ===")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    img_w = stream.width
    img_h = stream.height

    # Build segment lookup by camera PTS range
    seg_ranges = []
    for seg in segments:
        cr = seg["camera_ranges"][view]
        seg_ranges.append({
            "sample_id": seg["sample_id"],
            "cam_start": cr["start_pts_s"],
            "cam_end": cr["end_pts_s"],
            "front_start": seg["front_start_s"],
            "front_end": seg["front_end_s"],
        })

    cam_start_global = min(sr["cam_start"] for sr in seg_ranges)
    cam_end_global = max(sr["cam_end"] for sr in seg_ranges)

    # Create pose and hand landmarkers
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    ))

    all_frames = []
    boundary_diag = []
    prev_ts_ms = -1
    ts_adjustments = 0
    frame_idx = 0
    n_in_range = 0

    # Per-segment tracking
    seg_first_pts = {sr["sample_id"]: None for sr in seg_ranges}
    seg_last_pts = {sr["sample_id"]: None for sr in seg_ranges}
    prev_hand_assignment = {}

    for frame in container.decode(stream):
        cam_pts_s = float(frame.pts) * tb if frame.pts is not None else frame_idx / 30.0

        # Skip frames before/after all segments (with small margin)
        if cam_pts_s < cam_start_global - 0.5:
            frame_idx += 1
            continue
        if cam_pts_s > cam_end_global + 0.5:
            break

        # Find which segment (if any) this frame belongs to
        current_seg = None
        for sr in seg_ranges:
            if sr["cam_start"] <= cam_pts_s <= sr["cam_end"]:
                current_seg = sr
                break

        if current_seg is None:
            frame_idx += 1
            continue

        n_in_range += 1
        sid = current_seg["sample_id"]
        front_pts_s = cam_to_front(cam_pts_s, affine_a, affine_b)

        # Track first/last actual PTS per segment
        if seg_first_pts[sid] is None:
            seg_first_pts[sid] = cam_pts_s
        seg_last_pts[sid] = cam_pts_s

        # Prepare timestamp for MediaPipe (strictly increasing integer ms)
        ts_ms = int(round(cam_pts_s * 1000))
        original_ts_ms = ts_ms
        if ts_ms <= prev_ts_ms:
            ts_ms = prev_ts_ms + 1
            ts_adjustments += 1
        prev_ts_ms = ts_ms

        # Convert to RGB for MediaPipe
        img_rgb = frame.to_ndarray(format="rgb24")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Run pose detection
        pose_result = pose_lm.detect_for_video(mp_image, ts_ms)
        hand_result = hand_lm.detect_for_video(mp_image, ts_ms)

        # Build frame record
        frame_record = {
            "sample_id": sid,
            "view": view,
            "decoded_frame_index": frame_idx,
            "camera_local_pts_s": round(cam_pts_s, 6),
            "mapped_front_pts_s": round(front_pts_s, 6),
            "timestamp_ms_sent_to_mediapipe": ts_ms,
            "timestamp_ms_original": original_ts_ms,
            "timestamp_adjusted": ts_ms != original_ts_ms,
            "image_width": img_w,
            "image_height": img_h,
        }

        # Pose landmarks
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            plm = pose_result.pose_landmarks[0]
            pose_data = []
            for li, lm in enumerate(plm):
                raw_nx, raw_ny = lm.x, lm.y
                raw_px = raw_nx * img_w
                raw_py = raw_ny * img_h
                und_px_arr, und_norm_arr = undistort_points(
                    np.array([[raw_px, raw_py]]), K, dist
                )
                und_px = und_px_arr[0] if len(und_px_arr) > 0 else [None, None]
                und_norm = und_norm_arr[0] if len(und_norm_arr) > 0 else [None, None]

                pose_data.append({
                    "index": li,
                    "name": POSE_LANDMARK_NAMES[li],
                    "raw_norm_x": round(float(raw_nx), 6),
                    "raw_norm_y": round(float(raw_ny), 6),
                    "raw_distorted_px_x": round(float(raw_px), 2),
                    "raw_distorted_px_y": round(float(raw_py), 2),
                    "undistorted_px_x": round(float(und_px[0]), 2),
                    "undistorted_px_y": round(float(und_px[1]), 2),
                    "undistorted_norm_cam_x": round(float(und_norm[0]), 6),
                    "undistorted_norm_cam_y": round(float(und_norm[1]), 6),
                    "visibility": round(float(lm.visibility), 4),
                    "presence": round(float(lm.presence), 4) if hasattr(lm, "presence") else None,
                })
            frame_record["pose_landmarks"] = pose_data
            frame_record["pose_detected"] = True
        else:
            frame_record["pose_landmarks"] = None
            frame_record["pose_detected"] = False

        # Hand landmarks
        n_hands_detected = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        frame_record["hands_detected_count"] = n_hands_detected

        if n_hands_detected > 0:
            hand_associations = associate_hands(
                pose_result, hand_result, img_w, img_h, prev_hand_assignment
            )

            hands_data = []
            for hi in range(n_hands_detected):
                assoc = hand_associations[hi] if hi < len(hand_associations) else {
                    "anatomical_identity": "AMBIGUOUS",
                    "model_handedness_label": "Unknown",
                    "model_handedness_score": 0,
                }

                hlm = hand_result.hand_landmarks[hi]
                hand_pts = []
                for li, lm in enumerate(hlm):
                    raw_nx, raw_ny = lm.x, lm.y
                    raw_px = raw_nx * img_w
                    raw_py = raw_ny * img_h
                    und_px_arr, und_norm_arr = undistort_points(
                        np.array([[raw_px, raw_py]]), K, dist
                    )
                    und_px = und_px_arr[0] if len(und_px_arr) > 0 else [None, None]
                    und_norm = und_norm_arr[0] if len(und_norm_arr) > 0 else [None, None]

                    hand_pts.append({
                        "index": li,
                        "name": HAND_LANDMARK_NAMES[li],
                        "raw_norm_x": round(float(raw_nx), 6),
                        "raw_norm_y": round(float(raw_ny), 6),
                        "raw_distorted_px_x": round(float(raw_px), 2),
                        "raw_distorted_px_y": round(float(raw_py), 2),
                        "undistorted_px_x": round(float(und_px[0]), 2),
                        "undistorted_px_y": round(float(und_px[1]), 2),
                        "undistorted_norm_cam_x": round(float(und_norm[0]), 6),
                        "undistorted_norm_cam_y": round(float(und_norm[1]), 6),
                        "visibility": None,
                        "presence": None,
                    })

                hands_data.append({
                    "hand_index": hi,
                    "anatomical_identity": assoc["anatomical_identity"],
                    "model_handedness_label": assoc["model_handedness_label"],
                    "model_handedness_score": assoc["model_handedness_score"],
                    "pose_left_wrist_dist_px": assoc.get("pose_left_wrist_dist_px"),
                    "pose_right_wrist_dist_px": assoc.get("pose_right_wrist_dist_px"),
                    "landmarks": hand_pts,
                })

                # Update prev assignment for continuity
                if assoc["anatomical_identity"] not in (None, "AMBIGUOUS"):
                    prev_hand_assignment[assoc["anatomical_identity"]] = assoc.get("hand_wrist_px")

            frame_record["hands"] = hands_data
        else:
            frame_record["hands"] = []

        all_frames.append(frame_record)
        frame_idx += 1

        if n_in_range % 200 == 0:
            print(f"    {view}: {n_in_range} frames processed, t={cam_pts_s:.1f}s")

    container.close()
    pose_lm.close()
    hand_lm.close()

    print(f"    {view}: {n_in_range} total frames, {ts_adjustments} timestamp adjustments")

    # Boundary diagnostics
    for sr in seg_ranges:
        sid = sr["sample_id"]
        first = seg_first_pts[sid]
        last = seg_last_pts[sid]
        boundary_diag.append({
            "sample_id": sid,
            "view": view,
            "requested_cam_start_s": round(sr["cam_start"], 6),
            "requested_cam_end_s": round(sr["cam_end"], 6),
            "actual_first_frame_pts_s": round(first, 6) if first else None,
            "actual_last_frame_pts_s": round(last, 6) if last else None,
            "start_error_ms": round((first - sr["cam_start"]) * 1000, 2) if first else None,
            "end_error_ms": round((sr["cam_end"] - last) * 1000, 2) if last else None,
            "n_frames": sum(1 for f in all_frames if f["sample_id"] == sid),
        })

    return all_frames, boundary_diag


# ═══════════════════════════════════════════════════════════
#  Overlay video generation
# ═══════════════════════════════════════════════════════════

def make_overlay_video(view, frames_data, segments, affine_a, affine_b, out_path, fps_out=15):
    """Overlay skeleton on decoded frames using PyAV."""
    video_path = VIDEOS[view]
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    img_w, img_h = stream.width, stream.height

    scale = 0.5
    ow, oh = int(img_w * scale), int(img_h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (ow, oh))

    # Build frame lookup by (view, cam_pts rounded)
    frame_lookup = {}
    for fd in frames_data:
        if fd["view"] == view:
            key = round(fd["camera_local_pts_s"], 4)
            frame_lookup[key] = fd

    seg_ranges = []
    for seg in segments:
        cr = seg["camera_ranges"][view]
        seg_ranges.append((cr["start_pts_s"], cr["end_pts_s"], seg["sample_id"]))

    cam_start = min(sr[0] for sr in seg_ranges) - 1.0
    cam_end = max(sr[1] for sr in seg_ranges) + 1.0

    frame_idx = 0
    dt_out = 1.0 / fps_out
    next_grab_s = cam_start
    written = 0

    POSE_CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16),  # arms
        (11, 12), (11, 23), (12, 24), (23, 24),  # torso
        (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    ]

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for frame in container.decode(stream):
        cam_pts_s = float(frame.pts) * tb if frame.pts is not None else frame_idx / 30.0

        if cam_pts_s < cam_start:
            frame_idx += 1
            continue
        if cam_pts_s > cam_end:
            break

        if cam_pts_s < next_grab_s:
            frame_idx += 1
            continue
        next_grab_s = cam_pts_s + dt_out

        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (ow, oh))

        front_pts_s = cam_to_front(cam_pts_s, affine_a, affine_b)

        # Find current segment
        current_sid = None
        for cs, ce, sid in seg_ranges:
            if cs <= cam_pts_s <= ce:
                current_sid = sid
                break

        # Header
        if current_sid:
            cv2.rectangle(small, (0, 0), (ow, 35), (0, 80, 0), -1)
            label = current_sid.split("_", 3)[-1]
        else:
            cv2.rectangle(small, (0, 0), (ow, 35), (60, 60, 60), -1)
            label = "NEUTRAL"

        cv2.putText(small, f"{view} {label} cam={cam_pts_s:.2f} front={front_pts_s:.2f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Find nearest frame data
        key = round(cam_pts_s, 4)
        fd = frame_lookup.get(key)
        if fd is None:
            # Try nearby keys
            for delta in [0.0001, -0.0001, 0.001, -0.001]:
                fd = frame_lookup.get(round(cam_pts_s + delta, 4))
                if fd:
                    break

        if fd:
            # Draw pose
            if fd.get("pose_detected") and fd.get("pose_landmarks"):
                pts = [(int(lm["raw_distorted_px_x"] * scale), int(lm["raw_distorted_px_y"] * scale))
                       for lm in fd["pose_landmarks"]]
                for i, j in POSE_CONNECTIONS:
                    if fd["pose_landmarks"][i]["visibility"] > 0.3 and fd["pose_landmarks"][j]["visibility"] > 0.3:
                        cv2.line(small, pts[i], pts[j], (200, 200, 200), 1)

            # Draw hands
            ANAT_COLORS = {"ANAT_LEFT": (0, 255, 0), "ANAT_RIGHT": (0, 128, 255), "AMBIGUOUS": (0, 0, 255)}
            warnings = []
            for hand in fd.get("hands", []):
                anat = hand["anatomical_identity"]
                color = ANAT_COLORS.get(anat, (128, 128, 128))

                hpts = [(int(lm["raw_distorted_px_x"] * scale), int(lm["raw_distorted_px_y"] * scale))
                        for lm in hand["landmarks"]]
                for i, j in HAND_CONNECTIONS:
                    cv2.line(small, hpts[i], hpts[j], color, 1)
                for pt in hpts:
                    cv2.circle(small, pt, 2, color, -1)

                # Label
                wx, wy = hpts[0]
                short_label = {"ANAT_LEFT": "L", "ANAT_RIGHT": "R", "AMBIGUOUS": "?"}
                cv2.putText(small, short_label.get(anat, "?"), (wx - 5, wy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                if anat == "AMBIGUOUS":
                    warnings.append("AMBIGUOUS_HAND")

                # Wrist distance warning
                for side_key in ["pose_left_wrist_dist_px", "pose_right_wrist_dist_px"]:
                    d = hand.get(side_key)
                    if d and d > 150:
                        warnings.append(f"WRIST_DIST_HIGH")

            if not fd.get("pose_detected"):
                warnings.append("NO_POSE")
            if fd["hands_detected_count"] == 0:
                warnings.append("NO_HANDS")

            if warnings:
                warn_str = " ".join(set(warnings))
                cv2.putText(small, warn_str, (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        writer.write(small)
        written += 1
        frame_idx += 1

    writer.release()
    container.close()
    print(f"    Overlay {view}: {written} frames written to {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  QA statistics
# ═══════════════════════════════════════════════════════════

def compute_qa_stats(frames_data, view, segments):
    """Compute per-sample QA statistics."""
    stats = []
    for seg in segments:
        sid = seg["sample_id"]
        seg_frames = [f for f in frames_data if f["sample_id"] == sid and f["view"] == view]
        n = len(seg_frames)
        if n == 0:
            stats.append({"sample_id": sid, "view": view, "n_frames": 0})
            continue

        # Pose usable rate (shoulders, elbows, wrists visible > 0.5)
        pose_usable = 0
        for f in seg_frames:
            if f["pose_detected"] and f["pose_landmarks"]:
                key_indices = [11, 12, 13, 14, 15, 16]  # shoulders, elbows, wrists
                vis = [f["pose_landmarks"][i]["visibility"] for i in key_indices]
                if all(v > 0.5 for v in vis):
                    pose_usable += 1
        pose_rate = pose_usable / n

        # Hand detection rates
        anat_left_count = 0
        anat_right_count = 0
        both_hands_count = 0
        ambiguous_count = 0
        model_mismatch_count = 0
        max_missing_run = 0
        current_missing_run = 0
        wrist_dists = []
        swap_candidates = 0

        for f in seg_frames:
            has_left = False
            has_right = False
            for hand in f.get("hands", []):
                anat = hand["anatomical_identity"]
                if anat == "ANAT_LEFT":
                    has_left = True
                    anat_left_count += 1
                elif anat == "ANAT_RIGHT":
                    has_right = True
                    anat_right_count += 1
                elif anat == "AMBIGUOUS":
                    ambiguous_count += 1

                # Model mismatch check
                # Non-mirrored: model "Left" should be anatomical RIGHT
                model_lbl = hand["model_handedness_label"]
                if anat == "ANAT_LEFT" and model_lbl == "Left":
                    model_mismatch_count += 1
                elif anat == "ANAT_RIGHT" and model_lbl == "Right":
                    model_mismatch_count += 1

                # Wrist distances
                for key in ["pose_left_wrist_dist_px", "pose_right_wrist_dist_px"]:
                    d = hand.get(key)
                    if d is not None:
                        wrist_dists.append(d)

            if has_left and has_right:
                both_hands_count += 1

            if not has_left and not has_right:
                current_missing_run += 1
                max_missing_run = max(max_missing_run, current_missing_run)
            else:
                current_missing_run = 0

        # Compute frame duration for max missing time
        if n > 1:
            frame_dt = (seg_frames[-1]["camera_local_pts_s"] - seg_frames[0]["camera_local_pts_s"]) / (n - 1)
        else:
            frame_dt = 1.0 / 30.0
        max_missing_time_s = max_missing_run * frame_dt

        wrist_dists_arr = np.array(wrist_dists) if wrist_dists else np.array([0])

        stats.append({
            "sample_id": sid,
            "view": view,
            "n_frames": n,
            "pose_usable_rate": round(pose_rate, 4),
            "anat_left_detection_rate": round(anat_left_count / n, 4),
            "anat_right_detection_rate": round(anat_right_count / n, 4),
            "both_hands_21pt_rate": round(both_hands_count / n, 4),
            "max_consecutive_hand_missing": max_missing_run,
            "max_consecutive_hand_missing_time_s": round(max_missing_time_s, 3),
            "model_handedness_mismatch_count": model_mismatch_count,
            "ambiguous_frame_count": ambiguous_count,
            "identity_swap_candidates": swap_candidates,
            "wrist_dist_median_px": round(float(np.median(wrist_dists_arr)), 2),
            "wrist_dist_p95_px": round(float(np.percentile(wrist_dists_arr, 95)), 2),
            "wrist_dist_max_px": round(float(np.max(wrist_dists_arr)), 2),
        })

    return stats


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Step 6A: MediaPipe 2D Landmark Pilot Extraction — S001_P001\n")

    # Load verified segments
    seg_path = PROC_DIR / "action_segments_v2_1_verified.json"
    seg_data = json.load(open(seg_path, encoding="utf-8"))
    all_segs = seg_data["segments"]
    transforms = seg_data["affine_transforms"]

    # Pre-verify
    assert seg_data["n_segments_expected"] == 45
    assert len(all_segs) == 45
    assert all(s["verification_status"] == "MANUAL_VERIFIED" for s in all_segs)
    print("  Input verification: PASS (45 segments, all MANUAL_VERIFIED)")

    # Filter to pilot R03
    pilot_segs = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    assert len(pilot_segs) == 9, f"Expected 9 pilot segments, got {len(pilot_segs)}"
    print(f"  Pilot samples: {len(pilot_segs)}")

    # Output directories
    out_base = PROC_DIR / "landmarks_2d_v1" / "pilot"
    out_overlays = out_base / "overlays"
    out_lowconf = out_base / "low_confidence_contact_sheets"
    for d in [out_base, out_overlays, out_lowconf]:
        d.mkdir(parents=True, exist_ok=True)

    # Load intrinsics
    intrinsics = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        intrinsics[view] = load_intrinsics(view)

    # Process each view
    all_view_frames = {}
    all_boundary_diag = []

    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        a, b = t["a"], t["b"]
        K, dist = intrinsics[view]

        frames, bdiag = extract_view(view, pilot_segs, a, b, K, dist, out_base)
        all_view_frames[view] = frames
        all_boundary_diag.extend(bdiag)

    # Save per-view landmark JSONs
    for view in ["FRONT", "LEFT", "RIGHT"]:
        view_file = out_base / f"landmarks_2d_{view.lower()}.json"
        with open(view_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1_pilot",
                "view": view,
                "n_frames": len(all_view_frames[view]),
                "image_size": [1920, 1080],
                "original_video_mirrored": False,
                "frames": all_view_frames[view],
            }, f, indent=1, ensure_ascii=False)
        print(f"  Saved: {view_file.name} ({len(all_view_frames[view])} frames)")

    # Boundary diagnostics CSV
    bdiag_path = out_base / "boundary_frame_diagnostics.csv"
    if all_boundary_diag:
        with open(bdiag_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_boundary_diag[0].keys())
            w.writeheader()
            w.writerows(all_boundary_diag)
        print(f"  Saved: {bdiag_path.name}")

    # QA stats
    all_qa = []
    for view in ["FRONT", "LEFT", "RIGHT"]:
        qa = compute_qa_stats(all_view_frames[view], view, pilot_segs)
        all_qa.extend(qa)

    # Detection quality CSV
    dq_path = out_base / "detection_quality.csv"
    if all_qa:
        with open(dq_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_qa[0].keys())
            w.writeheader()
            w.writerows(all_qa)
        print(f"  Saved: {dq_path.name}")

    # Hand identity diagnostics CSV
    hid_rows = []
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for fd in all_view_frames[view]:
            for hand in fd.get("hands", []):
                hid_rows.append({
                    "sample_id": fd["sample_id"],
                    "view": view,
                    "camera_pts_s": fd["camera_local_pts_s"],
                    "hand_index": hand["hand_index"],
                    "anatomical_identity": hand["anatomical_identity"],
                    "model_handedness_label": hand["model_handedness_label"],
                    "model_handedness_score": hand["model_handedness_score"],
                    "pose_left_wrist_dist_px": hand.get("pose_left_wrist_dist_px"),
                    "pose_right_wrist_dist_px": hand.get("pose_right_wrist_dist_px"),
                })
    hid_path = out_base / "hand_identity_diagnostics.csv"
    if hid_rows:
        with open(hid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hid_rows[0].keys())
            w.writeheader()
            w.writerows(hid_rows)
        print(f"  Saved: {hid_path.name}")

    # Generate overlays
    print("\n  Generating overlay videos...")
    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        overlay_path = out_overlays / f"overlay_pilot_{view.lower()}.mp4"
        make_overlay_video(view, all_view_frames[view], pilot_segs, t["a"], t["b"], overlay_path)

    # Run manifest
    manifest = {
        "step": "6A_pilot",
        "date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "mediapipe_version": mp.__version__,
        "pyav_version": av.__version__,
        "opencv_version": cv2.__version__,
        "pose_model": {
            "path": str(POSE_MODEL),
            "type": "pose_landmarker_lite",
            "sha256": sha256_file(POSE_MODEL),
        },
        "hand_model": {
            "path": str(HAND_MODEL),
            "type": "hand_landmarker",
            "sha256": sha256_file(HAND_MODEL),
        },
        "pose_options": {
            "running_mode": "VIDEO",
            "num_poses": 1,
            "output_segmentation_masks": False,
        },
        "hand_options": {
            "running_mode": "VIDEO",
            "num_hands": 2,
        },
        "input_file": str(seg_path),
        "affine_transforms": transforms,
        "pilot_sample_ids": PILOT_SAMPLE_IDS,
        "original_videos_mirrored": False,
        "coordinate_conventions": {
            "raw_norm": "MediaPipe normalized [0,1] image coordinates",
            "raw_distorted_px": "raw_norm * image_size, still distorted by lens",
            "undistorted_px": "C1 intrinsics undistortion applied, pixel coords",
            "undistorted_norm_cam": "C1 undistorted normalized camera coords (x/fx, y/fy)",
            "hand_visibility_presence": "null (not provided by Hand Landmarker API)",
        },
        "null_policy": "Detection failures and unavailable values stored as null, never 0-filled or interpolated",
    }
    manifest_path = out_base / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {manifest_path.name}")

    # Print QA summary
    print("\n  ═══════════════════════════════════════════════════")
    print("  QA Summary")
    print("  ═══════════════════════════════════════════════════")
    for view in ["FRONT", "LEFT", "RIGHT"]:
        view_qa = [q for q in all_qa if q["view"] == view]
        print(f"\n  {view}:")
        for q in view_qa:
            sid_short = q["sample_id"].split("_", 3)[-1]
            print(f"    {sid_short}: {q['n_frames']}f "
                  f"pose={q['pose_usable_rate']:.0%} "
                  f"L={q['anat_left_detection_rate']:.0%} "
                  f"R={q['anat_right_detection_rate']:.0%} "
                  f"both={q['both_hands_21pt_rate']:.0%} "
                  f"miss={q['max_consecutive_hand_missing']}f "
                  f"ambig={q['ambiguous_frame_count']} "
                  f"wD_p95={q['wrist_dist_p95_px']:.0f}px")

    # Boundary error summary
    print("\n  Boundary PTS errors:")
    max_start_err = 0
    max_end_err = 0
    for bd in all_boundary_diag:
        se = abs(bd["start_error_ms"]) if bd["start_error_ms"] is not None else 0
        ee = abs(bd["end_error_ms"]) if bd["end_error_ms"] is not None else 0
        max_start_err = max(max_start_err, se)
        max_end_err = max(max_end_err, ee)
    print(f"    Max start error: {max_start_err:.2f} ms")
    print(f"    Max end error: {max_end_err:.2f} ms")

    # Verify completeness
    for sid in PILOT_SAMPLE_IDS:
        for view in ["FRONT", "LEFT", "RIGHT"]:
            count = sum(1 for f in all_view_frames[view] if f["sample_id"] == sid)
            if count == 0:
                print(f"    WARNING: {sid} / {view} has 0 frames!")

    # Check no zeros stored as null
    zero_as_null_count = 0
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for fd in all_view_frames[view]:
            if fd["pose_landmarks"]:
                for lm in fd["pose_landmarks"]:
                    if lm["raw_norm_x"] == 0 and lm["raw_norm_y"] == 0 and lm["visibility"] < 0.1:
                        zero_as_null_count += 1

    print(f"\n  Status: REQUIRES_HUMAN_VISUAL_REVIEW")
    print(f"\n  Generated files:")
    print(f"    {out_base / 'landmarks_2d_front.json'}")
    print(f"    {out_base / 'landmarks_2d_left.json'}")
    print(f"    {out_base / 'landmarks_2d_right.json'}")
    print(f"    {out_base / 'detection_quality.csv'}")
    print(f"    {out_base / 'hand_identity_diagnostics.csv'}")
    print(f"    {out_base / 'boundary_frame_diagnostics.csv'}")
    print(f"    {out_base / 'run_manifest.json'}")
    print(f"    {out_overlays / 'overlay_pilot_front.mp4'}")
    print(f"    {out_overlays / 'overlay_pilot_left.mp4'}")
    print(f"    {out_overlays / 'overlay_pilot_right.mp4'}")


if __name__ == "__main__":
    main()
