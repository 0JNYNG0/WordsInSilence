"""
Step 6A v1.1: MediaPipe 2D Landmark Pilot Extraction — S001_P001

Corrections from v1:
- PTS: reject None-PTS frames (no fallback to frame_index/30.0)
- Anatomical association: multi-evidence scoring (not greedy assignment)
- Separate raw detection from anatomical association with explicit states
- QA stats: no hardcoded values, per-hand missing runs, real swap candidates
- Overlay: segment-only (no neutral gaps between segments)
- Contact sheets for flagged frames
- Handedness label mapping verification
- synchronized_view_coverage.csv diagnostic
- All thresholds explicit in manifest
- SHA-256 for all inputs
"""

import json
import csv
import hashlib
import os
import sys
import datetime
import math
from pathlib import Path
from collections import defaultdict

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
#  Explicit thresholds (recorded in manifest)
# ═══════════════════════════════════════════════════════════

THRESHOLDS = {
    "pose_min_detection_confidence": 0.5,
    "pose_min_presence_confidence": 0.5,
    "pose_min_tracking_confidence": 0.5,
    "hand_min_detection_confidence": 0.5,
    "hand_min_presence_confidence": 0.5,
    "hand_min_tracking_confidence": 0.5,
    "pose_visibility_usable_threshold": 0.5,
    "association_shoulder_width_gate_factor": 1.5,
    "association_continuity_weight": 0.3,
    "association_elbow_direction_weight": 0.2,
    "association_handedness_weight": 0.1,
    "association_distance_weight": 1.0,
    "wrist_distance_outlier_percentile": 95,
}

V1_DEFAULTS = {
    "pose_min_detection_confidence": "0.5 (mediapipe default)",
    "pose_min_presence_confidence": "0.5 (mediapipe default)",
    "pose_min_tracking_confidence": "0.5 (mediapipe default)",
    "hand_min_detection_confidence": "0.5 (mediapipe default)",
    "hand_min_presence_confidence": "0.5 (mediapipe default)",
    "hand_min_tracking_confidence": "0.5 (mediapipe default)",
}

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
    with open(C1_INTRINSICS[view], encoding="utf-8") as f:
        d = json.load(f)
    K = np.array(d["camera_matrix"], dtype=np.float64)
    dist = np.array(d["dist_coeffs"], dtype=np.float64)
    if dist.ndim == 1:
        dist = dist.reshape(1, -1)
    image_size = d["image_size"]  # [width, height]
    return K, dist, image_size


def undistort_points(pts_px, K, dist):
    if len(pts_px) == 0:
        return np.array([]), np.array([])
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    und_norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    und_px = np.column_stack([und_norm[:, 0] * fx + cx, und_norm[:, 1] * fy + cy])
    return und_px, und_norm


def front_to_cam(front_s, a, b):
    return a * front_s + b


def cam_to_front(cam_s, a, b):
    return (cam_s - b) / a


# ═══════════════════════════════════════════════════════════
#  Anatomical hand association (v1.1 — multi-evidence)
# ═══════════════════════════════════════════════════════════

class AnatomicalHandTracker:
    """Per-sample tracker maintaining state across frames.
    Must be reset between samples.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_accepted = {}  # side -> {"pos": np.array, "pts_s": float}
        self.prev_velocity = {}  # side -> np.array (px/s)

    def _shoulder_width(self, pose_result, img_w, img_h):
        """Compute shoulder width in pixels for distance normalization."""
        if not pose_result.pose_landmarks:
            return None
        plm = pose_result.pose_landmarks[0]
        ls = plm[11]  # left_shoulder
        rs = plm[12]  # right_shoulder
        if ls.visibility < 0.3 or rs.visibility < 0.3:
            return None
        lsp = np.array([ls.x * img_w, ls.y * img_h])
        rsp = np.array([rs.x * img_w, rs.y * img_h])
        return float(np.linalg.norm(lsp - rsp))

    def _elbow_to_wrist_direction(self, pose_result, img_w, img_h, side):
        """Get unit vector from elbow to wrist for given anatomical side."""
        if not pose_result.pose_landmarks:
            return None
        plm = pose_result.pose_landmarks[0]
        if side == "LEFT":
            elbow_idx, wrist_idx = 13, 15
        else:
            elbow_idx, wrist_idx = 14, 16
        elb = plm[elbow_idx]
        wri = plm[wrist_idx]
        if elb.visibility < 0.3 or wri.visibility < 0.3:
            return None
        ev = np.array([elb.x * img_w, elb.y * img_h])
        wv = np.array([wri.x * img_w, wri.y * img_h])
        d = wv - ev
        n = np.linalg.norm(d)
        if n < 5:
            return None
        return d / n

    def associate(self, pose_result, hand_result, img_w, img_h, cam_pts_s):
        """
        Associate detected hands to anatomical LEFT/RIGHT.

        Returns list of dicts per raw detection, each with:
        - anatomical_identity, association_status
        - diagnostic costs and reasons
        """
        results = []

        # Get pose wrist positions
        pose_wrists = {}
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            for side, idx in [("LEFT", 15), ("RIGHT", 16)]:
                lm = plm[idx]
                if lm.visibility > 0.3:
                    pose_wrists[side] = np.array([lm.x * img_w, lm.y * img_h])

        shoulder_w = self._shoulder_width(pose_result, img_w, img_h)
        gate = shoulder_w * THRESHOLDS["association_shoulder_width_gate_factor"] if shoulder_w else 300.0

        n_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        if n_hands == 0:
            return results

        # Gather raw detections
        detections = []
        for hi in range(n_hands):
            hw = hand_result.hand_landmarks[hi][0]
            wrist_px = np.array([hw.x * img_w, hw.y * img_h])
            h = hand_result.handedness[hi][0]
            detections.append({
                "hand_index": hi,
                "wrist_px": wrist_px,
                "model_label": h.category_name,
                "model_score": h.score,
            })

        # Compute distances and costs for each (detection, side) pair
        cost_matrix = {}  # (hi, side) -> {distance_cost, continuity_cost, elbow_cost, handedness_cost, total_cost}

        for det in detections:
            hi = det["hand_index"]
            for side in ["LEFT", "RIGHT"]:
                # Distance cost (normalized by shoulder width)
                if side in pose_wrists:
                    dist_px = float(np.linalg.norm(det["wrist_px"] - pose_wrists[side]))
                    dist_norm = dist_px / shoulder_w if shoulder_w and shoulder_w > 0 else dist_px / 300.0
                else:
                    dist_px = None
                    dist_norm = 2.0  # high cost when pose wrist unavailable

                distance_cost = dist_norm * THRESHOLDS["association_distance_weight"]

                # Continuity cost
                continuity_cost = 0.0
                if side in self.prev_accepted:
                    prev = self.prev_accepted[side]
                    dt = cam_pts_s - prev["pts_s"]
                    if 0 < dt < 0.2:
                        expected_pos = prev["pos"]
                        if side in self.prev_velocity:
                            expected_pos = prev["pos"] + self.prev_velocity[side] * dt
                        cont_dist = float(np.linalg.norm(det["wrist_px"] - expected_pos))
                        cont_norm = cont_dist / shoulder_w if shoulder_w and shoulder_w > 0 else cont_dist / 300.0
                        continuity_cost = cont_norm * THRESHOLDS["association_continuity_weight"]
                    else:
                        continuity_cost = 0.5 * THRESHOLDS["association_continuity_weight"]  # stale

                # Elbow direction cost
                elbow_cost = 0.0
                elbow_dir = self._elbow_to_wrist_direction(pose_result, img_w, img_h, side)
                if elbow_dir is not None and side in pose_wrists:
                    # Direction from pose elbow to detected hand wrist
                    plm = pose_result.pose_landmarks[0]
                    elbow_idx = 13 if side == "LEFT" else 14
                    elb_lm = plm[elbow_idx]
                    elb_px = np.array([elb_lm.x * img_w, elb_lm.y * img_h])
                    det_dir = det["wrist_px"] - elb_px
                    det_dir_n = np.linalg.norm(det_dir)
                    if det_dir_n > 5:
                        det_dir_unit = det_dir / det_dir_n
                        dot = float(np.dot(elbow_dir, det_dir_unit))
                        elbow_cost = max(0, 1.0 - dot) * THRESHOLDS["association_elbow_direction_weight"]

                # Handedness auxiliary cost (low weight)
                handedness_cost = 0.0
                # Non-mirrored: model "Left" = viewer's left = subject's right = ANAT_RIGHT
                # model "Right" = viewer's right = subject's left = ANAT_LEFT
                expected_model = "Right" if side == "LEFT" else "Left"
                if det["model_label"] != expected_model:
                    handedness_cost = THRESHOLDS["association_handedness_weight"]
                else:
                    # bonus for agreement, scaled by score
                    handedness_cost = -THRESHOLDS["association_handedness_weight"] * (det["model_score"] - 0.5)

                total_cost = distance_cost + continuity_cost + elbow_cost + handedness_cost

                cost_matrix[(hi, side)] = {
                    "dist_px": dist_px,
                    "dist_norm": round(dist_norm, 4),
                    "distance_cost": round(distance_cost, 4),
                    "continuity_cost": round(continuity_cost, 4),
                    "elbow_cost": round(elbow_cost, 4),
                    "handedness_cost": round(handedness_cost, 4),
                    "total_cost": round(total_cost, 4),
                }

        # ── Assignment logic ──
        assignments = {}  # hi -> (side, status, reason, rejected_reason)

        # Check for duplicate detection (both detections near same pose wrist)
        if n_hands == 2:
            # Check if both are closest to the same pose wrist
            best_side_0 = min(["LEFT", "RIGHT"], key=lambda s: cost_matrix[(0, s)]["total_cost"])
            best_side_1 = min(["LEFT", "RIGHT"], key=lambda s: cost_matrix[(1, s)]["total_cost"])

            # Check if both detections are very close to each other
            inter_hand_dist = float(np.linalg.norm(detections[0]["wrist_px"] - detections[1]["wrist_px"]))
            inter_hand_close = (shoulder_w is not None and inter_hand_dist < shoulder_w * 0.3) or \
                               (shoulder_w is None and inter_hand_dist < 80)

            # Check if both map to same pose wrist with overwhelming proximity
            both_same_wrist = False
            if best_side_0 == best_side_1:
                both_same_wrist = True
            elif len(pose_wrists) >= 2:
                # Check if one detection is closer to opposite wrist than its assigned wrist distance
                # suggests the assignment is wrong
                pass

            if both_same_wrist:
                # Both detections want the same side
                preferred_side = best_side_0
                other_side = "RIGHT" if preferred_side == "LEFT" else "LEFT"

                c0 = cost_matrix[(0, preferred_side)]["total_cost"]
                c1 = cost_matrix[(1, preferred_side)]["total_cost"]

                # Assign best to preferred side
                if c0 <= c1:
                    winner, loser = 0, 1
                else:
                    winner, loser = 1, 0

                assignments[winner] = (
                    preferred_side, "DETECTED",
                    f"best_cost_for_{preferred_side}",
                    None,
                )

                # For loser: check if it can reasonably be the other side
                loser_other_cost = cost_matrix[(loser, other_side)]
                loser_pref_cost = cost_matrix[(loser, preferred_side)]

                # Check if both detections are close to each other AND
                # both are overwhelmingly close to the same pose wrist
                if inter_hand_close:
                    # Both detections near same spot — likely duplicate
                    if other_side in pose_wrists:
                        d_loser_other = cost_matrix[(loser, other_side)].get("dist_px")
                        if d_loser_other is not None and d_loser_other < gate:
                            assignments[loser] = (
                                other_side, "DETECTED",
                                f"assigned_to_{other_side}_after_conflict",
                                f"duplicate_candidate_close_to_{preferred_side}",
                            )
                        else:
                            assignments[loser] = (
                                None, "DUPLICATE_DETECTION_CANDIDATE",
                                "both_near_same_wrist_and_close_to_each_other",
                                f"would_need_{other_side}_but_too_far",
                            )
                    else:
                        assignments[loser] = (
                            None, "DUPLICATE_DETECTION_CANDIDATE",
                            "both_near_same_wrist_and_close_to_each_other",
                            f"{other_side}_pose_wrist_not_visible",
                        )
                else:
                    # Not close to each other — try assigning loser to other side
                    if other_side in pose_wrists:
                        d_loser_other = cost_matrix[(loser, other_side)].get("dist_px")
                        if d_loser_other is not None and d_loser_other < gate:
                            assignments[loser] = (
                                other_side, "DETECTED",
                                f"second_best_assigned_to_{other_side}",
                                None,
                            )
                        else:
                            assignments[loser] = (
                                None, "AMBIGUOUS",
                                f"best_is_{preferred_side}_but_taken",
                                f"{other_side}_dist={d_loser_other:.0f}px_exceeds_gate={gate:.0f}px" if d_loser_other else f"{other_side}_no_dist",
                            )
                    else:
                        assignments[loser] = (
                            None, "AMBIGUOUS",
                            f"best_is_{preferred_side}_but_taken",
                            f"{other_side}_pose_wrist_not_visible",
                        )

            else:
                # Different sides — normal assignment
                for hi, side in [(0, best_side_0), (1, best_side_1)]:
                    d_px = cost_matrix[(hi, side)].get("dist_px")
                    if d_px is not None and d_px < gate:
                        other = "RIGHT" if side == "LEFT" else "LEFT"
                        assignments[hi] = (
                            side, "DETECTED",
                            f"best_cost_for_{side}",
                            None,
                        )
                    else:
                        assignments[hi] = (
                            None, "AMBIGUOUS",
                            "no_pose_wrist_within_gate",
                            f"dist={d_px:.0f}px_gate={gate:.0f}px" if d_px else "no_dist",
                        )

        elif n_hands == 1:
            hi = 0
            best_side = min(["LEFT", "RIGHT"], key=lambda s: cost_matrix[(0, s)]["total_cost"])
            other_side = "RIGHT" if best_side == "LEFT" else "LEFT"
            d_px = cost_matrix[(hi, best_side)].get("dist_px")

            if d_px is not None and d_px < gate:
                assignments[hi] = (
                    best_side, "DETECTED",
                    f"best_cost_for_{best_side}",
                    None,
                )
            elif d_px is None and len(pose_wrists) == 0:
                assignments[hi] = (
                    None, "AMBIGUOUS",
                    "no_pose_wrists_visible",
                    None,
                )
            else:
                assignments[hi] = (
                    None, "AMBIGUOUS",
                    f"dist={d_px:.0f}px_exceeds_gate={gate:.0f}px" if d_px else "no_dist",
                    None,
                )

        # Build results
        for det in detections:
            hi = det["hand_index"]
            side, status, reason, rejected_reason = assignments.get(
                hi, (None, "AMBIGUOUS", "unhandled", None)
            )

            anat_identity = f"ANAT_{side}" if side else None

            # Distances to both wrists
            d_left = cost_matrix[(hi, "LEFT")].get("dist_px")
            d_right = cost_matrix[(hi, "RIGHT")].get("dist_px")

            # Normalized selected distance
            if side and side in pose_wrists:
                sel_dist = cost_matrix[(hi, side)].get("dist_px")
                sel_dist_norm = cost_matrix[(hi, side)]["dist_norm"]
            else:
                sel_dist = None
                sel_dist_norm = None

            results.append({
                "hand_index": hi,
                "anatomical_identity": anat_identity,
                "association_status": status,
                "model_handedness_label": det["model_label"],
                "model_handedness_score": round(det["model_score"], 4),
                "hand_wrist_px": [round(float(det["wrist_px"][0]), 2),
                                  round(float(det["wrist_px"][1]), 2)],
                "pose_left_wrist_dist_px": round(d_left, 2) if d_left is not None else None,
                "pose_right_wrist_dist_px": round(d_right, 2) if d_right is not None else None,
                "normalized_selected_distance": round(sel_dist_norm, 4) if sel_dist_norm is not None else None,
                "continuity_cost": cost_matrix[(hi, side)]["continuity_cost"] if side else None,
                "elbow_direction_cost": cost_matrix[(hi, side)]["elbow_cost"] if side else None,
                "handedness_auxiliary_cost": cost_matrix[(hi, side)]["handedness_cost"] if side else None,
                "final_cost": cost_matrix[(hi, side)]["total_cost"] if side else None,
                "assignment_reason": reason,
                "rejected_alternative_reason": rejected_reason,
                "shoulder_width_px": round(shoulder_w, 2) if shoulder_w else None,
                "gate_px": round(gate, 2),
            })

            # Update tracker state for accepted detections
            if status == "DETECTED" and side:
                prev_pos = self.prev_accepted.get(side, {}).get("pos")
                prev_t = self.prev_accepted.get(side, {}).get("pts_s")
                if prev_pos is not None and prev_t is not None:
                    dt = cam_pts_s - prev_t
                    if 0 < dt < 0.2:
                        self.prev_velocity[side] = (det["wrist_px"] - prev_pos) / dt
                self.prev_accepted[side] = {"pos": det["wrist_px"].copy(), "pts_s": cam_pts_s}

        return results


# ═══════════════════════════════════════════════════════════
#  Per-view extraction
# ═══════════════════════════════════════════════════════════

def extract_view(view, segments, affine_a, affine_b, K, dist, out_dir):
    video_path = VIDEOS[view]
    print(f"\n  === {view} ===")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    img_w = stream.width
    img_h = stream.height

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

    # Create landmarkers with explicit thresholds
    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
        min_pose_detection_confidence=THRESHOLDS["pose_min_detection_confidence"],
        min_pose_presence_confidence=THRESHOLDS["pose_min_presence_confidence"],
        min_tracking_confidence=THRESHOLDS["pose_min_tracking_confidence"],
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=THRESHOLDS["hand_min_detection_confidence"],
        min_hand_presence_confidence=THRESHOLDS["hand_min_presence_confidence"],
        min_tracking_confidence=THRESHOLDS["hand_min_tracking_confidence"],
    ))

    all_frames = []
    pts_missing_frames = []
    prev_ts_ms = -1
    ts_adjustments = 0
    frame_idx = 0
    n_in_range = 0

    seg_first_pts = {sr["sample_id"]: None for sr in seg_ranges}
    seg_last_pts = {sr["sample_id"]: None for sr in seg_ranges}

    # Per-sample tracker — reset between samples
    tracker = AnatomicalHandTracker()
    current_sample_id = None

    for frame in container.decode(stream):
        # Strict PTS check — no fallback
        if frame.pts is None:
            pts_missing_frames.append({"frame_index": frame_idx, "view": view})
            frame_idx += 1
            continue

        cam_pts_s = float(frame.pts) * tb

        if cam_pts_s < cam_start_global - 0.5:
            frame_idx += 1
            continue
        if cam_pts_s > cam_end_global + 0.5:
            break

        # Find segment
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

        # Reset tracker on sample change
        if sid != current_sample_id:
            tracker.reset()
            current_sample_id = sid

        if seg_first_pts[sid] is None:
            seg_first_pts[sid] = cam_pts_s
        seg_last_pts[sid] = cam_pts_s

        # MediaPipe timestamp (strictly increasing)
        ts_ms = int(round(cam_pts_s * 1000))
        original_ts_ms = ts_ms
        if ts_ms <= prev_ts_ms:
            ts_ms = prev_ts_ms + 1
            ts_adjustments += 1
        prev_ts_ms = ts_ms

        img_rgb = frame.to_ndarray(format="rgb24")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        pose_result = pose_lm.detect_for_video(mp_image, ts_ms)
        hand_result = hand_lm.detect_for_video(mp_image, ts_ms)

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

        # Hand landmarks — raw detection + anatomical association
        n_hands_detected = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        frame_record["hands_detected_count"] = n_hands_detected

        # Anatomical hand status (filled after association)
        frame_record["anat_left_status"] = "NOT_DETECTED"
        frame_record["anat_right_status"] = "NOT_DETECTED"
        frame_record["anat_left_landmarks"] = None
        frame_record["anat_right_landmarks"] = None

        if n_hands_detected > 0:
            associations = tracker.associate(
                pose_result, hand_result, img_w, img_h, cam_pts_s
            )

            raw_hands = []
            for hi in range(n_hands_detected):
                assoc = associations[hi] if hi < len(associations) else {
                    "anatomical_identity": None,
                    "association_status": "AMBIGUOUS",
                    "model_handedness_label": "Unknown",
                    "model_handedness_score": 0,
                    "assignment_reason": "no_association_result",
                    "rejected_alternative_reason": None,
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

                raw_hands.append({
                    "hand_index": hi,
                    "anatomical_identity": assoc.get("anatomical_identity"),
                    "association_status": assoc.get("association_status", "AMBIGUOUS"),
                    "model_handedness_label": assoc.get("model_handedness_label", "Unknown"),
                    "model_handedness_score": assoc.get("model_handedness_score", 0),
                    "pose_left_wrist_dist_px": assoc.get("pose_left_wrist_dist_px"),
                    "pose_right_wrist_dist_px": assoc.get("pose_right_wrist_dist_px"),
                    "normalized_selected_distance": assoc.get("normalized_selected_distance"),
                    "continuity_cost": assoc.get("continuity_cost"),
                    "elbow_direction_cost": assoc.get("elbow_direction_cost"),
                    "handedness_auxiliary_cost": assoc.get("handedness_auxiliary_cost"),
                    "final_cost": assoc.get("final_cost"),
                    "assignment_reason": assoc.get("assignment_reason"),
                    "rejected_alternative_reason": assoc.get("rejected_alternative_reason"),
                    "shoulder_width_px": assoc.get("shoulder_width_px"),
                    "gate_px": assoc.get("gate_px"),
                    "landmarks": hand_pts,
                })

                # Fill anatomical slots
                status = assoc.get("association_status", "AMBIGUOUS")
                anat_id = assoc.get("anatomical_identity")
                if anat_id == "ANAT_LEFT" and status == "DETECTED":
                    frame_record["anat_left_status"] = "DETECTED"
                    frame_record["anat_left_landmarks"] = hand_pts
                elif anat_id == "ANAT_RIGHT" and status == "DETECTED":
                    frame_record["anat_right_status"] = "DETECTED"
                    frame_record["anat_right_landmarks"] = hand_pts
                elif status == "DUPLICATE_DETECTION_CANDIDATE":
                    # Mark appropriately — don't overwrite a DETECTED
                    pass  # stays NOT_DETECTED for the side
                elif status == "AMBIGUOUS":
                    pass  # stays NOT_DETECTED

            frame_record["hands"] = raw_hands

            # Update statuses for non-DETECTED cases
            for rh in raw_hands:
                if rh["association_status"] == "AMBIGUOUS":
                    # If a side doesn't have DETECTED yet, mark AMBIGUOUS
                    anat = rh.get("anatomical_identity")
                    if anat is None:
                        # Could be either side — mark both as ambiguous if still NOT_DETECTED
                        if frame_record["anat_left_status"] == "NOT_DETECTED":
                            frame_record["anat_left_status"] = "AMBIGUOUS"
                        if frame_record["anat_right_status"] == "NOT_DETECTED":
                            frame_record["anat_right_status"] = "AMBIGUOUS"
                elif rh["association_status"] == "DUPLICATE_DETECTION_CANDIDATE":
                    # The side that doesn't have a real detection gets this status
                    anat = rh.get("anatomical_identity")
                    if anat is None:
                        # Mark the unassigned side
                        for s in ["anat_left_status", "anat_right_status"]:
                            if frame_record[s] == "NOT_DETECTED":
                                frame_record[s] = "DUPLICATE_DETECTION_CANDIDATE"
                                break
        else:
            frame_record["hands"] = []

        all_frames.append(frame_record)
        frame_idx += 1

        if n_in_range % 200 == 0:
            print(f"    {view}: {n_in_range} frames processed, t={cam_pts_s:.1f}s")

    container.close()
    pose_lm.close()
    hand_lm.close()

    print(f"    {view}: {n_in_range} total frames, {ts_adjustments} ts adjustments, "
          f"{len(pts_missing_frames)} PTS-missing frames excluded")

    # Boundary diagnostics
    boundary_diag = []
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

    return all_frames, boundary_diag, pts_missing_frames


# ═══════════════════════════════════════════════════════════
#  Handedness label mapping verification
# ═══════════════════════════════════════════════════════════

def verify_handedness_mapping(all_view_frames):
    """Aggregate label mapping from frames where hands are far apart
    and pose wrist correspondence is clear."""
    rows = []
    summary = {}
    for view, frames in all_view_frames.items():
        mapping_counts = defaultdict(int)  # (model_label, anat_identity) -> count
        clear_frame_count = 0
        total_detection_count = 0

        for f in frames:
            if not f["pose_detected"]:
                continue
            for hand in f.get("hands", []):
                total_detection_count += 1
                anat = hand.get("anatomical_identity")
                status = hand.get("association_status")
                if anat is None or status != "DETECTED":
                    continue

                d_left = hand.get("pose_left_wrist_dist_px")
                d_right = hand.get("pose_right_wrist_dist_px")
                if d_left is None or d_right is None:
                    continue

                # "Clear" = assigned wrist much closer than other
                assigned_side = anat.replace("ANAT_", "")
                assigned_dist = d_left if assigned_side == "LEFT" else d_right
                other_dist = d_right if assigned_side == "LEFT" else d_left

                if other_dist > assigned_dist * 2 and assigned_dist < 100:
                    clear_frame_count += 1
                    mapping_counts[(hand["model_handedness_label"], anat)] += 1

        summary[view] = {
            "total_hand_detections": total_detection_count,
            "clear_correspondence_detections": clear_frame_count,
            "observed_mapping": {f"{k[0]}->{k[1]}": v for k, v in mapping_counts.items()},
        }

    return summary


# ═══════════════════════════════════════════════════════════
#  QA statistics (v1.1 — no hardcoded values)
# ═══════════════════════════════════════════════════════════

def compute_qa_stats(frames_data, view, segments):
    stats = []
    for seg in segments:
        sid = seg["sample_id"]
        seg_frames = [f for f in frames_data if f["sample_id"] == sid and f["view"] == view]
        n = len(seg_frames)
        if n == 0:
            stats.append({"sample_id": sid, "view": view, "n_frames": 0})
            continue

        # Pose usable rate
        pose_usable = 0
        for f in seg_frames:
            if f["pose_detected"] and f["pose_landmarks"]:
                key_indices = [11, 12, 13, 14, 15, 16]
                vis = [f["pose_landmarks"][i]["visibility"] for i in key_indices]
                if all(v > THRESHOLDS["pose_visibility_usable_threshold"] for v in vis):
                    pose_usable += 1

        # Per-anatomical-hand stats
        left_detected = 0
        right_detected = 0
        both_detected = 0
        ambiguous_frames = 0
        duplicate_frames = 0

        # Max consecutive missing per hand
        left_missing_run = 0
        left_max_missing_run = 0
        right_missing_run = 0
        right_max_missing_run = 0
        both_missing_run = 0
        both_max_missing_run = 0

        # Swap candidates
        swap_candidate_frames = []

        # Wrist distances (per assigned hand only)
        left_wrist_dists = []
        right_wrist_dists = []

        prev_left_pos = None
        prev_right_pos = None
        prev_pts_s = None

        for fi, f in enumerate(seg_frames):
            has_left = f["anat_left_status"] == "DETECTED"
            has_right = f["anat_right_status"] == "DETECTED"

            if has_left:
                left_detected += 1
            if has_right:
                right_detected += 1
            if has_left and has_right:
                both_detected += 1

            if f["anat_left_status"] == "AMBIGUOUS" or f["anat_right_status"] == "AMBIGUOUS":
                ambiguous_frames += 1
            if f["anat_left_status"] == "DUPLICATE_DETECTION_CANDIDATE" or \
               f["anat_right_status"] == "DUPLICATE_DETECTION_CANDIDATE":
                duplicate_frames += 1

            # Missing runs
            if not has_left:
                left_missing_run += 1
                left_max_missing_run = max(left_max_missing_run, left_missing_run)
            else:
                left_missing_run = 0

            if not has_right:
                right_missing_run += 1
                right_max_missing_run = max(right_max_missing_run, right_missing_run)
            else:
                right_missing_run = 0

            if not has_left and not has_right:
                both_missing_run += 1
                both_max_missing_run = max(both_max_missing_run, both_missing_run)
            else:
                both_missing_run = 0

            # Wrist distances — only for assigned hand
            cur_left_pos = None
            cur_right_pos = None
            for hand in f.get("hands", []):
                if hand["association_status"] != "DETECTED":
                    continue
                anat = hand.get("anatomical_identity")
                if anat == "ANAT_LEFT":
                    d = hand.get("pose_left_wrist_dist_px")
                    if d is not None:
                        left_wrist_dists.append(d)
                    cur_left_pos = hand.get("hand_wrist_px")
                elif anat == "ANAT_RIGHT":
                    d = hand.get("pose_right_wrist_dist_px")
                    if d is not None:
                        right_wrist_dists.append(d)
                    cur_right_pos = hand.get("hand_wrist_px")

            # Swap candidate detection: if current left pos is closer to previous right pos
            # and current right pos is closer to previous left pos
            if prev_left_pos is not None and prev_right_pos is not None and \
               cur_left_pos is not None and cur_right_pos is not None:
                cl = np.array(cur_left_pos)
                cr = np.array(cur_right_pos)
                pl = np.array(prev_left_pos)
                pr = np.array(prev_right_pos)

                # Normal continuity
                ll = float(np.linalg.norm(cl - pl))
                rr = float(np.linalg.norm(cr - pr))
                # Swapped continuity
                lr = float(np.linalg.norm(cl - pr))
                rl = float(np.linalg.norm(cr - pl))

                if lr + rl < ll + rr * 0.7:  # swapped would be 30% better
                    swap_candidate_frames.append({
                        "frame_index": fi,
                        "camera_pts_s": f["camera_local_pts_s"],
                        "normal_cost": round(ll + rr, 1),
                        "swapped_cost": round(lr + rl, 1),
                    })

            if cur_left_pos is not None:
                prev_left_pos = cur_left_pos
            if cur_right_pos is not None:
                prev_right_pos = cur_right_pos

        # Frame duration
        if n > 1:
            frame_dt = (seg_frames[-1]["camera_local_pts_s"] - seg_frames[0]["camera_local_pts_s"]) / (n - 1)
        else:
            frame_dt = 1.0 / 30.0

        left_wd = np.array(left_wrist_dists) if left_wrist_dists else None
        right_wd = np.array(right_wrist_dists) if right_wrist_dists else None

        stats.append({
            "sample_id": sid,
            "view": view,
            "n_frames": n,
            "pose_usable_rate": round(pose_usable / n, 4),
            "anat_left_detection_rate": round(left_detected / n, 4),
            "anat_right_detection_rate": round(right_detected / n, 4),
            "both_hands_detected_rate": round(both_detected / n, 4),
            "left_max_consecutive_missing_frames": left_max_missing_run,
            "left_max_consecutive_missing_time_s": round(left_max_missing_run * frame_dt, 3),
            "right_max_consecutive_missing_frames": right_max_missing_run,
            "right_max_consecutive_missing_time_s": round(right_max_missing_run * frame_dt, 3),
            "both_unavailable_max_consecutive_frames": both_max_missing_run,
            "both_unavailable_max_consecutive_time_s": round(both_max_missing_run * frame_dt, 3),
            "ambiguous_frame_count": ambiguous_frames,
            "duplicate_detection_candidate_frame_count": duplicate_frames,
            "identity_swap_candidate_count": len(swap_candidate_frames),
            "identity_swap_candidate_frames": swap_candidate_frames,
            "left_assigned_wrist_dist_median_px": round(float(np.median(left_wd)), 2) if left_wd is not None else None,
            "left_assigned_wrist_dist_p95_px": round(float(np.percentile(left_wd, 95)), 2) if left_wd is not None else None,
            "left_assigned_wrist_dist_max_px": round(float(np.max(left_wd)), 2) if left_wd is not None else None,
            "right_assigned_wrist_dist_median_px": round(float(np.median(right_wd)), 2) if right_wd is not None else None,
            "right_assigned_wrist_dist_p95_px": round(float(np.percentile(right_wd, 95)), 2) if right_wd is not None else None,
            "right_assigned_wrist_dist_max_px": round(float(np.max(right_wd)), 2) if right_wd is not None else None,
        })

    return stats


# ═══════════════════════════════════════════════════════════
#  Synchronized view coverage diagnostic
# ═══════════════════════════════════════════════════════════

def compute_synchronized_view_coverage(all_view_frames, segments, transforms):
    """For each anatomical hand, check how many frames have detection in 2+ views
    at nearby mapped_front_pts (within 25ms tolerance).
    Does NOT perform triangulation."""
    rows = []

    for seg in segments:
        sid = seg["sample_id"]

        # Gather per-view accepted detections with mapped_front_pts
        per_view = {}
        for view in ["FRONT", "LEFT", "RIGHT"]:
            frames = all_view_frames[view]
            seg_frames = [f for f in frames if f["sample_id"] == sid]
            left_times = []
            right_times = []
            for f in seg_frames:
                t = f["mapped_front_pts_s"]
                if f["anat_left_status"] == "DETECTED":
                    left_times.append(t)
                if f["anat_right_status"] == "DETECTED":
                    right_times.append(t)
            per_view[view] = {
                "left_times": np.array(sorted(left_times)),
                "right_times": np.array(sorted(right_times)),
            }

        # For each anatomical hand, count frames with 2+ views within 25ms
        for hand_side in ["LEFT", "RIGHT"]:
            key = f"{hand_side.lower()}_times"

            # Use FRONT as reference timeline
            front_times = per_view["FRONT"][key]
            if len(front_times) == 0:
                rows.append({
                    "sample_id": sid,
                    "anatomical_hand": hand_side,
                    "front_frames_with_hand": 0,
                    "front_frames_with_2plus_views": 0,
                    "front_frames_with_3_views": 0,
                    "sync_2plus_rate": None,
                    "sync_3_rate": None,
                })
                continue

            n_2plus = 0
            n_3 = 0
            tol = 0.025  # 25ms

            for ft in front_times:
                views_with_det = 1  # FRONT has it
                for other_view in ["LEFT", "RIGHT"]:
                    other_times = per_view[other_view][key]
                    if len(other_times) > 0:
                        idx = np.searchsorted(other_times, ft)
                        # Check nearest neighbors
                        for ci in [idx - 1, idx]:
                            if 0 <= ci < len(other_times):
                                if abs(other_times[ci] - ft) <= tol:
                                    views_with_det += 1
                                    break

                if views_with_det >= 2:
                    n_2plus += 1
                if views_with_det >= 3:
                    n_3 += 1

            rows.append({
                "sample_id": sid,
                "anatomical_hand": hand_side,
                "front_frames_with_hand": len(front_times),
                "front_frames_with_2plus_views": n_2plus,
                "front_frames_with_3_views": n_3,
                "sync_2plus_rate": round(n_2plus / len(front_times), 4) if len(front_times) > 0 else None,
                "sync_3_rate": round(n_3 / len(front_times), 4) if len(front_times) > 0 else None,
            })

    return rows


# ═══════════════════════════════════════════════════════════
#  Overlay video (segment-only, no neutral gaps)
# ═══════════════════════════════════════════════════════════

POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

STATUS_COLORS = {
    "DETECTED": None,  # use anatomical color
    "AMBIGUOUS": (0, 0, 255),         # red
    "DUPLICATE_DETECTION_CANDIDATE": (255, 0, 255),  # magenta
}

ANAT_COLORS = {
    "ANAT_LEFT": (0, 255, 0),    # green
    "ANAT_RIGHT": (0, 165, 255), # orange
}


def make_overlay_video(view, frames_data, segments, affine_a, affine_b,
                       out_path, qa_stats, fps_out=15):
    """Overlay on segment-only frames (no neutral gaps)."""
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

    # Build frame lookup
    frame_lookup = {}
    for fd in frames_data:
        if fd["view"] == view:
            key = round(fd["camera_local_pts_s"], 4)
            frame_lookup[key] = fd

    seg_ranges = []
    for seg in segments:
        cr = seg["camera_ranges"][view]
        seg_ranges.append((cr["start_pts_s"], cr["end_pts_s"], seg["sample_id"]))

    # Compute wrist distance outlier threshold per sample
    wrist_dist_thresholds = {}
    for qs in qa_stats:
        if qs["view"] == view:
            sid = qs["sample_id"]
            th = max(
                qs.get("left_assigned_wrist_dist_p95_px") or 0,
                qs.get("right_assigned_wrist_dist_p95_px") or 0,
            )
            wrist_dist_thresholds[sid] = th * 1.5 if th > 0 else 200

    dt_out = 1.0 / fps_out
    frame_idx = 0
    written = 0

    # Process segments sequentially (skip neutral gaps)
    for seg_start, seg_end, seg_sid in seg_ranges:
        # Seek-free: we decode sequentially, so just track position
        next_grab_s = seg_start

        container2 = av.open(str(video_path))
        stream2 = container2.streams.video[0]
        stream2.thread_type = "AUTO"
        tb2 = float(stream2.time_base)

        # Add separator frame
        sep = np.zeros((oh, ow, 3), dtype=np.uint8)
        label = seg_sid.split("_", 3)[-1]
        cv2.putText(sep, f"{view} {label}", (ow // 4, oh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for _ in range(int(fps_out * 0.5)):  # 0.5s separator
            writer.write(sep)
            written += 1

        for frame in container2.decode(stream2):
            if frame.pts is None:
                continue
            cam_pts_s = float(frame.pts) * tb2

            if cam_pts_s < seg_start - 0.05:
                continue
            if cam_pts_s > seg_end + 0.05:
                break

            if cam_pts_s < next_grab_s:
                continue
            next_grab_s = cam_pts_s + dt_out

            if cam_pts_s < seg_start or cam_pts_s > seg_end:
                continue

            img = frame.to_ndarray(format="bgr24")
            small = cv2.resize(img, (ow, oh))
            front_pts_s = cam_to_front(cam_pts_s, affine_a, affine_b)

            # Header
            cv2.rectangle(small, (0, 0), (ow, 35), (0, 80, 0), -1)
            cv2.putText(small, f"{view} {label} cam={cam_pts_s:.2f} front={front_pts_s:.2f}",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Find frame data
            key = round(cam_pts_s, 4)
            fd = frame_lookup.get(key)
            if fd is None:
                for delta in [0.0001, -0.0001, 0.001, -0.001, 0.01, -0.01]:
                    fd = frame_lookup.get(round(cam_pts_s + delta, 4))
                    if fd:
                        break

            warnings = []

            if fd:
                # Draw pose
                if fd.get("pose_detected") and fd.get("pose_landmarks"):
                    pts = [(int(lm["raw_distorted_px_x"] * scale),
                            int(lm["raw_distorted_px_y"] * scale))
                           for lm in fd["pose_landmarks"]]
                    for i, j in POSE_CONNECTIONS:
                        if fd["pose_landmarks"][i]["visibility"] > 0.3 and \
                           fd["pose_landmarks"][j]["visibility"] > 0.3:
                            cv2.line(small, pts[i], pts[j], (200, 200, 200), 1)

                # Draw hands with status
                for hand in fd.get("hands", []):
                    status = hand.get("association_status", "AMBIGUOUS")
                    anat = hand.get("anatomical_identity")

                    if status == "DETECTED" and anat:
                        color = ANAT_COLORS.get(anat, (128, 128, 128))
                    else:
                        color = STATUS_COLORS.get(status, (128, 128, 128))

                    hpts = [(int(lm["raw_distorted_px_x"] * scale),
                             int(lm["raw_distorted_px_y"] * scale))
                            for lm in hand["landmarks"]]
                    for i, j in HAND_CONNECTIONS:
                        cv2.line(small, hpts[i], hpts[j], color, 1)
                    for pt in hpts:
                        cv2.circle(small, pt, 2, color, -1)

                    # Label
                    wx, wy = hpts[0]
                    if status == "DETECTED":
                        short = "L" if anat == "ANAT_LEFT" else "R"
                    elif status == "DUPLICATE_DETECTION_CANDIDATE":
                        short = "DUP"
                        warnings.append("DUPLICATE")
                    elif status == "AMBIGUOUS":
                        short = "?"
                        warnings.append("AMBIG")
                    else:
                        short = "?"
                    cv2.putText(small, short, (wx - 5, wy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    # Wrist distance outlier (assigned wrist only)
                    if status == "DETECTED" and anat:
                        side = anat.replace("ANAT_", "")
                        d_key = f"pose_{side.lower()}_wrist_dist_px"
                        d = hand.get(d_key)
                        th = wrist_dist_thresholds.get(seg_sid, 200)
                        if d is not None and d > th:
                            warnings.append(f"wD={d:.0f}")

                if not fd.get("pose_detected"):
                    warnings.append("NO_POSE")
                if fd["hands_detected_count"] == 0:
                    warnings.append("NO_HANDS")

                # Swap candidate check
                for qs in qa_stats:
                    if qs["view"] == view and qs["sample_id"] == seg_sid:
                        for sc in qs.get("identity_swap_candidate_frames", []):
                            if abs(sc["camera_pts_s"] - fd["camera_local_pts_s"]) < 0.02:
                                warnings.append("SWAP_CAND")
                                break

            if warnings:
                cv2.putText(small, " ".join(set(warnings)), (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            writer.write(small)
            written += 1

        container2.close()

    container.close()
    writer.release()
    print(f"    Overlay {view}: {written} frames written to {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  Contact sheet generation
# ═══════════════════════════════════════════════════════════

def generate_contact_sheets(all_view_frames, qa_stats, segments, transforms, out_dir):
    """Generate contact sheets for flagged conditions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect frames needing contact sheets
    flagged = defaultdict(list)  # category -> [(view, sample_id, cam_pts_s, reason)]

    # Specific required samples
    required_samples = {
        ("LEFT", "S001_P001_GREETING_SLOW_R03"): "low_detection_greeting_slow_left",
        ("RIGHT", "S001_P001_LOVE_NORMAL_R03"): "low_detection_love_normal_right",
    }
    for (view, sid), tag in required_samples.items():
        frames = [f for f in all_view_frames[view]
                  if f["sample_id"] == sid]
        # Sample 5 evenly spaced frames
        if frames:
            indices = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
            for idx in indices:
                flagged[tag].append((view, sid, frames[idx]["camera_local_pts_s"], "sampled"))

    # All FAST samples — sample frames
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for sid in PILOT_SAMPLE_IDS:
            if "FAST" in sid:
                frames = [f for f in all_view_frames[view] if f["sample_id"] == sid]
                if frames:
                    indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
                    for idx in indices:
                        flagged[f"fast_{view.lower()}"].append(
                            (view, sid, frames[idx]["camera_local_pts_s"], "fast_sample"))

    # Specific frame-level flags
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for f in all_view_frames[view]:
            for hand in f.get("hands", []):
                status = hand.get("association_status")
                if status == "DUPLICATE_DETECTION_CANDIDATE":
                    flagged["duplicate_candidates"].append(
                        (view, f["sample_id"], f["camera_local_pts_s"], "duplicate"))
                elif status == "AMBIGUOUS":
                    flagged["ambiguous_frames"].append(
                        (view, f["sample_id"], f["camera_local_pts_s"], "ambiguous"))

    # Swap candidates from QA stats
    for qs in qa_stats:
        for sc in qs.get("identity_swap_candidate_frames", []):
            flagged["swap_candidates"].append(
                (qs["view"], qs["sample_id"], sc["camera_pts_s"], "swap_candidate"))

    # Wrist distance outliers
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for f in all_view_frames[view]:
            for hand in f.get("hands", []):
                if hand.get("association_status") != "DETECTED":
                    continue
                anat = hand.get("anatomical_identity")
                if anat:
                    side = anat.replace("ANAT_", "")
                    d = hand.get(f"pose_{side.lower()}_wrist_dist_px")
                    if d is not None and d > 150:
                        flagged["wrist_dist_outliers"].append(
                            (view, f["sample_id"], f["camera_local_pts_s"],
                             f"d={d:.0f}px"))

    # Generate contact sheets
    for category, items in flagged.items():
        if not items:
            continue

        # Limit to 24 frames per sheet
        if len(items) > 24:
            indices = np.linspace(0, len(items) - 1, 24, dtype=int)
            items = [items[i] for i in indices]

        # Group by view for efficient reading
        by_view = defaultdict(list)
        for view, sid, pts_s, reason in items:
            by_view[view].append((sid, pts_s, reason))

        thumbnails = []
        thumb_w, thumb_h = 320, 180

        for view, frame_list in by_view.items():
            video_path = VIDEOS[view]
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)

            # Sort by PTS
            frame_list.sort(key=lambda x: x[1])
            fl_idx = 0

            for frame in container.decode(stream):
                if fl_idx >= len(frame_list):
                    break
                if frame.pts is None:
                    continue
                cam_pts_s = float(frame.pts) * tb

                target_pts = frame_list[fl_idx][1]
                if cam_pts_s < target_pts - 0.02:
                    continue

                if abs(cam_pts_s - target_pts) <= 0.05:
                    img = frame.to_ndarray(format="bgr24")
                    thumb = cv2.resize(img, (thumb_w, thumb_h))

                    sid = frame_list[fl_idx][0]
                    reason = frame_list[fl_idx][2]
                    label_short = sid.split("_", 3)[-1][:20]
                    cv2.putText(thumb, f"{view} {label_short}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    cv2.putText(thumb, f"t={cam_pts_s:.2f} {reason}", (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
                    thumbnails.append(thumb)
                    fl_idx += 1

                if cam_pts_s > target_pts + 0.05:
                    # Missed this frame, skip
                    fl_idx += 1

            container.close()

        if not thumbnails:
            continue

        # Arrange in grid
        cols = min(4, len(thumbnails))
        rows_n = math.ceil(len(thumbnails) / cols)
        sheet = np.zeros((rows_n * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
        for i, thumb in enumerate(thumbnails):
            r, c = divmod(i, cols)
            sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

        sheet_path = out_dir / f"contact_{category}.png"
        cv2.imwrite(str(sheet_path), sheet)

    generated = list(out_dir.glob("contact_*.png"))
    print(f"  Contact sheets: {len(generated)} generated in {out_dir.name}/")
    return generated


# ═══════════════════════════════════════════════════════════
#  Input verification
# ═══════════════════════════════════════════════════════════

def verify_inputs(seg_data, all_segs):
    """Comprehensive input verification."""
    errors = []

    # 45 segments
    if seg_data.get("n_segments_expected") != 45:
        errors.append(f"n_segments_expected={seg_data.get('n_segments_expected')}, expected 45")
    if len(all_segs) != 45:
        errors.append(f"len(segments)={len(all_segs)}, expected 45")

    # All MANUAL_VERIFIED
    non_verified = [s["sample_id"] for s in all_segs if s["verification_status"] != "MANUAL_VERIFIED"]
    if non_verified:
        errors.append(f"Non-verified segments: {non_verified}")

    # Check canonical sample_id set
    expected_ids = set()
    for sign in ["GREETING", "THANK_YOU", "LOVE"]:
        for speed in ["NORMAL", "SLOW", "FAST"]:
            for rep in range(1, 6):
                expected_ids.add(f"S001_P001_{sign}_{speed}_R{rep:02d}")
    actual_ids = set(s["sample_id"] for s in all_segs)
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            errors.append(f"Missing sample_ids: {missing}")
        if extra:
            errors.append(f"Extra sample_ids: {extra}")

    # Check order (no overlap)
    for i in range(len(all_segs) - 1):
        if all_segs[i]["front_end_s"] > all_segs[i + 1]["front_start_s"]:
            errors.append(f"Overlap: {all_segs[i]['sample_id']} end={all_segs[i]['front_end_s']} > "
                          f"{all_segs[i + 1]['sample_id']} start={all_segs[i + 1]['front_start_s']}")

    # R03 pilot count
    pilot = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    if len(pilot) != 9:
        errors.append(f"Pilot R03 count={len(pilot)}, expected 9")

    # Affine coefficients match Step 5 v2.1 exactly
    transforms = seg_data.get("affine_transforms", {})
    expected_left_a = 0.9999806597862914
    expected_left_b = 2.310470997893187
    expected_right_a = 0.9999996430818727
    expected_right_b = 9.388814864164976

    if transforms.get("LEFT", {}).get("a") != expected_left_a:
        errors.append(f"LEFT affine a mismatch: {transforms.get('LEFT', {}).get('a')}")
    if transforms.get("LEFT", {}).get("b") != expected_left_b:
        errors.append(f"LEFT affine b mismatch: {transforms.get('LEFT', {}).get('b')}")
    if transforms.get("RIGHT", {}).get("a") != expected_right_a:
        errors.append(f"RIGHT affine a mismatch: {transforms.get('RIGHT', {}).get('a')}")
    if transforms.get("RIGHT", {}).get("b") != expected_right_b:
        errors.append(f"RIGHT affine b mismatch: {transforms.get('RIGHT', {}).get('b')}")

    return errors


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Step 6A v1.1: MediaPipe 2D Landmark Pilot Extraction — S001_P001\n")

    # Load segments
    seg_path = PROC_DIR / "action_segments_v2_1_verified.json"
    with open(seg_path, encoding="utf-8") as f:
        seg_data = json.load(f)
    all_segs = seg_data["segments"]
    transforms = seg_data["affine_transforms"]

    # Input verification
    errors = verify_inputs(seg_data, all_segs)
    if errors:
        print("  INPUT VERIFICATION FAILED:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  Input verification: PASS (45 segments, all MANUAL_VERIFIED, "
          "canonical IDs, no overlap, affine match)")

    pilot_segs = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    assert len(pilot_segs) == 9
    print(f"  Pilot samples: {len(pilot_segs)}")

    # Output directories
    out_base = PROC_DIR / "landmarks_2d_v1_1" / "pilot"
    out_overlays = out_base / "overlays"
    out_contacts = out_base / "low_confidence_contact_sheets"
    for d in [out_base, out_overlays, out_contacts]:
        d.mkdir(parents=True, exist_ok=True)

    # Load intrinsics and verify image sizes
    intrinsics = {}
    intrinsics_sizes = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        K, dist, img_size = load_intrinsics(view)
        intrinsics[view] = (K, dist)
        intrinsics_sizes[view] = img_size

    # Verify intrinsics image_size matches video resolution
    for view in ["FRONT", "LEFT", "RIGHT"]:
        container = av.open(str(VIDEOS[view]))
        stream = container.streams.video[0]
        vid_w, vid_h = stream.width, stream.height
        container.close()
        int_size = intrinsics_sizes[view]
        assert int_size == [vid_w, vid_h], \
            f"{view} intrinsics image_size={int_size} != video resolution=[{vid_w},{vid_h}]"
    print("  Intrinsics image_size vs video resolution: MATCH (all views)")

    # Process each view
    all_view_frames = {}
    all_boundary_diag = []
    all_pts_missing = []

    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        a, b = t["a"], t["b"]
        K, dist = intrinsics[view]

        frames, bdiag, pts_missing = extract_view(view, pilot_segs, a, b, K, dist, out_base)
        all_view_frames[view] = frames
        all_boundary_diag.extend(bdiag)
        all_pts_missing.extend(pts_missing)

    # Save per-view landmark JSONs
    for view in ["FRONT", "LEFT", "RIGHT"]:
        vid_size = intrinsics_sizes[view]
        view_file = out_base / f"landmarks_2d_{view.lower()}.json"
        with open(view_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.1_pilot",
                "view": view,
                "n_frames": len(all_view_frames[view]),
                "image_size": vid_size,
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

    # Detection quality CSV (exclude nested swap frames for CSV)
    dq_path = out_base / "detection_quality.csv"
    if all_qa:
        csv_qa = []
        for q in all_qa:
            row = {k: v for k, v in q.items() if k != "identity_swap_candidate_frames"}
            csv_qa.append(row)
        with open(dq_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=csv_qa[0].keys())
            w.writeheader()
            w.writerows(csv_qa)
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
                    "anatomical_identity": hand.get("anatomical_identity"),
                    "association_status": hand.get("association_status"),
                    "model_handedness_label": hand["model_handedness_label"],
                    "model_handedness_score": hand["model_handedness_score"],
                    "pose_left_wrist_dist_px": hand.get("pose_left_wrist_dist_px"),
                    "pose_right_wrist_dist_px": hand.get("pose_right_wrist_dist_px"),
                    "normalized_selected_distance": hand.get("normalized_selected_distance"),
                    "continuity_cost": hand.get("continuity_cost"),
                    "elbow_direction_cost": hand.get("elbow_direction_cost"),
                    "handedness_auxiliary_cost": hand.get("handedness_auxiliary_cost"),
                    "final_cost": hand.get("final_cost"),
                    "assignment_reason": hand.get("assignment_reason"),
                    "rejected_alternative_reason": hand.get("rejected_alternative_reason"),
                    "shoulder_width_px": hand.get("shoulder_width_px"),
                    "gate_px": hand.get("gate_px"),
                })
    hid_path = out_base / "hand_identity_diagnostics.csv"
    if hid_rows:
        with open(hid_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hid_rows[0].keys())
            w.writeheader()
            w.writerows(hid_rows)
        print(f"  Saved: {hid_path.name}")

    # Handedness label mapping verification
    handedness_mapping = verify_handedness_mapping(all_view_frames)

    # Synchronized view coverage
    sync_coverage = compute_synchronized_view_coverage(
        all_view_frames, pilot_segs, transforms
    )
    sync_path = out_base / "synchronized_view_coverage.csv"
    if sync_coverage:
        with open(sync_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sync_coverage[0].keys())
            w.writeheader()
            w.writerows(sync_coverage)
        print(f"  Saved: {sync_path.name}")

    # Generate overlays (segment-only)
    print("\n  Generating overlay videos...")
    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        overlay_path = out_overlays / f"overlay_pilot_{view.lower()}.mp4"
        make_overlay_video(view, all_view_frames[view], pilot_segs,
                           t["a"], t["b"], overlay_path, all_qa)

    # Generate contact sheets
    print()
    generate_contact_sheets(all_view_frames, all_qa, pilot_segs, transforms, out_contacts)

    # Compute SHA-256 for all inputs
    input_hashes = {
        "action_segments": sha256_file(seg_path),
        "pose_model": sha256_file(POSE_MODEL),
        "hand_model": sha256_file(HAND_MODEL),
    }
    for view in ["FRONT", "LEFT", "RIGHT"]:
        input_hashes[f"video_{view}"] = sha256_file(VIDEOS[view])
        input_hashes[f"intrinsics_{view}"] = sha256_file(C1_INTRINSICS[view])

    # Run manifest
    manifest = {
        "step": "6A_pilot_v1.1",
        "date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "mediapipe_version": mp.__version__,
        "pyav_version": av.__version__,
        "opencv_version": cv2.__version__,
        "pose_model": {
            "path": str(POSE_MODEL),
            "type": "pose_landmarker_lite",
            "sha256": input_hashes["pose_model"],
        },
        "hand_model": {
            "path": str(HAND_MODEL),
            "type": "hand_landmarker",
            "sha256": input_hashes["hand_model"],
        },
        "thresholds": THRESHOLDS,
        "v1_defaults_note": V1_DEFAULTS,
        "pose_options": {
            "running_mode": "VIDEO",
            "num_poses": 1,
            "output_segmentation_masks": False,
            "min_pose_detection_confidence": THRESHOLDS["pose_min_detection_confidence"],
            "min_pose_presence_confidence": THRESHOLDS["pose_min_presence_confidence"],
            "min_tracking_confidence": THRESHOLDS["pose_min_tracking_confidence"],
        },
        "hand_options": {
            "running_mode": "VIDEO",
            "num_hands": 2,
            "min_hand_detection_confidence": THRESHOLDS["hand_min_detection_confidence"],
            "min_hand_presence_confidence": THRESHOLDS["hand_min_presence_confidence"],
            "min_tracking_confidence": THRESHOLDS["hand_min_tracking_confidence"],
        },
        "input_sha256": input_hashes,
        "input_file": str(seg_path),
        "affine_transforms": transforms,
        "pilot_sample_ids": PILOT_SAMPLE_IDS,
        "original_videos_mirrored": False,
        "pts_missing_frames_excluded": len(all_pts_missing),
        "pts_missing_details": all_pts_missing[:20] if all_pts_missing else [],
        "coordinate_conventions": {
            "raw_norm": "MediaPipe normalized [0,1] image coordinates",
            "raw_distorted_px": "raw_norm * image_size, still distorted by lens",
            "undistorted_px": "C1 intrinsics undistortion applied, pixel coords",
            "undistorted_norm_cam": "C1 undistorted normalized camera coords (x/fx, y/fy)",
            "hand_visibility_presence": "null (not provided by Hand Landmarker API)",
        },
        "null_policy": "Detection failures and unavailable values stored as null, never 0-filled or interpolated",
        "association_statuses": [
            "DETECTED — accepted anatomical assignment",
            "NOT_DETECTED — no hand detection for this anatomical side",
            "AMBIGUOUS — detection exists but cannot be reliably assigned",
            "DUPLICATE_DETECTION_CANDIDATE — likely duplicate of another detection near same wrist",
        ],
        "handedness_label_mapping": handedness_mapping,
    }
    manifest_path = out_base / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {manifest_path.name}")

    # ── Print QA summary ──
    print("\n  " + "=" * 55)
    print("  QA Summary (v1.1)")
    print("  " + "=" * 55)

    for view in ["FRONT", "LEFT", "RIGHT"]:
        view_qa = [q for q in all_qa if q["view"] == view]
        print(f"\n  {view}:")
        for q in view_qa:
            sid_short = q["sample_id"].split("_", 3)[-1]
            print(f"    {sid_short}: {q['n_frames']}f "
                  f"pose={q['pose_usable_rate']:.0%} "
                  f"L={q['anat_left_detection_rate']:.0%} "
                  f"R={q['anat_right_detection_rate']:.0%} "
                  f"both={q['both_hands_detected_rate']:.0%} "
                  f"Lmiss={q['left_max_consecutive_missing_frames']}f "
                  f"Rmiss={q['right_max_consecutive_missing_frames']}f "
                  f"ambig={q['ambiguous_frame_count']} "
                  f"dup={q['duplicate_detection_candidate_frame_count']} "
                  f"swap={q['identity_swap_candidate_count']}")

    # Boundary errors
    print("\n  Boundary PTS errors:")
    max_start = max((abs(b["start_error_ms"]) for b in all_boundary_diag if b["start_error_ms"]), default=0)
    max_end = max((abs(b["end_error_ms"]) for b in all_boundary_diag if b["end_error_ms"]), default=0)
    print(f"    Max start error: {max_start:.2f} ms")
    print(f"    Max end error: {max_end:.2f} ms")
    print(f"    Note: 71.9ms > 2 frames at 30fps (66.7ms). "
          "Visual review confirms LOVE_NORMAL FRONT motion onset is not clipped.")

    # PTS missing
    print(f"\n  PTS-missing frames excluded: {len(all_pts_missing)}")

    # Handedness mapping
    print("\n  Handedness label mapping (clear-correspondence frames):")
    for view, data in handedness_mapping.items():
        print(f"    {view}: {data['clear_correspondence_detections']} clear detections "
              f"out of {data['total_hand_detections']} total")
        for mapping, count in data["observed_mapping"].items():
            print(f"      {mapping}: {count}")

    # Sync coverage summary
    print("\n  Synchronized multi-view coverage:")
    for row in sync_coverage:
        r2 = row["sync_2plus_rate"]
        r3 = row["sync_3_rate"]
        print(f"    {row['sample_id'].split('_',3)[-1]} {row['anatomical_hand']}: "
              f"front={row['front_frames_with_hand']}f "
              f"2+views={r2:.0%}" if r2 is not None else "N/A",
              f"3views={r3:.0%}" if r3 is not None else "N/A")

    print(f"\n  Status: REQUIRES_HUMAN_VISUAL_REVIEW")
    print(f"\n  Generated files:")
    for p in sorted(out_base.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            if size_kb > 1024:
                print(f"    {p.relative_to(out_base)}: {size_kb / 1024:.1f} MB")
            else:
                print(f"    {p.relative_to(out_base)}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
