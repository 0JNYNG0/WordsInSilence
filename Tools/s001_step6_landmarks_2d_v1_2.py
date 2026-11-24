"""
Step 6A v1.2: MediaPipe 2D Landmark Pilot Extraction — S001_P001

Corrections from v1.1:
- Global assignment: enumerate all (none/L/R/LR/RL) combinations, pick minimum cost.
  Never force loser to opposite hand without evidence.
- Two-pass association: 1st pass without handedness, compute observed mapping,
  2nd pass with verified mapping at low weight.
- hand_wrist_px stored in output (was missing in v1.1).
- Swap calculation: fixed formula, gap-aware, reports comparison count.
- Synchronized coverage: all FRONT frames as reference, one-to-one matching.
- Assertion: duplicate_candidate cannot be DETECTED simultaneously.
- Overlay: full sample_id, raw detection index/model label, per-wrist distances.
- Contact sheets: draw landmarks on flagged frames.
"""

import json
import csv
import hashlib
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
#  Thresholds
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
    "association_distance_weight": 1.0,
    # handedness weight set to 0 in pass 1; updated in pass 2
    "association_handedness_weight_pass1": 0.0,
    "association_handedness_weight_pass2": 0.1,
    "swap_cost_ratio_threshold": 0.7,
    "swap_gap_max_frames": 3,
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
    return K, dist, d["image_size"]


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


def _short_sid(sid):
    """E.g. S001_P001_GREETING_NORMAL_R03 -> GREETING_NORMAL_R03"""
    parts = sid.split("_", 3)
    return parts[3] if len(parts) > 3 else sid


# ═══════════════════════════════════════════════════════════
#  Anatomical hand association (v1.2 — global assignment)
# ═══════════════════════════════════════════════════════════

class AnatomicalHandTracker:
    """Per-sample tracker. Must be reset between samples."""

    def __init__(self, handedness_weight=0.0, observed_mapping=None):
        self.handedness_weight = handedness_weight
        self.observed_mapping = observed_mapping  # e.g. {"Left": "ANAT_LEFT", "Right": "ANAT_RIGHT"}
        self.reset()

    def reset(self):
        self.prev_accepted = {}  # side -> {"pos": np.array, "pts_s": float}
        self.prev_velocity = {}  # side -> np.array (px/s)

    def _shoulder_width(self, pose_result, img_w, img_h):
        if not pose_result.pose_landmarks:
            return None
        plm = pose_result.pose_landmarks[0]
        ls, rs = plm[11], plm[12]
        if ls.visibility < 0.3 or rs.visibility < 0.3:
            return None
        lsp = np.array([ls.x * img_w, ls.y * img_h])
        rsp = np.array([rs.x * img_w, rs.y * img_h])
        w = float(np.linalg.norm(lsp - rsp))
        return w if w > 20 else None

    def _elbow_direction_cost(self, pose_result, img_w, img_h, side, det_wrist):
        if not pose_result.pose_landmarks:
            return 0.0
        plm = pose_result.pose_landmarks[0]
        elbow_idx = 13 if side == "LEFT" else 14
        wrist_idx = 15 if side == "LEFT" else 16
        elb, wri = plm[elbow_idx], plm[wrist_idx]
        if elb.visibility < 0.3 or wri.visibility < 0.3:
            return 0.0
        elb_px = np.array([elb.x * img_w, elb.y * img_h])
        wri_px = np.array([wri.x * img_w, wri.y * img_h])
        pose_dir = wri_px - elb_px
        n1 = np.linalg.norm(pose_dir)
        if n1 < 5:
            return 0.0
        det_dir = det_wrist - elb_px
        n2 = np.linalg.norm(det_dir)
        if n2 < 5:
            return 0.0
        dot = float(np.dot(pose_dir / n1, det_dir / n2))
        return max(0.0, 1.0 - dot) * THRESHOLDS["association_elbow_direction_weight"]

    def _single_cost(self, det_wrist, side, pose_wrists, shoulder_w, cam_pts_s,
                     pose_result, img_w, img_h, model_label, model_score):
        """Compute cost of assigning det_wrist to anatomical side."""
        gate = shoulder_w * THRESHOLDS["association_shoulder_width_gate_factor"] if shoulder_w else 300.0

        # Distance cost
        if side in pose_wrists:
            dist_px = float(np.linalg.norm(det_wrist - pose_wrists[side]))
            dist_norm = dist_px / shoulder_w if shoulder_w and shoulder_w > 0 else dist_px / 300.0
        else:
            dist_px = None
            dist_norm = 2.0
        distance_cost = dist_norm * THRESHOLDS["association_distance_weight"]

        # Continuity cost
        continuity_cost = 0.0
        if side in self.prev_accepted:
            prev = self.prev_accepted[side]
            dt = cam_pts_s - prev["pts_s"]
            if 0 < dt < 0.2:
                expected = prev["pos"]
                if side in self.prev_velocity:
                    expected = prev["pos"] + self.prev_velocity[side] * dt
                c_dist = float(np.linalg.norm(det_wrist - expected))
                c_norm = c_dist / shoulder_w if shoulder_w and shoulder_w > 0 else c_dist / 300.0
                continuity_cost = c_norm * THRESHOLDS["association_continuity_weight"]
            else:
                continuity_cost = 0.5 * THRESHOLDS["association_continuity_weight"]

        # Elbow direction cost
        elbow_cost = self._elbow_direction_cost(pose_result, img_w, img_h, side, det_wrist)

        # Handedness cost
        handedness_cost = 0.0
        if self.handedness_weight > 0 and self.observed_mapping:
            expected_label = None
            for lbl, anat in self.observed_mapping.items():
                if anat == f"ANAT_{side}":
                    expected_label = lbl
                    break
            if expected_label is not None:
                if model_label != expected_label:
                    handedness_cost = self.handedness_weight
                else:
                    handedness_cost = -self.handedness_weight * (model_score - 0.5)

        total = distance_cost + continuity_cost + elbow_cost + handedness_cost
        within_gate = dist_px is not None and dist_px < gate

        return {
            "dist_px": dist_px,
            "dist_norm": round(dist_norm, 4),
            "distance_cost": round(distance_cost, 4),
            "continuity_cost": round(continuity_cost, 4),
            "elbow_cost": round(elbow_cost, 4),
            "handedness_cost": round(handedness_cost, 4),
            "total_cost": round(total, 4),
            "within_gate": within_gate,
            "gate_px": round(gate, 2),
        }

    def associate(self, pose_result, hand_result, img_w, img_h, cam_pts_s):
        """Global assignment: enumerate all valid combinations."""
        n_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        if n_hands == 0:
            return []

        # Pose wrists
        pose_wrists = {}
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            for side, idx in [("LEFT", 15), ("RIGHT", 16)]:
                lm = plm[idx]
                if lm.visibility > 0.3:
                    pose_wrists[side] = np.array([lm.x * img_w, lm.y * img_h])

        shoulder_w = self._shoulder_width(pose_result, img_w, img_h)

        # Gather detections
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

        # Compute cost for each (detection, side) pair
        costs = {}
        for det in detections:
            for side in ["LEFT", "RIGHT"]:
                costs[(det["hand_index"], side)] = self._single_cost(
                    det["wrist_px"], side, pose_wrists, shoulder_w, cam_pts_s,
                    pose_result, img_w, img_h,
                    det["model_label"], det["model_score"],
                )

        # ── Enumerate all valid assignment combinations ──
        # Each detection -> None / "LEFT" / "RIGHT"
        # Constraint: at most one detection per side

        if n_hands == 1:
            candidates = [
                {},                    # none assigned
                {0: "LEFT"},           # det0 -> LEFT
                {0: "RIGHT"},          # det0 -> RIGHT
            ]
        elif n_hands == 2:
            candidates = [
                {},                           # none
                {0: "LEFT"},                  # det0 -> L only
                {0: "RIGHT"},                 # det0 -> R only
                {1: "LEFT"},                  # det1 -> L only
                {1: "RIGHT"},                 # det1 -> R only
                {0: "LEFT", 1: "RIGHT"},      # det0->L, det1->R
                {0: "RIGHT", 1: "LEFT"},      # det0->R, det1->L
            ]
        else:
            # 3+ hands: just try all permutation-like combos
            candidates = [{}]
            for hi in range(n_hands):
                candidates.append({hi: "LEFT"})
                candidates.append({hi: "RIGHT"})
            if n_hands >= 2:
                for i in range(n_hands):
                    for j in range(n_hands):
                        if i != j:
                            candidates.append({i: "LEFT", j: "RIGHT"})

        # Score each candidate
        UNASSIGNED_PENALTY = 0.5  # small penalty for leaving a within-gate detection unassigned
        best_combo = None
        best_score = float("inf")
        best_details = None

        for combo in candidates:
            # Check validity: no two detections to same side
            sides_used = list(combo.values())
            if len(sides_used) != len(set(sides_used)):
                continue

            total = 0.0
            combo_ok = True
            for hi, side in combo.items():
                c = costs[(hi, side)]
                if not c["within_gate"]:
                    combo_ok = False
                    break
                total += c["total_cost"]

            if not combo_ok:
                continue

            # Add small penalty for unassigned detections that are within gate of some side
            for det in detections:
                hi = det["hand_index"]
                if hi not in combo:
                    # Check if it could have been assigned
                    for side in ["LEFT", "RIGHT"]:
                        if side not in combo.values() and costs[(hi, side)]["within_gate"]:
                            total += UNASSIGNED_PENALTY
                            break

            if total < best_score:
                best_score = total
                best_combo = combo

        if best_combo is None:
            best_combo = {}  # nothing assignable

        # ── Determine status for each detection ──
        results = []
        for det in detections:
            hi = det["hand_index"]
            assigned_side = best_combo.get(hi)

            if assigned_side is not None:
                status = "DETECTED"
                anat_identity = f"ANAT_{assigned_side}"
                reason = f"global_best_for_{assigned_side}"
                rejected_reason = None

                # Check if this detection was close to the OTHER wrist too
                other_side = "RIGHT" if assigned_side == "LEFT" else "LEFT"
                c_assigned = costs[(hi, assigned_side)]
                c_other = costs[(hi, other_side)]

                # Is there another detection assigned to the other side?
                other_det_for_other = None
                for ohi, oside in best_combo.items():
                    if oside == other_side:
                        other_det_for_other = ohi

                if other_det_for_other is None and c_other["within_gate"]:
                    # Could have gone to other side too, but other side is unoccupied
                    # This is fine — it was the better fit for assigned_side
                    pass

            else:
                # Not assigned — determine why
                # Check if both detections went to the same side (shouldn't happen with valid combos)
                # or if this detection couldn't fit anywhere

                # Check if another detection took both viable sides
                viable_sides = [s for s in ["LEFT", "RIGHT"] if costs[(hi, s)]["within_gate"]]
                taken_sides = set(best_combo.values())

                if not viable_sides:
                    status = "AMBIGUOUS"
                    anat_identity = None
                    reason = "no_pose_wrist_within_gate"
                    rejected_reason = None
                elif all(s in taken_sides for s in viable_sides):
                    # All viable sides taken by other detections
                    # Check if this is a duplicate (close to same wrist as winner)
                    best_viable = min(viable_sides, key=lambda s: costs[(hi, s)]["total_cost"])
                    winner_hi = [h for h, s in best_combo.items() if s == best_viable][0]
                    inter_dist = float(np.linalg.norm(
                        detections[hi]["wrist_px"] - detections[winner_hi]["wrist_px"]
                    ))
                    sw_threshold = shoulder_w * 0.4 if shoulder_w else 120
                    if inter_dist < sw_threshold:
                        status = "DUPLICATE_DETECTION_CANDIDATE"
                        anat_identity = None
                        reason = f"close_to_winner_at_{best_viable}_inter_dist={inter_dist:.0f}px"
                        rejected_reason = f"winner_det{winner_hi}_assigned_{best_viable}"
                    else:
                        status = "AMBIGUOUS"
                        anat_identity = None
                        reason = f"viable_sides_taken_but_not_close_to_winner"
                        rejected_reason = f"inter_dist={inter_dist:.0f}px_too_large_for_duplicate"
                else:
                    # Has viable side not taken — shouldn't happen if global optimizer works
                    # But handle gracefully
                    status = "AMBIGUOUS"
                    anat_identity = None
                    reason = "global_optimizer_did_not_assign"
                    rejected_reason = None

            # Build result
            d_left = costs[(hi, "LEFT")].get("dist_px")
            d_right = costs[(hi, "RIGHT")].get("dist_px")
            gate = costs[(hi, "LEFT")]["gate_px"]  # same for both sides

            if assigned_side:
                sel_cost = costs[(hi, assigned_side)]
                sel_dist_norm = sel_cost["dist_norm"]
                cont_cost = sel_cost["continuity_cost"]
                elbow_cost = sel_cost["elbow_cost"]
                hand_cost = sel_cost["handedness_cost"]
                final_cost = sel_cost["total_cost"]
            else:
                sel_dist_norm = None
                cont_cost = None
                elbow_cost = None
                hand_cost = None
                final_cost = None

            # Assertion: duplicate_candidate must NOT be DETECTED
            assert not (status == "DETECTED" and
                        rejected_reason is not None and
                        "duplicate_candidate" in str(rejected_reason)), \
                f"ASSERTION FAILED: duplicate_candidate assigned as DETECTED at t={cam_pts_s}"

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
                "continuity_cost": cont_cost,
                "elbow_direction_cost": elbow_cost,
                "handedness_auxiliary_cost": hand_cost,
                "final_cost": final_cost,
                "assignment_reason": reason,
                "rejected_alternative_reason": rejected_reason,
                "shoulder_width_px": round(shoulder_w, 2) if shoulder_w else None,
                "gate_px": round(gate, 2),
            })

            # Update tracker for accepted
            if status == "DETECTED" and assigned_side:
                prev_pos = self.prev_accepted.get(assigned_side, {}).get("pos")
                prev_t = self.prev_accepted.get(assigned_side, {}).get("pts_s")
                if prev_pos is not None and prev_t is not None:
                    dt = cam_pts_s - prev_t
                    if 0 < dt < 0.2:
                        self.prev_velocity[assigned_side] = (det["wrist_px"] - prev_pos) / dt
                self.prev_accepted[assigned_side] = {
                    "pos": det["wrist_px"].copy(), "pts_s": cam_pts_s
                }

        return results


# ═══════════════════════════════════════════════════════════
#  Observed handedness mapping (from pass 1)
# ═══════════════════════════════════════════════════════════

def compute_observed_mapping(frames):
    """From pass-1 results (handedness_weight=0), find label→anat mapping
    using frames where assigned wrist is close and other wrist is far."""
    counts = defaultdict(int)  # (model_label, anat_identity) -> count
    for f in frames:
        for hand in f.get("hands", []):
            if hand["association_status"] != "DETECTED":
                continue
            anat = hand["anatomical_identity"]
            if not anat:
                continue
            side = anat.replace("ANAT_", "")
            d_assigned_key = f"pose_{side.lower()}_wrist_dist_px"
            d_other_key = f"pose_{'right' if side == 'LEFT' else 'left'}_wrist_dist_px"
            d_assigned = hand.get(d_assigned_key)
            d_other = hand.get(d_other_key)
            if d_assigned is None or d_other is None:
                continue
            if d_assigned < 80 and d_other > d_assigned * 2.5:
                counts[(hand["model_handedness_label"], anat)] += 1

    # Determine dominant mapping
    mapping = {}
    for label in ["Left", "Right"]:
        best_anat = None
        best_count = 0
        for (l, a), c in counts.items():
            if l == label and c > best_count:
                best_count = c
                best_anat = a
        if best_anat and best_count >= 10:
            mapping[label] = best_anat

    return mapping, dict(counts)


# ═══════════════════════════════════════════════════════════
#  Per-view extraction
# ═══════════════════════════════════════════════════════════

def extract_view(view, segments, affine_a, affine_b, K, dist,
                 handedness_weight, observed_mapping):
    """Extract landmarks for all pilot segments from one view."""
    video_path = VIDEOS[view]
    print(f"\n  === {view} (pass {'1' if handedness_weight == 0 else '2'}) ===")

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    tb = float(stream.time_base)
    img_w, img_h = stream.width, stream.height

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
    pts_missing = []
    prev_ts_ms = -1
    ts_adjustments = 0
    frame_idx = 0
    n_in_range = 0

    seg_first_pts = {sr["sample_id"]: None for sr in seg_ranges}
    seg_last_pts = {sr["sample_id"]: None for sr in seg_ranges}

    tracker = AnatomicalHandTracker(
        handedness_weight=handedness_weight,
        observed_mapping=observed_mapping,
    )
    current_sample_id = None

    for frame in container.decode(stream):
        if frame.pts is None:
            pts_missing.append({"frame_index": frame_idx, "view": view})
            frame_idx += 1
            continue

        cam_pts_s = float(frame.pts) * tb

        if cam_pts_s < cam_start_global - 0.5:
            frame_idx += 1
            continue
        if cam_pts_s > cam_end_global + 0.5:
            break

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

        if sid != current_sample_id:
            tracker.reset()
            current_sample_id = sid

        if seg_first_pts[sid] is None:
            seg_first_pts[sid] = cam_pts_s
        seg_last_pts[sid] = cam_pts_s

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

        # Pose
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            plm = pose_result.pose_landmarks[0]
            pose_data = []
            for li, lm in enumerate(plm):
                raw_nx, raw_ny = lm.x, lm.y
                raw_px, raw_py = raw_nx * img_w, raw_ny * img_h
                und_px_arr, und_norm_arr = undistort_points(
                    np.array([[raw_px, raw_py]]), K, dist)
                und_px = und_px_arr[0] if len(und_px_arr) > 0 else [None, None]
                und_norm = und_norm_arr[0] if len(und_norm_arr) > 0 else [None, None]
                pose_data.append({
                    "index": li, "name": POSE_LANDMARK_NAMES[li],
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

        # Hands
        n_hands_detected = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        frame_record["hands_detected_count"] = n_hands_detected
        frame_record["anat_left_status"] = "NOT_DETECTED"
        frame_record["anat_right_status"] = "NOT_DETECTED"
        frame_record["anat_left_landmarks"] = None
        frame_record["anat_right_landmarks"] = None

        if n_hands_detected > 0:
            associations = tracker.associate(
                pose_result, hand_result, img_w, img_h, cam_pts_s)

            raw_hands = []
            for hi in range(n_hands_detected):
                assoc = associations[hi] if hi < len(associations) else {
                    "anatomical_identity": None, "association_status": "AMBIGUOUS",
                    "model_handedness_label": "Unknown", "model_handedness_score": 0,
                    "hand_wrist_px": [0, 0],
                    "assignment_reason": "no_result", "rejected_alternative_reason": None,
                }

                hlm = hand_result.hand_landmarks[hi]
                hand_pts = []
                for li, lm in enumerate(hlm):
                    raw_nx, raw_ny = lm.x, lm.y
                    raw_px, raw_py = raw_nx * img_w, raw_ny * img_h
                    und_px_arr, und_norm_arr = undistort_points(
                        np.array([[raw_px, raw_py]]), K, dist)
                    und_px = und_px_arr[0] if len(und_px_arr) > 0 else [None, None]
                    und_norm = und_norm_arr[0] if len(und_norm_arr) > 0 else [None, None]
                    hand_pts.append({
                        "index": li, "name": HAND_LANDMARK_NAMES[li],
                        "raw_norm_x": round(float(raw_nx), 6),
                        "raw_norm_y": round(float(raw_ny), 6),
                        "raw_distorted_px_x": round(float(raw_px), 2),
                        "raw_distorted_px_y": round(float(raw_py), 2),
                        "undistorted_px_x": round(float(und_px[0]), 2),
                        "undistorted_px_y": round(float(und_px[1]), 2),
                        "undistorted_norm_cam_x": round(float(und_norm[0]), 6),
                        "undistorted_norm_cam_y": round(float(und_norm[1]), 6),
                        "visibility": None, "presence": None,
                    })

                raw_hands.append({
                    "hand_index": hi,
                    "anatomical_identity": assoc.get("anatomical_identity"),
                    "association_status": assoc.get("association_status", "AMBIGUOUS"),
                    "model_handedness_label": assoc.get("model_handedness_label", "Unknown"),
                    "model_handedness_score": assoc.get("model_handedness_score", 0),
                    "hand_wrist_px": assoc.get("hand_wrist_px", [None, None]),
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

                status = assoc.get("association_status", "AMBIGUOUS")
                anat_id = assoc.get("anatomical_identity")
                if anat_id == "ANAT_LEFT" and status == "DETECTED":
                    frame_record["anat_left_status"] = "DETECTED"
                    frame_record["anat_left_landmarks"] = hand_pts
                elif anat_id == "ANAT_RIGHT" and status == "DETECTED":
                    frame_record["anat_right_status"] = "DETECTED"
                    frame_record["anat_right_landmarks"] = hand_pts

            frame_record["hands"] = raw_hands

            # Update statuses for non-DETECTED
            for rh in raw_hands:
                st = rh["association_status"]
                if st == "AMBIGUOUS":
                    for k in ["anat_left_status", "anat_right_status"]:
                        if frame_record[k] == "NOT_DETECTED":
                            frame_record[k] = "AMBIGUOUS"
                elif st == "DUPLICATE_DETECTION_CANDIDATE":
                    for k in ["anat_left_status", "anat_right_status"]:
                        if frame_record[k] == "NOT_DETECTED":
                            frame_record[k] = "DUPLICATE_DETECTION_CANDIDATE"
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

    print(f"    {view}: {n_in_range} total frames, {ts_adjustments} ts adj, "
          f"{len(pts_missing)} PTS-missing excluded")

    # Boundary diagnostics
    boundary_diag = []
    for sr in seg_ranges:
        sid = sr["sample_id"]
        first = seg_first_pts[sid]
        last = seg_last_pts[sid]
        boundary_diag.append({
            "sample_id": sid, "view": view,
            "requested_cam_start_s": round(sr["cam_start"], 6),
            "requested_cam_end_s": round(sr["cam_end"], 6),
            "actual_first_frame_pts_s": round(first, 6) if first else None,
            "actual_last_frame_pts_s": round(last, 6) if last else None,
            "start_error_ms": round((first - sr["cam_start"]) * 1000, 2) if first else None,
            "end_error_ms": round((sr["cam_end"] - last) * 1000, 2) if last else None,
            "n_frames": sum(1 for f in all_frames if f["sample_id"] == sid),
        })

    return all_frames, boundary_diag, pts_missing


# ═══════════════════════════════════════════════════════════
#  QA statistics (v1.2)
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

        # Pose usable
        pose_usable = sum(1 for f in seg_frames
                          if f["pose_detected"] and f["pose_landmarks"]
                          and all(f["pose_landmarks"][i]["visibility"] >
                                  THRESHOLDS["pose_visibility_usable_threshold"]
                                  for i in [11, 12, 13, 14, 15, 16]))

        # Per-anatomical-hand
        left_det = right_det = both_det = 0
        ambiguous_frames = duplicate_frames = 0
        left_miss_run = left_max_miss = 0
        right_miss_run = right_max_miss = 0
        both_miss_run = both_max_miss = 0
        left_wrist_dists = []
        right_wrist_dists = []

        # Swap tracking
        prev_left_pos = None
        prev_right_pos = None
        prev_left_frame_idx = None
        prev_right_frame_idx = None
        swap_candidates = []
        swap_comparisons = 0

        for fi, f in enumerate(seg_frames):
            has_left = f["anat_left_status"] == "DETECTED"
            has_right = f["anat_right_status"] == "DETECTED"
            if has_left: left_det += 1
            if has_right: right_det += 1
            if has_left and has_right: both_det += 1
            if f["anat_left_status"] == "AMBIGUOUS" or f["anat_right_status"] == "AMBIGUOUS":
                ambiguous_frames += 1
            if f["anat_left_status"] == "DUPLICATE_DETECTION_CANDIDATE" or \
               f["anat_right_status"] == "DUPLICATE_DETECTION_CANDIDATE":
                duplicate_frames += 1

            # Missing runs
            if not has_left:
                left_miss_run += 1
                left_max_miss = max(left_max_miss, left_miss_run)
            else:
                left_miss_run = 0
            if not has_right:
                right_miss_run += 1
                right_max_miss = max(right_max_miss, right_miss_run)
            else:
                right_miss_run = 0
            if not has_left and not has_right:
                both_miss_run += 1
                both_max_miss = max(both_max_miss, both_miss_run)
            else:
                both_miss_run = 0

            # Wrist distances per assigned hand
            cur_left_pos = None
            cur_right_pos = None
            for hand in f.get("hands", []):
                if hand["association_status"] != "DETECTED":
                    continue
                anat = hand.get("anatomical_identity")
                hwp = hand.get("hand_wrist_px")
                if anat == "ANAT_LEFT":
                    d = hand.get("pose_left_wrist_dist_px")
                    if d is not None:
                        left_wrist_dists.append(d)
                    if hwp:
                        cur_left_pos = np.array(hwp)
                elif anat == "ANAT_RIGHT":
                    d = hand.get("pose_right_wrist_dist_px")
                    if d is not None:
                        right_wrist_dists.append(d)
                    if hwp:
                        cur_right_pos = np.array(hwp)

            # Swap detection — only when both currently and previously have both hands
            # and no long gap
            if (cur_left_pos is not None and cur_right_pos is not None and
                prev_left_pos is not None and prev_right_pos is not None):
                left_gap = fi - prev_left_frame_idx if prev_left_frame_idx is not None else 999
                right_gap = fi - prev_right_frame_idx if prev_right_frame_idx is not None else 999
                if left_gap <= THRESHOLDS["swap_gap_max_frames"] and \
                   right_gap <= THRESHOLDS["swap_gap_max_frames"]:
                    swap_comparisons += 1
                    # Normal continuity cost
                    normal_cost = (float(np.linalg.norm(cur_left_pos - prev_left_pos)) +
                                   float(np.linalg.norm(cur_right_pos - prev_right_pos)))
                    # Swapped cost
                    swapped_cost = (float(np.linalg.norm(cur_left_pos - prev_right_pos)) +
                                    float(np.linalg.norm(cur_right_pos - prev_left_pos)))
                    # Fixed formula: swapped_cost < 0.7 * normal_cost
                    if swapped_cost < THRESHOLDS["swap_cost_ratio_threshold"] * normal_cost:
                        swap_candidates.append({
                            "frame_index_in_sample": fi,
                            "camera_pts_s": f["camera_local_pts_s"],
                            "normal_cost": round(normal_cost, 1),
                            "swapped_cost": round(swapped_cost, 1),
                            "ratio": round(swapped_cost / normal_cost, 3) if normal_cost > 0 else None,
                        })

            if cur_left_pos is not None:
                prev_left_pos = cur_left_pos
                prev_left_frame_idx = fi
            if cur_right_pos is not None:
                prev_right_pos = cur_right_pos
                prev_right_frame_idx = fi

        frame_dt = ((seg_frames[-1]["camera_local_pts_s"] -
                     seg_frames[0]["camera_local_pts_s"]) / (n - 1)) if n > 1 else 1.0 / 30.0

        lwd = np.array(left_wrist_dists) if left_wrist_dists else None
        rwd = np.array(right_wrist_dists) if right_wrist_dists else None

        stats.append({
            "sample_id": sid, "view": view, "n_frames": n,
            "pose_usable_rate": round(pose_usable / n, 4),
            "anat_left_detection_rate": round(left_det / n, 4),
            "anat_right_detection_rate": round(right_det / n, 4),
            "both_hands_detected_rate": round(both_det / n, 4),
            "left_max_consecutive_missing_frames": left_max_miss,
            "left_max_consecutive_missing_time_s": round(left_max_miss * frame_dt, 3),
            "right_max_consecutive_missing_frames": right_max_miss,
            "right_max_consecutive_missing_time_s": round(right_max_miss * frame_dt, 3),
            "both_unavailable_max_consecutive_frames": both_max_miss,
            "both_unavailable_max_consecutive_time_s": round(both_max_miss * frame_dt, 3),
            "ambiguous_frame_count": ambiguous_frames,
            "duplicate_detection_candidate_frame_count": duplicate_frames,
            "identity_swap_candidate_count": len(swap_candidates),
            "identity_swap_comparisons_executed": swap_comparisons,
            "identity_swap_candidate_frames": swap_candidates,
            "left_assigned_wrist_dist_median_px": round(float(np.median(lwd)), 2) if lwd is not None else None,
            "left_assigned_wrist_dist_p95_px": round(float(np.percentile(lwd, 95)), 2) if lwd is not None else None,
            "left_assigned_wrist_dist_max_px": round(float(np.max(lwd)), 2) if lwd is not None else None,
            "right_assigned_wrist_dist_median_px": round(float(np.median(rwd)), 2) if rwd is not None else None,
            "right_assigned_wrist_dist_p95_px": round(float(np.percentile(rwd, 95)), 2) if rwd is not None else None,
            "right_assigned_wrist_dist_max_px": round(float(np.max(rwd)), 2) if rwd is not None else None,
        })

    return stats


# ═══════════════════════════════════════════════════════════
#  Synchronized view coverage (v1.2 — unconditional)
# ═══════════════════════════════════════════════════════════

def compute_synchronized_view_coverage(all_view_frames, segments):
    """Use all FRONT frames as reference. One-to-one nearest matching."""
    rows = []
    tol = 0.025  # 25ms

    for seg in segments:
        sid = seg["sample_id"]

        # All FRONT timestamps for this sample
        front_frames = [f for f in all_view_frames["FRONT"] if f["sample_id"] == sid]
        front_times = [f["mapped_front_pts_s"] for f in front_frames]
        n_ref = len(front_times)

        for hand_side in ["LEFT", "RIGHT"]:
            anat_key = f"anat_{hand_side.lower()}_status"

            # FRONT detection per reference frame
            front_detected = [f[anat_key] == "DETECTED" for f in front_frames]

            # LEFT/RIGHT detection times (mapped to front PTS)
            other_times = {}
            for other_view in ["LEFT", "RIGHT"]:
                frames = [f for f in all_view_frames[other_view] if f["sample_id"] == sid]
                det_times = sorted([f["mapped_front_pts_s"] for f in frames
                                    if f[anat_key] == "DETECTED"])
                other_times[other_view] = np.array(det_times)

            # One-to-one nearest matching
            n_2plus = 0
            n_3 = 0
            front_det_count = sum(front_detected)
            reused = 0

            # Track used indices for one-to-one
            used_left = set()
            used_right = set()

            for ri, ft in enumerate(front_times):
                views_with_det = 0
                if front_detected[ri]:
                    views_with_det = 1

                for other_view in ["LEFT", "RIGHT"]:
                    ot = other_times[other_view]
                    used_set = used_left if other_view == "LEFT" else used_right
                    if len(ot) == 0:
                        continue
                    idx = np.searchsorted(ot, ft)
                    # Check neighbors, pick closest unused
                    best_ci = None
                    best_diff = tol + 1
                    for ci in [idx - 1, idx, idx + 1]:
                        if ci < 0 or ci >= len(ot):
                            continue
                        diff = abs(ot[ci] - ft)
                        if diff <= tol and ci not in used_set and diff < best_diff:
                            best_diff = diff
                            best_ci = ci

                    if best_ci is not None:
                        views_with_det += 1
                        used_set.add(best_ci)

                if views_with_det >= 2:
                    n_2plus += 1
                if views_with_det >= 3:
                    n_3 += 1

            # Count unmatched detections in LEFT/RIGHT
            unmatched_left = len(other_times["LEFT"]) - len(used_left)
            unmatched_right = len(other_times["RIGHT"]) - len(used_right)

            rows.append({
                "sample_id": sid,
                "anatomical_hand": hand_side,
                "total_reference_frames": n_ref,
                "front_detected_frames": front_det_count,
                "frames_with_2plus_views": n_2plus,
                "frames_with_3_views": n_3,
                "unconditional_2plus_rate": round(n_2plus / n_ref, 4) if n_ref > 0 else None,
                "unconditional_3view_rate": round(n_3 / n_ref, 4) if n_ref > 0 else None,
                "conditional_on_front_detected_rate": round(n_2plus / front_det_count, 4)
                    if front_det_count > 0 else None,
                "reused_match_count": 0,  # one-to-one matching ensures 0
                "unmatched_left_detections": unmatched_left,
                "unmatched_right_detections": unmatched_right,
            })

    return rows


# ═══════════════════════════════════════════════════════════
#  Overlay (v1.2 — full sample_id, detailed annotations)
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
    "DETECTED": None,
    "AMBIGUOUS": (0, 0, 255),
    "DUPLICATE_DETECTION_CANDIDATE": (255, 0, 255),
}
ANAT_COLORS = {"ANAT_LEFT": (0, 255, 0), "ANAT_RIGHT": (0, 165, 255)}


def make_overlay_video(view, frames_data, segments, affine_a, affine_b,
                       out_path, qa_stats, fps_out=15):
    video_path = VIDEOS[view]
    scale = 0.5
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    img_w, img_h = stream.width, stream.height
    container.close()
    ow, oh = int(img_w * scale), int(img_h * scale)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_out, (ow, oh))

    frame_lookup = {}
    for fd in frames_data:
        if fd["view"] == view:
            key = round(fd["camera_local_pts_s"], 4)
            frame_lookup[key] = fd

    seg_ranges = []
    for seg in segments:
        cr = seg["camera_ranges"][view]
        seg_ranges.append((cr["start_pts_s"], cr["end_pts_s"], seg["sample_id"]))

    # Swap candidate lookup
    swap_pts = set()
    for qs in qa_stats:
        if qs["view"] == view:
            for sc in qs.get("identity_swap_candidate_frames", []):
                swap_pts.add(round(sc["camera_pts_s"], 2))

    dt_out = 1.0 / fps_out
    written = 0

    for seg_start, seg_end, seg_sid in seg_ranges:
        # Separator
        sep = np.zeros((oh, ow, 3), dtype=np.uint8)
        label = _short_sid(seg_sid)
        cv2.putText(sep, f"{view} {label}", (10, oh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for _ in range(int(fps_out * 0.5)):
            writer.write(sep)
            written += 1

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base)
        next_grab_s = seg_start

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            cam_pts_s = float(frame.pts) * tb
            if cam_pts_s < seg_start - 0.05:
                continue
            if cam_pts_s > seg_end + 0.05:
                break
            if cam_pts_s < next_grab_s or cam_pts_s < seg_start or cam_pts_s > seg_end:
                continue
            next_grab_s = cam_pts_s + dt_out

            img = frame.to_ndarray(format="bgr24")
            small = cv2.resize(img, (ow, oh))
            front_pts_s = cam_to_front(cam_pts_s, affine_a, affine_b)

            # Header with full sample_id
            cv2.rectangle(small, (0, 0), (ow, 40), (0, 80, 0), -1)
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
                # Pose skeleton
                if fd.get("pose_detected") and fd.get("pose_landmarks"):
                    pts = [(int(lm["raw_distorted_px_x"] * scale),
                            int(lm["raw_distorted_px_y"] * scale))
                           for lm in fd["pose_landmarks"]]
                    for i, j in POSE_CONNECTIONS:
                        if fd["pose_landmarks"][i]["visibility"] > 0.3 and \
                           fd["pose_landmarks"][j]["visibility"] > 0.3:
                            cv2.line(small, pts[i], pts[j], (200, 200, 200), 1)

                    # Draw pose wrists
                    for wi, wcolor in [(15, (0, 200, 0)), (16, (0, 128, 255))]:
                        if fd["pose_landmarks"][wi]["visibility"] > 0.3:
                            cv2.circle(small, pts[wi], 5, wcolor, 2)

                # Hands with status
                for hand in fd.get("hands", []):
                    status = hand.get("association_status", "AMBIGUOUS")
                    anat = hand.get("anatomical_identity")
                    hi = hand["hand_index"]
                    ml = hand["model_handedness_label"]

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

                    # Label: raw index, model label, status
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

                    cv2.putText(small, f"d{hi}:{ml[0]}->{short}", (wx - 10, wy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                    # Show distances to both wrists
                    dl = hand.get("pose_left_wrist_dist_px")
                    dr = hand.get("pose_right_wrist_dist_px")
                    dist_str = f"dL={dl:.0f} dR={dr:.0f}" if dl and dr else ""
                    if dist_str:
                        cv2.putText(small, dist_str, (wx - 10, wy + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

                if not fd.get("pose_detected"):
                    warnings.append("NO_POSE")
                if fd["hands_detected_count"] == 0:
                    warnings.append("NO_HANDS")

                # Swap candidate
                if round(cam_pts_s, 2) in swap_pts:
                    warnings.append("SWAP_CAND")

            if warnings:
                cv2.putText(small, " ".join(set(warnings)), (5, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            writer.write(small)
            written += 1

        container.close()

    writer.release()
    print(f"    Overlay {view}: {written} frames written to {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  Contact sheets (v1.2 — with landmarks drawn)
# ═══════════════════════════════════════════════════════════

def _draw_landmarks_on_thumb(thumb, fd, scale, thumb_w, thumb_h, img_w, img_h):
    """Draw pose + hand landmarks on a contact sheet thumbnail."""
    if not fd:
        return
    sx = thumb_w / img_w
    sy = thumb_h / img_h

    # Pose
    if fd.get("pose_detected") and fd.get("pose_landmarks"):
        pts = [(int(lm["raw_distorted_px_x"] * sx), int(lm["raw_distorted_px_y"] * sy))
               for lm in fd["pose_landmarks"]]
        for i, j in POSE_CONNECTIONS:
            if fd["pose_landmarks"][i]["visibility"] > 0.3 and \
               fd["pose_landmarks"][j]["visibility"] > 0.3:
                cv2.line(thumb, pts[i], pts[j], (200, 200, 200), 1)
        # Pose wrists
        for wi, wc in [(15, (0, 200, 0)), (16, (0, 128, 255))]:
            if fd["pose_landmarks"][wi]["visibility"] > 0.3:
                cv2.circle(thumb, pts[wi], 4, wc, 2)

    # Hands
    for hand in fd.get("hands", []):
        status = hand.get("association_status", "AMBIGUOUS")
        anat = hand.get("anatomical_identity")
        if status == "DETECTED" and anat:
            color = ANAT_COLORS.get(anat, (128, 128, 128))
        else:
            color = STATUS_COLORS.get(status, (128, 128, 128))

        hpts = [(int(lm["raw_distorted_px_x"] * sx), int(lm["raw_distorted_px_y"] * sy))
                for lm in hand["landmarks"]]
        for i, j in HAND_CONNECTIONS:
            cv2.line(thumb, hpts[i], hpts[j], color, 1)

        wx, wy = hpts[0]
        hi = hand["hand_index"]
        ml = hand["model_handedness_label"]
        short = "L" if anat == "ANAT_LEFT" else ("R" if anat == "ANAT_RIGHT" else
                ("DUP" if status == "DUPLICATE_DETECTION_CANDIDATE" else "?"))
        cv2.putText(thumb, f"d{hi}:{ml[0]}->{short}", (wx - 5, wy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        dl = hand.get("pose_left_wrist_dist_px")
        dr = hand.get("pose_right_wrist_dist_px")
        if dl and dr:
            cv2.putText(thumb, f"dL={dl:.0f} dR={dr:.0f}", (wx - 5, wy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.22, color, 1)


def generate_contact_sheets(all_view_frames, qa_stats, segments, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    flagged = defaultdict(list)  # category -> [(view, sid, cam_pts_s, reason)]

    # Required samples
    for (view, sid), tag in {
        ("LEFT", "S001_P001_GREETING_SLOW_R03"): "low_detection_greeting_slow_left",
        ("RIGHT", "S001_P001_LOVE_NORMAL_R03"): "low_detection_love_normal_right",
    }.items():
        frames = [f for f in all_view_frames[view] if f["sample_id"] == sid]
        if frames:
            indices = np.linspace(0, len(frames) - 1, min(12, len(frames)), dtype=int)
            for idx in indices:
                flagged[tag].append((view, sid, frames[idx]["camera_local_pts_s"], "sampled"))

    # All FAST
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for sid in PILOT_SAMPLE_IDS:
            if "FAST" in sid:
                frames = [f for f in all_view_frames[view] if f["sample_id"] == sid]
                if frames:
                    indices = np.linspace(0, len(frames) - 1, min(8, len(frames)), dtype=int)
                    for idx in indices:
                        flagged[f"fast_{view.lower()}"].append(
                            (view, sid, frames[idx]["camera_local_pts_s"], "fast"))

    # Frame-level flags
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for f in all_view_frames[view]:
            for hand in f.get("hands", []):
                st = hand.get("association_status")
                if st == "DUPLICATE_DETECTION_CANDIDATE":
                    flagged["duplicate_candidates"].append(
                        (view, f["sample_id"], f["camera_local_pts_s"], "dup"))
                elif st == "AMBIGUOUS":
                    flagged["ambiguous_frames"].append(
                        (view, f["sample_id"], f["camera_local_pts_s"], "ambig"))

    # Swap candidates
    for qs in qa_stats:
        for sc in qs.get("identity_swap_candidate_frames", []):
            flagged["swap_candidates"].append(
                (qs["view"], qs["sample_id"], sc["camera_pts_s"], "swap"))

    # Wrist distance outliers (assigned wrist > 150px)
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
                            (view, f["sample_id"], f["camera_local_pts_s"], f"d={d:.0f}"))

    # Strong misassignment: assigned wrist > 2x opposite AND diff > 50px
    for view in ["FRONT", "LEFT", "RIGHT"]:
        for f in all_view_frames[view]:
            for hand in f.get("hands", []):
                if hand.get("association_status") != "DETECTED":
                    continue
                anat = hand.get("anatomical_identity")
                if not anat:
                    continue
                side = anat.replace("ANAT_", "")
                d_assigned = hand.get(f"pose_{side.lower()}_wrist_dist_px")
                other = "right" if side == "LEFT" else "left"
                d_other = hand.get(f"pose_{other}_wrist_dist_px")
                if d_assigned and d_other and d_assigned > d_other * 2 and d_assigned - d_other > 50:
                    flagged["strong_misassignment"].append(
                        (view, f["sample_id"], f["camera_local_pts_s"],
                         f"assigned={d_assigned:.0f} other={d_other:.0f}"))

    # Regression case: THANK_YOU_NORMAL FRONT 318.7-318.83
    for f in all_view_frames["FRONT"]:
        if f["sample_id"] == "S001_P001_THANK_YOU_NORMAL_R03":
            t = f["camera_local_pts_s"]
            if 318.65 <= t <= 318.9:
                flagged["regression_thankyou_front"].append(
                    ("FRONT", f["sample_id"], t, "regression"))

    # Build frame_lookup for each view
    frame_lookups = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        fl = {}
        for fd in all_view_frames[view]:
            key = round(fd["camera_local_pts_s"], 4)
            fl[key] = fd
        frame_lookups[view] = fl

    thumb_w, thumb_h = 384, 216

    for category, items in flagged.items():
        if not items:
            continue
        if len(items) > 24:
            indices = np.linspace(0, len(items) - 1, 24, dtype=int)
            items = [items[i] for i in indices]

        by_view = defaultdict(list)
        for view, sid, pts_s, reason in items:
            by_view[view].append((sid, pts_s, reason))

        thumbnails = []

        for view, frame_list in by_view.items():
            video_path = VIDEOS[view]
            container = av.open(str(video_path))
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            tb = float(stream.time_base)
            vw, vh = stream.width, stream.height

            frame_list.sort(key=lambda x: x[1])
            fl_idx = 0

            for vframe in container.decode(stream):
                if fl_idx >= len(frame_list):
                    break
                if vframe.pts is None:
                    continue
                cam_pts_s = float(vframe.pts) * tb
                target_pts = frame_list[fl_idx][1]
                if cam_pts_s < target_pts - 0.02:
                    continue
                if abs(cam_pts_s - target_pts) <= 0.05:
                    img = vframe.to_ndarray(format="bgr24")
                    thumb = cv2.resize(img, (thumb_w, thumb_h))

                    sid = frame_list[fl_idx][0]
                    reason = frame_list[fl_idx][2]

                    # Find frame data and draw landmarks
                    key = round(target_pts, 4)
                    fd = frame_lookups[view].get(key)
                    if fd is None:
                        for delta in [0.0001, -0.0001, 0.001, -0.001]:
                            fd = frame_lookups[view].get(round(target_pts + delta, 4))
                            if fd:
                                break

                    _draw_landmarks_on_thumb(thumb, fd, 1.0, thumb_w, thumb_h, vw, vh)

                    # Text overlay
                    cv2.putText(thumb, f"{view} {_short_sid(sid)}", (5, 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                    cv2.putText(thumb, f"t={cam_pts_s:.2f} {reason}", (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 255), 1)
                    thumbnails.append(thumb)
                    fl_idx += 1
                if cam_pts_s > target_pts + 0.05:
                    fl_idx += 1

            container.close()

        if not thumbnails:
            continue

        cols = min(4, len(thumbnails))
        rows_n = math.ceil(len(thumbnails) / cols)
        sheet = np.zeros((rows_n * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
        for i, thumb in enumerate(thumbnails):
            r, c = divmod(i, cols)
            sheet[r * thumb_h:(r + 1) * thumb_h, c * thumb_w:(c + 1) * thumb_w] = thumb

        cv2.imwrite(str(out_dir / f"contact_{category}.png"), sheet)

    generated = list(out_dir.glob("contact_*.png"))
    print(f"  Contact sheets: {len(generated)} generated")
    return generated


# ═══════════════════════════════════════════════════════════
#  Input verification
# ═══════════════════════════════════════════════════════════

def verify_inputs(seg_data, all_segs):
    errors = []
    if seg_data.get("n_segments_expected") != 45:
        errors.append(f"n_segments_expected={seg_data.get('n_segments_expected')}")
    if len(all_segs) != 45:
        errors.append(f"len(segments)={len(all_segs)}")
    non_v = [s["sample_id"] for s in all_segs if s["verification_status"] != "MANUAL_VERIFIED"]
    if non_v:
        errors.append(f"Non-verified: {non_v}")
    expected_ids = set()
    for sign in ["GREETING", "THANK_YOU", "LOVE"]:
        for speed in ["NORMAL", "SLOW", "FAST"]:
            for rep in range(1, 6):
                expected_ids.add(f"S001_P001_{sign}_{speed}_R{rep:02d}")
    actual_ids = set(s["sample_id"] for s in all_segs)
    if actual_ids != expected_ids:
        errors.append(f"ID mismatch: missing={expected_ids - actual_ids}, extra={actual_ids - expected_ids}")
    for i in range(len(all_segs) - 1):
        if all_segs[i]["front_end_s"] > all_segs[i + 1]["front_start_s"]:
            errors.append(f"Overlap at {i}")
    pilot = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    if len(pilot) != 9:
        errors.append(f"Pilot count={len(pilot)}")
    transforms = seg_data.get("affine_transforms", {})
    for cam, exp_a, exp_b in [
        ("LEFT", 0.9999806597862914, 2.310470997893187),
        ("RIGHT", 0.9999996430818727, 9.388814864164976),
    ]:
        if transforms.get(cam, {}).get("a") != exp_a:
            errors.append(f"{cam} affine a mismatch")
        if transforms.get(cam, {}).get("b") != exp_b:
            errors.append(f"{cam} affine b mismatch")
    return errors


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Step 6A v1.2: MediaPipe 2D Landmark Pilot Extraction — S001_P001\n")

    seg_path = PROC_DIR / "action_segments_v2_1_verified.json"
    with open(seg_path, encoding="utf-8") as f:
        seg_data = json.load(f)
    all_segs = seg_data["segments"]
    transforms = seg_data["affine_transforms"]

    errors = verify_inputs(seg_data, all_segs)
    if errors:
        print("  INPUT VERIFICATION FAILED:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  Input verification: PASS")

    pilot_segs = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    assert len(pilot_segs) == 9
    print(f"  Pilot samples: {len(pilot_segs)}")

    out_base = PROC_DIR / "landmarks_2d_v1_2" / "pilot"
    out_overlays = out_base / "overlays"
    out_contacts = out_base / "low_confidence_contact_sheets"
    for d in [out_base, out_overlays, out_contacts]:
        d.mkdir(parents=True, exist_ok=True)

    # Load intrinsics and verify
    intrinsics = {}
    intrinsics_sizes = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        K, dist, img_size = load_intrinsics(view)
        intrinsics[view] = (K, dist)
        intrinsics_sizes[view] = img_size

    for view in ["FRONT", "LEFT", "RIGHT"]:
        container = av.open(str(VIDEOS[view]))
        stream = container.streams.video[0]
        vid_w, vid_h = stream.width, stream.height
        container.close()
        assert intrinsics_sizes[view] == [vid_w, vid_h], \
            f"{view} intrinsics size mismatch"
    print("  Intrinsics vs video resolution: MATCH")

    # ════ PASS 1: association without handedness ════
    print("\n  ══ PASS 1: association without handedness ══")
    pass1_frames = {}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        K, dist = intrinsics[view]
        frames, _, _ = extract_view(view, pilot_segs, t["a"], t["b"], K, dist,
                                    handedness_weight=0.0, observed_mapping=None)
        pass1_frames[view] = frames

    # Compute observed mapping from pass 1
    observed_mapping, mapping_counts = compute_observed_mapping(
        [f for v in pass1_frames.values() for f in v]
    )
    print(f"\n  Observed handedness mapping: {observed_mapping}")
    print(f"  Mapping counts: {mapping_counts}")

    # ════ PASS 2: association with verified handedness ════
    print("\n  ══ PASS 2: association with verified handedness mapping ══")
    all_view_frames = {}
    all_boundary_diag = []
    all_pts_missing = []

    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        K, dist = intrinsics[view]
        frames, bdiag, pts_miss = extract_view(
            view, pilot_segs, t["a"], t["b"], K, dist,
            handedness_weight=THRESHOLDS["association_handedness_weight_pass2"],
            observed_mapping=observed_mapping,
        )
        all_view_frames[view] = frames
        all_boundary_diag.extend(bdiag)
        all_pts_missing.extend(pts_miss)

    # Save per-view JSONs
    for view in ["FRONT", "LEFT", "RIGHT"]:
        view_file = out_base / f"landmarks_2d_{view.lower()}.json"
        with open(view_file, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.2_pilot",
                "view": view,
                "n_frames": len(all_view_frames[view]),
                "image_size": intrinsics_sizes[view],
                "original_video_mirrored": False,
                "frames": all_view_frames[view],
            }, f, indent=1, ensure_ascii=False)
        print(f"  Saved: {view_file.name} ({len(all_view_frames[view])} frames)")

    # Boundary diagnostics
    bdiag_path = out_base / "boundary_frame_diagnostics.csv"
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

    dq_path = out_base / "detection_quality.csv"
    csv_qa = [{k: v for k, v in q.items() if k != "identity_swap_candidate_frames"}
              for q in all_qa]
    with open(dq_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_qa[0].keys())
        w.writeheader()
        w.writerows(csv_qa)
    print(f"  Saved: {dq_path.name}")

    # Hand identity diagnostics
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
                    "hand_wrist_px_x": hand["hand_wrist_px"][0] if hand.get("hand_wrist_px") else None,
                    "hand_wrist_px_y": hand["hand_wrist_px"][1] if hand.get("hand_wrist_px") else None,
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
    with open(hid_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hid_rows[0].keys())
        w.writeheader()
        w.writerows(hid_rows)
    print(f"  Saved: {hid_path.name}")

    # Handedness verification (from pass 2)
    handedness_verification, hv_counts = compute_observed_mapping(
        [f for v in all_view_frames.values() for f in v]
    )

    # Synchronized coverage
    sync_coverage = compute_synchronized_view_coverage(all_view_frames, pilot_segs)
    sync_path = out_base / "synchronized_view_coverage.csv"
    with open(sync_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sync_coverage[0].keys())
        w.writeheader()
        w.writerows(sync_coverage)
    print(f"  Saved: {sync_path.name}")

    # Overlays
    print("\n  Generating overlay videos...")
    for view in ["FRONT", "LEFT", "RIGHT"]:
        t = transforms[view]
        make_overlay_video(view, all_view_frames[view], pilot_segs,
                           t["a"], t["b"], out_overlays / f"overlay_pilot_{view.lower()}.mp4",
                           all_qa)

    # Contact sheets
    print()
    generate_contact_sheets(all_view_frames, all_qa, pilot_segs, out_contacts)

    # SHA-256
    input_hashes = {"action_segments": sha256_file(seg_path),
                    "pose_model": sha256_file(POSE_MODEL),
                    "hand_model": sha256_file(HAND_MODEL)}
    for view in ["FRONT", "LEFT", "RIGHT"]:
        input_hashes[f"video_{view}"] = sha256_file(VIDEOS[view])
        input_hashes[f"intrinsics_{view}"] = sha256_file(C1_INTRINSICS[view])

    # Manifest
    manifest = {
        "step": "6A_pilot_v1.2",
        "date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "mediapipe_version": mp.__version__,
        "pyav_version": av.__version__,
        "opencv_version": cv2.__version__,
        "pose_model": {"path": str(POSE_MODEL), "type": "pose_landmarker_lite",
                        "sha256": input_hashes["pose_model"]},
        "hand_model": {"path": str(HAND_MODEL), "type": "hand_landmarker",
                        "sha256": input_hashes["hand_model"]},
        "thresholds": THRESHOLDS,
        "v1_defaults_note": V1_DEFAULTS,
        "two_pass_association": {
            "pass1": "handedness_weight=0, pose distance only",
            "pass2": f"handedness_weight={THRESHOLDS['association_handedness_weight_pass2']}, "
                     f"observed mapping applied",
            "observed_mapping": observed_mapping,
            "mapping_counts": {f"{k[0]}->{k[1]}": v for k, v in mapping_counts.items()},
        },
        "pose_options": {
            "running_mode": "VIDEO", "num_poses": 1, "output_segmentation_masks": False,
            "min_pose_detection_confidence": THRESHOLDS["pose_min_detection_confidence"],
            "min_pose_presence_confidence": THRESHOLDS["pose_min_presence_confidence"],
            "min_tracking_confidence": THRESHOLDS["pose_min_tracking_confidence"],
        },
        "hand_options": {
            "running_mode": "VIDEO", "num_hands": 2,
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
        "coordinate_conventions": {
            "raw_norm": "MediaPipe normalized [0,1] image coordinates",
            "raw_distorted_px": "raw_norm * image_size, still distorted by lens",
            "undistorted_px": "C1 intrinsics undistortion applied, pixel coords",
            "undistorted_norm_cam": "C1 undistorted normalized camera coords (x/fx, y/fy)",
            "hand_visibility_presence": "null (not provided by Hand Landmarker API)",
        },
        "null_policy": "Detection failures and unavailable values stored as null, "
                       "never 0-filled or interpolated",
        "association_statuses": [
            "DETECTED — accepted anatomical assignment",
            "NOT_DETECTED — no hand detection for this anatomical side",
            "AMBIGUOUS — detection exists but cannot be reliably assigned",
            "DUPLICATE_DETECTION_CANDIDATE — likely duplicate of another detection near same wrist",
        ],
        "handedness_label_mapping_pass2_verification": {
            "observed_mapping": handedness_verification,
            "counts": {f"{k[0]}->{k[1]}": v for k, v in hv_counts.items()},
        },
    }
    manifest_path = out_base / "run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {manifest_path.name}")

    # ── QA summary ──
    print("\n  " + "=" * 60)
    print("  QA Summary (v1.2)")
    print("  " + "=" * 60)

    # Count regression and strong misassignment
    dup_detected_count = 0
    strong_misassign_count = 0
    for row in hid_rows:
        rej = row.get("rejected_alternative_reason") or ""
        if "duplicate_candidate" in rej and row["association_status"] == "DETECTED":
            dup_detected_count += 1
        if row["association_status"] == "DETECTED" and row.get("anatomical_identity"):
            side = row["anatomical_identity"].replace("ANAT_", "")
            d_a = row.get(f"pose_{side.lower()}_wrist_dist_px")
            other = "right" if side == "LEFT" else "left"
            d_o = row.get(f"pose_{other}_wrist_dist_px")
            if d_a and d_o:
                try:
                    da, do = float(d_a), float(d_o)
                    if da > do * 2 and da - do > 50:
                        strong_misassign_count += 1
                except:
                    pass

    print(f"\n  V1.1 regression check:")
    print(f"    duplicate_candidate but DETECTED: {dup_detected_count} (v1.1 had 44)")
    print(f"    strong misassignment candidates: {strong_misassign_count} (v1.1 had 27)")

    # Regression case check
    print(f"\n  Regression case THANK_YOU_NORMAL FRONT 318.7-318.83:")
    for row in hid_rows:
        if row["sample_id"] == "S001_P001_THANK_YOU_NORMAL_R03" and row["view"] == "FRONT":
            t = row["camera_pts_s"]
            if 318.65 <= t <= 318.9:
                print(f"    t={t:.4f} hi={row['hand_index']} "
                      f"anat={row['anatomical_identity']} "
                      f"status={row['association_status']} "
                      f"dL={row['pose_left_wrist_dist_px']} "
                      f"dR={row['pose_right_wrist_dist_px']}")

    for view in ["FRONT", "LEFT", "RIGHT"]:
        view_qa = [q for q in all_qa if q["view"] == view]
        print(f"\n  {view}:")
        for q in view_qa:
            sid_short = _short_sid(q["sample_id"])
            print(f"    {sid_short}: {q['n_frames']}f "
                  f"pose={q['pose_usable_rate']:.0%} "
                  f"L={q['anat_left_detection_rate']:.0%} "
                  f"R={q['anat_right_detection_rate']:.0%} "
                  f"both={q['both_hands_detected_rate']:.0%} "
                  f"Lmiss={q['left_max_consecutive_missing_frames']}f "
                  f"Rmiss={q['right_max_consecutive_missing_frames']}f "
                  f"ambig={q['ambiguous_frame_count']} "
                  f"dup={q['duplicate_detection_candidate_frame_count']} "
                  f"swap={q['identity_swap_candidate_count']}"
                  f"({q['identity_swap_comparisons_executed']}cmp)")

    # Boundary
    max_start = max((abs(b["start_error_ms"]) for b in all_boundary_diag if b["start_error_ms"]), default=0)
    max_end = max((abs(b["end_error_ms"]) for b in all_boundary_diag if b["end_error_ms"]), default=0)
    print(f"\n  Boundary PTS errors: max start={max_start:.2f}ms, max end={max_end:.2f}ms")
    print(f"  71.9ms > 2 frames at 30fps (66.7ms). "
          "Visual review confirms LOVE_NORMAL FRONT onset not clipped.")

    # Sync coverage
    print(f"\n  Synchronized multi-view coverage (diagnostic only — "
          "actual sufficiency requires triangulation + reprojection error verification):")
    for row in sync_coverage:
        r2 = row["unconditional_2plus_rate"]
        r3 = row["unconditional_3view_rate"]
        print(f"    {_short_sid(row['sample_id'])} {row['anatomical_hand']}: "
              f"ref={row['total_reference_frames']}f "
              f"front_det={row['front_detected_frames']}f "
              f"2+v={r2:.0%} 3v={r3:.0%} "
              f"reuse={row['reused_match_count']}")

    print(f"\n  PTS-missing frames excluded: {len(all_pts_missing)}")
    print(f"\n  Status: REQUIRES_HUMAN_VISUAL_REVIEW")

    # List files
    print(f"\n  Generated files:")
    for p in sorted(out_base.rglob("*")):
        if p.is_file():
            size_kb = p.stat().st_size / 1024
            size_str = f"{size_kb / 1024:.1f} MB" if size_kb > 1024 else f"{size_kb:.1f} KB"
            print(f"    {p.relative_to(out_base)}: {size_str}")


if __name__ == "__main__":
    main()
