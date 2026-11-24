"""
Step 6A v1.3: MediaPipe 2D Landmark Pilot Extraction — S001_P001

Final correction. Changes from v1.2:
- Pre-assignment duplicate check (before global enumeration)
- Bidirectional temporal association for single-detection frames
- Enhanced per-detection diagnostics
- Fixed conditional coverage (rate <= 1.0)
- Fixed _short_sid (THANK_YOU not truncated)
- Fixed null fallback (no [0,0])
- Comprehensive regression assertions
"""

import json, csv, hashlib, sys, datetime, math
from pathlib import Path
from collections import defaultdict

import av, cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker, PoseLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions, RunningMode,
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
    "S001_P001_GREETING_NORMAL_R03", "S001_P001_GREETING_SLOW_R03",
    "S001_P001_GREETING_FAST_R03", "S001_P001_THANK_YOU_NORMAL_R03",
    "S001_P001_THANK_YOU_SLOW_R03", "S001_P001_THANK_YOU_FAST_R03",
    "S001_P001_LOVE_NORMAL_R03", "S001_P001_LOVE_SLOW_R03",
    "S001_P001_LOVE_FAST_R03",
]
POSE_LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]
HAND_LANDMARK_NAMES = [
    "WRIST","THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP",
]

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
    "association_handedness_weight_pass1": 0.0,
    "association_handedness_weight_pass2": 0.1,
    "duplicate_inter_hand_sw_ratio": 0.4,
    "swap_cost_ratio_threshold": 0.7,
    "swap_gap_max_frames": 3,
    "bidirectional_anchor_max_gap_s": 0.5,
    "anchor_separation_sw_ratio": 0.6,
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
    if dist.ndim == 1: dist = dist.reshape(1, -1)
    return K, dist, d["image_size"]

def undistort_points(pts_px, K, dist):
    if len(pts_px) == 0: return np.array([]), np.array([])
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    und_norm = cv2.undistortPoints(pts, K, dist).reshape(-1, 2)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    und_px = np.column_stack([und_norm[:,0]*fx+cx, und_norm[:,1]*fy+cy])
    return und_px, und_norm

def front_to_cam(front_s, a, b): return a * front_s + b
def cam_to_front(cam_s, a, b): return (cam_s - b) / a

def _short_sid(sid):
    prefix = "S001_P001_"
    return sid[len(prefix):] if sid.startswith(prefix) else sid

POSE_CONNECTIONS = [
    (11,13),(13,15),(12,14),(14,16),(11,12),(11,23),(12,24),(23,24),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

# ═══════════════════════════════════════════════════════════
#  Association (v1.3)
# ═══════════════════════════════════════════════════════════

class AnatomicalHandTracker:
    def __init__(self, handedness_weight=0.0, observed_mapping=None):
        self.handedness_weight = handedness_weight
        self.observed_mapping = observed_mapping
        self.reset()

    def reset(self):
        self.prev_accepted = {}
        self.prev_velocity = {}

    def _shoulder_width(self, pose_result, img_w, img_h):
        if not pose_result.pose_landmarks: return None
        plm = pose_result.pose_landmarks[0]
        ls, rs = plm[11], plm[12]
        if ls.visibility < 0.3 or rs.visibility < 0.3: return None
        w = float(np.linalg.norm(
            np.array([ls.x*img_w, ls.y*img_h]) - np.array([rs.x*img_w, rs.y*img_h])))
        return w if w > 20 else None

    def _single_cost(self, det_wrist, side, pose_wrists, sw, cam_pts_s,
                     pose_result, img_w, img_h, model_label, model_score):
        gate = sw * THRESHOLDS["association_shoulder_width_gate_factor"] if sw else 300.0
        if side in pose_wrists:
            dist_px = float(np.linalg.norm(det_wrist - pose_wrists[side]))
            dist_norm = dist_px / sw if sw and sw > 0 else dist_px / 300.0
        else:
            dist_px = None
            dist_norm = 2.0
        distance_cost = dist_norm * THRESHOLDS["association_distance_weight"]

        continuity_cost = 0.0
        if side in self.prev_accepted:
            prev = self.prev_accepted[side]
            dt = cam_pts_s - prev["pts_s"]
            if 0 < dt < 0.2:
                expected = prev["pos"]
                if side in self.prev_velocity:
                    expected = prev["pos"] + self.prev_velocity[side] * dt
                c = float(np.linalg.norm(det_wrist - expected))
                continuity_cost = (c / sw if sw and sw > 0 else c / 300.0) * \
                                  THRESHOLDS["association_continuity_weight"]
            else:
                continuity_cost = 0.5 * THRESHOLDS["association_continuity_weight"]

        elbow_cost = 0.0
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            ei = 13 if side == "LEFT" else 14
            wi = 15 if side == "LEFT" else 16
            elb, wri = plm[ei], plm[wi]
            if elb.visibility > 0.3 and wri.visibility > 0.3:
                ep = np.array([elb.x*img_w, elb.y*img_h])
                wp = np.array([wri.x*img_w, wri.y*img_h])
                pd = wp - ep; n1 = np.linalg.norm(pd)
                dd = det_wrist - ep; n2 = np.linalg.norm(dd)
                if n1 > 5 and n2 > 5:
                    dot = float(np.dot(pd/n1, dd/n2))
                    elbow_cost = max(0.0, 1.0 - dot) * THRESHOLDS["association_elbow_direction_weight"]

        handedness_cost = 0.0
        if self.handedness_weight > 0 and self.observed_mapping:
            expected_label = None
            for lbl, anat in self.observed_mapping.items():
                if anat == f"ANAT_{side}":
                    expected_label = lbl; break
            if expected_label is not None:
                if model_label != expected_label:
                    handedness_cost = self.handedness_weight
                else:
                    handedness_cost = -self.handedness_weight * (model_score - 0.5)

        total = distance_cost + continuity_cost + elbow_cost + handedness_cost
        within_gate = dist_px is not None and dist_px < gate
        return {
            "dist_px": dist_px, "dist_norm": round(dist_norm, 4),
            "distance_cost": round(distance_cost, 4),
            "continuity_cost": round(continuity_cost, 4),
            "elbow_cost": round(elbow_cost, 4),
            "handedness_cost": round(handedness_cost, 4),
            "total_cost": round(total, 4),
            "within_gate": within_gate, "gate_px": round(gate, 2),
        }

    def associate(self, pose_result, hand_result, img_w, img_h, cam_pts_s):
        n_hands = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        if n_hands == 0: return []

        pose_wrists = {}
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            for side, idx in [("LEFT", 15), ("RIGHT", 16)]:
                lm = plm[idx]
                if lm.visibility > 0.3:
                    pose_wrists[side] = np.array([lm.x*img_w, lm.y*img_h])

        sw = self._shoulder_width(pose_result, img_w, img_h)

        detections = []
        for hi in range(n_hands):
            hw = hand_result.hand_landmarks[hi][0]
            wrist_px = np.array([hw.x*img_w, hw.y*img_h])
            h = hand_result.handedness[hi][0]
            detections.append({"hand_index": hi, "wrist_px": wrist_px,
                               "model_label": h.category_name, "model_score": h.score})

        costs = {}
        for det in detections:
            for side in ["LEFT", "RIGHT"]:
                costs[(det["hand_index"], side)] = self._single_cost(
                    det["wrist_px"], side, pose_wrists, sw, cam_pts_s,
                    pose_result, img_w, img_h, det["model_label"], det["model_score"])

        # ── Pre-assignment conflict detection ──
        close_same_wrist_conflict = False
        conflict_nearest = None
        inter_det_dist = None
        norm_inter_det = None

        if n_hands == 2:
            d0l = costs[(0, "LEFT")].get("dist_px") or 9999
            d0r = costs[(0, "RIGHT")].get("dist_px") or 9999
            d1l = costs[(1, "LEFT")].get("dist_px") or 9999
            d1r = costs[(1, "RIGHT")].get("dist_px") or 9999
            nearest_0 = "LEFT" if d0l < d0r else "RIGHT"
            nearest_1 = "LEFT" if d1l < d1r else "RIGHT"

            inter_det_dist = float(np.linalg.norm(
                detections[0]["wrist_px"] - detections[1]["wrist_px"]))
            norm_inter_det = inter_det_dist / sw if sw and sw > 0 else inter_det_dist / 300.0

            if nearest_0 == nearest_1:
                conflict_nearest = nearest_0
                if norm_inter_det < THRESHOLDS["duplicate_inter_hand_sw_ratio"]:
                    close_same_wrist_conflict = True

        # ── Enumerate valid candidates ──
        UNASSIGNED_PENALTY = 0.5

        if n_hands == 1:
            candidates = [{}, {0: "LEFT"}, {0: "RIGHT"}]
        elif n_hands == 2:
            if close_same_wrist_conflict:
                # Restrict: NO dual-hand assignment allowed
                candidates = [
                    {}, {0: "LEFT"}, {0: "RIGHT"}, {1: "LEFT"}, {1: "RIGHT"},
                ]
            else:
                candidates = [
                    {}, {0: "LEFT"}, {0: "RIGHT"}, {1: "LEFT"}, {1: "RIGHT"},
                    {0: "LEFT", 1: "RIGHT"}, {0: "RIGHT", 1: "LEFT"},
                ]
        else:
            candidates = [{}]
            for hi in range(n_hands):
                candidates.append({hi: "LEFT"})
                candidates.append({hi: "RIGHT"})
            if not close_same_wrist_conflict:
                for i in range(n_hands):
                    for j in range(n_hands):
                        if i != j:
                            candidates.append({i: "LEFT", j: "RIGHT"})

        # Score each candidate
        scored = []
        for combo in candidates:
            sides_used = list(combo.values())
            if len(sides_used) != len(set(sides_used)): continue
            total = 0.0; ok = True
            for hi, side in combo.items():
                c = costs[(hi, side)]
                if not c["within_gate"]: ok = False; break
                total += c["total_cost"]
            if not ok: continue
            for det in detections:
                hi = det["hand_index"]
                if hi not in combo:
                    for side in ["LEFT", "RIGHT"]:
                        if side not in combo.values() and costs[(hi, side)]["within_gate"]:
                            total += UNASSIGNED_PENALTY; break
            scored.append((total, combo))

        scored.sort(key=lambda x: x[0])
        best_score, best_combo = scored[0] if scored else (float("inf"), {})
        second_score = scored[1][0] if len(scored) > 1 else None
        second_combo = scored[1][1] if len(scored) > 1 else None

        # Cost margin
        assignment_margin = round(second_score - best_score, 4) if second_score is not None else None

        # Single vs dual margin
        single_combos = [s for s in scored if len(s[1]) <= 1]
        dual_combos = [s for s in scored if len(s[1]) >= 2]
        best_single = single_combos[0][0] if single_combos else None
        best_dual = dual_combos[0][0] if dual_combos else None
        single_vs_dual_margin = round(best_dual - best_single, 4) \
            if best_single is not None and best_dual is not None else None

        # ── Build results ──
        results = []
        for det in detections:
            hi = det["hand_index"]
            assigned_side = best_combo.get(hi)

            cost_left = costs[(hi, "LEFT")]
            cost_right = costs[(hi, "RIGHT")]
            dl = cost_left.get("dist_px")
            dr = cost_right.get("dist_px")
            dist_ratio = round(dl / dr, 4) if dl and dr and dr > 0 else None
            gate = cost_left["gate_px"]

            dup_rule = "not_applicable"
            if close_same_wrist_conflict:
                dup_rule = f"close_same_wrist_{conflict_nearest}_inter={inter_det_dist:.1f}px_norm={norm_inter_det:.3f}"

            if assigned_side is not None:
                status = "DETECTED"
                anat = f"ANAT_{assigned_side}"
                reason = f"global_best_for_{assigned_side}"
                rejected_reason = None
                sel_cost = costs[(hi, assigned_side)]
            else:
                # Determine why unassigned
                viable = [s for s in ["LEFT", "RIGHT"] if costs[(hi, s)]["within_gate"]]
                taken = set(best_combo.values())

                if close_same_wrist_conflict and conflict_nearest:
                    # Check if this is the loser in a duplicate conflict
                    winner_hi = [h for h, s in best_combo.items() if s == conflict_nearest]
                    if winner_hi:
                        inter = inter_det_dist
                        sw_th = sw * THRESHOLDS["duplicate_inter_hand_sw_ratio"] if sw else 120
                        if inter is not None and inter < sw_th:
                            status = "DUPLICATE_DETECTION_CANDIDATE"
                            anat = None
                            reason = f"duplicate_of_det{winner_hi[0]}_at_{conflict_nearest}"
                            rejected_reason = f"inter={inter:.1f}px<threshold={sw_th:.1f}px"
                        else:
                            status = "AMBIGUOUS"
                            anat = None
                            reason = f"same_wrist_{conflict_nearest}_but_inter={inter:.1f}px"
                            rejected_reason = None
                    else:
                        status = "AMBIGUOUS"
                        anat = None
                        reason = "conflict_but_no_winner"
                        rejected_reason = None
                elif not viable:
                    status = "AMBIGUOUS"
                    anat = None
                    reason = "no_pose_wrist_within_gate"
                    rejected_reason = None
                elif all(s in taken for s in viable):
                    winner_for_viable = None
                    for s in viable:
                        for h, vs in best_combo.items():
                            if vs == s: winner_for_viable = h
                    if winner_for_viable is not None and inter_det_dist is not None:
                        sw_th = sw * THRESHOLDS["duplicate_inter_hand_sw_ratio"] if sw else 120
                        if inter_det_dist < sw_th:
                            status = "DUPLICATE_DETECTION_CANDIDATE"
                            anat = None
                            reason = f"viable_taken_close_inter={inter_det_dist:.1f}px"
                            rejected_reason = f"winner_det{winner_for_viable}"
                        else:
                            status = "AMBIGUOUS"
                            anat = None
                            reason = "viable_taken_not_close"
                            rejected_reason = None
                    else:
                        status = "AMBIGUOUS"
                        anat = None
                        reason = "viable_sides_taken"
                        rejected_reason = None
                else:
                    status = "AMBIGUOUS"
                    anat = None
                    reason = "global_optimizer_did_not_assign"
                    rejected_reason = None

                sel_cost = None

            # Best/second-best candidate names
            c_left_total = cost_left["total_cost"]
            c_right_total = cost_right["total_cost"]
            if c_left_total <= c_right_total:
                best_cand_name, best_cand_score = "LEFT", c_left_total
                second_cand_name, second_cand_score = "RIGHT", c_right_total
            else:
                best_cand_name, best_cand_score = "RIGHT", c_right_total
                second_cand_name, second_cand_score = "LEFT", c_left_total

            # For AMBIGUOUS/DUP: still record costs to both sides
            if sel_cost:
                sel_dist_norm = sel_cost["dist_norm"]
                cont = sel_cost["continuity_cost"]
                elb = sel_cost["elbow_cost"]
                hc = sel_cost["handedness_cost"]
                fc = sel_cost["total_cost"]
            else:
                sel_dist_norm = None
                # Record best viable cost anyway
                cont = cost_left["continuity_cost"]
                elb = cost_left["elbow_cost"]
                hc = cost_left["handedness_cost"]
                fc = None

            results.append({
                "hand_index": hi,
                "anatomical_identity": anat,
                "association_status": status,
                "model_handedness_label": det["model_label"],
                "model_handedness_score": round(det["model_score"], 4),
                "hand_wrist_px": [round(float(det["wrist_px"][0]), 2),
                                  round(float(det["wrist_px"][1]), 2)],
                "pose_left_wrist_dist_px": round(dl, 2) if dl is not None else None,
                "pose_right_wrist_dist_px": round(dr, 2) if dr is not None else None,
                "normalized_selected_distance": round(sel_dist_norm, 4) if sel_dist_norm is not None else None,
                "cost_to_left": round(c_left_total, 4),
                "cost_to_right": round(c_right_total, 4),
                "distance_ratio_left_vs_right": dist_ratio,
                "inter_detection_distance_px": round(inter_det_dist, 2) if inter_det_dist is not None else None,
                "normalized_inter_detection_distance": round(norm_inter_det, 4) if norm_inter_det is not None else None,
                "best_candidate_name": best_cand_name,
                "best_candidate_score": round(best_cand_score, 4),
                "second_best_candidate_name": second_cand_name,
                "second_best_candidate_score": round(second_cand_score, 4),
                "assignment_cost_margin": assignment_margin,
                "single_vs_dual_assignment_margin": single_vs_dual_margin,
                "continuity_cost": cont,
                "elbow_direction_cost": elb,
                "handedness_auxiliary_cost": hc,
                "final_cost": fc,
                "assignment_reason": reason,
                "rejected_alternative_reason": rejected_reason,
                "rejection_rule": "close_same_wrist_conflict" if close_same_wrist_conflict and status != "DETECTED" else ("gate" if status == "AMBIGUOUS" and "gate" in (reason or "") else "none"),
                "duplicate_rule_result": dup_rule,
                "shoulder_width_px": round(sw, 2) if sw else None,
                "gate_px": round(gate, 2),
            })

            if status == "DETECTED" and assigned_side:
                prev_pos = self.prev_accepted.get(assigned_side, {}).get("pos")
                prev_t = self.prev_accepted.get(assigned_side, {}).get("pts_s")
                if prev_pos is not None and prev_t is not None:
                    dt = cam_pts_s - prev_t
                    if 0 < dt < 0.2:
                        self.prev_velocity[assigned_side] = (det["wrist_px"] - prev_pos) / dt
                self.prev_accepted[assigned_side] = {"pos": det["wrist_px"].copy(), "pts_s": cam_pts_s}

        return results


# ═══════════════════════════════════════════════════════════
#  Bidirectional temporal refinement
# ═══════════════════════════════════════════════════════════

def bidirectional_refine(sample_frames):
    """Post-process single-detection frames using forward+backward anchors."""
    if not sample_frames:
        return sample_frames

    # Find anchor frames: both hands DETECTED with good separation
    anchors = []
    for fi, f in enumerate(sample_frames):
        if f["anat_left_status"] != "DETECTED" or f["anat_right_status"] != "DETECTED":
            continue
        # Get wrist positions
        l_pos = r_pos = None
        sw = None
        for hand in f.get("hands", []):
            if hand["association_status"] != "DETECTED": continue
            if hand["anatomical_identity"] == "ANAT_LEFT":
                l_pos = np.array(hand["hand_wrist_px"])
            elif hand["anatomical_identity"] == "ANAT_RIGHT":
                r_pos = np.array(hand["hand_wrist_px"])
            if hand.get("shoulder_width_px"):
                sw = hand["shoulder_width_px"]

        if l_pos is None or r_pos is None: continue
        sep = float(np.linalg.norm(l_pos - r_pos))
        sw_val = sw if sw else 200
        if sep / sw_val > THRESHOLDS["anchor_separation_sw_ratio"]:
            anchors.append({
                "fi": fi, "pts_s": f["camera_local_pts_s"],
                "left_pos": l_pos, "right_pos": r_pos,
            })

    if not anchors:
        return sample_frames

    max_gap = THRESHOLDS["bidirectional_anchor_max_gap_s"]

    # Check single-detection frames
    for fi, f in enumerate(sample_frames):
        n_detected = sum(1 for h in f.get("hands", []) if h["association_status"] == "DETECTED")
        if n_detected != 1: continue

        detected_hand = None
        for h in f.get("hands", []):
            if h["association_status"] == "DETECTED":
                detected_hand = h; break
        if not detected_hand: continue

        det_pos = np.array(detected_hand["hand_wrist_px"])
        det_anat = detected_hand["anatomical_identity"]
        det_side = det_anat.replace("ANAT_", "") if det_anat else None
        pts_s = f["camera_local_pts_s"]

        # Find nearest anchor before and after
        prev_anchor = None
        next_anchor = None
        for a in anchors:
            if a["fi"] < fi:
                if prev_anchor is None or a["fi"] > prev_anchor["fi"]:
                    prev_anchor = a
            elif a["fi"] > fi:
                if next_anchor is None or a["fi"] < next_anchor["fi"]:
                    next_anchor = a

        # Check if anchors support the current assignment
        should_change = False
        new_identity = None

        for anchor in [prev_anchor, next_anchor]:
            if anchor is None: continue
            gap = abs(pts_s - anchor["pts_s"])
            if gap > max_gap: continue

            # Which anchor hand is closer to det_pos?
            d_to_left = float(np.linalg.norm(det_pos - anchor["left_pos"]))
            d_to_right = float(np.linalg.norm(det_pos - anchor["right_pos"]))

            # Determine anchor-supported identity
            if d_to_left < d_to_right:
                anchor_supports = "LEFT"
            else:
                anchor_supports = "RIGHT"

            if det_side and anchor_supports != det_side:
                # Anchor disagrees with current assignment
                should_change = True
                new_identity = anchor_supports

        if should_change and new_identity:
            # Check if both anchors agree (if both exist within gap)
            prev_supports = next_supports = None
            if prev_anchor and abs(pts_s - prev_anchor["pts_s"]) <= max_gap:
                d_l = float(np.linalg.norm(det_pos - prev_anchor["left_pos"]))
                d_r = float(np.linalg.norm(det_pos - prev_anchor["right_pos"]))
                prev_supports = "LEFT" if d_l < d_r else "RIGHT"
            if next_anchor and abs(pts_s - next_anchor["pts_s"]) <= max_gap:
                d_l = float(np.linalg.norm(det_pos - next_anchor["left_pos"]))
                d_r = float(np.linalg.norm(det_pos - next_anchor["right_pos"]))
                next_supports = "LEFT" if d_l < d_r else "RIGHT"

            if prev_supports and next_supports:
                if prev_supports == next_supports and prev_supports != det_side:
                    # Both anchors agree on different identity
                    new_anat = f"ANAT_{prev_supports}"
                    detected_hand["anatomical_identity"] = new_anat
                    detected_hand["association_status"] = "DETECTED"
                    detected_hand["assignment_reason"] = \
                        f"bidirectional_correction_from_{det_anat}_to_{new_anat}"
                    old_side_key = f"anat_{det_side.lower()}_status" if det_side else None
                    new_side_key = f"anat_{prev_supports.lower()}_status"
                    if old_side_key:
                        f[old_side_key] = "NOT_DETECTED"
                        f[f"anat_{det_side.lower()}_landmarks"] = None
                    f[new_side_key] = "DETECTED"
                    f[f"anat_{prev_supports.lower()}_landmarks"] = detected_hand["landmarks"]
                elif prev_supports != next_supports:
                    # Anchors disagree → AMBIGUOUS
                    detected_hand["anatomical_identity"] = None
                    detected_hand["association_status"] = "AMBIGUOUS"
                    detected_hand["assignment_reason"] = \
                        f"bidirectional_conflict_prev={prev_supports}_next={next_supports}"
                    if det_side:
                        f[f"anat_{det_side.lower()}_status"] = "NOT_DETECTED"
                        f[f"anat_{det_side.lower()}_landmarks"] = None
                    f["anat_left_status"] = "AMBIGUOUS" if f["anat_left_status"] == "NOT_DETECTED" else f["anat_left_status"]
                    f["anat_right_status"] = "AMBIGUOUS" if f["anat_right_status"] == "NOT_DETECTED" else f["anat_right_status"]
            elif prev_supports or next_supports:
                single_anchor = prev_supports or next_supports
                if single_anchor != det_side:
                    # Single anchor disagrees → change or AMBIGUOUS
                    new_anat = f"ANAT_{single_anchor}"
                    detected_hand["anatomical_identity"] = new_anat
                    detected_hand["assignment_reason"] = \
                        f"bidirectional_single_anchor_correction_to_{new_anat}"
                    if det_side:
                        f[f"anat_{det_side.lower()}_status"] = "NOT_DETECTED"
                        f[f"anat_{det_side.lower()}_landmarks"] = None
                    f[f"anat_{single_anchor.lower()}_status"] = "DETECTED"
                    f[f"anat_{single_anchor.lower()}_landmarks"] = detected_hand["landmarks"]

    return sample_frames


# ═══════════════════════════════════════════════════════════
#  Per-view extraction
# ═══════════════════════════════════════════════════════════

def extract_view(view, segments, affine_a, affine_b, K, dist,
                 handedness_weight, observed_mapping):
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
            "sample_id": seg["sample_id"], "cam_start": cr["start_pts_s"],
            "cam_end": cr["end_pts_s"], "front_start": seg["front_start_s"],
            "front_end": seg["front_end_s"],
        })

    cam_start_global = min(sr["cam_start"] for sr in seg_ranges)
    cam_end_global = max(sr["cam_end"] for sr in seg_ranges)

    pose_lm = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False,
        min_pose_detection_confidence=THRESHOLDS["pose_min_detection_confidence"],
        min_pose_presence_confidence=THRESHOLDS["pose_min_presence_confidence"],
        min_tracking_confidence=THRESHOLDS["pose_min_tracking_confidence"],
    ))
    hand_lm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=THRESHOLDS["hand_min_detection_confidence"],
        min_hand_presence_confidence=THRESHOLDS["hand_min_presence_confidence"],
        min_tracking_confidence=THRESHOLDS["hand_min_tracking_confidence"],
    ))

    all_frames = []
    pts_missing = []
    prev_ts_ms = -1; ts_adj = 0; frame_idx = 0; n_in = 0
    seg_first = {sr["sample_id"]: None for sr in seg_ranges}
    seg_last = {sr["sample_id"]: None for sr in seg_ranges}
    tracker = AnatomicalHandTracker(handedness_weight, observed_mapping)
    current_sid = None

    for frame in container.decode(stream):
        if frame.pts is None:
            pts_missing.append({"frame_index": frame_idx, "view": view})
            frame_idx += 1; continue
        cam_pts_s = float(frame.pts) * tb
        if cam_pts_s < cam_start_global - 0.5: frame_idx += 1; continue
        if cam_pts_s > cam_end_global + 0.5: break

        current_seg = None
        for sr in seg_ranges:
            if sr["cam_start"] <= cam_pts_s <= sr["cam_end"]:
                current_seg = sr; break
        if current_seg is None: frame_idx += 1; continue

        n_in += 1
        sid = current_seg["sample_id"]
        front_pts_s = cam_to_front(cam_pts_s, affine_a, affine_b)
        if sid != current_sid: tracker.reset(); current_sid = sid
        if seg_first[sid] is None: seg_first[sid] = cam_pts_s
        seg_last[sid] = cam_pts_s

        ts_ms = int(round(cam_pts_s * 1000))
        orig_ts = ts_ms
        if ts_ms <= prev_ts_ms: ts_ms = prev_ts_ms + 1; ts_adj += 1
        prev_ts_ms = ts_ms

        img_rgb = frame.to_ndarray(format="rgb24")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        pose_result = pose_lm.detect_for_video(mp_image, ts_ms)
        hand_result = hand_lm.detect_for_video(mp_image, ts_ms)

        fr = {
            "sample_id": sid, "view": view, "decoded_frame_index": frame_idx,
            "camera_local_pts_s": round(cam_pts_s, 6),
            "mapped_front_pts_s": round(front_pts_s, 6),
            "timestamp_ms_sent_to_mediapipe": ts_ms,
            "timestamp_ms_original": orig_ts,
            "timestamp_adjusted": ts_ms != orig_ts,
            "image_width": img_w, "image_height": img_h,
        }

        # Pose
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            plm = pose_result.pose_landmarks[0]
            pd = []
            for li, lm in enumerate(plm):
                rx, ry = lm.x, lm.y
                px, py = rx*img_w, ry*img_h
                up, un = undistort_points(np.array([[px, py]]), K, dist)
                upx = up[0] if len(up) > 0 else [None, None]
                unx = un[0] if len(un) > 0 else [None, None]
                pd.append({
                    "index": li, "name": POSE_LANDMARK_NAMES[li],
                    "raw_norm_x": round(float(rx), 6), "raw_norm_y": round(float(ry), 6),
                    "raw_distorted_px_x": round(float(px), 2), "raw_distorted_px_y": round(float(py), 2),
                    "undistorted_px_x": round(float(upx[0]), 2), "undistorted_px_y": round(float(upx[1]), 2),
                    "undistorted_norm_cam_x": round(float(unx[0]), 6), "undistorted_norm_cam_y": round(float(unx[1]), 6),
                    "visibility": round(float(lm.visibility), 4),
                    "presence": round(float(lm.presence), 4) if hasattr(lm, "presence") else None,
                })
            fr["pose_landmarks"] = pd; fr["pose_detected"] = True
        else:
            fr["pose_landmarks"] = None; fr["pose_detected"] = False

        # Hands
        nh = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0
        fr["hands_detected_count"] = nh
        fr["anat_left_status"] = "NOT_DETECTED"; fr["anat_right_status"] = "NOT_DETECTED"
        fr["anat_left_landmarks"] = None; fr["anat_right_landmarks"] = None

        if nh > 0:
            assocs = tracker.associate(pose_result, hand_result, img_w, img_h, cam_pts_s)
            raw_hands = []
            for hi in range(nh):
                a = assocs[hi] if hi < len(assocs) else {
                    "anatomical_identity": None, "association_status": "AMBIGUOUS",
                    "model_handedness_label": "Unknown", "model_handedness_score": 0,
                    "hand_wrist_px": [None, None],
                    "assignment_reason": "no_result", "rejected_alternative_reason": None,
                }
                hlm = hand_result.hand_landmarks[hi]
                hpts = []
                for li, lm in enumerate(hlm):
                    rx, ry = lm.x, lm.y; px, py = rx*img_w, ry*img_h
                    up, un = undistort_points(np.array([[px, py]]), K, dist)
                    upx = up[0] if len(up) > 0 else [None, None]
                    unx = un[0] if len(un) > 0 else [None, None]
                    hpts.append({
                        "index": li, "name": HAND_LANDMARK_NAMES[li],
                        "raw_norm_x": round(float(rx), 6), "raw_norm_y": round(float(ry), 6),
                        "raw_distorted_px_x": round(float(px), 2), "raw_distorted_px_y": round(float(py), 2),
                        "undistorted_px_x": round(float(upx[0]), 2), "undistorted_px_y": round(float(upx[1]), 2),
                        "undistorted_norm_cam_x": round(float(unx[0]), 6), "undistorted_norm_cam_y": round(float(unx[1]), 6),
                        "visibility": None, "presence": None,
                    })

                rh = dict(a)
                rh["landmarks"] = hpts
                raw_hands.append(rh)

                st = a.get("association_status", "AMBIGUOUS")
                aid = a.get("anatomical_identity")
                if aid == "ANAT_LEFT" and st == "DETECTED":
                    fr["anat_left_status"] = "DETECTED"; fr["anat_left_landmarks"] = hpts
                elif aid == "ANAT_RIGHT" and st == "DETECTED":
                    fr["anat_right_status"] = "DETECTED"; fr["anat_right_landmarks"] = hpts

            fr["hands"] = raw_hands
            for rh in raw_hands:
                st = rh["association_status"]
                if st == "AMBIGUOUS":
                    for k in ["anat_left_status", "anat_right_status"]:
                        if fr[k] == "NOT_DETECTED": fr[k] = "AMBIGUOUS"
                elif st == "DUPLICATE_DETECTION_CANDIDATE":
                    for k in ["anat_left_status", "anat_right_status"]:
                        if fr[k] == "NOT_DETECTED": fr[k] = "DUPLICATE_DETECTION_CANDIDATE"; break
        else:
            fr["hands"] = []

        all_frames.append(fr)
        frame_idx += 1
        if n_in % 200 == 0: print(f"    {view}: {n_in} frames, t={cam_pts_s:.1f}s")

    container.close(); pose_lm.close(); hand_lm.close()
    print(f"    {view}: {n_in} total, {ts_adj} ts adj, {len(pts_missing)} PTS-miss")

    # Bidirectional refinement per sample (pass 2 only)
    if handedness_weight > 0:
        for sid in [sr["sample_id"] for sr in seg_ranges]:
            sf = [f for f in all_frames if f["sample_id"] == sid]
            bidirectional_refine(sf)
        print(f"    {view}: bidirectional refinement applied")

    # Boundary diagnostics
    bdiag = []
    for sr in seg_ranges:
        sid = sr["sample_id"]; f0 = seg_first[sid]; fl = seg_last[sid]
        bdiag.append({
            "sample_id": sid, "view": view,
            "requested_cam_start_s": round(sr["cam_start"], 6),
            "requested_cam_end_s": round(sr["cam_end"], 6),
            "actual_first_frame_pts_s": round(f0, 6) if f0 else None,
            "actual_last_frame_pts_s": round(fl, 6) if fl else None,
            "start_error_ms": round((f0 - sr["cam_start"]) * 1000, 2) if f0 else None,
            "end_error_ms": round((sr["cam_end"] - fl) * 1000, 2) if fl else None,
            "n_frames": sum(1 for f in all_frames if f["sample_id"] == sid),
        })
    return all_frames, bdiag, pts_missing


# ═══════════════════════════════════════════════════════════
#  Observed mapping
# ═══════════════════════════════════════════════════════════

def compute_observed_mapping(frames):
    counts = defaultdict(int)
    for f in frames:
        for h in f.get("hands", []):
            if h["association_status"] != "DETECTED": continue
            a = h.get("anatomical_identity")
            if not a: continue
            s = a.replace("ANAT_", "")
            da = h.get(f"pose_{s.lower()}_wrist_dist_px")
            ot = "right" if s == "LEFT" else "left"
            do = h.get(f"pose_{ot}_wrist_dist_px")
            if da and do and da < 80 and do > da * 2.5:
                counts[(h["model_handedness_label"], a)] += 1
    mapping = {}
    for label in ["Left", "Right"]:
        best_a, best_c = None, 0
        for (l, a), c in counts.items():
            if l == label and c > best_c: best_c = c; best_a = a
        if best_a and best_c >= 10: mapping[label] = best_a
    return mapping, dict(counts)


# ═══════════════════════════════════════════════════════════
#  QA stats
# ═══════════════════════════════════════════════════════════

def compute_qa_stats(frames_data, view, segments):
    stats = []
    for seg in segments:
        sid = seg["sample_id"]
        sf = [f for f in frames_data if f["sample_id"] == sid and f["view"] == view]
        n = len(sf)
        if n == 0: stats.append({"sample_id": sid, "view": view, "n_frames": 0}); continue

        pu = sum(1 for f in sf if f["pose_detected"] and f["pose_landmarks"]
                 and all(f["pose_landmarks"][i]["visibility"] > THRESHOLDS["pose_visibility_usable_threshold"]
                         for i in [11,12,13,14,15,16]))
        ld = rd = bd = af = df = 0
        lmr = lmm = rmr = rmm = bmr = bmm = 0
        lwd = []; rwd = []
        plp = prp = None; plfi = prfi = None
        swap_cands = []; swap_cmps = 0

        # Conflict counts
        same_nearest = close_same = dual_det_conflict = 0

        for fi, f in enumerate(sf):
            hl = f["anat_left_status"] == "DETECTED"
            hr = f["anat_right_status"] == "DETECTED"
            if hl: ld += 1
            if hr: rd += 1
            if hl and hr: bd += 1
            if f["anat_left_status"] == "AMBIGUOUS" or f["anat_right_status"] == "AMBIGUOUS": af += 1
            if f["anat_left_status"] == "DUPLICATE_DETECTION_CANDIDATE" or \
               f["anat_right_status"] == "DUPLICATE_DETECTION_CANDIDATE": df += 1

            if not hl: lmr += 1; lmm = max(lmm, lmr)
            else: lmr = 0
            if not hr: rmr += 1; rmm = max(rmm, rmr)
            else: rmr = 0
            if not hl and not hr: bmr += 1; bmm = max(bmm, bmr)
            else: bmr = 0

            clp = crp = None
            for h in f.get("hands", []):
                if h["association_status"] != "DETECTED": continue
                a = h.get("anatomical_identity")
                wp = h.get("hand_wrist_px")
                if a == "ANAT_LEFT":
                    d = h.get("pose_left_wrist_dist_px")
                    if d is not None: lwd.append(d)
                    if wp: clp = np.array(wp)
                elif a == "ANAT_RIGHT":
                    d = h.get("pose_right_wrist_dist_px")
                    if d is not None: rwd.append(d)
                    if wp: crp = np.array(wp)

            # Conflict tracking for this frame
            hands = f.get("hands", [])
            if len(hands) == 2:
                dl0 = hands[0].get("pose_left_wrist_dist_px") or 9999
                dr0 = hands[0].get("pose_right_wrist_dist_px") or 9999
                dl1 = hands[1].get("pose_left_wrist_dist_px") or 9999
                dr1 = hands[1].get("pose_right_wrist_dist_px") or 9999
                n0 = "L" if dl0 < dr0 else "R"
                n1 = "L" if dl1 < dr1 else "R"
                if n0 == n1:
                    same_nearest += 1
                    idd = hands[0].get("inter_detection_distance_px")
                    nid = hands[0].get("normalized_inter_detection_distance")
                    if nid is not None and nid < THRESHOLDS["duplicate_inter_hand_sw_ratio"]:
                        close_same += 1
                        if hands[0]["association_status"] == "DETECTED" and \
                           hands[1]["association_status"] == "DETECTED":
                            dual_det_conflict += 1

            # Swap
            if clp is not None and crp is not None and plp is not None and prp is not None:
                lg = fi - plfi if plfi is not None else 999
                rg = fi - prfi if prfi is not None else 999
                if lg <= THRESHOLDS["swap_gap_max_frames"] and rg <= THRESHOLDS["swap_gap_max_frames"]:
                    swap_cmps += 1
                    nc = float(np.linalg.norm(clp-plp)) + float(np.linalg.norm(crp-prp))
                    sc = float(np.linalg.norm(clp-prp)) + float(np.linalg.norm(crp-plp))
                    if nc > 0 and sc < THRESHOLDS["swap_cost_ratio_threshold"] * nc:
                        swap_cands.append({
                            "frame_index_in_sample": fi, "camera_pts_s": f["camera_local_pts_s"],
                            "normal_cost": round(nc, 1), "swapped_cost": round(sc, 1),
                            "ratio": round(sc/nc, 3),
                        })
            if clp is not None: plp = clp; plfi = fi
            if crp is not None: prp = crp; prfi = fi

        dt = (sf[-1]["camera_local_pts_s"] - sf[0]["camera_local_pts_s"]) / (n-1) if n > 1 else 1/30
        la = np.array(lwd) if lwd else None
        ra = np.array(rwd) if rwd else None

        stats.append({
            "sample_id": sid, "view": view, "n_frames": n,
            "pose_usable_rate": round(pu/n, 4),
            "anat_left_detection_rate": round(ld/n, 4),
            "anat_right_detection_rate": round(rd/n, 4),
            "both_hands_detected_rate": round(bd/n, 4),
            "left_max_consecutive_missing_frames": lmm,
            "left_max_consecutive_missing_time_s": round(lmm*dt, 3),
            "right_max_consecutive_missing_frames": rmm,
            "right_max_consecutive_missing_time_s": round(rmm*dt, 3),
            "both_unavailable_max_consecutive_frames": bmm,
            "both_unavailable_max_consecutive_time_s": round(bmm*dt, 3),
            "ambiguous_frame_count": af,
            "duplicate_detection_candidate_frame_count": df,
            "same_nearest_wrist_conflict_frames": same_nearest,
            "close_same_wrist_conflict_frames": close_same,
            "accepted_dual_hand_conflict_count": dual_det_conflict,
            "identity_swap_candidate_count": len(swap_cands),
            "identity_swap_comparisons_executed": swap_cmps,
            "identity_swap_candidate_frames": swap_cands,
            "left_assigned_wrist_dist_median_px": round(float(np.median(la)),2) if la is not None else None,
            "left_assigned_wrist_dist_p95_px": round(float(np.percentile(la,95)),2) if la is not None else None,
            "left_assigned_wrist_dist_max_px": round(float(np.max(la)),2) if la is not None else None,
            "right_assigned_wrist_dist_median_px": round(float(np.median(ra)),2) if ra is not None else None,
            "right_assigned_wrist_dist_p95_px": round(float(np.percentile(ra,95)),2) if ra is not None else None,
            "right_assigned_wrist_dist_max_px": round(float(np.max(ra)),2) if ra is not None else None,
        })
    return stats


# ═══════════════════════════════════════════════════════════
#  Synchronized coverage (v1.3)
# ═══════════════════════════════════════════════════════════

def compute_sync_coverage(all_view_frames, segments):
    rows = []; tol = 0.025
    for seg in segments:
        sid = seg["sample_id"]
        ff = [f for f in all_view_frames["FRONT"] if f["sample_id"] == sid]
        n_ref = len(ff)
        for hand in ["LEFT", "RIGHT"]:
            ak = f"anat_{hand.lower()}_status"
            fd_count = sum(1 for f in ff if f[ak] == "DETECTED")
            ot = {}
            for ov in ["LEFT", "RIGHT"]:
                frames = [f for f in all_view_frames[ov] if f["sample_id"] == sid]
                ot[ov] = np.array(sorted([f["mapped_front_pts_s"] for f in frames if f[ak] == "DETECTED"]))

            used_l = set(); used_r = set()
            n2 = n3 = 0; fd_with_other = 0

            for ri, ff_i in enumerate(ff):
                ft = ff_i["mapped_front_pts_s"]
                front_has = ff_i[ak] == "DETECTED"
                views = 0
                if front_has: views = 1
                matched_other = False
                for ov in ["LEFT", "RIGHT"]:
                    arr = ot[ov]; us = used_l if ov == "LEFT" else used_r
                    if len(arr) == 0: continue
                    idx = np.searchsorted(arr, ft)
                    bc = None; bd = tol + 1
                    for ci in [idx-1, idx, idx+1]:
                        if 0 <= ci < len(arr) and ci not in us:
                            diff = abs(arr[ci] - ft)
                            if diff <= tol and diff < bd: bd = diff; bc = ci
                    if bc is not None:
                        views += 1; us.add(bc); matched_other = True
                if views >= 2: n2 += 1
                if views >= 3: n3 += 1
                if front_has and matched_other: fd_with_other += 1

            # Verify no reuse
            reuse_l = len(used_l) - len(set(used_l))  # always 0 by set
            reuse_r = len(used_r) - len(set(used_r))
            reuse = reuse_l + reuse_r

            u2r = round(n2/n_ref, 4) if n_ref > 0 else None
            u3r = round(n3/n_ref, 4) if n_ref > 0 else None
            cfr = round(fd_with_other/fd_count, 4) if fd_count > 0 else None

            assert u2r is None or 0 <= u2r <= 1, f"u2r={u2r}"
            assert u3r is None or 0 <= u3r <= 1, f"u3r={u3r}"
            assert cfr is None or 0 <= cfr <= 1, f"cfr={cfr}"
            assert n3 <= n2 <= n_ref
            assert fd_with_other <= fd_count

            rows.append({
                "sample_id": sid, "anatomical_hand": hand,
                "total_reference_frames": n_ref, "front_detected_frames": fd_count,
                "frames_with_2plus_views": n2, "frames_with_3_views": n3,
                "unconditional_2plus_rate": u2r, "unconditional_3view_rate": u3r,
                "front_detected_with_other_view_frames": fd_with_other,
                "conditional_on_front_detected_rate": cfr,
                "reused_match_count": reuse,
                "unmatched_left_detections": len(ot["LEFT"]) - len(used_l),
                "unmatched_right_detections": len(ot["RIGHT"]) - len(used_r),
            })
    return rows


# ═══════════════════════════════════════════════════════════
#  Overlay & contact sheets (abbreviated — same structure as v1.2)
# ═══════════════════════════════════════════════════════════

def make_overlay_video(view, frames_data, segments, aa, ab, out_path, qa_stats, fps=15):
    video_path = VIDEOS[view]
    c0 = av.open(str(video_path)); s0 = c0.streams.video[0]
    iw, ih = s0.width, s0.height; c0.close()
    sc = 0.5; ow, oh = int(iw*sc), int(ih*sc)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (ow, oh))
    fl = {}
    for fd in frames_data:
        if fd["view"] == view: fl[round(fd["camera_local_pts_s"], 4)] = fd
    srs = [(seg["camera_ranges"][view]["start_pts_s"], seg["camera_ranges"][view]["end_pts_s"],
            seg["sample_id"]) for seg in segments]
    swap_pts = set()
    for qs in qa_stats:
        if qs["view"] == view:
            for sc2 in qs.get("identity_swap_candidate_frames", []): swap_pts.add(round(sc2["camera_pts_s"], 2))
    dt = 1.0/fps; written = 0
    for ss, se, ssid in srs:
        sep = np.zeros((oh, ow, 3), dtype=np.uint8)
        cv2.putText(sep, f"{view} {_short_sid(ssid)}", (10, oh//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        for _ in range(int(fps*0.5)): writer.write(sep); written += 1
        con = av.open(str(video_path)); st = con.streams.video[0]; st.thread_type = "AUTO"
        tb = float(st.time_base); ng = ss
        for frame in con.decode(st):
            if frame.pts is None: continue
            cps = float(frame.pts)*tb
            if cps < ss-0.05: continue
            if cps > se+0.05: break
            if cps < ng or cps < ss or cps > se: continue
            ng = cps + dt
            img = frame.to_ndarray(format="bgr24"); small = cv2.resize(img, (ow, oh))
            fps2 = cam_to_front(cps, aa, ab)
            cv2.rectangle(small, (0,0), (ow,40), (0,80,0), -1)
            cv2.putText(small, f"{view} {_short_sid(ssid)} cam={cps:.2f} front={fps2:.2f}",
                        (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
            key = round(cps, 4); fd = fl.get(key)
            if fd is None:
                for d in [0.0001,-0.0001,0.001,-0.001,0.01,-0.01]:
                    fd = fl.get(round(cps+d, 4))
                    if fd: break
            warns = []
            if fd:
                if fd.get("pose_detected") and fd.get("pose_landmarks"):
                    pts = [(int(lm["raw_distorted_px_x"]*sc), int(lm["raw_distorted_px_y"]*sc)) for lm in fd["pose_landmarks"]]
                    for i,j in POSE_CONNECTIONS:
                        if fd["pose_landmarks"][i]["visibility"]>0.3 and fd["pose_landmarks"][j]["visibility"]>0.3:
                            cv2.line(small, pts[i], pts[j], (200,200,200), 1)
                    for wi,wc in [(15,(0,200,0)),(16,(0,128,255))]:
                        if fd["pose_landmarks"][wi]["visibility"]>0.3: cv2.circle(small, pts[wi], 5, wc, 2)
                for h in fd.get("hands", []):
                    st2 = h.get("association_status","AMBIGUOUS"); a2 = h.get("anatomical_identity")
                    if st2=="DETECTED" and a2: color = {"ANAT_LEFT":(0,255,0),"ANAT_RIGHT":(0,165,255)}.get(a2,(128,128,128))
                    elif st2=="DUPLICATE_DETECTION_CANDIDATE": color = (255,0,255)
                    else: color = (0,0,255)
                    hp = [(int(lm["raw_distorted_px_x"]*sc), int(lm["raw_distorted_px_y"]*sc)) for lm in h["landmarks"]]
                    for i,j in HAND_CONNECTIONS: cv2.line(small, hp[i], hp[j], color, 1)
                    wx,wy = hp[0]; hi2 = h["hand_index"]; ml = h["model_handedness_label"]
                    short = "L" if a2=="ANAT_LEFT" else ("R" if a2=="ANAT_RIGHT" else ("DUP" if st2=="DUPLICATE_DETECTION_CANDIDATE" else "?"))
                    cv2.putText(small, f"d{hi2}:{ml[0]}->{short}", (wx-10,wy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    dl2 = h.get("pose_left_wrist_dist_px"); dr2 = h.get("pose_right_wrist_dist_px")
                    if dl2 and dr2: cv2.putText(small, f"dL={dl2:.0f} dR={dr2:.0f}", (wx-10,wy+15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
                    if st2=="DUPLICATE_DETECTION_CANDIDATE": warns.append("DUP")
                    elif st2=="AMBIGUOUS": warns.append("AMBIG")
                if not fd.get("pose_detected"): warns.append("NO_POSE")
                if fd["hands_detected_count"]==0: warns.append("NO_HANDS")
                if round(cps,2) in swap_pts: warns.append("SWAP")
            if warns: cv2.putText(small, " ".join(set(warns)), (5,35), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            writer.write(small); written += 1
        con.close()
    writer.release()
    print(f"    Overlay {view}: {written} frames -> {out_path.name}")


def _draw_lm(thumb, fd, tw, th, iw, ih):
    if not fd: return
    sx, sy = tw/iw, th/ih
    if fd.get("pose_detected") and fd.get("pose_landmarks"):
        pts = [(int(lm["raw_distorted_px_x"]*sx), int(lm["raw_distorted_px_y"]*sy)) for lm in fd["pose_landmarks"]]
        for i,j in POSE_CONNECTIONS:
            if fd["pose_landmarks"][i]["visibility"]>0.3 and fd["pose_landmarks"][j]["visibility"]>0.3:
                cv2.line(thumb, pts[i], pts[j], (200,200,200), 1)
        for wi,wc in [(15,(0,200,0)),(16,(0,128,255))]:
            if fd["pose_landmarks"][wi]["visibility"]>0.3: cv2.circle(thumb, pts[wi], 4, wc, 2)
    for h in fd.get("hands",[]):
        st = h.get("association_status","AMBIGUOUS"); a = h.get("anatomical_identity")
        if st=="DETECTED" and a: c = {"ANAT_LEFT":(0,255,0),"ANAT_RIGHT":(0,165,255)}.get(a,(128,128,128))
        elif st=="DUPLICATE_DETECTION_CANDIDATE": c = (255,0,255)
        else: c = (0,0,255)
        hp = [(int(lm["raw_distorted_px_x"]*sx), int(lm["raw_distorted_px_y"]*sy)) for lm in h["landmarks"]]
        for i,j in HAND_CONNECTIONS: cv2.line(thumb, hp[i], hp[j], c, 1)
        wx,wy = hp[0]; short = "L" if a=="ANAT_LEFT" else ("R" if a=="ANAT_RIGHT" else ("DUP" if st=="DUPLICATE_DETECTION_CANDIDATE" else "?"))
        cv2.putText(thumb, f"d{h['hand_index']}:{h['model_handedness_label'][0]}->{short}", (wx-5,wy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1)
        dl = h.get("pose_left_wrist_dist_px"); dr = h.get("pose_right_wrist_dist_px")
        if dl and dr: cv2.putText(thumb, f"dL={dl:.0f} dR={dr:.0f}", (wx-5,wy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.22, c, 1)


def gen_contact_sheets(avf, qa, segs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    flagged = defaultdict(list)
    for (v,s),tag in {("LEFT","S001_P001_GREETING_SLOW_R03"):"low_det_greeting_slow_left",
                      ("RIGHT","S001_P001_LOVE_NORMAL_R03"):"low_det_love_normal_right"}.items():
        fs = [f for f in avf[v] if f["sample_id"]==s]
        if fs:
            for idx in np.linspace(0,len(fs)-1,min(12,len(fs)),dtype=int):
                flagged[tag].append((v,s,fs[idx]["camera_local_pts_s"],"sampled"))
    for v in ["FRONT","LEFT","RIGHT"]:
        for s in PILOT_SAMPLE_IDS:
            if "FAST" in s:
                fs = [f for f in avf[v] if f["sample_id"]==s]
                if fs:
                    for idx in np.linspace(0,len(fs)-1,min(8,len(fs)),dtype=int):
                        flagged[f"fast_{v.lower()}"].append((v,s,fs[idx]["camera_local_pts_s"],"fast"))
    for v in ["FRONT","LEFT","RIGHT"]:
        for f in avf[v]:
            for h in f.get("hands",[]):
                if h["association_status"]=="DUPLICATE_DETECTION_CANDIDATE":
                    flagged["duplicate_candidates"].append((v,f["sample_id"],f["camera_local_pts_s"],"dup"))
                elif h["association_status"]=="AMBIGUOUS":
                    flagged["ambiguous_frames"].append((v,f["sample_id"],f["camera_local_pts_s"],"ambig"))
    for qs in qa:
        for sc in qs.get("identity_swap_candidate_frames",[]):
            flagged["swap_candidates"].append((qs["view"],qs["sample_id"],sc["camera_pts_s"],"swap"))
    # Swap context: ±5 frames
    for qs in qa:
        for sc in qs.get("identity_swap_candidate_frames",[]):
            sf = [f for f in avf[qs["view"]] if f["sample_id"]==qs["sample_id"]]
            for fi,f in enumerate(sf):
                if abs(f["camera_local_pts_s"]-sc["camera_pts_s"])<0.02:
                    for di in range(max(0,fi-5), min(len(sf),fi+6)):
                        flagged["swap_context"].append((qs["view"],qs["sample_id"],sf[di]["camera_local_pts_s"],"swap_ctx"))
                    break
    # Regression: THANK_YOU_NORMAL FRONT 318.65-318.9
    for f in avf["FRONT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_NORMAL_R03" and 318.65<=f["camera_local_pts_s"]<=318.9:
            flagged["regression_thankyou_front"].append(("FRONT",f["sample_id"],f["camera_local_pts_s"],"reg"))
    # Regression: THANK_YOU_FAST LEFT 391.05-391.4
    for f in avf["LEFT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_FAST_R03" and 391.05<=f["camera_local_pts_s"]<=391.4:
            flagged["regression_thankyou_fast_left"].append(("LEFT",f["sample_id"],f["camera_local_pts_s"],"reg"))
    # Strong misassignment check
    for v in ["FRONT","LEFT","RIGHT"]:
        for f in avf[v]:
            for h in f.get("hands",[]):
                if h["association_status"]!="DETECTED": continue
                a = h.get("anatomical_identity")
                if not a: continue
                s = a.replace("ANAT_",""); ot = "right" if s=="LEFT" else "left"
                da = h.get(f"pose_{s.lower()}_wrist_dist_px"); do = h.get(f"pose_{ot}_wrist_dist_px")
                if da and do and da > do*2 and da-do > 50:
                    flagged["strong_misassignment"].append((v,f["sample_id"],f["camera_local_pts_s"],f"da={da:.0f}do={do:.0f}"))

    fls = {}
    for v in ["FRONT","LEFT","RIGHT"]:
        d = {}
        for fd in avf[v]: d[round(fd["camera_local_pts_s"],4)] = fd
        fls[v] = d
    tw,th = 384,216
    for cat, items in flagged.items():
        if not items: continue
        if len(items) > 24: items = [items[i] for i in np.linspace(0,len(items)-1,24,dtype=int)]
        bv = defaultdict(list)
        for v,s,p,r in items: bv[v].append((s,p,r))
        thumbs = []
        for v, flist in bv.items():
            con = av.open(str(VIDEOS[v])); st = con.streams.video[0]; st.thread_type="AUTO"
            tb = float(st.time_base); vw,vh = st.width,st.height
            flist.sort(key=lambda x:x[1]); fi2 = 0
            for vf in con.decode(st):
                if fi2>=len(flist): break
                if vf.pts is None: continue
                cps = float(vf.pts)*tb; tp = flist[fi2][1]
                if cps < tp-0.02: continue
                if abs(cps-tp) <= 0.05:
                    img = vf.to_ndarray(format="bgr24"); t = cv2.resize(img,(tw,th))
                    k = round(tp,4); fd = fls[v].get(k)
                    if fd is None:
                        for d in [0.0001,-0.0001,0.001,-0.001]: fd = fls[v].get(round(tp+d,4));
                        if fd: pass
                    _draw_lm(t, fd, tw, th, vw, vh)
                    cv2.putText(t, f"{v} {_short_sid(flist[fi2][0])}", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
                    cv2.putText(t, f"t={cps:.2f} {flist[fi2][2]}", (5,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,200,255), 1)
                    thumbs.append(t); fi2 += 1
                if cps > tp+0.05: fi2 += 1
            con.close()
        if not thumbs: continue
        cols = min(4,len(thumbs)); rn = math.ceil(len(thumbs)/cols)
        sheet = np.zeros((rn*th, cols*tw, 3), dtype=np.uint8)
        for i,t in enumerate(thumbs):
            r,c = divmod(i,cols); sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
        cv2.imwrite(str(out_dir/f"contact_{cat}.png"), sheet)
    gen = list(out_dir.glob("contact_*.png"))
    print(f"  Contact sheets: {len(gen)} generated")
    return gen


# ═══════════════════════════════════════════════════════════
#  Input verification & Assertions
# ═══════════════════════════════════════════════════════════

def verify_inputs(seg_data, all_segs):
    errors = []
    if seg_data.get("n_segments_expected") != 45: errors.append("n_segments!=45")
    if len(all_segs) != 45: errors.append(f"len={len(all_segs)}")
    nv = [s["sample_id"] for s in all_segs if s["verification_status"] != "MANUAL_VERIFIED"]
    if nv: errors.append(f"non-verified={nv}")
    expected = set()
    for sg in ["GREETING","THANK_YOU","LOVE"]:
        for sp in ["NORMAL","SLOW","FAST"]:
            for r in range(1,6): expected.add(f"S001_P001_{sg}_{sp}_R{r:02d}")
    actual = set(s["sample_id"] for s in all_segs)
    if actual != expected: errors.append(f"ID mismatch")
    for i in range(len(all_segs)-1):
        if all_segs[i]["front_end_s"] > all_segs[i+1]["front_start_s"]: errors.append(f"overlap@{i}")
    if len([s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]) != 9: errors.append("pilot!=9")
    t = seg_data.get("affine_transforms",{})
    for c,ea,eb in [("LEFT",0.9999806597862914,2.310470997893187),("RIGHT",0.9999996430818727,9.388814864164976)]:
        if t.get(c,{}).get("a") != ea: errors.append(f"{c} a mismatch")
        if t.get(c,{}).get("b") != eb: errors.append(f"{c} b mismatch")
    return errors


def run_assertions(avf, qa, sync, all_bdiag):
    """Run all regression and consistency assertions. Returns list of (name, pass/fail, detail)."""
    results = []

    def check(name, cond, detail=""):
        results.append((name, "PASS" if cond else "FAIL", detail))
        return cond

    # Frame counts match v1.2
    check("frame_count_front", sum(1 for f in avf["FRONT"])==1001, f"got {sum(1 for f in avf['FRONT'])}")
    check("frame_count_left", sum(1 for f in avf["LEFT"])==1177)
    check("frame_count_right", sum(1 for f in avf["RIGHT"])==1178)

    # Boundary match v1.2
    for bd in all_bdiag:
        if bd["view"]=="FRONT" and "LOVE_NORMAL" in bd["sample_id"]:
            check("boundary_love_normal_front_start", abs(bd["start_error_ms"]-71.9)<0.1,
                  f"got {bd['start_error_ms']}")

    # accepted_dual_hand_conflict_count == 0
    total_dual = sum(q.get("accepted_dual_hand_conflict_count",0) for q in qa)
    check("accepted_dual_hand_conflict_count==0", total_dual==0, f"got {total_dual}")

    # THANK_YOU_NORMAL FRONT 318.7-318.83: ANAT_RIGHT must be null
    for f in avf["FRONT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_NORMAL_R03":
            t = f["camera_local_pts_s"]
            if 318.7 <= t <= 318.8333:
                check(f"thankyou_normal_front_{t:.4f}_right_null",
                      f["anat_right_landmarks"] is None,
                      f"status={f['anat_right_status']}")

    # THANK_YOU_FAST LEFT 391.222489: ANAT_LEFT must be null
    for f in avf["LEFT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_FAST_R03":
            t = f["camera_local_pts_s"]
            if abs(t - 391.222489) < 0.002:
                check("thankyou_fast_left_391.222_left_null",
                      f["anat_left_landmarks"] is None,
                      f"status={f['anat_left_status']}")

    # 391.255833, 391.289167: NOT explicitly ANAT_LEFT maintained
    for f in avf["LEFT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_FAST_R03":
            t = f["camera_local_pts_s"]
            for target in [391.255833, 391.289167]:
                if abs(t - target) < 0.002:
                    # Should not be ANAT_LEFT with DETECTED status
                    is_anat_left = False
                    for h in f.get("hands",[]):
                        if h["association_status"]=="DETECTED" and h["anatomical_identity"]=="ANAT_LEFT":
                            is_anat_left = True
                    check(f"thankyou_fast_left_{target:.6f}_not_anat_left",
                          not is_anat_left,
                          f"anat_left_status={f['anat_left_status']}")

    # All rates 0-1
    for q in qa:
        for k in ["pose_usable_rate","anat_left_detection_rate","anat_right_detection_rate","both_hands_detected_rate"]:
            v = q.get(k)
            if v is not None:
                check(f"rate_0_1_{q['sample_id']}_{q['view']}_{k}", 0 <= v <= 1, f"{k}={v}")

    # Coverage assertions
    for s in sync:
        check(f"sync_{s['sample_id']}_{s['anatomical_hand']}_3<=2<=ref",
              s["frames_with_3_views"] <= s["frames_with_2plus_views"] <= s["total_reference_frames"])
        if s["front_detected_with_other_view_frames"] is not None and s["front_detected_frames"] is not None:
            check(f"sync_{s['sample_id']}_{s['anatomical_hand']}_fd_other<=fd",
                  s["front_detected_with_other_view_frames"] <= s["front_detected_frames"])
        check(f"sync_{s['sample_id']}_{s['anatomical_hand']}_reuse==0",
              s["reused_match_count"] == 0)
        for rk in ["unconditional_2plus_rate","unconditional_3view_rate","conditional_on_front_detected_rate"]:
            rv = s.get(rk)
            if rv is not None:
                check(f"sync_rate_{s['sample_id']}_{s['anatomical_hand']}_{rk}", 0<=rv<=1, f"{rk}={rv}")

    # DETECTED ↔ landmarks consistency
    for v in ["FRONT","LEFT","RIGHT"]:
        for f in avf[v]:
            for side in ["left","right"]:
                st = f[f"anat_{side}_status"]
                lm = f[f"anat_{side}_landmarks"]
                if st == "DETECTED":
                    check(f"detected_has_landmarks_{v}_{f['camera_local_pts_s']:.4f}_{side}",
                          lm is not None)
                else:
                    check(f"non_detected_null_landmarks_{v}_{f['camera_local_pts_s']:.4f}_{side}",
                          lm is None,
                          f"status={st},lm={'not None' if lm else 'None'}")

    # No [0,0] fallback
    for v in ["FRONT","LEFT","RIGHT"]:
        for f in avf[v]:
            for h in f.get("hands",[]):
                wp = h.get("hand_wrist_px")
                if wp is not None:
                    check(f"no_zero_fallback_{v}_{f['camera_local_pts_s']:.4f}_{h['hand_index']}",
                          wp != [0, 0] and wp != [0.0, 0.0])

    return results


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Step 6A v1.3: MediaPipe 2D Landmark Pilot Extraction — S001_P001\n")

    seg_path = PROC_DIR / "action_segments_v2_1_verified.json"
    with open(seg_path, encoding="utf-8") as f: seg_data = json.load(f)
    all_segs = seg_data["segments"]; transforms = seg_data["affine_transforms"]
    errs = verify_inputs(seg_data, all_segs)
    if errs: print("  INPUT FAILED:", errs); sys.exit(1)
    print("  Input verification: PASS")
    pilot_segs = [s for s in all_segs if s["sample_id"] in PILOT_SAMPLE_IDS]
    assert len(pilot_segs) == 9

    out_base = PROC_DIR / "landmarks_2d_v1_3" / "pilot"
    out_ov = out_base / "overlays"; out_cs = out_base / "low_confidence_contact_sheets"
    for d in [out_base, out_ov, out_cs]: d.mkdir(parents=True, exist_ok=True)

    intrinsics = {}; isizes = {}
    for v in ["FRONT","LEFT","RIGHT"]:
        K,d,sz = load_intrinsics(v); intrinsics[v]=(K,d); isizes[v]=sz
    for v in ["FRONT","LEFT","RIGHT"]:
        c = av.open(str(VIDEOS[v])); s = c.streams.video[0]
        assert isizes[v]==[s.width,s.height]; c.close()
    print("  Intrinsics vs video: MATCH")

    # Pass 1
    print("\n  == PASS 1: no handedness ==")
    p1f = {}
    for v in ["FRONT","LEFT","RIGHT"]:
        t = transforms[v]; K,d = intrinsics[v]
        fr,_,_ = extract_view(v, pilot_segs, t["a"], t["b"], K, d, 0.0, None)
        p1f[v] = fr
    om, mc = compute_observed_mapping([f for vs in p1f.values() for f in vs])
    print(f"\n  Observed mapping: {om}")

    # Pass 2
    print("\n  == PASS 2: with handedness + bidirectional ==")
    avf = {}; abd = []; apm = []
    for v in ["FRONT","LEFT","RIGHT"]:
        t = transforms[v]; K,d = intrinsics[v]
        fr, bd, pm = extract_view(v, pilot_segs, t["a"], t["b"], K, d,
                                  THRESHOLDS["association_handedness_weight_pass2"], om)
        avf[v] = fr; abd.extend(bd); apm.extend(pm)

    # Save JSONs
    for v in ["FRONT","LEFT","RIGHT"]:
        vf = out_base / f"landmarks_2d_{v.lower()}.json"
        with open(vf,"w",encoding="utf-8") as f:
            json.dump({"version":"v1.3_pilot","view":v,"n_frames":len(avf[v]),
                       "image_size":isizes[v],"original_video_mirrored":False,"frames":avf[v]},
                      f, indent=1, ensure_ascii=False)
        print(f"  Saved: {vf.name} ({len(avf[v])} frames)")

    # Boundary
    bp = out_base / "boundary_frame_diagnostics.csv"
    with open(bp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=abd[0].keys()); w.writeheader(); w.writerows(abd)
    print(f"  Saved: {bp.name}")

    # QA
    aqa = []
    for v in ["FRONT","LEFT","RIGHT"]: aqa.extend(compute_qa_stats(avf[v], v, pilot_segs))
    dp = out_base / "detection_quality.csv"
    cq = [{k:v2 for k,v2 in q.items() if k!="identity_swap_candidate_frames"} for q in aqa]
    with open(dp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cq[0].keys()); w.writeheader(); w.writerows(cq)
    print(f"  Saved: {dp.name}")

    # Hand identity diagnostics
    hid = []
    hid_fields = [
        "sample_id","view","camera_pts_s","hand_index","anatomical_identity","association_status",
        "model_handedness_label","model_handedness_score","hand_wrist_px_x","hand_wrist_px_y",
        "pose_left_wrist_dist_px","pose_right_wrist_dist_px",
        "normalized_selected_distance","cost_to_left","cost_to_right",
        "distance_ratio_left_vs_right","inter_detection_distance_px",
        "normalized_inter_detection_distance","best_candidate_name","best_candidate_score",
        "second_best_candidate_name","second_best_candidate_score",
        "assignment_cost_margin","single_vs_dual_assignment_margin",
        "continuity_cost","elbow_direction_cost","handedness_auxiliary_cost","final_cost",
        "assignment_reason","rejected_alternative_reason","rejection_rule","duplicate_rule_result",
        "shoulder_width_px","gate_px",
    ]
    for v in ["FRONT","LEFT","RIGHT"]:
        for fd in avf[v]:
            for h in fd.get("hands",[]):
                wp = h.get("hand_wrist_px")
                row = {
                    "sample_id": fd["sample_id"], "view": v,
                    "camera_pts_s": fd["camera_local_pts_s"],
                    "hand_wrist_px_x": wp[0] if wp else None,
                    "hand_wrist_px_y": wp[1] if wp else None,
                }
                for k in hid_fields:
                    if k not in row: row[k] = h.get(k)
                hid.append(row)
    hp = out_base / "hand_identity_diagnostics.csv"
    with open(hp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=hid_fields); w.writeheader(); w.writerows(hid)
    print(f"  Saved: {hp.name}")

    # Sync coverage
    sync = compute_sync_coverage(avf, pilot_segs)
    sp = out_base / "synchronized_view_coverage.csv"
    with open(sp,"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sync[0].keys()); w.writeheader(); w.writerows(sync)
    print(f"  Saved: {sp.name}")

    # Overlays
    print("\n  Generating overlays...")
    for v in ["FRONT","LEFT","RIGHT"]:
        t = transforms[v]
        make_overlay_video(v, avf[v], pilot_segs, t["a"], t["b"], out_ov/f"overlay_pilot_{v.lower()}.mp4", aqa)

    # Contact sheets
    print()
    gen_contact_sheets(avf, aqa, pilot_segs, out_cs)

    # Handedness verification
    hv, hvc = compute_observed_mapping([f for vs in avf.values() for f in vs])

    # SHA-256
    ih2 = {"action_segments": sha256_file(seg_path), "pose_model": sha256_file(POSE_MODEL),
           "hand_model": sha256_file(HAND_MODEL)}
    for v in ["FRONT","LEFT","RIGHT"]:
        ih2[f"video_{v}"] = sha256_file(VIDEOS[v])
        ih2[f"intrinsics_{v}"] = sha256_file(C1_INTRINSICS[v])

    # Manifest
    manifest = {
        "step": "6A_pilot_v1.3", "date": datetime.datetime.now().isoformat(),
        "python_version": sys.version, "mediapipe_version": mp.__version__,
        "pyav_version": av.__version__, "opencv_version": cv2.__version__,
        "pose_model": {"path": str(POSE_MODEL), "sha256": ih2["pose_model"]},
        "hand_model": {"path": str(HAND_MODEL), "sha256": ih2["hand_model"]},
        "thresholds": THRESHOLDS,
        "v1_defaults_note": {"all_thresholds": "0.5 (mediapipe default)"},
        "two_pass_association": {
            "pass1": "handedness_weight=0", "pass2": "handedness_weight=0.1 + bidirectional refinement",
            "observed_mapping": om,
            "mapping_counts": {f"{k[0]}->{k[1]}":v for k,v in mc.items()},
        },
        "input_sha256": ih2, "pilot_sample_ids": PILOT_SAMPLE_IDS,
        "original_videos_mirrored": False, "pts_missing": len(apm),
        "null_policy": "No 0-fill, no interpolation, no forced assignment",
        "handedness_verification": {"mapping": hv, "counts": {f"{k[0]}->{k[1]}":v for k,v in hvc.items()}},
    }
    mp2 = out_base / "run_manifest.json"
    with open(mp2,"w",encoding="utf-8") as f: json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {mp2.name}")

    # ══ ASSERTIONS ══
    print("\n  " + "="*60)
    print("  REGRESSION ASSERTIONS")
    print("  " + "="*60)

    ar = run_assertions(avf, aqa, sync, abd)
    n_pass = sum(1 for _,s,_ in ar if s=="PASS")
    n_fail = sum(1 for _,s,_ in ar if s=="FAIL")
    # Print only failures and key passes
    for name, status, detail in ar:
        if status == "FAIL":
            print(f"    FAIL: {name} — {detail}")
    print(f"\n  Assertions: {n_pass} PASS, {n_fail} FAIL out of {len(ar)}")

    if n_fail > 0:
        print(f"\n  STATUS: ASSERTION_FAILURE ({n_fail} failures)")
        sys.exit(1)

    # ══ QA SUMMARY ══
    print("\n  " + "="*60)
    print("  QA Summary (v1.3)")
    print("  " + "="*60)

    # Conflict counts
    total_same = sum(q.get("same_nearest_wrist_conflict_frames",0) for q in aqa)
    total_close = sum(q.get("close_same_wrist_conflict_frames",0) for q in aqa)
    total_dual_conf = sum(q.get("accepted_dual_hand_conflict_count",0) for q in aqa)
    total_dup = sum(q.get("duplicate_detection_candidate_frame_count",0) for q in aqa)
    total_ambig = sum(q.get("ambiguous_frame_count",0) for q in aqa)
    total_swap_cmp = sum(q.get("identity_swap_comparisons_executed",0) for q in aqa)
    total_swap = sum(q.get("identity_swap_candidate_count",0) for q in aqa)

    print(f"\n  same_nearest_wrist_conflict: {total_same}")
    print(f"  close_same_wrist_conflict: {total_close}")
    print(f"  accepted_dual_hand_conflict: {total_dual_conf}")
    print(f"  DUPLICATE frames: {total_dup}")
    print(f"  AMBIGUOUS frames: {total_ambig}")
    print(f"  swap comparisons: {total_swap_cmp}")
    print(f"  swap candidates: {total_swap}")

    # Regression cases
    print(f"\n  THANK_YOU_NORMAL FRONT 318.7-318.83:")
    for f in avf["FRONT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_NORMAL_R03" and 318.65<=f["camera_local_pts_s"]<=318.9:
            for h in f.get("hands",[]):
                print(f"    t={f['camera_local_pts_s']:.4f} hi={h['hand_index']} anat={h.get('anatomical_identity')} "
                      f"status={h['association_status']} dL={h.get('pose_left_wrist_dist_px')} dR={h.get('pose_right_wrist_dist_px')}")

    print(f"\n  THANK_YOU_FAST LEFT 391.05-391.4:")
    for f in avf["LEFT"]:
        if f["sample_id"]=="S001_P001_THANK_YOU_FAST_R03" and 391.05<=f["camera_local_pts_s"]<=391.4:
            for h in f.get("hands",[]):
                print(f"    t={f['camera_local_pts_s']:.6f} hi={h['hand_index']} anat={h.get('anatomical_identity')} "
                      f"status={h['association_status']} dL={h.get('pose_left_wrist_dist_px')} dR={h.get('pose_right_wrist_dist_px')}")

    for v in ["FRONT","LEFT","RIGHT"]:
        vq = [q for q in aqa if q["view"]==v]
        print(f"\n  {v}:")
        for q in vq:
            print(f"    {_short_sid(q['sample_id'])}: {q['n_frames']}f "
                  f"pose={q['pose_usable_rate']:.0%} L={q['anat_left_detection_rate']:.0%} "
                  f"R={q['anat_right_detection_rate']:.0%} both={q['both_hands_detected_rate']:.0%} "
                  f"ambig={q['ambiguous_frame_count']} dup={q['duplicate_detection_candidate_frame_count']} "
                  f"swap={q['identity_swap_candidate_count']}({q['identity_swap_comparisons_executed']}cmp) "
                  f"conflict={q['close_same_wrist_conflict_frames']}/{q['same_nearest_wrist_conflict_frames']}")

    # Coverage
    print(f"\n  Synchronized coverage (diagnostic — sufficiency requires triangulation verification):")
    for s in sync:
        print(f"    {_short_sid(s['sample_id'])} {s['anatomical_hand']}: ref={s['total_reference_frames']} "
              f"fd={s['front_detected_frames']} 2+v={s['unconditional_2plus_rate']:.0%} "
              f"3v={s['unconditional_3view_rate']:.0%} "
              f"cond={s['conditional_on_front_detected_rate']:.0%}" if s['conditional_on_front_detected_rate'] is not None else f"cond=N/A",
              f"reuse={s['reused_match_count']}")

    print(f"\n  Status: REQUIRES_HUMAN_VISUAL_REVIEW")
    print(f"\n  Files:")
    for p in sorted(out_base.rglob("*")):
        if p.is_file():
            sk = p.stat().st_size/1024
            print(f"    {p.relative_to(out_base)}: {sk/1024:.1f} MB" if sk>1024 else f"    {p.relative_to(out_base)}: {sk:.1f} KB")


if __name__ == "__main__":
    main()
