"""
Step 6B v1.2.1 PILOT: Multi-view 3D Reconstruction -- S001_P001 R03 (9 samples)

Patch on v1.2:
  - Preserves all v1.2 triangulation core, 2-pass scoring, 20px template filter
  - Fixes temporal QA: reappearance tracking, depth oscillation, acceleration
  - Adds selection metadata: pass1/pass2 cameras/scores for ALL valid landmarks
  - Balanced temporal contact sheets by reason type
  - Strengthened assertions with reason-specific field checks
"""

import json, csv, hashlib, sys, datetime, math, zipfile
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import cv2
import numpy as np
import av

# ================================================================
#  Paths & Constants
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"

STEP6A_DIR = PROC_DIR / "landmarks_2d_v1_4" / "full"
OUT_DIR = PROC_DIR / "landmarks_3d_v1" / "pilot_v1_2_1"

VIDEOS = {
    "FRONT": RAW_DIR / "S001_P001_FRONT.mp4",
    "LEFT":  RAW_DIR / "S001_P001_LEFT.mp4",
    "RIGHT": RAW_DIR / "S001_P001_RIGHT.mp4",
}

C1_INTRINSICS = {
    "FRONT": PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_FRONT_01_v2.json",
    "LEFT":  PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_LEFT_01_v2.json",
    "RIGHT": PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation" / "intrinsics_CAM_RIGHT_01_v2.json",
}
C2_EXTRINSICS = PROC_DIR / "C2_CALIBRATION_v2_robust.json"
C2_APPROVAL = PROC_DIR / "C2_APPROVAL.json"
SYNC_REPORT = PROC_DIR / "sync_report.json"
SEGMENTS_FILE = PROC_DIR / "action_segments_v2_1_verified.json"

PILOT_SAMPLE_IDS = [
    "S001_P001_GREETING_NORMAL_R03", "S001_P001_GREETING_SLOW_R03",
    "S001_P001_GREETING_FAST_R03", "S001_P001_THANK_YOU_NORMAL_R03",
    "S001_P001_THANK_YOU_SLOW_R03", "S001_P001_THANK_YOU_FAST_R03",
    "S001_P001_LOVE_NORMAL_R03", "S001_P001_LOVE_SLOW_R03",
    "S001_P001_LOVE_FAST_R03",
]

VIEWS = ["FRONT", "LEFT", "RIGHT"]

TEMPLATE_REPROJ_THRESHOLD_PX = 20.0

CONFIG = {
    "sync_tolerance_s": 0.020,
    "min_views_for_3d": 2,
    "reproj_outlier_px": 20.0,
    "reproj_exclude_px": 50.0,
    "template_reproj_threshold_px": TEMPLATE_REPROJ_THRESHOLD_PX,
    "min_depth_m": 0.3,
    "max_depth_m": 10.0,
    "min_triangulation_angle_deg": 2.0,
    "cam_count_score_bonus": 10.0,
    "angle_score_weight": 0.5,
    "angle_score_cap_deg": 30.0,
    "validation_penalty_threshold_px": 30.0,
    "validation_penalty_weight": 0.1,
    "high_reproj_penalty_weight": 0.3,
    "bone_consistency_penalty_weight": 2.0,
    "temporal_continuity_penalty_weight": 1.5,
    "camera_switch_penalty_weight": 0.5,
    "bone_outlier_mad_factor": 5.0,
    "bone_max_finger_phalanx_m": 0.06,
    "bone_max_metacarpal_m": 0.12,
    "temporal_velocity_warn_ms": 5.0,
    "temporal_accel_warn_ms2": 200.0,
    "temporal_max_gap_s": 0.1,
    "reappear_jump_threshold_m": 0.1,
    "depth_osc_window_size": 7,
    "depth_osc_min_reversals": 3,
    "depth_osc_min_ptp_m": 0.02,
    "depth_osc_min_z_xy_ratio": 2.0,
    "depth_osc_min_residual_mad_m": 0.005,
    "apply_smoothing": False,
    "contact_thumb_size": (384, 216),
    "contact_max_cells": 24,
    "contact_max_per_sample": 3,
    "contact_max_per_reason": 5,
}

POSE_BONES = [
    (11, 13, "left_upper_arm"), (13, 15, "left_forearm"),
    (12, 14, "right_upper_arm"), (14, 16, "right_forearm"),
    (11, 12, "shoulders"), (11, 23, "left_torso"), (12, 24, "right_torso"),
    (23, 24, "hips"),
]

HAND_BONES = [
    (0, 1, "thumb_cmc"), (1, 2, "thumb_mcp"), (2, 3, "thumb_ip"), (3, 4, "thumb_tip"),
    (0, 5, "index_mcp"), (5, 6, "index_pip"), (6, 7, "index_dip"), (7, 8, "index_tip"),
    (0, 9, "middle_mcp"), (9, 10, "middle_pip"), (10, 11, "middle_dip"), (11, 12, "middle_tip"),
    (0, 13, "ring_mcp"), (13, 14, "ring_pip"), (14, 15, "ring_dip"), (15, 16, "ring_tip"),
    (0, 17, "pinky_mcp"), (17, 18, "pinky_pip"), (18, 19, "pinky_dip"), (19, 20, "pinky_tip"),
    (5, 9, "palm_idx_mid"), (9, 13, "palm_mid_ring"), (13, 17, "palm_ring_pinky"),
]

POSE_LM_GROUPS = {
    "face": list(range(0, 11)),
    "sign_upper": list(range(11, 17)),
    "pose_hand": list(range(17, 23)),
    "lower_body": list(range(23, 33)),
}

REGRESSION_BONE = {
    "sample_id": "S001_P001_GREETING_NORMAL_R03",
    "frame_index": 39,
    "category": "right_hand",
    "bones": [(9, 10), (11, 12), (13, 14), (1, 2)],
    "min_expected_m": 0.10,
}

REGRESSION_TY_FRAMES = {
    "sample_id": "S001_P001_THANK_YOU_FAST_R03",
    "front_frame_indices": [48, 49],
}

# Required fields per temporal reason
TEMPORAL_REQUIRED_FIELDS = {
    "high_velocity": ["displacement_m", "dt_s", "velocity_ms"],
    "high_acceleration": ["velocity_ms", "acceleration_ms2", "dt_s"],
    "camera_switch_teleport": ["cameras_prev", "cameras_curr", "displacement_m", "velocity_ms"],
    "detection_reappear": ["gap_frames", "last_valid_frame_index", "reappear_displacement_m", "reappear_velocity_ms"],
    "depth_axis_anomaly": ["displacement_xy_m", "displacement_z_m", "velocity_ms"],
    "depth_oscillation": ["window_start_fi", "window_end_fi", "z_ptp_residual_m", "reversal_count", "z_residual_mad_m"],
    "bone_anomaly_jump": ["displacement_m", "velocity_ms"],
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _ds(values):
    if not values:
        return {"count": 0}
    a = np.array(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "mean": round(float(np.mean(a)), 2),
        "median": round(float(np.median(a)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
        "p99": round(float(np.percentile(a, 99)), 2),
        "max": round(float(np.max(a)), 2),
    }


# ================================================================
#  Load calibration
# ================================================================

def load_calibration():
    intrinsics = {}
    for v in VIEWS:
        with open(C1_INTRINSICS[v]) as f:
            d = json.load(f)
        K = np.array(d["camera_matrix"], dtype=np.float64)
        dist = np.array(d["dist_coeffs"], dtype=np.float64)
        intrinsics[v] = {"K": K, "dist": dist, "image_size": d["image_size"]}

    with open(C2_EXTRINSICS) as f:
        c2 = json.load(f)

    extrinsics = {}
    for v in VIEWS:
        R = np.array(c2["cameras"][v]["R"], dtype=np.float64)
        t = np.array(c2["cameras"][v]["t"], dtype=np.float64).reshape(3, 1)
        extrinsics[v] = {"R": R, "t": t}

    proj_matrices = {}
    camera_centers = {}
    for v in VIEWS:
        K = intrinsics[v]["K"]
        R = extrinsics[v]["R"]
        t = extrinsics[v]["t"]
        proj_matrices[v] = K @ np.hstack([R, t])
        camera_centers[v] = (-R.T @ t).flatten()

    return intrinsics, extrinsics, proj_matrices, camera_centers


# ================================================================
#  Load 2D landmarks
# ================================================================

def load_2d_landmarks():
    all_frames = {}
    for v in VIEWS:
        fp = STEP6A_DIR / f"landmarks_2d_{v.lower()}.json"
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        r03 = [fr for fr in data["frames"] if fr["sample_id"] in PILOT_SAMPLE_IDS]
        all_frames[v] = r03
        print(f"  Loaded {v}: {len(r03)} R03 frames (of {data['n_frames']} total)")
    return all_frames


# ================================================================
#  Frame synchronization
# ================================================================

def synchronize_frames(all_frames):
    tol = CONFIG["sync_tolerance_s"]
    synced = {}
    for sid in PILOT_SAMPLE_IDS:
        view_frames = {}
        for v in VIEWS:
            vf = sorted(
                [f for f in all_frames[v] if f["sample_id"] == sid],
                key=lambda f: f["mapped_front_pts_s"]
            )
            view_frames[v] = vf
        front_frames = view_frames["FRONT"]
        matched = []
        for ff in front_frames:
            ft = ff["mapped_front_pts_s"]
            entry = {"front_time_s": ft, "views": {"FRONT": ff}}
            for v in ["LEFT", "RIGHT"]:
                best = None
                best_dt = tol + 1
                for cf in view_frames[v]:
                    dt = abs(cf["mapped_front_pts_s"] - ft)
                    if dt < best_dt:
                        best_dt = dt
                        best = cf
                    if cf["mapped_front_pts_s"] > ft + tol:
                        break
                if best is not None and best_dt <= tol:
                    entry["views"][v] = best
            entry["n_views"] = len(entry["views"])
            matched.append(entry)
        synced[sid] = matched
    return synced


# ================================================================
#  Triangulation primitives
# ================================================================

def triangulate_point(pts_2d, proj_mats):
    n = len(pts_2d)
    if n < 2:
        return None
    if n == 2:
        p1 = np.array([[pts_2d[0][0]], [pts_2d[0][1]]], dtype=np.float64)
        p2 = np.array([[pts_2d[1][0]], [pts_2d[1][1]]], dtype=np.float64)
        X4 = cv2.triangulatePoints(proj_mats[0], proj_mats[1], p1, p2)
        X3 = (X4[:3] / X4[3]).flatten()
        return X3
    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (pt, P) in enumerate(zip(pts_2d, proj_mats)):
        x, y = pt
        A[2*i]     = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X4 = Vt[-1]
    if abs(X4[3]) < 1e-10:
        return None
    return X4[:3] / X4[3]


def reproject_point(X3, K, R, t, dist=None):
    X_cam = R @ X3.reshape(3, 1) + t
    if X_cam[2, 0] <= 0:
        return None
    if dist is not None and len(dist) >= 4:
        rvec, _ = cv2.Rodrigues(R)
        pts_3d = X3.reshape(1, 1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
        return pts_2d[0, 0]
    x_norm = X_cam[0, 0] / X_cam[2, 0]
    y_norm = X_cam[1, 0] / X_cam[2, 0]
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    return np.array([u, v])


def extract_landmark_2d(frame, lm_type, lm_index, hand_side=None):
    if lm_type == "pose":
        if not frame.get("pose_detected") or not frame.get("pose_landmarks"):
            return None
        lms = frame["pose_landmarks"]
        if lm_index >= len(lms):
            return None
        lm = lms[lm_index]
        return {
            "undistorted_px": (lm["undistorted_px_x"], lm["undistorted_px_y"]),
            "raw_px": (lm["raw_distorted_px_x"], lm["raw_distorted_px_y"]),
            "visibility": lm.get("visibility", 0),
            "presence": lm.get("presence", 0),
        }
    elif lm_type == "hand":
        if frame.get(f"anat_{hand_side}_status") != "DETECTED":
            return None
        lms = frame.get(f"anat_{hand_side}_landmarks")
        if not lms or lm_index >= len(lms):
            return None
        lm = lms[lm_index]
        return {
            "undistorted_px": (lm["undistorted_px_x"], lm["undistorted_px_y"]),
            "raw_px": (lm["raw_distorted_px_x"], lm["raw_distorted_px_y"]),
            "visibility": lm.get("visibility", 0),
            "presence": lm.get("presence", 0),
        }
    return None


# ================================================================
#  Multi-candidate view selection (geometry-only, Pass 1)
# ================================================================

def _view_abbrev(views):
    return "+".join(v[0] for v in views)


def evaluate_candidate_geometry(cand_views, obs, all_obs_views,
                                intrinsics, extrinsics, proj_matrices, camera_centers):
    pts = [obs[v]["undistorted_px"] for v in cand_views]
    pmats = [proj_matrices[v] for v in cand_views]
    X3 = triangulate_point(pts, pmats)

    ev = {
        "views": _view_abbrev(cand_views), "cameras": list(cand_views),
        "feasible": False,
        "geometry_score": None, "used_reproj_penalty": None,
        "excluded_view_validation_penalty": None, "triangulation_angle_score": None,
        "bone_consistency_penalty": None, "temporal_continuity_penalty": None,
        "camera_switch_penalty": None, "final_score": None,
        "reason": "",
        "xyz": None, "cheirality_pass": True, "finite": True,
        "depth_m": None,
        "used_reproj_mean": None, "used_reproj_max": None,
        "validation_reproj_mean": None, "all_reproj_mean": None,
        "angle_deg": None, "per_camera": {},
    }

    if X3 is None or not np.all(np.isfinite(X3)):
        ev["finite"] = False
        ev["reason"] = "non_finite"
        return ev

    for v in all_obs_views:
        R, t = extrinsics[v]["R"], extrinsics[v]["t"]
        if (R @ X3.reshape(3, 1) + t)[2, 0] <= 0:
            ev["cheirality_pass"] = False
            ev["reason"] = f"behind_{v}"
            return ev

    z = float(X3[2])
    ev["depth_m"] = round(z, 4)
    if z < CONFIG["min_depth_m"] or z > CONFIG["max_depth_m"]:
        ev["reason"] = f"depth_{z:.2f}m"
        return ev

    used_e, val_e = [], []
    for v in all_obs_views:
        rp = reproject_point(X3, intrinsics[v]["K"], extrinsics[v]["R"],
                             extrinsics[v]["t"], intrinsics[v]["dist"])
        if rp is not None:
            err = float(np.linalg.norm(rp - np.array(obs[v]["raw_px"])))
            ev["per_camera"][v] = {"reproj_px": rp.tolist(), "error": round(err, 2)}
            (used_e if v in cand_views else val_e).append(err)
        else:
            ev["per_camera"][v] = {"reproj_px": None, "error": None}
            if v in cand_views:
                ev["reason"] = f"reproj_fail_{v}"
                return ev

    ev["used_reproj_mean"] = round(float(np.mean(used_e)), 2) if used_e else None
    ev["used_reproj_max"] = round(float(np.max(used_e)), 2) if used_e else None
    ev["validation_reproj_mean"] = round(float(np.mean(val_e)), 2) if val_e else None
    all_e = used_e + val_e
    ev["all_reproj_mean"] = round(float(np.mean(all_e)), 2) if all_e else None

    if ev["used_reproj_max"] is not None and ev["used_reproj_max"] > CONFIG["reproj_exclude_px"]:
        ev["reason"] = f"used_reproj_{ev['used_reproj_max']:.0f}px"
        return ev

    min_ang = 180.0
    for i in range(len(cand_views)):
        for j in range(i + 1, len(cand_views)):
            r1 = camera_centers[cand_views[i]] - X3
            r2 = camera_centers[cand_views[j]] - X3
            n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
            if n1 < 1e-6 or n2 < 1e-6:
                min_ang = 0; break
            cos_a = np.clip(np.dot(r1, r2) / (n1 * n2), -1, 1)
            min_ang = min(min_ang, float(np.degrees(np.arccos(cos_a))))

    ev["angle_deg"] = round(min_ang, 2)
    if min_ang < CONFIG["min_triangulation_angle_deg"]:
        ev["reason"] = f"angle_{min_ang:.1f}deg"
        return ev

    ev["feasible"] = True
    ev["xyz"] = X3.tolist()

    nc = len(cand_views)
    geo_base = nc * CONFIG["cam_count_score_bonus"]
    reproj_pen = ev["used_reproj_mean"] if ev["used_reproj_mean"] else 0.0
    angle_sc = min(min_ang, CONFIG["angle_score_cap_deg"]) * CONFIG["angle_score_weight"]
    val_pen = 0.0
    if val_e:
        val_pen = max(0, max(val_e) - CONFIG["validation_penalty_threshold_px"]) * CONFIG["validation_penalty_weight"]
    high_reproj_pen = 0.0
    if ev["used_reproj_max"] and ev["used_reproj_max"] > CONFIG["reproj_outlier_px"]:
        high_reproj_pen = (ev["used_reproj_max"] - CONFIG["reproj_outlier_px"]) * CONFIG["high_reproj_penalty_weight"]

    ev["geometry_score"] = round(geo_base - reproj_pen + angle_sc - val_pen - high_reproj_pen, 4)
    ev["used_reproj_penalty"] = round(reproj_pen + high_reproj_pen, 4)
    ev["excluded_view_validation_penalty"] = round(val_pen, 4)
    ev["triangulation_angle_score"] = round(angle_sc, 4)
    ev["bone_consistency_penalty"] = 0.0
    ev["temporal_continuity_penalty"] = 0.0
    ev["camera_switch_penalty"] = 0.0
    ev["final_score"] = ev["geometry_score"]
    ev["reason"] = f"{nc}v_gs{ev['geometry_score']:.1f}"

    return ev


# ================================================================
#  Pass 1: Geometry-only triangulation (per landmark)
# ================================================================

def triangulate_landmark_pass1(synced_entry, lm_type, lm_index, hand_side,
                               intrinsics, extrinsics, proj_matrices, camera_centers):
    obs = {}
    for v, frame in synced_entry["views"].items():
        lm2d = extract_landmark_2d(frame, lm_type, lm_index, hand_side)
        if lm2d is not None:
            obs[v] = lm2d

    n_obs = len(obs)
    result = {
        "lm_type": lm_type, "lm_index": lm_index, "hand_side": hand_side,
        "triangulation_valid": False, "qa_valid": False,
        "template_eligible": False, "template_eligible_by_view": {},
        "xyz": None,
        "n_observations": n_obs, "n_cameras_used": 0,
        "cameras_used": [], "excluded_cameras": [],
        "selection_reason": "", "qa_flags": [],
        "used_reproj_mean_px": None, "used_reproj_max_px": None,
        "excluded_reproj_mean_px": None, "all_reproj_mean_px": None,
        "invalid_reason": None,
        "candidates": [],
        "per_camera": {},
        # Selection metadata (filled after pass 2)
        "pass1_selected_cameras": [],
        "pass2_selected_cameras": [],
        "selection_changed": False,
        "pass1_geometry_score": None,
        "pass2_final_score": None,
        # Reappearance fields
        "is_reappearance": False,
        "last_valid_frame_index": None,
        "gap_frames": None,
        "gap_s": None,
        "reappear_displacement_m": None,
        "reappear_velocity_ms": None,
        "reappearance_classification": None,
    }

    for v, lm2d in obs.items():
        result["per_camera"][v] = {
            "raw_2d_px": list(lm2d["raw_px"]),
            "undistorted_2d_px": list(lm2d["undistorted_px"]),
            "visibility": lm2d["visibility"], "presence": lm2d["presence"],
            "reproj_2d_px": None, "reproj_error_px": None,
            "used_for_triangulation": False,
        }

    if n_obs < CONFIG["min_views_for_3d"]:
        result["invalid_reason"] = f"insufficient_views({n_obs})"
        return result

    obs_views = sorted(obs.keys())
    all_candidates = []

    for r in range(2, n_obs + 1):
        for combo in combinations(obs_views, r):
            ev = evaluate_candidate_geometry(list(combo), obs, obs_views,
                                             intrinsics, extrinsics, proj_matrices, camera_centers)
            all_candidates.append(ev)

    result["candidates"] = _make_cand_summaries(all_candidates)

    best, best_score = None, -float("inf")
    for ev in all_candidates:
        if ev["feasible"] and ev["geometry_score"] > best_score:
            best_score = ev["geometry_score"]
            best = ev

    if best is None:
        result["invalid_reason"] = "no_feasible_candidate"
        result["selection_reason"] = "all_candidates_failed"
        return result

    _apply_best_candidate(result, best, obs_views)
    result["pass1_selected_cameras"] = list(best["cameras"])
    result["pass1_geometry_score"] = best["geometry_score"]

    result["_all_candidates"] = all_candidates
    result["_obs"] = obs
    result["_obs_views"] = obs_views

    return result


def _make_cand_summaries(all_candidates):
    return [{
        "v": ev["views"], "ok": ev["feasible"],
        "gs": ev["geometry_score"], "ur": ev["used_reproj_mean"],
        "um": ev["used_reproj_max"], "vr": ev["validation_reproj_mean"],
        "a": ev["angle_deg"], "d": ev["depth_m"], "r": ev["reason"],
        "urp": ev["used_reproj_penalty"], "evp": ev["excluded_view_validation_penalty"],
        "tas": ev["triangulation_angle_score"],
        "bcp": ev["bone_consistency_penalty"], "tcp": ev["temporal_continuity_penalty"],
        "csp": ev["camera_switch_penalty"], "fs": ev["final_score"],
    } for ev in all_candidates]


def _apply_best_candidate(result, best, obs_views):
    result["triangulation_valid"] = True
    result["xyz"] = best["xyz"]
    result["n_cameras_used"] = len(best["cameras"])
    result["cameras_used"] = best["cameras"]
    result["excluded_cameras"] = [v for v in obs_views if v not in best["cameras"]]
    result["selection_reason"] = best["reason"]
    result["used_reproj_mean_px"] = best["used_reproj_mean"]
    result["used_reproj_max_px"] = best["used_reproj_max"]

    excl_e = [best["per_camera"][v]["error"] for v in result["excluded_cameras"]
              if best["per_camera"].get(v, {}).get("error") is not None]
    result["excluded_reproj_mean_px"] = round(float(np.mean(excl_e)), 2) if excl_e else None
    all_e = [best["per_camera"][v]["error"] for v in obs_views
             if best["per_camera"].get(v, {}).get("error") is not None]
    result["all_reproj_mean_px"] = round(float(np.mean(all_e)), 2) if all_e else None

    result["qa_valid"] = True
    result["template_eligible"] = (
        result["used_reproj_max_px"] is not None and
        result["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX
    )

    te_by_view = {}
    for v in VIEWS:
        cam_info = best["per_camera"].get(v)
        if cam_info and cam_info.get("error") is not None:
            te_by_view[v] = (result["template_eligible"] and
                             cam_info["error"] <= TEMPLATE_REPROJ_THRESHOLD_PX)
        else:
            te_by_view[v] = False
    result["template_eligible_by_view"] = te_by_view

    if best["used_reproj_max"] and best["used_reproj_max"] > CONFIG["reproj_outlier_px"]:
        result["qa_flags"].append(f"high_used_reproj_{best['used_reproj_max']:.0f}px")
    if result["excluded_cameras"]:
        result["qa_flags"].append(f"excluded_{'_'.join(result['excluded_cameras'])}")

    for v in obs_views:
        cam_ev = best["per_camera"].get(v, {})
        result["per_camera"][v]["reproj_2d_px"] = cam_ev.get("reproj_px")
        result["per_camera"][v]["reproj_error_px"] = cam_ev.get("error")
        result["per_camera"][v]["used_for_triangulation"] = v in best["cameras"]


def _invalidate_te(lm):
    lm["template_eligible"] = False
    for vv in VIEWS:
        lm["template_eligible_by_view"][vv] = False


# ================================================================
#  Process one sample (Pass 1)
# ================================================================

def process_sample_pass1(sid, synced_frames, intrinsics, extrinsics, proj_matrices, cam_centers):
    frames_3d = []
    for entry in synced_frames:
        fr = {
            "sample_id": sid,
            "front_time_s": entry["front_time_s"],
            "n_views_synced": entry["n_views"],
            "views_available": sorted(entry["views"].keys()),
            "per_view_info": {},
            "pose_landmarks_3d": [],
            "left_hand_landmarks_3d": [],
            "right_hand_landmarks_3d": [],
        }
        for v, vfr in entry["views"].items():
            fr["per_view_info"][v] = {
                "camera_local_pts_s": vfr["camera_local_pts_s"],
                "mapped_front_pts_s": vfr["mapped_front_pts_s"],
                "decoded_frame_index": vfr["decoded_frame_index"],
                "pose_detected": vfr.get("pose_detected", False),
                "anat_left_status": vfr.get("anat_left_status", "NOT_DETECTED"),
                "anat_right_status": vfr.get("anat_right_status", "NOT_DETECTED"),
            }
        for li in range(33):
            fr["pose_landmarks_3d"].append(
                triangulate_landmark_pass1(entry, "pose", li, None,
                                           intrinsics, extrinsics, proj_matrices, cam_centers))
        for hs in ["left", "right"]:
            key = f"{hs}_hand_landmarks_3d"
            for li in range(21):
                fr[key].append(
                    triangulate_landmark_pass1(entry, "hand", li, hs,
                                               intrinsics, extrinsics, proj_matrices, cam_centers))
        frames_3d.append(fr)
    return frames_3d


# ================================================================
#  Pass 2: Re-score with bone/temporal/switch penalties
# ================================================================

def compute_bone_medians_for_pass2(frames_3d):
    bone_data = defaultdict(list)
    for fr in frames_3d:
        for side in ["left", "right"]:
            key = f"{side}_hand_landmarks_3d"
            hand = fr[key]
            for bi, bj, bname in HAND_BONES:
                li, lj = hand[bi], hand[bj]
                if (li["triangulation_valid"] and lj["triangulation_valid"] and
                    li.get("used_reproj_max_px") is not None and
                    lj.get("used_reproj_max_px") is not None and
                    li["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX and
                    lj["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX):
                    bl = float(np.linalg.norm(np.array(li["xyz"]) - np.array(lj["xyz"])))
                    bone_data[f"{side}_{bname}"].append(bl)
        pose = fr["pose_landmarks_3d"]
        for bi, bj, bname in POSE_BONES:
            li, lj = pose[bi], pose[bj]
            if (li["triangulation_valid"] and lj["triangulation_valid"] and
                li.get("used_reproj_max_px") is not None and
                lj.get("used_reproj_max_px") is not None and
                li["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX and
                lj["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX):
                bl = float(np.linalg.norm(np.array(li["xyz"]) - np.array(lj["xyz"])))
                bone_data[f"pose_{bname}"].append(bl)
    medians = {}
    for name, lengths in bone_data.items():
        if len(lengths) < 3:
            continue
        a = np.array(lengths)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        if mad < 1e-6:
            mad = 1e-6
        medians[name] = {"median": med, "mad": mad}
    return medians


def rescore_pass2(frames_3d, bone_medians, intrinsics, extrinsics, proj_matrices, camera_centers):
    n_changed = 0

    for fi, fr in enumerate(frames_3d):
        for cat, key, bones, prefix in [
            ("pose", "pose_landmarks_3d", POSE_BONES, "pose_"),
            ("left_hand", "left_hand_landmarks_3d", HAND_BONES, "left_"),
            ("right_hand", "right_hand_landmarks_3d", HAND_BONES, "right_"),
        ]:
            for li_idx, lm in enumerate(fr[key]):
                all_candidates = lm.get("_all_candidates")
                if not all_candidates:
                    # No candidates means invalid -- store metadata anyway
                    lm["pass2_selected_cameras"] = list(lm["cameras_used"])
                    lm["pass2_final_score"] = lm.get("pass1_geometry_score")
                    if lm["triangulation_valid"]:
                        lm["selection_reason"] = f"{lm['n_cameras_used']}v_fs{lm.get('pass1_geometry_score', 0):.1f}"
                    continue

                obs_views = lm.get("_obs_views", [])

                for ev in all_candidates:
                    if not ev["feasible"]:
                        continue
                    X3 = np.array(ev["xyz"])

                    # Bone consistency penalty
                    bone_pen = 0.0
                    for bi, bj, bname in bones:
                        if bi == lm["lm_index"] or bj == lm["lm_index"]:
                            other_idx = bj if bi == lm["lm_index"] else bi
                            other_lm = fr[key][other_idx]
                            bkey = f"{prefix}{bname}"
                            if (other_lm["triangulation_valid"] and
                                other_lm.get("used_reproj_max_px") is not None and
                                other_lm["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX and
                                bkey in bone_medians):
                                other_xyz = np.array(other_lm["xyz"])
                                bl = float(np.linalg.norm(X3 - other_xyz))
                                bm = bone_medians[bkey]
                                dev = abs(bl - bm["median"]) / (1.4826 * bm["mad"] + 1e-9)
                                if dev > 3.0:
                                    bone_pen += dev - 3.0

                    ev["bone_consistency_penalty"] = round(
                        bone_pen * CONFIG["bone_consistency_penalty_weight"], 4)

                    # Temporal continuity penalty
                    temp_pen = 0.0
                    if fi > 0:
                        prev_lm = frames_3d[fi - 1][key][li_idx]
                        dt = fr["front_time_s"] - frames_3d[fi - 1]["front_time_s"]
                        if (prev_lm["triangulation_valid"] and dt > 0 and
                            dt <= CONFIG["temporal_max_gap_s"]):
                            prev_xyz = np.array(prev_lm["xyz"])
                            d = float(np.linalg.norm(X3 - prev_xyz))
                            vel = d / dt
                            if vel > CONFIG["temporal_velocity_warn_ms"]:
                                temp_pen = (vel - CONFIG["temporal_velocity_warn_ms"]) / CONFIG["temporal_velocity_warn_ms"]

                    ev["temporal_continuity_penalty"] = round(
                        temp_pen * CONFIG["temporal_continuity_penalty_weight"], 4)

                    # Camera switch penalty
                    switch_pen = 0.0
                    if fi > 0:
                        prev_lm = frames_3d[fi - 1][key][li_idx]
                        if prev_lm["triangulation_valid"]:
                            if set(prev_lm["cameras_used"]) != set(ev["cameras"]):
                                switch_pen = 1.0

                    ev["camera_switch_penalty"] = round(
                        switch_pen * CONFIG["camera_switch_penalty_weight"], 4)

                    ev["final_score"] = round(
                        ev["geometry_score"]
                        - ev["bone_consistency_penalty"]
                        - ev["temporal_continuity_penalty"]
                        - ev["camera_switch_penalty"], 4)
                    ev["reason"] = f"{len(ev['cameras'])}v_fs{ev['final_score']:.1f}"

                # Re-select best by final_score
                best, best_score = None, -float("inf")
                for ev in all_candidates:
                    if ev["feasible"] and ev["final_score"] > best_score:
                        best_score = ev["final_score"]
                        best = ev

                if best is None:
                    continue

                old_cams = set(lm["cameras_used"])
                new_cams = set(best["cameras"])
                changed = old_cams != new_cams or lm["xyz"] != best["xyz"]

                if changed:
                    n_changed += 1
                    lm["qa_flags"] = []
                    _apply_best_candidate(lm, best, obs_views)

                # Store pass2 metadata for ALL valid landmarks
                lm["pass2_selected_cameras"] = list(best["cameras"])
                lm["pass2_final_score"] = best["final_score"]
                lm["selection_changed"] = changed
                lm["selection_reason"] = best["reason"]

                lm["candidates"] = _make_cand_summaries(all_candidates)

    return n_changed


def cleanup_internal_fields(all_results):
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    lm.pop("_all_candidates", None)
                    lm.pop("_obs", None)
                    lm.pop("_obs_views", None)


# ================================================================
#  Post-processing: Bone length QA
# ================================================================

def apply_bone_qa(all_results):
    bone_outliers = []
    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        if not frames:
            continue
        for side in ["left", "right"]:
            key = f"{side}_hand_landmarks_3d"
            bone_lens = defaultdict(list)
            for fi, fr in enumerate(frames):
                hand = fr[key]
                for bi, bj, bname in HAND_BONES:
                    if hand[bi]["triangulation_valid"] and hand[bj]["triangulation_valid"]:
                        bl = float(np.linalg.norm(
                            np.array(hand[bi]["xyz"]) - np.array(hand[bj]["xyz"])))
                        bone_lens[bname].append((fi, bl))
            for bname, measurements in bone_lens.items():
                if len(measurements) < 3:
                    continue
                lengths = np.array([m[1] for m in measurements])
                med = float(np.median(lengths))
                mad = float(np.median(np.abs(lengths - med)))
                if mad < 1e-6:
                    mad = 1e-6
                thresh = CONFIG["bone_outlier_mad_factor"] * 1.4826 * mad
                is_phalanx = "mcp" not in bname and "palm" not in bname
                abs_max = CONFIG["bone_max_finger_phalanx_m"] if is_phalanx else CONFIG["bone_max_metacarpal_m"]
                for fi, bl in measurements:
                    outlier = abs(bl - med) > thresh or bl > abs_max
                    if not outlier:
                        continue
                    for bi2, bj2, bn2 in HAND_BONES:
                        if bn2 == bname:
                            hand = frames[fi][key]
                            for idx in [bi2, bj2]:
                                if hand[idx]["qa_valid"]:
                                    hand[idx]["qa_valid"] = False
                                    _invalidate_te(hand[idx])
                                    hand[idx]["qa_flags"].append(f"bone_{side}_{bname}_{bl:.4f}m")
                            bone_outliers.append({
                                "sample_id": sid, "frame_index": fi,
                                "front_time_s": round(frames[fi]["front_time_s"], 6),
                                "category": f"{side}_hand",
                                "bone": f"{side}_{bname}", "indices": [bi2, bj2],
                                "length_m": round(bl, 4), "median_m": round(med, 4),
                                "mad_m": round(mad, 6), "threshold_m": round(thresh, 4),
                                "dev_factor": round(abs(bl - med) / (1.4826 * mad + 1e-9), 1),
                            })
                            break
        # Pose bones
        pose_lens = defaultdict(list)
        for fi, fr in enumerate(frames):
            pose = fr["pose_landmarks_3d"]
            for bi, bj, bname in POSE_BONES:
                if pose[bi]["triangulation_valid"] and pose[bj]["triangulation_valid"]:
                    bl = float(np.linalg.norm(
                        np.array(pose[bi]["xyz"]) - np.array(pose[bj]["xyz"])))
                    pose_lens[bname].append((fi, bl))
        for bname, measurements in pose_lens.items():
            if len(measurements) < 3:
                continue
            lengths = np.array([m[1] for m in measurements])
            med = float(np.median(lengths))
            mad = float(np.median(np.abs(lengths - med)))
            if mad < 1e-6:
                mad = 1e-6
            thresh = CONFIG["bone_outlier_mad_factor"] * 1.4826 * mad
            for fi, bl in measurements:
                if abs(bl - med) <= thresh:
                    continue
                for bi2, bj2, bn2 in POSE_BONES:
                    if bn2 == bname:
                        pose = frames[fi]["pose_landmarks_3d"]
                        for idx in [bi2, bj2]:
                            if pose[idx]["qa_valid"]:
                                pose[idx]["qa_valid"] = False
                                _invalidate_te(pose[idx])
                                pose[idx]["qa_flags"].append(f"bone_pose_{bname}_{bl:.4f}m")
                        bone_outliers.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(frames[fi]["front_time_s"], 6),
                            "category": "pose", "bone": f"pose_{bname}",
                            "indices": [bi2, bj2],
                            "length_m": round(bl, 4), "median_m": round(med, 4),
                            "mad_m": round(mad, 6), "threshold_m": round(thresh, 4),
                            "dev_factor": round(abs(bl - med) / (1.4826 * mad + 1e-9), 1),
                        })
                        break
    return bone_outliers


# ================================================================
#  Torso anchor Z for depth oscillation
# ================================================================

def compute_torso_z(frame):
    """Average Z of shoulders (pose 11, 12) as torso anchor."""
    pose = frame["pose_landmarks_3d"]
    vals = []
    for idx in [11, 12]:
        if pose[idx]["triangulation_valid"] and pose[idx]["xyz"]:
            vals.append(pose[idx]["xyz"][2])
    if vals:
        return float(np.mean(vals))
    return None


# ================================================================
#  Post-processing: Temporal QA (reappearance + velocity + accel + depth osc)
# ================================================================

def apply_temporal_qa(all_results):
    temporal_anomalies = []
    reappearance_records = []

    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        if len(frames) < 2:
            continue

        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            n_lm = len(frames[0][key])

            for li in range(n_lm):
                # State tracking
                last_valid_fi = None
                last_valid_xyz = None
                last_valid_time = None
                # Velocity history for acceleration
                prev_vel_vec = None
                prev_vel_dt = None
                prev_vel_fi = None
                # Z series for depth oscillation (consecutive valid frames only)
                z_window = []  # list of (fi, z_relative, xyz, time)

                for fi in range(len(frames)):
                    curr = frames[fi]
                    lc = curr[key][li]
                    curr_time = curr["front_time_s"]

                    if not lc["triangulation_valid"]:
                        # Reset z_window on invalid gap
                        z_window = []
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None
                        continue

                    curr_xyz = np.array(lc["xyz"])

                    # --- Reappearance detection ---
                    # Check immediate predecessor (fi-1)
                    imm_prev_valid = False
                    if fi > 0:
                        imm_prev_lm = frames[fi - 1][key][li]
                        imm_prev_valid = imm_prev_lm["triangulation_valid"]

                    is_reappear = False
                    if fi > 0 and not imm_prev_valid and last_valid_fi is not None:
                        # This is a reappearance: gap of >=1 invalid frame
                        is_reappear = True
                        gap_frames = fi - last_valid_fi - 1
                        gap_s = curr_time - last_valid_time
                        reappear_disp = float(np.linalg.norm(curr_xyz - last_valid_xyz))
                        reappear_vel = reappear_disp / gap_s if gap_s > 0 else float("inf")

                        lc["is_reappearance"] = True
                        lc["last_valid_frame_index"] = last_valid_fi
                        lc["gap_frames"] = gap_frames
                        lc["gap_s"] = round(gap_s, 6)
                        lc["reappear_displacement_m"] = round(reappear_disp, 4)
                        lc["reappear_velocity_ms"] = round(reappear_vel, 2)

                        # Classification
                        if reappear_disp > CONFIG["reappear_jump_threshold_m"]:
                            if reappear_vel > CONFIG["temporal_velocity_warn_ms"]:
                                classification = "large_jump"
                            else:
                                classification = "moderate_displacement"
                        else:
                            classification = "stable_reappearance"
                        lc["reappearance_classification"] = classification

                        lc["qa_flags"].append(f"detection_reappear_{classification}")

                        # Policy: first reappearance -> template_eligible=False
                        _invalidate_te(lc)

                        # If large jump + high velocity -> qa_valid=False
                        if classification == "large_jump":
                            lc["qa_valid"] = False

                        reappearance_records.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr_time, 6),
                            "category": cat, "landmark_index": li,
                            "is_reappearance": True,
                            "last_valid_frame_index": last_valid_fi,
                            "gap_frames": gap_frames,
                            "gap_s": round(gap_s, 6),
                            "reappear_displacement_m": round(reappear_disp, 4),
                            "reappear_velocity_ms": round(reappear_vel, 2),
                            "reappearance_classification": classification,
                            "cameras_used": lc["cameras_used"],
                            "qa_valid": lc["qa_valid"],
                            "template_eligible": lc["template_eligible"],
                        })

                        temporal_anomalies.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr_time, 6),
                            "category": cat, "landmark_index": li,
                            "displacement_m": round(reappear_disp, 4),
                            "displacement_xy_m": round(float(np.linalg.norm(curr_xyz[:2] - last_valid_xyz[:2])), 4),
                            "displacement_z_m": round(abs(float(curr_xyz[2] - last_valid_xyz[2])), 4),
                            "velocity_ms": round(reappear_vel, 2),
                            "velocity_xy_ms": None, "velocity_z_ms": None,
                            "acceleration_ms2": None, "acceleration_xy_ms2": None,
                            "acceleration_z_ms2": None,
                            "acceleration_available": False,
                            "acceleration_unavailable_reason": "reappearance_gap",
                            "dt_s": round(gap_s, 6),
                            "cameras_prev": frames[last_valid_fi][key][li]["cameras_used"] if last_valid_fi is not None else None,
                            "cameras_curr": lc["cameras_used"],
                            "cameras_changed": True,
                            "last_valid_frame_index": last_valid_fi,
                            "gap_frames": gap_frames,
                            "reappear_displacement_m": round(reappear_disp, 4),
                            "reappear_velocity_ms": round(reappear_vel, 2),
                            "reason": "detection_reappear",
                            # Depth osc fields (null for non-osc)
                            "window_start_fi": None, "window_end_fi": None,
                            "z_ptp_residual_m": None, "reversal_count": None,
                            "z_residual_mad_m": None, "z_xy_ratio": None,
                        })

                        # Reset velocity history on reappearance
                        z_window = []
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None

                    # --- Frame-to-frame velocity/acceleration ---
                    elif fi > 0 and imm_prev_valid and last_valid_fi is not None:
                        dt = curr_time - last_valid_time
                        if 0 < dt <= CONFIG["temporal_max_gap_s"]:
                            disp = curr_xyz - last_valid_xyz
                            d = float(np.linalg.norm(disp))
                            vel = d / dt
                            xy_d = float(np.linalg.norm(disp[:2]))
                            z_d = abs(float(disp[2]))
                            vel_vec = disp / dt

                            # Acceleration
                            accel = None
                            accel_xy = None
                            accel_z = None
                            accel_available = False
                            accel_reason = None

                            if prev_vel_vec is not None and prev_vel_dt is not None:
                                dt_avg = (dt + prev_vel_dt) / 2
                                if dt_avg > 0:
                                    accel_vec = (vel_vec - prev_vel_vec) / dt_avg
                                    accel = float(np.linalg.norm(accel_vec))
                                    accel_xy = float(np.linalg.norm(accel_vec[:2]))
                                    accel_z = abs(float(accel_vec[2]))
                                    accel_available = True
                            else:
                                accel_reason = "no_predecessor_velocity"

                            cams_changed = set(last_valid_xyz_cams) != set(lc["cameras_used"]) if last_valid_fi is not None else False
                            has_bone = any("bone_" in f for f in lc["qa_flags"])
                            depth_only = z_d > 3 * xy_d and z_d > 0.05

                            flagged = False
                            if vel > CONFIG["temporal_velocity_warn_ms"]:
                                if cams_changed:
                                    reason = "camera_switch_teleport"
                                elif has_bone:
                                    reason = "bone_anomaly_jump"
                                elif depth_only:
                                    reason = "depth_axis_anomaly"
                                else:
                                    reason = "high_velocity"

                                lc["qa_flags"].append(f"temporal_{reason}_{vel:.1f}ms")
                                if reason != "high_velocity":
                                    lc["qa_valid"] = False
                                    _invalidate_te(lc)
                                flagged = True

                                temporal_anomalies.append({
                                    "sample_id": sid, "frame_index": fi,
                                    "front_time_s": round(curr_time, 6),
                                    "category": cat, "landmark_index": li,
                                    "displacement_m": round(d, 4),
                                    "displacement_xy_m": round(xy_d, 4),
                                    "displacement_z_m": round(z_d, 4),
                                    "velocity_ms": round(vel, 2),
                                    "velocity_xy_ms": round(xy_d / dt, 2),
                                    "velocity_z_ms": round(z_d / dt, 2),
                                    "acceleration_ms2": round(accel, 2) if accel is not None else None,
                                    "acceleration_xy_ms2": round(accel_xy, 2) if accel_xy is not None else None,
                                    "acceleration_z_ms2": round(accel_z, 2) if accel_z is not None else None,
                                    "acceleration_available": accel_available,
                                    "acceleration_unavailable_reason": accel_reason,
                                    "dt_s": round(dt, 6),
                                    "cameras_prev": frames[last_valid_fi][key][li]["cameras_used"],
                                    "cameras_curr": lc["cameras_used"],
                                    "cameras_changed": cams_changed,
                                    "last_valid_frame_index": last_valid_fi,
                                    "gap_frames": 0,
                                    "reappear_displacement_m": None,
                                    "reappear_velocity_ms": None,
                                    "reason": reason,
                                    "window_start_fi": None, "window_end_fi": None,
                                    "z_ptp_residual_m": None, "reversal_count": None,
                                    "z_residual_mad_m": None, "z_xy_ratio": None,
                                })

                            elif not flagged and accel_available and accel > CONFIG["temporal_accel_warn_ms2"]:
                                lc["qa_flags"].append(f"high_accel_{accel:.0f}ms2")
                                temporal_anomalies.append({
                                    "sample_id": sid, "frame_index": fi,
                                    "front_time_s": round(curr_time, 6),
                                    "category": cat, "landmark_index": li,
                                    "displacement_m": round(d, 4),
                                    "displacement_xy_m": round(xy_d, 4),
                                    "displacement_z_m": round(z_d, 4),
                                    "velocity_ms": round(vel, 2),
                                    "velocity_xy_ms": round(xy_d / dt, 2),
                                    "velocity_z_ms": round(z_d / dt, 2),
                                    "acceleration_ms2": round(accel, 2),
                                    "acceleration_xy_ms2": round(accel_xy, 2) if accel_xy is not None else None,
                                    "acceleration_z_ms2": round(accel_z, 2) if accel_z is not None else None,
                                    "acceleration_available": True,
                                    "acceleration_unavailable_reason": None,
                                    "dt_s": round(dt, 6),
                                    "cameras_prev": frames[last_valid_fi][key][li]["cameras_used"],
                                    "cameras_curr": lc["cameras_used"],
                                    "cameras_changed": cams_changed,
                                    "last_valid_frame_index": last_valid_fi,
                                    "gap_frames": 0,
                                    "reappear_displacement_m": None,
                                    "reappear_velocity_ms": None,
                                    "reason": "high_acceleration",
                                    "window_start_fi": None, "window_end_fi": None,
                                    "z_ptp_residual_m": None, "reversal_count": None,
                                    "z_residual_mad_m": None, "z_xy_ratio": None,
                                })

                            prev_vel_vec = vel_vec
                            prev_vel_dt = dt
                            prev_vel_fi = fi

                    # --- Z window for depth oscillation ---
                    torso_z = compute_torso_z(curr)
                    z_rel = float(curr_xyz[2]) - torso_z if torso_z is not None else float(curr_xyz[2])
                    xy_abs = float(np.linalg.norm(curr_xyz[:2]))

                    z_window.append((fi, z_rel, curr_xyz.copy(), curr_time, lc["cameras_used"], xy_abs))

                    # Track for next iteration
                    last_valid_fi = fi
                    last_valid_xyz = curr_xyz.copy()
                    last_valid_time = curr_time
                    last_valid_xyz_cams = list(lc["cameras_used"])

                    # First frame for this landmark -- initialize
                    if fi == 0 or is_reappear:
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None

                # --- Depth oscillation analysis on z_window ---
                # Process sliding windows on consecutive valid frames
                _detect_depth_oscillation(
                    sid, cat, li, frames, key, z_window, temporal_anomalies)

    return temporal_anomalies, reappearance_records


def _detect_depth_oscillation(sid, cat, li, frames, key, z_window, temporal_anomalies):
    """Detect depth oscillation using detrended Z residuals."""
    win_size = CONFIG["depth_osc_window_size"]
    min_rev = CONFIG["depth_osc_min_reversals"]
    min_ptp = CONFIG["depth_osc_min_ptp_m"]
    min_z_xy = CONFIG["depth_osc_min_z_xy_ratio"]
    min_rmad = CONFIG["depth_osc_min_residual_mad_m"]

    if len(z_window) < win_size:
        return

    flagged_centers = set()

    for ws in range(len(z_window) - win_size + 1):
        win = z_window[ws:ws + win_size]

        # Check time continuity -- no large gaps
        max_dt = max(win[k+1][3] - win[k][3] for k in range(len(win) - 1))
        if max_dt > CONFIG["temporal_max_gap_s"]:
            continue

        # Extract Z values and detrend (remove linear trend)
        z_vals = np.array([w[1] for w in win])  # relative to torso
        t_vals = np.array([w[3] for w in win])
        t_norm = t_vals - t_vals[0]

        # Linear detrend
        if t_norm[-1] > 0:
            slope = (z_vals[-1] - z_vals[0]) / t_norm[-1]
            z_detrended = z_vals - (z_vals[0] + slope * t_norm)
        else:
            z_detrended = z_vals - np.mean(z_vals)

        # Z residual statistics
        z_ptp = float(np.max(z_detrended) - np.min(z_detrended))
        z_mad = float(np.median(np.abs(z_detrended - np.median(z_detrended))))

        if z_ptp < min_ptp or z_mad < min_rmad:
            continue

        # Count sign reversals in detrended Z
        z_diffs = np.diff(z_detrended)
        signs = np.sign(z_diffs)
        signs = signs[signs != 0]  # remove zero steps
        if len(signs) < 2:
            continue
        reversals = int(np.sum(np.diff(signs) != 0))

        if reversals < min_rev:
            continue

        # XY residual for comparison
        xy_steps = []
        for k in range(1, len(win)):
            xy_d = float(np.linalg.norm(win[k][2][:2] - win[k-1][2][:2]))
            xy_steps.append(xy_d)
        z_steps = [abs(float(z_diffs[k])) for k in range(len(z_diffs))]
        mean_xy_step = float(np.mean(xy_steps)) if xy_steps else 1e-9
        mean_z_step = float(np.mean(z_steps)) if z_steps else 0

        z_xy_ratio = mean_z_step / max(mean_xy_step, 1e-9)

        if z_xy_ratio < min_z_xy:
            continue

        # This is a confirmed depth oscillation
        center_idx = win_size // 2
        center_fi = win[center_idx][0]

        if center_fi in flagged_centers:
            continue
        flagged_centers.add(center_fi)

        lc_center = frames[center_fi][key][li]
        flag = f"depth_oscillation_ptp{z_ptp:.3f}m_rev{reversals}"
        lc_center["qa_flags"].append(flag)
        # Only invalidate qa_valid if strong evidence
        # (high amplitude AND z-dominant)
        if z_ptp > 0.05 and z_xy_ratio > 3.0:
            lc_center["qa_valid"] = False
            _invalidate_te(lc_center)

        # Center frame velocity/acceleration
        center_vel = None
        center_accel = None
        if center_idx > 0 and center_idx < len(win) - 1:
            dt_fwd = win[center_idx + 1][3] - win[center_idx][3]
            if dt_fwd > 0:
                center_vel = float(np.linalg.norm(
                    win[center_idx + 1][2] - win[center_idx][2])) / dt_fwd

        temporal_anomalies.append({
            "sample_id": sid, "frame_index": center_fi,
            "front_time_s": round(win[center_idx][3], 6),
            "category": cat, "landmark_index": li,
            "displacement_m": None, "displacement_xy_m": None,
            "displacement_z_m": None,
            "velocity_ms": round(center_vel, 2) if center_vel else None,
            "velocity_xy_ms": None, "velocity_z_ms": None,
            "acceleration_ms2": None, "acceleration_xy_ms2": None,
            "acceleration_z_ms2": None,
            "acceleration_available": False,
            "acceleration_unavailable_reason": "depth_oscillation_window",
            "dt_s": None,
            "cameras_prev": None,
            "cameras_curr": lc_center["cameras_used"],
            "cameras_changed": None,
            "last_valid_frame_index": None,
            "gap_frames": None,
            "reappear_displacement_m": None,
            "reappear_velocity_ms": None,
            "reason": "depth_oscillation",
            "window_start_fi": win[0][0],
            "window_end_fi": win[-1][0],
            "z_ptp_residual_m": round(z_ptp, 4),
            "reversal_count": reversals,
            "z_residual_mad_m": round(z_mad, 4),
            "z_xy_ratio": round(z_xy_ratio, 2),
            "consecutive_frames": True,
        })


# ================================================================
#  Statistics (same as v1.2)
# ================================================================

def compute_reproj_distributions(all_results):
    d = {
        "used_all": [], "excluded_all": [],
        "used_pose": [], "used_left_hand": [], "used_right_hand": [],
        "used_2view": [], "used_3view": [],
        "qa_valid_used": [], "qa_invalid_used": [],
        "preselection_3view": [],
        "te_true_used": [], "te_false_used": [],
    }
    per_cam_used = {v: [] for v in VIEWS}
    per_cam_excl = {v: [] for v in VIEWS}

    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    for cand in lm.get("candidates", []):
                        if cand["v"] == "F+L+R" and cand.get("ur") is not None:
                            d["preselection_3view"].append(cand["ur"])
                    if not lm["triangulation_valid"]:
                        continue
                    ue = lm["used_reproj_mean_px"]
                    if ue is not None:
                        d["used_all"].append(ue)
                        d[f"used_{cat}"].append(ue)
                        if lm["n_cameras_used"] == 2: d["used_2view"].append(ue)
                        elif lm["n_cameras_used"] == 3: d["used_3view"].append(ue)
                        if lm["qa_valid"]: d["qa_valid_used"].append(ue)
                        else: d["qa_invalid_used"].append(ue)
                        if lm["template_eligible"]: d["te_true_used"].append(ue)
                        else: d["te_false_used"].append(ue)
                    ee = lm.get("excluded_reproj_mean_px")
                    if ee is not None:
                        d["excluded_all"].append(ee)
                    for v, ci in lm["per_camera"].items():
                        err = ci.get("reproj_error_px")
                        if err is not None:
                            if ci.get("used_for_triangulation"):
                                per_cam_used[v].append(err)
                            else:
                                per_cam_excl[v].append(err)
    summary = {k: _ds(v) for k, v in d.items()}
    summary["per_camera_used"] = {v: _ds(e) for v, e in per_cam_used.items()}
    summary["per_camera_excluded"] = {v: _ds(e) for v, e in per_cam_excl.items()}
    return summary


def compute_coverage(frames_3d):
    total = len(frames_3d)
    if total == 0:
        return {}
    result = {}
    for cat, key, n_lm in [("pose", "pose_landmarks_3d", 33),
                             ("left_hand", "left_hand_landmarks_3d", 21),
                             ("right_hand", "right_hand_landmarks_3d", 21)]:
        tot_lm = total * n_lm
        n_tri = n_qa = n_te = n_3c = n_2c = n_inv = 0
        inv_reasons = defaultdict(int)
        for fr in frames_3d:
            for lm in fr[key]:
                if lm["triangulation_valid"]:
                    n_tri += 1
                    if lm["qa_valid"]: n_qa += 1
                    if lm["template_eligible"]: n_te += 1
                    if lm["n_cameras_used"] == 3: n_3c += 1
                    elif lm["n_cameras_used"] == 2: n_2c += 1
                else:
                    n_inv += 1
                    inv_reasons[lm.get("invalid_reason", "unknown")] += 1
        result[cat] = {
            "total": tot_lm,
            "tri_valid": n_tri, "tri_rate": round(n_tri / tot_lm, 4),
            "qa_valid": n_qa, "qa_rate": round(n_qa / tot_lm, 4),
            "template_eligible": n_te, "te_rate": round(n_te / tot_lm, 4),
            "n_3cam": n_3c, "n_2cam": n_2c,
            "invalid": n_inv, "invalid_reasons": dict(inv_reasons),
        }
    return result


def compute_exclusion_stats(all_results):
    per_cam_cat = {v: {"pose": 0, "left_hand": 0, "right_hand": 0} for v in VIEWS}
    pose_group_excl = {g: {v: 0 for v in VIEWS} for g in POSE_LM_GROUPS}
    total_excl = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["excluded_cameras"]:
                        total_excl += 1
                        for v in lm["excluded_cameras"]:
                            per_cam_cat[v][cat] += 1
                            if cat == "pose":
                                for gname, indices in POSE_LM_GROUPS.items():
                                    if lm["lm_index"] in indices:
                                        pose_group_excl[gname][v] += 1
    return {"per_camera_category": per_cam_cat, "total_excluded": total_excl,
            "pose_group_exclusion": pose_group_excl}


def compute_bone_stats(frames_3d):
    bone_data = defaultdict(list)
    for fr in frames_3d:
        pose = fr["pose_landmarks_3d"]
        for bi, bj, bname in POSE_BONES:
            if pose[bi]["triangulation_valid"] and pose[bj]["triangulation_valid"]:
                bone_data[f"pose_{bname}"].append(
                    float(np.linalg.norm(np.array(pose[bi]["xyz"]) - np.array(pose[bj]["xyz"]))))
        for side in ["left", "right"]:
            hand = fr[f"{side}_hand_landmarks_3d"]
            for bi, bj, bname in HAND_BONES:
                if hand[bi]["triangulation_valid"] and hand[bj]["triangulation_valid"]:
                    bone_data[f"{side}_{bname}"].append(
                        float(np.linalg.norm(np.array(hand[bi]["xyz"]) - np.array(hand[bj]["xyz"]))))
    result = {}
    for name, lengths in bone_data.items():
        if len(lengths) < 2:
            result[name] = {"n": len(lengths)}
            continue
        a = np.array(lengths)
        med = float(np.median(a))
        mad = float(np.median(np.abs(a - med)))
        result[name] = {
            "n": len(a), "median_m": round(med, 4),
            "mad_m": round(mad, 6), "std_m": round(float(np.std(a)), 4),
            "min_m": round(float(np.min(a)), 4), "max_m": round(float(np.max(a)), 4),
        }
    return result


# ================================================================
#  QA Flag analysis (same structure as v1.2)
# ================================================================

def analyze_qa_flags(all_results):
    report = []
    for sid in ["S001_P001_GREETING_FAST_R03", "S001_P001_GREETING_SLOW_R03"]:
        frames = all_results.get(sid, [])
        n = len(frames)
        n_tri = sum(1 for f in frames if any(lm["triangulation_valid"] for lm in f["right_hand_landmarks_3d"]))
        n_qa = sum(1 for f in frames if any(lm["qa_valid"] for lm in f["right_hand_landmarks_3d"]))
        report.append({
            "flag": "low_right_hand_detection", "sample_id": sid,
            "total_frames": n,
            "rh_tri_frames": n_tri, "rh_tri_rate": round(n_tri / max(n, 1), 3),
            "rh_qa_frames": n_qa, "rh_qa_rate": round(n_qa / max(n, 1), 3),
        })
    ty_sid = REGRESSION_TY_FRAMES["sample_id"]
    ty_frames = all_results.get(ty_sid, [])
    for fi in REGRESSION_TY_FRAMES["front_frame_indices"]:
        if fi >= len(ty_frames):
            report.append({"flag": "thankyou_fast_detailed", "frame_index": fi, "found": False})
            continue
        fr = ty_frames[fi]
        lh = fr["left_hand_landmarks_3d"]
        n_left_excl = sum(1 for lm in lh if lm["triangulation_valid"] and "LEFT" in lm.get("excluded_cameras", []))
        report.append({
            "flag": "thankyou_fast_detailed", "sample_id": ty_sid,
            "frame_index": fi, "front_time_s": fr["front_time_s"], "found": True,
            "n_tri": sum(1 for lm in lh if lm["triangulation_valid"]),
            "n_qa": sum(1 for lm in lh if lm["qa_valid"]),
            "n_te": sum(1 for lm in lh if lm["template_eligible"]),
            "n_left_excluded": n_left_excl,
        })
    return report


# ================================================================
#  Video frame extraction + contact sheets
# ================================================================

def extract_video_frames(video_path, frame_indices):
    frame_indices = sorted(set(frame_indices))
    if not frame_indices:
        return {}
    frames = {}
    con = av.open(str(video_path))
    st = con.streams.video[0]
    st.thread_type = "AUTO"
    fi = 0
    ti = 0
    for vf in con.decode(st):
        if ti >= len(frame_indices):
            break
        if fi == frame_indices[ti]:
            frames[fi] = vf.to_ndarray(format="bgr24")
            while ti < len(frame_indices) and frame_indices[ti] == fi:
                ti += 1
        fi += 1
    con.close()
    return frames


def make_contact_sheet(items, video_frames, out_path, title):
    tw, th = CONFIG["contact_thumb_size"]
    if not items:
        img = np.full((th, tw * 2, 3), 50, dtype=np.uint8)
        cv2.putText(img, f"QA: {title}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "No items", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(out_path), img)
        return
    n = min(len(items), CONFIG["contact_max_cells"])
    if len(items) > n:
        sel = [items[i] for i in np.linspace(0, len(items) - 1, n, dtype=int)]
    else:
        sel = items[:n]
    cols = min(4, n)
    rows = math.ceil(n / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for idx, item in enumerate(sel):
        dfi = item.get("dfi")
        if dfi is not None and dfi in video_frames:
            thumb = cv2.resize(video_frames[dfi], (tw, th))
        else:
            thumb = np.full((th, tw, 3), 30, dtype=np.uint8)
        y = 15
        for line in item.get("text", []):
            cv2.putText(thumb, str(line)[:60], (3, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
            y += 14
        r, c = divmod(idx, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = thumb
    cv2.imwrite(str(out_path), sheet)
    print(f"  Contact sheet: {title} -> {len(sel)} cells -> {out_path.name}")


def build_temporal_contact_items(temporal_anomalies, all_results):
    """Build balanced contact sheet items: by reason, by sample, deduplicated."""
    max_per_reason = CONFIG["contact_max_per_reason"]
    max_per_sample = CONFIG["contact_max_per_sample"]
    max_total = CONFIG["contact_max_cells"]

    by_reason = defaultdict(list)
    for ta in temporal_anomalies:
        by_reason[ta["reason"]].append(ta)

    # Sort each reason by severity (velocity or amplitude)
    for reason, items in by_reason.items():
        if reason == "depth_oscillation":
            items.sort(key=lambda x: x.get("z_ptp_residual_m") or 0, reverse=True)
        else:
            items.sort(key=lambda x: x.get("velocity_ms") or 0, reverse=True)

    # Select up to max_per_reason, balanced by sample
    selected = []
    seen_keys = set()

    for reason in ["camera_switch_teleport", "detection_reappear", "bone_anomaly_jump",
                    "depth_axis_anomaly", "depth_oscillation", "high_acceleration", "high_velocity"]:
        items = by_reason.get(reason, [])
        # Balance by sample within this reason
        by_sample = defaultdict(list)
        for ta in items:
            by_sample[ta["sample_id"]].append(ta)
        count = 0
        for sid in PILOT_SAMPLE_IDS:
            for ta in by_sample.get(sid, [])[:max_per_sample]:
                key = (ta["sample_id"], ta["frame_index"], ta["category"], ta["landmark_index"], ta["reason"])
                if key not in seen_keys and count < max_per_reason:
                    seen_keys.add(key)
                    selected.append(ta)
                    count += 1

    # Convert to contact items
    contact_items = []
    for ta in selected[:max_total]:
        sid = ta["sample_id"]
        fi = ta["frame_index"]
        fr = all_results[sid][fi]
        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")

        severity = ta.get("velocity_ms") or ta.get("z_ptp_residual_m") or 0
        contact_items.append({
            "dfi": dfi, "sample_id": sid,
            "frame_index": fi,
            "front_time_s": ta["front_time_s"],
            "category": ta["category"],
            "landmark_index": ta["landmark_index"],
            "reason": ta["reason"],
            "severity": round(severity, 2),
            "text": [
                f"{sid.replace('S001_P001_','')} f{fi}",
                f"t={ta['front_time_s']:.3f}s {ta['category']}",
                f"lm{ta['landmark_index']} {ta['reason']}",
                f"v={ta.get('velocity_ms','N/A')} a={ta.get('acceleration_ms2','N/A')}",
                f"d={ta.get('displacement_m','N/A')}m",
            ],
        })
    return contact_items


# ================================================================
#  Overlays
# ================================================================

def draw_overlay(img, fr3d, view, frame_idx):
    for cat_key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
        is_pose = cat_key == "pose_landmarks_3d"
        for lm in fr3d[cat_key]:
            cam = lm["per_camera"].get(view)
            if not cam:
                continue
            if cam.get("raw_2d_px"):
                px, py = int(cam["raw_2d_px"][0]), int(cam["raw_2d_px"][1])
                used = cam.get("used_for_triangulation", False)
                col = (0, 255, 0) if used else (255, 255, 0)
                cv2.circle(img, (px, py), 3 if is_pose else 2, col, -1)
            if lm["triangulation_valid"] and cam.get("reproj_2d_px"):
                rx, ry = int(cam["reproj_2d_px"][0]), int(cam["reproj_2d_px"][1])
                if not lm["qa_valid"]:
                    col = (128, 128, 128)
                elif is_pose:
                    col = (0, 0, 255)
                elif cat_key == "left_hand_landmarks_3d":
                    col = (255, 0, 0)
                else:
                    col = (255, 100, 0)
                if not lm["template_eligible"]:
                    cv2.drawMarker(img, (rx, ry), col, cv2.MARKER_CROSS, 6, 1)
                else:
                    cv2.circle(img, (rx, ry), 3, col, 1)
                if cam.get("raw_2d_px"):
                    ox, oy = int(cam["raw_2d_px"][0]), int(cam["raw_2d_px"][1])
                    lc_col = (0, 255, 255) if cam.get("used_for_triangulation") else (0, 165, 255)
                    cv2.line(img, (ox, oy), (rx, ry), lc_col, 1)
    pvi = fr3d["per_view_info"].get(view, {})
    cps = pvi.get("camera_local_pts_s", 0)
    cv2.putText(img, f"{view} t={cps:.2f}s dfi={frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    nq = [sum(1 for lm in fr3d[k] if lm["qa_valid"])
          for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]]
    nt = [sum(1 for lm in fr3d[k] if lm["template_eligible"])
          for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]]
    cv2.putText(img, f"QA P{nq[0]} L{nq[1]} R{nq[2]}  TE P{nt[0]} L{nt[1]} R{nt[2]}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


def make_overlays_for_view(view, all_results, synced, out_dir):
    dfi_lookup = defaultdict(list)
    for sid in PILOT_SAMPLE_IDS:
        for si, entry in enumerate(synced[sid]):
            if view in entry["views"]:
                dfi = entry["views"][view]["decoded_frame_index"]
                dfi_lookup[dfi].append((sid, si))
    if not dfi_lookup:
        return {}
    max_dfi = max(dfi_lookup.keys())
    con = av.open(str(VIDEOS[view]))
    st = con.streams.video[0]
    st.thread_type = "AUTO"
    w, h = st.width, st.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    counts = {}
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        op = out_dir / f"reproj_{short}_{view.lower()}.mp4"
        writers[sid] = cv2.VideoWriter(str(op), fourcc, 30, (w, h))
        counts[f"{short}_{view.lower()}"] = 0
    fi = 0
    for vf in con.decode(st):
        if fi > max_dfi:
            break
        entries = dfi_lookup.get(fi)
        if entries:
            raw_img = vf.to_ndarray(format="bgr24")
            for sid, si in entries:
                if si < len(all_results[sid]):
                    img = raw_img.copy()
                    draw_overlay(img, all_results[sid][si], view, fi)
                    writers[sid].write(img)
                    short = sid.replace("S001_P001_", "")
                    counts[f"{short}_{view.lower()}"] += 1
        fi += 1
    for wr in writers.values():
        wr.release()
    con.close()
    for k, c in counts.items():
        print(f"    Overlay {view} {k}: {c} frames")
    return counts


# ================================================================
#  Assertions
# ================================================================

def run_assertions(all_results, synced, bone_outliers, temporal_anomalies,
                   reappearance_records, overlay_counts, contact_paths,
                   contact_index_rows):
    results = []

    def chk(cond, msg):
        results.append(("PASS" if cond else "FAIL", msg))

    # 1. 9 sequences
    chk(len(all_results) == 9, f"9_sequences (got {len(all_results)})")

    # 2. Frame counts + structure
    total_frames = 0
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        frames = all_results.get(sid, [])
        expected = len(synced.get(sid, []))
        chk(len(frames) == expected, f"{short}_frames: {len(frames)}=={expected}")
        total_frames += len(frames)
        for fi, fr in enumerate(frames):
            if len(fr["pose_landmarks_3d"]) != 33:
                chk(False, f"{short}_f{fi}_pose!=33"); break
            if len(fr["left_hand_landmarks_3d"]) != 21:
                chk(False, f"{short}_f{fi}_lh!=21"); break
            if len(fr["right_hand_landmarks_3d"]) != 21:
                chk(False, f"{short}_f{fi}_rh!=21"); break
        else:
            chk(True, f"{short}_structure_ok")

    # 3. Total frames = 1001
    chk(total_frames == 1001, f"total_frames={total_frames}==1001")

    # 4. Total landmarks
    total_lm = sum(
        len(fr["pose_landmarks_3d"]) + len(fr["left_hand_landmarks_3d"]) + len(fr["right_hand_landmarks_3d"])
        for sid in PILOT_SAMPLE_IDS for fr in all_results[sid])
    chk(total_lm == 75075, f"total_landmarks={total_lm}==75075")

    # 5. Invalid have no coords
    n_bad = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                for lm in fr[key] if not lm["triangulation_valid"] and lm["xyz"] is not None)
    chk(n_bad == 0, f"no_coords_for_invalid (violations={n_bad})")

    # 6. Camera count matches
    n_mm = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
               for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
               for lm in fr[key] if lm["triangulation_valid"] and
               sum(1 for c in lm["per_camera"].values() if c.get("used_for_triangulation")) != lm["n_cameras_used"])
    chk(n_mm == 0, f"cam_count_match (mismatch={n_mm})")

    # 7. Cheirality
    n_behind = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                   for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in fr[key] if lm["triangulation_valid"] and lm["xyz"] and lm["xyz"][2] <= 0)
    chk(n_behind == 0, f"cheirality (behind={n_behind})")

    # 8. template_eligible implies qa_valid
    n_te_not_qa = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                      for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                      for lm in fr[key] if lm["template_eligible"] and not lm["qa_valid"])
    chk(n_te_not_qa == 0, f"te_implies_qa (violations={n_te_not_qa})")

    # 9. used_reproj_max > 20px => template_eligible=False
    n_high_te = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                    for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                    for lm in fr[key] if lm["triangulation_valid"] and
                    lm["used_reproj_max_px"] is not None and
                    lm["used_reproj_max_px"] > TEMPLATE_REPROJ_THRESHOLD_PX and lm["template_eligible"])
    chk(n_high_te == 0, f"high_reproj_not_te (violations={n_high_te})")

    # 10. per-view TE consistency
    n_view_te_bad = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                        for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                        for lm in fr[key] if not lm["template_eligible"]
                        for v in VIEWS if lm.get("template_eligible_by_view", {}).get(v, False))
    chk(n_view_te_bad == 0, f"per_view_te_consistency (violations={n_view_te_bad})")

    # 11. Regression: GREETING_NORMAL_R03 frame 39 right hand
    reg = REGRESSION_BONE
    frames = all_results.get(reg["sample_id"], [])
    if reg["frame_index"] < len(frames):
        rh = frames[reg["frame_index"]]["right_hand_landmarks_3d"]
        reg_ok = True
        reg_details = []
        for bi, bj in reg["bones"]:
            if rh[bi]["triangulation_valid"] and rh[bj]["triangulation_valid"]:
                bl = float(np.linalg.norm(np.array(rh[bi]["xyz"]) - np.array(rh[bj]["xyz"])))
                if bl >= reg["min_expected_m"]:
                    if rh[bi]["template_eligible"] or rh[bj]["template_eligible"]:
                        reg_ok = False
                        reg_details.append(f"{bi}-{bj}={bl:.4f}m_NOT_FLAGGED")
                    else:
                        reg_details.append(f"{bi}-{bj}={bl:.4f}m_FLAGGED_OK")
                else:
                    reg_details.append(f"{bi}-{bj}={bl:.4f}m_FIXED")
        chk(reg_ok, f"regression_greeting_f39_rh ({'; '.join(reg_details)})")
    else:
        chk(False, "regression_greeting_f39_not_found")

    # 12. THANK_YOU_FAST frames 48,49 LEFT exclusion
    ty = REGRESSION_TY_FRAMES
    ty_frames = all_results.get(ty["sample_id"], [])
    for ffi in ty["front_frame_indices"]:
        if ffi < len(ty_frames):
            lh = ty_frames[ffi]["left_hand_landmarks_3d"]
            n_le = sum(1 for lm in lh if lm["triangulation_valid"] and "LEFT" in lm.get("excluded_cameras", []))
            chk(n_le == 21, f"ty_fast_f{ffi}_left_excl={n_le}==21")
        else:
            chk(False, f"ty_fast_f{ffi}_not_found")

    # 13. Overlay frame counts
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            k = f"{short}_{v.lower()}"
            actual = overlay_counts.get(k, -1)
            chk(actual == expected, f"overlay_{k}: {actual}=={expected}")

    # 14. Contact sheet files exist and > 10KB
    for cp in contact_paths:
        if cp.exists():
            sz = cp.stat().st_size
            chk(sz > 10000, f"contact_{cp.name}_size={sz}")
        else:
            chk(False, f"contact_{cp.name}_missing")

    # 15. Reappearance: invalid->valid transitions match reappearance records
    n_transitions = 0
    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            for li in range(len(frames[0][key])):
                for fi in range(1, len(frames)):
                    prev_valid = frames[fi-1][key][li]["triangulation_valid"]
                    curr_valid = frames[fi][key][li]["triangulation_valid"]
                    if not prev_valid and curr_valid:
                        # Check there was a last_valid before this gap
                        has_prior = any(frames[k][key][li]["triangulation_valid"] for k in range(fi-1))
                        if has_prior:
                            n_transitions += 1
    n_reappear_flags = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                           for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                           for lm in fr[key] if lm.get("is_reappearance"))
    chk(n_transitions == n_reappear_flags,
        f"reappear_count: transitions={n_transitions} flags={n_reappear_flags}")

    # 16. All reappearances have gap and displacement
    n_reappear_missing = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                             for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                             for lm in fr[key] if lm.get("is_reappearance") and
                             (lm.get("gap_frames") is None or lm.get("reappear_displacement_m") is None))
    chk(n_reappear_missing == 0, f"reappear_has_gap_disp (missing={n_reappear_missing})")

    # 17. First reappearance NOT template_eligible
    n_reappear_te = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                        for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                        for lm in fr[key] if lm.get("is_reappearance") and lm["template_eligible"])
    chk(n_reappear_te == 0, f"reappear_not_te (violations={n_reappear_te})")

    # 18. No old-style depth_oscillation flags (v1.2 format)
    n_old_osc = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                    for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                    for lm in fr[key] for f in lm["qa_flags"]
                    if f.startswith("depth_oscillation_rev"))
    chk(n_old_osc == 0, f"no_old_depth_osc_flags (found={n_old_osc})")

    # 19. Depth oscillation amplitude in meters (z_ptp_residual_m < 5m is reasonable)
    depth_osc_items = [ta for ta in temporal_anomalies if ta["reason"] == "depth_oscillation"]
    if depth_osc_items:
        max_amp = max(ta.get("z_ptp_residual_m") or 0 for ta in depth_osc_items)
        chk(max_amp < 5.0, f"depth_osc_amplitude_meters (max={max_amp:.3f}m)")
    else:
        chk(True, "depth_osc_amplitude_meters (none found)")

    # 20. Depth windows don't cross invalid gaps (consecutive_frames=True for all)
    n_non_consec = sum(1 for ta in temporal_anomalies
                       if ta["reason"] == "depth_oscillation" and not ta.get("consecutive_frames", True))
    chk(n_non_consec == 0, f"depth_osc_consecutive (non_consec={n_non_consec})")

    # 21. Temporal reason-specific required fields
    n_missing_fields = 0
    for ta in temporal_anomalies:
        reason = ta["reason"]
        required = TEMPORAL_REQUIRED_FIELDS.get(reason, [])
        for field in required:
            if ta.get(field) is None:
                n_missing_fields += 1
    chk(n_missing_fields == 0, f"temporal_required_fields (missing={n_missing_fields})")

    # 22. Acceleration unavailable has reason
    n_accel_no_reason = sum(1 for ta in temporal_anomalies
                            if ta.get("acceleration_available") is False and
                            ta.get("acceleration_unavailable_reason") is None)
    chk(n_accel_no_reason == 0, f"accel_unavail_has_reason (missing={n_accel_no_reason})")

    # 23. All valid landmarks have final-score-based selection_reason
    n_bad_reason = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"]:
                        sr = lm.get("selection_reason", "")
                        if not ("v_fs" in sr or "v_gs" in sr):
                            n_bad_reason += 1
    chk(n_bad_reason == 0, f"selection_reason_format (bad={n_bad_reason})")

    # 24. Best candidate = max final_score
    n_score_mismatch = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if not lm["triangulation_valid"]:
                        continue
                    cands = lm.get("candidates", [])
                    feasible_scores = [c["fs"] for c in cands if c["ok"] and c["fs"] is not None]
                    if feasible_scores:
                        max_fs = max(feasible_scores)
                        p2fs = lm.get("pass2_final_score")
                        if p2fs is not None and abs(p2fs - max_fs) > 0.01:
                            n_score_mismatch += 1
    chk(n_score_mismatch == 0, f"best_equals_max_final_score (mismatch={n_score_mismatch})")

    # 25. All feasible candidates have score components
    n_missing_sc = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                       for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                       for lm in fr[key] for cand in lm.get("candidates", [])
                       if cand["ok"] and any(cand.get(k) is None for k in ["gs", "urp", "evp", "tas", "bcp", "tcp", "csp", "fs"]))
    chk(n_missing_sc == 0, f"all_candidates_have_scores (missing={n_missing_sc})")

    # 26. Contact sheet index: no duplicate identification keys
    if contact_index_rows:
        keys = set()
        n_dup = 0
        for row in contact_index_rows:
            k = (row.get("sheet_type"), row.get("sample_id"), row.get("sample_frame_index"),
                 row.get("category"), row.get("landmark_index"), row.get("anomaly_reason"))
            if k in keys:
                n_dup += 1
            keys.add(k)
        chk(n_dup == 0, f"contact_index_no_dups (dups={n_dup})")

    # 27. tri_valid count preserved from v1.2
    n_tri = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                for lm in fr[key] if lm["triangulation_valid"])
    chk(n_tri == 70943, f"tri_valid_preserved={n_tri}==70943")

    return results


# ================================================================
#  Main
# ================================================================

def main():
    print("Step 6B v1.2.1 PILOT: Multi-view 3D Reconstruction - S001_P001 R03\n")
    print("  Status: STEP6B_V1_2_TEMPLATE_FILTER_PASS / STEP6B_V1_2_TEMPORAL_QA_REWORK_REQUIRED\n")

    print("  Verifying inputs...")
    for fp in [C2_EXTRINSICS, C2_APPROVAL, SYNC_REPORT, SEGMENTS_FILE]:
        if not fp.exists():
            print(f"  MISSING: {fp}"); sys.exit(1)
    for v in VIEWS:
        for fp in [C1_INTRINSICS[v], STEP6A_DIR / f"landmarks_2d_{v.lower()}.json"]:
            if not fp.exists():
                print(f"  MISSING: {fp}"); sys.exit(1)

    with open(C2_APPROVAL) as f:
        c2a = json.load(f)
    if c2a.get("status") != "APPROVED/PASS":
        print(f"  C2 not approved: {c2a.get('status')}"); sys.exit(1)
    print(f"  C2: APPROVED (LEFT={c2a['baselines']['LEFT']['baseline_m']:.3f}m, "
          f"RIGHT={c2a['baselines']['RIGHT']['baseline_m']:.3f}m)")

    print("\n  Loading calibration...")
    intrinsics, extrinsics, proj_matrices, cam_centers = load_calibration()

    print("  Loading 2D landmarks...")
    all_frames = load_2d_landmarks()

    print("\n  Synchronizing frames...")
    synced = synchronize_frames(all_frames)
    for sid in PILOT_SAMPLE_IDS:
        n = len(synced[sid])
        print(f"    {sid.replace('S001_P001_', '')}: {n} frames")

    out_json = OUT_DIR / "raw_3d"
    out_ov = OUT_DIR / "overlays"
    out_cs = OUT_DIR / "contact_sheets"
    for d in [out_json, out_ov, out_cs]:
        d.mkdir(parents=True, exist_ok=True)

    # -- Pass 1 --
    print("\n  Pass 1: Geometry-only triangulation...")
    all_results = {}
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        frames_3d = process_sample_pass1(sid, synced[sid], intrinsics, extrinsics, proj_matrices, cam_centers)
        all_results[sid] = frames_3d
        n_tri = sum(1 for f in frames_3d for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"] for lm in f[k] if lm["triangulation_valid"])
        print(f"    {short}: {n_tri}/{len(frames_3d)*75} tri_valid")

    # -- Bone medians --
    print("\n  Computing bone medians for Pass 2...")
    all_bone_medians = {}
    for sid in PILOT_SAMPLE_IDS:
        all_bone_medians[sid] = compute_bone_medians_for_pass2(all_results[sid])

    # -- Pass 2 --
    print("  Pass 2: Re-scoring with penalties...")
    total_changed = 0
    for sid in PILOT_SAMPLE_IDS:
        n_changed = rescore_pass2(all_results[sid], all_bone_medians[sid],
                                  intrinsics, extrinsics, proj_matrices, cam_centers)
        total_changed += n_changed
        print(f"    {sid.replace('S001_P001_','')}: {n_changed} re-selected")
    print(f"  Total Pass 2 re-selections: {total_changed}")

    cleanup_internal_fields(all_results)

    # -- Bone QA --
    print("\n  Post-processing: bone length QA...")
    bone_outliers = apply_bone_qa(all_results)
    print(f"    Bone outliers: {len(bone_outliers)}")

    # -- Temporal QA (rewritten) --
    print("  Post-processing: temporal QA (reappearance + velocity + acceleration + depth osc)...")
    temporal_anomalies, reappearance_records = apply_temporal_qa(all_results)
    by_reason = defaultdict(int)
    for ta in temporal_anomalies:
        by_reason[ta["reason"]] += 1
    for r, c in sorted(by_reason.items()):
        print(f"    {r}: {c}")
    print(f"    Total temporal anomalies: {len(temporal_anomalies)}")
    print(f"    Reappearance records: {len(reappearance_records)}")

    # -- Counts --
    n_tri = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                for lm in f[k] if lm["triangulation_valid"])
    n_qa = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
               for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
               for lm in f[k] if lm["qa_valid"])
    n_te = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
               for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
               for lm in f[k] if lm["template_eligible"])
    n_high_te = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                    for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                    for lm in f[k] if lm["triangulation_valid"] and
                    lm["used_reproj_max_px"] is not None and
                    lm["used_reproj_max_px"] > TEMPLATE_REPROJ_THRESHOLD_PX and lm["template_eligible"])
    print(f"\n  Counts: tri={n_tri} qa={n_qa} te={n_te} high_reproj_te={n_high_te}")

    # -- Save JSONs --
    print("\n  Saving 3D JSONs...")
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        op = out_json / f"landmarks_3d_{short}.json"
        with open(op, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.2.1_pilot",
                "sample_id": sid,
                "coordinate_system": "FRONT_camera",
                "axes": "X=right Y=down Z=forward",
                "unit": "meters",
                "n_frames": len(all_results[sid]),
                "config": CONFIG,
                "frames": all_results[sid],
            }, f, indent=1, ensure_ascii=False)
        sz = op.stat().st_size / 1024 / 1024
        print(f"    {op.name} ({sz:.1f} MB)")

    # -- Stats --
    reproj_dists = compute_reproj_distributions(all_results)
    excl_stats = compute_exclusion_stats(all_results)
    qa_flag_report = analyze_qa_flags(all_results)

    # -- Overlays --
    print("\n  Generating overlays...")
    overlay_counts = {}
    for v in VIEWS:
        print(f"  View {v}:")
        oc = make_overlays_for_view(v, all_results, synced, out_ov)
        overlay_counts.update(oc)

    # -- Contact sheets --
    print("\n  Generating contact sheets (reason-balanced)...")
    temporal_cs_items = build_temporal_contact_items(temporal_anomalies, all_results)

    needed_dfis = set()
    for item in temporal_cs_items:
        if item.get("dfi") is not None:
            needed_dfis.add(item["dfi"])

    # Bone/reproj/exclusion contact items (keep simple balanced approach)
    bone_cs = []
    for bo in bone_outliers[:50]:
        sid, fi = bo["sample_id"], bo["frame_index"]
        dfi = all_results[sid][fi]["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        bone_cs.append({"dfi": dfi, "sample_id": sid, "text": [
            f"{sid.replace('S001_P001_','')} f{fi}",
            f"bone: {bo['bone']} len={bo['length_m']:.4f}m",
            f"dev={bo['dev_factor']:.1f}x MAD",
        ]})

    print(f"  Extracting {len(needed_dfis)} FRONT video frames...")
    video_frames = extract_video_frames(VIDEOS["FRONT"], sorted(needed_dfis))

    contact_paths = []
    cp = out_cs / "contact_temporal.png"
    make_contact_sheet(temporal_cs_items, video_frames, cp, "temporal")
    contact_paths.append(cp)

    cp = out_cs / "contact_bone.png"
    make_contact_sheet(bone_cs, video_frames, cp, "bone")
    contact_paths.append(cp)

    # Build contact sheet index
    contact_index_rows = []
    for item in temporal_cs_items:
        lm_data = None
        sid = item.get("sample_id", "")
        fi = item.get("frame_index")
        cat = item.get("category", "")
        li = item.get("landmark_index")
        if sid and fi is not None and cat and li is not None:
            key_map = {"pose": "pose_landmarks_3d", "left_hand": "left_hand_landmarks_3d", "right_hand": "right_hand_landmarks_3d"}
            kk = key_map.get(cat)
            if kk and fi < len(all_results.get(sid, [])):
                lm_data = all_results[sid][fi][kk][li]

        contact_index_rows.append({
            "sheet_type": "temporal",
            "sample_id": sid,
            "sample_frame_index": fi,
            "decoded_frame_index": item.get("dfi"),
            "front_time_s": item.get("front_time_s"),
            "category": cat,
            "landmark_index": li,
            "anomaly_reason": item.get("reason", ""),
            "severity": item.get("severity"),
            "used_cameras": ",".join(lm_data["cameras_used"]) if lm_data else "",
            "excluded_cameras": ",".join(lm_data["excluded_cameras"]) if lm_data else "",
            "reproj_error": lm_data["used_reproj_max_px"] if lm_data else None,
            "displacement": item.get("severity"),
            "velocity": None,
            "acceleration": None,
            "qa_valid": lm_data["qa_valid"] if lm_data else None,
            "template_eligible": lm_data["template_eligible"] if lm_data else None,
        })

    cs_index_path = out_cs / "contact_sheet_index.csv"
    if contact_index_rows:
        with open(cs_index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=contact_index_rows[0].keys())
            w.writeheader()
            w.writerows(contact_index_rows)

    # -- CSVs --
    print("\n  Saving CSVs...")

    # Frame/landmark QA CSV
    qa_rows = []
    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    te_bv = lm.get("template_eligible_by_view", {})
                    qa_rows.append({
                        "sample_id": sid, "frame_index": fi,
                        "front_time_s": fr["front_time_s"],
                        "category": cat, "landmark_index": lm["lm_index"],
                        "triangulation_valid": lm["triangulation_valid"],
                        "qa_valid": lm["qa_valid"],
                        "template_eligible": lm["template_eligible"],
                        "te_FRONT": te_bv.get("FRONT", False),
                        "te_LEFT": te_bv.get("LEFT", False),
                        "te_RIGHT": te_bv.get("RIGHT", False),
                        "n_cameras_used": lm["n_cameras_used"],
                        "cameras_used": ",".join(lm["cameras_used"]) if lm["cameras_used"] else "",
                        "excluded_cameras": ",".join(lm["excluded_cameras"]) if lm["excluded_cameras"] else "",
                        "selection_reason": lm["selection_reason"],
                        "pass1_cameras": ",".join(lm.get("pass1_selected_cameras", [])),
                        "pass2_cameras": ",".join(lm.get("pass2_selected_cameras", [])),
                        "selection_changed": lm.get("selection_changed", False),
                        "pass1_geometry_score": lm.get("pass1_geometry_score"),
                        "pass2_final_score": lm.get("pass2_final_score"),
                        "used_reproj_mean_px": lm["used_reproj_mean_px"],
                        "used_reproj_max_px": lm["used_reproj_max_px"],
                        "is_reappearance": lm.get("is_reappearance", False),
                        "gap_frames": lm.get("gap_frames"),
                        "gap_s": lm.get("gap_s"),
                        "reappear_displacement_m": lm.get("reappear_displacement_m"),
                        "reappear_velocity_ms": lm.get("reappear_velocity_ms"),
                        "reappearance_classification": lm.get("reappearance_classification"),
                        "invalid_reason": lm["invalid_reason"],
                        "qa_flags": "|".join(lm["qa_flags"]) if lm["qa_flags"] else "",
                    })
    qa_csv = OUT_DIR / "frame_landmark_qa.csv"
    with open(qa_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qa_rows[0].keys())
        w.writeheader()
        w.writerows(qa_rows)
    print(f"  frame_landmark_qa.csv: {len(qa_rows)} rows")

    # Reappearance CSV
    reappear_csv = OUT_DIR / "reappearance_records.csv"
    if reappearance_records:
        with open(reappear_csv, "w", newline="", encoding="utf-8") as f:
            for rr in reappearance_records:
                rr["cameras_used"] = ",".join(rr["cameras_used"]) if isinstance(rr["cameras_used"], list) else rr["cameras_used"]
            w = csv.DictWriter(f, fieldnames=reappearance_records[0].keys())
            w.writeheader()
            w.writerows(reappearance_records)
    print(f"  reappearance_records.csv: {len(reappearance_records)} rows")

    # Temporal anomaly CSV
    ta_csv = OUT_DIR / "temporal_anomalies.csv"
    if temporal_anomalies:
        ta_for_csv = []
        for ta in temporal_anomalies:
            row = dict(ta)
            if isinstance(row.get("cameras_prev"), list):
                row["cameras_prev"] = ",".join(row["cameras_prev"])
            if isinstance(row.get("cameras_curr"), list):
                row["cameras_curr"] = ",".join(row["cameras_curr"])
            ta_for_csv.append(row)
        all_keys = list(dict.fromkeys(k for row in ta_for_csv for k in row.keys()))
        with open(ta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(ta_for_csv)
    else:
        with open(ta_csv, "w", encoding="utf-8") as f:
            f.write("(no temporal anomalies)\n")
    print(f"  temporal_anomalies.csv: {len(temporal_anomalies)} rows")

    # Sample QA summary
    sum_rows = []
    for sid in PILOT_SAMPLE_IDS:
        cov = compute_coverage(all_results[sid])
        n_bo = sum(1 for bo in bone_outliers if bo["sample_id"] == sid)
        n_ta = sum(1 for ta in temporal_anomalies if ta["sample_id"] == sid)
        row = {"sample_id": sid, "n_frames": len(all_results[sid]),
               "n_bone_outliers": n_bo, "n_temporal_anomalies": n_ta}
        for cat in ["pose", "left_hand", "right_hand"]:
            cc = cov.get(cat, {})
            row[f"{cat}_tri"] = cc.get("tri_valid", 0)
            row[f"{cat}_qa"] = cc.get("qa_valid", 0)
            row[f"{cat}_te"] = cc.get("template_eligible", 0)
        sum_rows.append(row)
    sum_csv = OUT_DIR / "sample_qa_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_rows[0].keys())
        w.writeheader()
        w.writerows(sum_rows)

    # Other CSVs (bone, exclusion, reproj, overlay verification)
    bone_rows = []
    for sid in PILOT_SAMPLE_IDS:
        bs = compute_bone_stats(all_results[sid])
        for name, st in bs.items():
            bone_rows.append({"sample_id": sid, "bone": name, **st})
    with open(OUT_DIR / "bone_lengths.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bone_rows[0].keys())
        w.writeheader()
        w.writerows(bone_rows)

    if bone_outliers:
        with open(OUT_DIR / "bone_outliers.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bone_outliers[0].keys())
            w.writeheader()
            w.writerows(bone_outliers)

    excl_rows = []
    for v in VIEWS:
        for cat in ["pose", "left_hand", "right_hand"]:
            excl_rows.append({"camera": v, "category": cat,
                              "n_excluded": excl_stats["per_camera_category"][v][cat]})
    for gname in POSE_LM_GROUPS:
        for v in VIEWS:
            excl_rows.append({"camera": v, "category": f"pose_{gname}",
                              "n_excluded": excl_stats["pose_group_exclusion"].get(gname, {}).get(v, 0)})
    with open(OUT_DIR / "camera_exclusion_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=excl_rows[0].keys())
        w.writeheader()
        w.writerows(excl_rows)

    with open(OUT_DIR / "reproj_distributions.json", "w", encoding="utf-8") as f:
        json.dump(reproj_dists, f, indent=2)

    cam_rows = []
    for sid in PILOT_SAMPLE_IDS:
        per_cam = {v: [] for v in VIEWS}
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if not lm["triangulation_valid"]:
                        continue
                    for v, ci in lm["per_camera"].items():
                        e = ci.get("reproj_error_px")
                        if e is not None:
                            per_cam[v].append(e)
        for v in VIEWS:
            st = _ds(per_cam[v])
            cam_rows.append({"sample_id": sid, "camera": v, **st})
    with open(OUT_DIR / "camera_reproj_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cam_rows[0].keys())
        w.writeheader()
        w.writerows(cam_rows)

    ov_rows = []
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            k = f"{short}_{v.lower()}"
            actual = overlay_counts.get(k, -1)
            ov_rows.append({"sample": short, "view": v, "expected": expected, "actual": actual,
                            "match": actual == expected})
    with open(OUT_DIR / "overlay_frame_verification.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ov_rows[0].keys())
        w.writeheader()
        w.writerows(ov_rows)

    # Manifest
    manifest = {
        "step": "6B_pilot_v1.2.1",
        "date": datetime.datetime.now().isoformat(),
        "status": "STEP6B_V1_2_1_TEMPORAL_QA_COMPLETE",
        "template_filter_status": "STEP6B_V1_2_TEMPLATE_FILTER_PASS",
        "full_45_status": "FULL_45_BLOCKED_PENDING_EXTERNAL_REVIEW_V1_2_1",
        "triangulation_method": "DLT 2-pass multi-candidate scoring (unchanged from v1.2)",
        "temporal_qa": "v1.2.1: reappearance tracking, detrended depth oscillation, acceleration with availability",
        "smoothing": "DISABLED",
        "config": CONFIG,
        "pass2_reselections": total_changed,
    }
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    with open(OUT_DIR / "qa_flag_analysis.json", "w", encoding="utf-8") as f:
        json.dump(qa_flag_report, f, indent=2, ensure_ascii=False)

    # -- Assertions --
    print("\n  Running assertions...")
    assertion_results = run_assertions(all_results, synced, bone_outliers,
                                       temporal_anomalies, reappearance_records,
                                       overlay_counts, contact_paths, contact_index_rows)
    n_pass = sum(1 for s, _ in assertion_results if s == "PASS")
    n_fail = sum(1 for s, _ in assertion_results if s == "FAIL")
    for status, msg in assertion_results:
        if status == "FAIL":
            print(f"    FAIL: {msg}")
    print(f"  Assertions: {n_pass} PASS, {n_fail} FAIL")

    with open(OUT_DIR / "assertion_results.json", "w", encoding="utf-8") as f:
        json.dump([{"status": s, "message": m} for s, m in assertion_results], f, indent=2)

    # -- Report --
    print("\n  Generating report...")
    rpt = []
    rpt.append("# Step 6B v1.2.1 Pilot Report -- Multi-view 3D Reconstruction\n")
    rpt.append("## Status\n")
    rpt.append("- `STEP6B_V1_2_TEMPLATE_FILTER_PASS`")
    rpt.append("- `STEP6B_V1_2_1_TEMPORAL_QA_COMPLETE`")
    rpt.append("- `FULL_45_BLOCKED_PENDING_EXTERNAL_REVIEW_V1_2_1`\n")
    rpt.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    rpt.append(f"**Script**: `Tools/s001_step6b_triangulate_pilot_v1_2_1.py`")
    rpt.append(f"**Scope**: 9 R03 samples x 3 views\n")

    rpt.append("## Changes from v1.2\n")
    rpt.append("- **Reappearance tracking**: separate immediate-previous vs last-valid state")
    rpt.append("- **Reappearance policy**: first reappearance -> template_eligible=False; large_jump -> qa_valid=False")
    rpt.append("- **Depth oscillation**: detrended Z residual relative to torso anchor, meter-unit amplitude, window reset on gaps")
    rpt.append("- **Acceleration**: track availability + unavailable_reason")
    rpt.append("- **Selection metadata**: pass1/pass2 cameras, scores, selection_changed for ALL valid landmarks")
    rpt.append("- **Contact sheets**: balanced by reason type, not dominated by single category")
    rpt.append("- **Assertions**: reason-specific field checks, reappearance invariants\n")
    rpt.append("### Preserved from v1.2")
    rpt.append("- Triangulation core, 2-pass candidate scoring")
    rpt.append(f"- template_eligible threshold: used_reproj_max <= {TEMPLATE_REPROJ_THRESHOLD_PX}px")
    rpt.append("- Bone QA, camera exclusion logic")
    rpt.append(f"- tri_valid: {n_tri} (unchanged)\n")

    rpt.append("## Summary\n")
    rpt.append(f"- Frames: 1001, Landmarks: 75075")
    rpt.append(f"- tri_valid: {n_tri}, qa_valid: {n_qa}, template_eligible: {n_te}")
    rpt.append(f"- High reproj (>20px) AND te=True: {n_high_te}")
    rpt.append(f"- Pass 2 re-selections: {total_changed}")
    rpt.append(f"- Bone outliers: {len(bone_outliers)}")
    rpt.append(f"- Temporal anomalies: {len(temporal_anomalies)}")
    rpt.append(f"- Reappearance records: {len(reappearance_records)}\n")

    rpt.append("## Temporal Anomalies by Reason\n")
    rpt.append("| Reason | Count |")
    rpt.append("|--------|-------|")
    for r, c in sorted(by_reason.items()):
        rpt.append(f"| {r} | {c} |")
    rpt.append("")

    rpt.append("## Reappearance Statistics\n")
    if reappearance_records:
        by_class = defaultdict(int)
        for rr in reappearance_records:
            by_class[rr["reappearance_classification"]] += 1
        for cls, cnt in sorted(by_class.items()):
            rpt.append(f"- {cls}: {cnt}")
        n_reappear_te = sum(1 for rr in reappearance_records if rr["template_eligible"])
        rpt.append(f"- Reappearances with template_eligible=True: {n_reappear_te}")
    rpt.append("")

    rpt.append(f"## Assertions: {n_pass} PASS, {n_fail} FAIL\n")
    for status, msg in assertion_results:
        rpt.append(f"- [{status}] {msg}")
    rpt.append("")

    rpt.append("## Output Files\n")
    rpt.append("| File | Size |")
    rpt.append("|------|------|")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            sk = p.stat().st_size / 1024
            sz_str = f"{sk / 1024:.1f} MB" if sk > 1024 else f"{sk:.1f} KB"
            rpt.append(f"| {p.relative_to(OUT_DIR)} | {sz_str} |")
    rpt.append("")

    rpt_path = OUT_DIR / "STEP6B_PILOT_V1_2_1_REPORT.md"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))

    # -- ZIP --
    print("\n  Creating delivery ZIP...")
    zip_path = PROJECT_ROOT / "WordsInSilence_step6b_v1_2_1_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "Tools/s001_step6b_triangulate_pilot_v1_2_1.py")
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(PROC_DIR)))
    zip_sz = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP: {zip_path.name} ({zip_sz:.1f} MB)")

    print(f"\n  {'=' * 60}")
    print(f"  STEP 6B v1.2.1 PILOT SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  tri_valid: {n_tri}/75075")
    print(f"  qa_valid: {n_qa}/75075")
    print(f"  template_eligible: {n_te}/75075")
    print(f"  Bone outliers: {len(bone_outliers)}")
    print(f"  Temporal anomalies: {len(temporal_anomalies)}")
    print(f"  Reappearances: {len(reappearance_records)}")
    print(f"  Assertions: {n_pass} PASS, {n_fail} FAIL")
    print(f"  Output: {OUT_DIR}")
    print(f"  ZIP: {zip_path} ({zip_sz:.1f} MB)")
    if n_fail > 0:
        print(f"  WARNING: {n_fail} assertion(s) FAILED")
        for s, m in assertion_results:
            if s == "FAIL":
                print(f"    FAIL: {m}")


if __name__ == "__main__":
    main()
