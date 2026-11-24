"""
Step 6B v1.2.3 PILOT: Multi-view 3D Reconstruction -- S001_P001 R03 (9 samples)

Patch on v1.2.3:
  - Observation reappearance: per-view 2D observation state transitions
  - Initial appearance: explicit unavailable reason and gap metadata
  - High acceleration: previous/current velocity metadata
  - Depth anomaly center acceleration: computed from window coordinates
  - Assertions: real FAIL-able checks, no pass-through stubs
  - Report: precise QA invalidation breakdown
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
OUT_DIR = PROC_DIR / "landmarks_3d_v1" / "pilot_v1_2_3"

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

def _has_obs_2d(frame_2d_view, lm_type, lm_index, hand_side):
    """Check if a 2D observation exists for this landmark in this view's frame."""
    if lm_type == "pose":
        if not frame_2d_view.get("pose_detected"):
            return False
        lms = frame_2d_view.get("pose_landmarks")
        if not lms or lm_index >= len(lms):
            return False
        return True
    elif lm_type == "hand":
        if frame_2d_view.get(f"anat_{hand_side}_status") != "DETECTED":
            return False
        lms = frame_2d_view.get(f"anat_{hand_side}_landmarks")
        if not lms or lm_index >= len(lms):
            return False
        return True
    return False


def apply_temporal_qa(all_results, synced_data):
    temporal_anomalies = []
    appearance_records = []  # ALL invalid->valid transitions (1,161 expected)
    observation_records = []  # per-view observation transitions
    depth_stats = {"segments_total": 0, "windows_possible": 0,
                   "windows_evaluated": 0, "windows_skipped": 0,
                   "windows_skipped_by_reason": defaultdict(int),
                   "anomalies_detected": 0}

    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        synced_frames = synced_data[sid]
        if len(frames) < 2:
            continue

        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            n_lm = len(frames[0][key])
            lm_type = "pose" if cat == "pose" else "hand"
            hand_side = None
            if cat == "left_hand":
                hand_side = "left"
            elif cat == "right_hand":
                hand_side = "right"

            for li in range(n_lm):
                # State tracking
                last_valid_fi = None
                last_valid_xyz = None
                last_valid_time = None
                last_valid_xyz_cams = None
                # Velocity history for acceleration
                prev_vel_vec = None
                prev_vel_dt = None
                prev_vel_fi = None
                prev_vel_mag = None  # scalar for high_accel metadata
                # Z series for depth oscillation (consecutive valid frames only)
                z_window = []

                # Observation state per view
                obs_last_valid = {v: None for v in VIEWS}  # fi of last valid obs

                def _flush_depth_segment():
                    if z_window:
                        depth_stats["segments_total"] += 1
                        _detect_depth_oscillation(
                            sid, cat, li, frames, key, z_window,
                            temporal_anomalies, depth_stats)

                for fi in range(len(frames)):
                    curr = frames[fi]
                    lc = curr[key][li]
                    curr_time = curr["front_time_s"]
                    synced_entry = synced_frames[fi]

                    # --- Observation reappearance per view ---
                    obs_curr_by_view = {}
                    for v in VIEWS:
                        v_frame = synced_entry["views"].get(v)
                        if v_frame is not None:
                            has_obs = _has_obs_2d(v_frame, lm_type, li, hand_side)
                        else:
                            has_obs = False
                        obs_curr_by_view[v] = has_obs

                        if fi > 0:
                            prev_synced = synced_frames[fi - 1]
                            pv_frame = prev_synced["views"].get(v)
                            prev_has_obs = _has_obs_2d(pv_frame, lm_type, li, hand_side) if pv_frame else False

                            if not prev_has_obs and has_obs:
                                if obs_last_valid[v] is not None:
                                    obs_type = "observation_reappearance"
                                else:
                                    obs_type = "initial_observation_after_start_gap"
                                observation_records.append({
                                    "sample_id": sid, "frame_index": fi,
                                    "front_time_s": round(curr_time, 6),
                                    "category": cat, "landmark_index": li,
                                    "view": v,
                                    "observation_type": obs_type,
                                    "last_valid_obs_fi": obs_last_valid[v],
                                    "gap_frames": fi - obs_last_valid[v] - 1 if obs_last_valid[v] is not None else fi,
                                })

                        if has_obs:
                            obs_last_valid[v] = fi

                    if not lc["triangulation_valid"]:
                        _flush_depth_segment()
                        z_window = []
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None
                        prev_vel_mag = None
                        continue

                    curr_xyz = np.array(lc["xyz"])

                    # --- Appearance/Reappearance detection ---
                    imm_prev_valid = False
                    if fi > 0:
                        imm_prev_lm = frames[fi - 1][key][li]
                        imm_prev_valid = imm_prev_lm["triangulation_valid"]

                    is_reappear = False

                    # Build observation metadata for this transition
                    obs_prev_by_view_tri = {}
                    obs_curr_by_view_tri = {}
                    obs_reappeared_views = []
                    has_obs_reappearance = False
                    if fi > 0 and not imm_prev_valid:
                        for v in VIEWS:
                            prev_synced = synced_frames[fi - 1]
                            pv_frame = prev_synced["views"].get(v)
                            prev_obs = _has_obs_2d(pv_frame, lm_type, li, hand_side) if pv_frame else False
                            obs_prev_by_view_tri[v] = prev_obs
                            obs_curr_by_view_tri[v] = obs_curr_by_view[v]
                            if not prev_obs and obs_curr_by_view[v]:
                                obs_reappeared_views.append(v)
                        has_obs_reappearance = len(obs_reappeared_views) > 0

                    # Case 1: invalid->valid with prior valid (true reappearance)
                    if fi > 0 and not imm_prev_valid and last_valid_fi is not None:
                        is_reappear = True
                        gap_frames = fi - last_valid_fi - 1
                        gap_s = curr_time - last_valid_time
                        reappear_disp = float(np.linalg.norm(curr_xyz - last_valid_xyz))
                        reappear_vel = reappear_disp / gap_s if gap_s > 0 else float("inf")
                        prev_cams = list(last_valid_xyz_cams)
                        curr_cams = list(lc["cameras_used"])
                        cams_changed = set(prev_cams) != set(curr_cams)

                        lc["is_reappearance"] = True
                        lc["last_valid_frame_index"] = last_valid_fi
                        lc["gap_frames"] = gap_frames
                        lc["gap_s"] = round(gap_s, 6)
                        lc["reappear_displacement_m"] = round(reappear_disp, 4)
                        lc["reappear_velocity_ms"] = round(reappear_vel, 2)

                        if reappear_disp > CONFIG["reappear_jump_threshold_m"]:
                            if reappear_vel > CONFIG["temporal_velocity_warn_ms"]:
                                classification = "large_jump"
                            else:
                                classification = "moderate_displacement"
                        else:
                            classification = "stable_reappearance"
                        lc["reappearance_classification"] = classification
                        lc["qa_flags"].append(f"detection_reappear_{classification}")
                        _invalidate_te(lc)
                        if classification == "large_jump":
                            lc["qa_valid"] = False

                        appearance_records.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr_time, 6),
                            "category": cat, "landmark_index": li,
                            "appearance_type": "triangulation_reappearance",
                            "last_valid_frame_index": last_valid_fi,
                            "gap_frames": gap_frames,
                            "gap_s": round(gap_s, 6),
                            "reappear_displacement_m": round(reappear_disp, 4),
                            "reappear_velocity_ms": round(reappear_vel, 2),
                            "reappearance_classification": classification,
                            "reappearance_metrics_available": True,
                            "reappearance_metrics_unavailable_reason": None,
                            "cameras_prev": prev_cams,
                            "cameras_curr": curr_cams,
                            "cameras_changed": cams_changed,
                            "observation_prev_by_view": obs_prev_by_view_tri,
                            "observation_curr_by_view": obs_curr_by_view_tri,
                            "observation_reappeared_views": obs_reappeared_views,
                            "has_observation_reappearance": has_obs_reappearance,
                            "qa_valid": lc["qa_valid"],
                            "template_eligible": lc["template_eligible"],
                        })

                        reappear_disp_xy = float(np.linalg.norm(curr_xyz[:2] - last_valid_xyz[:2]))
                        reappear_disp_z = abs(float(curr_xyz[2] - last_valid_xyz[2]))
                        temporal_anomalies.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr_time, 6),
                            "category": cat, "landmark_index": li,
                            "displacement_m": round(reappear_disp, 4),
                            "displacement_xy_m": round(reappear_disp_xy, 4),
                            "displacement_z_m": round(reappear_disp_z, 4),
                            "velocity_ms": round(reappear_vel, 2),
                            "velocity_xy_ms": round(reappear_disp_xy / gap_s, 2) if gap_s > 0 else None,
                            "velocity_z_ms": round(reappear_disp_z / gap_s, 2) if gap_s > 0 else None,
                            "acceleration_ms2": None, "acceleration_xy_ms2": None,
                            "acceleration_z_ms2": None,
                            "acceleration_available": False,
                            "acceleration_unavailable_reason": "reappearance_gap",
                            "dt_s": round(gap_s, 6),
                            "cameras_prev": prev_cams,
                            "cameras_curr": curr_cams,
                            "cameras_changed": cams_changed,
                            "last_valid_frame_index": last_valid_fi,
                            "gap_frames": gap_frames,
                            "reappear_displacement_m": round(reappear_disp, 4),
                            "reappear_velocity_ms": round(reappear_vel, 2),
                            "previous_velocity_ms": None,
                            "current_velocity_ms": None,
                            "previous_velocity_dt_s": None,
                            "current_velocity_dt_s": None,
                            "reason": "detection_reappear",
                            "window_start_fi": None, "window_end_fi": None,
                            "z_ptp_residual_m": None, "reversal_count": None,
                            "z_residual_mad_m": None, "z_xy_ratio": None,
                        })

                        _flush_depth_segment()
                        z_window = []
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None
                        prev_vel_mag = None

                    # Case 2: invalid->valid with NO prior valid
                    elif fi > 0 and not imm_prev_valid and last_valid_fi is None:
                        gap_from_start = fi
                        gap_s_from_start = curr_time - frames[0]["front_time_s"]
                        lc["is_reappearance"] = False
                        appearance_records.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr_time, 6),
                            "category": cat, "landmark_index": li,
                            "appearance_type": "initial_valid_after_start_gap",
                            "last_valid_frame_index": None,
                            "gap_frames": gap_from_start,
                            "gap_s": None,
                            "gap_frames_from_sequence_start": gap_from_start,
                            "gap_s_from_sequence_start": round(gap_s_from_start, 6),
                            "reappear_displacement_m": None,
                            "reappear_velocity_ms": None,
                            "reappearance_classification": None,
                            "reappearance_metrics_available": False,
                            "reappearance_metrics_unavailable_reason": "no_prior_valid_coordinate",
                            "cameras_prev": None,
                            "cameras_curr": list(lc["cameras_used"]),
                            "cameras_changed": None,
                            "observation_prev_by_view": obs_prev_by_view_tri,
                            "observation_curr_by_view": obs_curr_by_view_tri,
                            "observation_reappeared_views": obs_reappeared_views,
                            "has_observation_reappearance": has_obs_reappearance,
                            "qa_valid": lc["qa_valid"],
                            "template_eligible": lc["template_eligible"],
                        })

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

                            cams_changed = set(last_valid_xyz_cams) != set(lc["cameras_used"])
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
                                    "cameras_prev": list(last_valid_xyz_cams),
                                    "cameras_curr": list(lc["cameras_used"]),
                                    "cameras_changed": cams_changed,
                                    "last_valid_frame_index": last_valid_fi,
                                    "gap_frames": 0,
                                    "reappear_displacement_m": None,
                                    "reappear_velocity_ms": None,
                                    "previous_velocity_ms": round(prev_vel_mag, 2) if prev_vel_mag is not None else None,
                                    "current_velocity_ms": round(vel, 2),
                                    "previous_velocity_dt_s": round(prev_vel_dt, 6) if prev_vel_dt is not None else None,
                                    "current_velocity_dt_s": round(dt, 6),
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
                                    "cameras_prev": list(last_valid_xyz_cams),
                                    "cameras_curr": list(lc["cameras_used"]),
                                    "cameras_changed": cams_changed,
                                    "last_valid_frame_index": last_valid_fi,
                                    "gap_frames": 0,
                                    "reappear_displacement_m": None,
                                    "reappear_velocity_ms": None,
                                    "previous_velocity_ms": round(prev_vel_mag, 2) if prev_vel_mag is not None else None,
                                    "current_velocity_ms": round(vel, 2),
                                    "previous_velocity_dt_s": round(prev_vel_dt, 6) if prev_vel_dt is not None else None,
                                    "current_velocity_dt_s": round(dt, 6),
                                    "reason": "high_acceleration",
                                    "window_start_fi": None, "window_end_fi": None,
                                    "z_ptp_residual_m": None, "reversal_count": None,
                                    "z_residual_mad_m": None, "z_xy_ratio": None,
                                })

                            prev_vel_vec = vel_vec
                            prev_vel_dt = dt
                            prev_vel_fi = fi
                            prev_vel_mag = vel

                    # --- Z window for depth oscillation ---
                    torso_z = compute_torso_z(curr)
                    z_rel = float(curr_xyz[2]) - torso_z if torso_z is not None else float(curr_xyz[2])
                    xy_abs = float(np.linalg.norm(curr_xyz[:2]))

                    z_window.append((fi, z_rel, curr_xyz.copy(), curr_time, lc["cameras_used"], xy_abs))

                    last_valid_fi = fi
                    last_valid_xyz = curr_xyz.copy()
                    last_valid_time = curr_time
                    last_valid_xyz_cams = list(lc["cameras_used"])

                    if fi == 0 or is_reappear:
                        prev_vel_vec = None
                        prev_vel_dt = None
                        prev_vel_fi = None
                        prev_vel_mag = None

                _flush_depth_segment()

    depth_stats["windows_skipped_by_reason"] = dict(depth_stats["windows_skipped_by_reason"])
    return temporal_anomalies, appearance_records, observation_records, depth_stats


def _detect_depth_oscillation(sid, cat, li, frames, key, z_window,
                              temporal_anomalies, depth_stats):
    """Detect depth oscillation using detrended Z residuals on a single consecutive segment."""
    win_size = CONFIG["depth_osc_window_size"]
    min_rev = CONFIG["depth_osc_min_reversals"]
    min_ptp = CONFIG["depth_osc_min_ptp_m"]
    min_z_xy = CONFIG["depth_osc_min_z_xy_ratio"]
    min_rmad = CONFIG["depth_osc_min_residual_mad_m"]

    if len(z_window) < win_size:
        return

    n_possible = len(z_window) - win_size + 1
    depth_stats["windows_possible"] += n_possible

    flagged_centers = set()

    for ws in range(n_possible):
        win = z_window[ws:ws + win_size]

        # Check time continuity -- no large gaps
        max_dt = max(win[k+1][3] - win[k][3] for k in range(len(win) - 1))
        if max_dt > CONFIG["temporal_max_gap_s"]:
            depth_stats["windows_skipped"] += 1
            depth_stats["windows_skipped_by_reason"]["time_gap"] += 1
            continue

        # Check frame index continuity
        fi_consecutive = all(
            win[k+1][0] - win[k][0] == 1
            for k in range(len(win) - 1)
        )

        depth_stats["windows_evaluated"] += 1

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

        depth_stats["anomalies_detected"] += 1

        lc_center = frames[center_fi][key][li]
        flag = f"depth_oscillation_ptp{z_ptp:.3f}m_rev{reversals}"
        lc_center["qa_flags"].append(flag)
        # Only invalidate qa_valid if strong evidence
        if z_ptp > 0.05 and z_xy_ratio > 3.0:
            lc_center["qa_valid"] = False
            _invalidate_te(lc_center)

        # Center frame velocity and acceleration from window coordinates
        ci = center_idx
        center_prev_vel = None
        center_curr_vel = None
        center_accel = None
        center_accel_xy = None
        center_accel_z = None
        center_accel_available = False
        center_accel_reason = None
        center_prev_vel_dt = None
        center_curr_vel_dt = None

        # Previous velocity: (center - prev) / dt
        if ci >= 1:
            dt_prev = win[ci][3] - win[ci - 1][3]
            if dt_prev > 0:
                disp_prev = win[ci][2] - win[ci - 1][2]
                center_prev_vel = float(np.linalg.norm(disp_prev)) / dt_prev
                center_prev_vel_dt = dt_prev
                vel_vec_prev = disp_prev / dt_prev

        # Current velocity: (next - center) / dt
        if ci < len(win) - 1:
            dt_curr = win[ci + 1][3] - win[ci][3]
            if dt_curr > 0:
                disp_curr = win[ci + 1][2] - win[ci][2]
                center_curr_vel = float(np.linalg.norm(disp_curr)) / dt_curr
                center_curr_vel_dt = dt_curr
                vel_vec_curr = disp_curr / dt_curr

        # Acceleration from velocity change
        if ci >= 1 and ci < len(win) - 1 and center_prev_vel is not None and center_curr_vel is not None:
            dt_avg = (center_prev_vel_dt + center_curr_vel_dt) / 2
            if dt_avg > 0:
                accel_vec = (vel_vec_curr - vel_vec_prev) / dt_avg
                center_accel = float(np.linalg.norm(accel_vec))
                center_accel_xy = float(np.linalg.norm(accel_vec[:2]))
                center_accel_z = abs(float(accel_vec[2]))
                center_accel_available = True

        if not center_accel_available:
            center_accel_reason = "insufficient_window_neighbors"

        temporal_anomalies.append({
            "sample_id": sid, "frame_index": center_fi,
            "front_time_s": round(win[ci][3], 6),
            "category": cat, "landmark_index": li,
            "displacement_m": None, "displacement_xy_m": None,
            "displacement_z_m": None,
            "velocity_ms": round(center_curr_vel, 2) if center_curr_vel is not None else None,
            "velocity_xy_ms": None, "velocity_z_ms": None,
            "acceleration_ms2": round(center_accel, 2) if center_accel is not None else None,
            "acceleration_xy_ms2": round(center_accel_xy, 2) if center_accel_xy is not None else None,
            "acceleration_z_ms2": round(center_accel_z, 2) if center_accel_z is not None else None,
            "acceleration_available": center_accel_available,
            "acceleration_unavailable_reason": center_accel_reason,
            "dt_s": None,
            "cameras_prev": None,
            "cameras_curr": lc_center["cameras_used"],
            "cameras_changed": None,
            "last_valid_frame_index": None,
            "gap_frames": None,
            "reappear_displacement_m": None,
            "reappear_velocity_ms": None,
            "center_previous_velocity_ms": round(center_prev_vel, 2) if center_prev_vel is not None else None,
            "center_current_velocity_ms": round(center_curr_vel, 2) if center_curr_vel is not None else None,
            "center_velocity_ms": round(center_curr_vel, 2) if center_curr_vel is not None else None,
            "center_acceleration_ms2": round(center_accel, 2) if center_accel is not None else None,
            "previous_velocity_ms": round(center_prev_vel, 2) if center_prev_vel is not None else None,
            "current_velocity_ms": round(center_curr_vel, 2) if center_curr_vel is not None else None,
            "previous_velocity_dt_s": round(center_prev_vel_dt, 6) if center_prev_vel_dt is not None else None,
            "current_velocity_dt_s": round(center_curr_vel_dt, 6) if center_curr_vel_dt is not None else None,
            "reason": "depth_oscillation",
            "window_start_fi": win[0][0],
            "window_end_fi": win[-1][0],
            "z_ptp_residual_m": round(z_ptp, 4),
            "reversal_count": reversals,
            "z_residual_mad_m": round(z_mad, 4),
            "z_xy_ratio": round(z_xy_ratio, 2),
            "consecutive_frames": fi_consecutive,
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
    """Build balanced contact sheet items per spec:
    1. Each existing reason gets at least 1
    2. Each sample with anomalies gets at least 1
    3. reason x sample round-robin
    4. Remaining by severity
    5. No duplicate identification keys
    6. Max 3 per sample (soft limit)
    """
    max_total = CONFIG["contact_max_cells"]

    by_reason = defaultdict(list)
    by_sample = defaultdict(list)
    for ta in temporal_anomalies:
        by_reason[ta["reason"]].append(ta)
        by_sample[ta["sample_id"]].append(ta)

    # Sort each reason by severity
    for reason, items in by_reason.items():
        if reason == "depth_oscillation":
            items.sort(key=lambda x: x.get("z_ptp_residual_m") or 0, reverse=True)
        else:
            items.sort(key=lambda x: x.get("velocity_ms") or 0, reverse=True)

    selected = []
    seen_keys = set()
    sample_count = defaultdict(int)

    def _key(ta):
        return (ta["sample_id"], ta["frame_index"], ta["category"], ta["landmark_index"], ta["reason"])

    def _add(ta):
        k = _key(ta)
        if k in seen_keys:
            return False
        seen_keys.add(k)
        selected.append(ta)
        sample_count[ta["sample_id"]] += 1
        return True

    # Phase 1: each reason at least 1
    for reason in sorted(by_reason.keys()):
        for ta in by_reason[reason]:
            if _add(ta):
                break

    # Phase 2: each sample with anomalies at least 1
    for sid in PILOT_SAMPLE_IDS:
        if sid in by_sample and sample_count[sid] == 0:
            for ta in by_sample[sid]:
                if _add(ta):
                    break

    # Phase 3: reason x sample round-robin
    reasons = sorted(by_reason.keys())
    samples_with_anomalies = [s for s in PILOT_SAMPLE_IDS if s in by_sample]
    reason_idx = defaultdict(lambda: defaultdict(int))  # reason -> sample -> next_idx

    for _round in range(5):  # multiple rounds
        if len(selected) >= max_total:
            break
        for reason in reasons:
            if len(selected) >= max_total:
                break
            for sid in samples_with_anomalies:
                if len(selected) >= max_total:
                    break
                if sample_count[sid] >= 3:
                    continue
                items = [ta for ta in by_reason[reason] if ta["sample_id"] == sid]
                idx = reason_idx[reason][sid]
                while idx < len(items):
                    if _add(items[idx]):
                        reason_idx[reason][sid] = idx + 1
                        break
                    idx += 1
                    reason_idx[reason][sid] = idx

    # Phase 4: remaining by severity (allow >3 per sample if needed)
    all_sorted = sorted(temporal_anomalies,
                        key=lambda x: x.get("velocity_ms") or x.get("z_ptp_residual_m") or 0,
                        reverse=True)
    for ta in all_sorted:
        if len(selected) >= max_total:
            break
        _add(ta)

    # Convert to contact items with ACTUAL temporal anomaly values
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
            # Actual values from temporal anomaly record
            "displacement_m": ta.get("displacement_m"),
            "displacement_xy_m": ta.get("displacement_xy_m"),
            "displacement_z_m": ta.get("displacement_z_m"),
            "velocity_ms": ta.get("velocity_ms"),
            "velocity_xy_ms": ta.get("velocity_xy_ms"),
            "velocity_z_ms": ta.get("velocity_z_ms"),
            "acceleration_ms2": ta.get("acceleration_ms2"),
            "acceleration_available": ta.get("acceleration_available"),
            "acceleration_unavailable_reason": ta.get("acceleration_unavailable_reason"),
            "dt_s": ta.get("dt_s"),
            "z_ptp_residual_m": ta.get("z_ptp_residual_m"),
            "z_residual_mad_m": ta.get("z_residual_mad_m"),
            "z_xy_ratio": ta.get("z_xy_ratio"),
            "reversal_count": ta.get("reversal_count"),
            "gap_frames": ta.get("gap_frames"),
            "gap_s": ta.get("gap_s") if ta.get("gap_s") is not None else ta.get("dt_s"),
            "reappear_displacement_m": ta.get("reappear_displacement_m"),
            "reappear_velocity_ms": ta.get("reappear_velocity_ms"),
            "cameras_prev": ta.get("cameras_prev"),
            "cameras_curr": ta.get("cameras_curr"),
            "cameras_changed": ta.get("cameras_changed"),
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
                   appearance_records, observation_records, overlay_counts,
                   contact_paths, contact_index_rows, depth_stats):
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

    # 13. Overlay frame counts (decode-verified)
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

    # 15. ALL invalid->valid transitions = 1,161 appearance events
    n_all_transitions = 0
    n_with_prior = 0
    n_no_prior = 0
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
                        n_all_transitions += 1
                        has_prior = any(frames[k][key][li]["triangulation_valid"] for k in range(fi-1))
                        if has_prior:
                            n_with_prior += 1
                        else:
                            n_no_prior += 1
    chk(len(appearance_records) == n_all_transitions,
        f"all_transitions_recorded: records={len(appearance_records)} transitions={n_all_transitions}")

    # 16. Reappearance vs initial appearance split
    n_reappear_type = sum(1 for ar in appearance_records if ar["appearance_type"] == "triangulation_reappearance")
    n_initial_type = sum(1 for ar in appearance_records if ar["appearance_type"] == "initial_valid_after_start_gap")
    chk(n_reappear_type == n_with_prior,
        f"reappearance_split: reappear={n_reappear_type}=={n_with_prior} initial={n_initial_type}=={n_no_prior}")

    # 17. Reappearances have gap/displacement/velocity; initial appearances do NOT have displacement
    n_reappear_missing = sum(1 for ar in appearance_records
                             if ar["appearance_type"] == "triangulation_reappearance" and
                             (ar.get("gap_frames") is None or ar.get("reappear_displacement_m") is None or
                              ar.get("reappear_velocity_ms") is None))
    chk(n_reappear_missing == 0, f"reappear_has_gap_disp_vel (missing={n_reappear_missing})")

    n_initial_has_disp = sum(1 for ar in appearance_records
                             if ar["appearance_type"] == "initial_valid_after_start_gap" and
                             ar.get("reappear_displacement_m") is not None)
    chk(n_initial_has_disp == 0, f"initial_no_displacement (violations={n_initial_has_disp})")

    # 18. is_reappearance flag matches appearance_type in landmark data
    n_reappear_flags = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                           for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                           for lm in fr[key] if lm.get("is_reappearance"))
    chk(n_reappear_flags == n_reappear_type,
        f"reappear_flags_match: flags={n_reappear_flags} records={n_reappear_type}")

    # 19. Reappearance NOT template_eligible
    n_reappear_te = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                        for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                        for lm in fr[key] if lm.get("is_reappearance") and lm["template_eligible"])
    chk(n_reappear_te == 0, f"reappear_not_te (violations={n_reappear_te})")

    # 20. cameras_changed matches actual camera set comparison
    n_cam_mismatch = 0
    for ar in appearance_records:
        if ar["appearance_type"] == "triangulation_reappearance":
            prev_cams = ar.get("cameras_prev")
            curr_cams = ar.get("cameras_curr")
            expected_changed = set(prev_cams) != set(curr_cams) if prev_cams and curr_cams else None
            if expected_changed is not None and ar.get("cameras_changed") != expected_changed:
                n_cam_mismatch += 1
    chk(n_cam_mismatch == 0, f"cameras_changed_actual (mismatch={n_cam_mismatch})")

    # 21. Verify reappearance gap/displacement by independent recalculation
    n_recompute_mismatch = 0
    for ar in appearance_records:
        if ar["appearance_type"] != "triangulation_reappearance":
            continue
        sid = ar["sample_id"]
        fi = ar["frame_index"]
        cat_key_map = {"pose": "pose_landmarks_3d", "left_hand": "left_hand_landmarks_3d", "right_hand": "right_hand_landmarks_3d"}
        kk = cat_key_map.get(ar["category"])
        if not kk:
            continue
        lv_fi = ar["last_valid_frame_index"]
        if lv_fi is None:
            continue
        curr_lm = all_results[sid][fi][kk][ar["landmark_index"]]
        prev_lm = all_results[sid][lv_fi][kk][ar["landmark_index"]]
        if curr_lm["xyz"] and prev_lm["xyz"]:
            recomp_disp = float(np.linalg.norm(np.array(curr_lm["xyz"]) - np.array(prev_lm["xyz"])))
            stored_disp = ar["reappear_displacement_m"]
            if abs(recomp_disp - stored_disp) > 0.001:
                n_recompute_mismatch += 1
    chk(n_recompute_mismatch == 0, f"reappear_disp_recompute (mismatch={n_recompute_mismatch})")

    # 22. large_jump consistent across CSV/JSON/manifest
    n_large_jump = sum(1 for ar in appearance_records
                       if ar.get("reappearance_classification") == "large_jump")
    n_large_jump_lm = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                          for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                          for lm in fr[key] if lm.get("reappearance_classification") == "large_jump")
    chk(n_large_jump == n_large_jump_lm,
        f"large_jump_consistent: records={n_large_jump} landmarks={n_large_jump_lm}")

    # 23. No old-style depth_oscillation flags
    n_old_osc = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                    for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                    for lm in fr[key] for f in lm["qa_flags"]
                    if f.startswith("depth_oscillation_rev"))
    chk(n_old_osc == 0, f"no_old_depth_osc_flags (found={n_old_osc})")

    # 24. Depth oscillation amplitude reasonable
    depth_osc_items = [ta for ta in temporal_anomalies if ta["reason"] == "depth_oscillation"]
    if depth_osc_items:
        max_amp = max(ta.get("z_ptp_residual_m") or 0 for ta in depth_osc_items)
        chk(max_amp < 5.0, f"depth_osc_amplitude_meters (max={max_amp:.3f}m)")
    else:
        chk(True, "depth_osc_amplitude_meters (none found)")

    # 25. All valid segments processed for depth oscillation
    chk(depth_stats["windows_possible"] == depth_stats["windows_evaluated"] + depth_stats["windows_skipped"],
        f"depth_windows_accounting: possible={depth_stats['windows_possible']} "
        f"eval={depth_stats['windows_evaluated']} skip={depth_stats['windows_skipped']}")

    # 26. consecutive_frames verified from actual frame indices
    n_consec_mismatch = 0
    for ta in depth_osc_items:
        ws_fi = ta.get("window_start_fi")
        we_fi = ta.get("window_end_fi")
        if ws_fi is not None and we_fi is not None:
            expected_consec = (we_fi - ws_fi) == (CONFIG["depth_osc_window_size"] - 1)
            if ta.get("consecutive_frames") != expected_consec:
                n_consec_mismatch += 1
    chk(n_consec_mismatch == 0,
        f"depth_consec_frames_verified (mismatch={n_consec_mismatch})")

    # 27. Temporal reason-specific required fields
    n_missing_fields = 0
    for ta in temporal_anomalies:
        reason = ta["reason"]
        required = TEMPORAL_REQUIRED_FIELDS.get(reason, [])
        for field in required:
            if ta.get(field) is None:
                n_missing_fields += 1
    chk(n_missing_fields == 0, f"temporal_required_fields (missing={n_missing_fields})")

    # 28. Acceleration unavailable has reason
    n_accel_no_reason = sum(1 for ta in temporal_anomalies
                            if ta.get("acceleration_available") is False and
                            ta.get("acceleration_unavailable_reason") is None)
    chk(n_accel_no_reason == 0, f"accel_unavail_has_reason (missing={n_accel_no_reason})")

    # 29. All valid landmarks have v_fs selection_reason (NOT v_gs)
    n_bad_reason = 0
    n_gs_format = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"]:
                        sr = lm.get("selection_reason", "")
                        if "v_gs" in sr:
                            n_gs_format += 1
                        elif "v_fs" not in sr:
                            n_bad_reason += 1
    chk(n_bad_reason == 0 and n_gs_format == 0,
        f"selection_reason_v_fs_only (bad={n_bad_reason} v_gs={n_gs_format})")

    # 30. Best candidate = max final_score
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

    # 31. All feasible candidates have score components
    n_missing_sc = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                       for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                       for lm in fr[key] for cand in lm.get("candidates", [])
                       if cand["ok"] and any(cand.get(k) is None for k in ["gs", "urp", "evp", "tas", "bcp", "tcp", "csp", "fs"]))
    chk(n_missing_sc == 0, f"all_candidates_have_scores (missing={n_missing_sc})")

    # 32. Contact sheet: all 9 samples with anomalies included
    anomaly_samples = set(ta["sample_id"] for ta in temporal_anomalies)
    contact_samples = set(row["sample_id"] for row in contact_index_rows) if contact_index_rows else set()
    missing_samples = anomaly_samples - contact_samples
    chk(len(missing_samples) == 0,
        f"contact_all_anomaly_samples (missing={len(missing_samples)}: {missing_samples})")

    # 33. Contact sheet: all existing reason types included
    anomaly_reasons = set(ta["reason"] for ta in temporal_anomalies)
    contact_reasons = set(row["anomaly_reason"] for row in contact_index_rows) if contact_index_rows else set()
    missing_reasons = anomaly_reasons - contact_reasons
    chk(len(missing_reasons) == 0,
        f"contact_all_reasons (missing={len(missing_reasons)}: {missing_reasons})")

    # 34. Contact index no duplicate keys
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

    # 35. Contact index values match source temporal anomaly
    n_idx_mismatch = 0
    ta_lookup = {}
    for ta in temporal_anomalies:
        tk = (ta["sample_id"], ta["frame_index"], ta["category"], ta["landmark_index"], ta["reason"])
        ta_lookup[tk] = ta
    for row in (contact_index_rows or []):
        tk = (row["sample_id"], row["sample_frame_index"], row["category"],
              row["landmark_index"], row["anomaly_reason"])
        src = ta_lookup.get(tk)
        if src:
            for field in ["displacement_m", "velocity_ms", "acceleration_ms2",
                          "z_ptp_residual_m", "z_residual_mad_m"]:
                rv = row.get(field)
                sv = src.get(field)
                if rv is not None and sv is not None and abs(rv - sv) > 0.01:
                    n_idx_mismatch += 1
                elif (rv is None) != (sv is None):
                    n_idx_mismatch += 1
    chk(n_idx_mismatch == 0, f"contact_index_values_match (mismatch={n_idx_mismatch})")

    # 36. Contact temporal image cell count matches index row count
    n_temporal_index = len([r for r in (contact_index_rows or []) if r.get("sheet_type") == "temporal"])
    contact_temporal_path = OUT_DIR / "contact_sheets" / "contact_temporal.png"
    if contact_temporal_path.exists():
        ct_img = cv2.imread(str(contact_temporal_path))
        if ct_img is not None:
            tw, th = CONFIG["contact_thumb_size"]
            ct_cols = ct_img.shape[1] // tw
            ct_rows = ct_img.shape[0] // th
            ct_cells = ct_cols * ct_rows
            # Actual cells = min(selected items, max_cells)
            chk(n_temporal_index <= ct_cells,
                f"contact_cells_vs_index: index={n_temporal_index} image_grid={ct_cells}")
        else:
            chk(False, "contact_cells_vs_index: could not read image")
    else:
        chk(False, "contact_cells_vs_index: image missing")

    # 37. tri_valid count preserved from v1.2
    n_tri = sum(1 for sid in PILOT_SAMPLE_IDS for fr in all_results[sid]
                for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                for lm in fr[key] if lm["triangulation_valid"])
    chk(n_tri == 70943, f"tri_valid_preserved={n_tri}==70943")

    # 38. Overlay decode verification
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            ov_path = OUT_DIR / "overlays" / f"reproj_{short}_{v.lower()}.mp4"
            if ov_path.exists():
                try:
                    con = av.open(str(ov_path))
                    st = con.streams.video[0]
                    decoded = sum(1 for _ in con.decode(st))
                    con.close()
                    chk(decoded == expected,
                        f"overlay_decode_{short}_{v.lower()}: {decoded}=={expected}")
                except Exception as e:
                    chk(False, f"overlay_decode_{short}_{v.lower()}: error={e}")

    # 39. Observation transitions match independent recount from synced data
    n_obs_recount = 0
    for sid in PILOT_SAMPLE_IDS:
        sf = synced[sid]
        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            lm_type = "pose" if cat == "pose" else "hand"
            hs = {"left_hand": "left", "right_hand": "right"}.get(cat)
            n_lm = len(all_results[sid][0][key])
            for li in range(n_lm):
                for v in VIEWS:
                    for fi in range(1, len(sf)):
                        pf = sf[fi - 1]["views"].get(v)
                        cf = sf[fi]["views"].get(v)
                        prev_obs = _has_obs_2d(pf, lm_type, li, hs) if pf else False
                        curr_obs = _has_obs_2d(cf, lm_type, li, hs) if cf else False
                        if not prev_obs and curr_obs:
                            n_obs_recount += 1
    chk(len(observation_records) == n_obs_recount,
        f"obs_transitions_recount: records={len(observation_records)} recount={n_obs_recount}")

    # 40. 42 initial appearances all have unavailable reason
    initial_apps = [ar for ar in appearance_records if ar["appearance_type"] == "initial_valid_after_start_gap"]
    n_init_no_reason = sum(1 for ar in initial_apps
                           if ar.get("reappearance_metrics_unavailable_reason") is None)
    chk(n_init_no_reason == 0,
        f"initial_appearance_has_reason (missing={n_init_no_reason})")

    # 41. 22 high_acceleration have previous/current velocity
    ha_items = [ta for ta in temporal_anomalies if ta["reason"] == "high_acceleration"]
    n_ha_missing = sum(1 for ta in ha_items
                       if ta.get("previous_velocity_ms") is None or ta.get("current_velocity_ms") is None)
    chk(n_ha_missing == 0,
        f"high_accel_has_velocities (missing={n_ha_missing} of {len(ha_items)})")

    # 42. 195 depth anomaly center acceleration computed
    n_depth_no_accel = sum(1 for ta in depth_osc_items
                           if not ta.get("acceleration_available", False))
    chk(n_depth_no_accel == 0,
        f"depth_center_accel_computed (unavailable={n_depth_no_accel} of {len(depth_osc_items)})")

    # 43. Depth center accel independently verified from coordinates
    n_depth_accel_mismatch = 0
    for ta in depth_osc_items:
        sid = ta["sample_id"]
        cfi = ta["frame_index"]
        cat = ta["category"]
        li = ta["landmark_index"]
        ws_fi = ta["window_start_fi"]
        we_fi = ta["window_end_fi"]
        cat_key = {"pose": "pose_landmarks_3d", "left_hand": "left_hand_landmarks_3d",
                    "right_hand": "right_hand_landmarks_3d"}[cat]
        # Find center's neighbors in the window
        win_fis = list(range(ws_fi, we_fi + 1))
        # Collect valid frame indices
        valid_fis = [f for f in win_fis
                     if all_results[sid][f][cat_key][li]["triangulation_valid"]]
        if cfi in valid_fis:
            ci_idx = valid_fis.index(cfi)
            if ci_idx >= 1 and ci_idx < len(valid_fis) - 1:
                prev_fi = valid_fis[ci_idx - 1]
                next_fi = valid_fis[ci_idx + 1]
                xyz_prev = np.array(all_results[sid][prev_fi][cat_key][li]["xyz"])
                xyz_curr = np.array(all_results[sid][cfi][cat_key][li]["xyz"])
                xyz_next = np.array(all_results[sid][next_fi][cat_key][li]["xyz"])
                t_prev = all_results[sid][prev_fi]["front_time_s"]
                t_curr = all_results[sid][cfi]["front_time_s"]
                t_next = all_results[sid][next_fi]["front_time_s"]
                dt1 = t_curr - t_prev
                dt2 = t_next - t_curr
                if dt1 > 0 and dt2 > 0:
                    v1 = (xyz_curr - xyz_prev) / dt1
                    v2 = (xyz_next - xyz_curr) / dt2
                    recomp_accel = float(np.linalg.norm((v2 - v1) / ((dt1 + dt2) / 2)))
                    stored_accel = ta.get("center_acceleration_ms2")
                    if stored_accel is not None and abs(recomp_accel - stored_accel) > 0.1:
                        n_depth_accel_mismatch += 1
    chk(n_depth_accel_mismatch == 0,
        f"depth_center_accel_recompute (mismatch={n_depth_accel_mismatch})")

    # 44. No stale velocity across timestamp gaps
    n_stale = 0
    for ta in temporal_anomalies:
        if ta["reason"] == "high_acceleration":
            prev_vel_dt = ta.get("previous_velocity_dt_s")
            curr_vel_dt = ta.get("current_velocity_dt_s")
            max_gap = CONFIG["temporal_max_gap_s"]
            if prev_vel_dt is not None and prev_vel_dt > max_gap:
                n_stale += 1
            if curr_vel_dt is not None and curr_vel_dt > max_gap:
                n_stale += 1
    chk(n_stale == 0, f"no_stale_velocity_across_gaps (stale={n_stale})")

    return results


# ================================================================
#  Main
# ================================================================

def main():
    print("Step 6B v1.2.3 PILOT: Multi-view 3D Reconstruction - S001_P001 R03\n")
    print("  Status: STEP6B_V1_2_TEMPLATE_FILTER_PASS / STEP6B_V1_2_2_TEMPORAL_CORE_PASS\n")

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
    temporal_anomalies, appearance_records, observation_records, depth_stats = apply_temporal_qa(all_results, synced)
    by_reason = defaultdict(int)
    for ta in temporal_anomalies:
        by_reason[ta["reason"]] += 1
    for r, c in sorted(by_reason.items()):
        print(f"    {r}: {c}")
    print(f"    Total temporal anomalies: {len(temporal_anomalies)}")
    n_reappear = sum(1 for ar in appearance_records if ar["appearance_type"] == "triangulation_reappearance")
    n_initial = sum(1 for ar in appearance_records if ar["appearance_type"] == "initial_valid_after_start_gap")
    print(f"    Appearance records: {len(appearance_records)} (reappear={n_reappear}, initial={n_initial})")
    print(f"    Depth stats: segments={depth_stats['segments_total']} windows_possible={depth_stats['windows_possible']} "
          f"evaluated={depth_stats['windows_evaluated']} skipped={depth_stats['windows_skipped']} "
          f"anomalies={depth_stats['anomalies_detected']}")
    # cameras_changed stats for reappearance
    reappear_records = [ar for ar in appearance_records if ar["appearance_type"] == "triangulation_reappearance"]
    n_cam_changed = sum(1 for ar in reappear_records if ar.get("cameras_changed") is True)
    n_cam_same = sum(1 for ar in reappear_records if ar.get("cameras_changed") is False)
    print(f"    Reappearance cameras_changed: changed={n_cam_changed} same={n_cam_same}")
    n_obs_reappear = sum(1 for o in observation_records if o["observation_type"] == "observation_reappearance")
    n_obs_initial = sum(1 for o in observation_records if o["observation_type"] == "initial_observation_after_start_gap")
    obs_by_view = defaultdict(int)
    for o in observation_records:
        obs_by_view[o["view"]] += 1
    print(f"    Observation transitions: {len(observation_records)} (reappear={n_obs_reappear}, initial={n_obs_initial})")
    print(f"    Observation by view: " + " ".join(f"{v}={obs_by_view.get(v,0)}" for v in VIEWS))

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
                "version": "v1.2.3_pilot",
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

    # Build contact sheet index with ACTUAL values from temporal anomaly records
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

        cam_prev = item.get("cameras_prev")
        cam_curr = item.get("cameras_curr")
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
            "displacement_m": item.get("displacement_m"),
            "displacement_xy_m": item.get("displacement_xy_m"),
            "displacement_z_m": item.get("displacement_z_m"),
            "velocity_ms": item.get("velocity_ms"),
            "velocity_xy_ms": item.get("velocity_xy_ms"),
            "velocity_z_ms": item.get("velocity_z_ms"),
            "acceleration_ms2": item.get("acceleration_ms2"),
            "acceleration_available": item.get("acceleration_available"),
            "acceleration_unavailable_reason": item.get("acceleration_unavailable_reason"),
            "dt_s": item.get("dt_s"),
            "z_ptp_residual_m": item.get("z_ptp_residual_m"),
            "z_residual_mad_m": item.get("z_residual_mad_m"),
            "z_xy_ratio": item.get("z_xy_ratio"),
            "reversal_count": item.get("reversal_count"),
            "gap_frames": item.get("gap_frames"),
            "gap_s": item.get("gap_s"),
            "reappear_displacement_m": item.get("reappear_displacement_m"),
            "reappear_velocity_ms": item.get("reappear_velocity_ms"),
            "cameras_prev": ",".join(cam_prev) if isinstance(cam_prev, list) else cam_prev,
            "cameras_curr": ",".join(cam_curr) if isinstance(cam_curr, list) else cam_curr,
            "cameras_changed": item.get("cameras_changed"),
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

    # Appearance/Reappearance CSV (all invalid->valid transitions)
    appear_csv = OUT_DIR / "appearance_records.csv"
    if appearance_records:
        appear_csv_rows = []
        for ar in appearance_records:
            row = dict(ar)
            if isinstance(row.get("cameras_prev"), list):
                row["cameras_prev"] = ",".join(row["cameras_prev"])
            if isinstance(row.get("cameras_curr"), list):
                row["cameras_curr"] = ",".join(row["cameras_curr"])
            appear_csv_rows.append(row)
        all_ar_keys = list(dict.fromkeys(k for row in appear_csv_rows for k in row.keys()))
        with open(appear_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_ar_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(appear_csv_rows)
    print(f"  appearance_records.csv: {len(appearance_records)} rows")

    # Observation appearance CSV
    obs_csv = OUT_DIR / "observation_appearance_records.csv"
    if observation_records:
        with open(obs_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=observation_records[0].keys())
            w.writeheader()
            w.writerows(observation_records)
    print(f"  observation_appearance_records.csv: {len(observation_records)} rows")

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
        "step": "6B_pilot_v1.2.3",
        "date": datetime.datetime.now().isoformat(),
        "status": "STEP6B_V1_2_3_TEMPORAL_QA_COMPLETE",
        "template_filter_status": "STEP6B_V1_2_TEMPLATE_FILTER_PASS",
        "temporal_core_status": "STEP6B_V1_2_2_TEMPORAL_CORE_PASS",
        "full_45_status": "FULL_45_BLOCKED_PENDING_EXTERNAL_REVIEW_V1_2_3",
        "triangulation_method": "DLT 2-pass multi-candidate scoring (unchanged from v1.2)",
        "temporal_qa": "v1.2.3: observation reappearance, center acceleration, velocity metadata, real stub assertions",
        "smoothing": "DISABLED",
        "config": CONFIG,
        "pass2_reselections": total_changed,
        "depth_oscillation_stats": depth_stats,
        "appearance_records_total": len(appearance_records),
        "reappearance_count": n_reappear,
        "initial_appearance_count": n_initial,
        "reappear_cameras_changed": n_cam_changed,
        "reappear_cameras_same": n_cam_same,
        "observation_records_total": len(observation_records),
        "observation_reappearance_count": n_obs_reappear,
        "observation_initial_count": n_obs_initial,
        "observation_by_view": {v: obs_by_view.get(v, 0) for v in VIEWS},
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
                                       temporal_anomalies, appearance_records,
                                       observation_records, overlay_counts,
                                       contact_paths, contact_index_rows, depth_stats)
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
    rpt.append("# Step 6B v1.2.3 Pilot Report -- Multi-view 3D Reconstruction\n")
    rpt.append("## Status\n")
    rpt.append("- `STEP6B_V1_2_TEMPLATE_FILTER_PASS`")
    rpt.append("- `STEP6B_V1_2_2_TEMPORAL_CORE_PASS`")
    rpt.append("- `STEP6B_V1_2_3_TEMPORAL_QA_COMPLETE`")
    rpt.append("- `FULL_45_BLOCKED_PENDING_EXTERNAL_REVIEW_V1_2_3`\n")
    rpt.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    rpt.append(f"**Script**: `Tools/s001_step6b_triangulate_pilot_v1_2_3.py`")
    rpt.append(f"**Scope**: 9 R03 samples x 3 views\n")

    rpt.append("## Changes from v1.2.2\n")
    rpt.append("- **Observation reappearance**: per-view 2D observation state transitions tracked")
    rpt.append("- **Initial appearance metadata**: reappearance_metrics_available=False, gap_from_sequence_start")
    rpt.append("- **High acceleration velocity**: previous_velocity_ms, current_velocity_ms, dt metadata")
    rpt.append("- **Depth center acceleration**: computed from window coordinates (no more acceleration_unavailable_reason)")
    rpt.append("- **Real stub assertions**: large_jump_count, consecutive_frames, contact_cells_vs_index now FAIL-able")
    rpt.append("- **QA invalidation breakdown**: strong depth, new qa_invalid, already invalid counts\n")
    rpt.append("### Preserved from v1.2.2")
    rpt.append("- Triangulation core, 2-pass candidate scoring")
    rpt.append("- All-segment depth oscillation, appearance/reappearance split, actual cameras_changed")
    rpt.append("- Contact sheet balancing, contact index actual values")
    rpt.append(f"- template_eligible threshold: used_reproj_max <= {TEMPLATE_REPROJ_THRESHOLD_PX}px")
    rpt.append(f"- tri_valid: {n_tri} (unchanged)")
    rpt.append(f"- Pass 2 re-selections: {total_changed} (unchanged)\n")

    rpt.append("## Summary\n")
    rpt.append(f"- Frames: 1001, Landmarks: 75075")
    rpt.append(f"- tri_valid: {n_tri}, qa_valid: {n_qa}, template_eligible: {n_te}")
    rpt.append(f"- High reproj (>20px) AND te=True: {n_high_te}")
    rpt.append(f"- Pass 2 re-selections: {total_changed}")
    rpt.append(f"- Bone outliers: {len(bone_outliers)}")
    rpt.append(f"- Temporal anomalies: {len(temporal_anomalies)}")
    rpt.append(f"- Appearance records: {len(appearance_records)} (reappear={n_reappear}, initial={n_initial})\n")

    rpt.append("## Temporal Anomalies by Reason\n")
    rpt.append("| Reason | Count |")
    rpt.append("|--------|-------|")
    for r, c in sorted(by_reason.items()):
        rpt.append(f"| {r} | {c} |")
    rpt.append("")

    rpt.append("## Depth Oscillation Statistics\n")
    rpt.append(f"- depth_valid_segments_total: {depth_stats['segments_total']}")
    rpt.append(f"- depth_windows_possible: {depth_stats['windows_possible']}")
    rpt.append(f"- depth_windows_evaluated: {depth_stats['windows_evaluated']}")
    rpt.append(f"- depth_windows_skipped: {depth_stats['windows_skipped']}")
    rpt.append(f"- depth_windows_skipped_by_reason: {depth_stats['windows_skipped_by_reason']}")
    rpt.append(f"- depth_anomalies_detected: {depth_stats['anomalies_detected']}\n")

    rpt.append("## Appearance/Reappearance Statistics\n")
    rpt.append(f"- Total invalid->valid transitions: {len(appearance_records)}")
    rpt.append(f"- triangulation_reappearance: {n_reappear}")
    rpt.append(f"- initial_valid_after_start_gap: {n_initial}")
    if reappear_records:
        by_class = defaultdict(int)
        for rr in reappear_records:
            by_class[rr["reappearance_classification"]] += 1
        for cls, cnt in sorted(by_class.items()):
            rpt.append(f"- {cls}: {cnt}")
        rpt.append(f"- cameras_changed=True: {n_cam_changed}")
        rpt.append(f"- cameras_changed=False: {n_cam_same}")

    rpt.append(f"\n## Observation Reappearance Statistics\n")
    rpt.append(f"- Total observation transitions (invalid->valid per view): {len(observation_records)}")
    rpt.append(f"- observation_reappearance (prior observation exists): {n_obs_reappear}")
    rpt.append(f"- initial_observation_after_start_gap: {n_obs_initial}")
    for v in VIEWS:
        rpt.append(f"- {v}: {obs_by_view.get(v, 0)}")
    rpt.append("")

    # QA invalidation breakdown (Section 8)
    # strong depth: z_ptp > threshold AND z_xy_ratio > threshold (the conditions that set qa_valid=False)
    n_depth_strong = sum(1 for ta in temporal_anomalies
                         if ta["reason"] == "depth_oscillation"
                         and ta.get("z_ptp_residual_m", 0) > CONFIG.get("depth_oscillation_z_ptp_threshold_m", 0.03)
                         and ta.get("z_xy_ratio", 0) > CONFIG.get("depth_oscillation_z_xy_ratio_threshold", 2.0))
    # Landmarks with depth_oscillation qa_flag AND qa_invalid
    depth_flagged_invalid = set()
    depth_flagged_already_invalid = set()
    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for li, lm in enumerate(fr[key]):
                    has_depth_flag = any("depth_oscillation" in f for f in lm["qa_flags"])
                    if has_depth_flag and not lm["qa_valid"]:
                        lm_id = (sid, fi, key, li)
                        # Check if this landmark would have been invalid even without depth flag
                        other_flags = [f for f in lm["qa_flags"] if "depth_oscillation" not in f]
                        if other_flags:
                            depth_flagged_already_invalid.add(lm_id)
                        else:
                            depth_flagged_invalid.add(lm_id)
    n_new_qa_invalid = len(depth_flagged_invalid)
    n_already_invalid = len(depth_flagged_already_invalid)
    n_total_depth_and_invalid = n_new_qa_invalid + n_already_invalid
    rpt.append(f"\n## QA Invalidation Breakdown\n")
    rpt.append(f"- Strong depth oscillation conditions met: {n_depth_strong}")
    rpt.append(f"- New qa_invalid from depth_oscillation: {n_new_qa_invalid}")
    rpt.append(f"- Already invalid (other flags): {n_already_invalid}")
    rpt.append(f"- Total landmarks with depth flag + qa_invalid: {n_total_depth_and_invalid}")
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

    rpt_path = OUT_DIR / "STEP6B_PILOT_V1_2_3_REPORT.md"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))

    # -- ZIP --
    print("\n  Creating delivery ZIP...")
    zip_path = PROJECT_ROOT / "WordsInSilence_step6b_v1_2_3_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "Tools/s001_step6b_triangulate_pilot_v1_2_3.py")
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(PROC_DIR)))
    zip_sz = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP: {zip_path.name} ({zip_sz:.1f} MB)")

    print(f"\n  {'=' * 60}")
    print(f"  STEP 6B v1.2.3 PILOT SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  tri_valid: {n_tri}/75075")
    print(f"  qa_valid: {n_qa}/75075")
    print(f"  template_eligible: {n_te}/75075")
    print(f"  Bone outliers: {len(bone_outliers)}")
    print(f"  Temporal anomalies: {len(temporal_anomalies)}")
    print(f"  Appearance records: {len(appearance_records)} (reappear={n_reappear} initial={n_initial})")
    print(f"  Depth segments: {depth_stats['segments_total']} windows: {depth_stats['windows_possible']} anomalies: {depth_stats['anomalies_detected']}")
    print(f"  Observation transitions: {len(observation_records)} (reappear={n_obs_reappear} initial={n_obs_initial})")
    print(f"  Observation by view: " + " ".join(f"{v}={obs_by_view.get(v,0)}" for v in VIEWS))
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
