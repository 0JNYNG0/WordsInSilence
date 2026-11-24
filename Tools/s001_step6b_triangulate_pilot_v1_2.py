"""
Step 6B v1.2 PILOT: Multi-view 3D Reconstruction -- S001_P001 R03 (9 samples)

Changes from v1.1:
  - template_eligible requires used_reproj_max_px <= 20px (was: any qa_valid)
  - template_eligible_by_view: per-camera reproj check
  - 2-pass candidate scoring: Pass 1 geometry -> bone medians; Pass 2 adds
    bone_consistency_penalty, temporal_continuity_penalty, camera_switch_penalty
  - All 8 score components stored per candidate
  - Actual acceleration computation in temporal QA
  - Depth oscillation via Z velocity sign reversals (not sequence-wide range)
  - Pose landmark group exclusion diagnostics (face/sign_upper/pose_hand/lower_body)
  - Balanced per-sample contact sheet sampling
  - Real assertion conditions (not chk(True, ...))
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
OUT_DIR = PROC_DIR / "landmarks_3d_v1" / "pilot_v1_2"

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
    "depth_z_reversal_window": 5,
    "depth_z_reversal_min_count": 3,
    "depth_z_reversal_min_amplitude_m": 0.01,
    "apply_smoothing": False,
    "contact_thumb_size": (384, 216),
    "contact_max_cells": 24,
    "contact_max_per_sample": 3,
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

HAND_CONNECTIONS_VIS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

POSE_CONNECTIONS_VIS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
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
    """Evaluate one candidate -- geometry only (Pass 1)."""
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
    per_cam_errors = {}
    for v in all_obs_views:
        rp = reproject_point(X3, intrinsics[v]["K"], extrinsics[v]["R"],
                             extrinsics[v]["t"], intrinsics[v]["dist"])
        if rp is not None:
            err = float(np.linalg.norm(rp - np.array(obs[v]["raw_px"])))
            ev["per_camera"][v] = {"reproj_px": rp.tolist(), "error": round(err, 2)}
            per_cam_errors[v] = err
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

    # Geometry score components
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
    # Pass 2 penalties initialized to 0
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

    result["candidates"] = [{
        "v": ev["views"], "ok": ev["feasible"],
        "gs": ev["geometry_score"], "ur": ev["used_reproj_mean"],
        "um": ev["used_reproj_max"], "vr": ev["validation_reproj_mean"],
        "a": ev["angle_deg"], "d": ev["depth_m"], "r": ev["reason"],
        "urp": ev["used_reproj_penalty"], "evp": ev["excluded_view_validation_penalty"],
        "tas": ev["triangulation_angle_score"],
        "bcp": ev["bone_consistency_penalty"], "tcp": ev["temporal_continuity_penalty"],
        "csp": ev["camera_switch_penalty"], "fs": ev["final_score"],
    } for ev in all_candidates]

    # Select best by geometry score for pass 1
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
    # Store all_candidates for pass 2 re-scoring
    result["_all_candidates"] = all_candidates
    result["_obs"] = obs
    result["_obs_views"] = obs_views

    return result


def _apply_best_candidate(result, best, obs_views):
    """Apply best candidate selection to result dict."""
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

    # QA starts as valid, bone/temporal post-processing may invalidate
    result["qa_valid"] = True

    # template_eligible: qa_valid AND used_reproj_max <= 20px
    result["template_eligible"] = (
        result["used_reproj_max_px"] is not None and
        result["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX
    )

    # template_eligible_by_view: per-camera check
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
    """Compute robust bone medians/MAD from Pass 1 high-confidence results."""
    bone_data = defaultdict(list)

    for fr in frames_3d:
        for side in ["left", "right"]:
            key = f"{side}_hand_landmarks_3d"
            hand = fr[key]
            for bi, bj, bname in HAND_BONES:
                li = hand[bi]
                lj = hand[bj]
                if (li["triangulation_valid"] and lj["triangulation_valid"] and
                    li.get("used_reproj_max_px") is not None and
                    lj.get("used_reproj_max_px") is not None and
                    li["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX and
                    lj["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX):
                    bl = float(np.linalg.norm(
                        np.array(li["xyz"]) - np.array(lj["xyz"])))
                    bone_data[f"{side}_{bname}"].append(bl)

        pose = fr["pose_landmarks_3d"]
        for bi, bj, bname in POSE_BONES:
            li = pose[bi]
            lj = pose[bj]
            if (li["triangulation_valid"] and lj["triangulation_valid"] and
                li.get("used_reproj_max_px") is not None and
                lj.get("used_reproj_max_px") is not None and
                li["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX and
                lj["used_reproj_max_px"] <= TEMPLATE_REPROJ_THRESHOLD_PX):
                bl = float(np.linalg.norm(
                    np.array(li["xyz"]) - np.array(lj["xyz"])))
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
    """Pass 2: re-score candidates with bone/temporal/switch penalties, re-select best."""
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
                    continue

                obs_views = lm.get("_obs_views", [])
                if not obs_views:
                    continue

                # Compute penalties for each feasible candidate
                for ev in all_candidates:
                    if not ev["feasible"]:
                        continue

                    X3 = np.array(ev["xyz"])

                    # Bone consistency penalty
                    bone_pen = 0.0
                    related_bones = []
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
                                related_bones.append((bname, bl, dev))

                    ev["bone_consistency_penalty"] = round(
                        bone_pen * CONFIG["bone_consistency_penalty_weight"], 4)

                    # Temporal continuity penalty
                    temp_pen = 0.0
                    if fi > 0:
                        prev_fr = frames_3d[fi - 1]
                        prev_lm = prev_fr[key][li_idx]
                        dt = fr["front_time_s"] - prev_fr["front_time_s"]
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
                            prev_cams = set(prev_lm["cameras_used"])
                            curr_cams = set(ev["cameras"])
                            if prev_cams != curr_cams:
                                switch_pen = 1.0

                    ev["camera_switch_penalty"] = round(
                        switch_pen * CONFIG["camera_switch_penalty_weight"], 4)

                    # Final score
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

                # Check if selection changed
                old_cams = set(lm["cameras_used"])
                new_cams = set(best["cameras"])
                if old_cams != new_cams or lm["xyz"] != best["xyz"]:
                    n_changed += 1
                    # Re-apply
                    lm["qa_flags"] = []  # reset flags before re-apply
                    _apply_best_candidate(lm, best, obs_views)

                # Update candidate summaries with pass 2 scores
                lm["candidates"] = [{
                    "v": ev["views"], "ok": ev["feasible"],
                    "gs": ev["geometry_score"], "ur": ev["used_reproj_mean"],
                    "um": ev["used_reproj_max"], "vr": ev["validation_reproj_mean"],
                    "a": ev["angle_deg"], "d": ev["depth_m"], "r": ev["reason"],
                    "urp": ev["used_reproj_penalty"], "evp": ev["excluded_view_validation_penalty"],
                    "tas": ev["triangulation_angle_score"],
                    "bcp": ev["bone_consistency_penalty"], "tcp": ev["temporal_continuity_penalty"],
                    "csp": ev["camera_switch_penalty"], "fs": ev["final_score"],
                } for ev in all_candidates]

    return n_changed


def cleanup_internal_fields(all_results):
    """Remove internal fields not needed in output."""
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
                                    hand[idx]["template_eligible"] = False
                                    for vv in VIEWS:
                                        hand[idx]["template_eligible_by_view"][vv] = False
                                    hand[idx]["qa_flags"].append(
                                        f"bone_{side}_{bname}_{bl:.4f}m")
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
                                pose[idx]["template_eligible"] = False
                                for vv in VIEWS:
                                    pose[idx]["template_eligible_by_view"][vv] = False
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
#  Post-processing: Temporal QA (velocity + acceleration + depth oscillation)
# ================================================================

def apply_temporal_qa(all_results):
    temporal_anomalies = []

    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        if len(frames) < 2:
            continue

        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            n_lm = len(frames[0][key])

            # Build per-landmark time series for depth oscillation
            for li in range(n_lm):
                z_vel_signs = []  # (fi, sign_of_z_velocity)
                prev_valid_fi = None
                prev_valid_xyz = None
                gap_since_last_valid = 0.0

                for fi in range(len(frames)):
                    curr = frames[fi]
                    lc = curr[key][li]

                    if not lc["triangulation_valid"]:
                        if prev_valid_fi is not None:
                            gap_since_last_valid += (curr["front_time_s"] -
                                frames[max(fi-1, 0)]["front_time_s"])
                        continue

                    curr_xyz = np.array(lc["xyz"])

                    if fi == 0 or prev_valid_fi is None:
                        prev_valid_fi = fi
                        prev_valid_xyz = curr_xyz
                        gap_since_last_valid = 0.0
                        continue

                    prev = frames[prev_valid_fi]
                    lp = prev[key][li]
                    dt = curr["front_time_s"] - prev["front_time_s"]

                    if dt <= 0 or dt > CONFIG["temporal_max_gap_s"]:
                        prev_valid_fi = fi
                        prev_valid_xyz = curr_xyz
                        gap_since_last_valid = 0.0
                        continue

                    p1 = prev_valid_xyz
                    p2 = curr_xyz
                    disp = p2 - p1
                    d = float(np.linalg.norm(disp))
                    vel = d / dt

                    xy_d = float(np.linalg.norm(disp[:2]))
                    z_d = abs(float(disp[2]))
                    z_vel = float(disp[2]) / dt

                    # Track Z velocity sign for oscillation detection
                    z_sign = 1 if z_vel > 0.001 else (-1 if z_vel < -0.001 else 0)
                    if z_sign != 0:
                        z_vel_signs.append((fi, z_sign, abs(z_vel)))

                    # Detection reappearance
                    if not lp["triangulation_valid"]:
                        lc["qa_flags"].append("detection_reappear")
                        reappear_dist = d
                        if reappear_dist > 0.1:
                            lc["qa_flags"].append(f"reappear_jump_{reappear_dist:.3f}m")
                            lc["qa_valid"] = False
                            lc["template_eligible"] = False
                            for vv in VIEWS:
                                lc["template_eligible_by_view"][vv] = False

                    # Acceleration (requires fi >= 2 with valid predecessor of predecessor)
                    accel = None
                    accel_xy = None
                    accel_z = None
                    if prev_valid_fi >= 1:
                        ppfi = None
                        for k in range(prev_valid_fi - 1, -1, -1):
                            if frames[k][key][li]["triangulation_valid"]:
                                ppfi = k
                                break
                        if ppfi is not None:
                            pp = frames[ppfi]
                            dt_prev = prev["front_time_s"] - pp["front_time_s"]
                            if 0 < dt_prev <= CONFIG["temporal_max_gap_s"]:
                                pp_xyz = np.array(pp[key][li]["xyz"])
                                vel_prev = (p1 - pp_xyz) / dt_prev
                                vel_curr = disp / dt
                                dt_avg = (dt + dt_prev) / 2
                                accel_vec = (vel_curr - vel_prev) / dt_avg
                                accel = float(np.linalg.norm(accel_vec))
                                accel_xy = float(np.linalg.norm(accel_vec[:2]))
                                accel_z = abs(float(accel_vec[2]))

                    cams_changed = set(lp["cameras_used"]) != set(lc["cameras_used"])
                    has_bone = any("bone_" in f for f in lc["qa_flags"])
                    depth_only = z_d > 3 * xy_d and z_d > 0.05

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
                            lc["template_eligible"] = False
                            for vv in VIEWS:
                                lc["template_eligible_by_view"][vv] = False

                        temporal_anomalies.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr["front_time_s"], 6),
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
                            "dt_s": round(dt, 6),
                            "cameras_prev": lp["cameras_used"],
                            "cameras_curr": lc["cameras_used"],
                            "cameras_changed": cams_changed,
                            "gap_since_last_valid_s": round(gap_since_last_valid, 6),
                            "reason": reason,
                        })

                    elif accel is not None and accel > CONFIG["temporal_accel_warn_ms2"]:
                        lc["qa_flags"].append(f"high_accel_{accel:.0f}ms2")
                        temporal_anomalies.append({
                            "sample_id": sid, "frame_index": fi,
                            "front_time_s": round(curr["front_time_s"], 6),
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
                            "dt_s": round(dt, 6),
                            "cameras_prev": lp["cameras_used"],
                            "cameras_curr": lc["cameras_used"],
                            "cameras_changed": cams_changed,
                            "gap_since_last_valid_s": round(gap_since_last_valid, 6),
                            "reason": "high_acceleration",
                        })

                    prev_valid_fi = fi
                    prev_valid_xyz = curr_xyz
                    gap_since_last_valid = 0.0

                # Depth oscillation: Z velocity sign reversals in sliding window
                win = CONFIG["depth_z_reversal_window"]
                min_rev = CONFIG["depth_z_reversal_min_count"]
                min_amp = CONFIG["depth_z_reversal_min_amplitude_m"]

                for w_start in range(len(z_vel_signs) - win + 1):
                    window = z_vel_signs[w_start:w_start + win]
                    signs = [s for _, s, _ in window]
                    amps = [a for _, _, a in window]
                    reversals = sum(1 for k in range(1, len(signs)) if signs[k] != signs[k-1])
                    mean_amp = np.mean(amps) if amps else 0

                    if reversals >= min_rev and mean_amp > min_amp:
                        center_fi = window[len(window)//2][0]
                        lc_center = frames[center_fi][key][li]
                        flag = f"depth_oscillation_rev{reversals}_amp{mean_amp:.3f}"
                        if flag not in lc_center["qa_flags"]:
                            lc_center["qa_flags"].append(flag)
                            temporal_anomalies.append({
                                "sample_id": sid, "frame_index": center_fi,
                                "front_time_s": round(frames[center_fi]["front_time_s"], 6),
                                "category": cat, "landmark_index": li,
                                "displacement_m": None, "displacement_xy_m": None,
                                "displacement_z_m": None,
                                "velocity_ms": None, "velocity_xy_ms": None,
                                "velocity_z_ms": None,
                                "acceleration_ms2": None, "acceleration_xy_ms2": None,
                                "acceleration_z_ms2": None,
                                "dt_s": None,
                                "cameras_prev": None, "cameras_curr": None,
                                "cameras_changed": None,
                                "gap_since_last_valid_s": None,
                                "reason": f"depth_oscillation",
                            })

    return temporal_anomalies


# ================================================================
#  Statistics
# ================================================================

def compute_reproj_distributions(all_results):
    d = {
        "used_all": [], "excluded_all": [],
        "used_pose": [], "used_left_hand": [], "used_right_hand": [],
        "excl_pose": [], "excl_left_hand": [], "excl_right_hand": [],
        "used_2view": [], "used_3view": [],
        "qa_valid_used": [], "qa_invalid_used": [],
        "preselection_3view": [],
        "te_true_used": [], "te_false_used": [],
    }
    per_cam_used = {v: [] for v in VIEWS}
    per_cam_excl = {v: [] for v in VIEWS}
    per_lm = defaultdict(list)

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
                        if lm["n_cameras_used"] == 2:
                            d["used_2view"].append(ue)
                        elif lm["n_cameras_used"] == 3:
                            d["used_3view"].append(ue)
                        if lm["qa_valid"]:
                            d["qa_valid_used"].append(ue)
                        else:
                            d["qa_invalid_used"].append(ue)
                        if lm["template_eligible"]:
                            d["te_true_used"].append(ue)
                        else:
                            d["te_false_used"].append(ue)
                        per_lm[f"{cat}_{lm['lm_index']}"].append(ue)

                    ee = lm.get("excluded_reproj_mean_px")
                    if ee is not None:
                        d["excluded_all"].append(ee)
                        d[f"excl_{cat}"].append(ee)

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
    summary["per_landmark"] = {k: _ds(v) for k, v in per_lm.items()}
    return summary


def compute_coverage_v12(frames_3d):
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
            "n_3cam": n_3c, "n_3cam_of_tri": round(n_3c / max(n_tri, 1), 4),
            "n_2cam": n_2c, "n_2cam_of_tri": round(n_2c / max(n_tri, 1), 4),
            "invalid": n_inv, "invalid_reasons": dict(inv_reasons),
        }
    return result


def compute_exclusion_stats_v12(all_results):
    """Detailed exclusion diagnostics by pose landmark groups."""
    # Per camera x category x landmark: exclusion counts and residuals
    excl_detail = defaultdict(lambda: {
        "n_excluded": 0, "n_total_tri": 0,
        "residual_norms": [], "residual_dx": [], "residual_dy": [],
        "n_3view_infeasible": 0, "n_score_based_2view": 0,
        "n_excl_over_20": 0, "n_excl_over_50": 0,
        "n_excl_over_80": 0, "n_excl_over_100": 0,
    })

    pose_group_excl = {g: {v: 0 for v in VIEWS} for g in POSE_LM_GROUPS}
    pose_group_total = {g: {v: 0 for v in VIEWS} for g in POSE_LM_GROUPS}

    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    if not lm["triangulation_valid"]:
                        continue

                    li = lm["lm_index"]

                    # Find pose landmark group
                    if cat == "pose":
                        group = None
                        for gname, indices in POSE_LM_GROUPS.items():
                            if li in indices:
                                group = gname
                                break
                    else:
                        group = cat

                    for v in VIEWS:
                        dkey = (v, cat, li)
                        excl_detail[dkey]["n_total_tri"] += 1

                        if cat == "pose" and group:
                            pose_group_total[group][v] = pose_group_total.get(group, {}).get(v, 0) + 1

                    if lm["excluded_cameras"]:
                        for v in lm["excluded_cameras"]:
                            dkey = (v, cat, li)
                            excl_detail[dkey]["n_excluded"] += 1

                            cam_info = lm["per_camera"].get(v, {})
                            err = cam_info.get("reproj_error_px")
                            if err is not None:
                                excl_detail[dkey]["residual_norms"].append(err)
                                if err > 20: excl_detail[dkey]["n_excl_over_20"] += 1
                                if err > 50: excl_detail[dkey]["n_excl_over_50"] += 1
                                if err > 80: excl_detail[dkey]["n_excl_over_80"] += 1
                                if err > 100: excl_detail[dkey]["n_excl_over_100"] += 1

                                reproj_px = cam_info.get("reproj_2d_px")
                                raw_px = cam_info.get("raw_2d_px")
                                if reproj_px and raw_px:
                                    dx = reproj_px[0] - raw_px[0]
                                    dy = reproj_px[1] - raw_px[1]
                                    excl_detail[dkey]["residual_dx"].append(dx)
                                    excl_detail[dkey]["residual_dy"].append(dy)

                            # Classify: 3-view infeasible vs score-based
                            has_3v_feasible = any(c["v"] == "F+L+R" and c["ok"] for c in lm.get("candidates", []))
                            if has_3v_feasible:
                                excl_detail[dkey]["n_score_based_2view"] += 1
                            else:
                                excl_detail[dkey]["n_3view_infeasible"] += 1

                            if cat == "pose" and group:
                                pose_group_excl[group][v] = pose_group_excl.get(group, {}).get(v, 0) + 1

    # Summarize per-camera-category
    per_cam_cat = {v: {"pose": 0, "left_hand": 0, "right_hand": 0} for v in VIEWS}
    total_excl = 0
    for (v, cat, li), info in excl_detail.items():
        per_cam_cat[v][cat] += info["n_excluded"]
        total_excl += info["n_excluded"]

    return {
        "per_camera_category": per_cam_cat,
        "total_excluded": total_excl // 3,  # each landmark counted per view above
        "pose_group_exclusion": pose_group_excl,
        "pose_group_total": pose_group_total,
        "detail": {f"{v}_{cat}_{li}": {
            "n_excluded": info["n_excluded"],
            "n_total_tri": info["n_total_tri"],
            "excl_rate": round(info["n_excluded"] / max(info["n_total_tri"], 1), 4),
            "residual_mean": round(float(np.mean(info["residual_norms"])), 2) if info["residual_norms"] else None,
            "residual_median": round(float(np.median(info["residual_norms"])), 2) if info["residual_norms"] else None,
            "dx_mean": round(float(np.mean(info["residual_dx"])), 2) if info["residual_dx"] else None,
            "dy_mean": round(float(np.mean(info["residual_dy"])), 2) if info["residual_dy"] else None,
            "n_3view_infeasible": info["n_3view_infeasible"],
            "n_score_based_2view": info["n_score_based_2view"],
            "n_excl_over_20": info["n_excl_over_20"],
            "n_excl_over_50": info["n_excl_over_50"],
            "n_excl_over_80": info["n_excl_over_80"],
            "n_excl_over_100": info["n_excl_over_100"],
        } for (v, cat, li), info in excl_detail.items() if info["n_excluded"] > 0},
    }


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
            "cv": round(float(np.std(a)) / med, 4) if med > 0.001 else None,
        }
    return result


# ================================================================
#  QA Flag detailed analysis
# ================================================================

def analyze_qa_flags_v12(all_results):
    report = []

    for sid in ["S001_P001_GREETING_FAST_R03", "S001_P001_GREETING_SLOW_R03"]:
        frames = all_results.get(sid, [])
        n = len(frames)
        n_tri = sum(1 for f in frames if any(lm["triangulation_valid"] for lm in f["right_hand_landmarks_3d"]))
        n_qa = sum(1 for f in frames if any(lm["qa_valid"] for lm in f["right_hand_landmarks_3d"]))
        n_3c = sum(1 for f in frames
                   if any(lm["n_cameras_used"] == 3 and lm["triangulation_valid"]
                          for lm in f["right_hand_landmarks_3d"]))
        report.append({
            "flag": "low_right_hand_detection", "sample_id": sid,
            "total_frames": n,
            "rh_tri_frames": n_tri, "rh_tri_rate": round(n_tri / max(n, 1), 3),
            "rh_qa_frames": n_qa, "rh_qa_rate": round(n_qa / max(n, 1), 3),
            "rh_3cam_frames": n_3c, "rh_3cam_rate": round(n_3c / max(n, 1), 3),
        })

    ty_sid = REGRESSION_TY_FRAMES["sample_id"]
    ty_frames = all_results.get(ty_sid, [])

    for fi in REGRESSION_TY_FRAMES["front_frame_indices"]:
        if fi >= len(ty_frames):
            report.append({"flag": "thankyou_fast_detailed", "frame_index": fi, "found": False})
            continue
        fr = ty_frames[fi]
        lh = fr["left_hand_landmarks_3d"]
        n_left_excl = sum(1 for lm in lh
                          if lm["triangulation_valid"] and "LEFT" in lm.get("excluded_cameras", []))

        per_lm = []
        for lm in lh:
            left_cam = lm["per_camera"].get("LEFT", {})
            per_lm.append({
                "i": lm["lm_index"],
                "tri": lm["triangulation_valid"], "qa": lm["qa_valid"], "te": lm["template_eligible"],
                "cams": lm["cameras_used"], "excl": lm["excluded_cameras"],
                "sel": lm["selection_reason"],
                "ur": lm["used_reproj_mean_px"],
                "um": lm["used_reproj_max_px"],
                "left_err": left_cam.get("reproj_error_px"),
                "left_used": left_cam.get("used_for_triangulation", False),
                "flags": lm["qa_flags"],
                "te_by_view": lm.get("template_eligible_by_view", {}),
            })

        report.append({
            "flag": "thankyou_fast_detailed", "sample_id": ty_sid,
            "frame_index": fi, "front_time_s": fr["front_time_s"], "found": True,
            "views_available": fr["views_available"],
            "left_status": {v: fr["per_view_info"].get(v, {}).get("anat_left_status", "N/A") for v in VIEWS},
            "n_tri": sum(1 for lm in lh if lm["triangulation_valid"]),
            "n_qa": sum(1 for lm in lh if lm["qa_valid"]),
            "n_te": sum(1 for lm in lh if lm["template_eligible"]),
            "n_left_excluded": n_left_excl,
            "per_landmark": per_lm,
            "assessment": (
                f"LEFT excluded for {n_left_excl}/21 left-hand landmarks. "
                "Hand crossover -- LEFT view unreliable. "
                "2-view (FRONT+RIGHT) used where LEFT excluded."
            ),
        })

    return report


# ================================================================
#  Video frame extraction for contact sheets
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


def balanced_sample_items(items_by_sample, max_per_sample, max_total):
    """Balanced sampling: up to max_per_sample per sample, then fill to max_total."""
    selected = []
    for sid in PILOT_SAMPLE_IDS:
        sitems = items_by_sample.get(sid, [])
        n = min(len(sitems), max_per_sample)
        if n > 0:
            if len(sitems) > n:
                indices = np.linspace(0, len(sitems) - 1, n, dtype=int)
                selected.extend([sitems[i] for i in indices])
            else:
                selected.extend(sitems[:n])
    if len(selected) > max_total:
        indices = np.linspace(0, len(selected) - 1, max_total, dtype=int)
        selected = [selected[i] for i in indices]
    return selected


# ================================================================
#  Overlays (one pass per view)
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
                    lc = (0, 255, 255) if cam.get("used_for_triangulation") else (0, 165, 255)
                    cv2.line(img, (ox, oy), (rx, ry), lc, 1)

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


def make_overlays_for_view(view, all_results, synced, intrinsics, extrinsics, out_dir):
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
                   overlay_counts, contact_paths, contact_index):
    results = []

    def chk(cond, msg):
        results.append(("PASS" if cond else "FAIL", msg))

    # 1. All 9 sequences
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

    # 3. Total landmarks = frames * 75
    total_lm_actual = sum(
        len(fr["pose_landmarks_3d"]) + len(fr["left_hand_landmarks_3d"]) + len(fr["right_hand_landmarks_3d"])
        for sid in PILOT_SAMPLE_IDS for fr in all_results[sid])
    chk(total_lm_actual == total_frames * 75,
        f"total_landmarks: {total_lm_actual}=={total_frames*75}")

    # 4. Invalid landmarks have no coords
    n_bad = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if not lm["triangulation_valid"] and lm["xyz"] is not None:
                        n_bad += 1
    chk(n_bad == 0, f"no_coords_for_invalid (violations={n_bad})")

    # 5. Camera count matches used flag
    n_mm = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"]:
                        n_u = sum(1 for c in lm["per_camera"].values() if c.get("used_for_triangulation"))
                        if n_u != lm["n_cameras_used"]:
                            n_mm += 1
    chk(n_mm == 0, f"cam_count_match (mismatch={n_mm})")

    # 6. Cheirality
    n_behind = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["xyz"] is not None:
                        if lm["xyz"][2] <= 0:
                            n_behind += 1
    chk(n_behind == 0, f"cheirality (behind={n_behind})")

    # 7. template_eligible implies qa_valid
    n_te_not_qa = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["template_eligible"] and not lm["qa_valid"]:
                        n_te_not_qa += 1
    chk(n_te_not_qa == 0, f"te_implies_qa (violations={n_te_not_qa})")

    # 8. used_reproj_max > 20px implies template_eligible=False
    n_high_te = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if (lm["triangulation_valid"] and
                        lm["used_reproj_max_px"] is not None and
                        lm["used_reproj_max_px"] > TEMPLATE_REPROJ_THRESHOLD_PX and
                        lm["template_eligible"]):
                        n_high_te += 1
    chk(n_high_te == 0, f"high_reproj_not_te (violations={n_high_te})")

    # 9. Per-view template eligibility consistency
    n_view_te_bad = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    te_by_view = lm.get("template_eligible_by_view", {})
                    if not lm["template_eligible"]:
                        # If template_eligible is False, all per-view must be False
                        for v in VIEWS:
                            if te_by_view.get(v, False):
                                n_view_te_bad += 1
    chk(n_view_te_bad == 0, f"per_view_te_consistency (violations={n_view_te_bad})")

    # 10. Regression: GREETING_NORMAL_R03 frame 39 right hand
    reg = REGRESSION_BONE
    frames = all_results.get(reg["sample_id"], [])
    if reg["frame_index"] < len(frames):
        rh = frames[reg["frame_index"]]["right_hand_landmarks_3d"]
        regression_ok = True
        regression_details = []
        for bi, bj in reg["bones"]:
            if rh[bi]["triangulation_valid"] and rh[bj]["triangulation_valid"]:
                bl = float(np.linalg.norm(
                    np.array(rh[bi]["xyz"]) - np.array(rh[bj]["xyz"])))
                if bl >= reg["min_expected_m"]:
                    if rh[bi]["template_eligible"] or rh[bj]["template_eligible"]:
                        regression_ok = False
                        regression_details.append(f"{bi}-{bj}={bl:.4f}m_NOT_FLAGGED")
                    else:
                        regression_details.append(f"{bi}-{bj}={bl:.4f}m_FLAGGED_OK")
                else:
                    regression_details.append(f"{bi}-{bj}={bl:.4f}m_FIXED")
        chk(regression_ok, f"regression_greeting_f39_rh ({'; '.join(regression_details)})")
    else:
        chk(False, "regression_greeting_f39_not_found")

    # 11. THANK_YOU_FAST frames 48,49 LEFT exclusion: real condition checks
    ty = REGRESSION_TY_FRAMES
    ty_frames = all_results.get(ty["sample_id"], [])
    for ffi in ty["front_frame_indices"]:
        if ffi < len(ty_frames):
            lh = ty_frames[ffi]["left_hand_landmarks_3d"]
            n_le = sum(1 for lm in lh if lm["triangulation_valid"] and "LEFT" in lm.get("excluded_cameras", []))
            chk(n_le == 21, f"ty_fast_f{ffi}_left_excl={n_le}==21")
        else:
            chk(False, f"ty_fast_f{ffi}_not_found")

    # 12. Overlay frame counts
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            k = f"{short}_{v.lower()}"
            actual = overlay_counts.get(k, -1)
            chk(actual == expected, f"overlay_{k}: {actual}=={expected}")

    # 13. Contact sheets have images (file size > 10KB)
    for cp in contact_paths:
        if cp.exists():
            sz = cp.stat().st_size
            chk(sz > 10000, f"contact_{cp.name}_size={sz}")
        else:
            chk(False, f"contact_{cp.name}_missing")

    # 14. Contact sheet balance: each sample represented
    if contact_index:
        samples_in_cs = set()
        for row in contact_index:
            samples_in_cs.add(row.get("sample_id", ""))
        n_samples_repr = sum(1 for sid in PILOT_SAMPLE_IDS if sid in samples_in_cs)
        chk(n_samples_repr >= min(len(PILOT_SAMPLE_IDS), 5),
            f"contact_balance: {n_samples_repr}/{len(PILOT_SAMPLE_IDS)} samples represented")

    # 15. Temporal anomalies have velocity AND acceleration values where applicable
    n_ta_missing_vel = 0
    for ta in temporal_anomalies:
        if ta["reason"] in ("high_velocity", "camera_switch_teleport", "bone_anomaly_jump",
                            "depth_axis_anomaly", "high_acceleration"):
            if ta["velocity_ms"] is None:
                n_ta_missing_vel += 1
    chk(n_ta_missing_vel == 0, f"temporal_has_velocity (missing={n_ta_missing_vel})")

    # 16. All feasible candidates have score components
    n_missing_sc = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    for cand in lm.get("candidates", []):
                        if cand["ok"]:
                            if any(cand.get(k) is None for k in ["gs", "urp", "evp", "tas", "bcp", "tcp", "csp", "fs"]):
                                n_missing_sc += 1
    chk(n_missing_sc == 0, f"all_candidates_have_scores (missing={n_missing_sc})")

    return results


# ================================================================
#  Main
# ================================================================

def main():
    print("Step 6B v1.2 PILOT: Multi-view 3D Reconstruction - S001_P001 R03\n")
    print("  Status: STEP6B_V1_1_TRIANGULATION_CORE_PASS / FULL_45_BLOCKED_PENDING_TEMPLATE_QA_V1_2\n")

    # -- Verify inputs --
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

    # -- Load --
    print("\n  Loading calibration...")
    intrinsics, extrinsics, proj_matrices, cam_centers = load_calibration()
    for v in VIEWS:
        print(f"    {v}: fx={intrinsics[v]['K'][0,0]:.1f} center={cam_centers[v].round(3)}")

    print("\n  Loading 2D landmarks...")
    all_frames = load_2d_landmarks()

    print("\n  Synchronizing frames...")
    synced = synchronize_frames(all_frames)
    for sid in PILOT_SAMPLE_IDS:
        n = len(synced[sid])
        n3 = sum(1 for e in synced[sid] if e["n_views"] == 3)
        print(f"    {sid.replace('S001_P001_', '')}: {n} frames ({n3} 3-view synced)")

    # -- Output dirs --
    out_json = OUT_DIR / "raw_3d"
    out_ov = OUT_DIR / "overlays"
    out_cs = OUT_DIR / "contact_sheets"
    for d in [out_json, out_ov, out_cs]:
        d.mkdir(parents=True, exist_ok=True)

    # -- Pass 1: Geometry-only triangulation --
    print("\n  Pass 1: Geometry-only triangulation...")
    all_results = {}
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        print(f"\n  === {short} (Pass 1) ===")
        frames_3d = process_sample_pass1(sid, synced[sid], intrinsics, extrinsics, proj_matrices, cam_centers)
        all_results[sid] = frames_3d
        n_tri = sum(1 for f in frames_3d for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"] for lm in f[k] if lm["triangulation_valid"])
        n_total = len(frames_3d) * 75
        print(f"    tri_valid: {n_tri}/{n_total} ({n_tri/max(n_total,1):.1%})")

    # -- Compute bone medians from pass 1 high-confidence results --
    print("\n  Computing bone medians for Pass 2...")
    all_bone_medians = {}
    for sid in PILOT_SAMPLE_IDS:
        bm = compute_bone_medians_for_pass2(all_results[sid])
        all_bone_medians[sid] = bm
        print(f"    {sid.replace('S001_P001_','')}: {len(bm)} bone types with median")

    # -- Pass 2: Re-score with penalties --
    print("\n  Pass 2: Re-scoring with bone/temporal/switch penalties...")
    total_changed = 0
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        n_changed = rescore_pass2(all_results[sid], all_bone_medians[sid],
                                  intrinsics, extrinsics, proj_matrices, cam_centers)
        total_changed += n_changed
        print(f"    {short}: {n_changed} landmarks re-selected")
    print(f"  Total Pass 2 re-selections: {total_changed}")

    # -- Cleanup internal fields --
    cleanup_internal_fields(all_results)

    # -- Post-processing QA --
    print("\n  Post-processing: bone length QA...")
    bone_outliers = apply_bone_qa(all_results)
    print(f"    Bone outliers flagged: {len(bone_outliers)}")

    print("  Post-processing: temporal QA (velocity + acceleration + depth oscillation)...")
    temporal_anomalies = apply_temporal_qa(all_results)
    print(f"    Temporal anomalies flagged: {len(temporal_anomalies)}")

    # Count template_eligible
    n_te = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
               for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
               for lm in f[k] if lm["template_eligible"])
    n_tri_total = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                      for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                      for lm in f[k] if lm["triangulation_valid"])
    n_qa_total = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                     for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                     for lm in f[k] if lm["qa_valid"])
    n_high_reproj_te = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                           for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                           for lm in f[k] if lm["triangulation_valid"] and
                           lm["used_reproj_max_px"] is not None and
                           lm["used_reproj_max_px"] > TEMPLATE_REPROJ_THRESHOLD_PX and
                           lm["template_eligible"])
    print(f"\n  Template eligibility check:")
    print(f"    tri_valid: {n_tri_total}, qa_valid: {n_qa_total}, template_eligible: {n_te}")
    print(f"    High reproj (>20px) AND template_eligible: {n_high_reproj_te} (should be 0)")

    # -- Save 3D JSONs --
    print("\n  Saving 3D JSONs...")
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        op = out_json / f"landmarks_3d_{short}.json"
        with open(op, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.2_pilot",
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

    # -- Compute all stats --
    print("\n  Computing statistics...")
    all_qa_summaries = []
    for sid in PILOT_SAMPLE_IDS:
        cov = compute_coverage_v12(all_results[sid])
        bs = compute_bone_stats(all_results[sid])
        n_bone_out = sum(1 for bo in bone_outliers if bo["sample_id"] == sid)
        n_temp_anom = sum(1 for ta in temporal_anomalies if ta["sample_id"] == sid)

        all_qa_summaries.append({
            "sample_id": sid,
            "n_frames": len(all_results[sid]),
            "n_synced_3view_frames": sum(1 for f in all_results[sid] if f["n_views_synced"] == 3),
            "coverage": cov,
            "n_bone_outliers": n_bone_out,
            "n_temporal_anomalies": n_temp_anom,
        })

    reproj_dists = compute_reproj_distributions(all_results)
    excl_stats = compute_exclusion_stats_v12(all_results)

    # -- QA flag analysis --
    print("\n  Analyzing QA flags...")
    qa_flag_report = analyze_qa_flags_v12(all_results)
    for item in qa_flag_report:
        flag = item.get("flag", "")
        if flag == "low_right_hand_detection":
            print(f"    {flag}: {item['sample_id'].replace('S001_P001_','')} "
                  f"tri={item['rh_tri_rate']:.1%} qa={item['rh_qa_rate']:.1%}")
        elif flag == "thankyou_fast_detailed" and item.get("found"):
            print(f"    {flag}: f{item['frame_index']} left_excl={item['n_left_excluded']}/21 "
                  f"qa={item['n_qa']}/21 te={item['n_te']}/21")

    # -- Overlays --
    print("\n  Generating overlays (one pass per view)...")
    overlay_counts = {}
    for v in VIEWS:
        print(f"  View {v}:")
        oc = make_overlays_for_view(v, all_results, synced, intrinsics, extrinsics, out_ov)
        overlay_counts.update(oc)

    # -- Contact sheets (balanced per-sample) --
    print("\n  Generating contact sheets with balanced per-sample sampling...")

    needed_dfis = set()
    cs_items_by_sample = {
        "temporal": defaultdict(list),
        "bone": defaultdict(list),
        "reproj": defaultdict(list),
        "exclusion": defaultdict(list),
        "qa_flag": defaultdict(list),
    }

    for ta in temporal_anomalies:
        sid = ta["sample_id"]
        fi = ta["frame_index"]
        fr = all_results[sid][fi]
        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items_by_sample["temporal"][sid].append({
            "dfi": dfi, "sample_id": sid,
            "text": [
                f"{sid.replace('S001_P001_','')} f{fi}",
                f"t={ta['front_time_s']:.3f}s {ta['category']}",
                f"lm{ta['landmark_index']} v={ta.get('velocity_ms','N/A')}m/s",
                f"a={ta.get('acceleration_ms2','N/A')}ms2",
                f"reason: {ta['reason']}",
            ],
        })

    for bo in bone_outliers:
        sid = bo["sample_id"]
        fi = bo["frame_index"]
        fr = all_results[sid][fi]
        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items_by_sample["bone"][sid].append({
            "dfi": dfi, "sample_id": sid,
            "text": [
                f"{sid.replace('S001_P001_','')} f{fi}",
                f"t={bo['front_time_s']:.3f}s {bo['category']}",
                f"bone: {bo['bone']}",
                f"len={bo['length_m']:.4f}m med={bo['median_m']:.4f}m",
                f"dev={bo['dev_factor']:.1f}x MAD",
            ],
        })

    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["used_reproj_max_px"] is not None:
                        if lm["used_reproj_max_px"] > CONFIG["reproj_outlier_px"]:
                            dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
                            if dfi is not None:
                                needed_dfis.add(dfi)
                            cs_items_by_sample["reproj"][sid].append({
                                "dfi": dfi, "sample_id": sid,
                                "text": [
                                    f"{sid.replace('S001_P001_','')} f{fi}",
                                    f"t={fr['front_time_s']:.3f}s {cat}",
                                    f"lm{lm['lm_index']} used_max={lm['used_reproj_max_px']:.1f}px",
                                    f"qa={lm['qa_valid']} te={lm['template_eligible']}",
                                ],
                            })

                    if lm["triangulation_valid"] and lm["excluded_cameras"]:
                        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
                        if dfi is not None:
                            needed_dfis.add(dfi)
                        cs_items_by_sample["exclusion"][sid].append({
                            "dfi": dfi, "sample_id": sid,
                            "text": [
                                f"{sid.replace('S001_P001_','')} f{fi}",
                                f"{cat} lm{lm['lm_index']}",
                                f"used: {','.join(c[0] for c in lm['cameras_used'])}",
                                f"excl: {','.join(c[0] for c in lm['excluded_cameras'])}",
                                f"ur={lm['used_reproj_mean_px']}px",
                            ],
                        })

    for item in qa_flag_report:
        if item.get("flag") == "thankyou_fast_detailed" and item.get("found"):
            fi = item["frame_index"]
            sid = item["sample_id"]
            fr = all_results[sid][fi]
            dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
            if dfi is not None:
                needed_dfis.add(dfi)
            cs_items_by_sample["qa_flag"][sid].append({
                "dfi": dfi, "sample_id": sid,
                "text": [
                    f"TY_FAST f{fi} t={item['front_time_s']:.3f}s",
                    f"left_excl={item['n_left_excluded']}/21",
                    f"qa={item['n_qa']}/21 te={item['n_te']}/21",
                    "hand crossover region",
                ],
            })

    reg = REGRESSION_BONE
    reg_frames = all_results.get(reg["sample_id"], [])
    if reg["frame_index"] < len(reg_frames):
        rfr = reg_frames[reg["frame_index"]]
        dfi = rfr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items_by_sample["qa_flag"][reg["sample_id"]].append({
            "dfi": dfi, "sample_id": reg["sample_id"],
            "text": [
                f"GREETING_NORMAL f{reg['frame_index']}",
                f"t={rfr['front_time_s']:.3f}s",
                "right hand bone outlier",
                "REGRESSION TEST",
            ],
        })

    print(f"  Extracting {len(needed_dfis)} FRONT video frames for contact sheets...")
    video_frames = extract_video_frames(VIDEOS["FRONT"], sorted(needed_dfis))
    print(f"    Extracted: {len(video_frames)} frames")

    contact_paths = []
    contact_index_rows = []
    max_ps = CONFIG["contact_max_per_sample"]
    max_cells = CONFIG["contact_max_cells"]

    for cs_type, items_by_sid in cs_items_by_sample.items():
        balanced = balanced_sample_items(dict(items_by_sid), max_ps, max_cells)
        cp = out_cs / f"contact_{cs_type}.png"
        make_contact_sheet(balanced, video_frames, cp, cs_type)
        contact_paths.append(cp)
        for item in balanced:
            contact_index_rows.append({
                "sheet_type": cs_type,
                "sample_id": item.get("sample_id", ""),
                "dfi": item.get("dfi"),
                "text_line_0": item.get("text", [""])[0],
            })

    # Save contact sheet index
    cs_index_path = out_cs / "contact_sheet_index.csv"
    if contact_index_rows:
        with open(cs_index_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=contact_index_rows[0].keys())
            w.writeheader()
            w.writerows(contact_index_rows)
    print(f"  contact_sheet_index.csv ({len(contact_index_rows)} rows)")

    # -- CSVs --
    print("\n  Saving CSVs...")

    qa_rows = []
    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    te_by_view = lm.get("template_eligible_by_view", {})
                    qa_rows.append({
                        "sample_id": sid, "frame_index": fi,
                        "front_time_s": fr["front_time_s"],
                        "category": cat, "landmark_index": lm["lm_index"],
                        "triangulation_valid": lm["triangulation_valid"],
                        "qa_valid": lm["qa_valid"],
                        "template_eligible": lm["template_eligible"],
                        "te_FRONT": te_by_view.get("FRONT", False),
                        "te_LEFT": te_by_view.get("LEFT", False),
                        "te_RIGHT": te_by_view.get("RIGHT", False),
                        "n_observations": lm["n_observations"],
                        "n_cameras_used": lm["n_cameras_used"],
                        "cameras_used": ",".join(lm["cameras_used"]) if lm["cameras_used"] else "",
                        "excluded_cameras": ",".join(lm["excluded_cameras"]) if lm["excluded_cameras"] else "",
                        "selection_reason": lm["selection_reason"],
                        "x": lm["xyz"][0] if lm["xyz"] else None,
                        "y": lm["xyz"][1] if lm["xyz"] else None,
                        "z": lm["xyz"][2] if lm["xyz"] else None,
                        "used_reproj_mean_px": lm["used_reproj_mean_px"],
                        "used_reproj_max_px": lm["used_reproj_max_px"],
                        "excluded_reproj_mean_px": lm["excluded_reproj_mean_px"],
                        "all_reproj_mean_px": lm["all_reproj_mean_px"],
                        "invalid_reason": lm["invalid_reason"],
                        "qa_flags": "|".join(lm["qa_flags"]) if lm["qa_flags"] else "",
                    })
    qa_csv = OUT_DIR / "frame_landmark_qa.csv"
    with open(qa_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qa_rows[0].keys())
        w.writeheader()
        w.writerows(qa_rows)
    print(f"  frame_landmark_qa.csv: {len(qa_rows)} rows")

    sum_rows = []
    for qs in all_qa_summaries:
        row = {
            "sample_id": qs["sample_id"],
            "n_frames": qs["n_frames"],
            "n_synced_3view_frames": qs["n_synced_3view_frames"],
            "n_bone_outliers": qs["n_bone_outliers"],
            "n_temporal_anomalies": qs["n_temporal_anomalies"],
        }
        for cat in ["pose", "left_hand", "right_hand"]:
            cc = qs["coverage"].get(cat, {})
            row[f"{cat}_tri_valid"] = cc.get("tri_valid", 0)
            row[f"{cat}_tri_rate"] = cc.get("tri_rate", 0)
            row[f"{cat}_qa_valid"] = cc.get("qa_valid", 0)
            row[f"{cat}_qa_rate"] = cc.get("qa_rate", 0)
            row[f"{cat}_te"] = cc.get("template_eligible", 0)
            row[f"{cat}_te_rate"] = cc.get("te_rate", 0)
            row[f"{cat}_3cam"] = cc.get("n_3cam", 0)
            row[f"{cat}_3cam_of_tri"] = cc.get("n_3cam_of_tri", 0)
            row[f"{cat}_2cam"] = cc.get("n_2cam", 0)
            row[f"{cat}_invalid"] = cc.get("invalid", 0)
        sum_rows.append(row)
    sum_csv = OUT_DIR / "sample_qa_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_rows[0].keys())
        w.writeheader()
        w.writerows(sum_rows)
    print(f"  sample_qa_summary.csv")

    # Camera exclusion CSV (detailed with pose groups)
    excl_rows = []
    for v in VIEWS:
        for cat in ["pose", "left_hand", "right_hand"]:
            excl_rows.append({
                "camera": v, "category": cat,
                "n_excluded": excl_stats["per_camera_category"][v][cat],
            })
    # Add pose group rows
    for gname in POSE_LM_GROUPS:
        for v in VIEWS:
            excl_rows.append({
                "camera": v, "category": f"pose_{gname}",
                "n_excluded": excl_stats["pose_group_exclusion"].get(gname, {}).get(v, 0),
            })
    excl_csv = OUT_DIR / "camera_exclusion_report.csv"
    with open(excl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=excl_rows[0].keys())
        w.writeheader()
        w.writerows(excl_rows)

    # Recalculate total_excluded properly
    total_excl_lm = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["excluded_cameras"]:
                        total_excl_lm += 1
    print(f"  camera_exclusion_report.csv (total_landmarks_with_exclusion={total_excl_lm})")

    # Detailed exclusion diagnostics JSON
    excl_detail_path = OUT_DIR / "camera_exclusion_detail.json"
    with open(excl_detail_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_camera_category": excl_stats["per_camera_category"],
            "pose_group_exclusion": excl_stats["pose_group_exclusion"],
            "detail": excl_stats.get("detail", {}),
        }, f, indent=2, ensure_ascii=False)

    bone_rows = []
    for sid in PILOT_SAMPLE_IDS:
        bs = compute_bone_stats(all_results[sid])
        for name, st in bs.items():
            bone_rows.append({"sample_id": sid, "bone": name, **st})
    bone_csv = OUT_DIR / "bone_lengths.csv"
    with open(bone_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bone_rows[0].keys())
        w.writeheader()
        w.writerows(bone_rows)
    print(f"  bone_lengths.csv")

    bo_csv = OUT_DIR / "bone_outliers.csv"
    if bone_outliers:
        with open(bo_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bone_outliers[0].keys())
            w.writeheader()
            w.writerows(bone_outliers)
    else:
        with open(bo_csv, "w", encoding="utf-8") as f:
            f.write("(no bone outliers)\n")
    print(f"  bone_outliers.csv ({len(bone_outliers)} rows)")

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
        with open(ta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ta_for_csv[0].keys())
            w.writeheader()
            w.writerows(ta_for_csv)
    else:
        with open(ta_csv, "w", encoding="utf-8") as f:
            f.write("(no temporal anomalies)\n")
    print(f"  temporal_anomalies.csv ({len(temporal_anomalies)} rows)")

    rdist_path = OUT_DIR / "reproj_distributions.json"
    with open(rdist_path, "w", encoding="utf-8") as f:
        json.dump(reproj_dists, f, indent=2, ensure_ascii=False)
    print(f"  reproj_distributions.json")

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
    cam_csv = OUT_DIR / "camera_reproj_errors.csv"
    with open(cam_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cam_rows[0].keys())
        w.writeheader()
        w.writerows(cam_rows)
    print(f"  camera_reproj_errors.csv")

    ov_verify_rows = []
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            k = f"{short}_{v.lower()}"
            actual = overlay_counts.get(k, -1)
            ov_verify_rows.append({
                "sample": short, "view": v,
                "expected": expected, "actual": actual,
                "match": actual == expected,
            })
    ov_csv = OUT_DIR / "overlay_frame_verification.csv"
    with open(ov_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ov_verify_rows[0].keys())
        w.writeheader()
        w.writerows(ov_verify_rows)
    print(f"  overlay_frame_verification.csv")

    # -- Manifest --
    input_sha = {
        "c1_front": sha256_file(C1_INTRINSICS["FRONT"]),
        "c1_left": sha256_file(C1_INTRINSICS["LEFT"]),
        "c1_right": sha256_file(C1_INTRINSICS["RIGHT"]),
        "c2_extrinsics": sha256_file(C2_EXTRINSICS),
        "2d_front": sha256_file(STEP6A_DIR / "landmarks_2d_front.json"),
        "2d_left": sha256_file(STEP6A_DIR / "landmarks_2d_left.json"),
        "2d_right": sha256_file(STEP6A_DIR / "landmarks_2d_right.json"),
    }
    manifest = {
        "step": "6B_pilot_v1.2",
        "date": datetime.datetime.now().isoformat(),
        "status": "STEP6B_V1_1_TRIANGULATION_CORE_PASS",
        "full_45_status": "FULL_45_BLOCKED_PENDING_TEMPLATE_QA_V1_2",
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "coordinate_system": "FRONT_camera (X=right, Y=down, Z=forward)",
        "unit": "meters",
        "triangulation_method": "DLT 2-pass multi-candidate scoring",
        "pass1": "geometry-only scoring -> bone medians",
        "pass2": "bone_consistency + temporal_continuity + camera_switch penalties",
        "template_eligible_threshold": f"used_reproj_max <= {TEMPLATE_REPROJ_THRESHOLD_PX}px",
        "smoothing": "DISABLED",
        "input_sha256": input_sha,
        "config": CONFIG,
        "pilot_sample_ids": PILOT_SAMPLE_IDS,
        "pass2_reselections": total_changed,
    }
    man_path = OUT_DIR / "run_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cfg_path = OUT_DIR / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)

    flag_path = OUT_DIR / "qa_flag_analysis.json"
    with open(flag_path, "w", encoding="utf-8") as f:
        json.dump(qa_flag_report, f, indent=2, ensure_ascii=False)

    # -- Assertions --
    print("\n  Running assertions...")
    assertion_results = run_assertions(all_results, synced, bone_outliers,
                                       temporal_anomalies, overlay_counts,
                                       contact_paths, contact_index_rows)
    n_pass = sum(1 for s, _ in assertion_results if s == "PASS")
    n_fail = sum(1 for s, _ in assertion_results if s == "FAIL")
    for status, msg in assertion_results:
        if status == "FAIL":
            print(f"    FAIL: {msg}")
    print(f"  Assertions: {n_pass} PASS, {n_fail} FAIL")

    asr_path = OUT_DIR / "assertion_results.json"
    with open(asr_path, "w", encoding="utf-8") as f:
        json.dump([{"status": s, "message": m} for s, m in assertion_results], f, indent=2)

    # -- Report --
    print("\n  Generating report...")
    rpt = []
    rpt.append("# Step 6B v1.2 Pilot Report -- Multi-view 3D Reconstruction\n")
    rpt.append("## Status: STEP6B_V1_1_TRIANGULATION_CORE_PASS / FULL_45_BLOCKED_PENDING_TEMPLATE_QA_V1_2\n")
    rpt.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    rpt.append(f"**Script**: `Tools/s001_step6b_triangulate_pilot_v1_2.py`")
    rpt.append(f"**Scope**: 9 R03 samples x 3 views")
    rpt.append(f"**v1.1 status**: STEP6B_V1_1_TRIANGULATION_CORE_PASS\n")

    rpt.append("## Changes from v1.1\n")
    rpt.append(f"- `template_eligible` requires `used_reproj_max_px <= {TEMPLATE_REPROJ_THRESHOLD_PX}px`")
    rpt.append("- `template_eligible_by_view`: per-camera reproj check")
    rpt.append("- 2-pass candidate scoring: Pass 1 geometry -> bone medians; Pass 2 adds penalties")
    rpt.append("- Score components per candidate: geometry_score, used_reproj_penalty, excluded_view_validation_penalty, triangulation_angle_score, bone_consistency_penalty, temporal_continuity_penalty, camera_switch_penalty, final_score")
    rpt.append("- Actual acceleration computation (not just velocity)")
    rpt.append("- Depth oscillation via Z velocity sign reversals (not sequence-wide range)")
    rpt.append("- Pose landmark group exclusion diagnostics (face/sign_upper/pose_hand/lower_body)")
    rpt.append("- Balanced per-sample contact sheet sampling + contact_sheet_index.csv")
    rpt.append("- Real assertion conditions (not chk(True, ...))")
    rpt.append(f"- Thresholds kept at {CONFIG['reproj_outlier_px']}/{CONFIG['reproj_exclude_px']}px\n")

    rpt.append("## Coordinate System\n")
    rpt.append("- **Reference**: FRONT camera (identity R, zero t)")
    rpt.append("- **Axes**: X=right, Y=down, Z=forward")
    rpt.append("- **Unit**: meters\n")

    rpt.append("## Triangulation Method\n")
    rpt.append("- DLT: `cv2.triangulatePoints` (2-view), SVD (3-view)")
    rpt.append("- **2-pass view selection**:")
    rpt.append("  - Pass 1: geometry-only scoring, compute bone medians/MAD from high-confidence results")
    rpt.append("  - Pass 2: add bone_consistency, temporal_continuity, camera_switch penalties; re-select")
    rpt.append(f"  - Pass 2 re-selections: {total_changed}")
    rpt.append("- Feasibility: cheirality, finite, depth 0.3-10m,")
    rpt.append(f"    angle >= {CONFIG['min_triangulation_angle_deg']}deg, used_reproj_max <= {CONFIG['reproj_exclude_px']}px")
    rpt.append("- Smoothing: **DISABLED**\n")

    rpt.append("## Template Eligibility\n")
    rpt.append(f"- `template_eligible`: `qa_valid AND used_reproj_max_px <= {TEMPLATE_REPROJ_THRESHOLD_PX}px`")
    rpt.append("- `template_eligible_by_view`: `template_eligible AND per_camera_reproj <= 20px`")
    rpt.append(f"- tri_valid: {n_tri_total}, qa_valid: {n_qa_total}, template_eligible: {n_te}")
    rpt.append(f"- High reproj (>{TEMPLATE_REPROJ_THRESHOLD_PX}px) AND template_eligible: {n_high_reproj_te} (must be 0)\n")

    # Per-sample table
    rpt.append("## Per-Sample Results\n")
    rpt.append("| Sample | Frames | Synced3V | P_tri | P_qa | P_te | LH_tri | LH_qa | LH_te | RH_tri | RH_qa | RH_te | BoneOut | TempAnom |")
    rpt.append("|--------|--------|----------|-------|------|------|--------|-------|-------|--------|-------|-------|---------|----------|")
    for qs in all_qa_summaries:
        short = qs["sample_id"].replace("S001_P001_", "")
        c = qs["coverage"]
        rpt.append(
            f"| {short} | {qs['n_frames']} | {qs['n_synced_3view_frames']} | "
            f"{c['pose']['tri_rate']:.0%} | {c['pose']['qa_rate']:.0%} | {c['pose']['te_rate']:.0%} | "
            f"{c['left_hand']['tri_rate']:.0%} | {c['left_hand']['qa_rate']:.0%} | {c['left_hand']['te_rate']:.0%} | "
            f"{c['right_hand']['tri_rate']:.0%} | {c['right_hand']['qa_rate']:.0%} | {c['right_hand']['te_rate']:.0%} | "
            f"{qs['n_bone_outliers']} | {qs['n_temporal_anomalies']} |"
        )
    rpt.append("")

    # Reproj distributions
    rpt.append("## Reprojection Error Distributions\n")
    rpt.append("| Distribution | Count | Mean | Median | P95 | P99 | Max |")
    rpt.append("|-------------|-------|------|--------|-----|-----|-----|")
    for name in ["preselection_3view", "used_all", "excluded_all",
                  "used_pose", "used_left_hand", "used_right_hand",
                  "used_2view", "used_3view", "qa_valid_used", "qa_invalid_used",
                  "te_true_used", "te_false_used"]:
        st = reproj_dists.get(name, {})
        if st.get("count", 0) > 0:
            rpt.append(f"| {name} | {st['count']} | {st['mean']:.1f} | {st['median']:.1f} | "
                       f"{st['p95']:.1f} | {st['p99']:.1f} | {st['max']:.1f} |")
    rpt.append("")

    rpt.append("### Per-Camera Used Reproj\n")
    rpt.append("| Camera | Count | Mean | Median | P95 | Max |")
    rpt.append("|--------|-------|------|--------|-----|-----|")
    for v in VIEWS:
        st = reproj_dists.get("per_camera_used", {}).get(v, {})
        if st.get("count", 0) > 0:
            rpt.append(f"| {v} | {st['count']} | {st['mean']:.1f} | {st['median']:.1f} | "
                       f"{st['p95']:.1f} | {st['max']:.1f} |")
    rpt.append("")

    # Camera exclusion with pose groups
    rpt.append("## Camera Exclusion Report\n")
    rpt.append(f"Total landmarks with camera exclusion: {total_excl_lm}\n")
    rpt.append("| Camera | Pose | Left Hand | Right Hand | Total |")
    rpt.append("|--------|------|-----------|------------|-------|")
    for v in VIEWS:
        ec = excl_stats["per_camera_category"][v]
        rpt.append(f"| {v} | {ec['pose']} | {ec['left_hand']} | {ec['right_hand']} | "
                   f"{sum(ec.values())} |")
    rpt.append("")

    rpt.append("### Pose Landmark Group Exclusion\n")
    rpt.append("| Group | FRONT | LEFT | RIGHT |")
    rpt.append("|-------|-------|------|-------|")
    for gname in POSE_LM_GROUPS:
        ge = excl_stats["pose_group_exclusion"].get(gname, {})
        rpt.append(f"| {gname} | {ge.get('FRONT', 0)} | {ge.get('LEFT', 0)} | {ge.get('RIGHT', 0)} |")
    rpt.append("")

    # QA flag analysis
    rpt.append("## QA Flag Analysis\n")
    for item in qa_flag_report:
        flag = item.get("flag", "")
        rpt.append(f"### {flag}\n")
        if flag == "low_right_hand_detection":
            rpt.append(f"- **sample**: {item['sample_id']}")
            rpt.append(f"- total_frames: {item['total_frames']}")
            rpt.append(f"- rh_tri: {item['rh_tri_frames']} ({item['rh_tri_rate']:.1%})")
            rpt.append(f"- rh_qa_valid: {item['rh_qa_frames']} ({item['rh_qa_rate']:.1%})")
            rpt.append(f"- rh_3cam: {item['rh_3cam_frames']} ({item['rh_3cam_rate']:.1%})")
        elif flag == "thankyou_fast_detailed" and item.get("found"):
            rpt.append(f"- **sample**: {item['sample_id']} frame {item['frame_index']}")
            rpt.append(f"- front_time_s: {item['front_time_s']:.6f}")
            rpt.append(f"- tri: {item['n_tri']}/21, qa: {item['n_qa']}/21, te: {item['n_te']}/21")
            rpt.append(f"- LEFT excluded: {item['n_left_excluded']}/21")
            rpt.append(f"- Assessment: {item['assessment']}")
        rpt.append("")

    # Regression test
    rpt.append("## Regression Test: GREETING_NORMAL_R03 Frame 39 Right Hand\n")
    reg_frames = all_results.get(REGRESSION_BONE["sample_id"], [])
    if REGRESSION_BONE["frame_index"] < len(reg_frames):
        rfr = reg_frames[REGRESSION_BONE["frame_index"]]
        rh = rfr["right_hand_landmarks_3d"]
        rpt.append("| Bone | Length(m) | QA_valid | Template_eligible | Flags |")
        rpt.append("|------|----------|---------|-------------------|-------|")
        for bi, bj, bname in HAND_BONES:
            if rh[bi]["triangulation_valid"] and rh[bj]["triangulation_valid"]:
                bl = float(np.linalg.norm(np.array(rh[bi]["xyz"]) - np.array(rh[bj]["xyz"])))
                qa = rh[bi]["qa_valid"] and rh[bj]["qa_valid"]
                te = rh[bi]["template_eligible"] and rh[bj]["template_eligible"]
                flags = rh[bi]["qa_flags"] + rh[bj]["qa_flags"]
                flag_str = "; ".join(set(flags))[:80] if flags else ""
                marker = " **OUTLIER**" if bl > 0.08 else ""
                rpt.append(f"| {bname} ({bi}-{bj}) | {bl:.4f}{marker} | {qa} | {te} | {flag_str} |")
    rpt.append("")

    rpt.append(f"## Bone Outliers: {len(bone_outliers)} total\n")
    if bone_outliers:
        per_sample = defaultdict(int)
        for bo in bone_outliers:
            per_sample[bo["sample_id"].replace("S001_P001_", "")] += 1
        for s, c in sorted(per_sample.items()):
            rpt.append(f"- {s}: {c}")
    rpt.append("")

    rpt.append(f"## Temporal Anomalies: {len(temporal_anomalies)} total\n")
    if temporal_anomalies:
        by_reason = defaultdict(int)
        for ta in temporal_anomalies:
            by_reason[ta["reason"]] += 1
        for r, c in sorted(by_reason.items()):
            rpt.append(f"- {r}: {c}")
    rpt.append("")

    # Assertions
    rpt.append(f"## Assertions: {n_pass} PASS, {n_fail} FAIL\n")
    for status, msg in assertion_results:
        rpt.append(f"- [{status}] {msg}")
    rpt.append("")

    # Output files
    rpt.append("## Output Files\n")
    rpt.append("| File | Size |")
    rpt.append("|------|------|")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            sk = p.stat().st_size / 1024
            sz_str = f"{sk / 1024:.1f} MB" if sk > 1024 else f"{sk:.1f} KB"
            rpt.append(f"| {p.relative_to(OUT_DIR)} | {sz_str} |")
    rpt.append("")

    rpt.append("## Null Policy\n")
    rpt.append("- `triangulation_valid=False`: no xyz coords, invalid_reason recorded")
    rpt.append("- `qa_valid=False`: coords preserved for analysis, not for template")
    rpt.append(f"- `template_eligible=False`: excluded if qa_valid=False OR used_reproj_max > {TEMPLATE_REPROJ_THRESHOLD_PX}px")
    rpt.append("- No interpolation, no gap-filling, no smoothing")
    rpt.append("- Per-camera 2D preserved even for invalid 3D\n")

    rpt_path = OUT_DIR / "STEP6B_PILOT_V1_2_REPORT.md"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))
    print(f"  Saved: {rpt_path.name}")

    # -- ZIP --
    print("\n  Creating delivery ZIP...")
    zip_path = PROJECT_ROOT / "WordsInSilence_step6b_v1_2_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "Tools/s001_step6b_triangulate_pilot_v1_2.py")
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(PROC_DIR)))
    zip_sz = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP: {zip_path.name} ({zip_sz:.1f} MB)")

    # -- Final summary --
    total_frames = sum(len(all_results[sid]) for sid in PILOT_SAMPLE_IDS)
    total_lm = total_frames * 75

    print(f"\n  {'=' * 60}")
    print(f"  STEP 6B v1.2 PILOT SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Samples: {len(PILOT_SAMPLE_IDS)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Triangulation valid: {n_tri_total}/{total_lm} ({n_tri_total / max(total_lm, 1):.1%})")
    print(f"  QA valid: {n_qa_total}/{total_lm} ({n_qa_total / max(total_lm, 1):.1%})")
    print(f"  Template eligible: {n_te}/{total_lm} ({n_te / max(total_lm, 1):.1%})")
    print(f"  Bone outliers: {len(bone_outliers)}")
    print(f"  Temporal anomalies: {len(temporal_anomalies)}")
    print(f"  Pass 2 re-selections: {total_changed}")
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
