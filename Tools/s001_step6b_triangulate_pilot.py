"""
Step 6B v1.0 PILOT: Multi-view 3D Reconstruction -- S001_P001 R03 (9 samples)

Reads frozen 2D landmarks from Step 6A full extraction.
Uses C1 intrinsics (v2) + C2 extrinsics (v2 robust) + sync data.
Triangulates pose (33) and hand (21x2) landmarks per synchronized frame.
"""

import json, csv, hashlib, sys, datetime, math, zipfile, copy
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import av

# ═══════════════════════════════════════════════════════════
#  Paths & Constants
# ═══════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"

STEP6A_DIR = PROC_DIR / "landmarks_2d_v1_4" / "full"
OUT_DIR = PROC_DIR / "landmarks_3d_v1" / "pilot"

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

# 3D QA flagged items
QA_FLAGS = {
    "low_right_hand_detection": ["S001_P001_GREETING_FAST_R03", "S001_P001_GREETING_SLOW_R03"],
    "low_confidence_frames": [
        # These are LEFT camera local times; converted to front_time below
        {"sample_id": "S001_P001_THANK_YOU_FAST_R03", "view": "LEFT",
         "camera_local_times_s": [391.255833, 391.289167],
         "reason": "hand_crossover_single_detection_ANAT_LEFT"},
    ],
}

# ═══════════════════════════════════════════════════════════
#  Configuration (separated for future tuning)
# ═══════════════════════════════════════════════════════════

CONFIG = {
    # Frame matching
    "sync_tolerance_s": 0.020,        # max time diff for frame matching across views

    # Triangulation
    "min_views_for_3d": 2,            # minimum views to attempt triangulation
    "reproj_outlier_px": 20.0,        # per-camera reproj error to flag as outlier
    "reproj_exclude_px": 50.0,        # per-camera reproj error to exclude view entirely
    "max_3view_reproj_to_keep_all": 10.0,  # if all 3 views under this, keep all

    # QA thresholds (initial -- will be refined based on actual distribution)
    "bone_length_variation_warn": 0.30,   # 30% variation from median bone length
    "temporal_jump_warn_m": 0.10,         # 10cm frame-to-frame jump
    "depth_oscillation_warn_m": 0.05,     # 5cm depth oscillation

    # Smoothing: DISABLED for pilot (raw only)
    "apply_smoothing": False,

    # Overlay
    "overlay_thumb_size": (384, 216),
}

# Pose connections for bone length analysis
POSE_BONES = [
    (11, 13, "left_upper_arm"), (13, 15, "left_forearm"),
    (12, 14, "right_upper_arm"), (14, 16, "right_forearm"),
    (11, 12, "shoulders"), (11, 23, "left_torso"), (12, 24, "right_torso"),
    (23, 24, "hips"),
]

# Hand connections
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm
]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════
#  Load calibration data
# ═══════════════════════════════════════════════════════════

def load_calibration():
    """Load C1 intrinsics, C2 extrinsics, build projection matrices."""
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

    # Build projection matrices: P = K @ [R|t]
    proj_matrices = {}
    for v in VIEWS:
        K = intrinsics[v]["K"]
        R = extrinsics[v]["R"]
        t = extrinsics[v]["t"]
        Rt = np.hstack([R, t])  # 3x4
        P = K @ Rt
        proj_matrices[v] = P

    return intrinsics, extrinsics, proj_matrices


# ═══════════════════════════════════════════════════════════
#  Load 2D landmarks
# ═══════════════════════════════════════════════════════════

def load_2d_landmarks():
    """Load frozen Step 6A 2D landmarks for R03 samples."""
    all_frames = {}
    for v in VIEWS:
        fp = STEP6A_DIR / f"landmarks_2d_{v.lower()}.json"
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        # Filter to R03 only
        r03 = [fr for fr in data["frames"] if fr["sample_id"] in PILOT_SAMPLE_IDS]
        all_frames[v] = r03
        print(f"  Loaded {v}: {len(r03)} R03 frames (of {data['n_frames']} total)")
    return all_frames


# ═══════════════════════════════════════════════════════════
#  Frame synchronization
# ═══════════════════════════════════════════════════════════

def synchronize_frames(all_frames):
    """Match frames across views by mapped_front_pts_s using DP monotonic matching."""
    tol = CONFIG["sync_tolerance_s"]
    synced = {}  # sample_id -> list of {front_time, views: {v: frame}}

    for sid in PILOT_SAMPLE_IDS:
        # Get frames per view, sorted by front time
        view_frames = {}
        for v in VIEWS:
            vf = sorted(
                [f for f in all_frames[v] if f["sample_id"] == sid],
                key=lambda f: f["mapped_front_pts_s"]
            )
            view_frames[v] = vf

        # Use FRONT as reference timeline
        front_frames = view_frames["FRONT"]
        matched = []

        for ff in front_frames:
            ft = ff["mapped_front_pts_s"]
            entry = {"front_time_s": ft, "views": {"FRONT": ff}}

            # Find closest LEFT/RIGHT frame within tolerance
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


# ═══════════════════════════════════════════════════════════
#  Triangulation
# ═══════════════════════════════════════════════════════════

def triangulate_point(pts_2d, proj_mats):
    """
    Triangulate a 3D point from 2+ views using DLT.
    pts_2d: list of (x, y) undistorted normalized camera coords
    proj_mats: list of 3x4 projection matrices
    Returns: (X, Y, Z) in FRONT camera coordinates, or None
    """
    n = len(pts_2d)
    if n < 2:
        return None

    if n == 2:
        # cv2.triangulatePoints expects 2xN arrays
        p1 = np.array([[pts_2d[0][0]], [pts_2d[0][1]]], dtype=np.float64)
        p2 = np.array([[pts_2d[1][0]], [pts_2d[1][1]]], dtype=np.float64)
        X4 = cv2.triangulatePoints(proj_mats[0], proj_mats[1], p1, p2)
        X3 = (X4[:3] / X4[3]).flatten()
        return X3

    # For 3+ views, use linear DLT
    A = np.zeros((2 * n, 4), dtype=np.float64)
    for i, (pt, P) in enumerate(zip(pts_2d, proj_mats)):
        x, y = pt
        A[2*i]     = x * P[2] - P[0]
        A[2*i + 1] = y * P[2] - P[1]

    _, _, Vt = np.linalg.svd(A)
    X4 = Vt[-1]
    if abs(X4[3]) < 1e-10:
        return None
    X3 = X4[:3] / X4[3]
    return X3


def reproject_point(X3, K, R, t, dist=None):
    """Reproject 3D point (in FRONT coords) to pixel coords in a camera."""
    X_cam = R @ X3.reshape(3, 1) + t  # Transform to camera coords
    if X_cam[2, 0] <= 0:
        return None  # Behind camera
    x_norm = X_cam[0, 0] / X_cam[2, 0]
    y_norm = X_cam[1, 0] / X_cam[2, 0]
    # Apply distortion if provided
    if dist is not None and len(dist) >= 4:
        pts_in = np.array([[[x_norm, y_norm]]], dtype=np.float64)
        # Use projectPoints for distortion
        rvec, _ = cv2.Rodrigues(R)
        pts_3d = X3.reshape(1, 1, 3)
        pts_2d, _ = cv2.projectPoints(pts_3d, rvec, t, K, dist)
        return pts_2d[0, 0]
    else:
        u = K[0, 0] * x_norm + K[0, 2]
        v = K[1, 1] * y_norm + K[1, 2]
        return np.array([u, v])


def extract_landmark_2d(frame, lm_type, lm_index, hand_side=None):
    """
    Extract 2D undistorted pixel coords and metadata for a landmark from a frame.
    lm_type: 'pose' or 'hand'
    lm_index: landmark index
    hand_side: 'left' or 'right' (for hand landmarks)
    Returns: dict with coords and metadata, or None if not detected
    """
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
            "norm_cam": (lm["undistorted_norm_cam_x"], lm["undistorted_norm_cam_y"]),
        }

    elif lm_type == "hand":
        status_key = f"anat_{hand_side}_status"
        lm_key = f"anat_{hand_side}_landmarks"
        if frame.get(status_key) != "DETECTED" or not frame.get(lm_key):
            return None
        lms = frame[lm_key]
        if lm_index >= len(lms):
            return None
        lm = lms[lm_index]
        return {
            "undistorted_px": (lm["undistorted_px_x"], lm["undistorted_px_y"]),
            "raw_px": (lm["raw_distorted_px_x"], lm["raw_distorted_px_y"]),
            "visibility": lm.get("visibility", 0),
            "presence": lm.get("presence", 0),
            "norm_cam": (lm["undistorted_norm_cam_x"], lm["undistorted_norm_cam_y"]),
        }

    return None


def triangulate_landmark(synced_entry, lm_type, lm_index, hand_side,
                          intrinsics, extrinsics, proj_matrices):
    """
    Triangulate a single landmark across available views.
    Returns a dict with 3D coords, quality info, per-camera reprojection.
    """
    views_data = synced_entry["views"]

    # Collect 2D observations
    obs = {}
    for v, frame in views_data.items():
        lm2d = extract_landmark_2d(frame, lm_type, lm_index, hand_side)
        if lm2d is not None:
            obs[v] = lm2d

    n_obs = len(obs)
    result = {
        "lm_type": lm_type,
        "lm_index": lm_index,
        "hand_side": hand_side,
        "valid": False,
        "n_cameras": 0,
        "cameras_used": [],
        "xyz": None,
        "reproj_error_mean_px": None,
        "reproj_error_max_px": None,
        "confidence": 0.0,
        "invalid_reason": None,
        "per_camera": {},
    }

    if n_obs < CONFIG["min_views_for_3d"]:
        result["invalid_reason"] = f"insufficient_views({n_obs})"
        # Still record available 2D data
        for v, lm2d in obs.items():
            result["per_camera"][v] = {
                "raw_2d_px": list(lm2d["raw_px"]),
                "undistorted_2d_px": list(lm2d["undistorted_px"]),
                "visibility": lm2d["visibility"],
                "presence": lm2d["presence"],
                "reproj_2d_px": None,
                "reproj_error_px": None,
                "used_for_triangulation": False,
            }
        return result

    # Prepare for triangulation
    view_list = sorted(obs.keys())
    pts_undist_px = []
    pmats = []

    for v in view_list:
        pts_undist_px.append(obs[v]["undistorted_px"])
        pmats.append(proj_matrices[v])

    # Triangulate
    X3 = triangulate_point(pts_undist_px, pmats)
    if X3 is None:
        result["invalid_reason"] = "triangulation_failed"
        return result

    # Check if point is in front of all cameras
    for v in view_list:
        R = extrinsics[v]["R"]
        t = extrinsics[v]["t"]
        X_cam = R @ X3.reshape(3, 1) + t
        if X_cam[2, 0] <= 0:
            result["invalid_reason"] = f"behind_camera_{v}"
            return result

    # Compute reprojection errors for ALL views with observations
    reproj_errors = {}
    for v in obs:
        K = intrinsics[v]["K"]
        R = extrinsics[v]["R"]
        t = extrinsics[v]["t"]
        dist = intrinsics[v]["dist"]
        reproj_px = reproject_point(X3, K, R, t, dist)
        if reproj_px is not None:
            orig_px = np.array(obs[v]["raw_px"])
            err = float(np.linalg.norm(reproj_px - orig_px))
            reproj_errors[v] = {"reproj_px": reproj_px.tolist(), "error": err}
        else:
            reproj_errors[v] = {"reproj_px": None, "error": float("inf")}

    # If 3 views and one is very bad, re-triangulate with best 2
    if n_obs == 3:
        errors = [(v, reproj_errors[v]["error"]) for v in view_list]
        worst_v, worst_err = max(errors, key=lambda x: x[1])
        best_two_max = max(e for v, e in errors if v != worst_v)

        if worst_err > CONFIG["reproj_exclude_px"]:
            # Exclude worst view, re-triangulate
            remaining = [v for v in view_list if v != worst_v]
            pts2 = [obs[v]["undistorted_px"] for v in remaining]
            pmats2 = [proj_matrices[v] for v in remaining]
            X3_new = triangulate_point(pts2, pmats2)
            if X3_new is not None:
                # Recompute reproj errors with new point
                for v in obs:
                    K = intrinsics[v]["K"]
                    R = extrinsics[v]["R"]
                    t_vec = extrinsics[v]["t"]
                    dist = intrinsics[v]["dist"]
                    rp = reproject_point(X3_new, K, R, t_vec, dist)
                    if rp is not None:
                        orig_px = np.array(obs[v]["raw_px"])
                        reproj_errors[v] = {"reproj_px": rp.tolist(),
                                             "error": float(np.linalg.norm(rp - orig_px))}
                X3 = X3_new
                view_list = remaining
                result["invalid_reason"] = f"excluded_{worst_v}_reproj_{worst_err:.1f}px"

        elif worst_err > CONFIG["reproj_outlier_px"] and best_two_max < CONFIG["max_3view_reproj_to_keep_all"]:
            # Flag but keep all 3
            result["invalid_reason"] = f"high_reproj_{worst_v}_{worst_err:.1f}px"

    # Populate result
    used_errors = [reproj_errors[v]["error"] for v in view_list if reproj_errors[v]["error"] < float("inf")]
    all_errors = [reproj_errors[v]["error"] for v in obs if reproj_errors[v]["error"] < float("inf")]

    result["valid"] = True
    result["xyz"] = X3.tolist()
    result["n_cameras"] = len(view_list)
    result["cameras_used"] = view_list
    result["reproj_error_mean_px"] = float(np.mean(used_errors)) if used_errors else None
    result["reproj_error_max_px"] = float(np.max(used_errors)) if used_errors else None
    result["reproj_error_all_mean_px"] = float(np.mean(all_errors)) if all_errors else None

    # Confidence: based on n_views and reproj error
    if len(view_list) >= 3 and result["reproj_error_mean_px"] < 5:
        result["confidence"] = 0.9
    elif len(view_list) >= 2 and result["reproj_error_mean_px"] < 10:
        result["confidence"] = 0.7
    elif len(view_list) >= 2:
        result["confidence"] = 0.4
    else:
        result["confidence"] = 0.2

    # Per-camera details
    for v in obs:
        rp = reproj_errors.get(v, {})
        result["per_camera"][v] = {
            "raw_2d_px": list(obs[v]["raw_px"]),
            "undistorted_2d_px": list(obs[v]["undistorted_px"]),
            "visibility": obs[v]["visibility"],
            "presence": obs[v]["presence"],
            "reproj_2d_px": rp.get("reproj_px"),
            "reproj_error_px": rp.get("error"),
            "used_for_triangulation": v in view_list,
        }

    return result


# ═══════════════════════════════════════════════════════════
#  Process one sample
# ═══════════════════════════════════════════════════════════

def process_sample(sid, synced_frames, intrinsics, extrinsics, proj_matrices):
    """Process all synchronized frames for one sample, triangulating all landmarks."""
    frames_3d = []

    for entry in synced_frames:
        frame_result = {
            "sample_id": sid,
            "front_time_s": entry["front_time_s"],
            "n_views_available": entry["n_views"],
            "views_available": sorted(entry["views"].keys()),
            "per_view_info": {},
            "pose_landmarks_3d": [],
            "left_hand_landmarks_3d": [],
            "right_hand_landmarks_3d": [],
        }

        # Record per-view timestamps
        for v, fr in entry["views"].items():
            frame_result["per_view_info"][v] = {
                "camera_local_pts_s": fr["camera_local_pts_s"],
                "mapped_front_pts_s": fr["mapped_front_pts_s"],
                "decoded_frame_index": fr["decoded_frame_index"],
                "pose_detected": fr.get("pose_detected", False),
                "anat_left_status": fr.get("anat_left_status", "NOT_DETECTED"),
                "anat_right_status": fr.get("anat_right_status", "NOT_DETECTED"),
            }

        # Triangulate pose landmarks (33)
        for li in range(33):
            r = triangulate_landmark(entry, "pose", li, None,
                                      intrinsics, extrinsics, proj_matrices)
            frame_result["pose_landmarks_3d"].append(r)

        # Triangulate hand landmarks (21 each)
        for hand_side in ["left", "right"]:
            key = f"{hand_side}_hand_landmarks_3d"
            for li in range(21):
                r = triangulate_landmark(entry, "hand", li, hand_side,
                                          intrinsics, extrinsics, proj_matrices)
                frame_result[key].append(r)

        frames_3d.append(frame_result)

    return frames_3d


# ═══════════════════════════════════════════════════════════
#  QA Analysis
# ═══════════════════════════════════════════════════════════

def compute_reproj_stats(frames_3d):
    """Compute per-sample reprojection error statistics."""
    all_errors = {"pose": [], "left_hand": [], "right_hand": []}
    per_camera = {v: [] for v in VIEWS}
    per_lm = defaultdict(list)

    for fr in frames_3d:
        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            for lm in fr[key]:
                if not lm["valid"]:
                    continue
                for v, cam_info in lm["per_camera"].items():
                    err = cam_info.get("reproj_error_px")
                    if err is not None and err < float("inf"):
                        all_errors[cat].append(err)
                        per_camera[v].append(err)
                        lm_key = f"{cat}_{lm['lm_index']}"
                        per_lm[lm_key].append(err)

    stats = {}
    for cat, errs in all_errors.items():
        if errs:
            stats[cat] = {
                "count": len(errs),
                "mean": float(np.mean(errs)),
                "median": float(np.median(errs)),
                "p95": float(np.percentile(errs, 95)),
                "max": float(np.max(errs)),
            }
        else:
            stats[cat] = {"count": 0}

    stats["per_camera"] = {}
    for v, errs in per_camera.items():
        if errs:
            stats["per_camera"][v] = {
                "count": len(errs),
                "mean": float(np.mean(errs)),
                "median": float(np.median(errs)),
                "p95": float(np.percentile(errs, 95)),
                "max": float(np.max(errs)),
            }

    return stats


def compute_coverage(frames_3d):
    """Compute 3D coverage statistics per landmark category."""
    total = len(frames_3d)
    if total == 0:
        return {}

    coverage = {"pose": [0]*33, "left_hand": [0]*21, "right_hand": [0]*21}
    view_counts = {"pose": defaultdict(int), "left_hand": defaultdict(int), "right_hand": defaultdict(int)}

    for fr in frames_3d:
        for cat, key, n_lm in [("pose", "pose_landmarks_3d", 33),
                                 ("left_hand", "left_hand_landmarks_3d", 21),
                                 ("right_hand", "right_hand_landmarks_3d", 21)]:
            for lm in fr[key]:
                if lm["valid"]:
                    coverage[cat][lm["lm_index"]] += 1
                    view_counts[cat][lm["n_cameras"]] += 1

    result = {}
    for cat in coverage:
        n_lm = len(coverage[cat])
        rates = [coverage[cat][i] / total for i in range(n_lm)]
        result[cat] = {
            "per_landmark_rate": rates,
            "min_rate": float(min(rates)),
            "mean_rate": float(np.mean(rates)),
            "max_rate": float(max(rates)),
            "view_count_distribution": dict(view_counts[cat]),
        }

    return result


def compute_bone_lengths(frames_3d):
    """Compute bone length statistics for pose and hand landmarks."""
    bone_data = defaultdict(list)

    for fr in frames_3d:
        # Pose bones
        pose = fr["pose_landmarks_3d"]
        for i, j, name in POSE_BONES:
            if pose[i]["valid"] and pose[j]["valid"]:
                p1 = np.array(pose[i]["xyz"])
                p2 = np.array(pose[j]["xyz"])
                bone_data[f"pose_{name}"].append(float(np.linalg.norm(p1 - p2)))

        # Hand bones
        for side in ["left", "right"]:
            hand = fr[f"{side}_hand_landmarks_3d"]
            for i, j in HAND_BONES:
                if hand[i]["valid"] and hand[j]["valid"]:
                    p1 = np.array(hand[i]["xyz"])
                    p2 = np.array(hand[j]["xyz"])
                    bone_data[f"{side}_hand_{i}_{j}"].append(float(np.linalg.norm(p1 - p2)))

    result = {}
    for name, lengths in bone_data.items():
        if len(lengths) < 2:
            result[name] = {"n": len(lengths), "stable": None}
            continue
        med = float(np.median(lengths))
        std = float(np.std(lengths))
        cv = std / med if med > 0.001 else float("inf")
        result[name] = {
            "n": len(lengths),
            "median_m": round(med, 4),
            "std_m": round(std, 4),
            "cv": round(cv, 4),
            "min_m": round(float(min(lengths)), 4),
            "max_m": round(float(max(lengths)), 4),
            "stable": cv < CONFIG["bone_length_variation_warn"],
        }

    return result


def compute_temporal_stability(frames_3d):
    """Compute frame-to-frame 3D jump statistics."""
    jumps = {"pose": defaultdict(list), "left_hand": defaultdict(list), "right_hand": defaultdict(list)}
    anomalies = []

    for fi in range(1, len(frames_3d)):
        prev = frames_3d[fi - 1]
        curr = frames_3d[fi]
        dt = curr["front_time_s"] - prev["front_time_s"]
        if dt <= 0 or dt > 0.1:  # skip gaps > 100ms
            continue

        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            for li in range(len(curr[key])):
                lm_prev = prev[key][li]
                lm_curr = curr[key][li]
                if lm_prev["valid"] and lm_curr["valid"]:
                    p1 = np.array(lm_prev["xyz"])
                    p2 = np.array(lm_curr["xyz"])
                    d = float(np.linalg.norm(p2 - p1))
                    jumps[cat][li].append(d)
                    if d > CONFIG["temporal_jump_warn_m"]:
                        anomalies.append({
                            "frame_index": fi,
                            "front_time_s": curr["front_time_s"],
                            "category": cat,
                            "landmark_index": li,
                            "jump_m": round(d, 4),
                            "dt_s": round(dt, 4),
                        })

    stats = {}
    for cat in jumps:
        cat_stats = {}
        for li, js in jumps[cat].items():
            if js:
                cat_stats[li] = {
                    "mean_m": round(float(np.mean(js)), 4),
                    "median_m": round(float(np.median(js)), 4),
                    "p95_m": round(float(np.percentile(js, 95)), 4),
                    "max_m": round(float(max(js)), 4),
                }
        stats[cat] = cat_stats

    return stats, anomalies


def compute_depth_stability(frames_3d):
    """Check for depth-axis oscillation (Z in FRONT camera coords)."""
    z_series = {"pose": defaultdict(list), "left_hand": defaultdict(list), "right_hand": defaultdict(list)}

    for fr in frames_3d:
        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            for lm in fr[key]:
                if lm["valid"]:
                    z_series[cat][lm["lm_index"]].append(lm["xyz"][2])

    oscillations = []
    for cat in z_series:
        for li, zs in z_series[cat].items():
            if len(zs) < 3:
                continue
            zs = np.array(zs)
            diffs = np.diff(zs)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
            z_range = float(np.max(zs) - np.min(zs))
            if z_range > CONFIG["depth_oscillation_warn_m"] * 5:
                oscillations.append({
                    "category": cat, "landmark_index": li,
                    "z_range_m": round(z_range, 4),
                    "sign_changes": int(sign_changes),
                    "n_frames": len(zs),
                })

    return oscillations


# ═══════════════════════════════════════════════════════════
#  QA Flag analysis
# ═══════════════════════════════════════════════════════════

def analyze_qa_flags(all_results, synced_data):
    """Analyze specific 3D QA flagged items from external review."""
    report = []

    # 1. GREETING_FAST_R03 and GREETING_SLOW_R03 right hand detection
    for sid in QA_FLAGS["low_right_hand_detection"]:
        frames = all_results.get(sid, [])
        n_total = len(frames)
        n_rh_valid = sum(1 for f in frames
                         if any(lm["valid"] for lm in f["right_hand_landmarks_3d"]))
        n_rh_3view = sum(1 for f in frames
                         if any(lm["valid"] and lm["n_cameras"] == 3
                                for lm in f["right_hand_landmarks_3d"]))
        report.append({
            "flag": "low_right_hand_detection",
            "sample_id": sid,
            "total_frames": n_total,
            "right_hand_3d_frames": n_rh_valid,
            "right_hand_3d_rate": round(n_rh_valid / max(n_total, 1), 3),
            "right_hand_3view_frames": n_rh_3view,
            "right_hand_3view_rate": round(n_rh_3view / max(n_total, 1), 3),
        })

    # 2. THANK_YOU_FAST_R03 specific frames
    # Load affine transforms to convert camera local times to front times
    with open(SEGMENTS_FILE, encoding="utf-8") as f:
        seg_data = json.load(f)
    affine = seg_data["affine_transforms"]

    for flag_item in QA_FLAGS["low_confidence_frames"]:
        sid = flag_item["sample_id"]
        view = flag_item["view"]
        frames = all_results.get(sid, [])
        # Convert camera local times to front times
        a_coeff = affine[view]["a"]
        b_coeff = affine[view]["b"]
        front_times = [(t - b_coeff) / a_coeff for t in flag_item["camera_local_times_s"]]

        for ti, target_t in enumerate(front_times):
            cam_local_t = flag_item["camera_local_times_s"][ti]
            # Find closest frame
            closest = None
            min_dt = float("inf")
            for f in frames:
                dt = abs(f["front_time_s"] - target_t)
                if dt < min_dt:
                    min_dt = dt
                    closest = f

            if closest is None or min_dt > 0.05:
                report.append({
                    "flag": "low_confidence_frame",
                    "sample_id": sid,
                    "camera_local_time_s": cam_local_t,
                    "front_time_s_converted": target_t,
                    "found": False,
                    "recommendation": "FRAME_NOT_FOUND",
                })
                continue

            # Check left hand 3D quality at this frame
            lh_valid = sum(1 for lm in closest["left_hand_landmarks_3d"] if lm["valid"])
            lh_reproj = []
            for lm in closest["left_hand_landmarks_3d"]:
                if lm["valid"] and lm["reproj_error_mean_px"] is not None:
                    lh_reproj.append(lm["reproj_error_mean_px"])

            views_available = closest["views_available"]
            left_status = {vv: closest["per_view_info"].get(vv, {}).get("anat_left_status", "N/A")
                           for vv in VIEWS}

            mean_reproj = float(np.mean(lh_reproj)) if lh_reproj else None
            recommendation = "INCLUDE" if lh_valid >= 15 and (mean_reproj is None or mean_reproj < 15) else "EXCLUDE_OR_FLAG"

            report.append({
                "flag": "low_confidence_frame",
                "sample_id": sid,
                "camera_local_time_s": cam_local_t,
                "front_time_s_converted": target_t,
                "found": True,
                "actual_time_s": closest["front_time_s"],
                "views_available": views_available,
                "left_hand_status_per_view": left_status,
                "left_hand_valid_3d_landmarks": lh_valid,
                "left_hand_mean_reproj_px": round(mean_reproj, 2) if mean_reproj else None,
                "recommendation": recommendation,
                "reason": flag_item["reason"],
            })

    return report


# ═══════════════════════════════════════════════════════════
#  Overlay generation
# ═══════════════════════════════════════════════════════════

HAND_CONNECTIONS_VIS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]

POSE_CONNECTIONS_VIS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
    (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),
]


def make_reproj_overlay(view, frames_3d, synced_data, sid, intrinsics, extrinsics, out_path):
    """Generate overlay video with original 2D (green) and reprojected 3D (red) landmarks."""
    video_path = VIDEOS[view]
    con = av.open(str(video_path))
    st = con.streams.video[0]
    st.thread_type = "AUTO"
    tb = float(st.time_base)
    w, h = st.width, st.height

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 30, (w, h))

    # Build lookup: camera_local_pts_s -> frame_3d index
    synced_lookup = {}
    for i, entry in enumerate(synced_data):
        if view in entry["views"]:
            local_t = entry["views"][view]["camera_local_pts_s"]
            synced_lookup[round(local_t, 4)] = i

    K = intrinsics[view]["K"]
    R = extrinsics[view]["R"]
    t_vec = extrinsics[view]["t"]
    dist = intrinsics[view]["dist"]

    written = 0
    for vf in con.decode(st):
        if vf.pts is None:
            continue
        cps = float(vf.pts) * tb

        # Find matching synced frame
        fi = None
        for delta in [0, 0.0001, -0.0001, 0.001, -0.001]:
            fi = synced_lookup.get(round(cps + delta, 4))
            if fi is not None:
                break

        if fi is None:
            continue
        if fi >= len(frames_3d):
            continue

        img = vf.to_ndarray(format="bgr24")
        fr3d = frames_3d[fi]

        # Draw original 2D landmarks (green circles)
        entry = synced_data[fi]
        if view in entry["views"]:
            vfr = entry["views"][view]
            # Pose
            if vfr.get("pose_landmarks"):
                for plm in vfr["pose_landmarks"]:
                    px = int(plm["raw_distorted_px_x"])
                    py = int(plm["raw_distorted_px_y"])
                    cv2.circle(img, (px, py), 3, (0, 255, 0), -1)
            # Hands
            for side, color in [("left", (0, 255, 0)), ("right", (0, 200, 0))]:
                lms = vfr.get(f"anat_{side}_landmarks")
                if lms:
                    for hlm in lms:
                        px = int(hlm["raw_distorted_px_x"])
                        py = int(hlm["raw_distorted_px_y"])
                        cv2.circle(img, (px, py), 2, color, -1)

        # Draw reprojected 3D (red markers)
        for cat, key, color in [
            ("pose", "pose_landmarks_3d", (0, 0, 255)),
            ("left_hand", "left_hand_landmarks_3d", (255, 0, 0)),
            ("right_hand", "right_hand_landmarks_3d", (255, 100, 0)),
        ]:
            for lm in fr3d[key]:
                if not lm["valid"]:
                    continue
                cam_info = lm["per_camera"].get(view)
                if cam_info and cam_info.get("reproj_2d_px"):
                    rx, ry = cam_info["reproj_2d_px"]
                    cv2.circle(img, (int(rx), int(ry)), 3, color, 1)
                    # Draw reproj error as line from original to reproj
                    if cam_info.get("raw_2d_px"):
                        ox, oy = cam_info["raw_2d_px"]
                        cv2.line(img, (int(ox), int(oy)), (int(rx), int(ry)), (0, 255, 255), 1)

        # Info text
        cv2.putText(img, f"{view} t={cps:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        n_valid_pose = sum(1 for lm in fr3d["pose_landmarks_3d"] if lm["valid"])
        n_valid_lh = sum(1 for lm in fr3d["left_hand_landmarks_3d"] if lm["valid"])
        n_valid_rh = sum(1 for lm in fr3d["right_hand_landmarks_3d"] if lm["valid"])
        cv2.putText(img, f"3D: P={n_valid_pose}/33 L={n_valid_lh}/21 R={n_valid_rh}/21",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        writer.write(img)
        written += 1

    writer.release()
    con.close()
    print(f"    Overlay {view}: {written} frames -> {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  Contact sheet for anomalies
# ═══════════════════════════════════════════════════════════

def make_anomaly_contact_sheet(anomalies, synced_data_all, out_path):
    """Generate contact sheet for temporal jump anomalies."""
    if not anomalies:
        # Generate QA placeholder
        img = np.full((216, 768, 3), 50, dtype=np.uint8)
        cv2.putText(img, "QA: Temporal Jump Anomalies", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Threshold: {CONFIG['temporal_jump_warn_m']}m", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Total anomalies: {len(anomalies)}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(out_path), img)
        return

    # Sample up to 24 anomalies
    items = anomalies[:24] if len(anomalies) <= 24 else [
        anomalies[i] for i in np.linspace(0, len(anomalies)-1, 24, dtype=int)
    ]

    tw, th = 384, 216
    cols = min(4, len(items))
    rows = math.ceil(len(items) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)

    for idx, anom in enumerate(items):
        thumb = np.full((th, tw, 3), 30, dtype=np.uint8)
        cv2.putText(thumb, f"t={anom['front_time_s']:.2f}s", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(thumb, f"{anom['category']} lm{anom['landmark_index']}", (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1)
        cv2.putText(thumb, f"jump={anom['jump_m']*100:.1f}cm dt={anom['dt_s']*1000:.0f}ms", (5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        r, c = divmod(idx, cols)
        sheet[r*th:(r+1)*th, c*tw:(c+1)*tw] = thumb

    cv2.imwrite(str(out_path), sheet)
    print(f"  Anomaly contact sheet: {len(items)} entries -> {out_path.name}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    print("Step 6B v1.0 PILOT: Multi-view 3D Reconstruction - S001_P001 R03\n")

    # ── Verify inputs ──
    print("  Verifying inputs...")
    for fp in [C2_EXTRINSICS, C2_APPROVAL, SYNC_REPORT, SEGMENTS_FILE]:
        if not fp.exists():
            print(f"  MISSING: {fp}"); sys.exit(1)
    for v in VIEWS:
        for fp in [C1_INTRINSICS[v], STEP6A_DIR / f"landmarks_2d_{v.lower()}.json"]:
            if not fp.exists():
                print(f"  MISSING: {fp}"); sys.exit(1)

    # Verify C2 approval
    with open(C2_APPROVAL) as f:
        c2a = json.load(f)
    if c2a.get("status") != "APPROVED/PASS":
        print(f"  C2 not approved: {c2a.get('status')}"); sys.exit(1)
    print(f"  C2 extrinsics: APPROVED (LEFT baseline={c2a['baselines']['LEFT']['baseline_m']:.3f}m, "
          f"RIGHT baseline={c2a['baselines']['RIGHT']['baseline_m']:.3f}m)")

    # Verify coordinate convention
    conv = c2a["conventions"]["coordinate_transform"]
    print(f"  Convention: {conv['formula']} (unit: {conv['front_axes']['unit']})")

    # ── Load calibration ──
    print("\n  Loading calibration...")
    intrinsics, extrinsics, proj_matrices = load_calibration()
    for v in VIEWS:
        print(f"    {v}: K fx={intrinsics[v]['K'][0,0]:.1f} fy={intrinsics[v]['K'][1,1]:.1f}")

    # ── Load 2D landmarks ──
    print("\n  Loading 2D landmarks (frozen Step 6A)...")
    all_frames = load_2d_landmarks()

    # ── Synchronize frames ──
    print("\n  Synchronizing frames across views...")
    synced = synchronize_frames(all_frames)
    for sid in PILOT_SAMPLE_IDS:
        n = len(synced[sid])
        n3 = sum(1 for e in synced[sid] if e["n_views"] == 3)
        n2 = sum(1 for e in synced[sid] if e["n_views"] == 2)
        print(f"    {sid.replace('S001_P001_', '')}: {n} frames ({n3} 3-view, {n2} 2-view)")

    # ── Create output directories ──
    out_json = OUT_DIR / "raw_3d"
    out_ov = OUT_DIR / "overlays"
    out_cs = OUT_DIR / "contact_sheets"
    for d in [out_json, out_ov, out_cs]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Process each sample ──
    print("\n  Triangulating 3D landmarks...")
    all_results = {}
    all_qa_summaries = []

    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        print(f"\n  === {short} ===")

        frames_3d = process_sample(sid, synced[sid], intrinsics, extrinsics, proj_matrices)
        all_results[sid] = frames_3d

        # Save raw 3D JSON
        out_path = out_json / f"landmarks_3d_{short}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.0_pilot",
                "sample_id": sid,
                "coordinate_system": "FRONT_camera",
                "unit": "meters",
                "n_frames": len(frames_3d),
                "config": CONFIG,
                "frames": frames_3d,
            }, f, indent=1, ensure_ascii=False)
        sz_mb = out_path.stat().st_size / 1024 / 1024
        print(f"    Saved: {out_path.name} ({sz_mb:.1f} MB)")

        # Quick stats
        n_pose_valid = sum(1 for f in frames_3d
                           for lm in f["pose_landmarks_3d"] if lm["valid"])
        n_lh_valid = sum(1 for f in frames_3d
                          for lm in f["left_hand_landmarks_3d"] if lm["valid"])
        n_rh_valid = sum(1 for f in frames_3d
                          for lm in f["right_hand_landmarks_3d"] if lm["valid"])
        n_total_lm = len(frames_3d) * (33 + 21 + 21)
        n_valid_total = n_pose_valid + n_lh_valid + n_rh_valid
        print(f"    3D valid: {n_valid_total}/{n_total_lm} "
              f"(pose={n_pose_valid} L={n_lh_valid} R={n_rh_valid})")

        # Reproj stats
        rs = compute_reproj_stats(frames_3d)
        for cat in ["pose", "left_hand", "right_hand"]:
            if rs[cat].get("count", 0) > 0:
                print(f"    reproj {cat}: mean={rs[cat]['mean']:.1f}px "
                      f"median={rs[cat]['median']:.1f}px p95={rs[cat]['p95']:.1f}px "
                      f"max={rs[cat]['max']:.1f}px")

        # Coverage
        cov = compute_coverage(frames_3d)

        # Bone lengths
        bones = compute_bone_lengths(frames_3d)
        unstable = [k for k, v in bones.items() if v.get("stable") is False]
        if unstable:
            print(f"    Unstable bones: {len(unstable)}")

        # Temporal stability
        tstab, anomalies = compute_temporal_stability(frames_3d)

        # Depth stability
        z_osc = compute_depth_stability(frames_3d)

        qa_summary = {
            "sample_id": sid,
            "n_frames": len(frames_3d),
            "n_3view_frames": sum(1 for f in frames_3d if f["n_views_available"] == 3),
            "reproj_stats": rs,
            "coverage": cov,
            "n_unstable_bones": len(unstable),
            "n_temporal_anomalies": len(anomalies),
            "n_depth_oscillations": len(z_osc),
        }
        all_qa_summaries.append(qa_summary)

    # ── QA Flag analysis ──
    print("\n  " + "="*60)
    print("  3D QA Flag Analysis")
    print("  " + "="*60)
    qa_flag_report = analyze_qa_flags(all_results, synced)
    for item in qa_flag_report:
        print(f"    {item['flag']}: {item.get('sample_id','')} -> {item.get('recommendation', item.get('right_hand_3d_rate',''))}")

    # ── Generate overlays ──
    print("\n  Generating overlays...")
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        for v in VIEWS:
            ov_path = out_ov / f"reproj_{short}_{v.lower()}.mp4"
            make_reproj_overlay(v, all_results[sid], synced[sid], sid,
                                intrinsics, extrinsics, ov_path)

    # ── Generate contact sheets for anomalies ──
    print("\n  Generating contact sheets...")
    all_anomalies = []
    for sid in PILOT_SAMPLE_IDS:
        _, anomalies = compute_temporal_stability(all_results[sid])
        for a in anomalies:
            a["sample_id"] = sid
        all_anomalies.extend(anomalies)
    make_anomaly_contact_sheet(all_anomalies, None, out_cs / "contact_temporal_anomalies.png")

    # ── Save QA CSVs ──
    print("\n  Saving QA artifacts...")

    # Per-frame per-landmark QA CSV
    qa_rows = []
    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    qa_rows.append({
                        "sample_id": sid,
                        "frame_index": fi,
                        "front_time_s": fr["front_time_s"],
                        "category": cat,
                        "landmark_index": lm["lm_index"],
                        "valid": lm["valid"],
                        "n_cameras": lm["n_cameras"],
                        "cameras_used": ",".join(lm["cameras_used"]) if lm["cameras_used"] else "",
                        "x": lm["xyz"][0] if lm["xyz"] else None,
                        "y": lm["xyz"][1] if lm["xyz"] else None,
                        "z": lm["xyz"][2] if lm["xyz"] else None,
                        "reproj_error_mean_px": lm["reproj_error_mean_px"],
                        "reproj_error_max_px": lm["reproj_error_max_px"],
                        "confidence": lm["confidence"],
                        "invalid_reason": lm["invalid_reason"],
                    })
    qa_csv = OUT_DIR / "frame_landmark_qa.csv"
    with open(qa_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=qa_rows[0].keys())
        w.writeheader(); w.writerows(qa_rows)
    print(f"  Saved: {qa_csv.name} ({len(qa_rows)} rows)")

    # Per-sample QA summary CSV
    summary_rows = []
    for qs in all_qa_summaries:
        row = {
            "sample_id": qs["sample_id"],
            "n_frames": qs["n_frames"],
            "n_3view_frames": qs["n_3view_frames"],
            "n_unstable_bones": qs["n_unstable_bones"],
            "n_temporal_anomalies": qs["n_temporal_anomalies"],
            "n_depth_oscillations": qs["n_depth_oscillations"],
        }
        for cat in ["pose", "left_hand", "right_hand"]:
            cs = qs["reproj_stats"].get(cat, {})
            row[f"reproj_{cat}_mean"] = cs.get("mean")
            row[f"reproj_{cat}_p95"] = cs.get("p95")
            row[f"reproj_{cat}_max"] = cs.get("max")
            cc = qs["coverage"].get(cat, {})
            row[f"coverage_{cat}_mean"] = cc.get("mean_rate")
            row[f"coverage_{cat}_min"] = cc.get("min_rate")
        summary_rows.append(row)
    sum_csv = OUT_DIR / "sample_qa_summary.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)
    print(f"  Saved: {sum_csv.name}")

    # Per-camera reproj CSV
    cam_rows = []
    for sid in PILOT_SAMPLE_IDS:
        rs = compute_reproj_stats(all_results[sid])
        for v in VIEWS:
            cs = rs.get("per_camera", {}).get(v, {})
            cam_rows.append({
                "sample_id": sid, "camera": v,
                "reproj_count": cs.get("count", 0),
                "reproj_mean": cs.get("mean"), "reproj_median": cs.get("median"),
                "reproj_p95": cs.get("p95"), "reproj_max": cs.get("max"),
            })
    cam_csv = OUT_DIR / "camera_reproj_errors.csv"
    with open(cam_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cam_rows[0].keys())
        w.writeheader(); w.writerows(cam_rows)
    print(f"  Saved: {cam_csv.name}")

    # Bone length CSV
    bone_rows = []
    for sid in PILOT_SAMPLE_IDS:
        bones = compute_bone_lengths(all_results[sid])
        for name, bs in bones.items():
            bone_rows.append({
                "sample_id": sid, "bone": name,
                "n": bs["n"], "median_m": bs.get("median_m"),
                "std_m": bs.get("std_m"), "cv": bs.get("cv"),
                "min_m": bs.get("min_m"), "max_m": bs.get("max_m"),
                "stable": bs.get("stable"),
            })
    bone_csv = OUT_DIR / "bone_lengths.csv"
    with open(bone_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=bone_rows[0].keys())
        w.writeheader(); w.writerows(bone_rows)
    print(f"  Saved: {bone_csv.name}")

    # ── Manifest ──
    input_sha = {
        "c1_front": sha256_file(C1_INTRINSICS["FRONT"]),
        "c1_left": sha256_file(C1_INTRINSICS["LEFT"]),
        "c1_right": sha256_file(C1_INTRINSICS["RIGHT"]),
        "c2_extrinsics": sha256_file(C2_EXTRINSICS),
        "c2_approval": sha256_file(C2_APPROVAL),
        "sync_report": sha256_file(SYNC_REPORT),
        "2d_front": sha256_file(STEP6A_DIR / "landmarks_2d_front.json"),
        "2d_left": sha256_file(STEP6A_DIR / "landmarks_2d_left.json"),
        "2d_right": sha256_file(STEP6A_DIR / "landmarks_2d_right.json"),
    }
    manifest = {
        "step": "6B_pilot_v1.0",
        "date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "coordinate_system": "FRONT_camera (X=right, Y=down, Z=forward)",
        "unit": "meters",
        "triangulation_method": "DLT (cv2.triangulatePoints for 2-view, SVD for 3-view)",
        "view_selection": "all valid views; worst excluded if reproj > 50px",
        "smoothing": "DISABLED (raw only)",
        "input_sha256": input_sha,
        "config": CONFIG,
        "pilot_sample_ids": PILOT_SAMPLE_IDS,
        "calibration_files": {
            "c1_intrinsics": {v: str(C1_INTRINSICS[v]) for v in VIEWS},
            "c2_extrinsics": str(C2_EXTRINSICS),
            "sync_report": str(SYNC_REPORT),
        },
        "c2_convention": c2a["conventions"],
    }
    man_path = OUT_DIR / "run_manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {man_path.name}")

    # ── Config file ──
    cfg_path = OUT_DIR / "config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"  Saved: {cfg_path.name}")

    # ── QA Flag report JSON ──
    flag_path = OUT_DIR / "qa_flag_analysis.json"
    with open(flag_path, "w", encoding="utf-8") as f:
        json.dump(qa_flag_report, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {flag_path.name}")

    # ══ REPORT ══
    print("\n  Generating report...")
    report = []
    report.append("# Step 6B v1.0 Pilot Report -- Multi-view 3D Reconstruction\n")
    report.append("## Status: REQUIRES_EXTERNAL_FULL_EXTRACTION_REVIEW\n")
    report.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    report.append(f"**Session**: S001_P001")
    report.append(f"**Script**: `Tools/s001_step6b_triangulate_pilot.py`")
    report.append(f"**Scope**: 9 R03 samples x 3 views\n")

    report.append("## Calibration Files Used\n")
    report.append(f"- C1 intrinsics (v2): `Tools/calibration/results/C1/validation/intrinsics_CAM_*_01_v2.json`")
    report.append(f"- C2 extrinsics (v2 robust): `data/processed/sessions/S001_P001/C2_CALIBRATION_v2_robust.json`")
    report.append(f"- Sync: `data/processed/sessions/S001_P001/sync_report.json`")
    report.append(f"- 2D landmarks: `data/processed/sessions/S001_P001/landmarks_2d_v1_4/full/`\n")

    report.append("## Coordinate System\n")
    report.append(f"- **Reference**: FRONT camera")
    report.append(f"- **Axes**: X=right, Y=down, Z=forward (into scene)")
    report.append(f"- **Unit**: meters")
    report.append(f"- **Transform**: X_secondary = R @ X_front + t\n")

    report.append("## Triangulation Method\n")
    report.append("- DLT (Direct Linear Transform)")
    report.append("- 2 views: `cv2.triangulatePoints`")
    report.append("- 3 views: SVD-based linear least squares")
    report.append(f"- View exclusion: reproj > {CONFIG['reproj_exclude_px']}px -> re-triangulate with best 2")
    report.append("- Smoothing: **DISABLED** (raw 3D only)\n")

    report.append("## Per-Sample Results\n")
    report.append("| Sample | Frames | 3-view | Pose reproj | LH reproj | RH reproj | Pose cov | LH cov | RH cov | Anom |")
    report.append("|--------|--------|--------|------------|-----------|-----------|---------|--------|--------|------|")
    for qs in all_qa_summaries:
        short = qs["sample_id"].replace("S001_P001_", "")
        pr = qs["reproj_stats"].get("pose", {})
        lr = qs["reproj_stats"].get("left_hand", {})
        rr = qs["reproj_stats"].get("right_hand", {})
        pc = qs["coverage"].get("pose", {})
        lc = qs["coverage"].get("left_hand", {})
        rc = qs["coverage"].get("right_hand", {})
        report.append(
            f"| {short} | {qs['n_frames']} | {qs['n_3view_frames']} | "
            f"{pr.get('median','N/A'):.1f}px | {lr.get('median','N/A'):.1f}px | {rr.get('median','N/A'):.1f}px | "
            f"{pc.get('mean_rate',0):.0%} | {lc.get('mean_rate',0):.0%} | {rc.get('mean_rate',0):.0%} | "
            f"{qs['n_temporal_anomalies']} |"
            if isinstance(pr.get('median'), (int, float)) else
            f"| {short} | {qs['n_frames']} | {qs['n_3view_frames']} | N/A | N/A | N/A | "
            f"{pc.get('mean_rate',0):.0%} | {lc.get('mean_rate',0):.0%} | {rc.get('mean_rate',0):.0%} | "
            f"{qs['n_temporal_anomalies']} |"
        )
    report.append("")

    report.append("## 3D QA Flag Analysis\n")
    for item in qa_flag_report:
        report.append(f"### {item['flag']}: {item.get('sample_id','')}\n")
        for k, v in item.items():
            if k not in ("flag",):
                report.append(f"- **{k}**: {v}")
        report.append("")

    report.append("## Recommended QA Thresholds for Full 45-Sample Extraction\n")
    report.append("| Parameter | Current | Recommended | Rationale |")
    report.append("|-----------|---------|-------------|-----------|")
    # Compute actual distributions across all samples
    all_reproj = []
    for sid in PILOT_SAMPLE_IDS:
        for f in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in f[key]:
                    if lm["valid"] and lm["reproj_error_mean_px"] is not None:
                        all_reproj.append(lm["reproj_error_mean_px"])
    if all_reproj:
        p50 = float(np.percentile(all_reproj, 50))
        p95 = float(np.percentile(all_reproj, 95))
        p99 = float(np.percentile(all_reproj, 99))
        report.append(f"| reproj_outlier_px | {CONFIG['reproj_outlier_px']} | {max(p95*1.5, 15):.0f} | p95={p95:.1f}px, p99={p99:.1f}px |")
        report.append(f"| reproj_exclude_px | {CONFIG['reproj_exclude_px']} | {max(p99*2, 30):.0f} | p99={p99:.1f}px |")
    report.append(f"| temporal_jump_warn_m | {CONFIG['temporal_jump_warn_m']} | TBD | Based on pilot distribution |")
    report.append(f"| bone_length_variation_warn | {CONFIG['bone_length_variation_warn']} | TBD | Based on pilot distribution |")
    report.append("")

    report.append("## Reprojection Error Distribution (all samples pooled)\n")
    if all_reproj:
        report.append(f"- Count: {len(all_reproj)}")
        report.append(f"- Mean: {float(np.mean(all_reproj)):.2f}px")
        report.append(f"- Median: {float(np.median(all_reproj)):.2f}px")
        report.append(f"- P95: {float(np.percentile(all_reproj, 95)):.2f}px")
        report.append(f"- P99: {float(np.percentile(all_reproj, 99)):.2f}px")
        report.append(f"- Max: {float(np.max(all_reproj)):.2f}px")
    report.append("")

    report.append("## Scalability Assessment for Full 45-Sample Extraction\n")
    report.append("- Pipeline is sample-independent; no hardcoded pilot IDs in core logic")
    report.append("- Calibration data (C1, C2, sync) is session-level, shared across all samples")
    report.append("- 2D landmarks already extracted for all 45 samples")
    report.append("- Estimated output size: ~5x pilot (JSON ~500MB total)")
    report.append("- Estimated runtime: ~5x pilot")
    report.append("")

    report.append("## Output Files\n")
    report.append("| File | Size |")
    report.append("|------|------|")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            sk = p.stat().st_size / 1024
            sz_str = f"{sk/1024:.1f} MB" if sk > 1024 else f"{sk:.1f} KB"
            report.append(f"| {p.relative_to(OUT_DIR)} | {sz_str} |")
    report.append("")

    report.append("## Null Policy\n")
    report.append("- Landmarks with < 2 views: not triangulated (invalid, null xyz)")
    report.append("- No interpolation, no gap-filling, no smoothing")
    report.append("- Per-camera 2D data preserved even for invalid 3D points")
    report.append("- Invalid reason recorded for every non-valid landmark")
    report.append("")

    rpt_path = OUT_DIR / "STEP6B_PILOT_REPORT.md"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"  Saved: {rpt_path.name}")

    # ══ ZIP ══
    print("\n  Creating delivery ZIP...")
    zip_path = PROJECT_ROOT / "WordsInSilence_step6b_v1_0_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "Tools/s001_step6b_triangulate_pilot.py")
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                arcname = str(p.relative_to(PROC_DIR))
                zf.write(p, arcname)
    print(f"  ZIP: {zip_path.name} ({zip_path.stat().st_size/1024/1024:.1f} MB)")

    # ── Final summary ──
    print(f"\n  {'='*60}")
    print(f"  STEP 6B PILOT SUMMARY")
    print(f"  {'='*60}")
    print(f"  Samples: {len(PILOT_SAMPLE_IDS)}")
    total_frames = sum(len(all_results[sid]) for sid in PILOT_SAMPLE_IDS)
    print(f"  Total frames: {total_frames}")
    total_valid = sum(
        1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
        for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
        for lm in f[key] if lm["valid"]
    )
    total_lm = total_frames * 75
    print(f"  Valid 3D landmarks: {total_valid}/{total_lm} ({total_valid/max(total_lm,1):.1%})")
    if all_reproj:
        print(f"  Reproj error (all): mean={float(np.mean(all_reproj)):.1f}px "
              f"median={float(np.median(all_reproj)):.1f}px p95={float(np.percentile(all_reproj, 95)):.1f}px")
    print(f"  Output: {OUT_DIR}")
    print(f"  STATUS: REQUIRES_EXTERNAL_FULL_EXTRACTION_REVIEW")


if __name__ == "__main__":
    main()
