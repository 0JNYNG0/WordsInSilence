"""
Step 6B v1.1 PILOT: Multi-view 3D Reconstruction -- S001_P001 R03 (9 samples)

Changes from v1.0:
  - Multi-candidate view selection (all 2/3-view combos scored)
  - Separate triangulation_valid / qa_valid / template_eligible
  - Cheirality, depth, finite, triangulation angle checks per candidate
  - Post-processing bone length QA (MAD-based)
  - Post-processing temporal QA (velocity/accel with actual dt, categorized)
  - Detailed QA flag re-evaluation with per-landmark camera usage
  - Real video thumbnail contact sheets
  - Color-coded overlays with decoded_frame_index matching
  - Thresholds kept at 20/50px
  - Comprehensive distribution reports and assertions
  - Regression: GREETING_NORMAL_R03 frame 39 right hand bone outliers
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
OUT_DIR = PROC_DIR / "landmarks_3d_v1" / "pilot_v1_1"

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

CONFIG = {
    "sync_tolerance_s": 0.020,
    "min_views_for_3d": 2,
    "reproj_outlier_px": 20.0,
    "reproj_exclude_px": 50.0,
    "min_depth_m": 0.3,
    "max_depth_m": 10.0,
    "min_triangulation_angle_deg": 2.0,
    "cam_count_score_bonus": 10.0,
    "angle_score_weight": 0.5,
    "angle_score_cap_deg": 30.0,
    "validation_penalty_threshold_px": 30.0,
    "validation_penalty_weight": 0.1,
    "high_reproj_penalty_weight": 0.3,
    "bone_outlier_mad_factor": 5.0,
    "bone_max_finger_phalanx_m": 0.06,
    "bone_max_metacarpal_m": 0.12,
    "temporal_velocity_warn_ms": 5.0,
    "temporal_max_gap_s": 0.1,
    "depth_oscillation_warn_m": 0.05,
    "apply_smoothing": False,
    "contact_thumb_size": (384, 216),
    "contact_max_cells": 24,
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
    """Distribution stats helper."""
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
#  Multi-candidate view selection
# ================================================================

def _view_abbrev(views):
    return "+".join(v[0] for v in views)


def evaluate_candidate(cand_views, obs, all_obs_views,
                       intrinsics, extrinsics, proj_matrices, camera_centers):
    """Evaluate one triangulation candidate."""
    pts = [obs[v]["undistorted_px"] for v in cand_views]
    pmats = [proj_matrices[v] for v in cand_views]
    X3 = triangulate_point(pts, pmats)

    ev = {
        "views": _view_abbrev(cand_views), "cameras": list(cand_views),
        "feasible": False, "score": None, "reason": "",
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

    # Cheirality -- all observed cameras
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

    # Reproj errors
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

    # Triangulation angle
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

    # Feasible -- compute score
    ev["feasible"] = True
    ev["xyz"] = X3.tolist()
    nc = len(cand_views)
    s = nc * CONFIG["cam_count_score_bonus"]
    s -= ev["used_reproj_mean"]
    s += min(min_ang, CONFIG["angle_score_cap_deg"]) * CONFIG["angle_score_weight"]
    if val_e:
        s -= max(0, max(val_e) - CONFIG["validation_penalty_threshold_px"]) * CONFIG["validation_penalty_weight"]
    if ev["used_reproj_max"] > CONFIG["reproj_outlier_px"]:
        s -= (ev["used_reproj_max"] - CONFIG["reproj_outlier_px"]) * CONFIG["high_reproj_penalty_weight"]
    ev["score"] = round(s, 2)
    ev["reason"] = f"{nc}v_s{s:.1f}"
    return ev


def triangulate_landmark_v11(synced_entry, lm_type, lm_index, hand_side,
                              intrinsics, extrinsics, proj_matrices, camera_centers):
    obs = {}
    for v, frame in synced_entry["views"].items():
        lm2d = extract_landmark_2d(frame, lm_type, lm_index, hand_side)
        if lm2d is not None:
            obs[v] = lm2d

    n_obs = len(obs)
    result = {
        "lm_type": lm_type, "lm_index": lm_index, "hand_side": hand_side,
        "triangulation_valid": False, "qa_valid": False, "template_eligible": False,
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
    best, best_score = None, -float("inf")

    for r in range(2, n_obs + 1):
        for combo in combinations(obs_views, r):
            ev = evaluate_candidate(list(combo), obs, obs_views,
                                     intrinsics, extrinsics, proj_matrices, camera_centers)
            result["candidates"].append({
                "v": ev["views"], "ok": ev["feasible"],
                "s": ev["score"], "ur": ev["used_reproj_mean"],
                "um": ev["used_reproj_max"], "vr": ev["validation_reproj_mean"],
                "a": ev["angle_deg"], "d": ev["depth_m"], "r": ev["reason"],
            })
            if ev["feasible"] and ev["score"] > best_score:
                best_score = ev["score"]
                best = ev

    if best is None:
        result["invalid_reason"] = "no_feasible_candidate"
        result["selection_reason"] = "all_candidates_failed"
        return result

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
    result["template_eligible"] = True

    if best["used_reproj_max"] > CONFIG["reproj_outlier_px"]:
        result["qa_flags"].append(f"high_used_reproj_{best['used_reproj_max']:.0f}px")
    if result["excluded_cameras"]:
        result["qa_flags"].append(f"excluded_{'_'.join(result['excluded_cameras'])}")

    for v in obs_views:
        cam_ev = best["per_camera"].get(v, {})
        result["per_camera"][v]["reproj_2d_px"] = cam_ev.get("reproj_px")
        result["per_camera"][v]["reproj_error_px"] = cam_ev.get("error")
        result["per_camera"][v]["used_for_triangulation"] = v in best["cameras"]

    return result


# ================================================================
#  Process one sample
# ================================================================

def process_sample(sid, synced_frames, intrinsics, extrinsics, proj_matrices, cam_centers):
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
                triangulate_landmark_v11(entry, "pose", li, None,
                                          intrinsics, extrinsics, proj_matrices, cam_centers))
        for hs in ["left", "right"]:
            key = f"{hs}_hand_landmarks_3d"
            for li in range(21):
                fr[key].append(
                    triangulate_landmark_v11(entry, "hand", li, hs,
                                              intrinsics, extrinsics, proj_matrices, cam_centers))
        frames_3d.append(fr)
    return frames_3d


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
            bone_lens = defaultdict(list)  # bone_name -> [(fi, length)]
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
#  Post-processing: Temporal QA
# ================================================================

def apply_temporal_qa(all_results):
    temporal_anomalies = []

    for sid in PILOT_SAMPLE_IDS:
        frames = all_results[sid]
        if len(frames) < 2:
            continue

        for fi in range(1, len(frames)):
            prev, curr = frames[fi - 1], frames[fi]
            dt = curr["front_time_s"] - prev["front_time_s"]
            if dt <= 0 or dt > CONFIG["temporal_max_gap_s"]:
                continue

            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for li in range(len(curr[key])):
                    lp, lc = prev[key][li], curr[key][li]
                    if lc["triangulation_valid"] and not lp["triangulation_valid"]:
                        lc["qa_flags"].append("detection_reappear")
                        continue
                    if not (lp["triangulation_valid"] and lc["triangulation_valid"]):
                        continue

                    p1, p2 = np.array(lp["xyz"]), np.array(lc["xyz"])
                    d = float(np.linalg.norm(p2 - p1))
                    vel = d / dt

                    if vel <= CONFIG["temporal_velocity_warn_ms"]:
                        continue

                    xy_d = float(np.linalg.norm(p2[:2] - p1[:2]))
                    z_d = abs(float(p2[2] - p1[2]))
                    cams_changed = set(lp["cameras_used"]) != set(lc["cameras_used"])
                    has_bone = any("bone_" in f for f in lc["qa_flags"])
                    depth_only = z_d > 3 * xy_d and z_d > 0.05

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

                    temporal_anomalies.append({
                        "sample_id": sid, "frame_index": fi,
                        "front_time_s": round(curr["front_time_s"], 6),
                        "category": cat, "landmark_index": li,
                        "distance_m": round(d, 4), "velocity_ms": round(vel, 2),
                        "dt_s": round(dt, 6),
                        "xy_dist_m": round(xy_d, 4), "z_dist_m": round(z_d, 4),
                        "cameras_prev": lp["cameras_used"],
                        "cameras_curr": lc["cameras_used"],
                        "reason": reason,
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
                    # Pre-selection 3-view residual
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


def compute_coverage_v11(frames_3d):
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


def compute_exclusion_stats(all_results):
    excl = {v: {"pose": 0, "left_hand": 0, "right_hand": 0} for v in VIEWS}
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
                            excl[v][cat] += 1
    return {"per_camera_category": excl, "total_excluded": total_excl}


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


def compute_depth_stability(frames_3d):
    z_series = defaultdict(list)
    for fr in frames_3d:
        for cat, key in [("pose", "pose_landmarks_3d"),
                          ("left_hand", "left_hand_landmarks_3d"),
                          ("right_hand", "right_hand_landmarks_3d")]:
            for lm in fr[key]:
                if lm["triangulation_valid"]:
                    z_series[(cat, lm["lm_index"])].append(lm["xyz"][2])
    oscs = []
    for (cat, li), zs in z_series.items():
        if len(zs) < 3:
            continue
        za = np.array(zs)
        zr = float(np.max(za) - np.min(za))
        if zr > CONFIG["depth_oscillation_warn_m"] * 5:
            oscs.append({"category": cat, "lm_index": li, "z_range_m": round(zr, 4), "n": len(zs)})
    return oscs


# ================================================================
#  QA Flag detailed analysis
# ================================================================

def analyze_qa_flags_v11(all_results):
    report = []

    # Right hand detection
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

    # THANK_YOU_FAST_R03 frames 48, 49 detailed per-landmark
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
                "left_err": left_cam.get("reproj_error_px"),
                "left_used": left_cam.get("used_for_triangulation", False),
                "flags": lm["qa_flags"],
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


# ================================================================
#  Overlays (one pass per view)
# ================================================================

def draw_overlay(img, fr3d, view, frame_idx):
    """Draw overlay annotations on one frame."""
    for cat_key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
        is_pose = cat_key == "pose_landmarks_3d"
        for lm in fr3d[cat_key]:
            cam = lm["per_camera"].get(view)
            if not cam:
                continue
            # Original 2D
            if cam.get("raw_2d_px"):
                px, py = int(cam["raw_2d_px"][0]), int(cam["raw_2d_px"][1])
                used = cam.get("used_for_triangulation", False)
                col = (0, 255, 0) if used else (255, 255, 0)
                cv2.circle(img, (px, py), 3 if is_pose else 2, col, -1)
            # Reprojected 3D
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
    """Generate overlay videos for all samples in one pass per view."""
    # Support multiple synced frames mapping to same decoded_frame_index
    dfi_lookup = defaultdict(list)  # decoded_frame_index -> [(sid, synced_index)]
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
                   overlay_counts, contact_paths):
    results = []

    def chk(cond, msg):
        results.append(("PASS" if cond else "FAIL", msg))

    # 1. All 9 sequences
    chk(len(all_results) == 9, f"9_sequences (got {len(all_results)})")

    # 2. Frame counts + structure
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        frames = all_results.get(sid, [])
        expected = len(synced.get(sid, []))
        chk(len(frames) == expected, f"{short}_frames: {len(frames)}=={expected}")
        for fi, fr in enumerate(frames):
            if len(fr["pose_landmarks_3d"]) != 33:
                chk(False, f"{short}_f{fi}_pose!=33"); break
            if len(fr["left_hand_landmarks_3d"]) != 21:
                chk(False, f"{short}_f{fi}_lh!=21"); break
            if len(fr["right_hand_landmarks_3d"]) != 21:
                chk(False, f"{short}_f{fi}_rh!=21"); break
        else:
            chk(True, f"{short}_structure_ok")

    # 3. Invalid landmarks have no coords
    n_bad = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if not lm["triangulation_valid"] and lm["xyz"] is not None:
                        n_bad += 1
    chk(n_bad == 0, f"no_coords_for_invalid (violations={n_bad})")

    # 4. Camera count matches used flag
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

    # 5. Cheirality
    n_behind = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["xyz"] is not None:
                        if lm["xyz"][2] <= 0:
                            n_behind += 1
    chk(n_behind == 0, f"cheirality (behind={n_behind})")

    # 6. qa_valid landmarks within reproj threshold
    n_over = 0
    for sid in PILOT_SAMPLE_IDS:
        for fr in all_results[sid]:
            for key in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]:
                for lm in fr[key]:
                    if lm["qa_valid"] and lm["used_reproj_max_px"] is not None:
                        if lm["used_reproj_max_px"] > CONFIG["reproj_exclude_px"]:
                            n_over += 1
    chk(n_over == 0, f"qa_valid_within_reproj (over={n_over})")

    # 7. Regression: GREETING_NORMAL_R03 frame 39 right hand
    #    v1.0 had anomalous bone lengths (0.13-0.20m) due to bad 3-view triangulation.
    #    v1.1 multi-candidate selection should fix this by excluding the bad view.
    #    Check: either bone lengths are now normal (fixed), or if still abnormal, they're flagged.
    reg = REGRESSION_BONE
    frames = all_results.get(reg["sample_id"], [])
    if reg["frame_index"] < len(frames):
        rh = frames[reg["frame_index"]]["right_hand_landmarks_3d"]
        import numpy as _np
        regression_ok = True
        regression_details = []
        for bi, bj in reg["bones"]:
            if rh[bi]["triangulation_valid"] and rh[bj]["triangulation_valid"]:
                bl = float(_np.linalg.norm(
                    _np.array(rh[bi]["xyz"]) - _np.array(rh[bj]["xyz"])))
                if bl >= reg["min_expected_m"]:
                    # Still abnormal -- must be flagged
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

    # 8. THANK_YOU_FAST frames 48,49 LEFT exclusion reported
    ty = REGRESSION_TY_FRAMES
    ty_frames = all_results.get(ty["sample_id"], [])
    for ffi in ty["front_frame_indices"]:
        if ffi < len(ty_frames):
            lh = ty_frames[ffi]["left_hand_landmarks_3d"]
            n_le = sum(1 for lm in lh if lm["triangulation_valid"] and "LEFT" in lm.get("excluded_cameras", []))
            chk(True, f"ty_fast_f{ffi}_left_excl={n_le}")
        else:
            chk(False, f"ty_fast_f{ffi}_not_found")

    # 9. Overlay frame counts
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        expected = len(all_results[sid])
        for v in VIEWS:
            k = f"{short}_{v.lower()}"
            actual = overlay_counts.get(k, -1)
            chk(actual == expected, f"overlay_{k}: {actual}=={expected}")

    # 10. Contact sheets have images (file size > 10KB)
    for cp in contact_paths:
        if cp.exists():
            sz = cp.stat().st_size
            chk(sz > 10000, f"contact_{cp.name}_size={sz}")
        else:
            chk(False, f"contact_{cp.name}_missing")

    # 11. FAIL count
    n_fail = sum(1 for s, _ in results if s == "FAIL")
    # (This is a meta-check; reported separately)

    return results


# ================================================================
#  Main
# ================================================================

def main():
    print("Step 6B v1.1 PILOT: Multi-view 3D Reconstruction - S001_P001 R03\n")
    print("  Status of v1.0: EXTERNAL_REVIEW_REVISION_REQUIRED_BEFORE_FULL_45\n")

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

    # -- Triangulate --
    print("\n  Triangulating 3D landmarks (multi-candidate view selection)...")
    all_results = {}
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        print(f"\n  === {short} ===")
        frames_3d = process_sample(sid, synced[sid], intrinsics, extrinsics, proj_matrices, cam_centers)
        all_results[sid] = frames_3d
        n_tri = sum(1 for f in frames_3d for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"] for lm in f[k] if lm["triangulation_valid"])
        n_total = len(frames_3d) * 75
        print(f"    tri_valid: {n_tri}/{n_total} ({n_tri/max(n_total,1):.1%})")

    # -- Post-processing QA --
    print("\n  Post-processing: bone length QA...")
    bone_outliers = apply_bone_qa(all_results)
    print(f"    Bone outliers flagged: {len(bone_outliers)}")

    print("  Post-processing: temporal QA...")
    temporal_anomalies = apply_temporal_qa(all_results)
    print(f"    Temporal anomalies flagged: {len(temporal_anomalies)}")

    # -- Save 3D JSONs --
    print("\n  Saving 3D JSONs...")
    for sid in PILOT_SAMPLE_IDS:
        short = sid.replace("S001_P001_", "")
        op = out_json / f"landmarks_3d_{short}.json"
        with open(op, "w", encoding="utf-8") as f:
            json.dump({
                "version": "v1.1_pilot",
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
        cov = compute_coverage_v11(all_results[sid])
        bs = compute_bone_stats(all_results[sid])
        ds = compute_depth_stability(all_results[sid])
        n_bone_out = sum(1 for bo in bone_outliers if bo["sample_id"] == sid)
        n_temp_anom = sum(1 for ta in temporal_anomalies if ta["sample_id"] == sid)

        all_qa_summaries.append({
            "sample_id": sid,
            "n_frames": len(all_results[sid]),
            "n_synced_3view_frames": sum(1 for f in all_results[sid] if f["n_views_synced"] == 3),
            "coverage": cov,
            "n_bone_outliers": n_bone_out,
            "n_temporal_anomalies": n_temp_anom,
            "n_depth_oscillations": len(ds),
        })

    reproj_dists = compute_reproj_distributions(all_results)
    excl_stats = compute_exclusion_stats(all_results)

    # -- QA flag analysis --
    print("\n  Analyzing QA flags...")
    qa_flag_report = analyze_qa_flags_v11(all_results)
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

    # -- Contact sheets --
    print("\n  Generating contact sheets with video thumbnails...")

    # Collect all FRONT decoded_frame_indices needed
    needed_dfis = set()
    cs_items = {"temporal": [], "bone": [], "reproj": [], "exclusion": [], "qa_flag": []}

    for ta in temporal_anomalies[:100]:
        sid = ta["sample_id"]
        fi = ta["frame_index"]
        fr = all_results[sid][fi]
        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items["temporal"].append({
            "dfi": dfi,
            "text": [
                f"{sid.replace('S001_P001_','')} f{fi}",
                f"t={ta['front_time_s']:.3f}s {ta['category']}",
                f"lm{ta['landmark_index']} v={ta['velocity_ms']:.1f}m/s",
                f"d={ta['distance_m']:.4f}m dt={ta['dt_s']*1000:.0f}ms",
                f"cams:{','.join(c[0] for c in ta['cameras_curr'])}",
                f"reason: {ta['reason']}",
            ],
        })

    for bo in bone_outliers[:100]:
        sid = bo["sample_id"]
        fi = bo["frame_index"]
        fr = all_results[sid][fi]
        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items["bone"].append({
            "dfi": dfi,
            "text": [
                f"{sid.replace('S001_P001_','')} f{fi}",
                f"t={bo['front_time_s']:.3f}s {bo['category']}",
                f"bone: {bo['bone']}",
                f"len={bo['length_m']:.4f}m med={bo['median_m']:.4f}m",
                f"dev={bo['dev_factor']:.1f}x MAD",
            ],
        })

    # Reproj outliers (used_reproj > outlier threshold)
    reproj_outlier_items = []
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
                            reproj_outlier_items.append({
                                "dfi": dfi,
                                "text": [
                                    f"{sid.replace('S001_P001_','')} f{fi}",
                                    f"t={fr['front_time_s']:.3f}s {cat}",
                                    f"lm{lm['lm_index']} cams:{','.join(c[0] for c in lm['cameras_used'])}",
                                    f"used_max={lm['used_reproj_max_px']:.1f}px",
                                    f"qa={lm['qa_valid']} te={lm['template_eligible']}",
                                ],
                            })
    cs_items["reproj"] = reproj_outlier_items[:100]

    # Camera exclusion items
    excl_items = []
    for sid in PILOT_SAMPLE_IDS:
        for fi, fr in enumerate(all_results[sid]):
            for cat, key in [("pose", "pose_landmarks_3d"),
                              ("left_hand", "left_hand_landmarks_3d"),
                              ("right_hand", "right_hand_landmarks_3d")]:
                for lm in fr[key]:
                    if lm["triangulation_valid"] and lm["excluded_cameras"]:
                        dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
                        if dfi is not None:
                            needed_dfis.add(dfi)
                        excl_items.append({
                            "dfi": dfi,
                            "text": [
                                f"{sid.replace('S001_P001_','')} f{fi}",
                                f"{cat} lm{lm['lm_index']}",
                                f"used: {','.join(c[0] for c in lm['cameras_used'])}",
                                f"excl: {','.join(c[0] for c in lm['excluded_cameras'])}",
                                f"ur={lm['used_reproj_mean_px']}px",
                                f"er={lm['excluded_reproj_mean_px']}px",
                            ],
                        })
    cs_items["exclusion"] = excl_items[:100]

    # QA flag frames
    for item in qa_flag_report:
        if item.get("flag") == "thankyou_fast_detailed" and item.get("found"):
            fi = item["frame_index"]
            fr = all_results[item["sample_id"]][fi]
            dfi = fr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
            if dfi is not None:
                needed_dfis.add(dfi)
            cs_items["qa_flag"].append({
                "dfi": dfi,
                "text": [
                    f"TY_FAST f{fi} t={item['front_time_s']:.3f}s",
                    f"views: {','.join(item['views_available'])}",
                    f"left_excl={item['n_left_excluded']}/21",
                    f"qa={item['n_qa']}/21 te={item['n_te']}/21",
                    "hand crossover region",
                ],
            })
    # Add regression bone frame
    reg = REGRESSION_BONE
    reg_frames = all_results.get(reg["sample_id"], [])
    if reg["frame_index"] < len(reg_frames):
        rfr = reg_frames[reg["frame_index"]]
        dfi = rfr["per_view_info"].get("FRONT", {}).get("decoded_frame_index")
        if dfi is not None:
            needed_dfis.add(dfi)
        cs_items["qa_flag"].append({
            "dfi": dfi,
            "text": [
                f"GREETING_NORMAL f{reg['frame_index']}",
                f"t={rfr['front_time_s']:.3f}s",
                "right hand bone outlier",
                f"bones: {reg['bones']}",
                "REGRESSION TEST",
            ],
        })

    # Extract FRONT video frames
    print(f"  Extracting {len(needed_dfis)} FRONT video frames for contact sheets...")
    video_frames = extract_video_frames(VIDEOS["FRONT"], sorted(needed_dfis))
    print(f"    Extracted: {len(video_frames)} frames")

    contact_paths = []
    for cs_type, items in cs_items.items():
        cp = out_cs / f"contact_{cs_type}.png"
        make_contact_sheet(items, video_frames, cp, cs_type)
        contact_paths.append(cp)

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
                    qa_rows.append({
                        "sample_id": sid, "frame_index": fi,
                        "front_time_s": fr["front_time_s"],
                        "category": cat, "landmark_index": lm["lm_index"],
                        "triangulation_valid": lm["triangulation_valid"],
                        "qa_valid": lm["qa_valid"],
                        "template_eligible": lm["template_eligible"],
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

    # Sample QA summary CSV
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

    # Camera exclusion CSV
    excl_rows = []
    for v in VIEWS:
        for cat in ["pose", "left_hand", "right_hand"]:
            excl_rows.append({
                "camera": v, "category": cat,
                "n_excluded": excl_stats["per_camera_category"][v][cat],
            })
    excl_csv = OUT_DIR / "camera_exclusion_report.csv"
    with open(excl_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=excl_rows[0].keys())
        w.writeheader()
        w.writerows(excl_rows)
    print(f"  camera_exclusion_report.csv (total_excluded={excl_stats['total_excluded']})")

    # Bone length CSV (per sample)
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

    # Bone outlier CSV
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

    # Temporal anomaly CSV
    ta_csv = OUT_DIR / "temporal_anomalies.csv"
    if temporal_anomalies:
        for ta in temporal_anomalies:
            ta["cameras_prev"] = ",".join(ta["cameras_prev"])
            ta["cameras_curr"] = ",".join(ta["cameras_curr"])
        with open(ta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=temporal_anomalies[0].keys())
            w.writeheader()
            w.writerows(temporal_anomalies)
    else:
        with open(ta_csv, "w", encoding="utf-8") as f:
            f.write("(no temporal anomalies)\n")
    print(f"  temporal_anomalies.csv ({len(temporal_anomalies)} rows)")

    # Reproj distribution JSON
    rdist_path = OUT_DIR / "reproj_distributions.json"
    with open(rdist_path, "w", encoding="utf-8") as f:
        json.dump(reproj_dists, f, indent=2, ensure_ascii=False)
    print(f"  reproj_distributions.json")

    # Per-camera reproj CSV
    cam_rows = []
    for sid in PILOT_SAMPLE_IDS:
        # Per-sample per-camera stats from all valid landmarks
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

    # Overlay frame count verification
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
        "step": "6B_pilot_v1.1",
        "date": datetime.datetime.now().isoformat(),
        "v1_0_status": "EXTERNAL_REVIEW_REVISION_REQUIRED_BEFORE_FULL_45",
        "python_version": sys.version,
        "opencv_version": cv2.__version__,
        "coordinate_system": "FRONT_camera (X=right, Y=down, Z=forward)",
        "unit": "meters",
        "triangulation_method": "DLT multi-candidate (all 2/3-view combos scored)",
        "view_selection": "score = cam_bonus - reproj + angle_bonus - penalties",
        "smoothing": "DISABLED",
        "input_sha256": input_sha,
        "config": CONFIG,
        "pilot_sample_ids": PILOT_SAMPLE_IDS,
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
                                       temporal_anomalies, overlay_counts, contact_paths)
    n_pass = sum(1 for s, _ in assertion_results if s == "PASS")
    n_fail = sum(1 for s, _ in assertion_results if s == "FAIL")
    for status, msg in assertion_results:
        if status == "FAIL":
            print(f"    FAIL: {msg}")
    print(f"  Assertions: {n_pass} PASS, {n_fail} FAIL")

    # Save assertion results
    asr_path = OUT_DIR / "assertion_results.json"
    with open(asr_path, "w", encoding="utf-8") as f:
        json.dump([{"status": s, "message": m} for s, m in assertion_results], f, indent=2)

    # -- Report --
    print("\n  Generating report...")
    rpt = []
    rpt.append("# Step 6B v1.1 Pilot Report -- Multi-view 3D Reconstruction\n")
    rpt.append("## Status: EXTERNAL_REVIEW_REVISION_REQUIRED_BEFORE_FULL_45\n")
    rpt.append(f"**Date**: {datetime.datetime.now().strftime('%Y-%m-%d')}")
    rpt.append(f"**Script**: `Tools/s001_step6b_triangulate_pilot_v1_1.py`")
    rpt.append(f"**Scope**: 9 R03 samples x 3 views")
    rpt.append(f"**v1.0 status**: EXTERNAL_REVIEW_REVISION_REQUIRED_BEFORE_FULL_45\n")

    rpt.append("## Changes from v1.0\n")
    rpt.append("- Multi-candidate view selection (all 2/3-view combos scored)")
    rpt.append("- Separate `triangulation_valid` / `qa_valid` / `template_eligible`")
    rpt.append("- Post-processing bone length QA (MAD-based, abs threshold)")
    rpt.append("- Post-processing temporal QA (velocity/accel, categorized reasons)")
    rpt.append("- Regression test: GREETING_NORMAL_R03 frame 39 right hand")
    rpt.append("- Detailed THANK_YOU_FAST_R03 frames 48/49 per-landmark analysis")
    rpt.append("- Real video thumbnail contact sheets")
    rpt.append("- Color-coded overlays with decoded_frame_index matching")
    rpt.append(f"- Thresholds kept at {CONFIG['reproj_outlier_px']}/{CONFIG['reproj_exclude_px']}px\n")

    rpt.append("## Coordinate System\n")
    rpt.append("- **Reference**: FRONT camera (identity R, zero t)")
    rpt.append("- **Axes**: X=right, Y=down, Z=forward")
    rpt.append("- **Unit**: meters\n")

    rpt.append("## Triangulation Method\n")
    rpt.append("- DLT: `cv2.triangulatePoints` (2-view), SVD (3-view)")
    rpt.append("- **View selection**: all candidate subsets scored:")
    rpt.append("  - `score = n_cams*10 - used_reproj_mean + angle_bonus - penalties`")
    rpt.append("  - Feasibility: cheirality (all obs cameras), finite coords, depth 0.3-10m,")
    rpt.append(f"    angle >= {CONFIG['min_triangulation_angle_deg']}deg, used_reproj_max <= {CONFIG['reproj_exclude_px']}px")
    rpt.append("- Smoothing: **DISABLED**\n")

    # Per-sample table
    rpt.append("## Per-Sample Results\n")
    rpt.append("| Sample | Frames | Synced3V | P_tri | P_qa | P_te | LH_tri | LH_qa | RH_tri | RH_qa | 3cam% | 2cam% | BoneOut | TempAnom |")
    rpt.append("|--------|--------|----------|-------|------|------|--------|-------|--------|-------|-------|-------|---------|----------|")
    for qs in all_qa_summaries:
        short = qs["sample_id"].replace("S001_P001_", "")
        c = qs["coverage"]
        rpt.append(
            f"| {short} | {qs['n_frames']} | {qs['n_synced_3view_frames']} | "
            f"{c['pose']['tri_rate']:.0%} | {c['pose']['qa_rate']:.0%} | {c['pose']['te_rate']:.0%} | "
            f"{c['left_hand']['tri_rate']:.0%} | {c['left_hand']['qa_rate']:.0%} | "
            f"{c['right_hand']['tri_rate']:.0%} | {c['right_hand']['qa_rate']:.0%} | "
            f"{c['pose']['n_3cam_of_tri']:.0%} | {c['pose']['n_2cam_of_tri']:.0%} | "
            f"{qs['n_bone_outliers']} | {qs['n_temporal_anomalies']} |"
        )
    rpt.append("")

    # Reproj distributions
    rpt.append("## Reprojection Error Distributions\n")
    rpt.append("All values are reprojection error in pixels.\n")
    rpt.append("| Distribution | Count | Mean | Median | P95 | P99 | Max |")
    rpt.append("|-------------|-------|------|--------|-----|-----|-----|")
    for name in ["preselection_3view", "used_all", "excluded_all",
                  "used_pose", "used_left_hand", "used_right_hand",
                  "used_2view", "used_3view", "qa_valid_used", "qa_invalid_used"]:
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

    rpt.append("### Per-Camera Excluded Reproj\n")
    rpt.append("| Camera | Count | Mean | Median | P95 | Max |")
    rpt.append("|--------|-------|------|--------|-----|-----|")
    for v in VIEWS:
        st = reproj_dists.get("per_camera_excluded", {}).get(v, {})
        if st.get("count", 0) > 0:
            rpt.append(f"| {v} | {st['count']} | {st['mean']:.1f} | {st['median']:.1f} | "
                       f"{st['p95']:.1f} | {st['max']:.1f} |")
    rpt.append("")

    # Camera exclusion
    rpt.append("## Camera Exclusion Report\n")
    rpt.append(f"Total landmarks with camera exclusion: {excl_stats['total_excluded']}\n")
    rpt.append("| Camera | Pose | Left Hand | Right Hand | Total |")
    rpt.append("|--------|------|-----------|------------|-------|")
    for v in VIEWS:
        ec = excl_stats["per_camera_category"][v]
        rpt.append(f"| {v} | {ec['pose']} | {ec['left_hand']} | {ec['right_hand']} | "
                   f"{sum(ec.values())} |")
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
            rpt.append(f"- views: {item['views_available']}")
            rpt.append(f"- left_status: {item['left_status']}")
            rpt.append(f"- tri: {item['n_tri']}/21, qa: {item['n_qa']}/21, te: {item['n_te']}/21")
            rpt.append(f"- LEFT excluded: {item['n_left_excluded']}/21")
            rpt.append(f"- Assessment: {item['assessment']}")
            rpt.append("")
            rpt.append("| LM | Tri | QA | TE | Cams | Excl | UsedReproj | LeftReproj | LeftUsed |")
            rpt.append("|----|-----|----|----|------|------|------------|------------|----------|")
            for pl in item["per_landmark"]:
                rpt.append(
                    f"| {pl['i']} | {pl['tri']} | {pl['qa']} | {pl['te']} | "
                    f"{''.join(c[0] for c in pl['cams'])} | "
                    f"{''.join(c[0] for c in pl['excl'])} | "
                    f"{pl['ur']} | {pl['left_err']} | {pl['left_used']} |"
                )
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

    # Threshold analysis
    rpt.append("## Threshold Sensitivity (informational, not auto-applied)\n")
    rpt.append(f"Current: reproj_outlier={CONFIG['reproj_outlier_px']}px, reproj_exclude={CONFIG['reproj_exclude_px']}px\n")
    rpt.append("These thresholds are NOT auto-adjusted. The distributions above are provided for manual review.\n")

    # Bone/temporal summaries
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
    rpt.append("- `template_eligible=False`: excluded from downstream template generation")
    rpt.append("- No interpolation, no gap-filling, no smoothing")
    rpt.append("- Per-camera 2D preserved even for invalid 3D\n")

    rpt_path = OUT_DIR / "STEP6B_PILOT_V1_1_REPORT.md"
    with open(rpt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rpt))
    print(f"  Saved: {rpt_path.name}")

    # -- ZIP --
    print("\n  Creating delivery ZIP...")
    zip_path = PROJECT_ROOT / "WordsInSilence_step6b_v1_1_pilot_delivery.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(Path(__file__), "Tools/s001_step6b_triangulate_pilot_v1_1.py")
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(PROC_DIR)))
    zip_sz = zip_path.stat().st_size / 1024 / 1024
    print(f"  ZIP: {zip_path.name} ({zip_sz:.1f} MB)")

    # -- Final summary --
    total_frames = sum(len(all_results[sid]) for sid in PILOT_SAMPLE_IDS)
    total_lm = total_frames * 75
    total_tri = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                    for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                    for lm in f[k] if lm["triangulation_valid"])
    total_qa = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                   for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in f[k] if lm["qa_valid"])
    total_te = sum(1 for sid in PILOT_SAMPLE_IDS for f in all_results[sid]
                   for k in ["pose_landmarks_3d", "left_hand_landmarks_3d", "right_hand_landmarks_3d"]
                   for lm in f[k] if lm["template_eligible"])

    print(f"\n  {'=' * 60}")
    print(f"  STEP 6B v1.1 PILOT SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Samples: {len(PILOT_SAMPLE_IDS)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Triangulation valid: {total_tri}/{total_lm} ({total_tri / max(total_lm, 1):.1%})")
    print(f"  QA valid: {total_qa}/{total_lm} ({total_qa / max(total_lm, 1):.1%})")
    print(f"  Template eligible: {total_te}/{total_lm} ({total_te / max(total_lm, 1):.1%})")
    print(f"  Bone outliers: {len(bone_outliers)}")
    print(f"  Temporal anomalies: {len(temporal_anomalies)}")
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
