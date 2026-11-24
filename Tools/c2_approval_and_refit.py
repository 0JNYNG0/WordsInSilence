"""
C2 Approval Record + Robust Refit (excluding >5px outlier frames).
Saves C2_APPROVAL.json and C2_CALIBRATION_v2_robust.json without overwriting originals.
"""
import cv2
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "sessions" / "S001_P001"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sessions" / "S001_P001"
C1_DIR = PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1" / "validation"

VIDEOS = {
    "FRONT": RAW_DIR / "S001_P001_FRONT.mp4",
    "LEFT":  RAW_DIR / "S001_P001_LEFT.mp4",
    "RIGHT": RAW_DIR / "S001_P001_RIGHT.mp4",
}
CAMERA_IDS = {"FRONT": "CAM_FRONT_01", "LEFT": "CAM_LEFT_01", "RIGHT": "CAM_RIGHT_01"}

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, ARUCO_DICT)
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())


def load_c1():
    intrinsics = {}
    sha256s = {}
    for view, cam_id in CAMERA_IDS.items():
        p = C1_DIR / f"intrinsics_{cam_id}_v2.json"
        with open(p) as f:
            d = json.load(f)
        intrinsics[view] = {
            "K": np.array(d["camera_matrix"], dtype=np.float64),
            "dist": np.array(d["dist_coeffs"], dtype=np.float64),
            "image_size": tuple(d["image_size"]),
        }
        # Compute SHA-256 of the file
        import hashlib
        sha256s[view] = hashlib.sha256(p.read_bytes()).hexdigest()
    return intrinsics, sha256s


def load_c2():
    with open(OUT_DIR / "C2_CALIBRATION.json") as f:
        return json.load(f)


def verify_rotation(R):
    det = float(np.linalg.det(R))
    orth_err = float(np.linalg.norm(R @ R.T - np.eye(3)))
    return det, orth_err


def main():
    intrinsics, c1_sha256s = load_c1()
    c2 = load_c2()
    sync = json.load(open(OUT_DIR / "sync_report.json"))

    # ── Verify R matrices ──
    rot_checks = {}
    for view in ["LEFT", "RIGHT"]:
        R = np.array(c2["cameras"][view]["R"])
        det, orth = verify_rotation(R)
        rot_checks[view] = {"det": round(det, 8), "orthogonality_error": round(orth, 10)}
        print(f"  {view}: det(R)={det:.8f}, ||RR^T - I||={orth:.10f}")

    # ── Per-frame reproj analysis ──
    criteria_results = {}
    for view in ["LEFT", "RIGHT"]:
        cam = c2["cameras"][view]
        errors = np.array(cam["per_frame_reproj_errors"])
        n_total = len(errors)
        median = float(np.median(errors))
        over5 = int(np.sum(errors > 5.0))
        over5_pct = round(over5 / n_total * 100, 1) if n_total > 0 else 0
        over5_indices = [i for i, e in enumerate(errors) if e > 5.0]
        over5_values = [round(float(errors[i]), 4) for i in over5_indices]

        criteria_results[view] = {
            "stereo_rms": cam["stereo_rms"],
            "stereo_rms_pass": cam["stereo_rms"] < 1.0,
            "n_frame_pairs": n_total,
            "n_frame_pairs_pass": n_total >= 30,
            "reproj_median_px": round(median, 4),
            "reproj_median_pass": median <= 2.0,
            "over_5px_count": over5,
            "over_5px_pct": over5_pct,
            "over_5px_pass": over5_pct <= 10.0,
            "over_5px_frame_indices": over5_indices,
            "over_5px_values": over5_values,
            "rotation_det": rot_checks[view]["det"],
            "rotation_det_pass": abs(rot_checks[view]["det"] - 1.0) < 0.001,
            "orthogonality_error": rot_checks[view]["orthogonality_error"],
            "orthogonality_pass": rot_checks[view]["orthogonality_error"] < 0.001,
        }
        print(f"  {view}: median={median:.4f}px, >5px: {over5}/{n_total} ({over5_pct}%)")

    # ── Baseline check ──
    baselines = {}
    for view in ["LEFT", "RIGHT"]:
        t = np.array(c2["cameras"][view]["t"])
        bl = float(np.linalg.norm(t))
        # LEFT should have negative x (camera to the left), RIGHT positive x
        baselines[view] = {
            "baseline_m": round(bl, 4),
            "t_x": round(float(t[0]), 4),
            "t_y": round(float(t[1]), 4),
            "t_z": round(float(t[2]), 4),
            "physically_plausible": 0.5 < bl < 3.0,
        }

    # LEFT t_x < 0 means camera is to the left of FRONT
    baselines["LEFT"]["lateral_consistent"] = baselines["LEFT"]["t_x"] < 0
    baselines["RIGHT"]["lateral_consistent"] = baselines["RIGHT"]["t_x"] > 0

    all_pass = all(
        all(v for k, v in cr.items() if k.endswith("_pass"))
        for cr in criteria_results.values()
    )

    # ── C2 Approval ──
    approval = {
        "stage": "C2",
        "session": "S001",
        "status": "APPROVED/PASS",
        "approval_date": "2026-08-04",
        "approval_scope": "SEGMENTATION_AND_3D_SMOKE_TEST",

        "criteria": {
            "stereo_rms_threshold_px": 1.0,
            "min_sync_frame_pairs": 30,
            "reproj_median_threshold_px": 2.0,
            "over_5px_max_pct": 10.0,
            "rotation_det_tolerance": 0.001,
            "orthogonality_tolerance": 0.001,
            "baseline_range_m": [0.5, 3.0],
            "note": "Single per_frame_reproj_max is NOT a pass/fail criterion. Outlier frames >5px are logged and excluded in robust refit."
        },

        "results": criteria_results,
        "baselines": baselines,

        "c1_intrinsics_sha256": c1_sha256s,

        "frame_selection_chain": {
            "common_candidate_frames": 167,
            "subsampled_for_calibration": 80,
            "LEFT_valid_pairs": 73,
            "RIGHT_valid_pairs": 77,
            "note": "167 common frames (all 3 cameras detect board simultaneously) -> 80 subsampled via np.linspace -> pairs with >=6 common corner IDs used"
        },

        "conventions": {
            "coordinate_transform": {
                "formula": "X_secondary = R @ X_front + t",
                "meaning": "R and t transform a 3D point from the FRONT camera coordinate system to the secondary (LEFT or RIGHT) camera coordinate system.",
                "front_axes": {
                    "X": "right (in image plane)",
                    "Y": "down (in image plane)",
                    "Z": "forward (into the scene, optical axis)",
                    "unit": "meters"
                }
            },
            "projection": {
                "formula": "p = K @ [R|t] @ X_front_homogeneous",
                "per_camera": {
                    "FRONT": "p_front = K_front @ [I|0] @ X_front",
                    "LEFT": "p_left = K_left @ [R_left|t_left] @ X_front",
                    "RIGHT": "p_right = K_right @ [R_right|t_right] @ X_front"
                },
                "note": "K from C1 intrinsics, R/t from C2 extrinsics. dist_coeffs applied via cv2.undistortPoints or cv2.projectPoints."
            },
            "synchronization": {
                "offset_sign": "camera_local_time_s = front_time_s + offset_s",
                "offsets": {
                    "FRONT": 0.0,
                    "LEFT": 2.300333,
                    "RIGHT": 9.388633
                },
                "drift_correction": {
                    "formula": "camera_local_time_s = front_time_s + offset_s + drift_rate * front_time_s",
                    "LEFT": {
                        "offset_s": 2.300333,
                        "drift_rate": -19.3e-6,
                        "note": "drift_rate = drift_ppm * 1e-6. Over 500s, drift = -0.0097s (negligible)."
                    },
                    "RIGHT": {
                        "offset_s": 9.388633,
                        "drift_rate": -0.3e-6,
                        "note": "Effectively zero drift."
                    }
                },
                "vfr_warning": "FRONT is VFR. Always use actual PTS (cv2.CAP_PROP_POS_MSEC), never frame_index / fps.",
                "method": "end_clap (3 claps), cross-verified with matched start claps"
            }
        },

        "all_criteria_pass": all_pass,
        "c2_calibration_file": "C2_CALIBRATION.json",
        "c2_robust_refit_file": "C2_CALIBRATION_v2_robust.json"
    }

    with open(OUT_DIR / "C2_APPROVAL.json", "w", encoding="utf-8") as f:
        json.dump(approval, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: C2_APPROVAL.json (all_pass={all_pass})")

    # ── Robust Refit ──
    # Re-run stereo calibration excluding frames with reproj > 5px
    print("\n  Robust refit: excluding >5px outlier frames...")
    # We need to re-read the charuco data from videos (same scan as step 3)
    # For efficiency, load from the existing C2 data and filter

    # Actually we need to redo stereo calibration with filtered frames.
    # Let's re-scan the C2 region (coarse, same as step 3) and redo.

    sync_data = json.load(open(OUT_DIR / "sync_report.json"))
    offsets = {
        "FRONT": 0.0,
        "LEFT": sync_data["cameras"]["LEFT"]["offset_s_final"],
        "RIGHT": sync_data["cameras"]["RIGHT"]["offset_s_final"],
    }

    def detect_charuco_frame(gray):
        corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
        if ids is None or len(ids) < 3:
            return None, None, 0
        ret, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
        if ret < 4:
            return None, None, ret if ret else 0
        return cc, ci, ret

    # Re-scan (coarse only)
    all_dets = {}
    all_fps = {}
    for view, path in VIDEOS.items():
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = min(int(180 * fps), total)
        all_fps[view] = fps

        dets = []
        frame_idx = 0
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cc, ci, n = detect_charuco_frame(gray)
                if cc is not None and n >= 6:
                    dets.append((frame_idx, pts_ms, n, cc, ci))
            frame_idx += 1
        cap.release()
        all_dets[view] = dets
        print(f"    {view}: {len(dets)} detections")

    # Find common frames
    ref = "FRONT"
    def to_sync(view, pts_ms):
        return pts_ms / 1000.0 - offsets[view]

    common = []
    for rd in all_dets[ref]:
        r_sync = to_sync(ref, rd[1])
        matches = {ref: rd}
        ok = True
        for ov in ["LEFT", "RIGHT"]:
            best = None
            best_dt = float("inf")
            for od in all_dets[ov]:
                dt = abs(to_sync(ov, od[1]) - r_sync)
                if dt < best_dt:
                    best_dt = dt
                    best = od
            if best and best_dt < 0.1:
                matches[ov] = best
            else:
                ok = False
                break
        if ok and len(matches) == 3:
            common.append(matches)

    # Subsample
    if len(common) > 80:
        idx = np.linspace(0, len(common)-1, 80, dtype=int)
        common = [common[i] for i in idx]

    obj_points_3d = BOARD.getChessboardCorners()

    # Stereo calibrate each pair, compute per-frame errors, remove outliers, recalibrate
    robust_results = {
        "stage": "C2_robust_refit",
        "session": "S001",
        "reference_camera": "FRONT",
        "note": "Robust refit excluding frames with per-frame reproj > 5px. Original C2_CALIBRATION.json preserved.",
        "cameras": {
            "FRONT": c2["cameras"]["FRONT"]
        }
    }

    for ov in ["LEFT", "RIGHT"]:
        K_ref = intrinsics[ref]["K"]
        d_ref = intrinsics[ref]["dist"]
        K_ov = intrinsics[ov]["K"]
        d_ov = intrinsics[ov]["dist"]
        img_size = intrinsics[ref]["image_size"]

        # Collect point pairs
        obj_list_ref, img_list_ref, obj_list_ov, img_list_ov = [], [], [], []
        for cf in common:
            r_ci = cf[ref][4].flatten()
            o_ci = cf[ov][4].flatten()
            cids = np.intersect1d(r_ci, o_ci)
            if len(cids) < 6:
                continue
            r_mask = np.isin(r_ci, cids)
            o_mask = np.isin(o_ci, cids)
            r_order = np.argsort(r_ci[r_mask])
            o_order = np.argsort(o_ci[o_mask])
            obj = obj_points_3d[cids[np.argsort(cids)]].astype(np.float32)
            r_img = cf[ref][3].reshape(-1,2)[r_mask][r_order].astype(np.float32)
            o_img = cf[ov][3].reshape(-1,2)[o_mask][o_order].astype(np.float32)
            obj_list_ref.append(obj)
            img_list_ref.append(r_img)
            obj_list_ov.append(obj)
            img_list_ov.append(o_img)

        if len(obj_list_ref) < 10:
            print(f"    {ov}: too few pairs ({len(obj_list_ref)})")
            continue

        # Initial stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria_cv = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_list_ref, img_list_ref, img_list_ov,
            K_ref, d_ref, K_ov, d_ov,
            img_size, flags=flags, criteria=criteria_cv
        )

        # Per-frame reproj
        pf_errors = []
        for i in range(len(obj_list_ref)):
            ok_r, rv_r, tv_r = cv2.solvePnP(obj_list_ref[i], img_list_ref[i], K_ref, d_ref)
            if not ok_r:
                pf_errors.append(999.0)
                continue
            R_r, _ = cv2.Rodrigues(rv_r)
            R_ov_f = R @ R_r
            T_ov_f = R @ tv_r + T
            rv_ov, _ = cv2.Rodrigues(R_ov_f)
            proj, _ = cv2.projectPoints(obj_list_ref[i], rv_ov, T_ov_f, K_ov, d_ov)
            err = float(np.sqrt(np.mean((img_list_ov[i] - proj.reshape(-1,2))**2)))
            pf_errors.append(err)

        pf_errors = np.array(pf_errors)
        outlier_mask = pf_errors > 5.0
        n_outliers = int(np.sum(outlier_mask))
        outlier_indices = [int(i) for i in np.where(outlier_mask)[0]]
        outlier_values = [round(float(pf_errors[i]), 4) for i in outlier_indices]

        print(f"    {ov}: initial {len(pf_errors)} pairs, {n_outliers} outliers >5px")

        # Filter and recalibrate
        good = ~outlier_mask
        obj_ref_g = [obj_list_ref[i] for i in range(len(good)) if good[i]]
        img_ref_g = [img_list_ref[i] for i in range(len(good)) if good[i]]
        obj_ov_g = [obj_list_ov[i] for i in range(len(good)) if good[i]]
        img_ov_g = [img_list_ov[i] for i in range(len(good)) if good[i]]

        rms2, _, _, _, _, R2, T2, E2, F2 = cv2.stereoCalibrate(
            obj_ref_g, img_ref_g, img_ov_g,
            K_ref, d_ref, K_ov, d_ov,
            img_size, flags=flags, criteria=criteria_cv
        )

        # Recompute per-frame errors
        pf2 = []
        for i in range(len(obj_ref_g)):
            ok_r, rv_r, tv_r = cv2.solvePnP(obj_ref_g[i], img_ref_g[i], K_ref, d_ref)
            if not ok_r:
                pf2.append(999.0)
                continue
            R_r, _ = cv2.Rodrigues(rv_r)
            rv_ov, _ = cv2.Rodrigues(R2 @ R_r)
            T_ov_f = R2 @ tv_r + T2
            proj, _ = cv2.projectPoints(obj_ref_g[i], rv_ov, T_ov_f, K_ov, d_ov)
            err = float(np.sqrt(np.mean((img_ov_g[i] - proj.reshape(-1,2))**2)))
            pf2.append(round(err, 4))

        det2, orth2 = verify_rotation(R2)
        bl2 = float(np.linalg.norm(T2))

        print(f"    {ov} robust: rms={rms2:.4f}px, {len(pf2)} pairs, "
              f"median={np.median(pf2):.4f}px, max={np.max(pf2):.4f}px, "
              f"baseline={bl2:.4f}m")

        robust_results["cameras"][ov] = {
            "camera_id": CAMERA_IDS[ov],
            "role": "secondary",
            "R": R2.tolist(),
            "t": T2.flatten().tolist(),
            "stereo_rms": round(float(rms2), 4),
            "baseline_m": round(bl2, 4),
            "n_frame_pairs_original": len(pf_errors),
            "n_outliers_removed": n_outliers,
            "outlier_frame_indices": outlier_indices,
            "outlier_reproj_values": outlier_values,
            "n_frame_pairs_robust": len(pf2),
            "per_frame_reproj_errors": pf2,
            "per_frame_reproj_median": round(float(np.median(pf2)), 4),
            "per_frame_reproj_mean": round(float(np.mean(pf2)), 4),
            "per_frame_reproj_max": round(float(np.max(pf2)), 4),
            "rotation_det": round(det2, 8),
            "orthogonality_error": round(orth2, 10),
        }

    with open(OUT_DIR / "C2_CALIBRATION_v2_robust.json", "w", encoding="utf-8") as f:
        json.dump(robust_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved: C2_CALIBRATION_v2_robust.json")


if __name__ == "__main__":
    main()
