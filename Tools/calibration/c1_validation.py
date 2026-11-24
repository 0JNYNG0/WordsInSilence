"""
C1 Validation — Re-calibrate with uniform frame selection, compare with original,
compute spatial coverage, split-half stability, held-out solvePnP validation.
"""

import cv2
import numpy as np
import json
import csv
import os
import sys
from pathlib import Path

# ─── Board spec (locked measured values) ───
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
BOARD = cv2.aruco.CharucoBoard((7, 5), 0.035, 0.026, ARUCO_DICT)
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()
ARUCO_DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)

MAX_CORNERS = 24  # (7-1)*(5-1)

# ─── Paths ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = PROJECT_ROOT / "LocalMedia" / "Calibration" / "C1_Internal"
RESULTS_DIR = PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1"
VALIDATION_DIR = RESULTS_DIR / "validation"

VIDEOS = {
    "FRONT": {
        "file": MEDIA_DIR / "C1_FRONT_20260804.mp4",
        "camera_id": "CAM_FRONT_01",
    },
    "LEFT": {
        "file": MEDIA_DIR / "C1_LEFT_20260804.mp4",
        "camera_id": "CAM_LEFT_01",
    },
    "RIGHT": {
        "file": MEDIA_DIR / "C1_RIGHT_20260804.mp4",
        "camera_id": "CAM_RIGHT_01",
    },
}

SAMPLE_INTERVAL = 10
MIN_CORNERS_FOR_USE = 6
BLUR_THRESHOLD = 80.0
MIN_CORNERS_FOR_CALIB = 8
MAX_FRAMES_PER_CAM = 200


def compute_blur(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_charuco(gray):
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 2:
        return None, None, len(ids) if ids is not None else 0
    retval, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, BOARD)
    if retval < MIN_CORNERS_FOR_USE:
        return None, None, retval if retval else 0
    return cc, ci, retval


def extract_all_accepted_frames(video_path, view_name):
    """Extract ALL accepted frames (no subsampling cap)."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n  {view_name}: {w}x{h}, {fps:.2f}fps, {total_frames} frames")

    all_corners = []
    all_ids = []
    all_indices = []
    all_grays_for_coverage = []  # store lightweight metadata, not full images

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE_INTERVAL != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = compute_blur(gray)

        if blur < BLUR_THRESHOLD:
            frame_idx += 1
            continue

        cc, ci, nc = detect_charuco(gray)
        if cc is not None and nc >= MIN_CORNERS_FOR_CALIB:
            all_corners.append(cc)
            all_ids.append(ci)
            all_indices.append(frame_idx)

        frame_idx += 1

    cap.release()
    print(f"  Total accepted: {len(all_corners)}")
    return all_corners, all_ids, all_indices, (w, h), fps, total_frames


def uniform_subsample(n_total, max_count):
    """Uniform selection using np.linspace, no duplicates."""
    if n_total <= max_count:
        return list(range(n_total))
    indices = np.linspace(0, n_total - 1, max_count, dtype=int)
    return sorted(set(indices.tolist()))


def calibrate(corners_list, ids_list, image_size):
    """Calibrate camera from charuco corners/ids."""
    obj_points = []
    img_points = []
    for cc, ci in zip(corners_list, ids_list):
        obj_pts = BOARD.getChessboardCorners()[ci.flatten()]
        obj_points.append(obj_pts.astype(np.float32))
        img_points.append(cc.reshape(-1, 2).astype(np.float32))

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    per_frame_errors = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], K, dist)
        err = np.sqrt(np.mean((img_points[i] - proj.reshape(-1, 2)) ** 2))
        per_frame_errors.append(float(err))

    return {
        "rms": float(rms),
        "K": K,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "per_frame_errors": per_frame_errors,
        "obj_points": obj_points,
        "img_points": img_points,
    }


def compute_spatial_coverage(corners_list, ids_list, rvecs, tvecs, K, dist, image_size):
    """Compute spatial distribution metrics for accepted frames."""
    w, h = image_size
    centers_norm = []
    projected_areas = []
    distances = []
    tilts_lr = []  # left-right tilt (rotation around Y in board frame)
    tilts_ud = []  # up-down tilt (rotation around X in board frame)

    for i, (cc, ci, rvec, tvec) in enumerate(zip(corners_list, ids_list, rvecs, tvecs)):
        pts = cc.reshape(-1, 2)
        cx_norm = np.mean(pts[:, 0]) / w
        cy_norm = np.mean(pts[:, 1]) / h
        centers_norm.append((cx_norm, cy_norm))

        # Projected area: convex hull of detected corners
        if len(pts) >= 3:
            hull = cv2.convexHull(pts.astype(np.float32))
            area = cv2.contourArea(hull)
            projected_areas.append(area)
        else:
            projected_areas.append(0)

        # Distance from camera to board (Z component of tvec)
        dist_to_board = float(tvec[2][0])
        distances.append(dist_to_board)

        # Board tilt from rotation vector
        R, _ = cv2.Rodrigues(rvec)
        # Board normal in camera frame = R @ [0,0,1]
        normal = R @ np.array([0, 0, 1.0])
        # Tilt angles: angle between normal and camera Z axis
        tilt_lr_rad = np.arctan2(normal[0], normal[2])
        tilt_ud_rad = np.arctan2(normal[1], normal[2])
        tilts_lr.append(np.degrees(tilt_lr_rad))
        tilts_ud.append(np.degrees(tilt_ud_rad))

    centers_norm = np.array(centers_norm)

    # Grid occupancy: 4 columns x 3 rows
    grid_cols, grid_rows = 4, 3
    grid = np.zeros((grid_rows, grid_cols), dtype=int)
    for cx, cy in centers_norm:
        gc = min(int(cx * grid_cols), grid_cols - 1)
        gr = min(int(cy * grid_rows), grid_rows - 1)
        grid[gr, gc] += 1
    occupied = int(np.sum(grid > 0))

    areas = np.array(projected_areas)
    dists = np.array(distances)

    return {
        "center_x_range": [float(centers_norm[:, 0].min()), float(centers_norm[:, 0].max())],
        "center_y_range": [float(centers_norm[:, 1].min()), float(centers_norm[:, 1].max())],
        "grid_4x3_occupied": occupied,
        "grid_4x3_total": grid_rows * grid_cols,
        "grid_4x3_counts": grid.tolist(),
        "projected_area_p10": float(np.percentile(areas, 10)) if len(areas) else 0,
        "projected_area_median": float(np.median(areas)) if len(areas) else 0,
        "projected_area_p90": float(np.percentile(areas, 90)) if len(areas) else 0,
        "distance_min": float(dists.min()),
        "distance_max": float(dists.max()),
        "distance_median": float(np.median(dists)),
        "tilt_lr_range": [float(np.min(tilts_lr)), float(np.max(tilts_lr))],
        "tilt_ud_range": [float(np.min(tilts_ud)), float(np.max(tilts_ud))],
    }


def compute_undistortion_diff(K1, dist1, K2, dist2, image_size):
    """Compute pixel mapping difference between two distortion models
    over the central 90% of the image."""
    w, h = image_size
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.05)

    # Create grid of points in central 90%
    xs = np.linspace(margin_x, w - margin_x, 50)
    ys = np.linspace(margin_y, h - margin_y, 30)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)

    # Undistort with model 1
    pts1 = cv2.undistortPoints(pts.reshape(-1, 1, 2), K1, dist1, P=K1)
    # Undistort with model 2
    pts2 = cv2.undistortPoints(pts.reshape(-1, 1, 2), K2, dist2, P=K2)

    diff = np.sqrt(np.sum((pts1.reshape(-1, 2) - pts2.reshape(-1, 2)) ** 2, axis=1))
    return {
        "mean_px": float(np.mean(diff)),
        "max_px": float(np.max(diff)),
        "p95_px": float(np.percentile(diff, 95)),
        "p50_px": float(np.median(diff)),
    }


def split_half_stability(all_corners, all_ids, all_indices, image_size):
    """Split accepted frames into two temporal halves, calibrate each, compare."""
    n = len(all_corners)
    mid = n // 2

    half_a_corners = all_corners[:mid]
    half_a_ids = all_ids[:mid]
    half_b_corners = all_corners[mid:]
    half_b_ids = all_ids[mid:]

    # Subsample each half to MAX_FRAMES_PER_CAM if needed
    for half_name, hc, hi in [("A", half_a_corners, half_a_ids), ("B", half_b_corners, half_b_ids)]:
        if len(hc) < 10:
            print(f"  WARNING: Half {half_name} has only {len(hc)} frames")

    sel_a = uniform_subsample(len(half_a_corners), MAX_FRAMES_PER_CAM)
    sel_b = uniform_subsample(len(half_b_corners), MAX_FRAMES_PER_CAM)

    ca = calibrate([half_a_corners[i] for i in sel_a],
                   [half_a_ids[i] for i in sel_a], image_size)
    cb = calibrate([half_b_corners[i] for i in sel_b],
                   [half_b_ids[i] for i in sel_b], image_size)

    return ca, cb


def solvepnp_validation(held_out_corners, held_out_ids, K, dist):
    """Validate calibration on held-out frames using solvePnP."""
    errors = []
    for cc, ci in zip(held_out_corners, held_out_ids):
        obj_pts = BOARD.getChessboardCorners()[ci.flatten()].astype(np.float64)
        img_pts = cc.reshape(-1, 2).astype(np.float64)
        if len(obj_pts) < 4:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
        err = np.sqrt(np.mean((img_pts - proj.reshape(-1, 2)) ** 2))
        errors.append(float(err))

    if not errors:
        return None
    ea = np.array(errors)
    return {
        "n_frames": len(errors),
        "p50": float(np.percentile(ea, 50)),
        "p95": float(np.percentile(ea, 95)),
        "max": float(np.max(ea)),
        "mean": float(np.mean(ea)),
    }


def load_original_intrinsics(camera_id):
    """Load original C1 intrinsics for comparison."""
    path = RESULTS_DIR / f"intrinsics_{camera_id}.json"
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])
    return K, dist, data["rms_reprojection_error"]


def main():
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for view_name, info in VIDEOS.items():
        video_path = info["file"]
        camera_id = info["camera_id"]
        print(f"\n{'='*60}")
        print(f"  VALIDATING: {view_name} ({camera_id})")
        print(f"{'='*60}")

        # 1. Extract ALL accepted frames
        all_corners, all_ids, all_indices, image_size, fps, total_frames = \
            extract_all_accepted_frames(video_path, view_name)

        n_accepted = len(all_corners)
        if n_accepted < 20:
            print(f"  ERROR: Only {n_accepted} accepted frames!")
            all_results[view_name] = {"status": "FAIL", "reason": "insufficient_frames"}
            continue

        # 2. Uniform subsample for recalibration
        sel_indices = uniform_subsample(n_accepted, MAX_FRAMES_PER_CAM)
        print(f"  Uniform subsample: {len(sel_indices)} frames from {n_accepted} accepted")

        sel_corners = [all_corners[i] for i in sel_indices]
        sel_ids = [all_ids[i] for i in sel_indices]
        sel_frame_idx = [all_indices[i] for i in sel_indices]

        # Show temporal coverage
        print(f"  Frame range: {sel_frame_idx[0]} .. {sel_frame_idx[-1]} "
              f"(total video: {total_frames})")

        # 3. Recalibrate
        print(f"  Recalibrating with {len(sel_corners)} uniformly selected frames...")
        recalib = calibrate(sel_corners, sel_ids, image_size)
        print(f"  New RMS: {recalib['rms']:.4f}")

        # 4. Load original and compare
        orig_K, orig_dist, orig_rms = load_original_intrinsics(camera_id)
        new_K = recalib["K"]
        new_dist = recalib["dist"]

        fx_old, fy_old = orig_K[0, 0], orig_K[1, 1]
        cx_old, cy_old = orig_K[0, 2], orig_K[1, 2]
        fx_new, fy_new = new_K[0, 0], new_K[1, 1]
        cx_new, cy_new = new_K[0, 2], new_K[1, 2]

        comparison = {
            "fx_old": fx_old, "fx_new": fx_new,
            "fx_change_pct": (fx_new - fx_old) / fx_old * 100,
            "fy_old": fy_old, "fy_new": fy_new,
            "fy_change_pct": (fy_new - fy_old) / fy_old * 100,
            "cx_old": cx_old, "cx_new": cx_new,
            "cx_change_px": cx_new - cx_old,
            "cy_old": cy_old, "cy_new": cy_new,
            "cy_change_px": cy_new - cy_old,
            "rms_old": orig_rms, "rms_new": recalib["rms"],
            "dist_old": orig_dist.flatten().tolist(),
            "dist_new": new_dist.flatten().tolist(),
        }

        print(f"  fx: {fx_old:.2f} -> {fx_new:.2f} ({comparison['fx_change_pct']:+.3f}%)")
        print(f"  fy: {fy_old:.2f} -> {fy_new:.2f} ({comparison['fy_change_pct']:+.3f}%)")
        print(f"  cx: {cx_old:.2f} -> {cx_new:.2f} ({comparison['cx_change_px']:+.2f}px)")
        print(f"  cy: {cy_old:.2f} -> {cy_new:.2f} ({comparison['cy_change_px']:+.2f}px)")
        print(f"  RMS: {orig_rms:.4f} -> {recalib['rms']:.4f}")

        # 5. Undistortion mapping difference (central 90%)
        undist_diff = compute_undistortion_diff(orig_K, orig_dist, new_K, new_dist, image_size)
        print(f"  Undistort diff (central 90%): mean={undist_diff['mean_px']:.4f}px, "
              f"max={undist_diff['max_px']:.4f}px, p95={undist_diff['p95_px']:.4f}px")

        # 6. Spatial coverage (all accepted frames, not just subsampled)
        # Need rvecs/tvecs for all accepted — do solvePnP on all
        print(f"  Computing spatial coverage on all {n_accepted} accepted frames...")
        all_rvecs = []
        all_tvecs = []
        for cc, ci in zip(all_corners, all_ids):
            obj_pts = BOARD.getChessboardCorners()[ci.flatten()].astype(np.float64)
            img_pts = cc.reshape(-1, 2).astype(np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, new_K, new_dist)
            if ok:
                all_rvecs.append(rvec)
                all_tvecs.append(tvec)
            else:
                all_rvecs.append(np.zeros((3, 1)))
                all_tvecs.append(np.zeros((3, 1)))

        coverage = compute_spatial_coverage(
            all_corners, all_ids, all_rvecs, all_tvecs, new_K, new_dist, image_size
        )
        print(f"  Center X range: [{coverage['center_x_range'][0]:.3f}, {coverage['center_x_range'][1]:.3f}]")
        print(f"  Center Y range: [{coverage['center_y_range'][0]:.3f}, {coverage['center_y_range'][1]:.3f}]")
        print(f"  Grid 4x3 occupied: {coverage['grid_4x3_occupied']}/{coverage['grid_4x3_total']}")
        print(f"  Projected area p10/median/p90: {coverage['projected_area_p10']:.0f} / "
              f"{coverage['projected_area_median']:.0f} / {coverage['projected_area_p90']:.0f}")
        print(f"  Distance range: {coverage['distance_min']:.3f} ~ {coverage['distance_max']:.3f}m "
              f"(median {coverage['distance_median']:.3f}m)")
        print(f"  Tilt LR: [{coverage['tilt_lr_range'][0]:.1f}, {coverage['tilt_lr_range'][1]:.1f}] deg")
        print(f"  Tilt UD: [{coverage['tilt_ud_range'][0]:.1f}, {coverage['tilt_ud_range'][1]:.1f}] deg")

        # 7. Split-half stability
        print(f"  Split-half stability test...")
        half_a, half_b = split_half_stability(all_corners, all_ids, all_indices, image_size)

        stability = {
            "half_a_rms": half_a["rms"],
            "half_b_rms": half_b["rms"],
            "half_a_fx": float(half_a["K"][0, 0]),
            "half_b_fx": float(half_b["K"][0, 0]),
            "fx_diff_pct": abs(half_a["K"][0, 0] - half_b["K"][0, 0]) / float(new_K[0, 0]) * 100,
            "half_a_fy": float(half_a["K"][1, 1]),
            "half_b_fy": float(half_b["K"][1, 1]),
            "fy_diff_pct": abs(half_a["K"][1, 1] - half_b["K"][1, 1]) / float(new_K[1, 1]) * 100,
            "half_a_cx": float(half_a["K"][0, 2]),
            "half_b_cx": float(half_b["K"][0, 2]),
            "cx_diff_px": abs(float(half_a["K"][0, 2]) - float(half_b["K"][0, 2])),
            "half_a_cy": float(half_a["K"][1, 2]),
            "half_b_cy": float(half_b["K"][1, 2]),
            "cy_diff_px": abs(float(half_a["K"][1, 2]) - float(half_b["K"][1, 2])),
            "half_a_dist": half_a["dist"].flatten().tolist(),
            "half_b_dist": half_b["dist"].flatten().tolist(),
        }

        # Undistort diff between halves
        half_undist_diff = compute_undistortion_diff(
            half_a["K"], half_a["dist"], half_b["K"], half_b["dist"], image_size
        )
        stability["undistort_diff_halves"] = half_undist_diff

        print(f"  Half A RMS: {half_a['rms']:.4f}, Half B RMS: {half_b['rms']:.4f}")
        print(f"  fx diff: {stability['fx_diff_pct']:.3f}%")
        print(f"  fy diff: {stability['fy_diff_pct']:.3f}%")
        print(f"  cx diff: {stability['cx_diff_px']:.2f}px, cy diff: {stability['cy_diff_px']:.2f}px")
        print(f"  Half undistort diff: mean={half_undist_diff['mean_px']:.4f}px, "
              f"max={half_undist_diff['max_px']:.4f}px")

        # 8. Held-out solvePnP validation
        # Use frames NOT in the calibration subsample
        sel_set = set(sel_indices)
        held_out_indices = [i for i in range(n_accepted) if i not in sel_set]
        print(f"  Held-out validation: {len(held_out_indices)} frames...")

        if held_out_indices:
            ho_corners = [all_corners[i] for i in held_out_indices]
            ho_ids = [all_ids[i] for i in held_out_indices]
            pnp_val = solvepnp_validation(ho_corners, ho_ids, new_K, new_dist)
            if pnp_val:
                print(f"  Held-out PnP: p50={pnp_val['p50']:.4f}, p95={pnp_val['p95']:.4f}, "
                      f"max={pnp_val['max']:.4f}")
        else:
            pnp_val = None
            print(f"  No held-out frames (all used for calibration)")

        # Save recalibrated intrinsics
        recalib_out = {
            "camera_id": camera_id,
            "view": view_name,
            "image_size": list(image_size),
            "camera_matrix": new_K.tolist(),
            "dist_coeffs": new_dist.flatten().tolist(),
            "rms_reprojection_error": round(recalib["rms"], 4),
            "n_frames_used": len(sel_corners),
            "n_accepted_total": n_accepted,
            "selection_method": "np.linspace uniform",
            "per_frame_reprojection_errors": [round(e, 4) for e in recalib["per_frame_errors"]],
            "board_spec": {
                "dictionary": "DICT_4X4_50",
                "squares_x": 7, "squares_y": 5,
                "square_length_m": 0.035, "marker_length_m": 0.026,
            },
        }
        with open(VALIDATION_DIR / f"intrinsics_{camera_id}_v2.json", "w") as f:
            json.dump(recalib_out, f, indent=2)

        all_results[view_name] = {
            "camera_id": camera_id,
            "comparison": comparison,
            "undistort_diff_central90": undist_diff,
            "coverage": coverage,
            "stability": stability,
            "held_out_pnp": pnp_val,
            "n_accepted": n_accepted,
            "n_calib_frames": len(sel_corners),
        }

    # Save full validation results JSON
    # Convert numpy types for JSON serialization
    def np_convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            return np_convert(obj)

    with open(VALIDATION_DIR / "c1_validation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, cls=NpEncoder)

    print(f"\n{'='*60}")
    print("  VALIDATION COMPLETE")
    print(f"{'='*60}")
    for vn, r in all_results.items():
        if "comparison" not in r:
            print(f"  {vn}: {r.get('status','?')} — {r.get('reason','')}")
            continue
        c = r["comparison"]
        print(f"\n  {vn}:")
        print(f"    RMS: {c['rms_old']:.4f} -> {c['rms_new']:.4f}")
        print(f"    fx change: {c['fx_change_pct']:+.3f}%")
        print(f"    fy change: {c['fy_change_pct']:+.3f}%")
        print(f"    cx change: {c['cx_change_px']:+.2f}px")
        print(f"    cy change: {c['cy_change_px']:+.2f}px")
        ud = r["undistort_diff_central90"]
        print(f"    Undistort diff (90%): mean={ud['mean_px']:.4f}, max={ud['max_px']:.4f}px")
        cov = r["coverage"]
        print(f"    Grid: {cov['grid_4x3_occupied']}/{cov['grid_4x3_total']}")
        st = r["stability"]
        print(f"    Split-half fx diff: {st['fx_diff_pct']:.3f}%")
        ho = r.get("held_out_pnp")
        if ho:
            print(f"    Held-out PnP: p50={ho['p50']:.4f}, p95={ho['p95']:.4f}, max={ho['max']:.4f}")

    print(f"\nSaved to: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
