"""
C1 — Camera Intrinsic Calibration using ChArUco board.
Processes three independent calibration videos (FRONT, LEFT, RIGHT).

Board spec (measured):
  Dictionary: DICT_4X4_50, 7x5 squares
  squareLength = 0.035m, markerLength = 0.026m

Usage:
  python Tools/calibration/c1_intrinsic_calibration.py
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

# Expected number of charuco corners = (7-1)*(5-1) = 24
MAX_CORNERS = 24

# ─── Paths ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MEDIA_DIR = PROJECT_ROOT / "LocalMedia" / "Calibration" / "C1_Internal"
WORK_DIR = PROJECT_ROOT / "Tools" / "calibration" / "work" / "C1"
RESULTS_DIR = PROJECT_ROOT / "Tools" / "calibration" / "results" / "C1"

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

# ─── Frame selection parameters ───
SAMPLE_INTERVAL = 10        # sample every N frames
MIN_CORNERS_FOR_USE = 6     # minimum charuco corners to consider a frame
BLUR_THRESHOLD = 80.0       # Laplacian variance below this = blurry
MIN_CORNERS_FOR_CALIB = 8   # minimum corners for calibration frame
MAX_FRAMES_PER_CAM = 200    # cap frames for calibration to avoid overfitting
REPROJ_OUTLIER_THRESHOLD = 1.5  # pixels — reject frames with RMS above this


def compute_blur(gray):
    """Laplacian variance as sharpness metric."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def detect_charuco(gray):
    """Detect ArUco markers then interpolate ChArUco corners."""
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) < 2:
        return None, None, len(ids) if ids is not None else 0

    charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners, ids, gray, BOARD
    )
    if charuco_retval < MIN_CORNERS_FOR_USE:
        return None, None, charuco_retval if charuco_retval else 0

    return charuco_corners, charuco_ids, charuco_retval


def extract_frames(video_path, view_name):
    """Extract and evaluate frames from a calibration video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\n{'='*60}")
    print(f"  {view_name}: {w}x{h}, {fps:.2f} fps, {total_frames} frames")
    print(f"  Sampling every {SAMPLE_INTERVAL} frames...")
    print(f"{'='*60}")

    all_charuco_corners = []
    all_charuco_ids = []
    frame_quality = []
    accepted_frame_indices = []

    frame_idx = 0
    sampled = 0
    detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_INTERVAL != 0:
            frame_idx += 1
            continue

        sampled += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = compute_blur(gray)
        charuco_corners, charuco_ids, n_corners = detect_charuco(gray)

        status = "rejected"
        reject_reason = ""

        if blur_score < BLUR_THRESHOLD:
            reject_reason = "blurry"
        elif charuco_corners is None:
            reject_reason = f"too_few_corners({n_corners})"
        elif n_corners < MIN_CORNERS_FOR_CALIB:
            reject_reason = f"corners_below_calib_min({n_corners})"
        else:
            status = "accepted"
            detected += 1
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            accepted_frame_indices.append(frame_idx)

        frame_quality.append({
            "frame_idx": frame_idx,
            "blur_score": round(blur_score, 2),
            "n_charuco_corners": n_corners,
            "status": status,
            "reject_reason": reject_reason,
        })

        frame_idx += 1

    cap.release()

    print(f"  Sampled: {sampled}, Detected (accepted): {detected}")

    # If too many accepted frames, subsample for diversity
    if detected > MAX_FRAMES_PER_CAM:
        step = detected // MAX_FRAMES_PER_CAM
        indices = list(range(0, detected, step))[:MAX_FRAMES_PER_CAM]
        all_charuco_corners = [all_charuco_corners[i] for i in indices]
        all_charuco_ids = [all_charuco_ids[i] for i in indices]
        accepted_frame_indices = [accepted_frame_indices[i] for i in indices]
        print(f"  Subsampled to {len(indices)} frames for calibration")

    return {
        "corners": all_charuco_corners,
        "ids": all_charuco_ids,
        "accepted_indices": accepted_frame_indices,
        "frame_quality": frame_quality,
        "image_size": (w, h),
        "fps": fps,
        "total_frames": total_frames,
    }


def calibrate_camera(data, view_name):
    """Run OpenCV camera calibration using ChArUco corners."""
    corners = data["corners"]
    ids = data["ids"]
    image_size = data["image_size"]

    if len(corners) < 5:
        print(f"  WARNING: Only {len(corners)} frames — insufficient for calibration!")
        return None

    print(f"\n  Calibrating {view_name} with {len(corners)} frames...")

    # Prepare object points and image points for calibrateCamera
    obj_points = []
    img_points = []
    for cc, ci in zip(corners, ids):
        obj_pts = BOARD.getChessboardCorners()[ci.flatten()]
        obj_points.append(obj_pts.astype(np.float32))
        img_points.append(cc.reshape(-1, 2).astype(np.float32))

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )

    print(f"  RMS reprojection error: {rms:.4f} px")

    # Per-frame reprojection errors
    per_frame_errors = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                         camera_matrix, dist_coeffs)
        err = np.sqrt(np.mean((img_points[i] - projected.reshape(-1, 2)) ** 2))
        per_frame_errors.append(round(float(err), 4))

    return {
        "rms": round(float(rms), 4),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "per_frame_errors": per_frame_errors,
        "accepted_frame_indices": data["accepted_indices"],
        "n_frames_used": len(corners),
        "image_size": list(image_size),
    }


def filter_outlier_frames(data, calib_result, view_name):
    """Remove frames with high reprojection error and recalibrate."""
    errors = calib_result["per_frame_errors"]
    good_mask = [e < REPROJ_OUTLIER_THRESHOLD for e in errors]
    n_outliers = sum(1 for g in good_mask if not g)

    if n_outliers == 0:
        print(f"  {view_name}: No outlier frames to remove.")
        return calib_result

    print(f"  {view_name}: Removing {n_outliers} outlier frames (>{REPROJ_OUTLIER_THRESHOLD}px)...")

    corners_filtered = [c for c, g in zip(data["corners"], good_mask) if g]
    ids_filtered = [i for i, g in zip(data["ids"], good_mask) if g]
    indices_filtered = [idx for idx, g in zip(data["accepted_indices"], good_mask) if g]

    data_filtered = {
        "corners": corners_filtered,
        "ids": ids_filtered,
        "accepted_indices": indices_filtered,
        "image_size": data["image_size"],
    }

    if len(corners_filtered) < 5:
        print(f"  WARNING: Only {len(corners_filtered)} frames remain after filtering!")
        return calib_result

    result = calibrate_camera(data_filtered, f"{view_name} (refined)")
    if result is None:
        return calib_result

    result["n_outliers_removed"] = n_outliers
    return result


def save_frame_quality_csv(frame_quality, view_name):
    """Save per-frame quality data to CSV."""
    csv_path = RESULTS_DIR / f"frame_quality_{view_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "blur_score", "n_charuco_corners", "status", "reject_reason"])
        writer.writeheader()
        writer.writerows(frame_quality)
    print(f"  Saved: {csv_path.name}")


def save_intrinsics_json(calib_result, camera_id, view_name):
    """Save camera intrinsics to JSON."""
    if calib_result is None:
        return

    output = {
        "camera_id": camera_id,
        "view": view_name,
        "image_size": calib_result["image_size"],
        "camera_matrix": calib_result["camera_matrix"],
        "dist_coeffs": calib_result["dist_coeffs"],
        "rms_reprojection_error": calib_result["rms"],
        "n_frames_used": calib_result["n_frames_used"],
        "n_outliers_removed": calib_result.get("n_outliers_removed", 0),
        "per_frame_reprojection_errors": calib_result["per_frame_errors"],
        "accepted_frame_indices": calib_result["accepted_frame_indices"],
        "board_spec": {
            "dictionary": "DICT_4X4_50",
            "squares_x": 7,
            "squares_y": 5,
            "square_length_m": 0.035,
            "marker_length_m": 0.026,
        },
    }

    json_path = RESULTS_DIR / f"intrinsics_{camera_id}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {json_path.name}")


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_summary = {}

    for view_name, info in VIDEOS.items():
        video_path = info["file"]
        camera_id = info["camera_id"]

        if not video_path.exists():
            print(f"BLOCKED_SOURCE_MEDIA: {video_path} not found!")
            sys.exit(1)

        # Extract and evaluate frames
        data = extract_frames(video_path, view_name)

        # Save frame quality CSV
        save_frame_quality_csv(data["frame_quality"], view_name)

        # Initial calibration
        calib = calibrate_camera(data, view_name)
        if calib is None:
            results_summary[view_name] = {"status": "FAIL", "reason": "insufficient_frames"}
            continue

        # Refine by removing outlier frames
        calib = filter_outlier_frames(data, calib, view_name)

        # Save intrinsics
        save_intrinsics_json(calib, camera_id, view_name)

        # Corner distribution analysis
        accepted_q = [fq for fq in data["frame_quality"] if fq["status"] == "accepted"]
        rejected_q = [fq for fq in data["frame_quality"] if fq["status"] == "rejected"]
        corner_counts = [fq["n_charuco_corners"] for fq in accepted_q]

        results_summary[view_name] = {
            "camera_id": camera_id,
            "status": "PASS" if calib["rms"] < 1.0 else "WARN",
            "total_sampled": len(data["frame_quality"]),
            "accepted_frames": len(accepted_q),
            "rejected_frames": len(rejected_q),
            "frames_used_for_calib": calib["n_frames_used"],
            "outliers_removed": calib.get("n_outliers_removed", 0),
            "rms": calib["rms"],
            "per_frame_error_min": min(calib["per_frame_errors"]),
            "per_frame_error_max": max(calib["per_frame_errors"]),
            "per_frame_error_mean": round(float(np.mean(calib["per_frame_errors"])), 4),
            "corner_count_min": min(corner_counts) if corner_counts else 0,
            "corner_count_max": max(corner_counts) if corner_counts else 0,
            "corner_count_mean": round(float(np.mean(corner_counts)), 1) if corner_counts else 0,
        }

        print(f"\n  {view_name} Summary:")
        print(f"    Accepted: {len(accepted_q)}, Rejected: {len(rejected_q)}")
        print(f"    Calib frames: {calib['n_frames_used']}, RMS: {calib['rms']:.4f}")
        print(f"    Per-frame error: min={min(calib['per_frame_errors']):.4f}, "
              f"max={max(calib['per_frame_errors']):.4f}, "
              f"mean={np.mean(calib['per_frame_errors']):.4f}")

    # Save source media report
    source_report = {
        "stage": "C1",
        "board_spec_used": "DICT_4X4_50, 7x5, 0.035m, 0.026m",
        "original_videos_modified": False,
    }
    for view_name, info in VIDEOS.items():
        vp = info["file"]
        cap = cv2.VideoCapture(str(vp))
        source_report[view_name] = {
            "path": str(vp.relative_to(PROJECT_ROOT)),
            "file_size_bytes": vp.stat().st_size,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_s": round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS), 2),
        }
        cap.release()

    with open(RESULTS_DIR / "source_media_report.json", "w") as f:
        json.dump(source_report, f, indent=2)

    # Print final summary
    print(f"\n{'='*60}")
    print("  C1 CALIBRATION SUMMARY")
    print(f"{'='*60}")
    for view_name, s in results_summary.items():
        print(f"\n  {view_name} ({s.get('camera_id','?')}):")
        print(f"    Status: {s['status']}")
        print(f"    Accepted/Rejected: {s.get('accepted_frames',0)}/{s.get('rejected_frames',0)}")
        print(f"    Calibration frames: {s.get('frames_used_for_calib',0)}")
        print(f"    RMS: {s.get('rms','N/A')}")
        print(f"    Per-frame error: {s.get('per_frame_error_min','?')} ~ {s.get('per_frame_error_max','?')} (mean {s.get('per_frame_error_mean','?')})")
        print(f"    Corners: {s.get('corner_count_min',0)} ~ {s.get('corner_count_max',0)} (mean {s.get('corner_count_mean',0)})")

    # Save summary JSON
    with open(RESULTS_DIR / "c1_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {RESULTS_DIR}")
    return results_summary


if __name__ == "__main__":
    main()
