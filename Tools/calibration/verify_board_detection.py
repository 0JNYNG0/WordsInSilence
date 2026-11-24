"""
C0 -- ChArUco board detection verifier.

Reads board_spec.json, constructs the same CharucoBoard, and verifies detection
on an image file, video file, or live camera feed.

Usage:
    python Tools/calibration/verify_board_detection.py --image charuco_board_7x5_A4.png
    python Tools/calibration/verify_board_detection.py --video calibration_clip.mp4
    python Tools/calibration/verify_board_detection.py --camera 0

Output:
    - Detection statistics (marker count, corner count, IDs)
    - Overlay image saved to Tools/calibration/detection_overlay_*.png
    - PASS or FAIL with reason
"""

import argparse, json, os, sys
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(SCRIPT_DIR, "board_spec.json")


def load_board():
    """Load board spec and create CharucoBoard."""
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        spec = json.load(f)

    sq_x = spec["squares_x"]
    sq_y = spec["squares_y"]
    sq_len = spec["square_length_mm"] / 1000.0
    mk_len = spec["marker_length_mm"] / 1000.0
    dict_name = spec["dictionary"]

    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((sq_x, sq_y), sq_len, mk_len, aruco_dict)
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    return spec, board, aruco_dict, aruco_detector


def detect_frame(frame, board, aruco_detector, spec):
    """Detect ArUco markers and interpolate ChArUco corners in a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    # Detect ArUco markers
    corners, ids, rejected = aruco_detector.detectMarkers(gray)

    result = {
        "aruco_detected": 0,
        "aruco_ids": [],
        "charuco_corners": 0,
        "charuco_ids": [],
    }

    if ids is not None and len(ids) > 0:
        result["aruco_detected"] = len(ids)
        result["aruco_ids"] = sorted(ids.flatten().tolist())

        # Interpolate ChArUco corners
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if retval > 0:
            result["charuco_corners"] = retval
            result["charuco_ids"] = sorted(charuco_ids.flatten().tolist())

    return result, corners, ids


def draw_overlay(frame, corners, ids, board, spec):
    """Draw detection overlay on frame."""
    overlay = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if retval > 0:
            cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)

    return overlay


def evaluate(result, spec):
    """Evaluate detection result against spec."""
    expected_markers = spec["aruco_marker_count"]   # 17
    expected_corners = spec["charuco_corner_count"]  # 24

    passed = True
    reasons = []

    if result["aruco_detected"] == 0:
        passed = False
        reasons.append("No ArUco markers detected")
    elif result["aruco_detected"] < expected_markers:
        passed = False
        reasons.append(f"ArUco markers: {result['aruco_detected']}/{expected_markers}")

    if result["charuco_corners"] == 0:
        passed = False
        reasons.append("No ChArUco corners interpolated")
    elif result["charuco_corners"] < expected_corners:
        passed = False
        reasons.append(f"ChArUco corners: {result['charuco_corners']}/{expected_corners}")

    # Check for unexpected marker IDs (should be 0-16 for 17 markers)
    max_valid_id = expected_markers - 1
    invalid_ids = [i for i in result["aruco_ids"] if i > max_valid_id]
    if invalid_ids:
        reasons.append(f"Unexpected marker IDs: {invalid_ids} (expected 0-{max_valid_id})")

    return passed, reasons


def process_image(path, board, aruco_detector, spec):
    """Process a single image file."""
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return False

    frame = cv2.imread(path)
    if frame is None:
        # Try grayscale
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"ERROR: cannot read image: {path}")
            return False

    print(f"Image: {os.path.basename(path)} ({frame.shape[1]}x{frame.shape[0]})")

    result, corners, ids = detect_frame(frame, board, aruco_detector, spec)
    passed, reasons = evaluate(result, spec)

    print(f"  ArUco markers detected: {result['aruco_detected']}/{spec['aruco_marker_count']}")
    print(f"  ArUco marker IDs: {result['aruco_ids']}")
    print(f"  ChArUco corners: {result['charuco_corners']}/{spec['charuco_corner_count']}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    if reasons:
        for r in reasons:
            print(f"    Reason: {r}")

    # Save overlay
    overlay = draw_overlay(frame, corners, ids, board, spec)
    base = os.path.splitext(os.path.basename(path))[0]
    overlay_path = os.path.join(SCRIPT_DIR, f"detection_overlay_{base}.png")
    cv2.imwrite(overlay_path, overlay)
    print(f"  Overlay saved: {overlay_path}")

    return passed


def process_video(path, board, aruco_detector, spec, sample_count=10):
    """Process a video file, sampling frames."""
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return False

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {os.path.basename(path)} ({total_frames} frames, {fps:.1f} fps)")

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames), dtype=int)

    best_result = None
    best_frame = None
    best_corners = None
    best_ids = None
    all_pass = True

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        result, corners, ids = detect_frame(frame, board, aruco_detector, spec)
        passed, reasons = evaluate(result, spec)

        status = "PASS" if passed else "FAIL"
        print(f"  Frame {idx}: markers={result['aruco_detected']}, "
              f"corners={result['charuco_corners']}  [{status}]")

        if not passed:
            all_pass = False

        if best_result is None or result["charuco_corners"] > best_result["charuco_corners"]:
            best_result = result
            best_frame = frame.copy()
            best_corners = corners
            best_ids = ids

    cap.release()

    # Save best overlay
    if best_frame is not None:
        overlay = draw_overlay(best_frame, best_corners, best_ids, board, spec)
        base = os.path.splitext(os.path.basename(path))[0]
        overlay_path = os.path.join(SCRIPT_DIR, f"detection_overlay_{base}.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"  Best overlay saved: {overlay_path}")

    print(f"  Overall: {'ALL PASS' if all_pass else 'SOME FRAMES FAILED'}")
    return all_pass


def process_camera(device_id, board, aruco_detector, spec, duration_s=5):
    """Process live camera for a few seconds."""
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera device {device_id}")
        return False

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Camera {device_id}: {w}x{h} @ {fps:.0f} fps")
    print(f"  Capturing for {duration_s}s...")

    max_frames = int(fps * duration_s)
    best_result = None
    best_frame = None
    best_corners = None
    best_ids = None
    frame_count = 0
    pass_count = 0

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        result, corners, ids = detect_frame(frame, board, aruco_detector, spec)
        passed, _ = evaluate(result, spec)
        if passed:
            pass_count += 1

        if best_result is None or result["charuco_corners"] > best_result["charuco_corners"]:
            best_result = result
            best_frame = frame.copy()
            best_corners = corners
            best_ids = ids

    cap.release()

    if best_result:
        print(f"  Frames captured: {frame_count}")
        print(f"  Frames passed: {pass_count}/{frame_count}")
        print(f"  Best: markers={best_result['aruco_detected']}, "
              f"corners={best_result['charuco_corners']}")
        print(f"  Best marker IDs: {best_result['aruco_ids']}")

        overlay = draw_overlay(best_frame, best_corners, best_ids, board, spec)
        overlay_path = os.path.join(SCRIPT_DIR, f"detection_overlay_camera{device_id}.png")
        cv2.imwrite(overlay_path, overlay)
        print(f"  Overlay saved: {overlay_path}")

        overall = pass_count > 0
        print(f"  Result: {'PASS' if overall else 'FAIL'}")
        return overall
    else:
        print(f"  No frames captured")
        return False


def main():
    parser = argparse.ArgumentParser(description="C0: ChArUco board detection verifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to image file")
    group.add_argument("--video", help="Path to video file")
    group.add_argument("--camera", type=int, help="Camera device index")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of frames to sample from video (default: 10)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Camera capture duration in seconds (default: 5)")
    args = parser.parse_args()

    spec, board, aruco_dict, aruco_detector = load_board()
    print(f"Board spec: {spec['squares_x']}x{spec['squares_y']}, "
          f"sq={spec['square_length_mm']}mm, mk={spec['marker_length_mm']}mm, "
          f"{spec['dictionary']}")
    print(f"Expected: {spec['aruco_marker_count']} ArUco markers, "
          f"{spec['charuco_corner_count']} ChArUco corners\n")

    if args.image:
        passed = process_image(args.image, board, aruco_detector, spec)
    elif args.video:
        passed = process_video(args.video, board, aruco_detector, spec, args.samples)
    elif args.camera is not None:
        passed = process_camera(args.camera, board, aruco_detector, spec, args.duration)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
