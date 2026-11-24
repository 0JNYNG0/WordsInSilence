"""
C0 — 소스 영상 메타데이터 점검 도구.

ffprobe 없이 OpenCV VideoCapture로 영상 메타데이터를 추출하고
촬영 일관성을 검증한다.

Usage:
    python Tools/calibration/check_source_media.py [--video-dir Tools/dataset]

Output:
    Tools/calibration/source_media_report.json
    콘솔에 요약 및 경고 출력
"""

import argparse, json, os, sys, glob
import cv2


def probe_video(path: str) -> dict:
    """OpenCV로 영상 메타데이터 추출."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"path": path, "error": "cannot_open"}

    info = {
        "path": os.path.basename(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 3),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fourcc": decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC))),
        "duration_s": None,
        "orientation": None,
    }

    fc = info["frame_count"]
    fps = info["fps"]
    if fc > 0 and fps > 0:
        info["duration_s"] = round(fc / fps, 2)

    info["orientation"] = "portrait" if info["height"] > info["width"] else "landscape"

    cap.release()
    return info


def decode_fourcc(v: int) -> str:
    """4CC 정수를 문자열로 변환."""
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def classify_view(filename: str) -> str:
    """파일명에서 뷰 식별."""
    fn = filename.lower()
    if "front" in fn:
        return "front"
    elif "left" in fn:
        return "left_oblique"
    elif "right" in fn:
        return "right_oblique"
    return "unknown"


def check_consistency(videos: list) -> list:
    """촬영 일관성 경고 생성."""
    warnings = []

    # Group by sign_id
    signs = {}
    for v in videos:
        if "error" in v:
            warnings.append(f"OPEN_FAILED: {v['path']}")
            continue
        parts = v["path"].split("_")
        sign_id = "_".join(parts[:2]) if len(parts) >= 2 else v["path"]
        signs.setdefault(sign_id, []).append(v)

    for sign_id, vids in signs.items():
        fps_set = set(v["fps"] for v in vids)
        if len(fps_set) > 1:
            warnings.append(f"FPS_MISMATCH ({sign_id}): {fps_set}")

        res_set = set(f"{v['width']}x{v['height']}" for v in vids)
        if len(res_set) > 1:
            # 다른 카메라이므로 해상도 차이는 경고 수준
            warnings.append(f"RESOLUTION_VARIES ({sign_id}): {res_set} -- expected for multi-camera setup")

        for v in vids:
            view = classify_view(v["path"])
            if view == "front" and v["orientation"] != "landscape":
                warnings.append(f"FRONT_NOT_LANDSCAPE: {v['path']} ({v['width']}x{v['height']})")
            if v["fps"] < 24:
                warnings.append(f"LOW_FPS: {v['path']} fps={v['fps']}")
            if v["frame_count"] < 30:
                warnings.append(f"VERY_SHORT: {v['path']} frames={v['frame_count']}")

    return warnings


def main():
    parser = argparse.ArgumentParser(description="C0: Source media metadata check")
    parser.add_argument("--video-dir", default="Tools/dataset",
                        help="Directory containing source videos")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: Tools/calibration/source_media_report.json)")
    args = parser.parse_args()

    video_dir = args.video_dir
    if not os.path.isdir(video_dir):
        print(f"ERROR: directory not found: {video_dir}")
        sys.exit(1)

    patterns = ["*.mp4", "*.mov", "*.avi"]
    files = []
    seen = set()
    for pat in patterns:
        for f in glob.glob(os.path.join(video_dir, pat)):
            norm = os.path.normcase(os.path.abspath(f))
            if norm not in seen:
                seen.add(norm)
                files.append(f)
    files.sort()

    if not files:
        print(f"WARNING: no video files found in {video_dir}")
        sys.exit(0)

    print(f"Scanning {len(files)} video(s) in {video_dir}...\n")

    results = []
    for f in files:
        info = probe_video(f)
        info["view"] = classify_view(info["path"])
        results.append(info)
        status = "OK" if "error" not in info else info["error"]
        print(f"  {info['path']:60s}  {info.get('width','?')}x{info.get('height','?')}  "
              f"fps={info.get('fps','?')}  frames={info.get('frame_count','?')}  "
              f"{info.get('orientation','?')}  [{status}]")

    warnings = check_consistency(results)

    print(f"\n=== Warnings ({len(warnings)}) ===")
    for w in warnings:
        print(f"  WARNING: {w}")
    if not warnings:
        print("  None")

    # Equipment summary
    front_vids = [v for v in results if v["view"] == "front" and "error" not in v]
    side_vids = [v for v in results if v["view"] != "front" and v["view"] != "unknown" and "error" not in v]

    print(f"\n=== Equipment Summary ===")
    if front_vids:
        v = front_vids[0]
        print(f"  Front: {v['width']}x{v['height']}, {v['fourcc']}, {v['fps']} fps")
    if side_vids:
        for v in side_vids[:2]:
            print(f"  {v['view']}: {v['width']}x{v['height']}, {v['fourcc']}, {v['fps']} fps")

    # Mirror check
    print(f"\n=== Mirror Status ===")
    print(f"  NOT FOUND in video metadata (OpenCV does not expose mirror flags).")
    print(f"  Manual verification required: compare left/right shoulder positions")
    print(f"  in front video frame with known anatomical left/right.")

    # Save report
    out_path = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "source_media_report.json")
    report = {
        "stage": "C0",
        "video_dir": os.path.abspath(video_dir),
        "file_count": len(results),
        "videos": results,
        "warnings": warnings,
        "mirror_status": "NOT_FOUND",
        "rotation_metadata": "NONE_DETECTED",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
