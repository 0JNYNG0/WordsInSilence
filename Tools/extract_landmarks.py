#!/usr/bin/env python3
"""
Tools/extract_landmarks.py
--------------------------
전문가 수어 MP4 영상에서 MediaPipe Hand + Pose 랜드마크를 추출하여
HandPoseRecording JSON (schemaVersion=2) 으로 저장한다.

Unity의 HandPoseRecordingRepository.Load() 가 그대로 읽을 수 있는 포맷.

사용법:
    python extract_landmarks.py --video clip.mp4 --motion-id hello_01 --display-name 안녕
    python extract_landmarks.py --video clip.mp4 --motion-id hello_01 --display-name 안녕 --output-dir ./output

필요 패키지 (Python 3.10~3.12 권장, 3.13은 mediapipe 미지원 가능):
    pip install mediapipe opencv-python

필요 모델 파일 → Tools/models/ 폴더에 배치:
    hand_landmarker.task
    pose_landmarker_lite.task
    (다운로드 URL 은 Tools/README.md 참조)
"""

import argparse
import json
import math
import os
import sys

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─── 경로 ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")
HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

# ─── MediaPipe Pose 랜드마크 인덱스 (BlazePose 33점) ─────────────────────────

LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16

# ─── FrameDataQuality enum 값 (HandPoseFrame.cs 와 일치) ─────────────────────
# Good=0, HandOnly=1, PoseOnly=2, InsufficientPose=3,
# AssociationUncertain=4, OutOfFrame=5, Invalid=6

QUALITY_GOOD              = 0
QUALITY_HAND_ONLY         = 1
QUALITY_POSE_ONLY         = 2
QUALITY_INSUFFICIENT_POSE = 3
QUALITY_INVALID           = 6

# PoseBodyMath.MinShoulderWidth 와 동일
MIN_SHOULDER_WIDTH = 0.05

# ─── 수학 유틸 (PoseBodyMath.cs 동일 로직) ───────────────────────────────────

def shoulder_center(lx, ly, rx, ry):
    return {"x": (lx + rx) * 0.5, "y": (ly + ry) * 0.5}

def shoulder_width_2d(lx, ly, rx, ry):
    return math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)

def body_relative(px, py, center, width):
    if width <= MIN_SHOULDER_WIDTH or math.isnan(width):
        return {"x": float("nan"), "y": float("nan")}
    return {"x": (px - center["x"]) / width, "y": (py - center["y"]) / width}

def lm_to_vec3(lm):
    """MediaPipe NormalizedLandmark → {"x", "y", "z"}"""
    return {"x": lm.x, "y": lm.y, "z": lm.z}

# ─── 프레임 변환 ─────────────────────────────────────────────────────────────

def make_frame(timestamp_ms, hand_result, pose_result):
    # ── Hand ──────────────────────────────────────────────────────────────────
    has_left  = False
    has_right = False
    left_pts  = None
    right_pts = None

    if hand_result and hand_result.hand_landmarks:
        for i, landmarks in enumerate(hand_result.hand_landmarks):
            # MediaPipe 반환 handedness: 이미지 좌우 기준 ("Left" / "Right")
            category = hand_result.handedness[i][0].category_name
            pts = [lm_to_vec3(lm) for lm in landmarks]
            if category == "Left":
                has_left = True
                left_pts = pts
            else:
                has_right = True
                right_pts = pts

    # ── Pose ──────────────────────────────────────────────────────────────────
    has_pose            = False
    has_valid_shoulders = False
    pose_pts            = None
    shoulder_ctr        = {"x": 0.0, "y": 0.0}
    s_width             = 0.0
    left_wrist_rel      = {"x": float("nan"), "y": float("nan")}
    right_wrist_rel     = {"x": float("nan"), "y": float("nan")}

    if pose_result and pose_result.pose_landmarks:
        lms      = pose_result.pose_landmarks[0]
        has_pose = True
        pose_pts = [lm_to_vec3(lm) for lm in lms]

        l_sh = lms[LEFT_SHOULDER]
        r_sh = lms[RIGHT_SHOULDER]
        shoulder_ctr        = shoulder_center(l_sh.x, l_sh.y, r_sh.x, r_sh.y)
        s_width             = shoulder_width_2d(l_sh.x, l_sh.y, r_sh.x, r_sh.y)
        has_valid_shoulders = s_width > MIN_SHOULDER_WIDTH

        l_wr = lms[LEFT_WRIST]
        r_wr = lms[RIGHT_WRIST]
        left_wrist_rel  = body_relative(l_wr.x, l_wr.y, shoulder_ctr, s_width)
        right_wrist_rel = body_relative(r_wr.x, r_wr.y, shoulder_ctr, s_width)

    # ── Quality ───────────────────────────────────────────────────────────────
    has_hand = has_left or has_right
    if has_hand and has_pose and has_valid_shoulders:
        quality = QUALITY_GOOD
    elif has_hand and has_pose:
        quality = QUALITY_INSUFFICIENT_POSE
    elif has_hand:
        quality = QUALITY_HAND_ONLY
    elif has_pose:
        quality = QUALITY_POSE_ONLY
    else:
        quality = QUALITY_INVALID

    return {
        "TimestampMs":            timestamp_ms,
        "HasLeftHand":            has_left,
        "HasRightHand":           has_right,
        "HasPose":                has_pose,
        "HasValidShoulders":      has_valid_shoulders,
        "LeftHandPoints":         left_pts,
        "RightHandPoints":        right_pts,
        "PosePoints":             pose_pts,
        "ShoulderCenter":         shoulder_ctr,
        "ShoulderWidth":          s_width,
        "LeftWristBodyRelative":  left_wrist_rel,
        "RightWristBodyRelative": right_wrist_rel,
        "Quality":                quality,
    }

# ─── 추출 메인 ───────────────────────────────────────────────────────────────

def extract(video_path, motion_id, display_name):
    for p in (HAND_MODEL, POSE_MODEL):
        if not os.path.exists(p):
            print(f"[오류] 모델 파일 없음: {p}")
            print("  Tools/models/ 폴더에 모델 파일을 배치하세요. (Tools/README.md 참조)")
            sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[오류] 영상 파일 열기 실패: {video_path}")
        sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[정보] {os.path.basename(video_path)}  |  {total}프레임  |  {fps:.1f}fps")

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
    )

    frames    = []
    frame_idx = 0

    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            timestamp_ms = int(frame_idx * 1000 / fps)
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_lm.detect_for_video(mp_img, timestamp_ms)
            pose_result = pose_lm.detect_for_video(mp_img, timestamp_ms)

            frames.append(make_frame(timestamp_ms, hand_result, pose_result))
            frame_idx += 1

            if frame_idx % 30 == 0:
                print(f"  {frame_idx}/{total} 프레임 처리 중...", end="\r")

    cap.release()
    print(f"\n[처리 완료] {frame_idx}프레임")

    # 통계
    valid_hand     = sum(1 for f in frames if f["HasLeftHand"] or f["HasRightHand"])
    valid_pose     = sum(1 for f in frames if f["HasPose"])
    valid_combined = sum(1 for f in frames if (f["HasLeftHand"] or f["HasRightHand"]) and f["HasPose"])
    duration_ms    = frame_idx * 1000.0 / fps
    source_type    = "hand_pose" if valid_pose > 0 else "hand_only"

    print(f"  Total:{frame_idx}  Hand:{valid_hand}  Pose:{valid_pose}  Combined:{valid_combined}")

    return {
        "SchemaVersion":       2,
        "MotionId":            motion_id,
        "DisplayName":         display_name if display_name else motion_id,
        "DurationMs":          duration_ms,
        "SourceType":          source_type,
        "Frames":              frames,
        "TotalFrames":         frame_idx,
        "ValidHandFrames":     valid_hand,
        "ValidPoseFrames":     valid_pose,
        "ValidCombinedFrames": valid_combined,
    }

# ─── 속도 변형 증강 ──────────────────────────────────────────────────────────

def speed_augment(recording, factor, suffix):
    """
    같은 프레임 데이터를 다른 속도로 리샘플링해서 변형 레퍼런스를 만든다.
    factor < 1 → 빠른 동작 시뮬레이션 (프레임 수 감소)
    factor > 1 → 느린 동작 시뮬레이션 (프레임 수 증가)
    DisplayName은 원본과 동일하게 유지 → Unity에서 같은 단어로 그룹화됨.
    """
    original = recording["Frames"]
    n        = len(original)
    new_n    = max(2, round(n * factor))

    resampled = []
    for i in range(new_n):
        t   = i / (new_n - 1)
        idx = round(t * (n - 1))
        resampled.append(dict(original[idx]))

    duration_ms = recording["DurationMs"]
    for i, frame in enumerate(resampled):
        frame["TimestampMs"] = round(i / (new_n - 1) * duration_ms)

    result = dict(recording)
    result["MotionId"]            = recording["MotionId"] + suffix
    result["Frames"]              = resampled
    result["TotalFrames"]         = new_n
    result["ValidHandFrames"]     = sum(1 for f in resampled if f["HasLeftHand"] or f["HasRightHand"])
    result["ValidPoseFrames"]     = sum(1 for f in resampled if f["HasPose"])
    result["ValidCombinedFrames"] = sum(1 for f in resampled if (f["HasLeftHand"] or f["HasRightHand"]) and f["HasPose"])
    return result

# ─── 진입점 ──────────────────────────────────────────────────────────────────

def save_recording(recording, out_dir, motion_id):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{motion_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recording, f, ensure_ascii=False, indent=2, allow_nan=True)
    print(f"[저장] {out_path}")


def save_with_augments(recording, out_dir):
    """원본 + 1.2x(slow) 2개 저장. _fast는 DTW 길이 불일치로 노이즈 발생 → 제외."""
    mid = recording["MotionId"]
    save_recording(recording, out_dir, mid)
    save_recording(speed_augment(recording, 1.2, "_slow"), out_dir, mid + "_slow")


def main():
    parser = argparse.ArgumentParser(description="수어 MP4 → HandPoseRecording JSON (schemaVersion=2)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",     help="입력 MP4 파일 경로 (단일 처리)")
    group.add_argument("--video-dir", help="MP4 파일이 들어있는 폴더 (일괄 처리)")
    parser.add_argument("--motion-id",    default="", help="MotionId (--video 단일 처리 시 사용)")
    parser.add_argument("--display-name", default="", help="표시 이름 (--video 단일 처리 시 사용)")
    parser.add_argument("--output-dir",   default="", help="출력 폴더 (기본: 입력 영상과 같은 폴더)")
    parser.add_argument("--no-augment",   action="store_true", help="속도 변형 증강 비활성화 (원본 1개만 저장)")
    args = parser.parse_args()

    if args.video_dir:
        # ── 일괄 처리: 폴더 내 모든 .mp4 파일 ────────────────────────────
        video_dir = args.video_dir
        mp4_files = sorted(f for f in os.listdir(video_dir) if f.lower().endswith(".mp4"))
        if not mp4_files:
            print(f"[오류] .mp4 파일이 없습니다: {video_dir}")
            sys.exit(1)

        print(f"[일괄 처리] {len(mp4_files)}개 파일 발견")
        out_dir = args.output_dir if args.output_dir else video_dir

        for filename in mp4_files:
            motion_id    = os.path.splitext(filename)[0]
            display_name = motion_id
            video_path   = os.path.join(video_dir, filename)
            print(f"\n{'='*50}")
            print(f"[{motion_id}] 처리 시작")
            recording = extract(video_path, motion_id, display_name)
            if args.no_augment:
                save_recording(recording, out_dir, motion_id)
            else:
                save_with_augments(recording, out_dir)

        print(f"\n[완료] {len(mp4_files)}개 파일 처리 완료")

    else:
        # ── 단일 처리 ─────────────────────────────────────────────────────
        motion_id = args.motion_id if args.motion_id else os.path.splitext(os.path.basename(args.video))[0]
        out_dir   = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.video))
        recording = extract(args.video, motion_id, args.display_name)
        if args.no_augment:
            save_recording(recording, out_dir, motion_id)
        else:
            save_with_augments(recording, out_dir)


if __name__ == "__main__":
    main()
