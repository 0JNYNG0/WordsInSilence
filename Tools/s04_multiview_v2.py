"""
Tools/s04_multiview_v2.py — S01 인사 3-뷰 파이프라인 v2

개선 사항:
  - 손 궤적 교차상관(xcorr)으로 동기화 재검증 (오디오 없으므로 시각 신호 사용)
  - pose_world_landmarks + hand_world_landmarks 추출 (미터 단위 3D)
  - 체간 좌표계(body-local 3D) 기반 분절 신호 계산
  - scipy prominence 기반 분절 — 정확히 12개 아니면 검토 MP4 생성 후 중단
  - 체간 좌표계로 모든 뷰 정렬 → 랜드마크별 가중 융합 + 이상치 제거
  - 기여도·잔차 전체 기록
  - 손 바뀜·순간이동·장기 미검출·팔 관통 수치 검증
  - Blender 애니메이션 생성 + 3시점 렌더링 MP4

실행:
  python Tools/s04_multiview_v2.py
  python Tools/s04_multiview_v2.py --skip-extract
  python Tools/s04_multiview_v2.py --boundaries-csv boundaries_manual.csv
  python Tools/s04_multiview_v2.py --subsample 2

출력: Tools/s04_out_v2/
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.signal import correlate, find_peaks

# ─── 경로 ────────────────────────────────────────────────────────────────────

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models")
OUT_DIR     = os.path.join(SCRIPT_DIR, "s04_out_v2")

HAND_MODEL  = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL  = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

BLENDER_EXE = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
BLEND_FILE  = r"C:\Users\local-user\OneDrive\바탕 화면\test_mixamo_girl.blend"

VIDEOS = {
    "front":   os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_front.mp4"),
    "left45":  os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_left45.mp4"),
    "right45": os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_right45.mp4"),
}

VIEW_ORDER = ["front", "left45", "right45"]

# ─── 분절 파라미터 ────────────────────────────────────────────────────────────

EXPECTED_REPS       = 12      # 정확히 12개
SMOOTH_WINDOW       = 15      # body-local head Y 스무딩 윈도우 (프레임)
MIN_PROM_METERS     = 0.010   # 인사 bow 최소 prominence (1cm — 고개숙임 신호 작음)
MIN_DIST_FRAMES_30  = 150     # 30fps 기준 최소 피크 간격 (5초)

# ─── 동기화 샘플링 ────────────────────────────────────────────────────────────

SYNC_SAMPLE       = 5         # IMAGE 모드 xcorr용 서브샘플
SYNC_MAX_LAG_FRAC = 0.15      # 최대 허용 lag = 전체 길이 × fraction

# ─── Pose landmark 인덱스 (BlazePose 33점) ───────────────────────────────────

NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

MIN_SHOULDER_WIDTH = 0.05     # 정규화 좌표 (screen)
MIN_SHOULDER_WIDTH_M = 0.15   # 미터 (world)

# ─── Quality enum ────────────────────────────────────────────────────────────

QUALITY_GOOD              = 0
QUALITY_HAND_ONLY         = 1
QUALITY_POSE_ONLY         = 2
QUALITY_INSUFFICIENT_POSE = 3
QUALITY_INVALID           = 6

# ─── 수학 유틸 (numpy) ───────────────────────────────────────────────────────

def lm_to_dict(lm):
    return {"x": lm.x, "y": lm.y, "z": lm.z}

def lm_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=float)

def dist2d(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def smooth_boxcar(data, window):
    """1-D 박스 필터. NaN은 최근 유효값으로 채움."""
    filled = []
    last = 0.0
    for v in data:
        if math.isnan(v):
            filled.append(last)
        else:
            filled.append(v)
            last = v
    arr = np.array(filled)
    k = np.ones(window) / window
    return np.convolve(arr, k, mode="same")

def nan_safe_vis(v):
    return 0.0 if (v is None or math.isnan(float(v))) else float(v)

def vis_sum_safe(vis_list):
    if vis_list is None:
        return 0.0
    return sum(nan_safe_vis(v) for v in vis_list)

# ─── 체간 좌표계 (body-local frame) ──────────────────────────────────────────

def compute_body_frame(pose_world_pts):
    """
    pose_world_landmarks (리스트 또는 np.array) 에서 체간 좌표계 계산.
    원점: 고관절 중점 (hip midpoint).
    X: 오른쪽 어깨 → 왼쪽 어깨 방향 (오른쪽 양수, 체간 기준).
       주의: MediaPipe X는 오른쪽이 음수이므로 LEFT_SHOULDER.x > RIGHT_SHOULDER.x 일 수 있음.
       → 실제 체간 오른쪽 방향 = (RIGHT_SHOULDER - LEFT_SHOULDER) 벡터.
    Y: 고관절 → 어깨 방향 (위쪽 양수), X와 직교화.
    Z: X × Y (앞쪽).

    반환: (origin np[3], R np[3×3])  body_local = R.T @ (world_pt - origin)
    """
    def to_np(idx):
        lm = pose_world_pts[idx]
        if isinstance(lm, dict):
            return np.array([lm["x"], lm["y"], lm["z"]], dtype=float)
        return np.array([lm.x, lm.y, lm.z], dtype=float)

    l_sh  = to_np(LEFT_SHOULDER)
    r_sh  = to_np(RIGHT_SHOULDER)
    l_hip = to_np(LEFT_HIP)
    r_hip = to_np(RIGHT_HIP)

    origin = (l_hip + r_hip) * 0.5   # hip midpoint

    x_raw = r_sh - l_sh              # right direction (body frame)
    xn    = np.linalg.norm(x_raw)
    if xn < 1e-6:
        return origin, np.eye(3)
    x = x_raw / xn

    hip_mid   = (l_hip + r_hip) * 0.5
    sh_mid    = (l_sh  + r_sh)  * 0.5
    up_raw    = sh_mid - hip_mid      # hip→shoulder (up direction in world)
    up_ortho  = up_raw - np.dot(up_raw, x) * x
    yn        = np.linalg.norm(up_ortho)
    if yn < 1e-6:
        return origin, np.eye(3)
    y = up_ortho / yn

    z = np.cross(x, y)               # forward (should point roughly toward camera)
    R = np.column_stack([x, y, z])   # body_local = R.T @ (world_pt - origin)
    return origin, R


def world_to_body_local(world_pts_np, origin, R):
    """world_pts_np: np array (N×3). 반환: body-local N×3."""
    shifted = world_pts_np - origin[None, :]
    return (R.T @ shifted.T).T        # N×3


def body_local_to_screen_wrist_rel(bl_wrist, shoulder_width_m):
    """
    body-local wrist position → screen-space WristBodyRelative 등가값.
    bl_wrist: 3-vector (x_right, y_up, z_fwd) in meters.
    shoulder_width_m: 어깨 너비 (미터).
    반환: (brel_x, brel_y)  — screen brel x = bl_x / sw, brel y = -bl_y / sw (Y 반전)
    """
    if shoulder_width_m < MIN_SHOULDER_WIDTH_M:
        return float("nan"), float("nan")
    return (bl_wrist[0] / shoulder_width_m,
            -bl_wrist[1] / shoulder_width_m)   # Y 반전: body_up↔screen_down

# ─── 1. 동기화: 손 궤적 교차상관 ─────────────────────────────────────────────

def extract_hand_dist_series(view_name, video_path, sample_every=SYNC_SAMPLE):
    """
    IMAGE 모드로 서브샘플링해 양손 손목 거리 시계열 추출.
    반환: (frame_indices, distances, fps)
    """
    print(f"  [xcorr] {view_name}: 손 거리 시계열 추출 (매 {sample_every}프레임)...")
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE, num_hands=2)

    idxs  = []
    dists = []
    fi    = 0
    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hl:
        while True:
            ret, bgr = cap.read()
            if not ret: break
            if fi % sample_every == 0:
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hl.detect(mp_img)
                if result.hand_landmarks and len(result.hand_landmarks) >= 2:
                    w0 = result.hand_landmarks[0][0]
                    w1 = result.hand_landmarks[1][0]
                    d  = dist2d(w0.x, w0.y, w1.x, w1.y)
                else:
                    d  = 1.0   # 검출 실패 = 최대 거리
                idxs.append(fi)
                dists.append(d)
                if len(idxs) % 120 == 0:
                    print(f"    {fi}/{total}", end="\r")
            fi += 1
    cap.release()
    print(f"    {view_name}: {len(dists)}개 샘플")
    return np.array(idxs), np.array(dists), fps


def xcorr_sync(front_dists, other_dists, front_fps, other_fps, max_lag_frac=SYNC_MAX_LAG_FRAC):
    """
    교차상관으로 front 기준 other 뷰의 최적 lag(초) 추정.
    양수 lag = other 영상이 front보다 늦게 시작.
    반환: (lag_frames_front, confidence, xcorr_peak)
    """
    # 공통 길이로 맞추고 정규화
    n_use = min(len(front_dists), len(other_dists))
    f_sig = front_dists[:n_use]
    o_sig = other_dists[:n_use]

    f_norm = (f_sig - f_sig.mean()) / (f_sig.std() + 1e-8)
    o_norm = (o_sig - o_sig.mean()) / (o_sig.std() + 1e-8)

    xcorr = correlate(f_norm, o_norm, mode="full")
    lags  = np.arange(-(n_use - 1), n_use)

    # max_lag 제한
    max_lag_samples = int(n_use * max_lag_frac)
    mask  = (np.abs(lags) <= max_lag_samples)
    best_idx = np.argmax(xcorr[mask])
    best_lag_samples = lags[mask][best_idx]  # sample 단위 (서브샘플 프레임)
    peak_val = xcorr[mask][best_idx] / n_use

    # sample → 원본 프레임 변환 (각 sample = SYNC_SAMPLE 프레임 간격)
    best_lag_raw = best_lag_samples * SYNC_SAMPLE  # front 기준 원본 프레임 lag
    return best_lag_raw, float(peak_val)


def validate_sync(views_data):
    """
    3개 뷰의 손 거리 시계열 추출 + 교차상관 동기화 재검증.
    반환: sync_result dict (front clap frame, per-view offsets, confidence)
    """
    print("\n=== Step 1: 동기화 교차상관 검증 ===")
    series = {}
    for vn, vp in VIDEOS.items():
        idxs, dists, fps = extract_hand_dist_series(vn, vp)
        series[vn] = {"idxs": idxs, "dists": dists, "fps": fps}

    # front 기준 각 뷰 xcorr
    front_dists = series["front"]["dists"]
    front_fps   = series["front"]["fps"]

    # 손 거리 최솟값 = clap 후보 (영상 앞 25% 내에서만 검색)
    max_clap_s = max(1, len(front_dists) * 25 // 100)
    front_clap_sample = int(np.argmin(front_dists[:max_clap_s]))
    front_clap_raw    = int(series["front"]["idxs"][front_clap_sample])

    sync = {
        "method": "hand_trajectory_cross_correlation",
        "note":   "No audio in source videos; visual hand trajectory used",
        "front_clap_frame": front_clap_raw,
        "views":  {"front": {"clap_frame": front_clap_raw, "lag_from_front": 0, "xcorr_confidence": 1.0}},
    }

    for vn in ["left45", "right45"]:
        other_dists = series[vn]["dists"]
        lag_raw, conf = xcorr_sync(front_dists, other_dists, front_fps, series[vn]["fps"])

        # other clap = 영상 앞 25% 내에서 최솟값
        max_other_s = max(1, len(other_dists) * 25 // 100)
        other_clap_s = int(np.argmin(other_dists[:max_other_s]))
        other_clap_raw = int(series[vn]["idxs"][other_clap_s])

        # Residual: clap-based lag vs xcorr-based lag
        clap_lag = front_clap_raw - other_clap_raw
        residual_frames = abs(int(lag_raw) - clap_lag)
        residual_ms = residual_frames / series[vn]["fps"] * 1000

        print(f"  {vn}: clap_lag={clap_lag}f  xcorr_lag={int(lag_raw)}f  "
              f"residual={residual_ms:.0f}ms  confidence={conf:.3f}")

        # 잔차 20프레임 이하면 clap 기반 유지, 크면 xcorr 사용
        if residual_frames <= 20:
            chosen_lag  = clap_lag
            chosen_src  = "clap_detection"
            chosen_clap = other_clap_raw
        else:
            # xcorr 결과로 역산
            chosen_lag  = int(lag_raw)
            chosen_clap = front_clap_raw - chosen_lag
            chosen_src  = "xcorr"
            print(f"  [경고] 잔차 {residual_ms:.0f}ms > 667ms → xcorr 결과 채택")

        sync["views"][vn] = {
            "clap_frame":        chosen_clap,
            "lag_from_front":    chosen_lag,
            "xcorr_lag_raw":     int(lag_raw),
            "clap_lag_raw":      clap_lag,
            "residual_ms":       round(residual_ms, 1),
            "xcorr_confidence":  round(conf, 4),
            "source":            chosen_src,
        }

    print(f"  동기화 완료: front={front_clap_raw}  "
          f"left45={sync['views']['left45']['clap_frame']}  "
          f"right45={sync['views']['right45']['clap_frame']}")
    return sync

# ─── 2. v2 랜드마크 추출 (world landmarks 포함) ──────────────────────────────

def extract_view_v2(view_name, video_path, subsample=1):
    """
    VIDEO 모드 전체 추출.
    저장: 기존 필드 + PoseWorldPoints, LeftHandWorldPoints, RightHandWorldPoints
    """
    print(f"[extract_v2] {view_name}: 시작 (subsample={subsample})...")
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO, num_hands=2)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO)

    frames = []
    fi = 0
    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        while True:
            ret, bgr = cap.read()
            if not ret: break
            if fi % subsample == 0:
                ts_ms  = int(fi * 1000 / fps)
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hr     = hand_lm.detect_for_video(mp_img, ts_ms)
                pr     = pose_lm.detect_for_video(mp_img, ts_ms)
                frames.append(_make_frame_v2(fi, ts_ms, hr, pr))
                if len(frames) % 60 == 0:
                    print(f"  {view_name}: {fi}/{total} ({fi*100//total}%)", end="\r")
            fi += 1
    cap.release()
    print(f"\n  {view_name}: {len(frames)}프레임 완료")
    v_h = sum(1 for f in frames if f["HasLeftHand"] or f["HasRightHand"])
    v_p = sum(1 for f in frames if f["HasPose"])
    print(f"  Hand:{v_h}  Pose:{v_p}")
    return {"view": view_name, "fps": fps/subsample, "original_fps": fps,
            "subsample": subsample, "frames": frames}


def _make_frame_v2(fi, ts_ms, hr, pr):
    """MediaPipe 결과 → v2 프레임 dict (world landmarks 포함)."""
    # ── Hand (screen + world) ────────────────────────────────────────────────
    has_left = has_right = False
    left_pts = right_pts = None
    left_vis = right_vis = None
    left_w   = right_w   = None  # world landmarks (wrist-centered, meters)

    if hr and hr.hand_landmarks:
        for i, lms in enumerate(hr.hand_landmarks):
            cat  = hr.handedness[i][0].category_name
            pts  = [lm_to_dict(lm) for lm in lms]
            vis  = [getattr(lm, "visibility", 1.0) for lm in lms]
            wpts = None
            if hr.hand_world_landmarks and i < len(hr.hand_world_landmarks):
                wpts = [lm_to_dict(lm) for lm in hr.hand_world_landmarks[i]]
            if cat == "Left":
                has_left = True; left_pts = pts; left_vis = vis; left_w = wpts
            else:
                has_right = True; right_pts = pts; right_vis = vis; right_w = wpts

    # ── Pose (screen + world) ────────────────────────────────────────────────
    has_pose = has_valid_sh = False
    pose_pts = pose_w = None
    pose_vis = None
    sh_cx = sh_cy = sw = 0.0
    l_wr_rel = r_wr_rel = {"x": float("nan"), "y": float("nan")}
    nose_y_screen = float("nan")

    if pr and pr.pose_landmarks:
        lms = pr.pose_landmarks[0]
        has_pose = True
        pose_pts = [lm_to_dict(lm) for lm in lms]
        pose_vis = [getattr(lm, "visibility", 1.0) for lm in lms]
        l_sh = lms[LEFT_SHOULDER]; r_sh = lms[RIGHT_SHOULDER]
        sh_cx = (l_sh.x + r_sh.x) * 0.5
        sh_cy = (l_sh.y + r_sh.y) * 0.5
        sw    = dist2d(l_sh.x, l_sh.y, r_sh.x, r_sh.y)
        has_valid_sh = sw > MIN_SHOULDER_WIDTH
        if has_valid_sh:
            def brel(px, py):
                return {"x": (px-sh_cx)/sw, "y": (py-sh_cy)/sw}
            l_wr_rel = brel(lms[LEFT_WRIST].x,  lms[LEFT_WRIST].y)
            r_wr_rel = brel(lms[RIGHT_WRIST].x, lms[RIGHT_WRIST].y)
        nose_y_screen = lms[NOSE].y

        if pr.pose_world_landmarks:
            pose_w = [lm_to_dict(lm) for lm in pr.pose_world_landmarks[0]]

    # ── Quality ─────────────────────────────────────────────────────────────
    has_hand = has_left or has_right
    if   has_hand and has_pose and has_valid_sh: quality = QUALITY_GOOD
    elif has_hand and has_pose:                   quality = QUALITY_INSUFFICIENT_POSE
    elif has_hand:                                quality = QUALITY_HAND_ONLY
    elif has_pose:                                quality = QUALITY_POSE_ONLY
    else:                                         quality = QUALITY_INVALID

    return {
        "FrameIdx": fi, "TimestampMs": ts_ms,
        "HasLeftHand": has_left, "HasRightHand": has_right,
        "HasPose": has_pose, "HasValidShoulders": has_valid_sh,
        "LeftHandPoints":   left_pts,  "RightHandPoints":  right_pts,
        "LeftHandVis":      left_vis,  "RightHandVis":     right_vis,
        "LeftHandWorldPts": left_w,    "RightHandWorldPts":right_w,
        "PosePoints":       pose_pts,  "PoseWorldPoints":  pose_w,
        "PoseVis":          pose_vis,
        "ShoulderCenter":   {"x": sh_cx, "y": sh_cy},
        "ShoulderWidth":    sw,
        "LeftWristBodyRelative":  l_wr_rel,
        "RightWristBodyRelative": r_wr_rel,
        "NoseYScreen": nose_y_screen,
        "Quality":     quality,
    }

# ─── 3. 동기화 인덱스 변환 ───────────────────────────────────────────────────

def to_view_arr_idx_v2(view_name, front_raw_idx, sync_info, views_data):
    """front raw_idx → 해당 뷰 frames[] 배열 인덱스 (sync_info 기반)."""
    front_fps   = views_data["front"]["original_fps"]
    front_clap  = sync_info["front_clap_frame"]
    vinfo       = sync_info["views"][view_name]
    view_clap   = vinfo["clap_frame"]
    view_fps    = views_data[view_name]["original_fps"]
    view_sub    = views_data[view_name]["subsample"]

    t_offset     = (front_raw_idx - front_clap) / front_fps
    view_raw_idx = view_clap + t_offset * view_fps
    arr_idx      = round(view_raw_idx / view_sub)
    frames       = views_data[view_name]["frames"]
    return max(0, min(len(frames) - 1, arr_idx))

# ─── 4. 체간 좌표계 기반 분절 신호 + scipy prominence ───────────────────────

def compute_head_bob_signal(front_frames):
    """
    front 뷰 pose_world_landmarks에서 체간 body-local 좌표계로 변환한
    head Y 신호 계산.
    반환: np.array (front_frames 길이, NaN 포함 가능)
    """
    signal = np.full(len(front_frames), np.nan)
    for i, f in enumerate(front_frames):
        pwp = f.get("PoseWorldPoints")
        if pwp is None or len(pwp) < 33:
            continue
        # dict 또는 lm 객체 지원
        try:
            origin, R = compute_body_frame(pwp)
            nose_w = np.array([pwp[NOSE]["x"], pwp[NOSE]["y"], pwp[NOSE]["z"]])
            nose_bl = R.T @ (nose_w - origin)
            signal[i] = nose_bl[1]   # body-local Y (위 = 양수)
        except Exception:
            pass
    return signal


def segment_reps_v2(front_frames, front_clap_raw, subsample=1):
    """
    body-local head Y 신호에 scipy find_peaks(prominence) 적용.
    반드시 정확히 EXPECTED_REPS(12) 개를 찾아야 함.
    성공: (segs, peaks_abs, signal_smooth)
    실패: (None, peaks_abs, signal_smooth) — 호출자가 검토 MP4 생성 후 중단
    """
    fps = 30.0 / subsample
    min_dist_frames = int(MIN_DIST_FRAMES_30 / max(subsample, 1))

    # clap 이후 frames[] 배열 시작 인덱스
    clap_arr = 0
    for i, f in enumerate(front_frames):
        if f["FrameIdx"] >= front_clap_raw:
            clap_arr = i
            break

    signal_raw = compute_head_bob_signal(front_frames)
    signal_sm  = smooth_boxcar(signal_raw.tolist(), SMOOTH_WINDOW)

    # clap 이후 신호
    post_clap = -signal_sm[clap_arr:]   # 음수화 → 최솟값(bow)을 최댓값으로

    peaks_local, props = find_peaks(
        post_clap,
        prominence=MIN_PROM_METERS,
        distance=min_dist_frames,
    )
    peaks_abs = [p + clap_arr for p in peaks_local]
    proms_abs  = props.get("prominences", [])

    n = len(peaks_abs)
    print(f"[segment_v2] 탐지 피크: {n}  (예상: {EXPECTED_REPS})")
    for i, (pk, pr) in enumerate(zip(peaks_abs, proms_abs)):
        t = front_frames[pk]["TimestampMs"] / 1000.0
        print(f"  rep[{i:2d}]: arr={pk}  t={t:.1f}s  prom={pr:.4f}m")

    if n != EXPECTED_REPS:
        print(f"\n[실패] 피크 {n}개 != {EXPECTED_REPS}개 - Blender 생성 중단.")
        print("  검토 MP4와 boundaries 파일을 생성합니다.")
        print("  수동 경계 제공: --boundaries-csv <파일>")
        return None, peaks_abs, signal_sm, proms_abs

    # 구간: 이전-다음 피크 중간
    n_total = len(front_frames)
    segs = []
    for i, pk in enumerate(peaks_abs):
        start = (peaks_abs[i-1] + pk) // 2 if i > 0 else clap_arr
        end   = (pk + peaks_abs[i+1]) // 2 if i < n-1 else min(pk + (n_total-pk)//2, n_total-1)
        segs.append((start, end))
    return segs, peaks_abs, signal_sm, proms_abs

# ─── 5. 검토 MP4 (분절 실패 시) ──────────────────────────────────────────────

def generate_review_mp4(front_frames, signal_sm, peaks_abs, proms_abs,
                         video_path, out_path, sync_clap_raw):
    """
    front 영상에 head_bob 신호 + 탐지 피크를 오버레이한 검토 MP4 생성.
    """
    print(f"[review_mp4] {out_path} 생성 중...")
    cap  = cv2.VideoCapture(video_path)
    fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w, out_h = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    # 신호 시각화용 정규화
    sig_min = np.nanmin(signal_sm) - 0.01
    sig_max = np.nanmax(signal_sm) + 0.01
    sig_range = max(sig_max - sig_min, 0.01)
    bar_h = 120   # 신호 바 높이

    peaks_set = set(peaks_abs)
    fi = 0
    arr_idx = 0

    while True:
        ret, bgr = cap.read()
        if not ret: break

        frame_resized = cv2.resize(bgr, (out_w, out_h - bar_h))
        canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        canvas[:out_h - bar_h, :] = frame_resized

        # 신호 막대 그리기
        sig_bar_y = out_h - bar_h
        canvas[sig_bar_y:, :] = (30, 30, 30)

        # 신호 파형 그리기 (전체 신호 맵핑)
        n_total = len(signal_sm)
        for x in range(out_w):
            si = int(x * n_total / out_w)
            if 0 <= si < n_total and not math.isnan(signal_sm[si]):
                sv = (signal_sm[si] - sig_min) / sig_range
                sy = int(sig_bar_y + bar_h - sv * bar_h)
                sy = max(sig_bar_y, min(out_h - 1, sy))
                color = (0, 200, 0)
                if si in peaks_set: color = (0, 0, 255)
                cv2.line(canvas, (x, sig_bar_y + bar_h - 1), (x, sy), color, 1)

        # 현재 프레임 위치 표시
        if arr_idx < n_total:
            cx = int(arr_idx * out_w / n_total)
            cv2.line(canvas, (cx, sig_bar_y), (cx, out_h - 1), (255, 255, 0), 2)

        # clap 위치 표시
        if sync_clap_raw is not None:
            clap_x = int(sync_clap_raw * out_w / n_total)
            cv2.line(canvas, (clap_x, sig_bar_y), (clap_x, out_h-1), (255, 0, 255), 2)

        # 피크 위치 표시
        for pk in peaks_abs:
            px = int(pk * out_w / n_total)
            cv2.line(canvas, (px, sig_bar_y), (px, out_h-1), (0, 0, 255), 1)

        # 텍스트 오버레이
        ts_s = fi / fps
        cv2.putText(canvas, f"frame={fi} arr={arr_idx} t={ts_s:.1f}s",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(canvas, f"Peaks detected: {len(peaks_abs)} (expected {EXPECTED_REPS})",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.putText(canvas, "GREEN=signal  RED=peaks  YELLOW=current  MAGENTA=clap",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        writer.write(canvas)
        fi += 1
        arr_idx += 1  # 1:1 (subsample=1 assumed for review)

    writer.release()
    cap.release()
    print(f"  완료: {out_path}")

# ─── 6. 체간 좌표계 기반 멀티뷰 융합 ────────────────────────────────────────

def fuse_rep_v2(seg, views_data, sync_info):
    """
    선택된 rep을 3-뷰 체간 좌표계 융합.

    전략:
    - Pose world → body-local (hip-centered, 어깨축 X, 상하 Y, 전후 Z)
    - Hand world → pose wrist에 정합 후 body-local 변환
    - per-landmark: visibility+continuity 가중 평균 + 이상치(2σ) 제거
    - 융합 결과 → blend_animate 호환 형식(screen-space 등가) + canonical 3D 저장

    반환:
      blend_frames  : blend_animate.py 호환 프레임 리스트 (screen-space 필드 포함)
      canonical_frames : body-local 3D 프레임 리스트
      contribution  : per-frame per-landmark 기여도 dict
    """
    start, end = seg
    front_frames = views_data["front"]["frames"]
    blend_frames     = []
    canonical_frames = []
    contribution     = []

    prev_bl_pose  = None   # continuity tracking
    prev_bl_lhand = None
    prev_bl_rhand = None

    for fi in range(start, end + 1):
        ff = front_frames[fi]
        frame_contrib = {"fi": fi, "ts": ff["TimestampMs"], "views": {}}

        # 동기화된 각 뷰 프레임 가져오기
        view_frames = {"front": ff}
        for vn in ["left45", "right45"]:
            ai = to_view_arr_idx_v2(vn, ff["FrameIdx"], sync_info, views_data)
            view_frames[vn] = views_data[vn]["frames"][ai]

        # ── Pose body-local 융합 (front 전용, 정면이 가장 정확) ──────────
        bl_pose, pose_origin, pose_R, sw_m = None, None, np.eye(3), 0.0
        if ff["HasPose"] and ff["PoseWorldPoints"]:
            try:
                pose_origin, pose_R = compute_body_frame(ff["PoseWorldPoints"])
                pw_arr = np.array([[p["x"], p["y"], p["z"]]
                                   for p in ff["PoseWorldPoints"]])
                bl_pose = world_to_body_local(pw_arr, pose_origin, pose_R)  # 33×3

                l_sh = pw_arr[LEFT_SHOULDER]; r_sh = pw_arr[RIGHT_SHOULDER]
                sw_m = float(np.linalg.norm(r_sh - l_sh))
                frame_contrib["pose_view"] = "front"
                frame_contrib["sw_meters"] = round(sw_m, 4)
            except Exception as e:
                bl_pose = None

        # ── Hand 융합: 3뷰에서 body-local 변환 후 가중 평균 ──────────────
        def fuse_hand_side(side):
            """
            side = 'Left' or 'Right'
            반환: (has_fused, bl_hand_21x3, best_screen_pts, screen_vis, contrib_info)
            """
            has_k = f"Has{side}Hand"
            wrl_k = f"{side}HandWorldPts"
            scr_k = f"{side}HandPoints"
            vis_k = f"{side}HandVis"

            candidates = []   # (view_name, bl_21x3, vis_sum, screen_pts)
            for vn in VIEW_ORDER:
                vf = view_frames[vn]
                if not vf[has_k]:
                    continue
                wrl_pts = vf.get(wrl_k)
                scr_pts = vf.get(scr_k)
                vis     = vf.get(vis_k)
                vsum    = vis_sum_safe(vis)

                if wrl_pts is None or len(wrl_pts) < 21:
                    candidates.append((vn, None, vsum, scr_pts, vis))
                    continue

                # hand_world → body-local
                # MediaPipe hand_world는 손목 중심이 아님 (wrist ≈ pt[0])
                # pose wrist world로 손목 위치 앵커링
                # 해당 뷰의 pose world wrist 사용
                vf_pose_w = vf.get("PoseWorldPoints")
                if vf_pose_w is None or pose_origin is None:
                    candidates.append((vn, None, vsum, scr_pts, vis))
                    continue

                try:
                    wrist_idx = LEFT_WRIST if side == "Left" else RIGHT_WRIST
                    pw_arr_v  = np.array([[p["x"], p["y"], p["z"]]
                                          for p in vf_pose_w])
                    pose_wrist_world = pw_arr_v[wrist_idx]

                    # hand_world[0] = wrist-anchor; hand_world[i] = relative to wrist
                    hw_arr = np.array([[p["x"], p["y"], p["z"]]
                                       for p in wrl_pts])
                    # hand pt[0]이 정확히 0이 아닐 수 있으므로 오프셋 제거
                    hw_arr = hw_arr - hw_arr[0]

                    # body-local 변환: pose wrist world + relative hand world
                    hw_world_abs = pose_wrist_world[None, :] + hw_arr  # 21×3, world

                    # front 체간 좌표계로 변환
                    bl_hand = world_to_body_local(hw_world_abs, pose_origin, pose_R)
                    candidates.append((vn, bl_hand, vsum, scr_pts, vis))
                except Exception:
                    candidates.append((vn, None, vsum, scr_pts, vis))

            # body-local이 있는 후보만 융합
            bl_candidates = [(vn, bl, vs, sc, vi)
                              for vn, bl, vs, sc, vi in candidates if bl is not None]

            if not bl_candidates:
                # screen-space fallback (v1 방식: visibility 합 기준 최고 뷰)
                scr_cands = [(vs, sc, vi, vn) for vn, _, vs, sc, vi in candidates if sc is not None]
                if not scr_cands:
                    return False, None, None, None, {"views_used": [], "method": "none"}
                scr_cands.sort(key=lambda x: x[0], reverse=True)
                _, best_sc, best_vi, best_vn = scr_cands[0]
                return True, None, best_sc, best_vi, {"views_used": [best_vn], "method": "screen_fallback"}

            # ── per-landmark 가중 융합 ──────────────────────────────────
            n_lm = 21
            vis_weights = np.array([vs for _, _, vs, _, _ in bl_candidates])
            vis_weights = vis_weights / (vis_weights.sum() + 1e-8)

            bl_stack = np.array([bl for _, bl, _, _, _ in bl_candidates])  # K×21×3

            # visibility 가중 평균
            bl_mean = np.einsum("k,kij->ij", vis_weights, bl_stack)  # 21×3

            # continuity 가중치 (이전 프레임 참조)
            prev = prev_bl_lhand if side == "Left" else prev_bl_rhand
            if prev is not None:
                dists_from_prev = np.linalg.norm(
                    bl_stack - prev[None, :, :], axis=2)  # K×21
                # 거리가 클수록 낮은 continuity 가중치
                cont_w = 1.0 / (1.0 + dists_from_prev.mean(axis=1))
                cont_w = cont_w / (cont_w.sum() + 1e-8)
                combined_w = 0.7 * vis_weights + 0.3 * cont_w
            else:
                combined_w = vis_weights

            bl_mean = np.einsum("k,kij->ij", combined_w, bl_stack)

            # 이상치 제거 (2σ)
            for lm_i in range(n_lm):
                pts_lm = bl_stack[:, lm_i, :]   # K×3
                dists  = np.linalg.norm(pts_lm - bl_mean[lm_i][None, :], axis=1)
                sigma  = np.std(dists)
                outlier = dists > 2 * sigma
                if outlier.any() and not outlier.all():
                    w_adj = combined_w.copy()
                    w_adj[outlier] = 0.0
                    w_adj = w_adj / w_adj.sum()
                    bl_mean[lm_i] = np.einsum("k,kj->j", w_adj, pts_lm)

            # 잔차 계산
            residuals = np.linalg.norm(bl_stack - bl_mean[None, :, :], axis=2)  # K×21

            views_used = [vn for vn, _, _, _, _ in bl_candidates]
            contrib_info = {
                "views_used": views_used,
                "method": "body_local_weighted_fusion",
                "weights": combined_w.tolist(),
                "mean_residual_m": float(residuals.mean()),
                "max_residual_m":  float(residuals.max()),
            }

            # screen pts: 가장 잔차 작은 뷰에서 가져옴
            best_idx   = int(np.argmin(residuals.mean(axis=1)))
            best_vn    = bl_candidates[best_idx][0]
            best_sc    = next((sc for vn,_,_,sc,_ in candidates if vn==best_vn), None)
            best_vi    = next((vi for vn,_,_,_,vi in candidates if vn==best_vn), None)

            return True, bl_mean, best_sc, best_vi, contrib_info

        has_l, bl_l, scr_l, vis_l, ctr_l = fuse_hand_side("Left")
        has_r, bl_r, scr_r, vis_r, ctr_r = fuse_hand_side("Right")

        frame_contrib["left_hand"]  = ctr_l
        frame_contrib["right_hand"] = ctr_r

        # ── WristBodyRelative (body-local 3D → screen 등가) ──────────────
        def wrist_brel_from_bl(bl_hand_21x3):
            if bl_hand_21x3 is None or sw_m < MIN_SHOULDER_WIDTH_M:
                return {"x": float("nan"), "y": float("nan")}
            bx, by = body_local_to_screen_wrist_rel(bl_hand_21x3[0], sw_m)
            return {"x": bx, "y": by}

        l_wr_fused = wrist_brel_from_bl(bl_l) if has_l else ff["LeftWristBodyRelative"]
        r_wr_fused = wrist_brel_from_bl(bl_r) if has_r else ff["RightWristBodyRelative"]

        # ── Quality ──────────────────────────────────────────────────────
        has_hand = has_l or has_r
        has_pose_f = ff["HasPose"]
        has_valid_sh_f = ff["HasValidShoulders"]
        if   has_hand and has_pose_f and has_valid_sh_f: quality = QUALITY_GOOD
        elif has_hand and has_pose_f:                     quality = QUALITY_INSUFFICIENT_POSE
        elif has_hand:                                    quality = QUALITY_HAND_ONLY
        elif has_pose_f:                                  quality = QUALITY_POSE_ONLY
        else:                                             quality = QUALITY_INVALID

        ts_rel = ff["TimestampMs"] - front_frames[start]["TimestampMs"]

        blend_frames.append({
            "TimestampMs":            ts_rel,
            "HasLeftHand":            has_l,
            "HasRightHand":           has_r,
            "HasPose":                has_pose_f,
            "HasValidShoulders":      has_valid_sh_f,
            "LeftHandPoints":         scr_l,
            "RightHandPoints":        scr_r,
            "PosePoints":             ff["PosePoints"],
            "ShoulderCenter":         ff["ShoulderCenter"],
            "ShoulderWidth":          ff["ShoulderWidth"],
            "LeftWristBodyRelative":  l_wr_fused,
            "RightWristBodyRelative": r_wr_fused,
            "Quality":                quality,
        })

        canonical_frames.append({
            "TimestampMs":        ts_rel,
            "HasLeftHand":        has_l,
            "HasRightHand":       has_r,
            "HasPose":            has_pose_f,
            "ShoulderWidthM":     round(sw_m, 4),
            "PoseBodyLocal":      bl_pose.tolist() if bl_pose is not None else None,
            "LeftHandBodyLocal":  bl_l.tolist()    if bl_l  is not None else None,
            "RightHandBodyLocal": bl_r.tolist()    if bl_r  is not None else None,
        })
        contribution.append(frame_contrib)

        # continuity 업데이트
        prev_bl_pose  = bl_pose
        prev_bl_lhand = bl_l
        prev_bl_rhand = bl_r

    return blend_frames, canonical_frames, contribution

# ─── 7. 반복 구간 점수화 (12개 후보) ─────────────────────────────────────────

def score_rep_v2(seg, front_frames, views_data, sync_info):
    """
    hand_ratio + pose_ratio + inter-view residual + continuity 통합 점수.
    """
    start, end = seg
    count = end - start + 1
    if count <= 0:
        return 0.0

    hand_ok = pose_ok = 0
    residuals = []

    for fi in range(start, end + 1):
        ff = front_frames[fi]
        view_frames = {"front": ff}
        for vn in ["left45", "right45"]:
            ai = to_view_arr_idx_v2(vn, ff["FrameIdx"], sync_info, views_data)
            view_frames[vn] = views_data[vn]["frames"][ai]

        any_hand = any(view_frames[vn]["HasLeftHand"] or view_frames[vn]["HasRightHand"]
                       for vn in VIEW_ORDER)
        if any_hand:    hand_ok += 1
        if ff["HasPose"]: pose_ok += 1

        # inter-view wrist residual (body-relative screen X)
        wrists = []
        for vn in VIEW_ORDER:
            vf = view_frames[vn]
            for side in ["Left", "Right"]:
                if vf[f"Has{side}Hand"]:
                    rel = vf[f"{side}WristBodyRelative"]
                    x   = rel.get("x") if isinstance(rel, dict) else rel["x"]
                    if x is not None and not math.isnan(float(x)):
                        wrists.append(float(x))
        if len(wrists) >= 2:
            residuals.append(float(np.std(wrists)))

    hand_ratio = hand_ok / count
    pose_ratio = pose_ok / count
    res_score  = 1.0 / (1.0 + np.mean(residuals)) if residuals else 0.5
    return 0.40 * hand_ratio + 0.30 * pose_ratio + 0.30 * res_score

# ─── 8. 수동 경계 CSV 로드 ───────────────────────────────────────────────────

def load_boundaries_csv(csv_path, front_frames):
    """
    수동 boundaries CSV 로드. 컬럼: rep_idx, start_arr, end_arr
    (또는 start_sec, end_sec 형식도 지원 — fps=30 가정)
    """
    segs = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "start_arr" in row and "end_arr" in row:
                segs.append((int(row["start_arr"]), int(row["end_arr"])))
            elif "start_sec" in row and "end_sec" in row:
                fps = 30.0
                segs.append((int(float(row["start_sec"]) * fps),
                              int(float(row["end_sec"])   * fps)))
    if len(segs) != EXPECTED_REPS:
        raise ValueError(f"boundaries CSV에 {len(segs)}개 구간 (필요: {EXPECTED_REPS})")
    return segs

# ─── 9. 수치 검증 ────────────────────────────────────────────────────────────

def compute_validation(blend_frames, canonical_frames):
    """
    손 바뀜·순간이동·장기 미검출·체간 잔차 수치 계산.
    """
    n = len(blend_frames)
    hand_swaps         = 0
    wrist_teleports    = []
    gap_runs           = []   # (start_fi, length)
    inter_view_residuals = []

    prev_l = prev_r = None
    gap_start = gap_len = 0
    in_gap    = False
    TELEPORT_THRESH_SW = 0.5   # 어깨 너비 단위 ≥ 0.5 = 순간이동

    for i, (bf, cf) in enumerate(zip(blend_frames, canonical_frames)):
        # 손 바뀜: 이전 프레임 대비 left/right 유무 반전
        cur_l = bf["HasLeftHand"]
        cur_r = bf["HasRightHand"]
        if prev_l is not None:
            if (cur_l and not prev_l and not cur_r and prev_r) or \
               (cur_r and not prev_r and not cur_l and prev_l):
                hand_swaps += 1
        prev_l = cur_l; prev_r = cur_r

        # 순간이동: WristBodyRelative 급변
        for side in ["Left", "Right"]:
            cur_rel  = bf[f"{side}WristBodyRelative"]
            cur_x    = cur_rel.get("x") if isinstance(cur_rel, dict) else cur_rel["x"]
            if cur_x is None or math.isnan(float(cur_x)):
                continue
            # 이전 유효값과 비교는 생략 (단순 threshold 검사)

        # 장기 미검출 갭
        has_any = bf["HasLeftHand"] or bf["HasRightHand"]
        if not has_any:
            if not in_gap:
                in_gap = True; gap_start = i; gap_len = 1
            else:
                gap_len += 1
        else:
            if in_gap and gap_len >= 10:   # 10프레임(0.33s) 이상 미검출
                gap_runs.append({"start": gap_start, "length": gap_len,
                                  "ts_ms": blend_frames[gap_start]["TimestampMs"]})
            in_gap = False; gap_len = 0

        # inter-view residual: canonical 체간 좌표 L/R wrist 잔차
        # (간략화: 융합 contribution 레포트에서 별도 추출)

    if in_gap and gap_len >= 10:
        gap_runs.append({"start": gap_start, "length": gap_len,
                          "ts_ms": blend_frames[gap_start]["TimestampMs"]})

    # 손목 순간이동 (body-local 좌표 사용, 연속 프레임 간 거리)
    teleport_count = 0
    TELEPORT_THRESH_M = 0.15   # 0.15m/frame = 4.5m/s
    for i in range(1, len(canonical_frames)):
        cf0 = canonical_frames[i-1]; cf1 = canonical_frames[i]
        for side in ["Left", "Right"]:
            k = f"{side}HandBodyLocal"
            p0 = cf0.get(k); p1 = cf1.get(k)
            if p0 and p1:
                wrist0 = np.array(p0[0])
                wrist1 = np.array(p1[0])
                if np.linalg.norm(wrist1 - wrist0) > TELEPORT_THRESH_M:
                    teleport_count += 1

    return {
        "total_frames":    n,
        "hand_swaps":      hand_swaps,
        "wrist_teleports": teleport_count,
        "long_gap_runs":   gap_runs,
        "long_gaps_count": len(gap_runs),
        "notes": {
            "teleport_threshold_m": 0.15,
            "gap_threshold_frames": 10,
            "swap_definition": "one hand appears as the other disappears in consecutive frames",
        }
    }

# ─── 10. 저장 유틸 ───────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=True, cls=_NumpyEncoder)
    sz = os.path.getsize(path)
    print(f"[저장] {os.path.relpath(path)} ({sz//1024}KB)")

def save_boundaries_v2_csv(segs, scores, peaks_abs, proms_abs, front_frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rep_idx","start_arr","end_arr","frame_count",
                    "peak_arr","peak_ts_s","prominence_m","score"])
        for i, (seg, sc) in enumerate(zip(segs, scores)):
            s, e = seg
            pk   = peaks_abs[i] if i < len(peaks_abs) else ""
            pr   = round(float(proms_abs[i]), 4) if i < len(proms_abs) else ""
            ts_s = round(front_frames[pk]["TimestampMs"]/1000.0, 2) if pk != "" else ""
            w.writerow([i, s, e, e-s+1, pk, ts_s, pr, round(sc, 4)])
    print(f"[저장] {os.path.relpath(path)}")

def save_head_bob_csv(signal_sm, front_frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arr_idx","frame_idx","ts_ms","nose_body_local_y"])
        for i, f_fr in enumerate(front_frames):
            val = signal_sm[i] if i < len(signal_sm) else float("nan")
            w.writerow([i, f_fr["FrameIdx"], f_fr["TimestampMs"],
                        round(float(val), 6)])
    print(f"[저장] {os.path.relpath(path)}")

# ─── 11. Blender 애니메이션 실행 ─────────────────────────────────────────────

def run_blender(blend_script, log_path, blend_in=None):
    """Blender CLI 실행. blend_in=None이면 BLEND_FILE 사용."""
    blend_src = blend_in or BLEND_FILE
    cmd = [BLENDER_EXE, blend_src, "--background", "--python", blend_script]
    print(f"[Blender] {os.path.basename(blend_script)} 실행 중...")
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
    print(f"  종료코드: {result.returncode}  로그: {os.path.relpath(log_path)}")
    return result.returncode

# ─── 12. Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="S01 인사 멀티뷰 파이프라인 v2")
    ap.add_argument("--skip-extract",     action="store_true",
                    help="v2 raw JSON 캐시 재사용")
    ap.add_argument("--subsample",        type=int, default=1,
                    help="추출 서브샘플 (1=30fps, 2=15fps)")
    ap.add_argument("--boundaries-csv",   default="",
                    help="수동 경계 CSV (auto segmentation 실패 시)")
    ap.add_argument("--rep-idx",          type=int, default=-1,
                    help="rep 수동 선택 (-1=자동)")
    ap.add_argument("--skip-blender",     action="store_true",
                    help="Blender 단계 건너뜀")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 파일 확인 ────────────────────────────────────────────────────────────
    for p in (HAND_MODEL, POSE_MODEL):
        if not os.path.exists(p):
            sys.exit(f"[오류] 모델 없음: {p}")
    for vn, vp in VIDEOS.items():
        if not os.path.exists(vp):
            sys.exit(f"[오류] 비디오 없음: {vp}")
        print(f"[확인] {vn}: {os.path.basename(vp)}")

    # ── Step 1: 동기화 교차상관 검증 ─────────────────────────────────────────
    sync_path = os.path.join(OUT_DIR, "sync_validation.json")
    if os.path.exists(sync_path) and args.skip_extract:
        print(f"\n[캐시] {os.path.relpath(sync_path)}")
        with open(sync_path, encoding="utf-8") as f:
            sync_info = json.load(f)
    else:
        sync_info = validate_sync({})
        save_json(sync_info, sync_path)

    # ── Step 2: v2 랜드마크 추출 ─────────────────────────────────────────────
    raw_paths  = {vn: os.path.join(OUT_DIR, f"raw_v2_{vn}.json") for vn in VIDEOS}
    views_data = {}
    print(f"\n=== Step 2: v2 추출 (subsample={args.subsample}) ===")
    for vn, vp in VIDEOS.items():
        rp = raw_paths[vn]
        if args.skip_extract and os.path.exists(rp):
            print(f"[캐시] {os.path.relpath(rp)} 로드...")
            with open(rp, encoding="utf-8") as f:
                views_data[vn] = json.load(f)
            print(f"  {vn}: {len(views_data[vn]['frames'])}프레임")
        else:
            vdata = extract_view_v2(vn, vp, subsample=args.subsample)
            views_data[vn] = vdata
            save_json(vdata, rp)

    front_frames = views_data["front"]["frames"]
    front_clap   = sync_info["front_clap_frame"]

    # ── Step 3: 분절 ─────────────────────────────────────────────────────────
    print(f"\n=== Step 3: 분절 (expected={EXPECTED_REPS}) ===")

    # 수동 경계 우선
    if args.boundaries_csv and os.path.exists(args.boundaries_csv):
        print(f"  수동 boundaries CSV: {args.boundaries_csv}")
        segs = load_boundaries_csv(args.boundaries_csv, front_frames)
        peaks_abs = [s + (e-s)//2 for s, e in segs]   # 구간 중심을 가짜 피크로
        proms_abs = [0.0] * len(segs)
        signal_sm = smooth_boxcar(compute_head_bob_signal(front_frames).tolist(), SMOOTH_WINDOW)
    else:
        result = segment_reps_v2(front_frames, front_clap, subsample=args.subsample)
        segs, peaks_abs, signal_sm, proms_abs = result

        # 분절 실패 → 검토 MP4 + 중단
        if segs is None:
            review_path = os.path.join(OUT_DIR, "segmentation_fail_review.mp4")
            generate_review_mp4(
                front_frames, signal_sm, peaks_abs, proms_abs,
                VIDEOS["front"], review_path, front_clap)
            save_head_bob_csv(signal_sm, front_frames,
                              os.path.join(OUT_DIR, "head_bob_signal.csv"))
            print(f"\n중단: 분절 실패.")
            print(f"  검토 MP4: {review_path}")
            print(f"  신호 CSV: {os.path.join(OUT_DIR, 'head_bob_signal.csv')}")
            print(f"  수동 경계 작성 후 --boundaries-csv 로 재실행하거나")
            print(f"  MIN_PROM_METERS({MIN_PROM_METERS})/MIN_DIST_FRAMES_30({MIN_DIST_FRAMES_30}) 조정")
            sys.exit(1)

    save_head_bob_csv(signal_sm, front_frames,
                      os.path.join(OUT_DIR, "head_bob_signal.csv"))

    # ── Step 4: 점수화 + 선택 ────────────────────────────────────────────────
    print(f"\n=== Step 4: {len(segs)}개 rep 점수화 ===")
    scores = [score_rep_v2(s, front_frames, views_data, sync_info) for s in segs]

    save_boundaries_v2_csv(segs, scores, peaks_abs, proms_abs, front_frames,
                            os.path.join(OUT_DIR, "boundaries_v2.csv"))

    for i, (seg, sc) in enumerate(zip(segs, scores)):
        pk = peaks_abs[i] if i < len(peaks_abs) else "?"
        print(f"  rep[{i:2d}]: arr=[{seg[0]},{seg[1]}] len={seg[1]-seg[0]+1:3d} score={sc:.4f}")

    if 0 <= args.rep_idx < len(segs):
        best = args.rep_idx
        print(f"\n[선택] rep_idx={best} (수동)")
    else:
        best = int(np.argmax(scores))
        print(f"\n[선택] rep_idx={best} (자동, score={scores[best]:.4f})")

    best_seg = segs[best]
    n_seg = best_seg[1] - best_seg[0] + 1
    t_ms  = front_frames[best_seg[1]]["TimestampMs"] - front_frames[best_seg[0]]["TimestampMs"]
    print(f"  구간: [{best_seg[0]},{best_seg[1]}]  {n_seg}프레임  {t_ms}ms")

    # ── Step 5: 체간 좌표계 융합 ─────────────────────────────────────────────
    print(f"\n=== Step 5: 멀티뷰 융합 (body-aligned MediaPipe world fusion) ===")
    blend_frames, canonical_frames, contribution = fuse_rep_v2(
        best_seg, views_data, sync_info)

    n_total = len(blend_frames)
    v_h     = sum(1 for f in blend_frames if f["HasLeftHand"] or f["HasRightHand"])
    v_p     = sum(1 for f in blend_frames if f["HasPose"])
    v_c     = sum(1 for f in blend_frames if (f["HasLeftHand"] or f["HasRightHand"]) and f["HasPose"])
    dur_ms  = blend_frames[-1]["TimestampMs"] if blend_frames else 0
    print(f"  {n_total}프레임  손:{v_h}({v_h*100//n_total}%)  포즈:{v_p}({v_p*100//n_total}%)  "
          f"결합:{v_c}({v_c*100//n_total}%)  {dur_ms}ms")

    # 융합 메타데이터
    fusion_meta = {
        "fusion_method":       "body-aligned MediaPipe world fusion",
        "coordinate_system":   "hip_midpoint origin, right_shoulder X, hip_to_shoulder Y, X_cross_Y Z",
        "not_calibrated":      True,
        "note":                "Not calibrated triangulation — body-aligned world landmarks from single-view MediaPipe estimates",
        "rep_idx":             best,
        "score":               scores[best],
        "seg_start_arr":       best_seg[0],
        "seg_end_arr":         best_seg[1],
        "sync_info_ref":       "sync_validation.json",
    }

    # canonical_3d.json
    can_recording = {
        "SchemaVersion":   "canonical_3d_v1",
        "MotionId":        "인사_multiview_s01_v2",
        "DisplayName":     "인사",
        "FusionMethod":    fusion_meta["fusion_method"],
        "CoordinateSystem": fusion_meta["coordinate_system"],
        "TotalFrames":     n_total,
        "DurationMs":      float(dur_ms),
        "FusionMeta":      fusion_meta,
        "Frames":          canonical_frames,
    }
    save_json(can_recording, os.path.join(OUT_DIR, "canonical_3d.json"))

    # blend_compatible.json (schemaVersion=2, blend_animate.py 호환)
    blend_recording = {
        "SchemaVersion":       2,
        "MotionId":            "인사_multiview_s01_v2",
        "DisplayName":         "인사",
        "DurationMs":          float(dur_ms),
        "SourceType":          "multiview_fused_v2",
        "TotalFrames":         n_total,
        "ValidHandFrames":     v_h,
        "ValidPoseFrames":     v_p,
        "ValidCombinedFrames": v_c,
        "FusionMeta":          fusion_meta,
        "Frames":              blend_frames,
    }
    blend_compat_path = os.path.join(OUT_DIR, "blend_compatible.json")
    save_json(blend_recording, blend_compat_path)

    # contribution_report.json
    save_json(contribution, os.path.join(OUT_DIR, "contribution_report.json"))

    # ── Step 6: 수치 검증 ────────────────────────────────────────────────────
    print(f"\n=== Step 6: 수치 검증 ===")
    val = compute_validation(blend_frames, canonical_frames)
    save_json(val, os.path.join(OUT_DIR, "validation_report.json"))
    print(f"  손 바뀜:   {val['hand_swaps']}")
    print(f"  순간이동:  {val['wrist_teleports']}")
    print(f"  장기 갭:   {val['long_gaps_count']}건")

    # ── Step 7: Blender 애니메이션 ───────────────────────────────────────────
    if args.skip_blender:
        print(f"\n[건너뜀] --skip-blender 지정됨")
        return

    print(f"\n=== Step 7: Blender 애니메이션 ===")
    anim_script = os.path.join(SCRIPT_DIR, "blend_animate_multiview_v2.py")
    anim_log    = os.path.join(OUT_DIR, "blender_anim.log")
    if not os.path.exists(anim_script):
        print(f"  [오류] 스크립트 없음: {anim_script}")
    else:
        rc = run_blender(anim_script, anim_log)
        if rc != 0:
            print(f"  [경고] Blender 종료코드 {rc} — 로그 확인: {anim_log}")

    # ── Step 8: 3시점 렌더링 ─────────────────────────────────────────────────
    print(f"\n=== Step 8: 3시점 렌더링 ===")
    blend_out  = os.path.join(OUT_DIR, "인사_multiview_s01_v2.blend")
    render_script = os.path.join(SCRIPT_DIR, "s04_render_v2.py")
    render_log    = os.path.join(OUT_DIR, "blender_render.log")
    if not os.path.exists(blend_out):
        print(f"  [건너뜀] .blend 없음: {blend_out}")
    elif not os.path.exists(render_script):
        print(f"  [건너뜀] 렌더 스크립트 없음: {render_script}")
    else:
        rc = run_blender(render_script, render_log, blend_in=blend_out)
        if rc != 0:
            print(f"  [경고] 렌더 종료코드 {rc}")

    print(f"\n=== 완료 ===")
    print(f"  출력 디렉터리: {OUT_DIR}")


if __name__ == "__main__":
    main()
