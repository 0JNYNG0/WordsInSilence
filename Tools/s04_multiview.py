"""
Tools/s04_multiview.py — S01 '인사' 3-뷰 랜드마크 추출 · 동기화 · 분절 · 융합

실행:
  python Tools/s04_multiview.py                     # 전체 실행
  python Tools/s04_multiview.py --skip-extract      # 캐시된 raw JSON 재사용
  python Tools/s04_multiview.py --subsample 2       # 15fps 추출 (속도 우선)
  python Tools/s04_multiview.py --rep-idx 3         # rep 3 강제 선택

출력 (Tools/s04_out/):
  sync_offsets.json       — 각 뷰의 clap 프레임 (동기화 기준)
  raw_front.json          — front 뷰 전체 랜드마크 (캐시)
  raw_left45.json         — left45 뷰 전체 랜드마크 (캐시)
  raw_right45.json        — right45 뷰 전체 랜드마크 (캐시)
  boundaries.csv          — 분절 결과 (rep별 start/end/score)
  multiview_fused.json    — 선택된 rep 융합 (HandPoseRecording schemaVersion=2)
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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "models")
OUT_DIR     = os.path.join(SCRIPT_DIR, "s04_out")

HAND_MODEL = os.path.join(MODEL_DIR, "hand_landmarker.task")
POSE_MODEL = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")

VIDEOS = {
    "front":   os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_front.mp4"),
    "left45":  os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_left45.mp4"),
    "right45": os.path.join(DATASET_DIR, "SIGN002_GREETING_P01_S01_B01_right45.mp4"),
}

# ─── 세그멘테이션 파라미터 ────────────────────────────────────────────────────

CLAP_SAMPLE_EVERY = 5     # clap 탐지 서브샘플 (매 N번째 프레임만 처리)
CLAP_SEARCH_SEC   = 30    # 영상 앞부분에서 clap 탐색 범위 (초)
EXPECTED_REPS     = 12    # 예상 반복 횟수 (불일치 시 경고)
SMOOTH_WINDOW     = 60    # nose Y 스무딩 윈도우 (프레임) — bow 주기(~2s@30fps=60f) 기준
PEAK_MIN_DIST     = 40    # 연속 피크 최소 간격 (프레임) — 최소 ~1.3s 간격
PEAK_MIN_PROM     = 0.003 # 피크 최소 두드러짐 (normalized Y)

# ─── Pose landmark 인덱스 (BlazePose 33점) ───────────────────────────────────

NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16
LEFT_HIP       = 23
RIGHT_HIP      = 24

MIN_SHOULDER_WIDTH = 0.05

# ─── Quality enum (HandPoseFrame.cs 동일) ────────────────────────────────────

QUALITY_GOOD              = 0
QUALITY_HAND_ONLY         = 1
QUALITY_POSE_ONLY         = 2
QUALITY_INSUFFICIENT_POSE = 3
QUALITY_INVALID           = 6

# ─── 수학 유틸 ───────────────────────────────────────────────────────────────

def dist2d(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

def lm_to_dict(lm):
    return {"x": lm.x, "y": lm.y, "z": lm.z}

def body_relative(px, py, cx, cy, sw):
    if sw <= MIN_SHOULDER_WIDTH or math.isnan(sw):
        return {"x": float("nan"), "y": float("nan")}
    return {"x": (px - cx) / sw, "y": (py - cy) / sw}

def smooth_boxcar(data, window):
    """박스 필터 스무딩. NaN은 0으로 대체."""
    cleaned = [0.0 if math.isnan(v) else v for v in data]
    result  = []
    half    = window // 2
    n       = len(cleaned)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result.append(sum(cleaned[lo:hi]) / (hi - lo))
    return result

def find_peaks(data, min_dist=5, min_prom=0.0):
    """로컬 최대값 인덱스. min_dist·min_prom 필터 적용."""
    n      = len(data)
    peaks  = [i for i in range(1, n - 1) if data[i] >= data[i-1] and data[i] >= data[i+1]]

    # min_prom: 전체 범위 대비 두드러짐
    baseline = min(data) if data else 0.0
    if min_prom > 0:
        peaks = [p for p in peaks if data[p] - baseline >= min_prom]

    # min_dist: 가까운 피크 중 더 높은 것만 유지
    filtered = []
    last = -min_dist - 1
    for p in peaks:
        if p - last >= min_dist:
            filtered.append(p)
            last = p
    return filtered

def vis_sum(vis_list):
    """visibility 합. None 또는 None 요소 안전 처리."""
    if vis_list is None:
        return 0.0
    return sum(v for v in vis_list if v is not None)

# ─── 1. Clap 탐지 (IMAGE 모드) ───────────────────────────────────────────────

def detect_clap_frame(view_name, video_path,
                      sample_every=CLAP_SAMPLE_EVERY,
                      search_sec=CLAP_SEARCH_SEC):
    """
    두 손목 사이 거리가 최솟값인 프레임 인덱스(raw)를 반환.
    CLAP_SEARCH_SEC 초 이내의 앞부분에서만 탐색.
    """
    print(f"[clap] {view_name}: 탐지 중 (처음 {search_sec}s, 매 {sample_every}프레임)...")
    cap       = cv2.VideoCapture(video_path)
    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frame = int(search_sec * fps)

    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )

    min_dist   = float("inf")
    best_frame = 0
    frame_idx  = 0

    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm:
        while frame_idx < max_frame:
            ret, bgr = cap.read()
            if not ret:
                break
            if frame_idx % sample_every == 0:
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hand_lm.detect(mp_img)
                if result and result.hand_landmarks and len(result.hand_landmarks) >= 2:
                    w0 = result.hand_landmarks[0][0]
                    w1 = result.hand_landmarks[1][0]
                    d  = dist2d(w0.x, w0.y, w1.x, w1.y)
                    if d < min_dist:
                        min_dist   = d
                        best_frame = frame_idx
            frame_idx += 1

    cap.release()
    t = best_frame / fps
    print(f"  -> clap_frame={best_frame}  dist={min_dist:.4f}  t={t:.2f}s")
    return best_frame

# ─── 2. 랜드마크 추출 (VIDEO 모드) ───────────────────────────────────────────

def extract_view(view_name, video_path, subsample=1):
    """
    VIDEO 모드 전체 랜드마크 추출. visibility 포함.
    반환: {"view", "fps", "original_fps", "subsample", "frames": [...]}
    """
    print(f"[extract] {view_name}: 시작 (subsample={subsample})...")
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
    processed = 0

    with mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_lm, \
         mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:

        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % subsample == 0:
                ts_ms  = int(frame_idx * 1000 / fps)
                rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hr     = hand_lm.detect_for_video(mp_img, ts_ms)
                pr     = pose_lm.detect_for_video(mp_img, ts_ms)
                frames.append(_make_frame(frame_idx, ts_ms, hr, pr))
                processed += 1
                if processed % 60 == 0:
                    pct = 100 * frame_idx // total
                    print(f"  {view_name}: {frame_idx}/{total} ({pct}%)", end="\r")

            frame_idx += 1

    cap.release()
    print(f"\n  {view_name}: {processed}프레임 추출 완료")
    valid_hand = sum(1 for f in frames if f["HasLeftHand"] or f["HasRightHand"])
    valid_pose = sum(1 for f in frames if f["HasPose"])
    print(f"  Hand:{valid_hand}  Pose:{valid_pose}  Total:{processed}")

    return {
        "view":         view_name,
        "fps":          fps / subsample,
        "original_fps": fps,
        "subsample":    subsample,
        "frames":       frames,
    }


def _make_frame(frame_idx, ts_ms, hand_result, pose_result):
    """MediaPipe 결과 → 프레임 딕셔너리 (visibility 포함)."""
    # ── Hand ────────────────────────────────────────────────────────────────
    has_left = has_right = False
    left_pts = right_pts = None
    left_vis = right_vis = None

    if hand_result and hand_result.hand_landmarks:
        for i, lms in enumerate(hand_result.hand_landmarks):
            cat = hand_result.handedness[i][0].category_name
            pts = [lm_to_dict(lm) for lm in lms]
            vis = [getattr(lm, "visibility", 1.0) for lm in lms]
            if cat == "Left":
                has_left = True; left_pts = pts; left_vis = vis
            else:
                has_right = True; right_pts = pts; right_vis = vis

    # ── Pose ────────────────────────────────────────────────────────────────
    has_pose = has_valid_shoulders = False
    pose_pts = pose_vis = None
    sh_cx = sh_cy = sw = 0.0
    l_wr_rel = r_wr_rel = {"x": float("nan"), "y": float("nan")}
    nose_y   = float("nan")

    if pose_result and pose_result.pose_landmarks:
        lms      = pose_result.pose_landmarks[0]
        has_pose = True
        pose_pts = [lm_to_dict(lm) for lm in lms]
        pose_vis = [getattr(lm, "visibility", 1.0) for lm in lms]

        l_sh = lms[LEFT_SHOULDER]; r_sh = lms[RIGHT_SHOULDER]
        sh_cx = (l_sh.x + r_sh.x) * 0.5
        sh_cy = (l_sh.y + r_sh.y) * 0.5
        sw    = dist2d(l_sh.x, l_sh.y, r_sh.x, r_sh.y)
        has_valid_shoulders = sw > MIN_SHOULDER_WIDTH

        l_wr_rel = body_relative(lms[LEFT_WRIST].x,  lms[LEFT_WRIST].y,  sh_cx, sh_cy, sw)
        r_wr_rel = body_relative(lms[RIGHT_WRIST].x, lms[RIGHT_WRIST].y, sh_cx, sh_cy, sw)
        nose_y   = lms[NOSE].y

    # ── Quality ─────────────────────────────────────────────────────────────
    has_hand = has_left or has_right
    if   has_hand and has_pose and has_valid_shoulders: quality = QUALITY_GOOD
    elif has_hand and has_pose:                          quality = QUALITY_INSUFFICIENT_POSE
    elif has_hand:                                       quality = QUALITY_HAND_ONLY
    elif has_pose:                                       quality = QUALITY_POSE_ONLY
    else:                                                quality = QUALITY_INVALID

    return {
        "FrameIdx":              frame_idx,
        "TimestampMs":           ts_ms,
        "HasLeftHand":           has_left,
        "HasRightHand":          has_right,
        "HasPose":               has_pose,
        "HasValidShoulders":     has_valid_shoulders,
        "LeftHandPoints":        left_pts,
        "RightHandPoints":       right_pts,
        "LeftHandVis":           left_vis,
        "RightHandVis":          right_vis,
        "PosePoints":            pose_pts,
        "PoseVis":               pose_vis,
        "ShoulderCenter":        {"x": sh_cx, "y": sh_cy},
        "ShoulderWidth":         sw,
        "LeftWristBodyRelative": l_wr_rel,
        "RightWristBodyRelative":r_wr_rel,
        "NoseY":                 nose_y,
        "Quality":               quality,
    }

# ─── 3. 동기화 헬퍼 ──────────────────────────────────────────────────────────

def to_view_arr_idx(view_name, front_raw_idx, clap_frames, views_data):
    """
    front 뷰 raw frame index → 해당 뷰의 frames[] 배열 인덱스.

    동일 시각 t_from_clap(초) 기준:
      front_raw_idx → (front_raw_idx - front_clap) / front_fps = t_offset_sec
      view_raw_idx  = view_clap + t_offset_sec * view_fps
      view_arr_idx  = round(view_raw_idx / view_subsample)
    """
    front_fps  = views_data["front"]["original_fps"]
    front_clap = clap_frames["front"]
    view_fps   = views_data[view_name]["original_fps"]
    view_clap  = clap_frames[view_name]
    view_sub   = views_data[view_name]["subsample"]

    t_offset     = (front_raw_idx - front_clap) / front_fps
    view_raw_idx = view_clap + t_offset * view_fps
    arr_idx      = round(view_raw_idx / view_sub)

    frames = views_data[view_name]["frames"]
    return max(0, min(len(frames) - 1, arr_idx))

# ─── 4. 세그멘테이션 ─────────────────────────────────────────────────────────

def segment_reps(front_frames, clap_raw_frame):
    """
    front 뷰 nose Y 피크 탐지 → 반복 구간 경계 추출.
    clap_raw_frame 이후의 데이터만 사용.

    반환: [(start_arr, end_arr), ...]  (front frames[] 배열 인덱스)
    """
    # clap 이후 frames 배열 시작 인덱스
    clap_arr = 0
    for i, f in enumerate(front_frames):
        if f["FrameIdx"] >= clap_raw_frame:
            clap_arr = i
            break

    nose_y_raw = [f["NoseY"] for f in front_frames]
    nose_y_sm  = smooth_boxcar(nose_y_raw, SMOOTH_WINDOW)

    # clap 이후 smoothed nose Y
    post_clap  = nose_y_sm[clap_arr:]

    peaks_local = find_peaks(post_clap, min_dist=PEAK_MIN_DIST, min_prom=PEAK_MIN_PROM)
    peaks_abs   = [p + clap_arr for p in peaks_local]

    n_peaks = len(peaks_abs)
    print(f"[segment] 탐지 피크: {n_peaks}  (예상: {EXPECTED_REPS})")
    if n_peaks != EXPECTED_REPS:
        print(f"  [경고] 피크 수 불일치 (실제={n_peaks}, 예상={EXPECTED_REPS})")
        print("  --rep-idx 로 수동 선택하거나 PEAK_MIN_DIST/PEAK_MIN_PROM 조정 필요.")

    if n_peaks < 1:
        print("  [오류] 피크 없음 — 세그멘테이션 실패.")
        return []

    # 구간: 이전 피크 중간 ~ 다음 피크 중간
    n_total = len(front_frames)
    segs = []
    for i, pk in enumerate(peaks_abs):
        if i == 0:
            prev_mid = clap_arr
        else:
            prev_mid = (peaks_abs[i - 1] + pk) // 2
        if i == n_peaks - 1:
            next_mid = min(pk + (n_total - pk) // 2, n_total - 1)
        else:
            next_mid = (pk + peaks_abs[i + 1]) // 2
        segs.append((prev_mid, next_mid))

    return segs

# ─── 5. 반복 구간 점수화 ─────────────────────────────────────────────────────

def score_rep(seg, front_frames, views_data, clap_frames):
    """
    멀티뷰 통합 hand ratio + front pose ratio + nose Y 안정성 → [0, 1] 점수.
    """
    start, end = seg
    count = end - start + 1
    if count <= 0:
        return 0.0

    left_data  = views_data["left45"]["frames"]
    right_data = views_data["right45"]["frames"]
    hand_ok = pose_ok = 0

    for fi in range(start, end + 1):
        ff = front_frames[fi]
        li = to_view_arr_idx("left45",  ff["FrameIdx"], clap_frames, views_data)
        ri = to_view_arr_idx("right45", ff["FrameIdx"], clap_frames, views_data)
        lf = left_data[li]; rf = right_data[ri]

        any_hand = (ff["HasLeftHand"] or ff["HasRightHand"] or
                    lf["HasLeftHand"] or lf["HasRightHand"] or
                    rf["HasLeftHand"] or rf["HasRightHand"])
        if any_hand:  hand_ok += 1
        if ff["HasPose"]: pose_ok += 1

    hand_ratio = hand_ok / count
    pose_ratio = pose_ok / count

    nose_ys = [front_frames[i]["NoseY"] for i in range(start, end + 1)
               if not math.isnan(front_frames[i]["NoseY"])]
    if len(nose_ys) > 1:
        mean = sum(nose_ys) / len(nose_ys)
        std  = math.sqrt(sum((v - mean) ** 2 for v in nose_ys) / len(nose_ys))
        # 활발한 움직임 = std 클수록 좋음 (bow 운동 포함)
        motion = min(std * 20, 1.0)
    else:
        motion = 0.0

    return 0.50 * hand_ratio + 0.30 * pose_ratio + 0.20 * motion

# ─── 6. 멀티뷰 융합 ──────────────────────────────────────────────────────────

def fuse_rep(seg, front_frames, views_data, clap_frames):
    """
    선택된 rep 구간을 3-뷰 융합하여 HandPoseRecording 프레임 리스트 반환.

    전략:
    - Pose (어깨·자세): front 뷰 전용 (정면 뷰가 가장 정확)
    - 손 포인트(21점): 3뷰 중 visibility 합이 가장 높은 뷰 채택
    - WristBodyRelative: front 포즈 손목 우선; front 없으면 hand pt0 변환
    """
    start, end = seg
    left_data  = views_data["left45"]["frames"]
    right_data = views_data["right45"]["frames"]
    result_frames = []
    t0_ms = None

    for fi in range(start, end + 1):
        ff = front_frames[fi]
        li = to_view_arr_idx("left45",  ff["FrameIdx"], clap_frames, views_data)
        ri = to_view_arr_idx("right45", ff["FrameIdx"], clap_frames, views_data)
        lf = left_data[li]
        rf = right_data[ri]

        ts = ff["TimestampMs"]
        if t0_ms is None:
            t0_ms = ts
        ts_rel = ts - t0_ms

        # ── Pose: front 우선 ────────────────────────────────────────────────
        has_pose            = ff["HasPose"]
        has_valid_shoulders = ff["HasValidShoulders"]
        pose_pts            = ff["PosePoints"]
        sh_ctr              = ff["ShoulderCenter"]
        sh_width            = ff["ShoulderWidth"]

        # ── 손 포인트: 가장 visibility 합 높은 뷰 ──────────────────────────
        def best_hand(side):
            has_k = f"Has{side}Hand"
            pts_k = f"{side}HandPoints"
            vis_k = f"{side}HandVis"
            candidates = []
            for fdata in (ff, lf, rf):
                if fdata[has_k] and fdata[pts_k]:
                    s = vis_sum(fdata[vis_k])
                    candidates.append((s, fdata[pts_k], fdata[vis_k]))
            if not candidates:
                return False, None, None
            candidates.sort(key=lambda x: x[0], reverse=True)
            return True, candidates[0][1], candidates[0][2]

        has_left,  left_pts,  left_vis  = best_hand("Left")
        has_right, right_pts, right_vis = best_hand("Right")

        # ── WristBodyRelative: front 포즈 손목 우선 ────────────────────────
        #   front 포즈 없고 손은 있으면 hand pt0 사용 (어깨 기준 변환 가능할 때만)
        def wrist_rel(side, has_hand_side, hand_pts):
            rel = ff[f"{side}WristBodyRelative"]
            if not math.isnan(rel["x"]):
                return rel
            # front 포즈 손목이 없을 때: hand pt0 → body-relative (front 어깨 기준)
            if has_hand_side and hand_pts and has_valid_shoulders:
                pt0  = hand_pts[0]
                return body_relative(pt0["x"], pt0["y"],
                                     sh_ctr["x"], sh_ctr["y"], sh_width)
            return {"x": float("nan"), "y": float("nan")}

        l_wr_rel = wrist_rel("Left",  has_left,  left_pts)
        r_wr_rel = wrist_rel("Right", has_right, right_pts)

        # ── Quality ─────────────────────────────────────────────────────────
        has_hand = has_left or has_right
        if   has_hand and has_pose and has_valid_shoulders: quality = QUALITY_GOOD
        elif has_hand and has_pose:                          quality = QUALITY_INSUFFICIENT_POSE
        elif has_hand:                                       quality = QUALITY_HAND_ONLY
        elif has_pose:                                       quality = QUALITY_POSE_ONLY
        else:                                                quality = QUALITY_INVALID

        result_frames.append({
            "TimestampMs":            ts_rel,
            "HasLeftHand":            has_left,
            "HasRightHand":           has_right,
            "HasPose":                has_pose,
            "HasValidShoulders":      has_valid_shoulders,
            "LeftHandPoints":         left_pts,
            "RightHandPoints":        right_pts,
            "PosePoints":             pose_pts,
            "ShoulderCenter":         sh_ctr,
            "ShoulderWidth":          sh_width,
            "LeftWristBodyRelative":  l_wr_rel,
            "RightWristBodyRelative": r_wr_rel,
            "Quality":                quality,
        })

    return result_frames

# ─── 저장 유틸 ───────────────────────────────────────────────────────────────

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=True)
    sz = os.path.getsize(path)
    print(f"[저장] {os.path.relpath(path)} ({sz//1024}KB)")

def save_boundaries_csv(segs, scores, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("rep_idx,start_arr,end_arr,frame_count,score\n")
        for i, (seg, sc) in enumerate(zip(segs, scores)):
            s, e = seg
            f.write(f"{i},{s},{e},{e-s+1},{sc:.4f}\n")
    print(f"[저장] {os.path.relpath(path)}")

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="S01 인사 멀티뷰 파이프라인")
    ap.add_argument("--skip-extract", action="store_true",
                    help="랜드마크 추출 건너뜀 (캐시된 raw JSON 재사용)")
    ap.add_argument("--subsample", type=int, default=1,
                    help="추출 서브샘플 (1=전체, 2=15fps, 3=10fps)")
    ap.add_argument("--rep-idx", type=int, default=-1,
                    help="반복 구간 수동 선택 (-1=자동)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 모델·비디오 확인 ────────────────────────────────────────────────────
    for p in (HAND_MODEL, POSE_MODEL):
        if not os.path.exists(p):
            sys.exit(f"[오류] 모델 없음: {p}")
    for vn, vp in VIDEOS.items():
        if not os.path.exists(vp):
            sys.exit(f"[오류] 비디오 없음: {vp}")
        print(f"[확인] {vn}: {os.path.basename(vp)}")

    # ── Step 1: Clap 탐지 ───────────────────────────────────────────────────
    sync_path = os.path.join(OUT_DIR, "sync_offsets.json")
    if os.path.exists(sync_path):
        print(f"\n[캐시] {os.path.relpath(sync_path)}")
        with open(sync_path) as f:
            clap_frames = json.load(f)
        print(f"  clap frames: {clap_frames}")
    else:
        print("\n=== Step 1: Clap 탐지 ===")
        clap_frames = {vn: detect_clap_frame(vn, vp) for vn, vp in VIDEOS.items()}
        save_json(clap_frames, sync_path)

    # ── Step 2: 랜드마크 추출 ────────────────────────────────────────────────
    raw_paths = {vn: os.path.join(OUT_DIR, f"raw_{vn}.json") for vn in VIDEOS}
    views_data = {}

    print(f"\n=== Step 2: 랜드마크 추출 (subsample={args.subsample}) ===")
    for vn, vp in VIDEOS.items():
        rp = raw_paths[vn]
        if args.skip_extract and os.path.exists(rp):
            print(f"[캐시] {os.path.relpath(rp)} 로드...")
            with open(rp) as f:
                views_data[vn] = json.load(f)
            print(f"  {vn}: {len(views_data[vn]['frames'])}프레임 로드")
        else:
            vdata = extract_view(vn, vp, subsample=args.subsample)
            views_data[vn] = vdata
            save_json(vdata, rp)

    # ── Step 3: 세그멘테이션 ─────────────────────────────────────────────────
    print("\n=== Step 3: 세그멘테이션 ===")
    front_frames = views_data["front"]["frames"]
    front_clap   = clap_frames["front"]
    segs = segment_reps(front_frames, front_clap)

    if not segs:
        sys.exit("[오류] 반복 구간 탐지 실패. --skip-extract 후 PEAK 파라미터 조정 필요.")

    # ── Step 4: 점수화 + 선택 ────────────────────────────────────────────────
    print("\n=== Step 4: 반복 구간 점수화 ===")
    scores = [score_rep(s, front_frames, views_data, clap_frames) for s in segs]

    boundaries_path = os.path.join(OUT_DIR, "boundaries.csv")
    save_boundaries_csv(segs, scores, boundaries_path)

    for i, (seg, sc) in enumerate(zip(segs, scores)):
        print(f"  rep[{i:2d}]: frames [{seg[0]:4d},{seg[1]:4d}] len={seg[1]-seg[0]+1:3d} score={sc:.4f}")

    if 0 <= args.rep_idx < len(segs):
        best = args.rep_idx
        print(f"\n[선택] rep_idx={best} (수동 지정)")
    else:
        best = max(range(len(scores)), key=lambda i: scores[i])
        print(f"\n[선택] rep_idx={best} (자동, score={scores[best]:.4f})")

    best_seg = segs[best]
    print(f"  구간: front arr [{best_seg[0]}, {best_seg[1]}]  ({best_seg[1]-best_seg[0]+1}프레임)")

    # ── Step 5: 멀티뷰 융합 ──────────────────────────────────────────────────
    print("\n=== Step 5: 멀티뷰 융합 ===")
    fused_frames = fuse_rep(best_seg, front_frames, views_data, clap_frames)

    total     = len(fused_frames)
    v_hand    = sum(1 for f in fused_frames if f["HasLeftHand"] or f["HasRightHand"])
    v_pose    = sum(1 for f in fused_frames if f["HasPose"])
    v_comb    = sum(1 for f in fused_frames if (f["HasLeftHand"] or f["HasRightHand"]) and f["HasPose"])
    dur_ms    = fused_frames[-1]["TimestampMs"] if fused_frames else 0

    print(f"  프레임: {total}  손:{v_hand}({v_hand*100//total}%)  포즈:{v_pose}({v_pose*100//total}%)  결합:{v_comb}({v_comb*100//total}%)")
    print(f"  길이: {dur_ms}ms ({dur_ms/1000:.1f}s)")

    recording = {
        "SchemaVersion":       2,
        "MotionId":            "인사_multiview_s01",
        "DisplayName":         "인사",
        "DurationMs":          float(dur_ms),
        "SourceType":          "multiview_fused",
        "TotalFrames":         total,
        "ValidHandFrames":     v_hand,
        "ValidPoseFrames":     v_pose,
        "ValidCombinedFrames": v_comb,
        "FusionMeta": {
            "views":         list(VIDEOS.keys()),
            "rep_idx":       best,
            "score":         scores[best],
            "seg_start_arr": best_seg[0],
            "seg_end_arr":   best_seg[1],
            "clap_frames":   clap_frames,
        },
        "Frames": fused_frames,
    }

    fused_path = os.path.join(OUT_DIR, "multiview_fused.json")
    save_json(recording, fused_path)

    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
    blend_file  = r"C:\Users\local-user\OneDrive\바탕 화면\test_mixamo_girl.blend"
    anim_script = os.path.join(SCRIPT_DIR, "blend_animate_multiview.py")

    print(f"\n=== 완료 ===")
    print(f"  multiview_fused.json: {fused_path}")
    print(f"\n다음 단계 — Blender 애니메이션 생성:")
    print(f'  "{blender_exe}" "{blend_file}" --background --python "{anim_script}"')


if __name__ == "__main__":
    main()
