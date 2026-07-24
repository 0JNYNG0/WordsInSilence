#!/usr/bin/env python3
"""
Tools/extract_features.py
--------------------------
TrainingSamples/ 폴더의 레이블된 JSON 파일에서 HandPoseEvaluator 피처를 추출하여 CSV로 저장한다.
추출된 CSV는 train_svm.py 로 학습에 사용한다.

JSON 파일의 "DisplayName" 필드가 레이블로 사용된다.
파일이 Unity Unity persistentDataPath/TrainingSamples/ 에 저장된 형식과 동일.

사용법:
    python extract_features.py --samples-dir "C:\\Users\\local-user\\TrainingSamples"
    python extract_features.py --samples-dir "C:\\Users\\local-user\\TrainingSamples" --recordings-dir "C:\\Users\\local-user\\Recordings"
    python extract_features.py --samples-dir /path/to/samples --output features.csv

--recordings-dir 옵션으로 Recordings/ 폴더(전문가 영상 데이터)도 함께 학습에 포함할 수 있다.
Recordings/ 의 WristBodyRelative 좌표는 Unity 와 x, y 부호가 반전되어 있어 자동으로 보정된다.
_slow 파일은 인위적 속도 조정 데이터이므로 자동 제외한다.

필요 패키지: 없음 (표준 라이브러리만 사용)
"""

import argparse
import csv
import json
import math
import os
import sys

NAN = float("nan")

FEATURE_NAMES = [
    "ApertureTrend",
    "WristTrajDx", "WristTrajDy",
    "OscCountX", "OscCountY",
    "PathClosure", "WristZAmp",
    "WristCenterY", "WristYOscillation",
    "FaceDistNose", "FaceDistEar", "FaceDistMouth",
    "IndexDirX", "IndexDirY", "IndexDirZ",
    "WristCenterX", "TwoHandDist", "TwoHandCorr",
    "PalmNormalX", "PalmNormalY", "PalmNormalZ",
    "FingerExtThumb", "FingerExtIndex", "FingerExtMiddle",
    "FingerExtRing", "FingerExtPinky",
]

# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def _is_nan(v):
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True

def _vec2_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def _vec3_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def _normalize3(v):
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 1e-8:
        return None
    return (v[0]/mag, v[1]/mag, v[2]/mag)

def _get_vec2(d, key):
    """dict에서 {x, y} 꺼내기. 없으면 (nan, nan)."""
    v = d.get(key) if isinstance(d, dict) else None
    if not isinstance(v, dict):
        return NAN, NAN
    x = v.get("x", NAN)
    y = v.get("y", NAN)
    return (NAN if x is None else x), (NAN if y is None else y)

def _wrist_xy(frame):
    """주 손목 body-relative (x, y). RightWrist 우선, 없으면 LeftWrist."""
    rx, ry = _get_vec2(frame, "RightWristBodyRelative")
    if not _is_nan(rx):
        return rx, ry
    return _get_vec2(frame, "LeftWristBodyRelative")

def _get_hand_pts(frame):
    """오른손 우선, 없으면 왼손. 21개 (x,y,z) 튜플 리스트 또는 None."""
    def _extract(key):
        pts = frame.get(key)
        if isinstance(pts, list) and len(pts) >= 21:
            result = []
            for p in pts[:21]:
                if isinstance(p, dict):
                    result.append((p.get("x", NAN), p.get("y", NAN), p.get("z", NAN)))
                else:
                    return None
            return result
        return None

    if frame.get("HasRightHand"):
        pts = _extract("RightHandPoints")
        if pts:
            return pts
    if frame.get("HasLeftHand"):
        return _extract("LeftHandPoints")
    return None

def _get_pose_pts(frame):
    """PosePoints 리스트. 없으면 None."""
    pp = frame.get("PosePoints")
    if not isinstance(pp, list) or len(pp) < 11:
        return None
    return pp

def _smoothed(vals, win=2):
    out = []
    n = len(vals)
    for i in range(n):
        s = max(0, i - win)
        e = min(n - 1, i + win)
        out.append(sum(vals[s:e+1]) / (e - s + 1))
    return out

# ─── 비활성 구간 제거 ──────────────────────────────────────────────────────────

def _trim_inactive(frames):
    has_hand = lambda f: f.get("HasLeftHand") or f.get("HasRightHand")
    start = next((i for i, f in enumerate(frames) if has_hand(f)), 0)
    end   = next((len(frames)-1-i for i, f in enumerate(reversed(frames)) if has_hand(f)), len(frames)-1)
    return frames[start:end+1] if start <= end else frames

# ─── 개별 피처 함수 ────────────────────────────────────────────────────────────

def feat_aperture_trend(frames):
    apts = []
    for f in frames:
        pts = _get_hand_pts(f)
        if not pts:
            continue
        tips = [4, 8, 12, 16, 20]
        apt = sum(math.sqrt((pts[0][0]-pts[t][0])**2 + (pts[0][1]-pts[t][1])**2) for t in tips) / 5.0
        apts.append(apt)
    if len(apts) < 4:
        return NAN
    mid    = len(apts) // 2
    first  = sum(apts[:mid]) / max(1, mid)
    second = sum(apts[mid:]) / max(1, len(apts) - mid)
    mn, mx = min(apts), max(apts)
    r = mx - mn
    if r < 1e-4:
        return 0.0
    return max(-1.0, min(1.0, (second - first) / r))


def feat_wrist_traj(frames):
    xs, ys = [], []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        wx, wy = _wrist_xy(f)
        if _is_nan(wx):
            continue
        xs.append(wx)
        ys.append(wy)
    if len(xs) < 4:
        return NAN, NAN
    return max(xs)-min(xs), max(ys)-min(ys)


def feat_osc_x(frames):
    xs = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        wx, _ = _wrist_xy(f)
        if _is_nan(wx):
            continue
        xs.append(wx)
    if len(xs) < 8:
        return NAN
    sm = _smoothed(xs)
    MIN_CH = 0.04
    rev, d = 0, 0
    for i in range(1, len(sm)):
        nd = 1 if sm[i]-sm[i-1] > MIN_CH else (-1 if sm[i]-sm[i-1] < -MIN_CH else 0)
        if nd != 0 and nd != d:
            if d != 0:
                rev += 1
            d = nd
    return float(rev)


def feat_osc_y(frames):
    ys = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        _, wy = _wrist_xy(f)
        if _is_nan(wy):
            continue
        ys.append(wy)
    if len(ys) < 8:
        return NAN
    sm = _smoothed(ys)
    MIN_CH = 0.04
    rev, d = 0, 0
    for i in range(1, len(sm)):
        nd = 1 if sm[i]-sm[i-1] > MIN_CH else (-1 if sm[i]-sm[i-1] < -MIN_CH else 0)
        if nd != 0 and nd != d:
            if d != 0:
                rev += 1
            d = nd
    return float(rev)


def feat_path_closure(frames):
    pos = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        wx, wy = _wrist_xy(f)
        if _is_nan(wx):
            continue
        pos.append((wx, wy))
    if len(pos) < 8:
        return NAN
    path_len = sum(_vec2_dist(pos[i], pos[i-1]) for i in range(1, len(pos)))
    if path_len < 0.01:
        return 0.0
    return min(1.0, _vec2_dist(pos[0], pos[-1]) / path_len)


def feat_wrist_z_amp(frames):
    RIGHT_WR, LEFT_WR = 16, 15
    zs = []
    for f in frames:
        pp = _get_pose_pts(f)
        if not pp:
            continue
        if RIGHT_WR < len(pp) and isinstance(pp[RIGHT_WR], dict):
            z = pp[RIGHT_WR].get("z", NAN)
        else:
            z = NAN
        if _is_nan(z) and LEFT_WR < len(pp) and isinstance(pp[LEFT_WR], dict):
            z = pp[LEFT_WR].get("z", NAN)
        if _is_nan(z):
            continue
        zs.append(z)
    if len(zs) < 8:
        return NAN
    return max(zs) - min(zs)


def feat_wrist_center_y(frames):
    ys = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        _, wy = _wrist_xy(f)
        if _is_nan(wy):
            continue
        ys.append(wy)
    if len(ys) < 4:
        return NAN
    return sum(ys) / len(ys)


def feat_wrist_y_std(frames):
    ys = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        _, wy = _wrist_xy(f)
        if _is_nan(wy):
            continue
        ys.append(wy)
    if len(ys) < 8:
        return NAN
    mean = sum(ys) / len(ys)
    var  = sum((y-mean)**2 for y in ys) / len(ys)
    return math.sqrt(var)


def feat_face_dists(frames):
    nose_s = ear_s = mouth_s = 0.0
    count = 0
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        wx, wy = _wrist_xy(f)
        if _is_nan(wx):
            continue

        pp = _get_pose_pts(f)
        sc = f.get("ShoulderCenter", {})
        sw = f.get("ShoulderWidth", 0.0)
        if not pp or not isinstance(sc, dict) or sw < 1e-4:
            continue

        cx = sc.get("x", NAN)
        cy = sc.get("y", NAN)
        if _is_nan(cx):
            continue

        def body_rel(idx):
            if idx >= len(pp) or not isinstance(pp[idx], dict):
                return NAN, NAN
            px = pp[idx].get("x", NAN)
            py = pp[idx].get("y", NAN)
            if _is_nan(px):
                return NAN, NAN
            return (px - cx) / sw, (py - cy) / sw

        nose  = body_rel(0)
        lear  = body_rel(7)
        rear  = body_rel(8)
        ml    = body_rel(9)
        mr    = body_rel(10)

        if _is_nan(nose[0]) or _is_nan(lear[0]) or _is_nan(ml[0]):
            continue

        wrist = (wx, wy)
        d_nose  = _vec2_dist(wrist, nose)
        d_ear   = min(_vec2_dist(wrist, lear), _vec2_dist(wrist, rear))
        d_mouth = (_vec2_dist(wrist, ml) + _vec2_dist(wrist, mr)) * 0.5

        nose_s += d_nose; ear_s += d_ear; mouth_s += d_mouth
        count  += 1

    if count < 4:
        return NAN, NAN, NAN
    return nose_s/count, ear_s/count, mouth_s/count


def feat_index_dir(frames):
    sx = sy = sz = 0.0
    count = 0
    for f in frames:
        pts = _get_hand_pts(f)
        if not pts or len(pts) < 9:
            continue
        dx = pts[8][0] - pts[5][0]
        dy = pts[8][1] - pts[5][1]
        dz = pts[8][2] - pts[5][2]
        mag = math.sqrt(dx*dx + dy*dy + dz*dz)
        if mag < 1e-6:
            continue
        sx += dx/mag; sy += dy/mag; sz += dz/mag
        count += 1
    if count < 4:
        return NAN, NAN, NAN
    n = _normalize3((sx/count, sy/count, sz/count))
    return n if n else (NAN, NAN, NAN)


def feat_wrist_center_x(frames):
    xs = []
    for f in frames:
        if not f.get("HasValidShoulders"):
            continue
        wx, _ = _wrist_xy(f)
        if _is_nan(wx):
            continue
        xs.append(wx)
    if len(xs) < 4:
        return NAN
    return sum(xs) / len(xs)


def feat_two_hand_dist(frames):
    dists = []
    for f in frames:
        if not (f.get("HasValidShoulders") and f.get("HasLeftHand") and f.get("HasRightHand")):
            continue
        lx, ly = _get_vec2(f, "LeftWristBodyRelative")
        rx, ry = _get_vec2(f, "RightWristBodyRelative")
        if _is_nan(lx) or _is_nan(rx):
            continue
        dists.append(_vec2_dist((lx, ly), (rx, ry)))
    if len(dists) < 4:
        return NAN
    return sum(dists) / len(dists)


def feat_two_hand_corr(frames):
    dots = []
    for i in range(1, len(frames)):
        prev, curr = frames[i-1], frames[i]
        if not (prev.get("HasLeftHand") and curr.get("HasLeftHand")):
            continue
        if not (prev.get("HasRightHand") and curr.get("HasRightHand")):
            continue
        plx, _ = _get_vec2(prev, "LeftWristBodyRelative")
        prx, _ = _get_vec2(prev, "RightWristBodyRelative")
        clx, _ = _get_vec2(curr, "LeftWristBodyRelative")
        crx, _ = _get_vec2(curr, "RightWristBodyRelative")
        if any(_is_nan(v) for v in [plx, prx, clx, crx]):
            continue
        ldx = clx - plx
        rdx = crx - prx
        if abs(ldx) + abs(rdx) < 0.001:
            continue
        dots.append(1.0 if ldx * rdx > 0 else -1.0)
    if len(dots) < 4:
        return NAN
    return sum(dots) / len(dots)


def feat_palm_normal(frames):
    sx = sy = sz = 0.0
    count = 0
    for f in frames:
        pts = _get_hand_pts(f)
        if not pts or len(pts) < 18:
            continue
        # lateral = pt17 - pt5, vertical = pt9 - pt0
        lat  = (pts[17][0]-pts[5][0], pts[17][1]-pts[5][1], pts[17][2]-pts[5][2])
        vert = (pts[9][0]-pts[0][0],  pts[9][1]-pts[0][1],  pts[9][2]-pts[0][2])
        # cross product
        nx = lat[1]*vert[2] - lat[2]*vert[1]
        ny = lat[2]*vert[0] - lat[0]*vert[2]
        nz = lat[0]*vert[1] - lat[1]*vert[0]
        mag = math.sqrt(nx*nx + ny*ny + nz*nz)
        if mag < 1e-8:
            continue
        sx += nx/mag; sy += ny/mag; sz += nz/mag
        count += 1
    if count < 4:
        return NAN, NAN, NAN
    n = _normalize3((sx/count, sy/count, sz/count))
    return n if n else (NAN, NAN, NAN)


def feat_finger_extension(frames):
    MCPS = [2, 5, 9, 13, 17]
    TIPS = [4, 8, 12, 16, 20]
    sums  = [0.0] * 5
    count = 0
    for f in frames:
        pts = _get_hand_pts(f)
        if not pts or len(pts) < 21:
            continue
        palm_scale = _vec3_dist(pts[0], pts[9])
        if palm_scale < 1e-4:
            continue
        for i in range(5):
            sums[i] += _vec3_dist(pts[MCPS[i]], pts[TIPS[i]]) / palm_scale
        count += 1
    if count < 4:
        return [NAN] * 5
    return [s/count for s in sums]

# ─── 전체 피처 벡터 추출 ──────────────────────────────────────────────────────

def _apply_recordings_coord_fix(recording):
    """
    Recordings/ 전문가 영상 데이터의 WristBodyRelative 좌표 보정.
    extract_landmarks.py (MediaPipe Python) 와 Unity 의 좌표계는
    x, y 부호가 반전되어 있으므로 부호를 뒤집어 TrainingSamples/ 와 맞춘다.
    원본 dict 를 직접 수정한다.
    """
    for frame in recording.get("Frames", []):
        for key in ("LeftWristBodyRelative", "RightWristBodyRelative"):
            v = frame.get(key)
            if isinstance(v, dict):
                if v.get("x") is not None:
                    v["x"] = -v["x"]
                if v.get("y") is not None:
                    v["y"] = -v["y"]
    return recording


def extract_features(recording):
    """
    HandPoseRecording JSON dict → 26개 float 리스트.
    C#의 HandPoseEvaluator.ExtractFeatureVector() 와 동일한 순서·로직.
    """
    frames = recording.get("Frames", [])
    frames = _trim_inactive(frames)

    feats = []

    feats.append(feat_aperture_trend(frames))

    dx, dy = feat_wrist_traj(frames)
    feats.append(dx)
    feats.append(dy)

    feats.append(feat_osc_x(frames))
    feats.append(feat_osc_y(frames))
    feats.append(feat_path_closure(frames))
    feats.append(feat_wrist_z_amp(frames))
    feats.append(feat_wrist_center_y(frames))
    feats.append(feat_wrist_y_std(frames))

    fn, fe, fm = feat_face_dists(frames)
    feats.append(fn)
    feats.append(fe)
    feats.append(fm)

    ix, iy, iz = feat_index_dir(frames)
    feats.append(ix)
    feats.append(iy)
    feats.append(iz)

    feats.append(feat_wrist_center_x(frames))
    feats.append(feat_two_hand_dist(frames))
    feats.append(feat_two_hand_corr(frames))

    px, py, pz = feat_palm_normal(frames)
    feats.append(px)
    feats.append(py)
    feats.append(pz)

    fep = feat_finger_extension(frames)
    feats.extend(fep)

    return feats

# ─── 진입점 ──────────────────────────────────────────────────────────────────

def _load_dir(folder, coord_fix=False, source_name=""):
    """JSON 폴더에서 (label, feats) 행 리스트를 반환한다."""
    rows = []
    skipped = 0
    files = sorted(f for f in os.listdir(folder) if f.endswith(".json"))
    print(f"[{source_name}] {len(files)}개 JSON 발견: {folder}")
    for fname in files:
        # _slow 파일 제외
        if fname.endswith("_slow.json"):
            print(f"  [제외] {fname}: _slow 파일")
            skipped += 1
            continue

        path = os.path.join(folder, fname)
        try:
            with open(path, encoding="utf-8") as fp:
                rec = json.load(fp)
        except Exception as e:
            print(f"  [건너뜀] {fname}: {e}")
            skipped += 1
            continue

        label = rec.get("DisplayName") or rec.get("MotionId") or ""
        if not label:
            print(f"  [건너뜀] {fname}: DisplayName/MotionId 없음")
            skipped += 1
            continue

        if coord_fix:
            _apply_recordings_coord_fix(rec)

        feats = extract_features(rec)

        nan_count = sum(1 for v in feats if _is_nan(v))
        if nan_count > 10:
            print(f"  [경고] {fname} ({label}): NaN {nan_count}/{len(feats)}개")

        rows.append((label, feats))
        skipped_msg = ""

    print(f"  → {len(rows)}개 추출, {skipped}개 건너뜀")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="TrainingSamples JSON → 피처 CSV (train_svm.py 입력용)"
    )
    parser.add_argument(
        "--samples-dir", required=True,
        help="Unity TrainingSamples 폴더 경로"
             " (보통 .../AppData/LocalLow/DefaultCompany/WordsInSilence/TrainingSamples)"
    )
    parser.add_argument(
        "--recordings-dir", default=None,
        help="전문가 영상 Recordings 폴더 경로 (선택). "
             "포함 시 각 단어의 기준 앵커로 활용됨. _slow 파일은 자동 제외."
    )
    parser.add_argument(
        "--output", default="features.csv",
        help="출력 CSV 파일 경로 (기본: features.csv)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.samples_dir):
        print(f"[오류] 폴더 없음: {args.samples_dir}")
        sys.exit(1)

    all_rows = []

    # TrainingSamples/ 로드 (좌표 보정 없음)
    all_rows += _load_dir(args.samples_dir, coord_fix=False, source_name="TrainingSamples")

    # Recordings/ 로드 (좌표 보정 적용)
    if args.recordings_dir:
        if not os.path.isdir(args.recordings_dir):
            print(f"[경고] Recordings 폴더 없음: {args.recordings_dir}")
        else:
            all_rows += _load_dir(args.recordings_dir, coord_fix=True, source_name="Recordings")

    if not all_rows:
        print("[오류] 추출된 샘플 없음.")
        sys.exit(1)

    label_counts = {}
    rows_out = []
    for label, feats in all_rows:
        rows_out.append([label] + feats)
        label_counts[label] = label_counts.get(label, 0) + 1

    out_path = args.output
    with open(out_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["label"] + FEATURE_NAMES)
        writer.writerows(rows_out)

    print(f"\n[완료] 총 {len(rows_out)}개 샘플 추출")
    print(f"       저장: {os.path.abspath(out_path)}")
    print("\n단어별 샘플 수:")
    for label, cnt in sorted(label_counts.items(), key=lambda x: x[0]):
        status = "[부족] 권장 20개 이상" if cnt < 10 else ("[보통]" if cnt < 20 else "[OK]")
        print(f"  {label:12s}: {cnt:3d}개  {status}")


if __name__ == "__main__":
    main()
