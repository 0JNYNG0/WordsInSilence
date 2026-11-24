"""
s01_pipeline.py  —  S01 데이터셋 처리 및 leave-one-repetition-out 기준선

S01 구조:
  - 감사합니다(THANKS): 10회 반복
  - 인사(GREETING): 12회 반복
  - 사랑(LOVE): 11회 반복
  - 각 수어는 front/left45/right45 동시 촬영 (긴 단일 MP4)
  - 시작 박수로 세 시점 동기화

전제:
  - 각 MP4는 모든 반복을 포함한 연속 영상
  - extract_landmarks.py 가 각 영상 전체를 하나의 JSON으로 추출한다
  - 본 스크립트는 그 JSON을 박수 동기화 + 반복 분리 후 LOO 기준선 계산
  - MediaPipe 설치 없이도 --skip-extract 로 기존 JSON 재사용 가능

참고사항:
  - left45/right45 뷰는 LOO 기준선에 포함하지 않는다 (front만 사용)
  - 자동 분리 결과가 기대 반복 수와 다르면 보고하고 수동 경계 CSV를 확인한다
  - 누락 좌표를 (0,0)으로 채우지 않는다; NaN 프레임은 DTW에서 제외
  - DTW 거리는 경로 길이로 정규화한다

실행:
  python Tools/s01_pipeline.py
  python Tools/s01_pipeline.py --skip-extract --out-dir Tools/s01_out
  python Tools/s01_pipeline.py --manual-boundaries Tools/s01_boundaries.csv
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys

# Windows 콘솔 UTF-8 출력
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
if hasattr(sys.stderr, "reconfigure"):
    try: sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

# ── 상수 ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DEFAULT_DATASET = os.path.join(SCRIPT_DIR, "dataset")
DEFAULT_OUT     = os.path.join(SCRIPT_DIR, "s01_out")
EXTRACT_SCRIPT  = os.path.join(SCRIPT_DIR, "extract_landmarks.py")

SIGNS = {
    "SIGN001": {"label": "감사합니다", "tag": "THANKS",  "expected_reps": 10},
    "SIGN002": {"label": "인사",       "tag": "GREETING","expected_reps": 12},
    "SIGN003": {"label": "사랑",       "tag": "LOVE",    "expected_reps": 11},
}
VIEWS   = ["front", "left45", "right45"]
PATTERN = "{sign}_{tag}_P01_S01_B01_{view}.mp4"


# ── 파일 경로 확인 ────────────────────────────────────────────────────────────
def find_files(dataset_dir):
    """
    S01 파일 목록을 확인한다.
    반환: {(sign, view): path} 또는 누락 목록
    """
    found = {}
    missing = []
    for sign_code, info in SIGNS.items():
        for view in VIEWS:
            fname = PATTERN.format(sign=sign_code, tag=info["tag"], view=view)
            path  = os.path.join(dataset_dir, fname)
            if os.path.isfile(path):
                found[(sign_code, view)] = path
            else:
                missing.append(f"{dataset_dir}/{fname}")
    return found, missing


# ── 추출 ───────────────────────────────────────────────────────────────────────
def extract_one(mp4_path, out_json, motion_id, display_name):
    if not os.path.isfile(EXTRACT_SCRIPT):
        print(f"  [오류] extract_landmarks.py 없음: {EXTRACT_SCRIPT}", file=sys.stderr)
        return False
    cmd = [
        sys.executable, EXTRACT_SCRIPT,
        "--video",        mp4_path,
        "--motion-id",    motion_id,
        "--display-name", display_name,
        "--out",          out_json,
    ]
    print(f"  추출: {os.path.basename(mp4_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [오류] {result.stderr.strip()[:200]}", file=sys.stderr)
        return False
    return True


# ── JSON 로드 ─────────────────────────────────────────────────────────────────
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── 손목 시퀀스 추출 ──────────────────────────────────────────────────────────
def wrist_sequence(frames, side="right"):
    """
    body-relative 손목 (x, y) 시퀀스.
    NaN이거나 손이 없는 프레임은 None으로 표시 (0으로 채우지 않음).
    """
    key = f"{'Left' if side == 'left' else 'Right'}WristBodyRelative"
    pts = []
    for fr in frames:
        has = fr.get(f"Has{'Left' if side == 'left' else 'Right'}Hand", False)
        v = fr.get(key, {})
        if has and isinstance(v, dict) and v.get("x") is not None:
            x, y = v.get("x", 0.0), v.get("y", 0.0)
            if not (math.isnan(x) or math.isnan(y)):
                pts.append((x, y))
                continue
        pts.append(None)
    return pts


# ── 클랩 동기화 ───────────────────────────────────────────────────────────────
def detect_clap_frame(frames):
    """
    양손이 갑자기 빠르게 접근했다가 멀어지는 프레임을 클랩으로 감지한다.
    감지 실패 시 0을 반환한다.
    """
    best_frame = 0
    best_score = -1.0

    for i in range(1, len(frames) - 1):
        fr = frames[i]
        has_both = fr.get("HasLeftHand", False) and fr.get("HasRightHand", False)
        if not has_both:
            continue

        lv = fr.get("LeftWristBodyRelative", {})
        rv = fr.get("RightWristBodyRelative", {})
        if not (isinstance(lv, dict) and isinstance(rv, dict)):
            continue

        lx = lv.get("x", None)
        rx = rv.get("x", None)
        if lx is None or rx is None:
            continue

        dist = abs(rx - lx)
        # 클랩: 손 간 거리가 최소에 가까울 때 (두 손이 가장 가까움)
        score = -dist  # 작을수록 높은 점수
        if score > best_score:
            best_score = score
            best_frame = i

    return best_frame


# ── 반복 분리 ─────────────────────────────────────────────────────────────────
def segment_repetitions(frames, expected_count, sign_label, fps_est=30.0):
    """
    연속 영상에서 개별 반복 동작을 분리한다.

    전략:
      1. 손목 이동 에너지를 계산한다.
      2. 에너지가 낮은 구간(준비/휴식)을 경계로 사용한다.
      3. expected_count 개 반복이 검출되지 않으면 경고하고 그대로 반환.

    반환: [(start_frame_idx, end_frame_idx), ...] (원본 frames 기준 인덱스)
    """
    if not frames:
        return []

    n = len(frames)
    # 간단한 에너지 계산: 연속 프레임 손목 이동량
    energy = [0.0] * n
    for i in range(1, n):
        r_seq = wrist_sequence([frames[i - 1], frames[i]], "right")
        l_seq = wrist_sequence([frames[i - 1], frames[i]], "left")
        e = 0.0
        if r_seq[0] and r_seq[1]:
            dx = r_seq[1][0] - r_seq[0][0]
            dy = r_seq[1][1] - r_seq[0][1]
            e += (dx * dx + dy * dy) ** 0.5
        if l_seq[0] and l_seq[1]:
            dx = l_seq[1][0] - l_seq[0][0]
            dy = l_seq[1][1] - l_seq[0][1]
            e += (dx * dx + dy * dy) ** 0.5
        energy[i] = e

    # 슬라이딩 평균으로 스무딩
    window = max(1, int(fps_est * 0.5))  # 0.5초
    smoothed = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n - 1, i + window // 2)
        chunk = energy[lo:hi + 1]
        smoothed.append(sum(chunk) / len(chunk) if chunk else 0)

    # 임계값: 평균의 20%
    threshold = sum(smoothed) / n * 0.20

    # 활성 구간 찾기
    in_rep = False
    segments = []
    rep_start = 0
    for i in range(n):
        if not in_rep and smoothed[i] >= threshold:
            in_rep = True
            rep_start = i
        elif in_rep and smoothed[i] < threshold:
            in_rep = False
            if i - rep_start >= int(fps_est * 0.3):  # 0.3초 이상
                segments.append((rep_start, i - 1))
    if in_rep:
        segments.append((rep_start, n - 1))

    # 반복 수 검증
    if len(segments) != expected_count:
        print(f"  [경고] {sign_label}: 기대 반복={expected_count}, "
              f"검출={len(segments)}. 수동 경계 CSV를 사용하거나 임계값 조정 필요.")

    return segments


# ── 수동 경계 CSV 로드 ────────────────────────────────────────────────────────
def load_manual_boundaries(csv_path):
    """
    수동 경계 CSV: sign_code,view,rep_idx,start_frame,end_frame
    반환: {(sign_code, view): [(start, end), ...]}
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    boundaries = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["sign_code"], row["view"])
            if key not in boundaries:
                boundaries[key] = []
            boundaries[key].append((int(row["start_frame"]), int(row["end_frame"])))
    # sort by rep_idx
    for key in boundaries:
        boundaries[key].sort()
    return boundaries


# ── DTW 거리 (정규화) ─────────────────────────────────────────────────────────
def dtw_distance_normalized(seq_a, seq_b, band_ratio=0.20):
    """
    Sakoe-Chiba band DTW.
    NaN/None 포인트를 건너뛴다.
    경로 길이로 정규화하여 시퀀스 길이 차이를 보정한다.
    """
    a = [p for p in seq_a if p is not None]
    b = [p for p in seq_b if p is not None]
    n, m = len(a), len(b)

    if n == 0 or m == 0:
        return float("inf")

    band = max(1, int(band_ratio * max(n, m)))
    INF  = float("inf")
    dp   = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            dx   = a[i - 1][0] - b[j - 1][0]
            dy   = a[i - 1][1] - b[j - 1][1]
            cost = (dx * dx + dy * dy) ** 0.5
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    raw = dp[n][m]
    # 경로 길이 정규화: 이론적 최대 경로 = n + m - 1
    return raw / (n + m - 1) if raw < INF else INF


# ── LOO 기준선 ────────────────────────────────────────────────────────────────
def run_loo_baseline(samples):
    """
    samples: [{sample_id, sign_code, label, rep_idx, seq}]
    Leave-one-repetition-out: 한 반복을 쿼리로, 나머지 front 반복을 레퍼런스로 사용.
    """
    results = []
    for i, query in enumerate(samples):
        refs = [s for j, s in enumerate(samples) if j != i]
        if not refs:
            continue

        best_label = None
        best_dist  = float("inf")
        for ref in refs:
            d = dtw_distance_normalized(query["seq"], ref["seq"])
            if d < best_dist:
                best_dist  = d
                best_label = ref["label"]

        correct = best_label == query["label"]

        # 같은 수어 vs 다른 수어 거리
        same_dists  = [dtw_distance_normalized(query["seq"], r["seq"])
                       for r in refs if r["label"] == query["label"]]
        diff_dists  = [dtw_distance_normalized(query["seq"], r["seq"])
                       for r in refs if r["label"] != query["label"]]

        results.append({
            "sample_id":   query["sample_id"],
            "sign_code":   query["sign_code"],
            "label":       query["label"],
            "rep_idx":     query["rep_idx"],
            "predicted":   best_label,
            "correct":     correct,
            "best_dist":   round(best_dist, 5),
            "same_sign_avg_dist": round(sum(same_dists) / len(same_dists), 5) if same_dists else None,
            "diff_sign_avg_dist": round(sum(diff_dists) / len(diff_dists), 5) if diff_dists else None,
        })

    return results


# ── 리포트 작성 ───────────────────────────────────────────────────────────────
def write_report(out_dir, loo_results, found_files, missing_files, seg_summary, errors):
    lines = ["=== S01 데이터셋 LOO 기준선 결과 ===", ""]

    # 파일 현황
    lines += [f"파일 발견: {len(found_files)}개 / 누락: {len(missing_files)}개"]
    if missing_files:
        lines.append("누락 파일:")
        for p in missing_files:
            lines.append(f"  {p}")
    lines.append("")

    # 반복 분리 요약
    lines.append("=== 반복 분리 결과 ===")
    for s in seg_summary:
        match = "O" if s["detected"] == s["expected"] else "X"
        lines.append(f"  [{match}] {s['sign_code']} ({s['label']}): "
                     f"기대={s['expected']}, 검출={s['detected']}")
    lines.append("")

    # LOO 결과
    if not loo_results:
        lines.append("[경고] LOO 기준선을 계산할 수 없었습니다.")
        lines.append("  (JSON 추출 실패 또는 --skip-extract 후 JSON 없음)")
    else:
        n_correct = sum(1 for r in loo_results if r["correct"])
        acc = n_correct / len(loo_results) if loo_results else 0.0
        lines += [
            "=== LOO 기준선 (front 반복 기준) ===",
            f"방법: Leave-one-repetition-out DTW (body-relative 손목, 경로길이 정규화)",
            f"좌표 규약: OfflineMediaPipe (CanonicalFrame 변환 미적용)",
            f"샘플 수: {len(loo_results)}, 정답: {n_correct}",
            f"정확도: {acc * 100:.1f}%",
            "",
            "--- 개별 결과 ---",
        ]
        for r in loo_results:
            mark = "O" if r["correct"] else "X"
            same = f"{r['same_sign_avg_dist']:.4f}" if r["same_sign_avg_dist"] else "N/A"
            diff = f"{r['diff_sign_avg_dist']:.4f}" if r["diff_sign_avg_dist"] else "N/A"
            lines.append(
                f"  [{mark}] {r['label']}(rep{r['rep_idx']}) "
                f"→ 예측:{r['predicted']}  best={r['best_dist']:.4f}  "
                f"같은수어_avg={same}  다른수어_avg={diff}"
            )
        # 수어 간 거리 요약
        lines += [
            "",
            "--- 수어 간 거리 요약 ---",
        ]
        for sign_code, info in SIGNS.items():
            label = info["label"]
            sign_results = [r for r in loo_results if r["sign_code"] == sign_code]
            if not sign_results:
                continue
            same_vals = [r["same_sign_avg_dist"] for r in sign_results if r["same_sign_avg_dist"] is not None]
            diff_vals = [r["diff_sign_avg_dist"] for r in sign_results if r["diff_sign_avg_dist"] is not None]
            same_avg = sum(same_vals) / len(same_vals) if same_vals else float("nan")
            diff_avg = sum(diff_vals) / len(diff_vals) if diff_vals else float("nan")
            lines.append(f"  {label}: 같은수어={same_avg:.4f}, 다른수어={diff_avg:.4f}")

    # 오류
    if errors:
        lines += ["", "=== 오류 목록 ==="]
        lines += [f"  {e}" for e in errors]

    lines += [
        "",
        "=== 주의사항 ===",
        "  - 이 결과는 front 뷰 전체 영상을 자동 분리한 반복 기반 기준선입니다.",
        "  - 합격 임계값은 아직 확정되지 않았습니다.",
        "  - ExpectedSignPrototype 모드는 CanonicalFrame 변환 후 비교합니다 (현재 stub).",
        "  - left45/right45 뷰는 기준선에 포함하지 않았습니다.",
    ]

    report_path = os.path.join(out_dir, "s01_baseline_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n리포트 저장: {report_path}")
    for line in lines:
        print(line)

    # LOO 결과 CSV
    if loo_results:
        csv_path = os.path.join(out_dir, "s01_loo_results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(loo_results[0].keys()))
            writer.writeheader()
            writer.writerows(loo_results)
        print(f"LOO 결과 CSV: {csv_path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="S01 데이터셋 처리 및 LOO 기준선")
    parser.add_argument("--dataset-dir",        default=DEFAULT_DATASET)
    parser.add_argument("--out-dir",            default=DEFAULT_OUT)
    parser.add_argument("--skip-extract",       action="store_true")
    parser.add_argument("--manual-boundaries",  default=None,
                        help="수동 경계 CSV (sign_code,view,rep_idx,start_frame,end_frame)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    errors = []

    # 1. 파일 확인
    print("=== S01 파일 확인 ===")
    found_files, missing_files = find_files(args.dataset_dir)
    for (sign, view), path in sorted(found_files.items()):
        print(f"  발견: {os.path.basename(path)}")
    for path in missing_files:
        print(f"  누락: {path}")

    if missing_files:
        print(f"\n[경고] {len(missing_files)}개 파일 없음. 발견된 파일만 처리합니다.")

    # 2. 추출
    json_paths = {}
    if args.skip_extract:
        print("\n추출 단계 건너뜀 (--skip-extract). 기존 JSON 재사용.")
        for (sign, view) in found_files:
            p = os.path.join(args.out_dir, f"{sign.lower()}_{view}.json")
            if os.path.isfile(p):
                json_paths[(sign, view)] = p
            else:
                errors.append(f"JSON 없음 (--skip-extract): {p}")
    else:
        print("\n=== 랜드마크 추출 ===")
        for (sign, view), mp4_path in sorted(found_files.items()):
            motion_id = f"s01_{sign.lower()}_{view}"
            label     = SIGNS[sign]["label"]
            out_json  = os.path.join(args.out_dir, f"{sign.lower()}_{view}.json")
            ok = extract_one(mp4_path, out_json, motion_id, label)
            if ok:
                json_paths[(sign, view)] = out_json
            else:
                errors.append(f"추출 실패: {os.path.basename(mp4_path)}")

    # 3. 수동 경계 로드
    manual_boundaries = load_manual_boundaries(args.manual_boundaries)

    # 4. front 반복 분리
    print("\n=== 반복 분리 (front 뷰) ===")
    seg_summary = []
    all_samples = []

    for sign_code, info in SIGNS.items():
        label    = info["label"]
        exp_reps = info["expected_reps"]
        view     = "front"
        key      = (sign_code, view)

        if key not in json_paths:
            print(f"  [{sign_code} front] JSON 없음 - 건너뜀")
            seg_summary.append({"sign_code": sign_code, "label": label,
                                 "expected": exp_reps, "detected": 0})
            continue

        data   = load_json(json_paths[key])
        frames = data.get("Frames", [])
        if not frames:
            print(f"  [{sign_code} front] 프레임 없음")
            seg_summary.append({"sign_code": sign_code, "label": label,
                                 "expected": exp_reps, "detected": 0})
            continue

        # 손 검출률, pose 검출률
        hand_cnt = sum(1 for f in frames
                       if f.get("HasLeftHand", False) or f.get("HasRightHand", False))
        pose_cnt = sum(1 for f in frames if f.get("HasPose", False))
        print(f"  [{sign_code} front] 총 {len(frames)}프레임  "
              f"hand={hand_cnt/len(frames):.2f}  pose={pose_cnt/len(frames):.2f}")

        # 경계 결정
        if key in manual_boundaries:
            segs = manual_boundaries[key]
            print(f"    수동 경계 사용: {len(segs)}개")
        else:
            # FPS 추정
            ts = [f.get("TimestampMs", i * 33) for i, f in enumerate(frames)]
            fps_est = ((len(ts) - 1) / ((ts[-1] - ts[0]) / 1000.0)
                       if len(ts) >= 2 and ts[-1] > ts[0] else 30.0)
            segs = segment_repetitions(frames, exp_reps, label, fps_est)
            print(f"    자동 분리: {len(segs)}개 (기대 {exp_reps}개)")

        seg_summary.append({"sign_code": sign_code, "label": label,
                             "expected": exp_reps, "detected": len(segs)})

        # 각 반복에 sample_id 부여
        for rep_idx, (seg_start, seg_end) in enumerate(segs):
            rep_frames = frames[seg_start:seg_end + 1]
            seq = wrist_sequence(rep_frames, "right")
            # 손이 없으면 left 시도
            valid_right = sum(1 for p in seq if p is not None)
            if valid_right < len(seq) * 0.3:
                seq = wrist_sequence(rep_frames, "left")

            valid_cnt = sum(1 for p in seq if p is not None)
            print(f"    rep{rep_idx:02d}: frames[{seg_start}-{seg_end}] "
                  f"길이={seg_end-seg_start+1}  유효손목={valid_cnt}/{len(seq)}")

            all_samples.append({
                "sample_id": f"{sign_code}_front_rep{rep_idx:02d}",
                "sign_code": sign_code,
                "label":     label,
                "rep_idx":   rep_idx,
                "seq":       seq,
            })

    # 5. LOO 기준선
    loo_results = []
    if len(all_samples) >= 2:
        print("\n=== LOO 기준선 계산 ===")
        loo_results = run_loo_baseline(all_samples)
        n_correct = sum(1 for r in loo_results if r["correct"])
        print(f"정확도: {n_correct}/{len(loo_results)} = "
              f"{n_correct/len(loo_results)*100:.1f}%")
    else:
        print("\n[경고] LOO 계산을 위한 샘플이 부족합니다.")

    # 6. 리포트
    write_report(args.out_dir, loo_results, found_files, missing_files, seg_summary, errors)


if __name__ == "__main__":
    main()
