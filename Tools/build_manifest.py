"""
build_manifest.py  —  WordsInSilence 데이터 매니페스트 생성

Recordings/ 와 TrainingSamples/ 의 모든 JSON 을 스캔하여
data_manifest.csv 를 생성한다.

출력 컬럼:
  sample_id, source_json, label, motion_id, performer_id, session_id,
  camera_view, source_type, coordinate_convention, applied_transform,
  is_derived, derivation_method, parent_source, original_fps, normalized_fps,
  total_frames, valid_hand_ratio, valid_pose_ratio, valid_combined_ratio,
  sha256, generated_feature_row_index, exclusion_reason, notes

valid_combined_ratio 계산:
  ValidCombinedFrames / TotalFrames (JSON root 저장값 사용).
  ValidCombinedFrames = (HasLeftHand OR HasRightHand) AND HasPose 인 프레임 수.
  root 저장값이 없으면 프레임 배열에서 재계산.

실행:
  python Tools/build_manifest.py
  python Tools/build_manifest.py --recordings-dir ... --training-dir ... --features-csv ... --out data_manifest.csv
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

# ── 기본 경로 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Unity persistentDataPath (Windows 기준)
PERSISTENT_ROOT = os.path.join(
    os.environ.get("LOCALAPPDATA", "").replace("Local", "LocalLow"),
    "DefaultCompany", "WordsInSilence"
)

DEFAULT_RECORDINGS = os.path.join(PERSISTENT_ROOT, "Recordings")
DEFAULT_TRAINING   = os.path.join(PERSISTENT_ROOT, "TrainingSamples")
DEFAULT_FEATURES   = os.path.join(PROJECT_ROOT, "features_with_expert.csv")
DEFAULT_OUT        = os.path.join(SCRIPT_DIR, "data_manifest.csv")

FIELDNAMES = [
    "sample_id",
    "source_json",
    "label",
    "motion_id",
    "performer_id",
    "session_id",
    "camera_view",
    "source_type",
    "coordinate_convention",
    "applied_transform",
    "is_derived",
    "derivation_method",
    "parent_source",
    "original_fps",
    "normalized_fps",
    "total_frames",
    "valid_hand_ratio",
    "valid_pose_ratio",
    "valid_combined_ratio",
    "sha256",
    "generated_feature_row_index",
    "exclusion_reason",
    "notes",
]


# ── SHA-256 ────────────────────────────────────────────────────────────────────
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── FPS 추정 ──────────────────────────────────────────────────────────────────
def estimate_fps(frames):
    """프레임 타임스탬프로부터 원본 FPS를 추정한다."""
    if len(frames) < 2:
        return "unknown"
    total_ms = frames[-1].get("TimestampMs", 0) - frames[0].get("TimestampMs", 0)
    if total_ms <= 0:
        return "unknown"
    fps = (len(frames) - 1) / (total_ms / 1000.0)
    return f"{fps:.1f}"


# ── ValidCombinedFrames 재계산 (root 저장값 없을 때) ───────────────────────────
def recompute_combined(frames):
    """
    (HasLeftHand OR HasRightHand) AND HasPose 인 프레임 수.
    JSON root의 ValidCombinedFrames 저장값과 동일한 기준.
    """
    count = 0
    for f in frames:
        has_hand = f.get("HasLeftHand", False) or f.get("HasRightHand", False)
        has_pose = f.get("HasPose", False)
        if has_hand and has_pose:
            count += 1
    return count


# ── features_with_expert.csv 로드 ─────────────────────────────────────────────
def load_feature_row_map(features_csv_path):
    """
    features_with_expert.csv 를 읽어 label → 행 인덱스 리스트 dict 를 반환한다.
    파일이 없으면 빈 dict 를 반환한다.
    """
    if not os.path.isfile(features_csv_path):
        print(f"[warn] features_with_expert.csv 없음: {features_csv_path}", file=sys.stderr)
        return {}

    label_rows = defaultdict(list)
    with open(features_csv_path, encoding="utf-8-sig") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            label_rows[row["label"]].append(idx)
    return dict(label_rows)


# ── 단일 JSON 파싱 ─────────────────────────────────────────────────────────────
def parse_recording(path, source_type, sample_id, label_row_map):
    filename = os.path.basename(path)
    sha = sha256_of(path)
    row = {f: "" for f in FIELDNAMES}
    row["sample_id"]  = sample_id
    row["source_json"] = filename
    row["sha256"]     = sha
    row["source_type"] = source_type

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        row["notes"] = f"parse_error: {e}"
        return row

    motion_id    = data.get("MotionId", "")
    display_name = data.get("DisplayName", "")
    total_frames = data.get("TotalFrames", 0) or 0
    valid_hand   = data.get("ValidHandFrames", None)
    valid_pose   = data.get("ValidPoseFrames", None)
    valid_comb   = data.get("ValidCombinedFrames", None)
    frames       = data.get("Frames", []) or []

    row["motion_id"]    = motion_id
    row["label"]        = display_name if display_name else motion_id

    # label: DisplayName 우선
    label = display_name if display_name else motion_id

    # _slow 판별
    is_slow = motion_id.endswith("_slow") or filename.endswith("_slow.json")

    # source_type별 메타데이터
    if source_type == "expert_video_extracted":
        row["performer_id"]          = "unknown"   # 전문가 ID 원본에 없음
        row["session_id"]            = "unknown"
        row["camera_view"]           = "front"
        row["coordinate_convention"] = "OfflineMediaPipe"
        row["applied_transform"]     = "flip_body_relative_x, flip_body_relative_y"
        if is_slow:
            row["is_derived"]         = "true"
            row["derivation_method"]  = "slow_playback_extraction"
            # parent_source: 같은 단어의 non-slow JSON
            parent_name = filename.replace("_slow.json", ".json")
            row["parent_source"]      = parent_name
            row["exclusion_reason"]   = "slow_variant_excluded_from_svm_training"
        else:
            row["is_derived"]         = "false"
            row["derivation_method"]  = ""
            row["parent_source"]      = ""
    else:  # user_recording (TrainingSamples)
        row["performer_id"]          = "user_01"   # 단일 사용자, 미공식 추적
        row["session_id"]            = "unknown"   # 두 세션 있었으나 미기록
        row["camera_view"]           = "front"
        row["coordinate_convention"] = "UnityRuntime"
        row["applied_transform"]     = "none"
        row["is_derived"]            = "false"
        row["derivation_method"]     = ""
        row["parent_source"]         = ""

    # 프레임 통계
    actual_frame_count = len(frames)
    tf = total_frames if total_frames > 0 else actual_frame_count
    row["total_frames"] = str(tf)

    # valid_hand_ratio, valid_pose_ratio: root 저장값 우선
    if tf > 0:
        if valid_hand is not None:
            row["valid_hand_ratio"] = f"{valid_hand / tf:.4f}"
        else:
            hand_cnt = sum(1 for fr in frames if fr.get("HasLeftHand", False) or fr.get("HasRightHand", False))
            row["valid_hand_ratio"] = f"{hand_cnt / tf:.4f}"

        if valid_pose is not None:
            row["valid_pose_ratio"] = f"{valid_pose / tf:.4f}"
        else:
            pose_cnt = sum(1 for fr in frames if fr.get("HasPose", False))
            row["valid_pose_ratio"] = f"{pose_cnt / tf:.4f}"

        # valid_combined_ratio: (HasHand AND HasPose) / TotalFrames
        # root 저장값 우선 (ValidCombinedFrames = HasHand AND HasPose per frame)
        if valid_comb is not None:
            row["valid_combined_ratio"] = f"{valid_comb / tf:.4f}"
        else:
            row["valid_combined_ratio"] = f"{recompute_combined(frames) / tf:.4f}"

    # FPS 추정
    row["original_fps"]   = estimate_fps(frames)
    row["normalized_fps"] = "N/A (statistical_features)"

    # features_with_expert.csv 행 연결
    # 구조: 행 0-199 = 사용자 훈련 샘플 (20 라벨 × 10)
    #       행 200-218 = 전문가 레퍼런스 (19 수어 × 1)
    feat_indices = label_row_map.get(label, [])
    if is_slow:
        row["generated_feature_row_index"] = ""  # 제외됨
    elif feat_indices:
        # expert rows: index >= 200; training rows: index < 200
        expert_feat = [i for i in feat_indices if i >= 200]
        train_feat  = [i for i in feat_indices if i < 200]
        if source_type == "expert_video_extracted":
            if expert_feat:
                row["generated_feature_row_index"] = str(expert_feat[0])
            else:
                row["generated_feature_row_index"] = "expert_row_not_found"
        else:
            if train_feat:
                row["generated_feature_row_index"] = f"{train_feat[0]}-{train_feat[-1]}"
            else:
                row["generated_feature_row_index"] = "training_rows_not_found"
    else:
        row["generated_feature_row_index"] = "not_in_features_csv"

    # notes
    notes = []
    if is_slow:
        notes.append("slow variant")
    if source_type == "expert_video_extracted" and not is_slow:
        notes.append("expert reference recording from KSL video (AIHub/국립국어원 계열)")
    if tf > 0 and valid_comb is not None and valid_comb / tf < 0.30:
        notes.append(f"low valid_combined_ratio ({valid_comb/tf:.2f})")
    row["notes"] = "; ".join(notes)

    return row


# ── 디렉터리 스캔 ──────────────────────────────────────────────────────────────
def scan_directory(dirpath, source_type, label_row_map, id_prefix, start_idx=0):
    rows = []
    if not os.path.isdir(dirpath):
        print(f"[warn] 디렉터리 없음: {dirpath}", file=sys.stderr)
        return rows, start_idx

    for i, fname in enumerate(sorted(os.listdir(dirpath))):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dirpath, fname)
        sample_id = f"{id_prefix}_{start_idx + i:04d}"
        row = parse_recording(fpath, source_type, sample_id, label_row_map)
        rows.append(row)

    return rows, start_idx + len(rows)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="WordsInSilence 데이터 매니페스트 생성")
    parser.add_argument("--recordings-dir",  default=DEFAULT_RECORDINGS)
    parser.add_argument("--training-dir",    default=DEFAULT_TRAINING)
    parser.add_argument("--features-csv",    default=DEFAULT_FEATURES)
    parser.add_argument("--out",             default=DEFAULT_OUT)
    args = parser.parse_args()

    label_row_map = load_feature_row_map(args.features_csv)
    print(f"features_with_expert.csv: {len(label_row_map)} labels 로드")

    # Recordings/ — expert 추출
    rec_rows, next_idx = scan_directory(
        args.recordings_dir, "expert_video_extracted", label_row_map, "REC")

    # TrainingSamples/ — Unity 런타임 녹화
    train_rows, _ = scan_directory(
        args.training_dir, "user_recording", label_row_map, "TRN")

    all_rows = rec_rows + train_rows

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    # 요약 통계
    print(f"\n매니페스트 생성 완료: {args.out}")
    print(f"  Recordings/    : {len(rec_rows):4d} 행")
    print(f"  TrainingSamples/: {len(train_rows):4d} 행")
    print(f"  합계            : {len(all_rows):4d} 행")

    # feature 연결 성공/실패
    linked   = sum(1 for r in all_rows if r["generated_feature_row_index"]
                   and r["generated_feature_row_index"] not in ("", "not_in_features_csv"))
    excluded = sum(1 for r in all_rows if r["exclusion_reason"])
    unlinked = sum(1 for r in all_rows if r["generated_feature_row_index"] == "not_in_features_csv")
    print(f"\n  feature 연결 성공: {linked}")
    print(f"  제외 (slow 등)  : {excluded}")
    print(f"  미연결          : {unlinked}")

    # valid_combined_ratio 분포
    ratios = []
    for r in all_rows:
        try: ratios.append(float(r["valid_combined_ratio"]))
        except: pass
    below_03 = sum(1 for v in ratios if v < 0.30)
    below_07 = sum(1 for v in ratios if v < 0.70)
    print(f"\n  valid_combined_ratio < 0.30: {below_03} / {len(ratios)}")
    print(f"  valid_combined_ratio < 0.70: {below_07} / {len(ratios)}")

    return len(all_rows), linked, excluded


if __name__ == "__main__":
    main()
