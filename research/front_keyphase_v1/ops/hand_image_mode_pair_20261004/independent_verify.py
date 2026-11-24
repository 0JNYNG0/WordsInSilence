"""Independent saved-observation verification. No decoder/MediaPipe/camera imports.

The prior independent verifier is reused solely for fixed official preprocessing
and 8 independent TCN forwards. Candidate runner/helper functions are not imported.
Run once after main execution; exclusive output creation prevents overwrite.
"""
from pathlib import Path
import copy
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "capture_cadence_pair_20261004/independent_verify.py"
spec = importlib.util.spec_from_file_location("independent_cadence_reference", HELPER)
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def main():
    output = HERE / "INDEPENDENT_RESULT.json"
    markdown = HERE / "INDEPENDENT_RESULT.md"
    if output.exists() or markdown.exists():
        raise RuntimeError("Exclusive verification outputs already exist")
    cfg_path = HERE / "PREPARED_CONFIG.json"
    cfg = V.read_json(cfg_path)
    result_dir = Path(cfg["output_dir"])
    result_path = result_dir / "RESULT.json"
    result = V.read_json(result_path)
    manifest_path = result_dir / "MANIFEST.json"
    manifest = V.read_json(manifest_path)
    protected = {str(cfg_path): V.sha256_file(cfg_path), str(result_path): V.sha256_file(result_path),
                 str(manifest_path): V.sha256_file(manifest_path), str(HELPER): V.sha256_file(HELPER)}
    protected.update(cfg["pins"])
    for name, sha in manifest.items():
        path = V.safe_result_file(result_dir, name)
        V.required(V.sha256_file(path) == sha, f"Manifest mismatch: {name}")
        protected[str(path)] = sha
    for path, sha in protected.items():
        V.required(V.sha256_file(path) == sha, f"Input pin mismatch: {path}")

    arms = {}
    clip_checks = []
    empty_hands = {"B_video": 0, "C_hand_image": 0}
    row_count = 0
    for clip in cfg["clips"]:
        cid = clip["clip_id"]
        stored = {}
        arms[cid] = {}
        for tag, alias in (("B_video", "A"), ("C_hand_image", "B")):
            artifact = result["clips"][cid]["artifacts"][tag]
            path = V.safe_result_file(result_dir, artifact["npz"])
            metadata = V.read_json(V.safe_result_file(result_dir, artifact["meta"]))
            V.required(V.sha256_file(path) == metadata["npz_sha256"], "NPZ metadata hash")
            raw = load_npz(path)
            obs = {k: raw["obs_" + k] for k in V.FIELDS}
            V.validate_observation(obs)
            ids = np.arange(0, clip["expected_frames_from_prior"], 2, dtype=np.int64)
            V.required(same_bytes(obs["frame_id"], ids), "Frozen even-frame IDs changed")
            V.required(same_bytes(raw["selected_frame_id"], ids), "Selected frame IDs changed")
            for section in (slice(33, 54), slice(54, 75)):
                empty_hands[tag] += int((~obs["usable"][:, section].any(axis=1)).sum())
            arms[cid][alias] = {"selected": obs}
            stored[tag] = raw
        b, c = stored["B_video"], stored["C_hand_image"]
        old = load_npz(clip["baseline_b_npz"])
        V.required(set(b) == set(old), "B baseline key mismatch")
        V.required(all(same_bytes(b[k], old[k]) for k in b), "B is not byte-identical to old B")
        for k in ("xy_px", "observed", "usable", "conf"):
            V.required(same_bytes(b["obs_" + k][:, :33], c["obs_" + k][:, :33]), "Pose changed: " + k)
        for k in b:
            if k.startswith("decoded_") or k in ("selected_frame_id", "obs_frame_id", "obs_timestamp_us"):
                V.required(same_bytes(b[k], c[k]), "B/C source exposure changed: " + k)
        row_count += len(b["obs_frame_id"])
        clip_checks.append({"clip_id": cid, "rows_per_arm": len(b["obs_frame_id"]),
                            "B_baseline_all_arrays_exact_bytes": True,
                            "BC_pose_all_fields_exact_bytes": True,
                            "BC_decoded_source_hashes_ids_timestamps_exact_bytes": True})

    # Schema-only aliases: old helper A receives new B_video, helper B receives
    # new C_hand_image. No arm's actual inputs or official preprocessing change.
    old_result = V.read_json(cfg["parent_cadence"]["result_path"])
    adapted = {"scoring": {"classes": old_result["scoring"]["classes"], "windows": []}}
    for w in result["windows"]:
        adapted["scoring"]["windows"].append({
            "clip_id": w["clip_id"], "label": w["label"], "start_us": w["start_us"],
            "end_us_exclusive": w["end_us_exclusive"], "window_frame_ids": w["frame_ids"],
            "window_timestamp_us": w["timestamp_us"], "A": w["B_video"], "B": w["C_hand_image"]})
    scored = V.independent_scores(cfg, adapted, arms)
    for item in scored["windows"]:
        item["arm"] = {"A": "B_video", "B": "C_hand_image"}[item["arm"]]
    V.required(scored["forwards"] == 8, "Exactly 8 independent forwards required")

    # Recalculate decisions independently using already checked scores, not runner metrics.
    summary = []
    totals = {"B_correct": 0, "C_correct": 0, "recovered": 0, "regressed": 0}
    classes = adapted["scoring"]["classes"]
    for w in result["windows"]:
        row = {"clip_id": w["clip_id"], "label": w["label"], "frames": len(w["frame_ids"])}
        truth = classes.index(w["label"])
        correct = []
        for tag in ("B_video", "C_hand_image"):
            logits = np.asarray(w[tag]["logits93"], dtype=np.float64)
            order = np.lexsort((np.arange(93), -logits))
            margin = float(logits[truth] - np.max(np.delete(logits, truth)))
            flag = int(order[0]) == truth
            correct.append(flag)
            row[tag] = {"top1": classes[int(order[0])], "correct": flag,
                        "rank": int(np.flatnonzero(order == truth)[0]) + 1, "margin": margin}
        totals["B_correct"] += int(correct[0]); totals["C_correct"] += int(correct[1])
        totals["recovered"] += int(not correct[0] and correct[1])
        totals["regressed"] += int(correct[0] and not correct[1])
        summary.append(row)
    for k, value in totals.items():
        V.required(result["decision"][k] == value, "Decision mismatch: " + k)
    for path, sha in protected.items():
        V.required(V.sha256_file(path) == sha, f"Protected file changed during verification: {path}")
    forbidden = [k for k in sys.modules if k == "mediapipe" or k.startswith("mediapipe.") or k == "av"]
    V.required(not forbidden, "Decoder/MediaPipe imported unexpectedly")
    report = {"status": "PASS", "scope": "Saved observation independent official preprocessing and 8 TCN forwards",
              "sources": {"config_sha256": V.sha256_file(cfg_path), "result_sha256": V.sha256_file(result_path),
                          "helper_sha256": V.sha256_file(HELPER), "verifier_sha256": V.sha256_file(__file__)},
              "pins_checked_before_after": len(cfg["pins"]), "manifest_files_checked": len(manifest),
              "clip_checks": clip_checks, "processed_rows_per_arm": row_count,
              "all21_missing_hand_slots": empty_hands, "total_hand_slots_per_arm": row_count * 2,
              "model": scored, "windows": summary, "decision": totals,
              "all_protected_files_unchanged": True,
              "execution": {"tcn_forwards": 8, "model_loads": 1, "video_decodes": 0,
                            "mediapipe_calls": 0, "camera_calls": 0, "training_runs": 0},
              "limits": ["Two words, known signers, exposed development examples; not independent generalization.",
                         "Image mode changes the hand extraction bundle; coordinate truth and precise causal component are not proved.",
                         "More observed points alone do not prove better coordinates.",
                         "Offline B-then-C timing does not measure actual camera throughput or production accuracy."]}
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# 손 IMAGE 비교 독립 검산", "", "판정: PASS. 저장 관측만으로 정확히 8회 TCN 추론을 독립 실행했습니다.",
             "영상 재해독·MediaPipe·카메라·학습은 모두 0회입니다.", "",
             f"- 6개 영상 B/C 12관측의 원본·출력 해시 확인. 입력핀 {len(cfg['pins'])}개와 결과파일 {len(manifest)}개가 검산 전후 같습니다.",
             "- B는 이전 B의 모든 저장 배열과 바이트 단위로 같습니다. B/C의 pose, 원본 프레임·시각·RGB 해시는 같습니다.",
             "- 고정 원구간, 전체 93개 클래스 순서, 입력 해시, 신체·손 크기 정규화, 744개 점수를 독립 재현했습니다. 최대 점수 차이 0.0.",
             "- 모델의 모든 가중치와 버퍼가 전후 같습니다.", "",
             "| 구간 | VIDEO 1위 / 정답 점수차 | IMAGE 1위 / 정답 점수차 |", "|---|---|---|"]
    for row in summary:
        b, c = row["B_video"], row["C_hand_image"]
        lines.append(f"| {row['clip_id']} ({row['label']}) | {b['top1']} / {b['margin']:+.6f} | {c['top1']} / {c['margin']:+.6f} |")
    lines += ["", f"전체 93개 기준 VIDEO {totals['B_correct']}/4 → IMAGE {totals['C_correct']}/4. 회복 {totals['recovered']}건, 회귀 {totals['regressed']}건.",
              f"손 21점이 전부 미관측인 칸은 {row_count*2}칸 중 B {empty_hands['B_video']}, C {empty_hands['C_hand_image']}입니다. 관측 증가 자체를 좌표 정확성으로 해석하지 않습니다.",
              "", "이 결과는 공개 개발 자료의 한 오답을 실제로 고친 제한된 근거입니다. 손 추출 모드 전체 변경의 효과이며, 좌표의 실제 정확성·새 사람·교차손·실제 웹캠 성능은 입증하지 않습니다.",
              "동일 기계의 순차 오프라인 측정에서 C의 처리 시간이 늘어났으므로 웹에 연결할 경우 실제 처리 간격이 다시 변할 수 있습니다. 현재 결과만으로 기본 설정을 교체하지 않습니다.",
              "", "재현 명령: Python 3.13으로 이 폴더의 independent_verify.py --run 실행. 출력이 이미 있으면 덮어쓰지 않고 중단합니다."]
    with markdown.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "forwards": 8, "decision": totals,
                      "missing_hand_slots": empty_hands, "max_logit_diff": scored["max_abs_logit_difference"]}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] != ["--run"]:
        raise SystemExit("Explicit --run is required; exactly 8 forwards")
    main()
