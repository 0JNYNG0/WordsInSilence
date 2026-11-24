"""Saved NIKL18 result verification only; explicit --run, at most 36 forwards.

No candidate module is imported. The result schema is read from nikl18_pair source,
but official preprocessing, ranking, counts, and the reference model are independent.
"""
from pathlib import Path
import json
import sys
import numpy as np
from independent_reference import (FIELDS, ReferenceScorer, array_hash, byte_equal,
                                   compare_raw, read_observation, require, sha256, validate_observation)

HERE = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe_file(folder, name):
    require(isinstance(name, str) and Path(name).name == name, "Result leaf path required")
    path = folder / name
    require(path.is_file(), "Missing result artifact: " + name)
    return path


def compare_scale(actual, saved, name):
    require(isinstance(saved, dict) and saved.get("shape") == [], "Scale schema: " + name)
    require(actual == saved["value"], "Scale mismatch: " + name)
    require(saved["nonfinite_count"] == int(actual is None), "Scale null count: " + name)
    require(saved["status"] == ("finite" if actual is not None else "nonfinite_serialized_null_missing"),
            "Scale missing status: " + name)


def check_input(prepared, actual, saved, obs, label):
    x, pad = prepared
    for key in ("span", "timestamp_us", "x_sha256", "x_shape", "x_dtype"):
        require(actual[key] == saved[key], f"{label}: input {key}")
    require(actual["frame_ids"] == saved["frame_id"], f"{label}: frame IDs")
    require(array_hash(pad) == saved["pad_sha256"], f"{label}: pad hash")
    lo, hi = actual["span"]
    # present contains the common body9 + hands42; other native slots stay absent.
    body = (0, 11, 12, 13, 14, 15, 16, 23, 24)
    source_usable = obs["usable"][lo:hi+1]
    count = int(source_usable[:, body].sum() + source_usable[:, 33:75].sum())
    require(saved["present_shape"] == [hi-lo+1, 67] and saved["present_count"] == count,
            f"{label}: native present summary")
    compare_scale(actual["body_scale"], saved["body_scale"], label + " body")
    require(set(actual["hand_scales"]) == set(saved["hand_scales"]), "Hand scale sides")
    for side in actual["hand_scales"]:
        compare_scale(actual["hand_scales"][side], saved["hand_scales"][side], label + " " + side)


def main():
    out_json, out_md = HERE / "INDEPENDENT_RESULT.json", HERE / "INDEPENDENT_RESULT.md"
    require(not out_json.exists() and not out_md.exists(), "Exclusive result already exists")
    cfg_path = HERE / "PREPARED_CONFIG.json"
    cfg = read_json(cfg_path)
    folder = Path(cfg["output_dir"])
    result_path, manifest_path = folder / "RESULT.json", folder / "MANIFEST.json"
    result = read_json(result_path)
    require(result["status"] == "COMPLETED_PENDING_INDEPENDENT_VERIFICATION", "No completed main result")
    require(result["evaluation_selection"] == "all_stored_obs", "Not original full-frame evaluation")
    require(result["hashes_before"] == result["hashes_after"], "Main-run source hashes changed")
    main_state = result["model_state"]
    require(main_state["unchanged"] is True and main_state["payload_state_sha256"] == main_state["before"] == main_state["after"],
            "Main-run model state invariance failed")
    require(main_state["payload_state_sha256"] == result["model"]["payload_state_sha256"], "Main model identity")
    manifest = read_json(manifest_path)["files"]
    protected = dict(cfg["pins"])
    protected.update({str(cfg_path): sha256(cfg_path), str(result_path): sha256(result_path),
                      str(manifest_path): sha256(manifest_path), str(Path(__file__)): sha256(__file__),
                      str(HERE / "independent_reference.py"): sha256(HERE / "independent_reference.py")})
    provenance = result["provenance"]
    for filename, expected in (("nikl18_pair.py", provenance["runner_sha256"]),
                               ("test_nikl18.py", provenance["test_sha256"]),
                               ("TEST_RECEIPT.json", provenance["receipt_sha256"]),
                               ("SCHEMA_EXAMPLE.json", result["hashes_before"]["schema_doc"])):
        protected[str(HERE / filename)] = expected
    for name, expected in manifest.items():
        protected[str(safe_file(folder, name))] = expected
    for path, expected in protected.items():
        require(sha256(path) == expected, "Pinned file mismatch: " + path)
    require(result["provenance"]["config_sha256"] == sha256(cfg_path), "Main config provenance")

    clips = cfg["clips"]
    ids = [c["clip_id"] for c in clips]
    require(len(ids) == len(set(ids)) == 18 and ids == sorted(ids), "Frozen 18 IDs/order")
    source = read_json(cfg["source_manifest_path"])
    ready = sorted(c["clip_id"] for c in source["clips"] if c["readiness"] in ("READY", "NEEDS_REVIEW"))
    require(ready == ids, "Source18 selection changed")
    hist = read_json(cfg["historical_evaluation_path"])
    historical = {r["id"]: r for r in hist["rows"] if r["group"] == "public" and r["condition"] == "intact"}
    require(set(historical) == set(ids), "Historical18 IDs")
    require(all(r["models"]["control"]["full93"]["correct"] for r in historical.values()), "Historical B18/18")
    rows = result["rows"]
    require([r["clip_id"] for r in rows] == ids, "Result18 IDs/order")
    loaded, raw_checks = {}, []
    for clip, row in zip(clips, rows):
        cid, n = clip["clip_id"], clip["expected_frames"]
        require(row["label"] == clip["label"] == historical[cid]["true"], "Label provenance")
        require(row["word_id"] == clip["word_id"] and row["n_frames"] == n, "Clip metadata")
        require(row["evaluation_selection"] == "all_stored_obs", "Per-clip frame selection")
        cache = read_observation(clip["baseline_npz_path"], clip["baseline_npz_sha256"])
        validate_observation(cache, n)
        pair, dec = {}, {}
        for arm in ("B_video", "C_hand_image"):
            artifact = row["persisted"][arm]
            path, meta_path = safe_file(folder, artifact["npz"]), safe_file(folder, artifact["meta"])
            meta = read_json(meta_path)
            require(meta["npz_sha256"] == sha256(path), "Output observation hash")
            require(meta["clip_id"] == cid and meta["tag"] == arm, "Output observation identity")
            require(meta["stored_rows"] == "all decoded rows", "Stored observation is not full cadence")
            obs = read_observation(path)
            validate_observation(obs, n)
            pair[arm] = obs
            with np.load(path, allow_pickle=False) as z:
                dec[arm] = {k: z[k].copy() for k in z.files if k.startswith("decoded_")}
            require(np.array_equal(dec[arm]["decoded_frame_id"], np.arange(n)), "Decoded full-frame IDs")
            require(byte_equal(dec[arm]["decoded_timestamp_us"], obs["timestamp_us"]), "Decoded original timestamps")
            require(dec[arm]["decoded_processed"].all(), "Not all RGB frames processed")
        compare_raw(pair["B_video"], pair["C_hand_image"], cache)
        require(set(dec["B_video"]) == set(dec["C_hand_image"]), "Decoded metadata keys differ")
        require(all(byte_equal(dec["B_video"][k], dec["C_hand_image"][k]) for k in dec["B_video"]),
                "B/C RGB/PTS/frame exposure differs")
        loaded[cid] = {"B": pair["B_video"], "C": pair["C_hand_image"], "cache": cache}
        raw_checks.append({"clip_id": cid, "frames": n, "B_cache_7fields_bytes_exact": True,
                           "BC_pose_bytes_exact": True, "BC_all_decoded_metadata_exact": True})

    # Only after all raw/cache/source gates: one model load and official preparations.
    scorer = ReferenceScorer(cfg)
    require(result["classes"] == scorer.classes, "93-class order differs")
    require(scorer.payload_state_sha256 == main_state["payload_state_sha256"], "Independent model differs from main model")
    require(result["model"]["checkpoint_sha256"] == cfg["scoring"]["checkpoint_sha256"], "Main checkpoint identity")
    prepared, feature_rows, available_c = {}, [], 0
    for clip, row in zip(clips, rows):
        cid = clip["clip_id"]
        prepared[cid] = {}
        cache_x, cache_detail = scorer.prepare(loaded[cid]["cache"])
        require(cache_x is not None, "Historical cache has no active hand span")
        rec = {"clip_id": cid}
        for arm in ("B", "C"):
            pair, detail = scorer.prepare(loaded[cid][arm])
            prepared[cid][arm] = pair
            if pair is None:
                require(arm == "C" and row[arm]["status"] == "UNAVAILABLE_NO_ACTIVE_HANDS", "Unexpected unavailable")
                require("score" not in row[arm] and "logits" not in row[arm], "Unavailable has fabricated result")
            else:
                check_input(pair, detail, row[arm]["input"], loaded[cid][arm], cid + arm)
                if arm == "B":
                    require(detail == cache_detail and all(byte_equal(a, b) for a, b in zip(pair, cache_x)),
                            "B/cache official feature mismatch")
                    require(detail["x_sha256"] == row[arm]["cache_input_x_sha256"], "Cached B input hash")
                else:
                    require(row[arm]["status"] == "AVAILABLE", "C available status")
                    available_c += 1
            rec[arm] = detail
        require(rec["B"]["status"] == "AVAILABLE", "B unavailable is technical failure")
        require(row["span_change"]["B_span"] == rec["B"]["span"] and row["span_change"]["C_span"] == rec["C"]["span"],
                "Saved span change incorrect")
        feature_rows.append(rec)

    score_rows, max_delta = {cid: {"clip_id": cid} for cid in ids}, 0.0
    # Mirror main arithmetic/order (B18 then all available C), independently score and rank.
    for arm in ("B", "C"):
        for clip, row in zip(clips, rows):
            cid = clip["clip_id"]
            pair = prepared[cid][arm]
            if pair is None:
                score_rows[cid][arm] = {"status": "UNAVAILABLE_NO_ACTIVE_HANDS", "correct": None}
                continue
            logits, score = scorer.score(pair, clip["label"])
            saved = np.asarray(row[arm]["logits"], dtype=np.float64)
            require(saved.shape == (93,) and np.array_equal(logits, saved), "93 independent logits mismatch")
            max_delta = max(max_delta, float(np.max(np.abs(logits-saved))))
            require(score == row[arm]["score"], "Independent full93 ranking differs")
            if arm == "B":
                expected = historical[cid]["models"]["control"]["full93"]["pred"]
                require(score["top1"] == expected == clip["label"], "Historical B prediction not reproduced")
            score_rows[cid][arm] = dict(score, status="AVAILABLE")
    execution = scorer.finalize(available_c)
    scores = [score_rows[cid] for cid in ids]
    b_ok = sum(r["B"]["correct"] for r in scores)
    c_ok = sum(r["C"]["correct"] is True for r in scores)
    unavailable = [r["clip_id"] for r in scores if r["C"]["status"] == "UNAVAILABLE_NO_ACTIVE_HANDS"]
    regressions = [r["clip_id"] for r in scores if r["B"]["correct"] and r["C"]["correct"] is False]
    decision = {"denominator": 18, "B_correct": b_ok, "C_correct": c_ok, "C_unavailable": len(unavailable),
                "C_unavailable_clips": unavailable, "C_regressions": regressions, "expected_forwards": 18+available_c,
                "code": "PASS_FROZEN_MODE_REGRESSION" if b_ok == c_ok == 18 and not unavailable else "FAIL_FROZEN_MODE_REGRESSION"}
    require(decision == result["decision"], "Independent decision mismatch")
    require(result["counters"]["tcn_forwards"] == 18+available_c, "Main actual forward count")
    for path, expected in protected.items():
        require(sha256(path) == expected, "Protected input/output changed during verification: " + path)
    report = {"status": "PASS", "decision": decision, "execution": execution,
              "input_pins_checked": len(cfg["pins"]), "result_manifest_files": len(manifest),
              "source_hashes": {"config": sha256(cfg_path), "result": sha256(result_path),
                                "verifier": sha256(__file__), "reference": sha256(HERE/"independent_reference.py")},
              "protected_files_unchanged": True, "raw_checks": raw_checks,
              "main_model_state_record_verified": main_state,
              "features": feature_rows, "scores": scores, "max_abs_logit_difference": max_delta,
              "historical_score_arithmetic": "Prior float64 affine; comparison only input identity and full93 top1, not logit bit equality.",
              "limits": ["Training demonstrations; preservation check, not new-person generalization.",
                         "Per-arm span and scales may change. Coordinate physical truth is not established.",
                         "No deployment or webcam throughput claim."]}
    with out_json.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ["# NIKL18 손 IMAGE 회귀 독립 검산", "", "독립 재현 검사: PASS.",
             f"실제 독립 추론 {execution['tcn_forwards']}회, 모델 로드1회. 새 영상해독·MediaPipe·카메라·학습0회.",
             f"전체93 기준 B {b_ok}/18, C {c_ok}/18, C 미가용{len(unavailable)}개. 후보 판정: {decision['code']}.",
             f"93점수 {execution['tcn_forwards']*93}개를 독립 재현했고 최대차이는{max_delta}입니다.",
             "B 원관측·공식입력은 기존 cache와 정확히 같고 B/C 몸 관측·원래 프레임·PTS·RGB hash가 같습니다.",
             "18개 분모 유지, 클래스 순서·span·입력 hash·몸/손 scale·순위·margin·모델전체상태·입력핀 보존을 검산했습니다.",
             "", "| 단어 | B 1위 / 정답 점수차 | C 1위 / 정답 점수차 |", "|---|---|---|"]
    for clip, r in zip(clips, scores):
        b, c = r["B"], r["C"]
        ct = "미가용" if c["correct"] is None else f"{c['top1']} / {c['true_vs_best_other_margin']:+.6f}"
        lines.append(f"| {clip['label']} | {b['top1']} / {b['true_vs_best_other_margin']:+.6f} | {ct} |")
    lines += ["", "이 검사는 이미 학습한 시범18의 보존 검사입니다. 통과하더라도 새 사람·교차손·실제 웹캠 성능이나 손 좌표의 물리적 정확성을 입증하지 않습니다.",
              "과거832의 점수는 float64 affine 산술이므로 과거점수 bit 일치는 요구하지 않았으며, 이번 본실행 full-forward와 독립검산은 동일 산술로 정확 대조했습니다."]
    with out_md.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print(json.dumps({"status": "PASS", "decision": decision, "execution": execution, "max_abs_logit_difference": max_delta}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] != ["--run"]:
        raise SystemExit("Explicit --run required; saved result only, at most 36 independent forwards")
    main()
