"""Independent local-only audit of one authorized completed session.

Reads saved JSON plus raw timestamp/frame-id/observed/usable arrays. No production
imports, restore call, inference, model reads, camera access or image decoding.
All writes stay beside this verifier; original session bytes are hashed twice.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
LAB = OUT.parents[2]
SESSION_ID = "ses_9fca646a923a"
SESSION = LAB / "runs" / SESSION_ID
MODEL_SHA = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"
TRANSFORM_SHA = "6ae29cea6ae030d8567a9c79749cf8717738a1b2c2b73eaa93d4b38fd539ac07"
MODEL_KEY = "tcn_control_83212868_handgap"
MAX_ENDPOINT_US = 66668
HANDS = {"left": slice(33, 54), "right": slice(54, 75)}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def inventory():
    return {p.relative_to(SESSION).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
            for p in sorted(SESSION.rglob("*")) if p.is_file()}


def stats(values):
    vals = [int(v) for v in values]
    return {"count": len(vals), "min_us": min(vals) if vals else None,
            "median_us": float(np.median(vals)) if vals else None,
            "max_us": max(vals) if vals else None,
            "mean_us": float(np.mean(vals)) if vals else None}


def main():
    before = inventory()
    write("SESSION_HASHES_BEFORE.json", before)
    checks = []

    def check(name, passed):
        checks.append({"check": name, "pass": bool(passed)})

    session = read(SESSION / "session.json")
    reviews = [json.loads(line) for line in (SESSION / "trial_reviews.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    latest = {}
    for review in reviews:
        if review["revision"] > latest.get(review["trial_id"], {}).get("revision", 0):
            latest[review["trial_id"]] = review
    tids = session["trial_ids"]
    check("session closed by client, camera false", session["state"] == "CLOSED"
          and session["closed_reason"] == "CLOSED_BY_CLIENT" and session["camera"] is False)
    check("exactly three result files, trial ids and reviews", len(tids) == len(set(tids)) == 3
          and {p.stem.removesuffix(".result") for p in (SESSION / "trials").glob("*.result.json")}
          == set(tids) == set(latest) and len(reviews) == 3)
    check("expected confirmed words in capture order",
          [latest[tid]["actual_word_id"] for tid in tids] == ["MUL", "NEO", "GAMSAHAMNIDA"])
    trials = []
    for tid in tids:
        result = read(SESSION / "trials" / f"{tid}.result.json")
        sidecar = read(SESSION / "trials" / f"{tid}.sidecar.json")
        raw_path = SESSION / "trials" / f"{tid}.npz"
        raw_sha = before[f"trials/{tid}.npz"]["sha256"]
        with np.load(raw_path, allow_pickle=False) as saved:
            usable = saved["usable"].copy()
            observed = saved["observed"].copy()
            ts = saved["timestamp_us"].copy()
            frame_ids = saved["frame_id"].copy()
        check(f"{tid}: raw hash agrees with sidecar and result",
              raw_sha == sidecar["observation_sha256"] == result["observation_sha256"])
        check(f"{tid}: saved mask/time/frame shape and ordering valid",
              usable.ndim == 2 and usable.shape[1] == 75 and usable.dtype == np.bool_
              and observed.shape == usable.shape and observed.dtype == np.bool_
              and ts.shape == frame_ids.shape == (len(usable),)
              and ts.dtype == np.int64 and np.all(np.diff(ts) > 0)
              and np.all(np.diff(frame_ids) > 0))
        hand_rows = np.flatnonzero(usable[:, 33:75].any(axis=1))
        lo, hi = int(hand_rows[0]), int(hand_rows[-1])
        span = {"source_row_first": lo, "source_row_last": hi, "span_frames": hi-lo+1,
                "timestamp_us_first": int(ts[lo]), "timestamp_us_last": int(ts[hi]),
                "source_frame_id_first": int(frame_ids[lo]),
                "source_frame_id_last": int(frame_ids[hi])}
        missing, run_rows, partial, observed_counts = [], [], {}, {}
        for side, sl in HANDS.items():
            counts = usable[:, sl].sum(axis=1)
            partial[side] = int(((counts[lo:hi+1] > 0) & (counts[lo:hi+1] < 21)).sum())
            observed_counts[side] = int(usable[lo:hi+1, sl].sum())
            absent = counts == 0
            cursor = lo
            while cursor <= hi:
                if not absent[cursor]:
                    cursor += 1
                    continue
                start = cursor
                while cursor + 1 <= hi and absent[cursor + 1]:
                    cursor += 1
                end = cursor
                run_rows.append({"side": side, "first_source_row": start,
                                 "last_source_row": end, "length_frames": end-start+1,
                                 "first_timestamp_us": int(ts[start]),
                                 "last_timestamp_us": int(ts[end])})
                for i in range(start, end + 1):
                    interior = lo < i < hi
                    prev_count = int(counts[i-1]) if i > 0 else None
                    next_count = int(counts[i+1]) if i+1 < len(ts) else None
                    endpoint_dt = int(ts[i+1]) - int(ts[i-1]) if 0 < i < len(ts)-1 else None
                    full_neighbours = prev_count == next_count == 21
                    isolated = interior and full_neighbours
                    time_ok = endpoint_dt is not None and 0 < endpoint_dt <= MAX_ENDPOINT_US
                    reasons = []
                    if not interior:
                        reasons.append("ACTIVE_SPAN_BOUNDARY")
                    if not full_neighbours:
                        reasons.append("NEIGHBOUR_NOT_ALL_21_USABLE")
                    if endpoint_dt is None or endpoint_dt <= 0:
                        reasons.append("NO_POSITIVE_ENDPOINT_INTERVAL")
                    elif not time_ok:
                        reasons.append("ENDPOINT_INTERVAL_EXCEEDS_66668_US")
                    missing.append({"side": side, "source_row": i,
                                    "source_frame_id": int(frame_ids[i]), "timestamp_us": int(ts[i]),
                                    "whole_hand_usable_points": 0,
                                    "raw_observed_points": int(observed[i, sl].sum()),
                                    "absent_run_first_row": start, "absent_run_last_row": end,
                                    "absent_run_frames": end-start+1,
                                    "interior_to_original_active_span": interior,
                                    "previous_usable_points": prev_count,
                                    "next_usable_points": next_count,
                                    "previous_timestamp_us": int(ts[i-1]) if i > 0 else None,
                                    "next_timestamp_us": int(ts[i+1]) if i+1 < len(ts) else None,
                                    "endpoint_interval_us": endpoint_dt,
                                    "isolated_with_two_full_neighbours": isolated,
                                    "time_gate_pass": time_ok,
                                    "eligible": isolated and time_ok, "rejection_reasons": reasons})
                cursor += 1
        rc = result["research_comparison"]
        original = rc["hand_gap_comparison"]["original"]
        audit = rc["hand_gap_comparison"]["logit_audit"]
        creation = result["research_model"]
        fixed = rc["identity_fixed_at_trial_creation"]
        tx = rc["input_transform"]
        provenance = tx["provenance"]
        raw_logits = np.asarray(audit["original_coordinates"], dtype=np.float64)
        repaired_logits = np.asarray(audit["short_gap_restored"], dtype=np.float64)
        labels = audit["label_order"]
        review = latest[tid]
        actual_label = creation["trained_class_set"]["project_word_to_model_label"][review["actual_word_id"]]
        raw_top = labels[int(raw_logits.argmax())]
        repaired_top = labels[int(repaired_logits.argmax())]
        eligible = [m for m in missing if m["eligible"]]
        isolated = [m for m in missing if m["isolated_with_two_full_neighbours"]]
        expected_points = {"observed": observed_counts, "estimated": {"left": 0, "right": 0},
                           "observed_total": sum(observed_counts.values()), "estimated_total": 0}
        record = {"transform_id": "wis_hand_gap_restore_v1", "source_sha256": TRANSFORM_SHA,
                  "span": span, "fills": []}
        expected_prov_sha = hashlib.sha256(json.dumps(record, sort_keys=True,
                                                     separators=(",", ":")).encode("utf-8")).hexdigest()
        check(f"{tid}: session, trial, self-report identity", result["trial_id"] == review["trial_id"] == tid
              and result["session_id"] == review["session_id"] == SESSION_ID
              and review["source"] == "participant_self_report" and review["is_expert_verification"] is False)
        check(f"{tid}: frozen checkpoint and selection identities", all(
            block["model_key"] == MODEL_KEY and block["checkpoint_sha256"] == MODEL_SHA
            for block in (creation, fixed, rc, original)))
        check(f"{tid}: frozen transform source and original arm", all(
            block["input_transform"]["loaded_source_sha256"] == TRANSFORM_SHA
            and block["input_transform"]["pinned_source_sha256"] == TRANSFORM_SHA
            for block in (creation, fixed, rc)) and original["input_transform"]["transform_id"] == "none"
              and original["arm"] == "original_coordinates" and rc["arm"] == "short_gap_restored")
        check(f"{tid}: full93 audit and labels", len(labels) == len(set(labels)) == 93
              and labels == creation["trained_class_set"]["labels"]
              and raw_logits.shape == repaired_logits.shape == (93,)
              and np.isfinite(raw_logits).all() and np.isfinite(repaired_logits).all()
              and original["scopes"]["full"]["candidate_count"] == rc["scopes"]["full"]["candidate_count"] == 93)
        check(f"{tid}: argmax matches display and same actual-label report",
              raw_top == original["scopes"]["full"]["top1"] == actual_label
              and repaired_top == rc["scopes"]["full"]["top1"] == actual_label)
        check(f"{tid}: independent eligibility empty and provenance consistent", not eligible
              and provenance["status"] == "IDENTITY_NO_FILL" and provenance["identity"] is True
              and provenance["fills"] == [] and provenance["filled_hand_frames"] == {"left": 0, "right": 0, "total": 0}
              and provenance["source_active_span"] == span and provenance["hand_points"] == expected_points
              and provenance["provenance_sha256"] == expected_prov_sha)
        check(f"{tid}: identical scores and support, no performance verdict",
              np.array_equal(raw_logits, repaired_logits) and audit["max_abs_difference"] == 0
              and original["support"] == rc["support"] and result["accepted"] is None
              and result["execution_assessment"] == "UNCONFIGURED")
        supported = int(usable[lo:hi+1, 33:75].any(axis=1).sum())
        check(f"{tid}: real support agrees", rc["support"]["span_frames"] == hi-lo+1
              and rc["support"]["observed_hand_frames"] == supported)
        trials.append({"trial_id": tid, "actual_word_id": review["actual_word_id"], "actual_label": actual_label,
                       "raw_top1": raw_top, "repaired_top1": repaired_top,
                       "max_abs_logit_difference": float(np.max(np.abs(raw_logits-repaired_logits))),
                       "raw_observation_sha256": raw_sha, "frames_total": len(ts), "source_active_span": span,
                       "whole_recording_step": stats(np.diff(ts)), "active_span_step": stats(np.diff(ts[lo:hi+1])),
                       "observed_hand_frames": supported, "partial_hand_frames_by_side": partial,
                       "whole_hand_absent_slots_by_side": {side: sum(m["side"] == side for m in missing) for side in HANDS},
                       "whole_hand_absent_runs": run_rows, "whole_hand_absent_slots": missing,
                       "isolated_with_full_neighbours_count": len(isolated),
                       "isolated_with_full_neighbours": isolated, "eligible_count": len(eligible),
                       "saved_filled_hand_frames": provenance["filled_hand_frames"],
                       "end_reason": result["end_reason"], "capture_status": result["diagnostic_capture"]["status"]})
    after = inventory()
    write("SESSION_HASHES_AFTER.json", after)
    check("all original session files unchanged", before == after)
    summary = {"status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
               "session_id": SESSION_ID, "session_state": session["state"],
               "verified_local_at": datetime.now().isoformat(),
               "verifier_sha256": sha(Path(__file__)), "original_files_hashed": len(before),
               "original_files_unchanged": before == after, "new_model_inference": 0,
               "camera_or_device_calls": 0, "image_or_video_decoding": 0, "parameters_changed": 0,
               "rule": {"whole_hand_points": 21, "interior_one_frame_only": True,
                        "both_neighbours_all_usable": True, "max_endpoint_interval_us": MAX_ENDPOINT_US,
                        "source_mask": "raw usable, hand slots33:54/54:75, original any-usable-hand active span"},
               "mask_semantics": "All-absent usable slots are not proof of physical occlusion, missing fingers, or bad signing; an intentionally unused hand also qualifies as absent in this inventory.",
               "full93_matches_self_report": sum(t["raw_top1"] == t["repaired_top1"] == t["actual_label"] for t in trials),
               "total_trials": len(trials), "eligible_hand_frames": sum(t["eligible_count"] for t in trials),
               "isolated_full_neighbour_hand_frames": sum(t["isolated_with_full_neighbours_count"] for t in trials),
               "verdict": "connection/save/identity success; all no-fill, no live repair effect demonstrated; self-report agreement not execution-quality or generalization evidence",
               "checks": checks, "trials": trials}
    write("INDEPENDENT_LIVE_AUDIT.json", summary)
    lines = ["# 실제 촬영3회 독립 저장·보완조건 감사", "", f"판정: **{summary['status']}**. 세션 `{SESSION_ID}`은 사용자 종료·카메라 해제 상태다.", "",
             "원본/보완 전체93점수에서 직접 계산한 1순위는 물·너·감사합니다로 본인확인3/3과 일치한다. 세 건 모두 전체93점수가 정확히 같고 최대차는0.0이다. 모델832·보완코드6ae29·생성시고정식별·원본NPZ/sidecar/result SHA 일치.", "",
             "| 본인확인 | 원본→보완 전체93 | 전체/활성 프레임 | 활성 간격 중앙값(ms) | 전후21점이 유효한 단일누락 | 시간조건까지 충족 |", "|---|---|---:|---:|---:|---:|"]
    for t in trials:
        lines.append(f"| {t['actual_label']} | {t['raw_top1']}→{t['repaired_top1']} | {t['frames_total']}/{t['source_active_span']['span_frames']} | {t['active_span_step']['median_us']/1000:.3f} | {t['isolated_with_full_neighbours_count']} | {t['eligible_count']} |")
    lines += ["", "원본 usable의 손33:54·54:75를 직접 검사했다. 보완함수나 모델은 호출하지 않았다. 시간 조건은 양끝차66,668us 이하로 그대로 유지했다. 모든 전후 관측 조건 충족 단일누락의 정확한 시각은 아래와 같다.", "",
              "| 표현 | 손 | 원본 행 / frame_id | 앞 시각(us) | 누락 시각(us) | 뒤 시각(us) | 양끝차(us) | 판정 |", "|---|---|---|---:|---:|---:|---:|---|"]
    for t in trials:
        for m in t["isolated_with_full_neighbours"]:
            lines.append(f"| {t['actual_label']} | {m['side']} | {m['source_row']} / {m['source_frame_id']} | {m['previous_timestamp_us']} | {m['timestamp_us']} | {m['next_timestamp_us']} | {m['endpoint_interval_us']} | {'보완가능' if m['eligible'] else '66,668us 초과'} |")
    lines += ["", "두 손의 완전비유효 슬롯·연속구간·부분손 프레임 수와 각 슬롯의 거절 이유는 INDEPENDENT_LIVE_AUDIT.json에 전수 기록했다. 유효점이 없는 손은 실제 가림/검출실패/비사용손 중 무엇인지 이 검사만으로 구별할 수 없다. 시간 간격의 물리적 원인도 판단하지 않았다.", "",
              "**세 건 모두 보완0·추정점0이다.** 실제 촬영→동일시도 원본/보완결과 저장→본인확인 연결은 확인됐지만, 실제 촬영에서 보완이 발동하거나 인식을 회복한다는 효과는 검증되지 않았다. 본인확인은 수행 정확성의 전문가 판정이나 새 사람 일반화 근거가 아니다.", "",
              f"검사 전후 원본 세션파일 {len(before)}개 SHA256/크기 동일: {before == after}. 원본 수정0, 새추론0, 장치호출0, 영상/사진 디코딩0, 파라미터변경0. 결과물은 이 폴더에만 작성했다.", "",
              "근거: SESSION_HASHES_BEFORE.json, SESSION_HASHES_AFTER.json, INDEPENDENT_LIVE_AUDIT.json, verify_live_session.py."]
    (OUT / "RESULT_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "checks": len(checks),
                      "failures": [c for c in checks if not c["pass"]],
                      "original_files_hashed": len(before), "trials": [
                          {k: t[k] for k in ("actual_label", "frames_total", "active_span_step",
                           "whole_hand_absent_slots_by_side", "partial_hand_frames_by_side",
                           "isolated_with_full_neighbours_count", "eligible_count")}
                          for t in trials]}, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
