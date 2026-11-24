from pathlib import Path
import hashlib
import importlib.util
import json

U = Path(__file__).resolve().parent
O = U.parent


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert not (U / "FINAL.json").exists()
    report = read(U / "SAVED_PUBLIC_COMPARISON.json")
    verified = read(U / "SAVED_PUBLIC_COMPARISON_VERIFIED.json")
    review = read(U / "COMPARATOR_CODE_REVIEW.json")
    wrapper = read(U / "RUNTIME_WRAPPER_REVIEW.json")
    nxt = read(U / "NEXT_TIME_BINDING_REVIEW.json")
    queue = read(U / "CONTEXT_REVIEW_QUEUE.json")
    assert queue["clips"] == 12 and queue["claims"] == 39 and queue["witness_queries"] == 129
    assert len({(r["clip_id"], f) for r in queue["rows"] for c in r["claims"] for f in c["witness_frames"]}) == 86
    for path, digest in queue["pins"].items():
        assert sha(path) == digest, path
    assert all(p["event_context"] is None for r in queue["rows"] for c in r["claims"] for p in c["context_labels"])
    assert report["cases"] == verified["cases"] == 16
    assert verified["original_native_points"] == 185
    assert report["point_states"] == verified["point_states"]
    assert review["checks"] == verified["checks"] == 1828
    for obj in [review, wrapper]:
        for p, digest in obj["pins"].items():
            assert sha(p) == digest, p
    evidence = [
        "SOURCE_FEATURE_CONSTRAINT_AUDIT.json", "PHYSICAL_GUIDE_BINDING_AUDIT.json",
        "NEXT_IMPLEMENTATION_REVIEW.json", "COMPARISON_CONTRACT.json",
        "COMPARATOR_CHECKS.json", "SAVED_PUBLIC_COMPARISON.json",
        "SAVED_PUBLIC_COMPARISON_VERIFIED.json", "COMPARATOR_CODE_REVIEW.json",
        "RUNTIME_INPUT_CHECKS.json", "RUNTIME_WRAPPER_REVIEW.json",
        "NEXT_TIME_BINDING_REVIEW.json", "CONTEXT_REVIEW_QUEUE.json", "RESULT_KO.txt",
    ]
    final = dict(status="SOURCE_FEATURE_COMPARATOR_IMPLEMENTED_AND_SAVED_PUBLIC_VALIDATED",
        implementer="Codex root", source_reviewer="Codex egypt_source_binding_review",
        physical_and_code_verifier="Codex predicate_contract_review",
        learning_direction_reviewer="Codex retention_direction_review",
        independent_KSL_expert_validation=False, whole_goal_complete=False,
        previous_goal_turn="no progress: answered user's scope question only; revalidated current artifacts and completed this implementation",
        source_clauses=11, direct_center_labels_audited=108, later_FIT_records_reused=550,
        public_saved_cases=16, native_points=185, point_states=report["point_states"],
        saved_verification_checks=verified["checks"], behavior_checks=28,
        independent_synthetic_checks=len(review["synthetic_checks"]), new_input_checks=20,
        comparison_only_seconds_median=report["comparison_seconds_median"],
        initial_time_and_schema_failures_preserved="guide_comparison_initial.py and COMPARATOR_CODE_REVIEW.json",
        new_training=0, new_model_inference=0, old_scores_recomputed=False,
        current_web_changed=False, C_T3F_changed=False, private_mentor_data_used=False,
        camera_calls=0, downloads=0, external_transmission=0, git_changes=False,
        accuracy_gain_established=False, end_to_end_latency_measured=False,
        next_review_status=nxt["status"],
        context_review_prepared=dict(clips=12, claims=39, queries=129, unique_clip_frames=86,
            labels_assigned=0, first_sheet_visually_inspected=1, large_images_opened=0),
        evidence={str(U / name): sha(U / name) for name in evidence})
    (U / "FINAL.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    spec = importlib.util.spec_from_file_location("current_work_sync", O / "word100_learned_flow_role_20261102/sync_flow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.U = U
    summary = (
        "전체 학습형 피드백 목표는 active이며 미완료다. 직전 질문 설명 turn은 no-progress였고 이번에는 AE를 실제 구현했다. "
        "출처11문구/기존108직접center/후속550FITtyped 근거를 감사하고 기존 학습 관측과 설명의 부분조건을 대조하는 공통 비교기를 추가했다. "
        "관측과 기대값, DTW시점 제안, 표현 변이 적용성을 분리한다. 낮은p 한점은 수어오류가 아니며 미검출·추정·짧은반대근거를 유지한다. "
        "공개16저장조건에 새필드만 단회 부착, 기존 모든출력/원시간/185point 보존: compatible150,unobserved25,estimate3,counter7. "
        "3개의 lowgeometry negative는 새필드에서 이미지시간추정으로 구분했다. root28검사와 별도1828저장검산+20합성검사 PASS. "
        "NaN시간/정수절삭/비boolean/출처URL검증 오류는 최초코드 보존 후 수정했다. 새 RuntimeGuideComparator는 word/videoSHA/framecount/중복clause를 검사; "
        "기존16 preflight+4오류입력20검사, 별도24artifact검토 PASS. 완료16재부착 없음. "
        "새학습/모델추론/기존점수재채점/웹적용0. 비교부만중앙값0.006초는 전체지연아님. 새정확도향상아니므로 HTML생략. "
        "C/T3F/8804/8808/8809파일 보존, 사용자·멘토/H14/15/ID16접근0. 동일Codex검산이며 독립KSL전문가정답아님. "
        "기존 AB/AC의 동작충분성실패도 해결로 바꾸지 않는다. 원시점차이가 해당설명의 적용시점인지 직접시간맥락 검토가 다음 핵심이다. "
        "다음12FITsheet(4표현×ID01/04/07),39claim/129contextquery(고유clip-frame86) 검토큐 준비완료, "
        "과거support/y/판정문구는 큐에서 제외했다. 새맥락라벨0,root0739ID01sheet사전열람1은12상한에 포함한다."
    )
    next_action = (
        "AE NEXT_TIME_BINDING_REVIEW.json과 준비완료 CONTEXT_REVIEW_QUEUE.json(12FIT/39claim/129query)을 먼저 읽고 Q BINDING_WITNESSES117 및 기존36 chronological sheet에서 "
        "FIT공개자료만으로 source-action event-context를 준비한다. 현재 physical supported/counter/unknown과 모델확률은 새맥락정답으로 복사하지 않는다. "
        "손모양이 맞아서 해당단계라고 정하거나, 다르다는 이유로 비교범위밖으로 밀어내는 순환을 금지. "
        "정확point/ordered witness만, 단계전체확장/DTW정답승격0. 전환·다른맥락·불명확을 보존하고 새교정정답/전문가의미오류로 부르지 않는다. "
        "검토지시의 한정 batch와 완료기준으로 직접이미지를 확인한 뒤 추가학습 필요성 및 기존학습prototype 연결을 결정한다. "
        "새base 통합은 RuntimeGuideComparator.attach만 사용하고 실제영상SHA/NPZ원축계보 검증, list/NumPy재사용축, identity미첨부freshbase 조건을 유지한다. "
        "AE16완료결과/AB·AC완료학습평가 재실행·재채점 금지. 현재 comparator는 웹 미적용이며 전체목표 미완료; "
        "지원단어 실제새입력/중요차이/정상허용/관측부족/전체지연과 분리평가까지 계속한다."
    )
    module.sync("출처 설명과 학습 관측의 시점별 비교 구현·공개 저장결과 검산", summary,
        next_action, final["status"], dict(training=None, evaluation=None,
        saved_attachment="complete_16_no_reinference", comparator="verified_offline_not_web",
        next_work="direct_source_action_context_binding"))
    root = U.parents[3]
    state = read(O / "STATE.json")
    assert state[U.name]["summary"] == summary and state["current_stage"] == final["status"]
    assert summary in (root / "WordsInSilence_PROJECT_CONTEXT.md").read_text(encoding="utf-8")
    assert next_action in (root / "NEW_CHAT_HANDOFF.md").read_text(encoding="utf-8")
    print(json.dumps(dict(status=final["status"], whole_goal_complete=False, evidence=len(evidence)), ensure_ascii=False))


if __name__ == "__main__":
    main()
