"""Synchronize actual research results after independent verification; no model runs."""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

U = Path(__file__).resolve().parent
ROOT = U.parents[3]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def put(p, d):
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    assert (ROOT / "AGENTS.md").exists()
    audit = read(U / "verification/KSL_BRIDGE_INDEPENDENT.json")
    assert audit["status"] == "PASS", audit["status"]
    assert read(U / "verification/ALL42_SCORES_INDEPENDENT.json")["status"] == "PASS"
    ui = read(U / "verification/UI_VERIFY.json")
    assert ui["status"] == "PASS"
    assert ui["final_html_sha256"] == hashlib.sha256((U / "RESULT_KO.html").read_bytes()).hexdigest()
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    rel = "research/front_keyphase_v1/ops/word100_feature_learning_pipeline_20261027"
    status = "COMPARISON_COMPLETE_FUSION_NOT_ADOPTED_KSL_FEEDBACK_UNVALIDATED"
    current = (
        "사용량 중단에서 재개하여 완료된 학습을 재사용하고 후속 비교·검산·보고를 마쳤다. 공식 DeepHand 수동 수형 3359장·45종을 원출처 촬영자9명에 연결하여 "
        "학습2693(01–05)/개발284(06–07)/최종382(08–09)로 고정했다. 최근 포함10후보를 조사하고 V3/V4/TinyNeXt 세 영상모델, "
        "관절·RGB·시간 네조건 및 존재표시 대조군을 3seed로 학습했다. 고정영상분류9+관측15+영상미세조정9+미세조정영상관측9=42개 실제 학습·최종평가 완료. "
        "최종점수 재학습/재선택 없음. 외부45수형 시험에서 RGB+시간은 41.45→68.24%(+26.79%p), RGB+관절+시간은72.25→67.89%(-4.36%p). "
        "최종전 개발macroF1로 선택한 RJ_F는 최종67.71%(3seed61.26–72.51), 관절+시간71.20%보다 낮아 운영 미채택. "
        "시험최고 후보로 사후선택하지 않음. 이 수치는 TCN100단어 인식률이 아니다. "
        "36공개KSL구간468프레임·456crop의 추정을24개 고정참고경로로 재생하고 세물리특징/관측보류/분모보존을 연결했다. "
        "기존AI초안은 독립정답이 아니므로 정상교차/교정정확도 미확립. 42점수400항목·미세조정93항목·bridge171계약 별도검산과 "
        "실제KSL재생 독립검산, 화면468표본/PC·모바일검증 통과. 같은Codex 별도검산이며 사람정답검토와 구분한다. "
        "C·T3-F·운영32핀 불변, 사용자멘토자료접근·학습·튜닝·선택0. 연구보고화면완료, 운영후보활성화0. 6단계전체운영완료가아니다."
    )
    next_action = (
        "다음 핵심은 관절이 정상일 때 유지하고 손은 보이는데 검출만 빠진 때 RGB로 보완하는 선택적 후보의 근거 확보다. "
        "최종382장은 이미소진되어 규칙/가중치/후보선택에재사용하지않는다. 미검출18장 RGB+시간15/18×3 결과는 작은사후부분집합가설근거이며 "
        "가림복구나KSL교정성공이아니다. 기존독립주석을우선재사용하고 부족한KSL의 정상수형/변화/회전/교차/미관측만 작은독립확인묶음으로 "
        "고정하여 필요한사람검토를집중한다. AI초안745/191을정답승격하지않는다. 새공개묶음으로 관측범위·오판·정상교차·짧은변화보존이 "
        "통과한뒤 DTW교정정확도와별도웹후보로이동한다. 현재검증실패후보를단순연결해완료처리하지않는다. "
        "기존C/T3F/운영유지·개인촬영미사용. 이번고정실험실행종료; 활성학습/추론없음."
    )
    docs = [ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "NEW_CHAT_HANDOFF.md", ROOT / "research/front_keyphase_v1/ops/STATE.json"]
    backup = U / "before/final_sync"
    backup.mkdir(parents=True, exist_ok=True)
    for p in docs:
        dest = backup / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
    ctx = docs[0].read_text(encoding="utf-8-sig")
    title = "### 2026-10-27 공개 수형 42회 학습·비교 및 DTW 기술연결 완료 — 통합 후보 미채택"
    assert title not in ctx, "Already synchronized"
    evidence = f"근거: {rel}/RESULT_KO.html, SUMMARY.json, FINAL_GATE.json, verification/ALL42_SCORES_INDEPENDENT.json, KSL_BRIDGE_INDEPENDENT.json, UI_VERIFY.json."
    ctx = ctx.replace("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인", f"## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n{title}\n{now}\n{current}\n{evidence}\n", 1)
    ctx = ctx.replace("## 9. 다음 단일 작업", f"## 9. 다음 단일 작업\n\n### 2026-10-27 선택적 RGB 보조의 새 공개 근거 확보\n{now}\n{next_action}\n{evidence}\n", 1)
    docs[0].write_text(ctx, encoding="utf-8")
    state = read(docs[2])
    record = {"status": status, "at": now, "unit": U.name, "training_completed": 42,
              "active_training_or_inference": False, "split": {"fit": 2693, "validation": 284, "test": 382},
              "final_evaluation_complete": True, "final_test_tuning": False,
              "observer_selected_before_test": "observer_RJ_F", "operating_candidate_enabled": False,
              "production_changed": False, "user_mentor_data_accessed_or_used": False,
              "all_six_operating_stages_complete": False,
              "summary": current, "next": next_action, "result": rel + "/RESULT_KO.html",
              "verification": ["verification/ALL42_SCORES_INDEPENDENT.json", "verification/KSL_BRIDGE_INDEPENDENT.json", "verification/UI_VERIFY.json", "verification/FINAL_PROTECTION.json"]}
    state["updated_date_kst"] = now
    state["current_stage"] = status
    state["next_single_action"] = next_action
    state[U.name] = record
    put(docs[2], state)
    execution = U / "CURRENT_EXECUTION.json"
    if not (backup / execution.name).exists():
        shutil.copy2(execution, backup / execution.name)
    put(execution, record)
    docs[1].write_text(
        f"# WordsInSilence 최신 인수인계\n\n{now}\n\n현재 목표: 수어 특징 관측의 신뢰도 확보와 DTW 피드백 검증.\n\n"
        f"{current}\n\n다음 한 작업: {next_action}\n\n작업단위: `{rel}`\n\n"
        "모델 선택: 최근10후보 검토, V3/V4/TinyNeXt 실측. 이번 고정 예산의 개발검증으로 V3 선택이며 전세계 최적 주장 금지. "
        "원래45수형 사람주석과 파생3물리특징/한국수어 의미정답은구분. DeepHand CC BY-NC-SA3.0 연구자산격리.\n\n"
        "재개 주의: 42완료점수재추론/재채점/재학습금지, 최종test로 선택/튜닝금지. "
        "KSL 기술재생의 최초호출mask누락 오류는 logs/KSL_BRIDGE_RECOVERY.json에 보존하고 동일조건으로복구; 최종평가는반복안함. "
        "진행중프로세스없음, 학습/추론 세션재시작불필요. 루트문서직전본은 before/final_sync.\n\n"
        f"결과: `{rel}/RESULT_KO.html`\n검산: `verification/ALL42_SCORES_INDEPENDENT.json`, `verification/KSL_BRIDGE_INDEPENDENT.json`, `verification/UI_VERIFY.json`.\n",
        encoding="utf-8")
    put(U / "verification/DOCUMENT_SYNC.json", {"status": "PASS", "at": now, "paths": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in docs}, "active_execution": False})
    print(json.dumps({"status": status, "synchronized": len(docs), "all_six_operating_stages_complete": False}))


if __name__ == "__main__":
    main()
