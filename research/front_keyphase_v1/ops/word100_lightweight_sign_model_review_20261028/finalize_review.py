"""Render a document-only review and synchronize its handoff. No model execution."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import re
import shutil
import sys

sys.stdout.reconfigure(encoding="utf-8")
U = Path(__file__).resolve().parent
ROOT = U.parents[3]
assert ROOT.name == "WordsInSilence"
OLD = U.parent / "word100_latency_input_gate_20261028"
review = U / "FINAL_REVIEW.txt"
assert review.exists() and "PASS" in review.read_text(encoding="utf-8-sig"), "Independent review required"
audit_old = json.loads((OLD / "FINAL_AUDIT.json").read_text(encoding="utf-8-sig"))
protected = {}
for name, digest in audit_old["evidence_sha256"].items():
    got = hashlib.sha256((OLD / name).read_bytes()).hexdigest()
    assert got == digest, f"Prior evidence changed: {name}"
    protected[name] = got

report = (U / "RESULT_KO.txt").read_text(encoding="utf-8-sig")
blocks = []
for block in report.strip().split("\n\n"):
    escaped = html.escape(block)
    escaped = re.sub(r'https://[^\s<]+', lambda m: '<a href="'+m[0]+'">'+m[0]+'</a>', escaped)
    blocks.append('<p>'+escaped.replace('\n','<br>')+'</p>')
notes = ["RESULT_KO.txt", "SKELETON_REVIEW.txt", "RGB_REVIEW.txt", "DECISION_AUDIT.txt", "LOCAL_HISTORY_REVIEW.txt", "FINAL_REVIEW.txt"]
assert all((U / n).exists() for n in notes)
doc = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHuBERT 유지와 경량 모델 전환 검토</title><style>
body{margin:0;background:#f3f5f8;color:#182435;font:17px/1.8 system-ui,"Malgun Gothic",sans-serif}
main{max-width:1040px;margin:30px auto;padding:30px;background:white;border-radius:16px}h1{line-height:1.4;font-size:30px}
.lead{padding:20px;background:#edf6ff;border-left:5px solid #2b69a5}p{margin:26px 0;overflow-wrap:anywhere}a{color:#155f9c}nav a{display:inline-block;margin:4px 15px 4px 0;font-size:14px}small{color:#526276}
</style><main><h1>SHuBERT를 계속 써야 할까?</h1><div class="lead"><b>제품 필수 모델로 확정하지 않는다.</b><br>7.16초는 속도 개선만 확인한 결과다. 경량 시간별 특징 판정부를 우선 비교하고, SHuBERT는 연구 기준으로 남긴다.</div>
<small>공식 문헌·기존 결과 검토 / 새 다운로드·설치·추론·학습·운영 교체 없음</small><nav>'''
doc += ''.join(f'<a href="{n}">{n}</a>' for n in notes) + '</nav>' + ''.join(blocks) + '</main></html>'
(U / "RESULT_KO.html").write_text(doc, encoding="utf-8")

at = datetime.now(timezone(timedelta(hours=9))).isoformat()
status = "LIGHTWEIGHT_SIGN_MODEL_REVIEW_COMPLETE_NO_MODEL_SELECTED_OR_TRAINING"
summary = "공식 수어 논문/코드, CPU 시간, 공개 가중치·사용 조건, 과거 ST-GCN 이력을 3개 별도 에이전트와 교차 검토. 15.3278→7.1595초는 DINO Linear INT8의 속도 개선이며 품질 유지 미확인·총 대기시간 아님. SHuBERT는 제품 필수 모델로 확정하지 않고 연구 기준으로 유지. 경량 시간별 특징 판정부가 주 후보 방향: 기존 TCN 구조의 별도 판정부를 비용 낮은 기준으로, SL-GCN을 새 비교 후보 한 개로 제한. TCN 특징 우위나 SL-GCN 한국수어 우위는 미검증. 4.9M/CPU225.3ms는 외부 논문의 관절→단어 구간이고 후속 속성 시험과 구성 다름. 최신 RGB 후보도 즉시 대체 근거 부족. 새 다운로드/설치/추론/학습/개인자료 접근/운영변경0."
next_action = "이번 모델 선택 보고를 사용자와 확인한 뒤, 기존 공개 KSL의 확인 가능한 손모양·관계·짧은 변화 정답과 사람 분리를 고정하고 작은 TCN 시간별 특징 판정부/저장 SHuBERT 특징 추가/SL-GCN 한 후보의 공정 비교 계약을 정한다. SL-GCN 입력·사용 조건·정답 지원 확인, 사전학습 차이 분리. 실제 관측 부족을 채우거나 보류 증가를 개선으로 숨기지 않는다. 품질 통과 후보만 전체 대기시간 측정. 지금은 새 다운로드·설치·학습 실행 대기, C/T3-F와 사용자·멘토자료 보존."
entry = {
    "at": at, "status": status, "unit": str(U), "summary": summary,
    "recommendation": "Separate small temporal attribute predictor first; SL-GCN as one comparator; SHuBERT research reference only",
    "selection_claim": "Engineering priority, not demonstrated best KSL feedback model",
    "new_downloads": 0, "installations": 0, "inference_runs": 0, "training_runs": 0,
    "private_access": False, "operating_changes": False,
    "report": str(U / "RESULT_KO.html"), "independent_review": str(review), "next_single_action": next_action
}
before = U / "before"
before.mkdir(exist_ok=True)
paths = [ROOT / "WordsInSilence_PROJECT_CONTEXT.md", ROOT / "NEW_CHAT_HANDOFF.md", U.parent / "STATE.json"]
for p in paths:
    b = before / p.name
    if not b.exists():
        shutil.copyfile(p, b)

state_path = U.parent / "STATE.json"
state = json.loads(state_path.read_text(encoding="utf-8-sig"))
state["updated_date_kst"] = at
state["current_stage"] = status
state["next_single_action"] = next_action
state[U.name] = entry
state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

context_path = ROOT / "WordsInSilence_PROJECT_CONTEXT.md"
context = context_path.read_text(encoding="utf-8-sig")
title = "### 2026-10-28 경량 수어 모델 근거 재검토 — SHuBERT 제품 필수 채택 보류"
assert title not in context, "Review already synchronized"
evidence = "근거: research/front_keyphase_v1/ops/" + U.name + "/RESULT_KO.txt, FINAL_REVIEW.txt, SKELETON_REVIEW.txt, RGB_REVIEW.txt, LOCAL_HISTORY_REVIEW.txt, DECISION_AUDIT.txt."
for heading, body in [("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인", summary), ("## 9. 다음 단일 작업", next_action)]:
    assert context.count(heading) == 1
    context = context.replace(heading, heading+"\n\n"+title+"\n"+at+"\n"+body+"\n"+evidence+"\n", 1)
context_path.write_text(context, encoding="utf-8")

handoff = f'''# WordsInSilence 최신 인수인계

{at}

목표: 한국수어의 동작 설명과 여러 사람의 참고 영상을 함께 배워, 사용자가 어느 손으로 어떤 모양·방향·위치·움직임을 어떤 순서로 수행했는지 파악하고, 중요한 차이와 허용 가능한 차이를 구분해 알려준다.

현재 단위: research/front_keyphase_v1/ops/{U.name}
현재 상태: {status}

{summary}

이번 최신 요청은 SHuBERT 가속만 반복하지 말고 아예 다른 경량 모델이 적합한지 근거로 판단해 달라는 것. 이 턴은 읽기/문헌/저장산술/보고만 수행. 최신 요청에 따라 모델 검색을 제한적으로 다시 수행했으며, 과거의 “새 모델 검색 금지”를 이번 직접 요청보다 우선하지 않았다. 외부 모델/데이터 다운로드와 새 학습은 이번에 실행하지 않음.

다음 한 작업: {next_action}

선택 근거 제한: SL-GCN 4.9M/CPU225.3ms는 OpenHands INCLUDE의 관절→gloss 측정으로 전처리·피드백 제외. 속성 연구의 SL-GCN과 동일 구성/성능 아님. OpenHands 27점 최소 입력→양손 전체/관측상태/시간별 출력 적응 필요. 작은 TCN 우선은 재사용/검증 비용 근거이며 정확도 우위 아님. 과거 ST-GCN+DPC27/49/localXY 단어 실험은 TCN보다 낮았으나 현재 시간별 특징 과제와 다르고 SL-GCN과도 다름. 최신 ASL→LSC의 MLP54.5/SHuBERT22.1/SLGCN14.5는 1명74영상 대표손모양 기댓정확도, KSL/단계 성능 아님. 최신 속성 weights 일부는 공개 예정. 라이선스는 코드/가중치/자료 별도.

최근 실행 근거: word100_latency_input_gate_20261028의 공개3영상×2회×2조건12회 CPU비교. cachedRGB→특징15.3278→7.1595초(53.29%감소), INT8 6.7161~9.4263초. 상대L2특징변화31.62~44.60%는 정확도하락률 아님. DINO가 비용 대부분. INT8품질 미확인으로 미채택. 이전 독립186검사와 관측입력 최종25검사 결과 유지. 현재손108+저장SHuBERT768=876 입력 부품이나 판정부 학습0. 과거복사RGB가 저장특징에 남는 문제 미해결. 직전 exec63880종료0, 새실행없음.

공개 KSL probe는4표현/3기록수행자12영상의 개발 비교. AI초안 합의는 전문가GT아님. 활동132중기존120/SHuBERT118,움직임38중34/32;SHuBERT고정특징 우위미확인,FT불가능 증명아님. 지연 개선과 의미 품질 개선을 분리한다.

환경: Samsung960XHA,Ultra7 258V,RAM32GB,IntelArc140V내장GPU/NPU. 현재Torch2.12CPU,가속 미설정. GPU없음이아니며XPU/OpenVINO지원문헌은있으나해당경로실측은없음. 눈/입 표정은목표제외,손-얼굴부위관계는유지. 총촬영후UI대기미측정. 가속/프레임감축/얼굴RGB제거/증류 품질 미보장.

담당: root 종합/보고; latency_path_audit 관절논문/속도/공개성; latency_quality_reviewer RGB와과거구현이력; observation_input_builder 산술·Intel·최종반대검토. 별도 에이전트 검토는 독립 수어전문가 확인과 다름. FINAL_REVIEW와FINAL_AUDIT 참조.

보호: C/T3-F·운영웹/DTW·W:/sources/이전동결읽기전용. 사용자·멘토자료접근/학습/튜닝/선택금지,카메라/외부전송/Git/임의다운로드금지. 이번정책/보고외코드변경없음. 진행중추론/학습/다운로드/자동후속없음.

{evidence}
'''
(ROOT / "NEW_CHAT_HANDOFF.md").write_text(handoff, encoding="utf-8")

# Documentation-only checks, not a new model test.
assert json.loads(state_path.read_text(encoding="utf-8"))[U.name]["status"] == status
assert context_path.read_text(encoding="utf-8").count(title) == 2
assert status in (ROOT / "NEW_CHAT_HANDOFF.md").read_text(encoding="utf-8")
assert "7.1595" in report and "225.3" in report
assert all((U / n).exists() for n in re.findall(r'href="([^"/:]+\.txt)"', doc))
audit = {
    "at": at, "status": "PASS_DOCUMENT_EVIDENCE_REVIEW_NOT_NEW_MODEL_QUALITY",
    "independent_review": str(review), "prior_evidence_hashes_verified": protected,
    "local_report_links": len(notes), "context_entries": 2, "state_parsed": True,
    "actual_browser_ui_test": False, "new_model_test": False,
    "new_downloads": 0, "training_runs": 0, "operating_changes": False,
    "report_sha256": hashlib.sha256((U / "RESULT_KO.txt").read_bytes()).hexdigest(),
    "notes": "Primary-source and saved-result review; same-system independent agents, not sign-expert labels."
}
(U / "FINAL_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
print(json.dumps({"status": status,"report":str(U / "RESULT_KO.html"),"prior_hashes_verified":len(protected),"downloads":0,"training":0},ensure_ascii=False))
