import json, hashlib, re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
U=Path(__file__).resolve().parent
O=U.parent
ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open("x",encoding="utf8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)

f=read(U/"FROZEN.json")
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
v=read(U/"VERIFY.json")
for p,h in v["outputs"].items():assert sha(p)==h,p
for name in ["VERIFY.json","STRESS.json","PATH_AUDIT.json","CONTRACTS_FINAL.json","STATIC_REPORT_CHECK.json"]:
    assert read(U/name)["status"]=="PASS",name
assert not (U/"RUNNING.json").exists()
roots=[ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]
for p in roots:assert sha(p)==sha(U/"before"/p.name),f"Concurrent root edit: {p}"
now=datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
summary=read(U/"SUMMARY.json")
audit=dict(status="PUBLIC_MULTIREREFERENCE_ANGLE_EVIDENCE_VERIFIED_RESEARCH_ONLY",
    at=now,unit=U.name,counts=summary,protected_count=len(f["protected"]),
    implementer="Codex",verification="Codex separate saved-output formulas and contracts, not a second reviewer",
    first_numeric_error=None,browser_preview="file URL blocked by browser policy; no alternate browser/serving workaround; static validation and standalone timeline figure inspected",
    conditions="mixed-coverage reference bank explicit:136 merged with angle evidence+763 fullframe masked unassessed; no model comparison",
    prohibitions_respected=dict(training=0,classifier_inference=0,user_access=0,detector=0,operational_changes=0),
    remaining=["763 reference quality sidecars missing","normal comparison abstention risk","known identity error unchanged",
               "existing raw-feature DTW alignment uncertainty","public annotation timing oracle, not live-web pipeline",
               "repetition/compression and3D correctness not validated by this adapter"],
    artifacts={str(p):sha(p) for p in U.iterdir() if p.is_file()})
write(U/"FINAL_AUDIT.json",audit)

next_action=("각도별 근거×다중참고 DTW 연구 결과를 사용자와 검토한다. 공개136질의/899참고풀1087쌍에서 좌표를 보존하고 손·각도·시간대 근거를 분리했다. "
 "4080항목 중 일부비교200/근거부족3820/비교대상아님60; 단순수치2509중2309는재확인근거부족. "
 "재확인자료136/899이고5명자료확보는2단어뿐이므로운영자동차단/학습READY로승격하지않는다. "
 "다음제안은나머지763공개참고의재확인준비범위·비용과기존다중참고구간추정/반복압축보호연결검증계획을확정하는것. "
 "단일관측을임의정답승격하거나최소5명기준완화금지. C·T3-F·운영웹/DTW 유지,사용자자료학습/튜닝/선택금지. 추가학습은결과공동검토후.")
entry=f"""### 2026-10-21 각도별 근거와 다중참고 DTW 연결 — 연구 검증 완료·운영 미적용
- 단위 `ops/{U.name}`; RESULT_KO.html/txt, SUMMARY.json, FINAL_AUDIT.json. 구현/별도수식검산 동일Codex, 다른검토자 없음.
- 공개TRAIN899영상/100단어/9명 참고풀, 기존각도근거136영상을질의로본인촬영자제외1087쌍 비교. 양쪽각도근거있는124쌍. 136통합관측+763기존fullframe→ROI관측/재확인없음 계보를명시; 같은스냅샷의단순각도/근거분리비교이며C모델성능실험아님.
- 전체저장113181프레임에서기존원시특징DTW정렬 후근거부착,좌표16977150값보존. 공개주석동작구간분모고정·참고고유시간가중·8프레임/50%/5명기준유지. 사용자구간을정답처럼아는실시간웹경로아님.
- 136×2손×15각도=4080항목: 일부차이표시200/근거부족3820/역할상비교안함60. 모든핵심시간5명확인0. 단순수치2509중2309재확인부족. 200도수어정답판정아니며가래떡·대지2단어12영상에서만5명참고확보. 전체100단어정밀준비완료아님.
- 실제미관측/단일관측재확인없음/관측상충/참고정렬부족을손·각도·원본시간으로분리. 낮은차이도빈구간있으면PARTIAL;비교불가None/NaN,정답/오류판정없음.
- 정상24/교차55장면좌표·각도·상태보존;기존소복f84오류1미교정. 엄격조건이정상비교도보류하므로자동운영채택금지.
- VERIFY1087쌍/160100대응/원시각도쌍1356212·엄격140595 검산PASS. PATH_AUDIT12쌍별도최적비용PASS. 통제272손전체각도누락+136한각도한프레임누락+136시간2배+136비핵심누락PASS;경로고정시험이며새검출/실사용인식률아님. 합성11계약최종코드PASS.
- C·T3-F등32보호해시유지. 새학습/분류/사용자접근/검출/운영변경0. 신규수치실패없음. 보고서브라우저file:미리보기정책차단;우회없음,정적HTML/JS검사와별도시간도표검토.
- 남은문제:763참고품질근거부재,정상보류,원시특징정렬오류,공개주석의실시간대체,반복/압축·3D정답미검증. 학습은사용자결과확인후.
"""
p=ROOT/"WordsInSilence_PROJECT_CONTEXT.md";text=p.read_text("utf-8-sig")
mark="## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인"
assert mark in text
text=text.replace(mark,mark+"\n\n"+entry,1)
mark9="## 9. 다음 단일 작업"
text=text.replace(mark9,mark9+"\n\n### 2026-10-21 각도별 다중참고 근거 결과 검토\n"+next_action+"\n",1)
p.write_text(text,encoding="utf8")
state=read(O/"STATE.json")
state.update(updated_date_kst=now,current_stage="WORD100_ANGLE_MULTIREF_DTW_VERIFIED_RESEARCH_ONLY",next_single_action=next_action)
state[U.name]=dict(status="VERIFIED_RESEARCH_ONLY_AWAITING_USER_REVIEW",at=now,unit=U.name,
    scope=dict(reference_clips=899,query_clips=136,words=100,people=9,pairs=1087,split="public TRAIN only"),
    counts=summary,protected=32,training=0,user_data_access=0,operational_change=False,
    report=str(U/"RESULT_KO.html"),audit=str(U/"FINAL_AUDIT.json"),next=next_action,
    limitations=audit["remaining"],running=None)
(O/"STATE.json").write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 각도별 비교 근거를 다중참고DTW에 연결하고 부족한근거를잘맞음으로오해하지않게한다.
마지막 완료: 공개136질의/899참고풀1087쌍 각도·시간근거 연결 및 별도수식 검산.
다음 한 작업: {next_action}

{entry}

실행 종료0: run.py freeze/run, verify.py, stress.py, path_audit.py, report.py, visual_check.py.
CONTRACTS_FINAL.json은최종어댑터로기존11합성계약재실행,원CONTRACTS.json보존.
완료추출·학습·전체검산반복금지. 진행프로세스/잠금없음.
보고서file:URL브라우저검사거절로새탭오픈미확인;다른브라우저/서버우회없음.정적자료검사와PNG시각검토완료.
관측상충없음은정답보증아니며,미관측·재확인부족·상충을잘못된동작과혼동하지않는다.
기존 C·T3-F·운영DTW/웹·원본·사용자자료 보존. 추가학습은함께결과확인후.
"""
(ROOT/"NEW_CHAT_HANDOFF.md").write_text(handoff,encoding="utf8")
assert read(O/"STATE.json")[U.name]["counts"]["query_clips"]==136
for p in roots:assert U.name in p.read_text("utf8")
for p,h in f["protected"].items():assert sha(p)==h,p
write(U/"SYNC_CHECK.json",dict(status="PASS",roots={str(p):sha(p) for p in roots},protected=32))
print("COMPLETE",U/"RESULT_KO.html")
