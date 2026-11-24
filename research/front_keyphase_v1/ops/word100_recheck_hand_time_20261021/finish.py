import re,json
from pathlib import Path
import study as d
U,O,ROOT=d.U,d.O,d.ROOT
f=d.read(U/"FROZEN.json")
d.pipe.check(f["pins"]);d.pipe.check(f["protected"])
for name in ("VERIFY.json","STRESS.json","CONTRACTS.json"):
    assert d.read(U/name)["status"]=="PASS"
assert not (U/"RUNNING.json").exists()
roots=[ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]
for p in roots:assert d.sha(p)==d.sha(U/"before"/p.name),"Concurrent root edit: "+str(p)
text=(U/"RESULT_KO.html").read_text("utf8")
for src in re.findall(r'src="([^"]+)"',text):assert (U/src).is_file(),src
assert "<script" not in text
result=d.read(U/"RESULT.json");verify=d.read(U/"VERIFY.json");diag=d.read(U/"DIAGNOSIS.json")
assert len(f["rows"])==158 and len({r["word_code"] for r in f["rows"]})==100
next_action=("사용자와 재확인·손별 시간대 결과 검토. 학습은 아직 승인되지 않음. 다음 후보는 공개 자료에서 "
    "참고로 확정한 손내림/동작/불확실 상태를 시간 정렬 근거 계산에도 분리 적용하는 비교 계획. "
    "참고 분모 고정·불확실 구간 유지·정상 교차 보존을 유지하며, 기존5명/50%/8프레임 기준을 결과로 완화하지 않는다. "
    "C·T3-F/운영 웹 유지, 사용자·멘토자료 학습/튜닝/선택 금지.")
at=d.now()
entry=f"""### 2026-10-21 기존 손 재확인·손별 시간대 분리 검증 완료
- 단위 `{U.name}`. 사용자 승인 범위 구현·검증 완료, 추가학습/운영교체 없음. Codex 직접 구현·별도수식검산·원본108패널 시각검토(독립 수어전문가 아님).
- 기존손 재확인을 새손 좌우배정 gate와 분리: 원래 geometry 임계값의 유일한 양방향 대응, 팔 반대근거/중복/모호함 veto. 기존좌표/좌우/기존유효각도/새손추가조건 보존. 형상 재확인≠좌우·해부학 정답 새 검증.
- 공개100단어·9TRAIN사람·158영상·19,611저장프레임 보존검사. 실제 다중참고는 기존3단어·9명·27영상·자기사람 제외8참고=216쌍,4조건=864저장쌍. 210손프레임 추가재확인,2,638각도프레임 state2→4. 기존좌표2,941,650수치·기존유효각도보존.
- 표시가능각도항목675개 중 기존353→재확인373, 시간대만353,결합373(20항목 증가). 후원7→27/270,수어238/270·멀다108/135그대로. 반대손135NOT_REQUIRED별도. 인식률/정답률/동작일치율 아님, 아직302근거부족.
- 손별 시간: 질의사람 제외 공개참고8명만 사용,기존 양성 안정손내림(.3초/8frame/.05어깨)·5명지지·어떤참고의 범위내손점도 veto. 모름은 필요범위 유지. 후원왼손 고유45frame,멀다왼손103frame(leave-one-out합집합) 분리. 후원750쌍-각도행의 관측비율상승이나 최종표시가능수 추가증가 없음. 임계값사후완화없음.
- 전체손제거54조건에서432경로 재정렬(357변경)해도 참고시간분모유지·누락손available0. 1각도1frame제거27조건 이웃/분모유지.27공개장면 라벨/후보순서변경 불변·중복후보확정차단. 구별동작 누락의 수어오류 판정 시험은 아님.
- 기존 정상/교차/알려진오류80대조 검산,알려진소복f84 보류유지. 시각108패널 정상손내림/교차보존 확인범위;모든손가락3D정답/모든수어보장 아님. 후원16/72·수어12/72정렬근거미달 그대로,손내림이 정렬앞단에 반영되지 않은 문제는 다음검토.
- 최초시각화 core키 KeyError는 과거36품질스키마차이. 영상생성 전 중단·VISUAL_FIRST_ERROR 보존,동결원래주석으로 시각화경로만정정. 비교/후보/검산결과 재실행·제외·튜닝 없음.
- 보호32핀 유지. 학습/분류추론/사용자멘토자료접근/운영변경0,모든실행종료. FROZEN/RESULT/VERIFY/STRESS/CONTRACTS/DIAGNOSIS/VISUAL_REVIEW/RESULT_KO.html 및 txt. HTML정적링크검사·그래프시각검사,브라우저렌더검사미실행.
"""
audit=dict(at=at,status="VERIFIED_MODEST_AVAILABILITY_GAIN_REMAINING_GAPS",implementer="Codex",
    verifier="same Codex separate arithmetic, saved-array counterfactuals and visual review; not independent semantic annotator",
    result=result,counts=verify["counts"],training=0,classifier_inference=0,user_data_access=0,
    runtime_changes=False,protected=len(f["protected"]),next=next_action,running=None,
    artifacts={str(p):d.sha(p) for p in U.iterdir() if p.is_file()})
d.write(U/"FINAL_AUDIT.json",audit)
p=roots[0];content=p.read_text("utf-8-sig")
mark="## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인";assert mark in content
content=content.replace(mark,mark+"\n\n"+entry,1)
mark="## 9. 다음 단일 작업";assert mark in content
content=content.replace(mark,mark+"\n\n### 2026-10-21 재확인·손별 시간대 결과 공동 검토\n"+next_action+"\n",1)
p.write_text(content,encoding="utf8")
state=d.read(roots[2]);state.update(updated_date_kst=at,
    current_stage="WORD100_RECHECK_HAND_TIME_VERIFIED_MODEST_GAIN",next_single_action=next_action)
state[U.name]=dict(status="VERIFIED_MODEST_AVAILABILITY_GAIN_AWAITING_REVIEW",at=at,
    scope=dict(preservation_words=100,preservation_clips=158,people=9,primary_clips=27,primary_words=3,pairs=216),
    results=result["comparison"],new_reconfirmed_handframes=210,new_eligible_angleframes=2638,
    training=0,user_data_access=0,protected=32,operational_change=False,
    report=str(U/"RESULT_KO.html"),audit=str(U/"FINAL_AUDIT.json"),next=next_action,running=None)
roots[2].write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
roots[1].write_text(f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 기존 손의 재확인과 손별 필요한 시간대를 분리해 정상 비교 가능성을 높이고, 질의 손누락이 참고 비교범위를 줄이지 않도록 검증한다.
마지막 완료: `{U.name}` 4조건 비교/검산/원본시각검토 완료. 표시가능각도353→373/675,302부족 지속. 정상 비교 충분성 완성/인식성능향상 아님.
다음 한 작업: {next_action}

{entry}

종료0: study.py freeze/run, verify.py verify/stress, contracts.py, visuals.py(시각화스키마경로 정정후), report.py.
모든 실행 종료, RUNNING 없음. 기존 비교/추출/검산 재실행하지 말고 저장근거 재사용.
신규 파일은 이번 연구 단위와 루트 상태문서만. 원본/W:/sources/과거동결자료 읽기 전용.
결과를 함께 검토하기 전 추가 학습 금지. 사용자 테스트는 실사용 평가 전용, 학습·튜닝·선택에 사용 금지.
""",encoding="utf8")
d.pipe.check(f["protected"])
for p in roots:assert U.name in p.read_text("utf8")
d.write(U/"SYNC_CHECK.json",dict(status="PASS",roots={str(p):d.sha(p) for p in roots},
                               protected=32,running=None))
print("FINISHED",U/"RESULT_KO.html")
