"""Seal checked artifacts, then synchronize current-state documents."""
from pathlib import Path
from html.parser import HTMLParser
import json,hashlib,re
from datetime import datetime
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open("x",encoding="utf8") as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
f=read(U/"FROZEN.json");s=read(U/"SUMMARY.json")
assert not (U/"RUNNING.json").exists()
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
for r in read(U/"PREPARED_INDEX.json"):
    for p,h in r["output_hashes"].items():assert sha(p)==h,p
v=read(U/"VERIFY.json")
for p,h in v["outputs"].items():assert sha(p)==h,p
for name in ("INPUT_VERIFY.json","VERIFY.json","STRESS.json","UNCERTAINTY_CONTROLS.json"):
    assert read(U/name)["status"]=="PASS",name
assert v["counts"]["pairs"]==216 and v["counts"]["query_clips"]==27
assert s["totals"]["strict_summary"]==353 and s["totals"]["status_INSUFFICIENT"]==322
assert sum(s["totals"].get("status_"+k,0) for k in ("OBSERVED","PARTIAL","INSUFFICIENT","NOT_REQUIRED"))==810
visual=read(U/"VISUAL_REVIEW.json")
assert visual["panels_reviewed"]==217 and len(visual["pages_reviewed"])==25
class Links(HTMLParser):
    links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ("src","href") and v:self.links.append(v)
check=Links();html=(U/"RESULT_KO.html").read_text("utf8");check.feed(html)
for link in check.links:
    assert "://" not in link
    assert (U/link).exists(),link
assert "__TABLE__" not in html and "__GALLERIES__" not in html
js=(U/"report_data.js").read_text("utf8")
data=json.loads(js[len("const clips = "):].strip().rstrip(";"))
assert len(data)==27
for c in data:
    assert len(c["rows"])==30 and len(c["times"])==len(c["votes"])==len(c["state"])
    assert c["required"] in (["L","R"],["R"])
write(U/"STATIC_REPORT_CHECK.json",dict(status="PASS",local_links=len(check.links),clips=27,
    syntax="node --check report_ui.js and report_data.js separately",
    visual="AVAILABILITY.png and 25 RGB contact sheets inspected; full HTML browser rendering not performed"))
roots=[ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]
for p in roots:assert sha(p)==sha(U/"before"/p.name),f"Concurrent edit {p}"
now=datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
next_action=(
    "공개3단어×9명 재확인 결과를 사용자와 검토한다. 27영상/216쌍에서 준비는 완료했으나 정상 비교 충분성은 미달: "
    "수어238/270·후원7/270·멀다108/135 각도 요약 가능(정답률 아님). "
    "다음 제안은 기존 손 재확인과 새 손 삽입의 좌우 연결 조건을 분리하고, "
    "여러 참고에서 시간별 손 역할을 확인하여 정상 손내림과 실제 관측 누락을 구분하는 것. "
    "공통 규칙 후보는 같은27영상과 기존 오류·교차 대조군으로 검증; 최소5명/50% 조건 임의완화·단어별 예외 금지. "
    "C·T3-F/운영웹 유지, 사용자·멘토 자료 학습·튜닝·후보선택 금지. 추가 학습은 결과 공동검토 후.")
entry=f"""### 2026-10-21 작은 다중 수행자 재확인 묶음 — 준비·검산 완료, 정상 비교 충분성 미달
- 단위 `ops/{U.name}`. 보고서 RESULT_KO.html/txt, SUMMARY.json, FINAL_AUDIT.json. 동일 Codex 구현/별도 수식 검산·RGB 판독, 독립 수어 전문가 검증 아님.
- 공개 TRAIN 수어·후원·멀다 3단어×ID01~09=27영상/3299저장프레임. 공개 역할/기존 교차 코드 순 선택 동결, 기존5 재사용·신규22 재확인(준비112초), 자기 촬영자 제외216쌍. DEV/보류/사용자/멘토 자료0.
- 원본 전체화면 기존 관측+동일이미지 재검출·필요 확대→기존 좌표 보존 통합→고정ROI→각도근거. 기존 알고리즘·크기·최소5명/8프레임/50% 유지. 추가116 손-프레임(핵심32), 기존 usable107218점 보존. legacy safe 출력은 생성되나 사용하지 않음.
- 동일 최종좌표/정렬에서 과거5개 품질메타만 둔 조건675개 요약 부족→완성27개 조건353 표시/322보류. 353중59전체시점근거·294부분. 멀다 비사용손135 NOT_REQUIRED. 인식률·수어정답률·학습향상 아님.
- 단어별 비교 각도 수어238/270,후원7/270,멀다108/135. 후원에서 한 손→양손 단계인데 전체 구간 양손요구, 겹침 시 current_arm_uncertain으로 기존 손 재확인 보류, 주석 경계/원시DTW정렬 한계. 216쌍중28 기존정렬근거하한 미달. 정상에도 넓은 보류라 자동운영채택/학습READY 아님.
- INPUT_VERIFY arccos/좌표보존, VERIFY216쌍/31701대응/원시238090·엄격161379각도쌍 검산PASS. 기존 정상2/교차11좌표·상태유지. RGB217패널/25면(정상81,추가116,교차11,보류9)판독, 21관절정답 보증 아님.
- 통제 손전체각도누락54/한각도한시점누락27/시간2배27/비핵심누락27 및 각도불확실27/참고손누락27 PASS. 모두 저장정렬 고정 정보 통제, 새 픽셀가림/실사용/분류성능 아님. 각도불확실은 다른 각도·좌표에 전파되지 않음.
- 최초 freeze에서 과거 재확인 JSON 경로 오연결로 추출 전 중단; FIRST_ERROR.txt 보존, records 경로 확인·수정 후 첫 동결, 모든27 누락 없이 완료. 준비/검산 수치 오류 없음.
- 보호32해시 유지. 학습/분류/사용자접근/운영변경0. 정적 HTML·JS 및 PNG 시각검토; 브라우저 전체 렌더 미실시. 남은 작업은 재확인 연결·손별 시간범위 검토 후 공통 후보 검증.
"""
audit=dict(status="PREPARED_VERIFIED_NORMAL_AVAILABILITY_LIMITED",at=now,summary=s,
    implementer="Codex",verifier="same Codex separate arithmetic and visual review, not independent annotator",
    protected=len(f["protected"]),training=0,classifier_inference=0,user_data_access=0,operational_change=False,
    first_error=dict(file="FIRST_ERROR.txt",cause="Before freeze, old record path assumed at wrong location",
                     resolution="Verified legacy records path; corrected routing before first successful freeze; no output rerun/exclusion"),
    semantic_gate="NORMAL_COMPARISON_SUFFICIENCY_NOT_PASSED",
    next=next_action,artifacts={str(p):sha(p) for p in U.iterdir() if p.is_file()},
    running=None)
write(U/"FINAL_AUDIT.json",audit)
p=roots[0];text=p.read_text("utf-8-sig")
mark="## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인";assert mark in text
text=text.replace(mark,mark+"\n\n"+entry,1)
mark9="## 9. 다음 단일 작업";assert mark9 in text
text=text.replace(mark9,mark9+"\n\n### 2026-10-21 작은 다중 수행자 자료 결과 검토\n"+next_action+"\n",1)
p.write_text(text,encoding="utf8")
state=read(O/"STATE.json")
state.update(updated_date_kst=now,current_stage="WORD100_SMALL_MULTIPEOPLE_ANGLE_VERIFIED_AVAILABILITY_LIMITED",
             next_single_action=next_action)
state[U.name]=dict(status="PREPARED_VERIFIED_NORMAL_AVAILABILITY_LIMITED_AWAITING_USER_REVIEW",
    at=now,scope=s["scope"],counts=s["totals"],training=0,user_data_access=0,
    protected=32,operational_change=False,report=str(U/"RESULT_KO.html"),
    audit=str(U/"FINAL_AUDIT.json"),next=next_action,running=None)
(O/"STATE.json").write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
roots[1].write_text(f"""# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 작은 공개 다중 수행자 묶음의 재확인 정보를 완성하고 정상 비교 가능성과 선택적 보류를 검증한다.
마지막 완료: 3단어·9명·27영상 준비,216쌍 검산. 자료 준비/선택적 보류는 완료지만 정상 비교 충분성은 미달.
다음 한 작업: {next_action}

{entry}

종료0: study.py freeze/prepare/compare,check_inputs.py,reuse_arithmetic.py,reuse_stress.py,
visuals.py,analyze.py,diagnose_and_uncertainty.py,report.py. Node JS 구문검사 완료.
모든 실행 종료·RUNNING 없음. 완료 추출/검산 반복금지.
신규 파일은 이번 단위와 루트 상태 문서만. 기존 C/T3-F·원본·과거 동결 결과 읽기 전용 유지.
이번 질의는 추가학습 승인이 아님. 정상 동작의 보류를 사용자 수행 오류로 표현하지 않는다.
""",encoding="utf8")
for p,h in f["protected"].items():assert sha(p)==h,p
for p in roots:assert U.name in p.read_text("utf8")
write(U/"SYNC_CHECK.json",dict(status="PASS",roots={str(p):sha(p) for p in roots},protected=32))
print("COMPLETE",U/"RESULT_KO.html")
