"""Synchronize latest handoff after immutable results, checking original snapshots."""
from pathlib import Path
import json,hashlib
U=Path(__file__).resolve().parent
R=U.parents[3]
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=read(U/"FINAL_AUDIT.json")
pairs=[(R/"WordsInSilence_PROJECT_CONTEXT.md",U/"before"/"WordsInSilence_PROJECT_CONTEXT.md"),
       (R/"NEW_CHAT_HANDOFF.md",U/"before"/"NEW_CHAT_HANDOFF.md"),
       (U.parent/"STATE.json",U/"before"/"STATE.json")]
for p,b in pairs:assert sha(p)==sha(b),f"Concurrent edit: {p}"
next_action=a["next_single_action"]
current="""### 2026-10-20 손가락 보류 기준 공개100 신규영상 확대 검증 — 연쇄 정보손실 확인·제외 후보 미채택
- 단위 `ops/word100_finger_preservation_20261020`, `RESULT_KO.html`/`FINAL_AUDIT.json`. 이전36과 겹치지 않는 공개TRAIN100영상/100단어/9명/12276프레임; 단어당 새1영상. 합계136영상/100단어/16917프레임. 전체899/1199나100단어 밖 검증 아님.
- 동결규칙 그대로 native 전체검출·재검출통합 후 고정ROI/기존손크기 유지. 단어/사람별예외·결과후계수변경 없음. 기존417526관절시점·기존유효입력·손목/MCP위치·좌우·시간 보존.
- 추가698손시점=상충미표시485/손가락일부불확실197/비교부족16. 손로컬186061→200003→보류후198583. 새13942중1420보류/12522유지; 정확도 개선 의미 아님.
- 핵심구간41영상/86손시점에서 일부굽힘감소. 108931각도시점중552보류: 직접불확실273/자체보류아니지만공유점제외277/비교부족2. 점별제외가 이웃각도까지 연쇄손실을 만듦.
- 3영상4손시점에서 당시한손의계산가능각도전체보류(장애인복지관1/여주시청2는15각도,북구주민지원센터1은내려가는손1각도). 양손전체동시각도소실0; 일부보류최장0.166667초. 의미상필수동작손실0은미증명.
- 원본92고유프레임표본검토(122배치중복포함). 정상20장면추가손실0/새교차표본2유지,과거정상3·교차53검증해시재사용.20정상은기존관측이라새형상정답보증아님. 열린손/검지/옆면/접촉에서도 보류발생;정밀관절GT없음.
- 입력10311840값/기하9023쌍별도수식검산PASS,각도손실별도복소수식PASS. 구현·검산동일Codex,다른검토자없음. 좌표흔들림과실제움직임분리필요.
- 진단·보존확인완료,점별형상제외후보미채택. C·T3-F/운영32보호해시유지. 학습0/분류0/사용자접근0,학습READY없음. 실행종료0·잠금없음.
"""
p=R/"WordsInSilence_PROJECT_CONTEXT.md"
context=p.read_text("utf-8-sig")
for marker,addition in [
    ("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인",current),
    ("## 9. 다음 단일 작업","### 2026-10-20 확대 검증 결과와 각도별 근거 보존 검토\n"+next_action+"\n")
]:
    assert context.count(marker)==1
    context=context.replace(marker,marker+"\n\n"+addition,1)
p.write_text(context,encoding="utf-8")
p=U.parent/"STATE.json";state=read(p)
state["current_stage"]="WORD100_FINGER_PRESERVATION_AUDIT_COMPLETE_HARD_HOLD_NOT_ADOPTED"
state["updated_date_kst"]=a["at"][:10]
state["next_single_action"]=next_action
state[U.name]=dict(a,evidence={
    n:str(U/n) for n in ["FROZEN.json","VERIFY.json","ANALYSIS.json","LOSS_VERIFY.json","LOSS_CAUSE.json",
        "VISUAL_REVIEW.json","RESULT_KO.html","FINAL_AUDIT.json"]},running_processes=[])
p.write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {a['at']}

현재 목표: 전체영상 검출·손 통합 후 고정상반신 입력에서 손 위치와 정밀 손가락 근거를 분리하고, 정상 동작 정보를 잃지 않도록 한다.
마지막 완료: 공개100 신규영상 확대 검증. 보존 검산은 통과했지만 점별 보류의 이웃각도 연쇄 손실을 확인해 입력 제외 후보는 미채택.
다음 한 작업: {next_action}

{current}

실행: pipeline.py freeze/run, verify.py, analysis.py, verify_loss.py, render.py, review_supplement.py, closeout.py 완료.
이전 detector/전수검산을 반복하지 않는다. CURRENT_EXECUTION.json은 COMPLETE, session29098 exit0. 신규 학습·분류 추론 없음.
구현/검산 담당: Codex 로컬. 별도 수식 검산이며 별도 에이전트·수어전문가의 정답 판정이 아니다.
사소한 복구: 추가 집계의 첫 읽기 명령이 기본 cp949로 실패해 UTF-8로 재실행. 입력·규칙·기존 결과 변경 없음.
보고서와 수치: `research/front_keyphase_v1/ops/{U.name}/RESULT_KO.html`, FINAL_AUDIT.json, LOSS_CAUSE.json.
운영 C·T3-F/DTW/웹·원본·사용자자료 보존. 사용자자료를 진단/학습/튜닝/후보선택에 접근하지 않았다.
"""
(R/"NEW_CHAT_HANDOFF.md").write_text(handoff,encoding="utf-8")
for p,_ in pairs:
    if p.suffix==".json":read(p)
    else:assert U.name in p.read_text("utf-8")
for p,h in read(U/"FROZEN.json")["protected"].items():assert sha(p)==h,p
out=dict(status="SYNCHRONIZED",unit=U.name,at=a["at"],
    files={str(p):sha(p) for p,_ in pairs},
    report=sha(U/"RESULT_KO.html"),protected=32)
with (U/"SYNC_CHECK.json").open("x",encoding="utf-8") as fp:json.dump(out,fp,ensure_ascii=False,indent=2)
print("SYNCHRONIZED",U.name)
