"""Finalize existing report and synchronize after explicit verifier metadata repair."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,re
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):
    with (U/name).open("x",encoding="utf8") as fp:json.dump(obj,fp,ensure_ascii=False,indent=2)
f=read(U/"FROZEN.json");s=read(U/"SUMMARY.json");v=read(U/"VERIFY.json")
now=datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
rec=read(U/"VERIFY_RECOVERY.json")
verifier=U/"verify_v2.py"
normalized=hashlib.sha256(verifier.read_text("utf-8").encode("utf-8")).hexdigest()
assert normalized==rec["verifier_v2_sha256"]
write("REPORT_FIRST_ERROR.json",dict(error="Final audit compared normalized-LF verifier text digest with Windows CRLF file bytes. Report generation and all numerical checks had finished; FINAL_AUDIT had not been emitted.",
    preserved_original_digest=rec["verifier_v2_sha256"],normalized_text_digest=normalized,
    actual_file_digest=sha(verifier),candidate_or_data_changed=False,
    resolution="Verified exact normalized text identity; record actual byte digest separately, without changing verifier or recovery original."))
write("VERIFY_RECOVERY_FILE_HASH.json",dict(path=str(verifier),sha256=sha(verifier),normalized_text_sha256=normalized))
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
for q in read(U/"INDEX.json"):assert sha(q["path"])==q["sha256"]
page=(U/"RESULT_KO.html").read_text("utf8")
for link in re.findall(r"(?:src|href)=[\"']([^\"'#]+)[\"']",page):assert (U/link).exists(),link
next_action=("좌표 보존·각도별 근거 후보의 공개136영상 결과를 사용자와 검토한다. 점 삭제의 연쇄손실912각도시점(핵심371)을 "
    "비교후보로 회복했고 원시좌표/기존입력은 불변. 단,교차55장면1650계산각도 중재확인부족735/상충37이므로 "
    "엄격한 비교마스크를 정상/오류 또는 자동 운영차단으로 채택하지 않는다. 다음 제안은 실제미관측·단일관측재확인부족·관측상충을 "
    "분리한 결과표시/비교범위 계약과 위치·궤적 독립 유지 검증이다. 단일관측을임의정답승격하거나문턱완화하지않는다. "
    "정상형태20표본중9장면45각도상충도동작오류가아님. C·T3-F/웹 유지,사용자자료접근·학습·튜닝·선택 금지. "
    "전체1199준비나 학습READY아니며 추가학습은 함께 결과확인후.")
audit=dict(status="ANGLE_EVIDENCE_SEPARATED_VERIFIED_RESEARCH_ONLY",at=now,unit=U.name,
    scope=dict(videos=136,words=100,people=9,frames=16917,split="TRAIN diagnostics only"),
    counts=s["totals"]["all"],cohorts=s["totals"],verification=v["counts"],control_totals=v["control_totals"],
    contracts=read(U/"CONTRACTS.json"),public_self_contract=read(U/"PUBLIC_COMPARISON_CONTRACT.json"),
    protected_files=32,model_changes=0,coordinate_changes=0,training_runs=0,classifier_runs=0,detector_runs=0,user_data_reads=0,
    implementation="Codex local",verification_role="Same Codex separate complex-angle equations; not separate reviewer",
    decision="Coordinate preservation/local per-angle evidence contract passed. Strict eligibility is diagnostic only, not an automatically adopted production gate.",
    running_processes=[],next_single_action=next_action,report=str(U/"RESULT_KO.html"),
    visual_review="Reused unchanged public raw92-frame prior inspection; new angle diagrams and control-state lists examined. No new finger GT labels.",
    recoveries=["Verifier accounting separated extra historical normal/cross/error controls; original checker preserved.",
        "Final audit hash metadata differentiated normalized LF text from actual Windows CRLF bytes; verifier/data unchanged."],
    evidence_hashes={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.suffix in (".py",".json",".html",".txt")})
write("FINAL_AUDIT.json",audit)
pairs=[(R/"WordsInSilence_PROJECT_CONTEXT.md",U/"before"/"WordsInSilence_PROJECT_CONTEXT.md"),
       (R/"NEW_CHAT_HANDOFF.md",U/"before"/"NEW_CHAT_HANDOFF.md"),(O/"STATE.json",U/"before"/"STATE.json")]
for p,b in pairs:assert sha(p)==sha(b),f"Concurrent root edit {p}"
current="""### 2026-10-21 座標保존 각도별 비교 근거 분리 — 연쇄손실 회복·연구 후보만
- 단위 `ops/word100_angle_evidence_20261021`, `RESULT_KO.html`/`FINAL_AUDIT.json`. 공개TRAIN기존36+확대100=136영상/100단어/9명/16917프레임 저장자료 재사용. 새검출0.
- 좌표·원래관측·280채널모델입력불변,각도별 미관측/기하퇴화/재확인부족/상충/상충미표시를별도저장. 비교불가NaN과고정참고분모계약구현;실제DTW정렬·운영미적용.
- 과거점별보류가잃던1774각도시점계산값보관. 같은품질기준비교후보167816→168728(+912,핵심371). 이전36+293/핵심94,확대100+619/핵심277. 좌표정답복원·인식률향상아님.
- 전체계산가능202919중상충21965/재확인부족12226/상충미표시168728. 기존불확실성도진단하지만원래입력은삭제안함.
- 정상대조24장면/교차55장면좌표·입력불변. 교차1650계산각도중878근거/735재확인부족/37상충. 체계표본20중9장면45상충. 엄격마스크자동채택시정상비교도많이보류될위험이남아운영미채택.
- 기존소복f84오류주석1은보존·미교정. 정상/교차가모든손가락각도의정밀정답은아님. 이전92원본표본시각근거재사용,새각도상태도식검토.
- 좌표2537550값·입력4736760값불변/각도상태507510개별도복소수식검산PASS(최대2.84e-14도),합성계약12PASS. 동일Codex구현/별도수식검산,다른검토자없음.
- 검산회복: 신규자료와과거주석교집합4장면을기존대조와분리(정상1/교차2/오류1);원검산기·최초오류보존. 해시기록LF텍스트/CRLF파일차이도별도정정근거. 후보·입력·계수변경없음.
- C·T3-F/운영32보호해시유지. 학습0/분류0/사용자접근0/학습READY없음. 연구단위완료·실행종료·잠금없음.
""".replace("座標保존","좌표 보존")
cp=R/"WordsInSilence_PROJECT_CONTEXT.md";text=cp.read_text("utf-8-sig")
for marker,body in [("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인",current),
                    ("## 9. 다음 단일 작업","### 2026-10-21 각도별 근거 보존 결과 검토\n"+next_action+"\n")]:
    assert text.count(marker)==1;text=text.replace(marker,marker+"\n\n"+body,1)
cp.write_text(text,encoding="utf8")
state=read(O/"STATE.json");state["updated_date_kst"]="2026-10-21"
state["current_stage"]="WORD100_ANGLE_EVIDENCE_SEPARATED_VERIFIED_RESEARCH_ONLY"
state["next_single_action"]=next_action;state[U.name]=audit
(O/"STATE.json").write_text(json.dumps(state,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
handoff=f"""# WordsInSilence 최신 인수인계

갱신 {now}

현재 목표: 손 위치·좌표를 보존하면서 정밀 손가락 각도별 비교 근거를 따로 표시한다.
마지막 완료: 공개136영상 각도별 근거 분리 후보 구현·저장수식 검산. 점별제외의 연쇄손실912각도시점 회복,정상/교차좌표보존. 엄격비교마스크는운영미채택.
다음 한 작업: {next_action}

{current}

실행/근거: candidate.py, contracts.py(12PASS), run.py freeze/run 완료. verify.py 초기대조개수실패 보존→verify_v2.py PASS.
report.py 그림·보고서 생성 후LF/CRLF해시메타데이터불일치로최종파일전중단; finish.py에서파일변경없이정정근거확인·최종감사완료.
자료·코드·조건을 결과에 맞춰 바꾸지 않았다. 완료된추출·학습·전체검산 반복금지.
위치: `research/front_keyphase_v1/ops/{U.name}/RESULT_KO.html`, FINAL_AUDIT.json, VERIFY.json.
구현/검산은 동일Codex이며 별도에이전트/전문가정답검토는없음. 실제새사람인식률,모든수어일반화,세부피드백정확성미검증.
기존 C·T3-F·DTW·웹·원본·사용자자료 보존. 추가학습은사용자결과검토후.
"""
(R/"NEW_CHAT_HANDOFF.md").write_text(handoff,encoding="utf8")
for p,_ in pairs:
    if p.suffix==".json":read(p)
    else:assert U.name in p.read_text("utf8")
for p,h in f["protected"].items():assert sha(p)==h,p
write("SYNC_CHECK.json",dict(status="PASS",roots={str(p):sha(p) for p,_ in pairs},protected=32))
print("COMPLETE",U/"RESULT_KO.html")
