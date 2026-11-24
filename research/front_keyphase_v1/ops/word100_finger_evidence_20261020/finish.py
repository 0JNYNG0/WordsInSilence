"""Publish evidence and update current handoff after completed verification."""
import ast
import hashlib
import html
import json
import re
from datetime import datetime,timedelta,timezone
from pathlib import Path
U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
Q=O/"word100_merge_scale_input_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open("x",encoding="utf8") as fp:json.dump(v,fp,ensure_ascii=False,indent=2,allow_nan=False)
def text(p,s):
    with Path(p).open("x",encoding="utf8") as fp:fp.write(s)

f=read(U/"FROZEN.json");s=read(U/"SUMMARY.json")
v=read(U/"VERIFY.json");effect=read(U/"EFFECT_AUDIT.json");visual=read(U/"VISUAL_TARGETS.json")
assert v["status"]=="GEOMETRY_AND_INPUT_PRESERVATION_PASS"
assert read(U/"CONTRACTS_V2.json")["count"]==10
assert len(s["prior12"])==12 and all(x["selected_angle_unresolved"] for x in s["prior12"])
assert not (U/"RUNNING.json").exists()
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
for item in read(U/"INDEX.json"):
    for p,h in item["outputs"].items():assert sha(p)==h,p
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
review=dict(at=at,reviewer="Codex same implementer; no external expert/agent",
    viewed_sheets=visual["sheets"],source_frame_panels=len(visual["targets"]),
    selection=visual["selection"],
    observations=[
        "Prior12 raw panels: closed, edge-on, blurred or overlapping fingers; uncertain finger marks correspond to conflicting predicted segments. Not proof of exact true joint error.",
        "10 additional clip-diverse marked panels retain the raw hand/arm position; marks affect fingers locally, and sometimes all5 when hypotheses disagree broadly.",
        "3 normal identity and4 distinct crossing panels reviewed; identity/position preserved. Some normal crossing poses have uncertain fingers. Correct identity does not establish exact finger correctness.",
        "3 unpaired new examples are near lowered-hand/ROI edge and retain available position, without inventing a full precise hand.",
        "3 paired non-flagged examples show no new forced correction. No-disagreement is not proof of hidden-joint correctness.",
    ],
    quantitative_controls=effect["controls"],
    limits="35 selected public frames, not every flagged frame manually annotated. No finger-level ground truth or precision/recall. No temporal smoothness/performance claim.")
write(U/"VISUAL_REVIEW.json",review)
status="FINGER_EVIDENCE_SEPARATED_VERIFIED_RESEARCH_ONLY"
next_action=("손 위치 보존·손가락별 불확실성 후보 결과를 사용자와 검토한다. "
 "일괄 삭제는 채택하지 않고, 새 관측의 모양 보류 후보와 진단 표시를 구분한다. "
 "승인 후 공통 규칙을 더 넓은 공개 입력에 적용해 필요한 손 모양·시간 연속성·실제 화면 밖 미관측을 검증하고, "
 "기존 크기 고정 조건에서 통합 전/통합 후/손가락 근거 반영의 비교 조건을 확정한다. "
 "현재36영상 진단·입력 후보이며 전체1199준비나 학습READY 아님. C·T3-F와 웹 유지, "
 "사용자 촬영은 접근·학습·튜닝·선택 금지. 추가 학습은 사용자 검토 후.")
audit=dict(status=status,at=at,unit=U.name,
    scope=dict(videos=36,word_types=23,people=9,frames=4641,split="Public TRAIN diagnostic set, reused known hard cases; no held-out accuracy claim."),
    implementation="Codex local",verification="Same Codex separate complex-number equations and saved-input reconstruction; not independent reviewer.",
    counts=s["counts"],effects=effect["counts"],controls=effect["controls"],
    independently_reconstructed_hand_comparisons=4095,reconstructed_input_values=1299480,
    previously_largest12_disagreement_angles_flagged=12,visual_frames=35,
    synthetic_contracts=10,protected_files=32,training_runs=0,classifier_evaluations=0,
    source_detector_reruns=0,user_data_reads=0,production_changes=0,
    conclusion="Position and baseline preserved; new finger uncertainty distinguished. True coordinate correction and recognition improvement unproven.",
    important_limits=[
        "Two predictions come from the same detector family; mutual agreement can still be wrong.",
        "Midpoint-disc envelope is a discrepancy diagnostic, not calibrated probabilistic confidence.",
        "Existing4133finger-time ambiguities are diagnosed only, deliberately not filtered; baseline errors can remain.",
        "10/53 normal-crossing control frames carry shape-uncertainty flags; none discarded or re-assigned.",
        "New-point filter removes637hand-local point-times /572body-distal point-times; may lose correct useful shape. No learned performance claim.",
        "No finger-level visual ground truth. 35 frames reviewed, not all88 flagged new-hand frames.",
        "Research280input only; C/T3-F/DTW/web unchanged; full1199 expansion and training not started."
    ],next_single_action=next_action)
write(U/"FINAL_AUDIT.json",audit)

report="""손 위치는 보존하고, 불확실한 손가락만 구분하는 연구 후보

결론
손 전체를 버리지 않고 손의 위치 근거와 정밀한 모양 근거를 나누는 후보를 구현했다.
공개 학습 자료36개 영상(23단어,9명),4,641프레임에서 입력 계산과 보존을 검증했다.
학습·인식률 측정은 하지 않았으며 기존 C·T3-F와 웹/DTW는 변경하지 않았다.

어떻게 판단하나
같은 원본 순간에서 얻은 두 검출을 비교한다. 손바닥을 기준으로 위치·회전·크기를
맞춘 뒤 손가락 마디에 비해 두 좌표가 얼마나 엇갈리는지 본다.
엇갈림이 커서 마디 방향이나 굽힘을 좁혀 말할 수 없으면 그 관절/손가락의 모양을 보류한다.
한 검출만 있거나 비교할 손바닥 기준점이 부족하면 '비교 근거 부족'이다.
잘못된 수어 동작이라고 판단하는 규칙이 아니다. 두 예측이 일치해도 둘 다 틀릴 수 있다.
진단 정렬만 할 뿐 원본 위치 좌표, ROI, 시간, 좌우 배정, 기존 크기 기준은 바꾸지 않는다.
단어명·단어ID·사람ID·특정 수어 각도는 판정에 들어가지 않는다.

수치
이전 단계에서 새로 얻은300개 손 시점 중:
-202: 비교 가능하고 이 규칙으로 큰 충돌이 발견되지 않음(정답 보장 아님).
-88: 일부 손가락의 모양을 확정하기 어려움. 총185개 손가락×프레임.
-10: 비교할 근거 부족. 손의 위치 근거는 보존.
핵심 동작 안에서 새로 얻은81손 시점 중30에 모양 불확실성, 비교불가0.
이전에 큰 각도 불일치가 확인된 상위12개 장면의 해당 각도 모두 보류로 구분.
새 비교가능각도4,146개 중245개는 방향/굽힘 범위를 좁혀 말할 수 없음.
300은 사람/영상 수가 아니라 한쪽 손×한 프레임 수이다.

보존 및 검증
원본 관측180,332관절×프레임, 기존 유효 입력, 손목·손바닥 위치 근거·좌우·시간 보존.
저장된 정상3장면과 교차53프레임 모두 유지. 교차10프레임에서는 손가락 불확실성도
검출됐으나 정상 손/교차를 삭제하거나 좌우를 바꾸지 않았다.
별도 복소수 수식으로4,095쌍 비교를 검산했고 연구 입력1,299,480값이 일치했다.
이동·회전·크기 변경, 접힌 손, 한 점 누락 등10개 모의 조건 검사 통과.
실제 공개35장면의 원본·확대·표시를 확인했다. 전체 의심 관절의 정답 검수는 아니다.
구현/검산/시각검토는 같은 Codex이며 별도 사람이나 에이전트의 검토가 아니다.

입력 후보의 실제 변화
진단은 기존/추가 관측 모두에 적용하되 기존 관측의 입력값은 그대로 두었다.
선택적으로 적용할 연구 입력에서는 새 관측의 불확실/비교불가 부분만 모양 근거에서 보류했다.
몸 기준의 손목·손바닥기준점은 보존하고, 불확실한 말단 손가락은 몸 기준/손 로컬 두 경로에서
함께 보류했다. 손바닥기준점도 모양 근거가 부족하면 손 로컬값만 보류한다.
추가된 손 로컬 유효점5,938→5,301(637보류), 몸 기준 추가 유효점5,938→5,366(572보류).
원시 좌표를 삭제하거나 다른 좌표로 채운 것은 아니다. 별도 연구 배열이며 운영 미적용.

남은 문제
불확실성 구분이 손가락 정답 복원이나 인식률 상승을 뜻하지 않는다.
현재 관측끼리의 불일치이지 '몇% 정확함'이라는 확률이 아니다.
기존 관측에서도4,133개 손가락 시점에 불확실성 표시가 있어 기존 문제는 남는다.
이를 일괄 삭제하면 정상 교차에서도 유효한 정보가 줄어들 수 있다.
새 관측에서도 올바른 모양을 일부 보류할 수 있으므로 학습 효과는 별도 검증해야 한다.
고정 상반신 범위·전체 원본 검출을 유지했고 이번에는 센서를 재실행하지 않았다.
모든 단어에 적용 가능한 공통 규칙이지만 실제 확인 범위는23단어36영상이다.
전체1,199영상 준비·학습 완료나 새 사용자 일반화를 주장하지 않는다.

다음 단계
이번 결과를 함께 확인한 뒤 더 넓은 공개 입력에서 필요한 모양과 시간 연속성을 보존하는지
확인한다. 크기 기준을 고정하고 통합 전/통합 후/손가락 근거 반영을 나눠 비교하는 조건을
확정한다. 추가 학습은 사용자 검토 후 진행한다. 사용자 촬영은 사용하지 않았다.
"""
text(U/"RESULT_KO.txt",report)
gallery=""
for i,t in enumerate(visual["targets"],1):
    gallery+=f'<figure><img loading="lazy" src="evidence/review_{i:02d}.jpg" alt="{html.escape(t["label"])}의 공개 관측 검토"><figcaption>{html.escape(t["group"])} · {html.escape(t["label"])} · 프레임 {t["frame_id"]}</figcaption></figure>'
page=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>손 위치 보존 · 손가락별 근거 검증</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:#172b42;font:17px/1.7 'Malgun Gothic',sans-serif}}
main{{max-width:1100px;margin:auto;padding:48px 24px 90px}}h1{{font-size:38px;line-height:1.35;letter-spacing:-1.4px}}h2{{font-size:25px;margin:0 0 18px}}
section{{background:white;border:1px solid #dce4ec;border-radius:18px;padding:28px;margin:22px 0}}p{{margin:12px 0}}
.badge{{color:#225fa8;font-weight:bold;letter-spacing:1px}}.note{{border-left:5px solid #e7a02b;background:#fff4df;padding:18px 22px;border-radius:6px}}
.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.metric{{background:#eaf1fa;border-radius:12px;padding:18px}}.metric b{{display:block;font-size:28px;color:#1c5ba2}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:13px 12px;border-bottom:1px solid #dce4ec;text-align:left;vertical-align:top}}th{{color:#416079;background:#f1f5fa}}img{{max-width:100%;height:auto;border-radius:8px}}figure{{margin:18px 0}}figcaption,.muted{{font-size:14px;color:#5c7085}}a{{color:#165f9d}}details{{margin-top:20px}}summary{{cursor:pointer;font-weight:bold}}.flow{{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap}}.flow div{{flex:1;min-width:180px;padding:18px;background:#edf4fb;border-radius:10px}}
@media(max-width:700px){{main{{padding:20px 14px}}h1{{font-size:29px}}section{{padding:20px}}.metrics{{grid-template-columns:1fr}}table{{font-size:14px}}}}
</style><main><div class="badge">WORDS IN SILENCE · 공개 자료 입력 검증</div>
<h1>손의 위치는 살리고,<br>확신하기 어려운 손가락은 따로 표시했다.</h1>
<div class="note"><b>연구 후보 구현·입력 검증 완료</b><br>손가락 정답을 복원하거나 인식률을 높였다는 결과는 아직 아니다. C·T3-F와 웹은 그대로이며 학습은 하지 않았다.</div>
<div class="metrics"><div class="metric"><b>36개 영상</b>공개 학습 자료 · 23단어 · 9명</div><div class="metric"><b>4,641프레임</b>완료된 관측을 같은 조건에서 대조</div><div class="metric"><b>기존 값 보존</b>손 위치·좌우·시간·기존 유효 입력</div></div>
<section><h2>무엇이 달라졌나?</h2><p>이전에는 손의 자리가 대략 맞으면 세부 손가락도 믿을 수 있다고 오해할 여지가 있었다. 이제 <b>“손은 이곳에 있다”</b>와 <b>“약지가 얼마나 굽었는지는 아직 모르겠다”</b>를 함께 표현한다.</p>
<div class="flow"><div><b>① 손 위치 유지</b><br>현재 손목·손바닥 위치를 보존한다.</div><div><b>② 손 모양만 대조</b><br>두 검출의 손바닥 위치·회전·크기를 맞춘다.</div><div><b>③ 손가락별 표시</b><br>마디에 비해 좌표가 크게 엇갈리면 모양 판단을 보류한다.</div></div>
<p>예를 들어 짧은 손가락 마디에서 좌표가 조금만 바뀌어도 굽힘 각도는 크게 달라질 수 있다. 이때 손 전체를 없애거나 다른 모양으로 억지로 채우지 않는다.</p>
<p class="muted">정렬은 비교용 계산이다. 원본 좌표나 화면 범위, 학습용 손 크기 기준을 바꾸는 작업이 아니다. 단어별 예외나 특정 각도 문턱은 사용하지 않았다.</p></section>
<section><h2>새로 찾았던 손 300개 시점의 결과</h2>
<table><tr><th>분류</th><th>수</th><th>뜻</th></tr>
<tr><td>큰 모양 충돌 미발견</td><td><b>202</b></td><td>두 검출을 비교할 수 있고 이번 규칙에 걸리지 않음. 정답 보장은 아님.</td></tr>
<tr><td>일부 손가락 판단 보류</td><td><b>88</b></td><td>총 185개 손가락 시점에서 방향·굽힘을 확정하기 어려움.</td></tr>
<tr><td>비교 근거 부족</td><td><b>10</b></td><td>비교할 손바닥 기준점 등이 부족함. 위치 근거는 유지.</td></tr></table>
<p><b>300은 사람 수나 영상 수가 아니라 ‘한쪽 손 × 한 프레임’ 수다.</b> 핵심 동작 안에서 추가된 81개 손 시점 중 30개에서도 일부 모양 불확실성이 있었다.</p>
<p>이전에 각도가 크게 엇갈린 상위 12개 장면은 해당 각도를 모두 보류로 구분했다. 아래는 그중 같은 장면의 약지가 약 172°와 5°로 달랐던 사례다.</p>
<img src="evidence/review_01.jpg" alt="냉커피 장면의 손 위치는 유지하고 불확실한 손가락 관절을 주황색으로 표시">
<p class="muted">주황색은 “수어를 틀렸다”는 표시가 아니다. 이 이미지에서 정밀한 손가락 모양을 확정할 근거가 약하다는 뜻이다. 확대 그림은 결과 확인용이다.</p></section>
<section><h2>정상 손과 교차는 유지됐나?</h2><table><tr><th>검사</th><th>결과</th></tr>
<tr><td>기존 정상 장면</td><td>3개 모두 유지</td></tr><tr><td>기존 정상 교차</td><td>53프레임 모두 유지</td></tr>
<tr><td>기존 유효 입력값·좌우 배정</td><td>변경 없음</td></tr><tr><td>별도 수식 검산</td><td>손 비교 4,095쌍 · 입력값 1,299,480개 일치</td></tr>
<tr><td>모의 조건 검사</td><td>이동·회전·크기·접힌 손·누락 등 10개 통과</td></tr></table>
<p><b>정상 교차 53프레임 중 10프레임에서도 손가락 불확실성이 나타났다.</b> 좌우가 맞게 배정됐어도 가려진 손가락까지 정확하다는 뜻은 아니기 때문이다. 이 장면들을 버리거나 좌우를 바꾸지 않았다.</p>
<p>기존 손은 진단만 표시하고 입력을 보존했다. 새 손만 대상으로 손가락 근거가 부족한 부분을 보류하는 <b>별도 연구 입력</b>을 준비했다.</p></section>
<section><h2>남은 문제와 적용 범위</h2><ul>
<li><b>정확한 손 모양 복원은 아직 아니다.</b> 두 검출이 같은 방식으로 틀리면 이번 규칙도 놓칠 수 있다.</li>
<li><b>정상 정보도 줄어들 수 있다.</b> 연구 입력에서 새 손 로컬 관절 시점 637개를 보류했다. 유효 정보가 얼마나 줄었는지 학습 전 추가 검토가 필요하다.</li>
<li><b>기존 관측의 문제는 남긴 상태다.</b> 기존 관측에도 4,133개 손가락 시점의 불확실성이 표시됐지만 일괄 삭제하지 않았다.</li>
<li><b>모든 수어 검증은 아니다.</b> 공통 규칙이지만 실제 확인 범위는 23단어·36영상이다. 전체 1,199개 준비나 새 사용자 성능 검증은 하지 않았다.</li></ul>
<p>원본 35장면을 직접 대조했다. 모든 관절의 정답을 사람이 표시한 자료는 없어 오류 검출 정확도나 인식률 개선 수치를 제시하지 않는다.</p></section>
<section><h2>다음 작업</h2><p>우선 이번 구분 방식과 정보 보류의 영향을 함께 확인한다. 이후 더 넓은 공개 입력에서 필요한 손 모양과 시간 연속성이 유지되는지 확인하고, <b>크기 기준을 고정한 통합 전·통합 후·손가락 근거 반영 비교</b>를 준비하는 것이 다음 순서다.</p>
<p><b>추가 학습은 결과 확인 후 진행한다.</b> 사용자 촬영은 접근·학습·튜닝·선택에 사용하지 않았다. 기존 C·T3-F, 웹, DTW는 변경하지 않았다.</p></section>
<section><h2>원본 대조와 근거 파일</h2><details><summary>검토한 35개 장면 보기</summary>{gallery}</details>
<p><a href="RESULT_KO.txt">자세한 설명</a> · <a href="VERIFY.json">별도 수식 검산</a> · <a href="EFFECT_AUDIT.json">정보 보류 영향</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="FINAL_AUDIT.json">최종 상태</a></p>
<p class="muted">구현·별도 수식 검산·시각 검토는 동일한 Codex가 수행했다. 별도 사람이나 에이전트가 독립 검토한 결과로 표현하지 않는다.</p></section></main></html>"""
text(U/"RESULT_KO.html",page)

# Check local evidence links, source syntax and JSON before root documentation update.
for p in U.glob("*.py"):ast.parse(p.read_text("utf8"),filename=str(p))
for p in U.glob("*.json"):read(p)
for rel in re.findall(r'(?:src|href)="([^"]+)"',page):
    assert (U/rel).is_file(),rel
for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
    assert sha(p)==sha(U/"before"/p.name),"Concurrent root change: "+str(p)
summary=f"""### 2026-10-20 손 위치 보존·손가락별 불확실성 후보 검증 — 연구 입력만, 학습 전 검토
- 단위 `ops/{U.name}`. `RESULT_KO.html`/`FINAL_AUDIT.json`. 공개TRAIN36영상·23단어·9명·4641프레임 재사용. 전체1199준비 아님.
- 손바닥 위치·회전·크기를 비교용으로 맞춘 뒤 마디길이 대비 관측불일치/전각도불확정을 구분. 단어·사람 예외/일률각도문턱 없음. 실제 위치/좌우/ROI/시간/기존정규화값 불변.
- 추가300손시점: 충돌미발견202/손가락일부불확실88/비교부족10. 불확실185손가락시점, 비교각도4146중245보류. 이전최대불일치12해당각도 모두표시. 정답복원/인식률향상 아님.
- 원시180332관절시점·기존유효입력·위치기준점 보존. 정상3/교차53유지. 교차10프레임형상불확실은좌우오류아니며삭제안함. 기존4133손가락불확실진단은입력유지.
- 별도선택연구280입력은 새관측의불확실/비교부족만보류. 추가local5938→5301(637보류),body5938→5366(572보류). 몸기준손목/MCP위치보존,말단은양경로일관보류. 실제유효형상손실가능성남음.
- 별도복소수수식4095비교·1299480입력값일치/모의10PASS/원본35장면검토. 같은Codex수행,별도검토자없음. 동일검출기상관오류·정밀정답부재·시간검증미완 한계.
- C·T3-F/운영32보호파일유지. 학습/분류평가/재검출/사용자접근0. 학습READY없음. 진행종료·잠금없음.

"""
cp=ROOT/"WordsInSilence_PROJECT_CONTEXT.md";contents=cp.read_text("utf8")
assert "## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n" in contents
assert "## 9. 다음 단일 작업\n" in contents
contents=contents.replace("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n",
    "## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n"+summary,1)
contents=contents.replace("## 9. 다음 단일 작업\n",
    "## 9. 다음 단일 작업\n\n### 2026-10-20 손가락 근거 후보 결과 검토\n"+next_action+"\n\n",1)
cp.write_text(contents,encoding="utf8")
state=read(O/"STATE.json")
state.update(updated_at=at,current_stage="WORD100_FINGER_EVIDENCE_VERIFIED_RESEARCH_ONLY",
             next_single_action=next_action)
state[U.name]={**audit,"report":str(U/"RESULT_KO.html")}
(O/"STATE.json").write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
(ROOT/"NEW_CHAT_HANDOFF.md").write_text(f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 전체원본 검출·통합 후 고정상반신 입력에서 손 위치와 정밀 손가락 근거를 분리한다.
마지막 완료: 공통 손가락불확실성 후보·연구입력 검증. 정상/교차/기존입력 보존,새관측 불확실구분. 학습·운영미적용.
다음 한 작업: {next_action}

{summary}
- 현재 산출물 `candidate.py`, `FROZEN.json`, `SUMMARY.json`, `VERIFY.json`, `EFFECT_AUDIT.json`, `VISUAL_REVIEW.json`, `FINAL_AUDIT.json`.
- `evidence/<clip>.npz`: 원시위치,손목/손바닥근거,관절/마디/각도별 비교가능·불확실·누락,기존/추가출처. angle_available만으로확정금지;angle_unresolved와함께판독.
- `inputs/<clip>.npz`: 고정손크기,새관측만모양보류한280차원 연구배열. 기존 C 입력이나 운영에 연결하지 않았다. 전체raw좌표는보존. 손목은local에서도보존;MCP는몸기준위치보존/local모양선택보류.
- 엄지~새끼손가락에동일규칙. 두예측중점/반차이반경이마디방향을좁히지못하면보류. 확률적신뢰도나해부학적오류기준아님. 손바닥정렬은진단만,원본정규화불변.
- 앞선 `word100_merge_scale_input_20261020` 세조건108배열과 `word100_fullframe_hand_merge_20261020` 통합결과는읽기전용. 검출·완료검산 반복불필요.
- 기존21충돌(미배정후보포함18/반대배정근거3),기존좌우/형상오류는이번작업으로수정되지않음.
- 검증전모의CONTRACTS.json10PASS 후각도전체범위보류조건추가;공개실행전CONTRACTS_V2.json10PASS와최종코드를동결. 공개결과후계수조정없음.
- 구현/별도수식검산/시각검토 같은Codex. 35장면시각검토이지모든88불확실시점의정답주석아님. 정상좌우보존과정밀형상정답을혼동하지않는다.
- 보호32파일해시불변,학습0/분류평가0/재검출0/사용자촬영접근0. 새프로세스/잠금없음. 문서/링크/JSON/Python구문검사완료.
""",encoding="utf8")
assert read(O/"STATE.json")["next_single_action"]==next_action
assert U.name in cp.read_text("utf8").split("## 8. 현재 상태",1)[1].split("###",2)[1]
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
write(U/"CLOSEOUT.json",dict(status="PASS",at=at,protected_files=32,static_links_verified=True,
    browser_layout_verified=False,report=str(U/"RESULT_KO.html"),
    code_hashes={str(p):sha(p) for p in U.glob("*.py")},
    docs_hashes={str(p):sha(p) for p in [cp,ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]}))
print(status)
