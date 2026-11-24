"""Publish verified input-only findings and synchronize the current handoff."""
import ast
import hashlib
import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

U=Path(__file__).resolve().parent;O=U.parent;ROOT=O.parents[2]
P=O/"word100_fullframe_hand_merge_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open("x",encoding="utf8") as fp:json.dump(x,fp,ensure_ascii=False,indent=2,allow_nan=False)
def text(p,x):
    with Path(p).open("x",encoding="utf8") as fp:fp.write(x)
f=read(U/"FROZEN.json");s=read(U/"INPUT_SUMMARY.json");v=read(U/"VERIFY.json")
assert v["status"]=="INPUT_ARITHMETIC_AND_PRESERVATION_PASS"
assert read(U/"CONTRACT_CHECKS.json")["count"]==6
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
for item in read(U/"INDEX.json"):
    for p,h in item["outputs"].items():assert sha(p)==h,p
at=datetime.now(timezone(timedelta(hours=9))).isoformat()
review=dict(
    at=at,reviewer="Codex same implementer; not external expert or independent agent",
    viewed_images=[str(U/"evidence"/f"angle_sheet_{n}.jpg") for n in range(1,5)],
    target_count=12,selection="12 highest full-frame vs local-crop angle disagreement hand-frames; not representative accuracy sample",
    reused_prior_review=str(P/"VISUAL_REVIEW.json"),
    observed="Hands remain in the corresponding source regions, while folded/foreshortened/blurred finger chains can disagree substantially. No anatomical ground-truth joint labels were created. Some views show fingers projected to short or overlapping segments; neither detector result is automatically accepted as true shape.",
    examples=[
        dict(clip="NIA_SL_WORD0106_REAL01_F",frame=53,side="left",finding="Folded hand; ring distal angle172.2 vs4.7 degrees, corresponding joint shifts up to22.87px. Position agreement did not ensure finger shape."),
        dict(clip="NIA_SL_WORD0185_REAL03_F",frame=120,side="right",finding="Downward fingers; one projected bone1.09px in full detection, angle15.5 vs173.9. Extremely short projected segments amplify angle differences."),
        dict(clip="NIA_SL_WORD1139_REAL01_F",frame=48,side="left",finding="Hand near edge-on; thumb geometry172.1 vs24.3. Current image does not establish both predictions as equivalent."),
    ],
    conclusion="Location availability and angular computability are separate. Do not use all added fingers as reliable precise-shape evidence. Existing baseline coordinates preserved, new semantic quality gate unresolved.",
)
write(U/"VISUAL_REVIEW.json",review)
status="INPUT_CONTRAST_VERIFIED_SHAPE_QUALITY_REQUIRES_REVIEW"
next_action=("세 조건 입력 대조 결과를 사용자와 검토한다. 통합후 크기고정은 기존값 보존, 크기재계산은정규화변화만분리됨. "
             "다음보완은 새관측의 손위치 근거와 정밀손가락 근거를 분리하고, 어깨폭기준 점거리합의만으로 굽힘신뢰를 인정하지 않는 공통품질 후보 검증. "
             "일률각도문턱/특정단어예외/좌표보간으로 덮지 않는다. 몸기준손가락XY와손로컬XY 모두의 미관측일관성을 고려한다. "
             "현재36영상/108배열 범위이며 전체1199준비·학습READY아님. C·T3-F/운영웹 보존, 사용자자료학습·튜닝·선택금지, 추가학습은결과확인후.")
audit=dict(at=at,status=status,scope=dict(clips=36,words=23,train_signers=9,frames=4641,
            arrays=108,full1199_ready=False),verification=v,summary=s["totals"],
            angle_disagreement=s["angle_disagreement_deg"],
            exact_shape_ground_truth_available=False,training_ready=False,
            production_changes=0,training_runs=0,classifier_evaluations=0,user_recording_reads=0,
            detector_reruns=0,protected_file_count=len(f["protected"]),
            visual_review=str(U/"VISUAL_REVIEW.json"),next_single_action=next_action)
write(U/"FINAL_AUDIT.json",audit)
report=f"""세 조건 입력 검증 결과
작성 {at}

결론: 세 조건 입력의 계산과 정상 관측 보존은 통과했다. 그러나 추가 손의 정밀 손가락 모양 신뢰는 아직 해결되지 않아 학습 준비 완료로 판정하지 않는다. 학습0/분류평가0/재검출0/사용자촬영접근0/운영변경0.

범위: 이전 재검출 완료 공개 TRAIN36영상/23단어/9명,4641프레임 전체.3조건108개 입력 배열. 전체1199개 학습 입력 준비 완료가 아니다.
full_before: 통합전관측+통합전영상별손크기.
merge_fixed_scale: 통합후관측+통합전의동일손크기. 기존유효입력변경0.
merge_recomputed_scale: 통합후관측+재계산손크기. fixed와몸기준/마스크/시간동일, 손로컬153442기존값변화(35영상). 이 변화는예상된정규화효과이며오류나성능저하판정아님.

독립수식검산:3898440입력값일치,최대차0. 미관측좌표0+마스크0. 시간/프레임/280차원보존. 기존정상3장면/정상교차53프레임 유지. 기존관측관절174394개시점·기존계산가능굽힘59153개시점보존. 기존오류수정이나새일반화검증을뜻하지않는다.
영역밖165624점교란에도통합후입력동일. 기존시각검토321개재사용+새각도불일치상위12개원본시각대조. 특정단어계수없음. 담당Codex,검산별도구현이나동일수행자.

손위치/손모양근거분리:
- 새관측300손·프레임,5938관절·프레임. 그중주석본동작81손·프레임.
- 새관측중손바닥5점모두가능290,완전하지않음10. 관절이하나라도있는것을완전한손위치로표현하지않음.
- 새관측중15개굽힘각도모두계산가능259,일부또는전부불가41.
- 계산가능굽힘59153→63313(+4160각도·프레임). 이는정답수/정밀도아님.
- 각도가능마스크,손목가능마스크,손바닥가능마스크,기존/추가관측출처를각각저장. 진짜손모양정답확인은아직없음.
- 위근거배열은진단용이며280학습채널에새기능으로주입하지않았다. 새품질마스크를임의로학습입력에적용하지않았으므로불확실한추가손가락문제는남아있다.

발견: 같은장면전체/부분검사의추가관절4160각도차는중앙값3.15도,95백분위23.51도,최대167.47도. 임계값이나사용자동작허용범위가아니라두검출결과의불일치다.
냉커피01f53왼손약지끝마디:172.2도와4.7도. 점위치는어깨폭대비허용거리안이어도손가락마디길이에비하면큰차이가될수있다. 가림/손회전/흐림/매우짧은투영뼈대에서민감하다. 어느검출기가정답인지이검사만으로확정하지않는다.
손크기분모변경은정규화전부터있는각도불일치를해결하지않는다. 기존C·T3-F인식률저하가발생했다고판정한것도아니다.

남은21충돌은기존과동일:미배정후보포함18/반대배정근거3,새로덮어쓰기없음. 전체영상에서관측한단서를실제웹캠화면밖에서도얻는다고가정하면안된다.

다음: {next_action}
"""
text(U/"RESULT_KO.txt",report)
page=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>세 조건 입력 검증 · WordsInSilence</title>
<style>body{{margin:0;background:#f3f6fb;color:#172538;font:17px/1.65 "Malgun Gothic",sans-serif}}main{{max-width:1080px;margin:40px auto;padding:0 24px 60px}}h1{{font-size:36px;line-height:1.3}}h2{{font-size:23px}}section{{background:white;padding:28px;margin:24px 0;border-radius:18px}}.badge{{color:#1c6a65;font-weight:bold}}.warning{{background:#fff1dc;border-left:5px solid #d48b17;padding:20px}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{text-align:left;padding:14px;border-bottom:1px solid #dfe5ef}}img{{width:100%;height:auto;border-radius:10px}}.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metric{{flex:1;padding:20px;background:#edf4fa;border-radius:12px;min-width:180px}}.metric b{{display:block;font-size:29px;color:#176c77}}a{{color:#176b98}}details{{margin:18px 0}}</style>
<main><p class="badge">WordsInSilence · 공개 자료 입력 검증</p><h1>관측 추가와 손 크기 변화는 분리됐다.<br>정밀한 손가락 모양에는 보완이 남았다.</h1>
<div class="warning"><b>입력 계산·기존 관측 보존 통과 / 정밀 손 모양 신뢰 검토 필요</b><br>이번에는 학습하지 않았다. 기존 C·T3-F와 웹 환경은 유지했다.</div>
<section><h2>같은 영상, 세 가지 입력</h2><table><tr><th>조건</th><th>손 관측</th><th>손 크기 기준</th><th>확인한 차이</th></tr>
<tr><td>통합 전</td><td>기존 관측</td><td>기존 기준</td><td>비교 기준</td></tr>
<tr><td>통합 후·크기 고정</td><td>빠진 손 추가</td><td>기존과 동일</td><td>기존에 유효했던 입력값 변화 0</td></tr>
<tr><td>통합 후·크기 재계산</td><td>빠진 손 추가</td><td>추가 관측으로 재계산</td><td>손 로컬 값 변화 / 몸 기준·마스크·시간 동일</td></tr></table>
<p>기존 C 모델과 구분되는 연구 입력 조건이다. 세 조건 모두 전체 원본에서 검출·통합한 뒤 고정 상반신 영역을 적용했다.</p>
<div class="metrics"><div class="metric"><b>36개 영상</b>23단어 · 공개 학습 촬영자 9명</div><div class="metric"><b>108개 입력</b>4,641프레임 × 3조건</div><div class="metric"><b>0회 학습</b>입력 검증만 완료</div></div>
<p>전체 1,199개 학습·개발 영상의 준비 완료를 뜻하지 않는다.</p></section>
<section><h2>정상 손과 교차는 유지됐나?</h2><p><b>저장된 정상 3장면과 정상 교차 53프레임이 유지됐다.</b> 기존 관절 좌표·좌우·시간을 바꾸지 않았고, 크기 고정 조건에서는 기존의 유효한 모델 입력값도 그대로였다.</p>
<p>108개 배열의 3,898,440개 입력값을 별도 수식으로 다시 계산해 일치했다. 크기 재계산 조건에서 달라진 기존 손 로컬 값 153,442개는 분모가 바뀌면서 생기는 예상된 변화다.</p>
<p>이는 기존 관측을 보존했다는 검사이며, 모든 손의 실제 좌우·손가락 정답을 새로 검증했다는 뜻은 아니다.</p></section>
<section><h2>손 위치와 손 모양의 근거를 따로 저장했다</h2><table><tr><th>추가된 300개 손 시점 중</th><th>결과</th><th>의미</th></tr>
<tr><td>손바닥의 기준 5점이 모두 있음</td><td>290</td><td>손바닥 위치를 계산할 근거가 있음</td></tr>
<tr><td>15개 손가락 굽힘 각도를 모두 계산할 수 있음</td><td>259</td><td>필요한 관절점이 있음. 정확한 모양 보장은 아님</td></tr>
<tr><td>일부 또는 전체 굽힘 각도를 계산할 수 없음</td><td>41</td><td>손이 추가됐어도 모든 굽힘을 확인할 수 없음</td></tr></table>
<p>300은 영상 수가 아니라 ‘한쪽 손 × 한 프레임’의 수다. 관측 여부·각도 계산 가능 여부·기존/추가 출처를 구분했다. 진단 정보는 모델의 새 채널로 추가하지 않았다.</p></section>
<section><h2>중요한 발견: 손의 자리가 비슷해도 굽힘은 다를 수 있다</h2>
<p>같은 장면을 전체 화면과 손 주변으로 검사한 결과를 비교했다. 계산 가능한 추가 각도 4,160개의 차이는 중앙값 <b>3.15°</b>, 95백분위 <b>23.51°</b>, 최대 <b>167.47°</b>였다.</p>
<img src="evidence/angle_01.jpg" alt="냉커피 같은 프레임의 전체검사와 부분검사 손가락 각도 불일치">
<p>위 장면의 약지 끝마디는 전체 검사에서 약 172°, 부분 검사에서 약 5°로 계산됐다. 같은 손의 같은 순간이다. 이는 사용자 동작 차이도, 동작 허용 기준도 아니다.</p>
<div class="warning">기존 합의 기준은 어깨 폭에 비해 관절점이 가까운지를 본다. 그 차이가 작은 손가락 마디에 비해서는 클 수 있다. 손 크기 정규화 이전부터 존재하는 차이여서, 크기 기준을 바꾸는 것만으로 해결되지 않는다.</div>
<p>불일치 상위 12장면의 원본을 확인했다. 접힌 손, 옆을 향한 손, 흐린 손에서 손가락 선이 다르게 잡힌다. 한 결과를 정답으로 자동 지정하지 않았다.</p>
<details><summary>불일치 상위 12장면 보기</summary>{''.join(f'<img src="evidence/angle_sheet_{n}.jpg" alt="각도 불일치 검토 {n}">' for n in range(1,5))}</details></section>
<section><h2>다음 작업</h2><ol><li>현재 손 위치 근거를 보존하면서, 새 손가락 관절의 불확실성을 구분하는 공통 규칙을 검증한다.</li>
<li>필요하다면 몸 기준 손가락 좌표와 손 로컬 좌표 모두에 같은 미관측 처리가 적용되도록 별도 후보를 준비한다.</li>
<li>정상 손·교차와 실제 웹캠 관측 제약을 확인한 뒤, 전체 입력 준비와 세 조건 학습 비교를 검토한다.</li></ol>
<p><b>학습 준비 완료로 올리지 않았다.</b> 사용자 촬영은 접근·학습·튜닝에 사용하지 않았다. 특정 단어만을 위한 계수나 예외도 추가하지 않았다.</p></section>
<section><h2>검증 자료</h2><p><a href="RESULT_KO.txt">전체 기록</a> · <a href="VERIFY.json">입력 검산</a> · <a href="CONTRACT_CHECKS.json">미관측 등 6개 조건 검사</a> · <a href="ANGLE_GEOMETRY_AUDIT.json">각도 대조</a> · <a href="VISUAL_REVIEW.json">시각 검토</a> · <a href="FINAL_AUDIT.json">최종 상태</a></p>
<p>구현·별도 수식 검산·시각 검토는 Codex가 수행했다. 별도의 사람이나 에이전트가 독립 검토한 것으로 표현하지 않는다.</p></section></main></html>"""
text(U/"RESULT_KO.html",page)
summary=f"""### 2026-10-20 세 조건 입력 대조 검증 — 계산·보존 통과, 추가 손가락 품질 문제로 학습 전 검토
- 단위 `ops/{U.name}`. `RESULT_KO.html`/`FINAL_AUDIT.json`. 기존통합완료공개TRAIN36/23단어/9명/4641프레임,세조건108배열. 전체1199준비아님.
- 통합전/통합후기존크기고정/통합후크기재계산.3898440값독립수식일치,기존크기고정의기존입력변화0. 재계산은기존로컬153442값변화·몸기준/마스크/시간불변.
- 기존정상3/교차53프레임보존. 기존좌표·각도보존이며새좌우정답검증아님. 기존321원본검토재사용,각도불일치상위12원본추가대조.
- 손위치/굽힘/출처분리진단:추가300손시점중손바닥5점290,15각도전부계산가능259/일부불가41. 각도계산가능은정답모양아님. 진단만저장,품질필터임의적용없음.
- 추가4160각도전체/부분검사불일치중앙3.15도/p9523.51도/최대167.47도. 냉커피01f53약지172.2vs4.7도. 점거리어깨비율합의가손가락모양을보증하지않음;정규화전불일치.
- 원본·C/T3-F/운영32보호파일유지,학습/분류평가/재검출/사용자접근0.6개미관측등검사통과. 학습READY발행안함. Codex별도수식검산·시각검토,동일수행자.

"""
for p in [ROOT/"WordsInSilence_PROJECT_CONTEXT.md",ROOT/"NEW_CHAT_HANDOFF.md",O/"STATE.json"]:
    assert sha(p)==sha(U/"before"/p.name),"Concurrent edit "+str(p)
p=ROOT/"WordsInSilence_PROJECT_CONTEXT.md";content=p.read_text("utf8")
content=content.replace("## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n","## 8. 현재 상태 — 반드시 새 대화에서 먼저 확인\n\n"+summary,1)
content=content.replace("## 9. 다음 단일 작업\n","## 9. 다음 단일 작업\n\n### 2026-10-20 세 조건 입력 검증 결과와 손가락 근거 검토\n"+next_action+"\n\n",1)
p.write_text(content,encoding="utf8")
state=read(O/"STATE.json")
state["updated_at"]=at;state["current_stage"]="WORD100_MERGE_SCALE_INPUT_VERIFIED_SHAPE_REVIEW"
state["next_single_action"]=next_action
state[U.name]={**audit,"report":str(U/"RESULT_KO.html")}
(O/"STATE.json").write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
(ROOT/"NEW_CHAT_HANDOFF.md").write_text(f"""# WordsInSilence 최신 인수인계

갱신 {at}

현재 목표: 전체원본 손통합→고정상반신 입력에서 관측추가/크기기준 효과를 분리하고 위치/손가락 근거를 검증한다.
마지막 완료: 세조건36영상108입력 검산 및 손가락불일치 원본대조. 계산·정상보존통과,정밀손모양신뢰미해결.
다음 한 작업: {next_action}

{summary}
- `word100_fullframe_hand_merge_20261020` 이전 완료결과는읽기전용.재검출/검산반복불필요.
- 이번 `inputs/<clip>/full_before.npz`, `merge_fixed_scale.npz`, `merge_recomputed_scale.npz`는모두280차원. 진단`evidence/<clip>.npz`의각도/손바닥/손목/출처를새학습채널로섞지않음.
- 손크기미계산2쌍은fixed에서도미관측,다른손/사람/DEV값보충없음. C와새세조건명혼동금지.
- 추가관측품질검증은전체/부분같은모델일치이며정밀정답아님. `ANGLE_GEOMETRY_AUDIT.json`의pending_visual은생성당시상태,완료는`VISUAL_REVIEW.json`참조.
- 새위험관측은기존형상불확실성을정량화한것. 학습전좌표가망가졌거나C·T3-F인식률이떨어졌다는뜻아님.
- 기존21충돌은미배정후보포함18/반대배정근거3. 기존관측보존으로기존오류도유지됨. 손가락중복/좌우를억지로정하지않음.
- 전체1199입력확대전에이번품질근거결과검토. 고정ROI밖단서와실제웹캠밖미관측차이검증도남음. 센서선크롭자료를새학습에자동혼합하지않음.
- 학습은사용자검토후. 사용자는실사용테스터이며촬영자료학습/튜닝/선택금지. 모델/운영웹/DTW기존보존.
- 실행종료/잠금없음. 구현·검산·시각검토Codex동일수행자. 보고서정적검사예정.
""",encoding="utf8")
# Validate changed report, evidence, Python syntax and synchronized JSON after writing.
for p in U.glob("*.py"):ast.parse(p.read_text("utf8"),filename=str(p))
for p in U.glob("*.json"):read(p)
assert read(O/"STATE.json")["next_single_action"]==next_action
for n in range(1,13):assert (U/"evidence"/f"angle_{n:02d}.jpg").is_file()
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
write(U/"CLOSEOUT.json",dict(status="PASS",at=at,protected_files=len(f["protected"]),
      result=str(U/"RESULT_KO.html"),report_verified="Static links/JSON/images/Python syntax; no browser screenshot claimed.",
      later_code_hashes={str(p):sha(p) for p in U.glob("*.py")}))
handoff=ROOT/"NEW_CHAT_HANDOFF.md"
handoff.write_text(handoff.read_text("utf8").replace("보고서정적검사예정.","보고서링크/JSON/이미지/새Python구문정적검사완료."),encoding="utf8")
print(status)
