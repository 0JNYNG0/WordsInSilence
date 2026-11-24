"""Persist broad validation findings; no detector, training, or deployment."""
import hashlib, html, json, re
from collections import Counter
import numpy as np
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
U=Path(__file__).resolve().parent
ROOT=U.parents[3]
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    with (U/name).open("x",encoding="utf8") as fp:
        if isinstance(value,str):fp.write(value)
        else:json.dump(value,fp,ensure_ascii=False,indent=2)
now=datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
f=read(U/"FROZEN.json");a=read(U/"ANALYSIS.json");c=a["counts"]
v=read(U/"VERIFY.json");lv=read(U/"LOSS_VERIFY.json")
t=read(U/"VISUAL_TARGETS.json");supp=read(U/"REVIEW_SUPPLEMENT.json")
cause=read(U/"LOSS_CAUSE.json")
count_all=Counter();count_core=Counter()
for r in f["rows"]:
    out=U/"outputs"/r["clip_id"]
    with np.load(out/"loss_audit.npz") as la,np.load(out/"evidence.npz") as ev:
        for i,s,k in np.argwhere(la["lost_angles"]):
            if not ev["pair_available"][i,s]:key="no_comparison_pair"
            elif not ev["angle_available"][i,s,k]:key="unassessed_angle_with_pair"
            elif ev["angle_unresolved"][i,s,k]:key="direct_unresolved"
            else:key="assessed_resolved_but_shared_point_held"
            count_all[key]+=1
            if la["core"][i]:count_core[key]+=1
assert dict(count_all)==cause["all_lost_angles"]
assert dict(count_core)==cause["core_lost_angles"]
seen=set((q["clip_id"],q["index"]) for q in t["normal_controls"])
placements=len(seen)
for key in ["loss_targets","motion_targets","allshape_targets"]:
    for q in t[key]:
        placements+=len(q["indices"])
        seen.update((q["clip_id"],i) for i in q["indices"])
for q in supp["motion_exact"]:
    placements+=len(q["exact_indices"])
    seen.update((q["clip_id"],i) for i in q["exact_indices"])
for q in supp["crossing_proxy"]:
    placements+=len(q["indices"])
    seen.update((q["clip_id"],i) for i in q["indices"])
visual=dict(at=now,operator="Codex image inspection, not expert or joint ground truth",
    systematic_raw_controls=20,controls_with_added_hold=supp["normal_controls_with_hold"],
    loss_spans=12,large_transition_examples=8,all_one_hand_angle_loss_events=4,
    sampled_frame_placements=placements,unique_clip_frames=len(seen),
    actual_viewed_files=[str(p) for pattern in ["normal_sheet_*.jpg","loss_sheet_*.jpg",
        "motion_sheet_*.jpg","allshape_sheet_*.jpg","motion_exact_sheet_*.jpg","cross_proxy_*.jpg"]
        for p in sorted((U/"visual").glob(pattern))],
    observations=[
        "Raw20 controls inspected before uncertainty details: open, curled, pointing, contacting/crossing, and inactive other-hand postures; held points0 at these sampled frames. All are existing observations, so this does not certify newly restored shapes.",
        "New wrist-X-order proxy2frames, 공항시장 f56/57: arms cross horizontally while hands remain visibly separate vertically. No position/angle loss; both-angle counts30 preserved across55..58. This is not an identity ground-truth benchmark.",
        "장애인복지관 f69: hand remains visible and approximately edge-on; all15 computable angles withheld. Exact joints not visually certified.",
        "여주시청 f58/f60: an open edge-on hand remains visible; each frame loses15 computable angles. Coarse shape survives in pixels despite uncertain exact joints.",
        "북구주민지원센터 f68: opposite hand is visibly descending by the body; only1 computable angle existed before hold. Not evidence of a missing mandatory sign phase.",
        "응접실 f44: approximately open horizontal hand visible; merged15angles becomes3. 상처 f46: extended-index configuration visible during orientation change. 믿다 f39: open hand approaching contact;15becomes9.",
        "Large measured angular changes have no matching evidence of huge physical articulation in several raw adjacent frames (여수/남양주/외상값); do not interpret removed variation sum as true movement information loss.",
        "Other sampled reductions include curled and edge-on shapes, hand contact and orientation changes. Broad shape visibility does not establish precise2D/3D joint truth.",
    ],
    limitations=["Sampled frames, not exhaustive semantic video annotation.",
        "Core interval is a clip annotation, not per-hand mandatory motion ground truth.",
        "Two same-detector predictions may share errors; unflagged does not imply correct.",
        "Existing errors retained by non-regression contract; no claim all identity errors solved."],
    conclusion="Existing-input/position preservation PASS; new shape withholding harmlessness NOT ESTABLISHED. Do not adopt hard hold or start training.")
write("VISUAL_REVIEW.json",visual)
next_action=("공개100신규영상의 손가락보류 검증 결과를 사용자와 검토한다. 손목·손바닥 위치와 기존입력 보존은 통과했으나 "
    "새 손가락 정보를 점 단위로 제외하면 이웃 각도까지 연쇄 보류됨(핵심구간552중277은 해당각도 자체 보류판정 없음). "
    "후속 제안은 원시좌표/위치/각도별 불확실성을 분리 보존하고 각도별 비교근거만 제한하는 후보를 먼저 검증하는 것. "
    "보류를 정상·오류 정답으로 쓰거나 좌표삭제 후보를 바로 학습하지 않는다. 현재100단어에서 새100영상 진단이며 "
    "전체899/1199 준비 또는 학습READY가 아니다. C·T3-F·웹 유지, 사용자촬영 접근·학습·튜닝·선택 금지. 추가 학습은 결과 확인 후.")
summary=f"""손가락 보류 기준 확대 검증

결론: 손 위치와 원래 입력은 지켰지만, 새로 찾은 손가락의 불확실한 점을 제외하는 후보는 아직 채택하기 어렵다.
이유는 한 점을 보류하면 그 점을 함께 쓰는 이웃 각도까지 계산할 수 없게 되기 때문이다.
단어별 예외나 기준 완화 없이 기존 규칙을 그대로 검증했다.

범위: AI Hub 공개 TRAIN에서 이전36영상과 겹치지 않는100영상,100단어,9명,12,276프레임.
단어당 새 영상1개를 균형 있게 선택했다. 단어마다9명을 모두 본 것은 아니다.
이전36영상과 합치면136영상/100단어/9명/16,917프레임이다.
현재100단어 밖의 수어, 전체899학습영상,1199전체입력,실사용 일반화 검증은 아니다.

보존: 기존 관측417,526관절시점과 기존 유효 입력값,손목·손바닥 기준점,좌우 배정,고정ROI,시간,손 크기 기준을 유지했다.
새 정상 대조20장면에서 추가 보류0. 추가 교차 표본2프레임의 위치/각도 유지. 과거 정상3/교차53검증은 해시로 재사용했다.
정상 대조20장은 기존 관측이므로 새로 복원한 손가락의 정확성을 보증하는 시험은 아니다.

새로운 손 관측:698손시점 중 비교상충 미표시485,손가락 일부 불확실197,비교쌍 부족16.
‘손시점’은 한 프레임의 한 손이며,698개 영상이나 사람을 뜻하지 않는다.
기존 손로컬 관측186,061점시점 → 통합200,003 → 보류후198,583.
새로 얻은13,942점시점 중1,420을 보류했고12,522를 남겼다. 기존값보다 늘었지만 정답좌표/인식률 향상이라는 뜻은 아니다.

남은 문제:41영상에서 핵심 구간의 일부 굽힘 근거 감소,총86손시점.
핵심구간 계산가능108,931각도시점 중552보류(약0.51%). 손 위치는 보존.
552중273은 해당각도 자체 불확실,277은 해당각도 자체는 보류판정이 없지만 공유점을 제외해서 함께 손실,
나머지2는 해당각도/비교쌍 정보부족이다. 보류판정 없음 역시 정답이라는 의미는 아니다.
3영상의4손시점에서는 그때 계산 가능했던 한 손의 각도가 모두 보류됐다. 양손의 모든각도가 동시에 사라진 프레임은0.
그중 북구주민지원센터1시점은 내려가는 손의1개 각도 보류이며,나머지 장애인복지관1/여주시청2시점은15각도 보류다.
핵심구간 일부각도 보류가 이어진 최장 길이는 약0.17초. 짧아도 구별동작일 수 있어 무해하다고 단정하지 않는다.

원본 관찰: 손을 옆으로 세운 열린 모양,검지를 세우는 모양,접촉 직전 등의 유효한 큰 윤곽이 보이면서도 정밀각도는 불확실했다.
화면에서 손 모양이 보이는 것과 그21개 좌표가 정확한 것은 다르다.
큰 계산각도 변화 중에는 실제손이 그만큼 움직였다고 보기 어려운 장면도 있어 잡음과 실제동작을 구분해야 한다.
정밀관절 정답 주석이 없어 ‘필수동작 손실0’이나‘잘못된점만 제거했다’는 결론은 낼 수 없다.

검증: 입력10,311,840값·비교기하9,023쌍을 별도 수식으로 검산,각도손실 합계도 별도 복소수식 검산.
원본 표본 {len(seen)}개 고유영상프레임을 검토했다(그림 배치{placements}개,중복포함).
구현과 별도수식 검산은 같은Codex가 수행했고 다른검토자/수어전문가의 독립판정은 없다.

다음: 원시 손가락 좌표는 보관하고, 위치·각도별 불확실성·실제 미관측을 따로 남기는 후보를 검토한다.
각도 하나가 불확실하다는 이유로 이웃각도까지 삭제하지 않는 처리부터 공개자료에서 비교한다.
사용자 검토 후 입력조건을 확정하고 그 다음에 학습 여부를 결정한다.

C·T3-F와 운영웹32보호파일 유지. 새학습0/분류채점0/사용자자료접근0. 추가 다운로드·외부전송·Git변경 없음.
"""
write("RESULT_KO.txt",summary)
def card(image,caption):
    return f'<figure><a href="visual/{image}"><img loading="lazy" src="visual/{image}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>'
normal_rows="".join(f"<tr><td>{html.escape(q['label'])}</td><td>{q['index']}</td><td>{q['angles_merged']}</td><td>{q['angles_safe']}</td></tr>" for q in supp["normal_controls"])
page=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>손가락 보류 기준 확대 검증</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf2f7;color:#182d44;font:17px/1.8 'Malgun Gothic',sans-serif}}
main{{max-width:1180px;margin:auto;padding:48px 24px 80px}}h1{{font-size:36px;line-height:1.4}}h2{{font-size:25px;margin-top:0}}
section{{background:#fff;border:1px solid #dce5ed;border-radius:18px;padding:28px;margin:22px 0}}p{{margin:12px 0}}
.lead{{font-size:22px;font-weight:bold}}.warn{{border-left:7px solid #c77922}}.good{{border-left:7px solid #238376}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.stat{{background:#eef4fa;border-radius:12px;padding:18px}}
.stat strong{{font-size:30px;display:block}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce5ed}}
th{{background:#edf4fa}}figure{{margin:24px 0}}img{{width:100%;border-radius:10px}}figcaption{{padding:12px;background:#f1f5fa}}
a{{color:#155caa}}.small{{font-size:14px;color:#50657a}}.bars div{{margin:10px 0;padding:8px 12px;background:#dae8fa;border-radius:7px}}
@media(max-width:700px){{main{{padding:20px 10px}}section{{padding:18px}}.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}
</style><main><p>WordsInSilence · 공개 자료 입력 검증 · 2026.10.20</p>
<h1>손은 보이는데, 손가락 정보를<br>너무 많이 보류하고 있지는 않을까?</h1>
<section class="warn"><p class="lead">기존 손 위치·입력은 보존됐다.<br>하지만 새로 찾은 손가락 점을 제외하는 후보는 아직 채택하기 어렵다.</p>
<p>한 관절점을 보류하면, 그 점을 함께 사용하는 이웃 각도도 계산할 수 없어진다. 더 넓은 자료에서 이 연쇄 손실을 확인했다.
기준을 느슨하게 바꾸지 않고, 불확실성 표시와 실제 입력 제외를 나누어 검토해야 한다.</p></section>
<section><h2>1. 무엇을 더 확인했나</h2><div class="grid"><div class="stat"><strong>100개</strong>이전36개와 겹치지 않는 영상</div>
<div class="stat"><strong>100단어 · 9명</strong>단어당 새 영상1개,촬영자 균형선택</div><div class="stat"><strong>12,276프레임</strong>누락 없이 입력 검산</div></div>
<p>AI Hub 공개 학습용 자료만 사용했다. 기존36개와 합치면136영상이지만, 현재100단어 밖의 자료까지 검증한 것은 아니다.
단어마다9명을 모두 조사하거나 전체899개 학습영상을 처리한 단계도 아니다.</p>
<p>원본 전체 영상 검출 → 같은 장면의 재검출 손 통합 → 고정 상반신 범위 적용 → 손 크기 기준 고정 → 손가락 근거 표시 순서를 유지했다.</p></section>
<section class="good"><h2>2. 보존된 것</h2><table><tr><th>확인 대상</th><th>결과</th></tr>
<tr><td>기존 관절·유효 입력</td><td>417,526관절시점 보존,기존 유효값 변화 없음</td></tr>
<tr><td>손 위치</td><td>손목·손바닥 기준점과 기존 좌우 배정 유지</td></tr>
<tr><td>정상 대조</td><td>미리 정한20장면에서 보류로 추가 손실0</td></tr>
<tr><td>교차 대조</td><td>추가 교차 표본2프레임 유지,이전53교차/3정상 검증 재사용</td></tr>
<tr><td>촬영 구간·범위</td><td>시간·고정ROI·손 크기 기준 변화 없음</td></tr></table>
<p>정상 대조20장면은 기존 관측이어서 보존 규칙의 확인이다. 새로 복원한 모든 손가락이 정확하다는 보증은 아니다.
손가락 끝의 일부 새 좌표는 연구 후보에서 제외될 수 있지만,손목·손바닥 위치 기준은 별도로 남긴다.</p>
{card("cross_proxy_01.jpg","팔이 교차해도 손은 서로 다른 높이에 있다. 이 장면에서 기존 양손의30개 계산각도가 유지됐다. X좌표 순서만으로 좌우를 다시 정하지 않았다.")}
<details><summary>정상 대조20장면 수치</summary><table><tr><th>단어</th><th>프레임</th><th>통합후 좌/우 각도수</th><th>보류후</th></tr>{normal_rows}</table></details></section>
<section><h2>3. 복원된 정보와 보류된 정보</h2><p>새로 추가된 <b>698손시점</b> 중485는 두 검출 결과의 상충이 표시되지 않았고,
197은 일부 손가락이 불확실했으며16은 비교할 관측쌍이 부족했다. 손시점은 ‘한 프레임의 한 손’이다.</p>
<div class="bars"><div style="width:93%">기존 손로컬 관측 · 186,061점시점</div><div style="width:100%">새 손 통합 후 · 200,003점시점</div>
<div style="width:99.29%">보류 후보 적용 후 · 198,583점시점</div></div>
<p>새로 얻은13,942점시점 중1,420을 보류하고12,522를 남겼다. 관측량이 늘어난 것은 확인했지만,
정답 좌표 수나 인식률이 늘었다는 뜻은 아니다.</p></section>
<section class="warn"><h2>4. 확인된 핵심 문제: 한 점의 보류가 이웃 각도로 퍼진다</h2>
<div class="grid"><div class="stat"><strong>41영상</strong>핵심구간 일부 굽힘 근거 감소</div>
<div class="stat"><strong>86손시점</strong>한 개 이상 각도가 보류된 순간</div><div class="stat"><strong>최장0.17초</strong>일부 각도 보류가 이어진 구간</div></div>
<p>핵심구간의 계산가능108,931각도시점 중552가 보류됐다(약0.51%). 전체 비율은 작아도 짧은 구별 동작이 포함되면 영향이 클 수 있다.
이 수치는 인식률 감소나 실제 동작 손실률이 아니다.</p>
<table><tr><th>552개가 보류된 직접 원인</th><th>각도시점 수</th></tr>
<tr><td>해당 각도 자체가 불확실함</td><td>273</td></tr>
<tr><td><b>자체 보류판정은 없지만,이웃과 공유하는 점이 제외됨</b></td><td><b>277</b></td></tr>
<tr><td>해당 각도/비교쌍의 근거 부족</td><td>2</td></tr></table>
<p>자체 보류판정이 없다는 것도 정답이라는 뜻은 아니다. 다만 불확실한 각도만 제한하려던 처리가 그보다 넓은 정보를 줄일 수 있다는 근거다.</p>
{card("loss_04.jpg","응접실 표본: 열린 손의 큰 윤곽은 보이지만, f44에서 계산 가능한15개 각도 중3개만 남았다. 정확한 각도 불확실성과 전체 손 모양 부재는 다르다.")}
<p>3영상의4손시점에서는 그때 계산 가능했던 한 손의 굽힘 각도가 모두 보류됐다.
장애인복지관1시점과 여주시청2시점은15각도,북구주민지원센터1시점은1각도였다.
후자는 내려가는 반대손으로 보여 필수동작 손실로 분류하지 않았다. 양손 전체각도가 동시에 모두 사라진 프레임은 없었다.</p>
{card("allshape_03.jpg","여주시청 표본: 옆으로 세운 손의 윤곽은 남아 있지만 f58의15개 계산각도는 모두 보류됐다. 손가락이 옆에서 겹쳐 보이는 상황의 한계다.")}
<details><summary>나머지 보류 장면과 시간 그래프</summary>
{''.join(card(f"loss_sheet_{i:02d}.jpg","회색=기존,파랑=통합후,주황=보류후의 계산가능 각도 수. 높을수록 정답이라는 뜻은 아니다.") for i in range(1,7))}
</details></section>
<section><h2>5. 실제 움직임과 좌표 흔들림도 구분해야 한다</h2>
<p>옆으로 보이는 손,펴진 손,검지를 세운 손,손끼리 닿기 직전 등에서도 보류가 발생했다.
영상에 큰 손 모양은 보여도 각 마디의 정확한 좌표가 보이는 것은 아니다.
큰 각도 변화 수치 역시 실제 손가락 움직임 대신 검출기의 좌표 흔들림일 수 있었다.</p>
{card("motion_exact_sheet_02.jpg","큰 각도 변화의 직전·해당·다음 원본 장면. 손의 큰 윤곽은 비교적 이어지지만 계산각도는 크게 바뀔 수 있다.")}
<p>따라서 ‘불확실한 좌표를 표시하는 기능’은 유용하지만,그 표시를 곧바로 ‘잘못된 동작’이나 ‘삭제해도 되는 정보’로 사용할 수 없다.</p></section>
<section><h2>6. 다음 단계 제안</h2><ol>
<li><b>원래 좌표와 손 위치는 보관한다.</b> 불확실한 손가락이 있어도 손 전체를 없애지 않는다.</li>
<li><b>각도별 비교 가능 여부를 따로 남긴다.</b> 한 점을 지워 이웃 각도까지 함께 잃는 처리를 분리한다.
DTW에서는 충분히 관측한 각도만 근거로 사용하고,비교하지 못한 항목은 ‘유사함’으로 처리하지 않는다.</li>
<li><b>학습 입력 방식은 그 결과를 함께 확인한 뒤 정한다.</b> 현재280채널 입력의 점별 제외 후보는 보류한다.
같은 공개자료에서 열린 손·굽힌 손·검지·정상교차·재등장 보존을 먼저 비교하고,그 다음 학습 조건을 확정한다.</li></ol>
<p>단어 이름에 따른 예외나 사용자 촬영에 맞춘 조정은 없다. C·T3-F와 운영웹은 그대로다.
이번 작업의 새학습·분류채점·사용자촬영 접근은 모두0이다.</p></section>
<section class="small"><h2>검증 범위와 근거</h2><p>별도 수식으로10,311,840입력값/9,023비교쌍 검산.
각도손실 합계도 별도 복소수식으로 검산했다. 원본 {len(seen)}고유영상프레임 표본 검토(그림 배치{placements}개,중복포함).
구현과 검산은 같은Codex가 수행했으며 다른 검토자나 수어전문가의 독립판정은 없다.</p>
<p>정밀 관절 정답 주석과 단어별 필수동작 정답이 없어 ‘필수정보 손실0’을 증명하지 못했다.
학습자료에서 한 입력검증이며 새로운 사람의 인식 성능 평가가 아니다.</p>
<p><a href="RESULT_KO.txt">텍스트 보고</a> · <a href="ANALYSIS.json">전수 집계</a> · <a href="LOSS_CAUSE.json">연쇄 보류 원인</a> ·
<a href="VERIFY.json">입력 검산</a> · <a href="LOSS_VERIFY.json">각도 검산</a> · <a href="VISUAL_REVIEW.json">원본 관찰 기록</a></p></section></main></html>"""
write("RESULT_KO.html",page)
for name in re.findall(r'(?:src|href)="([^"#]+)"',page):
    assert (U/name).exists(),name
for p,h in f["protected"].items():assert sha(p)==h,p
for p,h in f["pins"].items():
    if Path(p).suffix==".py":assert sha(p)==h,p
for row in read(U/"INDEX.json"):
    for p,h in row["outputs"].items():assert sha(p)==h,p
assert len(read(U/"INDEX.json"))==100
assert not (U/"RUNNING.json").exists()
execution=read(U/"CURRENT_EXECUTION.json")
execution.update(status="COMPLETE",exit_code=0,completed_at=now,validation_complete=True)
(U/"CURRENT_EXECUTION.json").write_text(json.dumps(execution,ensure_ascii=False,indent=2),encoding="utf8")
audit=dict(at=now,status="BROAD_INPUT_VALIDATION_COMPLETE_HARD_HOLD_NOT_ADOPTED",unit=U.name,
    implementation="Codex local",verification="Same Codex separate equations; no other agent/expert",
    scope=dict(new_clips=100,words=100,people=9,frames=12276,prior_clips_reused=36),
    input_verification=v,loss_verification=lv,counts=c,
    clips_with_core_loss=41,all_angle_loss_hand_frames=4,core_shared_point_collateral_angles=277,
    unique_visual_frames=len(seen),production_hashes_preserved=32,
    rules_changed_after_result=False,training_runs=0,classifier_runs=0,user_data_reads=0,
    decision="Preserve raw observations and diagnostic uncertainty; do not promote point-level shape hold into training/production.",
    limitation="No precise finger truth or mandatory-phase ground truth; existing errors retained; no recognition gain measured.",
    next_single_action=next_action,report=str(U/"RESULT_KO.html"),
    minor_recovery="Read-only supplementary count command initially used system cp949 and failed before reads/writes. Reissued with explicit UTF-8; no output/rule/data overwrite.",
    evidence_hashes={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.suffix in (".py",".json",".txt",".html")})
write("FINAL_AUDIT.json",audit)
write("CLOSEOUT_NEXT.txt",next_action)
print(json.dumps(dict(status=audit["status"],visual_unique=len(seen),placements=placements,report=audit["report"]),ensure_ascii=False))
