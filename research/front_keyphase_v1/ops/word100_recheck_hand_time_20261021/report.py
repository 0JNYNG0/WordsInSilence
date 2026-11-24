import html,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import study as d
U=d.U;f=d.read(U/"FROZEN.json");result=d.read(U/"RESULT.json");v=d.read(U/"VERIFY.json");stress=d.read(U/"STRESS.json")
primary=[r for r in f["rows"] if r["primary"]];summary={};total=Counter()
for qr in primary:
    cid=qr["clip_id"];word=qr["label"]
    z=d.load(U/"rechecked"/(cid+".npz"));q=d.load(qr["prepared"])
    primarymask=np.broadcast_to(q["core"][:,None,None],z["angle_state"].shape)
    c=summary.setdefault(word,Counter())
    for k,num in Counter(z["angle_state"][primarymask].ravel()).items():c["core_state_"+str(k)]+=num
    c["new_core_eligible"]+=int((primarymask&(z["before_state"]==2)&(z["angle_state"]==4)).sum())
    for rr in primary:
        if rr["word_code"]!=qr["word_code"] or rr["signer_id"]==qr["signer_id"]:continue
        pid=cid+"__"+rr["clip_id"]
        old=d.read(U/"pairs"/("recheck_only__"+pid+".json"))
        new=d.read(U/"pairs"/("combined__"+pid+".json"))
        if new["alignment_support"]<.35:c["low_alignment_pairs"]+=1
        for a,b in zip(old["rows"],new["rows"]):
            if not b["required"]:continue
            c["pair_angle_rows"]+=1
            c["smaller_denominator_rows"]+=b["expected_seconds"]<a["expected_seconds"]-1e-9
            if a["strict"]["coverage"] is not None and b["strict"]["coverage"] is not None:
                c["coverage_improved_rows"]+=b["strict"]["coverage"]>a["strict"]["coverage"]+1e-9
            if not b["strict"]["ready"]:
                if not b["alignment_ok"]:c["fail_alignment"]+=1
                if b["strict"]["coverage"] is None or b["strict"]["coverage"]<.5:c["fail_coverage"]+=1
                if min(b["strict"]["query_frames"],b["strict"]["reference_frames"])<8:c["fail_framecount"]+=1
union={}
for p in (U/"schedules").glob("*__NIA*.npz"):
    rid=p.stem.split("__")[1];z=d.load(p)["rest"]
    union[rid]=union.get(rid,np.zeros_like(z))|z
unique=[]
for rr in primary:
    rest=union[rr["clip_id"]];q=d.load(rr["prepared"]);w=d.adapter.cells(q["timestamp_us"])
    for h,side in enumerate(("L","R")):
        unique.append(dict(clip_id=rr["clip_id"],word=rr["label"],side=side,frames=int(rest[:,h].sum()),seconds=float(w@rest[:,h])))
d.write(U/"DIAGNOSIS.json",dict(words=summary,unique_reference_rest=unique,
    caution="Union is across leave-query-out profiles; overlap between profiles is not additional independent observations. Failure reasons overlap.",
    next="Before training: separate evidence for stable lowering, active hand and unknown in the alignment support/cost too; compare with frozen current reference denominator. Diagnose conservative rest misses and alignment mismatch with public references, not relax five-person criteria using outcomes."))
visual=d.read(U/"VISUAL_MANIFEST.json")
d.write(U/"VISUAL_REVIEW.json",dict(at=d.now(),reviewer="Codex visual inspection of all11 contact sheets,108 panels",
    raw_panels=108,pages=11,
    observations=[
        "Newly corroborated hands generally occupy actual visible hands, including overlapping palms, contacting arms and crossings. Existing coordinates not replaced.",
        "Observed lowering panels show the target left arm resting low while the right hand acts near the face. No visibly raised target hand in inspected lowering panels.",
        "Known erroneous duplicated identity at 소복 frame84 remains unassessed, not newly approved. Later clearly separated hands retain existing geometry.",
        "Crossing control panels preserve both arm/hand placements; previously valid angle states are not revoked.",
        "Dense fingertips/occlusion cannot be individually certified from these panels; geometric agreement of two outputs from the same detector is not independent anatomical truth.",
    ],
    limits="Not all210 changed handframes individually annotated; first-per-clip, long runs, old controls and lowering boundaries inspected. No 3D or semantic ground truth; no all-word generalization claim.",
    apparent_new_hand_deletion=False,observed_new_coordinate_mutation=False))
# A readable standalone figure, using actual stored outcomes.
plt.rcParams["font.family"]="Malgun Gothic";plt.rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(figsize=(10,4.2),layout="constrained")
labels=["기존","손 재확인","손별 시간대","두 보완 결합"]
vals=[result["comparison"][c]["total"]["PARTIAL_COMPARISON"] for c in f["conditions"]]
ax.barh(labels[::-1],vals[::-1],color=["#12a890","#3582c4","#3582c4","#7b8b9e"])
ax.set_xlim(0,675);ax.set_xlabel("차이를 표시할 근거가 확보된 각도 항목 / 675개")
for y,n in enumerate(vals[::-1]):ax.text(n+7,y,str(n),va="center",fontsize=12,fontweight="bold")
ax.set_title("비교 가능 항목은 20개 증가 · 단어 인식률을 측정한 결과는 아님",fontsize=14,pad=14)
ax.spines[["top","right"]].set_visible(False);fig.savefig(U/"COMPARISON.png",dpi=150);plt.close(fig)

names=dict(zip(f["conditions"],labels));table=""
for condition in f["conditions"]:
    t=result["comparison"][condition]["total"]
    table+=f'<tr><td>{names[condition]}</td><td>{t["PARTIAL_COMPARISON"]} / 675</td><td>{t["INSUFFICIENT"]}</td></tr>'
wordtable=""
for word in ("수어","후원","멀다"):
    a=result["comparison"]["baseline"]["words"][word].get("PARTIAL_COMPARISON",0)
    b=result["comparison"]["combined"]["words"][word].get("PARTIAL_COMPARISON",0)
    wordtable+=f'<tr><td>{word}</td><td>{a} → <b>{b}</b></td><td>{summary[word]["low_alignment_pairs"]} / 72</td></tr>'
restsummary={word:sum(x["frames"] for x in unique if x["word"]==word and x["side"]=="L") for word in ("수어","후원","멀다")}
body=f"""
<header><span class="tag">공개 자료 검증 · 2026.10.21</span>
<h1>손을 다시 확인하는 일과<br>비교할 시간을 정하는 일을 분리했습니다.</h1>
<p>기존 좌표와 좌우 배정은 보존하며, 이미 찾은 손의 모양을 불필요하게 보류하던 조건을 보완했습니다.<br>
비교 대상의 손이 사라져도 참고 자료에서 정해 둔 비교 범위는 줄어들지 않습니다.</p></header>
<main>
<section><h2>먼저 결과</h2><div class="cards">
<article><b>353 → 373</b><p>675개 각도 항목 중<br>차이를 표시할 근거가 생긴 항목</p></article>
<article><b>54 / 54</b><p>한 손 전체를 지운 시험에서<br>참고 비교 범위 유지·잘 맞음 판정 없음</p></article>
<article><b>302개 남음</b><p>여전히 비교 근거가 부족함<br>정상 동작도 보류될 수 있음</p></article></div>
<p class="notice">이번 숫자는 <strong>TCN 인식률·수어 정답률·동작 일치율이 아닙니다.</strong>
27개 영상에서 손별 15개 각도를 여러 참고 영상과 비교했을 때, 수치를 제시할 근거가 충분한지 센 결과입니다.</p></section>
<section><h2>무엇을 바꿨나</h2><div class="two">
<article><h3>① 이미 찾은 손은 별도로 재확인</h3>
<p>기존에는 이미 찾은 손도 팔 연결로 좌우를 다시 증명해야 손 모양을 비교할 수 있었습니다.
이제 원본 장면에서 다시 찾은 손이 기존 손과 <strong>하나씩 유일하게 대응</strong>하면 모양 근거로 사용할 수 있습니다.</p>
<p>다른 손 후보와 겹쳐 구별이 안 되거나, 현재 팔 연결이 반대쪽을 가리키면 보류합니다.
새 손을 추가하는 엄격한 조건은 그대로이며, 좌표나 좌우 배정을 바꾸지 않았습니다.</p>
<p class="muted">같은 물리적 손에 대한 재확인입니다. 기존 좌우 이름까지 새로 검증됐다는 뜻은 아닙니다.</p></article>
<article><h3>② 손마다 비교해야 할 시간을 따로 기록</h3>
<p>비교 대상 사람을 빼고 나머지 8명의 참고 자료만 봅니다.
최소 5명이 같은 단계에서 안정적으로 손을 내려놓았다는 근거가 있고, 다른 참고에서 그 손이 범위 안에 보이는 반대 근거가 없을 때만 내려놓은 구간으로 분리합니다.</p>
<p><strong>안 보임 = 손내림으로 처리하지 않습니다.</strong> 모르는 구간은 비교 범위 안에 남깁니다.
이 규칙은 단어 이름을 읽지 않는 공통 처리입니다.</p></article></div></section>
<section><h2>같은 자료에서 네 조건을 비교</h2>
<p>수어·후원·멀다, 각 9명씩 총 27개 영상. 한 영상을 나머지 8명과 비교한 216쌍을 사용했습니다.
재확인 보존 검사는 기존 공개 대조군까지 합쳐 100개 단어의 158개 영상·19,611개 저장 프레임에서 수행했습니다.</p>
<table><thead><tr><th>조건</th><th>차이 표시 가능</th><th>근거 부족</th></tr></thead><tbody>{table}</tbody></table>
<img class="chart" src="COMPARISON.png" alt="네 조건의 비교 가능한 각도 항목 수">
<p>시간대만 분리한 경우에는 최종 표시 가능 항목 수가 늘지 않았습니다.
손내림 구간 일부는 올바르게 제외했지만, 5명 이상·50% 이상 관측·8프레임 이상이라는 기존 기준을 새로 넘은 항목은 없었습니다.</p>
<table><thead><tr><th>단어</th><th>기존 → 두 보완 결합</th><th>시간 정렬 근거 부족 쌍</th></tr></thead><tbody>{wordtable}</tbody></table>
<p>멀다의 사용하지 않는 반대손 135개 항목은 별도로 ‘비교 대상 아님’을 유지했습니다. 675개 분모에 포함되지 않습니다.</p></section>
<section><h2>정상 손내림을 어떻게 확인했나</h2>
<p>참고별 조건을 합친 고유 저장 프레임 기준: 후원 왼손 {restsummary["후원"]}개, 멀다 왼손 {restsummary["멀다"]}개에서
여러 사람의 손내림 근거가 확인됐습니다. 같은 프레임이 여러 비교쌍에 반복 등장한 수는 중복 계산하지 않았습니다.</p>
<p>후원은 오른손을 얼굴 옆에서 움직이는 동안 왼손을 내려놓는 장면이 있습니다.
아래 원본 장면 확인에서도 내려놓은 왼손이 보입니다. 멀다 왼손은 이미 선택적 손이라 최종 지표는 그대로입니다.</p>
<details><summary>원본 장면과 관측점 확인: 108패널 · 11쪽</summary>
{''.join(f'<img class="sheet" loading="lazy" src="visuals/review_{i:02}.jpg" alt="원본 관측 검토 {i}">' for i in range(1,12))}
<p>청록색은 기존 왼손, 분홍색은 기존 오른손 좌표입니다. 노란 선은 고정된 학습 관측 범위입니다.
손목 경계 위만 따로 검출한 화면이 아니라 전체 원본 장면에 관측 결과를 겹쳐 표시했습니다.</p></details></section>
<section><h2>안전 확인과 남은 문제</h2><ul>
<li>전체 158개 영상에서 기존 좌표 2,941,650개 수치와 기존 비교 가능 각도를 보존했습니다. 새 좌표 삭제·좌우 교환은 없습니다.</li>
<li>이전 정상·교차·기존 오류 대조 80개 기록을 확인했습니다. 알려진 소복 중복 배정 장면은 여전히 보류됐습니다.</li>
<li>한 손을 모두 지운 54개 시험에서 정렬을 새로 계산했습니다. 총 432개 경로 중 357개가 바뀌어도 참고 시간 분모는 유지됐고, 사라진 손의 비교 점수가 나오지 않았습니다.</li>
<li>각도 하나의 한 프레임만 지운 27개 시험에서도 이웃 각도와 참고 범위는 유지됐습니다. 짧은 누락을 실제 수어 오류로 확정한 시험은 아닙니다.</li>
<li>후원은 여전히 270개 중 243개 항목이 근거 부족입니다. 겹침·미검출과 참고 간 정렬 차이가 남았습니다.
기존 정렬 근거 기준에 미달한 쌍도 그대로 보류했습니다.</li>
<li>두 번의 검출이 비슷하다고 실제 손가락 관절이 정확하다는 보장은 없습니다. 원래의 좌우 배정 오류를 이번 처리로 전부 고쳤다는 뜻도 아닙니다.</li>
</ul><p class="notice">정상 손내림과 한 손 동작을 곧바로 오답으로 만들지 않았습니다.
다만 정상 영상이 충분히 비교되는 수준에는 아직 도달하지 못했습니다. ‘보류 안전성 확인’과 ‘실사용 완성’을 구분합니다.</p></section>
<section><h2>다음 한 작업</h2>
<p><strong>손내림·동작 중·불확실이라는 구분을 시간 정렬의 근거 계산에도 연결할 필요가 있는지 공개 자료로 먼저 비교합니다.</strong>
지금은 최종 비교 범위만 분리했고, 앞 단계의 정렬은 기존 양손 기준을 유지했습니다.
이를 손이 안 보인다는 이유로 점수를 유리하게 만드는 방향이 아닌, 참고 자료에서 확인된 손별 역할을 반영하는 방향으로 검증하는 것이 다음 순서입니다.</p>
<p>기존 C·T3-F와 운영 웹은 그대로입니다. 사용자·멘토 촬영 자료를 읽거나 학습·튜닝에 사용하지 않았으며, 추가 학습도 하지 않았습니다.
이번 결과를 함께 확인한 뒤 다음 범위를 결정합니다.</p></section>
<section class="muted"><h2>검증 범위와 재현 근거</h2>
<p>원본과 이전 결과는 읽기 전용. 구현과 별도 수식 검산은 모두 Codex 수행이며 독립 수어 전문가 검증은 아닙니다.
추가 영상 검출 없이 저장된 공개 관측을 재사용했습니다. 영상 의미·3차원 관절 정답이나 모든 수어의 성능을 증명한 결과가 아닙니다.</p>
<p>FROZEN.json · RESULT.json · VERIFY.json · STRESS.json · DIAGNOSIS.json · VISUAL_REVIEW.json.
최초 시각화에서 과거 파일의 core 키 차이를 발견해 원래 주석 시간으로 읽기 경로만 고쳤습니다.
학습·비교 조건과 결과는 바꾸지 않았습니다.</p></section>
</main>"""
css="""*{box-sizing:border-box}body{margin:0;background:#f2f5f9;color:#172638;font:17px/1.8 'Malgun Gothic',sans-serif}
header{background:#102d49;color:white;padding:64px max(6vw,24px);border-bottom:7px solid #2bb9a5}
header h1{font-size:clamp(30px,3.6vw,46px);line-height:1.4;letter-spacing:-1.2px}header p{color:#d3e5f4;max-width:1050px}
main{max-width:1150px;margin:32px auto;padding:0 24px}section{background:white;border:1px solid #dee7f0;padding:30px;border-radius:18px;margin:24px 0}
h2{margin:0 0 18px;font-size:27px}h3{font-size:21px;margin-top:0}.tag{color:#8ae2d6;font-weight:bold}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.two{display:grid;grid-template-columns:1fr 1fr;gap:24px}
article{background:#f4f8fc;border-radius:13px;padding:22px}.cards b{font-size:32px;color:#0a786a}.cards p{margin-bottom:0}
.notice{background:#e9f5f2;border-left:4px solid #21a18e;padding:16px}.muted{color:#526579;font-size:15px}
table{width:100%;border-collapse:collapse;margin:20px 0}th,td{padding:15px;text-align:left;border-bottom:1px solid #dfe7ee}
th{background:#e9eff5}.chart{width:100%;max-width:950px}.sheet{display:block;width:100%;max-width:960px;margin:20px auto}
summary{cursor:pointer;color:#13677c;font-weight:bold}li{margin:12px 0}@media(max-width:760px){.cards,.two{grid-template-columns:1fr}section{padding:20px}}"""
text="<!doctype html><html lang='ko'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>손 재확인·손별 시간대 검증</title><style>"+css+"</style>"+body+"</html>"
(U/"RESULT_KO.html").write_text(text,encoding="utf8")
(U/"RESULT_KO.txt").write_text(
    "손 재확인·손별 시간대 분리 검증\n"
    "공개3단어9명27영상216쌍, 확대 보존검사100단어158영상19611프레임.\n"
    "기존353/675→재확인373/675, 시간대만353/675, 결합373/675. 단어 인식률 아님.\n"
    f"후원 왼손 손내림 {restsummary['후원']}개 고유 저장프레임 구분. 멀다 왼손 {restsummary['멀다']}개.\n"
    "정상/교차/기존오류80기록. 54개 손전체제거 시험 새정렬432개 중357개 변경해도 참고범위 고정, 누락손 점수 없음. 각도1프레임27시험 이웃/분모 보존.\n"
    "좌표/기존유효각도 보존. 신규210손프레임 재확인, 2638각도프레임 비교 가능. 원본108패널 검토.\n"
    "302개 근거부족 지속; 후원243/270. 형상일치≠해부학적 정확도/좌우 재검증. 시간정렬 앞단은 기존양손기준, 다음 검토 대상.\n"
    "기존 C/T3-F/웹 유지, 사용자자료 접근0, 학습0. 결과 공동 검토 후 다음 작업.\n",encoding="utf8")
assert len(visual["panels"])==108 and len(visual["pages"])==11
d.write(U/"REPORT_CHECK.json",dict(status="STATIC_CHECK_PASS",local_image_links=12,html_utf8=True,
    browser_rendered=False,figure_rendered=True,comparison=result["comparison"],next="public alignment support audit before training"))
print("REPORT",restsummary,dict(summary))
