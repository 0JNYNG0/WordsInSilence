"""Closeout with fair mask comparisons, limitations, and example diagrams."""
import hashlib,html,json,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from candidate import compare_aligned_angles
U=Path(__file__).resolve().parent;O=U.parent
W=O/"word100_finger_preservation_20261020"
read=lambda p:json.loads(Path(p).read_text("utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(name,obj):
    with (U/name).open("x",encoding="utf8") as fp:
        if isinstance(obj,str):fp.write(obj)
        else:json.dump(obj,fp,ensure_ascii=False,indent=2)
f=read(U/"FROZEN.json");s=read(U/"SUMMARY.json");v=read(U/"VERIFY.json")
rows={r["clip_id"]:r for r in f["rows"]}
now=datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
# Public replay of output masking: self-comparison is an API contract, not accuracy.
selfcheck=dict(clips=0,expected=0,compared=0,unknown=0)
for q in read(U/"INDEX.json"):
    z=load(q["path"])
    d=compare_aligned_angles(z,z,z["computable"])
    assert d["expected_count"]==int(z["computable"].sum())
    assert np.isnan(d["difference_deg"][~d["compared"]]).all()
    if d["compared_count"]:assert d["mean_difference_deg"]==0
    selfcheck["clips"]+=1;selfcheck["expected"]+=d["expected_count"];selfcheck["compared"]+=d["compared_count"]
    selfcheck["unknown"]+=d["expected_count"]-d["compared_count"]
selfcheck["meaning"]="Same-data self check verifies masks and fixed denominator only. Not user/reference accuracy."
write("PUBLIC_COMPARISON_CONTRACT.json",selfcheck)
(U/"visual").mkdir()
font=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",19)
small=ImageFont.truetype("C:/Windows/Fonts/malgun.ttf",15)
examples=[]
for cid,i,side in [
    ("NIA_SL_WORD0766_REAL06_F",44,1),("NIA_SL_WORD1691_REAL05_F",58,0),
    ("NIA_SL_WORD1139_REAL06_F",39,0),("NIA_SL_WORD0795_REAL08_F",69,0)]:
    z=load(U/"outputs"/(cid+".npz"));r=rows[cid]
    old=z["old_point_filter_computable"][i,side];raw=z["computable"][i,side]
    state=z["angle_state"][i,side];ok=state==4;rec=z["restored_eligible"][i,side]
    result=dict(clip_id=cid,label=r["label"],index=i,side=side,
        old_point_computable=int(old.sum()),raw_computable=int(raw.sum()),eligible=int(ok.sum()),
        fair_old_eligible=int((ok&old).sum()),restored_eligible=int(rec.sum()),states=state.tolist())
    examples.append(result)
    im=Image.new("RGB",(1200,305),"#f5f8fc");dr=ImageDraw.Draw(im)
    dr.text((20,10),f"{r['label']} · f{i} · {'왼손' if side==0 else '오른손'} · 각 손가락의3개 각도",font=font,fill="#18364d")
    for k,name in enumerate(["엄지","검지","중지","약지","소지"]):
        dr.text((310+k*168,55),name,font=font,fill="#18364d")
    for k,(title,mask) in enumerate([("이전: 점 제외 후 계산 가능",old),("현재: 원래 각도 보관",raw),("현재: 비교 근거 상태",state)]):
        y=95+k*54;dr.text((20,y+8),title,font=small,fill="#18364d")
        for j in range(15):
            x=300+j*56
            if k<2:color="#256d8f" if mask[j] else "#d8dfe7"
            else:color={0:"#d8dfe7",1:"#9da9b7",2:"#8e80b7",3:"#d89b36",4:"#238a77"}[int(mask[j])]
            dr.rectangle((x,y,x+47,y+32),fill=color)
            if k==2 and rec[j]:dr.text((x+15,y+3),"+",font=font,fill="white")
        count=int(mask.sum()) if k<2 else int(ok.sum())
        dr.text((1150,y+5),str(count),font=font,fill="#18364d")
    dr.text((20,264),"파랑: 보관된 각도 / 초록: 비교 근거 있음 / 주황: 관측 상충 / 보라: 추가 확인 부족 / +: 연쇄 손실에서 회복",font=small,fill="#38556f")
    im.save(U/"visual"/f"{cid}_f{i}.png")
write("EXAMPLES.json",examples)
normal9=[q for q in v["controls"] if q["group"]=="systematic20" and q["unresolved"]]
total=s["totals"]["all"];wide=s["totals"]["broader100"]
next_action=("좌표 보존·각도별 근거 후보의 공개136영상 결과를 사용자와 검토한다. 점 삭제의 연쇄손실912각도시점(핵심371)을 "
    "비교후보로 회복했고 원시좌표/기존입력은 불변. 단,교차55장면1650계산각도 중재확인부족735/상충37이므로 "
    "엄격한 비교마스크를 정상/오류 또는 자동 운영차단으로 채택하지 않는다. 다음 제안은 실제미관측·단일관측재확인부족·관측상충을 "
    "분리한 결과표시/비교범위 계약과 위치·궤적 독립 유지 검증이다. 단일관측을임의정답승격하거나문턱완화하지않는다. "
    "기존정상20중9장면45각도상충도동작오류가아님. C·T3-F/웹 유지,사용자자료접근·학습·튜닝·선택 금지. "
    "전체1199준비나 학습READY아니며 추가학습은 함께 결과확인후.")
text=f"""좌표 보존·각도별 비교 근거 분리 검증

변경: 관절점을 제외하지 않고 원래좌표·관측마스크·모델입력을 그대로 보존했다.
각도마다 실제미관측/기하퇴화/추가확인부족/관측상충/비교근거있음을 따로 기록한다.
한 각도가 보류돼도 그 점을 공유하는 이웃각도를 자동 보류하지 않는다.
실제로 같은 불확실한 마디를 쓰는 두 각도는 각각 보류될 수 있다.
비교할 수 없는 각도는차이0이 아닌미비교로 표시하고,참고가요구하는분모를사용자관측누락때문에줄이지않는다.

공개 TRAIN136영상/100단어/9명/16917프레임. 이전36과확대100의 저장자료재사용.
재검출·새다운로드·새학습·분류평가·사용자자료접근0.
미적용100단어 밖·전체899/1199·새사람성능검증은아니다.

전체원시좌표2537550값,모델입력4736760값불변. 좌우·ROI·시간·크기기준불변.
이전점제외가잃던1774각도시점의계산값은모두보존;그중912는각도별근거조건도충족해비교후보로회복(핵심371).
확대100만보면619회복(핵심277),이전36은293(핵심94).
이는저장입력의연쇄손실을줄인결과이며정답각도보장/인식률향상아님.
전체계산가능202919각도시점중상충21965,추가확인부족12226,상충미표시168728.
같은 엄격품질기준으로 점제외까지 적용하면167816,점제외를없애면168728(차이912).

대조: 기존정상3+추가정상1+체계표본20/기존교차53+추가교차2에서좌표·모델입력변경0.
정상수행이거나좌우배정이정상이라는것이모든손가락좌표정확성의정답은아니다.
체계표본20중9장면에서45계산각도가관측상충;원래값은그대로보관했다.
교차55장면계산가능1650각도중비교근거있음878,재확인부족735,상충37.
따라서두번째검출확인을모든비교의필수조건으로강제하면정상동작도많이보류될수있다.
소복f84 기존좌우오류주석1시점은유지되어있고이번각도작업으로교정된것이아니다.

예: 응접실f44 오른손 이전계산가능3→원래15개보관/비교근거10개.
여주시청f58 왼손 이전0→원래15개보관/비교근거6개.
정밀각도불확실은남아있지만손전체정보삭제와분리됐다.

검증: 합성계약12PASS,각도상태507510개와수식별도검산PASS,최대오차2.84e-14도.
공개136자기비교는미비교NaN/고정분모계약검사만수행(정확도실험아님).
초기검산대조개수불일치는신규100에과거주석4장면추가포함때문. 기존오류1을정상대조로포함하지않게분리,
원본검산기와최초오류보존. 후보·입력·계수변경없이verify_v2로검산완료.
구현과별도수식검산은동일Codex,다른검토자나정밀손가락GT없음.

다음: {next_action}
"""
write("RESULT_KO.txt",text)
def fig(q):
    img=f"visual/{q['clip_id']}_f{q['index']}.png"
    return f"<figure><img src='{img}' alt='각도별 비교 근거'><figcaption>{q['label']} f{q['index']}: 이전 점 제외 후 계산 {q['old_point_computable']}개 → 원래 계산값 {q['raw_computable']}개 보관 / 비교 근거 {q['eligible']}개. 같은 품질 조건으로 비교하면 {q['fair_old_eligible']} → {q['eligible']}개.</figcaption></figure>"
page=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>좌표 보존 · 각도별 근거 분리</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef3f8;color:#1a3045;font:17px/1.8 'Malgun Gothic',sans-serif}}
main{{max-width:1180px;margin:auto;padding:44px 24px 70px}}section{{background:white;border:1px solid #dce5ef;border-radius:16px;padding:26px;margin:20px 0}}
h1{{font-size:36px;line-height:1.4}}h2{{font-size:25px}}.lead{{font-size:22px;font-weight:700}}.good{{border-left:7px solid #238a77}}.warn{{border-left:7px solid #d89b36}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;border-bottom:1px solid #dce5ef;text-align:left}}th{{background:#eff5fa}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{background:#eff5fa;border-radius:12px;padding:16px}}.card strong{{font-size:29px;display:block}}
img{{width:100%}}figure{{margin:20px 0}}figcaption{{padding:12px;background:#f1f6fa}}a{{color:#155caa}}.small{{font-size:14px}}
@media(max-width:720px){{main{{padding:20px 12px}}.grid{{grid-template-columns:1fr}}h1{{font-size:27px}}section{{padding:18px}}}}
</style><main><p>WordsInSilence · 공개 입력 검증 · 2026.10.21</p><h1>좌표는 남기고,<br>각도마다 비교 근거를 구분했다.</h1>
<section class="good"><p class="lead">이웃 각도까지 함께 잃던 문제는 줄였다.<br>정상 손·교차의 좌표와 모델 입력은 그대로다.</p>
<p>관절점을 지워서 여러 각도를 한꺼번에 계산하지 못하던 방식을 분리했다.
원래 좌표는 보관하고 각도마다 비교 근거가 있는지 표시한다. 분류 모델이나 운영 DTW에는 아직 적용하지 않았다.</p></section>
<section><h2>1. 어떻게 바꿨나</h2>
<p>예를 들어 검지의 두 번째 각도가 불확실해도,그 이유만으로 첫 번째·세 번째 각도의 좌표를 없애지 않는다.
단,실제로 두 각도가 같은 불확실한 마디를 사용한다면 각각 보류될 수 있다.</p>
<table><tr><th>표시</th><th>의미</th><th>처리</th></tr>
<tr><td>비교 근거 있음</td><td>필요한 좌표가 있고 두 관측에서 현재 기준의 상충이 표시되지 않음</td><td>비교 후보로 남김. 정답 보장은 아님</td></tr>
<tr><td>관측 상충</td><td>같은 손가락 각도를 검출 결과들이 충분히 일치되게 설명하지 못함</td><td>좌표·각도 보관,정밀 비교는 보류</td></tr>
<tr><td>추가 확인 부족</td><td>원래 관측은 있으나 재확인할 두 번째 근거가 부족함</td><td>잘못된 동작으로 판정하지 않음</td></tr>
<tr><td>미관측·계산 불가</td><td>필요한 점이 없거나 겹쳐 각도를 계산할 수 없음</td><td>차이0점으로 채우지 않음</td></tr></table>
<p>미비교 각도는 ‘잘 맞았다’는0점이 아니다. 참고가 요구하는 범위는 유지하면서 실제 비교한 개수도 함께 남긴다.
이는 비교 부품의 계약 검증이며 새 DTW 시간정렬 실험은 아니다.</p></section>
<section><h2>2. 공개 자료 검증 결과</h2><div class="grid">
<div class="card"><strong>136영상</strong>100단어 · 9명 · 16,917프레임</div>
<div class="card"><strong>912각도시점</strong>연쇄 손실에서 비교 후보로 회복</div>
<div class="card"><strong>변경0</strong>원래 좌표·기존 모델 입력</div></div>
<p>이전36영상과 확대100영상의 저장된 공개 학습용 자료를 모두 재사용했다. 새 사람 인식률이나100단어 밖의 수어를 검증한 결과는 아니다.</p>
<table><tr><th>범위</th><th>회복된 비교 후보</th><th>그중 핵심 구간</th></tr>
<tr><td>이전36영상</td><td>293각도시점</td><td>94</td></tr><tr><td>확대100영상</td><td>619각도시점</td><td>277</td></tr>
<tr><td>합계</td><td>912각도시점</td><td>371</td></tr></table>
<p>‘각도시점’은 한 프레임의 한 각도다.912개 영상이나912개 정답을 뜻하지 않는다.
이전 점 제외가 줄이던1,774개 각도의 계산값은 모두 남겼고,그중912개가 각도별 근거 조건도 충족했다.</p>
<p>공정한 비교를 위해 같은 품질 조건을 양쪽에 적용하면 비교 후보는167,816 →168,728개다.
나머지 좌표를 버리거나 모두 확실한 값으로 승격한 것은 아니다.</p></section>
<section><h2>3. 실제로 달라진 사례</h2>{''.join(fig(q) for q in examples)}
<p><a href="../word100_finger_preservation_20261020/RESULT_KO.html">같은 원본 장면과 이전 보류 방식의 시간 그래프 보기</a></p></section>
<section class="warn"><h2>4. 남은 문제: 정상 동작도 재확인 근거가 부족할 수 있다</h2>
<p>정상 대조24장면과 교차55장면에서 좌표·기존 입력 변화는 없었다.
그러나 <b>좌표를 보존한 것과 모든 각도를 확실하게 비교할 수 있는 것은 다르다.</b></p>
<table><tr><th>교차55장면의 계산가능1,650각도</th><th>개수</th></tr>
<tr><td>비교 근거 있음</td><td>878</td></tr><tr><td>추가 확인 부족</td><td>735</td></tr><tr><td>관측 상충</td><td>37</td></tr></table>
<p>정상 형태 대조20장면 중9장면에서도45각도에 상충 표시가 있었다. 이는45개 동작 오류가 아니다.
같은 검출기의 두 결과만으로 정밀한 손가락 정답을 확정할 수 없다는 한계다.</p>
<p>이 엄격한 확인 조건을 운영 비교의 필수 조건으로 바로 적용하면 정상 동작도 많이 보류될 수 있다.
현재는 <b>좌표 보존과 상태 구분을 구현·검증한 단계</b>이며,자동 정답/오답 판정이나 운영 필터로 채택한 단계가 아니다.</p>
<p>별도 기존 좌우 오류 주석1장면(소복f84)은 그대로 남아 있다. 이번 작업이 좌우 배정 문제까지 해결한 것은 아니다.</p></section>
<section><h2>5. 다음에 확인할 것</h2><ol>
<li>화면에서 관측하지 못한 각도와,관측은 있지만 재확인 자료가 부족한 각도를 결과 표시에서 구분한다.</li>
<li>손가락 근거가 부족해도 손목 위치·이동 궤적의 관측 근거는 독립적으로 유지한다.</li>
<li>공개 자료에서 비교한 범위와 미비교 범위를 함께 표시하는 조건을 확인한 뒤,학습 입력에 반영할지 결정한다.</li></ol>
<p>단어별 예외·임계값 완화·사용자 자료 조정은 하지 않았다. C·T3-F·운영 웹은 보존했고,새 학습·분류채점·사용자 자료 접근은 모두0이다.</p></section>
<section class="small"><h2>검증 근거</h2><p>좌표2,537,550값/모델입력4,736,760값 동일,각도상태507,510개 별도수식 검산.
합성 계약12개 통과. 구현과 복소수식 검산은 같은Codex가 수행했으며 별도 검토자·정밀관절 정답은 없다.
변하지 않은 원본에 대한 이전92프레임 관찰은 재사용했고,이번에는 상태표를 다시 검토했다.</p>
<p>검산 초기 대조 개수 불일치는 확대100에 과거주석4장면이 더 포함돼 생겼다. 그중 기존오류1을 정상대조로 세지 않도록 구분했다.
최초 오류·원래 검산기는 보존했고,후보·자료·계수를 바꾸지 않은 수정 검산을 통과했다.</p>
<p><a href="RESULT_KO.txt">텍스트 보고서</a> · <a href="SUMMARY.json">집계</a> · <a href="VERIFY.json">별도 검산</a> ·
<a href="CONTRACTS.json">계약 검사</a> · <a href="PUBLIC_COMPARISON_CONTRACT.json">미비교 처리 검사</a></p></section></main></html>"""
write("RESULT_KO.html",page)
for name in re.findall(r"(?:src|href)=[\"']([^\"'#]+)[\"']",page):assert (U/name).exists(),name
for p,h in {**f["pins"],**f["protected"]}.items():assert sha(p)==h,p
for r in read(U/"INDEX.json"):assert sha(r["path"])==r["sha256"]
recovery=read(U/"VERIFY_RECOVERY.json")
assert sha(U/"verify_v2.py")==recovery["verifier_v2_sha256"]
write("FINAL_AUDIT.json",dict(status="ANGLE_EVIDENCE_SEPARATED_VERIFIED_RESEARCH_ONLY",at=now,unit=U.name,
    scope=dict(videos=136,words=100,people=9,frames=16917,split="TRAIN diagnostics only; no generalization estimate"),
    counts=total,cohorts=s["totals"],verification=v["counts"],control_totals=v["control_totals"],
    synthetic_contracts=12,public_self_comparison=selfcheck,protected=32,
    model_changes=0,coordinate_changes=0,training_runs=0,classifier_runs=0,detector_runs=0,user_data_reads=0,
    implementation="Codex local",independent_equations="Same Codex, separate complex-angle implementation, not independent reviewer.",
    decision="Coordinate preservation and local evidence separation PASS. Strict comparison eligibility remains diagnostic, not production or training adoption.",
    next_single_action=next_action,report=str(U/"RESULT_KO.html"),
    evidence_hashes={p.name:sha(p) for p in U.iterdir() if p.is_file() and p.suffix in (".json",".py",".html",".txt")}))
print("REPORT_READY",U/"RESULT_KO.html")
