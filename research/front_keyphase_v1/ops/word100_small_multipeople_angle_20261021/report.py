import json,html
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text("utf-8-sig"))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,v):
    with Path(p).open("x",encoding="utf8") as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
s=read(U/"SUMMARY.json");f=read(U/"FROZEN.json");vm=read(U/"VISUAL_MANIFEST.json")
write(U/"VISUAL_REVIEW.json",dict(
    reviewer="Codex visual contact-sheet inspection, not independent linguistic annotation",
    pages_reviewed=[p["path"] for p in vm["pages"]],panels_reviewed=217,
    groups=dict(Counter(x["kind"] for x in vm["items"])),
    findings=[
        "수어 9명 대표 장면에서 양손 움직임이 보이며 대부분 각도 재확인이 가능. 일부 손 회전/두 번째 검출 부재에서 보류.",
        "멀다 9명 대표 장면은 오른손 동작과 내려놓은 반대손. 반대손 135항목 NOT_REQUIRED이며 오류로 처리하지 않음.",
        "후원은 대부분 앞부분 한 손 동작 후 양손 근접/접촉. 양손이 전체 주석 구간에 필요하다는 보수적 규칙이 시간별 역할을 구분하지 못함. 사람7의 진행 모양도 차이가 있어 단일 순서 정답 단정 금지.",
        "후원 사람3 교차 11장면 기존 좌표 보존. 영상에 손/펼친 손가락이 보이지만 재확인은 전부 미확인으로 남아 정상 비교 가능성에 한계.",
        "추가116 손-프레임은 실제 손이 보이는 곳의 입력. 명백한 몸통/배경 손 삽입은 표본판독에서 못 찾았으나 21개 관절/가려진 손가락의 정확성 보증 아님. 84는 주석 핵심 구간 밖.",
        "WITHHELD #208 수어 사람5 f70은 손을 내리는 장면도 주석 핵심 구간에 포함. 공개 주석 구간을 완벽한 동작 경계로 가정하면 안 됨.",
        "WITHHELD #211~213 후원 사람8/6/5: visible hand/contact + current_arm_uncertain. 기존 좌표는 남아 있고 같은 손의 재확인 연결이 막힘. 관측 오류와 실제 수행 오류 구분 필요.",
        "WITHHELD #209 및 #214~216: 정상 동작 형태에서도 새 검출이 필요한 손을 못 찾거나 연결 못함. 손가락 근거 보류를 사용자 수행 실패로 부르면 안 됨."
    ],
    limitations="Raw frames and coordinate-based crops inspected; no dense human landmark ground truth or full linguistic certification. Normal motion preservation != validated motion correctness."))

plt.rcParams["font.family"]="Malgun Gothic"
plt.rcParams["axes.unicode_minus"]=False
fig,ax=plt.subplots(figsize=(11,4.7),layout="constrained")
labels=[x["label"] for x in s["words"]]
parts=[("status_OBSERVED","모든 핵심 시점 근거 확보","#1a8c83"),
       ("status_PARTIAL","일부 구간 차이 표시","#6bb6b1"),
       ("status_INSUFFICIENT","요약 비교 보류","#e3a656")]
den=np.array([270,270,135]);left=np.zeros(3)
for key,label,color in parts:
    vals=np.array([x["counts"].get(key,0) for x in s["words"]])
    ax.barh(labels,100*vals/den,left=left,label=label,color=color,height=.55)
    for i,v in enumerate(vals):
        if v:ax.text(left[i]+100*v/den[i]/2,i,str(v),va="center",ha="center",fontsize=12)
    left+=100*vals/den
ax.set_xlim(0,100);ax.invert_yaxis()
ax.set_xlabel("비교 대상 각도 항목 중 비율 (%) — 정답률 아님")
ax.set_title("같은 절차로 9명 자료를 준비해도, 동작에 따라 비교 가능성이 다름",loc="left",pad=20)
ax.legend(loc="upper center",bbox_to_anchor=(.5,-.22),ncol=3,frameon=False)
ax.spines[["top","right","left"]].set_visible(False)
fig.savefig(U/"AVAILABILITY.png",dpi=150);plt.close(fig)

clips=[]
for row in read(U/"PREPARED_INDEX.json"):
    cid=row["clip_id"];q=load(row["prepared"]);a=read(U/"queries"/(cid+".json"))
    votes=load(U/"queries"/(cid+".npz"))["votes"]
    clips.append(dict(id=cid,label=row["label"],person=row["signer_id"],video=Path(row["video"]).as_uri(),
        required=a["required"],times=(q["timestamp_us"]/1e6).tolist(),frames=q["frame_id"].tolist(),
        core=q["core"].tolist(),state=q["angle_state"].tolist(),votes=votes.tolist(),rows=a["rows"]))
data=json.dumps(clips,ensure_ascii=False,allow_nan=False)
(U/"report_data.js").write_text("const clips = "+data+";\n",encoding="utf8")
table="".join(f'<tr><td>{w["label"]}</td><td>9명 · 9영상</td><td>{270 if w["word_code"]!="WORD0949" else 135}</td><td>{w["counts"]["strict_summary"]}</td><td>{w["counts"].get("status_OBSERVED",0)}</td><td>{w["counts"].get("status_PARTIAL",0)}</td><td>{w["counts"].get("status_INSUFFICIENT",0)}</td></tr>' for w in s["words"])
galleries="".join(f'<details><summary>{p["kind"]} 장면 모음 {Path(p["path"]).stem}</summary><img loading="lazy" src="visual/{Path(p["path"]).name}" alt="공개 영상 원본과 손 확대 장면"></details>' for p in vm["pages"])
page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>여러 사람의 재확인 자료 — 작은 공개 묶음 검증</title>
<style>
:root{--ink:#203341;--muted:#5c7081;--teal:#187e79;--line:#dbe4e9}*{box-sizing:border-box}
body{margin:0;background:#eff3f5;color:var(--ink);font:16px/1.7 'Malgun Gothic',sans-serif}
main{max-width:1200px;margin:auto;padding:44px 30px}h1{font-size:34px;line-height:1.35;margin:12px 0}
h2{font-size:24px;margin-top:0}h3{font-size:19px}p{margin:12px 0}.eyebrow{color:var(--teal);font-size:13px;font-weight:bold;letter-spacing:2px}
section{background:white;padding:30px;border-radius:18px;margin:24px 0;box-shadow:0 4px 20px #263c4b08}
.lead{font-size:21px;max-width:950px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:25px 0}
.card{background:white;border:1px solid var(--line);padding:20px;border-radius:14px}.card b{font-size:32px;display:block;color:var(--teal)}
.warn{background:#fff5df;border-left:5px solid #d59b35;padding:16px 20px}.good{background:#eaf7f4;padding:16px 20px}
table{border-collapse:collapse;width:100%;font-size:15px}th,td{padding:13px 10px;border-bottom:1px solid var(--line);text-align:left}
th{background:#edf5f5}.scroll{overflow:auto}img{max-width:100%;height:auto}select{font:inherit;padding:9px;border:1px solid var(--line);border-radius:8px}
.detail{display:grid;grid-template-columns:1fr 1.15fr;gap:22px}video{width:100%;background:#15212a;border-radius:10px}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:13px}.legend span:before{content:"";display:inline-block;width:13px;height:13px;background:var(--c);margin-right:5px}
canvas{width:100%;height:auto;border:1px solid var(--line)}details{border-top:1px solid var(--line);padding:12px 0}summary{cursor:pointer;font-weight:600}
.small{font-size:14px;color:var(--muted)}a{color:var(--teal)}li{margin:8px 0}
@media(max-width:800px){main{padding:20px 12px}section{padding:20px}.cards{grid-template-columns:1fr 1fr}.detail{grid-template-columns:1fr}h1{font-size:27px}}
</style><main>
<div class="eyebrow">WORDS IN SILENCE · 연구용 공개 자료 검증 · 2026.10.21</div>
<h1>여러 사람의 자료를 모두 재확인하면<br>정상 동작을 충분히 비교할 수 있을까?</h1>
<p class="lead"><b>자료 준비와 보류 처리 검증은 완료.</b><br>비교 가능한 부분은 늘었지만, 손이 겹치는 정상 동작까지 넓게 보류되는 문제가 남았다.</p>
<div class="cards"><div class="card"><b>27</b>3개 단어 × 9명 영상</div><div class="card"><b>216</b>다른 사람끼리의 비교</div><div class="card"><b>353 / 675</b>일부라도 차이를 표시한 각도</div><div class="card"><b>322 / 675</b>요약 비교가 보류된 각도</div></div>
<section><h2>무엇을 준비했나</h2>
<p>‘수어’(양손), ‘후원’(손 교차·근접), ‘멀다’(한 손)의 공개 TRAIN 촬영자 1~9번 자료를 모두 준비했다. 기존 5개는 해시를 확인해 재사용하고, 나머지 22개에 같은 재확인 절차를 적용했다. 새 영상 다운로드는 없었다.</p>
<p class="good"><b>원본 전체 화면의 기존 관절 → 같은 원본 장면에서 손 재검출·필요한 경우 확대 확인 → 기존 손을 보존하며 합침 → 고정 상반신 범위 적용 → 각도별 근거 → 사람별 DTW 비교</b></p>
<p>검출·합치기 규칙, 화면 범위, 손 크기 기준, 최소 5명·8프레임·50% 비교 조건은 바꾸지 않았다. 각 영상은 자기 자신과 같은 사람을 제외한 8명의 같은 단어 영상과 비교했다. 추가된 손 관측은 116 손-프레임이며 그중 32는 주석상 핵심 구간 안이다. 모두 정확한 손가락 좌표를 복구했다는 뜻은 아니다.</p>
</section>
<section><h2>정상 자료에서 얼마나 비교할 수 있었나</h2>
<img src="AVAILABILITY.png" alt="단어별 비교 가능한 각도 항목 비율">
<div class="scroll"><table><thead><tr><th>단어</th><th>자료</th><th>대상 각도</th><th>차이 표시</th><th>모든 핵심 시점 근거</th><th>일부 구간만 비교</th><th>요약 보류</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<p><b>‘각도 항목’은 영상 1개에서 특정 손가락의 특정 굽힘을 비교한 하나의 결과다.</b> 양손은 30항목, 한 손은 15항목이다. ‘멀다’의 내려놓은 반대손 135항목은 비교 대상에서 제외했고 수행 오류로 처리하지 않았다.</p>
<p>재확인 정보가 없던 상태를 같은 최종 좌표·같은 시간 정렬에서 재현하면 675항목 모두 요약 근거가 부족했다. 정보를 완성한 뒤 353항목은 차이를 표시할 수 있게 됐다. 이 중 294항목에는 비교 못한 시간대가 있고, 59항목만 모든 핵심 시점에 최소 5명의 근거가 있다.</p>
<p class="warn"><b>이 수치는 인식률·수어 정답률이 아니다.</b> 353항목 모두 ‘잘한 동작’이라는 뜻도 아니다. 현재 확인된 구간의 차이를 설명할 수 있다는 뜻이다.</p>
</section>
<section><h2>왜 ‘후원’은 여전히 비교가 어려운가</h2>
<ol><li><b>시간에 따라 사용하는 손이 달라진다.</b> 대표 원본 장면에는 앞부분에 한 손을 쓰고 뒤에서 양손이 가까워지는 과정이 있다. 현재 보수적 비교 규칙은 이 단어의 전체 핵심 구간에서 양손 모두를 요구한다. 정상적으로 내려놓은 시간까지 근거 부족에 포함될 수 있다.</li>
<li><b>영상에 보이는 손도 재확인 연결이 막힌다.</b> 양손이 가까울 때 현재 팔 연결만으로 같은 손인지 확정하지 못하면 기존 좌표는 남겨두되 재확인에는 사용하지 않는다. 사람 8·6·5의 예시에서 실제로 이 이유가 확인됐다. 손가락이 보인다고 좌표가 정확하다고 보장할 수도 없지만, 모두 비교 불가로 두면 활용성이 낮다.</li>
<li><b>주석 경계와 시간 정렬에도 한계가 있다.</b> ‘수어’ 사람5의 일부 주석 구간에는 이미 손을 내리는 장면이 들어 있다. 전체 원시 특징 기반 정렬은 216쌍 중 28쌍에서 기존 근거 하한을 넘지 못했다. 이번에 그 규칙을 완화하거나 정답처럼 고치지 않았다.</li></ol>
<p class="warn">따라서 “실제 누락·불확실한 부분만 보류하고 정상 동작은 충분히 비교한다”는 목표는 <b>아직 완전히 달성하지 못했다.</b> 자료 미준비와 현재 재확인 방식의 한계를 구분해 확인한 단계다.</p>
</section>
<section><h2>한 영상의 손·각도·시간대를 직접 보기</h2>
<p class="small">왼손·오른손은 수행자의 몸 기준. 원본 영상과 저장 관측 시점을 보여준다. 색은 수행 점수가 아니라 비교 근거 상태다.</p>
<label>공개 영상 <select id="clip"></select></label>
<div class="detail" style="margin-top:20px"><div><video id="video" controls preload="metadata"></video><p id="description"></p></div>
<div><h3>재확인 상태</h3><div class="legend"><span style="--c:#c0c7cd">누락</span><span style="--c:#dbb55a">재확인 없음</span><span style="--c:#d98983">관측 상충</span><span style="--c:#238f9b">비교 근거 있음</span></div><canvas id="state" width="900" height="530"></canvas>
<h3>다른 사람 5명 이상과 비교 가능한 시간</h3><canvas id="votes" width="900" height="530"></canvas>
<p class="small">파란색: 해당 시점에 참고 5명 이상. 회색: 부족. 흐린 배경: 공개 주석 핵심 구간 밖 또는 비교하지 않는 손. 중간의 빈 곳을 성공으로 채우지 않는다.</p></div></div>
<div class="scroll"><table id="angles"></table></div>
</section>
<section><h2>안전하게 보류되는지 확인한 결과</h2><ul>
<li>기존에 사용하던 관절 107,218점은 좌표와 좌우 칸 그대로 보존했다. 정상·교차 기존 표본 13장면도 좌표와 각도 상태가 유지됐다.</li>
<li>한 손 전체 각도 정보를 지운 54회, 한 각도 한 시점만 지운 27회, 한 각도만 불확실하게 만든 27회에서 누락을 0도 차이·성공으로 처리하지 않았다. 다른 손과 다른 각도는 유지됐다.</li>
<li>참고 영상 한 손의 정보가 사라져도 비교해야 할 범위는 줄이지 않았다(27회). 정상 한 손 동작의 반대손은 계속 비교 대상이 아니었다.</li>
<li>시간 표시를 2배로 늘린 27회와 핵심 구간 밖의 근거를 지운 27회도 확인했다. 이 시험들은 <b>저장된 시간 정렬을 고정한 정보 제거 시험</b>이다. 새로운 촬영·픽셀 가림·실시간 검출 실험은 아니다.</li></ul>
<p>원본 81개 대표 장면, 새 손 관측 116개, 기존 교차 11개, 보류 대표 9개를 이미지로 확인했다. 같은 Codex가 별도 수식으로 수치도 검산했다. 독립된 수어 전문가 판정이나 모든 관절의 정답 좌표 검증은 아니다.</p>
</section>
<section><h2>다음에 할 일</h2><ol>
<li><b>기존 손의 재확인과 새 손 삽입 조건을 분리해 검토.</b> 현재 팔 연결이 불확실하다는 이유만으로 기존 손의 모든 각도 확인이 막히는 사례를 먼저 다룬다. 좌우가 실제로 불분명한 손을 억지로 확정하지 않는다.</li>
<li><b>여러 참고 영상에서 손별 사용 시간대를 검증.</b> 정상적으로 한 손을 내려놓은 구간과 반드시 보여야 할 구간을 구분하되, 비교하는 사람이 손을 숨겼다고 기준 범위가 줄어들게 만들지 않는다.</li>
<li><b>같은 27개 공개 묶음과 기존 오류·교차 대조군으로 다시 확인.</b> 정상 비교 가능성과 누락 감지 보존을 동시에 확인한 뒤에 넓은 자료·웹 적용·학습을 판단한다.</li></ol>
<p class="good">기존 C·T3-F·운영 웹·운영 DTW는 변경하지 않았다. 사용자·멘토 촬영 자료 접근, 학습, 튜닝, 후보 선택은 없었다. 이번 결과를 같이 확인한 뒤 다음 단계로 진행한다.</p></section>
<section><h2>원본 장면 검토 자료</h2><p class="small">전체 원본 화면과 손 부분 확대를 함께 배치했다. 확대는 검토 화면이며 입력 화면 범위를 바꾼 것이 아니다.</p>__GALLERIES__</section>
<p class="small">근거: <a href="SUMMARY.json">집계</a> · <a href="INPUT_VERIFY.json">입력 검산</a> · <a href="VERIFY.json">DTW 수치 검산</a> · <a href="STRESS.json">누락 통제</a> · <a href="UNCERTAINTY_CONTROLS.json">각도 불확실성 통제</a> · <a href="DIAGNOSIS.json">보류 이유</a></p>
</main><script src="report_data.js"></script><script src="report_ui.js"></script></html>'''
(U/"RESULT_KO.html").write_text(page.replace("__TABLE__",table).replace("__GALLERIES__",galleries),encoding="utf8")
(U/"RESULT_KO.txt").write_text("""공개 3단어×9명=27영상 재확인 자료 준비 및 검증 완료.
대상: 수어·후원·멀다. 기존5 재사용/신규22. 저장3299프레임, 다른 사람끼리216쌍.
전체화면 검출·같은 이미지 재확인·합친 후 고정ROI. 계수/역할/최소5명 기준 변경 없음.
수어238/270, 후원7/270, 멀다108/135 각도 항목에서 비교 차이 표시. 정답률 아님.
전체675중353 표시(59모든시점/294부분),322보류. 멀다 비사용손135 비교대상아님.
새 손116 손-프레임 중32 주석핵심. 기존107218 관절점 보존. 숨겨진 손가락 정확성은 보장 불가.
자료만 완성하면 모두 해결된다는 가설은 지지되지 않음. 후원의 한 손 단계에도 양손 요구,
겹침 중 current_arm_uncertain으로 기존 손 재확인 전체 보류, 주석/정렬 한계가 남음.
입력·수치 검산 PASS, 정보 누락·각도 불확실성의 국소 보류 PASS. 정상 비교 충분성은 미달.
다음: 기존 손 재확인 vs 새 손 삽입 조건 분리, 여러 참고로 손별 시간대 검증, 같은 공개 묶음+오류대조군 재검증.
C/T3-F/운영웹 유지. 사용자자료 접근/학습/튜닝/분류평가0. 추가학습은 공동 결과검토 후.
최초 freeze 경로 연결 오류는 추출 전 중단; FIRST_ERROR.txt 보존, 과거 records 경로로 수정 후 최초 동결 및 정상 실행.
""",encoding="utf8")
print("REPORT",U/"RESULT_KO.html")
