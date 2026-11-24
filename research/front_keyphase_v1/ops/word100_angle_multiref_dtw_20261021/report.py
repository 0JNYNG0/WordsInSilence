import json,html
from collections import Counter
import numpy as np
from run import U, A, read, load, write

def main():
    assert read(U/"VERIFY.json")["status"]=="PASS" and read(U/"STRESS.json")["status"]=="PASS"
    assert read(U/"PATH_AUDIT.json")["status"]=="PASS"
    data=[];counts=Counter()
    for ix in read(U/"INDEX.json"):
        cid=ix["clip_id"];r=read(U/"queries"/(cid+".json"))
        q=load(U/"prepared"/(cid+".npz"));v=load(U/"queries"/(cid+".npz"))
        counts["raw_values"]+=sum(x["raw_only_median_deg"] is not None for x in r["rows"])
        counts["strict_values"]+=sum(x["compared_median_deg"] is not None for x in r["rows"])
        counts["raw_but_insufficient"]+=sum(x["raw_only_median_deg"] is not None and x["compared_median_deg"] is None for x in r["rows"])
        data.append(dict(**ix,required=r["required"],role_reviewed=r["role_reviewed"],rows=r["rows"],
                         ts=((q["timestamp_us"]-q["timestamp_us"][0])/1e6).tolist(),
                         frame_id=q["frame_id"].tolist(),state=q["angle_state"].tolist(),
                         core=q["core"].tolist(),votes=v["votes"].tolist(),
                         expected_votes=v["expected_votes"].tolist()))
    c=read(U/"VERIFY.json")["counts"];controls=read(U/"VERIFY.json")["controls"]
    summary=dict(**c,**dict(counts),controls=controls,stress=read(U/"STRESS.json")["counts"],
                 comparison_meaning="one item = one clip × one hand × one angle, not one word or one video",
                 all_signs_correctness_verified=False,training=0,classifier_inference=0,
                 user_data_access=0,operational_change=False)
    write(U/"SUMMARY.json",summary)
    (U/"report_data.js").write_text("const DATA="+json.dumps(data,ensure_ascii=False,separators=(",",":"))+";",encoding="utf8")
    prose=f"""각도별 근거 × 다중 참고 DTW — 연구용 연결·공개 검증

결론: 어느 손·각도·시각을 어떤 참고 사람과 비교했는지 추적할 수 있다.
비교하지 못한 각도를 0으로 채우지 않고, 낮은 차이도 일부 구간만 확인되면 '일부만 비교됨'으로 표시한다.
정확도 개선이나 수어 정답 판정은 이번 작업에서 측정하지 않았다.

자료: 공개 TRAIN899영상/100단어/9명 참고 풀. 그중 각도 근거가 준비된136영상을 질의로 사용해 본인 촬영자 제외7~8명과1087쌍 비교.
기존136영상은 전체 화면 검출·재탐색 통합 후 상반신 제한 관측; 나머지763은 저장된 전체 화면 검출 후 상반신 제한 관측이지만 재확인 정보 없음.
두 경로를 같은 자료처럼 숨기지 않았고, 두 경우 모두 동일 스냅샷에서 원시 각도 차이와 근거를 구분한 차이를 함께 계산했다.
같은 좌표 조건에서 보고 방식만 비교한 것으로 기존 C 대 새 모델 비교가 아니다.

방법: 저장된 모든 프레임 유지 → 기존 비용식·DTW 시간 정렬 → 각도별 관측/미관측/재확인 부족/상충 구분 → 다른 참고 사람별 차이 → 중앙값.
분모는 공개 주석의 동작 구간으로 고정. 질의나 참고 손이 사라져도 해당 시간을 지우지 않는다.
새로 동작을 잘라낸 것이 아니며 정렬에는 처음~끝의 저장 프레임을 사용한다. 수치 근거 평가 범위만 공개 주석으로 고정한 오프라인 검증이다.
각 참고에서 8개 이상 서로 다른 프레임과 50% 이상 실제 참고 시간, 정렬 근거0.35 조건을 유지하고 최소5명의 참고가 필요하다.
5명 조건을 만족해도 중간 빈 시간대가 있으면 '일부만 비교됨'; 수어가 맞았다는 판정은 아예 제공하지 않는다.
재확인은 같은 검출기 계열의 다른 관측 근거 비교일 뿐 전문가 정답이나 독립 센서의 정답 확인이 아니다.

결과: 136영상×양손×15각도=4080개 항목.
- 근거 있는 참고5명 이상으로 일부 구간의 차이를 표시:200개.
- 비교 근거 부족:3820개.
- 기존 참고 역할상 비교 대상이 아닌 손의 각도:60개.
- 모든 핵심 시간에5명 근거가 이어진 항목:0개.
- 단순 관측값만으로 숫자를 내던2509개 중2309개는 재확인 근거가 충분하지 않았다.
이것은2309번 오인식을 고쳤다는 뜻도, 사용자가 틀렸다는 뜻도 아니다. 근거 부족을 숫자 뒤에 숨기지 않게 된 것이다.
재확인 자료가 사람별5개 이상 모인 단어는 현재 가래떡·대지2개뿐이므로200항목도 이12영상에서 나왔다.
100단어 전체의 정밀 다중참고 비교가 준비됐다는 결론은 낼 수 없다.

검증: 899영상의 좌표16977150값 보존. 이전 정상24장면·교차55장면의 좌표/각도/근거 상태 그대로 유지.
기존 소복84프레임의 좌우 오류 주석1건도 그대로 남아 있으며 이번 연결로 해결하지 않았다.
전체1087쌍의 경로·차이·고유 참고시간 분모·중앙값·시간대 독립 수식 검산.12쌍 최적경로 비용 별도 검산.
공개 자료에서 손 전체 각도 근거 제거272회, 한 각도 한 프레임 누락136회, 시간 간격2배 변환136회, 비핵심 구간 누락136회 통과.
이는 같은 저장 DTW 경로 위의 통제 시험이며 픽셀 가림·검출기 재실행·실제 반복 수행의 정답률 검증은 아니다.
한 각도 누락은 해당 각도의 시간대에만 표시됐고 이웃29각도 비교 결과는 유지됐다.
합성11계약은 동일 동작, 단일관측, 참고수 부족, 중복사람, 사용하지 않는 손, 느린 수행·중복프레임 등 포함.
구현과 별도 수식 검산은 모두 같은 Codex 담당이며 제3자 수어 전문가 검토가 아니다.

남은 문제:
1.763개 참고의 정밀 재확인 근거가 없어 엄격 비교는 대부분 보류된다. 기준을 낮춰 숫자를 만드는 해결은 하지 않았다.
2.정상 손 모양/교차도 재확인 부족·관측 상충으로 보류될 수 있다. 좌표는 보존되지만 모든 정상 비교 수치가 유지되는 것은 아니다.
3.시간 정렬에는 기존 원시 특징이 쓰이므로 잘못된 관절이 정렬에 미치는 영향은 별도로 남는다. 반복/과도압축의 기존 연구를 대체하거나 해결한 작업이 아니다.
4.동작 범위를 고정할 때 공개 주석을 썼다. 사용자 영상에는 이 정답 주석이 없으므로 실제 웹 적용 전 다중 참고의 구간 추정과 연결해 검증해야 한다.
5.현재2D각도는 실제3D손가락각도나 올바른 수어의 점수가 아니다. 원래 가려진 손 모양을 복원한 것이 아니다.

다음: 결과를 사용자와 검토한 뒤, 재확인 근거가 없는 나머지 공개 참고의 준비 범위/비용을 정하고 기존 구간 추정·반복/압축 보호와 함께 검증.
그전까지 연구 결과 표시 후보로 유지한다. 학습0·분류추론0·사용자자료접근0·C/T3-F/운영DTW/웹 변경0.
"""
    (U/"RESULT_KO.txt").write_text(prose,encoding="utf8")
    page=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>각도별 근거와 다중 참고 DTW</title><style>
:root{color-scheme:light;--ink:#183046;--muted:#637487;--green:#168377;--amber:#b66b12;--line:#dde5ec}
*{box-sizing:border-box}body{margin:0;background:#f3f6f9;color:var(--ink);font-family:"Malgun Gothic",sans-serif;line-height:1.65}
main{max-width:1300px;margin:auto;padding:38px 28px}h1{font-size:32px;line-height:1.35;margin:8px 0 14px}h2{font-size:22px;margin:0 0 14px}h3{font-size:17px}
.tag{color:#287c76;font-size:13px;font-weight:bold}.lead{max-width:950px;font-size:18px}.panel{background:white;border:1px solid var(--line);border-radius:16px;padding:25px;margin:22px 0}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:18px}.num{font-size:32px;font-weight:bold}.small{font-size:13px;color:var(--muted)}.warn{border-left:5px solid #cc8b36;background:#fff8eb;padding:14px 18px}.info{border-left:5px solid #3c8e93;background:#edf7f6;padding:14px 18px}
select{max-width:100%;padding:10px;border:1px solid #b9c8d4;border-radius:7px;font-size:15px;margin:5px 12px 10px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:10px 8px;text-align:left}th{color:var(--muted);background:#f5f8fa}tr.sel{background:#e8f5f3}tbody tr{cursor:pointer}tbody tr:hover{background:#eef4f7}.badge{font-size:12px;border-radius:5px;background:#eef2f5;padding:4px 6px;white-space:nowrap}
.metric{font-size:27px;font-weight:bold}.timeline{width:100%;height:auto;display:block}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.sq{display:inline-block;width:11px;height:11px;margin-right:5px}.details{max-height:340px;overflow:auto}.muted{color:var(--muted)}pre{white-space:pre-wrap;font:14px/1.75 "Malgun Gothic",sans-serif}a{color:#177887}
@media(max-width:820px){.cards{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}main{padding:20px 12px}h1{font-size:27px}}
</style></head><body><main>
<div class="tag">2026.10.21 · 공개 자료 검증 · 학습/운영 적용 없음</div>
<h1>“차이가 작다”와 “충분히 봤다”를<br>나누어 보여줍니다.</h1>
<p class="lead">여러 사람의 참고 동작과 비교하되, <b>어느 손의 어느 각도를 언제 비교했는지</b> 남겼습니다.
보이지 않거나 관측이 엇갈린 부분은 0점으로 채우지 않고 <b>비교 근거 부족</b>으로 구분합니다.</p>
<div class="cards"><div class="card"><div class="num">136</div>검증 영상 / 100단어<div class="small">참고 풀 899영상 · 9명</div></div>
<div class="card"><div class="num">1,087</div>다른 사람과의 비교쌍<div class="small">같은 촬영자 제외</div></div>
<div class="card"><div class="num">200</div>일부 차이 표시 항목<div class="small">5명 근거 확보 · 빈 구간은 남음</div></div>
<div class="card"><div class="num">3,820</div>비교 근거 부족 항목<div class="small">오류 동작이라는 뜻 아님</div></div></div>
<p class="small">항목 = 영상 1개 × 한 손 × 한 각도. 총 4,080항목 중 나머지 60개는 기존 참고 역할상 비교 대상이 아닌 손입니다.</p>
<div class="warn"><b>정확도가 높아졌다는 결과가 아닙니다.</b> 단순 계산으로 숫자가 나오는 2,509항목 중 2,309항목은 재확인 근거가 부족했습니다.
재확인 정보는 136영상에만 있으므로, 100단어 전체의 정밀 비교가 준비된 상태는 아닙니다.</div>
<section class="panel"><h2>손·각도·시간대별로 확인</h2>
<label>공개 비교 영상 <select id="clip"></select></label><label>손 <select id="hand"><option value="0">왼손</option><option value="1">오른손</option></select></label>
<p id="source" class="small"></p><div class="grid"><div><table><thead><tr><th>각도</th><th>근거 있는 차이</th><th>참고 수</th><th>상태</th></tr></thead><tbody id="angles"></tbody></table></div>
<div><h3 id="title"></h3><div id="headline"></div><p id="context"></p>
<div id="timeline"></div><div class="legend">
<span><i class="sq" style="background:#168377"></i>5명 이상과 비교</span><span><i class="sq" style="background:#ddad55"></i>1~4명과 비교</span><span><i class="sq" style="background:#a9b4c0"></i>참고 근거 없음</span>
</div><p class="small">전체 원본 시간축입니다. 파란 선은 공개 주석의 동작 구간이며, 연한 구간은 이번 근거 평가 범위 밖입니다. 이 범위는 손이 사라졌다고 줄어들지 않습니다.</p>
<h3>비교하지 못한 이유와 시간</h3><div id="gaps" class="details"></div>
<div class="info small">보이지 않음, 재확인 자료 없음, 두 관측이 엇갈림은 서로 다릅니다. 모두 사용자 동작이 틀렸다는 뜻은 아닙니다.</div>
</div></div></section>
<section class="panel"><h2>이번에 확인한 것</h2><div class="grid"><div>
<h3>이웃 각도는 그대로</h3><p>한 각도 한 프레임만 누락시킨 공개 자료 136회에서, 그 시간대를 표시하고 이웃 29각도의 결과는 유지했습니다.</p>
<h3>손이 통째로 비교 불가하면 빈 값</h3><p>손 전체 각도 근거를 제거한 272회에서 해당 손의 정밀 차이는 보류하고, 반대 손 결과는 유지했습니다.</p>
</div><div><h3>좌표·정상 교차 유지</h3><p>참고 풀의 좌표 16,977,150값을 바꾸지 않았습니다. 이전 정상 24장면·교차 55장면의 좌표와 각도·근거 상태도 유지했습니다.</p>
<h3>수치와 근거를 따로 검산</h3><p>1,087쌍의 비교 마스크·시간 분모·중앙값·누락 시간대를 별도 수식으로 확인했습니다. 기존 좌우 오류 주석 1건은 이번 작업으로 해결하지 않았습니다.</p></div></div>
<p class="small">통제 시험은 같은 저장 DTW 경로 위에서 각도 근거를 제거한 시험입니다. 실제 영상 가림·재검출·분류 정확도 시험과는 다릅니다. 구현과 검산 담당은 동일 Codex입니다.</p></section>
<section class="panel"><h2>남은 문제와 다음 단계</h2>
<ol><li><b>참고 쪽 근거 부족:</b> 나머지 763영상에는 각도 재확인 정보가 없습니다. 부족한 근거를 임의로 확실하다고 바꾸지 않았습니다.</li>
<li><b>정상 동작도 보류될 수 있음:</b> 좌표는 유지했지만 엄격한 비교 조건 때문에 정상 손 모양·교차의 수치도 보류될 수 있습니다.</li>
<li><b>시간 정렬과 실제 웹 적용:</b> 기존 원시 특징의 정렬 오류는 별도 문제입니다. 공개 주석으로 고정한 동작 구간을 실제 웹에서도 안다고 가정할 수 없습니다.</li>
<li><b>다음:</b> 결과를 함께 검토하고, 공개 참고의 근거를 확장할 범위와 기존 구간 추정·반복/압축 보호의 연결 검증을 먼저 정합니다. 그 뒤 학습 여부를 판단합니다.</li></ol>
<div class="info">C·T3-F, 운영 DTW와 웹을 보존했습니다. 사용자 촬영 자료 접근·학습·튜닝은 하지 않았습니다.</div>
</section><details class="panel"><summary>자료 조건·검증 범위·한계 전문</summary><pre>__PROSE__</pre></details>
<p class="small"><a href="SUMMARY.json">집계 근거</a> · <a href="VERIFY.json">독립 수식 검산</a> · <a href="STRESS.json">공개 자료 통제 시험</a> · <a href="FROZEN.json">고정한 비교 조건</a></p>
</main><script src="report_data.js"></script><script>
const $=id=>document.getElementById(id);let angle=0;
const states={INSUFFICIENT:"근거 부족",PARTIAL:"일부만 비교",OBSERVED:"관측 구간 비교",NOT_REQUIRED:"비교 대상 아님"};
const reasons={query_missing:"해당 각도 관절이 보이지 않음",query_degenerate:"각도 계산 기하 부족",query_unassessed:"재확인 관측 없음",query_conflict:"관측끼리 엇갈림",reference_or_alignment_insufficient:"참고 또는 시간 정렬의 근거 부족"};
DATA.forEach((d,i)=>{$("clip").add(new Option(`${d.label} · 촬영자 ${d.signer_id} · ${d.word_code}`,i))});
$("clip").value=Math.max(DATA.findIndex(d=>d.rows.some(r=>r.compared_median_deg!==null)),0);
const fmt=x=>x===null?"—":x.toFixed(1)+"°";
function render(){
 const d=DATA[+$("clip").value],h=+$("hand").value,rows=d.rows.slice(h*15,h*15+15),r=rows[angle];
 $("source").textContent=`${d.clip_id} · 같은 단어의 다른 촬영자 ${r.reference_people.join(", ")} 참고 · 원본 저장 프레임 전체 정렬. ${d.role_reviewed?"기존 손 역할 정보 사용":"필수 손 역할 미확정: 양손을 보수적으로 따로 기술하며 정답 판정하지 않음"}`;
 $("angles").innerHTML=rows.map((x,j)=>`<tr data-angle="${j}" class="${j===angle?"sel":""}"><td>${x.name}</td><td>${fmt(x.compared_median_deg)}</td><td>${x.strict_reference_count}명</td><td><span class="badge">${states[x.status]}</span></td></tr>`).join("");
 $("angles").querySelectorAll("tr").forEach(tr=>tr.onclick=()=>{angle=+tr.dataset.angle;render()});
 $("title").textContent=(h===0?"왼손":"오른손")+" · "+r.name;
 $("headline").innerHTML=`<div class="metric">${fmt(r.compared_median_deg)} <span class="badge">${states[r.status]}</span></div>`;
 $("context").textContent=`실제 비교를 뒷받침한 참고: ${r.actual_compared_reference_people.length?r.actual_compared_reference_people.join(", ")+"번 촬영자":"없음"}. 참고를 엄격히 구분하기 전 단순 관측값 중앙값은 ${fmt(r.raw_only_median_deg)}. 낮은 차이는 비교한 부분에 한정되며 수어 정답률이 아닙니다.`;
 const w=540,x0=15,dur=d.ts.at(-1)||1,scale=(w-30)/dur;
 let svg=`<svg class="timeline" viewBox="0 0 540 125" role="img" aria-label="각도별 비교 근거 시간대"><text x="15" y="15" font-size="12">다른 참고와 비교한 근거</text>`;
 d.ts.forEach((t,i)=>{const left=i?(d.ts[i-1]+t)/2:0,right=i<d.ts.length-1?(d.ts[i+1]+t)/2:dur,v=d.votes[i][h][angle];
  const color=!d.core[i]?"#edf1f5":v>=5?"#168377":v?"#ddad55":"#a9b4c0";
  svg+=`<rect x="${x0+left*scale}" y="23" width="${Math.max(.5,(right-left)*scale)}" height="24" fill="${color}"><title>${left.toFixed(2)}~${right.toFixed(2)}초 · ${v}명 · 프레임${d.frame_id[i]}</title></rect>`;
  if(d.core[i])svg+=`<rect x="${x0+left*scale}" y="51" width="${Math.max(.5,(right-left)*scale)}" height="3" fill="#3b78b7"/>`;
  const s=d.state[i][h][angle],sc=["#d9dfe5","#663f7b","#8aa8c5","#c77856","#168377"][s];
  svg+=`<rect x="${x0+left*scale}" y="80" width="${Math.max(.5,(right-left)*scale)}" height="16" fill="${sc}"><title>관측 상태${s} · ${t.toFixed(2)}초</title></rect>`;
 });
 svg+=`<text x="15" y="74" font-size="12">질의 자체: 회색 미관측 / 파랑 재확인 없음 / 주황 상충 / 초록 상충 미표시</text><text x="15" y="116" font-size="12">0초</text><text x="490" y="116" font-size="12">${dur.toFixed(2)}초</text></svg>`;
 $("timeline").innerHTML=svg;
 $("gaps").innerHTML=Object.entries(r.reasons).filter(([k,v])=>v.length).map(([k,v])=>`<p><b>${reasons[k]}</b><br>${v.map(t=>`${t.start_s.toFixed(2)}–${t.end_s.toFixed(2)}초 (${t.frames}프레임)`).join(" · ")}</p>`).join("")||"<p>이 공개 주석 구간에서 추가 누락 표시가 없습니다. 올바른 동작이라는 판정은 아닙니다.</p>";
}
$("clip").onchange=()=>{angle=0;render()};$("hand").onchange=render;render();
</script></body></html>'''
    (U/"RESULT_KO.html").write_text(page.replace("__PROSE__",html.escape(prose)),encoding="utf8")
    print("REPORT",dict(counts))
if __name__=="__main__":main()
