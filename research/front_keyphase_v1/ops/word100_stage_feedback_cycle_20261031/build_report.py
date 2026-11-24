from learning import *
from feedback import load_cases,CONDITIONS
import html
KO={'R_index_only':'오른손 검지만 펴짐','R_open_four':'오른손 네 손가락 펴짐','L_open_four':'왼손 네 손가락 펴짐','R_face_overlap':'오른손·얼굴의 화면상 겹침','hands_overlap':'양손의 화면상 겹침','R_motion_up':'오른손 위로 이동','R_motion_down':'오른손 아래로 이동','R_motion_left':'오른손 화면 왼쪽으로 이동','R_motion_right':'오른손 화면 오른쪽으로 이동','L_motion_down':'왼손 아래로 이동'}
NAMES={'legacy_dtw':'기존 DTW 기준','pose_capacity_matched':'관절 + 시간별 판정부','pose_plus_signrep':'관절 + SignRep + 시간별 판정부'}
def pct(v):return f'{v*100:.2f}%'
def main():
 scores=read(U/'SCORES.json');paired=read(U/'PAIRED_FEEDBACK.json');control=read(U/'CONTROLS.json');temporal=read(U/'TEMPORAL_SCORES.json');rt=read(U/'RUNTIME_RESULT.json');audit=read(U/'LABEL_AUDIT.json');info,arrays,cases=load_cases();rows={r['alias']:r for r in info['rows']}
 data=dict(conditions=NAMES,features=KO,targets=info['targets'],queries=paired['queries'],results=paired['rows'],cases={},guide_status='표현 전체의 필수 단계 확인 0/12. 일부 설명 연결 4/12. 교육적 수정 지시 없음.')
 for c in CONDITIONS:
  data['cases'][c]={}
  for a,case in cases[c].items():
   r=rows[a];data['cases'][c][a]=dict(alias=a,label=r['label'],person=r['person'],time=(case['seq']['timestamp_us']/1e6).tolist(),frames=case['seq']['frame_id'].tolist(),prob=None if case['prob'] is None else case['prob'].tolist(),support=case['support'].astype(int).tolist(),video='/video/'+a,video_file=Path(r['source']).as_uri(),box=r['box'])
 (U/'VIEW_DATA.json').write_text(json.dumps(data,ensure_ascii=False),encoding='utf8')
 (U/'VIEW_DATA.js').write_text('window.RESEARCH_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf8')
 table=''
 for group,name in [('held_person','배운 표현 · 새 기록 ID 2명 / 24영상'),('held_expression_person','새 표현 4개 · 새 기록 ID / 8영상')]:
  s=[next(r for r in scores['summary'] if r['condition']==c and r['group']==group) for c in CONDITIONS[1:]]
  table+=f'<tr><td>{name}</td><td>{pct(s[0]["mean"])}<small>{pct(s[0]["range"][0])}~{pct(s[0]["range"][1])}</small></td><td>{pct(s[1]["mean"])}<small>{pct(s[1]["range"][0])}~{pct(s[1]["range"][1])}</small></td><td>{(s[1]["mean"]-s[0]["mean"])*100:+.2f}%p</td></tr>'
 featuretable=''
 for j,f in enumerate(info['targets']):
  vals=[np.mean([r['targets'][j]['balanced_accuracy'] for r in scores['rows'] if r['condition']==c and r['group']=='held_person']) for c in CONDITIONS[1:]]
  featuretable+=f'<tr><td>{KO[f]}</td><td>{pct(vals[0])}</td><td>{pct(vals[1])}</td><td>{(vals[1]-vals[0])*100:+.2f}%p</td></tr>'
 tempos=[]
 for c in CONDITIONS[1:]:
  rs=[r for r in temporal['rows'] if r['condition']==c];tempos.append(dict(condition=c,correct_brackets=[r['both_endpoints_and_bracket'] for r in rs],stable_changes=[r['stable_pair_predicted_changes'] for r in rs]))
 runt=''.join(f'<tr><td>{NAMES[r["condition"]]}</td><td>{r["median"]:.2f}초</td><td>{r["p95"]:.2f}초</td><td>{r["range"][0]:.2f}~{r["range"][1]:.2f}초</td></tr>' for r in rt['summary'])
 text=f'''요청한 1~4단계 비교 주기 실행 완료 — 2026-10-31
목표: 어느 손이 어떤 모양과 움직임을 어떤 순서로 수행했고, 참고와 어느 시간대에서 다른지 근거를 보여주는 연구 경로.

1. 입력과 설명 연결
공개16표현·80영상·9817프레임. 기존12표현의학습36/개발12/새사람24, 새표현4개의8영상. 학습기록ID1~3,개발4,평가5~6. 원래36영상은이번학습/개발용이며 새평가라고부르지않음. 모두과거TCN TRAIN자료로 C/T3F에독립인시험아님.
각도·위치69항목과SignRep768영상특징의시간대응을검산. 이전36배열bitwise일치. 17개후보특징중양성/음성각4표현·3기록ID이상지원하는10개만학습. 449개bodywrist-only ROI허용라벨제외후3384개확인라벨(전체17항목기준). 영상/프레임삭제0. 같은Codex초안교차확인1103개중35개충돌은보류. 독립사람/수어교육자정답아님.
설명전체형태일치0/12,부분근거연결4/12. 멀다의한손/두손불일치등때문에설명으로영상을정답화하지않음. 손바닥방향/필수반복/실제접촉은학습완료아님.

2. 6회학습과잠금평가
SignRep본체고정. 두판정부각47914파라미터,동일160epoch·seed42/43/44. 체크포인트고정최종epoch,평가점수로모델/임계값선택없음. C/T3F학습/교체없음.
배운표현새사람10특징균형정확도: 관절80.47%(78.40~81.78)→영상추가79.37%(74.70~82.06), -1.09%p.
새표현새사람:62.70%(61.87~64.31)→61.32%(59.41~63.29), -1.38%p.
오른손검지만펴짐+7.59%p/양손화면상겹침+12.30%p/오른손왼쪽이동+10.00%p. 왼손펴짐-16.67%p/오른손위이동-15.82%p. 특징마다상반된결과이므로전체채택하지않음.
영상별평균은84.21→86.35%로높아짐. 각영상에보이는특징종류와단일class지원이다르기때문에10특징균형평균과방향이다를수있음. 주지표를바꿔성공선언하지않음.
고정6영상의RGB변화구간6개: 관절{tempos[0]['correct_brackets']}→영상{tempos[1]['correct_brackets']}. 동일하게보이는인접샘플144쌍에서모델이바뀌었다고출력한수{tempos[0]['stable_changes']}→{tempos[1]['stable_changes']}. 일부짧은모양변화에는좋은신호. 정확한변화프레임/모든표현성능증명아님.

3. 여러참고와시간정렬·연구화면
새사람24영상각각학습사람3개의참고와따로비교,3경로216쌍. 평균궤적/다른사람단계이어붙이기없음. 시점별특징확률·관측지원·대응시간·차이후보표시. 관측0참고는낮은거리로정상합의하지않도록수정. 손누락으로참고분모가줄지않도록고정.
거리경고개수는기존3/24,관절2/24,영상2/24. 정상오류교정률은아님. 후보특징차이는관절24/24영상·726셀,영상23/24·398셀. 셀은인접프레임중복포함,독립오류사건수가아님. 정상공개자료에도많으므로바로'틀린수어'로표시불가.
속도0.75/1.25·양손각각미관측·짧은중간삭제360시험. 미관측144시험분모보존PASS. 모든216참고쌍과합성시험에서참고프레임범위보존. 알려진시간변형정렬오차평균은3경로모두약3.34~3.35ms:추가영상의시간정렬개선은확인못함. 저장특징의시간변형시험이며실제빠른촬영정확도아님. 삭제구간보존이곧필수동작누락검출성공은아님.

4. 원본부터처리시간
원본전체MediaPipe관절검출·좌우배정(CPU)과원본영상SignRep(GPU)를병행. 학습판정부+3참고DTW+JSON기록포함. 시작준비{rt['setup_seconds']:.2f}초별도. 표본3영상각2회.
SignRep경로중앙값{rt['summary'][2]['median']:.2f}초,95백분위{rt['summary'][2]['p95']:.2f}초. detector단계가대부분을차지하여GPU추가처리는병행중가려짐. CPU기준시간도동시GPU작업하에서측정한검출시간공유합계이므로독립CPU전용성능아님. 고정ROI/사전참고캐시전제,실제웹캠촬영종료부터의출시완료시간측정아님. 로컬웹전달/표시는별도측정.

판정과남은일
자료→6회학습→개별참고비교→원본처리시간/연구화면까지한주기완료. 기존C/T3F·운영웹/DTW보존. 사용자/멘토자료접근·학습·튜닝·선택없음.
제품채택보류:10특징주평가개선없음,교육적으로옳고그른수어/허용범위정답부족,전체대기시간목표미달. 일부변화구간개선은실재하지만전체피드백정확도개선으로확대해석하지않음.
다음은새모델탐색보다①현재실패가많은공통특징의관측/주석지원·시간범위를개발자료에서점검②가이드와형태가실제로맞는표현의필수단계와허용변이근거확보③촬영중기존관절추출을진행해종료후남는시간을실측하는순서. 이번잠금평가32영상은이제개발이력으로남기며다음후보최종선택에반복사용하지않음. 독립교육정답없이교정정확도90%/오류교정5%목표를달성했다고선언하지않음.
'''
 cpu=read(U/'CPU_RUNTIME_RESULT.json');cs=cpu['summary'][1]
 text+='\n로컬 웹 화면 검증: 세 경로×두 표현6시험에서 영상 시간대응/영상 디코딩/그래프표시 PASS. 공개영상사용, 별도 Chrome시험프로필. 최초 결과JSON전달·파싱·2화면갱신 약0.109초(1회), 전체웹캠제품시간은아님.\n'
 text+=f'\n별도 CPU 단독 관절 경로: 중앙 {cs["median"]:.2f}초, p95 {cs["p95"]:.2f}초, 범위 {cs["range"][0]:.2f}~{cs["range"][1]:.2f}초(6회). 준비 {cpu["setup_seconds"]:.2f}초별도. GPU추론없음. SignRep CPU단독은미측정.\n'
 (U/'RESULT_KO.txt').write_text(text,encoding='utf8')
 summary=f'''<h1>동작을 시간 순서로 비교하는 연구 경로</h1><p class="lead">1~4단계 비교는 끝났습니다. <b>SignRep를 추가하면 일부 손모양 변화는 더 잘 찾았지만, 전체 특징 정확도는 개선되지 않았습니다.</b> 기존 모델과 운영 웹은 유지합니다.</p>
 <div class="cards"><div><b>80개 공개 영상</b><span>학습36 · 개발12 · 평가32<br>사용자 촬영 0</span></div><div><b>두 조건 × 3회</b><span>같은 크기 판정부 · 본체 고정<br>10개 물리 특징</span></div><div><b>연구용 차이 표시</b><span>틀린 수어라고 단정하지 않음<br>기존 C·T3-F 보존</span></div></div>
 <h2>무엇이 달라졌나</h2><table><tr><th>평가 조건</th><th>관절만</th><th>SignRep 추가</th><th>변화</th></tr>{table}</table><p>10개 특징의 균형 정확도, 3회 평균(아래는 범위)입니다. <b>단어 인식률·교육 피드백 정답률이 아닙니다.</b> 새 사람은 이번 판정부에서 미사용인 기록 ID이며, 과거 TCN에는 사용된 공개 자료입니다.</p>
 <details><summary>어떤 특징이 좋아지고 나빠졌는지 보기</summary><table><tr><th>특징 / 배운 표현·새 사람</th><th>관절</th><th>영상 추가</th><th>변화</th></tr>{featuretable}</table></details>
 <p>6개 영상에서 확인한 모양 변화 구간 6개를 맞힌 수: <b>관절 4·3·3개 → 영상 추가 6·5·4개</b>. 일부 변화에는 이득이 있으나 표본이 작습니다. 같은 물리 상태로 관측한 인접144쌍에서 두 시점의 예측 상태가 달랐던 수도 5·6·5 → 1·2·4로 줄었습니다.</p>
 <h2>두 영상을 직접 비교하세요</h2><p>아래의 새 사람 영상은 <b>공개 자료</b>입니다. 슬라이더를 움직이면 DTW가 짝지은 참고 시점이 함께 이동합니다. 그래프는 관측된 손 특징에 대한 <b>모델의 추정값</b>이며, 1이면 맞는 수어라는 뜻이 아닙니다. 회색 점은 비교 근거가 부족합니다.</p>
 <div class="controls"><label>영상 <select id="query"></select></label><label>비교 경로 <select id="condition"></select></label><label>참고 <select id="reference"></select></label><label>특징 <select id="feature"></select></label></div>
 <div class="videos"><div><h3 id="qlabel"></h3><video id="qvideo" controls muted playsinline preload="metadata"></video></div><div><h3 id="rlabel"></h3><video id="rvideo" controls muted playsinline preload="metadata"></video></div></div>
 <input id="seek" type="range" min="0" value="0" step="1"><p id="times"></p><svg id="graph" viewBox="0 0 1000 210" role="img" aria-label="시간 정렬 후 손 특징 확률 비교"></svg><p id="current"></p><p id="evidence"></p><div id="events"></div>
 <details><summary>범위·손내림·관측 부족 검증</summary><p>24영상 × 3개 참고 × 3경로의 216쌍에서 참고 시간 범위를 유지했습니다. 한 손을 지운 144개 대조 시험에서도 필요한 비교 분모를 줄이지 않았습니다. 참고에서 낮게 쉬는 손이라는 근거가 있는 경우만 별도로 처리합니다. 이 검사는 비교 범위와 분모 보존을 확인합니다. 기존 좌표·손 배정은 수정하지 않았지만, 모든 정상 교차의 의미 보존이나 교육적 허용성을 별도 정답으로 검증한 것은 아닙니다.</p><p>속도를 바꾼 저장 특징 시험의 시간 오차는 세 경로 모두 약3.34~3.35ms였습니다. 영상 특징 추가로 시간 정렬 자체가 좋아졌다는 결과는 없었습니다. 짧은 구간을 지운 시험은 그 시간대를 비교에서 없애지 않는지 확인했으며, 실제 필수 동작을 놓쳤는지에 대한 정답 시험은 아닙니다.</p><p>손 관절 없이 몸의 손목만으로 손 전체가 범위 안이라고 처리하던449개 타깃을 제외했습니다. 원본80영상과9817프레임은 유지했습니다. 낮게 내려간 손의 중요한 의미가 잘리지 않았다고 전부 보장하지는 못합니다.</p></details>
 <h2>시간은 얼마나 걸렸나</h2><table><tr><th>경로</th><th>중앙값</th><th>95백분위</th><th>범위 / 6회</th></tr>{runt}</table><p>CPU 원본 관절 추출과 GPU 영상 특징 추출을 병행하고, 판정부·3참고 DTW·결과 기록 시간을 더한 값입니다. 준비 {rt['setup_seconds']:.2f}초는 별도입니다. 기존/관절 경로도 이때의 공통 검출 시간을 쓰므로 독립 CPU 단독 측정은 아닙니다. 참고 특징·ROI는 미리 준비했습니다. 실제 촬영 중 선처리나 배포 게임의 최종 대기시간은 아직 측정하지 않았습니다.</p>
 <h2>현재 할 수 있는 것과 남은 것</h2><p><b>할 수 있음:</b> 어느 손의 어떤 특징이 어느 시간에 참고와 다르게 추정됐는지, 어디는 근거가 부족한지 영상과 함께 확인하기.</p><p><b>아직 못함:</b> 코와 광대의 실제 접촉, 필수 반복 횟수, 표현마다 허용되는 차이를 검증된 교육 기준으로 판정하기. 12표현의 설명 중 전체 형태가 확인된 연결은0개이며 4개는 일부 설명만 연결됩니다. 손바닥 방향 등7개 후보 항목은 자료가 부족해 학습하지 않았습니다.</p><p>다음은 실패한 공통 특징의 관측·라벨과 시간 범위를 개발 자료에서 점검하고, 실제 표현과 맞는 가이드의 필수 단계·허용 변이를 확인하는 작업입니다. 속도는 촬영 중 관절 추출을 이어가 종료 후에 남는 처리시간부터 검증합니다. 이번 평가 점수에 맞춰 규칙을 바꾸거나 좋은 항목만 골라 채택하지 않습니다.</p>
 <p><a href="RESULT_KO.txt">전체 쉬운 한국어 결과</a> · <a href="SCORES.json">저장 점수</a> · <a href="review/INPUTS_INDEPENDENT.json">입력 별도 검산</a> · <a href="review/RESULTS_VERIFICATION_FINAL.json">결과 별도 검산</a> · <a href="review/guide_stage_map.json">가이드 연결 근거</a></p>'''
 summary+='<p>로컬 웹 화면: 세 경로×두 표현의6개 조작 시험 통과. 결과 전달·화면 갱신 약0.109초(1회) 별도 측정. 이 값만으로 실제 게임의 최종 대기시간을 보장하지 않습니다.</p>'
 summary+=f'<p><b>별도 CPU 단독 관절 경로</b>: 중앙 {cs["median"]:.2f}초 · 95백분위 {cs["p95"]:.2f}초 · 범위 {cs["range"][0]:.2f}~{cs["range"][1]:.2f}초(6회). GPU 모델은 실행하지 않았으며, 최초 준비 {cpu["setup_seconds"]:.2f}초는 별도입니다.</p>'
 css='''body{font-family:"Malgun Gothic",sans-serif;margin:0;background:#f5f7fb;color:#192a42;line-height:1.7}main{max-width:1120px;margin:auto;padding:28px}h1{font-size:30px}h2{margin-top:38px}p{max-width:1000px}.lead{font-size:19px}.cards,.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards>div{background:#e6edf7;padding:20px;border-radius:14px}.cards span,small{display:block;font-size:13px;color:#4f637d}.cards b{font-size:20px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:13px;border-bottom:1px solid #d9e1ec;text-align:left}details{margin:20px 0;padding:16px;background:white;border-radius:10px}summary{cursor:pointer;font-weight:bold}.videos{grid-template-columns:1fr 1fr}video{width:100%;background:#111;border-radius:10px}.controls{display:flex;flex-wrap:wrap;gap:12px}select{padding:9px;max-width:290px}#seek{width:100%;margin-top:18px}#graph{width:100%;background:white;border:1px solid #d9e1ec}#events{background:#fff5da;padding:14px;border-radius:10px}a{color:#165db0}@media(max-width:750px){main{padding:15px}.videos,.cards{grid-template-columns:1fr}h1{font-size:25px}}'''
 js=r'''let D;const $=id=>document.getElementById(id);const el=id=>$(id).value;let active;function opts(id,items){$(id).innerHTML=items.map(([v,t])=>`<option value="${v}">${t}</option>`).join('')}
 function seekVideo(v,t){v.dataset.seekTime=t;if(v.readyState>=1){v.currentTime=t}else{v.addEventListener('loadedmetadata',()=>v.currentTime=Number(v.dataset.seekTime),{once:true})}}function caseFor(c,a){return D.cases[c][a]}function setup(){opts('query',D.queries.map(q=>[q.alias,`${q.label} · 기록 ID${D.cases.legacy_dtw[q.alias].person}`]));opts('condition',Object.entries(D.conditions));$('condition').value='pose_plus_signrep';opts('feature',D.targets.map(f=>[f,D.features[f]]));$('feature').value='R_open_four';updateQuery();$('query').onchange=updateQuery;$('condition').onchange=updateQuery;$('reference').onchange=updatePair;$('feature').onchange=draw;$('seek').oninput=()=>{const i=+$('seek').value;seekVideo($('qvideo'),active.q.time[i]);seekVideo($('rvideo'),active.r.time[active.p.query_to_reference[i]]);draw()}}
 function updateQuery(){const q=D.queries.find(q=>q.alias===el('query'));opts('reference',q.references.map(r=>[r.alias,`기록 ID${r.person}`]));updatePair()}
 function updatePair(){const c=el('condition'),a=el('query'),b=el('reference');const result=D.results.find(r=>r.query===a&&r.condition===c);active={q:caseFor(c,a),r:caseFor(c,b),p:result.pairs.find(p=>p.reference===b),result};$('qvideo').src=location.protocol==='file:'?active.q.video_file:active.q.video;$('rvideo').src=location.protocol==='file:'?active.r.video_file:active.r.video;$('qlabel').textContent=`공개 새 사람: ${active.q.label} / ID${active.q.person}`;$('rlabel').textContent=`개별 참고: ${active.r.label} / ID${active.r.person}`;$('seek').max=active.q.time.length-1;$('seek').value=0;seekVideo($('qvideo'),active.q.time[0]);seekVideo($('rvideo'),active.r.time[active.p.query_to_reference[0]]);draw()}
 function draw(){const {q,r,p,result}=active;const i=+$('seek').value,j=p.query_to_reference[i],f=D.targets.indexOf(el('feature'));$('times').textContent=`대응 시점: 새 사람 ${q.time[i].toFixed(3)}초 → 참고 ${r.time[j].toFixed(3)}초 (원본 프레임 ${q.frames[i]} / ${r.frames[j]})`;
 $('evidence').textContent=`이 참고의 비교 지원 ${p.compared_cells}/${p.reference_expected_cells}칸 (${(100*p.coverage).toFixed(1)}%). 부족 ${p.missing_cells}칸. 원본 시간의 추가 미관측 ${p.query_direct_unobserved_cells}칸. 낮은 거리만으로 잘했다고 판정하지 않습니다.`;
 let s='<line x1="35" y1="170" x2="980" y2="170" stroke="#b5c2d4"/><line x1="35" y1="25" x2="35" y2="170" stroke="#b5c2d4"/><text x="5" y="30" font-size="14">1</text><text x="5" y="175" font-size="14">0</text>';
 if(q.prob){const xx=k=>35+945*k/(q.time.length-1),yy=v=>170-145*v;for(const [isref,color] of [[false,'#1964b3'],[true,'#d57914']]){const pts=q.time.map((t,k)=>{let z=isref?p.query_to_reference[k]:k,cas=isref?r:q;return `${xx(k)},${yy(cas.prob[z][f])}`}).join(' ');s+=`<polyline points="${pts}" stroke="${color}" fill="none" stroke-width="2"/>`}
 q.time.forEach((t,k)=>{if(!q.support[k][f]||!r.support[p.query_to_reference[k]][f])s+=`<circle cx="${xx(k)}" cy="185" r="3" fill="#999"/>`});s+=`<line x1="${xx(i)}" x2="${xx(i)}" y1="18" y2="190" stroke="#111"/>`;const yes=q.support[i][f]&&r.support[j][f];$('current').textContent=`${D.features[el('feature')]} — 파랑(새 사람) ${q.prob[i][f].toFixed(2)}, 주황(참고) ${r.prob[j][f].toFixed(2)}. ${yes?'검출 근거를 갖춘 비교 후보':'이 시점은 비교 근거 부족'}. 확률은 교정의 확신도가 아닙니다.`;
 }else{$('current').textContent='기존 DTW는 이 학습 특징의 확률을 출력하지 않습니다. 시간 대응과 관측 근거를 비교하세요.';s+='<text x="80" y="95" fill="#65768c">기존 좌표·각도 DTW 시간 대응</text>'}$('graph').innerHTML=s;
 const ev=result.events.filter(e=>e.feature===el('feature')).sort((a,b)=>a.query_seconds-b.query_seconds);$('events').textContent=ev.length?`3명 중 2명 이상 참고와 차이가 있다고 추정한 시점: ${ev.slice(0,12).map(e=>e.query_seconds.toFixed(2)+'초').join(', ')}${ev.length>12?' …':''}. 실제로 틀린 수어라는 판정이 아닙니다.`:'이 특징에서 여러 참고가 함께 지지하는 차이 후보는 없습니다. 근거 부족까지 일치로 해석하면 안 됩니다.'}
 const started=performance.now();(location.protocol==='file:'?Promise.resolve(JSON.stringify(window.RESEARCH_DATA)):fetch('VIEW_DATA.json').then(r=>r.text())).then(t=>{D=JSON.parse(t);setup();requestAnimationFrame(()=>requestAnimationFrame(()=>location.protocol!=='file:'&&fetch('/ui-measurement',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({payload_fetch_and_two_animation_frames_ms:performance.now()-started,payload_bytes:new Blob([t]).size,ui_queries:D.queries.length,user_agent:navigator.userAgent})})))})
 '''
 (U/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>시간별 동작 피드백 비교 결과</title><style>'+css+'</style><main>'+summary+'</main><script src="VIEW_DATA.js"></script><script>'+js+'</script></html>',encoding='utf8')
 print('RESULT_KO.html/txt and VIEW_DATA.json created')
if __name__=='__main__':main()
