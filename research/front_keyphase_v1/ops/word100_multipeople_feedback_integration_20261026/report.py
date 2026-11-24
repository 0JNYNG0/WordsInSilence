from pathlib import Path
import json,html,hashlib,datetime,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path('C:/MyUnityProjects/WordsInSilence');O=ROOT/'research/front_keyphase_v1/ops';U=O/'word100_multipeople_feedback_integration_20261026';FC=O/'word100_feedback_evidence_consumer_20261026'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=read(U/'VERIFIED.json');cv=read(U/'CAUSE_VERIFIED.json');a=read(U/'CAUSE_AUDIT.json');r=read(U/'RESULT.json');f=read(U/'FROZEN.json');fv=read(FC/'VERIFIED.json');assert v['status']=='PASS_REAL600_EVIDENCE_PRESERVATION' and cv['status']=='PASS_CAUSES_ALL600'
meta={x['clip_id']:x for x in f['rows']};causes={x['name']:x for x in a['rows']};by={};allreceipts={}
for row in r['rows']:
 qid=row['query'];receipt=read(Path(row['receipt']));allreceipts[row['name']]=receipt
 if qid not in by:by[qid]=dict(id=qid,word=meta[qid]['word_code'],label=meta[qid]['label'],person=meta[qid]['signer_id'],recognition=receipt['recognition'],references=[])
 item=dict(id=row['reference'],person=meta[row['reference']]['signer_id'],groups=[],cause={k:causes[row['name']][k] for k in ['uncorroborated','only_no_second_prediction','prediction_disagreement','identity_disagreement']},receipt=Path(row['receipt']).relative_to(U).as_posix())
 for h in receipt['hands']:
  gs=[]
  for g in h['groups']:
   gs.append({k:g[k] for k in ['group','unit','audit_cells','point_compared_cells','sequence_compared_cells','corroborated_sequence_cells','point_coverage','sequence_coverage','corroborated_coverage','point_mean_difference','sequence_level_mean_difference','sequence_change_mean_difference','timeline']})
  item['groups'].append(gs)
 by[qid]['references'].append(item)
data=sorted(by.values(),key=lambda x:(x['label'],x['person']));assert len(data)==300 and all(len(x['references'])==2 for x in data)
plt.rcParams['font.family']='Malgun Gothic';plt.rcParams['axes.unicode_minus']=False
example=v['reference_disagreements'][0];qid=example['query'];selected=by[qid];side=0 if example['side']=='L' else 1
fig,axes=plt.subplots(2,2,figsize=(14,7),gridspec_kw={'height_ratios':[3,1]},constrained_layout=True)
for col,ref in enumerate(selected['references']):
 g=ref['groups'][side][0];t=g['timeline'];x=np.array(t['reference_seconds']);axes[0,col].plot(x,t['point_difference'],color='#1555b6',lw=1.6,label='원래 순간별 각도 차이');axes[0,col].plot(x,t['sequence_level_difference'],color='#e67630',lw=1.3,label='연속 구간의 각도 차이')
 axes[0,col].set(title=f"참고 촬영 ID {ref['person']} · {example['side']} 손",ylabel='영상에서 계산한 각도 차이 (도)',xlabel='해당 참고 영상 시각 (초)');axes[0,col].legend(fontsize=10);axes[0,col].grid(alpha=.18)
 ex=np.array(t['audit_channels']);ok=ex>0;pc=np.array(t['point_channels']);sc=np.array(t['sequence_channels']);qc=np.array(t['corroborated_channels'])
 axes[1,col].plot(x,np.where(ok,pc/np.maximum(ex,1)*100,np.nan),label='순간 계산 가능',color='#1555b6');axes[1,col].plot(x,np.where(ok,sc/np.maximum(ex,1)*100,np.nan),label='연속 계산 가능',color='#e67630');axes[1,col].plot(x,np.where(ok,qc/np.maximum(ex,1)*100,np.nan),label='재확인 근거 있음',color='#16836a');axes[1,col].set(ylim=(-5,105),ylabel='각도별 근거 (%)',xlabel='해당 참고 영상 시각 (초)');axes[1,col].grid(alpha=.18)
fig.suptitle(f"공개 자료 예시: {selected['label']} / 촬영 ID {selected['person']}을 두 참고와 각각 비교\n수치가 다르다고 동작 오류를 확정하지 않음 · 재확인 없음은 틀렸다는 뜻이 아님",fontsize=15);axes[1,1].legend(fontsize=9,loc='lower left');fig.savefig(U/'reference_example.png',dpi=135);plt.close(fig)
groups=v['groups'];domain=sum(x['domain'] for x in groups.values());seq=sum(x['sequence'] for x in groups.values());quality=sum(x['quality'] for x in groups.values());pct=cv['only_missing_second_percent']
txt=f'''인식 결과와 여러 참고의 동작 비교를 연결한 연구 결과

이번에 한 작업
공개100개 단어의300개 영상(촬영 기록ID11·12·13)을 사용했다. 한 영상을 나머지 두ID의 같은 단어 영상과 각각 비교하여600쌍을 만들었다. 참고들의 관절을 평균내거나, 작은 차이가 나온 참고만 골라내지 않았다. 기존 C·T3-F가 같은 원본 영상에 내놓았던 인식 결과와 손가락·손바닥·팔꿈치·손목 차이, 관측 범위, 부족한 시간대를 한 연구 결과로 연결했다.

검증한 것
앞선4500개 합성시간·삭제·미관측 시험의 결과 표시를 전수 검산했다. 손이 통째로 없는600조건에서도 손가락과 손 위치 차이를0으로 기록하지 않고 비교불가로 표시했으며 팔 정보는 보존했다.
이번 실제 다른ID600쌍에서도 기존 연속 비교가 가능한 {seq:,}개 채널·시점과 기존 품질 근거 {quality:,}개를 모두 보존했다. 새 시간 대응에서 근거가 줄던134개는 원래 대응의 정보를 유지했다. 원래 순간별 차이도 유지했다.
각도와 위치의 평균차이는 작게 줄었지만, 실제 다른 사람 간에 어느 시점이 정답인지 판정한 자료가 없으므로 이를 정확도 개선으로 발표하지 않는다.

새로 확인한 핵심 문제
손가락은 연속 차이를 계산할 수 있는543,231개 중 기존 재확인 조건을 충족한 것이1,014개였다. 이것은 나머지 손가락이 모두 틀렸다는 뜻이 아니다.
재확인이 부족한542,217개 중527,230개({pct:.2f}%)는 두 번째 관측이 없다는 이유만으로 재확인하지 못한 경우였다. 실제 예측 불일치가 포함된 것은11,887개, 좌우 근거 불일치가 포함된 것은4,971개이며 이 두 원인은 서로 겹칠 수 있다.
현재 방식은 앞·중간·뒤의 모든 구간에서 양쪽 참고의 추가 관측이 있어야 재확인 조건을 충족한다. 따라서 한 순간만 재확인 자료가 없어도 연속 구간 전체는 재확인 완료로 부르지 않는다. 좌표로 계산한 원래 차이는 버리지 않는다.
좌표 계산 가능 / 두 번째 관측 없음 / 관측끼리 불일치 / 재확인 근거 있음은 서로 다른 말이다. 어느 상태도 자동으로 올바른 수어의 정답 판정이 되지는 않는다.

인식 결과와 YOLO의 현재 위치
같은 원본300개에 이미 저장된 C 294/300(98.00%), T3-F 295/300(98.33%) 결과를 연결했다. 새 학습이나 새 인식률 측정은0회이며 이번 작업이 인식률을 올렸다는 의미가 아니다.
MediaPipe로 몸과 손을 기본 관측하고, 재검출도 손을 놓친 경우 손 전용YOLO를 보조로 쓰는 후보는 별도로 검증해왔다. 손목 위치 보완과 정밀 손가락 모양 확인을 구분한다. 보완 손목을 넣은 W 학습은 일관된 인식 향상이 없어 미채택했다. 이번600쌍은 원래C의 저장 관절을 사용하며 YOLO 좌표를 운영모델에 이미 적용한 결과가 아니다.

보는 방법
아래 탐색기에서 단어와 촬영ID를 고르면, C·T3-F의 저장 인식 결과와 두 참고별 차이가 나온다. 손가락 각도는 도(°), 위치는 어깨폭으로 나눈 거리이다. 100점이나 올바름 비율이 아니다.
파란 선은 순간별 차이, 주황 선은 앞·중간·뒤 연속 구간의 차이다. 끊긴 선은 해당 비교를 계산할 수 없었던 구간이다. 관측 비율은 지정한 참고의 점검 범위에서 계산 가능한 비율이고, 두 손이 반드시 그 시간 내내 동작해야 한다는 뜻이 아니다.

다음 작업
재확인이 없던 구간과 실제 예측 불일치 구간을 구분한 채, 공개 자료에서 현재 이미지로 다시 확인할 가치가 있는 시간대를 공통 기준으로 선정한다. 같은 프레임을 무조건 두 번 본다는 이유로 정답처럼 취급하지 않고, 눈으로 확인한 공개 표본의 위치·손가락 근거와 비교한다. 좌표와 순간별 비교는 계속 보존하고, 정상 손·교차·짧은 변화가 훼손되는지 함께 확인한다. 그 후 유효한 보완만 인식과 DTW에서 각각 고정 비교한다.

한계와 보호 범위
이300개는 기존 개발자료이며 새 사용자 최종 성능평가가 아니다. 2D 각도는 가려진3D 손 모양을 보장하지 않는다. 이 연구 표시가 올바른 수어 또는 교정 방향을 검증한 단계는 아니다. 구현·별도산술검산은 같은 Codex이며 독립적인 사람의 정답 검증과 다르다. 기존 C·T3-F·E3·운영 웹·DTW·원본은 보존했다. 사용자·멘토·이미 노출된 추가199/200자료 접근/학습/튜닝/후보선택0. 전체 목표는 계속 진행 중이다.
'''
(U/'RESULT_KO.txt').write_text(txt,encoding='utf-8')
template='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>공개 수어 자료 · 인식과 비교 근거</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf2f7;color:#193047;font:16px/1.7 system-ui}main{max-width:1230px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.4}h2{font-size:24px;margin-top:32px}.lead{font-size:19px}.notice{background:#fff4db;border-left:4px solid #d59a36;padding:16px 20px}.grid,.refs{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}.card{background:white;padding:23px;border-radius:14px;margin-top:16px}.pill{display:inline-block;padding:4px 10px;background:#e3eee9;border-radius:8px;margin:4px 8px 4px 0}label{display:inline-block;margin-right:14px}select{font:inherit;padding:8px;border:1px solid #a8bac8;border-radius:8px;background:white;max-width:100%}strong{font-size:20px}small,.muted{color:#53697e}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:9px 6px;text-align:left;border-bottom:1px solid #dfe7ee}.bar{height:10px;background:#e3e9ee;border-radius:5px;margin:7px 0}.bar i{display:block;height:100%;background:#e67630;border-radius:5px}.bar.quality i{background:#16836a}.chart{width:100%;height:auto;background:#f6f8fb;border-radius:8px}.legend{font-size:13px}.legend b{padding-right:14px}pre{white-space:pre-wrap;font:inherit}a{color:#145da0}img{width:100%;border-radius:12px}summary{cursor:pointer;color:#145da0;font-weight:600}.metric{font-size:29px;display:block}.weak{color:#866520}@media(max-width:800px){.grid,.refs{grid-template-columns:1fr}main{padding:18px}h1{font-size:26px}}
</style><main><p>WordsInSilence · 운영과 분리된 연구 결과</p><h1>단어 인식과 동작 비교를<br>관측 근거와 함께 보기</h1><p class="lead">인식 결과와 세밀한 동작 비교는 별개입니다. 보이지 않는 손을 ‘잘 맞음’으로 표시하지 않습니다.</p><div class="notice">공개 100개 단어 · 300개 개발 영상 · 서로 다른 촬영 ID 600쌍. 새 실사용 정확도나 검증된 교정 판정은 아닙니다.</div>
<div class="grid"><div class="card"><strong>계산할 수 있는 연속 비교</strong><span class="metric">__SEQ__개 보존</span>손가락·손 위치·팔의 원래 비교 근거를 모두 유지했습니다.</div><div class="card"><strong>재확인이 부족한 손가락 비교</strong><span class="metric">__PCT__%는 두 번째 관측 없음</span>‘자료가 아직 없음’과 ‘서로 다르게 검출됨’을 구분했습니다.</div></div>
<h2>공개 영상 비교 탐색</h2><p>각 촬영본을 다른 두 참고와 각각 비교합니다. 여러 참고를 평균낸 가상의 손 동작은 만들지 않습니다.</p><div class="card"><label>단어 <select id="word"></select></label><label>촬영 기록 ID <select id="person"></select></label><label>손 <select id="hand"><option value="0">왼손</option><option value="1">오른손</option></select></label><label>비교 부위 <select id="group"><option value="0">손가락 관절각</option><option value="1">손바닥 위치</option><option value="2">팔꿈치 위치</option><option value="3">몸 검출의 손목 위치</option><option value="4">손 검출의 손목 위치</option></select></label><p id="rank"></p><small>인식은 같은 원본 영상에 저장된 C·T3-F 결과입니다. 이 페이지에서 모델을 재실행하지 않습니다.</small></div><div id="refs" class="refs"></div>
<p class="muted">두 손 모두가 항상 필요하다고 가정한 정답률이 아닙니다. 참고의 주석 구간을 점검 범위로 유지하며, 정상 손내림과 미검출을 자동으로 동작 오류라고 판정하지 않습니다. 각 참고의 시간축도 서로 다릅니다.</p>
<h2>두 참고가 서로 다른 차이를 보이는 예</h2><img src="reference_example.png" alt="공개 오억 표현의 두 참고별 차이와 비교 가능한 범위"><p>손가락 각도는 카메라에 투영된 값입니다. 자세·카메라 각도·검출·동작 자체의 차이가 모두 영향을 줄 수 있으므로 큰 수치 하나로 교정 방향을 확정하지 않습니다.</p>
<details><summary>작업 내용, 수치의 의미와 남은 문제 자세히 보기</summary><pre>__TEXT__</pre></details><p><a href="VERIFIED.json">600쌍 전수 검산</a> · <a href="CAUSE_VERIFIED.json">재확인 부족 원인 검산</a> · <a href="../word100_feedback_evidence_consumer_20261026/RESULT_KO.html">4,500조건 연결부 검산</a></p></main><script id="data" type="application/json">__DATA__</script><script>
const rows=JSON.parse(document.getElementById('data').textContent),$=s=>document.getElementById(s),words=[...new Map(rows.map(x=>[x.word,x.label])).entries()];
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(v,unit=''){return v==null?'비교 근거 없음':v.toFixed(unit==='°'?1:3)+unit;}
function pct(v){return v==null?'범위 없음':(v*100).toFixed(1)+'%';}
function plot(t){const n=t.reference_seconds.length,W=520,H=180,p=35,max=Math.max(1,...t.point_difference.filter(x=>x!=null),...t.sequence_level_difference.filter(x=>x!=null));function line(a,col){let d='',open=false;for(let i=0;i<n;i++){if(a[i]==null){open=false;continue}let x=p+i/Math.max(n-1,1)*(W-p-10),y=H-p-a[i]/max*(H-2*p);d+=(open?'L':'M')+x.toFixed(2)+','+y.toFixed(2);open=true}return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="1.7"/>';}return '<svg class="chart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="차이와 비교 가능 구간"><line x1="35" y1="145" x2="510" y2="145" stroke="#b4c1ce"/>'+line(t.point_difference,'#1555b6')+line(t.sequence_level_difference,'#e67630')+'<text x="3" y="38" font-size="11">'+max.toFixed(1)+'</text><text x="8" y="146" font-size="11">0</text><text x="35" y="170" font-size="11">'+t.reference_seconds[0].toFixed(2)+'초</text><text x="450" y="170" font-size="11">'+t.reference_seconds[n-1].toFixed(2)+'초</text></svg>';}
function person(){const items=rows.filter(x=>x.word==$('word').value);$('person').innerHTML=items.map(x=>'<option value="'+x.id+'">'+x.person+'</option>').join('');render();}
function render(){const q=rows.find(x=>x.id==$('person').value),h=+$('hand').value,gi=+$('group').value; if(!q)return;
 $('rank').innerHTML=Object.entries(q.recognition).map(([k,v])=>'<span class="pill">'+esc(k)+' · '+(v.correct?'목표 단어 1위':v.top5_correct?'목표 단어 상위 5개 포함':'목표 단어 상위 5개 밖')+'</span>').join('');
 $('refs').innerHTML=q.references.map(r=>{const g=r.groups[h][gi],unit=gi==0?'°':' 어깨폭',c=r.cause;return '<article class="card"><h3>참고 촬영 ID '+r.person+'</h3><p><strong>순간별 차이 '+fmt(g.point_mean_difference,unit)+'</strong></p>'+plot(g.timeline)+'<p class="legend"><b style="color:#1555b6">파랑: 순간별 차이</b><b style="color:#e67630">주황: 연속 구간 차이</b></p><table><tr><th>앞·중간·뒤의 위치/각도 차이</th><td>'+fmt(g.sequence_level_mean_difference,unit)+'</td></tr><tr><th>변화량 차이</th><td>'+fmt(g.sequence_change_mean_difference,unit)+'</td></tr></table><p>순간 비교 가능 '+pct(g.point_coverage)+'<br>연속 비교 가능 '+pct(g.sequence_coverage)+'</p><div class="bar"><i style="width:'+(100*(g.sequence_coverage||0))+'%"></i></div><p>기존 재확인 조건 충족 '+pct(g.corroborated_coverage)+'</p><div class="bar quality"><i style="width:'+(100*(g.corroborated_coverage||0))+'%"></i></div><small>녹색 비율은 정확도·올바름 비율이 아닙니다. 손가락은 추가 관측의 일치 조건이며, 위치는 기존 좌표/좌우 근거 조건입니다.</small><details><summary>이 비교의 양손 손가락 재확인 부족 원인</summary><p>두 번째 관측 없음만 해당: '+c.only_no_second_prediction.toLocaleString()+'<br>예측 불일치 포함: '+c.prediction_disagreement.toLocaleString()+'<br>좌우 근거 불일치 포함: '+c.identity_disagreement.toLocaleString()+'</p><small>채널·시점 수이며 원인은 겹칠 수 있습니다. 선택 부위가 아닌 양손 손가락 전체 집계입니다. 모두 동작 오류 판정과 구분합니다.</small></details><p><a href="'+r.receipt+'">각도별 수치·부족 시간대 전체 기록</a></p></article>';}).join('');}
 $('word').innerHTML=words.map(([w,l])=>'<option value="'+w+'">'+esc(l)+' · '+w+'</option>').join('');$('word').onchange=person;for(const id of ['person','hand','group'])$(id).onchange=render;person();
</script></html>'''
embedded=json.dumps(data,ensure_ascii=False,separators=(',',':'),allow_nan=False).replace('</','<\\/')
doc=template.replace('__SEQ__',f'{seq:,}').replace('__PCT__',f'{pct:.2f}').replace('__TEXT__',html.escape(txt)).replace('__DATA__',embedded)
(U/'RESULT_KO.html').write_text(doc,encoding='utf-8')
fctxt=f'''차이 수치와 관측 근거를 연결한 연구용 결과 표시

공개100단어·300영상에서 파생한4500개 시험의 저장 결과를 새 연구용 결과 형식으로 연결했다. 손가락15각도·손바닥·팔꿈치·몸의손목·손의손목을 좌우로 나누고, 순간별 차이·연속구간 차이·변화량·비교가능비율·재확인비율·근거부족 시간대를 각각 보존했다.
전수 별도 검산: 차이요약225,000개, 시간구간585,000개, 시각별 값16,782,750개. 손이 통째로 없는600조건에서 손 모양·손 위치는 비교불가(null)로 남기고 팔 정보는 유지했다. 원본300개만 저장 C·T3-F 인식 결과를 연결했다.
이4500개는 같은 영상의 시간변형·삭제·미관측 시험이다. 서로 다른 사람의 새 정확도 시험은 아니며, 후속600쌍 연구에서 별도로 다른 촬영ID의 실제 자료를 비교했다. 기존 C·T3-F·E3와 운영 웹·DTW는 변경하지 않았다. 학습0·새추론0·사용자자료0.
별도 산술검산은 같은 Codex이며 독립 사람 정답 검증과는 다르다. 올바른 수어 판정과 교정 방향은 아직 미검증이며 임의로 생성하지 않는다.
'''
(FC/'RESULT_KO.txt').write_text(fctxt,encoding='utf-8');(FC/'RESULT_KO.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>피드백 근거 연결부 검산</title><style>body{font:18px/1.9 system-ui;max-width:1000px;margin:40px auto;padding:25px;color:#193047;background:#edf2f7}pre{white-space:pre-wrap;font:inherit}a{color:#145da0}</style><h1>피드백 근거 연결부 검산</h1><pre>'+html.escape(fctxt)+'</pre><a href="../word100_multipeople_feedback_integration_20261026/RESULT_KO.html">서로 다른 촬영 ID의600쌍 비교 탐색기</a></html>',encoding='utf-8')
checks=dict(at=datetime.datetime.now().astimezone().isoformat(),query_count=len(data),refs=sum(len(x['references']) for x in data),rank_count=300,support_cells=seq,quality_cells=quality,diagnostic_not_accuracy=True,no_server_or_camera=True,html_visual_rendered=False,static_png_requires_view=True,files={str(p):sha(p) for p in [U/'RESULT_KO.html',U/'RESULT_KO.txt',U/'reference_example.png',FC/'RESULT_KO.html',FC/'RESULT_KO.txt']})
(U/'REPORT_CHECK.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print('REPORTS_CREATED',len(data),seq,quality,pct,(U/'RESULT_KO.html').stat().st_size)
