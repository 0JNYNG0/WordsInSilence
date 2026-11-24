from pathlib import Path
import json,hashlib,re
from collections import Counter
U=Path(__file__).resolve().parent;P=U.parent/'word100_temporal_shape_context_20261027'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
d=read(U/'COMPARISON.json')
idx={r['case_id']:r for p in (U,P) for r in read(p/'visual_inputs/INDEX.json')['rows']}
center={r['case_id']:r for r in read(U/'independent/CENTER_LABELS.json')['rows']}
agree=Counter()
for s in d['sequences']:
 if not s['fresh_dense']:continue
 for f in ('index_only','open_four'):
  a,b=s['frames'][6]['direct'][f],center[s['case_id']][f]
  agree['both_unknown' if a is None and b is None else 'one_unknown' if a is None or b is None else 'known_agree' if a==b else 'known_conflict']+=1
dump(U/'CENTER_AGREEMENT.json',{'status':'COMPARED_FIRST_DRAFTS_UNCHANGED','counts':dict(agree),'scope':'Same-AI current image reviews, not human truth','secondary_sha256':sha(U/'independent/CENTER_LABELS.json')})
web=[]
for s in d['sequences']:
 r=idx[s['case_id']];it={k:v for k,v in s.items() if k!='edges'}
 it['edges']=[{'edge_index':x['edge_index'],'status':x['status'],'rgb_score':x['rgb']['score'],'pose_score':x['pose']['score']} for x in s['edges']]
 for i,x in enumerate(it['frames']):
  x['image']=Path(r['frames'][i]['focus_file']).as_uri();x['full']=Path(r['frames'][i]['full_file']).as_uri()
 web.append(it)
payload={'sequences':web,'trials':d['masked_trials'],'events':d['events']}
style=re.search(r'<style>(.*?)</style>',(P/'RESULT_KO.html').read_text(encoding='utf-8'),re.S).group(1)
style+=''' .triplet{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}.shot{background:#142a46;color:white;border-radius:10px;overflow:hidden}.shot img{display:block;width:100%;height:300px;object-fit:contain}.shot p{padding:8px 12px;margin:0;font-size:14px}.shot.current{outline:3px solid #3c79d6}.strip{display:grid;grid-template-columns:repeat(7,1fr);gap:8px}.strip img{width:100%;display:block}.strip button{padding:4px;font-size:11px}.edgebar{display:flex;gap:4px}.edge{flex:1;height:14px;border-radius:4px;background:#92b5a4}.edge.change{background:#e7b36c}.edge.missing{background:#bfc8d0}.loading{color:#775911;min-height:26px}.boundary{border-left:4px solid #e7b36c;padding:10px 14px;background:#fff8ee;margin:10px 0}.trioresult{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.trioresult .resultbox{margin:0}.statusnote{font-size:14px}button:disabled{opacity:.5;cursor:wait}.rowhead{font-weight:700}.footerlinks{display:flex;gap:18px;flex-wrap:wrap}@media(max-width:780px){.triplet{grid-template-columns:1fr 1fr 1fr;gap:5px}.shot img{height:190px}.shot p{font-size:11px;padding:5px}.trioresult{grid-template-columns:1fr}.strip{grid-template-columns:repeat(4,1fr)}}'''
def rows(summary):
 return ''.join(f'<tr><th>{name}</th><td>{summary[key]["draft_match"]}</td><td>{summary[key]["draft_conflict"]}</td><td>{summary[key]["withheld"]}</td></tr>' for key,name in [('baseline','앞뒤 상태 연결'),('guard','RGB·관절 변화 차단 추가')])
page='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손모양과 변화 구간을 함께 보존하기</title><style>__STYLE__</style></head><body><main>
<header><div class="eyebrow">WORDS IN SILENCE · 2026.10.27</div><h1>손모양은 보존하고,<br>바뀌는 과정은 따로 기록하기</h1><p class="lead">펼친 손과 굽힌 손을 확인한 순간은 그대로 둔다.<br>그 사이를 억지로 같은 모양으로 채우거나 영상에서 지우지 않는다.</p><span class="tag">실제 원본 프레임 확인</span><span class="tag">변화 구간 표시 구현</span><span class="tag">강제 차단 후보는 미채택</span></header>
<section><h2>이전·다음 버튼은 정말 한 프레임인가?</h2><p><b>맞다. 원본의 연속한 한 프레임, 약 0.033초씩 이동한다.</b> 13장 모아보기도 같은 이미지다. 432번 버튼 이동과 450개 표시 상태, 234개 모아보기 타일을 확인했다. ‘별’의 58→59번 프레임도 원본 영상과 픽셀이 일치했다.</p><p>기존 K번호는 순서를 숨기기 위한 임의 식별자였다. 새 화면은 <b>실제 프레임 번호와 시간</b>을 표시하며 이전·현재·다음을 함께 보여준다. 사진을 다 읽은 뒤 사진·번호·시간을 동시에 바꾸도록 개선했다.</p><p class="notice">저장된 30fps 영상 사이에 담기지 않은 순간은 알 수 없다. 기존 화면에서 로딩보다 설명이 먼저 바뀌는 상태는 관찰됐지만, 잘못된 이미지가 안정적으로 표시되거나 원본 순서가 뒤바뀐 문제는 발견하지 못했다.</p></section>
<section><h2>손모양 ‘상태’와 ‘변화’를 구분했다</h2><div class="cards"><div class="card"><h3>확인된 손모양</h3><p>현재 화면에서 직접 확인한 특징을 보존한다. 이번 특징은 ‘검지만 폄’, ‘네 손가락 폄’이다.</p></div><div class="card"><h3>상태가 달라진 구간</h3><p>앞에서 없던 특징이 뒤에 나타나거나 반대가 확인되면, 그 사이에 변화가 있었다고 기록한다. 정확히 어느 순간인지 억지로 정하지 않는다.</p></div><div class="card"><h3>관측이 부족한 구간</h3><p>앞뒤가 같아도 중간을 못 봤다면 같은 모양이었다고 확정하지 않는다. 영상과 시간을 남기고 추정 근거를 구분한다.</p></div></div><p class="callout">‘네 손가락 폄이 아님’은 <b>‘주먹’이라는 뜻이 아니다.</b> 중간 굽힘이나 다른 손모양도 포함될 수 있다. 중간 모양을 이름 붙이지 못해도 펼치는 순서·방향·움직임 기록은 보존한다. 그 변화가 수어 의미상 필수인지는 이번 시험에서 판정하지 않았다.</p></section>
<section id="viewer"><h2>원본 한 프레임씩, 나란히 확인</h2><div class="controls"><button data-example="T15,3,open_four,1">별 · 짧은 펼침</button><button data-example="T17,10,open_four,0">순종 · 점진적인 굽힘</button><button data-example="T11,6,open_four,1">감사 · 유지된 모양</button><button data-example="N07,6,index_only,1">새 구간 · 이동하는 검지</button></div>
<div class="controls"><select id="case" aria-label="영상 구간"></select><select id="feature" aria-label="특징"><option value="open_four">네 손가락 폄</option><option value="index_only">검지만 폄</option></select><label><input id="mask" type="checkbox">현재 특징값을 가린 시험 보기</label></div><div id="load" class="loading" aria-live="polite"></div>
<div class="triplet"><div class="shot"><img id="im0" alt="이전 원본 프레임"><p id="cap0"></p></div><div class="shot current"><img id="im1" alt="현재 원본 프레임"><p id="cap1"></p></div><div class="shot"><img id="im2" alt="다음 원본 프레임"><p id="cap2"></p></div></div>
<input id="time" class="range" type="range" min="0" max="12" value="3" aria-label="프레임"><div class="controls"><button id="prev">이전 1프레임</button><button id="next">다음 1프레임</button><button id="play">천천히 연속 보기</button><a id="full" target="_blank">전체 원본 화면</a></div>
<div id="outputs" class="trioresult"></div><p id="masknote" class="muted"></p><h3>현재 관측으로 확인한 특징</h3><div class="timelinewrap"><div id="timeline" class="timeline"></div></div><p class="muted">있음 / 아님 / ?는 손모양 특징의 관찰이다. 맞는 수어 / 틀린 수어 판정이 아니다.</p><h3>인접 프레임의 영상·관절 변화 신호</h3><div id="edges" class="edgebar"></div><div class="legend"><span><i class="dot" style="background:#92b5a4"></i>기준보다 큰 변화 신호 없음</span><span><i class="dot" style="background:#e7b36c"></i>변화·회전·관측 흔들림 가능</span><span><i class="dot gray"></i>추적 근거 부족</span></div><div id="events"></div><p class="notice">주황색은 잘못된 수어라는 뜻이 아니다. 평면상의 이동·회전을 일부 보정했어도 3차원 회전, 가림, 관절 검출 흔들림이 섞일 수 있다.</p><details><summary>같은 13프레임을 시간순으로 모두 보기</summary><div id="strip" class="strip"></div></details></section>
<section><h2>보완 후보를 다른 공개 구간에서도 시험했다</h2><p>지난 6표현을 제외한 수어·화상·마음에들다·잘하다·조상·이집트, 기록 ID 3개씩 총 <b>18개의 새 연속 구간</b>을 결과 전에 고정했다. 영상은 기존 공개 TRAIN 자료이며 새로운 사용자 인식률 시험은 아니다.</p><p>손 영역의 이미지 움직임에서 전체 이동·평면 회전·크기 변화를 보정하고 남은 변화량, 현재 관절의 2D 각도 변화를 계산했다. 이전의 안정 구간 2개에서 기준을 정한 후 바꾸지 않았다. 두 신호가 모두 안정적일 때만 시간 추정을 허용하는 후보를 비교했다.</p><div class="scroll"><table><thead><tr><th>새 348개 특징값 한 칸 가림 시험</th><th>원래 판단과 일치</th><th>충돌</th><th>보류</th></tr></thead><tbody>__NEW_ROWS__</tbody></table></div><p class="callout"><b>이번 차단 후보는 채택하지 않았다.</b> 새 구간에서 줄일 기존 충돌은 없었고, 원래 판단과 맞던 220개까지 보류했다. 현재 영상 신호를 더 보는 것 자체보다, 그 신호를 어떻게 해석하는지가 중요했다.</p><div class="scroll"><table><thead><tr><th>과거 진단 279개 시험</th><th>일치</th><th>충돌</th><th>보류</th></tr></thead><tbody>__OLD_ROWS__</tbody></table></div><p>과거 ‘별’의 짧은 펼침을 덮어쓰는 충돌 1개는 보류했지만, 기존 일치 102개도 보류했다. 이것을 전반적인 개선으로 보고하지 않는다.</p><p class="notice">현재 RGB에 AI가 붙인 특징값을 매번 한 칸씩 따로 가린 대조 시험이다. 수어 인식률·손가락 복원 정확도·교정 정확도가 아니다. 인접 프레임은 서로 연관된다. 결과를 보고 기준을 높이거나 단어 예외를 만들지 않았다.</p></section>
<section><h2>무엇을 남기고, 무엇을 적용하지 않았나?</h2><div class="twocol"><div><h3>구현·검증한 부분</h3><ul><li>원본 프레임과 시간, 세 장 비교, 로딩 완료 후 동시 표시</li><li>직접 확인한 모양과 모양 변화의 시간 범위 분리</li><li>변화와 관측 부족의 근거 기록</li><li>신규 직접 관측 411개와 기존 334개 보존</li></ul></div><div><h3>적용하지 않은 부분</h3><ul><li>변화 신호가 크면 무조건 시간 추정을 막는 규칙</li><li>관측 부족을 정답 또는 수어 오류로 바꾸는 판정</li><li>기존 C·T3-F·운영 웹·DTW 교체</li><li>검증되지 않은 추정값을 정답으로 삼는 학습</li></ul></div></div><p>실제 새 불확실 항목 57개에서 기존 방법의 추정은 7개, 차단 후보는 0개였다. 직접 관측한 현재 모습이 지워진 것은 아니지만, 놓친 부분을 되찾는 이득도 확인하지 못했다.</p></section>
<section><h2>다음 작업은 무엇이 달라야 하나?</h2><p>중간의 모든 프레임을 주먹·펼침 중 하나로 채우는 것을 목표로 두지 않는다. <b>확인된 모양과 그 사이의 변화 순서를 보존하면서, 필요한 핵심 상태를 놓치지 않는지</b>를 기준으로 삼는다.</p><ol><li>손모양이 유지되는 동안 손 전체가 이동하는 경우와, 실제 손가락이 펴지거나 접히는 경우를 분리한다. 영상 변화량이나 2D 관절각 하나를 변화의 정답으로 쓰지 않는다.</li><li>기존 공통 특징 묶음에 모양 유지·변화·관측 불가의 시간 범위와 근거를 연결한다. 이전 시간문맥 시험의 4개 진단 사례와 새 결과는 진단 이력으로 보존한다.</li><li>검증 가능한 특징 정답을 확보한 뒤, 현재 RGB와 앞뒤 흐름을 함께 쓰는 작은 특징 모델을 비교한다. 목표 단어를 정답 힌트로 넣지 않으며 네 촬영은 학습·튜닝·선택에 사용하지 않는다.</li></ol><p>중간 구간의 정확한 손모양을 이름 붙이지 못해도 영상과 순서는 남길 수 있다. 다만 특정 표현에서 그 구간을 무시해도 되는지는 별도의 언어적 근거가 필요하다.</p></section>
<section><h2>근거와 검증</h2><p>원본 234프레임·81입력핀 보존, 432인접구간의 저장 잔차·관절각 산술을 별도 검산했다. 현재 이미지 초안과 시간순·수치 검토의 담당을 나눴다. 모두 동일 AI 계열 검토이며 독립 사람의 정답 검증과 구분한다.</p><p>새 18중심 이미지의 36항목 별도 비교: __CENTER__. 같은 표현의 서로 다른 기록 ID를 사용했으며, 이미지 외형으로 사람 신원을 판정하지 않았다.</p><div class="footerlinks"><a href="frame_audit/FRAME_AUDIT.json">원본·버튼 감사</a><a href="COMPARISON.json">비교 결과</a><a href="METRICS_ROOT_VERIFY.json">점수 산술 검산</a><a href="review/GUARD_VERIFY.json">보완 규칙 별도 검산</a><a href="FROZEN.json">사전 조건</a><a href="RESULT_KO.txt">보고 원문</a></div><p class="notice">기존 모델·웹·DTW 보호 해시 32개 유지. 학습 0회, 사용자·멘토 자료 사용 0건. 화면의 손·팔 확대는 보기용이며 모델 입력 화면을 새로 자른 것이 아니다.</p></section></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=x=>document.getElementById(x);let seq,i=3,ticket=0,playing=false,busy=false;
const featureText={index_only:'검지만 폄',open_four:'네 손가락 폄'},valueText=v=>v===null?'확인 불가':v===1?'해당 특징 있음':'해당 특징 아님';
D.sequences.forEach(s=>{let o=document.createElement('option');o.value=s.case_id;o.textContent=(s.fresh_dense?'새 비교':'이전 진단')+' · '+s.case_id+' '+s.label+' · ID '+s.person_id;$('case').append(o)});
function stop(){playing=false;$('play').textContent='천천히 연속 보기'}
async function render(j){const mine=++ticket;busy=true;$('prev').disabled=$('next').disabled=true;$('load').textContent='원본 프레임을 불러오는 중…';const s=D.sequences.find(s=>s.case_id===$('case').value);j=Math.max(0,Math.min(12,j));const slots=[Math.max(0,j-1),j,Math.min(12,j+1)];const images=await Promise.all(slots.map(async n=>{let im=new Image();im.src=s.frames[n].image;await im.decode();return im}));if(mine!==ticket)return;seq=s;i=j;$('time').value=i;const f=$('feature').value,x=s.frames[i],trial=D.trials.find(t=>t.case_id===s.case_id&&t.index===i&&t.feature===f),masked=$('mask').checked&&!!trial;
 slots.forEach((n,k)=>{$('im'+k).src=images[k].src;$('cap'+k).textContent=(k===0?'이전':k===1?'현재':'다음')+' · 원본 '+s.frames[n].frame_id+'프레임 · '+(s.frames[n].pts_us/1e6).toFixed(3)+'초';$('im'+k).dataset.frameId=s.frames[n].frame_id});
 $('full').href=x.full;const b=masked?trial.baseline:x.baseline[f],g=masked?trial.guard:x.guard[f];const msg=o=>o.kind==='direct'?'현재 이미지의 관측을 유지':o.value!==null?'시간 흐름으로 추정':o.reason==='change_or_viewpoint'?'변화·회전·관측 흔들림 신호로 보류':o.reason==='insufficient_transition_evidence'?'영상 또는 관절 근거가 부족해서 보류':'앞뒤의 충분하고 일치하는 근거가 없음';
 $('outputs').innerHTML='<div class="resultbox direct"><small>원래 이미지 판단 · '+featureText[f]+'</small><strong>'+valueText(x.direct[f])+'</strong><small>현재 모양에 대한 AI 관찰 초안</small></div>'+[['앞뒤 상태 연결',b],['변화 차단 후보 · 미채택',g]].map(([n,o])=>{const conflict=masked&&o.value!==null&&o.value!==trial.hidden_draft;return '<div class="resultbox '+(conflict?'conflict':o.kind==='direct'?'direct':o.value!==null?'estimate':'')+'"><small>'+n+'</small><strong>'+valueText(o.value)+(conflict?' · 원래 판단과 충돌':'')+'</strong><small>'+msg(o)+'</small></div>'}).join('');
 $('masknote').textContent=masked?'원래 이미지 판단 한 칸만 가리고 시험 중. RGB 이미지는 보완 기능의 입력으로 남아 있다.':$('mask').checked?'원래 보류 상태 또는 끝점이므로 특징값 한 칸 가림 시험 대상이 아니다. 실제 입력 결과 표시 중.':'실제 입력 결과. 직접 관측된 모양은 두 방법 모두 보존한다.';
 $('timeline').innerHTML=s.frames.map((t,n)=>'<button class="cell '+(t.direct[f]===null?'':'v'+t.direct[f])+(n===i?' on':'')+'" data-index="'+n+'">'+(t.direct[f]===null?'?':t.direct[f]===1?'있음':'아님')+'<em>f'+t.frame_id+'</em></button>').join('');
 $('edges').innerHTML=s.edges.map(e=>'<span class="edge '+(e.status==='insufficient_evidence'?'missing':e.status==='no_change_signal'?'':'change')+'" title="프레임 '+s.frames[e.edge_index].frame_id+' → '+s.frames[e.edge_index+1].frame_id+'"></span>').join('');
 const events=D.events.filter(e=>e.case_id===s.case_id&&e.feature===f);$('events').innerHTML=events.length?events.map(e=>'<div class="boundary">'+(e.from_pts_us/1e6).toFixed(3)+'~'+(e.to_pts_us/1e6).toFixed(3)+'초: '+(e.kind==='observed_state_change_interval'?'확인된 특징 상태가 달라진 구간':'앞뒤 상태는 같지만 중간 모양은 미확인')+'<small> · 정확한 중간 모양과 언어적 중요도는 미확정</small></div>').join(''):'<p class="muted">이 짧은 구간의 확인된 특징에서 서로 다른 끝 상태가 기록되지 않았다. 보이지 않은 순간까지 같았다는 뜻은 아니다.</p>';
 $('strip').innerHTML=s.frames.map((t,n)=>'<button data-index="'+n+'"><img loading="lazy" src="'+t.image+'" alt="원본 '+t.frame_id+'프레임">f'+t.frame_id+' · '+(t.pts_us/1e6).toFixed(3)+'초</button>').join('');
 document.querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>{stop();render(+b.dataset.index)});$('load').textContent='원본 한 프레임 간격 약 0.033초 · 사진과 시간 표시 동기화 완료';$('prev').disabled=i===0;$('next').disabled=i===12;busy=false;history.replaceState(null,'','#'+s.case_id);window.renderedState={caseId:s.case_id,index:i,frameId:x.frame_id,ptsUs:x.pts_us,image:x.image,done:true};}
 $('prev').onclick=()=>{stop();render(i-1)};$('next').onclick=()=>{stop();render(i+1)};$('time').oninput=()=>{stop();render(+$('time').value)};['case','feature','mask'].forEach(id=>$(id).onchange=()=>{stop();render(id==='case'?6:i)});
 $('play').onclick=async()=>{if(playing){stop();return}playing=true;$('play').textContent='정지';while(playing){await render(i===12?0:i+1);await new Promise(r=>setTimeout(r,180));}};
 document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{stop();let[c,j,f,m]=b.dataset.example.split(',');$('case').value=c;$('feature').value=f;$('mask').checked=m==='1';render(+j)});
 const requested=location.hash.slice(1);$('case').value=D.sequences.some(s=>s.case_id===requested)?requested:'T17';$('mask').checked=false;render($('case').value==='T17'?8:6).catch(e=>{$('load').textContent='이미지 로드 실패: '+e.message});
</script></body></html>'''
centertext='확정 일치 '+str(agree['known_agree'])+'개, 양쪽 보류 '+str(agree['both_unknown'])+'개, 한쪽 보류 '+str(agree['one_unknown'])+'개, 서로 다른 확정 '+str(agree['known_conflict'])+'개'
page=page.replace('__STYLE__',style).replace('__NEW_ROWS__',rows(d['fresh_summary'])).replace('__OLD_ROWS__',rows(d['old_diagnostic_summary'])).replace('__CENTER__',centertext).replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
(U/'RESULT_KO.html').write_text(page,encoding='utf-8')
text='''손모양 유지·변화 구간 보존 및 원본 프레임 감사 — 2026-10-27

사용자 질문 답변
이전/다음은 실제연속원본1프레임(33333/33334us)이다. 432버튼클릭/450완료상태·234contact타일·T15/T17원본26PNG대조일치. K식별자는시간순이아니다. 13장모아보기와같은이미지다. 원본30fps사이의추가실제장면/촬영전누락은확인불가. 기존로드중설명먼저변경확인,잘못된고정이미지순서오류는없음. 새뷰어에서프레임/시간/이미지를decode후동시반영,이전/현재/다음과시간순13장·느린연속보기제공.
주먹→펼침양끝모양을보존하면서중간을변화구간으로구분하는방향은반영. 중간을한상태로강제채우거나삭제하지않는다. 변화의언어적중요성은자동으로무시하지않는다. 이번open_four=0은주먹정답이아니며여러다른모양을포함한다.

입력과비교
새6표현(수어·화상·마음에들다·잘하다·조상·이집트)×3기록ID=18고정구간,각13원본프레임총234. 2특징468칸셔플RGB판정411known/57unknown. 이전18구간은진단전용. 전체원본관측을읽고표시crop은보기용. 사용자/멘토자료사용0.
고정과거C시간규칙(운영TCN C와별개)에현재RGB/2D관절변화차단추가. 전체손이동/평면회전/scale보정후LK잔차90백분위/손바닥폭, 12관절각변화90백분위를계산. 옛T07/T10의기록ID01안정24연결에서최대값고정. RGB0.1079912488458561 손바닥폭비,2D각3.034099536097995도. 평가후기준변경없음.

결과
새348개독립단일특징마스킹: baseline259일치0충돌89보류→guard39일치0충돌309보류. 기존일치220개보류. calibration과다른기록ID02/03만224개:160일치0충돌64보류→17일치0충돌207보류. 과거279개진단:194일치1충돌84보류→92일치0충돌187보류. 지난1충돌을막았으나기존일치102도보류. 실제새unknown57에서기존7추정→후보0. 추정정답없어실제복원정확도는말할수없음.
새216인접구간:42큰변화신호없음/124변화또는시점불확실신호/50근거부족. 둘다유효166중각도초과117. 정상움직임전체의허용범위가아닌두안정구간기준을all-edge에적용해과도보류됨. 영상변화는3D회전/가림,2D각변화는투영/검출흔들림도가능. 새오답감소입증0. 하드차단규칙미채택. 원래direct411+334보존은설계보존검사이며정답증명아님.

반영한부분
확인된특징상태와관측상태변화의시간범위/같은끝상태사이unknown구간을별도기록(20구간). 모든원본시간/관측값보존. 변화/근거부족을수어정오로바꾸지않음. 연구뷰어원본프레임표시보완. 운영TCN/DTW/웹은변경안함.

다음한작업
모든중간프레임을확정복원하는대신기존공통특징의유지·변화·관측불가와실제모양근거를연결하고, 핵심손모양상태와순서가남는지평가한다. 정상손전체이동·회전과손가락변화를구분할검증가능한특징정답이필요. 같은AI초안/관절값/변화량을정답으로승격하거나새계수탐색으로과보류를숨기지않음. 공개사람/표현분리유지. 신뢰정답확보후작은RGB+시간특징학습비교. 새검출기추가/현재TCN교체/사용자자료튜닝금지.

검증
root: 동결/guard집계/원시432edge수학검산/보고UI.
local_feature_inventory: 실제프레임·버튼·픽셀감사,18중심이미지별도판정.
linguistic_eval_review: 원본추출·RGB/pose metric구현·별도scalar산술검산.
ksl_source_audit: 234셔플이미지+focus23장판정,guard독립구현검산.
모두같은AI계열이며독립사람정답아님. 351RGB잔차·415pose점수·4962각쌍재산술검산PASS. 기존모델/운영32핀불변. 학습0/새검출0/운영교체0/원본수정0/개인자료사용0.
'''
(U/'RESULT_KO.txt').write_text(text,encoding='utf-8')
dump(U/'DECISION.json',{'status':'COMPLETE_COMMON_TRANSITION_REPRESENTATION_AND_CANDIDATE_EVALUATION',
 'hard_guard_adopted':False,'why':'220 previously matching masked feature trials withheld on fresh windows; no fresh conflicts corrected.',
 'implemented':['observed state preservation','bounded observed state change events','unresolved interval provenance','synchronized actual-frame viewer'],
 'training':False,'production_changed':False,'center_agreement':dict(agree),'event_counts':dict(Counter(x['kind'] for x in d['events']))})
print(json.dumps({'html_bytes':(U/'RESULT_KO.html').stat().st_size,'center':dict(agree),'events':len(d['events'])},ensure_ascii=False))
