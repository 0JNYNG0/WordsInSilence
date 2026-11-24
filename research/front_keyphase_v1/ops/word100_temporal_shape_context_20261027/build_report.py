from pathlib import Path
import json, hashlib, html
from collections import Counter
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

d=read(U/'COMPARISON.json'); idx=read(U/'visual_inputs/INDEX.json')
seqrev=read(U/'sequence_review/SEQUENCE_REVIEW.json')
secondary=read(U/'independent/SECOND_IMAGE_LABELS.json')
primary=read(U/'review/IMAGE_LABELS.json')
lut={(r['case_id'],r['image_key']):r for r in primary['rows']}
second={(r['case_id'],r['image_key']):r for r in secondary['rows']}
ix={r['case_id']:r for r in idx['rows']}
counter=Counter(); confirmed=[]
for key,r in second.items():
    for f in ('index_only','open_four'):
        a,b=lut[key][f],r[f]
        counter['both_unknown' if a is None and b is None else 'one_unknown' if a is None or b is None else 'known_agree' if a==b else 'known_conflict']+=1
for r in d['masked_trials']:
    conflicts=[m for m in ('B','C') if r['outputs'][m]['value'] is not None and r['outputs'][m]['value']!=r['hidden_draft']]
    if conflicts:
        key=(r['case_id'],ix[r['case_id']]['frames'][r['index']]['image_key'])
        value=second[key][r['feature']]
        confirmed.append({'case_id':r['case_id'],'image_key':key[1],'feature':r['feature'],
           'condition_conflicts':conflicts,'original_draft':r['hidden_draft'],
           'second_image_draft':value,'two_drafts_agree':value==r['hidden_draft']})
protected=read(U.parent/'word100_face_relation_alignment_20261027/FROZEN.json')['protected']
assert all(sha(p)==h for p,h in protected.items())
assert all(sha(p)==h for p,h in d['pins'].items())
assert all(not r['identity_concern'] for r in seqrev['rows'])
extra={'status':'PASS_WITH_UNRESOLVED_REAL_SHAPES','secondary_field_agreement':dict(counter),
       'masked_conflict_rechecks':confirmed,'operating_pins_preserved':len(protected),
       'comparison_pins_preserved':len(d['pins']),
       'secondary_labels_sha256':sha(U/'independent/SECOND_IMAGE_LABELS.json'),
       'sequence_review_sha256':sha(U/'sequence_review/SEQUENCE_REVIEW.json'),
       'identity_concerns_in_selected_rgb_windows':0,
       'not_independent_human_truth':True,'training':False,'web_model_replaced':False}
dump(U/'ADDITIONAL_REVIEW.json',extra)

notes={r['case_id']:r for r in seqrev['rows']}
web=[]
for s in d['sequences']:
    r=ix[s['case_id']]
    item={**s,'observation':notes[s['case_id']]['observation_ko'],'review_note':notes[s['case_id']]['notes']}
    item['contact']='visual_inputs/'+s['case_id']+'/ordered_contact.png'
    for j,f in enumerate(item['frames']):
        f['image']='visual_inputs/'+s['case_id']+'/'+r['frames'][j]['image_key']+'_focus.png'
        f['full_image']='visual_inputs/'+s['case_id']+'/'+r['frames'][j]['image_key']+'_full.png'
    web.append(item)
payload={'sequences':web,'masked_trials':d['masked_trials']}
tables=''
names={'A':'① 현재 화면만','B':'② 이전 흐름 추가','C':'③ 앞뒤 흐름 추가'}
for m in 'ABC':
    r=d['masked_summary'][m]
    bar=''.join(f'<i class="{cl}" style="width:{r[k]/r["total"]*100}%"></i>' for k,cl in [('draft_match','green'),('draft_conflict','red'),('withheld','gray')])
    tables+=f'<tr><th>{names[m]}</th><td>{r["draft_match"]}</td><td>{r["draft_conflict"]}</td><td>{r["withheld"]}</td><td><div class="bar">{bar}</div></td></tr>'
htmltext='''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>손모양을 앞뒤 흐름으로 확인하기</title><style>
:root{--ink:#182738;--muted:#53677a;--line:#dce5eb;--green:#218672;--blue:#3c79d6;--red:#c95950;--bg:#f3f6f8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 'Malgun Gothic',sans-serif}main{max-width:1180px;margin:0 auto;padding:36px 24px 70px}header{padding:30px 0}h1{font-size:38px;line-height:1.3;letter-spacing:-1.3px;margin:12px 0}h2{font-size:25px;margin:0 0 16px}h3{font-size:19px;margin:0 0 10px}p{margin:10px 0}.eyebrow{font-size:13px;letter-spacing:2px;color:var(--green);font-weight:700}.lead{font-size:21px;max-width:900px}.muted,small{color:var(--muted)}section{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin:22px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card{border:1px solid var(--line);border-radius:12px;padding:18px}.big{font-size:32px;font-weight:750;line-height:1.35}.tag{font-size:13px;display:inline-block;border-radius:30px;padding:3px 10px;background:#e7f4ef;color:#146353;margin-right:6px}.blue{background:var(--blue)}.green{background:var(--green)}.red{background:var(--red)}.gray{background:#b9c4ce}.callout{border-left:4px solid var(--blue);background:#edf4ff;padding:14px 18px;border-radius:4px}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px;border-bottom:1px solid var(--line)}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}.bar{height:14px;display:flex;min-width:160px;border-radius:7px;overflow:hidden}i{display:inline-block;height:100%}.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:14px;margin-top:14px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:16px 0}button,select{font:inherit;border:1px solid #c6d3df;border-radius:8px;padding:8px 12px;background:white;color:var(--ink);cursor:pointer}button:hover{background:#edf4ff}select{max-width:100%}.viewer{display:grid;grid-template-columns:1fr 1fr;gap:24px}.imagebox{background:#0d223e;border-radius:12px;overflow:hidden}.imagebox img{width:100%;height:355px;object-fit:contain;display:block}.imagebox p{color:white;padding:8px 16px;margin:0;font-size:14px}.range{width:100%;accent-color:var(--blue)}.resultbox{padding:12px 16px;margin:9px 0;border-radius:10px;background:#f3f6f9;border:1px solid var(--line)}.resultbox strong{display:block;font-size:20px}.resultbox small{display:block;margin-top:4px}.resultbox.estimate{border-color:#79a5eb;background:#eff5ff}.resultbox.direct{border-color:#7fb6a9;background:#edf7f2}.resultbox.conflict{border-color:#d17b72;background:#fff0ee}.timeline{display:grid;grid-template-columns:repeat(13,minmax(32px,1fr));gap:4px;margin:12px 0;min-width:480px}.cell{padding:7px 0;text-align:center;font-size:14px;border:2px solid transparent;border-radius:7px;background:#e9eef2;cursor:pointer}.cell.on{border-color:var(--ink)}.cell.v1{background:#d7eee3}.cell.v0{background:#dde6f4}.cell em{display:block;font-size:10px;font-style:normal}.timelinewrap{overflow-x:auto}.hidden{display:none}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:18px}ol{padding-left:22px}li{padding:6px 0}a{color:#245cad;text-underline-offset:3px}details{margin:14px 0}summary{cursor:pointer;font-weight:700}.contact{width:100%;margin-top:14px}.notice{font-size:14px;border-top:1px solid var(--line);padding-top:12px;margin-top:16px}.status{font-weight:700;color:#254d75}footer{font-size:14px;color:var(--muted)}@media(max-width:780px){main{padding:16px 14px 50px}h1{font-size:30px}section{padding:20px 16px}.cards,.viewer,.twocol{grid-template-columns:1fr}.imagebox img{height:300px}.lead{font-size:18px}th,td{padding:9px}.bar{min-width:100px}}
</style></head><body><main>
<header><div class="eyebrow">WORDS IN SILENCE · 2026.10.27</div><h1>손모양을 한 장이 아닌<br>앞뒤 흐름으로 확인하기</h1><p class="lead">앞뒤 정보는 도움이 된다. 다만 <b>중간의 짧은 변화까지 지워서는 안 된다.</b><br>실제 공개 영상 18구간에서 세 가지 방법을 비교했다.</p><span class="tag">공개 영상만 사용</span><span class="tag">기존 C · T3-F 보존</span><span class="tag">연구용 비교 완료</span></header>
<section><h2>이번에 확인한 것</h2><div class="cards"><div class="card"><div class="big">18구간</div><p>6표현 × 기록 ID 3개<br>겹침 · 옆면 · 손내림 · 펼침 · 변화 구간</p></div><div class="card"><div class="big">234프레임</div><p>각 구간 원본 13프레임<br>약 0.033초 간격 / 약 0.4초 구간</p></div><div class="card"><div class="big">2가지 특징</div><p>검지만 편 상태인가?<br>엄지를 제외한 네 손가락을 폈는가?</p></div></div><p class="notice">시간순을 섞은 이미지에서 각 상태를 먼저 기록한 뒤 원래 시간으로 연결했다. 자동 손모양 인식기를 학습한 시험이 아니라, <b>RGB 관찰 초안에 시간 보완 규칙을 적용한 시험</b>이다. AI 관찰 초안은 독립 사람 정답이 아니다.</p></section>
<section><h2>세 가지 방법</h2><div class="cards"><div class="card"><h3>① 현재 화면만</h3><p>현재 손모양이 확인되면 기록하고, 안 보이면 확인 불가로 남긴다.</p></div><div class="card"><h3>② 이전 흐름 추가</h3><p>직전 0.2초 안에서 최소 두 번 확인되고, 그 범위의 모든 확인된 상태가 같으면 현재를 그 상태로 <b>추정</b>한다.</p></div><div class="card"><h3>③ 앞뒤 흐름 추가</h3><p>이전과 이후 각각 0.2초 안에서 최소 두 번씩 같은 상태가 확인되어야 추정한다. 범위 안에 반대 상태가 하나라도 보이면 보류한다.</p></div></div><p>현재 직접 확인된 모양은 세 방법 모두 그대로 둔다. 추정으로 채운 값을 다시 다른 추정의 근거로 사용하지 않는다.</p><p class="muted">여기서 ③은 이번 시간 비교 조건이다. 기존 TCN의 ‘C 후보 모델’과는 다른 이름이다.</p></section>
<section><h2>원래 보였던 순간을 가렸을 때</h2><p>원래 판단할 수 있었던 중간 특징값 279개를 <b>한 번에 하나씩</b> 가렸다. 앞뒤 정보가 그 순간의 원래 이미지 판단을 되찾는지 확인했다.</p><div class="scroll"><table><thead><tr><th>방법</th><th>원래 판단과 일치</th><th>충돌</th><th>보류</th><th>전체 279개</th></tr></thead><tbody>__TABLE__</tbody></table></div><div class="legend"><span><i class="dot green"></i>일치</span><span><i class="dot red"></i>충돌</span><span><i class="dot gray"></i>보류</span></div><p class="callout">앞뒤를 모두 본 방법은 이전만 본 방법보다 충돌이 4개에서 1개로 줄었다. 하지만 보류는 38개에서 84개로 늘었다. <b>더 조심스럽게 판단했으며, 완벽히 복원한 것은 아니다.</b></p><p class="notice">이는 수어 인식률이나 실제 가려진 손의 복원 정확도가 아니다. 279개 중 해당 모양이 ‘있음’은 64개, ‘아님’은 215개로 불균형하다. 인접 프레임은 서로 비슷하므로 279명을 시험한 것과도 다르다. 충돌한 실제 이미지 4개는 다른 검토자도 별도로 보고 원래 모양 판단에 동의했다.</p></section>
<section id="viewer"><h2>직접 확인: 어느 순간을 어떻게 채웠나?</h2><p>먼저 ‘짧은 변화가 사라진 사례’를 확인해 보자. 손가락을 잠깐 편 순간을 가리면, 앞뒤의 접힌 상태로 채울 수 있다.</p><div class="controls"><button data-preset="T15,3,open_four,1">짧은 변화가 사라진 사례</button><button data-preset="T10,6,open_four,1">안정된 모양을 되찾은 사례</button><button data-preset="T01,6,index_only,0">앞뒤도 가려져 있는 사례</button><button data-preset="T17,10,open_four,1">변화 앞뒤가 달라 보류한 사례</button></div><div class="controls"><select id="case" aria-label="공개 구간"></select><select id="feature" aria-label="손모양 특징"><option value="open_four">네 손가락을 편 상태</option><option value="index_only">검지만 편 상태</option></select><label><input type="checkbox" id="mask"> 현재 특징값을 가리고 비교</label></div><div class="viewer"><div><div class="imagebox"><img id="frame" alt="선택한 공개 수어 원본 프레임의 손과 팔"><p id="framecap"></p></div><input class="range" id="time" type="range" min="0" max="12" value="3" aria-label="원본 프레임 시간"><div class="controls"><button id="prev">이전 프레임</button><button id="next">다음 프레임</button><a id="full" target="_blank">전체 원본 프레임 보기</a></div></div><div><p id="truth" class="callout"></p><div id="outputs"></div><p id="masknote" class="muted"></p></div></div><div class="timelinewrap"><div id="timeline" class="timeline"></div></div><p class="muted">아래 시간표는 원래 이미지 판단: ‘있음 / 아님 / ?’. ‘아님’은 그 손모양 특징이 없다는 뜻이며, 수어를 틀렸다는 뜻이 아니다.</p><p id="seqnote"></p><details><summary>13프레임을 시간순으로 한 번에 보기</summary><img id="contact" class="contact" loading="lazy" alt="13개 원본 프레임 시간순 비교"></details></section>
<section><h2>처음부터 모호했던 장면은?</h2><div class="twocol"><div><h3>실제 불확실한 중심 특징 11개</h3><p>이전 흐름만 사용하면 1개를 추정했다. 앞뒤 근거를 함께 요구하면 새로 확정할 수 있는 항목은 0개였다.</p><p>전체 468항목 중 불확실했던 134항목도, 이전만 사용하면 12개를 추정했지만 앞뒤 방식은 0개였다.</p></div><div><h3>왜 앞뒤를 봐도 그대로인가?</h3><p>‘믿다’의 손 겹침과 ‘실명’의 옆면은 주변 프레임에서도 계속 모호했다. <b>앞뒤에 새로운 시각 정보가 없다면, 프레임 수가 늘어도 정확한 손가락 상태를 확인할 수 없다.</b></p><p>따라서 이번 결과로 원래의 측면 손모양 문제가 해결됐다고 보고하지 않는다.</p></div></div><p class="notice">중심 이미지 36항목을 별도 검토자도 확인: 양쪽 확정 일치 22개, 양쪽 보류 11개, 한쪽만 확정 3개. 원래 라벨은 바꾸지 않고 이 차이를 보존했다.</p></section>
<section><h2>정상 동작을 해치지 않았나?</h2><ul><li>현재 화면으로 기록한 334개 특징값은 덮어쓰지 않았다. 이는 코드의 보존 규칙 검증이며 관찰 자체가 항상 옳다는 증명은 아니다.</li><li>시간순 시각 검토에서 모양 변화가 보이는 7구간, 뚜렷한 변화가 없는 6구간, 가림·회전으로 판단이 제한되는 5구간을 구분했다.</li><li>선택한 18구간에서 팔과 손의 연결이 다른 손으로 갑자기 이어지는 징후는 보이지 않았다. 자동 추적 오류율 검증은 별도다.</li><li>내려놓은 손과 교차한 손을 수어 오류로 판정하거나 학습 자료에서 제거하지 않았다.</li></ul></section>
<section><h2>사용자의 아이디어를 어떻게 반영할까?</h2><p><b>“현재 한 장이 애매하면 앞뒤 흐름을 참고한다”는 방향은 타당하다.</b> 다만 뜻을 알아보는 것과, 그 순간 손가락이 정확히 어떤 모양인지 증명하는 것은 다르다.</p><p>이번 ‘별’ 구간처럼 짧게 펼쳤다가 다시 접는 모습이 실제 이미지에 있다. 이것이 1~2프레임 안에 완전히 굽혔다 펴는 일이 항상 가능하다는 뜻은 아니다. <b>‘짧으니 변하지 않았을 것’이라는 가정만으로 중간을 지우면 안 된다는 근거</b>다.</p><ol><li><b>현재 이미지와 관절 근거를 우선한다.</b> 손모양 변화가 직접 보이면 앞뒤의 다수 상태로 덮어쓰지 않는다.</li><li><b>앞뒤의 같은 손이 유지됐다는 근거를 추가한다.</b> 방향 전환, 손 겹침, 손가락 벌어짐·굽힘 변화가 의심되면 단순 연결을 중단한다.</li><li><b>관측과 추정을 함께 저장한다.</b> 직접 보임 / 시간 흐름으로 추정 / 판단 어려움을 분리한다. 추정만으로 정밀한 교정 지시를 내리지 않는다.</li><li><b>다음 비교는 현재 RGB의 ‘변화 신호’를 함께 사용한다.</b> 카메라·손 전체의 이동과 손가락 모양 변화를 구분하는 작은 검사부터 한다. 이번 충돌 4개는 진단 자료로 남기고, 성능 확인은 별도로 고정한 공개 구간에서 한다.</li></ol><p>이번 결과는 확인된 상태를 앞뒤로 연결하는 고정 규칙의 한계다. 현재 RGB의 변화까지 함께 배우는 시간 모델 전체가 효과 없다는 뜻은 아니다.</p><p class="notice">관측 모델의 학습은 검증된 손모양 정답이 확보되어야 진행할 수 있다. 이번 AI 초안과 시간 추정을 그대로 정답으로 재학습시키지 않았다. 기존 모델의 인식률 향상·실시간 웹 적용·교육적 허용 기준 검증은 이번 결과에 포함되지 않는다.</p></section>
<section><h2>검증 기록</h2><p>구현·집계: Codex root. 원본 추출·시각 연속성 검토, 순서를 섞은 이미지 판정, 중심·추가 이미지 판정, 별도 산술 검산을 역할별로 나눴다. 모두 같은 AI 계열이며 독립 사람 정답 검증과 구분한다.</p><div class="controls"><a href="COMPARISON.json">전체 계산 결과</a><a href="ADDITIONAL_REVIEW.json">추가 이미지 확인</a><a href="review/COMPARISON_VERIFY.json">별도 코드 검산</a><a href="FROZEN.json">사전 고정 조건</a><a href="RESULT_KO.txt">보고서 원문</a></div><p>기존 C·T3-F·운영 웹·DTW 보호 해시 32개 불변. 새 학습 0회, 사용자·멘토 자료 사용 0건.</p><p class="muted">관련 연구: <a href="https://openaccess.thecvf.com/content/ICCV2023/html/Fu_Deformer_Dynamic_Fusion_Transformer_for_Robust_Hand_Pose_Estimation_ICCV_2023_paper.html">Deformer, ICCV 2023</a>는 가림·흐림 앞뒤의 영상 정보를 이용한다. 시간 정보를 활용할 연구 근거이며, 그 논문 성능이 이 프로젝트에서 재현됐다는 뜻은 아니다.</p></section>
<footer>18개 짧은 공개 구간의 연구 결과 · 전체 수어 일반화나 사용자 피드백 정확도 확정 아님</footer></main>
<script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const modes=['A','B','C'], names={A:'① 현재 화면만',B:'② 이전 흐름 추가',C:'③ 앞뒤 흐름 추가'};
const reason={no_current_evidence:'현재 모양이 확인되지 않음',past_insufficient:'이전의 확실한 관측이 부족함',past_conflict:'이전 구간에서 모양이 달라짐',future_insufficient:'이후의 확실한 관측이 부족함',direction_conflict:'앞뒤 모양이 다르거나 변화가 있음',past_agreement:'이전 모양이 같아서 추정',bidirectional_agreement:'앞뒤 모양이 같아서 추정',feature_conflict:'다른 손모양 근거와 충돌',identity_uncertain:'같은 손인지 불확실'};
const val=x=>x===null?'확인 불가':x===1?'이 모양이 보임':'이 모양은 아님';
D.sequences.forEach(s=>{let o=document.createElement('option');o.value=s.case_id;o.textContent=s.case_id+' · '+s.label+' · 기록 ID '+s.person_id+' · '+(s.side==='R'?'오른손':'왼손');$('case').append(o)});
function render(){let s=D.sequences.find(x=>x.case_id===$('case').value),i=+$('time').value,f=$('feature').value,x=s.frames[i],trial=D.masked_trials.find(t=>t.case_id===s.case_id&&t.index===i&&t.feature===f),masked=$('mask').checked&&!!trial;
$('frame').src=x.image;$('full').href=x.full_image;$('contact').src=s.contact;$('framecap').textContent='원본 '+x.frame_id+'프레임 · '+(x.pts_us/1e6).toFixed(3)+'초 · 중심 대비 '+(x.offset_frames>=0?'+':'')+x.offset_frames+'프레임';
$('truth').textContent='현재 이미지를 직접 본 원래 판단: '+val(x.direct[f])+(masked?' — 이 값만 가린 상태로 비교 중':'');
$('masknote').textContent=masked?'다른 프레임의 원래 판단만 사용한다. 현재 숨긴 판단은 채점에만 쓴다.':$('mask').checked?'원래 확인 불가이거나 끝점이라 인위적 가림 시험 대상이 아니다. 실제 입력 결과를 표시한다.':'실제 관측과 그 주변 흐름으로 계산한 결과다.';
$('outputs').innerHTML=modes.map(m=>{let o=masked?trial.outputs[m]:x.outputs[m][f],conflict=masked&&o.value!==null&&o.value!==trial.hidden_draft;return '<div class="resultbox '+(conflict?'conflict':o.kind==='direct'?'direct':o.kind==='temporal_estimate'?'estimate':'')+'"><small>'+names[m]+'</small><strong>'+val(o.value)+(conflict?' · 원래 판단과 충돌':'')+'</strong><small>'+(o.kind==='direct'?'현재 화면에서 직접 관찰한 값 유지':reason[o.reason]||o.reason||'')+'</small><small>사용한 시점: '+o.support_indices.map(n=>(s.frames[n].pts_us/1e6).toFixed(3)+'초').join(', ')+'</small></div>'}).join('');
$('timeline').innerHTML=s.frames.map((t,n)=>'<button class="cell '+(t.direct[f]===null?'':'v'+t.direct[f])+(i===n?' on':'')+'" data-index="'+n+'">'+(t.direct[f]===null?'?':t.direct[f]===1?'있음':'아님')+'<em>'+((t.pts_us-s.frames[6].pts_us)/1000).toFixed(0)+'ms</em></button>').join('');
$('seqnote').textContent=s.observation+' '+s.review_note;document.querySelectorAll('[data-index]').forEach(b=>b.onclick=()=>{$('time').value=b.dataset.index;render()});history.replaceState(null,'','#'+s.case_id);}
['case','feature','mask'].forEach(id=>$(id).addEventListener('change',render));$('time').addEventListener('input',render);$('prev').onclick=()=>{$('time').value=Math.max(0,+$('time').value-1);render()};$('next').onclick=()=>{$('time').value=Math.min(12,+$('time').value+1);render()};
document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{let[c,i,f,m]=b.dataset.preset.split(',');$('case').value=c;$('time').value=i;$('feature').value=f;$('mask').checked=m==='1';render()});
const requested=location.hash.slice(1);$('case').value=D.sequences.some(s=>s.case_id===requested)?requested:'T15';$('time').value=$('case').value==='T15'?3:6;$('mask').checked=$('case').value==='T15';render();
</script></body></html>'''
htmltext=htmltext.replace('__TABLE__',tables).replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/'))
(U/'RESULT_KO.html').write_text(htmltext,encoding='utf-8')
txt='''앞뒤 시간 흐름으로 손모양 근거를 보완한 공개 시험 — 2026-10-27

결론
이번 인위적 마스킹 시험에서 앞뒤 정보가 안정된 모양을 되찾는 경우를 확인했다. 그러나 단순히 앞뒤 모양이 같다는 이유로 중간도 같다고 확정하면 짧은 변화를 덮어버릴 수 있다. 원래 측면/가림 때문에 모호했던 중심 11항목은 앞뒤 방식에서 새로 해결된 항목이 없었다. TCN 인식률이나 DTW 피드백 성능 개선으로 보고하지 않는다.

이번 고정 상태 일치 규칙의 한계이며, RGB 변화까지 배우는 시간 모델 전체가 불가능하다는 뜻이 아니다.

수행
6표현×기록ID3개=18개 공개영상에서 동결된 0.4초 구간 각13프레임, 총234원본RGB를 추출했다. 검지단독폄/네손가락폄 2항목 총468칸을 순서가 섞인 이미지로 먼저 판정했다. 현재이미지만(A), 이전200ms(B), 앞뒤각200ms(C)의 공통규칙을 적용했다. 알려진현재관측은 보존하고 최소2개앵커/방향, 충돌보류, 추정재사용금지를 적용했다. 기존TCN C모델과 시간조건C는 별개다.

결과
직접관찰초안334칸/보류134칸. 중심36칸은직접25/보류11; B새추정1/C새추정0. 전체보류134중B새추정12/C새추정0. 실제unknown에는정답이없으므로복원정확도주장불가.
알려진중간항목279개를한개씩독립적으로가린시험: A일치0/충돌0/보류279, B일치237/충돌4/보류38, C일치194/충돌1/보류84. B/C각각추정241/195. 원래모양있음64/아님215로불균형. 같은영상인접항목들이어서 독립표본279개 아님.
충돌4실제이미지는별도검토자도원래초안과동의했다. '별'의짧은펼침을가리면앞뒤접힌상태로채워지는C실패가있다. 원래불확실한주변프레임도포함된시험이며1프레임동안완전굽힘/폄가능성을측정한시험아님.
원래확정334칸은모두그대로보존(설계불변검사). 시간순영상검토: 변화보임7/뚜렷변화없음6/가림회전판단제한5, 갑작스러운손연결문제관찰0/18. 자동추적정확도보증은아님. 손내림/교차를오류로판정하거나제거하지않음.
별도중심36칸검토: 양쪽확정일치22/양쪽unknown11/한쪽unknown3. 추가14이미지28항목검토는ADDITIONAL_REVIEW참조. 같은AI계열초안이며독립사람정답이아님.

다음단계
현재RGB/관절에손가락변화신호가있는지확인하는공통보조를작게검토한다. 손/카메라전체이동과손가락변화구분,같은손연결,현재관측우선,추정별도기록을유지한다. 이번4충돌은진단용이며새평가는별도공개구간을미리동결한다. 결과를좋게만들려고시간범위확대/임계값수정/단어예외/초안수정안함. 검증된손모양정답확보전동일AI추정재학습안함.

보존·범위
기존C/T3F/운영웹/DTW보호32핀불변. 사용자/멘토접근0,학습0,추가검출기0,모델추론0,다운로드0,Git0. 공개원본234 RGB정확일치/실제PTS오차0us/원본핀78유지. 원본좌표나운영코드미변경. 학습입력과모델의개선완료가아니라시간근거비교단위완료.

담당
root: 동결·시간보완코드·집계·보고/UI·최종보존검사
local_feature_inventory: 원본추출·해시/PTS검사·18시퀀스RGB연속성검토
linguistic_eval_review: 순서가섞인234이미지직접판정·별도산술검산
ksl_source_audit: 중심18이미지및추가14이미지별도판정
실행모두로컬,별도Codex검산과독립사람정답을구분.
'''
(U/'RESULT_KO.txt').write_text(txt,encoding='utf-8')
print(json.dumps({'additional':extra,'html_bytes':(U/'RESULT_KO.html').stat().st_size},ensure_ascii=False))
