from pathlib import Path
import json,hashlib,copy
import numpy as np
from source_bridge import SourceBridge
X=Path(__file__).resolve().parent;O=X.parent
G=O/'word100_learned_feedback_integration_20261102';V=O/'word100_learned_body_relation_20261102'
RET=O/'word100_learned_feature_retention_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=X/n;p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(n,v):
 with (X/n).open('x',encoding='utf8') as f:f.write(v)

def main():
 oldfiles=[G/n for n in ['learner_adapter.py','run_demo.py','demo.html','demo.js']]
 oldfiles += [O/'word100_learned_rgb_motion_20261102/runs/rgb_residual_42/model.pt',V/'runs/rgb_geometry_42/model.pt']
 pins={str(p):sha(p) for p in oldfiles}
 save('DEMO_FROZEN.json',dict(goal='Use the already learned neural evidence on a fresh video and link observations to sourced phrases and native replay times',
   decisions='Keep existing DTW windows and frozen observers. Do not optimize all-witness coverage or train a completion model from nonexistent completion labels.',
   scope='Add source-linked partial observations; whole-order/repeat/educational correctness remain unverified and full goal stays active',
   evaluation='12 saved actual public normal/cut/reversed-main cases transformed only by additive bridge, missing-image guards, then one fresh public end-to-end runtime test and browser checks',
   new_model_training=False,new_model_download=False,old_models_and_demos_read_only=True,private_inputs_never_used_for_development=True,
   port=8808,pins=pins))
 server=(G/'run_demo.py').read_text(encoding='utf8').replace('PORT=8807','PORT=8808')
 server=server.replace("U/'order_videos'","O/'word100_learned_feedback_integration_20261102'/'order_videos'")
 text('run_demo.py',server)
 html=(G/'demo.html').read_text(encoding='utf8').replace('배운 동작으로 확인하는 수어 피드백','설명과 내 동작을 연결하는 수어 피드백')
 html=html.replace('배운 손모양·관계·이동을 내 영상에서 확인해요','수어 설명의 어떤 부분이 내 영상에 보였나요?')
 html=html.replace('여러 사람의 공개 영상으로 학습한 모델이 내 영상의 손모양·얼굴과 손·양손 관계 및 동작 흐름을 확인합니다.',
  '여러 사람의 공개 영상으로 학습한 모델이 관측한 특징을 사전 설명의 해당 문구와 내 영상의 시간에 연결합니다. 확인하지 못한 부분은 별도로 표시합니다.')
 html=html.replace('</style>', '.source-timeline{display:flex;height:18px;border-radius:5px;overflow:hidden;margin:10px 0}.source-timeline button{padding:0;border:0;border-radius:0;min-width:1px}.source-yes{background:#17806f}.source-no{background:#d18412}.source-unknown{background:#a1afba}.source-cards h3{font-size:17px}.source-cards .source a{color:#155f80}</style>')
 text('demo.html',html)
 js=(G/'demo.js').read_text(encoding='utf8').replace('function render(r,label){','function renderBase(r,label){',1)
 js+='''
function render(r,label){
 renderBase(r,label);const root=$('result');const previous=[...root.childNodes];root.replaceChildren(el('h2','설명과 연결해서 확인한 동작'));
 root.append(el('p',label,'small'));
 const cached=Boolean(r.latency?.cached_example_not_new_end_to_end_measurement);
 root.append(el('p',cached?'저장된 공개 결과 · 이번에 새로 측정한 대기시간은 아닙니다.':`영상 분석부터 결과까지 ${(r.latency?.total_analysis_seconds||0).toFixed(2)}초`,'small'));
 const section=el('div',undefined,'source-cards');const unavailable=[];
 for(const item of r.source_components||[]){
  if(['COMPONENT_NOT_RUNTIME_VALIDATED','EXACT_SCOPE_OBSERVER_UNAVAILABLE'].includes(item.status)){unavailable.push(item);continue;}
  const card=el('div',undefined,'stage');const phrases=item.source_clauses.map(c=>c.text).join(' / ');
  card.append(el('h3','설명: “'+phrases+'”'),el('p','확인할 부분: '+item.name));
  const play=el('button',`${item.start_seconds.toFixed(2)}–${item.end_seconds.toFixed(2)}초 다시 보기`);
  play.onclick=()=>{$('query').currentTime=item.start_seconds;clipEnd=item.end_seconds+.08;$('query').play();};card.append(play);
  if(item.point_evidence){
   const rows=item.point_evidence;const bar=el('div',undefined,'source-timeline');const runs=[];
   for(const p of rows){let last=runs.at(-1);if(last&&last.status===p.status){last.end=p.seconds;last.n++;}else runs.push({status:p.status,start:p.seconds,end:p.seconds,n:1});}
   for(const run of runs){let cls=run.status==='FEATURE_OBSERVED'?'source-yes':run.status==='FEATURE_NOT_CONFIRMED'?'source-no':'source-unknown';let b=el('button',undefined,cls);b.style.flex=run.n;b.title=`${run.start.toFixed(2)}–${run.end.toFixed(2)}초 · `+(cls==='source-yes'?'이 특징이 관측됨':cls==='source-no'?'이 특징을 확인하지 못함':'관측 또는 확신 부족');b.setAttribute('aria-label',b.title);b.onclick=()=>{$('query').currentTime=run.start;clipEnd=run.end+.08;$('query').play();};bar.append(b);}
   card.append(bar,el('p','초록: 특징 관측 / 주황: 해당 특징 미확인 / 회색: 관측·확신 부족. 주황색 자체가 틀린 수어라는 뜻은 아닙니다.','small'));
  }else card.append(el('p',item.status==='RELATED_MOVEMENT_OBSERVED'?'관련된 이동이 보입니다.':item.status==='INSUFFICIENT_OBSERVATION'?'영상 근거가 부족합니다.':'관련된 이동은 확인이 더 필요합니다.'));
  card.append(el('p',item.message,'small'));
  const unchecked=[...new Set(item.source_clauses.flatMap(c=>c.unchecked||[]))];
  card.append(el('p','실제 접촉·정확한 끝 위치 등 설명의 모든 조건과 전체 동작 완료를 검사한 것은 아닙니다.','small'));
  let sources=el('div',undefined,'source');for(const c of item.source_clauses){const a=el('a',c.text+' · 사전 출처');a.href=c.url;a.target='_blank';a.rel='noopener';sources.append(a,document.createTextNode(' '));}card.append(sources);section.append(card);
 }
 root.append(section);
 if(unavailable.length){let d=el('details');d.append(el('summary','아직 확인할 수 없는 설명의 부분'));for(const i of unavailable)d.append(el('p','“'+i.source_clauses.map(c=>c.text).join(' / ')+'” — '+i.message,'small'));root.append(d);}
 root.append(el('div','현재는 관측한 부분을 설명과 연결합니다. 중요한 동작의 누락·전체 순서·반복 횟수까지 검증된 정오 판정은 아직 아닙니다.','notice'));
 let detail=el('details');detail.append(el('summary','기존 손모양·관계·이동 분석 전체 보기'));for(const node of previous)detail.append(node);root.append(detail);
 $('progress').textContent='관련 문구와 시간대를 확인해 주세요. 색상이나 낮은 확신만으로 동작 오류를 확정하지 않습니다.';
}
'''
 text('demo.js',js)
 bridge=SourceBridge();checks=[];cases=[]
 for word in ['WORD0739','WORD1133','WORD1567','WORD2523']:
  for kind,folder in [('reencoded_normal','demo_public'),('action_cut','demo_controls'),('reverse_main','demo_order')]:
   src=G/folder/word/'RESULT.json';base=read(src)
   ap=G/'order_inputs'/word/'analysis.npz' if kind=='reverse_main' else RET/'actual_controls'/f'{word}_{kind}'/'analysis.npz'
   a=dict(np.load(ap,allow_pickle=False))
   if 'relation_probability' not in a:
    a['relation_probability']=np.load(V/'control_predictions'/f'rgb_geometry_42_{word}_{kind}.npz',allow_pickle=False)['probability']
   out=bridge.attach(base,a)
   assert all(out[k]==v for k,v in base.items())
   save(f'{folder}/{word}/RESULT.json',out)
   # Replay with image evidence unavailable: high cached probabilities cannot
   # turn the source readout into observed/successful evidence.
   blank={k:v.copy() for k,v in a.items()};blank['image_available']=np.zeros(len(a['frame_id']),bool)
   held=bridge.attach(base,blank)
   assert all(not c.get('confirmed_frames') and c['status']!='RELATED_MOVEMENT_OBSERVED' for c in held['source_components'])
   checks.append(dict(word=word,kind=kind,old_output_exact=True,missing_image_never_confirmed=True,
     all_source_frame_extents_preserved=all(c['frame_ids']==list(range(c['frame_ids'][0],c['frame_ids'][-1]+1)) for c in out['source_components'])))
   cases.append(dict(word=word,kind=kind,source=str(src),source_sha256=sha(src),input=str(ap),input_sha256=sha(ap),
     output=str(X/folder/word/'RESULT.json'),component_statuses=[dict(component=c['component_id'],status=c['status']) for c in out['source_components']]))
 for p,h in pins.items():assert sha(p)==h,p
 save('PUBLIC_REPLAY_VERIFIED.json',dict(cases=cases,checks=checks,old_outputs_exact=True,new_model_inference=0,new_training=0,
   meaningful_change='Learned physical observations now have explicit sourced clause, native replay time and unverified requirements in the primary view.',
   not_accuracy_gain=True,goal_complete=False))
 print('Prepared8808 source-linked demo and12public replays; old outputs exact; missing-image guards pass')
if __name__=='__main__':main()
