from targets import *
from collections import defaultdict
import traceback,os

def variants(p):
 return {k:dict(p,physical_event_complete=p['complete_'+k]) for k in ['strict','joint']}

def bind_features(saved,pred,old_binding,contracts):
 """Completion is evidence about a physical interval, not a whole-sign verdict.
 No word-specific logic, expected label, perturbation kind or ground truth input.
 Frozen direction/shape observations remain separate from new completeness.
 """
 cards=copy.deepcopy(saved['cards']);out=[]
 for c,s in zip(cards,contracts):
  if s['axis'] is None:continue
  es=[]
  for i,e in enumerate(saved['ledger']):
   if e['axis']!=s['axis'] or e['family'] not in s['families'] or e['moving'] not in s['movement_subjects'] or e['role_coverage']<.65 or e['valid']<.8:continue
   key=(e['axis'],e['moving'],e['a'],e['b']);d=old_binding.get(key)
   es.append(dict(span=[e['a'],e['b']],moving=e['moving'],axis=e['axis'],direction=d['direction'] if d else None,direction_readable=bool(d and d['direction_readable']),physical_complete=pred[i]['physical_event_complete'],semantic_complete=False,quality=pred[i]))
  signs={e['direction'] for e in es if e['direction'] is not None};expected=s['sign']
  state='not_specified' if expected is None else 'unknown'
  if expected is not None and signs:state='contradicted' if signs=={-expected} else 'mixed' if len(signs)>1 else 'supported'
  completed=any(e['physical_complete'] for e in es)
  c['completion_evidence']=dict(physical_interval_complete=completed,sign_completion_confirmed=False,state=state,observed=es)
  c['verdict_scope']='observed_features_only'
  c['completion_message']='움직임 구간의 시작과 끝이 함께 관측됐어요. 단어 전체의 완료 여부는 별도 확인이 필요해요.' if completed else '보이는 모양·방향과 별도로, 이 동작을 끝까지 했는지는 아직 확인하지 못했어요.'
  if not c.get('observation_quarantine_overlap'):
   if state=='contradicted':jc.integration.add_warning(c,'bilateral_direction','표시된 손의 이동 방향을 확인해 보세요. '+s['instruction'])
   elif state=='mixed' and c['display_status']=='similar':c.update(display_status='uncertain',message='이 구간에 서로 다른 이동 방향이 섞여 있어요.',educational_wrong=False)
  out.append(dict(card=c['id'],state=state,events=es))
 return cards,out

def main():
 assert (U/'PREFLIGHT.json').exists() and (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);cfg=read(U/'FROZEN.json')
 for p,h in cfg['pins'].items():assert sha(p)==h,p
 pinfiles=[U/'evaluate_boundary.py',U/'preflight.py',U/'EXTRA_EVAL_LABELS.json',P/'DIRECT_SCORES.json',P/'CROSS_SCORES.json',P/'FULLPATH_COMPLETE.json',R/'EVALUATION_INPUTS.json',U/'TRAINING_COMPLETE.json']
 write(U/'EVAL_FROZEN.json',dict(pins={str(p):sha(p) for p in pinfiles},primary=cfg['primary'],conditions=[m+'_'+v for m in cfg['models'] for v in ['strict','joint']],gates=cfg['gates'],additional_gate='No completed-partial false positives beyond deployed baseline; normal whole recall per side and cross not below deployed baseline, all seeds. No best-seed selection.',display_contract='Observed feature similarity and physical interval completeness remain separate. No semantic whole-sign success.',scope='Previously exposed DEV and nonfocus videos. Not fresh test or natural learner errors.',no_post_score_tuning=True))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING',pid=os.getpid()))
 nets={}
 for rr in read(U/'TRAINING_COMPLETE.json')['runs']:
  assert sha(rr['path'])==rr['sha256'];n=EndpointNet(rr['seed'],cfg['norm'],rr['mode']);n.load_state_dict(torch.load(rr['path'],map_location='cpu',weights_only=True));n.eval();nets[(rr['seed'],rr['mode'])]=n
 cached={r['key']:r for r in read(R/'EVALUATION_INPUTS.json')['rows']};direct=[]
 groups=defaultdict(list)
 for r in read(P/'DIRECT_SCORES.json'):groups[(r['seed'],r['clip_id'])].append(r)
 for (seed,clip),rows in groups.items():
  src=cached[clip+'_full_normal'];assert sha(src['path'])==src['sha256'];z=np.load(src['path'])
  for mode in cfg['models']:
   pp=predictions(nets[seed,mode],z['x'],z['ts'],[r['event'] for r in rows])
   for r,p in zip(rows,pp):direct.append(dict(previous=r,mode=mode,after=p))
 write(U/'DIRECT_SCORES.json',direct);event('direct_evaluated',rows=len(direct))
 # Reuse previously inferred directions, DTW, handshapes and candidate spans.
 sources={(r['seed'],r['clip_id'],r['kind']):r for r in read(P/'FULLPATH_COMPLETE.json')['rows'] if r['mode']=='combined'}
 guide=read(P/'GUIDE_SUBJECTS.json');full=[]
 for rr in read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']:
  seed=rr['seed'];key=rr['clip_id']+'_'+rr['kind'];src=cached[key];assert sha(src['path'])==src['sha256'];assert sha(rr['result'])==rr['sha256'];saved=read(rr['result']);z=np.load(src['path'])
  pr=sources[seed,rr['clip_id'],rr['kind']];assert sha(pr['result'])==pr['sha256'];pd=read(pr['result']);directions={}
  for b in pd['bindings']:
   for e in b['events']:directions[(e['quality']['axis'],e['moving'],*e['span'])]=e
  for mode in cfg['models']:
   start=time.perf_counter();pp=predictions(nets[seed,mode],z['x'],z['ts'],saved['ledger']);elapsed=time.perf_counter()-start
   for variant in ['strict','joint']:
    pred=[variants(p)[variant] for p in pp];cards,bindings=bind_features(saved,pred,directions,guide[rr['clip_id'].split('_')[2]])
    condition=mode+'_'+variant;path=U/'feedback'/str(seed)/condition/f'{key}.json'
    write(path,dict(cards=cards,bindings=bindings,predictions=pp,previous=rr['result'],previous_sha256=rr['sha256'],protected=saved['protected']))
    full.append(dict(seed=seed,clip_id=rr['clip_id'],kind=rr['kind'],condition=condition,before=rc.compact(saved['cards']),previous_candidate=pr['after'],after=rc.compact(cards),result=str(path),sha256=sha(path),additional_seconds=elapsed))
 write(U/'FULLPATH_COMPLETE.json',dict(rows=full));event('fullpath_evaluated',outputs=len(full))
 # Nonfocus public data: frozen proposals/labels, unchanged old outputs.
 sys.modules['motion_core']=mc;reader=load('_boundary_public_convert',M/'evaluate.py')
 raws={r['clip_id']:r for r in read(A/'EXTRACTION_COMPLETE.json')['rows'] if r['person']==14};cross=[];extra=[];crossgroups=defaultdict(list)
 for r in read(P/'CROSS_SCORES.json'):crossgroups[(r['clip_id'],r['kind'],r['seed'])].append(r)
 additional={r['clip_id']:r for r in read(U/'EXTRA_EVAL_LABELS.json')['rows']}
 for clip,src in raws.items():
  assert sha(src['raw'])==src['raw_sha256']
  for kind in ['normal','whole_reverse','speed075','speed125']:
   x,ts=reader.convert(src['raw'],kind=='whole_reverse');ts=np.asarray(ts)/dict(speed075=.75,speed125=1.25).get(kind,1)
   for seed in cfg['seeds']:
    rows=crossgroups[clip,kind,seed]
    for mode in cfg['models']:
     pp=predictions(nets[seed,mode],x,ts,[r['event'] for r in rows])
     for r,p in zip(rows,pp):cross.append(dict(previous=r,mode=mode,after=p))
    if kind=='normal' and clip in additional:
     ann=additional[clip];spans=[r for r in supervised_spans(ann,len(x)) if r['quality_mask']>0];events=[dict(a=r['a'],b=r['b'],axis=ann['axis'],moving=ann['moving']) for r in spans]
     # First ever baseline inference for newly annotated left spans, not rerun of prior completed evaluation.
     oldnet=prev.CompletionNet(seed,read(P/'FROZEN.json')['norm']);oldnet.load_state_dict(torch.load(P/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));oldnet.eval();oldpp=prev.infer_batch(oldnet,x,ts,events)
     for mode in cfg['models']:
      pp=predictions(nets[seed,mode],x,ts,events)
      for row,e,b,p in zip(spans,events,oldpp,pp):extra.append(dict(clip_id=clip,seed=seed,mode=mode,event=e,label=row,before=b,after=p))
 write(U/'CROSS_SCORES.json',cross);write(U/'EXTRA_LEFT_SCORES.json',extra)
 write(U/'EVALUATION_COMPLETE.json',dict(direct_rows=len(direct),full_outputs=len(full),cross_rows=len(cross),extra_left_rows=len(extra),base_TCN_DTW_RTMW_reruns=0,new_models=6,threshold_tuned=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_VERIFICATION',pid=None));event('evaluation_complete',cross_rows=len(cross),extra_left_rows=len(extra))

if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
