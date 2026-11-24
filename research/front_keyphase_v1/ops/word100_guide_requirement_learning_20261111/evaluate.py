from requirement_core import *
from collections import Counter,defaultdict
import os,time,traceback,copy

def main():
 assert (U/'TRAINING_COMPLETE.json').exists() and not (U/'EVALUATION_COMPLETE.json').exists()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 cfg=read(U/'FROZEN.json');assert all(sha(p)==s for p,s in cfg['pins'].items())
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATING_FIXED_COMPARISON',pid=os.getpid()))
 rr={r['clip_id']:r for r in public_rows() if r['role']=='development'}
 sources=read(R/'EVALUATION_INPUTS.json')['rows']; cached={r['key']:r for r in sources}
 oldscores=read(B/'scores/coupled_42.json')
 baseline_rows=[q for q in oldscores if q['variant']=='normal' and q['task'] in ['reviewed_shape','movement','order']]
 protected_inputs={str(B/'scores/coupled_42.json'):sha(B/'scores/coupled_42.json')}
 previous=read(R/'safety_reconcile/FULLPATH_COMPLETE.json')['rows']
 # Fix all inputs/decoding before looking at the new model's scores.
 for r in sources:protected_inputs[r['path']]=r['sha256']
 for r in previous:protected_inputs[r['result']]=r['sha256']
 write(U/'EVAL_FROZEN.json',dict(pins=protected_inputs,primary='Reviewed supplied-window component compatibility on12publicDEVvideos; separate full78condition diagnostics',auto_scope='New matcher on prior automatic intervals; unchanged detector/TCN/DTW/role locator. Not autonomous end-to-end correctness gold.',unknown_gate='Insufficient observations, absent mapped requirements, or no linked candidate -> unknown; no blanket partial==error',metric_units='source annotations and derived compatibility queries counted separately',no_fresh_holdout=True))
 assert all(sha(p)==s for p,s in protected_inputs.items())
 arrcache={}
 def cache(key):
  if key not in arrcache:arrcache[key]={k:v for k,v in np.load(cached[key]['path'],allow_pickle=False).items()}
  return arrcache[key]
 base=frozen_observer();gold=[];basecheck=[]
 for q in baseline_rows:
  cid=q['clip_id'];r=rr[cid];c=cache(cid+'_full_normal');h=c['fh'];x=c['x'];ts=c['ts']; task=q['task'];inp=[]
  if task=='reviewed_shape':
   a=int(np.flatnonzero(r['frame_id']==q['frame'])[0]);j=q['feature'];pr=q['probability'];
   with torch.inference_mode():check=float(base.shared.base.attributes(torch.tensor(h[a]))[j].sigmoid())
   basecheck.append(abs(check-pr))
   for want in [0,1]:inp.append((observation(h,a,a),requirement('shape',feature=j,expected=want),pr if want else 1-pr,int(q['target']==want),dict(task='shape',a=a,b=a,feature=j,expected=want)))
  elif task=='movement':
   p=r['pairs'][q['pair']];a,b=p['a'],p['b'];g,ok,info=prior.geometry(x,ts,a,b,p['axis'],p.get('moving','R'),p.get('anchor','L'))
   for want in range(3):inp.append((observation(h,a,b,g),requirement('direction',axis=p['axis'],direction=want),q['probability'][want],int(q['target']==want),dict(task='direction',a=a,b=b,pair=q['pair'],axis=p['axis'],expected=want)))
  else:
   p=r['orders'][q['pair']];a,b=p['a'],p['b']
   inp.append((observation(h,a,b),requirement('order',guides=[p['guide_a'],p['guide_b']]),q['probability'],q['target'],dict(task='order',a=a,b=b,pair=q['pair'],guides=[p['guide_a'],p['guide_b']])))
  for ob,req,pr,y,meta in inp:gold.append(dict(ob=ob,req=req,base=logit(pr),meta=dict(clip_id=cid,target=y,eligible=q['eligible'],baseline_probability=float(pr),annotation=q,**meta)))
 assert max(basecheck)<1e-4,('Cached observer lineage mismatch',max(basecheck))
 write(U/'CACHE_LINEAGE_CHECK.json',dict(shape_cells=len(basecheck),max_probability_difference=max(basecheck),frozen_encoder_rerun=False))
 # Audit the joint semantic contract rather than treating short clips as errors.
 labels=read(U/'REQUIREMENT_LABELS.json')['rows']; support=Counter()
 for l in labels:
  if l['task']=='shape':
   for cid in l['guide_clause_ids']:support[(l['clip_id'].split('_')[2],cid,l['feature'],l['observed'])]+=1
 capability=[]
 for word,cards in CARDS.items():
  for card in cards:
   for j,y in card.get('features',{}).items():capability.append(dict(word=word,card=card['id'],feature=int(j),expected=y,direct_positive_cells=support[(word,card['id'],int(j),y)],qualified=bool(support[(word,card['id'],int(j),y)])))
 write(U/'GUIDE_CAPABILITY.json',dict(rows=capability,unqualified_policy='No descriptor-specific verification; do not assert that a classifier output proves this requirement',exact_contact=False,educational_tolerance=False))
 results=[];full=[];controls=[]
 for seed in cfg['seeds']:
  n=RequirementAdapter(np.array(cfg['mean'],np.float32),np.array(cfg['std'],np.float32));n.load_state_dict(torch.load(U/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'));n.eval()
  def predict(ob,req,pr):
   with torch.inference_mode():return float(n(torch.tensor(ob[None]),torch.tensor(req[None]),torch.tensor([logit(pr)])).sigmoid()[0])
  ob=torch.tensor(np.stack([a['ob'] for a in gold]));rq=torch.tensor(np.stack([a['req'] for a in gold]));bl=torch.tensor([a['base'] for a in gold])
  with torch.inference_mode():prob=n(ob,rq,bl).sigmoid().tolist()
  for a,p in zip(gold,prob):
   meta=a['meta'];results.append(dict(seed=seed,**meta,probability=p,before=decide(meta['baseline_probability'],meta['eligible']),after=decide(p,meta['eligible'])))
  for rec in sources:
   c=cache(rec['key']);h=c['fh'];x=c['x'];ts=c['ts'];origin=next(v for v in previous if v['seed']==seed and v['clip_id']==rec['clip_id'] and v['kind']==rec['kind']);old=read(origin['result']); word=rec['clip_id'].split('_')[2]
   rows=[];start=time.perf_counter();fh=torch.tensor(h[None]); av=prior.shared.available(x)
   for card in old['cards']:
    reqs=[];range_=card.get('query_indices');assoc=range_ is not None and len(range_)==2
    if not assoc:
     rows.append(dict(card=card['id'],before=card['display_status'],after='association_unknown',requirements=[],reason='No located interval; absent detection is not demonstrated omission'));continue
    a,b=map(int,range_);a=max(0,a);b=min(len(h)-1,b)
    # Use the previous card's observed expected probability (not its correctness verdict).
    for se in card.get('shape_evidence',[]):
     j=int(se['feature']);want=int(se['expected']);pr=se.get('mean_expected_probability');eligible=bool(av[a:b+1,j].mean()>=.8)
     qualified=bool(support[(word,card['id'],j,want)])
     if pr is None:
      reqs.append(dict(task='shape',feature=j,state='observation_unknown'));continue
     p=predict(observation(h,a,b),requirement('shape',feature=j,expected=want),float(pr))
     state=decide(p,eligible,qualified)
     reqs.append(dict(task='shape',feature=j,expected=want,probability=p,baseline_probability=pr,state=state,span=[a,b],qualified=qualified))
    spec=next((s for s in FAMILIES[word] if s['id']==card['id']),None)
    if spec and spec['axis'] is not None:
     # Common family/axis matching; no expected-direction-based event selection.
     es=[e for e in old['selected'] if e['family'] in spec['families'] and e['axis']==spec['axis']]
     if not es:reqs.append(dict(task='direction',state='association_unknown',reason='No linked motion on required relation axis'))
     else:
      for e in es:
       aa,bb=e['a'],e['b'];g,ok,_=prior.geometry(x,ts,aa,bb,e['axis'],e['moving'],'L' if e['moving']=='R' else 'R')
       if spec['sign'] is None:
        # Alternation requires a sequence of directions, not one preferred sign.
        reqs.append(dict(task='direction',state='sequence_required',span=[aa,bb]));continue
       want=int(spec['sign']+1)
       with torch.inference_mode():pr=float(base.movement(fh,torch.tensor(g[None]),torch.tensor([[0,aa,bb]])).softmax(-1)[0,want])
       p=predict(observation(h,aa,bb,g),requirement('direction',axis=e['axis'],direction=want),pr)
       reqs.append(dict(task='direction',state=decide(p,ok),probability=p,baseline_probability=pr,span=[aa,bb],physical_complete=e.get('complete'),short_is_wrong=False))
    st=[r['state'] for r in reqs]
    # Both opposing/return movements in the interval -> association ambiguity, not a confident error.
    motion_states={r['state'] for r in reqs if r['task']=='direction'}
    if {'difference','supported'}.issubset(motion_states):status='association_unknown'
    elif 'difference' in st:status='difference'
    elif st and all(s=='supported' for s in st):status='supported_features_only'
    else:status='insufficient_evidence'
    rows.append(dict(card=card['id'],before=card['display_status'],after=status,requirements=reqs,not_sign_completion=True))
   full.append(dict(seed=seed,clip_id=rec['clip_id'],kind=rec['kind'],cards=rows,seconds=time.perf_counter()-start,previous=origin['result'],previous_sha256=origin['sha256'],protected=old['protected']))
  # Fixed metamorphic controls: elapsed time/scale are not requirements; absence cannot pass.
  for a in gold:
   if a['meta']['task']!='direction':continue
   m=a['meta'];c=cache(m['clip_id']+'_full_normal');r=rr[m['clip_id']];p=r['pairs'][m['pair']];aa,bb=m['a'],m['b'];baseline=m['baseline_probability']
   g,ok,_=prior.geometry(c['x'],c['ts']*.7,aa,bb,p['axis'],p.get('moving','R'),p.get('anchor','L'));new_ob=observation(c['fh'],aa,bb,g)
   before=predict(a['ob'],a['req'],baseline);after=predict(new_ob,a['req'],baseline)
   # Translation/scale transform only global joint coordinates, not masks or local geometry.
   scaled=c['x'].copy();scaled[:,:102]=scaled[:,:102]*1.2+np.tile([.2,-.15],51)
   sg,sok,_=prior.geometry(scaled,c['ts'],aa,bb,p['axis'],p.get('moving','R'),p.get('anchor','L'))
   sp=predict(observation(c['fh'],aa,bb,sg),a['req'],baseline)
   controls.append(dict(seed=seed,clip_id=m['clip_id'],pair=m['pair'],desired=m['expected'],shorter_time_delta=abs(before-after),body_scale_translation_delta=abs(before-sp),masked_state=decide(after,False),unknown_association=decide(after,True,False),scope='Comparator invariance with frozen latent; not full-observer altered-video accuracy or amplitude tolerance'))
 write(U/'SCORES.json',results);write(U/'FULLPATH_DIAGNOSTICS.json',full);write(U/'CONTROLS.json',controls)
 summary={}
 for seed in cfg['seeds']:
  ss=[r for r in results if r['seed']==seed];summary[str(seed)]={}
  for task in ['shape','direction','order']:
   t=[r for r in ss if r['task']==task]
   summary[str(seed)][task]=dict(queries=len(t),before_correct=sum((r['baseline_probability']>=.5)==bool(r['target']) for r in t),after_correct=sum((r['probability']>=.5)==bool(r['target']) for r in t),before_false_difference=sum(r['target']==1 and r['before']=='difference' for r in t),after_false_difference=sum(r['target']==1 and r['after']=='difference' for r in t),before_false_support=sum(r['target']==0 and r['before']=='supported' for r in t),after_false_support=sum(r['target']==0 and r['after']=='supported' for r in t),before_withheld=sum(r['before'] not in ['supported','difference'] for r in t),after_withheld=sum(r['after'] not in ['supported','difference'] for r in t))
 write(U/'EVALUATION_COMPLETE.json',dict(summary=summary,gold_original_annotations=len(baseline_rows),derived_queries_per_seed=len(gold),public_dev_videos=len({q['clip_id'] for q in baseline_rows}),fullpath_conditions=len(sources),fullpath_seed_runs=len(full),capability_qualified=sum(q['qualified'] for q in capability),capability_total=len(capability),encoder_rerun=False,baseline_saved=True,fullpath_old_heads_on_new_requirement_queries=True,new_model_scores_tuned=False,fullpath_no_educational_truth=True))
 write(U/'CURRENT_EXECUTION.json',dict(status='EVALUATED_PENDING_SEPARATE_VERIFICATION',pid=None))
 print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'EVAL_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
