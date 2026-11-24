"""Saved K rank arithmetic only: no torch, model imports, or learned forwards."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, itertools, json, traceback
import numpy as np

E=Path(__file__).resolve().parent; K=E.parent; H=K.parent; G=H.parent; S=H/'stability_rank_blend'
MODES=['rate_head','old_local_full_plus_rate_half']; AX={2:0,3:1,4:2,5:4,7:3}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(r):return r['seed'],r['clip_id'],r['kind']
checks=0
def ck(value,label):
 global checks
 checks+=1
 if not value:raise AssertionError(label)

def select(card,spans,mask,score,qr,mode):
 chosen=None
 positions=[i for i,v in enumerate(mask)if v>=0]
 if positions:
  # Strict greater retains the first tied candidate independently of np.argmax.
  chosen_index=positions[0]
  for i in positions[1:]:
   if score[i]>score[chosen_index]:chosen_index=i
  a,b=map(int,spans[chosen_index]);axis=AX[card['family']]*2
  observed=[i for i in range(a,b+1)if qr[i,axis,8]>.5]
  chosen=dict(span=[a,b],source=mode,body_observed=observed,hand_qualified=[i for i in observed if qr[i,axis,9]>.5],score=float(score[chosen_index]),semantic_completion=None)
 result=dict(card=card['card'],family=card['family'],chosen=chosen,semantic_completion=None)
 if 'truth'in card:
  t=card['truth'];seen=set(chosen['body_observed'])if chosen else set()
  result['truth']=dict(t,all_witness=set(t['witness']).issubset(seen),wrong_points=len(set(t['wrong_witness']).intersection(seen)))
 return result

def orders(cards,ts):
 raw=[];pts=[]
 for a,b in itertools.combinations(cards,2):
  if sorted([a['family'],b['family']])!=[2,3]:continue
  spans=[c['chosen']['span']if c['chosen']else None for c in[a,b]];names=[a['card'],b['card']]
  if any(s is None for s in spans):
   rawstate=state=detail='unobserved';overlap=None;times=[None if s is None else[int(ts[s[0]]),int(ts[s[1]])]for s in spans]
  else:
   u,v=spans;times=[[int(ts[s[0]]),int(ts[s[1]])]for s in spans];x,y=times
   ck(x[0]<x[1]and y[0]<y[1],'strict positive extent')
   rawstate='ambiguous_overlap'if max(u[0],v[0])<=min(u[1],v[1])else'forward'if u[1]<v[0]else'reverse'
   overlap=max(0,min(x[1],y[1])-max(x[0],y[0]))
   if x[1]<=y[0]:state='forward';detail='forward_shared_boundary'if x[1]==y[0]else'forward_separated'
   elif y[1]<=x[0]:state='reverse';detail='reverse_shared_boundary'if y[1]==x[0]else'reverse_separated'
   else:state='ambiguous_overlap';detail='positive_duration_overlap';ck(overlap>0,'actual overlap positive')
  raw.append(dict(cards=names,state=rawstate));pts.append(dict(cards=names,state=state,detail=detail,overlap_duration_us=overlap,spans=spans,time_extents_us=times))
 return raw,pts

def main():
 dst=E/'INDEPENDENT_SAVED_VERIFICATION.json';ck(not dst.exists(),'one saved arithmetic verification')
 cfg=read(E/'FROZEN.json');execution=read(E/'EXECUTION_FROZEN.json');r=read(E/'COMPARISON.json');old=read(S/'RESULTS.json');training=read(K/'TRAINING_COMPLETE.json')
 ck(r['frozen_sha256']==sha(E/'FROZEN.json'),'freeze hash');ck(r['training_sha256']==sha(K/'TRAINING_COMPLETE.json'),'training hash')
 for pins in[cfg['pins'],execution['pins']]:
  for p,h in pins.items():ck(sha(p)==h,('preserved pin',p))
 protections=read(G/'PROTECTIONS.json');ck(len(protections)==477 and all(cfg['pins'].get(p)==h for p,h in protections.items()),'477 original protected pins')
 ck(execution['models_fixed_before_inference']and len(execution['pins'])==4,'three models plus training manifest fixed')
 ck({x['model']:x['model_sha256']for x in training['rows']}=={p:h for p,h in execution['pins'].items()if p!=str((K/'TRAINING_COMPLETE.json').resolve())},'execution models exact completed models')
 ck(cfg['at']<execution['at']<r['at'],'freeze precedes execution/comparison')
 rm={key(x):x for x in r['rows']};om={key(x):x for x in old['rows']};pm={key(x):x for x in r['predictions']};op={key(x):x for x in old['predictions']}
 expected={(seed,s['clip_id'],s['kind'])for s in cfg['scenarios']for seed in cfg['seeds']}
 ck(len(cfg['scenarios'])==82 and len(expected)==246,'82 scenarios times3seeds')
 ck(len(r['rows'])==len(r['predictions'])==246 and set(rm)==set(om)==set(pm)==expected,'unique exact246 keys')
 ck(cfg['conditions']==MODES,'two frozen conditions')
 for s in cfg['scenarios']:
  signatures=[[(c['card'],c['family'])for c in om[seed,s['clip_id'],s['kind']]['conditions']['local_full']['cards']]for seed in cfg['seeds']]
  ck(signatures[0]==signatures[1]==signatures[2],'identical seed card order')
 source_cache={};span_cache={};protected=0;total=0;changes=[];normalcases=[];regressions=[];feature_rows=0
 earliest={key(x):x for x in read(G/'LINKER_RESULTS.json')}
 for k,row in rm.items():
  original=om[k];meta=pm[k];oldmeta=op[k]
  ck(len(original['conditions'])==6,'six old conditions')
  for mode,c in original['conditions'].items():ck(row['conditions'][mode]==c,('old condition exact',mode,k))
  for mode,v in original['pts_order'].items():ck(row['pts_order'][mode]==v,('old PTS exact',mode,k))
  ck(set(row['conditions'])==set(original['conditions'])|set(MODES),'only fixed two modes added')
  ck(meta['previous_blend_prediction']==oldmeta['path']and meta['previous_blend_sha256']==oldmeta['sha256'],'old blend prediction lineage')
  for field,hfield in [('path','sha256'),('original_prediction','original_prediction_sha256')]:ck(sha(meta[field])==meta[hfield],('prediction hash',field,k))
  if meta['source']not in source_cache:
   with np.load(meta['source'],allow_pickle=False)as z:source_cache[meta['source']]=(z['z'][:,292:392].reshape(-1,10,10).copy(),z['ts'].copy())
  qr,ts=source_cache[meta['source']];ck(np.issubdtype(ts.dtype,np.integer)and np.all(np.diff(ts)>0),'actual integer timestamps')
  if meta['feature_path']not in span_cache:
   with np.load(meta['feature_path'],allow_pickle=False)as z:span_cache[meta['feature_path']]=z['spans'].copy()
  cards=original['conditions']['local_full']['cards'];out={mode:[]for mode in MODES}
  with np.load(meta['path'],allow_pickle=False)as new,np.load(meta['original_prediction'],allow_pickle=False)as prev:
   union=np.asarray(sorted({tuple(v)for i in range(len(cards))for v in prev[f'spans_{i}']}),np.int32).reshape(-1,2)
   ck(np.array_equal(union,span_cache[meta['feature_path']]),'exact candidate union')
   for i,c in enumerate(cards):
    sp=new[f'spans_{i}'];pr=new[f'presence_{i}'];a=prev[f'local_full_logits_{i}'];b=new[f'rate_head_logits_{i}'];blend=new[f'old_local_full_plus_rate_half_logits_{i}']
    ck(np.array_equal(sp,prev[f'spans_{i}'])and np.array_equal(pr,prev[f'presence_{i}']),'candidates and presence unchanged')
    ck(a.shape==b.shape==blend.shape==pr.shape and np.isfinite(b).all()and np.isfinite(blend).all(),'finite matching score shapes')
    if c['family']in[2,3]:ck(np.array_equal(blend,.5*a.astype(np.float64)+.5*b.astype(np.float64)),'fixed float64 half blend allcandidate logits')
    else:ck(np.array_equal(a,b)and np.array_equal(a,blend),'other family logits exact')
    for mode,score in zip(MODES,[b,blend]):
     actual=row['conditions'][mode]['cards'][i]
     if c['family']in[2,3]:after=select(c,sp,pr,score,qr,mode)
     else:after=c;protected+=1
     ck(actual==after,('independent selection/card evidence',mode,k,c['card']));total+=1;out[mode].append(after)
     before_span=c['chosen']['span']if c['chosen']else None;after_span=after['chosen']['span']if after['chosen']else None
     if before_span!=after_span:changes.append(dict(seed=k[0],clip_id=k[1],kind=k[2],condition=mode,card=c['card'],old_span=before_span,new_span=after_span))
  for mode in MODES:
   raw,pts=orders(out[mode],ts);ck(raw==row['conditions'][mode]['order']and pts==row['pts_order'][mode],('independent raw/PTS order',mode,k))
  if 'WORD0739'in k[1]and k[2]in['full_normal','speed075','speed125']:
   first=earliest[k];case=dict(seed=k[0],clip_id=k[1],kind=k[2],first_midpoint=[v for v in first['baseline_pairs']if v['guides']==[1,2]],first_role=[v for v in first['methods']['old_learned_roles_independent']['pairs']if v['families']==[2,3]],current={})
   for mode,cond in row['conditions'].items():
    _,pts=orders(cond['cards'],ts);case['current'][mode]=pts
   normalcases.append(case)
   for mode in MODES:
    state=row['pts_order'][mode][0]['state'];priorstate=original['pts_order']['local_full'][0]['state']
    if priorstate=='forward'and state!='forward':regressions.append(dict(seed=k[0],clip_id=k[1],kind=k[2],condition=mode,relative_to='local_full',prior=priorstate,current=state,pts=row['pts_order'][mode][0]))
 ck(total==636 and protected==348 and protected==r['protected_other_family_cards'],'318cards per newmode,174protected per mode')
 ck(r['summary'][:len(old['summary'])]==old['summary'],'all old summary rows exact')
 recomputed=[]
 for seed,mode,kind in itertools.product(cfg['seeds'],MODES,sorted({x['kind']for x in r['rows']})):
  rr=[x for x in r['rows']if x['seed']==seed and x['kind']==kind];cc=[c for x in rr for c in x['conditions'][mode]['cards']];tt=[c['truth']for c in cc if'truth'in c]
  recomputed.append(dict(seed=seed,condition=mode,kind=kind,scenarios=len(rr),cards=len(cc),missing=sum(c['chosen']is None for c in cc),truth_cards=len(tt),all_witness=sum(t['all_witness']for t in tt),wrong_points=sum(t['wrong_points']for t in tt),PTS_order=dict(Counter(p['state']for x in rr for p in x['pts_order'][mode])),raw_discrete_order=dict(Counter(p['state']for x in rr for p in x['conditions'][mode]['order']))))
 ck(recomputed==r['summary'][len(old['summary']):],'all66 new summaries exact')
 ck(r['semantic_completion']is None and r['G2_pass']is None,'no completion/pass promotion')
 timing={key(x):x for x in r['timing']};ck(len(r['timing'])==246 and set(timing)==expected,'one timing record per scenario seed')
 inferred=sum(any(c['family']in[2,3]for c in x['conditions']['local_full']['cards'])for x in r['rows'])
 ck(sum(x['scope']=='new small rank head only'for x in r['timing'])==inferred,'recorded one newhead schedule, two fixed conditions reuse scores')
 aggregate=[]
 for mode in MODES:
  for kind in ['full_normal','speed075','speed125','dynamic_complete_blocks','reversed_core_auto_order','omit_final_event','mask_final_event','all_hand_missing']:
   rr=[x for x in r['rows']if x['kind']==kind];tt=[c['truth']for x in rr for c in x['conditions'][mode]['cards']if'truth'in c]
   aggregate.append(dict(condition=mode,kind=kind,order=dict(Counter(p['state']for x in rr for p in x['pts_order'][mode])),witness_all=sum(t['all_witness']for t in tt),witness_total=len(tt),wrong_points=sum(t['wrong_points']for t in tt)))
 report=dict(at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),status='SAVED_RANK_ARITHMETIC_AND_PRESERVATION_PASS_BOTH_CANDIDATES_HAVE_NORMAL_REGRESSION',checks=checks,scenarios=82,rows=246,new_conditions=MODES,new_cards=636,protected_other_family_cards=348,protected_original_files=477,freeze_pins=len(cfg['pins']),execution_pins=len(execution['pins']),old_conditions_exact=6,old_PTS_conditions_exact=3,new_summary_rows=len(recomputed),newhead_schedule_rows=inferred,new_model_inferences_by_verifier=0,encoder_inferences_by_verifier=0,normal_speed_cases=normalcases,regressions_against_local_full=regressions,changed_semantic_spans=changes,aggregate=aggregate,original_G2_reference=str(H/'G2_REFERENCE_LINEAGE_REVIEW.json'),decision_notes=['Neither fixed condition can be declared G2 PASS by choosing the better condition separately for each case.','Original midpoint/role reference and intermediate baseline/local_full remain separate; point and interval decisions are not the same metric.','Each changed interval still needs semantic shadow verification. Old shape/direction weights unchanged does not prove new interval feedback preserved.','All-hand-missing raw body spans and relations are not whole-action positive supports; this rank evaluation does not emit semantic completion.','This is exposed public development, not fresh or educational correctness evaluation.'],G2_pass=None,G3_pass=None,new_training=False,new_neural_inference=False,threshold_or_alpha_selection=False,pins={str(p):sha(p)for p in[Path(__file__),E/'COMPARISON.json',E/'FROZEN.json',E/'EXECUTION_FROZEN.json',S/'RESULTS.json',H/'G2_REFERENCE_LINEAGE_REVIEW.json']})
 with dst.open('x',encoding='utf-8')as f:json.dump(report,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in report.items()if k not in['pins','normal_speed_cases','changed_semantic_spans','decision_notes']},ensure_ascii=False))

if __name__=='__main__':
 try:main()
 except BaseException:
  with(E/f'INDEPENDENT_VERIFICATION_FAILURE_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}.txt').open('x',encoding='utf-8')as f:f.write(traceback.format_exc())
  raise
