"""Saved-output diagnostic. Gold/known edit cuts only explain failure, never select runtime spans."""
from local_sequence_core import *
E=Q/'evaluation'
d=read(E/'COMPARISON.json');pm={(r['seed'],r['clip_id'],r['kind']):r for r in d['predictions']}
ext={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']}
norm=read(O/'word100_rtmw_joint_learning_20261108/feedback_viewfix/FEEDBACK_NORMALIZATION.json');mean=np.asarray(norm['mean'])[102:186];std=np.asarray(norm['std'])[102:186]
def inp(z):
 x=z['x'].copy();x[:,102:186]=np.where(np.repeat(x[:,237:279]>.5,2,1),(x[:,102:186]-mean)/std,0)
 return local_input(z['z'],x)
rows=[];invariants=[]
for r in d['rows']:
 if r['kind'] not in ['full_normal','dynamic_complete_blocks']:continue
 key=r['seed'],r['clip_id'],r['kind'];p=pm[key];stored=np.load(p['path']);source=np.load(p['source']);q=source['z'][:,292:392].reshape(-1,10,10)
 cuts=[s['query_right'] for s in ext.get(r['clip_id'],{}).get('splice_joins',[]) if s['synthetic_discontinuity']] if r['kind']=='dynamic_complete_blocks' else []
 for i,c in enumerate(r['conditions']['local_sequence']['cards']):
  if 'truth' not in c:continue
  sp=stored[f'spans_{i}'];pp=stored[f'p_{i}'];score=np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf)
  w=set(c['truth']['witness']);bad=set(c['truth']['wrong_witness']);valid=q[:,AX[c['family']]*2,8]>.5
  good=np.array([w<={k for k in range(a,b+1) if valid[k]} and not bad&{k for k in range(a,b+1) if valid[k]} for a,b in sp])
  candidate=None
  if good.any():
   inds=np.flatnonzero(good);j=inds[np.argmax(score[inds])];candidate=dict(span=sp[j].tolist(),p=pp[j].tolist(),score=float(score[j]) if np.isfinite(score[j]) else None,rank=int((score>score[j]).sum()+1),crosses_known_edit=any(sp[j,0]<cut<=sp[j,1] for cut in cuts))
  chosen=c['chosen'];rows.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],selected=chosen['span'] if chosen else None,p=chosen['candidate_probability'] if chosen else None,selected_crosses_known_edit=any(chosen['span'][0]<cut<=chosen['span'][1] for cut in cuts) if chosen else False,all_witness=c['truth']['all_witness'],wrong_points=c['truth']['wrong_points'],source_witness_admissible_candidate_count=int(good.sum()),best_source_witness_candidate=candidate,semantic_completion=None))
 if r['kind']=='dynamic_complete_blocks' and r['clip_id'] in ext:
  # Compare exactly preserved source blocks, never interpolate the edit seam.
  extrow=ext[r['clip_id']];normal=pm[r['seed'],r['clip_id'],'full_normal'];nn=np.load(normal['path']);nz=np.load(normal['source']);normal_inp=inp(nz);query_inp=inp(source)
  for m in extrow['motions'].values():
   a,b=m['block']['source_span'];qa,qb=m['block']['query_frame_span'];a,b,qa,qb=map(int,[a,b,qa,qb]);delta=float(np.max(abs(normal_inp[a:b+1]-query_inp[qa:qb+1])))
   for i,c in enumerate(r['conditions']['local_sequence']['cards']):
    sp=stored[f'spans_{i}'];osp=nn[f'spans_{i}'];lookup={tuple(v):j for j,v in enumerate(osp)};pairs=[(j,lookup[(aa-qa+a,bb-qa+a)]) for j,(aa,bb) in enumerate(sp) if aa>=qa and bb<=qb and (aa-qa+a,bb-qa+a) in lookup]
    if pairs:
     ix,oi=np.array(pairs).T;pdiff=float(np.max(abs(stored[f'p_{i}'][ix]-nn[f'p_{i}'][oi])))
     invariants.append(dict(seed=r['seed'],clip_id=r['clip_id'],card=c['card'],source_block=[a,b],query_block=[qa,qb],max_input_difference=delta,window_pairs=len(pairs),max_probability_difference=pdiff))
tim=np.asarray([r['seconds'] for r in d['timing']]);result=dict(at=stamp(),scope='Diagnostic gold candidates/cuts only, no adoption or rescoring. Source witness coverage is not educational completion.',rows=rows,local_block_invariance=invariants,latency=dict(median_seconds=float(np.median(tim)),p95_seconds=float(np.quantile(tim,.95)),max_seconds=float(tim.max()),includes='exhaustive candidate enumeration + CPU local sequence ranking',excludes='RTMW/capture/web/semantic feedback heads'),no_new_forward=True,private_access=False,final8=False)
write(Q/'FAILURE_DIAGNOSIS.json',result)
print('Known-motion selections',len(rows),'exact-block comparisons',len(invariants))
for r in rows:
 if r['seed']==42:print({k:v for k,v in r.items() if k not in ['p','semantic_completion']})
print('invariance',max(r['max_input_difference'] for r in invariants),max(r['max_probability_difference'] for r in invariants));print(result['latency'])
