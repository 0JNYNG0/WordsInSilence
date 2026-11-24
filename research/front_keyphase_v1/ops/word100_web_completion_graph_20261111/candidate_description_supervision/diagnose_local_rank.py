"""Explain saved choices; reviewed witnesses/edit seams never enter runtime selection."""
from local_rank_core import *
from transfer_local_timefix_core import normalized_pose
from collections import Counter
E=L/'evaluation';comp=read(E/'COMPARISON.json')
assert not (L/'FAILURE_DIAGNOSIS.json').exists()
pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
ext={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']}
rows=[];invariants=[]
for r in comp['rows']:
 if r['kind']not in ['full_normal','dynamic_complete_blocks']:continue
 p=pm[r['seed'],r['clip_id'],r['kind']];saved=np.load(p['path']);source=np.load(p['source']);q=source['z'][:,292:392].reshape(-1,10,10)
 cuts=[s['query_right']for s in ext.get(r['clip_id'],{}).get('splice_joins',[])if s['synthetic_discontinuity']]if r['kind']=='dynamic_complete_blocks'else []
 for mode in ['local_full','local_no_hands']:
  for i,c in enumerate(r['conditions'][mode]['cards']):
   if 'truth'not in c:continue
   sp=saved[f'spans_{i}'];logit=saved[f'{mode}_logits_{i}'];admit=saved[f'presence_{i}']>=0;score=np.where(admit,logit,-np.inf)
   w=set(c['truth']['witness']);bad=set(c['truth']['wrong_witness']);valid=q[:,AX[c['family']]*2,8]>.5
   good=np.array([w<={k for k in range(a,b+1)if valid[k]}and not bad&{k for k in range(a,b+1)if valid[k]}for a,b in sp])
   candidate=None
   if (good&admit).any():
    ix=np.flatnonzero(good&admit);j=ix[np.argmax(score[ix])];candidate=dict(span=sp[j].tolist(),raw_score=float(score[j]),rank=int((score>score[j]).sum()+1),crosses_known_edit=any(sp[j,0]<cut<=sp[j,1]for cut in cuts))
   chosen=c['chosen'];rows.append(dict(condition=mode,seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=c['card'],selected=chosen['span']if chosen else None,score=chosen['score']if chosen else None,selected_crosses_known_edit=any(chosen['span'][0]<cut<=chosen['span'][1]for cut in cuts)if chosen else False,all_witness=c['truth']['all_witness'],wrong_points=c['truth']['wrong_points'],source_witness_candidates=int(good.sum()),admitted_source_witness_candidates=int((good&admit).sum()),best_source_witness_candidate=candidate,semantic_completion=None))
 if r['kind']=='dynamic_complete_blocks'and r['clip_id']in ext:
  normal=pm[r['seed'],r['clip_id'],'full_normal'];nn=np.load(normal['path']);nz=np.load(normal['source']);np_=normalized_pose(nz['x']);qp=normalized_pose(source['x']);nq=nz['z'][:,292:392]
  for m in ext[r['clip_id']]['motions'].values():
   a,b=m['block']['source_span'];qa,qb=m['block']['query_frame_span'];npb=np_[a:b+1].copy();qpb=qp[qa:qb+1].copy();npb[0,279]=qpb[0,279]=0
   delta=float(max(np.max(abs(npb-qpb)),np.max(abs(nq[a:b+1]-q.reshape(-1,100)[qa:qb+1]))))
   for mode in ['local_full','local_no_hands']:
    for i,c in enumerate(r['conditions'][mode]['cards']):
     sp=saved[f'spans_{i}'];osp=nn[f'spans_{i}'];lu={tuple(v):j for j,v in enumerate(osp)};pairs=[(j,lu[(aa-qa+a,bb-qa+a)])for j,(aa,bb)in enumerate(sp)if aa>=qa and bb<=qb and(aa-qa+a,bb-qa+a)in lu]
     if pairs:
      ix,oi=np.array(pairs).T;pdiff=float(np.max(abs(saved[f'{mode}_logits_{i}'][ix]-nn[f'{mode}_logits_{i}'][oi])))
      invariants.append(dict(condition=mode,seed=r['seed'],clip_id=r['clip_id'],card=c['card'],source_block=[a,b],query_block=[qa,qb],max_input_difference=delta,window_pairs=len(pairs),max_raw_score_difference=pdiff))
groups=[]
for mode in ['local_full','local_no_hands']:
 for kind in ['full_normal','dynamic_complete_blocks']:
  rr=[r for r in rows if r['condition']==mode and r['kind']==kind]
  groups.append(dict(condition=mode,kind=kind,cards=len(rr),all_witness=sum(r['all_witness']for r in rr),selected_crosses_edit=sum(r['selected_crosses_known_edit']for r in rr),admissible_good_exists=sum(r['admitted_source_witness_candidates']>0 for r in rr),good_rank_median=float(np.median([r['best_source_witness_candidate']['rank']for r in rr if r['best_source_witness_candidate']]))))
write(L/'FAILURE_DIAGNOSIS.json',dict(at=stamp(),rows=rows,summary=groups,local_block_invariance=invariants,no_inference=True,no_runtime_oracle=True,limits='Exact preservation of source witnesses is a diagnostic, not educational correctness. Edited joins are not natural filmed errors.'))
for r in groups:print(r)
if invariants:print('invariance input',max(r['max_input_difference']for r in invariants),'score',max(r['max_raw_score_difference']for r in invariants))
