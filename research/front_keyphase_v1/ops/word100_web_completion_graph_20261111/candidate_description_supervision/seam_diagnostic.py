from core import *
import itertools
base=read(U/'episode_stage_contrast/BINDING_V2_SCORES.json')['rows'];comp=read(W/'COMPARISON.json');truth={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']};out=[]
for e in comp['predictions']:
 if e['kind']!='dynamic_complete_blocks' or e['clip_id'] not in truth:continue
 g=truth[e['clip_id']];cuts=[v['query_right'] for v in g['splice_joins'] if v['synthetic_discontinuity']];d=np.load(e['path']);rr=[r for r in base if (r['seed'],r['clip_id'],r['kind'])==(e['seed'],e['clip_id'],e['kind'])];choices=[]
 for i,r in enumerate(rr):
  cs=[c for c in r['candidates'] if c['has_motion']];new=[]
  for (ri,ci,j),p in zip(d['spec'],d['p']):
   if ri==i:new.append(dict(cs[ci],p=p[j].tolist(),crosses_known_edit=any(cs[ci]['span'][0]<k<=cs[ci]['span'][1] for k in cuts)))
  choices.append(new)
 for condition in ['presence_purity','presence_purity_coverage']:
  selected=[]
  for r,cs in zip(rr,choices):
   eligible=[c for c in cs if not c['crosses_known_edit']];n=2 if condition=='presence_purity' else 3;best=max(eligible,key=lambda c:np.log(np.clip(c['p'][:n],1e-6,1-1e-6)).sum()) if eligible else None;t=r['truth'];obs=set(best['body_observed']) if best else set();selected.append(dict(card=r['card'],family=r['family'],span=best['span'] if best else None,all_witness=set(t['witness'])<=obs,wrong_points=len(set(t['wrong_witness'])&obs)))
  a,b=[r['span'] for r in selected];order='unknown' if a is None or b is None else 'reverse' if b[1]<a[0] else 'forward' if a[1]<b[0] else 'overlap';out.append(dict(seed=e['seed'],clip_id=e['clip_id'],condition=condition,known_edit_cuts=cuts,cards=selected,order=order))
summary=[]
for seed,cond in itertools.product([42,43,44],['presence_purity','presence_purity_coverage']):
 rr=[r for r in out if r['seed']==seed and r['condition']==cond];s=dict(seed=seed,condition=cond,groups=sum(len(r['cards']) for r in rr),all_witness=sum(c['all_witness'] for r in rr for c in r['cards']),wrong_points=sum(c['wrong_points'] for r in rr for c in r['cards']),reverse=sum(r['order']=='reverse' for r in rr));summary.append(s);print(s)
write(W/'SEAM_DIAGNOSTIC.json',dict(at=stamp(),rows=out,summary=summary,diagnostic_only=True,ground_truth_edit_positions_used=True,not_runtime_method=True,not_improvement_claim=True,no_model_rerun=True,purpose='Counterfactual isolate synthetic editing discontinuities from ranking errors. Full unchanged benchmark remains failed. Do not infer edits in real capture or exclude short real movements.'))
