"""Fresh-process saved-artifact checks; no inference or score tuning."""
from source_boundary_core import *
from collections import Counter
import math
E=B/'evaluation';meta=read(B/'INPUTS_COMPLETE.json');cfg=read(B/'FROZEN.json');a=np.load(meta['arrays'])
assert sha(meta['arrays'])==meta['arrays_sha256']
for p,h in {**cfg['pins'],**meta['pins'],**read(E/'FROZEN.json')['pins']}.items():assert sha(p)==h,p
assert {r['person'] for r in meta['sequences'] if r['split']=='train'}=={2,4,5,6}
assert {r['person'] for r in meta['sequences'] if r['split']!='train'}=={8,9}
assert not ({r['source_sha256'] for r in meta['sequences'] if r['split']=='train'}&{r['source_sha256'] for r in meta['sequences'] if r['split']!='train'})
assert len({r['clip_id'] for r in meta['sequences']})==24 and len(meta['sequences'])==144
assert all(v['exact_same_features'] for v in meta['shape_invariance'])
for r in meta['sequences']:
 assert sha(r['path'])==r['sha256'];z=np.load(r['path']);assert all(np.isfinite(z[k]).all() for k in z.files);assert (z['bands']<=z['valid']).all()
for r in meta['boundary_audit']:assert r['onset'][0]<=min(r['progression'])<=max(r['progression'])<=r['offset'][1]
weights=np.load(B/'TRAIN_WEIGHTS.npz');assert all(np.isclose(weights[k].sum(),1) for k in weights.files)
comp=read(E/'COMPARISON.json');assert len(comp['rows'])==246;pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
ls=lambda v: -max(0.,-float(v))-math.log1p(math.exp(-abs(float(v))))
sig=lambda v:math.exp(ls(v))
argmax_checked=0
for r in comp['rows']:
 p=pm[r['seed'],r['clip_id'],r['kind']];assert sha(p['path'])==p['sha256'] and sha(p['source'])==p['source_sha256'];z=np.load(p['path'])
 for mode in ['body_window','direct_boundary']:
  for ci,c in enumerate(r['conditions'][mode]['cards']):
   sp=z[f'spans_{ci}'];j=F.index(c['family']);scores=[]
   if mode=='body_window':scores=[sum(map(ls,v)) if v[0]>=0 else -math.inf for v in z[f'window_logits_{ci}']]
   else:
    bl=z['boundary_logits'][:,j];lp=[ls(v) for v in bl[:,2]];pp=[sig(v) for v in bl[:,2]]
    for lo,hi in sp:
     n=int(hi-lo+1);ok=math.fsum(pp[lo:hi+1])/n>=.5;scores.append(float(bl[lo,0])+float(bl[hi,1])+math.fsum(lp[lo:hi+1])/n if ok else -math.inf)
   if scores and max(scores)>-math.inf:
    k=max(range(len(scores)),key=scores.__getitem__);assert c['chosen']['span']==sp[k].tolist(),(r['clip_id'],r['kind'],r['seed'],mode,c['card']);assert math.isclose(c['chosen']['score'],scores[k],abs_tol=1e-9)
   else:assert c['chosen'] is None
   assert c['semantic_completion'] is None;argmax_checked+=1
for s in comp['summary']:
 rr=[r for r in comp['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']];cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']];tt=[c for c in cc if 'truth'in c];obs=lambda c:set(c['chosen']['body_observed']) if c['chosen'] else set()
 assert sum(set(c['truth']['witness'])<=obs(c) for c in tt)==s['all_witness'];assert sum(len(set(c['truth']['wrong_witness'])&obs(c)) for c in tt)==s['wrong_points'];assert sum(c['chosen'] is None for c in cc)==s['missing'];assert dict(Counter(o['state'] for r in rr for o in r['conditions'][s['condition']]['order']))==s['order']
shadow=read(E/'shadow_components/RESULTS.json')
for s in shadow['summary']:
 cc=[c for r in shadow['rows'] if r['seed']==s['seed'] and r['kind']==s['kind'] for c in r['conditions'][s['condition']]];assert dict(Counter(c['status'] for c in cc))==s['statuses']
 for task in ['shape','direction']:assert dict(Counter(v['state'] for c in cc for v in c['evidence'] if v['task']==task))==s[task]
assert all(v['state']=='observation_unknown' for r in shadow['rows'] if r['kind']=='all_hand_missing' for cs in r['conditions'].values() for c in cs for v in c['evidence'] if v['task']=='shape')
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
write(B/'VERIFICATION.json',dict(at=stamp(),saved_arithmetic='PASS',argmax_checked=argmax_checked,protected_files=len(protected),protected_unchanged=True,no_new_inference=True,comparison_rows=246,source_clips=24,views=3,shape_only_body_input_invariant=True,reviewer='same Codex separate process, not independent human/KSL truth',first_failures_preserved=[str(W/'source_boundary_candidate/INPUT_REJECTED.json'),str(W/'source_boundary_candidate_v2/PRETRAIN_REJECTED.json')],expert_gold=False,private_access=False,final8_access=False,web_changed=False))
print('Saved source-boundary arithmetic verified',argmax_checked,'cards;',len(protected),'protected files')
