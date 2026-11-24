"""Recount saved outputs; no training, forward pass, or rescoring."""
from local_sequence_core import *
from collections import Counter

E=Q/'evaluation'
meta=read(N/'INPUTS_COMPLETE.json'); arr=np.load(meta['arrays'])
assert sha(meta['arrays'])==meta['arrays_sha256']
assert {r['person'] for r,t in zip(meta['rows'],arr['train']) if t}=={2,4,5,6}
assert {r['person'] for r,t in zip(meta['rows'],arr['train']) if not t}=={8,9}
assert len({r['clip_id'] for r in meta['rows']})==24
cfg=read(Q/'FROZEN.json')
for p,h in cfg['pins'].items(): assert sha(p)==h,p
metrics=[]
for run in read(Q/'TRAINING_COMPLETE.json')['rows']:
 for k in ['model','predictions']: assert sha(run[k])==run[k+'_sha256']
 pp=np.load(run['predictions'])['p']; assert pp.shape==arr['y'].shape and np.isfinite(pp).all()
 for role,take in [('fit',arr['train']),('probe',~arr['train'])]:
  for j,f in enumerate(F):
   for h,name in enumerate(['present','no_other_reviewed_action','all_progression_witnesses']):
    use=take&(arr['mask'][:,j,h]>0); y=arr['y'][use,j,h]; p=pp[use,j,h]>=.5
    metrics.append(dict(seed=run['seed'],role=role,family=f,target=name,known=int(use.sum()),positive=int(y.sum()),correct=int((p==y).sum()),tp=int((p&(y==1)).sum()),tn=int((~p&(y==0)).sum())))
comp=read(E/'COMPARISON.json'); assert len(comp['rows'])==246
pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
for r in comp['rows']:
 saved=pm[r['seed'],r['clip_id'],r['kind']]
 for k in ['source','path']: assert sha(saved[k])==saved['source_sha256' if k=='source' else 'sha256']
 zz=np.load(saved['path'])
 for i,c in enumerate(r['conditions']['local_sequence']['cards']):
  spans=zz[f'spans_{i}']; p=zz[f'p_{i}']; assert len(spans)==len(p) and np.isfinite(p).all()
  # Reproduce the fixed argmax from stored probabilities only.
  scores=np.where(p[:,0]>=.5,np.log(np.clip(p,1e-6,1-1e-6)).sum(1),-np.inf)
  eligible=len(scores)>0 and np.isfinite(scores.max())
  if eligible:
   at=int(np.argmax(scores)); assert c['chosen']['span']==spans[at].tolist()
   assert np.allclose(c['chosen']['candidate_probability'],p[at])
  else: assert c['chosen'] is None
  assert c['semantic_completion'] is None
for s in comp['summary']:
 rr=[r for r in comp['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']]
 cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']]
 truth=[c for c in cc if 'truth' in c]
 observed=lambda c:set(c['chosen']['body_observed']) if c['chosen'] else set()
 assert sum(set(c['truth']['witness'])<=observed(c) for c in truth)==s['all_witness']
 assert sum(len(set(c['truth']['wrong_witness'])&observed(c)) for c in truth)==s['wrong_points']
 assert sum(c['chosen'] is None for c in cc)==s['missing']
 assert dict(Counter(o['state'] for r in rr for o in r['conditions'][s['condition']]['order']))==s['order']
shadow=read(E/'shadow_components/RESULTS.json')
for s in shadow['summary']:
 cc=[c for r in shadow['rows'] if r['seed']==s['seed'] and r['kind']==s['kind'] for c in r['conditions'][s['condition']]]
 assert dict(Counter(c['status'] for c in cc))==s['statuses']
 for task in ['shape','direction']: assert dict(Counter(v['state'] for c in cc for v in c['evidence'] if v['task']==task))==s[task]
assert all(c['completion'] is None for r in shadow['rows'] for cc in r['conditions'].values() for c in cc)
assert all(v['state']=='observation_unknown' for r in shadow['rows'] if r['kind']=='all_hand_missing' for cc in r['conditions'].values() for c in cc for v in c['evidence'] if v['task']=='shape')
protected=read(U/'PROTECTIONS.json')
assert all(Path(p).is_file() and sha(p)==h for p,h in protected.items())
write(Q/'VERIFICATION.json',dict(at=stamp(),arithmetic='PASS',new_model_forward=False,source_clips=24,train_persons=[2,4,5,6],probe_persons=[8,9],probe_upstream_exposed=True,metrics=metrics,comparison_rows=len(comp['rows']),fixed_argmax_verified=True,protected_files=len(protected),protected_unchanged=True,reviewer='same Codex separate saved-artifact calculation; not an independent person or KSL expert',private_access=False,final8_access=False,web_changed=False))
print('Saved local sequence artifacts verified; protected',len(protected))
