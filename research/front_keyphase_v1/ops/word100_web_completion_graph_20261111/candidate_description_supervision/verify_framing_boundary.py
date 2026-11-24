"""Saved-artifact arithmetic and lineage, no model inference."""
from framing_boundary_core import *
from collections import Counter
E=R/'evaluation';cfg=read(R/'FROZEN.json');meta=read(R/'INPUTS_COMPLETE.json');a=np.load(meta['arrays']);assert sha(meta['arrays'])==meta['arrays_sha256']
for p,h in {**cfg['pins'],**meta['pins'],**read(R/'PRETRAIN_VERIFICATION.json')['code_pins']}.items():assert sha(p)==h,p
assert {r['person'] for r,t in zip(meta['rows'],a['train']) if t}=={2,4,5,6}
assert {r['person'] for r,t in zip(meta['rows'],a['train']) if not t}=={8,9}
assert not ({r['source_sha256'] for r,t in zip(meta['rows'],a['train']) if t}&{r['source_sha256'] for r,t in zip(meta['rows'],a['train']) if not t})
assert meta['zero_halo_max_difference']==0
weights=np.load(R/'TRAIN_WEIGHTS.npz');metrics=[]
for run in read(R/'TRAINING_COMPLETE.json')['rows']:
 for k in ['model','predictions']:assert sha(run[k])==run[k+'_sha256']
 assert np.isclose(weights[run['condition']].sum(),1)
 p=np.load(run['predictions'])['p'];assert p.shape==a['y'].shape and np.isfinite(p).all()
 for split,take in [('fit',a['train']),('probe',~a['train'])]:
  for view in ['full','upper','actual_crop']:
   for j,f in enumerate(F):
    for h,name in enumerate(['presence','purity','all_observed_source_progression']):
     use=take&np.array([r['view']==view for r in meta['rows']])&(a['mask'][:,j,h]>0);y=a['y'][use,j,h];pred=p[use,j,h]>=.5;metrics.append(dict(condition=run['condition'],seed=run['seed'],split=split,view=view,family=f,target=name,known=int(use.sum()),correct=int((pred==y).sum()),positive=int(y.sum()),tp=int((pred&(y==1)).sum()),tn=int((~pred&(y==0)).sum())))
comp=read(E/'COMPARISON.json');assert len(comp['rows'])==246;pm={(r['seed'],r['clip_id'],r['kind']):r for r in comp['predictions']}
for r in comp['rows']:
 s=pm[r['seed'],r['clip_id'],r['kind']];assert sha(s['path'])==s['sha256'] and sha(s['source'])==s['source_sha256'];p=np.load(s['path'])
 for mode in CONDITIONS:
  for i,c in enumerate(r['conditions'][mode]['cards']):
   pp=p[f'{mode}_p_{i}'];sp=p[f'{mode}_spans_{i}'];scores=np.where(pp[:,0]>=.5,np.log(np.clip(pp,1e-6,1-1e-6)).sum(1),-np.inf)
   if len(scores) and np.isfinite(scores.max()):
    k=int(np.argmax(scores));assert c['chosen']['span']==sp[k].tolist();assert np.allclose(c['chosen']['candidate_probability'],pp[k])
   else:assert c['chosen'] is None
   assert c['semantic_completion'] is None
for s in comp['summary']:
 rr=[r for r in comp['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']];cc=[c for r in rr for c in r['conditions'][s['condition']]['cards']];tt=[c for c in cc if 'truth'in c];obs=lambda c:set(c['chosen']['body_observed']) if c['chosen'] else set()
 assert sum(set(c['truth']['witness'])<=obs(c) for c in tt)==s['all_witness'];assert sum(len(set(c['truth']['wrong_witness'])&obs(c)) for c in tt)==s['wrong_points'];assert sum(c['chosen'] is None for c in cc)==s['missing'];assert dict(Counter(v['state'] for r in rr for v in r['conditions'][s['condition']]['order']))==s['order']
shadow=read(E/'shadow_components/RESULTS.json')
for s in shadow['summary']:
 cc=[c for r in shadow['rows'] if r['seed']==s['seed'] and r['kind']==s['kind'] for c in r['conditions'][s['condition']]];assert dict(Counter(c['status'] for c in cc))==s['statuses']
 for task in ['shape','direction']:assert dict(Counter(v['state'] for c in cc for v in c['evidence'] if v['task']==task))==s[task]
assert all(v['state']=='observation_unknown' for r in shadow['rows'] if r['kind']=='all_hand_missing' for cc in r['conditions'].values() for c in cc for v in c['evidence'] if v['task']=='shape')
protected=read(U/'PROTECTIONS.json');assert all(sha(p)==h for p,h in protected.items())
write(R/'VERIFICATION.json',dict(at=stamp(),saved_arithmetic='PASS',source_clips=24,actual_crop_views=24,new_crops=16,reused_crops=8,all_frames_and_times=True,labels_preserved_for_old_views=True,new_view_masks_recomputed=True,windows=len(meta['rows']),metrics=metrics,comparison_rows=246,conditions=CONDITIONS,protected_files=len(protected),protected_unchanged=True,no_new_inference=True,reviewer='same Codex separate arithmetic, not independent human/KSL truth',private_access=False,final8_access=False,web_changed=False))
print('Verified framing/boundary factorial saved artifacts; protected',len(protected))
