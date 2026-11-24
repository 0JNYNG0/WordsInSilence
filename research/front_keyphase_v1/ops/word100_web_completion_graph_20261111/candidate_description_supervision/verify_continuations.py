"""Saved-artifact audit only: no model forward, training or rescore."""
from core import *
from collections import Counter
checked=[];metrics=[];folders=['direction_neutral_candidates','boundary_direction_repair','dense_window_supervision']
for folder in folders:
 p=W/folder;meta=read(p/'INPUTS_COMPLETE.json');assert sha(meta['arrays'])==meta['arrays_sha256'];a=np.load(meta['arrays']);train=a['train'];ids=[r['person'] for r in meta['rows']];assert {v for v,t in zip(ids,train) if t}=={2,4,5,6};assert {v for v,t in zip(ids,train) if not t}=={8,9};assert set(ids)=={2,4,5,6,8,9}
 for r in read(p/'TRAINING_COMPLETE.json')['rows']:
  assert sha(r['model'])==r['model_sha256'];assert sha(r['predictions'])==r['predictions_sha256'];z=np.load(r['predictions']);pp=z['p'];y=a['y'];m=a['mask'];assert pp.shape==y.shape and np.isfinite(pp).all();use=(~train).reshape((-1,)+(1,)*(y.ndim-1))&(m>0);metrics.append(dict(folder=folder,seed=r['seed'],probe_known_items=int(use.sum()),probe_correct=int(((pp>=.5)==y)[use].sum()),unit='correlated supervised binary items, not clips or feedback accuracy'))
 checked.append(dict(folder=folder,source_clips=len({r['clip_id'] for r in meta['rows']}),windows_or_points=len(train),training_rows=int(train.sum()),probe_rows=int((~train).sum())))
for folder in ['direction_neutral_candidates','boundary_direction_repair/evaluation','dense_proposal_probe','dense_window_supervision/evaluation']:
 p=W/folder;d=read(p/'COMPARISON.json');assert len(d['rows'])==246
 for s in d['summary']:
  rr=[r for r in d['rows'] if r['seed']==s['seed'] and r['kind']==s['kind']];cards=[c for r in rr for c in r['conditions'][s['condition']]['cards']];hit=wrong=0
  for c in cards:
   if 'truth'not in c:continue
   observed=set(c['chosen']['body_observed']) if c['chosen'] else set();hit+=set(c['truth']['witness'])<=observed;wrong+=len(set(c['truth']['wrong_witness'])&observed)
  assert hit==s['all_witness'] and wrong==s['wrong_points'];assert sum(c['chosen'] is None for c in cards)==s['missing'];assert dict(Counter(o['state'] for r in rr for o in r['conditions'][s['condition']]['order']))==s['order']
 for r in d['predictions']:assert sha(r['source'])==r['source_sha256']
 ss=p/'shadow_components/RESULTS.json'
 if ss.exists():
  sh=read(ss)
  for s in sh['summary']:
   cards=[c for r in sh['rows'] if r['seed']==s['seed'] and r['kind']==s['kind'] for c in r['conditions'][s['condition']]];assert dict(Counter(c['status'] for c in cards))==s['statuses']
   for task in ['shape','direction']:assert dict(Counter(v['state'] for c in cards for v in c['evidence'] if v['task']==task))==s[task]
  assert all(c['completion'] is None for r in sh['rows'] for cards in r['conditions'].values() for c in cards)
  assert all(v['state']=='observation_unknown' for r in sh['rows'] if r['kind']=='all_hand_missing' for cards in r['conditions'].values() for c in cards for v in c['evidence'] if v['task']=='shape')
 checked.append(dict(folder=folder,rows=len(d['rows']),summary_recomputed=True,all_completion_claims_disabled=True,artifacts_sha256=sha(p/'COMPARISON.json')))
protected=read(U/'PROTECTIONS.json');assert all(Path(p).exists() and sha(p)==h for p,h in protected.items())
write(W/'CONTINUATION_VERIFICATION.json',dict(at=stamp(),status='ARITHMETIC_AND_PROTECTION_PASS_MODEL_GATE_REWORK',checked=checked,metrics=metrics,protected_files=len(protected),protected_unchanged=True,reviewer='same Codex separate stored-array arithmetic; delegated agents unavailable due quota; not independent human truth',new_inference_calls=0,private_access=False,final8_access=False,web_changed=False))
print('Verified continuation artifacts; protected files',len(protected))
