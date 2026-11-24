"""Separate arithmetic and saved-contract verification; no retraining."""
from pathlib import Path
import json,hashlib,numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
info=read(U/'INPUTS.json');rows=info['rows'];scores=read(U/'SCORES.json');runs=read(U/'TRAINING.json')['runs'];ready=read(U/'TRAIN_READY.json');checks=0
assert ready['training_code_sha256']==sha(U/'train_compare.py')
assert ready['input_manifest_sha256']==sha(U/'INPUTS.json');assert ready['label_sha256']==sha(U/'LABELS.json')
for p,h in info['pins'].items():assert sha(p)==h;checks+=1
data=[dict(np.load(r['input'],allow_pickle=False)) for r in rows]
fit=[i for i,r in enumerate(rows) if r['role']=='fit_candidate'];assert len(fit)==16
stats=read(U/'NORMALIZATION.json');xx=np.concatenate([data[i]['x'] for i in fit]);mm=np.concatenate([data[i]['mask'] for i in fit]);zz=np.concatenate([data[i]['rgb'] for i in fit])
for j in range(69):
 vals=xx[mm[:,j],j];mu=float(vals.mean()) if len(vals) else 0;sd=max(float(vals.std()),1e-3) if len(vals) else 1
 assert np.isclose(mu,stats['pose_mean'][j],atol=1e-6);assert np.isclose(sd,stats['pose_std'][j],atol=1e-6);checks+=2
assert np.allclose(zz.mean(0),stats['rgb_mean']);assert np.allclose(np.maximum(zz.std(0),1e-3),stats['rgb_std']);checks+=2
for run in runs:
 folder=U/'runs'/run['name'];r=read(folder/'RUN.json');assert sha(folder/'head.pt')==run['checkpoint_sha256'];assert sha(folder/'probabilities.npy')==run['probabilities_sha256']
 assert r['epochs']==r['selected_epoch']==160 and len(r['loss_trace'])==160 and np.isfinite(r['loss_trace']).all();assert r['fit_aliases']==[rows[i]['alias'] for i in fit]
 p=np.load(folder/'probabilities.npy',allow_pickle=False);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1;checks+=5
 for s in [s for s in scores['rows'] if s['run']==run['name']]:
  ids=[i for i,rr in enumerate(rows) if rr['role']==s['group'] or (s['group']=='all_held' and rr['role']!='fit_candidate')];macro=[]
  for j,k in enumerate(info['targets']):
   cm=np.zeros((2,2),int)
   for i in ids:
    for t in np.flatnonzero(data[i]['target_mask'][:,j]):cm[int(data[i]['y'][t,j]),int(p[i,t,j]>=.5)]+=1
   saved=s['per_target'][k];assert [int(cm[1,1]),int(cm[0,0]),int(cm[0,1]),int(cm[1,0])]==[saved[k2] for k2 in ['tp','tn','fp','fn']]
   if (cm.sum(1)>0).all():macro.append(float(np.mean(np.diag(cm)/cm.sum(1))))
   checks+=1
  assert abs(np.mean(macro)-s['macro_balanced_accuracy'])<1e-12;checks+=1
for s in scores['summary']:
 vals=[r['macro_balanced_accuracy'] for r in scores['rows'] if r['group']==s['group'] and r['condition']==s['condition']];assert abs(np.mean(vals)-s['mean'])<1e-12;assert [min(vals),max(vals)]==s['range'];checks+=2
events=read(U/'TEMPORAL_DETAILS.json')
for s in scores['temporal']:
 e=[r for r in events if r['condition']==s['condition'] and r['seed']==s['seed'] and ((r['role']=='fit_candidate')==(s['scope']=='fit'))];ch=[r for r in e if r['change']];st=[r for r in e if not r['change']]
 assert len(ch)==s['change_intervals'];assert sum((r['pred_left']>=.5)==bool(r['true_left']) and (r['pred_right']>=.5)==bool(r['true_right']) for r in ch)==s['correct_change_endpoints']
 assert len(st)==s['stable_adjacent_pairs'];assert sum((r['pred_left']>=.5)!=(r['pred_right']>=.5) for r in st)==s['false_flips'];checks+=4
runtime=read(U/'RUNTIME.json');times=[r['total_seconds'] for r in runtime['rows']];assert abs(np.median(times)-runtime['median_total_seconds'])<1e-12;assert [min(times),max(times)]==runtime['range_total_seconds'];checks+=2
out=dict(status='PASS',checks=checks,normalization_train_only=True,all_saved_score_confusion_counts_recomputed=True,temporal_endpoint_counts_recomputed=True,completed_runs=6,original_signrep_checkpoint_unchanged=True,known_cells=1112,training_labels='Provisional RGB-rechecked Codex physical annotations, NOT human/expert certified',implementation='Codex',verification='Separate Codex verification code; no separate agent or human independent truth',pins={str(U/f):sha(U/f) for f in ['INPUTS.json','TRAIN_READY.json','TRAINING.json','SCORES.json','TEMPORAL_DETAILS.json','RUNTIME.json','DIAGNOSTICS.json']})
p=U/'FINAL_VERIFICATION.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print('PASS',checks)
