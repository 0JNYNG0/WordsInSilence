"""Separate arithmetic/provenance audit of saved outputs; no new inference."""
from collections import Counter
import numpy as np
import torch
from prepare_real_fit import N,M,P,V,O,read,sha,exclusive

def balanced(y,m,p):
 values=[]
 for j in range(5):
  recalls=[]
  for c in [0,1]:
   q=m[:,j]&(y[:,j]==c)
   if q.any():recalls.append(float(((p[q,j]>=.5)==c).mean()))
  values.append(float(np.mean(recalls)) if recalls else None)
 return values

def main():
 results=read(N/'COMPARISON.json');rows=results['rows'];tr=read(N/'TRAINING_RESULT.json');ds=read(N/'DATASET.json')
 assert len(rows)==288 and len(tr['runs'])==6
 assert len({(r['condition'],r['seed'],r['case']) for r in rows})==288
 assert {r['person'] for r in ds['rows']}==set(range(1,10))
 summary={};physical=[];latency=[]
 for condition in read(N/'TRAINING_CONTRACT.json')['conditions']:
  rr=[r for r in rows if r['condition']==condition]
  summary[condition]={k:dict(Counter(m['status'] for r in rr if r['kind']==k for m in r['motions'])) for k in sorted({r['kind'] for r in rr})}
  for seed in [42,43,44]:
   d=N/'runs'/f'{condition}_{seed}';run=read(d/'COMPLETE.json')
   assert run['checkpoint_sha256']==sha(d/'model.pt')
   initial=M/'runs'/f'ordered_gru_{seed}/model.pt'
   assert run['initialized_from_sha256']==sha(initial)
   old=torch.load(initial,map_location='cpu',weights_only=True);new=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
   delta=sum(float(((new[k]-old[k])**2).sum()) for k in old)**.5
   assert delta>0 and abs(delta-run['weight_update_L2'])<1e-8
   a=np.load(d/'development_predictions.npz');b=np.load(d/'actual_reverse_predictions.npz')
   original=read(M/'DATASET.json')['rows'];idx=a['old_indices'];native=np.array([original[int(i)]['kind'] in ['native_directed','native_stable'] for i in idx]);assert native.sum()==27
   yy=np.concatenate([a['y'][native],b['y']]);mm=np.concatenate([a['mask'][native],b['mask']]);pp=np.concatenate([a['p'][native],b['p']])
   physical.append(dict(condition=condition,seed=seed,original_rows=int(native.sum()),actual_reverse_rows=len(b['p']),
    per_target_balanced_accuracy=balanced(yy,mm,pp),mean_balanced_accuracy=float(np.mean(balanced(yy,mm,pp)))))
 for r in rows:
  f=read(N/'feedback'/f"{r['condition']}_{r['seed']}_{r['case']}.json")
  if r['kind']=='native':base=read(V/'development_feedback'/f"rgb_geometry_42_{r['case']}.json")
  elif r['kind']=='actual_reverse':base=read(M/'reverse_extracted'/r['case'].removesuffix('_reverse')/'RESULT.json')
  else:base=read(V/'control_feedback'/f"rgb_geometry_42_{r['case']}.json")
  assert f['path']==base['path'] and len(f['stages'])==len(base['stages'])
  assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(f['stages'],base['stages']))
  assert r['motions']==f['motion_observations'] and len(f['evidence_gaps'])==r['gaps']
  assert r['old_outputs_preserved'] and all(m['whole_sign_error'] is False for m in r['motions'])
  latency.append(r['seconds'])
 assert summary==read(N/'COMPARISON_SUMMARY.json')
 for path,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(path)==h
 primary=read(N/'DEPLOYMENT_CANDIDATE.json')['primary_condition'];rr=[r for r in rows if r['condition']==primary]
 def count(kinds):return sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in rr if r['kind'] in kinds for m in r['motions'])
 counts=dict(normal=count(['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']),native=count(['native']),
  removed_arm=sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in rr if r['kind']=='action_cut' and r['word']=='WORD1133' for m in r['motions']),
  blackout=count(['main_blackout']))
 gate=read(N/'DEVELOPMENT_GATE.json');assert counts['normal']==gate['normal_confirmed'] and counts['native']==gate['native_confirmed']
 exclusive(N/'SAVED_VERIFICATION.json',dict(status='PASS_SAVED_ARITHMETIC_AND_HASHES',comparisons=len(rows),training_runs=6,counts=counts,
  physical_episode_scores=physical,median_motion_head_ms=float(np.median(latency)*1000),old_pins_preserved=True,
  implementer='Codex',verifier='Same Codex separate saved-output program; not independent human annotation',
  no_reinference=True,no_private_data=True,scope='Development controls, not independent KSL semantic error accuracy'))
 print('Verified',counts)

if __name__=='__main__':main()
