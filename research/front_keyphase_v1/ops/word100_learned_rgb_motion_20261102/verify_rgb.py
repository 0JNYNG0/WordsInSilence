from collections import Counter
import torch,numpy as np
from rgb_model import R,Q,N,M,P,O,RGBMotionNet,read
from prepare_real_fit import sha,exclusive,V
from verify_saved import balanced

def main():
 rows=read(R/'COMPARISON.json')['rows'];assert len(rows)==288
 assert len({(r['condition'],r['seed'],r['case']) for r in rows})==288
 plan=read(R/'PLAN.json');assert sha(N/'DATASET.json')==plan['data_sha256'];scores=[];updates=[];summary={}
 for cond in plan['conditions']:
  summary[cond]={k:dict(Counter(m['status'] for r in rows if r['condition']==cond and r['kind']==k for m in r['motions'])) for k in sorted({r['kind'] for r in rows})}
  for seed in [42,43,44]:
   d=R/'runs'/f'{cond}_{seed}';r=read(d/'COMPLETE.json');assert sha(d/'model.pt')==r['checkpoint_sha256']
   assert sha(Q/'runs'/f'separated_relative_{seed}/model.pt')==r['initial_sha256']
   model=RGBMotionNet(cond);model.initialize(seed);initial=model.state_dict();final=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
   delta=sum(float((v-initial[k]).square().sum()) for k,v in final.items())**.5;assert abs(delta-r['weight_update_L2'])<1e-8
   visual=float(final['direction.rgb.weight'].norm()+final['continuity.rgb.weight'].norm())
   assert visual==0 if cond=='no_rgb_replay' else visual>0
   updates.append(dict(condition=cond,seed=seed,total_update_L2=delta,rgb_update_norm=visual))
   a=np.load(d/'development_predictions.npz');b=np.load(d/'actual_reverse_predictions.npz');ds=read(M/'DATASET.json')['rows'];native=np.array([ds[int(i)]['kind'] in ['native_directed','native_stable'] for i in a['old_indices']]);assert native.sum()==27
   yy=np.concatenate([a['y'][native],b['y']]);mm=np.concatenate([a['mask'][native],b['mask']]);pp=np.concatenate([a['p'][native],b['p']]);score=balanced(yy,mm,pp)
   scores.append(dict(condition=cond,seed=seed,per_target_balanced_accuracy=score,mean_balanced_accuracy=float(np.mean(score))))
 for r in rows:
  out=read(R/'feedback'/f"{r['condition']}_{r['seed']}_{r['case']}.json")
  if r['kind']=='native':base=read(V/'development_feedback'/f"rgb_geometry_42_{r['case']}.json")
  elif r['kind']=='actual_reverse':base=read(M/'reverse_extracted'/r['case'].removesuffix('_reverse')/'RESULT.json')
  else:base=read(V/'control_feedback'/f"rgb_geometry_42_{r['case']}.json")
  assert out['path']==base['path'] and len(out['stages'])==len(base['stages'])
  assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],base['stages']))
  assert out['motion_observations']==r['motions'] and len(out['evidence_gaps'])==r['gaps']
 assert summary==read(R/'COMPARISON_SUMMARY.json')
 for p,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(p)==h
 primary=plan['primary'];rr=[r for r in rows if r['condition']==primary]
 def count(kinds):return sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in rr if r['kind'] in kinds for m in r['motions'])
 counts=dict(normal=count(['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']),native=count(['native']),blackout=count(['main_blackout']),
  removed_arm=sum(m['status']=='DIRECTED_MOTION_OBSERVED' for r in rr if r['kind']=='action_cut' and r['word']=='WORD1133' for m in r['motions']))
 gate=read(R/'DEVELOPMENT_GATE.json');assert counts['normal']==gate['normal_confirmed'] and counts['native']==gate['native_confirmed']
 exclusive(R/'SAVED_VERIFICATION.json',dict(status='PASS_SAVED_ARITHMETIC_AND_PRESERVATION',counts=counts,physical_episode_scores=scores,weight_updates=updates,
  motion_head_ms_median=float(np.median([r['seconds'] for r in rr])*1000),old_models_outputs_preserved=True,
  verifier='Same Codex separate program; not independent human/KSL ground truth',new_inference=False,no_private_data_access=True))
 print(counts);print(scores)

if __name__=='__main__':main()
