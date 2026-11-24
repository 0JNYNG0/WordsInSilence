from pathlib import Path
from collections import Counter
import sys,time,json,torch,numpy as np
from prepare_real_fit import N,M,P,V,O,read,exclusive
U=O/'word100_learned_feature_retention_20261102';OLD=O/'word100_multipeople_live_feedback_20261101'
sys.path.insert(0,str(M))
from motion_model import features,normalize,predict,public_arrays
from motion_runtime import MotionObserver
from train_motion import scores

class InputMasked(torch.nn.Module):
 def __init__(self,model):
  super().__init__();self.model=model;keep=read(N/'TRAINING_CONTRACT.json')['motion_value_indices'];v=torch.zeros(146);v[keep]=1;v[np.array(keep)+73]=1;self.register_buffer('valid',v)
 def forward(self,x,lengths):return self.model(x*self.valid,lengths)
class RealMotionObserver(MotionObserver):
 def __init__(self,condition,seed):
  super().__init__('ordered_gru',seed);self.model.load_state_dict(torch.load(N/'runs'/f'{condition}_{seed}'/'model.pt',map_location='cpu',weights_only=True));self.condition=condition
  if condition=='real_variants_motion_inputs':self.model=InputMasked(self.model)
  self.model.eval();self.model.requires_grad_(False)

def cases():
 result=[]
 for r in read(P/'INPUT_MANIFEST.json')['rows']:
  if r['role']!='development':continue
  a,z,m=public_arrays(r);s=np.load(OLD/'baseline_probabilities'/f"{r['clip_id']}.npz")['probabilities'];rel=np.load(V/'runs/rgb_geometry_42/predictions'/f"{r['clip_id']}.npz")['probability']
  result.append(dict(id=r['clip_id'],kind='native',a=a,shape=s,relation=rel,base=read(V/'development_feedback'/f"rgb_geometry_42_{r['clip_id']}.json")))
 for r in read(U/'VIDEO_CONTROLS.json')['rows']:
  cid=f"{r['word_id']}_{r['kind']}";a=dict(np.load(U/'actual_controls'/cid/'analysis.npz'));rel=np.load(V/'control_predictions'/f'rgb_geometry_42_{cid}.npz')['probability']
  result.append(dict(id=cid,kind=r['kind'],a=a,shape=a['retained_probability'],relation=rel,base=read(V/'control_feedback'/f'rgb_geometry_42_{cid}.json')))
 for r in read(M/'REVERSE_MANIFEST.json')['rows']:
  d=M/'reverse_extracted'/r['clip_id'];a=dict(np.load(d/'analysis.npz'));result.append(dict(id=r['clip_id']+'_reverse',kind='actual_reverse',a=a,shape=a['retained_probability'],relation=a['relation_probability'],base=read(d/'RESULT.json')))
 return result

def direct_cases():
 ds=read(M/'DATASET.json');old={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']};stats=read(M/'NORMALIZATION.json');cache={};seq=[];y=[];mask=[];indices=[]
 for i,r in enumerate(ds['rows']):
  if r['role']!='development':continue
  cid=r['clip_id']
  if cid not in cache:
   a,z,m=public_arrays(old[cid]);cache[cid]=normalize(z,m,stats)
  seq.append(cache[cid][r['frame_ids']]);y.append(ds['y'][i]);mask.append(ds['mask'][i]);indices.append(i)
 reverse=[]
 for r in read(M/'ACTUAL_REVERSE_EPISODES.json')['rows']:
  a=dict(np.load(M/'reverse_extracted'/r['clip_id']/'analysis.npz'));z,m=features(a,a['retained_probability'],a['relation_probability']);reverse.append(normalize(z,m,stats)[r['frame_ids']])
 gt=np.load(M/'actual_reverse_predictions/ordered_gru_42.npz')
 return seq,np.array(y),np.array(mask),indices,reverse,gt['y'],gt['mask']

def main():
 torch.set_num_threads(4);assert (N/'TRAINING_RESULT.json').exists();assert not (N/'COMPARISON.json').exists();rr=cases();seq,y,mask,ix,reverse,ry,rm=direct_cases();rows=[];direct=[]
 for cond in read(N/'TRAINING_CONTRACT.json')['conditions']:
  for seed in [42,43,44]:
   observer=RealMotionObserver(cond,seed);p=predict(observer.model,seq);rp=predict(observer.model,reverse);d=N/'runs'/f'{cond}_{seed}';np.savez_compressed(d/'development_predictions.npz',p=p,y=y,mask=mask,old_indices=ix)
   np.savez_compressed(d/'actual_reverse_predictions.npz',p=rp,y=ry,mask=rm)
   direct.append(dict(condition=cond,seed=seed,cached=scores(y,mask,p),actual_reverse=scores(ry,rm,rp)))
   for c in rr:
    out=observer.attach(c['base'],c['a'],c['shape'],c['relation']);name=f'{cond}_{seed}_{c["id"]}';exclusive(N/'feedback'/f'{name}.json',out)
    preserved=out['path']==c['base']['path'] and all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],c['base']['stages']))
    assert preserved
    rows.append(dict(condition=cond,seed=seed,case=c['id'],kind=c['kind'],word=out['word_id'],motions=out['motion_observations'],seconds=out['motion_head_seconds'],old_outputs_preserved=preserved,gaps=len(out['evidence_gaps'])))
   print('evaluated',cond,seed,flush=True)
 exclusive(N/'COMPARISON.json',dict(rows=rows,direct=direct,baseline=str(M/'COMPARISON.json'),new_video_extraction=False,new_training_on_development=False))
 summary={}
 for cond in read(N/'TRAINING_CONTRACT.json')['conditions']:
  summary[cond]={kind:dict(Counter(m['status'] for r in rows if r['condition']==cond and r['kind']==kind for m in r['motions'])) for kind in sorted({r['kind'] for r in rows})}
 exclusive(N/'COMPARISON_SUMMARY.json',summary);print(json.dumps(summary,ensure_ascii=False))
 # Fixed primary candidate, not the better-looking condition or seed.
 primary=read(N/'DEPLOYMENT_CANDIDATE.json')['primary_condition'];normals=['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'];pr=[r for r in rows if r['condition']==primary]
 def positive(r):return sum(m['status']=='DIRECTED_MOTION_OBSERVED' for m in r['motions'])
 normal_confirmed=sum(positive(r) for r in pr if r['kind'] in normals);native_confirmed=sum(positive(r) for r in pr if r['kind']=='native')
 cut=[r for r in pr if r['kind']=='action_cut' and r['word']=='WORD1133'];black=[r for r in pr if r['kind']=='main_blackout']
 checks=dict(normal_confirmed_at_least54_of60=normal_confirmed>=54,native_confirmed_at_least40_of45=native_confirmed>=40,
  removed_arm_travel_never_confirmed=all(not positive(r) for r in cut),blackout_never_confirmed=all(not positive(r) and r['gaps']>0 for r in black),all_prior_outputs_preserved=all(r['old_outputs_preserved'] for r in rows))
 exclusive(N/'DEVELOPMENT_GATE.json',dict(status='PASS_TO_LOCKED_EVALUATION' if all(checks.values()) else 'NOT_READY_FOR_WEB_DEFAULT',primary=primary,checks=checks,
  normal_confirmed=normal_confirmed,native_confirmed=native_confirmed,old_normal_confirmed=47,old_native_confirmed=40,
  no_best_seed_selection=True,locked_evaluation_not_run=True))
 print('gate',checks,flush=True)
if __name__=='__main__':main()
