from collections import Counter
import numpy as np,torch
from rgb_model import R,Q,N,M,P,read,predict,rgb_input
from rgb_runtime import RGBMotionObserver
from prepare_real_fit import exclusive
from evaluate_real_motion import cases,direct_cases

def rgb_direct_cases():
 seq,y,mask,indices,rev,ry,rm=direct_cases();rows=read(M/'DATASET.json')['rows'];public={r['clip_id']:r for r in read(P/'INPUT_MANIFEST.json')['rows']}
 stats=read(R/'RGB_NORMALIZATION.json');cache={};out=[]
 for s,i in zip(seq,indices):
  r=rows[int(i)];cid=r['clip_id']
  if cid not in cache:cache[cid]=rgb_input(dict(np.load(public[cid]['input'])),stats)
  video=cache[cid][r['frame_ids']];assert len(video)==len(s);out.append(np.concatenate([s,video],1))
 rout=[]
 for s,r in zip(rev,read(M/'ACTUAL_REVERSE_EPISODES.json')['rows']):
  cid=r['clip_id']+'_reverse'
  if cid not in cache:cache[cid]=rgb_input(dict(np.load(M/'reverse_extracted'/r['clip_id']/'analysis.npz')),stats)
  video=cache[cid][r['frame_ids']];assert len(video)==len(s);rout.append(np.concatenate([s,video],1))
 return out,y,mask,indices,rout,ry,rm

def main():
 torch.set_num_threads(4);assert (R/'TRAINING_RESULT.json').exists();assert not (R/'COMPARISON.json').exists()
 cc=cases();seq,y,m,ix,rev,ry,rm=rgb_direct_cases();rows=[]
 for condition in read(R/'PLAN.json')['conditions']:
  for seed in [42,43,44]:
   obs=RGBMotionObserver(condition,seed);d=R/'runs'/f'{condition}_{seed}'
   np.savez_compressed(d/'development_predictions.npz',p=predict(obs.model,seq),y=y,mask=m,old_indices=ix)
   np.savez_compressed(d/'actual_reverse_predictions.npz',p=predict(obs.model,rev),y=ry,mask=rm)
   for c in cc:
    out=obs.attach(c['base'],c['a'],c['shape'],c['relation']);assert out['path']==c['base']['path']
    assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],c['base']['stages']))
    exclusive(R/'feedback'/f'{condition}_{seed}_{c["id"]}.json',out)
    rows.append(dict(condition=condition,seed=seed,case=c['id'],kind=c['kind'],word=out['word_id'],motions=out['motion_observations'],seconds=out['motion_head_seconds'],old_outputs_preserved=True,gaps=len(out['evidence_gaps'])))
   print('evaluated',condition,seed,flush=True)
 exclusive(R/'COMPARISON.json',dict(rows=rows,baseline=str(Q/'COMPARISON.json'),new_video_extraction=False))
 summary={c:{k:dict(Counter(m['status'] for r in rows if r['condition']==c and r['kind']==k for m in r['motions'])) for k in sorted({r['kind'] for r in rows})} for c in read(R/'PLAN.json')['conditions']}
 exclusive(R/'COMPARISON_SUMMARY.json',summary)
 primary=read(R/'PLAN.json')['primary'];rr=[r for r in rows if r['condition']==primary]
 def positive(r):return sum(m['status']=='DIRECTED_MOTION_OBSERVED' for m in r['motions'])
 normal=sum(positive(r) for r in rr if r['kind'] in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']);native=sum(positive(r) for r in rr if r['kind']=='native')
 cut=[r for r in rr if r['kind']=='action_cut' and r['word']=='WORD1133'];black=[r for r in rr if r['kind']=='main_blackout']
 checks=dict(normal_confirmed_at_least54_of60=normal>=54,native_confirmed_at_least40_of45=native>=40,
  removed_arm_travel_never_confirmed=all(not positive(r) for r in cut),blackout_never_confirmed=all(not positive(r) and r['gaps']>0 for r in black),all_prior_outputs_preserved=True)
 exclusive(R/'DEVELOPMENT_GATE.json',dict(status='PASS_TO_LOCKED_EVALUATION' if all(checks.values()) else 'NOT_READY_FOR_WEB_DEFAULT',primary=primary,checks=checks,normal_confirmed=normal,native_confirmed=native,no_best_seed_selection=True,locked_evaluation_not_run=True))
 print(summary);print('Gate',normal,native,checks)

if __name__=='__main__':main()
