import torch,numpy as np
from collections import Counter
from path_model import Q,N,PathNet,read,predict
from prepare_real_fit import exclusive
from evaluate_real_motion import cases,direct_cases
from motion_runtime import MotionObserver

class PathObserver(MotionObserver):
 def __init__(self,condition,seed):
  super().__init__('ordered_gru',seed);self.model=PathNet(condition);self.model.load_state_dict(torch.load(Q/'runs'/f'{condition}_{seed}/model.pt',map_location='cpu',weights_only=True))
  self.model.eval();self.model.requires_grad_(False);self.condition=condition

def main():
 torch.set_num_threads(4);assert (Q/'TRAINING_RESULT.json').exists();assert not (Q/'COMPARISON.json').exists()
 cc=cases();seq,y,m,ix,rev,ry,rm=direct_cases();rows=[]
 for condition in read(Q/'CONTRACT.json')['conditions']:
  for seed in [42,43,44]:
   obs=PathObserver(condition,seed);d=Q/'runs'/f'{condition}_{seed}'
   np.savez_compressed(d/'development_predictions.npz',p=predict(obs.model,seq),y=y,mask=m,old_indices=ix)
   np.savez_compressed(d/'actual_reverse_predictions.npz',p=predict(obs.model,rev),y=ry,mask=rm)
   for c in cc:
    out=obs.attach(c['base'],c['a'],c['shape'],c['relation']);assert out['path']==c['base']['path']
    assert all(s['features']==b['features'] and s['relations']==b['relations'] for s,b in zip(out['stages'],c['base']['stages']))
    exclusive(Q/'feedback'/f'{condition}_{seed}_{c["id"]}.json',out)
    rows.append(dict(condition=condition,seed=seed,case=c['id'],kind=c['kind'],word=out['word_id'],motions=out['motion_observations'],seconds=out['motion_head_seconds'],old_outputs_preserved=True,gaps=len(out['evidence_gaps'])))
   print('evaluated',condition,seed,flush=True)
 exclusive(Q/'COMPARISON.json',dict(rows=rows,baseline=str(N/'COMPARISON.json'),new_video_extraction=False))
 summary={c:{k:dict(Counter(m['status'] for r in rows if r['condition']==c and r['kind']==k for m in r['motions'])) for k in sorted({r['kind'] for r in rows})} for c in read(Q/'CONTRACT.json')['conditions']}
 exclusive(Q/'COMPARISON_SUMMARY.json',summary)
 primary=read(Q/'CONTRACT.json')['primary'];rr=[r for r in rows if r['condition']==primary]
 def positive(r):return sum(m['status']=='DIRECTED_MOTION_OBSERVED' for m in r['motions'])
 normal=sum(positive(r) for r in rr if r['kind'] in ['reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']);native=sum(positive(r) for r in rr if r['kind']=='native')
 cut=[r for r in rr if r['kind']=='action_cut' and r['word']=='WORD1133'];black=[r for r in rr if r['kind']=='main_blackout']
 checks=dict(normal_confirmed_at_least54_of60=normal>=54,native_confirmed_at_least40_of45=native>=40,
  removed_arm_travel_never_confirmed=all(not positive(r) for r in cut),blackout_never_confirmed=all(not positive(r) and r['gaps']>0 for r in black),all_prior_outputs_preserved=True)
 exclusive(Q/'DEVELOPMENT_GATE.json',dict(status='PASS_TO_LOCKED_EVALUATION' if all(checks.values()) else 'NOT_READY_FOR_WEB_DEFAULT',primary=primary,checks=checks,normal_confirmed=normal,native_confirmed=native,no_best_seed_selection=True,locked_evaluation_not_run=True))
 print(summary);print('Gate',normal,native,checks)

if __name__=='__main__':main()
