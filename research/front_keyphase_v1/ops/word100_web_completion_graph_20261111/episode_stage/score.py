"""Saved-output arithmetic only. Motion evidence is not semantic completion."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
from occurrence_binding import contiguous_runs,union_indices
FAMILIES=[2,3,4,5,7];AXIS={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def propose(p,z,family):
 j=FAMILIES.index(family);q=z[:,292:392].reshape(-1,10,10);valid=q[:,AXIS[family]*2,8]>.5
 return dict(motion_intervals=contiguous_runs((p[:,j,1]>=.5)&valid),hold_intervals=contiguous_runs((p[:,j,2]>=.5)&valid),semantic_completion=None)

def main():
 assert not (W/'SCORES.json').exists();ev=read(W/'EVALUATION_COMPLETE.json');base={(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='source_context_control'};gt={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']};targets=[];behaviors=[]
 for r in ev['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256'];prob=np.load(r['path'])['p'];z=np.load(r['source'])['z'];old=base[r['seed'],r['clip_id'],r['kind']]
  for card in old['cards']:
   if card['parent'] or card['family'] not in FAMILIES:continue
   proposal=propose(prob,z,card['family']);motion=set(union_indices(proposal['motion_intervals']));hold=set(union_indices(proposal['hold_intervals']));oldset=set(card['observed_indices']);behaviors.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=card['id'],family=card['family'],old_span=card['span'],old_samples=len(oldset),new_motion=len(motion),new_hold=len(hold),motion_inside_old=len(motion&oldset),hold_inside_old=len(hold&oldset),**proposal))
   if r['clip_id'] not in gt or r['kind'] not in ['full_normal','dynamic_complete_blocks'] or card['family'] not in [2,3]:continue
   g=gt[r['clip_id']];key='B' if card['family']==2 else 'C';other='C' if key=='B' else 'B';m=g['motions'][key];idxkey='normal_query_indices' if r['kind']=='full_normal' else 'permuted_query_indices_in_source_order';witness=set(m['definite_moving_witness'][idxkey]);wrong=set(g['motions'][other]['definite_moving_witness'][idxkey]);held=set(i for h in m['post_motion_hold_ranges'] for i in h[idxkey]);on=set(m['onset_uncertainty'][idxkey]);off=set(m['offset_uncertainty'][idxkey]);runs=proposal['motion_intervals']
   targets.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],action=key,witnesses=sorted(witness),old_motion_witness=len(oldset&witness),new_motion_witness=len(motion&witness),witness_count=len(witness),old_opposite_witness=len(oldset&wrong),new_opposite_witness=len(motion&wrong),old_hold_in_action=len(oldset&held),new_hold_as_motion=len(motion&held),held_samples=len(held),hold_correct=len(hold&held),onset_supported=bool(motion&on),offset_supported=bool(motion&off),one_run_covers_all_witnesses=any(witness<=set(range(a,b+1)) for a,b in runs),proposal=proposal))
 summaries=[]
 for seed in [42,43,44]:
  for kind in ['full_normal','dynamic_complete_blocks']:
   rr=[r for r in targets if r['seed']==seed and r['kind']==kind];summaries.append(dict(seed=seed,kind=kind,groups=len(rr),old_groups_with_all_witnesses=sum(r['old_motion_witness']==r['witness_count'] for r in rr),new_groups_with_all_witnesses=sum(r['new_motion_witness']==r['witness_count'] for r in rr),old_opposite_action_points=sum(r['old_opposite_witness'] for r in rr),new_opposite_action_points=sum(r['new_opposite_witness'] for r in rr),hold_mislabeled_motion=sum(r['new_hold_as_motion'] for r in rr),hold_total=sum(r['held_samples'] for r in rr),hold_correct=sum(r['hold_correct'] for r in rr),onset_and_offset_support=sum(r['onset_supported'] and r['offset_supported'] for r in rr)))
  normal=[r for r in behaviors if r['seed']==seed and r['kind'] in ['full_normal','speed075','speed125']];missing=[r for r in behaviors if r['seed']==seed and r['kind']=='all_hand_missing'];print(seed,'normal/speed',len(normal),'old-supported-with-no-motion',sum(r['old_samples']>0 and r['new_motion']==0 for r in normal),'all_missing_motion',sum(r['new_motion'] for r in missing))
 write(W/'SCORES.json',dict(at=stamp(),motion_comparison=summaries,targets=targets,behaviors=behaviors,limits=['Only3 existing DEV source videos provide B/C motion/hold diagnostic truth. Other expressions are behavior regression screens, not accuracy labels.','A source witness is relative observed movement, not a complete sign or finger/contact truth.','New motion mask is compared to old whole action interval including hold; separating hold is intentional, not proof old selector made a handshape error.','Synthetic reordering is diagnostic; splice artifacts and source boundary uncertainty remain.','No policy adopted; G2 remains unpassed.'],web_changed=False));print(json.dumps(summaries,ensure_ascii=False))
if __name__=='__main__':main()
