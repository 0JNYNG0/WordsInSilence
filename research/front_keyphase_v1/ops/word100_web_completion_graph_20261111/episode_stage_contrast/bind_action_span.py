"""Corrected action-span diagnostic: motion and its terminal hold both support an action.

The per-point class probability is evidence about the action, not an exact boundary.
Rank existing candidates by summed log likelihood ratio; no expected-order filter.
This is not a completion/correctness rule and never changes runtime files.
"""
from pathlib import Path
import sys,hashlib
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
F=[2,3,4,5,7];A={2:0,3:1,4:2,5:4,7:3}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 frozen=W/'BINDING_V2_FROZEN.json';out=W/'BINDING_V2_SCORES.json';assert not frozen.exists() and not out.exists()
 paths=[Path(__file__),W/'EVALUATION_COMPLETE.json',U/'boundary_candidate_probe/RESULTS.json',U/'reviews/sequence_target/MOTION_EXTENTS.json',U/'context_role_training/RESULTS.json']
 write(frozen,dict(at=stamp(),rule='Correct aggregation mismatch in v1: select highest sum of log((p_motion+p_hold)/p_other) times observed elapsed frame duration, over unchanged candidate extent. Motion evidence p_motion>=0.5 admits a candidate for comparison; the whole-span likelihood ranks candidates but is not a second detection/completion threshold. Use all candidate types with the correct relation axis/moving side. Existing threshold p_motion>=0.5 must occur at least once. No desired-order or word-name input. No candidate endpoint shrinking/expansion. No semantic completion inference. Body support alone cannot qualify handshape/contact.',numeric_eps=1e-6,selection_threshold=None,scope='Saved exposed DEV diagnostic only. Compare source witness coverage, other-action contamination, normal/speed proposals and all-hand-missing qualification. Not automatic adoption.',pins={str(p):sha(p) for p in paths}))
 candidates={(r['clip_id'],r['kind']):r['candidates'] for r in read(U/'boundary_candidate_probe/RESULTS.json')};base={(r['seed'],r['clip_id'],r['kind']):r for r in read(U/'context_role_training/RESULTS.json') if r['condition']=='source_context_control'};gt={r['clip_id']:r for r in read(U/'reviews/sequence_target/MOTION_EXTENTS.json')['rows']};rows=[]
 for r in read(W/'EVALUATION_COMPLETE.json')['rows']:
  assert sha(r['path'])==r['sha256'] and sha(r['source'])==r['source_sha256'];d=np.load(r['source']);p=np.load(r['path'])['p'];q=d['z'][:,292:392].reshape(-1,10,10);ts=d['ts'];dt=np.r_[np.diff(ts),np.median(np.diff(ts))]/1e6
  for card in base[r['seed'],r['clip_id'],r['kind']]['cards']:
   family=card['family']
   if card['parent'] or family not in F:continue
   j=F.index(family);axis=A[family];valid=q[:,axis*2,8]>.5;hand=q[:,axis*2,9]>.5;pm=p[:,j,1];compatible=np.clip(p[:,j,1]+p[:,j,2],1e-6,1-1e-6);llr=np.log(compatible/(1-compatible));scores=[]
   for c in candidates[r['clip_id'],r['kind']]:
    if c['axis']!=axis or c['moving']!='R':continue
    ix=np.arange(c['a'],c['b']+1);observed=ix[valid[ix]];score=float(np.sum(llr[observed]*dt[observed]));qual=observed[hand[observed]]
    scores.append(dict(span=[c['a'],c['b']],source=c['source'],score=score,has_motion=bool(np.any(pm[observed]>=.5)),body_observed=observed.tolist(),hand_qualified=qual.tolist(),semantic_completion=None))
   eligible=[s for s in scores if s['has_motion']];chosen=max(eligible,key=lambda s:s['score']) if eligible else None
   row=dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],card=card['id'],family=family,old_span=card['span'],chosen=chosen,candidates=scores)
   if r['clip_id'] in gt and r['kind'] in ['full_normal','dynamic_complete_blocks'] and family in [2,3]:
    g=gt[r['clip_id']];key='B' if family==2 else 'C';other='C' if key=='B' else 'B';idx='normal_query_indices' if r['kind']=='full_normal' else 'permuted_query_indices_in_source_order';w=set(g['motions'][key]['definite_moving_witness'][idx]);wrong=set(g['motions'][other]['definite_moving_witness'][idx]);picked=set(chosen['body_observed']) if chosen else set();old=set(card['observed_indices'])
    row['truth']=dict(witness=sorted(w),wrong_witness=sorted(wrong),new_all_witness=w<=picked,old_all_witness=w<=old,new_wrong_points=len(wrong&picked),old_wrong_points=len(wrong&old),semantic_completion=None)
   rows.append(row)
 summary=[]
 for seed in [42,43,44]:
  for kind in ['full_normal','dynamic_complete_blocks']:
   rr=[r for r in rows if r['seed']==seed and r['kind']==kind and 'truth' in r];summary.append(dict(seed=seed,kind=kind,groups=len(rr),old_all_witness=sum(r['truth']['old_all_witness'] for r in rr),new_all_witness=sum(r['truth']['new_all_witness'] for r in rr),old_wrong_points=sum(r['truth']['old_wrong_points'] for r in rr),new_wrong_points=sum(r['truth']['new_wrong_points'] for r in rr)))
  rr=[r for r in rows if r['seed']==seed and r['kind'] in ['full_normal','speed075','speed125']];print('seed',seed,'normal/speed',len(rr),'no new candidate',sum(r['chosen'] is None for r in rr),flush=True)
 missing=[r for r in rows if r['kind']=='all_hand_missing'];assert all(not r['chosen'] or not r['chosen']['hand_qualified'] for r in missing)
 write(out,dict(at=stamp(),summary=summary,rows=rows,all_missing_hand_qualified=0,web_changed=False,model_inference_calls=0,limitations=['Action candidate with compatible motion/terminal hold. Interior evidence does not prove every boundary point; boundaries remain model proposals, not source truth. Shape module unchanged, but effect of choosing a new interval on shape/direction still requires integration regression.','Only3 known DEV source clips have B/C diagnostic witness targets. No new-user accuracy claim.','Synthetic reorder has seams and prior exposure; final8 untouched.','Repeated actions may require multiple candidates, one maximum does not establish full repeated episode.']))
 print(summary)
if __name__=='__main__':main()
