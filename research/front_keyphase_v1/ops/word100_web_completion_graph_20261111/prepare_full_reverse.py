from graph import *
from prepare_linker import load
import numpy as np,torch,sys,time,traceback
O=U.parent;R=O/'word100_candidate_role_learning_20261111';G=O/'word100_guide_requirement_learning_20261111'
def main():
 assert not (U/'FULL_REVERSE_INPUT_COMPLETE.json').exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rc=load('_full_reverse_req',G/'requirement_core.py');role=load('_full_reverse_role',R/'role_core.py');rows={r['clip_id']:r for r in rc.public_rows() if r['role']=='development'};base=rc.frozen_observer();frozen=role.frozen_model();src=[r for r in read(R/'EVALUATION_INPUTS.json')['rows'] if r['kind']=='full_normal']
 write(U/'FULL_REVERSE_FROZEN.json',dict(scope='New whole-recording temporal reverse of12public DEV videos. Compound source-time reversal and direction change; NOT natural learner-error or pure-order linguistic truth.',rationale='Prior partial-core reversal leaves original action fragments; original nine supports cannot all be labelled false.',source_operation='ii=arange(T)[::-1]; x_new=x[ii]; elapsed timestamps remain increasing',direct_truth='Every source anchor pair with a<b maps to T-1-a>T-1-b. This proves reversed SOURCE evidence order only.',model_selection='Existing source-phase network, old role heads and fixed G order heads retained; no training/threshold modification based on new scores.',normal_comparator='saved normal/speed records; no old inference replay',unknown_control='saved all-hand-missing',params=dict(confidence=.65,order_negative=.35)))
 out=[]
 for s in src:
  assert rc.sha(s['path'])==s['sha256'];z=np.load(s['path']);ii=np.arange(len(z['x']))[::-1];x=z['x'][ii].copy();ts=z['ts'].copy();r=rows[s['clip_id']]
  fh,_=role.lc.be.ev.frozen_forward(base,x,r['word']);ob=role.features(frozen,x,ts,fh);p=U/'full_reverse_inputs'/f"{s['clip_id']}.npz";p.parent.mkdir(exist_ok=True)
  np.savez_compressed(p,x=x,ts=ts,frame_id=np.arange(len(x)),source_indices=ii,fh=fh,z=ob['z'])
  out.append(dict(clip_id=s['clip_id'],word_id=s['clip_id'].split('_')[2],path=str(p),sha256=rc.sha(p),normal_path=s['path'],normal_sha256=s['sha256'],source_sha256=s['source_sha256'],frames=len(x)))
  print(s['clip_id'],len(x),flush=True)
 write(U/'FULL_REVERSE_INPUT_COMPLETE.json',dict(rows=out,new_inputs_only=True,public_only=True))
if __name__=='__main__':
 try:main()
 except BaseException:(U/f'FULL_REVERSE_PREP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
