"""FIT-only executable checks before optimizer updates; not a new evaluation."""
import hashlib
import numpy as np
import torch
from real_motion import N,M,read,MotionNet,batch,filter_input,records_and_sequences
from prepare_real_fit import exclusive,sha

def main():
 torch.set_num_threads(2)
 seq,y,mask,rows,groups=records_and_sequences()
 assert len(seq)==len(rows)==len(y)==len(mask)
 assert {r['person'] for r in rows}==set(range(1,10))
 assert len(groups)==81 and all(len(g)==4 for g in groups)
 assert all(np.isfinite(x).all() and x.shape[1]==146 and len(x)>=2 for x in seq)
 assert np.isin(y,[0,1]).all() and mask.dtype==bool
 assert all(np.array_equal(y[g[0]],y[i]) and np.array_equal(mask[g[0]],mask[i]) for g in groups for i in g)
 stopped=[i for i,r in enumerate(rows) if r['kind']=='feature_stopped']
 assert all(np.array_equal(seq[i],np.repeat(seq[i][:1],len(seq[i]),axis=0)) and not y[i,:5].any() for i in stopped)
 cut=[i for i,r in enumerate(rows) if r['kind']=='feature_middle_omitted']
 assert all(not mask[i,:5].any() and mask[i,5] and y[i,5]==0 for i in cut)
 model=MotionNet('ordered_gru');model.load_state_dict(torch.load(M/'runs/ordered_gru_42/model.pt',map_location='cpu',weights_only=True));model.eval()
 data=[seq[0],seq[1]];x,l=batch(data)
 with torch.inference_mode():
  a=model(x,l);b=model(torch.cat([x,torch.randn(len(x),17,146)],dim=1),l)
 assert torch.allclose(a,b,atol=1e-6)
 keep=read(N/'TRAINING_CONTRACT.json')['motion_value_indices'];mult=np.zeros(146,np.float32);mult[keep]=1;mult[np.array(keep)+73]=1
 assert all(np.array_equal(filter_input(s,'real_variants_motion_inputs'),s*mult) for s in seq)
 contract=read(N/'TRAINING_CONTRACT.json')
 assert sha(M/'MOTION_TARGETS.json')==contract['motion_labels_sha256']
 exclusive(N/'PREFLIGHT.json',dict(status='PASS_FIT_ONLY',windows=len(rows),reviewed_fit_episodes=len(groups),
  real_variants_per_episode=4,stopped_controls=len(stopped),middle_omitted_controls=len(cut),fit_recorded_ids=list(range(1,10)),
  unknown_motion_labels_not_set_to_wrong=True,padding_invariance=True,training_runtime_mask_equivalence=True,
  source_label_hash_preserved=True,private_data_access=False,development_predictions_read=False))
 print('PASS FIT preflight:',len(rows),'windows;',len(groups),'reviewed events')

if __name__=='__main__':main()
