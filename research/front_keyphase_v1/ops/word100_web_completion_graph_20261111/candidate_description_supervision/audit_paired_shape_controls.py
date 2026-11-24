"""Controlled same-crop, same-interval input audit; no thresholds or training chosen from this."""
from transfer_local_timefix_core import *
import time

def main():
 dest=TQ/'PAIRED_SHAPE_AUDIT.json';assert not dest.exists();torch.set_num_threads(4);torch.set_num_interop_threads(1)
 rp=O/'word100_positive_direction_20261109/COMPARISON.json';records={(r['clip_id'],r['kind']):r for r in read(rp)['rows']};role=load('_paired_shape_role',O/'word100_candidate_role_learning_20261111/role_core.py');frozen=role.frozen_model();base=encoder();nets={}
 for run in read(TQ/'TRAINING_COMPLETE.json')['rows']:
  s=torch.load(run['model'],weights_only=True,map_location='cpu');n=TransferHead(s['mean'].numpy(),s['std'].numpy());n.load_state_dict(s);n.eval();nets[run['seed']]=n
 out=[]
 for cid,kind in records:
  if kind!='right_closed_hand_main':continue
  normal=records[cid,'crop_normal'];changed=records[cid,kind];nd=read(normal['decision'])['after']['cards'][0];span=nd['motion_evidence']['span'];evidence=[];qq=[];body=[]
  for label,src in [('normal',normal),('changed_hand',changed)]:
   assert sha(src['prediction'])==src['sha256'];a=np.load(src['prediction']);x=a['x'];q=role.old.relation_features(x);qn=(q-frozen.qm.numpy())/frozen.qs.numpy();qn=np.concatenate([qn[...,:4]*q[...,8,None],qn[...,4:8]*q[...,9,None],q[...,8:]],-1);feat=encode_windows(base,normalized_pose(x),qn.reshape(len(x),100),np.array([span]));p={}
   for seed,n in nets.items():
    with torch.inference_mode():p[seed]=n(torch.tensor(feat)).sigmoid().numpy()[0,F.index(4)].tolist()
   qq.append(q);body.append(x[:,:18]);evidence.append(dict(condition=label,source=src['prediction'],source_sha256=src['sha256'],span=span,probabilities=p))
  assert np.array_equal(body[0],body[1]);assert np.array_equal(qq[0][...,:4],qq[1][...,:4]);out.append(dict(clip_id=cid,same_crop=True,same_body_coordinates=True,same_body_relation_features=True,max_all_relation_change=float(np.max(abs(qq[0]-qq[1]))),conditions=evidence))
 write(dest,dict(at=stamp(),purpose='Does handshape-only change alter a body-motion candidate score on the exact same observed path?',rows=out,source_span_basis='Previously stored crop-normal movement interval; fixed before candidate predictions, not exact expert motion-boundary truth',no_new_training=True,no_threshold_tuning=True,private=False,final8=False))
 for r in out:print(r)
if __name__=='__main__':main()
