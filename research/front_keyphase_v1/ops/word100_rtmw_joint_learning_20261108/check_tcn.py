from common import *
import numpy as np,torch
from train_tcn import batch,summary
from pilot.model_masked_v1 import MaskedSignTCN
def main():
 assert not (U/'TCN_CODE_CHECK.json').exists();f=read(U/'FROZEN.json');cfg=dict(f['tcn']['model']);cfg.pop('type');torch.manual_seed(42);torch.set_num_threads(4)
 m=MaskedSignTCN(**cfg).eval();a=np.random.default_rng(42).normal(size=(31,280)).astype(np.float32);b=np.zeros((70,280),np.float32)
 r=dict(full=a,upper_mask=a*.5,y=7,parity=0);s=dict(full=b,upper_mask=b,y=3,parity=1)
 with torch.inference_mode():
  x,p,y=batch([r]);u=m(x,p)[0];x,p,y=batch([r,s]);v=m(x,p)[0];assert torch.allclose(u,v,atol=1e-5),'Padding leakage'
 x,p,y=batch([r],epoch=1);assert np.array_equal(x.numpy()[0],a*.5)
 q=summary([dict(person=11,logits=None,top1_correct=False,top3_correct=False)]);assert q['total']==q['input_failures']==1 and q['top1']==0
 ck=torch.load(O/'word100_body_boundary_20261017/training_C_20261013_best.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['state_dict'],strict=True)
 assert ck['word_codes']==f['classes'];assert sum(p.numel() for p in m.parameters())==37636
 write(U/'TCN_CODE_CHECK.json',dict(status='PASS',tests=['padding invariance','deterministic view augmentation','failed inputs denominator','strict C transfer dimensions/class order'],
  pins={str(U/n):sha(U/n) for n in ['train_tcn.py','check_tcn.py']},frozen_sha256=sha(U/'FROZEN.json')))
 event('tcn_checks_pass')
if __name__=='__main__':main()
