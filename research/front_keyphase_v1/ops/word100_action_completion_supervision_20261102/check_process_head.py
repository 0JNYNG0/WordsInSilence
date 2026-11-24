"""Synthetic architecture invariants; no real-clip training or prediction."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from process_head import ProcessEvidenceNet,make_input,batch,INPUT_DIM
U=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.manual_seed(42)
rng=np.random.default_rng(42)
def arrays(n):
 return dict(x=rng.normal(size=(n,69)).astype('float32'),mask=np.ones((n,69),bool),
  rgb=rng.normal(size=(n,768)).astype('float32'),image_available=np.ones(n,bool),
  frame_id=np.arange(n),timestamp_us=np.arange(n)*33333)
stats=dict(mean=np.zeros(768).tolist(),std=np.ones(768).tolist())
a=arrays(12);base=make_input(a,stats)
other={k:v.copy() for k,v in a.items()};other['x'][:,59:]=1e8
other['frame_id']+=100;other['timestamp_us']*=3
assert np.array_equal(base,make_input(other,stats))
a['image_available'][4:7]=False
b={k:v.copy() for k,v in a.items()};b['x'][4:7]=1e7;b['rgb'][4:7]=-1e7
assert np.array_equal(make_input(a,stats),make_input(b,stats))
model=ProcessEvidenceNet().eval()
short=make_input(a,stats);long=make_input(arrays(41),stats)
x,l=batch([short]);q=torch.tensor([0])
with torch.no_grad():
 single=model(x,l,q)
 xb,lb=batch([short,long]);both=model(xb,lb,torch.tensor([0,2]))
 assert torch.allclose(single['process_logits'][0],both['process_logits'][0],atol=2e-7)
 assert torch.allclose(single['point_logits'][0],both['point_logits'][0,:12],atol=2e-7)
 xb[0,12:]=1e7;changed=model(xb,lb,torch.tensor([0,2]))
 assert torch.equal(changed['process_logits'][0],both['process_logits'][0])
 assert torch.equal(changed['point_logits'][0,:12],both['point_logits'][0,:12])
 for n in [2,8,15,16,40]:
  xx,ll=batch([make_input(arrays(n),stats)])
  out=model(xx,ll,q)
  assert out['point_logits'].shape==(1,n,2) and out['process_logits'].shape==(1,)
  assert all(torch.isfinite(v).all() for v in out.values())
# Unknown targets must contribute neither a target value nor a gradient.
logits=torch.tensor([.1,.2,.3],requires_grad=True);known=torch.tensor([True,False,True])
target=torch.tensor([1.,0.,0.])
loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[known],target[known]);loss.backward()
assert logits.grad[1].item()==0
report=dict(status='PASS_SYNTHETIC_ARCHITECTURE_CONTRACT',parameters=sum(p.numel() for p in model.parameters()),
 input_dimensions=INPUT_DIM,tests=['excluded diagnostic/velocity/frame index/clock invariance',
 'blank pixels cannot retain geometry/RGB evidence','batch padded values do not alter real-time outputs',
 'native real length2/8/15/16/40 preserved','masked unknown target gradientzero'],
 real_clip_predictions=0,training=0,accuracy_validated=False,
 code_pins={str(U/n):hashlib.sha256((U/n).read_bytes()).hexdigest() for n in ['process_head.py','check_process_head.py']})
(U/'PROCESS_HEAD_SYNTHETIC_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('Synthetic process head invariants PASS; parameters',report['parameters'])
