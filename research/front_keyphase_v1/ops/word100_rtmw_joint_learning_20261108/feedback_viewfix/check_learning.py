from common import *
import torch,numpy as np
from feedback_model import PoseGuideNet
def main():
 assert not (U/'LEARNING_CODE_CHECK.json').exists()
 stats=read(U/'FEEDBACK_NORMALIZATION.json');torch.set_num_threads(4);torch.manual_seed(42)
 m=PoseGuideNet(stats['guide_vectors'],stats['guide_words'],stats['prototype_lengths']).eval()
 x=torch.randn(1,19,280);length=torch.tensor([19])
 with torch.no_grad():
  a=m(x,length,torch.tensor([0]));b=m(x,length,torch.tensor([1]))
  assert torch.equal(a['attributes'],b['attributes']),'Goal label leaked into observed attributes'
  padded=torch.cat([x,torch.randn(1,11,280)*1000],1);c=m(padded,length,torch.tensor([0]))
  assert torch.allclose(a['attributes'],c['attributes'][:,:19],atol=1e-6),'Padding influenced observed frames'
 logits=torch.randn(2,7);y=torch.zeros(2,7);mask=torch.zeros(2,7);mask[0,2]=1
 loss=lambda z:(torch.nn.functional.binary_cross_entropy_with_logits(logits,z,reduction='none')*mask).sum()
 altered=y.clone();altered[1]=1;assert torch.equal(loss(y),loss(altered))
 ready=read(U/'FEEDBACK_READY.json');assert ready['status']=='PASS';assert read(U/'FEEDBACK_INPUT_VERIFICATION.json')['status']=='PASS'
 for r in ready['rows']:
  z=np.load(r['path']);assert np.isfinite(z['x']).all()
  if r['role']=='development':assert np.all(z['ci']==-1) and np.all(z['sy']==-1) and np.all(z['sw']==0)
 write(U/'LEARNING_CODE_CHECK.json',dict(status='PASS',tests=['target does not alter observations','padding cannot influence valid frames','unknown label loss excluded','all DEV has no training correspondence or stage labels'],
  pins={str(U/n):sha(U/n) for n in ['feedback_model.py','prepare_feedback.py','train_feedback.py','check_learning.py']},frozen_sha256=sha(U/'FROZEN.json')))
 event('learning_checks_pass')
if __name__=='__main__':main()
