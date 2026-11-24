from pathlib import Path
import sys,json,numpy as np,torch
sys.dont_write_bytecode=True
from prepare import U,read,save,sha
from train import assemble
from model import GuideFeedbackNet,collate
def main():
 assert not (U/'TRAIN_READY.json').exists()
 rows,stats=assemble();fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
 assert len(fit)==36 and len(dev)==12
 assert set(r['person'] for r in fit).isdisjoint(set(r['person'] for r in dev))
 for r in rows:
  assert all(np.isfinite(r[k]).all() for k in ['pose','rgb','prelinear','y','ym'])
  if r['role']=='fit':assert ((r['canonical_index']>=0)&(r['canonical_index']<stats['prototype_lengths'][r['word']])).all()
  else:assert (r['canonical_index']==-1).all() and (r['stage_y']==-1).all(), 'No DEV-derived alignment training'
 torch.manual_seed(8);torch.set_num_threads(4)
 w=np.load(U/'PRETRAINED_PROJECTION.npy');m=GuideFeedbackNet(w,np.array(stats['guide_vectors']),np.array(stats['guide_words']),stats['prototype_lengths'],True)
 m.rgb_mean.copy_(torch.tensor(stats['rgb_mean']));m.rgb_std.copy_(torch.tensor(stats['rgb_std']));m.eval()
 b=collate([fit[0],fit[10]])
 with torch.no_grad():
  a=m(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word'])
  changed=m(b['pose'],b['rgb'],b['prelinear'],b['lengths'],(b['word']+1)%4)
  assert torch.equal(a['attributes'],changed['attributes']) and torch.equal(a['latent'],changed['latent'])
  assert not torch.equal(a['stage'],changed['stage'])
  base=m.tuned_weight().clone();assert torch.equal(base,torch.from_numpy(w))
 opt=torch.optim.SGD([m.update_a,m.update_b],lr=.01)
 loss=m(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word'])['attributes'].square().mean();loss.backward()
 assert torch.isfinite(m.update_b.grad).all() and m.update_b.grad.abs().sum()>0
 opt.step();assert not torch.equal(m.tuned_weight(),base);assert torch.equal(m.pretrained_projection,base)
 x=torch.randn(9,768);delta=(x@m.update_a.T)@m.update_b.T/m.rank
 assert torch.allclose(x@m.tuned_weight().T,x@base.T+delta,atol=1e-4,rtol=1e-4)
 # Padded future frames cannot change real-frame output in the batched path.
 single=collate([fit[0]]);m.eval()
 with torch.no_grad():
  one=m(single['pose'],single['rgb'],single['prelinear'],single['lengths'],single['word'])
  two=m(b['pose'],b['rgb'],b['prelinear'],b['lengths'],b['word'])
  assert torch.allclose(one['attributes'][0],two['attributes'][0,:len(fit[0]['pose'])],atol=2e-5,rtol=2e-5)
 record=dict(status='PASS_INPUT_MODEL_SUPERVISION_AND_PARTIAL_FINETUNE_CONTRACT',fit_clips=36,development_clips=12,
  source_whole_stage_truth_claimed=False,dev_transferred_alignment_used_for_training=False,
  goal_change_does_not_change_observation=True,pretrained_layer_update_and_merged_weight_equivalence=True,
  code_pins={n:sha(U/n) for n in ['prepare.py','prepare_projection.py','model.py','train.py','FROZEN_TRAINING.json']},
  input_manifest_sha256=sha(U/'INPUT_MANIFEST.json'),projection_manifest_sha256=sha(U/'PROJECTION_PREPARED.json'),
  note='Synthetic parameter/gradient check only; not a fitted candidate or recognition accuracy test.')
 save('TRAIN_READY.json',record);print(record['status'])
if __name__=='__main__':main()
