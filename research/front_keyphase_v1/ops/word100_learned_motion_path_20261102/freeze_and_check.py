import numpy as np,torch
from path_model import Q,N,M,POSITIONS,PathNet,relative_features,records_and_sequences,batch,read
from prepare_real_fit import exclusive,sha,clock

def main():
 torch.set_num_threads(2)
 contract=dict(at=clock(),conditions=['coupled_relative','separated_relative'],seeds=[42,43,44],epochs=120,lr=.0003,
  primary='separated_relative',deployment_seed=42,data=str(N/'DATASET.json'),data_sha256=sha(N/'DATASET.json'),
  fit_recorded_ids=list(range(1,10)),motion_target_source_sha256=sha(M/'MOTION_TARGETS.json'),
  changes=['Add per-window within-person displacement and adjacent-frame changes for25 position/arm-relation channels',
   'Compare shared and separate direction/continuity encoders; both start at exact frozen previous outputs'],
  hypotheses_not_conclusions=['Static-position shortcut may limit transfer','Negative continuity examples may interfere with direction learning'],
  losses={'masked_BCE':1.0,'same_event_actual_speed_consistency':.1},normalization='unchanged FIT-only',
  no_target_word_or_signer_input=True,final_epoch_only=True,unknown_targets_masked=True,
  labels_scope='Physical direction and synthetic interruption, not linguistic correctness',
  runtime_support='No new scan, window search, DTW change, threshold change or best seed selection',
  development_exposed=True,heldout_excluded=True,old_models_and_demos_preserved=True,
  promotion_criteria=str(N/'PROMOTION_CRITERIA.json'),new_heldout_test_only_after_development_gate=True)
 if (Q/'CONTRACT.json').exists():
  existing=read(Q/'CONTRACT.json');contract['at']=existing['at'];assert existing==contract
 else:exclusive(Q/'CONTRACT.json',contract)
 seq,y,m,records,groups=records_and_sequences();assert len(seq)==1044 and {r['person'] for r in records}==set(range(1,10))
 x,l=batch([seq[0],seq[12],seq[20]])
 model=PathNet('separated_relative');model.initialize(42);model.eval()
 from evaluate_real_motion import RealMotionObserver
 old=RealMotionObserver('real_variants_motion_inputs',42)
 with torch.inference_mode():
  expected=old.model(x,l);got=model(x,l);assert torch.allclose(got,expected,atol=1e-6)
  padded=torch.cat([x,torch.randn(len(x),9,146)],dim=1);assert torch.allclose(got,model(padded,l),atol=1e-6)
  a=relative_features(x,l);shift=x.clone();shift[:,:,POSITIONS]+=3
  b=relative_features(shift,l);assert torch.allclose(a,b,atol=1e-5)
 model.zero_grad();model(x,l)[:,:5].sum().backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.continuity.parameters())
 model.zero_grad();model(x,l)[:,5].sum().backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.direction.parameters())
 exclusive(Q/'PREFLIGHT.json',dict(status='PASS',fit_windows=1044,identity_at_initialization=True,padding_invariance=True,
  displacement_translation_invariance=True,gradient_separation=True,source_dataset_sha256=sha(N/'DATASET.json'),
  no_new_development_predictions=True,no_new_private_access=True))
 print('PASS: frozen plan, exact initial-output preservation, masked displacement, independent gradients')

if __name__=='__main__':main()
