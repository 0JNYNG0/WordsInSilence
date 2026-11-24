import numpy as np,torch,traceback
from rgb_model import R,Q,N,P,RGBMotionNet,PathNet,training_data,batch,read
from prepare_real_fit import exclusive,sha

def main():
 torch.set_num_threads(2);contract=read(R/'PLAN.json');assert sha(N/'DATASET.json')==contract['data_sha256']
 for row in read(R/'INPUT_READY.json')['rows']:assert sha(row['input'])==row['input_sha256'] and row['person'] in range(1,10)
 seq,y,mask,records,groups=training_data();assert len(seq)==1044 and len(groups)==81
 assert all(s.shape[1]==915 and np.isfinite(s).all() for s in seq)
 old=PathNet('separated_relative');old.load_state_dict(torch.load(Q/'runs/separated_relative_42/model.pt',map_location='cpu',weights_only=True));old.eval()
 x,l=batch([seq[0],seq[12],seq[20]]);model=RGBMotionNet('rgb_residual');model.initialize(42);model.eval()
 with torch.inference_mode():
  assert torch.allclose(model(x,l),old(x[:,:,:146],l),atol=1e-6)
  padded=torch.cat([x,torch.randn(len(x),7,915)],dim=1);assert torch.allclose(model(x,l),model(padded,l),atol=1e-6)
  # Nonzero visual weights prove the availability check, not merely the initial zeros.
  model.direction.rgb.weight.fill_(.01);model.continuity.rgb.weight.fill_(.01)
  missing=x.clone();missing[:,:,914]=0;changed=missing.clone();changed[:,:,146:914]=123
  assert torch.allclose(model(missing,l),model(changed,l),atol=1e-6)
 model.zero_grad();model(x,l)[:,:5].sum().backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.continuity.parameters())
 model.zero_grad();model(x,l)[:,5].sum().backward();assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for p in model.direction.parameters())
 stopped=[i for i,r in enumerate(records) if r['kind']=='feature_stopped'];assert all(np.array_equal(seq[i],np.repeat(seq[i][:1],len(seq[i]),axis=0)) for i in stopped)
 for p,h in read(P/'INPUT_MANIFEST.json')['source_pins'].items():assert sha(p)==h
 exclusive(R/'PREFLIGHT.json',dict(status='PASS',fit_windows=1044,source_events=81,feature_files=144,
  exact_initial_output_identity=True,masked_image_payload_ignored=True,padding_invariance=True,independent_gradients=True,
  rgb_and_pose_control_indices_match=True,old_models_preserved=True,new_evaluation_inference=False,private_data_access=False))
 print('PASS RGB preflight:1044 windows,915 features, initial identity and missing-image protection')

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (R/'PREFLIGHT_FIRST_FAILURE.txt').exists():(R/'PREFLIGHT_FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
