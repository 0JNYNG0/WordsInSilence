"""Prespecified public TRAIN sensitivity check after selection; never a selection criterion."""
from common import *
import torch,traceback
from multi_position_aux import MultiPositionAux,phase_readout
from temporal import phase_weights
sys.path.insert(0,str(BASE))
from training_core import padded_arrays
CONDITIONS=('original','swap_first_two_thirds','reverse_thirds','suffix_third')
def main():
 assert not (U/'DIAGNOSTIC.json').exists() and not (U/'TRAIN_RUNNING.json').exists()
 assert read(U/'EXIT.json')['exit_code']==0
 ctx=read(U/'TRAIN_CONTEXT.json');check(ctx['pins']);plan=read(U/'DIAGNOSTIC_PLAN.json')
 assert list(CONDITIONS)==plan['conditions']
 rows=sorted((r for r in ctx['rows'] if r['role']=='train_candidate'),key=lambda r:r['clip_id'])
 assert len(rows)==899 and {r['signer_id'] for r in rows}==set(range(1,10))
 xs={v:[] for v in VIEWS}
 for r in rows:
  for v in VIEWS:
   path=r['views'][v]['channel_file'];assert sha(path)==r['views'][v]['sha256']
   with np.load(path,allow_pickle=False) as z:xs[v].append(z['x_whole'].copy())
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 scores=np.zeros((3,2,899,4,100),np.float32);base=np.zeros((3,2,899,100),np.float32);maxerr=0.;epochs=[]
 with torch.inference_mode():
  for si,seed in enumerate(SEEDS):
   rp=read(U/f'{ARM}_{seed}_RESULT.json');cp=U/f'{ARM}_{seed}_best.pt';assert sha(cp)==rp['checkpoint_sha256']
   ck=torch.load(cp,map_location='cpu',weights_only=True);bs={k[5:]:v for k,v in ck['state_dict'].items() if k.startswith('base.')}
   m=MultiPositionAux(ck['model_config'],bs);m.load_state_dict(ck['state_dict']);m.requires_grad_(False);m.eval();before=statehash(m.state_dict());epochs.append(rp['selected_epoch'])
   for vi,v in enumerate(VIEWS):
    for start in range(0,899,32):
     data=xs[v][start:start+32];xx,pp=padded_arrays(data);x=torch.from_numpy(xx);pad=torch.from_numpy(pp)
     b,a,_,h,lengths=m.components(x,pad);base[si,vi,start:start+len(data)]=b.numpy()
     for ci,condition in enumerate(CONDITIONS):
      streams=[];dts=[]
      for i,n in enumerate(lengths):
       first,middle,last=np.array_split(np.arange(int(n)),3)
       ids={'original':np.arange(int(n)),'swap_first_two_thirds':np.r_[middle,first,last],'reverse_thirds':np.r_[last,middle,first],'suffix_third':last}[condition]
       streams.append(h[i,ids]);dt=x[i,ids if condition=='suffix_third' else np.arange(int(n)),279].clone();dt[0]=0;dts.append(dt)
      lens=torch.tensor([len(s) for s in streams]);features=torch.nn.utils.rnn.pad_sequence(streams,batch_first=True)
      pm=(torch.arange(features.shape[1])[None,:]<lens[:,None]).float()
      clock=torch.zeros(len(streams),features.shape[1],280)
      for i,dt in enumerate(dts):clock[i,:len(dt),279]=dt
      states=m.memory_sequence(features,lens);readout=phase_readout(states,pm,phase_weights(clock,pm));z=m.residual(readout)
      scores[si,vi,start:start+len(data),ci]=z.numpy()
      if ci==0:
       err=float((z-a).abs().max());maxerr=max(maxerr,err);assert err==0
   assert statehash(m.state_dict())==before
 y=np.array([r['label_index'] for r in rows]);np.savez_compressed(U/'DIAGNOSTIC_SCORES.npz',aux=scores,base=base,labels=y,ids=np.array([r['clip_id'] for r in rows]))
 # Re-read saved values for the diagnostic summary, separate from the model execution loop.
 with np.load(U/'DIAGNOSTIC_SCORES.npz',allow_pickle=False) as z:a=z['aux'].astype('float64');bb=z['base'];y=z['labels']
 result=[]
 for si,seed in enumerate(SEEDS):
  for vi,v in enumerate(VIEWS):
   orig=a[si,vi,:,0];op=orig.argmax(-1);oc=orig-orig.mean(-1,keepdims=True);scale=np.sqrt(np.mean(oc**2,axis=-1));assert np.all(scale>0)
   for ci,cond in enumerate(CONDITIONS):
    z=a[si,vi,:,ci];cc=z-z.mean(-1,keepdims=True);rel=np.sqrt(np.mean((cc-oc)**2,axis=-1))/scale
    result.append({'seed':seed,'view':v,'condition':cond,'same_aux_class':int((z.argmax(-1)==op).sum()),'same_aux_class_pct':float(np.mean(z.argmax(-1)==op)*100),
     'aux_label_hits':int((z.argmax(-1)==y).sum()),'relative_centered_rms_median_pct':float(np.median(rel)*100),
     'original_final_train_hits':int(((bb[si,vi]+a[si,vi,:,0]).argmax(-1)==y).sum())})
 old=read(D/'ANALYSIS.json');check(ctx['pins'])
 write(U/'DIAGNOSTIC.json',{'status':'COMPLETE_FIXED_TRAIN_DIAGNOSIS_NO_SELECTION','at':now(),'plan_sha256':sha(U/'DIAGNOSTIC_PLAN.json'),
  'scores_sha256':sha(U/'DIAGNOSTIC_SCORES.npz'),'code_sha256':sha(__file__),'rows':result,'prior_aux_rows':[r for r in old['probe_rows'] if r['condition'] in CONDITIONS],
  'prior_analysis_sha256':sha(D/'ANALYSIS.json'),'selected_epochs':epochs,'original_component_exact_error':maxerr,
  'clips':899,'words':100,'DEV_new_inference':0,'user_data_read':0,'additional_training':0,'selection_affected':False,
  'limits':'Contextual features, not raw-video correctness. Same words/inputs but different trained candidates and selected epochs. Sensitivity change is not semantic-stage accuracy or causal isolation.'},new=True)
 print('PASS prespecified fixed public TRAIN diagnosis; original exact, no fitting or DEV inference.')
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (U/'FIRST_DIAGNOSTIC_ERROR.json').exists():write(U/'FIRST_DIAGNOSTIC_ERROR.json',{'at':now(),'traceback':traceback.format_exc()},new=True)
  raise
