from pathlib import Path
import sys,json,hashlib,time,traceback
U=Path(__file__).resolve().parent;O=U.parent;BASE=O/'word100_component_learning_20261102'
sys.path.insert(0,str(BASE))
import common as c
import train_evaluate as e
import torch,numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
read=c.read;sha=c.sha
def save(n,d):
 with (U/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
POSITIONS=list(range(10))+list(range(28,38))+[56,57,58,61,62]

def relative(x,l):
 v=x[:,:,POSITIONS];m=x[:,:,np.asarray(POSITIONS)+63]>.5
 tm=torch.arange(x.shape[1])[None,:]<l[:,None];m=m&tm[:,:,None]
 first=m.long().argmax(1);anchor=v.gather(1,first[:,None,:]).squeeze(1);has=m.any(1)
 disp=torch.where(m&has[:,None,:],v-anchor[:,None,:],0)
 adjacent=torch.zeros_like(m);adjacent[:,1:]=m[:,1:]&m[:,:-1]
 step=torch.zeros_like(v);step[:,1:]=v[:,1:]-v[:,:-1];step=torch.where(adjacent,step,0)
 return torch.cat([disp,step,m.float(),adjacent.float()],-1)

class Model(c.TemporalNet):
 def __init__(self,condition):
  super().__init__('ordered');self.load_state_dict(torch.load(BASE/'runs/ordered.pt',map_location='cpu',weights_only=True))
  for p in self.parameters():p.requires_grad_(False)
  for p in self.out.parameters():p.requires_grad_(True)
  self.adaptation_condition=condition
  self.relative=nn.Linear(len(POSITIONS)*4,32,bias=False);nn.init.zeros_(self.relative.weight)
  self.relative.weight.requires_grad_(condition=='relative')
 def forward(self,x,l):
  h=self.embed[1](self.embed[0](x)+self.relative(relative(x,l)))
  _,last=self.temporal(pack_padded_sequence(h,l.cpu(),batch_first=True,enforce_sorted=False))
  return self.out(torch.cat([last[0],last[1]],-1))

def prepare():
 assert not (U/'FROZEN.json').exists()
 rows,meta,arrays=c.load();samples=c.build_temporal(rows,arrays,read(BASE/'NORMALIZATION.json'))
 a=Model('output_only').eval();b=Model('relative').eval();old=c.TemporalNet('ordered').eval()
 old.load_state_dict(torch.load(BASE/'runs/ordered.pt',map_location='cpu',weights_only=True))
 x,l=c.batch(samples[:3]);padded=torch.cat([x,torch.zeros(3,11,126)],1)
 with torch.inference_mode():
  assert torch.allclose(a(x,l),old(x,l),atol=1e-6)
  assert torch.allclose(b(x,l),old(x,l),atol=1e-6)
  assert torch.allclose(b(x,l),b(padded,l),atol=1e-6)
 for s in samples:
  if s['variant'] not in ['still','image_gap']:continue
  xx,ll=c.batch([s]);r=relative(xx,ll)
  assert torch.count_nonzero(r[:,:,:len(POSITIONS)*2])==0
 save('FROZEN.json',dict(conditions=['output_only','relative'],primary='relative',epochs=80,seed=42,lr=.001,weight_decay=.01,
  reason='Constant input falsely implies motion and two normal arm sweeps get confident negative; test explicit within-person changes, not another role-only loss.',
  point_model_unchanged=True,temporal_gru_embedding_frozen=True,shared_initial_outputs_exact=True,
  trainable='shared copied output layer; relative condition also zero-initialized displacement+adjacent-step/mask branch',
  gates='All original component comparison gates plus no worse known native classification/confirmed coverage than matched control; no increase in confident natural errors; at least one original failed gate improves.',
  scope='Same repeated DEV; not independent confirmation or actual RGB speed/occlusion test',
  pins={str(p):sha(p) for p in [BASE/'FROZEN.json',BASE/'common.py',BASE/'train_evaluate.py',BASE/'NORMALIZATION.json',BASE/'runs/ordered.pt',BASE/'POINT_RESULT.json',BASE/'ORDERED_RESULT.json',BASE/'DIAGNOSIS.json',U/'experiment.py']}))
 save('PREFLIGHT.json',dict(zero_initial_outputs_equal=True,padding_invariant=True,
  static_and_missing_displacement_steps_zero=True,original_records_unchanged=True,new_training=False))
 print('Relative-motion pair frozen and preflight passed')

def train():
 torch.set_num_threads(4);p=read(U/'FROZEN.json')
 for path,h in p['pins'].items():assert sha(path)==h,path
 for path,h in read(BASE/'FROZEN.json')['pins'].items():assert sha(path)==h,path
 rows,meta,arrays=c.load();samples=c.build_temporal(rows,arrays,read(BASE/'NORMALIZATION.json'))
 fit=[s for s in samples if s['record']['role']=='fit' and s['mask'] and s['variant'] in ['native','reverse','still']]
 x,l=c.batch(fit);y=torch.tensor([s['y'] for s in fit],dtype=torch.float32);j=torch.tensor([c.TEMP.index(s['record']['target']) for s in fit]);weights=c.grouped_weights(fit,c.TEMP)
 root=U/'runs';assert not root.exists();root.mkdir();collected={}
 for cond in p['conditions']:
  torch.manual_seed(42);m=Model(cond);opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=p['lr'],weight_decay=p['weight_decay']);hist=[];t=time.perf_counter()
  for epoch in range(p['epochs']):
   m.train();logits=m(x,l)[torch.arange(len(fit)),j];loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*weights).sum()
   assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
   if epoch%20==0 or epoch==79:hist.append(dict(epoch=epoch+1,loss=float(loss.detach())));print(cond,epoch+1,float(loss.detach()),flush=True)
  m.eval();torch.save(m.state_dict(),root/(cond+'.pt'));rr=e.predict_temporal(m,samples);met=e.metrics(rr);collected[cond]=met
  old=torch.load(BASE/'runs/ordered.pt',map_location='cpu',weights_only=True)
  assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.items() if not k.startswith('out.'))
  save(cond.upper()+'_RESULT.json',dict(rows=rr,metrics=met,history=hist,training_seconds=time.perf_counter()-t,checkpoint_sha256=sha(root/(cond+'.pt'))))
 point=read(BASE/'POINT_RESULT.json')['metrics'];base=read(BASE/'POOLED_RESULT.json')['metrics']
 comparison=e.gates(point,base,collected['relative'])
 for target in c.TEMP:
  a=collected['relative'][target+'::native'];b=collected['output_only'][target+'::native']
  comparison['checks'][target+':paired_recall_retained']=a['usable_positive_recall']>=b['usable_positive_recall']
  comparison['checks'][target+':paired_confirmed_recall_retained']=a['confirmed_positive_recall']>=b['confirmed_positive_recall']
  comparison['checks'][target+':paired_confident_wrong_not_increased']=a['confident_wrong']<=b['confident_wrong']
  if b['negative']:
   comparison['checks'][target+':paired_specificity_retained']=a['usable_negative_specificity']>=b['usable_negative_specificity']
   comparison['checks'][target+':paired_confirmed_specificity_retained']=a['confirmed_negative_specificity']>=b['confirmed_negative_specificity']
 oldchecks=read(BASE/'COMPARISON.json')['checks'];comparison['checks']['at_least_one_failed_gate_improved']=any(not v and comparison['checks'][k] for k,v in oldchecks.items())
 fields=['usable_positive_recall','usable_negative_specificity','confirmed_positive_recall','confirmed_negative_specificity']
 comparison['checks']['relative_has_added_contribution_over_matched_control']=any(
   collected['relative'][key][f]>collected['output_only'][key][f]
   for key in collected['relative'] if key.endswith(('::native','::reverse','::still'))
   for f in fields if collected['output_only'][key][f] is not None)
 comparison['passed']=all(comparison['checks'].values());comparison['status']='SCOPED_RELATIVE_GATES_PASSED' if comparison['passed'] else 'RELATIVE_GATES_NOT_MET'
 for path,h in p['pins'].items():assert sha(path)==h,path
 for path,h in read(BASE/'FROZEN.json')['pins'].items():assert sha(path)==h,path
 save('COMPARISON.json',comparison);save('COMPLETE.json',dict(status=comparison['status'],whole_goal_complete=False,new_temporal_runs=2))
 print(comparison,flush=True)
if __name__=='__main__':
 try:
  if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare()
  elif len(sys.argv)>1 and sys.argv[1]=='train':train()
  else:raise ValueError('prepare or train required')
 except BaseException:
  if not (U/'FIRST_FAILURE.txt').exists():(U/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
