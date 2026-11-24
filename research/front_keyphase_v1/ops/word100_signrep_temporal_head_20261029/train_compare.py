"""Fixed six-run experiment. No evaluation-based checkpoint selection."""
from pathlib import Path
import json,hashlib,time,traceback,sys
import numpy as np,torch
from torch import nn
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,x):
 p=U/n
 if p.exists():raise FileExistsError(p)
 p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
class Head(nn.Module):
 def __init__(self,use_rgb):
  super().__init__();self.use_rgb=use_rgb
  self.pose=nn.Linear(138,32);self.c1=nn.Conv1d(64,64,3,padding=1);self.c2=nn.Conv1d(64,32,3,padding=1);self.out=nn.Linear(32,5)
  self.rgb=nn.Linear(768,32) if use_rgb else None
 def forward(self,x,z,valid):
  p=torch.nn.functional.gelu(self.pose(x));r=torch.nn.functional.gelu(self.rgb(z)) if self.use_rgb else torch.zeros_like(p)
  h=torch.cat([p,r],-1)*valid[:,:,None];h=torch.nn.functional.gelu(self.c1(h.transpose(1,2)))*valid[:,None,:]
  h=torch.nn.functional.gelu(self.c2(h))*valid[:,None,:]
  return self.out(h.transpose(1,2))
def masked_loss(logits,y,m,w):
 # Unknown placeholders are not used even inside BCE evaluation.
 safe=torch.where(m,y,torch.zeros_like(y));loss=nn.functional.binary_cross_entropy_with_logits(logits,safe,reduction='none')
 weights=torch.where(safe>.5,w[1],w[0]);loss=torch.where(m,loss*weights,torch.zeros_like(loss))
 counts=m.sum(1);per=(loss.sum(1)/counts.clamp_min(1));ok=counts>0
 return (per*ok).sum()/ok.sum()
def assemble(rows,stats=None):
 arrays=[dict(np.load(r['input'],allow_pickle=False)) for r in rows]
 if stats is None:
  fit=[a for r,a in zip(rows,arrays) if r['role']=='fit_candidate'];xf=np.concatenate([a['x'] for a in fit]);mf=np.concatenate([a['mask'] for a in fit]);zf=np.concatenate([a['rgb'] for a in fit])
  means=np.array([xf[mf[:,j],j].mean() if mf[:,j].any() else 0 for j in range(69)],np.float32)
  stds=np.array([max(float(xf[mf[:,j],j].std()),1e-3) if mf[:,j].any() else 1 for j in range(69)],np.float32)
  stats=dict(pose_mean=means.tolist(),pose_std=stds.tolist(),rgb_mean=zf.mean(0).tolist(),rgb_std=np.maximum(zf.std(0),1e-3).tolist())
 n=len(arrays);T=max(len(a['x']) for a in arrays);x=np.zeros((n,T,138),np.float32);z=np.zeros((n,T,768),np.float32);y=np.zeros((n,T,5),np.float32);m=np.zeros_like(y,bool);v=np.zeros((n,T),np.float32)
 for i,a in enumerate(arrays):
  t=len(a['x']);px=np.where(a['mask'],(a['x']-np.array(stats['pose_mean']))/np.array(stats['pose_std']),0)
  x[i,:t]=np.concatenate([px,a['mask']],1);z[i,:t]=(a['rgb']-np.array(stats['rgb_mean']))/np.array(stats['rgb_std']);y[i,:t]=a['y'];m[i,:t]=a['target_mask'];v[i,:t]=1
 return [torch.from_numpy(a) for a in [x,z,y,m,v]],stats,arrays
def checks(batch):
 x,z,y,m,v=batch;torch.manual_seed(42);h=Head(True);l=h(x,z,v);w=torch.ones((2,5))
 a=masked_loss(l,y,m,w);ga=torch.autograd.grad(a,tuple(h.parameters()),retain_graph=True)
 altered=y.clone();altered[~m]=9876;b=masked_loss(l,altered,m,w);gb=torch.autograd.grad(b,tuple(h.parameters()))
 assert torch.equal(a,b) and all(torch.equal(u,vv) for u,vv in zip(ga,gb))
 h.eval()
 with torch.no_grad():
  whole=h(x,z,v)
  for i in range(len(x)):
   n=int(v[i].sum());single=h(x[i:i+1,:n],z[i:i+1,:n],v[i:i+1,:n])
   assert torch.allclose(whole[i,:n],single[0],atol=2e-6,rtol=2e-5)
 return dict(unknown_loss_and_gradients_identical=True,padded_batch_matches_unpadded=True)
def main():
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
 info=read(U/'INPUTS.json');rows=info['rows'];assert len(rows)==36
 assert read(U/'INPUT_VERIFICATION.json')['status']=='PASS'
 for p,h in info['pins'].items():assert sha(p)==h,(p,'pin')
 for r in rows:assert sha(r['input'])==r['input_sha256']
 assert read(U/'LABEL_REVIEW.json')['status']=='PASS_PROVISIONAL'
 fit=[i for i,r in enumerate(rows) if r['role']=='fit_candidate'];assert len(fit)==16
 assert {rows[i]['person'] for i in fit}.isdisjoint({r['person'] for r in rows if r['role']=='held_person'})
 batch,stats,arrays=assemble(rows);x,z,y,m,v=batch
 save('TRAIN_READY.json',dict(status='PASS_PROVISIONAL',checks=checks(batch),roles={role:sum(r['role']==role for r in rows) for role in sorted({r['role'] for r in rows})},input_manifest_sha256=sha(U/'INPUTS.json'),label_sha256=sha(U/'LABELS.json'),training_code_sha256=sha(__file__),fit_aliases=[rows[i]['alias'] for i in fit],no_semantic_stage_ground_truth=True))
 save('NORMALIZATION.json',stats)
 counts=torch.stack([((y[fit]==c)&m[fit]).sum((0,1)) for c in [0,1]]).float();assert (counts>0).all();w=counts.sum(0)[None]/(2*counts)
 runs=[]
 for condition in ['pose_only','pose_plus_signrep']:
  for seed in [42,43,44]:
   name=f'{condition}_{seed}';dest=U/'runs'/name;dest.mkdir(parents=True,exist_ok=False)
   torch.manual_seed(seed);h=Head(condition=='pose_plus_signrep');opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01);trace=[];start=time.perf_counter()
   for epoch in range(160):
    h.train();opt.zero_grad();logits=h(x[fit],z[fit],v[fit]);loss=masked_loss(logits,y[fit],m[fit],w);assert torch.isfinite(loss)
    loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),1);opt.step();trace.append(float(loss))
   seconds=time.perf_counter()-start;h.eval()
   with torch.no_grad():prob=h(x,z,v).sigmoid().numpy()
   np.save(dest/'probabilities.npy',prob,allow_pickle=False)
   torch.save(h.state_dict(),dest/'head.pt')
   result=dict(condition=condition,seed=seed,epochs=160,selected_epoch=160,loss_trace=trace,seconds=seconds,parameters=sum(p.numel() for p in h.parameters()),class_counts=counts.tolist(),class_weights=w.tolist(),checkpoint_sha256=sha(dest/'head.pt'),probabilities_sha256=sha(dest/'probabilities.npy'),backbone_frozen=True,fit_aliases=[rows[i]['alias'] for i in fit])
   save(f'runs/{name}/RUN.json',result);runs.append(dict(name=name,**{k:result[k] for k in ['condition','seed','seconds','parameters','checkpoint_sha256','probabilities_sha256']}));print(json.dumps(runs[-1]),flush=True)
 save('TRAINING.json',dict(status='COMPLETE',runs=runs,selected_by_evaluation=False,signrep_weight_updates=0))
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/'TRAIN_FIRST_FAILURE.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
