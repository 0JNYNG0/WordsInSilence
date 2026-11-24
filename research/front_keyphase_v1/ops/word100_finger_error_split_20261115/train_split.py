from split_model import *
import time,collections
from torch.nn import functional as F
cfg=read(U/'TRAIN_CONFIG.json');assert not(U/'TRAINED.json').exists()
for p,h in cfg['pins'].items():assert sha(p)==h
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
data=np.load(U/'TRAIN_INPUTS.npz');meta=read(U/'TRAIN_INPUTS.json');runs=[]
def weights(y,mask,rows):
 counts=collections.Counter((r['clip_id'],f,int(y[i,f]))for i,r in enumerate(rows)for f in range(4)if mask[i,f]);cl=collections.Counter((f,t)for c,f,t in counts);w=np.zeros(y.shape,np.float32)
 for i,r in enumerate(rows):
  for f in range(4):
   if mask[i,f]:w[i,f]=1/(counts[r['clip_id'],f,int(y[i,f])]*cl[f,int(y[i,f])])
 return w
for kind in ['shape','quality']:
 y=data[kind+'_y'];mask=data[kind+'_mask'];rows=meta[kind+'_rows'];w=weights(y,mask,rows)
 if kind=='shape':
  old=np.array([r['origin']=='original_fit'for r in rows]);w[old]*=.7/w[old].sum();w[~old]*=.3/w[~old].sum()
 else:w/=w.sum()
 wt=torch.tensor(w);target=torch.tensor(y.clip(0),dtype=torch.float32);outdir=U/'models'/kind;outdir.mkdir(parents=True,exist_ok=True)
 for seed in cfg['seeds']:
  start=time.perf_counter();torch.manual_seed(seed)
  if kind=='shape':
   ck=torch.load(P/'shared_finger_chain'/f'{seed}.pt',map_location='cpu',weights_only=False);model=SharedFinger();model.load_state_dict(ck['state_dict']);mu=ck['mean'];sd=ck['std']
  else:model=TrackQuality();mu=data['quality_mean'];sd=data['quality_std']
  x=torch.tensor((data[kind+'_x']-mu)/sd,dtype=torch.float32)
  if kind=='shape':
   with torch.no_grad():teacher=model(x).sigmoid().detach().clone()
   kw=w.copy();kw[~old]=0;kw/=kw.sum();kw=torch.tensor(kw)
  optimizer=torch.optim.AdamW(model.parameters(),lr=cfg[kind+'_lr'],weight_decay=cfg['weight_decay'])
  for step in range(cfg[kind+'_steps']):
   logits=model(x);loss=(F.binary_cross_entropy_with_logits(logits,target,reduction='none')*wt).sum()
   if kind=='shape':loss+=cfg['distillation_weight']*(F.binary_cross_entropy_with_logits(logits,teacher,reduction='none')*kw).sum()
   optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
  model.eval()
  with torch.inference_mode():prob=model(x).sigmoid().numpy()
  pred=np.where(prob>=.85,1,np.where(prob<=.15,0,-1));metrics=dict(known=int(mask.sum()),correct=int(((pred==y)&mask).sum()),wrong=int(((pred>=0)&(pred!=y)&mask).sum()),unknown=int(((pred<0)&mask).sum()))
  path=outdir/f'{seed}.pt';torch.save(dict(state_dict=model.state_dict(),mean=mu,std=sd,kind=kind,seed=seed,training_config_sha=sha(U/'TRAIN_CONFIG.json')),path)
  r=dict(kind=kind,seed=seed,path=str(path),sha256=sha(path),seconds=time.perf_counter()-start,fit_metrics=metrics,final_loss=float(loss.detach()));runs.append(r);write(U/'TRAIN_PROGRESS.json',dict(runs=runs));print(json.dumps(r),flush=True)
write(U/'TRAINED.json',dict(runs=runs,fit_only=True,base_models_unmodified=True))
