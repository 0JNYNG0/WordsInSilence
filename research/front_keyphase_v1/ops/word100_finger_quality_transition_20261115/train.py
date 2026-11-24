from quality_common import *
import torch,collections,time
from quality_model import QualityChange,features,windows
from torch.nn import functional as F
assert not(U/'TRAIN_FROZEN.json').exists();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);rows=read(U/'FIT_LABELS.json')['rows'];seq=[];targets=[];cache={}
for r in rows:
 assert r['role']=='fit' and r['person']in[3,6,9];key=r['clip_id'],r['hand']
 if key not in cache:
  assert sha(r['channel_path'])==r['channel_sha256'];z=np.load(r['channel_path']);x=z['feedback']if'feedback'in z.files else z['full'];cache[key]=(features(x,r['hand']),z['frame_id'])
 (f,ok),ids=cache[key];t=int(np.flatnonzero(ids==r['frame'])[0]);seq.append(windows(f,[t])[0]);q=np.array(r['quality']);phase=np.array(r['phase']);y=np.stack([np.where(q>=0,1-q,-1),phase],1);y[~ok[t]]=-1;targets.append(y)
x=np.array(seq);y=np.array(targets);mu=x.reshape(-1,x.shape[-1]).mean(0);sd=x.reshape(-1,x.shape[-1]).std(0).clip(.05);mask=y>=0;counts=collections.Counter((rows[i]['clip_id'],j,k,int(y[i,j,k]))for i in range(len(rows))for j in range(4)for k in range(2)if mask[i,j,k]);classes=collections.Counter((k,v)for cid,j,k,v in counts);w=np.zeros(y.shape,np.float32)
for i in range(len(rows)):
 for j in range(4):
  for k in range(2):
   if mask[i,j,k]:w[i,j,k]=1/(counts[rows[i]['clip_id'],j,k,int(y[i,j,k])]*classes[k,int(y[i,j,k])])
w/=w.sum();np.savez_compressed(U/'TRAIN_INPUTS.npz',x=x,y=y,weights=w,mean=mu,std=sd)
cfg=dict(seeds=[42,43,44],steps=600,lr=.001,confidence=.85,choice='One common quality/change architecture; final600steps/all3seeds. Fixed .85 all-seedagreement to suppress; no confident-state overrides.',features=x.shape[-1],inputs='handlocal normalized relative finger geometry + adjacentfinger distances + temporal differences; no word/person/guide targets',pins={str(U/p):sha(U/p)for p in['FIT_LABELS.json','DIAGNOSTIC_LABELS.json','quality_model.py','TRAIN_INPUTS.npz']});write(U/'TRAIN_FROZEN.json',cfg);xx=torch.tensor((x-mu)/sd);yy=torch.tensor(y.clip(0),dtype=torch.float32);ww=torch.tensor(w);runs=[];dest=U/'models';dest.mkdir(exist_ok=True)
for seed in cfg['seeds']:
 torch.manual_seed(seed);m=QualityChange(x.shape[-1]);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=.001);t=time.perf_counter()
 for step in range(cfg['steps']):
  pred=m(xx);loss=(F.binary_cross_entropy_with_logits(pred,yy,reduction='none')*ww).sum();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
 m.eval();path=dest/f'{seed}.pt';torch.save(dict(state_dict=m.state_dict(),mean=mu,std=sd,features=x.shape[-1]),path);runs.append(dict(seed=seed,model=str(path),sha256=sha(path),seconds=time.perf_counter()-t,loss=float(loss.detach())));print(runs[-1],flush=True)
write(U/'TRAINED.json',dict(runs=runs,fit_anchors=len(rows),fit_clips=len(cache),quality_known=int(mask[:,:,0].sum()),quality_bad=int((y[:,:,0]==1).sum()),phase_known=int(mask[:,:,1].sum()),phase_changing=int((y[:,:,1]==1).sum()),web_changed=False))
