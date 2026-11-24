"""Frontal-only posture observer; used for inspection, never completion."""
from probe_front import *
from core import PhaseNet,states,Features
from visible_lesson import adapter

def geometry(f):
 ids=[1,2,3,4,5,6,7,12];loc=f[:,:34].reshape(-1,17,2)[:,ids];av=f[:,34:51][:,ids]>.5
 pairs=np.concatenate([loc[:,6]-loc[:,7],loc[:,6]-loc[:,2],loc[:,7]-loc[:,3],loc[:,4]-loc[:,5]],axis=1)
 distance=np.linalg.norm(loc[:,6]-loc[:,7],axis=1,keepdims=True);x=np.c_[loc.reshape(len(f),-1),av,pairs,distance].astype(np.float32);good=av.all(1)&np.isfinite(x).all(1)
 return np.nan_to_num(x),good

class FrontGeometry:
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   c=torch.load(N/'geometry'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(c['mean']),'frame_mlp');m.load_state_dict(c['state']);m.eval();self.models.append((m,c['mean'],c['std']))
 def __call__(self,f):
  x,good=geometry(f);ss=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor(((x-mu)/sd)[None])).softmax(-1)[0].numpy();ss.append(states(p.copy(),good))
  ss=np.array(ss);return np.where((ss==ss[0]).all(0),ss[0],-1)

def train():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);dest=N/'geometry';dest.mkdir(exist_ok=True);assert not(dest/'FROZEN.json').exists()
 truth=read(O/'basic_demo_feedback_20261115/source_corrected/SOURCE_TRUTH.json')['rows'];src={r['id']:r for r in read(N/'INPUTS.json')['rows']};net=Features();cache={};xx=[];yy=[];meta=[]
 for r in truth:
  if r['split']!='fit':continue
  cid=r['clip_id'];require_front(src[cid])
  if cid not in cache:
   z=dict(np.load(src[cid]['raw']));ch=adapter.convert(z['xy_px'],z['score'],z['timestamp_us'],z['frame_id'],int(z['source_width']),int(z['source_height']));f,_,_=net(ch);x,good=geometry(f);cache[cid]=(ch,x,good)
  ch,x,good=cache[cid];i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  if good[i]:xx.append(x[i]);yy.append(r['state']);meta.append(dict(clip_id=cid,frame=r['frame'],state=r['state'],split='fit'))
 x=np.array(xx,np.float32);y=np.array(yy,np.int64);mu=x.mean(0);sd=np.maximum(x.std(0),.02);weights=torch.tensor(len(y)/(3*np.maximum(np.bincount(y,minlength=3),1)),dtype=torch.float32)
 write(dest/'FROZEN.json',dict(rows=meta,steps=800,lr=.001,weight_decay=.01,seeds=[42,43,44],confidence=.65,source=str(O/'basic_demo_feedback_20261115/source_corrected/SOURCE_TRUTH.json'),purpose='Preserve shape inspection when changed fingers invalidate the original joint relation. Proposal only, never adds completed events.',features='33 body-relative shoulder/elbow/wrist/root geometry values; no finger shape, word, target, elapsed phase or GRU embedding',selection='Same fixed hyperparameters as earlier geometry head; train from random initialization on original front FIT only. No mixed checkpoint initialization or DEV optimization.'))
 runs=[]
 for seed in [42,43,44]:
  torch.manual_seed(seed);m=PhaseNet(x.shape[1],'frame_mlp');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y);start=time.perf_counter()
  for _ in range(800):
   loss=torch.nn.functional.cross_entropy(m(tx[None])[0],ty,weight=weights);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  p=dest/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd),p);runs.append(dict(seed=seed,seconds=time.perf_counter()-start,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(dest/'TRAINED.json',dict(runs=runs,fit_rows=len(x),fit_clips=len(cache),private_data=False));print('FRONT_GEOMETRY_TRAINED',len(x),len(cache),flush=True)
if __name__=='__main__':train()
