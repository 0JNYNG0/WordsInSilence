"""Endpoint posture needs palm placement; distal finger flexion is excluded."""
from common import *
from qualified_motion import QualifiedMotionLesson

def palm_geometry(f):
 base,good=geometry(f);loc=f[:,:34].reshape(-1,17,2);av=f[:,34:51]>.5
 hands=[loc[:,7:12],loc[:,12:17]];centers=[x.mean(1)for x in hands];axes=[x[:,1:].mean(1)-x[:,0]for x in hands]
 delta=centers[1]-centers[0];rel=[]
 for axis in axes:
  scale=np.linalg.norm(axis,axis=1);unit=axis/np.maximum(scale[:,None],.02);rel.append(np.c_[(delta*unit).sum(1),delta[:,0]*unit[:,1]-delta[:,1]*unit[:,0],scale])
 distance=np.linalg.norm(hands[0][:,:,None]-hands[1][:,None,:],axis=-1)
 x=np.c_[base,centers[0],centers[1],axes[0],axes[1],delta,*rel,distance.min((1,2)),distance.mean((1,2))].astype(np.float32)
 good &=av[:,7:17].all(1)&np.isfinite(x).all(1)
 return np.nan_to_num(x),good

class PalmRelation(GeometryRelation):
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   c=torch.load(E/'palm_relation'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(c['mean']),'frame_mlp');m.load_state_dict(c['state']);m.eval();self.models.append((m,c['mean'],c['std']))
 def __call__(self,f,quality,ts):
  x,good=palm_geometry(f);ss=[];pp=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor(((x-mu)/sd)[None])).softmax(-1)[0].numpy();pp.append(p);ss.append(states(p.copy(),good))
  ss=np.array(ss);return np.where((ss==ss[0]).all(0),ss[0],-1),np.array(pp)

class PalmMotionLesson(QualifiedMotionLesson):
 def __init__(self):super().__init__();self.body_relation=PalmRelation()

def train():
 from core import Features
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);dest=E/'palm_relation';assert not(dest/'TRAINED.json').exists()
 frozen=read(E/'relation/FROZEN.json');rows={r['id']:r for r in sources()};net=Features();cache={};xx=[];yy=[];meta=[]
 for r in frozen['rows']:
  cid=r['clip']
  if cid not in cache:
   ch=channels(dict(np.load(rows[cid]['raw'])));f,_,_=net(ch);x,good=palm_geometry(f);cache[cid]=(ch,x,good)
  ch,x,good=cache[cid];i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  if good[i]:xx.append(x[i]);yy.append(r['state']);meta.append(r)
 x=np.array(xx,np.float32);y=np.array(yy,np.int64);mu=x.mean(0);sd=np.maximum(x.std(0),.02);cw=torch.tensor(len(y)/(3*np.maximum(np.bincount(y,minlength=3),1)),dtype=torch.float32)
 write(dest/'FROZEN.json',dict(rows=meta,features='Body/arm/wrist plus palm root/MCP center and plane; no distal finger joints, word, person or GRU embedding',seeds=[42,43,44],steps=800,lr=.001,confidence=.65,known_development_addition='REAL18 L/R have now been evaluated and are development, not untouched holdout',training_additions=0,feature_count=x.shape[1]))
 runs=[]
 for seed in[42,43,44]:
  torch.manual_seed(seed);m=PhaseNet(x.shape[1],'frame_mlp');opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y)
  for _ in range(800):
   loss=torch.nn.functional.cross_entropy(m(tx[None])[0],ty,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  p=dest/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(dest/'TRAINED.json',dict(runs=runs,fit_rows=len(x),fit_clips=len(cache),feature_count=x.shape[1],class_counts=np.bincount(y).tolist(),private_data=False));print('PALM_RELATION',len(x),x.shape[1],flush=True)
if __name__=='__main__':train()
