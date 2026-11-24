"""Source-trained projected near-state support. Never certify physical contact."""
from pathlib import Path
import sys
K=Path(__file__).resolve().parent;J=K.parent/'public_view_observation_repair_20261116'
sys.path.insert(0,str(J))
from view_common import *
from final_candidate import PlaneViewLesson
from palm_geometry import PalmRelation,palm_geometry
from core import Features,PhaseNet
import cv2

def projected_features(f,ch):
 base,good=palm_geometry(f);p=ch['body_relative_xy'];av=ch['point_available'];extra=np.zeros((len(p),8),np.float32)
 ids=np.r_[91+np.array([0,5,9,13,17,18,19,20]),112+np.array([0,5,9,13,17,18,19,20])]
 good &= av[:,ids].all(1)&np.isfinite(p[:,ids]).all((1,2))
 for i in np.flatnonzero(good):
  vals=[]
  for active,other in[(91,112),(112,91)]:
   edge=p[i,active+np.array([17,18,19,20])];palm=p[i,other+np.array([0,5,9,13,17])];diam=float(np.linalg.norm(palm[:,None]-palm[None,:],axis=-1).max());hull=cv2.convexHull(palm.astype(np.float32));d=np.array([max(0.,-cv2.pointPolygonTest(hull,tuple(v.astype(float)),True))for v in edge])
   if diam<=1e-5:good[i]=False;break
   vals.extend([float(d.min()),float(d.mean()),float(d.min()/diam),float(d.mean()/diam)])
  if good[i]:extra[i]=vals
 return np.concatenate([base,extra],axis=1),good

class ArrivalRelation(PalmRelation):
 def __init__(self,run_dir=None):
  super().__init__();self.extra=[];self.channels=None
  for seed in[42,43,44]:
   ck=torch.load((run_dir or K/'runs')/f'{seed}.pt',map_location='cpu',weights_only=False);m=PhaseNet(len(ck['mean']),'frame_mlp');m.load_state_dict(ck['state']);m.eval();self.extra.append((m,ck['mean'],ck['std']))
 def __call__(self,f,unused,ts):
  original,prob=super().__call__(f,unused,ts);x,good=projected_features(f,self.channels);pred=[];conf=[]
  for m,mean,std in self.extra:
   with torch.no_grad():p=torch.softmax(m(torch.tensor((x-mean)/std,dtype=torch.float32)[None])[0],-1).numpy()
   pred.append(p.argmax(1));conf.append(p.max(1))
  pred=np.array(pred);conf=np.array(conf);near=good&(pred==2).all(0)&(conf>=.65).all(0)
  # Add only a source-trained near endpoint; retain every prior state otherwise.
  self.added=np.flatnonzero(near&(original!=2)).tolist();original=original.copy();original[near]=2
  return original,prob

class ArrivalLesson(PlaneViewLesson):
 def __init__(self):super().__init__();self.body_relation=ArrivalRelation()
 def analyze(self,raw,lesson=None):
  self.body_relation.channels=channels(raw);q=super().analyze(raw,lesson);q['projected_arrival_added']=self.body_relation.added
  return q

def train(extra=None,extra_sources=None,output='runs'):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=K/output;out.mkdir(exist_ok=True);assert not(out/'TRAINED.json').exists();rows=read(J/'relation/FROZEN.json')['rows']+(extra or []);by={r['id']:r for r in sources()+(extra_sources or [])};net=Features();cache={};xx=[];yy=[];included=[]
 if not(K/'CONTRACT.json').exists():write(K/'CONTRACT.json',dict(cause='Observed approach/separation present but frame posture stays separated in11_L. Explicit observed edge/palm distance is missing from51-feature representation.',features='51body/palm +8 symmetric edge/hull distances, body-normalized and observed palm-diameter normalized; no word/person/duration desired-count features.',split='Existing public FIT162 state anchors only; no DEV or private labels.',training='Fixed3seeds,800steps,lr.001; old51weights initialization/new columns zero, confidence.65 unchanged.',gate='No new normal warning/lost support; no extra completion on partial/constant/missing; preserve known error corrections and latency.',scope='Independent candidate. Deployed J unchanged. No hidden-finger interpolation or physical contact inference.'))
 for r in rows:
  cid=r['clip']
  if cid not in cache:
   ch=channels(dict(np.load(by[cid]['raw'])));f,_,_=net(ch);x,good=projected_features(f,ch);cache[cid]=(ch,x,good)
  ch,x,good=cache[cid];i=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  if good[i]:xx.append(x[i]);yy.append(r['state']);included.append(r)
 x=np.array(xx,np.float32);y=torch.tensor(yy);mean=x.mean(0);std=np.maximum(x.std(0),.01);tx=torch.tensor((x-mean)/std);counts=np.bincount(yy,minlength=3);cw=torch.tensor(len(y)/(3*counts),dtype=torch.float32)
 write(out/'FROZEN.json',dict(rows=included,features=59,steps=800,lr=.001,confidence=.65,seeds=[42,43,44],class_counts=counts.tolist(),private_data=False))
 runs=[]
 for seed in[42,43,44]:
  ck=torch.load(E/'palm_relation'/f'{seed}.pt',map_location='cpu',weights_only=False);torch.manual_seed(seed);m=PhaseNet(59,'frame_mlp');state=m.state_dict()
  for key,value in ck['state'].items():
   if state[key].shape==value.shape:state[key]=value
   else:
    assert value.ndim==2 and value.shape[1]==51 and state[key].shape[1]==59
    state[key].zero_();state[key][:,:51]=value
  m.load_state_dict(state);runmean=np.r_[ck['mean'],mean[51:]];runstd=np.r_[ck['std'],std[51:]];tx=torch.tensor((x-runmean)/runstd,dtype=torch.float32);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
  for _ in range(800):
   loss=torch.nn.functional.cross_entropy(m(tx[None])[0],y,weight=cw);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),2);opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=runmean,std=runstd),p);runs.append(dict(seed=seed,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 write(out/'TRAINED.json',dict(runs=runs,fit_rows=len(x),private_data=False,initialized='Existing51-feature head copied,8 added columns initiallyzero; backbone frozen.'));print('ARRIVAL_TRAINED',len(x),counts.tolist(),flush=True)
if __name__=='__main__':train()
