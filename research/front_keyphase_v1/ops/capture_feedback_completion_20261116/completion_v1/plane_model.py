"""Observed projected hand-plane classifier, NOT a 3D palm-normal estimator.

Shared hands, word-independent input. Classes refer to visible upright vs flat
forms reviewed in RGB. A prediction is not proof of front/back surface contact.
"""
from train_shape import *
from torch import nn

def hand_plane_features(ch,hand):
 start=91 if hand=='L'else 112
 p=ch['body_relative_xy'][:,start:start+21].copy();av=ch['point_available'][:,start:start+21]
 # Keep image orientation, normalize size/translation, canonicalize handedness.
 z=p-p[:,0:1];scale=np.linalg.norm(z[:,9],axis=1)
 good=av.all(1)&np.isfinite(p).all((1,2))&(scale>1e-5)
 z/=np.maximum(scale[:,None,None],1e-5)
 if hand=='L':z[:,:,0]*=-1
 z=np.clip(np.nan_to_num(z),-6,6)
 return z.reshape(len(z),42).astype(np.float32),good

class PlaneHead(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(42,24),nn.GELU(),nn.Linear(24,2))
 def forward(self,x):return self.net(x)

class PlaneObserver:
 def __init__(self):
  self.models=[]
  for seed in[42,43,44]:
   ck=torch.load(V/'plane'/f'{seed}.pt',map_location='cpu',weights_only=False);m=PlaneHead();m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))
 def sequence(self,ch,hand):
  x,a=hand_plane_features(ch,hand);qs=[]
  with torch.inference_mode():
   for m,mu,sd in self.models:
    p=m(torch.tensor((x-mu)/sd)).softmax(-1).numpy();q=p.argmax(-1);q[p.max(-1)<.85]=-1;qs.append(q)
  s=np.stack(qs);s=np.where((s==s[0]).all(0),s[0],-1);s[~a]=-1;return s

def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);out=V/'plane';out.mkdir(exist_ok=True);assert not(out/'FROZEN.json').exists()
 records=read(V/'REVIEW_INPUTS.json')['rows'];xx=[];yy=[];truth=[]
 for r in records:
  ch=dict(np.load(r['channels']));ix=int(np.flatnonzero(ch['frame_id']==r['frame'])[0])
  for h,y in [('R',0),('L',1)]:
   x,a=hand_plane_features(ch,h)
   if not a[ix]:continue
   xx.append(x[ix]);yy.append(y);truth.append(dict(clip_id=r['clip_id'],frame=r['frame'],hand=h,plane=y,
      source_sha256=r['source_sha256'],review='Native RGB p01..09 reviewed: upright right hand and flat left hand. Projected visible form only.'))
 x=np.array(xx);y=np.array(yy);mu=x.mean(0);sd=x.std(0).clip(.05)
 write(out/'SOURCE_TRUTH.json',dict(rows=truth,scope='Projected upright vs flat hand form, not arbitrary palm-front/back orientation'))
 write(out/'FROZEN.json',dict(steps=400,seeds=[42,43,44],lr=.001,confidence=.85,truth_sha=sha(out/'SOURCE_TRUTH.json'),
   input='Hand wrist-relative 21x2, scale-normalized, L mirrored to R; no side ID/body/word/time input.',
   all_people_fit=list(range(1,10)),acceptance='Public DEV observed forms; missing must abstain, swapping observed plane forms must not be accepted as unchanged.',
   limitation='Specific observed forms and camera views; no 3D surface certification. Whole word truth remains null.'))
 tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y);runs=[]
 for seed in[42,43,44]:
  torch.manual_seed(seed);m=PlaneHead();opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001);t=time.perf_counter()
  for _ in range(400):
   loss=F.cross_entropy(m(tx),ty);opt.zero_grad();loss.backward();opt.step()
  p=out/f'{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd),p);runs.append(dict(seed=seed,sha256=sha(p),seconds=time.perf_counter()-t))
 write(out/'TRAINED.json',dict(runs=runs));print('PLANE_TRAINED',len(truth))
if __name__=='__main__':main()
