"""Source-labelled shared region and directed-progress probes on frozen100 features.

No word ID, requested goal or normalized clip time is fed to the heads.
Region finding and shape judging are deliberately independent.
"""
from apple_io import *
sys.path.insert(0,str(V))
from view_conditioned_shape import ViewConditionedShape
from core import Features
import adapter,torch,time
from torch import nn
from torch.nn import functional as F
class Head(nn.Module):
 def __init__(self,n,out):super().__init__();self.net=nn.Sequential(nn.Linear(n,48),nn.GELU(),nn.Linear(48,out))
 def forward(self,x):return self.net(x)
def relation(ch):
 xy=ch['body_relative_xy'];av=ch['point_available'];rp=xy[:,[112,117,121,125,129]].mean(1);lp=xy[:,[91,96,100,104,108]].mean(1)
 mouth=xy[:,[71,74,77,80]].mean(1)
 return np.c_[rp-mouth,rp-lp].astype('float32'),np.c_[av[:,[71,74,77,80,112,117,121,125,129]].all(1),av[:,[91,96,100,104,108,112,117,121,125,129]].all(1)]
def progress_input(values):
 # Endpoint change plus actual middle observations. No duration or word input.
 v=np.asarray(values);idx=np.linspace(0,len(v)-1,7).round().astype(int);z=v[idx]-v[0]
 return np.r_[z.ravel(),v[-1]-v[0]].astype('float32')
def train(name,x,y,steps=500):
 path=U/'runs';path.mkdir(exist_ok=True);mu=x.mean(0);sd=x.std(0).clip(.03);tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y);nclass=int(y.max())+1
 ww=np.array([1/max(1,(y==i).sum())for i in range(nclass)],np.float32);w=torch.tensor(ww/ww.mean());info=[]
 for seed in [42,43,44]:
  torch.manual_seed(seed);m=Head(x.shape[-1],nclass);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.005);st=time.perf_counter()
  for _ in range(steps):
   loss=F.cross_entropy(m(tx),ty,weight=w);opt.zero_grad();loss.backward();opt.step()
  p=path/f'{name}_{seed}.pt';torch.save(dict(state=m.state_dict(),mean=mu,std=sd,out=nclass),p);info.append(dict(seed=seed,seconds=time.perf_counter()-st,sha256=sha(p)))
 return info
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);assert not(U/'FROZEN.json').exists()
 rows=read(U/'INPUTS.json')['rows'];truth={r['person']:r for r in read(U/'SOURCE_TRUTH.json')['rows']};rx=[];ry=[];px=[];py=[];records=[]
 for r in rows:
  if r['role']!='fit':continue
  t=truth[r['person']];z=np.load(U/'features'/f'p{r["person"]:02d}.npz');f=z['features'];ch=dict(np.load(r['channels']));rel,av=relation(ch)
  anchors={0:list(range(0,max(1,t['mouth_anchors'][0]-10),5))+list(range(t['pair_anchors'][-1]+7,len(f),3)),1:t['mouth_anchors'],2:t['pair_anchors']}
  for y,ids in anchors.items():
   for i in ids:
    if y>0 and not av[i,y-1]:continue
    # Explicit normalized geometry + frozen common Conv/GRU context.
    rx.append(np.r_[np.nan_to_num(rel[i]),av[i],f[i]]);ry.append(y);records.append(dict(person=r['person'],frame=i,region=y))
  for action,(a,b) in enumerate([t['mouth_sweep'],t['wipe_sweeps'][0]]):
   v=rel[a:b+1,action*2:action*2+2];valid=av[a:b+1,action]
   if not valid.all():continue
   for scale in [.8,1.,1.2]:
    v2=v*scale
    # Labels describe directed observed displacement, not 'correct apple'.
    good=1 if action==0 else 2;bad=3 if action==0 else 4
    for vals,label in [(v2,good),(v2[::-1],bad),(np.repeat(v2[0:1],len(v2),0),0)]:px.append(progress_input(vals));py.append(label)
 write(U/'FROZEN.json',dict(source_sha=sha(U/'SOURCE_TRUTH.json'),fit_people=[1,2,3,4,6,7,8,9],development_people=[11,12,13,14,15],
  steps=500,seeds=[42,43,44],thresholds=dict(region=.70,progress=.75),class_names=dict(region=['background','near_mouth','at_opposite_hand'],progress=['hold','signer_right','down','signer_left','up']),
  immutable_backbone='100-word shared Conv+BiGRU frozen; no overwrite',expected=['normal five public DEV: no new wrong shape claims; shape changes/partial/reverse/order/missing controls'],
  scope='Small shared observation probes, not whole-word correctness. Existing gratitude and original four unchanged. Original-body normalized, no absolute pixels.',
  exposure='All existing DEV previously related-model-exposed. Need additional public held person before web.',training_records=records,counts=dict(region=len(rx),progress=len(px))))
 runs={name:train(name,np.array(x),np.array(y))for name,x,y in[('region',rx,ry),('progress',px,py)]};write(U/'TRAINED.json',dict(runs=runs,web_admitted=False));print('TRAINED',len(rx),len(px))
if __name__=='__main__':main()
