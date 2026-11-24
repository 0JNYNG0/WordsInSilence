"""Small learned direction/order readouts over the existing Conv+GRU observations."""
from common import *
import torch
from torch import nn
sys.path.insert(0,str(J))
from feedback_model import PoseGuideNet,collate

BASE=V/'feedback_runs/pose_transfer_42/model.pt'
class SemanticNet(nn.Module):
 def __init__(self):
  super().__init__();s=read(V/'FEEDBACK_NORMALIZATION.json')
  self.base=PoseGuideNet(s['guide_vectors'],s['guide_words'],s['prototype_lengths'])
  self.direction=nn.Sequential(nn.Linear(192,64),nn.GELU(),nn.Linear(64,15))
  self.order=nn.Sequential(nn.Linear(224,48),nn.GELU(),nn.Linear(48,1))
 def pair(self,h,a,b,word):
  q=torch.cat([h[a],h[b],h[b]-h[a]],-1)
  # Observation direction never sees the requested word or expected direction.
  direction=self.direction(q).reshape(5,3)
  g=self.base.guide_encoder(self.base.guide_vectors)+self.base.stage_embedding
  g=g[self.base.guide_words==word].mean(0)
  order=self.order(torch.cat([q,g],-1)).squeeze(-1)
  return direction,order

def availability(x):
 v=x[:,186:237]>.5;out=[]
 for j in range(7):
  pts=[9,30] if j in [5,6] else ([9]+list(range(14,30)) if j in [2,4] else [30]+list(range(35,51)))
  out.append(v[:,pts].all(1))
 return np.stack(out,1)

def pair_available(x,a,b,axis):
 v=x[:,186:237]>.5
 pts=[[1,2,6],[2,4,6],[3,5,30,35,39,43,47],[9,30],[9,30]][axis]
 return bool(v[a:b+1,pts].all(1).mean()>=.8)

def order_available(x,a,b,word):
 axes={0:[0,1],1:[2],2:[4],3:[3]}[int(word)]
 return all(pair_available(x,a,b,j) for j in axes)

def load_rows():
 out=[]
 for r in read(U/'READY.json')['rows']:
  assert sha(r['path'])==r['sha256'];z=np.load(r['path'],allow_pickle=False)
  out.append(dict(r,**{k:z[k].copy() for k in z.files}))
 return out

def transform(r,reverse=False,scale=1.,view='full',gap=0):
 q=dict(r);x=r['x_full'] if view=='full' else r['x_upper'];n=len(x)
 ids=np.arange(n)
 direct=np.flatnonzero(r['stage_gold']>=0)
 if reverse and len(direct):
  lo=int(min([direct[0]]+[p['a'] for p in r['pairs']]));hi=int(max([direct[-1]]+[p['b'] for p in r['pairs']]))
  ids[lo:hi+1]=ids[lo:hi+1][::-1]
 inverse=np.argsort(ids)
 pairs=[];orders=[]
 for p in r['pairs']:
  a,b=int(inverse[p['a']]),int(inverse[p['b']]);target=p['target'] if a<b else 2-p['target']
  pairs.append(dict(p,a=min(a,b),b=max(a,b),target=target))
 for p in r['orders']:
  a,b=int(inverse[p['a']]),int(inverse[p['b']]);orders.append(dict(p,a=min(a,b),b=max(a,b),target=int(a<b)))
 for k in ['y','ym','sy','sw','ci','old_y','old_ym','direct_ym','stage_gold','raw_x']:
  q[k]=r[k][ids].copy()
 q['x']=x[ids].copy();q['frame_id']=r['frame_id'].copy();q['timestamp_us']=r['timestamp_us'].copy()
 if reverse:q['ci'][:]=-1
 if scale!=1:
  # Nearest-neighbour temporal resampling preserves every short original event
  # for slow control; fast control explicitly may skip frames and is reported.
  ix=np.rint(np.linspace(0,n-1,max(2,round(n*scale)))).astype(int)
  for k in ['x','y','ym','sy','sw','ci','old_y','old_ym','direct_ym','stage_gold','raw_x']:q[k]=q[k][ix]
  mp={i:int(np.argmin(abs(ix-i))) for i in range(n)}
  for p in pairs+orders:p['a'],p['b']=mp[p['a']],mp[p['b']]
  q['frame_id']=np.arange(len(ix));q['timestamp_us']=np.arange(len(ix))*np.median(np.diff(r['timestamp_us']))
  q['source_indices']=ix
 else:q['source_indices']=np.arange(n)
 q['pairs']=pairs;q['orders']=orders;q['reversed']=reverse
 if gap:
  middle=int(np.median(np.flatnonzero(q['stage_gold']>=0))) if len(direct) else len(q['x'])//2
  lo=max(0,middle-gap//2);hi=min(len(q['x']),lo+gap)
  # Explicitly absent hand/arm evidence. No interpolation or fake ground truth.
  q['x'][lo:hi,:279]=0;q['raw_x'][lo:hi,:279]=0;q['gap_span']=[lo,hi]
 return q

def batch_losses(model,rows,pw):
 from torch.nn import functional as F
 b=collate(rows);o=model.base(b['x'],b['lengths'],b['word'])
 mask=b['ym'].clone()
 for i,r in enumerate(rows):mask[i,:len(r['x'])]*=torch.from_numpy(availability(r['x']))
 attr=(F.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*mask).sum()/mask.sum().clamp_min(1)
 ce=F.cross_entropy(o['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none')
 stage=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
 valid=(b['ci']>=0)&o['valid'];proto=model.base.prototype[b['word'][:,None],b['ci'].clamp_min(0)]
 align=((o['latent']-proto).square().mean(-1)*valid).sum()/valid.sum().clamp_min(1)
 dl=[];ol=[]
 for i,r in enumerate(rows):
  h=o['latent'][i]
  for p in r['pairs']:
   if pair_available(r['x'],p['a'],p['b'],p['axis']):
    d,_=model.pair(h,p['a'],p['b'],r['word']);dl.append(F.cross_entropy(d[p['axis']][None],torch.tensor([p['target']])))
  for p in r['orders']:
   if order_available(r['x'],p['a'],p['b'],r['word']):
    _,order=model.pair(h,p['a'],p['b'],r['word']);ol.append(F.binary_cross_entropy_with_logits(order,torch.tensor(float(p['target']))))
 zero=o['latent'].sum()*0
 return dict(attributes=attr,stage=stage,prototype=align,direction=torch.stack(dl).mean() if dl else zero,order=torch.stack(ol).mean() if ol else zero)
