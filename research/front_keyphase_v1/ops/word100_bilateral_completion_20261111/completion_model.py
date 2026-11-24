from bilateral_core import *

def observed_features(x,ts,a,b,axis,side):
 """Common bilateral geometry. No word, expected sign, identity or test-kind input."""
 vals,valid=prior.wrist_series(x,prior.motion.KINDS[axis],side,opposite(side))
 ts=np.asarray(ts,dtype=np.float64);a=int(a);b=int(b)
 if not (0<=a<b<len(ts)):raise ValueError('invalid span')
 # Two fixed real-time context windows + the candidate, separately resampled.
 lo=max(0,int(np.searchsorted(ts,ts[a]-160000)));hi=min(len(ts)-1,int(np.searchsorted(ts,ts[b]+160000)))
 sample=[];mask=[];ranges=[(lo,a,7),(a,b,17),(b,hi,7)]
 good=np.flatnonzero(valid & np.isfinite(vals).all(1));coverage=float(valid[a:b+1].mean())
 for l,r,n in ranges:
  tt=np.linspace(ts[l],ts[r],n);near=np.searchsorted(ts,tt).clip(0,len(ts)-1);prev=np.maximum(near-1,0);near=np.where(abs(ts[prev]-tt)<abs(ts[near]-tt),prev,near)
  vv=np.stack([np.interp(tt,ts[good],vals[good,k]) if len(good)>1 else np.zeros(n) for k in range(4)],1)
  mm=valid[near] & (abs(ts[near]-tt)<=100000);sample.append(vv);mask.append(mm.astype(np.float32))
 seq=np.concatenate(sample);obs=np.concatenate(mask);seq=(seq-seq[7])*obs[:,None]
 context=np.array([valid[lo:a+1].mean(),coverage,valid[b:hi+1].mean(),(ts[a]-ts[lo])/160000,(ts[hi]-ts[b])/160000,(ts[b]-ts[a])/1000000],np.float32)
 gaps=bc.groups(~valid[a:b+1]);maxgap=max((ts[a+d]-ts[a+c] for c,d in gaps),default=0)
 info=dict(coverage=coverage,max_missing_us=float(maxgap),before_ms=float((ts[a]-ts[lo])/1000),after_ms=float((ts[hi]-ts[b])/1000),eligible=bool(coverage>=.8 and valid[a] and valid[b] and maxgap<100000),side=side,axis=axis,span=[a,b],semantic_complete=False)
 return seq.astype(np.float32),obs, np.r_[np.eye(5,dtype=np.float32)[axis],context],info

class CompletionNet(nn.Module):
 def __init__(self,seed,norm):
  super().__init__();torch.manual_seed(seed)
  src=mc.MotionNet();src.load_state_dict(torch.load(M/'runs'/str(seed)/'model.pt',weights_only=True,map_location='cpu'),strict=False)
  self.conv=copy.deepcopy(src.motion_conv);self.gru=copy.deepcopy(src.motion_gru)
  self.register_buffer('scale',torch.tensor(norm,dtype=torch.float32))
  self.shared=nn.Sequential(nn.Linear(110,64),nn.GELU())
  self.direction=nn.Linear(64,3);self.boundaries=nn.Linear(64,2);self.quality=nn.Linear(64,4)
 def forward(self,seq,mask,meta):
  v=self.conv((seq/self.scale).transpose(1,2)).transpose(1,2);h,_=self.gru(v)
  pools=[]
  for a,b in [(0,7),(7,24),(24,31)]:
   m=mask[:,a:b,None];pools.append((h[:,a:b]*m).sum(1)/m.sum(1).clamp_min(1))
  z=self.shared(torch.cat(pools+[meta,torch.stack([mask[:,:7].mean(1),mask[:,7:24].mean(1),mask[:,24:].mean(1)],1)],1))
  return self.direction(z),self.boundaries(z),self.quality(z)

def infer_batch(net,x,ts,events):
 ff=[observed_features(x,ts,e['a'],e['b'],e['axis'],e['moving']) for e in events]
 if not ff:return []
 with torch.inference_mode():
  d,b,q=net(torch.tensor(np.stack([f[0] for f in ff])),torch.tensor(np.stack([f[1] for f in ff])),torch.tensor(np.stack([f[2] for f in ff])))
  dp,bp,qp=d.softmax(-1).numpy(),b.sigmoid().numpy(),q.softmax(-1).numpy()
 out=[]
 for f,dd,bb,qq in zip(ff,dp,bp,qp):
  i=f[3];direction_ok=i['eligible'] and dd.max()>=.8
  complete=i['eligible'] and i['before_ms']>=100 and i['after_ms']>=100 and min(bb)>=.8 and qq[2]>=.8
  out.append(dict(**i,direction_probability=dd.tolist(),direction=int(dd.argmax()),direction_readable=bool(direction_ok),boundary_probability=bb.tolist(),quality_probability=qq.tolist(),quality=int(qq.argmax()),physical_event_complete=bool(complete)))
 return out
