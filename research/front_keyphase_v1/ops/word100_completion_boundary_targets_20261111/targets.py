from boundary_core import *

def boundary_target(a,b,r):
 if b<r[0] or a>r[1]:return 0.,1.
 if a<=r[0] and b>=r[1]:return 1.,1.
 return .5,0. # Inside the annotation uncertainty band: do not invent certainty.

def supervised_spans(ann,n):
 spans={}
 def add(a,b,quality,direction,kind,evs=None,qmask=1.):
  a=max(0,int(a));b=min(n-1,int(b))
  if b-a<3:return
  if evs:
   s,sm=boundary_target(a,b,evs[0]['onset']);e,em=boundary_target(a,b,evs[-1]['offset'])
  else:s=e=sm=em=0.
  key=(a,b);row=dict(a=a,b=b,quality=quality,quality_mask=qmask,boundaries=[s,e],boundary_mask=[sm,em],direction=direction,kind=kind)
  if key in spans:
   oldrow=spans[key]
   if oldrow['quality']!=quality:oldrow['quality_mask']=0.
   for i in range(2):
    if oldrow['boundaries'][i]!=row['boundaries'][i]:oldrow['boundary_mask'][i]=0.
  else:spans[key]=row
 events=ann['events']
 for j,e in enumerate(events):
  a,b=e['onset'][0],e['offset'][1];mid=(e['onset'][1]+e['offset'][0])//2
  # Full examples span the uncertain boundary bands, including modest context.
  for da,db in [(0,0),(-2,0),(0,2),(-2,2)]:
   aa,bb=a+da,b+db
   other=any(k!=j and min(bb,f['offset'][0])-max(aa,f['onset'][1])>=2 for k,f in enumerate(events))
   if not other:add(aa,bb,2,e['direction'],'whole', [e])
  if e['onset'][1]<mid<e['offset'][0]:
   add(a,mid,1,e['direction'],'prefix',[e]);add(mid,b,1,e['direction'],'suffix',[e])
  add(e['onset'][1]+1,e['offset'][0]-1,1,e['direction'],'middle',[e])
  # Borderline crops contribute only unambiguous boundary/direction targets.
  add((sum(e['onset'])//2),b,2,e['direction'],'onset_ambiguous',[e],0.)
  add(a,(sum(e['offset'])//2),2,e['direction'],'offset_ambiguous',[e],0.)
 for e,f in zip(events,events[1:]):add(e['onset'][0],f['offset'][1],3,-100,'multiple',[e,f])
 for a,b in ann['holds']:add(a,b,0,1,'hold')
 for a,b in ann['background']:add(a,b,0,-100,'background')
 return list(spans.values())

class EndpointNet(prev.CompletionNet):
 def __init__(self,seed,norm,mode):
  super().__init__(seed,norm);self.load_state_dict(torch.load(P/'runs'/str(seed)/'model.pt',map_location='cpu',weights_only=True));self.mode=mode
  if mode=='endpoints':
   torch.manual_seed(seed+1000)
   self.endpoint=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Linear(64,64));nn.init.zeros_(self.endpoint[-1].weight);nn.init.zeros_(self.endpoint[-1].bias)
   self.endpoint_boundary=nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,2))
 def forward(self,seq,mask,meta):
  v=self.conv((seq/self.scale).transpose(1,2)).transpose(1,2);h,_=self.gru(v);pools=[]
  for a,b in [(0,7),(7,24),(24,31)]:
   m=mask[:,a:b,None];pools.append((h[:,a:b]*m).sum(1)/m.sum(1).clamp_min(1))
  z=self.shared(torch.cat(pools+[meta,torch.stack([mask[:,:7].mean(1),mask[:,7:24].mean(1),mask[:,24:].mean(1)],1)],1))
  if self.mode=='endpoints':
   local=torch.cat([z,h[:,7],h[:,23]],-1);return self.direction(z),self.endpoint_boundary(local),self.quality(z+self.endpoint(local))
  return self.direction(z),self.boundaries(z),self.quality(z)

def predictions(net,x,ts,events):
 ff=[prev.observed_features(x,ts,e['a'],e['b'],e['axis'],e['moving']) for e in events]
 if not ff:return []
 with torch.inference_mode():
  d,b,q=net(torch.tensor(np.stack([f[0] for f in ff])),torch.tensor(np.stack([f[1] for f in ff])),torch.tensor(np.stack([f[2] for f in ff])))
  dp,bp,qp=d.softmax(-1).numpy(),b.sigmoid().numpy(),q.softmax(-1).numpy()
 out=[]
 for f,dd,bb,qq in zip(ff,dp,bp,qp):
  i=f[3];common=i['eligible'] and i['before_ms']>=100 and i['after_ms']>=100
  out.append(dict(**i,direction_probability=dd.tolist(),direction=int(dd.argmax()),boundary_probability=bb.tolist(),quality_probability=qq.tolist(),quality=int(qq.argmax()),complete_strict=bool(common and min(bb)>=.8 and qq[2]>=.8),complete_joint=bool(common and qq[2]>=.8)))
 return out
