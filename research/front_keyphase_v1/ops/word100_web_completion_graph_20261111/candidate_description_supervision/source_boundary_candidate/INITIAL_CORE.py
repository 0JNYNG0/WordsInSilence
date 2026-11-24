from framing_boundary_core import *
B=W/'source_boundary_candidate'

def body_context(q,ts):
 q=np.asarray(q).reshape(-1,10,10);v=np.concatenate([q[...,:4],q[...,8:9]],-1).reshape(-1,50)
 tt=np.asarray(ts);lo=np.searchsorted(tt,tt-HALO_US);hi=np.searchsorted(tt,tt+HALO_US,side='right')
 before=np.stack([v[a:i+1].mean(0) for i,a in enumerate(lo)]);after=np.stack([v[i:b].mean(0) for i,b in enumerate(hi)])
 return np.c_[before,v,after].astype(np.float32)

def pool_windows(x,spans):
 # Temporal thirds and variation; no length, word, absolute time, or finger input.
 cs=np.r_[np.zeros((1,x.shape[1])),np.cumsum(x,axis=0,dtype=np.float64)];cs2=np.r_[np.zeros((1,x.shape[1])),np.cumsum(x*x,axis=0,dtype=np.float64)]
 a=spans[:,0];b=spans[:,1]+1;n=b-a;c1=a+(n+2)//3;c2=c1+(n+1)//3
 avg=lambda l,r:(cs[r]-cs[l])/(r-l)[:,None]
 m=avg(a,b);var=np.maximum((cs2[b]-cs2[a])/n[:,None]-m*m,0)
 return np.c_[avg(a,c1),avg(c1,c2),avg(c2,b),np.sqrt(var)].astype(np.float32)

class BodyWindow(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(600,96),nn.GELU(),nn.Linear(96,15))
 def forward(self,x):return self.head((x-self.mean)/self.std).reshape(-1,5,3)

class BoundaryHead(nn.Module):
 def __init__(self,mean,std):
  super().__init__();self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32));self.register_buffer('std',torch.tensor(std,dtype=torch.float32));self.head=nn.Sequential(nn.Linear(150,96),nn.GELU(),nn.Linear(96,15))
 def forward(self,x):return self.head((x-self.mean)/self.std).reshape(*x.shape[:-1],5,3)

def rank_boundary(logits,spans,j):
 # Start/end log-softmax normalization would be constant for all candidates.
 # Keep stable raw logits for ordering, log-sigmoid for interior presence.
 z=np.asarray(logits,dtype=np.float64);p=1/(1+np.exp(-np.clip(z[:,j,2],-700,700)));logp=-np.logaddexp(0,-z[:,j,2]);cs=np.r_[0,np.cumsum(logp)];ps=np.r_[0,np.cumsum(p)];a,b=spans.T;n=b-a+1
 score=z[a,j,0]+z[b,j,1]+(cs[b+1]-cs[a])/n
 return np.where((ps[b+1]-ps[a])/n>=.5,score,-np.inf)

def rank_window(logits):
 z=np.asarray(logits,dtype=np.float64)
 return np.where(z[:,0]>=0,(-np.logaddexp(0,-z)).sum(1),-np.inf)
