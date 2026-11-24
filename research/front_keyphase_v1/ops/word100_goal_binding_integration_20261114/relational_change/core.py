"""Two independent body-relative changes; no absolute forearm pose classes."""
import numpy as np
import torch
from torch import nn

def values(x,hand='R',palm=True):
    x=np.asarray(x);p=x[:,:102].reshape(-1,51,2);valid=x[:,186:237]>.5
    center=(p[:,1]+p[:,2])/2;axis=p[:,2]-p[:,1];scale=np.linalg.norm(axis,axis=1)
    right=axis/np.maximum(scale[:,None],1e-8);down=np.stack([-right[:,1],right[:,0]],1);down*=np.where(down[:,1:2]>=0,1,-1)
    e,w,h=(4,6,30)if hand=='R'else(3,5,9)
    palm_ids=[h,h+5,h+9,h+13,h+17];handpoint=p[:,palm_ids].mean(1)if palm else p[:,w]
    height=((handpoint-center)*down).sum(1)/np.maximum(scale,1e-8)
    forearm=p[:,w]-p[:,e];length=np.linalg.norm(forearm,axis=1)
    upright=-(forearm*down).sum(1)/np.maximum(length,1e-8)
    shared=valid[:,[1,2]].all(1)&(scale>1e-6)
    hv=shared&(valid[:,palm_ids].all(1)if palm else valid[:,w])
    fv=shared&valid[:,[e,w]].all(1)&(length>1e-6)
    return np.stack([height,upright],1),np.stack([hv,fv],1)

def measurement(x,ts,span,axis,hand='R',palm=True):
    a,b=span;v,ok=values(x[a:b+1],hand,palm);tt=np.asarray(ts)[a:b+1]
    valid=ok[:,axis]&np.isfinite(v[:,axis]);ids=np.flatnonzero(valid)
    available=bool(len(ids)>=2 and valid[0]and valid[-1] and valid.mean()>=.8 and np.max(np.diff(tt[ids]))<=120000)
    delta=float(v[-1,axis]-v[0,axis])if available else 0.
    return delta,available,dict(span=[int(a),int(b)],observed_fraction=float(valid.mean()),
      endpoint_values=v[[0,-1],axis].tolist(),native_observations=len(tt),anatomical_hand=hand)

class ChangeHeads(nn.Module):
    def __init__(self):
        super().__init__();self.raw_scale=nn.Parameter(torch.zeros(2));self.raw_band=nn.Parameter(torch.full((2,),-3.))
    def forward(self,delta,axis):
        scale=torch.nn.functional.softplus(self.raw_scale[axis]);band=torch.nn.functional.softplus(self.raw_band[axis])
        return torch.stack([-scale*delta,scale*band.expand_as(delta),scale*delta],-1)

def consensus(prob,available):
    p=np.asarray(prob);votes=p.argmax(-1)
    good=np.asarray(available)&(votes==votes[:1]).all(0)&(p.max(-1)>=.8).all(0)
    return [int(votes[0,i])if good[i]else None for i in range(len(good))]
