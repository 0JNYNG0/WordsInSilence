"""Projected hand/face extent, not facial expression or physical contact."""
import numpy as np
import torch
from torch import nn
from body_core import body_frame

def features(x, face, face_ok, hand):
    p=x[:,:102].reshape(-1,51,2);v=x[:,186:237]>.5
    o,s,ex,ey=body_frame(p);r=30 if hand=='R' else 9
    q=p[:,r:r+21]-o[:,None]
    h=np.stack([(q*ex[:,None]).sum(-1),(q*ey[:,None]).sum(-1)],-1)/np.maximum(s[:,None,None],1e-8)
    xmin,xmax,ymin,ymax=face[:,2:].T;size=np.c_[xmax-xmin,ymax-ymin]
    c=np.c_[(xmin+xmax)/2,(ymin+ymax)/2]
    z=(h-c[:,None])/np.maximum(size[:,None],1e-6)
    # Distance of each observed hand point from the face rectangle. Positive
    # outside, negative inside. Minimum captures fingers reaching the face even
    # when the palm/wrist are below it. Same function for left and right.
    dx=np.abs(z[:,:,0])-.5;dy=np.abs(z[:,:,1])-.5
    outside=np.linalg.norm(np.maximum(np.stack([dx,dy],-1),0),axis=-1)
    signed=outside+np.minimum(np.maximum(dx,dy),0)
    palm=z[:,[0,5,9,13,17]].mean(1)
    f=np.c_[np.quantile(signed,[0,.25,.5],axis=1).T,
            z[:,:,1].min(1)-.5,z[:,:,1].max(1)-.5,
            np.maximum(np.abs(palm[:,0])-.5,0),palm[:,1]-.5]
    ok=v[:,list(range(r,r+21))+[1,2]].all(1)&face_ok&(s>1e-6)&np.isfinite(f).all(1)
    f=np.clip(np.nan_to_num(f),-10,10).astype('float32');f[~ok]=0
    return f,ok

class RegionExtent(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Linear(7,2)
    def forward(self,x):return self.net(x)
