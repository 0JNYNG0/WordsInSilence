import numpy as np
import torch
from torch import nn
CLASSES={'region':['face_front','upper_body'],'forearm':['across','upright']}

def body_frame(p):
    axis=p[:,2]-p[:,1];size=np.linalg.norm(axis,axis=1);ex=axis/np.maximum(size[:,None],1e-8)
    ey=np.stack([-ex[:,1],ex[:,0]],1);ey*=np.where(ey[:,1:2]>=0,1,-1)
    return (p[:,1]+p[:,2])/2,size,ex,ey

def face_geometry(xy,available):
    """Reuse COCO133 cached observations; no facial expression or contact inference."""
    body=np.asarray(xy)[:,[0,5,6]];o,s,ex,ey=body_frame(body)
    q=xy[:,23:91]-o[:,None]
    local=np.stack([(q*ex[:,None]).sum(-1),(q*ey[:,None]).sum(-1)],-1)/np.maximum(s[:,None,None],1e-8)
    ok=np.asarray(available)[:,[0,5,6]+list(range(23,91))].all(1)&(s>1e-6)&np.isfinite(local).all((1,2))
    # face8 is the central lower-jaw point. Keep the complete face observation
    # for verification; using its position is not verified physical visibility.
    jaw=local[:,:17];box=np.c_[local[:,:,0].min(1),local[:,:,0].max(1),local[:,:,1].min(1),jaw[:,:,1].max(1)]
    ok&=(box[:,1]-box[:,0]>1e-6)&(box[:,3]-box[:,2]>1e-6)
    return np.c_[local[:,8],box],ok

def features(x,domain,hand,face=None,face_ok=None):
    p=np.asarray(x)[:,:102].reshape(-1,51,2);v=np.asarray(x)[:,186:237]>.5
    root=30 if hand=='R'else 9;elbow=4 if hand=='R'else 3;shoulder=2 if hand=='R'else 1
    o,s,ex,ey=body_frame(p);mirror=1 if hand=='R'else-1
    def local(q):return np.stack([(q*ex).sum(1)*mirror,(q*ey).sum(1)],1)/np.maximum(s[:,None],1e-8)
    def unit(q):return q/np.maximum(np.linalg.norm(q,axis=1)[:,None],1e-8)
    upper=local(p[:,elbow]-p[:,shoulder]);lower=local(p[:,root]-p[:,elbow])
    if domain=='forearm':
        u=unit(upper);w=unit(lower)
        f=np.c_[u,w,upper,local(p[:,root]-p[:,shoulder]),(u*w).sum(1),np.linalg.norm(lower,axis=1)]
        ok=v[:,[1,2,elbow,root]].all(1)&(s>1e-6)
    else:
        ids=[root,root+5,root+9,root+13,root+17];palm=p[:,ids].mean(1)
        a=local(palm-o);b=local(palm-p[:,0]);c=local(palm-p[:,shoulder]);wrist=local(p[:,root]-o)
        face=np.asarray(face).copy();face[:,0]*=mirror
        if mirror<0:face[:,2:4]=-face[:,3:1:-1]
        chin=face[:,:2];xmin,xmax,ymin,ymax=face[:,2:].T
        face_center=np.c_[(xmin+xmax)/2,(ymin+ymax)/2];face_size=np.c_[xmax-xmin,ymax-ymin]
        f=np.c_[a,b,c,wrist,a-chin,(a-face_center)/np.maximum(face_size,1e-6),
                np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1),upper,face_size]
        ok=v[:,ids+[0,1,2,elbow]].all(1)&(s>1e-6)&np.asarray(face_ok)
    ok&=np.isfinite(f).all(1);f=np.clip(np.nan_to_num(f),-10,10).astype('float32');f[~ok]=0
    return f,ok

class BodyHeads(nn.Module):
    def __init__(self,dims):
        super().__init__();self.heads=nn.ModuleDict({d:nn.Sequential(nn.Linear(n,24),nn.Tanh(),nn.Linear(24,2))for d,n in dims.items()})
    def forward(self,x,d):return self.heads[d](x)

def consensus(pp,ok,d):
    vote=pp.argmax(-1);known=ok&(vote==vote[:1]).all(0)&(pp.max(-1)>=.8).all(0)
    return [CLASSES[d][int(vote[0,i])]if known[i]else None for i in range(len(ok))]
