"""Long-finger observations shared across hands and vocabulary.

Geometric relations are INPUT features, not angle-threshold classification.
No word, expected handshape, description or synthetic-condition input.
"""
import numpy as np,torch
from torch import nn

def features(x,hand):
    p=x[:,:102].reshape(-1,51,2);v=x[:,186:237]>.5;start=30 if hand=='R' else 9
    p=p[:,start:start+21];v=v[:,start:start+21]
    scale=np.linalg.norm(p[:,9]-p[:,0],axis=1).clip(1e-5)
    out=[];valid=[]
    def norm(a):return np.linalg.norm(a,axis=-1).clip(1e-5)
    def cosine(a,b):return np.clip((a*b).sum(-1)/norm(a)/norm(b),-1,1)
    for k in [5,9,13,17]:
        q=p[:,k:k+4];u=np.diff(q,axis=1);lens=norm(u);base=q[:,0]-p[:,0];end=q[:,3]-q[:,0]
        # Direction-independent bending/extension ratios plus distance to palm.
        f=np.stack([cosine(u[:,0],u[:,1]),cosine(u[:,1],u[:,2]),cosine(u[:,0],u[:,2]),norm(end)/lens.sum(1),norm(q[:,3]-p[:,0])/scale,norm(q[:,1]-p[:,0])/scale,norm(q[:,2]-p[:,0])/scale,norm(base)/scale,cosine(base,end),cosine(base,u[:,0]),lens[:,0]/scale,lens[:,1]/scale,lens[:,2]/scale],-1)
        ok=v[:,0]&v[:,9]&v[:,k:k+4].all(1);f=np.clip(f,-4,4);f[~ok]=0
        out.append(np.c_[f,ok].astype(np.float32));valid.append(ok)
    return np.stack(out,1),np.stack(valid,1)

def windows(x,frames,radius=3):
    ids=np.asarray(frames)[:,None]+np.arange(-radius,radius+1);return x[np.clip(ids,0,len(x)-1)]

class SharedFinger(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Sequential(nn.Conv1d(14,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,16,bidirectional=True,batch_first=True);self.head=nn.Linear(32,1)
    def forward(self,x):
        # B,T,4,F -> B*4,T,F. Each finger uses the same learned weights.
        b,t,n,f=x.shape;q=x.permute(0,2,1,3).reshape(b*n,t,f);q=self.conv(q.transpose(1,2)).transpose(1,2);h,_=self.gru(q);return self.head(h[:,t//2]).reshape(b,n)
