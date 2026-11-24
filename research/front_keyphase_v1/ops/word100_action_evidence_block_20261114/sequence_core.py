"""Hand-only ordered evidence; no word ID, source speed or expected action input."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'word100_transition_observation_audit_20261114'))
from change_core import geometry
CLASSES=['opening','shape_maintained','closing']

def features(x,hand):
    x=np.asarray(x);g,ok=geometry(x,hand);base=30 if hand=='R'else 9
    p=x[:,:102].reshape(-1,51,2)[:,base:base+21]
    q=p-p[:,:1];axis=q[:,9];size=np.maximum(np.linalg.norm(axis,axis=1),1e-6)
    ey=axis/size[:,None];ex=np.stack([ey[:,1],-ey[:,0]],1)
    # Right/left are canonicalized, without absolute body/screen positions.
    local=np.stack([(q*ex[:,None]).sum(-1),(q*ey[:,None]).sum(-1)],-1)/size[:,None,None]
    if hand=='L':local[:,:,0]*=-1
    ids=[0]+list(range(5,21));coords=local[:,ids].reshape(len(x),-1)
    palm=np.stack([np.linalg.norm(q[:,5]-q[:,17],axis=1)/size,
        np.linalg.norm(q[:,5],axis=1)/size,np.linalg.norm(q[:,17],axis=1)/size,
        (local[:,5,0]*local[:,17,1]-local[:,5,1]*local[:,17,0])],1)
    f=np.concatenate([g,coords,palm,ok[:,None]],1)
    f=np.clip(np.nan_to_num(f),-10,10).astype('float32');f[~ok]=0
    return f,ok

def sequence(x,hand,length=16):
    f,ok=features(x,hand);ids=np.rint(np.linspace(0,len(x)-1,length)).astype(int)
    return f[ids],bool(len(x)>=2 and ok[0]and ok[-1]and ok.mean()>=.8)

class SequenceHead(nn.Module):
    def __init__(self,dim=59):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(dim,32),nn.Tanh())
        self.conv=nn.Conv1d(32,32,3,padding=1)
        self.gru=nn.GRU(32,24,batch_first=True)
        self.change=nn.Linear(24+64,3)
        self.pose=nn.Linear(32,3)
    def forward(self,x):
        z=self.embed(x);h=torch.tanh(self.conv(z.transpose(1,2))).transpose(1,2)
        _,last=self.gru(h)
        return self.change(torch.cat([last[-1],z[:,-1]-z[:,0],(z[:,-1]+z[:,0])/2],1)),self.pose(z)

def decide(pp,available):
    votes=pp.argmax(-1)
    return CLASSES[int(votes[0])]if available and len(set(votes.tolist()))==1 and pp.max(-1).min()>=.8 else None
