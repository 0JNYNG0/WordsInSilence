import numpy as np,torch
from torch import nn
FINGERS=['thumb','index','middle','ring','little']
def local_sequence(x,hand):
 p=x[:,:102].reshape(-1,51,2);v=x[:,186:237]>.5;b=30 if hand=='R' else 9;p=p[:,b:b+21];v=v[:,b:b+21]
 axis=p[:,9]-p[:,0];length=np.linalg.norm(axis,axis=1);base=v[:,0]&v[:,9]&(length>1e-6);ey=axis/np.maximum(length[:,None],1e-6);ex=np.c_[ey[:,1],-ey[:,0]];sgn=np.where(((p[:,5]-p[:,17])*ex).sum(1)<0,-1,1);ex*=sgn[:,None]
 scale=np.median(np.linalg.norm(p[:,[5,9,13,17]]-p[:,0:1],axis=-1),axis=1).clip(1e-6);rel=p-p[:,0:1];q=np.stack([(rel*ex[:,None]).sum(-1),(rel*ey[:,None]).sum(-1)],-1)/scale[:,None,None];q[~v]=0;q[~base]=0
 known=np.stack([base&v[:,i:i+4].all(1)for i in[1,5,9,13,17]],1)
 return np.c_[q.reshape(-1,42),v].astype(np.float32),known
def windows(z,indices,radius=3):
 ids=np.asarray(indices)[:,None]+np.arange(-radius,radius+1)[None];return z[np.clip(ids,0,len(z)-1)]
class FingerHead(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(127,48,3,padding=1),nn.GELU());self.gru=nn.GRU(48,24,bidirectional=True,batch_first=True);self.output=nn.Linear(48,5)
 def forward(self,x):
  h=self.conv(x.transpose(1,2)).transpose(1,2);h,_=self.gru(h);return self.output(h[:,x.shape[1]//2])
def baseline_fingers(prob,hand):
 op=prob[:,1 if hand=='R' else 2];cl=prob[:,3 if hand=='R' else 4];n=len(prob);e=np.repeat(op[:,None],5,1);f=np.repeat(cl[:,None],5,1);e[:,0]=0;f[:,0]=0
 if hand=='R':e[:,1]=np.maximum(e[:,1],prob[:,0]);f[:,2:]=np.maximum(f[:,2:],prob[:,0:1])
 return np.where((e>=.85)&(f<.35),1,np.where((f>=.85)&(e<.35),0,-1))
