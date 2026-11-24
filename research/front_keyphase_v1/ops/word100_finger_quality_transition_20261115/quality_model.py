import sys
from pathlib import Path
import numpy as np,torch
from torch import nn
O=Path(__file__).resolve().parent.parent;sys.path[:0]=[str(O/'word100_finger_repair_20261115'),str(O/'word100_clause_finger_diagnostics_20261114')]
from finger_model import features as geometry,windows
from finger_core import local_sequence
def features(x,hand):
 f,valid=geometry(x,hand);loc,_=local_sequence(x,hand);p=loc[:,:42].reshape(-1,21,2);out=[]
 for j,k in enumerate([5,9,13,17]):
  q=p[:,k:k+4];others=[b for b in [5,9,13,17]if b!=k];d=np.stack([np.linalg.norm(q[:,3]-p[:,b+3],axis=1)for b in others],1);lengths=np.linalg.norm(np.diff(q,axis=1),axis=-1);other_lengths=np.stack([np.linalg.norm(np.diff(p[:,b:b+4],axis=1),axis=-1).sum(1)for b in others],1);rel=lengths.sum(1)[:,None]/other_lengths.clip(.01)
  base=np.concatenate([f[:,j],q.reshape(-1,8),d,rel],1);delta=np.zeros_like(base);delta[1:]=base[1:]-base[:-1];ok=valid[:,j];delta[1:]*=(ok[1:]&ok[:-1])[:,None];z=np.c_[base,np.clip(delta,-2,2)];z[~ok]=0;out.append(z)
 return np.clip(np.stack(out,1),-8,8).astype(np.float32),valid
class QualityChange(nn.Module):
 def __init__(self,features=56):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(features,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,16,batch_first=True,bidirectional=True);self.out=nn.Linear(32,2)
 def forward(self,x):
  b,t,n,f=x.shape;z=x.permute(0,2,1,3).reshape(b*n,t,f);z=self.conv(z.transpose(1,2)).transpose(1,2);z,_=self.gru(z);return self.out(z[:,t//2]).reshape(b,n,2)
