import torch,numpy as np
from torch import nn
def reverse_geometry(g):
 z=g.clone();q=g[:,:36].reshape(-1,9,4)+g[:,36:40,None].transpose(1,2);r=q.flip(1)
 z[:,:36]=(r-r[:,:1]).flatten(1);z[:,36:40]=r[:,0];z[:,40:44]=-g[:,40:44]
 return z
def endpoint_pause(g,fraction):
 q=g[:36].reshape(9,4)+g[36:40];t=np.linspace(-fraction,1+fraction,9).clip(0,1);r=np.stack([np.interp(t,np.linspace(0,1,9),q[:,j])for j in range(4)],1)
 z=g.copy();z[:36]=(r-r[0]).ravel();z[36:40]=r[0];z[40:44]=r[-1]-r[0];z[44:48]=np.abs(np.diff(r,axis=0)).mean(0);z[48:52]=np.ptp(r,axis=0);return z
class SymmetricMotion(nn.Module):
 def __init__(self,old):
  super().__init__();import copy
  self.conv=copy.deepcopy(old.motion_conv);self.gru=copy.deepcopy(old.motion_gru);self.head=copy.deepcopy(old.motion_head)
  self.register_buffer('mean',old.gm.clone());self.register_buffer('std',old.gs.clone())
  self.requires_grad_(True)
 def raw(self,g):
  z=(g-self.mean)/self.std;seq=z[:,:36].reshape(-1,9,4);seq=self.conv(seq.transpose(1,2)).transpose(1,2);_,h=self.gru(seq);q=torch.cat([h[0],h[1]],1)
  # Motion inputs are body-relative trajectories only. No handshape/word/target/time flag.
  return self.head(torch.cat([q,z,torch.zeros(len(g),192,device=g.device)],1))
 def forward(self,g):
  a=self.raw(g);b=self.raw(reverse_geometry(g));return .5*(a+b[:,[2,1,0]])
