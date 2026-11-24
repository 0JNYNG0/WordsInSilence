from quality_model import features,windows
from torch import nn
import torch
class Branch(nn.Module):
 def __init__(self,n):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(n,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,16,batch_first=True,bidirectional=True);self.out=nn.Linear(32,1)
 def forward(self,x):
  z=self.conv(x.transpose(1,2)).transpose(1,2);z,_=self.gru(z);return self.out(z[:,x.shape[1]//2])
class QualityChange(nn.Module):
 def __init__(self,features=56):
  super().__init__();self.quality=Branch(features);self.change=Branch(28)
 def forward(self,x):
  b,t,n,f=x.shape;z=x.permute(0,2,1,3).reshape(b*n,t,f)
  # Physical shape-change branch excludes local absolute XY and tip proximity.
  # 2D projection still changes these geometry ratios; stability controls needed.
  p=torch.cat([z[:,:,:14],z[:,:,28:42]],-1)
  return torch.cat([self.quality(z),self.change(p)],-1).reshape(b,n,2)
