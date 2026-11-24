from common_split import *
import torch
from torch import nn
sys.path[:0]=[str(P),str(D),str(Q)]
from finger_model import SharedFinger,features as shape_features,windows
from quality_model import features as quality_features
class TrackQuality(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(56,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,16,bidirectional=True,batch_first=True);self.head=nn.Linear(32,1)
 def forward(self,x):
  b,t,n,f=x.shape;z=x.permute(0,2,1,3).reshape(b*n,t,f);z=self.conv(z.transpose(1,2)).transpose(1,2);z,_=self.gru(z);return self.head(z[:,t//2]).reshape(b,n)
def load_channels(path):
 z=np.load(path);key=next(k for k in ['tcn280','feedback','full']if k in z.files);return z[key],z['frame_id']
def consensus(p,available):
 z=np.where(p>=.85,1,np.where(p<=.15,0,-1));z[:,~available]=-1
 return np.where((z==z[0]).all(0),z[0],-1)
