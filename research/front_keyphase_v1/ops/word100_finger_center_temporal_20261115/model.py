from common import *
sys.path.insert(0,str(P))
from finger_model import SharedFinger,features,windows
import torch
from torch import nn
class ChangeHead(nn.Module):
 """Whether source-visible finger posture changes around the central instant.

 Not a word recognizer, quality repair model, or center-state classifier.
 """
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv1d(28,32,3,padding=1),nn.GELU());self.gru=nn.GRU(32,16,batch_first=True,bidirectional=True);self.head=nn.Linear(32,1)
 def forward(self,x):
  b,t,n,f=x.shape;q=x.permute(0,2,1,3).reshape(b*n,t,f);q=torch.cat([q,torch.cat([torch.zeros_like(q[:,:1]),q[:,1:]-q[:,:-1]],1)],-1)
  h,_=self.gru(self.conv(q.transpose(1,2)).transpose(1,2));return self.head(h[:,t//2]).reshape(b,n)

def consensus(ps,a,threshold=.9):
 q=np.where(ps>=threshold,1,np.where(ps<=1-threshold,0,-1));v=np.where((q==q[0]).all(0),q[0],-1);v[~a]=-1;return v
