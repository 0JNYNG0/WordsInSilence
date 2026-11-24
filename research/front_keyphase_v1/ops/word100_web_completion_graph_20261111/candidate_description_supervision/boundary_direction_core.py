"""Copied direction head only; common100 encoder and all deployed weights stay frozen."""
from local_rank_core import *
from transfer_local_timefix_core import encoder,normalized_pose
import copy
M=W/'boundary_direction_learning'
class DirectionHead(nn.Module):
 def __init__(self,observer):
  super().__init__();self.register_buffer('gm',observer.gm.detach().clone());self.register_buffer('gs',observer.gs.detach().clone())
  self.motion_conv=copy.deepcopy(observer.motion_conv);self.motion_gru=copy.deepcopy(observer.motion_gru);self.motion_head=copy.deepcopy(observer.motion_head)
  self.requires_grad_(True)
 def forward(self,context,g):
  z=(g-self.gm)/self.gs;seq=z[:,:36].reshape(-1,9,4);seq=self.motion_conv(seq.transpose(1,2)).transpose(1,2);_,h=self.motion_gru(seq)
  return self.motion_head(torch.cat([h[0],h[1],z,context],-1))

def hidden(base,pose):
 with torch.inference_mode():
  h=torch.nn.functional.gelu(base.input(torch.tensor(pose)[None]));h=base.temporal(h.transpose(1,2)).transpose(1,2);h,_=base.context(h)
  return h[0].numpy()
