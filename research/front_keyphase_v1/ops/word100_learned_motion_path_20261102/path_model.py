"""Within-person displacement and independent continuity supervision."""
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
Q=Path(__file__).resolve().parent;O=Q.parent;N=O/'word100_learned_motion_invariance_20261102';M=O/'word100_learned_motion_order_20261102'
sys.path.insert(0,str(N));sys.path.insert(0,str(M))
from motion_model import MotionNet,batch,predict,read
from real_motion import records_and_sequences

POSITIONS=list(range(10))+list(range(28,38))+[56,57,58,61,62]

def relative_features(x,lengths):
 v=x[:,:,POSITIONS];mask=x[:,:,np.array(POSITIONS)+73]>.5
 time_mask=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
 mask=mask&time_mask[:,:,None]
 first=mask.to(torch.int64).argmax(dim=1)
 anchor=v.gather(1,first[:,None,:]).squeeze(1)
 has=mask.any(dim=1)
 displacement=torch.where(mask&has[:,None,:],v-anchor[:,None,:],0)
 adjacent=torch.zeros_like(mask);adjacent[:,1:]=mask[:,1:]&mask[:,:-1]
 step=torch.zeros_like(v);step[:,1:]=v[:,1:]-v[:,:-1];step=torch.where(adjacent,step,0)
 return torch.cat([displacement,step,(mask&has[:,None,:]).to(v.dtype),adjacent.to(v.dtype)],dim=-1)

class RelativeBranch(MotionNet):
 def __init__(self):
  super().__init__('ordered_gru');self.relative=nn.Linear(len(POSITIONS)*4,32,bias=False);nn.init.zeros_(self.relative.weight)
 def forward(self,x,lengths):
  h=self.embed[1](self.embed[0](x)+self.relative(relative_features(x,lengths)))
  _,last=self.temporal(pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False))
  return self.out(torch.cat([last[0],last[1]],dim=-1))

class PathNet(nn.Module):
 def __init__(self,condition):
  super().__init__();assert condition in ['coupled_relative','separated_relative'];self.condition=condition
  self.direction=RelativeBranch()
  if condition=='separated_relative':self.continuity=RelativeBranch()
  keep=read(N/'TRAINING_CONTRACT.json')['motion_value_indices'];v=torch.zeros(146);v[keep]=1;v[np.array(keep)+73]=1;self.register_buffer('valid',v)
 def initialize(self,seed):
  s=torch.load(N/'runs'/f'real_variants_motion_inputs_{seed}/model.pt',map_location='cpu',weights_only=True)
  a=self.direction.load_state_dict(s,strict=False);assert a.missing_keys==['relative.weight'] and not a.unexpected_keys
  if self.condition=='separated_relative':
   a=self.continuity.load_state_dict(s,strict=False);assert a.missing_keys==['relative.weight'] and not a.unexpected_keys
 def forward(self,x,lengths):
  x=x*self.valid;d=self.direction(x,lengths)
  if self.condition=='coupled_relative':return d
  c=self.continuity(x,lengths);return torch.cat([d[:,:5],c[:,5:6]],dim=-1)

