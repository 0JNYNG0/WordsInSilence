import torch
from torch import nn
class EvidenceHeads(nn.Module):
 def __init__(self,guide_words):
  super().__init__();self.register_buffer('guide_words',torch.tensor([-1]+guide_words))
  self.temporal=nn.Sequential(nn.Conv1d(71,48,3,padding=1),nn.GELU(),nn.Conv1d(48,48,3,padding=1),nn.GELU())
  self.phase=nn.Linear(48,len(guide_words)+1)
  self.difference=nn.Sequential(nn.Linear(78,48),nn.GELU(),nn.Linear(48,7))
 def forward(self,h,p,word,valid,expected=None):
  z=torch.cat([h,p],-1)*valid[:,:,None];t=z.transpose(1,2)
  for layer in self.temporal:t=layer(t)*valid[:,None,:]
  phase=self.phase(t.transpose(1,2));allowed=(self.guide_words[None,:]==word[:,None])|(self.guide_words[None,:]<0)
  phase=phase.masked_fill(~allowed[:,None,:],-1e4)
  diff=None if expected is None else self.difference(torch.cat([z,expected],-1))
  return phase,diff
