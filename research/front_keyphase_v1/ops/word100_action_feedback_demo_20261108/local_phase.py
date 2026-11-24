"""A local coordinate check, without a whole-clip latent shortcut."""
import torch
from torch import nn
class LocalPhase(nn.Module):
 def __init__(self,guide_words):
  super().__init__();self.register_buffer('guide_words',torch.tensor([-1]+guide_words))
  self.net=nn.Sequential(nn.Conv1d(280,64,5,padding=2),nn.GELU(),nn.Conv1d(64,64,5,padding=2),nn.GELU(),nn.Conv1d(64,14,1))
 def forward(self,x,word,valid):
  h=x.transpose(1,2)
  for layer in self.net:h=layer(h)*valid[:,None,:]
  h=h.transpose(1,2);allowed=(self.guide_words[None,:]==word[:,None])|(self.guide_words[None,:]<0)
  return h.masked_fill(~allowed[:,None,:],-1e4)
