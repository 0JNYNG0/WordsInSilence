"""Pose-only observation network; target affects stage selection, never attributes."""
import torch
from torch import nn
import numpy as np
class PoseGuideNet(nn.Module):
 def __init__(self,vectors,words,lengths):
  super().__init__()
  self.register_buffer('guide_vectors',torch.as_tensor(vectors,dtype=torch.float32))
  self.register_buffer('guide_words',torch.as_tensor(words,dtype=torch.long))
  self.register_buffer('prototype_lengths',torch.as_tensor(lengths,dtype=torch.long))
  self.input=nn.Linear(280,96)
  self.temporal=nn.Sequential(nn.Conv1d(96,64,3,padding=1),nn.GELU(),nn.Conv1d(64,64,3,padding=2,dilation=2),nn.GELU())
  self.context=nn.GRU(64,32,batch_first=True,bidirectional=True)
  self.attributes=nn.Linear(64,7);self.stage_query=nn.Linear(64,32)
  self.guide_encoder=nn.Sequential(nn.Linear(np.asarray(vectors).shape[1],32),nn.GELU(),nn.Linear(32,32))
  self.stage_embedding=nn.Parameter(torch.randn(len(words),32)*.02)
  self.prototype=nn.Parameter(torch.randn(len(lengths),max(lengths),64)*.02)
 def forward(self,pose,lengths,word):
  valid=torch.arange(pose.shape[1],device=pose.device)[None]<lengths[:,None]
  h=nn.functional.gelu(self.input(pose))*valid[:,:,None];h=h.transpose(1,2)
  for layer in self.temporal:h=layer(h)*valid[:,None,:]
  pack=nn.utils.rnn.pack_padded_sequence(h.transpose(1,2),lengths.cpu(),batch_first=True,enforce_sorted=False)
  h,_=self.context(pack);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=pose.shape[1])
  attr=self.attributes(h);keys=self.guide_encoder(self.guide_vectors)+self.stage_embedding
  stage=(self.stage_query(h)@keys.T)/(32**.5)
  stage=stage.masked_fill(self.guide_words[None,None,:]!=word[:,None,None],-1e4)
  return dict(attributes=attr,stage=stage,latent=h,valid=valid)

def collate(rows):
 n=len(rows);T=max(len(r['x']) for r in rows);out={}
 for k,d in [('x',280),('y',7),('ym',7)]:
  a=np.zeros((n,T,d),np.float32)
  for i,r in enumerate(rows):a[i,:len(r[k])]=r[k]
  out[k]=torch.from_numpy(a)
 for k,fill,dt in [('sy',-1,np.int64),('sw',0,np.float32),('ci',-1,np.int64)]:
  a=np.full((n,T),fill,dt)
  for i,r in enumerate(rows):a[i,:len(r[k])]=r[k]
  out[k]=torch.from_numpy(a)
 out['lengths']=torch.tensor([len(r['x']) for r in rows]);out['word']=torch.tensor([r['word'] for r in rows])
 return out
