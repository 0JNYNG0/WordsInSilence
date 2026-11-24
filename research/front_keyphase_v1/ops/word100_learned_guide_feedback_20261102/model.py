"""Trainable public-guide feedback network. Goal-blind observation branch.

Pretrained SignRep fc.1 receives a genuine rank-8 weight update. Its frozen
backbone/normalization outputs are cached; this is not full-backbone tuning.
"""
import torch
from torch import nn
import numpy as np

class GuideFeedbackNet(nn.Module):
 def __init__(self, projection, guide_vectors, guide_words, prototype_lengths, finetune=False):
  super().__init__();self.finetune=finetune;self.rank=8
  self.register_buffer('pretrained_projection',torch.as_tensor(projection).float())
  self.register_buffer('guide_vectors',torch.as_tensor(guide_vectors).float())
  self.register_buffer('guide_words',torch.as_tensor(guide_words).long())
  self.register_buffer('prototype_lengths',torch.as_tensor(prototype_lengths).long())
  self.update_a=nn.Parameter(torch.randn(self.rank,768)*.02,requires_grad=finetune)
  self.update_b=nn.Parameter(torch.zeros(768,self.rank),requires_grad=finetune)
  self.rgb=nn.Linear(768,64);self.pose=nn.Linear(138,32)
  self.temporal=nn.Sequential(nn.Conv1d(96,64,3,padding=1),nn.GELU(),nn.Conv1d(64,64,3,padding=2,dilation=2),nn.GELU())
  self.context=nn.GRU(64,32,batch_first=True,bidirectional=True)
  self.attributes=nn.Linear(64,7)
  self.pose_reconstruction=nn.Linear(64,69)
  self.stage_query=nn.Linear(64,32)
  self.guide_encoder=nn.Sequential(nn.Linear(guide_vectors.shape[1],32),nn.GELU(),nn.Linear(32,32))
  self.stage_embedding=nn.Parameter(torch.randn(len(guide_words),32)*.02)
  self.prototype=nn.Parameter(torch.randn(len(prototype_lengths),max(prototype_lengths),64)*.02)
  self.register_buffer('rgb_mean',torch.zeros(768));self.register_buffer('rgb_std',torch.ones(768))
 def tuned_weight(self):return self.pretrained_projection+self.update_b@self.update_a/self.rank
 def forward(self,pose,rgb,prelinear,lengths,word):
  # Learned correction is exactly x @ (W0+B@A/rank).T - x@W0.T.
  z=rgb
  if self.finetune:z=z+(prelinear@self.update_a.T)@self.update_b.T/self.rank
  z=(z-self.rgb_mean)/self.rgb_std
  t=torch.arange(pose.shape[1],device=pose.device)[None];valid=t<lengths[:,None]
  h=torch.cat([nn.functional.gelu(self.rgb(z)),nn.functional.gelu(self.pose(pose))],-1)*valid[:,:,None]
  h=h.transpose(1,2)
  for layer in self.temporal:h=layer(h)*valid[:,None,:]
  h=h.transpose(1,2)
  packed=nn.utils.rnn.pack_padded_sequence(h,lengths.cpu(),batch_first=True,enforce_sorted=False)
  h,_=self.context(packed);h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=pose.shape[1])
  attr=self.attributes(h)
  keys=self.guide_encoder(self.guide_vectors)+self.stage_embedding
  scores=self.stage_query(h)@keys.T/(32**.5)
  scores=scores.masked_fill(self.guide_words[None,None,:]!=word[:,None,None],-1e4)
  return dict(attributes=attr,latent=h,stage=scores,pose=self.pose_reconstruction(h),valid=valid)

def collate(items):
 n=len(items);T=max(len(r['pose']) for r in items)
 out={}
 for k,d in [('pose',138),('rgb',768),('prelinear',768),('y',7),('ym',7),('pose_y',69),('pose_m',69)]:
  a=np.zeros((n,T,d),np.float32)
  for i,r in enumerate(items):a[i,:len(r[k])]=r[k]
  out[k]=torch.from_numpy(a)
 for k,fill,dtype in [('stage_y',-1,np.int64),('stage_w',0,np.float32),('canonical_index',-1,np.int64)]:
  a=np.full((n,T),fill,dtype=dtype)
  for i,r in enumerate(items):a[i,:len(r[k])]=r[k]
  out[k]=torch.from_numpy(a)
 out['lengths']=torch.tensor([len(r['pose']) for r in items]);out['word']=torch.tensor([r['word'] for r in items])
 return out
