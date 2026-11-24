"""Fixed pretrained C and trainable T3 auxiliary head. Never updates C statistics."""
import torch
import torch.nn.functional as F
from temporal import TemporalPhaseTCN,phase_weights,summarize

class FrozenCTemporal(TemporalPhaseTCN):
 model_type='wis_t3f_frozen_c_temporal'
 model_version='frozen_c_soft_elapsed3_v1'

 def __init__(self,**kwargs):
  super().__init__(**kwargs)
  for name,p in self.named_parameters():p.requires_grad_(name=='phase_weight')
  self.train(False)

 def train(self,mode=True):
  super().train(mode)
  self.proj.eval();self.blocks.eval();self.head.eval()
  return self

 def frozen_state(self):return {k:v for k,v in self.state_dict().items() if k!='phase_weight'}

 def assert_frozen(self):
  assert all(not m.training for child in (self.proj,self.blocks,self.head) for m in child.modules())
  assert all(p.requires_grad==(n=='phase_weight') for n,p in self.named_parameters())

 def base_and_contrast(self,x,pad_mask):
  self.assert_frozen()
  with torch.no_grad():
   mask=pad_mask.unsqueeze(1);h=self.proj(x.transpose(1,2)*mask)*mask
   for block in self.blocks:h=block(h,mask)
   g,d=summarize(h,pad_mask,phase_weights(x,pad_mask))
   base=self.head(g)
  return base,d

 def forward(self,x,pad_mask):
  base,d=self.base_and_contrast(x,pad_mask)
  return base+F.linear(d,self.phase_weight)
