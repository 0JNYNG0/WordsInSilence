"""Matched queried-event heads; uniquely named to avoid old model.py imports."""
from pathlib import Path
import sys,copy
Y=Path(__file__).resolve().parent; O=Y.parent
OLD=O/'word100_guided_event_order_20261102'
sys.path.insert(0,str(OLD))
from order_model import *
OUT=Path('W:/WordsInSilence_derived/word100_direct_occurrence_order_20261102')

class ReadoutNet(nn.Module):
 def __init__(self,condition):
  super().__init__();assert condition in ['tied_readout','direct_readout']
  self.condition=condition;self.base=OccurrenceOrderNet()
  for p in self.base.parameters():p.requires_grad_(False)
  self.visual_embed=copy.deepcopy(self.base.role_embed)
  self.visual_context=copy.deepcopy(self.base.role_context)
  self.query=copy.deepcopy(self.base.occurrence_query)
  for layer in [self.visual_embed,self.visual_context,self.query]:
   for p in layer.parameters():p.requires_grad_(True)
  self.base.eval()

 def train(self,mode=True):
  super().train(mode);self.base.eval();return self

 def forward(self,x,rgb,lengths,flow):
  with torch.no_grad():
   feature,h=self.base.encode(x,rgb,lengths)
   logits=self.base.actor.query(feature)@(self.base.actor.guide(self.base.actor.guide_vectors)+self.base.actor.embedding).T/(32**.5)
   families=torch.stack([torch.logsumexp(logits[:,:,ids],-1)-np.log(len(ids)) for ids in episode.ALIASES],-1)
   action=torch.cat([families,self.base.actor.rest(feature)],-1)
   visual=torch.cat([feature,h,x[:,:,64:74]],-1)
   fixed=self.base.role_embed(visual+self.base.flow_bridge(flow))
   fixed,_=self.base.role_context(nn.utils.rnn.pack_padded_sequence(fixed,lengths.cpu(),batch_first=True,enforce_sorted=False))
   fixed,_=nn.utils.rnn.pad_packed_sequence(fixed,batch_first=True,total_length=x.shape[1])
   role=self.base.role_output(fixed)
  z=self.visual_embed(visual)
  z,_=self.visual_context(nn.utils.rnn.pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False))
  z,_=nn.utils.rnn.pad_packed_sequence(z,batch_first=True,total_length=x.shape[1])
  occurrence=self.query(z)@self.base.family_guide_keys.T/(32**.5)
  if self.condition=='tied_readout':occurrence=occurrence+(role[:,:,1]-role[:,:,0])[:,:,None]
  return action,role,occurrence

def order_probability(physical,occurrence,available,first,second,condition):
 assert condition in ['tied_readout','direct_readout']
 multiplier=physical if condition=='tied_readout' else torch.ones_like(physical)
 return directed_probability(multiplier,occurrence,available,first,second)

def pair_loss(physical,occurrence,rr,condition):
 losses=[];used=0
 for i,r in enumerate(rr):
  n=len(r['x']);av=physical.new_tensor(r['available']);terms=[]
  for q in r['order_queries']:
   if q['target']<0:continue
   p,_=order_probability(physical[i,:n],occurrence[i,:n],av,q['asked_before'],q['asked_after'],condition)
   terms.append(nn.functional.binary_cross_entropy(p.clamp(1e-6,1-1e-6),p.new_tensor(float(q['target']))));used+=1
  losses.append(torch.stack(terms).mean() if terms else occurrence[i].sum()*0)
 return torch.stack(losses).mean(),used
