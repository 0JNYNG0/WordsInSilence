"""Label-agnostic auxiliary supervision; no word/person-specific branches."""
import torch
from torch.nn import functional as F
LAMBDA_AUX=0.1
def objective(base,aux,target,weight=LAMBDA_AUX):
 if base.ndim!=2 or base.shape!=aux.shape or target.shape!=(base.shape[0],):
  raise ValueError('Expected equal [batch,classes] logits and [batch] labels')
 if not torch.isfinite(base).all() or not torch.isfinite(aux).all():raise ValueError('Nonfinite logits')
 final=F.cross_entropy(base.detach()+aux,target,reduction='mean')
 extra=F.cross_entropy(aux,target,reduction='mean')
 total=final+weight*extra
 if not torch.isfinite(total):raise ValueError('Nonfinite loss')
 return total,final,extra
