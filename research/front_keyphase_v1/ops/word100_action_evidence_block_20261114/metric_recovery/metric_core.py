import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from sequence_core import SequenceHead
import torch
from torch import nn
class MetricHead(SequenceHead):
    def __init__(self,dim=59):
        super().__init__(dim)
        self.metric=nn.Linear(32,12)
    def contrast(self,x,y):
        z=torch.nn.functional.normalize(self.metric(self.embed(x)),dim=-1)
        sim=z@z.T/.15
        eye=torch.eye(len(y),device=y.device,dtype=torch.bool)
        pos=y[:,None].eq(y[None,:])&~eye
        logits=sim.masked_fill(eye,-1e9)
        logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
        good=pos.sum(1)>0
        return -(logp*pos).sum(1)[good]/pos.sum(1)[good]
