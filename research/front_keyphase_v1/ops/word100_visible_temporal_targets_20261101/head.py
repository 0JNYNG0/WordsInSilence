"""Small observation head. No text, gloss, word identity, or clip progress input."""
import torch
from torch import nn

class Head(nn.Module):
    def __init__(self, targets):
        super().__init__()
        self.pose=nn.Linear(138,32)
        self.aux=nn.Linear(768,32)
        self.c1=nn.Conv1d(64,48,3,padding=1)
        self.c2=nn.Conv1d(48,32,3,padding=2,dilation=2)
        self.out=nn.Linear(128,targets)
    def forward(self,x,z):
        assert x.shape[1]==z.shape[1]==13
        h=torch.cat([nn.functional.gelu(self.pose(x)),nn.functional.gelu(self.aux(z))],-1)
        h=nn.functional.gelu(self.c1(h.transpose(1,2)))
        h=nn.functional.gelu(self.c2(h)).transpose(1,2)
        # Ordered parts remain separate; a centre-state label does not supervise each frame.
        summary=torch.cat([h[:,:4].mean(1),h[:,4:9].mean(1),h[:,9:].mean(1),h[:,6]],-1)
        return self.out(summary)

def masked_loss(logits,y,mask,class_weights,event_weights):
    safe=torch.where(mask,y,torch.zeros_like(y))
    raw=nn.functional.binary_cross_entropy_with_logits(logits,safe,reduction='none')
    w=torch.where(safe>.5,class_weights[1],class_weights[0])*event_weights
    return torch.where(mask,raw*w,torch.zeros_like(raw)).sum()/event_weights[mask].sum().clamp_min(1)
