"""Isolated T3: original C backbone/global head plus zero-initialized phase contrasts."""
import torch
import torch.nn.functional as F
from pilot.model_masked_v1 import MaskedSignTCN

def phase_weights(x, pad_mask):
    """[B,T,3], normalized elapsed dt; real missing frames retain time."""
    valid=pad_mask.bool()
    dt=torch.where(valid,x[:,:,279],torch.zeros_like(x[:,:,279]))
    elapsed=torch.cumsum(dt,dim=1)
    first=elapsed.masked_fill(~valid,float('inf')).amin(dim=1,keepdim=True)
    last=elapsed.masked_fill(~valid,float('-inf')).amax(dim=1,keepdim=True)
    any_real=valid.any(dim=1,keepdim=True)
    first=torch.where(any_real,first,torch.zeros_like(first))
    last=torch.where(any_real,last,torch.zeros_like(last))
    span=last-first
    u=torch.where(span>0,(elapsed-first)/span.clamp_min(torch.finfo(x.dtype).tiny),torch.full_like(elapsed,.5))
    u=u.clamp(0,1)
    return torch.stack([(1-2*u).clamp_min(0),1-(2*u-1).abs(),(2*u-1).clamp_min(0)],dim=2)*pad_mask.unsqueeze(2)

def summarize(h, pad_mask, weights):
    """Keep original C mean arithmetic. Zero-mass phase falls back to mean."""
    m=pad_mask.unsqueeze(1)
    g=(h*m).sum(dim=2)/pad_mask.sum(dim=1).clamp(min=1.).unsqueeze(1)
    mass=weights.sum(dim=1)
    parts=torch.einsum('btc,bkt->bck',weights,h*m)/mass.clamp_min(torch.finfo(h.dtype).tiny).unsqueeze(2)
    parts=torch.where((mass>0).unsqueeze(2),parts,g.unsqueeze(1))
    return g,(parts-g.unsqueeze(1)).flatten(1)

class TemporalPhaseTCN(MaskedSignTCN):
    model_type='wis_t3_masked_tcn'
    model_version='soft_elapsed3_v1'

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # torch.zeros consumes no RNG; original shared initializer order is exact.
        self.phase_weight=torch.nn.Parameter(torch.zeros(self.head.out_features,3*self.head.in_features))

    def forward(self,x,pad_mask):
        h=x.transpose(1,2);m=pad_mask.unsqueeze(1)
        h=self.proj(h*m)*m
        for blk in self.blocks:h=blk(h,m)
        g,contrast=summarize(h,pad_mask,phase_weights(x,pad_mask))
        return self.head(g)+F.linear(contrast,self.phase_weight)
