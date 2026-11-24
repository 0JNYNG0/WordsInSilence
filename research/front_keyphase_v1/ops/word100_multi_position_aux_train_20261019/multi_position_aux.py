"""Isolated multi-position GRU readout; no production integration or training on import."""
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
P=Path(__file__).resolve().parent.parent/'word100_frozen_order_aux_preflight_20261019'
sys.path.insert(0,str(P))
from frozen_order_aux import FrozenOrderAux
from frozen_temporal import FrozenCTemporal
from temporal import phase_weights,summarize

def phase_readout(hidden,pad_mask,weights):
    """Read every real GRU output, including frames with missing hands.

    A zero-mass temporal bin falls back to the whole real-frame mean. Only
    trailing batch padding is excluded. Bins are elapsed time, not semantic stages.
    """
    mask=pad_mask.to(hidden.dtype)
    weights=weights.to(hidden.dtype)*mask.unsqueeze(-1)
    mass=weights.sum(1)
    mean=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(-1)
    parts=torch.einsum('btc,bth->bch',weights,hidden*mask.unsqueeze(-1))
    parts=parts/mass.clamp_min(torch.finfo(hidden.dtype).tiny).unsqueeze(-1)
    parts=torch.where((mass>0).unsqueeze(-1),parts,mean.unsqueeze(1))
    return parts.flatten(1)

class MultiPositionAux(FrozenOrderAux):
    model_type='wis_frozen_t3f_multi_position_aux'
    model_version='gru16_elapsed3_zero_residual_v1'

    def __init__(self,model_config,base_state):
        nn.Module.__init__(self)
        if model_config['width']!=32 or model_config['n_classes']!=100:
            raise ValueError('This fixed experiment uses width32/classes100')
        # Same construction order before the new output head preserves GRU RNG.
        self.base=FrozenCTemporal(**model_config)
        self.base.load_state_dict(base_state,strict=True)
        self.base.requires_grad_(False)
        self.memory=nn.GRU(32,16,num_layers=1,batch_first=True)
        self.residual=nn.Linear(48,100,bias=False)
        nn.init.zeros_(self.residual.weight)
        self.train(False)

    def memory_sequence(self,features,lengths):
        packed=pack_padded_sequence(features,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_out,_=self.memory(packed)
        out,returned=pad_packed_sequence(packed_out,batch_first=True,total_length=features.shape[1])
        if not torch.equal(returned.cpu(),lengths.cpu()):
            raise RuntimeError('Packed sequence lengths/order changed')
        return out

    def components(self,x,pad_mask):
        lengths=self.validate(x,pad_mask);self.assert_frozen()
        pad_mask=pad_mask.to(x.dtype)
        with torch.no_grad():
            mask=pad_mask.unsqueeze(1)
            h=self.base.proj(x.transpose(1,2)*mask)*mask
            for block in self.base.blocks:h=block(h,mask)
            weights=phase_weights(x,pad_mask)
            g,contrast=summarize(h,pad_mask,weights)
            b=self.base.head(g)+torch.nn.functional.linear(contrast,self.base.phase_weight)
        sequence=self.memory_sequence(h.transpose(1,2),lengths)
        readout=phase_readout(sequence,pad_mask,weights)
        a=self.residual(readout)
        return b,a,readout,h.transpose(1,2),lengths
