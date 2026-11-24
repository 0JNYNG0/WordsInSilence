"""Candidate native-time movement readout. Untrained until a frozen data gate.

One sequence forward emits physical query scores at every original time index.
No word ID, reference-video lookup, start/end-state summarization, or searched
subwindow enters this network. Per-time scores are estimates, not dense truth.
"""
import torch
from torch import nn

INPUT_DIM=915
TARGETS=5
RECEPTIVE_FIELD=15

class LocalBlock(nn.Module):
    def __init__(self,dilation):
        super().__init__()
        self.conv=nn.Conv1d(32,32,3,padding=dilation,dilation=dilation)
        # Normalize channels at a single time only. Global temporal statistics
        # would reintroduce dependence on unrelated preparation/ending frames.
        self.norm=nn.LayerNorm(32)

    def forward(self,h,time_valid):
        z=self.conv(h.transpose(1,2)).transpose(1,2)
        return (h+torch.nn.functional.gelu(self.norm(z)))*time_valid[:,:,None]

class LocalEventNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry=nn.Linear(146,32)
        self.rgb=nn.Linear(768,32,bias=False)
        self.input_norm=nn.LayerNorm(32)
        self.blocks=nn.ModuleList([LocalBlock(d) for d in [1,2,4]])
        self.output=nn.Linear(32,TARGETS)
        self.register_buffer('geometry_valid',torch.ones(146))

    def initialize_from_frozen_motion(self,state):
        """Transfer existing projections, not its last-state GRU or classifier.

        The caller must verify checkpoint provenance before passing its state.
        This function does not load files or enable any training by itself.
        """
        self.geometry.load_state_dict({k:state['direction.embed.0.'+k] for k in ['weight','bias']})
        self.rgb.load_state_dict({'weight':state['direction.rgb.weight']})
        with torch.no_grad():self.geometry_valid.copy_(state['valid'])

    def forward(self,x,lengths):
        if x.ndim!=3 or x.shape[-1]!=INPUT_DIM or lengths.shape!=(x.shape[0],):
            raise ValueError('Sequence/length shape mismatch')
        if lengths.dtype not in [torch.int32,torch.int64] or (lengths<1).any() or (lengths>x.shape[1]).any():
            raise ValueError('Invalid native lengths')
        if not torch.isfinite(x).all():raise ValueError('Input must carry finite values and explicit masks')
        time_valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        rgb_available=(x[:,:,914]>.5)&time_valid
        g=x[:,:,:146]*self.geometry_valid
        v=torch.where(rgb_available[:,:,None],x[:,:,146:914],0)
        h=torch.nn.functional.gelu(self.input_norm(self.geometry(g)+self.rgb(v)))
        h=h*time_valid[:,:,None]
        for block in self.blocks:h=block(h,time_valid)
        logits=self.output(h)
        # Preserve the time axis. Padded rows have no event prediction.
        logits=torch.where(time_valid[:,:,None],logits,torch.zeros_like(logits))
        return dict(logits=logits,time_valid=time_valid,rgb_available=rgb_available,
                    receptive_field=RECEPTIVE_FIELD)

def bag_logit(output,start,end,target):
    """Weak interval target: an event exists somewhere in reviewed support.

    This maximum is a fixed readout of one native-time forward, not a search over
    independently scored input crops. Positive interval labels are NOT copied
    to every time point. No unavailable bag is silently labelled normal.
    """
    if type(start) is not int or type(end) is not int or type(target) is not int:
        raise ValueError('Exact integer query coordinates required')
    logits=output['logits'];valid=output['time_valid']
    if logits.shape[0]!=1 or not (0<=start<=end<logits.shape[1]) or target not in range(TARGETS):
        raise ValueError('Bag outside single original sequence')
    if not valid[0,start:end+1].all():raise ValueError('Padding cannot supply event evidence')
    if not output['rgb_available'][0,start:end+1].any():
        raise ValueError('No image observation in bag; retain insufficient state')
    return logits[0,start:end+1,target].amax()
