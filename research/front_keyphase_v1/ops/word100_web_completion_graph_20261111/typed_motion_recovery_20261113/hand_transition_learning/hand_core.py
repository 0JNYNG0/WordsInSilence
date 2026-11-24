"""Shared hand-local observer. No word, body location, expected order or clip time input.

This research branch supplements the frozen common100 observer. It never changes
the deployed shape head, extent ranker, C/T3-F or the common100 encoder.
"""
from pathlib import Path
import sys
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent))
from common import *
from torch import nn
T=HERE/'typed_shape_transition_review'
ATTRIBUTES=['four_open','four_closed','index_only','intermediate_flexing']

def hand_input(x,side):
    assert side in ('L','R')
    a,b,c,d=(102,144,237,258) if side=='L' else (144,186,258,279)
    coords=x[:,a:b].copy();mask=x[:,c:d].copy()
    coords=np.where(np.repeat(mask>.5,2,axis=1),coords,0.)
    return np.c_[coords,mask].astype(np.float32)

class HandObserver(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32))
        self.spatial=nn.Sequential(nn.Linear(63,48),nn.GELU())
        self.temporal=nn.Sequential(nn.Conv1d(48,32,3,padding=1),nn.GELU(),nn.Conv1d(32,32,3,padding=2,dilation=2),nn.GELU())
        self.context=nn.GRU(32,16,batch_first=True,bidirectional=True)
        self.attributes=nn.Linear(32,4)
        self.transition=nn.Sequential(nn.Linear(128,48),nn.GELU(),nn.Linear(48,3))
    def forward(self,x,lengths):
        valid=torch.arange(x.shape[1],device=x.device)[None]<lengths[:,None]
        h=self.spatial((x-self.mean)/self.std)*valid[:,:,None]
        h=h.transpose(1,2)
        for layer in self.temporal:h=layer(h)*valid[:,None]
        pack=nn.utils.rnn.pack_padded_sequence(h.transpose(1,2),lengths.cpu(),batch_first=True,enforce_sorted=False)
        h,_=self.context(pack)
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.shape[1])
        first=h[:,0];last=h[torch.arange(len(h)),lengths-1]
        avg=(h*valid[:,:,None]).sum(1)/lengths[:,None]
        return dict(attributes=self.attributes(h),transition=self.transition(torch.cat([first,avg,last,last-first],-1)),latent=h,valid=valid)

def padded(sequences):
    lens=np.asarray([len(x) for x in sequences],np.int64)
    out=np.zeros((len(lens),int(lens.max()),63),np.float32)
    for i,x in enumerate(sequences):out[i,:len(x)]=x
    return torch.tensor(out),torch.tensor(lens)

def preserve_failure(stage):
    import traceback,time
    U.mkdir(exist_ok=True,parents=True)
    p=U/f'{stage}_FAILURE_{time.time_ns()}.txt'
    p.write_text(traceback.format_exc(),'utf-8')
