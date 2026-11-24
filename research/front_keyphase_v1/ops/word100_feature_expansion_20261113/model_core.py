"""Research-only temporal finger observation branch. No production imports/writes."""
from common import *
import torch
from torch import nn
from torch.nn import functional as F
old=load('_expansion_shared',B/'core.py')

def hand_features(xy,available):
    # Normalize shape in the palm plane; world orientation stays in the 280 inputs.
    xy=np.asarray(xy,np.float32);m=np.asarray(available,bool)
    axis=xy[:,9]-xy[:,0];length=np.linalg.norm(axis,axis=-1)
    valid=m[:,[0,5,9,13,17]].all(1)&(length>1)&np.isfinite(xy).all((1,2))
    ey=axis/np.maximum(length[:,None],1);ex=np.stack([ey[:,1],-ey[:,0]],1)
    ex*=np.where(((xy[:,5]-xy[:,17])*ex).sum(1)<0,-1.,1.)[:,None]
    scale=np.median(np.linalg.norm(xy[:,[5,9,13,17]]-xy[:,0,None],axis=-1),axis=1)
    delta=xy-xy[:,0,None]
    local=np.stack([(delta*ex[:,None]).sum(-1),(delta*ey[:,None]).sum(-1)],-1)/np.maximum(scale[:,None,None],1.)
    local=np.where((m&valid[:,None])[:,:,None],local,0)
    out=np.concatenate([np.clip(local.reshape(len(xy),42),-8,8),(m&valid[:,None]).astype(np.float32)],-1)
    return out.astype(np.float32),valid

class FineNet(old.SharedNet):
    def __init__(self,extra_count=7):
        super().__init__()
        self.finger_local=nn.Sequential(nn.Linear(63,32),nn.GELU())
        self.finger_context=nn.Linear(64,16)
        self.fingers=nn.Linear(48,15) # same weights for either hand
        self.extra_word=nn.Linear(128,extra_count)
    def forward(self,x,lengths,word,hand=None):
        o=super().forward(x,lengths,word);h=o['latent'];v=o['valid']
        avg=(h*v[:,:,None]).sum(1)/lengths[:,None];mx=h.masked_fill(~v[:,:,None],-1e4).max(1).values
        o['extra_logits']=self.extra_word(torch.cat([avg,mx],-1))
        if hand is not None:
            local=self.finger_local(hand);ctx=self.finger_context(h)[:,:,None].expand(-1,-1,2,-1)
            o['fingers']=self.fingers(torch.cat([local,ctx],-1)).reshape(*hand.shape[:3],5,3)
        return o

def rows_load(role=None):
    out=[]
    for m in read(U/'DATA_READY.json')['rows']:
        if role and m['role']!=role:continue
        assert sha(m['path'])==m['sha256'],m['clip_id']+' changed cached input'
        z=np.load(m['path'],allow_pickle=False)
        out.append(dict(m,**{k:z[k].copy() for k in z.files}))
    return out

def batch(rows,view='full',mask=False,rng=None):
    b=old.batch(rows,view,False)
    T=b['x'].shape[1];n=len(rows)
    hand=np.zeros((n,T,2,63),np.float32);fy=np.full((n,T,2,5),-1,np.int64)
    for i,r in enumerate(rows):
        N=len(r['x_full']);hand[i,:N]=r['hand'];fy[i,:N]=r['fy']
        if view=='upper':
            # Input masks are missingness evidence; never invent invisible truth.
            vis=r['x_upper'][:,186:237]>.5
            for side,s in enumerate((9,30)):
                visible=vis[:,s:s+21]
                hand[i,:N,side,:42]*=np.repeat(visible,2,1)
                hand[i,:N,side,42:]*=visible
                reference_visible=visible[:,[0,5,9,13,17]].all(1)
                hand[i,:N,side]*=reference_visible[:,None]
        if mask and N>8:
            start=int(rng.integers(2,N-3));side=int(rng.integers(0,2))
            b['x'][i,:N]=torch.from_numpy(old.mask_hand(b['x'][i,:N].numpy(),start,start+2,side))
            hand[i,start:start+2,side]=0
            # No newly inferred labels are created; evaluated masking is separate.
    b['hand']=torch.from_numpy(hand);b['fy']=torch.from_numpy(fy)
    b['extra']=torch.tensor([r.get('extra_index',-1) for r in rows]);return b

def fine_loss(o,b,weights):
    z=o['fingers'].reshape(-1,3);y=b['fy'].reshape(-1)
    if not (y>=0).any():return z.sum()*0
    return F.cross_entropy(z,y,weight=weights,ignore_index=-1)

def retention(student,teacher,b):
    valid=student['valid'];h=(student['latent']-teacher['latent']).square().mean(-1)
    a=(student['attributes']-teacher['attributes']).square().mean(-1)
    return ((h+a)*valid).sum()/valid.sum().clamp_min(1)

def supervision(o,b,pw):
    m=b['ym'];a=(F.binary_cross_entropy_with_logits(o['attributes'],b['y'],reduction='none',pos_weight=pw)*m).sum()/m.sum().clamp_min(1)
    ce=F.cross_entropy(o['stage'].transpose(1,2),b['sy'],ignore_index=-1,reduction='none')
    st=(ce*b['sw']).sum()/b['sw'].sum().clamp_min(1)
    return a+.35*st
