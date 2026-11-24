"""Research-only evidence/label eligibility. Never a deployed signing judge.

Tokens below are HUMAN-REVIEWED observable phases, not word chunks inferred from
spelling, softmax predictions, equal thirds or geometric nearest neighbours.
"""
from functools import lru_cache
import numpy as np

def assess_tokens(tokens,contract,fully_observed=True):
    # Unknown observations cannot establish that a required action was omitted.
    if not fully_observed or '?' in tokens:
        return {'status':'INSUFFICIENT_OBSERVATION','error':None}
    if not contract.get('expert_reviewed',False):
        return {'status':'REVIEW_REQUIRED','error':None}
    # Consecutive identical tokens are dwell/measurement samples of one phase,
    # not distinct cycles. Phase recurrence A-B-A-B remains distinct.
    compact=[]
    for t in tokens:
        if not compact or compact[-1]!=t:compact.append(t)
    tokens=tuple(compact);blocks=contract['blocks']
    @lru_cache(None)
    def match(b,pos):
        if b==len(blocks):return pos==len(tokens)
        block=blocks[b];seq=tuple(block['sequence']);lo=block['minimum_cycles'];hi=block['maximum_cycles']
        # "unbounded" means reviewer-declared permissible count variation.
        hi=len(tokens)//max(1,len(seq)) if hi is None else hi
        for count in range(lo,hi+1):
            expected=seq*count
            if tokens[pos:pos+len(expected)]==expected and match(b+1,pos+len(expected)):return True
        return False
    return {'status':'MATCHES_REVIEWED_STRUCTURE' if match(0,0) else 'STRUCTURE_REVIEW_NEEDED',
            'error':None,'complete_sign_correctness':None}

def arrays(x,axis):
    if axis=='global':return x[:,18:102].reshape(-1,42,2).astype('float64'),x[:,195:237]>0
    if axis=='local':return x[:,102:186].reshape(-1,42,2).astype('float64'),x[:,237:279]>0
    raise ValueError(axis)

def costs(a,b,axis):
    aa,av=arrays(a,axis);bb,bv=arrays(b,axis)
    both=av[:,None,:]&bv[None,:,:]
    n=both.sum(-1)
    dist=np.linalg.norm(aa[:,None,:,:]-bb[None,:,:,:],axis=-1)
    c=np.full(n.shape,np.inf)
    np.divide((dist*both).sum(-1),n,out=c,where=n>0)
    return c,n

def directed(cost):
    # Directed monotone coverage: every reference frame contributes once.
    # Unlike unrestricted nearest-neighbour matching, query indices may not
    # decrease. Duplicate query frames/dwell do not create an order penalty.
    dp=cost[0].copy()
    for row in cost[1:]:dp=row+np.minimum.accumulate(dp)
    return float(dp.min()/len(cost))

def compare(a,b,axis):
    cost,support=costs(a,b,axis)
    # Empty rows/columns have no geometric evidence. Exclude ONLY from the
    # descriptive distance and explicitly retain their counts/status. This
    # does not join gaps into a claim of complete, correctly ordered signing.
    ra=np.isfinite(cost).any(1);rb=np.isfinite(cost).any(0)
    out={'reference_frames':len(a),'query_frames':len(b),'reference_uncomparable':int((~ra).sum()),
         'query_uncomparable':int((~rb).sum()),'semantic_verdict':None,'distance_scope':'observed_geometry_only'}
    if not ra.any() or not rb.any():return dict(out,status='INSUFFICIENT',bag=None,ordered=None,order_gap=None)
    cc=cost[np.ix_(ra,rb)];ss=support[np.ix_(ra,rb)]
    bag=(np.min(cc,1).mean()+np.min(cc,0).mean())/2
    ordered=(directed(cc)+directed(cc.T))/2
    if not np.isfinite(ordered):return dict(out,status='INSUFFICIENT_MONOTONE_EVIDENCE',bag=float(bag),ordered=None,order_gap=None)
    assert ordered+1e-12>=bag
    return dict(out,status='OBSERVED_DISTANCE_ONLY',bag=float(bag),ordered=float(ordered),order_gap=float(max(0,ordered-bag)),
                median_joint_support=float(np.median(ss[ss>0]))/42)

CONDITIONS=('original','clock_slow','clock_fast','clock_local_speed','dwell_samples','omit_middle','swap_first_two','repeat_middle','mask_middle')
def variant(x,name):
    a=x.copy();n=len(a);chunks=np.array_split(np.arange(n),3)
    idx=np.arange(n)
    if name=='clock_slow':a[1:,279]/=.7
    elif name=='clock_fast':a[1:,279]/=1.3
    elif name=='clock_local_speed':a[1:,279]*=np.linspace(.7,1.3,n-1)
    elif name=='dwell_samples':idx=np.repeat(idx,np.where(idx%3==0,2,1));a=x[idx].copy()
    elif name=='omit_middle':idx=np.r_[chunks[0],chunks[2]];a=x[idx].copy()
    elif name=='swap_first_two':idx=np.r_[chunks[1],chunks[0],chunks[2]];a=x[idx].copy()
    elif name=='repeat_middle':idx=np.r_[chunks[0],chunks[1],chunks[1],chunks[2]];a=x[idx].copy()
    elif name=='mask_middle':
        a[chunks[1],18:186]=0;a[chunks[1],195:279]=0
    elif name!='original':raise ValueError(name)
    if name in ('dwell_samples','omit_middle','swap_first_two','repeat_middle'):
        # These are synthetic diagnostics with provenance. No resampled or
        # fabricated coordinates; a uniform clock avoids transferring old dt
        # in a different order. No classifier reads these variants.
        a[:,279]=np.median(x[1:,279]);a[0,279]=0
    assert np.isfinite(a).all() and a[0,279]==0 and np.all(a[1:,279]>0)
    return a,idx
