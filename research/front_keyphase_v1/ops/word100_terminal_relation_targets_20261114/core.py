"""Independent visible traits. No word, phase, source ID or duration inputs."""
import copy
import numpy as np
import torch
from torch import nn

CLASSES = {'shape':['four_open','four_closed'],
           'region':['face_front','upper_body'],
           'forearm':['across','upright']}
DIMS={'shape':12,'region':8,'forearm':2}

def features(x, domain, hand):
    x=np.asarray(x);p=x[:,:102].reshape(-1,51,2).astype('float64');v=x[:,186:237]>.5
    root=30 if hand=='R' else 9; elbow=4 if hand=='R' else 3
    axis=p[:,2]-p[:,1];scale=np.linalg.norm(axis,axis=1)
    right=axis/np.maximum(scale[:,None],1e-8);down=np.stack([-right[:,1],right[:,0]],1)
    down*=np.where(down[:,1:2]>=0,1,-1);center=(p[:,1]+p[:,2])/2
    def local(d):
        return np.stack([(d*right).sum(1),(d*down).sum(1)],1)/np.maximum(scale[:,None],1e-8)
    if domain=='shape':
        vals=[];ids=[root];size=np.linalg.norm(p[:,root+9]-p[:,root],axis=1)
        for start in [5,9,13,17]:
            ids+=list(range(root+start,root+start+4))
            a,b,c,d=[p[:,root+start+k] for k in range(4)]
            def cos(u,w):return (u*w).sum(1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(w,axis=1),1e-8)
            vals += [cos(b-a,c-b),cos(c-b,d-c),np.linalg.norm(d-a,axis=1)/np.maximum(size,1e-8)]
        f=np.stack(vals,1);ok=v[:,ids].all(1)&(size>1e-6)
    elif domain=='region':
        ids=[root,root+5,root+9,root+13,root+17];palm=p[:,ids].mean(1)
        a=local(palm-center);b=local(palm-p[:,0]);s=local(palm-p[:,2 if hand=='R' else 1])
        if hand=='L':a[:,0]*=-1;b[:,0]*=-1;s[:,0]*=-1
        f=np.c_[a,b,s,np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1)]
        ok=v[:,ids+[0,1,2]].all(1)&(scale>1e-6)
    else:
        a=local(p[:,root]-p[:,elbow]);n=np.linalg.norm(a,axis=1)
        a/=np.maximum(n[:,None],1e-8)
        if hand=='L':a[:,0]*=-1
        f=a;ok=v[:,[1,2,elbow,root]].all(1)&(scale>1e-6)&(n>1e-6)
    ok &=np.isfinite(f).all(1);f=np.clip(np.nan_to_num(f),-10,10);f[~ok]=0
    return f.astype('float32'),ok

class TraitHeads(nn.Module):
    def __init__(self):
        super().__init__();self.heads=nn.ModuleDict({d:nn.Sequential(nn.Linear(n,16),nn.Tanh(),nn.Linear(16,2))for d,n in DIMS.items()})
    def forward(self,x,d):return self.heads[d](x)

def consensus(pp, available, domain):
    pp=np.asarray(pp);votes=pp.argmax(-1)
    known=np.asarray(available)&(votes==votes[:1]).all(0)&(pp.max(-1)>=.8).all(0)
    return [CLASSES[domain][int(votes[0,i])]if known[i]else None for i in range(len(known))]

def assess(states, available, ts, requirement, association='confirmed'):
    """Assess visible requirements within a supplied interval, never find desired frames.

    A positive statement concerns this feature sequence only, not whole-sign correctness.
    A known start posture retained at the tail differs from a missing/unknown tail.
    A long unobserved interval cannot be filled by the expected guide.
    """
    ts=np.asarray(ts);n=len(ts)
    if n==0 or (n>1 and not np.all(np.diff(ts)>0)):raise ValueError('Native increasing timestamps required')
    reqdomains=set(requirement['start'])|set(requirement['goal'])
    for d in reqdomains:
        if len(states[d])!=n or len(available[d])!=n:raise ValueError('Trait/time mismatch')
    def match(i,req):return all(states[d][i]==value and bool(available[d][i])for d,value in req.items())
    start=[i for i in range(n)if match(i,requirement['start'])]
    goal=[i for i in range(n)if match(i,requirement['goal'])]
    result=dict(status='observation_uncertain',start_indices=start,goal_indices=goal,witness_indices=[],
        feature_sequence_observed=False,whole_action_complete=None,educational_error=None,
        comparison_retained=True,minimum_duration=None,minimum_displacement=None,display_enabled=False)
    if association!='confirmed':return dict(result,status='action_association_uncertain')
    if start and goal:
        pairs=[(a,b)for a in start for b in goal if a<b]
        if pairs:
            a,b=min(pairs,key=lambda ab:ab[1])
            visible=np.array([all(bool(available[d][i])for d in reqdomains)for i in range(a,b+1)])
            t=ts[a:b+1][visible]
            maxgap=int(np.diff(t).max())if len(t)>1 else None
            # Observability continuity, not a minimum action duration. Unknown
            # classification during a visible transition is allowed, not imputed.
            linked=maxgap is not None and maxgap<=120000 and visible.mean()>=.8
            return dict(result,status='required_feature_sequence_observed'if linked else'endpoint_pair_only',
                witness_indices=[a,b],feature_sequence_observed=linked,maximum_visible_gap_us=maxgap)
        return dict(result,status='reverse_feature_order')
    if goal:return dict(result,status='goal_posture_without_start')
    if all(available[d][-1]and states[d][-1]is not None for d in reqdomains):
        if match(n-1,requirement['start']):return dict(result,status='still_at_start_state')
        return dict(result,status='required_goal_not_confirmed')
    return result

def attach(original,evidence):
    out=copy.deepcopy(original);out['terminal_trait_diagnostics']=copy.deepcopy(evidence)
    return out
