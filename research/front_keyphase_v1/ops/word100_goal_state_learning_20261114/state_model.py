"""Observation-only state component. Never uses source index, word or time."""
import copy
import numpy as np
import torch
from torch import nn
CLASSES={'forearm':['wrist_above_elbow','forearm_across','wrist_below_elbow'],
 'arm_region':['upper_arm_side','wrist_side'], 'separation':['hands_near','hands_apart'],
 'height':['active_hand_higher','opposite_hand_higher']}
DIMS={'forearm':4,'arm_region':6,'separation':3,'height':3}
def features(x,domain,hand='R'):
    x=np.asarray(x);p=x[:,:102].reshape(-1,51,2).astype('float64');v=x[:,186:237]>.5
    shoulder=p[:,2]-p[:,1];scale=np.linalg.norm(shoulder,axis=1)
    right=shoulder/np.maximum(scale[:,None],1e-8)
    # Shoulder axis gives in-plane body rotation invariance. y down in image.
    down=np.stack([-right[:,1],right[:,0]],1)
    down*=np.where(down[:,1:2]>=0,1,-1)
    center=(p[:,1]+p[:,2])/2
    def local(vec):return np.stack([(vec*right).sum(1),(vec*down).sum(1)],1)/np.maximum(scale[:,None],1e-8)
    def palm(side):
        idx=[30,35,39,43,47]if side=='R'else[9,14,18,22,26]
        return p[:,idx].mean(1),v[:,idx].all(1)
    active,opp=('R','L')if hand=='R'else('L','R')
    s,e,w=(2,4,6)if hand=='R'else(1,3,5)
    os,oe,ow=(1,3,5)if hand=='R'else(2,4,6)
    ok=v[:,[1,2]].all(1)&(scale>1e-6)
    sign=1 if hand=='R'else -1
    if domain=='forearm':
        a=local(p[:,w]-p[:,e]);b=local(p[:,w]-center);a[:,0]*=sign;b[:,0]*=sign
        a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8)
        f=np.concatenate([a,b],1);ok &=v[:,[s,e,w]].all(1)
    elif domain=='arm_region':
        palm_xy,pok=palm(active);d=local(p[:,ow]-p[:,oe]);q=local(palm_xy-p[:,oe]);n=np.linalg.norm(d,axis=1)
        projection=(q*d).sum(1)/np.maximum(n*n,1e-8)
        lateral=np.abs(q[:,0]*d[:,1]-q[:,1]*d[:,0])/np.maximum(n,1e-8)
        f=np.stack([projection,lateral,np.linalg.norm(local(palm_xy-p[:,os]),axis=1),
          np.linalg.norm(q,axis=1),np.linalg.norm(local(palm_xy-p[:,ow]),axis=1),n],1)
        ok &=pok&v[:,[os,oe,ow]].all(1)&(n>1e-6)
    else:
        a,va=palm(active);b,vb=palm(opp);d=local(a-b);d[:,0]*=sign
        if domain=='separation':f=np.stack([np.abs(d[:,0]),np.abs(d[:,1]),np.linalg.norm(d,axis=1)],1)
        else:f=np.stack([d[:,1],np.abs(d[:,0]),np.linalg.norm(d,axis=1)],1)
        ok &=va&vb
    ok &=np.isfinite(f).all(1);f=np.clip(np.nan_to_num(f),-10,10);f[~ok]=0
    return f.astype('float32'),ok
class StateHeads(nn.Module):
    def __init__(self,kind='mlp'):
        super().__init__();self.kind=kind
        self.heads=nn.ModuleDict({k:nn.Sequential(nn.Linear(n,16),nn.Tanh(),nn.Linear(16,len(CLASSES[k])))if kind=='mlp'else nn.Linear(n,len(CLASSES[k]))for k,n in DIMS.items()})
    def forward(self,x,domain):return self.heads[domain](x)
def consensus(probabilities,available,domain,confidence=.8):
    # All three fixed seeds must agree; unavailable observations abstain.
    pp=np.asarray(probabilities);votes=pp.argmax(-1)
    sure=np.asarray(available)&np.all(votes==votes[:1],axis=0)&np.all(pp.max(-1)>=confidence,axis=0)
    return [CLASSES[domain][int(votes[0,i])]if sure[i]else None for i in range(len(sure))]
def transition_evidence(states,path,available):
    """Evidence sequence, NOT semantic completion. Unknowns never invent states."""
    observed=[(i,s)for i,s in enumerate(states)if s is not None]
    result=dict(goal_observed=False,ordered_transition_observed=False,semantic_completion=None,
                educational_error=None,comparison_retained=True,required_duration=None)
    if not observed:return dict(result,status='observation_insufficient',witness_indices=[])
    result['goal_observed']=any(s==path[-1]for _,s in observed)
    cursor=0;witness=[]
    for i,s in observed:
        if s==path[cursor]:
            witness.append(i);cursor+=1
            if cursor==len(path):break
    if cursor==len(path):return dict(result,status='ordered_state_change_observed',ordered_transition_observed=True,witness_indices=witness)
    # Unseen tail cannot be called an early stop. Do not use confidence as truth.
    if not available[-1]or states[-1]is None:return dict(result,status='observation_insufficient',witness_indices=witness)
    if result['goal_observed']:
        status='goal_pose_only'if all(s==path[-1]for _,s in observed)else'order_not_supported'
    elif states[-1]==path[0]:status='observed_before_goal'
    else:status='unresolved_state'
    return dict(result,status=status,witness_indices=witness)
def attach_component(located,components):
    out=copy.deepcopy(located)
    # No card deletion, reordering, threshold/extent modification or attribute overwrite.
    out['goal_state_diagnostics']=copy.deepcopy(components)
    out['goal_state_diagnostics_display_enabled']=False
    return out
