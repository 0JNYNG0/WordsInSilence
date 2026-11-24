"""Learned ambiguity check, shared across vocabulary and hands.

It can abstain from an existing finger claim, never invent a new finger state.
No word label, target shape, guide sentence or source identity is input.
"""
from pathlib import Path
import sys,numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent;P=O/'word100_finger_repair_20261115';D=O/'word100_clause_finger_diagnostics_20261114'
sys.path[:0]=[str(P),str(D)]
from finger_runtime import FingerObserver as PreviousObserver,refine
from finger_core import FingerHead,local_sequence
from finger_model import windows,features

class Guard:
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(U/'guard_models'/f'{seed}.pt',map_location='cpu',weights_only=False);m=FingerHead();m.load_state_dict(ck['state_dict']);m.eval();self.models.append((m,np.array(ck['mean']),np.array(ck['std'])))
 def infer_windows(self,x,available):
  ps=[]
  for m,mu,sd in self.models:
   v=(x-mu)/sd;v[:,:,63:]=0
   with torch.inference_mode():p=m(torch.tensor(v,dtype=torch.float32))[:,1:].sigmoid().numpy()
   ps.append(p)
  probs=np.array(ps);q=np.where(probs>=.85,1,np.where(probs<=.15,0,-1));q[:,~available]=-1
  states=np.where((q==q[0]).all(0),q[0],-1)
  return states,probs
 def infer(self,x,hand):
  f,_=local_sequence(x,hand);_,a=features(x,hand);xx=windows(np.c_[f,np.zeros((len(x),64),np.float32)],np.arange(len(x)))
  return self.infer_windows(xx,a)

def gate(old,new,mode):
 if mode=='veto_only':blocked=(old>=0)&(new>=0)&(old!=new)
 elif mode=='require_agreement':blocked=(old>=0)&(new!=old)
 else:raise ValueError(mode)
 q=old.copy();q[blocked]=-1;return q,blocked

class GuardedObserver:
 def __init__(self,mode):self.previous=PreviousObserver();self.guard=Guard();self.mode=mode
 def infer(self,x,hand):
  q=self.previous.infer(x,hand);new,prob=self.guard.infer(x,hand);old=q['states'].copy();q['states'],blocked=gate(old,new,self.mode);q['guard_blocked']=blocked;q['previous_states']=old;q['guard_states']=new;q['probabilities']['learned_guard']=prob
  return q
