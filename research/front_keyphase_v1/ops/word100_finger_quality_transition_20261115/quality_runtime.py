from pathlib import Path
import sys,numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent;sys.path[:0]=[str(O/'word100_finger_repair_20261115'),str(O/'word100_clause_finger_diagnostics_20261114')]
from finger_runtime import FingerObserver,refine
from quality_model import QualityChange,features,windows
class QualityObserver:
 def __init__(self):
  self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(U/'models'/f'{seed}.pt',map_location='cpu',weights_only=False);m=QualityChange(ck['features']);m.load_state_dict(ck['state_dict']);m.eval();self.models.append((m,ck['mean'],ck['std']))
 def infer(self,x,hand):
  f,valid=features(x,hand);z=windows(f,np.arange(len(x)));ps=[]
  for m,mu,sd in self.models:
   with torch.inference_mode():ps.append(m(torch.tensor((z-mu)/sd,dtype=torch.float32)).sigmoid().numpy())
  ps=np.array(ps);yes=(ps>=.85).all(0)&valid[:,:,None];no=(ps<=.15).all(0)&valid[:,:,None];states=np.where(yes,1,np.where(no,0,-1));return dict(probabilities=ps,states=states,available=valid)
def gate(states,q,mode):
 if mode=='quality_only':blocked=q['states'][:,:,0]==1
 elif mode=='phase_only':blocked=q['states'][:,:,1]==1
 elif mode=='combined':blocked=(q['states']==1).any(-1)
 else:raise ValueError(mode)
 out=states.copy();out[blocked]=-1;return out,blocked
class GuardedObserver:
 def __init__(self,mode):self.previous=FingerObserver();self.quality=QualityObserver();self.mode=mode
 def infer(self,x,hand):
  old=self.previous.infer(x,hand);q=self.quality.infer(x,hand);old['prior_states']=old['states'].copy();old['states'],old['quality_phase_blocked']=gate(old['states'],q,self.mode);old['quality_phase']=q
  return old
