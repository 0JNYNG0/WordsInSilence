"""Hand-local coarse evidence supplement; individual-finger facts stay separate."""
from local_shape import *
sys.path.insert(0,str(SOURCE));from finger_runtime import FingerObserver

class ExpandedShape(LocalShape):
 def __init__(self):
  self.kind='temporal';self.models=[]
  for seed in [42,43,44]:
   ck=torch.load(V/'expanded'/f'{seed}.pt',map_location='cpu',weights_only=False)
   m=HandHead('temporal');m.load_state_dict(ck['state']);m.eval();self.models.append((m,ck['mean'],ck['std']))

def combine(fingers,before,expanded):
 allknown=(fingers>=0).all(-1)
 base=np.where((fingers==1).all(-1),0,np.where((fingers==0).all(-1),1,np.where(allknown,2,-1)))
 agree=np.where((before==expanded)&(before>=0),before,-1)
 # A coarse class cannot overrule any known individual finger.
 conflict=((agree==0)&(fingers==0).any(-1))|((agree==1)&(fingers==1).any(-1))
 agree[conflict]=-1
 return np.where(allknown,base,agree),base

class ShapeConsensus:
 def __init__(self):
  self.finger=FingerObserver();self.before=LocalShape('temporal');self.expanded=ExpandedShape()
 def sequence(self,x,hand):
  f=self.finger.infer(x,hand);a,_=self.before.sequence(x,hand);b,_=self.expanded.sequence(x,hand);s,base=combine(f['states'],a,b)
  # Coarse agreement cannot restore an unobserved collapsed finger chain.
  usable=f['available'].all(-1);s[~usable]=-1
  reason=np.full(len(s),'model_confidence_insufficient',dtype='<U40')
  disagreement=((a>=0)&(b>=0)&(a!=b))|(f['reasons']=='model_disagreement').any(-1)
  reason[disagreement]='model_disagreement';reason[s>=0]='model_decided'
  reason[~usable]='geometry_unavailable'
  return dict(states=s,baseline=base,fingers=f['states'],available=f['available'],
              reasons=reason,finger_reasons=f['reasons'],geometry_reasons=f['geometry_reasons'])

def freeze():
 path=V/'CONSENSUS_FROZEN.json';assert not path.exists()
 write(path,dict(rule='Keep existing complete individual-finger observation. Otherwise add only agreement of original and FIT-expanded hand-local temporal ensembles, with no conflict against any known individual finger. Never fill individual-finger labels from coarse class.',
  independent_for_this_candidate=[16,17],prior_seen_diagnostics=[11,12,13,14,15],thresholds_changed=False,
  check='No additional wrong known DEV labels; original normal source not called flexed; synthetic hand replacement detected; missing unknown; opposite hand invariant. Full word orientation/contact still separate.',
  inputs={str(p):sha(p)for folder in ['temporal','expanded']for p in sorted((V/folder).glob('*.pt'))}))
 print('CONSENSUS FROZEN')
if __name__=='__main__':freeze()
