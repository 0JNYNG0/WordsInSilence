from split_model import *
from finger_runtime import FingerObserver,merge_observers,classify,refine
class SplitObserver:
 def __init__(self,mode='compatible_shape'):
  self.mode=mode;self.base=FingerObserver();self.models={}
  for kind in ['shape','quality']:
   mm=[]
   for seed in [42,43,44]:
    ck=torch.load(U/'models'/kind/f'{seed}.pt',map_location='cpu',weights_only=False);m=SharedFinger()if kind=='shape'else TrackQuality();m.load_state_dict(ck['state_dict']);m.eval();mm.append((m,np.array(ck['mean']),np.array(ck['std'])))
   self.models[kind]=mm
 def all(self,x,hand):
  base=self.base.infer(x,hand);f,a=shape_features(x,hand);q,_=quality_features(x,hand);probs={}
  for kind,features in [('shape',f),('quality',q)]:
   xx=windows(features,np.arange(len(x)));ps=[]
   for model,mu,sd in self.models[kind]:
    with torch.inference_mode():ps.append(model(torch.tensor((xx-mu)/sd,dtype=torch.float32)).sigmoid().numpy())
   probs[kind]=np.array(ps)
  p=consensus(probs['shape'],a);quality=consensus(probs['quality'],a);bp=base['probabilities'];compatible,conflict=merge_observers(np.array([classify(z,a)for z in bp['expanded_whole_hand']]),np.array([classify(z,a)for z in probs['shape']]),np.array([classify(z,a)for z in bp['old_v2']]))
  qonly=base['states'].copy();qonly[quality==0]=-1;combined=compatible.copy();combined[quality==0]=-1
  return dict(base=base['states'],shape_only=p,compatible_shape=compatible,quality_only=qonly,combined=combined,quality=quality,available=a,conflicts=conflict,probabilities=probs)
 def infer(self,x,hand):
  z=self.all(x,hand);z['states']=z[self.mode];return z
