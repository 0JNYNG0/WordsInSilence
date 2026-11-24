from model import *
from finger_runtime import FingerObserver,refine
class CenterTemporalObserver:
 def __init__(self,mode='change_fill'):
  assert mode in ['center_only','change_override','change_fill'];self.mode=mode;self.base=FingerObserver();self.models={};self.cfg=read(U/'TRAIN_CONFIG.json')
  for kind in ['state','phase']:
   mm=[]
   for seed in self.cfg['seeds']:
    ck=torch.load(U/'models'/kind/f'{seed}.pt',map_location='cpu',weights_only=False);m=SharedFinger()if kind=='state'else ChangeHead();m.load_state_dict(ck['state_dict']);m.eval();mm.append((m,ck['mean'],ck['std']))
   self.models[kind]=mm
 def all(self,x,hand):
  base=self.base.infer(x,hand);f,a=features(x,hand);xx=windows(f,np.arange(len(f)));probs={}
  for kind,models in self.models.items():
   value=f[:,None]if kind=='state'else xx;ps=[]
   for m,mu,sd in models:
    with torch.inference_mode():ps.append(m(torch.tensor((value-mu)/sd,dtype=torch.float32)).sigmoid().numpy())
   probs[kind]=np.array(ps)
  state=consensus(probs['state'],a,self.cfg['state_threshold']);phase=consensus(probs['phase'],a,self.cfg['phase_threshold']);override=base['states'].copy();fill=base['states'].copy();ok=(state>=0)&(phase==1)&a
  override[ok]=state[ok];fill_ok=ok&(fill<0)&~base['conflicts'];fill[fill_ok]=state[fill_ok]
  return dict(base=base['states'],center_only=state,change_override=override,change_fill=fill,phase=phase,available=a,conflicts=base['conflicts'],probabilities=probs)
 def infer(self,x,hand):
  z=self.all(x,hand);return z|dict(states=z[self.mode])
