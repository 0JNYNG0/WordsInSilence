from runtime import *
from finger_runtime import FingerHead,local_sequence
class DualCenterObserver(CenterTemporalObserver):
 def __init__(self,mode='dual_change_fill'):
  super().__init__('change_fill');assert mode in ['dual_center_only','dual_change_override','dual_change_fill'];self.mode=mode;self.whole=[]
  for seed in [42,43,44]:
   ck=torch.load(U/'models'/'whole_center'/f'{seed}.pt',map_location='cpu',weights_only=False);m=FingerHead();m.load_state_dict(ck['state_dict']);m.eval();self.whole.append((m,np.array(ck['mean']),np.array(ck['std'])))
 def all(self,x,hand):
  z=super().all(x,hand);fl,_=local_sequence(x,hand);xx=np.c_[fl,np.zeros((len(x),64),np.float32)][:,None];ps=[]
  for m,mu,sd in self.whole:
   v=(xx-mu)/sd;v[:,:,63:]=0
   with torch.inference_mode():ps.append(m(torch.tensor(v,dtype=torch.float32)).sigmoid().numpy()[:,1:])
  z['probabilities']['whole_center']=np.array(ps);whole=consensus(np.array(ps),z['available'],.9);dual=np.where((whole==z['center_only'])&(whole>=0),whole,-1);over=z['base'].copy();fill=over.copy();ok=(dual>=0)&(z['phase']==1)&z['available'];over[ok]=dual[ok];f=ok&(fill<0)&~z['conflicts'];fill[f]=dual[f]
  return z|dict(whole_center=whole,dual_center_only=dual,dual_change_override=over,dual_change_fill=fill)
