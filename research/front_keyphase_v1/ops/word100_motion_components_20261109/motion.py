"""Word-independent body-relation measurements and a small neural readout.

Inputs: observed coordinates, side bindings, relation kind, and time interval.
No word ID, target sign, person ID, guide sentence, or absolute frame/time index.
"""
import numpy as np
import torch
from torch import nn

KINDS=('wrist_height','forearm_upright','along_other_forearm','hand_separation','relative_hand_height')
SIDES={'L':dict(shoulder=1,elbow=3,wrist=5,hand=9),'R':dict(shoulder=2,elbow=4,wrist=6,hand=30)}

def body_points(x):
 p=np.asarray(x[:,:102],float).reshape(-1,51,2);valid=x[:,186:237]>.5
 shoulder=valid[:,1]&valid[:,2];dist=np.linalg.norm(p[:,1]-p[:,2],axis=1);scale=float(np.median(dist[shoulder])) if shoulder.any() else 0.
 if scale<1e-5:return np.zeros_like(p),np.zeros_like(valid)
 return (p-(p[:,1]+p[:,2])[:,None]/2)/scale,valid&shoulder[:,None]

def series(x,kind,moving='R',anchor='L'):
 p,v=body_points(x);m=SIDES[moving];r=SIDES[anchor];hand=m['hand'];other=r['hand']
 # Anatomical side binding, not camera-left or a word-dependent sign convention.
 orientation=np.sign(p[:,2,0]-p[:,1,0])*(1 if moving=='R' else -1)
 p[:,:,0]*=orientation[:,None]
 palm=p[:,[hand,hand+5,hand+9,hand+13,hand+17]].mean(1)
 if kind=='wrist_height':
  q=p[:,m['wrist']];out=np.column_stack([-q[:,1],q[:,0],-palm[:,1],palm[:,0]]);pts=[m['wrist'],hand,hand+5,hand+9,hand+13,hand+17]
 elif kind=='forearm_upright':
  arm=p[:,m['wrist']]-p[:,m['elbow']];length=np.linalg.norm(arm,axis=1);out=np.column_stack([-arm[:,1]/np.maximum(length,1e-5),arm[:,0]/np.maximum(length,1e-5),length,-p[:,m['wrist'],1]]);pts=[m['wrist'],m['elbow']]
 elif kind=='along_other_forearm':
  arm=p[:,r['wrist']]-p[:,r['elbow']];length=np.linalg.norm(arm,axis=1);rel=palm-p[:,r['elbow']]
  out=np.column_stack([(rel*arm).sum(1)/np.maximum(length**2,1e-5),rel[:,0],rel[:,1],np.linalg.norm(palm-p[:,r['wrist']],axis=1)]);pts=[r['elbow'],r['wrist'],hand,hand+5,hand+9,hand+13,hand+17]
 elif kind in ('hand_separation','relative_hand_height'):
  diff=p[:,hand]-p[:,other];dist=np.linalg.norm(diff,axis=1);height=-diff[:,1]
  out=np.column_stack([dist if kind=='hand_separation' else height,abs(diff[:,0]),height if kind=='hand_separation' else dist,-palm[:,1]]);pts=[hand,other,hand+5,hand+9,hand+13,hand+17]
 else:raise ValueError('Unsupported relation')
 good=v[:,pts].all(1)&np.isfinite(out).all(1)
 if kind in ('forearm_upright','along_other_forearm'):good&=length>.05
 return out,good

def features(x,ts,a,b,kind,moving='R',anchor='L',reverse=False):
 values,good=series(x,kind,moving,anchor);values=values[a:b+1];good=good[a:b+1];time=np.asarray(ts[a:b+1],float)
 if reverse:values=values[::-1];good=good[::-1];time=time[-1]-time[::-1]
 coverage=float(good.mean());eligible=coverage>=.8 and int(good.sum())>=4 and good[0] and good[-1]
 if not eligible:return np.zeros(58,np.float32),False,dict(coverage=coverage,observed=int(good.sum()),state='unknown')
 tg=time[good];q=values[good];uniform=np.linspace(tg[0],tg[-1],9)
 sampled=np.stack([np.interp(uniform,tg,q[:,j]) for j in range(4)],1)
 rel=sampled-sampled[0];delta=sampled[-1]-sampled[0]
 data=np.r_[rel.ravel(),sampled[0],delta,np.abs(np.diff(sampled,axis=0)).mean(0),np.ptp(sampled,axis=0),np.eye(5)[KINDS.index(kind)],coverage]
 assert data.shape==(58,) and np.isfinite(data).all()
 return data.astype(np.float32),True,dict(coverage=coverage,observed=int(good.sum()),signed_change=float(delta[0]),trajectory=sampled[:,0].tolist(),state='observed_not_correctness')

class MotionNet(nn.Module):
 def __init__(self,input_dim=58):
  super().__init__();self.layers=nn.Sequential(nn.Linear(input_dim,48),nn.GELU(),nn.Linear(48,24),nn.GELU(),nn.Linear(24,3))
 def forward(self,x):return self.layers(x)
