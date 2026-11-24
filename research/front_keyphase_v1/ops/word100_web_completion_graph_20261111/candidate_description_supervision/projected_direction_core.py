"""Shoulder-normalized projection, avoiding division by squared projected arm length.
One global measurement change for axis 2 only, not a word exception. Does not
restore unseen coordinates or prove contact; arm orientation can still be noisy.
"""
from boundary_direction_core import *
V=W/'projected_direction_learning'
prior=load('_proj_prior',O/'word100_broad_factorized_learning_20261110/core.py')

def geometry(x,ts,a,b,axis,moving='R',anchor='L',reverse=False):
 if axis!=2:return prior.geometry(x,ts,a,b,axis,moving,anchor,reverse)
 p,valid=prior.motion.body_points(x);m=prior.motion.SIDES[moving];r=prior.motion.SIDES[anchor]
 p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if moving=='R'else -1))[:,None]
 q=p[:,m['wrist']];other=p[:,r['wrist']];arm=other-p[:,r['elbow']];length=np.linalg.norm(arm,axis=1);rel=q-p[:,r['elbow']]
 values=np.column_stack([(rel*arm).sum(1)/np.maximum(length,1e-5),rel[:,0],rel[:,1],np.linalg.norm(q-other,axis=1)])
 good=valid[:,[m['wrist'],r['wrist'],r['elbow']]].all(1)&np.isfinite(values).all(1)&(length>.05)
 values=values[a:b+1];good=good[a:b+1];tt=np.asarray(ts[a:b+1],float)
 if len(tt)<1:return np.zeros(58,np.float32),False,dict(coverage=0.,observed=0)
 if reverse:values=values[::-1];good=good[::-1];tt=tt[-1]-tt[::-1]
 cov=float(good.mean());ok=cov>=.8 and int(good.sum())>=4 and bool(good[0]and good[-1])
 if not ok:return np.zeros(58,np.float32),False,dict(coverage=cov,observed=int(good.sum()))
 q=values[good];tg=tt[good];sampled=np.stack([np.interp(np.linspace(tg[0],tg[-1],9),tg,q[:,j])for j in range(4)],1)
 delta=sampled[-1]-sampled[0];g=np.r_[(sampled-sampled[0]).ravel(),sampled[0],delta,np.abs(np.diff(sampled,axis=0)).mean(0),np.ptp(sampled,axis=0),np.eye(5)[axis],cov]
 return g.astype(np.float32),True,dict(coverage=cov,observed=int(good.sum()),signed_change=float(delta[0]),trajectory=sampled[:,0].tolist(),body_wrist_only=True,contact_verified=False,units='shoulder_widths_along_current_anchor_direction',anchor_length_range=[float(length[a:b+1][good].min()),float(length[a:b+1][good].max())])
