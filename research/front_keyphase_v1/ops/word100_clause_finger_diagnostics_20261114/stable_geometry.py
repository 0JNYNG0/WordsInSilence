import numpy as np
def geometry(prior,x,ts,a,b,axis,moving='R',anchor='L',reverse=False):
 if prior.motion.KINDS[axis]!='along_other_forearm':return prior.geometry(x,ts,a,b,axis,moving,anchor,reverse)
 p,v=prior.motion.body_points(x);m=prior.motion.SIDES[moving];r=prior.motion.SIDES[anchor]
 p[:,:,0]*=(np.sign(p[:,2,0]-p[:,1,0])*(1 if moving=='R'else -1))[:,None]
 ids=[m['wrist'],r['wrist'],r['elbow']];good=v[:,ids].all(1)[a:b+1];p=p[a:b+1];tt=np.asarray(ts[a:b+1],float)
 cov=float(good.mean());ok=cov>=.8 and good.sum()>=4 and good[0] and good[-1]
 if not ok:return np.zeros(58,np.float32),False,dict(coverage=cov,observed=int(good.sum()))
 # A shared coordinate frame over this observed action, never a desired direction.
 # Uniform time sampling keeps a duplicated/slow frame from defining the arm axis.
 tg=tt[good];q=p[good];sample=np.stack([np.stack([np.interp(np.linspace(tg[0],tg[-1],9),tg,q[:,i,j])for j in range(2)],1)for i in ids],1)
 arm=sample[:,1]-sample[:,2];ref=np.median(arm,axis=0);length=float(np.linalg.norm(ref))
 if length<=.05:return np.zeros(58,np.float32),False,dict(coverage=cov,observed=int(good.sum()),reference_arm_too_short=True)
 rel=sample[:,0]-sample[:,2];unit=ref/length
 values=np.c_[rel@unit/length,rel,np.linalg.norm(sample[:,0]-sample[:,1],axis=1)]
 if reverse:values=values[::-1]
 delta=values[-1]-values[0];g=np.r_[(values-values[0]).ravel(),values[0],delta,np.abs(np.diff(values,axis=0)).mean(0),np.ptp(values,axis=0),np.eye(5)[axis],cov]
 return g.astype(np.float32),True,dict(coverage=cov,observed=int(good.sum()),signed_change=float(delta[0]),reference_arm_length=length,axis_mode='interval_median_observed_forearm',contact_verified=False)
