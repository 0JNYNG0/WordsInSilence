"""Read-only, label-free recheck scheduling. Trigger is not an error verdict."""
import numpy as np
ANCHORS=np.array([0,5,9,13,17])
TRIPLES=np.array([(ch[k],ch[k+1],ch[k+2]) for ch in [(0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20)] for k in range(3)])

def angles(x,v):
 a=x[...,TRIPLES[:,0],:]-x[...,TRIPLES[:,1],:];b=x[...,TRIPLES[:,2],:]-x[...,TRIPLES[:,1],:]
 good=v[...,TRIPLES].all(-1)&(np.linalg.norm(a,axis=-1)>1e-6)&(np.linalg.norm(b,axis=-1)>1e-6)
 val=np.degrees(np.arctan2(abs(a[...,0]*b[...,1]-a[...,1]*b[...,0]),(a*b).sum(-1)))
 return np.where(good,val,np.nan)

def descriptors(x,valid,time,roi):
 x=np.asarray(x,float);v=np.asarray(valid,bool)&np.isfinite(x).all(-1);n=len(x);assert x.shape==(n,2,21,2) and time.shape==(n,) and (np.diff(time)>0).all()
 inroi=v&(x[...,0]>=roi[0])&(x[...,0]<roi[2])&(x[...,1]>=roi[1])&(x[...,1]<roi[3]);visible=inroi.any(-1)
 a=angles(x,v);resid=np.full((n,2),np.nan);rate=resid.copy();returning=np.zeros((n,2),bool);partial=visible&(inroi.sum(-1)<21);overlap=np.zeros((n,2),bool)
 for i in range(n):
  if visible[i].all():
   p=[x[i,h,inroi[i,h]] for h in (0,1)];lo=np.maximum(p[0].min(0),p[1].min(0));hi=np.minimum(p[0].max(0),p[1].max(0))
   if (hi>lo).all():overlap[i]=True
  if i==0:
   returning[i]=visible[i];continue
  returning[i]=visible[i]&~visible[i-1]
  dt=(time[i]-time[i-1])/1e6
  # A long unobserved time interval is not a measured fast motion.
  if dt>0.2:continue
  for h in (0,1):
   if not visible[i,h] or not visible[i-1,h]:continue
   common=v[i,h]&v[i-1,h]
   da=np.abs(a[i,h]-a[i-1,h]);good=np.isfinite(da)
   if good.any():rate[i,h]=np.median(da[good])/dt
   if not common[ANCHORS].all() or common.sum()<5:continue
   pa=x[i-1,h,ANCHORS];pb=x[i,h,ANCHORS];ac=pa.mean(0);bc=pb.mean(0);aa=pa-ac;bb=pb-bc
   denom=np.sum(aa*aa);scale=np.sqrt(np.mean(np.sum(bb*bb,axis=1)))
   if denom<=1e-8 or scale<=1e-6:continue
   u,s,vt=np.linalg.svd(aa.T@bb);cor=np.eye(2);cor[1,1]=np.sign(np.linalg.det(u@vt));rot=u@cor@vt;factor=np.trace(np.diag(s)@cor)/denom
   if factor<=1e-8:continue
   projected=(x[i-1,h,common]-ac)@rot*factor+bc
   resid[i,h]=np.sqrt(np.mean(np.sum((projected-x[i,h,common])**2,axis=1)))/scale
 return dict(visible=visible,inroi=inroi,returning=returning,partial=partial,overlap=overlap,shape_residual=resid,angle_rate=rate)

def schedule(d,thresholds):
 # Every bit asks for an observation, never modifies an existing observation.
 reason=(d['returning'].astype(np.uint8)+2*d['partial'].astype(np.uint8)+4*d['overlap'].astype(np.uint8)+8*(d['shape_residual']>thresholds['shape_residual']).astype(np.uint8)+16*(d['angle_rate']>thresholds['angle_rate']).astype(np.uint8))
 reason=np.where(d['visible'],reason,0).astype(np.uint8)
 return reason

def crop_box(x,v,w,h):
 p=np.asarray(x)[v&np.isfinite(x).all(-1)]
 if not len(p):return None
 lo,hi=p.min(0),p.max(0);c=(lo+hi)/2;side=max(float(np.max(hi-lo))*1.6,64.)
 a=np.floor(c-side/2).astype(int);b=np.ceil(c+side/2).astype(int)
 return [max(0,int(a[0])),max(0,int(a[1])),min(w,int(b[0])),min(h,int(b[1]))]
