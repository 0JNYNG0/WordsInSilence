"""Causal, diagnostic-only region/center separation. Does not modify C assignments."""
import cv2
import numpy as np

PALM=[0,5,9,13,17]
PARAMS=dict(image_width=640,max_age_s=.20,max_points=32,min_points=6,quality=.01,
    min_distance=3,win=21,levels=3,fb_px=1.5,lk_error=20.,keep_fraction=.60,
    pose_gate=.25,pose_margin=.10,other_hand_gate=.16,
    ransac_px=1.5,inlier_fraction=.70,scale_min=.75,scale_max=1.3333333333,
    rotation_max_deg=45.,residual_shoulder=.02,covariance_ratio=.05,
    feature_area_fraction=.10,extrapolation_palm_fraction=.25,rotation_event_deg=15.)

def inside(p,w,h):
    a=np.asarray(p);return np.isfinite(a).all(-1)&(a[...,0]>=0)&(a[...,0]<w)&(a[...,1]>=0)&(a[...,1]<h)

def center_transform(p,q,center,palm):
    """Similarity transform only; anatomical center remains explicitly an estimate."""
    p=np.asarray(p,np.float32);q=np.asarray(q,np.float32)
    log=dict(previous=p.tolist(),current=q.tolist(),previous_center=center,
             previous_palm=None if palm is None else np.asarray(palm).tolist())
    if center is None or palm is None:return None,dict(log,reason='no_anatomical_anchor')
    cv2.setRNGSeed(31)
    m,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,
        ransacReprojThreshold=PARAMS['ransac_px'],maxIters=2000,confidence=.99,refineIters=10)
    if m is None or not np.isfinite(m).all():return None,dict(log,reason='no_transform')
    keep=inliers.reshape(-1).astype(bool);a=p[keep];b=q[keep]
    log.update(matrix=m.tolist(),inliers=keep.tolist())
    if keep.sum()<PARAMS['min_points'] or keep.mean()<PARAMS['inlier_fraction']:
        return None,dict(log,reason='insufficient_inliers')
    hull=cv2.convexHull(a);area=cv2.contourArea(hull)
    palm=np.asarray(palm,np.float32);palm_area=cv2.contourArea(cv2.convexHull(palm))
    span=max(float(np.linalg.norm(palm.max(0)-palm.min(0))),1.)
    cov=np.linalg.eigvalsh(np.cov(a.T));ratio=float(cov[0]/max(cov[1],1e-12))
    outside=max(0.,-cv2.pointPolygonTest(hull,tuple(map(float,center)),True))/span
    residual=np.linalg.norm(a@m[:,:2].T+m[:,2]-b,axis=1)
    scale=float(np.linalg.norm(m[:,0]));rotation=float(np.degrees(np.arctan2(m[1,0],m[0,0])))
    log.update(scale=scale,rotation_deg=rotation,residual_p90=float(np.quantile(residual,.9)),
        covariance_ratio=ratio,area_fraction=float(area/max(palm_area,1.)),
        extrapolation_fraction=outside,palm_span=span)
    why=None
    if not PARAMS['scale_min']<=scale<=PARAMS['scale_max']:why='scale_uncertain'
    elif abs(rotation)>PARAMS['rotation_max_deg']:why='rotation_uncertain'
    elif ratio<PARAMS['covariance_ratio'] or area<PARAMS['feature_area_fraction']*max(palm_area,1.):why='localized_texture'
    elif outside>PARAMS['extrapolation_palm_fraction']:why='center_extrapolation'
    if why:return None,dict(log,reason=why)
    nc=np.asarray(center)@m[:,:2].T+m[:,2];poly=palm@m[:,:2].T+m[:,2]
    return dict(center=nc,palm=poly),dict(log,reason='transform_candidate',predicted_center=nc.tolist())

def flow(prev,gray,points):
    p=np.asarray(points,np.float32).reshape(-1,1,2)
    args=dict(winSize=(PARAMS['win'],)*2,maxLevel=PARAMS['levels'],
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    q,st,err=cv2.calcOpticalFlowPyrLK(prev,gray,p,None,**args)
    if q is None:return None,dict(reason='flow_missing')
    back,bs,_=cv2.calcOpticalFlowPyrLK(gray,prev,q,None,**args)
    if back is None:return None,dict(reason='backward_missing')
    p=p[:,0];q=q[:,0];back=back[:,0];err=err[:,0]
    good=(st[:,0]==1)&(bs[:,0]==1)&np.isfinite(err)&(err<=PARAMS['lk_error'])
    good&=inside(q,gray.shape[1],gray.shape[0])&(np.linalg.norm(back-p,axis=1)<=PARAMS['fb_px'])
    log=dict(previous=p.tolist(),forward=q.tolist(),backward=back.tolist(),
        forward_status=st[:,0].tolist(),backward_status=bs[:,0].tolist(),errors=err.tolist(),good=good.tolist())
    if good.sum()<PARAMS['min_points'] or good.mean()<PARAMS['keep_fraction']:
        return None,dict(log,reason='flow_quality')
    return (p[good],q[good]),dict(log,reason='region_points_tracked')

def arm_anchors(hands,pose,usable,scale,w,h,suppressed):
    valid=[bool(inside(x[PALM],w,h).all()) and i not in suppressed for i,x in enumerate(hands)]
    anchors=[-1]*len(hands)
    if scale>1:
        for i,x in enumerate(hands):
            if not valid[i]:continue
            distances=[float(np.linalg.norm(x[0]-pose[wr])/scale)
                if usable[el] and usable[wr] and inside(pose[[el,wr]],w,h).all()
                else float('inf') for el,wr in [(13,15),(14,16)]]
            s=int(np.argmin(distances))
            if distances[s]<=PARAMS['pose_gate'] and distances[1-s]-distances[s]>=PARAMS['pose_margin']:anchors[i]=s
        for s in (0,1):
            ix=[i for i,a in enumerate(anchors) if a==s]
            if len(ix)>1:
                for i in ix:anchors[i]=-1
    return valid,anchors

class Tracker:
    def __init__(self):
        self.prev=None;self.time=None;self.tracks={};self.missing=[0,0];self.last_palm=[None,None]
    def step(self,gray,hands,baseline,pose,usable,time_s,measured,suppressed=()):
        if self.time is not None and time_s<=self.time:raise ValueError('noncausal timestamp')
        self.time=time_s;w,h=gray.shape[1],gray.shape[0]
        hands=np.asarray(hands);pose=np.asarray(pose)
        scale=float(np.linalg.norm(pose[11]-pose[12])) if usable[11] and usable[12] else 0.
        valid,anchors=arm_anchors(hands,pose,usable,scale,w,h,suppressed)
        # observed centers are preserved, even when identity lacks independent arm confirmation.
        out=dict(proposal=list(map(int,baseline)),anchors=anchors,
            observed_center=measured,center_estimate=[None,None],region=[None,None],
            center_used=list(measured),source=['observed' if v is not None else 'missing' for v in measured],
            seed=[False,False],seed_reason=['no_arm_observation']*2,
            recheck=[None,None],traces={},observed_finger_points_added=0)
        predictions={}
        for s,tr in list(self.tracks.items()):
            age=time_s-tr['anchor_time']
            log=dict(anchor_time=tr['anchor_time'],age=age,shoulder=scale)
            out['traces'][str(s)]=log
            if age>PARAMS['max_age_s']+1e-8 or scale<=1 or self.prev is None:
                log['reason']='expired_or_no_scale';del self.tracks[s];continue
            pair,fl=flow(self.prev,gray,tr['points']);log['flow']=fl
            if pair is None:log['reason']='flow_rejected';del self.tracks[s];continue
            p,q=pair;region=np.median(q,axis=0)
            if any(valid[i] and anchors[i]==1-s and np.linalg.norm(hands[i,PALM].mean(0)-region)<PARAMS['other_hand_gate']*scale for i in range(len(hands))):
                log['reason']='overlap_identity_uncertain';del self.tracks[s];continue
            hull=cv2.convexHull(q).reshape(-1,2)
            out['region'][s]=dict(points=q.tolist(),hull=hull.tolist(),center=region.tolist(),
                                  definition='tracked_texture_region_not_full_hand')
            center,cl=center_transform(p,q,tr['center'],tr['palm']);log['center_fit']=cl
            if center is not None:
                if cl['residual_p90']>PARAMS['residual_shoulder']*scale:cl['reason']='residual_uncertain';center=None
                elif not inside(center['center'],w,h):cl['reason']='center_outside';center=None
            if center is not None:
                log['reason']='region_and_center_estimate'
                out['center_estimate'][s]=center['center'].tolist()
                predictions[s]=center['center']
            else:log['reason']='region_only_center_uncertain'
            self.tracks[s]=dict(points=q,center=None if center is None else center['center'].tolist(),
                palm=None if center is None else center['palm'],anchor_time=tr['anchor_time'])
        for s in (0,1):
            obs=measured[s]
            if obs is None:
                self.missing[s]+=1
                if out['center_estimate'][s] is not None:
                    out['center_used'][s]=out['center_estimate'][s];out['source'][s]='estimated'
                elif out['region'][s] is not None:out['source'][s]='region_only'
                continue
            # Current evidence takes priority; no smoothing that drags it toward old texture.
            ix=[i for i,a in enumerate(anchors) if a==s and int(baseline[i])==s]
            re=dict(current_observation_priority=True,gap_frames=self.missing[s],
                arm_confirmed=len(ix)==1,
                old_estimate_discrepancy=None if s not in predictions or scale<=1
                    else float(np.linalg.norm(np.asarray(obs)-predictions[s])/scale),
                rotation_deg=None)
            out['recheck'][s]=re
            self.missing[s]=0
            prior_region_track=self.tracks.pop(s,None)  # anatomy always reset; pixels may survive current revalidation
            if len(ix)!=1:
                out['seed_reason'][s]='current_identity_unconfirmed'
                self.last_palm[s]=None
                continue
            i=ix[0];palm=hands[i,PALM]
            if self.last_palm[s] is not None:
                a=self.last_palm[s][2]-self.last_palm[s][0];b=palm[2]-palm[0]
                if np.linalg.norm(a)>1 and np.linalg.norm(b)>1:
                    re['rotation_deg']=float(np.degrees(np.arctan2(a[0]*b[1]-a[1]*b[0],np.dot(a,b))))
            self.last_palm[s]=palm.copy()
            mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,cv2.convexHull(palm.astype(np.float32)).astype(np.int32),255)
            mask=cv2.erode(mask,np.ones((3,3),np.uint8))
            pts=cv2.goodFeaturesToTrack(gray,PARAMS['max_points'],PARAMS['quality'],PARAMS['min_distance'],mask=mask)
            points=None if pts is None else pts[:,0]
            out['seed_reason'][s]='current_arm_and_palm_reanchored'
            if points is None or len(points)<PARAMS['min_points']:
                # Preserve only successfully tracked pixels still inside the CURRENT palm.
                # No prior center is carried forward. No guessed points or relaxed gates.
                hull=cv2.convexHull(palm.astype(np.float32))
                old_points=None if prior_region_track is None else prior_region_track['points']
                retained=[] if old_points is None else [p for p in old_points
                    if cv2.pointPolygonTest(hull,tuple(map(float,p)),False)>=0]
                out['retained_region_points']=out.get('retained_region_points',{})
                out['retained_region_points'][str(s)]=[np.asarray(p).tolist() for p in retained]
                if len(retained)<PARAMS['min_points']:
                    out['seed_reason'][s]='insufficient_texture';continue
                points=np.asarray(retained,np.float32)
                out['seed_reason'][s]='current_palm_revalidated_existing_pixels'
            self.tracks[s]=dict(points=points,center=list(obs),palm=palm.copy(),anchor_time=time_s)
            out['seed'][s]=True
        self.prev=gray.copy()
        return out
