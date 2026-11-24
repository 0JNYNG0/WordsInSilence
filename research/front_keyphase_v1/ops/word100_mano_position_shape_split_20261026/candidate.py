"""Offline public research: region / arm / wrist / finger evidence stay separate.

Region correspondences are image features, never reconstructed joint coordinates.
No class, signer, target-word, or camera-side heuristics enter these functions.
"""
import cv2
import numpy as np

PARAMS=dict(width=640, window_s=.5, max_points=32, min_points=6,
            quality=.01, min_distance=3, win=21, levels=3, fb_px=1.5,
            lk_error=20., keep_fraction=.60, residual_shoulder=.10,
            match_shoulder=.12, arm_gate=.25, arm_margin=.10)
PALM=[0,5,9,13,17]

def hull(xy):
    p=np.asarray(xy,np.float32)
    if len(p)<3 or not np.isfinite(p).all():return None
    h=cv2.convexHull(p)
    return h if cv2.contourArea(h)>1 else None

def seed_points(gray,hand,other,ratio):
    h=hull(hand*ratio)
    if h is None:return np.empty((0,2),np.float32)
    mask=np.zeros_like(gray);cv2.fillConvexPoly(mask,h.astype(np.int32),255)
    # Do not seed pixels shared with the other observed hand's projection.
    oh=hull(other*ratio) if other is not None else None
    if oh is not None:cv2.fillConvexPoly(mask,oh.astype(np.int32),0)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8))
    p=cv2.goodFeaturesToTrack(gray,PARAMS['max_points'],PARAMS['quality'],PARAMS['min_distance'],mask=mask)
    return np.empty((0,2),np.float32) if p is None else p[:,0]

def region_flow(gray,start,end,hand,other,scale,ratio=1/3):
    p=seed_points(gray[start],hand,other,ratio);initial=p.copy();n=len(p)
    if n<PARAMS['min_points']:
        return dict(anchor=start,target=end,initial=initial,xy=initial/ratio,valid=np.zeros(n,bool),steps=[],qualified=False,reason='few_initial_region_features')
    valid=np.ones(n,bool);steps=[];direction=1 if end>start else -1
    args=dict(winSize=(PARAMS['win'],)*2,maxLevel=PARAMS['levels'],criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
    failure=None
    for t in range(start,end,direction):
        q,st,e=cv2.calcOpticalFlowPyrLK(gray[t],gray[t+direction],p[:,None],None,**args)
        if q is None:failure='flow_unavailable';break
        back,bst,_=cv2.calcOpticalFlowPyrLK(gray[t+direction],gray[t],q,None,**args)
        if back is None:failure='reverse_flow_unavailable';break
        q=q[:,0];back=back[:,0];st=st[:,0];bst=bst[:,0];e=e[:,0]
        good=(st==1)&(bst==1)&np.isfinite(q).all(-1)&np.isfinite(e)&(e<=PARAMS['lk_error'])&(np.linalg.norm(back-p,axis=-1)<=PARAMS['fb_px'])
        good&=(q[:,0]>=0)&(q[:,0]<gray[t].shape[1])&(q[:,1]>=0)&(q[:,1]<gray[t].shape[0])
        valid &= good
        shift=q-p;med=np.median(shift[valid],axis=0) if valid.any() else np.zeros(2)
        residual=float(np.percentile(np.linalg.norm(shift[valid]-med,axis=-1),90)/(scale*ratio)) if valid.any() and scale>1 else None
        qualified=int(valid.sum())>=PARAMS['min_points'] and float(valid.mean())>=PARAMS['keep_fraction'] and residual is not None and residual<=PARAMS['residual_shoulder']
        steps.append(dict(frame_from=t,frame_to=t+direction,previous=p.copy(),forward=q.copy(),backward=back.copy(),status=st.copy(),back_status=bst.copy(),error=e.copy(),valid=valid.copy(),residual=residual,qualified=bool(qualified)))
        p=np.where(np.isfinite(q),q,p)
        if not qualified:failure='region_support_lost';break
    return dict(anchor=start,target=end,initial=initial,xy=p/ratio,valid=valid,steps=steps,qualified=failure is None,reason=failure or 'tracked_region_not_joint')

def nearest_anchors(t,anchors,times):
    result={}
    for s in (0,1):
        b=[i for i in range(t) if anchors[i,s] and 0<times[t]-times[i]<=PARAMS['window_s']+1e-8]
        a=[i for i in range(t+1,len(times)) if anchors[i,s] and 0<times[i]-times[t]<=PARAMS['window_s']+1e-8]
        result[s]=([max(b)] if b else [])+([min(a)] if a else [])
    return result

def assess(hand,pose,pv,scale,tracks,contour):
    arm_distances=[float(np.linalg.norm(hand[0]-pose[15+s])/scale) if scale>1 and pv[[11+s,13+s,15+s]].all() and np.isfinite(hand[0]).all() else None for s in (0,1)]
    arm=-1
    if all(d is not None for d in arm_distances):
        s=int(np.argmin(arm_distances))
        if arm_distances[s]<=PARAMS['arm_gate'] and arm_distances[1-s]-arm_distances[s]>=PARAMS['arm_margin']:arm=s
    h=hull(hand);regions=[]
    for s in (0,1):
        details=[]
        for tr in tracks[s]:
            distances=[max(0.,-cv2.pointPolygonTest(h,tuple(map(float,p)),True))/scale for p in tr['xy'][tr['valid']]] if h is not None and scale>1 else []
            cost=float(np.median(distances)) if distances else None
            ok=tr['qualified'] and cost is not None and cost<=PARAMS['match_shoulder']
            details.append(dict(anchor=tr['anchor'],target=tr['target'],initial_points=len(tr['initial']),surviving_points=int(tr['valid'].sum()),flow_qualified=tr['qualified'],reason=tr['reason'],distance_to_region=cost,region_match=bool(ok)))
        both=len(details)==2 and details[0]['anchor']<details[0]['target']<details[1]['anchor']
        regions.append(dict(side=s,both=both,qualified=bool(both and all(x['region_match'] for x in details)),details=details))
    supported=[s for s in (0,1) if regions[s]['qualified']]
    region_side=supported[0] if len(supported)==1 else -1
    cs=[s for s in (0,1) if contour[s]['pass']];contour_side=cs[0] if len(cs)==1 else -1
    # A single contour is not enough; same-frame arm and clean bilateral image
    # support must agree. Either competing region or conflicting contour vetoes.
    identity=arm if arm>=0 and region_side==arm and contour_side in [-1,arm] else -1
    return dict(arm_side=arm,arm_distances=arm_distances,contour_side=contour_side,
                contour=contour,region_side=region_side,regions=regions,identity_proposal=identity,
                position_evidence=bool(arm>=0 or contour_side>=0 or any(x['qualified'] for x in regions)),
                precise_palm_center_confirmed=False,finger_shape_confirmed=False,
                coordinates_changed=False,region_is_not_anatomical_ground_truth=True)
