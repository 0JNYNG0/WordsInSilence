"""Full-source current-frame hand union. No labels/word IDs/ROI are inputs."""
import numpy as np
PALM=[0,5,9,13,17]
PARAMS=dict(arm_gate=.25,arm_margin=.10,point_agreement=.08,
            duplicate_median=.10,duplicate_wrist=.20,
            ambiguous_overlap=.50,ambiguous_wrist=.30,
            crop_half_shoulder=.60,detector_confidence=.5)

def inside(p,w,h):
    p=np.asarray(p)
    return np.isfinite(p).all(-1)&(p[...,0]>=0)&(p[...,0]<w)&(p[...,1]>=0)&(p[...,1]<h)

def context(pose,usable,w,h):
    good=np.asarray(usable)&inside(pose,w,h)
    scale=float(np.linalg.norm(pose[11]-pose[12])) if good[[11,12]].all() else 0.
    return good,scale

def arm(points,pose,good,scale):
    if scale<=1 or not np.isfinite(points[0]).all():return -1
    d=[float(np.linalg.norm(points[0]-pose[k])/scale) if good[[j,k]].all() else np.inf
       for j,k in [(13,15),(14,16)]]
    s=int(np.argmin(d))
    return s if d[s]<=PARAMS['arm_gate'] and d[1-s]-d[s]>=PARAMS['arm_margin'] else -1

def overlap(a,b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():return 0.
    lo=np.maximum(a.min(0),b.min(0));hi=np.minimum(a.max(0),b.max(0))
    denominator=min(float(np.prod(a.max(0)-a.min(0))),float(np.prod(b.max(0)-b.min(0))))
    return float(np.prod(np.maximum(hi-lo,0))/denominator) if denominator>0 else 0.

def relation(a,b,aside,bside,scale):
    """Different current arms can cross; box overlap alone never merges them."""
    if scale<=1:return 'unknown'
    if aside>=0 and bside>=0 and aside!=bside:return 'different_current_arms'
    if not np.isfinite(a[PALM]).all() or not np.isfinite(b[PALM]).all():return 'unknown'
    wrist=float(np.linalg.norm(a[0]-b[0])/scale)
    median=float(np.median(np.linalg.norm(a[PALM]-b[PALM],axis=1))/scale)
    if wrist<=PARAMS['duplicate_wrist'] and median<=PARAMS['duplicate_median']:
        return 'same_geometry'
    if aside>=0 and aside==bside and overlap(a,b)>.25:
        return 'same_current_arm_overlap'
    if overlap(a,b)>=PARAMS['ambiguous_overlap'] and wrist<=PARAMS['ambiguous_wrist']:
        return 'ambiguous_overlap'
    return 'separate'

def crop_boxes(pose,usable,base_sides,w,h):
    good,scale=context(pose,usable,w,h);out={}
    if scale<=1:return out
    for s,(j,k) in enumerate([(13,15),(14,16)]):
        if s in base_sides or not good[[j,k]].all():continue
        x,y=pose[k];d=scale*PARAMS['crop_half_shoulder']
        b=[max(0,int(np.floor(x-d))),max(0,int(np.floor(y-d))),
           min(w,int(np.ceil(x+d))),min(h,int(np.ceil(y+d)))]
        if min(b[2]-b[0],b[3]-b[1])>=32:out[s]=b
    return out

def merge(full_detections,crop_detections,base_hands,base_sides,pose,usable,w,h):
    """Keep baseline. Confirm its physical hand or propose only genuinely new side.

    New fingers require same-frame full/crop agreement, not copied/interpolated
    from another time. Agreement is a quality proxy, not anatomical truth.
    """
    pose=np.asarray(pose);good,scale=context(pose,usable,w,h)
    old=[np.asarray(p) for p in base_hands]
    old_arms=[arm(p,pose,good,scale) for p in old]
    checks=[]
    for i,d in enumerate(full_detections):
        p=np.asarray(d['xy']);s=arm(p,pose,good,scale) if p.shape==(21,2) else -1
        q=dict(index=i,side=s,accepted=False,reason='current_arm_uncertain',
               matched_old=[],relations=[],joint_supported=[False]*21,crop_index=None)
        if s<0:
            checks.append(q);continue
        if not inside(p[PALM],w,h).all():
            q['reason']='palm_outside_source';checks.append(q);continue
        rel=[relation(p,b,s,bs,scale) for b,bs in zip(old,old_arms)]
        q['relations']=rel
        matches=[j for j,x in enumerate(rel) if x.startswith('same_')]
        q['matched_old']=matches
        if matches:
            q['reason']=('existing_identity_conflict' if any(base_sides[j]!=s for j in matches)
                         else 'confirm_existing_hand')
        elif 'ambiguous_overlap' in rel:
            q['reason']='unresolved_existing_overlap'
        elif s in base_sides:
            q['reason']='existing_side_preserved'
        else:
            corroborated=[]
            for j,c in enumerate(crop_detections):
                b=np.asarray(c['xy']);box=np.asarray(c['box'])
                if c['requested_side']!=s or arm(b,pose,good,scale)!=s:continue
                valid=inside(p,w,h)&inside(b,w,h)&inside(b-box[:2],box[2]-box[0],box[3]-box[1])
                valid &= np.linalg.norm(p-b,axis=1)<=PARAMS['point_agreement']*scale
                if valid[PALM].all():corroborated.append((j,valid))
            if len(corroborated)==1:
                j,valid=corroborated[0]
                q.update(accepted=True,reason='new_hand_current_arm_and_two_image_scales',
                         crop_index=j,joint_supported=valid.tolist())
            else:q['reason']='image_scales_disagree_or_ambiguous'
        checks.append(q)
    for s in (0,1):
        tied=[q for q in checks if q['accepted'] and q['side']==s]
        if len(tied)>1:
            for q in tied:q.update(accepted=False,reason='multiple_new_same_side')
    return checks
