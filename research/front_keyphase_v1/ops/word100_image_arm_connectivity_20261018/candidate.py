"""Research-only, single-frame image connectivity evidence.

No learned appearance identity, fixed skin-color classifier, word IDs, future
frames, or out-of-sensor pixels. Connected color pixels are supporting evidence,
not anatomical ground truth. All original Hand coordinates remain untouched.
"""
import cv2,numpy as np
from collections import deque

PARAMS={'max_width':640,'chroma_radius':20.,'patch_radius_shoulders':.02,'connectivity':8,'max_path_chebyshev_ratio':1.5,'pose_min':.5}
OFFSETS=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def shortest(mask,start,end):
    if not mask[start[1],start[0]] or not mask[end[1],end[0]]:return []
    if start==end:return [start]
    max_len=int(np.floor(PARAMS['max_path_chebyshev_ratio']*max(abs(start[0]-end[0]),abs(start[1]-end[1]))))
    queue=deque([(start,0)]);parent={start:None};h,w=mask.shape
    while queue:
        (x,y),dist=queue.popleft()
        if dist>=max_len:continue
        for dx,dy in OFFSETS:
            q=(x+dx,y+dy)
            if q in parent or not (0<=q[0]<w and 0<=q[1]<h) or not mask[q[1],q[0]]:continue
            parent[q]=(x,y)
            if q==end:
                path=[q]
                while parent[path[-1]] is not None:path.append(parent[path[-1]])
                return path[::-1]
            queue.append((q,dist+1))
    return []

def pixel_evidence(image,elbows,wrists,scale):
    """Coordinates in resized cropped image; None means unavailable endpoint."""
    h,w=image.shape[:2];lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)[:,:,1:].astype(np.float32)
    radius=max(1,int(round(scale*PARAMS['patch_radius_shoulders'])))
    centers=[]
    for pt in elbows+wrists:
        if pt is None:continue
        x,y=pt;patch=lab[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
        centers.append(np.median(patch.reshape(-1,2),axis=0))
    support=np.zeros((h,w),bool)
    for center in centers:support|=((lab-center)**2).sum(2)<=PARAMS['chroma_radius']**2
    _,cc=cv2.connectedComponents(support.astype(np.uint8),connectivity=8)
    get=lambda p:0 if p is None else int(cc[p[1],p[0]])
    ec=[get(p) for p in elbows];wc=[get(p) for p in wrists];proposed=[];why=[];paths=[]
    for i,pt in enumerate(wrists):
        comp=wc[i]
        if pt is None or comp==0:proposed.append(-1);why.append('unobserved_or_no_color_support');paths.append([]);continue
        sides=[s for s,q in enumerate(ec) if q==comp and q!=0]
        if len(sides)!=1:proposed.append(-1);why.append('both_arms_connected' if len(sides)>1 else 'no_elbow_connected');paths.append([]);continue
        if wc.count(comp)!=1:proposed.append(-1);why.append('multiple_hand_candidates_connected');paths.append([]);continue
        s=sides[0];path=shortest(support,elbows[s],pt)
        if not path:proposed.append(-1);why.append('path_not_direct_enough');paths.append([]);continue
        proposed.append(s);why.append('single_visible_arm_path');paths.append(path)
    return {'rgb_side':proposed,'pixel_reason':why,'paths':paths,'support':support,'components':cc,'elbow_components':ec,'wrist_components':wc,'centers_ab':[x.tolist() for x in centers]}

def propose(frame,width,height,pose,pose_usable,hands,point_mask,sides,identity):
    # Crop BEFORE resize, color conversion, seed collection, or graph traversal.
    assert 0<height<=frame.shape[0] and 0<width<=frame.shape[1]
    source=np.ascontiguousarray(frame[:height,:width]);w=min(width,PARAMS['max_width']);h=max(1,int(round(height*w/width)))
    image=cv2.resize(source,(w,h),interpolation=cv2.INTER_AREA);sx=w/width;sy=h/height
    p=np.asarray(pose,float);pm=np.asarray(pose_usable,bool);hh=np.asarray(hands,float);mask=np.asarray(point_mask,bool);n=len(hh)
    def inside(q):return np.isfinite(q).all() and 0<=q[0]<width and 0<=q[1]<height
    def pix(q):return (min(w-1,int(round(q[0]*sx))),min(h-1,int(round(q[1]*sy))))
    allowed=all(pm[k] and inside(p[k]) for k in (11,12,13,14))
    elbows=[pix(p[k]) if allowed else None for k in (13,14)]
    wrists=[pix(q[0]) if mask[i,0] and inside(q[0]) else None for i,q in enumerate(hh)]
    scale=float(np.linalg.norm(p[11]-p[12]))*sx if allowed else 0.
    if allowed and scale>1e-6:e=pixel_evidence(image,elbows,wrists,scale)
    else:e={'rgb_side':[-1]*n,'pixel_reason':['arm_anchors_unusable']*n,'paths':[[] for _ in range(n)],'support':np.zeros((h,w),bool),'components':np.zeros((h,w),np.int32),'elbow_components':[0,0],'wrist_components':[0]*n,'centers_ab':[]}
    old=[int(x) for x in sides];ids=[int(x) for x in identity];out=old.copy();reason=[]
    for i,s in enumerate(e['rgb_side']):
        if ids[i]>=0:reason.append('keep_previous_confirmed');continue
        if old[i]<0:reason.append('keep_previous_withheld');continue
        if s<0:reason.append(e['pixel_reason'][i]);continue
        if any(j!=i and old[j]==s for j in range(n)):reason.append('destination_occupied');continue
        out[i]=s;reason.append('rgb_same_hint' if s==old[i] else 'rgb_changed_hint')
    return {**e,'sides':out,'reason':reason,'elbows':elbows,'wrists':wrists,'image':image,'scale':scale,'raw_points_modified':False}
