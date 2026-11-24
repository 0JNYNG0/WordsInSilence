"""Research descriptors for hand / wrist / forearm relationships to facial regions.

No expression, identity, sign label, or correctness model. Preserve raw measurements
and distinguish image-plane projection from reliable shape/contact evidence.
"""
from __future__ import annotations
import numpy as np
import cv2
from .face_body_relations import region_geometry

ENDPOINTS=('thumb_tip','index_tip','middle_tip','ring_tip','little_tip','hand_wrist','palm_center','pose_wrist','elbow','forearm_nearest')
FINGERS=(4,8,12,16,20)

def build_evidence(obs,face):
    if not np.array_equal(obs.frame_id,face['frame_id']):raise ValueError('Frame mismatch')
    n=obs.n_frames
    distance=np.full((n,2,10,6),np.nan,np.float32)
    signed=np.full((n,2,10,6,2),np.nan,np.float32)
    xy=np.full((n,2,9,2),np.nan,np.float32)
    known=np.zeros((n,2,9),bool)
    centers=np.full((n,6,2),np.nan,np.float32)
    geometry=np.zeros((n,6),bool)
    projected=np.zeros((n,2,2,6),bool) # hand hull / forearm capsule
    projection_known=np.zeros((n,2,2,6),bool)
    for t in range(n):
        c,axes,radii,valid=region_geometry(face['xy_px'][t],obs.xy_px[t],obs.usable[t]);c=c[:6];radii=radii[:6];valid=valid[:6]
        centers[t]=c;geometry[t]=valid
        sw=np.linalg.norm(obs.xy_px[t,11]-obs.xy_px[t,12]) if obs.usable[t,11:13].all() else np.nan
        for side,base in enumerate((33,54)):
            mapping=list(base+np.array(FINGERS))+[base,None,15+side,13+side]
            for k,p in enumerate(mapping):
                if p is None:
                    ids=base+np.array((0,5,9,13,17))
                    if obs.usable[t,ids].all():xy[t,side,k]=obs.xy_px[t,ids].mean(axis=0);known[t,side,k]=True
                elif obs.usable[t,p] and np.isfinite(obs.xy_px[t,p]).all():
                    xy[t,side,k]=obs.xy_px[t,p];known[t,side,k]=True
            for k in range(9):
                if not known[t,side,k]:continue
                delta=(xy[t,side,k]-c)@axes.T/radii
                signed[t,side,k,valid]=delta[valid]
                distance[t,side,k,valid]=np.linalg.norm(delta[valid],axis=1)
            if known[t,side,7:9].all():
                elbow,wrist=xy[t,side,8],xy[t,side,7]
                v=wrist-elbow;length2=np.dot(v,v)
                if length2>1:
                    along=np.clip(((c-elbow)@v)/length2,0,1)
                    closest=elbow+along[:,None]*v
                    delta=(closest-c)@axes.T/radii
                    distance[t,side,9,valid]=np.linalg.norm(delta[valid],axis=1)
                    signed[t,side,9,valid]=delta[valid]
                    if np.isfinite(sw):
                        projection_known[t,side,1]=valid
                        projected[t,side,1]=valid&(np.linalg.norm(closest-c,axis=1)<=.04*sw)
            hand_ok=obs.usable[t,base:base+21]&np.isfinite(obs.xy_px[t,base:base+21]).all(axis=1)
            # All points needed to establish an absence of projection. Partial hands remain unknown.
            if hand_ok.all():
                hull=cv2.convexHull(obs.xy_px[t,base:base+21].astype(np.float32))
                for g in np.flatnonzero(valid):
                    projection_known[t,side,0,g]=True
                    projected[t,side,0,g]=cv2.pointPolygonTest(hull,tuple(c[g].astype(float)),True)>=-min(radii[g])*.25
    available=np.isfinite(distance)
    any_projection=projected.any(axis=(1,2))
    direct=available&~any_projection[:,None,None,:]
    return dict(distance=distance,signed=signed,xy=xy,known=known,centers=centers,geometry=geometry,
                projected=projected,projection_known=projection_known,available=available,
                direct_comparable=direct,frame_id=obs.frame_id.copy(),timestamp_us=obs.timestamp_us.copy())


def reference_projection_status(query_flags,query_known,ref_flags,ref_known):
    """Per side / part / face patch. Agreement is NOT correct signing.

    codes: 0=insufficient; 1=no projection in compared refs/query;
    2=projection also supported by both refs; 3=references vary;
    4=projection pattern differs from both refs (review, never automatically wrong).
    """
    if len(ref_flags)<2:raise ValueError('At least two reference executions required')
    valid=query_known&np.all(ref_known,axis=0)
    yes=np.all(ref_flags,axis=0);no=~np.any(ref_flags,axis=0)
    out=np.zeros(query_flags.shape,dtype=np.int8)
    out[valid&~yes&~no]=3
    out[valid&yes&query_flags]=2
    out[valid&no&~query_flags]=1
    out[valid&((yes&~query_flags)|(no&query_flags))]=4
    return out


def arm_motion_direction(obs,window_us=100000,min_displacement=.03):
    """Direction only from currently measured pose elbows/wrists; no inferred hidden hands.

    A symmetric +/-100 ms neighborhood is an offline diagnostic. No gap >100ms
    or invalid intermediate arm point is bridged. Speed magnitude isn't scored.
    """
    n=obs.n_frames;ids=(13,15,14,16)
    direction=np.full((n,4,2),np.nan,np.float32);good=np.zeros((n,4),bool)
    widths=np.linalg.norm(obs.xy_px[:,11]-obs.xy_px[:,12],axis=1)
    ok=obs.usable[:,11]&obs.usable[:,12]&np.isfinite(widths)&(widths>1)
    if not ok.any():return direction,good
    scale=float(np.median(widths[ok]));origin=obs.xy_px[:,[11,12]].mean(axis=1)
    body=(obs.xy_px[:,ids]-origin[:,None,:])/scale
    for t in range(n):
        a=max(0,int(np.searchsorted(obs.timestamp_us,obs.timestamp_us[t]-window_us)))
        b=min(n-1,int(np.searchsorted(obs.timestamp_us,obs.timestamp_us[t]+window_us)))
        if b<=a or np.diff(obs.timestamp_us[a:b+1]).max(initial=0)>100000:continue
        for k,p in enumerate(ids):
            if not obs.usable[a:b+1,p].all() or not ok[a:b+1].all():continue
            v=body[b,k]-body[a,k];norm=np.linalg.norm(v)
            if np.isfinite(norm) and norm>=min_displacement:
                direction[t,k]=v/norm;good[t,k]=True
    return direction,good


def motion_cost(a_dir,a_ok,b_dir,b_ok):
    both=a_ok[:,None,:]&b_ok[None,:,:]
    cosine=np.nansum(a_dir[:,None,:,:]*b_dir[None,:,:,:],axis=-1)
    d=np.clip((1-cosine)/2,0,1)
    count=both.sum(axis=-1)
    # No evidence contributes no additional cost, never an observation match verdict.
    cost=np.divide(np.where(both,d,0).sum(axis=-1),count,out=np.zeros(count.shape,float),where=count>0)
    return cost,count
