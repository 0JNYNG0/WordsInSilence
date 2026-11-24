"""Isolated metric evidence. No lexical rule, model, coordinate repair or hand exclusion."""
import numpy as np

CHAINS=((0,1,2,3,4),(0,5,6,7,8),(0,9,10,11,12),(0,13,14,15,16),(0,17,18,19,20))
TRIPLETS=tuple((c[i-1],c[i],c[i+1]) for c in CHAINS for i in (1,2,3))
ANGLE_NAMES=tuple(f'{finger}_{joint}' for finger in ('thumb','index','middle','ring','little') for joint in ('proximal','middle','distal'))
EPS=1e-6
MIN_SUPPORT=.5
MIN_FRAMES=8
MIN_REFS=5

def angles(xy,usable):
    """Each projected 2D angle needs exactly its three observed nondegenerate points."""
    p=np.asarray(xy,dtype=np.float64);ok=np.asarray(usable,bool)&np.isfinite(p).all(-1)
    out=np.full((len(p),15),np.nan)
    for j,(a,b,c) in enumerate(TRIPLETS):
        u=p[:,a]-p[:,b];v=p[:,c]-p[:,b]
        valid=ok[:,a]&ok[:,b]&ok[:,c]&(np.linalg.norm(u,axis=1)>=EPS)&(np.linalg.norm(v,axis=1)>=EPS)
        dot=(u*v).sum(1);cross=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        out[valid,j]=np.degrees(np.arctan2(np.abs(cross[valid]),dot[valid]))
    return out,np.isfinite(out)

def compared_angles(q,r,path,alignment_ok=True):
    """Frozen alignment and denominator. Partial output never implies a whole-hand score."""
    qi,ri=np.asarray(path,int).T
    values=[];supports=[];qn=[];rn=[]
    for j in range(15):
        good=np.isfinite(q[qi,j])&np.isfinite(r[ri,j]);n=int(good.sum())
        nq=len(np.unique(qi[good]));nr=len(np.unique(ri[good]));support=n/len(path)
        ready=alignment_ok and support>=MIN_SUPPORT and min(nq,nr)>=MIN_FRAMES
        values.append(float(np.abs(q[qi[good],j]-r[ri[good],j]).mean()) if ready else None)
        supports.append(support);qn.append(nq);rn.append(nr)
    return {'values':values,'support':supports,'query_frames':qn,'reference_frames':rn}

def aggregate_angles(pairs):
    vals=[];refs=[]
    for j in range(15):
        vv=[p['values'][j] for p in pairs if p['values'][j] is not None]
        refs.append(len(vv));vals.append(float(np.median(vv)) if len(vv)>=MIN_REFS else None)
    ready=sum(x is not None for x in vals)
    return {'angles':vals,'valid_reference_counts':refs,'angles_ready':ready,
            'whole_hand':float(np.mean(vals)) if ready==15 else None,
            'status':'ALL_ANGLES_AVAILABLE' if ready==15 else 'PARTIAL' if ready else 'UNAVAILABLE',
            'is_correctness_verdict':False}

def trajectory(q,r,qvalid,rvalid,path):
    """Compare identical retained pairs with two origins; never bridge a gap or alter path."""
    qi,ri=np.asarray(path,int).T;good=qvalid[qi]&rvalid[ri]
    nq=len(np.unique(qi[good]));nr=len(np.unique(ri[good]));support=float(good.mean())
    out={'support':support,'query_frames':nq,'reference_frames':nr,'old':None,'paired_start':None,'anchor_difference':None,'first_pair':None}
    if not good.any():return out
    g=np.flatnonzero(good);a=q[qi[good]].astype(float);b=r[ri[good]].astype(float)
    oq=q[np.flatnonzero(qvalid)[0]].astype(float);orr=r[np.flatnonzero(rvalid)[0]].astype(float)
    cq=q[qi[g[0]]].astype(float);cr=r[ri[g[0]]].astype(float)
    old=(a-oq)-(b-orr);new=(a-cq)-(b-cr)
    out.update(old=float(np.linalg.norm(old,axis=1).mean()*100),paired_start=float(np.linalg.norm(new,axis=1).mean()*100),
               anchor_difference=float(np.linalg.norm((oq-orr)-(cq-cr))*100),first_pair=[int(qi[g[0]]),int(ri[g[0]])])
    return out

def source_states(native,view,ts,bottom,width,origin,scale,intervals):
    """Two evidence layers: view-local masks and explicitly offline full-view diagnostics.

    Rest needs full observed hand geometry below the boundary plus a stable causal window.
    A missing hand never becomes a resting hand. Full view is not available to a cropped webcam.
    """
    n=len(ts);out={}
    core=np.zeros(n,bool)
    for start,end in intervals:core|=(ts>=start*1e6)&(ts<=end*1e6)
    for side,sl,elbow,wrist in [('L',slice(33,54),13,15),('R',slice(54,75),14,16)]:
        p=native['xy_px'][:,sl].astype(float);valid=native['usable'][:,sl]&np.isfinite(p).all(-1)
        full=valid.all(1)
        below=full&(p[:,:,1].min(1)>=bottom)
        inside=full&(p[:,:,0].min(1)>=0)&(p[:,:,0].max(1)<width)&(p[:,:,1].min(1)>=0)&(p[:,:,1].max(1)<bottom)
        rest=np.zeros(n,bool);body=np.isfinite(origin).all(1)&native['usable'][:,11]&native['usable'][:,12]
        rel=(p[:,0]-origin)/scale
        # Existing C calibration stability (.05 shoulder widths), .3 sec, 8 frames, .1 sec max gap.
        for t in range(n):
            lo=int(np.searchsorted(ts,ts[t]-300000,side='left'))
            if t-lo+1<8 or ts[t]-ts[lo]<250000:continue
            if not (below[lo:t+1]&body[lo:t+1]).all() or np.diff(ts[lo:t+1]).max()>100000:continue
            a=rel[lo:t+1];center=np.median(a,axis=0)
            rest[t]=center[1]>1.0 and np.linalg.norm(a-center,axis=1).max()<=.05
        vp=view['xy_px'][:,sl];vu=view['usable'][:,sl]&np.isfinite(vp).all(-1)
        aa,av=angles(vp,vu);anypoints=vu.any(1)
        # Arm evidence is a hint of in-frame location, never a substitute for hand geometry.
        arm=view['usable'][:,[elbow,wrist]].all(1)&np.isfinite(view['xy_px'][:,[elbow,wrist]]).all((1,2))
        pw=view['xy_px'][:,wrist]
        arm&=(pw[:,0]>=0)&(pw[:,0]<width)&(pw[:,1]>=0)&(pw[:,1]<bottom)
        gap=np.zeros(n,bool);ids=np.flatnonzero(anypoints)
        if len(ids):gap[ids[0]:ids[-1]+1]=~anypoints[ids[0]:ids[-1]+1]
        offline=np.full(n,0,np.int8) # 0 unknown/partial, 1 observed below, 2 observed inside, 3 lowered stable
        offline[below]=1;offline[inside]=2;offline[rest]=3
        out[side]={'angle_count':av.sum(1).astype(np.int8),'position_observed':vu[:,0],
                   'any_hand_points':anypoints,'arm_in_frame_hint':arm,'internal_gap':gap,
                   'offline_full_view_state':offline,'inside_source_but_view_missing':inside&~anypoints,
                   'annotated_action':core}
    return out
