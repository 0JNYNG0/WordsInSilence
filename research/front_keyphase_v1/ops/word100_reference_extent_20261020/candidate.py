"""Reference-projected evidence extent. Offline only, no sign correctness decision."""
import numpy as np
from wis_lab.core.features import body_frame,FeatureError
from wis_lab.core.dtw import _optimal_path

BUDGET=96
BAND=.2
MIN_REFERENCES=5
MIN_PATH_SUPPORT=.35
MIN_DISTINCT=8
SIDES=('L','R')
AXES=('handshape','location','movement')

def nearest(ts,target):
    ts=np.asarray(ts,dtype=float);target=np.asarray(target,dtype=float)
    right=np.clip(np.searchsorted(ts,target),0,len(ts)-1);left=np.maximum(0,right-1)
    return np.where(np.abs(ts[left]-target)<=np.abs(ts[right]-target),left,right)

def pose_sample(obs):
    """No hand21 landmarks or active_span used. Entire recording is kept."""
    ts=obs.timestamp_us
    picks=np.unique(nearest(ts,np.linspace(ts[0],ts[-1],min(BUDGET,len(ts)))))
    xy=np.full((len(ts),4,2),np.nan);valid=np.zeros((len(ts),4),bool)
    try:
        origin,scale,body=body_frame(obs)
        for j,k in enumerate((13,15,14,16)):
            valid[:,j]=body&obs.usable[:,k]&np.isfinite(obs.xy_px[:,k]).all(1)&np.isfinite(origin).all(1)
            xy[valid[:,j],j]=(obs.xy_px[valid[:,j],k]-origin[valid[:,j]])/scale
    except FeatureError:pass
    return dict(timestamp_us=np.asarray(ts[picks]),picked_rows=picks,xy=xy[picks],valid=valid[picks],full_timestamp_us=np.asarray(ts))

def align(q,r,required):
    groups=[0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]
    qa=q['xy'][:,groups];ra=r['xy'][:,groups];qv=q['valid'][:,groups];rv=r['valid'][:,groups]
    joint=qv[:,None,:]&rv[None,:,:]
    delta=np.linalg.norm(np.nan_to_num(qa[:,None]-ra[None,:]),axis=3)
    # Unknown arms never create a zero-cost match;1.5 matches existing missing-group units.
    cost=np.where(joint,delta,1.5).mean(2);support=joint.mean(2)
    path=_optimal_path(cost,BAND)
    if not path:return dict(status='UNAVAILABLE',reason='NO_PATH',path=[],support=0.)
    p=np.asarray(path,int);qi,ri=p.T;good=support[qi,ri]>=.5
    enough=(support[qi,ri].mean()>=MIN_PATH_SUPPORT and
            min(len(np.unique(qi[good])),len(np.unique(ri[good])))>=MIN_DISTINCT)
    return dict(status='AVAILABLE' if enough else 'UNAVAILABLE',reason=None if enough else 'ARM_ALIGNMENT_UNCERTAIN',
        path=p.tolist(),support=float(support[qi,ri].mean()),good=good.tolist(),cost=float(cost[qi,ri].mean()))

def project(q,refs,required):
    """At least5 distinct people must support a time label. No nearest-ref selection."""
    if len({r['signer_id'] for r in refs})!=len(refs):raise ValueError('DUPLICATE_REFERENCE_PERSON')
    n=len(q['timestamp_us']);corevotes=np.zeros(n,int);outsidevotes=np.zeros(n,int)
    expected={s:{k:np.zeros(n,int) for k in AXES} for s in SIDES}
    restvotes={s:np.zeros(n,int) for s in SIDES};pairs=[]
    for ref in refs:
        r=ref['pose'];a=align(q,r,required);pairs.append(dict(reference_clip=ref['clip_id'],reference_signer=ref['signer_id'],**a))
        if a['status']!='AVAILABLE':continue
        path=np.asarray(a['path'],int);good=np.asarray(a['good'],bool)
        for qi in range(n):
            ri=path[(path[:,0]==qi)&good,1]
            if not len(ri):continue
            core=ref['core'][ri]
            # Mixed boundary alignment is unknown, not a forced endpoint vote.
            if core.all():corevotes[qi]+=1
            elif not core.any():outsidevotes[qi]+=1
            else:continue
            for side in SIDES:
                if not core.all():continue
                rest=ref['rest'][side][ri]
                if rest.all():restvotes[side][qi]+=1
                for key in AXES:
                    ev=ref['expected'][side][key][ri]
                    if ev.all():expected[side][key][qi]+=1
    # Lifting labels to actual times copies nearest sampled labels, never coordinates.
    lift=nearest(q['timestamp_us'],q['full_timestamp_us'])
    core=(corevotes>=MIN_REFERENCES)[lift];outside=(outsidevotes>=MIN_REFERENCES)[lift]
    out=dict(schema='wis_public_reference_extent_candidate_v1',core=core,outside=outside,unknown=~(core|outside),
        core_votes=corevotes[lift],outside_votes=outsidevotes[lift],pairs=pairs,expected={},rest={},
        semantic_verdict=None,optional_sides=[s for s in SIDES if s not in required])
    for s in SIDES:
        out['rest'][s]=(restvotes[s]>=MIN_REFERENCES)[lift] & core
        out['expected'][s]={k:((expected[s][k]>=MIN_REFERENCES)[lift]&core) if s in required else np.zeros(len(lift),bool) for k in AXES}
    return out

def weights(ts):
    t=(np.asarray(ts,dtype=float)-ts[0])/1e6
    return np.diff(np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]]) if len(t)>1 else np.zeros(len(t))

def describe(ts,flags,extent,required,role_known=True):
    w=weights(ts);rows=[]
    for side in SIDES:
        for key in AXES:
            expect=extent['expected'][side][key];missing=expect&~flags[side][key];duration=float(w@expect);lost=float(w@missing)
            optional=side not in required
            status='NOT_REQUIRED' if optional else 'UNLOCATED' if not duration else 'UNOBSERVED_REFERENCE_ACTION' if lost>1e-8 else 'OBSERVED_REFERENCE_ACTION'
            rows.append(dict(side=side,key=key,status=status,expected_s=duration,missing_expected_s=lost,
                observed_expected_percent=100*(duration-lost)/duration if duration else None,
                reference_rest_s=float(w@extent['rest'][side]),core_but_unclassified_s=float(w@(extent['core']&~(expect|extent['rest'][side]))),
                role_known=role_known,wrong_action=None,whole_action_pass=None))
    return rows
