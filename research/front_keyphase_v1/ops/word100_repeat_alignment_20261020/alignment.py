"""Offline one/multiple-performance alignment; no classifier or sign verdict."""
import numpy as np
from numba import njit
from wis_lab.core.features import body_frame,FeatureError
SIDES=('L','R');AXES=('handshape','location','movement');CYCLES=(1,2,3)
MIN_REFS=5;BAND=.2;MIN_SUPPORT=.35;MIN_DISTINCT=8

@njit(cache=False)
def dp(cost):
    n,m=cost.shape;band=max(1,int(round(BAND*max(n,m))))
    acc=np.full((n+1,m+1),np.inf);back=np.full((n+1,m+1),-1,np.int8);acc[0,0]=0.
    for i in range(1,n+1):
        centre=(i-1)*(m-1)/(n-1)
        lo=max(1,int(np.floor(centre-band))+1);hi=min(m,int(np.ceil(centre+band))+1)
        for j in range(lo,hi+1):
            a,b,c=acc[i-1,j-1],acc[i-1,j],acc[i,j-1]
            k=0;v=a
            if b<v:k=1;v=b
            if c<v:k=2;v=c
            if np.isfinite(v):acc[i,j]=v+cost[i-1,j-1];back[i,j]=k
    path=[];i=n;j=m
    while i>0 and j>0 and back[i,j]>=0:
        path.append((i-1,j-1));k=back[i,j]
        if k==0:i-=1;j-=1
        elif k==1:i-=1
        else:j-=1
    return path[::-1],acc[n,m]

def full_pose(obs):
    n=obs.n_frames;xy=np.full((n,4,2),np.nan);valid=np.zeros((n,4),bool)
    try:
        origin,scale,body=body_frame(obs)
        for j,k in enumerate((13,15,14,16)):
            ok=body&obs.usable[:,k]&np.isfinite(obs.xy_px[:,k]).all(1)&np.isfinite(origin).all(1)
            xy[ok,j]=(obs.xy_px[ok,k]-origin[ok])/scale;valid[:,j]=ok
    except FeatureError:pass
    return dict(xy=xy,valid=valid,ts=obs.timestamp_us)

def costs(q,r,required):
    groups=[0,1] if required==['L'] else [2,3] if required==['R'] else [0,1,2,3]
    qa=q['xy'][:,groups];ra=r['xy'][:,groups];joint=q['valid'][:,None,groups]&r['valid'][None,:,groups]
    distance=np.linalg.norm(np.nan_to_num(qa[:,None]-ra[None,:]),axis=3)
    return np.where(joint,distance,1.5).mean(2),joint.mean(2)

def align_all(q,ref,required):
    cost,support=costs(q,ref['pose'],required);m=cost.shape[1];out=[]
    for cycles in CYCLES:
        repeated=np.tile(cost,(1,cycles));path,total=dp(repeated)
        p=np.asarray(path,int);qi,ri=p.T;local=ri%m;ev=support[qi,local];good=ev>=.5
        # Each whole reference traversal needs actual supported core evidence.
        supported=[]
        for c in range(cycles):
            sel=good&(ri//m==c)&ref['core'][local]
            supported.append(min(len(np.unique(qi[sel])),len(np.unique(local[sel]))))
        valid=bool(ev.mean()>=MIN_SUPPORT and min(supported)>=MIN_DISTINCT)
        terms=cost[qi,local];qcost=np.bincount(qi,terms,minlength=len(q['ts']))/np.bincount(qi,minlength=len(q['ts']))
        out.append(dict(cycles=cycles,path=p,good=good,support=float(ev.mean()),supported_core_distinct=supported,
            score=float(qcost.mean()),accumulated_cost=float(total),available=valid))
    return out

def votes(q,refs,bank,k,required):
    n=len(q['ts']);cv=np.zeros(n,int);av=np.zeros(n,int);ov=np.zeros(n,int)
    rest={s:np.zeros(n,int) for s in SIDES}
    strong={s:{a:np.zeros(n,int) for a in AXES} for s in SIDES};possible={s:{a:np.zeros(n,int) for a in AXES} for s in SIDES}
    available=0
    for ref,entries in zip(refs,bank):
        pair=entries[k-1]
        if not pair['available']:continue
        available+=1;p=pair['path'];good=pair['good'];m=len(ref['core'])
        for qi in range(n):
            ri=p[(p[:,0]==qi)&good,1]%m
            if not len(ri):continue
            core=ref['core'][ri]
            cv[qi]+=int(core.all());av[qi]+=int(core.any());ov[qi]+=int(not core.any())
            for s in SIDES:
                rest[s][qi]+=int(core.all() and ref['rest'][s][ri].all())
                for a in AXES:
                    e=ref['expected'][s][a][ri]
                    strong[s][a][qi]+=int(e.all());possible[s][a][qi]+=int(e.any())
    enough=available>=MIN_REFS
    core=(cv>=MIN_REFS)&enough;maybe=(av>0)&enough
    outside=(ov>=MIN_REFS)&~maybe&enough
    result=dict(core=core,possible_core=maybe,outside=outside,unknown=~(core|outside),available=available,
        core_votes=cv,possible_core_votes=av,outside_votes=ov,expected={},possible_expected={},rest={},
        semantic_verdict=None,whole_action_pass=None)
    for s in SIDES:
        result['rest'][s]=(rest[s]>=MIN_REFS)&core
        result['expected'][s]={a:(strong[s][a]>=MIN_REFS)&core&~result['rest'][s] if s in required else np.zeros(n,bool) for a in AXES}
        result['possible_expected'][s]={a:(possible[s][a]>0)&maybe&~result['rest'][s] if s in required else np.zeros(n,bool) for a in AXES}
    return result

def project(q,refs,required):
    assert len({r['signer_id'] for r in refs})==len(refs)
    bank=[align_all(q,r,required) for r in refs]
    scores=[]
    for k in CYCLES:
        # Include all reference costs, never select the easiest subset for model choice.
        valid=sum(b[k-1]['available'] for b in bank)
        score=float(np.median([b[k-1]['score'] for b in bank])) if valid>=MIN_REFS else None
        scores.append(dict(cycles=k,score=score,valid_refs=valid))
    options=[(r['score'],r['cycles']) for r in scores if r['score'] is not None]
    chosen=min(options)[1] if options else 1
    multi=votes(q,refs,bank,chosen,required);single=multi if chosen==1 else votes(q,refs,bank,1,required)
    return dict(chosen_cycles=chosen,cycle_scores=scores,multi=multi,single=single,bank=bank)

def weights(ts):
    t=(np.asarray(ts,float)-ts[0])/1e6
    return np.diff(np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]]) if len(t)>1 else np.zeros(len(t))

def describe(ts,flags,p,required):
    w=weights(ts);out=[]
    for side in SIDES:
        for key in AXES:
            strong=p['expected'][side][key];possible=p['possible_expected'][side][key]
            lost=strong&~flags[side][key];uncertain=possible&~strong&~flags[side][key]
            status='NOT_REQUIRED' if side not in required else 'UNLOCATED' if not strong.any() else 'MISSING_SUPPORTED_INTERVAL' if lost.any() else 'UNCERTAIN_MISSING_INTERVAL' if uncertain.any() else 'OBSERVED_SUPPORTED_INTERVAL_ONLY'
            out.append(dict(side=side,key=key,status=status,expected_s=float(w@strong),possible_s=float(w@possible),
                missing_s=float(w@lost),uncertain_missing_s=float(w@uncertain),
                unclassified_s=float(w@(p['possible_core']&~(possible|p['rest'][side]))),
                wrong_action=None,whole_action_pass=None))
    return out
