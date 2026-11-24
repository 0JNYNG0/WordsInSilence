"""Coherent named-reference local descriptors. Research sidecar only."""
import numpy as np
EPS=1e-6
MIN_PEERS=5
RADIUS=2
GROUPS={'angle':slice(0,15),'palm':slice(15,17),'elbow':slice(17,19),'wrist':slice(19,21)}

def channels(e):
    pos=np.concatenate([e['centre'],e['arm_xy'].reshape(-1,2,4)],2)
    rad=np.concatenate([np.repeat(e['radius'][:,:,None],2,2),np.zeros_like(pos[:,:,2:])],2)
    pv=np.concatenate([np.repeat(e['palm_valid'][:,:,None],2,2),np.repeat(e['arm_valid'],2,2)],2)
    lo=np.concatenate([e['angle_low'],pos-rad],2);hi=np.concatenate([e['angle_high'],pos+rad],2)
    valid=np.concatenate([e['angle_valid'],pv],2)&np.isfinite(lo)&np.isfinite(hi)
    return {'lo':np.where(valid,lo,np.nan),'hi':np.where(valid,hi,np.nan),'valid':valid}

def expected(e,req):
    hand=e['core'][:,None]&np.array([s in req for s in ('L','R')])[None,:]&~e['source_rest']
    return np.broadcast_to(hand[:,:,None],(len(hand),2,21)).copy()

def resample(c,ix):
    a=np.floor(ix).astype(int);b=np.ceil(ix).astype(int);w=(ix-a)[:,None,None]
    ok=c['valid'][a]&c['valid'][b]
    return {key:np.where(ok,c[key][a]*(1-w)+c[key][b]*w,np.nan) for key in ('lo','hi')}|{'valid':ok}

def mapped(c,path,n,rest):
    """One chronological sample per reference time, never an interval hull of frames."""
    ids=[path[path[:,1]==j,0] for j in range(n)]
    assert all(len(x)>0 for x in ids)
    mid=np.array([np.median(x) for x in ids]);assert np.all(np.diff(mid)>=0)
    out=resample(c,mid)
    out.update(first=np.array([x.min() for x in ids]),last=np.array([x.max() for x in ids]),index=mid,source_valid=c['valid'],source_rest=rest)
    return out

def descriptor(m,domain):
    n=len(domain);shape=(n,2,21)
    eligible=np.zeros(shape,bool);valid=np.zeros(shape,bool);span_bad=np.zeros(shape,bool);rest_bad=np.zeros(shape,bool)
    lo=np.full(shape+(5,),np.nan);hi=lo.copy();phase_index=np.full((n,3),np.nan);span=np.full((n,2),-1,int)
    for j in range(RADIUS,n-RADIUS):
        eligible[j]=domain[j-RADIUS:j+RADIUS+1].all(0)
        start,end=m['first'][j-RADIUS],m['last'][j+RADIUS]
        vb=m['source_valid'][start:end+1].all(0)
        rb=m['source_rest'][start:end+1].any(0)[:,None]
        span_bad[j]=eligible[j]&~vb;rest_bad[j]=eligible[j]&rb
        ix=[j-RADIUS,j,j+RADIUS]
        phase_index[j]=m['index'][ix];span[j]=[start,end]
        good=eligible[j]&vb&~rb&m['valid'][ix].all(0)
        l=np.moveaxis(m['lo'][ix],0,-1);h=np.moveaxis(m['hi'][ix],0,-1)
        dl=np.stack([l[...,1]-h[...,0],l[...,2]-h[...,1]],-1)
        dh=np.stack([h[...,1]-l[...,0],h[...,2]-l[...,1]],-1)
        lo[j]=np.concatenate([l,dl],-1);hi[j]=np.concatenate([h,dh],-1);valid[j]=good
    lo=np.where(valid[...,None],lo,np.nan);hi=np.where(valid[...,None],hi,np.nan)
    collapsed=np.isfinite(phase_index).all(1)&(np.diff(phase_index,axis=1)==0).any(1)
    return dict(lo=lo,hi=hi,valid=valid,eligible=eligible,span_bad=span_bad,rest_bad=rest_bad,phase_index=phase_index,span=span,collapsed=collapsed)

def distance(a,b,mode):
    gap=np.maximum(np.maximum(a['lo']-b['hi'],b['lo']-a['hi']),0.)
    sl=slice(0,3) if mode=='levels' else slice(0,5)
    return np.max(gap[...,sl],-1)

def calibrate(anchor,peers):
    out={}
    mask=np.stack([p['valid']&anchor['valid'] for p in peers]);count=mask.sum(0);support=count>=MIN_PEERS
    for mode in ('levels','sequence'):
        d=np.stack([distance(p,anchor,mode) for p in peers])
        threshold=np.max(np.where(mask,d,-np.inf),0)
        out[mode+'_threshold']=np.where(support,threshold,np.nan)
    return out|{'count':count,'support':support}

def compare(query,anchor,cal,domain):
    ok=domain&query['valid']&anchor['valid']&cal['support'];out={}
    for mode in ('levels','sequence'):
        d=distance(query,anchor,mode);excess=d-cal[mode+'_threshold']
        bad=ok&(excess>EPS)
        out.update({mode+'_distance':np.where(ok,d,np.nan),mode+'_bad':bad,mode+'_within':ok&~bad,mode+'_excess':np.where(ok,excess,np.nan)})
    return out|{'expected':domain.copy(),'comparable':ok,'unknown':domain&~ok,
       'context_unavailable':domain&~anchor['eligible'],'anchor_incomplete':domain&anchor['eligible']&~anchor['valid'],
       'query_incomplete':domain&query['eligible']&~query['valid'],'peer_insufficient':domain&~cal['support'],
       'query_span_bad':query['span_bad'],'query_rest_bad':query['rest_bad']}
