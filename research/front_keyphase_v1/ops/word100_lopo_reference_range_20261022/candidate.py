"""Leave-person-out descriptive envelopes; no semantic pass/fail or model edits."""
import numpy as np
MIN_PEOPLE=5
EPS=1e-6

def channels(e):
    pos=np.concatenate([e['centre'],e['arm_xy'].reshape(len(e['core']),2,4)],axis=-1)
    pv=np.concatenate([np.repeat(e['palm_valid'][:,:,None],2,axis=-1),np.repeat(e['arm_valid'],2,axis=-1)],axis=-1)
    rad=np.concatenate([np.repeat(e['radius'][:,:,None],2,axis=-1),np.zeros((len(e['core']),2,4))],axis=-1)
    lo=np.concatenate([e['angle_low'],pos-rad],axis=-1)
    hi=np.concatenate([e['angle_high'],pos+rad],axis=-1)
    valid=np.concatenate([e['angle_valid'],pv],axis=-1)&np.isfinite(lo)&np.isfinite(hi)
    return dict(lo=np.where(valid,lo,np.nan),hi=np.where(valid,hi,np.nan),valid=valid)

def project(c,path,n,neighbors):
    shape=(n,2,21);lo=np.full(shape,np.nan);hi=lo.copy();complete=np.zeros(shape,bool);partial=np.zeros(shape,bool)
    for j in range(n):
        ids=np.unique(path[path[:,1]==j,0]);vv=c['valid'][ids];complete[j]=vv.all(0);partial[j]=vv.any(0)&~complete[j]
        for h,a in np.argwhere(vv.any(0)):
            good=set(int(i) for i in ids if c['valid'][i,h,a])
            if neighbors:
                for i in list(good):
                    for k in (i-1,i+1):
                        if 0<=k<len(c['valid']) and c['valid'][k,h,a]:good.add(k)
            ix=sorted(good);lo[j,h,a]=c['lo'][ix,h,a].min();hi[j,h,a]=c['hi'][ix,h,a].max()
    return dict(lo=lo,hi=hi,complete=complete,partial=partial)

def pool(projected):
    valid=np.stack([p['complete'] for p in projected]);lo=np.stack([p['lo'] for p in projected]);hi=np.stack([p['hi'] for p in projected])
    count=valid.sum(0);support=count>=MIN_PEOPLE
    low=np.min(np.where(valid,lo,np.inf),axis=0);high=np.max(np.where(valid,hi,-np.inf),axis=0)
    return dict(low=np.where(support,low,np.nan),high=np.where(support,high,np.nan),count=count,support=support,
                member_low=lo,member_high=hi,member_complete=valid)

def persistent(mask):
    out=np.zeros_like(mask)
    for h in range(2):
        for a in range(21):
            start=None
            for j in range(len(mask)+1):
                hit=j<len(mask) and mask[j,h,a]
                if hit and start is None:start=j
                if not hit and start is not None:
                    if j-start>=3:out[start:j,h,a]=True
                    start=None
    return out

def compare(query,anchor,template,path,expected):
    p=project(query,path,len(expected),True)
    comparable=expected&template['support']&p['complete']
    gap=np.maximum(np.maximum(template['low']-p['hi'],p['lo']-template['high']),0.)
    gap=np.where(comparable,gap,np.nan);mismatch=comparable&(gap>EPS)
    base_comparable=expected&anchor['valid']&p['complete']
    base_gap=np.maximum(np.maximum(anchor['lo']-p['hi'],p['lo']-anchor['hi']),0.)
    base_gap=np.where(base_comparable,base_gap,np.nan);base_bad=base_comparable&(base_gap>EPS)
    return dict(expected=expected.copy(),comparable=comparable,unknown=expected&~comparable,
                reference_insufficient=expected&~template['support'],query_incomplete=expected&~p['complete'],query_partial=expected&p['partial'],
                gap=gap,mismatch=mismatch,persistent=persistent(mismatch),within_range=comparable&~mismatch,
                base_comparable=base_comparable,base_gap=base_gap,base_mismatch=base_bad,base_persistent=persistent(base_bad),
                query_low=p['lo'],query_high=p['hi'],query_complete=p['complete'])
