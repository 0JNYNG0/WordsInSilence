"""Pure fixed-rate mapping and conservative inherited extent-label arithmetic."""
import numpy as np

def rate_mapping(timestamp_us,rate):
    """Nearest original observation on a uniform source-time grid; no pose interpolation.

    Rate>1 is faster. Query duration is original duration/rate. Choose interval
    count nearest to duration/(median cadence*rate), retaining both endpoints.
    Exact nearest-time ties choose the earlier observation. Labels never affect it.
    """
    ts=np.asarray(timestamp_us,np.int64)
    assert len(ts)>=2 and np.all(np.diff(ts)>0) and rate>0
    duration=int(ts[-1]-ts[0]);cadence=float(np.median(np.diff(ts)))
    intervals=max(1,int(np.rint(duration/(cadence*rate))))
    wanted=np.linspace(float(ts[0]),float(ts[-1]),intervals+1)
    right=np.searchsorted(ts,wanted,side='left').clip(0,len(ts)-1)
    left=np.maximum(0,right-1)
    ids=np.where(wanted-ts[left]<=ts[right]-wanted,left,right).astype(np.int32)
    query=np.rint(np.linspace(0,duration/rate,intervals+1)).astype(np.int64)
    assert np.all(np.diff(query)>0)and np.all(np.diff(ids)>=0)
    assert ids[0]==0 and ids[-1]==len(ts)-1
    return dict(base_query_index=ids,timestamp_us=query,source_time_requested_us=wanted,
                base_query_pts=ts[ids],median_original_interval_us=cadence)

def enumerate_spans(n):
    return np.asarray([(a,b)for a in range(n-3)for b in range(a+3,n)],np.int32).reshape(-1,2)

def labels(source,baseline_index,block_id,family_index,base_kind='normal'):
    """Require EVERY original observed witness, surviving boundaries and >=4
    distinct observed source frames. Sampling removal cannot create full truth.
    Known splice joins remain negative; ordinary skipped sampling is NOT a join.
    Returns only math audit labels, not encoder-ready training features.
    """
    assert base_kind in ['normal','splice']
    m=np.asarray(baseline_index);block=np.asarray(block_id);grid=enumerate_spans(len(m))
    va=source['valid'][:,family_index].any(1);bs,be=source['bands'][:,family_index].T
    pos=(source['mask'][:,family_index]>0)&(source['y'][:,family_index]>0)
    neg=(source['mask'][:,family_index]>0)&(source['y'][:,family_index]==0)
    required=np.flatnonzero(pos);retained=np.unique(m)
    missing=np.setdiff1d(required,retained)
    onset_retained=np.intersect1d(np.flatnonzero(bs),retained)
    offset_retained=np.intersect1d(np.flatnonzero(be),retained)
    if not len(grid):
        return dict(spans=grid,positive=np.zeros(0,bool),negative=np.zeros(0,bool),eligible=np.zeros(0,bool),
                    crossing=np.zeros(0,bool),partial=np.zeros(0,bool),distinct_observed=np.zeros(0,np.int32),
                    required_witnesses=required,missing_witnesses=missing,onset_retained=onset_retained,offset_retained=offset_retained)
    a,b=grid.T;ma,mb=m[a],m[b];length=b-a+1
    def count(values,aa,bb):
        cs=np.r_[0,np.cumsum(values)];return cs[bb+1]-cs[aa]
    observed=va[m];eligible=observed[a]&observed[b]&(count(observed,a,b)>=.8*length)
    crossing=block[a]!=block[b]
    unique_start=np.r_[True,m[1:]!=m[:-1]]
    distinct=count(unique_start&observed,a,b)+(~unique_start[a]&observed[a])
    # Original-space calculations are meaningful only inside one source block.
    aa=np.minimum(ma,mb);bb=np.maximum(ma,mb);old_n=bb-aa+1
    old_eligible=va[aa]&va[bb]&(count(va,aa,bb)>=.8*old_n)&(old_n>=4)
    old_full=old_eligible&bs[aa]&be[bb]&(count(pos,aa,bb)==pos.sum())&(count(neg,aa,bb)==0)
    # Requiring all original witnesses globally makes lost short motion explicit.
    positive=eligible&~crossing&(ma<=mb)&old_full&(distinct>=4)&(len(required)>0)&(len(missing)==0)
    # Mapped observed disagreement only; lost witness alone is never a negative.
    mapped_pos=count(pos[m],a,b)>0;mapped_neg=count(neg[m],a,b)>0
    inherited_disagreement=old_eligible&~old_full&(mapped_neg|(mapped_pos&(~bs[ma]|~be[mb])))
    negative=eligible&~positive&(crossing|(~crossing&inherited_disagreement&(base_kind=='normal')))
    partial=eligible&~positive&~negative&mapped_pos
    assert not np.any(positive&negative)
    return dict(spans=grid,positive=positive,negative=negative,eligible=eligible,crossing=crossing,partial=partial,
                distinct_observed=distinct.astype(np.int32),required_witnesses=required,missing_witnesses=missing,
                onset_retained=onset_retained,offset_retained=offset_retained)
