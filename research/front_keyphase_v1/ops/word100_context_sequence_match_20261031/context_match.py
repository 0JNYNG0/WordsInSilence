"""Ordered local traces; no word labels or forced order across occurrences."""
import numpy as np
from numba import njit

EPS = 1e-9
CONTEXT_SECONDS = .20
MIN_FRAME_COUNT_RATIO = .5
MAX_FRAME_COUNT_RATIO = 2.

def frame_cost(q, r, groups):
    n = len(q[groups[0]][0]); m = len(r[groups[0]][0])
    total = np.zeros((n, m)); active = np.zeros(m)
    for g in groups:
        x, qm = q[g]; y, rm = r[g]
        den = rm.sum(1)
        shared = qm[:, None] & rm[None]
        diff = np.minimum(np.abs(x[:, None] - y[None]), 1)
        numerator = np.where(shared, diff, 0).sum(2) + (rm[None] & ~qm[:, None]).sum(2)
        total += np.divide(numerator, den[None], out=np.zeros((n, m)), where=den[None] > 0)
        active += den > 0
    return np.divide(total, active[None], out=np.zeros_like(total), where=active[None] > 0)

@njit(cache=False)
def window_dp(cost, start, max_length, return_back=False):
    # State 0 diagonal, 1/2 consecutive query-only, 3/4 reference-only.
    m = cost.shape[1]
    dp = np.full((max_length, m, 5), np.inf)
    back = np.full((max_length, m, 5), -1, np.int8)
    dp[0, 0, 0] = 2 * cost[start, 0]
    for i in range(max_length):
        for j in range(m):
            c = cost[start+i, j]
            if i > 0 and j > 0:
                k = np.argmin(dp[i-1, j-1])
                dp[i, j, 0] = dp[i-1, j-1, k] + 2*c
                back[i, j, 0] = k
            if i > 0:
                k = 0
                if dp[i-1,j,3] < dp[i-1,j,k]: k = 3
                if dp[i-1,j,4] < dp[i-1,j,k]: k = 4
                dp[i,j,1] = dp[i-1,j,k] + c; back[i,j,1] = k
                dp[i,j,2] = dp[i-1,j,1] + c; back[i,j,2] = 1
            if j > 0:
                k = 0
                if dp[i,j-1,1] < dp[i,j-1,k]: k = 1
                if dp[i,j-1,2] < dp[i,j-1,k]: k = 2
                dp[i,j,3] = dp[i,j-1,k] + c; back[i,j,3] = k
                dp[i,j,4] = dp[i,j-1,3] + c; back[i,j,4] = 3
    return dp, back

def recover(cost, start, length):
    dp, back = window_dp(cost, start, length)
    i, j = length-1, cost.shape[1]-1
    k = int(np.argmin(dp[i,j])); path = []
    assert np.isfinite(dp[i,j,k])
    while True:
        path.append((start+i,j))
        if i == 0 and j == 0: break
        previous = int(back[i,j,k]); assert previous >= 0
        if k == 0: i-=1; j-=1
        elif k in (1,2): i-=1
        else: j-=1
        k = previous
    return np.array(path[::-1], int)

def windows(cost):
    n, m = cost.shape
    lo = max(2, int(np.ceil(m*MIN_FRAME_COUNT_RATIO)))
    hi = min(n, int(np.floor(m*MAX_FRAME_COUNT_RATIO)))
    all_windows = []
    for start in range(n-lo+1):
        top = min(hi,n-start)
        dp, _ = window_dp(cost,start,top)
        for length in range(lo,top+1):
            value = float(dp[length-1,-1].min()/(length+m))
            if np.isfinite(value): all_windows.append((value,start,length))
    return sorted(all_windows)

def trace_match(q,r,segments,reference_pts,groups,summary_fn):
    n = len(q[groups[0]][0]); m = len(r[groups[0]][0])
    full_cost = frame_cost(q,r,groups)
    anchors = []
    for s in segments:
        if s['role'] != 'core_candidate': continue
        a,b = s['core']; t = np.asarray(reference_pts)/1e6
        ra = int(np.searchsorted(t,t[a]-CONTEXT_SECONDS,'left'))
        rb = int(np.searchsorted(t,t[b]+CONTEXT_SECONDS,'right'))-1
        rb = min(m-1,rb)
        costs = full_cost[:,ra:rb+1]
        candidates = windows(costs)
        if not candidates:
            anchors.append(dict(segment_id=s['segment_id'],reference_core=[a,b],reference_context=[ra,rb],state='no_duration_candidate',best=None,alternatives=[])); continue
        def pack(item):
            value,start,length = item
            path = recover(costs,start,length); path[:,1] += ra
            assert set(path[:,0]) == set(range(start,start+length))
            assert set(path[:,1]) == set(range(ra,rb+1))
            coreq = np.unique(path[(path[:,1]>=a)&(path[:,1]<=b),0])
            phases = {}
            for name,ids in [('entry',np.arange(ra,a)),('core',np.arange(a,b+1)),('exit',np.arange(b+1,rb+1))]:
                phases[name] = summary_fn(q,r,path,ids) if len(ids) else None
            expected_core = [v for v in phases['core'].values() if v['expected']]
            supported = bool(expected_core) and all(v['sufficient'] for v in expected_core)
            context_support = {}
            for phase in ['entry','exit']:
                vals=[] if phases[phase] is None else [v for v in phases[phase].values() if v['expected']]
                context_support[phase]=bool(vals) and all(v['sufficient'] for v in vals)
            return dict(cost=value,query_window=[start,start+length-1],query_core=[int(coreq[0]),int(coreq[-1])],path=path.tolist(),phases=phases,core_supported=supported,context_support=context_support)
        best = pack(candidates[0])
        alternatives = []
        # Search every window, not just the closest adjacent windows.
        for item in candidates[1:]:
            _,st,ln = item
            if st >= best['query_core'][0] and st+ln-1 <= best['query_core'][1]: continue
            pp=recover(costs,st,ln); pp[:,1]+=ra
            cq=pp[(pp[:,1]>=a)&(pp[:,1]<=b),0]
            if cq.max()>=best['query_core'][0] and cq.min()<=best['query_core'][1]: continue
            alt = pack(item); alternatives.append(alt); break
        # All equally optimal windows: check core spans, including overlapping contexts.
        tied_cores = []
        for item in candidates:
            if item[0] > best['cost']+EPS: break
            pp = recover(costs,item[1],item[2]); pp[:,1] += ra
            cq=pp[(pp[:,1]>=a)&(pp[:,1]<=b),0]
            span=[int(cq.min()),int(cq.max())]
            if span not in tied_cores: tied_cores.append(span)
        disjoint_tie=any(z[1]<best['query_core'][0] or z[0]>best['query_core'][1] for z in tied_cores)
        context_complete=all(best['context_support'].values()) and t[a]-.2>=t[0] and t[b]+.2<=t[-1]
        state='observation_insufficient' if not best['core_supported'] else 'context_incomplete' if not context_complete else 'ambiguous_occurrence' if disjoint_tie else 'supported_candidate'
        anchors.append(dict(segment_id=s['segment_id'],reference_core=[a,b],reference_context=[ra,rb],requested_context_seconds=[float(t[a]-.2),float(t[b]+.2)],available_context_seconds=[float(t[ra]),float(t[rb])],clipped_at_video_edge=[bool(t[a]-.2<t[0]),bool(t[b]+.2>t[-1])],best=best,alternatives=alternatives,tied_core_spans=tied_cores,disjoint_exact_tie=disjoint_tie,state=state,searched_windows=len(candidates),second_disjoint_gap=alternatives[0]['cost']-best['cost'] if alternatives else None,within_window_path_ties_enumerated=False))
    relations=[]
    for a,b in zip(anchors,anchors[1:]):
        if any(x['state'] in ('observation_insufficient','context_incomplete') or x['best'] is None for x in [a,b]): state='insufficient'
        elif any(x['state']=='ambiguous_occurrence' for x in [a,b]): state='ambiguous'
        else:
            aa,bb=a['best']['query_core'],b['best']['query_core']
            state='forward_candidate' if aa[1]<bb[0] else 'reverse_candidate' if bb[1]<aa[0] else 'overlapping_occurrences'
        relations.append(dict(first=a['segment_id'],second=b['segment_id'],state=state))
    return dict(anchors=anchors,relations=relations,query_frames=n,reference_frames=m,query_search_scope=n,reference_scope=m,semantic_correctness=False,forced_interstage_order=False)
