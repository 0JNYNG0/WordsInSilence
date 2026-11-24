"""Descriptive time coverage only: never scoring, hand roles, imputation or a verdict."""
import numpy as np
from ..core import features as F

SCHEMA = 'wis_metric_time_coverage_v1'
AXES = ('handshape', 'location', 'movement')

def metric_flags(obs):
    """Eligibility of the immutable legacy DTW groups at every original frame.

    This intentionally retains the current 19-angle gate; research partial-angle
    measurements are not silently substituted for the existing displayed score.
    """
    n = obs.n_frames
    valid = obs.usable & np.isfinite(obs.xy_px).all(axis=2)
    try:
        _, scale, _ = F.body_frame(obs)
        body = valid[:, 11] & valid[:, 12] & np.isfinite(scale)
    except F.FeatureError:
        body = np.zeros(n, bool)
    out = {}
    for side, sl in [('L', slice(33,54)), ('R', slice(54,75))]:
        p = obs.xy_px[:, sl];ok = valid[:, sl]
        position = body & ok[:, 0]
        handscale = np.median(np.linalg.norm(p[:, F.HAND_MCP] - p[:, :1], axis=2), axis=1)
        bend = body & ok.all(axis=1) & np.isfinite(handscale) & (handscale > 1e-6)
        vectors = []
        for chain in F.FINGER_CHAINS:
            for i in (1,2,3):
                a,b,c = chain[i-1:i+2]
                vectors.extend([p[:,a]-p[:,b], p[:,c]-p[:,b]])
        directions = [p[:,chain[1]]-p[:,chain[4]] for chain in F.FINGER_CHAINS]
        for i,j in F.SPREAD_PAIRS:vectors.extend([directions[i], directions[j]])
        for vec in vectors:
            length = np.linalg.norm(vec, axis=1)
            bend &= np.isfinite(length) & (length >= 1e-6)
        out[side] = {'handshape':bend, 'location':position, 'movement':position.copy()}
    return out

def timeline(timestamp_us, valid, bounds=None):
    """Duration-weighted observed/unknown intervals. No extension across long gaps.

    Each frame owns its nearest-time cell, capped to1.5 median frame intervals
    on either side. This defines an estimate of temporal coverage, not an error
    threshold or interpolation of missing positions. Uniform speed scaling is invariant.
    """
    ts = np.asarray(timestamp_us, dtype=np.float64) / 1e6
    flags = np.asarray(valid, dtype=bool)
    if ts.ndim!=1 or flags.shape!=ts.shape or not np.isfinite(ts).all():raise ValueError('INVALID_COVERAGE_ARRAY')
    if len(ts)>1 and not (np.diff(ts)>0).all():raise ValueError('NON_MONOTONIC_COVERAGE_TIME')
    if len(ts)<2:return {'status':'UNAVAILABLE','duration_s':0.,'observed_s':0.,'observed_percent':None,'longest_unobserved_s':0.,'intervals':[],'missing_intervals':[]}
    origin=ts[0];t=ts-origin
    start,end=(0.,float(t[-1])) if bounds is None else tuple(float(v) for v in bounds)
    if start<0 or end>t[-1]+1e-8 or end<=start:
        return {'status':'UNAVAILABLE','duration_s':0.,'observed_s':0.,'observed_percent':None,'longest_unobserved_s':0.,'intervals':[],'missing_intervals':[]}
    middle=(t[:-1]+t[1:])/2;radius=1.5*float(np.median(np.diff(t)))
    left=np.maximum(np.r_[0.,middle], t-radius);right=np.minimum(np.r_[middle,t[-1]],t+radius)
    intervals=[]
    def add(a,b,state):
        a=max(start,float(a));b=min(end,float(b))
        if b<=a:return
        if intervals and intervals[-1]['state']==state and abs(intervals[-1]['end_s']-a)<1e-8:
            intervals[-1]['end_s']=b
        else:intervals.append({'start_s':a,'end_s':b,'state':state})
    cursor=start
    for i in range(len(t)):
        if right[i]<=start or left[i]>=end:continue
        add(cursor,left[i],'timing_gap')
        add(left[i],right[i],'observed' if flags[i] else 'unobserved')
        cursor=max(cursor,min(end,right[i]))
    add(cursor,end,'timing_gap')
    observed=sum(x['end_s']-x['start_s'] for x in intervals if x['state']=='observed')
    missing=[]
    for x in intervals:
        if x['state']=='observed':continue
        if missing and abs(missing[-1]['end_s']-x['start_s'])<1e-8:
            missing[-1]['end_s']=x['end_s']
        else:missing.append({'start_s':x['start_s'],'end_s':x['end_s']})
    for x in missing:x['duration_s']=x['end_s']-x['start_s']
    duration=end-start
    return {'status':'AVAILABLE','start_s':start,'end_s':end,'duration_s':duration,'observed_s':observed,
            'observed_percent':min(100.,max(0.,100*observed/duration)),
            'longest_unobserved_s':max([x['duration_s'] for x in missing],default=0.),
            'intervals':intervals,'missing_intervals':missing}

def summarize_flags(ts, flags, active_bounds, required_sides=None):
    metrics=[]
    for key in AXES:
        sides=[]
        for side in ('L','R'):
            whole=timeline(ts,flags[side][key])
            active=timeline(ts,flags[side][key],active_bounds) if active_bounds is not None else timeline([],[])
            sides.append({'side':side,'required':side in required_sides if required_sides else None,'whole':whole,'analysis_span':active})
        metrics.append({'key':key,'sides':sides})
    return {'schema':SCHEMA,'status':'AVAILABLE','metrics':metrics,'is_a_verdict':False,'accepted':None,
            'basis':'Query-side raw-frame eligibility, duration weighted; not paired DTW support, correctness or reference agreement.',
            'scope':'Whole recording and original first-to-last any usable hand span. Missing outside span still displayed in whole recording.',
            'time_estimate':'Nearest-frame cells; long timing gaps remain unknown; no coordinate interpolation.',
            'missing_meaning':'May include normal lowering, unused hand, occlusion or detector loss. Cause and correctness are not inferred.'}

def summarize(obs, required_sides=None):
    flags=metric_flags(obs);active=None
    if obs.n_frames>1:
        try:
            lo,hi=F.active_span(obs)
            if hi>lo:active=tuple(float((obs.timestamp_us[i]-obs.timestamp_us[0])/1e6) for i in (lo,hi))
        except F.FeatureError:pass
    return summarize_flags(obs.timestamp_us,flags,active,required_sides)
