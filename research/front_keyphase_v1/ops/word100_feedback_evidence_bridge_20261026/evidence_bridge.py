"""Read-only research DTW result adapter; no inferred coordinates or verdicts.

Reference-required cells determine denominators, independently of query visibility.
Momentary and sequence evidence are intentionally separate. Detector model names
are provenance, never evidence of correctness by themselves.
"""
import numpy as np

GROUPS = {'finger_angles': list(range(15)), 'palm_xy': [15,16],
          'elbow_xy': [17,18], 'wrist_xy': [19,20]}

def time_cells(time_us):
    t=np.asarray(time_us,dtype=np.float64)/1e6
    if len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('MONOTONIC_REFERENCE_TIMESTAMPS_REQUIRED')
    edges=np.r_[t[0]-(t[1]-t[0])/2, (t[:-1]+t[1:])/2, t[-1]+(t[-1]-t[-2])/2]
    return edges, np.diff(edges)

def runs(mask,edges):
    m=np.asarray(mask,bool)
    starts=np.flatnonzero(m & ~np.r_[False,m[:-1]])
    stops=np.flatnonzero(m & ~np.r_[m[1:],False])+1
    return [{'first_index':int(a),'end_index_exclusive':int(b),
             'reference_start_s':float(edges[a]),'reference_end_s':float(edges[b])}
            for a,b in zip(starts,stops)]

def evidence_gate(raw_supported, identity_confirmed, kind):
    """Kind 0 absent, 1 current image-backed point, 2 inferred, 3 region only.

    Identity confirmation is channel-specific: an arm point must not license a
    finger point. Recovered YOLO/HaMeR points need the same image checks as MP.
    This gate DOES NOT certify image accuracy; the caller supplies that evidence.
    """
    raw=np.asarray(raw_supported,bool); ids=np.asarray(identity_confirmed,bool)
    kind=np.asarray(kind)
    if raw.shape!=ids.shape or raw.shape!=kind.shape:
        raise ValueError('EVIDENCE_SHAPE_MISMATCH')
    if not np.isin(kind,[0,1,2,3]).all():raise ValueError('UNKNOWN_EVIDENCE_KIND')
    return raw & ids & (kind==1)

def build(meta,z,time_us):
    expected=np.asarray(z['expected'],bool)
    if expected.ndim!=3 or expected.shape[1:]!=(2,21):raise ValueError('CHANNEL_SCHEMA')
    edges,w=time_cells(time_us)
    if len(w)!=len(expected):raise ValueError('REFERENCE_TIME_LENGTH')
    p=np.asarray(z['point_comparable'],bool);s=np.asarray(z['sequence_comparable'],bool)
    bad=np.asarray(z['point_bad'],bool)
    if (p&~expected).any() or (s&~expected).any() or (bad&~p).any():
        raise ValueError('EVIDENCE_OUTSIDE_DOMAIN')
    channels=[]
    for h,side in enumerate(('L','R')):
        for c in range(21):
            e=expected[:,h,c];pv=p[:,h,c];sv=s[:,h,c]
            needed=float(w[e].sum());observed=float(w[pv].sum());sequence=float(w[sv].sum())
            status=('NOT_REQUIRED' if needed==0 else 'INSUFFICIENT' if observed==0 else
                    'PARTIAL' if (e&~pv).any() else 'OBSERVED')
            channels.append(dict(side=side,channel=c,status=status,
                required_seconds=needed,point_supported_seconds=observed,
                point_coverage=None if needed==0 else observed/needed,
                sequence_supported_seconds=sequence,
                sequence_coverage=None if needed==0 else sequence/needed,
                point_difference_cells=int(bad[:,h,c].sum()),
                point_only_cells=int((pv&~sv).sum()),
                # No zero-valued error score is fabricated for an absent angle.
                mean_sequence_component_gap=[float(np.average(z['component_distance'][sv,h,c,k],weights=w[sv])) if sv.any() else None for k in range(5)],
                point_unknown_intervals=runs(e&~pv,edges),
                sequence_unknown_intervals=runs(e&~sv,edges),
                point_difference_intervals=runs(bad[:,h,c],edges),
                automatic_correction=None))
    groups={}
    for name,cs in GROUPS.items():
        a=[x for x in channels if x['channel'] in cs]
        den=sum(x['required_seconds'] for x in a)
        groups[name]=dict(required_channel_seconds=den,
            point_coverage=None if not den else sum(x['point_supported_seconds'] for x in a)/den,
            sequence_coverage=None if not den else sum(x['sequence_supported_seconds'] for x in a)/den,
            point_cells=int(p[:,:,cs].sum()),point_only_cells=int((p[:,:,cs]&~s[:,:,cs]).sum()),
            retained_point_differences=int(bad[:,:,cs].sum()),
            point_difference_without_sequence=int((bad[:,:,cs]&~s[:,:,cs]).sum()))
    return dict(schema='wis_dtw_evidence_bridge_v1',query=meta['query'],reference=meta['reference'],
        kind=meta['kind'],channels=channels,groups=groups,
        whole_action_correct=None,automatic_correction=None,
        sequence_flags_diagnostic_only=True,
        sequence_gap_units='angle: original input units; position: shoulder-width coordinates; never mixed into one score',
        limitation='Image agreement is not human ground truth; unknown is neither correct nor incorrect. Sequence difference flags have known speed sensitivity.',
        evidence_counts={k:int(z[k].sum()) for k in ['expected','point_comparable','point_bad','point_unknown','sequence_comparable','sequence_unknown']})

def selftest():
    raw=np.ones((2,21),bool);identity=raw.copy();kind=np.ones((2,21),int)
    kind[0,:15]=2;kind[1,15:17]=3;identity[1,0]=False
    gate=evidence_gate(raw,identity,kind)
    assert not gate[0,:15].any() and gate[0,19:21].all()
    assert not gate[1,15:17].any() and not gate[1,0] and gate[1,1]
    # Confident model output with no image-backed landmark remains unsupported.
    assert not evidence_gate(raw,identity,np.full_like(kind,2)).any()
    return dict(inferred_fingers_not_measured=True,region_not_exact_joint=True,
                identity_conflict_withheld=True,neighbor_angle_and_arm_preserved=True)
