"""Offline evidence labels. Never a sign-correctness or model-selection rule."""
import numpy as np
from wis_lab.service.word100_observation_coverage import timeline

AXES = ('handshape', 'location', 'movement')
MIN_FRAMES = 8
MIN_REFS = 5

def describe(ts, flags, position, fixed_bounds=None):
    """Per-hand position envelope. Boundary absence is retained separately.

    Trimming boundaries is not proof that no essential action occurred there.
    Shape gaps keep the wrist envelope, so an invisible finger cannot shrink it.
    """
    flags=np.asarray(flags,bool);position=np.asarray(position,bool)
    t=(np.asarray(ts,dtype=float)-ts[0])/1e6
    ids=np.flatnonzero(position)
    bounds=fixed_bounds
    if bounds is None and len(ids)>1:bounds=(float(t[ids[0]]),float(t[ids[-1]]))
    whole=timeline(ts,flags)
    x=timeline(ts,flags,bounds) if bounds and bounds[1]>bounds[0] else timeline([],[])
    return dict(observed_frames=int(flags.sum()),position_frames=int(position.sum()),
        coverage=x['observed_percent'],gap_fraction=x['longest_unobserved_s']/x['duration_s'] if x['duration_s'] else None,
        missing_s=sum(a['duration_s'] for a in x['missing_intervals']),duration_s=x['duration_s'],
        missing_intervals=x['missing_intervals'],bounds=bounds,
        whole_coverage=whole['observed_percent'],boundary_s=whole['duration_s']-x['duration_s'],
        boundary_meaning='Unassessed boundary; may be preparation, lowering, or missed action, not certified rest.')

def decide(d, valid_refs, required, thresholds=None):
    if required is False:return dict(status='NOT_REQUIRED',reasons=['REFERENCE_DECLARED_OPTIONAL'],wrong_action=None)
    reasons=[]
    if d['coverage'] is None:reasons.append('NO_POSITION_TIME_ENVELOPE')
    if d['observed_frames']<MIN_FRAMES:reasons.append('FEWER_THAN_8_OBSERVED_FRAMES')
    if valid_refs<MIN_REFS:reasons.append('FEWER_THAN_5_SUPPORTED_REFERENCES')
    if reasons:return dict(status='INSUFFICIENT',reasons=reasons,wrong_action=None)
    if thresholds is not None:
        if d['coverage']<thresholds['coverage_floor']-1e-7:reasons.append('BELOW_TRAIN_COVERAGE_Q05')
        if d['gap_fraction']>thresholds['gap_ceiling']+1e-7:reasons.append('ABOVE_TRAIN_GAP_Q95')
    if reasons:return dict(status='INSUFFICIENT',reasons=reasons,wrong_action=None)
    if d['missing_s']>1e-8:return dict(status='PARTIAL',reasons=['INTERNAL_UNOBSERVED_TIME'],wrong_action=None)
    return dict(status='OBSERVED_ENVELOPE',reasons=['NO_GAP_WITHIN_OBSERVED_ENVELOPE'],wrong_action=None)

def calibrate(rows):
    out={}
    for key in AXES:
        chosen=[r for r in rows if r['key']==key and r['required'] is not False and
                decide(r['evidence'],r['valid_refs'],r['required'])['status']!='INSUFFICIENT']
        out[key]=dict(n=len(chosen),coverage_floor=float(np.quantile([r['evidence']['coverage'] for r in chosen],.05)),
                      gap_ceiling=float(np.quantile([r['evidence']['gap_fraction'] for r in chosen],.95)))
    return out
