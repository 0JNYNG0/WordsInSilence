"""Observe signed strokes without bridging unrelated phase islands.

Model association remains estimated. A signed stroke is not a complete sign.
All observation/decision thresholds are inherited unchanged from the baseline.
"""
from pathlib import Path
import importlib.util,copy
import numpy as np
O=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('localized_frozen_positive_evidence',O/'word100_positive_direction_20261109/positive_evidence.py')
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
original_along_forearm=legacy.along_forearm

def along_forearm(x,ts,phase,contract):
    cfg=legacy.CONFIG['direction'];x=np.asarray(x);ts=np.asarray(ts);phase=np.asarray(phase)
    if len(ts)<2 or len(x)!=len(ts) or len(phase)!=len(ts) or not (np.diff(ts)>0).all():
        raise ValueError('Aligned observations and increasing native timestamps required')
    groups=np.asarray(contract['phase_groups'],dtype=int)+1
    active=phase[:,groups].sum(1)>=cfg['phase_support']
    ids=np.flatnonzero(active)
    cuts=np.flatnonzero((np.diff(ids)>1)|(np.diff(ts[ids])>cfg['max_gap_ms']*1000))+1
    islands=np.split(ids,cuts) if len(ids) else []
    events=[]
    for indices in islands:
        if not len(indices):continue
        a,b=int(indices[0]),int(indices[-1])
        # The unmodified numerical observer only sees this contiguous island.
        # All other model-supported islands remain in the output, not filtered out.
        event=original_along_forearm(x[a:b+1],ts[a:b+1],phase[a:b+1],contract)
        for point in event['trajectory']:point['index']+=a
        event['span']=[a,b];event['source_phase_indices']=indices.astype(int).tolist()
        event['unknown_reason']=('too_short_for_direction' if len(indices)<cfg['min_frames'] else 'insufficient_signed_observation') if event['state']=='unknown' else None
        event['whole_action_complete']=None
        events.append(event)
    opposite=[e for e in events if e['state']=='contradicted']
    state='contradicted' if opposite else 'supported' if events and all(e['state']=='supported' for e in events) else 'unknown'
    out=dict(kind='along_forearm',state=state,moving_hand=contract['moving_hand'],anchor_arm=contract['anchor_arm'],expected_sign=contract['expected_sign'],
        span=[int(ids[0]),int(ids[-1])] if len(ids) else None,contact_verified=False,exact_extent_required=False,
        events=events,contradicted_spans=[e['span'] for e in opposite],
        supported_spans=[e['span'] for e in events if e['state']=='supported'],
        unknown_spans=[e['span'] for e in events if e['state']=='unknown'],
        trajectory=[p for e in events for p in e['trajectory']],
        available_fraction=float(np.mean([e['available_fraction'] for e in events])) if events else 0.,
        observation_version='contiguous_phase_direction_v1',phase_association='model_estimate',
        expected_order_used=False,whole_action_complete=None)
    # Keep exact baseline scalar fields for a single run. Multiple runs have no
    # single signed displacement; reporting one would hide cancellation.
    if len(events)==1:
        for k in ['start_progress','end_progress','signed_change','trend','noise','deadband']:
            if k in events[0]:out[k]=events[0][k]
    return out

legacy.along_forearm=along_forearm

def refine(base,attrs,x,ts,fid,phase):
    out=legacy.refine(base,attrs,x,ts,fid,phase)
    for c in out['cards']:
        m=c.get('motion_evidence')
        if not m:continue
        shape_spans=[s for e in c['shape_evidence'] if e['state']=='contradicted' for s in e['contradicted_spans']]
        c['difference_spans']=shape_spans+m['contradicted_spans']
    out['evidence_version']='positive_shape_and_contiguous_direction_v2'
    return out

def localize_cues(result,compose):
    """Use the existing evidence-bound wording once per actual bad stroke."""
    prepared=copy.deepcopy(result);expanded=[]
    for c in prepared.get('cards',[]):
        motion=c.get('motion_evidence')
        if not motion or 'events' not in motion:
            expanded.append(c);continue
        shape_card=copy.deepcopy(c);shape_card['motion_evidence']=None;expanded.append(shape_card)
        for event in motion['events']:
            if event['state']!='contradicted':continue
            ec=copy.deepcopy(c);ec['shape_evidence']=[];ec['motion_evidence']=event
            expanded.append(ec)
    prepared['cards']=expanded
    cues=compose(prepared)
    # Same guide/component can occur in multiple actual strokes. Keep each
    # distinct native-time cue; the original composer deduplicates shared cards.
    ids={cid for c in cues['cues'] for cid in c['card_ids']}
    cues['notes']=[n for n in cues['notes'] if not(n['card_id'] in ids and n['status']=='uncertain')]
    cues['version']='evidence_bound_contiguous_coaching_v2'
    return cues
