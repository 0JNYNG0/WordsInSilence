"""Description-linked measurement evidence, never a linguistic correctness classifier.

Only saved reviewed source anchors attach prose to a reference. All DTW links
and all required cells remain visible. Missing values are not zero differences.
"""
import numpy as np

GROUP_MAP = {
    **{f'{h}_finger_angles14': f'{h}_shape' for h in 'LR'},
    **{f'{h}_palm_axis2d': f'{h}_axis' for h in 'LR'},
    **{f'{h}_hand_position2d': f'{h}_location' for h in 'LR'},
    **{f'{h}_arm2d': f'{h}_arm' for h in 'LR'},
    'two_hand_relation2d': 'hands_relation',
    'L_to_R_forearm2d': 'L_to_R_forearm',
    'R_to_L_forearm2d': 'R_to_L_forearm',
}

def runs(values):
    out=[]
    for v in sorted(set(map(int,values))):
        if out and v == out[-1][1]+1: out[-1][1]=v
        else: out.append([v,v])
    return out

def trace(q, r, path, core):
    """One node per ORIGINAL reference frame, every linked query retained.

    The difference reducer exactly matches legacy observed_summary. The new
    trace only locates saved values; it introduces no alignment or fit threshold.
    Raw values/min/max are for inspection, not recovered handshape labels.
    """
    x,qm=q; y,rm=r
    x=np.asarray(x); y=np.asarray(y)
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1]
    assert np.shape(qm)==x.shape and np.shape(rm)==y.shape
    qm=np.asarray(qm,bool)&np.isfinite(x); rm=np.asarray(rm,bool)&np.isfinite(y)
    raw=np.asarray(path)
    assert raw.ndim==2 and raw.shape[1]==2 and len(raw) and np.isfinite(raw).all()
    assert np.equal(raw,np.floor(raw)).all()
    pp=raw.astype(int)
    assert (pp>=0).all() and (pp[:,0]<len(x)).all() and (pp[:,1]<len(y)).all()
    steps=np.diff(pp,axis=0)
    assert (steps>=0).all() and (steps<=1).all() and (steps.sum(1)>0).all()
    assert len(core)==2 and all(type(v) is int for v in core) and 0<=core[0]<=core[1]<len(y)
    out=[]
    for j in range(core[0],core[1]+1):
        ids=pp[pp[:,1]==j,0]
        assert len(ids), 'Reference scope must not shrink'
        shared=qm[ids]&rm[j]; observed=shared.any(0)
        delta=np.where(shared,np.minimum(np.abs(x[ids]-y[j]),1),0)
        per=delta.max(0)
        out.append(dict(reference_frame=j,query_frames=ids.tolist(),
            reference_valid=rm[j].tolist(),query_valid=qm[ids].tolist(),
            reference_values=[float(v) if ok else None for v,ok in zip(y[j],rm[j])],
            query_values=[[float(v) if ok else None for v,ok in zip(x[i],qm[i])] for i in ids],
            difference_by_component=[float(v) if ok else None for v,ok in zip(per,observed)],
            observed_components=int(observed.sum()),expected_components=int(rm[j].sum()),
            possible_components=y.shape[1],
            path_missing_cells=int((rm[j]&~qm[ids]).sum()),
            difference=float(per.sum()/observed.sum()) if observed.any() else None))
    return out

def observation_status(metric):
    # Legacy sufficient alone can be true when only two of six arm channels exist.
    # Keep those measurements, but never call the WHOLE group well observed.
    if metric['compared']==0:return 'unobserved'
    if metric['reference_unobserved'] or metric['path_missing_cells'] or not metric['sufficient']:
        return 'partial_observation'
    return 'complete_observation'

def feedback(metric, saved_decision=None):
    obs=observation_status(metric)
    state=(saved_decision or {}).get('state')
    if obs!='complete_observation': status='comparison_evidence_insufficient'
    elif state=='larger_than_fit_reference_range': status='observed_difference_to_review'
    elif state=='within_small_reference_range': status='within_small_reference_sample'
    else: status='measurement_only_no_range'
    return dict(status=status,observation=obs,original_range_state=state,
        educational_correct=None,educational_error=None,correction_instruction=None,
        important_difference_proven=False,allowable_difference_proven=False)

def source_cards(segment, reference, reference_word, clauses):
    cards=[]
    for link in segment.get('source_links',[]):
        c=clauses[link['clause_id']]
        assert c['word_id']==reference_word, 'Wrong expression/source linkage'
        assert c['runtime_scope']!='inactive_different_form', 'Different form cannot be merged'
        assert link['text']==c['source_span'] and link['url']==c['source_url'], 'Source text/URL mismatch'
        assert link['status']==c['previous_visual_match_status'], 'Source support status mismatch'
        assert link['anchor_frames'] and all(type(v) is int for v in link['anchor_frames'])
        assert len(set(link['anchor_frames']))==len(link['anchor_frames'])
        fit=next((a for a in c['anchors'] if a['alias']==reference and a['role']=='fit'),None)
        assert fit is not None, 'Word ID is insufficient for source propagation'
        known={a['frame_id'] for a in fit['anchors']}
        assert set(link['anchor_frames'])<=known
        assert all(segment['core'][0]<=f<=segment['core'][1] for f in link['anchor_frames'])
        groups=list(dict.fromkeys(GROUP_MAP[g] for g in c['groups'] if g in GROUP_MAP))
        cards.append(dict(clause_id=c['id'],text=link['text'],url=link['url'],
            source_entry_id=c['source_entry_id'],source=c['source'],
            reference_segment_id=segment['segment_id'],numeric_context_core=segment['core'],
            full_span_semantic_verified=False,
            source_status=link['status'],reference_anchor_frames=link['anchor_frames'],
            applicable_numeric_groups=groups,
            context_only_groups=[g for g in c['groups'] if g not in GROUP_MAP],
            limitations=c['unsupported_components'],restrictions=c['mapping_stage_restrictions'],
            scope='reference anchor context only; no framewise semantic target',
            named_shape_verified=False,source_action_verified=False,training_target=False))
    return cards
