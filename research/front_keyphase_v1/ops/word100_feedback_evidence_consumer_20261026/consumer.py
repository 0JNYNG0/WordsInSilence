"""Research evidence consumer. No recognition, alignment, truth labels or pose editing."""
import numpy as np

GROUPS=(('finger_angles',0,15,'projected_degrees'),('palm_position',15,17,'shoulder_width'),('elbow_position',17,19,'shoulder_width'),('pose_wrist_position',19,21,'shoulder_width'),('hand_wrist_position',21,23,'shoulder_width'))
SCHEMA='wis_research_feedback_evidence_v1'

def finite_mean(x):
    values=np.asarray(x,float);values=values[np.isfinite(values)]
    return float(values.mean()) if len(values) else None

def spans(mask,times):
    """Inclusive sample endpoints; not inferred visibility between missing frames."""
    indices=np.flatnonzero(mask)
    if not len(indices):return []
    cuts=np.flatnonzero(np.diff(indices)>1)+1
    return [dict(first=int(p[0]),last=int(p[-1]),samples=len(p),reference_start_s=float(times[p[0]]/1e6),reference_end_s=float(times[p[-1]]/1e6)) for p in np.split(indices,cuts)]

def category(needed,compared,corroborated):
    if needed==0:return 'outside_audit_scope'
    if compared==0:return 'no_comparable_evidence'
    if compared<needed:return 'partial_evidence'
    if corroborated<compared:return 'computable_not_fully_corroborated'
    return 'corroboration_available'

def build(*,domain,point_ok,point_delta,sequence_ok,sequence_delta,sequence_quality,source,reference_missing,reference_times_us,reference_id,query_id,kind,recognition=None):
    """Input deltas are descriptive differences; nothing here labels sign correctness.

    Source1 refined,2 retained original,0 insufficient. Quality corroboration is
    evidence agreement, not proof of anatomical correctness. Neither target word
    nor classifier output participates in comparison status or coverage.
    """
    shape=np.shape(domain)
    if len(shape)!=3 or shape[1:]!=(2,23):raise ValueError('expected [reference_time,2hands,23channels]')
    n=shape[0];times=np.asarray(reference_times_us,np.int64)
    if times.shape!=(n,) or not np.all(np.diff(times)>0):raise ValueError('invalid reference time')
    for arr in (point_ok,point_delta,sequence_ok,sequence_quality,source,reference_missing):
        if np.shape(arr)!=shape:raise ValueError('evidence shape mismatch')
    if np.shape(sequence_delta)!=shape+(5,):raise ValueError('sequence descriptor must have 3levels+2changes')
    domain=np.asarray(domain,bool);po=np.asarray(point_ok,bool);so=np.asarray(sequence_ok,bool);sq=np.asarray(sequence_quality,bool)
    if (po&~domain).any() or (so&~domain).any() or (sq&~so).any():raise ValueError('evidence outside scope')
    if not np.array_equal(np.isfinite(point_delta),po):raise ValueError('unknown point cannot carry zero or a number')
    if not np.array_equal(np.isfinite(sequence_delta).all(-1),so) or not np.array_equal(np.isfinite(sequence_delta).any(-1),so):raise ValueError('unknown sequence must have all components absent')
    if (np.asarray(point_delta)[po]<0).any() or (np.asarray(sequence_delta)[so]<0).any():raise ValueError('differences cannot be negative')
    if not np.isin(source,[0,1,2]).all() or (so&(source==0)).any():raise ValueError('invalid provenance')
    hands=[]
    for hand in (0,1):
        groups=[];angles=[]
        for name,a,b,unit in GROUPS:
            ex=domain[:,hand,a:b];p=po[:,hand,a:b];s=so[:,hand,a:b];q=sq[:,hand,a:b];pd=np.asarray(point_delta)[:,hand,a:b];sd=np.asarray(sequence_delta)[:,hand,a:b]
            required=int(ex.sum());pc=int(p.sum());sc=int(s.sum());qc=int(q.sum())
            unavailable=ex&~p;context_unavailable=ex&p&~s;unconfirmed=s&~q
            src=source[:,hand,a:b]
            groups.append(dict(group=name,unit=unit,audit_cells=required,point_compared_cells=pc,sequence_compared_cells=sc,corroborated_sequence_cells=qc,
                point_coverage=(pc/required if required else None),sequence_coverage=(sc/required if required else None),corroborated_coverage=(qc/required if required else None),
                evidence_state=category(required,sc,qc),point_mean_difference=finite_mean(pd[p]),sequence_level_mean_difference=finite_mean(sd[s,:3]),sequence_change_mean_difference=finite_mean(sd[s,3:]),
                corroborated_level_mean_difference=finite_mean(sd[q,:3]),corroborated_change_mean_difference=finite_mean(sd[q,3:]),
                source_cells=dict(refined=int((s&(src==1)).sum()),retained_original=int((s&(src==2)).sum())),
                missing_point_intervals=spans(unavailable.any(-1),times),sequence_context_unavailable_intervals=spans(context_unavailable.any(-1),times),uncorroborated_intervals=spans(unconfirmed.any(-1),times),
                reference_missing_intervals=spans((reference_missing[:,hand,a:b]&ex).any(-1),times),
                timeline=dict(reference_seconds=(times/1e6).tolist(),audit_channels=ex.sum(-1).tolist(),point_channels=p.sum(-1).tolist(),sequence_channels=s.sum(-1).tolist(),corroborated_channels=q.sum(-1).tolist(),
                    point_difference=[finite_mean(pd[i,p[i]]) for i in range(n)],sequence_level_difference=[finite_mean(sd[i,s[i],:3]) for i in range(n)],sequence_change_difference=[finite_mean(sd[i,s[i],3:]) for i in range(n)]),
                gesture_correct=None,correction_direction=None))
        for angle in range(15):
            ex=domain[:,hand,angle];p=po[:,hand,angle];s=so[:,hand,angle];q=sq[:,hand,angle]
            angles.append(dict(angle_index=angle,audit_cells=int(ex.sum()),point_compared_cells=int(p.sum()),sequence_compared_cells=int(s.sum()),corroborated_cells=int(q.sum()),
                point_mean_degrees=finite_mean(point_delta[:,hand,angle][p]),sequence_level_mean_degrees=finite_mean(sequence_delta[:,hand,angle,:3][s]),sequence_change_mean_degrees=finite_mean(sequence_delta[:,hand,angle,3:][s]),
                missing_point_intervals=spans(ex&~p,times),sequence_context_unavailable_intervals=spans(ex&p&~s,times),uncorroborated_intervals=spans(s&~q,times)))
        hands.append(dict(side=('L','R')[hand],groups=groups,angles=angles))
    return dict(schema=SCHEMA,query_id=query_id,reference_id=reference_id,condition=kind,reference_clock='original_video_seconds',sample_interval_convention='inclusive_sample_centres',
        domain_meaning='Fixed reference annotation audit grid; not a claim that both hands are required throughout.',
        temporal_comparison='Each channel may retain original or refined time correspondence; no coherent whole-pose claim.',
        quality_meaning='Corroborating observation available, not anatomical ground truth.',
        recognition=recognition,recognition_scope='Saved original-input predictions only; transformed conditions are not re-inferred.',
        hands=hands,whole_gesture_correct=None,skill_score=None,validated_correction=None,
        limitations=['No sign-correctness thresholds applied','No anatomical truth labels','2D projected angle differences','Missing evidence is not a performance error','Original point and sequence differences both retained'])
