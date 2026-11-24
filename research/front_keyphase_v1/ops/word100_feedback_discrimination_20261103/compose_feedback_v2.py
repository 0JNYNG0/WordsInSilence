"""Combine learned observations without turning occurrence into completion.

No word-specific branches, score thresholds, alignment or model calls. Inputs
are exact saved evidence from independently identified source videos/axes.
The output retains each factor and its native extent; it is not a correctness
classifier and cannot invent an unobserved required action or a teaching rule.
"""
from copy import deepcopy
import hashlib
import json


VERSION = 'factorized-feedback-20261103-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def require(ok, text):
    if not ok:
        raise ValueError(text)


def high_confirmed(observation):
    if 'high_confirmation' in observation:
        require(type(observation['high_confirmation']) is bool, 'High confirmation must be boolean')
        return observation['high_confirmation']
    require(observation['status']=='INSUFFICIENT_OBSERVATION' and observation['probability'] is None,
            'High confirmation absent outside explicit no-observation schema')
    require(observation['prediction_coverage']==0 and observation['anchor'] is None and
            observation['unavailable_native_frames']==observation['fixed_scope']['frame_ids'],
            'No-observation schema requires zero coverage and the whole fixed scope unavailable')
    return False


def attach(base, local, identity):
    """Add a joint view to an existing source+partial-order comparison.

    identity comes from verified manifests, never from labels or a score.
    local is one original row of AJ or the explicitly new reverse evaluation.
    All probability/extent/unknown fields are copied, never recomputed.
    """
    require('combined_evidence_feedback' not in base, 'Already combined')
    require(base['word_id'] == local['word_id'] == identity['word_id'], 'Word differs')
    require(base['source_sha256'] == identity['video_sha256'], 'Source differs')
    require(digest(base) == identity['base_value_sha256'], 'Base changed')
    require(digest(local) == identity['local_value_sha256'], 'Local evidence changed')
    require(local['case_id'] == identity['local_case_id'], 'Local case differs')
    require(local['cache_binding']['path'] == identity['analysis_path'] and
            local['cache_binding']['sha256'] == identity['analysis_sha256'], 'Local analysis cache differs')
    frames = identity['native_frame_ids']; times = identity['timestamp_us']
    require(len(frames) == len(times) == local['native_frames'] == base['frames'], 'Axis length differs')
    require(all(type(v) is int and v >= 0 for v in frames + times), 'Native axes must be integers')
    require(all(b > a for a,b in zip(frames,frames[1:])) and all(b > a for a,b in zip(times,times[1:])), 'Axes not monotone')
    pos = {v:i for i,v in enumerate(frames)}
    ledger_key=identity['local_timestamp_key']
    require(ledger_key in {'control_timestamp_us','output_timestamp_us'} and
            local['source_ledger'][ledger_key] == times, 'Local ledger timestamps differ')
    for key in ['image_unavailable_frames','unobserved_frames']:
        require(all(type(f) is int and f in pos for f in local[key]), 'Local missing time outside input')
    def time_check(ff, seconds):
        require(all(f in pos for f in ff), 'Evidence outside native input')
        require(len(ff) == len(seconds), 'Evidence times differ')
        require(all(abs(t-times[pos[f]]/1e6) < 1e-9 for f,t in zip(ff,seconds)), 'Evidence not at original time')
    comparisons = base['guide_feature_comparison']['items']
    orders = base['partial_order_feedback']['items']
    stage_ids = {s['guide_id'] for s in base['stages']}
    for item in comparisons:
        ff=item.get('frame_ids',[])
        if ff:
            require(all(type(f) is int and f in pos for f in ff), 'Source interval outside axis')
            require(ff==frames[pos[ff[0]]:pos[ff[-1]]+1], 'Source interval omitted native rows')
            time_check([ff[0],ff[-1]],[item['start_seconds'],item['end_seconds']])
        for p in item.get('native_points', []):
            time_check([p['frame']], [p['seconds']])
    for item in orders:
        for key in ['first_evidence','second_evidence']:
            e=item[key]; indices=e['input_frames']; seconds=e['native_seconds']
            require(all(type(i) is int and 0 <= i < len(frames) for i in indices), 'Order input index outside axis')
            require(len(indices)==len(seconds) and all(abs(t-times[i]/1e6)<1e-9 for i,t in zip(indices,seconds)), 'Order time differs')
    events=[]
    for obs in local['observations']:
        require(obs['guide_id'] in stage_ids, 'Motion guide missing')
        scope = obs['fixed_scope']; ff=scope['frame_ids']; tt=scope['timestamp_us']
        require(ff == frames[pos[ff[0]]:pos[ff[-1]]+1], 'Motion scope omitted input frames')
        require(tt == [times[pos[f]] for f in ff], 'Motion source axis differs')
        require(obs['frames'] == [ff[0],ff[-1]], 'Motion bracket differs')
        require(all(type(f) is int and f in ff for f in obs['unavailable_native_frames']), 'Scope gap outside motion scope')
        require(obs.get('whole_action_complete') is None and obs.get('educational_error') is None,
                'Occurrence cannot be a correctness result')
        linked=[]
        for i,order in enumerate(orders):
            gids=order['first']['guide_ids']+order['second']['guide_ids']
            if obs['guide_id'] in gids:
                linked.append(i)
        flags=[]
        if high_confirmed(obs):flags.append('MOVEMENT_EVIDENCE_PRESENT')
        if obs['unavailable_native_frames']:flags.append('OBSERVATION_GAP_RETAINED')
        if any(orders[i]['observation_status']=='OBSERVED_B_BEFORE_A' for i in linked):
            flags.append('RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER')
        if any(orders[i]['observation_status'] not in {'OBSERVED_A_BEFORE_B','OBSERVED_B_BEFORE_A'} for i in linked):
            flags.append('RELATED_PAIR_ORDER_UNRESOLVED')
        if not linked:flags.append('NO_DISTINCT_PAIR_QUERY')
        # No conjunctive "passed" label: existence, sequence and missingness
        # answer different questions. In particular there is no absence rule.
        msg=('이 구간에서 해당 움직임의 근거가 관측됐습니다. ' if high_confirmed(obs)
             else '이 구간의 해당 움직임은 확인 보류입니다. ')
        if 'RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER' in flags:
            msg+='별도로 관련된 두 동작이 참고와 반대 순서로 관측된 근거가 있습니다. 전체 표현의 정오 판정은 아닙니다.'
        elif 'RELATED_PAIR_ORDER_UNRESOLVED' in flags:
            msg+='관련 동작의 앞뒤 근거가 겹치거나 부족해 순서를 확정하지 못했습니다.'
        elif 'NO_DISTINCT_PAIR_QUERY' in flags:
            msg+='반복 과정과 횟수를 판정할 순서 비교는 아직 없습니다.'
        else:
            msg+='일부 앞뒤 관계는 참고에서 본 순서로 관측됐습니다. 빠진 동작이나 전체 수행 완료를 보증하지 않습니다.'
        if 'OBSERVATION_GAP_RETAINED' in flags:
            msg+=' 보이지 않은 시간은 그대로 남겨 두었습니다.'
        events.append(dict(guide_id=obs['guide_id'], target_index=obs['target_index'],
            motion=deepcopy(obs), related_order_item_indices=linked, flags=flags, message_ko=msg,
            link_semantics='Shared guide identity only; distinct scopes/estimates do not establish the same occurrence or full source clause',
            motion_scope_family=identity['motion_scope_family'],
            exact_event_boundary=False, whole_action_complete=None, missing_action=None,
            educational_error=None, correction_enabled=False))
    payload=dict(version=VERSION, identity=deepcopy(identity),
        source_comparisons=deepcopy(comparisons), partial_order_items=deepcopy(orders), motion_events=events,
        order_frame_semantics='input index into native_frame_ids/timestamp_us; not source frame ID',
        global_image_unavailable_frames=deepcopy(local['image_unavailable_frames']),
        native_denominator=len(frames), selected_word_does_not_change_observations=True,
        no_score_averaging=True, no_scope_expansion=True, no_missing_frame_removal=True,
        normality=None, whole_sign_correctness=None, all_requirements_verified=False,
        wrong_handshape_in_applicable_context_validated=False,
        source_local_applicability= 'SOURCE_VARIANT_AND_LOCAL_PHASE_NOT_CONFIRMED',
        new_training=0, new_model_forwards=0, new_DTW=0, correction_enabled=False)
    out=deepcopy(base);out['combined_evidence_feedback']=payload
    require(digest({k:v for k,v in out.items() if k!='combined_evidence_feedback'})==digest(base), 'Old result changed')
    return out
