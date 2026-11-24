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
    frames = identity['native_frame_ids']; times = identity['timestamp_us']
    require(len(frames) == len(times) == local['native_frames'] == base['frames'], 'Axis length differs')
    require(all(type(v) is int and v >= 0 for v in frames + times), 'Native axes must be integers')
    require(all(b > a for a,b in zip(frames,frames[1:])) and all(b > a for a,b in zip(times,times[1:])), 'Axes not monotone')
    pos = {v:i for i,v in enumerate(frames)}
    def time_check(ff, seconds):
        require(all(f in pos for f in ff), 'Evidence outside native input')
        require(len(ff) == len(seconds), 'Evidence times differ')
        require(all(abs(t-times[pos[f]]/1e6) < 1e-9 for f,t in zip(ff,seconds)), 'Evidence not at original time')
    comparisons = base['guide_feature_comparison']['items']
    orders = base['partial_order_feedback']['items']
    stage_ids = {s['guide_id'] for s in base['stages']}
    for item in comparisons:
        for p in item.get('native_points', []):
            time_check([p['frame']], [p['seconds']])
    for item in orders:
        for key in ['first_evidence','second_evidence']:
            e=item[key]; time_check(e['input_frames'],e['native_seconds'])
    events=[]
    for obs in local['observations']:
        require(obs['guide_id'] in stage_ids, 'Motion guide missing')
        scope = obs['fixed_scope']; ff=scope['frame_ids']; tt=scope['timestamp_us']
        require(ff == frames[pos[ff[0]]:pos[ff[-1]]+1], 'Motion scope omitted input frames')
        require(tt == [times[pos[f]] for f in ff], 'Motion source axis differs')
        require(obs['frames'] == [ff[0],ff[-1]], 'Motion bracket differs')
        require(obs.get('whole_action_complete') is None and obs.get('educational_error') is None,
                'Occurrence cannot be a correctness result')
        linked=[]
        for i,order in enumerate(orders):
            gids=order['first']['guide_ids']+order['second']['guide_ids']
            if obs['guide_id'] in gids:
                linked.append(i)
        flags=[]
        if obs['high_confirmation']:flags.append('MOVEMENT_EVIDENCE_PRESENT')
        if obs['unavailable_native_frames']:flags.append('OBSERVATION_GAP_RETAINED')
        if any(orders[i]['observation_status']=='OBSERVED_B_BEFORE_A' for i in linked):
            flags.append('RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER')
        if any(orders[i]['observation_status'] not in {'OBSERVED_A_BEFORE_B','OBSERVED_B_BEFORE_A'} for i in linked):
            flags.append('RELATED_PAIR_ORDER_UNRESOLVED')
        if not linked:flags.append('NO_DISTINCT_PAIR_QUERY')
        # No conjunctive "passed" label: existence, sequence and missingness
        # answer different questions. In particular there is no absence rule.
        if 'RELATED_PAIR_OBSERVED_IN_REVERSE_ORDER' in flags:
            msg='움직임 근거와 함께, 관련된 두 동작이 참고와 반대 순서로 관측된 근거가 있습니다. 전체 표현의 정오 판정은 아닙니다.'
        elif 'RELATED_PAIR_ORDER_UNRESOLVED' in flags:
            msg='움직임 여부와 별개로 관련 동작의 앞뒤가 겹치거나 부족해 순서를 확정하지 못했습니다.'
        elif 'NO_DISTINCT_PAIR_QUERY' in flags:
            msg='이 움직임은 별도로 표시합니다. 반복 과정과 횟수를 판정할 순서 비교는 아직 없습니다.'
        else:
            msg='관련 움직임과 일부 앞뒤 관계를 각각 확인했습니다. 빠진 동작이나 전체 수행 완료를 보증하지 않습니다.'
        if 'OBSERVATION_GAP_RETAINED' in flags:
            msg+=' 보이지 않은 시간은 그대로 남겨 두었습니다.'
        events.append(dict(guide_id=obs['guide_id'], target_index=obs['target_index'],
            motion=deepcopy(obs), related_order_item_indices=linked, flags=flags, message_ko=msg,
            motion_scope_family=identity['motion_scope_family'],
            exact_event_boundary=False, whole_action_complete=None, missing_action=None,
            educational_error=None, correction_enabled=False))
    payload=dict(version=VERSION, identity=deepcopy(identity),
        source_comparisons=deepcopy(comparisons), partial_order_items=deepcopy(orders), motion_events=events,
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
