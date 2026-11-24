"""Source-grounded short wording. No model inference, thresholds, or deployment.

The caller supplies independently verified capability and event-binding decisions.
This module never derives an expected handshape from a word name.
"""
from math import isfinite

FINGER_KO={'thumb':'엄지','index':'검지','middle':'중지','ring':'약지','little':'새끼손가락'}
STATE_KO={'E':'펴진','B':'구부러진','F':'접힌'}
SUPPORTED_STATES={'thumb':{'E'},'index':{'E','B','F'},'middle':{'E','B','F'},'ring':{'E','B','F'},'little':{'E','B','F'}}

def _interval(v):
    if not isinstance(v,(list,tuple)) or len(v)!=2:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not isfinite(x) for x in v):return None
    if v[0]<0 or v[1]<v[0]:return None
    return list(v)

def format_feedback(observation,binding=None,validation=None):
    """Return observation / reference_match / reference_difference / unknown.

    No input probability is interpreted as correctness. `validation` defaults off.
    `synthetic_example` explicitly labels toy data and never bypasses any gate.
    """
    o=observation;v=validation or {};side=o.get('hand');finger=o.get('finger')
    side_ko={'R':'오른손','L':'왼손'}.get(side,'손');finger_ko=FINGER_KO.get(finger,'손가락');name=f'{side_ko} {finger_ko}'
    subject=name+('이' if finger=='little' else '가')
    topic=name+('은' if finger=='little' else '는')
    interval=_interval(o.get('native_interval_us'));state=o.get('state','U')
    result=dict(status='unknown',text=f'{name}의 모양은 확인하기 어려워요.',hint=None,highlight=None,evidence_interval_us=interval,educational_correctness=None,synthetic_example=bool(o.get('synthetic_example',False)),release_status='not_web_connected')
    def stop(reason):result['reason']=reason;return result
    if side not in ['L','R'] or finger not in FINGER_KO:return stop('invalid_hand_or_finger')
    if interval is None:return stop('invalid_native_interval')
    if o.get('attribute','finger_state')!='finger_state':return stop('unsupported_attribute')
    if state not in SUPPORTED_STATES[finger]:return stop('unknown_or_unsupported_state')
    if o.get('visibility')!='observed':return stop('visibility_uncertain')
    if o.get('evidence_type') not in ['direct_rgb','validated_model'] or not o.get('evidence_ids'):return stop('missing_direct_or_validated_evidence')
    if not v.get('attribute_validated',False) or state not in v.get('validated_states',[]):return stop('attribute_not_yet_validated')
    result.update(status='observation',text=f'{subject} {STATE_KO[state]} 것으로 보여요.',highlight={'hand':side,'fingers':[finger]},reason='observation_only')
    if not binding:return result
    if (o.get('word_id'),o.get('stage_id'),side,finger)!=(binding.get('word_id'),binding.get('stage_id'),binding.get('hand'),binding.get('finger')):return stop('binding_identity_mismatch')
    if not v.get('phase_binding_validated',False) or not v.get('reference_comparison_validated',False):return stop('comparison_or_phase_not_validated')
    if not binding.get('reference_anchors') or not binding.get('stage_interval_provenance'):return stop('missing_reference_provenance')
    if binding.get('comparison_scope')!='sparse_reference_anchor_not_mandatory_word_rule':return stop('unsupported_comparison_scope')
    target=binding.get('reference_state')
    if target not in SUPPORTED_STATES[finger] or target not in v.get('validated_states',[]):return stop('target_state_not_supported')
    phrase=binding.get('action_phrase')
    if not isinstance(phrase,str) or not phrase.strip():return stop('missing_action_phrase')
    if state==target:
        result.update(status='reference_match',text=f'“{phrase}” 동작에서 {topic} 참고 장면처럼 {STATE_KO[state]} 것으로 보여요.',reason='same_observed_component_not_whole_word_correctness')
    else:
        result.update(status='reference_difference',text=f'“{phrase}” 동작에서 {subject} 참고 장면과 다르게 {STATE_KO[state]} 것으로 보여요.',hint=f'참고 장면에서 {subject} {STATE_KO[target]} 모습을 다시 확인해 보세요.',reason='different_observed_component_not_educational_error')
    result['source_binding_id']=binding.get('binding_id');result['reference_anchor_provenance']=binding['reference_anchors'];result['source_clause_ids']=binding.get('clause_ids',[])
    return result
