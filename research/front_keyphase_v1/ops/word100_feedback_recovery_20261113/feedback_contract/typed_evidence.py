"""Keep physical observations separate from description membership.

Offline integration contract only. This does not infer, align, classify sign
correctness, or change the current web. Missing support must not become a hold.
"""

def explain_window(physical,clause,required=None):
    result={'physical':'unknown','clause':'unknown','completion':'not_established',
            'educational_correctness':None,'can_issue_correction':False}
    for source in [physical,clause]:
        if not isinstance(source,dict):raise ValueError('Typed evidence object required')
    clip=physical.get('clip_id');digest=physical.get('source_sha256')
    if (not isinstance(clip,str) or not clip or not isinstance(digest,str) or
        len(digest)!=64 or any(x not in '0123456789abcdef' for x in digest) or
        physical.get('hand') not in ['L','R']):
        result['reason']='missing_source_or_hand_identity';return result
    result['clip_id']=clip;result['source_sha256']=digest;result['hand']=physical['hand']
    interval=physical.get('native_interval_us')
    if not isinstance(interval,list) or len(interval)!=2 or not all(isinstance(x,int) and not isinstance(x,bool) for x in interval) or interval[0]<0 or interval[1]<=interval[0]:
        result['reason']='invalid_native_window';return result
    result['native_interval_us']=interval
    if physical.get('attribute')!='window_contains_torso_relative_wrist_translation':
        result['reason']='unsupported_physical_attribute';return result
    if physical.get('validated') is True and physical.get('observed') is True and physical.get('state') in ['M','H']:
        result['physical']='wrist_translation_present' if physical['state']=='M' else 'wrist_location_held'
    # Membership is independent: not_this_clause never overwrites physical M/H.
    same=(clause.get('native_interval_us')==interval and clause.get('hand')==physical.get('hand')
          and clause.get('clip_id')==clip and clause.get('source_sha256')==digest)
    clause_id=clause.get('clause_id')
    if same and isinstance(clause_id,str) and clause_id and clause.get('validated') is True and clause.get('state') in ['this_clause','not_this_clause']:
        result['clause']=clause['state'];result['clause_id']=clause_id
    result['reason']='separate_observations_not_action_completion'
    if required is not None:
        if not isinstance(required,dict):raise ValueError('Typed requirement object required')
        # Required motion/hold may define which observation is relevant, but is
        # never substituted for observed reality or forced into temporal order.
        result['requested_requirement']=required.get('attribute')
        result['requirement_supported']=(result['clause']=='this_clause' and
            required.get('clause_id')==result.get('clause_id') and
            isinstance(required.get('clause_id'),str) and bool(required['clause_id']) and
            ((required.get('attribute')=='wrist_translation' and result['physical']=='wrist_translation_present') or
             (required.get('attribute')=='wrist_location_hold' and result['physical']=='wrist_location_held')))
        result['reason']='local_component_only_not_full_stage_or_word_completion'
    return result
