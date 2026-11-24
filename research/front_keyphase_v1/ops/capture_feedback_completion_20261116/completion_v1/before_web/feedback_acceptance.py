"""Game acceptance is stricter than a word-name match.

This contract never invents completion evidence. Only a qualified server-side
lesson evaluator can provide positive criterion results. Missing data is a hold.
"""

def feedback_gate(result,target):
    cards=result.get('qualified_evidence',{}).get('joint_cards',[])
    visible_cards=result.get('cards',[])
    if any(c.get('state')=='attributable_feature_difference' for c in cards) or any(c.get('display_status')=='difference' for c in visible_cards):
        return dict(accepted=False,reason='feedback_correction_required')
    verdict=result.get('lesson_acceptance')
    if not isinstance(verdict,dict):
        return dict(accepted=False,reason='feedback_completion_unconfirmed')
    if verdict.get('target')!=target or verdict.get('schema')!='observed_lesson_requirements_v1':
        return dict(accepted=False,reason='feedback_contract_mismatch')
    if verdict.get('qualified') is not True:
        return dict(accepted=False,reason='feedback_not_qualified')
    criteria=verdict.get('criteria',[])
    required=verdict.get('required_ids',[])
    if not isinstance(required,list) or not required or any(not isinstance(x,str) or not x for x in required) or len(required)!=len(set(required)):
        return dict(accepted=False,reason='feedback_missing_requirements')
    if not isinstance(criteria,list) or any(not isinstance(c,dict) for c in criteria):
        return dict(accepted=False,reason='feedback_invalid_criteria')
    ids=[c.get('id') for c in criteria]
    if any(not isinstance(x,str) for x in ids) or len(ids)!=len(set(ids)) or set(ids)!=set(required):
        return dict(accepted=False,reason='feedback_missing_requirements')
    if any(c.get('status')=='difference' for c in criteria):
        return dict(accepted=False,reason='feedback_correction_required')
    n=len(result.get('timestamp_us',[]))
    for c in criteria:
        if c.get('status')!='supported':
            return dict(accepted=False,reason='feedback_observation_unconfirmed')
        spans=c.get('source_spans')
        if not isinstance(spans,list) or not spans:
            return dict(accepted=False,reason='feedback_missing_source_span')
        for s in spans:
            if not isinstance(s,list) or len(s)!=2 or any(type(x) is not int for x in s) or not 0<=s[0]<=s[1]<n:
                return dict(accepted=False,reason='feedback_invalid_source_span')
    return dict(accepted=True,reason='observed_lesson_requirements_met')
