"""Isolated output view. Does not change the frozen comparison or web service.

Old probability vectors must not be presented as if they produced an updated
verdict. Keep their provenance, and bind current text to resolved evidence.
"""
import copy

BLOCKED={'evidence_conflict','action_link_uncertain','component_link_uncertain'}

def prepare(result,specs,draft):
    out=copy.deepcopy(result)
    for card in out['joint_cards']:
        for e in card['observations']:
            if e['task']=='direction' and 'supplementary_motion'in e:
                e['baseline_probabilities']=e.pop('probabilities',None)
                e['resolution_source']='three_shared_motion_models' if 'baseline_state'in e else 'baseline_retained'
                e['resolved_probabilities_by_seed']=[{'seed':p['seed'],'probabilities':p.get('probabilities'),'span':p.get('span')}for p in e['supplementary_motion']]
        # Display only explicitly attributed evidence. A disagreement is gray.
        if card['state']in BLOCKED:
            card['presentation']={'tone':'uncertain','messages':[{'kind':'uncertain','text':'이 구간은 판단이 엇갈리거나 설명과의 연결이 불확실해, 교정을 보류했어요.','evidence':'joint_cards/state'}]}
        else:
            instruction=next(s['instruction']for s in specs if s['id']==card['card'])
            card['presentation']={'tone':'difference'if card['state']=='attributable_feature_difference'else'uncertain'if card['state']not in ['measured_features_agree','measured_features_and_goal_agree']else'observed_agreement','messages':draft(card,instruction)}
        card['presentation']['whole_sign_correct']=None
        card['presentation']['new_individual_finger_feedback_enabled']=False
    out['presentation_scope']='isolated public replay; not deployed to web'
    return out
