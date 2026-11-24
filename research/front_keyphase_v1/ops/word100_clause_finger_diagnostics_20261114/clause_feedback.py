"""Evidence-to-clause contract. This is not a Korean sentence parser or an LLM.

The learned observer supplies action intervals and finger states. A reviewed
descriptor supplies requirements. Neither target shape nor sentence order may
select the observed interval. No word-specific conditions in this compositor.
"""
FINGER_KO={'thumb':'엄지','index':'검지','middle':'중지','ring':'약지','little':'새끼손가락'}
HAND_KO={'R':'오른손','L':'왼손'}
SUPPORTED={'index','middle','ring','little'}
def compose(action,finger_evidence,direction_evidence,association_known):
    cues=[];unknown=[]
    for e in finger_evidence:
        if e['finger']not in SUPPORTED or not association_known or e['state']in ['uncertain','observation_unknown','association_unknown','evidence_conflict']:
            unknown.append(dict(kind='finger',hand=e['hand'],finger=e['finger'],span=e['span'],reason='unconfirmed_observation_or_action_link'));continue
        if e['state']!='difference':continue
        required=next((q for q in action['finger_requirements']if q['hand']==e['hand']and q['finger']==e['finger']),None)
        if required is None:continue
        verb='펴 보세요'if required['expected']=='extended'else'굽혀 보세요'
        cues.append(dict(kind='finger',hand=e['hand'],finger=e['finger'],span=e['span'],clause_id=action['action_id']+':shape',
            instruction=action['display_instruction'],text=f"이 동작에서 {HAND_KO[e['hand']]} {FINGER_KO[e['finger']]}를 {verb}.",
            evidence=e,expected=required['expected'],whole_sign_correct=None))
    for e in direction_evidence:
        state=e.get('state',e.get('after'))
        if not association_known or state in ['uncertain','observation_unknown','association_unknown','evidence_conflict','movement_not_established']:
            unknown.append(dict(kind='direction',span=e.get('span'),reason='unconfirmed_direction_or_action_link'));continue
        if state=='opposite_direction':
            cues.append(dict(kind='direction',span=e['span'],clause_id=action['action_id']+':movement',instruction=action['display_instruction'],
                text='움직인 방향을 확인해 보세요. '+action['display_instruction'],evidence=e,whole_sign_correct=None))
    return dict(action_id=action['action_id'],cues=cues,unknown=unknown,whole_action_complete=None,whole_sign_correct=None)
