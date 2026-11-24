"""Evidence-keyed, short Korean drafts; never infer unobserved mistakes."""
SHAPE={0:('오른손','검지 모양'),1:('오른손','네 손가락 펴기'),2:('왼손','네 손가락 펴기'),3:('오른손','네 손가락 접기'),4:('왼손','네 손가락 접기')}
def draft(card,instruction):
    result=[]
    for i,e in enumerate(card['differences']):
        if e['task']=='shape' and e['feature']in SHAPE and e['expected']==1:
            hand,feature=SHAPE[e['feature']]
            result.append(dict(kind='observed_difference',text=f'{hand}의 {feature} 부분을 다시 확인해 보세요.',evidence=f'differences/{i}',span=e['span'],hand=hand))
        elif e['task']=='direction':
            result.append(dict(kind='observed_difference',text=f'이동 방향이 참고와 반대로 관측됐어요. “{instruction}”의 방향을 확인해 보세요.',evidence=f'differences/{i}',span=e['span']))
    for i,pair in enumerate(card['order_evidence']):
        if pair['state']=='reverse' and card['state']=='attributable_feature_difference':result.append(dict(kind='observed_order',text='표시한 두 행동이 반대 순서로 연결됐어요. 순서를 확인해 보세요.',evidence=f'order_evidence/{i}',spans=pair['witness_spans'],whole_sign_error=False))
    if result:return result
    state=card['state']
    if state=='measured_features_and_goal_agree':text='손을 접으며 얼굴 앞에서 상체 쪽으로 옮기는 흐름이 관측됐어요.';kind='observed_agreement'
    elif state=='measured_features_agree':text='확인 가능한 손모양·움직임은 설명과 비슷해요.';kind='observed_agreement'
    elif state=='goal_pose_without_observed_start':text='끝 자세는 보이지만, 그 자세로 바뀌는 과정은 확인하지 못했어요.';kind='uncertain'
    elif state in ['action_link_uncertain','component_link_uncertain']:text='이 움직임과 설명을 아직 확실히 연결하지 못했어요.';kind='uncertain'
    elif state=='unimplemented_sequence_requirement':text='손모양은 확인했지만, 반복 동작 전체는 아직 판정하지 못해요.';kind='unsupported'
    elif state in ['joint_goal_unconfirmed','final_configuration_unconfirmed']:text='설명에서 요구한 마지막 손모양·위치까지는 확실하지 않아요.';kind='uncertain'
    else:text='이 구간은 근거가 부족해 교정 여부를 보류했어요.';kind='uncertain'
    return [dict(kind=kind,text=text,evidence='joint_cards/state',whole_sign_correct=False)]
