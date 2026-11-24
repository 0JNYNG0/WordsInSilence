"""Goal observations cannot substitute for action, shape or linguistic validity."""
import copy
def summarize_component(evidence,available,labels,domain):
    e=copy.deepcopy(evidence);visible=list(available);states=list(labels)
    status=e['status'];reason=None
    goal={'forearm':'팔의 도달 자세','arm_region':'반대팔의 손목 쪽 위치',
          'separation':'양손이 벌어진 모습','height':'양손 높이가 바뀌는 모습'}[domain]
    if status=='ordered_state_change_observed':
        text=f'{goal}까지 이어지는 변화가 관측됐어요.'
    elif status=='observed_before_goal':
        text=f'{goal}는 아직 확인되지 않았어요. 끝부분을 함께 확인해 보세요.'
        reason='observed_state_before_goal_not_educational_error'
    elif status=='goal_pose_only':
        text='마지막 자세는 보이지만, 앞 동작에서 이어졌는지는 확인하지 못했어요.'
        reason='goal_snapshot_without_required_state_change'
    elif status=='order_not_supported':
        text='관측된 자세의 앞뒤 순서를 다시 확인해야 해요.'
        reason='observed_state_order_difference'
    else:
        if not visible or not any(visible)or not visible[-1]:
            reason='missing_required_landmarks';text='끝부분의 손·팔 관측이 부족해 확인을 보류했어요.'
        else:
            reason='state_classifier_uncertain';text='손·팔은 관측됐지만 끝 자세를 확실히 구분하지 못했어요.'
    return dict(status=status,reason=reason,text_ko=text,component_goal_observed=e['goal_observed'],
        ordered_component_change=e['ordered_transition_observed'],semantic_completion=None,educational_error=None,
        shape_correctness=None,trajectory_correctness=None,required_repetitions=None,
        localization_may_be_deleted=False,localized_span_may_be_replaced=False,display_enabled=False,
        scope='Research component observation. Existing shape/direction comparison remains separate; no whole-sign completion.')
