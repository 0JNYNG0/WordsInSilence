"""Keep separate observations separate; do not turn uncertainty into failure."""
from core import assess

def describe_component(states,available,ts,requirement,association='confirmed'):
    result=assess(states,available,ts,requirement,association)
    status=result['status']
    texts={
     'required_feature_sequence_observed':'손을 펴다가 접는 변화가 확인됐어요.',
     'still_at_start_state':'손을 펼친 모습은 보이지만, 접는 변화는 아직 확인되지 않았어요.',
     'goal_posture_without_start':'접힌 손은 보이지만, 펴진 손에서 바뀌는 과정은 확인되지 않았어요.',
     'reverse_feature_order':'이 구간에서는 접힌 손이 펴지는 순서로 관측됐어요.',
     'endpoint_pair_only':'앞뒤 손모양은 보이지만, 사이 동작은 충분히 보이지 않았어요.',
     'action_association_uncertain':'이 손모양 변화가 설명의 어느 행동인지 더 확인해야 해요.',
     'required_goal_not_confirmed':'손을 접는 마지막 모습은 아직 확실히 확인되지 않았어요.',
     'observation_uncertain':'손모양이 충분히 보이지 않아 판단을 보류했어요.'}
    # Text is bound to this specific feature query, not to a word ID.
    shape_query=requirement=={'start':{'shape':'four_open'},'goal':{'shape':'four_closed'}}
    return dict(evidence=result,text_ko=texts[status]if shape_query else None,
      category='observed'if result['feature_sequence_observed']else'needs_review',
      educational_error=None,whole_action_complete=None,display_enabled=False)

def combine(parts,additional_requirements):
    """No majority vote: supported subparts cannot erase unresolved requirements."""
    return dict(parts=parts,additional_requirements=additional_requirements,
      whole_action_complete=None,educational_error=None,comparison_retained=True,
      wording_policy='Confirm only the observed feature; never say the whole action was correct.',
      display_enabled=False)
