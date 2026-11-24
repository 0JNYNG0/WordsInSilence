LABELS={
 'sweep_shape':('편 손 준비','팔을 쓰는 손의 모양이 달라 보여요.','오른손 손가락을 펴서 반대쪽 팔에 놓아 주세요.',['R']),
 'sweep_direction':('팔을 따라 손목 쪽으로','팔을 쓰는 방향이 반대로 보여요.','팔 위쪽에서 손목 쪽으로 쓸어 주세요.',['R','L']),
 'sweep_endpoint':('손목 쪽까지 이어 하기','손목 쪽까지 이어진 모습을 확인하지 못했어요.','손목 쪽까지 쓴 다음 두 주먹을 준비해 주세요.',['R','L']),
 'left_fist':('왼손 구부리기','위아래 동작에서 왼손 손가락이 펴져 보여요.','왼손 손가락을 구부려 주먹을 유지해 주세요.',['L']),
 'right_fist':('오른손 구부리기','위아래 동작에서 오른손 손가락이 펴져 보여요.','오른손 손가락을 구부려 주먹을 유지해 주세요.',['R']),
 'repeat_motion':('두 손 위아래로','두 손의 반복 움직임을 충분히 확인하지 못했어요.','두 주먹을 나란히 두고 위아래로 두 번 움직여 주세요.',['L','R']),
 'order':('두 동작 이어 하기','두 동작의 연결을 확인하지 못했어요.','팔을 쓴 다음 두 주먹을 위아래로 움직여 주세요.',['R','L'])}
def present(q,ref):
 ts=q['timestamp_us'];cards=[];cues=[];notes=[]
 for c in q['criteria']:
  title,message,action,hands=LABELS[c['id']];status=c['status'];spans=c['difference_spans']if status=='difference'else c['source_spans'];display={'supported':'similar','difference':'difference','unknown':'uncertain'}[status]
  if status=='supported':message='이 부분의 동작 특징을 확인했어요.'
  if status=='unknown':message='이 부분은 확인이 부족해요. 틀렸다는 뜻은 아니에요.'
  key='sweep'if c['id'].startswith('sweep_')else'whole'if c['id']=='order'else'fists';a=min((s[0]for s in spans),default=None);b=max((s[1]for s in spans),default=None)
  cards.append(dict(id=c['id'],title=title,instruction=action,hands=hands,reference=ref[key],display_status=display,status=status,message=message,query_spans=spans,query_indices=[a,b]if a is not None else None,query_seconds=[ts[a]/1e6,ts[b]/1e6]if a is not None else None,difference_spans=c['difference_spans'],whole_action_complete=status=='supported',educational_wrong=False))
  notes.append(dict(card_id=c['id'],status=display,message=message))
  if status=='difference':
   a,b=c['difference_spans'][0];cues.append(dict(kind='shape'if'shape'in c['id']or'fist'in c['id']else'direction',card_ids=[c['id']],start_seconds=ts[a]/1e6,end_seconds=ts[b]/1e6,observation=message,action=action,hands=hands,spans=c['difference_spans'],evidence='Observed learned features in a guided prototype, single public reference'))
 summary='연습의 두 동작을 확인했어요. 팔을 쓴 뒤 두 주먹을 위아래로 움직였어요.'if all(c['status']=='supported'for c in q['criteria'])else cues[0]['observation']+' '+cues[0]['action']if cues else'회색 부분은 확인이 더 필요해요. 안내를 보고 다시 해 보세요.'
 return cards,dict(summary=summary,cues=cues,notes=notes,whole_sign_correct=None)
