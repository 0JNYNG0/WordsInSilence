LABELS={
 'mouth_motion':('입술 아래 스치기','입술 아래에서 손이 반대 방향으로 움직인 구간이 보여요.','검지를 입술 아래에서 오른쪽으로 스쳐 주세요.',['R']),
 'index_shape':('검지 펴기','입술 아래 동작의 손가락 모양이 달라 보여요.','검지를 펴고 중지·약지·새끼손가락은 접어 주세요.',['R']),
 'support_shape':('왼손 주먹 준비','받치는 왼손의 모양이 주먹과 달라 보여요.','왼손은 주먹을 쥐고 준비해 주세요.',['L']),
 'wipe_shape':('오른 손바닥 준비','닦는 오른손의 모양이 손바닥을 펴는 모습과 달라 보여요.','오른손 손가락을 펴서 왼 주먹을 감싸듯 놓아 주세요.',['R']),
 'wipe_motion':('아래로 닦기','닦는 손이 아래가 아닌 반대 방향으로 움직였어요.','오른 손바닥으로 왼 주먹을 위에서 아래로 닦아 주세요.',['R','L']),
 'wipe_terminal':('닦기 마무리','닦는 동작의 마지막 모습을 확인하지 못했어요.','손바닥으로 왼 주먹을 닦아 내린 뒤 마무리해 주세요.',['R','L']),
 'sequence':('두 동작 이어 하기','두 동작의 순서가 참고와 달라요.','입술 아래를 스친 다음, 왼 주먹을 닦아 주세요.',['R','L'])}
def present(q,reference):
 ts=q['timestamp_us'];cards=[];cues=[];notes=[]
 for c in sorted(q['criteria'],key=lambda c:list(LABELS).index(c['id'])):
  title,message,action,hands=LABELS[c['id']];spans=c['difference_spans']if c['status']=='difference'else c['source_spans'];diff=c['difference_spans']
  if c['status']=='supported':status='similar';message='이 부분의 필요한 특징을 확인했어요.'
  elif c['status']=='difference':status='difference'
  else:status='uncertain';message='이 부분은 확인이 부족해요. 틀렸다는 뜻은 아니에요.'
  key='mouth'if c['id']in['index_shape','mouth_motion']else'whole'if c['id']=='sequence'else'wipe';ref=reference[key]
  a=min((s[0]for s in spans),default=None);b=max((s[1]for s in spans),default=None)
  cards.append(dict(id=c['id'],title=title,instruction=action,hands=hands,reference=ref,display_status=status,status=c['status'],message=message,query_spans=spans,query_indices=[a,b]if a is not None else None,query_seconds=[ts[a]/1e6,ts[b]/1e6]if a is not None else None,difference_spans=diff,whole_action_complete=c['status']=='supported',educational_wrong=False))
  notes.append(dict(card_id=c['id'],status=status,message=message))
  if diff:
   a,b=diff[0];cues.append(dict(kind='shape'if'shape'in c['id']else'order'if c['id']=='sequence'else'direction',card_ids=[c['id']],start_seconds=ts[a]/1e6,end_seconds=ts[b]/1e6,observation=message,action=action,hands=hands,spans=diff,evidence='Observed source-linked features; no DTW-filled completion.'))
 summary='검지 동작과 손바닥으로 닦는 동작을 순서대로 확인했어요. 잘 따라 했어요!'if all(c['status']=='supported'for c in q['criteria'])else cues[0]['observation']+' '+cues[0]['action']if cues else'회색으로 표시된 부분은 확인이 더 필요해요. 안내 동작을 보고 다시 해 보세요.'
 return cards,dict(summary=summary,cues=cues,notes=notes,whole_sign_correct=None)
