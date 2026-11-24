"""Text is tied to observed clauses, not generated or learned language."""
LABELS={
 'right_shape':('오른손 펼치기','오른손 손가락이 구부러진 구간이 보여요.','오른손의 손가락을 펴서 유지해 주세요.',['R']),
 'left_shape':('왼손 펼치기','왼손 손가락이 구부러진 구간이 보여요.','왼손의 손가락을 편 채 받쳐 주세요.',['L']),
 'right_orientation':('오른손 세우기','오른손을 세운 모양과 다른 구간이 보여요.','오른손을 모로 세워 손 옆면이 아래를 향하게 해 주세요.',['R']),
 'left_orientation':('왼손 눕히기','왼손을 눕힌 모양과 다른 구간이 보여요.','왼손을 펴고 손등이 위를 향하도록 눕혀 주세요.',['L']),
 'relative_target':('왼 손등 쪽에 닿기','손의 도착 위치를 확인해 주세요.','오른손 옆면을 왼 손등 쪽으로 가져가 주세요.',['R','L']),
 'repeat_order':('두 번 이어 하기','두 번 이어지는 동작을 모두 확인하지 못했어요.','한 번 댄 다음 떼고, 같은 위치에 한 번 더 대 주세요.',['R','L'])}

def present(q,reference_span):
 ts=q['timestamp_us'];cards=[];cues=[];notes=[]
 for c in q['criteria']:
  title,message,action,hands=LABELS[c['id']];spans=c.get('source_spans',[]);diff=c.get('difference_spans',[])
  if c['status']=='difference':spans=diff or spans;status='difference'
  elif c['status']=='supported':message='이 부분의 필요한 특징을 확인했어요.';status='similar'
  else:
   status='uncertain';message='이 부분은 확인이 부족해요. 틀렸다는 뜻은 아니에요.'
   if c['id']=='repeat_order':message=f"반복 동작은 {c['observed_count']}회 확인했어요. 두 번 모두 했는지는 확인이 더 필요해요."
  a=min((s[0]for s in spans),default=None);b=max((s[1]for s in spans),default=None)
  cards.append(dict(id=c['id'],title=title,instruction=action,hands=hands,reference=reference_span,
   display_status=status,status=c['status'],message=message,query_spans=spans,query_indices=[a,b]if a is not None else None,
   query_seconds=[ts[a]/1e6,ts[b]/1e6]if a is not None else None,difference_spans=diff,
   whole_action_complete=c['status']=='supported',educational_wrong=False))
  notes.append(dict(card_id=c['id'],status=status,message=message))
  if diff:
   a,b=diff[0];cues.append(dict(kind='shape'if c['kind']=='shape'else'orientation',card_ids=[c['id']],
    start_seconds=ts[a]/1e6,end_seconds=ts[b]/1e6,observation=message,action=action,hands=hands,spans=diff,
    evidence='Source-validated neural observations in the native recording; not a DTW-filled event.'))
 all_supported=all(c['status']=='supported'for c in q['criteria'])
 summary='손모양과 두 번의 동작을 확인했어요. 잘 따라 했어요!'if all_supported else(cues[0]['observation']+' '+cues[0]['action']if cues else'확인이 부족한 부분이 있어요. 회색 구간을 보고 다시 해 보세요.')
 return cards,dict(summary=summary,notes=notes,cues=cues,whole_sign_correct=None)
