"""Use the existing cards, with ambiguity attached to its actual source interval."""
from presentation import present as previous_present

def present(q,reference_span):
 cards,coaching=previous_present(q,reference_span);ts=q['timestamp_us'];changed=False
 for c,card in zip(q['criteria'],cards):
  if c.get('uncertainty_reason')!='projection_observers_disagree' or c['status']!='unknown':continue
  spans=c['uncertain_spans'];a=spans[0][0];b=spans[-1][1];message='이 구간은 손가락을 펼쳤는지 굽혔는지 확실히 구분하지 못했어요.'
  card.update(message=message,instruction='손 전체가 잘 보이도록 같은 동작을 다시 보여 주세요.',query_spans=spans,query_indices=[a,b],query_seconds=[ts[a]/1e6,ts[b]/1e6],uncertainty_reason=c['uncertainty_reason'])
  for note in coaching['notes']:
   if note['card_id']==c['id']:note['message']=message
  changed=True
 if changed and not coaching['cues']:coaching['summary']='손가락 모양이 불확실한 구간이 있어요. 회색으로 표시한 부분을 확인해 주세요.'
 return cards,coaching
