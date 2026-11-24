"""Preserve precise uncertainty locations without loading side-view models."""
from presentation import present as base_present

def present(q,reference_span):
 cards,coaching=base_present(q,reference_span);by={c['id']:c for c in q['criteria']};ts=q['timestamp_us']
 for card in cards:
  c=by[card['id']]
  if c['status']=='unknown':
   spans=c.get('uncertain_spans')or c.get('unconfirmed_events')
   if spans:
    a=min(s[0]for s in spans);b=max(s[1]for s in spans)
    card.update(query_spans=spans,query_indices=[a,b],query_seconds=[ts[a]/1e6,ts[b]/1e6],uncertainty_location='observed_unconfirmed_interval')
  elif c['status']=='difference'and c['kind']=='shape':
   hand='오른손'if c['hand']=='R'else'왼손';message=f'이 구간의 {hand} 모양이 펼친 손과 다르게 보여요.';card['message']=message
   for note in coaching['notes']:
    if note['card_id']==c['id']:note['message']=message
   for cue in coaching['cues']:
    if c['id']in cue['card_ids']:cue['observation']=message
 if coaching['cues']:coaching['summary']=coaching['cues'][0]['observation']+' '+coaching['cues'][0]['action']
 return cards,coaching
