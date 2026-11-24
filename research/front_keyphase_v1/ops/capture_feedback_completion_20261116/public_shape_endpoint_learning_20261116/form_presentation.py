"""Use hand highlighting and do not invent the identity of a bent finger."""
import copy
from repair_presentation import present as previous_present
def present(q,reference_span):
 view=copy.deepcopy(q)
 for c in view['criteria']:
  if c.get('kind')=='hand_form':c['kind']='shape'
 cards,coaching=previous_present(view,reference_span)
 for c,card in zip(q['criteria'],cards):
  if not c.get('inspection')or c['status']!='difference':continue
  hand='오른손'if c['hand']=='R'else'왼손';message=f'이 구간의 {hand} 모양이 펼친 손과 다르게 보여요.'
  card['message']=message
  for note in coaching['notes']:
   if note['card_id']==c['id']:note['message']=message
  for cue in coaching['cues']:
   if c['id']in cue['card_ids']:cue['observation']=message;cue['kind']='shape'
 if coaching['cues']:coaching['summary']=coaching['cues'][0]['observation']+' '+coaching['cues'][0]['action']
 return cards,coaching
