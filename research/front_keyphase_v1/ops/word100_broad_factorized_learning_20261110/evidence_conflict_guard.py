"""Keep conflicting direction observations uncertain, not a confident correction.
Neither source is treated as independent ground truth. No word-specific rules.
"""
import copy
def apply(result):
 out=copy.deepcopy(result);changed=[]
 for card in out['cards']:
  m=card.get('motion_evidence')
  if not m:continue
  old=m.get('old_geometry_state');new=m.get('state')
  if {old,new}!={'supported','contradicted'}:continue
  m.update(pre_conflict_state=new,state='unknown',evidence_conflict=True)
  card['difference_components']=[s for s in card.get('difference_components',[]) if s!='direction']
  card['checked_components']=[s for s in card.get('checked_components',[]) if s!='팔을 따른 이동 방향']
  bad=[s for s in card.get('shape_evidence',[]) if s['state']=='contradicted']
  card['difference_spans']=[span for s in bad for span in s['contradicted_spans']]
  if card.get('observation_quarantine_overlap'):
   card.update(status='unknown',display_status='uncertain',message='손 관측이 부족하여 이 구간은 확인하기 어려워요.')
  elif bad:
   card.update(status='difference_candidate',display_status='difference',message='손모양 차이가 보여요. 이동 방향은 근거가 서로 달라 다시 확인이 필요해요.')
  else:
   card.update(status='unknown',display_status='uncertain',message='이동 방향의 근거가 서로 달라 확인하기 어려워요. 틀렸다는 뜻은 아니에요.')
  card['candidate_message']=card['message'];card['educational_wrong']=False;changed.append(card['id'])
 out['direction_conflict_guard']=dict(changed=changed,posthoc_safety_only=True,not_accuracy_gain=True)
 return out
