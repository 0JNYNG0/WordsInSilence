"""Common observation-only order veto; does not change pose, direction, DTW or completion.
Caller supplies order evidence computed on these exact observed query windows.
"""
import copy
def apply(result,checks):
 out=copy.deepcopy(result);cards={c['id']:c for c in out['cards']};validated=[];changed=[]
 for source in checks:
  q=copy.deepcopy(source);pair=q.get('pair',[])
  if len(pair)!=2 or any(k not in cards for k in pair):
   q.update(state='unknown',eligible=False,reason='unmatched_action_pair');validated.append(q);continue
  a,b=(cards[k] for k in pair);sa=a.get('query_indices');sb=b.get('query_indices')
  expected=[min(sa[0],sb[0]),max(sa[1],sb[1])] if sa and sb else None
  if expected is None or q.get('span')!=expected:
   q.update(state='unknown',eligible=False,reason='absent_or_different_observation_window')
  if a.get('observation_quarantine_overlap') or b.get('observation_quarantine_overlap'):q.update(state='unknown',eligible=False,reason='insufficient_observation')
  if q['state']=='contradicted' and q.get('eligible') and b['display_status'] in ['similar','difference']:
   b.update(status='difference_candidate',display_status='difference',message='동작 순서가 참고와 다르게 보여요. 표시된 두 구간을 확인해 보세요.',educational_wrong=False);b['candidate_message']=b['message'];b.setdefault('difference_components',[])
   if 'order' not in b['difference_components']:b['difference_components'].append('order')
   changed.append(b['id'])
  validated.append(q)
 out['learned_order_checks']=validated;out['sequence_has_observed_order_conflict']=bool(changed);out['order_only_guard']=dict(changed=changed,not_completion_certificate=True);return out
