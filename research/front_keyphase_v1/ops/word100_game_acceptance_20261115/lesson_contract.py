"""Application-level evidence contract. No new recognition or correctness model.

Acceptance requires explicit positive task completion, never merely no warning.
This module does not set completion_verified from word identity or duration.
"""
AGREE={'measured_features_agree','measured_features_and_goal_agree'}
def assess(result,required_ids):
 cards=result.get('qualified_evidence',result).get('joint_cards',[])
 by={c['card']:c for c in cards};required=list(required_ids)
 details=[]
 for name in required:
  c=by.get(name)
  if c is None:details.append(dict(card=name,status='check',reason='missing_evidence'));continue
  if c.get('state')=='attributable_feature_difference':
   details.append(dict(card=name,status='correct',reason='observed_difference',differences=c.get('differences',[]),order=c.get('order_evidence',[])));continue
  if c.get('state')not in AGREE:
   details.append(dict(card=name,status='check',reason=c.get('state','missing_evidence')));continue
  orders=c.get('order_evidence',[])
  if any(e.get('state')!='forward'for e in orders):
   details.append(dict(card=name,status='check',reason='order_unconfirmed'));continue
  # A validated completion witness must be populated by the observation path.
  witness=c.get('completion_evidence',{})
  complete=witness.get('verified')is True and bool(witness.get('source_spans')) and bool(witness.get('requirements'))
  details.append(dict(card=name,status='pass'if complete else'match',reason='required_content_completed'if complete else'features_match_completion_unconfirmed'))
 status='correct'if any(d['status']=='correct'for d in details)else'pass'if required and all(d['status']=='pass'for d in details)else'check'
 return dict(status=status,can_advance=status=='pass',features_match=bool(required)and all(d['status']in ['pass','match']for d in details),cards=details,whole_sign_correct=None,basis='Explicit declared lesson requirements; not unrestricted KSL correctness')
