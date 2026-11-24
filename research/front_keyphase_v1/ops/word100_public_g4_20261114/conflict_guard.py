"""A disagreement between independent action readings cannot be shown as agreement.

Preserve each hypothesis, span and difference. Do not vote, relabel, rescore,
search for a desired action, or modify model weights. This is a presentation
qualification, not an improved action recognizer.
"""
import copy
def signatures(card):
 out={}
 for e in card['observations']:
  if e['task']=='shape':key=('shape',e.get('feature'),e.get('expected'));sign={'supported':1,'difference':-1}.get(e['state'])
  elif e['task']=='direction':key=('direction',);sign={'supported':1,'opposite_direction':-1}.get(e['state'])
  else:continue
  if sign is not None:out.setdefault(key,set()).add(sign)
 for e in card['order_evidence']:
  sign={'forward':1,'reverse':-1}.get(e['state'])
  if sign is not None:out.setdefault(('order',*e['cards']),set()).add(sign)
 return out
def qualify(seed_rows,reference_seed=42):
 if sorted(r['seed']for r in seed_rows)!=[42,43,44]:raise ValueError('All fixed three runs required')
 base=next(r for r in seed_rows if r['seed']==reference_seed);result=copy.deepcopy(base)
 others={r['seed']:{c['card']:c for c in r['joint_cards']}for r in seed_rows}
 for c in result['joint_cards']:
  union={};evidence=[]
  for seed,cc in others.items():
   other=cc[c['card']];evidence.append(dict(seed=seed,state=other['state'],observations=copy.deepcopy(other['observations']),order_evidence=copy.deepcopy(other['order_evidence']),differences=copy.deepcopy(other['differences'])))
   # An unbound measurement must not veto an otherwise attributable result.
   if other['state']=='action_link_uncertain':continue
   for k,v in signatures(other).items():union.setdefault(k,set()).update(v)
  conflicts=[list(k)for k,v in union.items()if v=={-1,1}]
  c['cross_run_evidence']=evidence;c['conflicts']=conflicts
  if conflicts:
   c['state_before_conflict_guard']=c['state'];c['state']='evidence_conflict';c['drafts_before_conflict_guard']=c.get('drafts',[])
   c['drafts']=[dict(kind='uncertain',text='이 구간의 동작 판단이 엇갈려요. 맞거나 틀렸다고 확정하지 않을게요.',evidence='cross_run_evidence',whole_sign_correct=False)]
  c['show_as_correct_sign']=False;c['whole_action_complete']=None
 result['conflict_guard_applied']=True;result['reference_seed_fixed_before_G4']=42
 return result
