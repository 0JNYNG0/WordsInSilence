"""A separate identity-conflict guard, not a replacement for observed shape/motion intervals.

Absence of a confident core is not evidence of an unfinished action. A positive
conflicting identity can withhold similarity; it cannot pronounce educational error.
No word, performer, duration template, or expected answer is an input to the network.
"""
import copy
import numpy as np
GROUPS=[[0],[1],[2],[3],[4],[5,7],[6,8,9],[10],[11,12]]
MAP={s:g for g,ss in enumerate(GROUPS) for s in ss}
def has_run(mask,ts,min_ms=60):
 dt=float(np.median(np.diff(ts))) if len(ts)>1 else 0;start=None
 for i,on in enumerate(np.r_[mask,False]):
  if on and start is None:start=i
  elif not on and start is not None:
   if ts[i-1]-ts[start]+dt>=min_ms*1000:return True
   start=None
 return False
def apply_guard(decision,prob,raw,ts):
 out=copy.deepcopy(decision);threshold=.65
 hand=np.stack([raw[:,237:258].sum(1)>=5,raw[:,258:279].sum(1)>=5],1)
 for c in out['cards']:
  span=c.get('query_indices');audit=[]
  if span is not None:
   available=hand[:,[0 if h=='L' else 1 for h in c['hands']]].all(1)
   for e in c['phase_evidence']:
    s=e.get('span')
    if s is None:continue
    a,b=s;gs={MAP[g] for g in e['group']};other=[g for g in range(9) if g not in gs]
    expected=prob[:,[2+2*g for g in gs]].sum(1);alternatives=prob[:,[2+2*g for g in other]];idx=alternatives.argmax(1);otherprob=alternatives.max(1)
    support=available[a:b+1]&(otherprob[a:b+1]>=threshold);fraction=float(support.mean());consistent=has_run(support,ts[a:b+1]);own=has_run(available[a:b+1]&(expected[a:b+1]>=threshold),ts[a:b+1])
    conflict=bool(fraction>=threshold and consistent and not own)
    audit.append(dict(group=e['group'],span=s,other_core_fraction=fraction,own_core_observed=bool(own),conflict=conflict,other_core_ids=sorted({int(other[j]) for j in idx[a:b+1][support]})))
  c['completion_identity_audit']=audit
  if c.get('display_status')=='similar' and any(a['conflict'] for a in audit):
   c.update(previous_completion_status=c['status'],status='unknown',display_status='uncertain',uncertainty_reason='conflicting_observed_event_identity',message='이 부분이 앞뒤의 다른 동작으로 잡혀, 끝까지 했는지 아직 확인하기 어려워요.',educational_wrong=False)
   c['candidate_message']=c['message']
 out['completion_guard_version']='positive_identity_conflict_v1';out['completion_guard_is_not_full_correctness']=True
 return out
