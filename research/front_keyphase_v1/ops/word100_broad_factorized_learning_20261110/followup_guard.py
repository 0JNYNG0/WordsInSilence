from core import *
from evidence_conflict_guard import apply
def main():
 assert not (U/'CONFLICT_GUARD_CHECK.json').exists()
 write(U/'CONFLICT_GUARD_FROZEN.json',dict(purpose='After fixed evaluation, prevent a learned positive direction claim where old geometry strongly disagrees. Conflict becomes unknown, not opposite truth. No retraining, thresholds, seed choice or baseline result rewrite.',input=str(U/'FULLPATH_COMPLETE.json'),input_sha256=sha(U/'FULLPATH_COMPLETE.json'),code_sha256=sha(U/'evidence_conflict_guard.py'),scope='Saved outputs only; isolated prototype, not deployed',independent_validation=False))
 rows=read(U/'FULLPATH_COMPLETE.json')['rows'];out=[];checks=0
 for r in rows:
  d=read(r['result']);f=apply(d);assert d==read(r['result']);checks+=1
  for key in ['path','frame_id','tcn_top3']:assert d.get(key)==f.get(key);checks+=1
  n=len(f['direction_conflict_guard']['changed']);changed=[]
  for a,b in zip(d['cards'],f['cards']):
   if b['id'] in f['direction_conflict_guard']['changed']:
    assert b['motion_evidence']['state']=='unknown';assert 'direction' not in b['difference_components'];assert b['display_status']!='similar';checks+=3;changed.append(b)
   else:assert a==b;checks+=1
  if r['kind'] in ['full_normal','speed075','speed125','all_hand_missing','omit_final_event','mask_final_event']:assert n==0;checks+=1
  out.append(dict(model=r['model'],clip_id=r['clip_id'],kind=r['kind'],changed_cards=changed))
 # Both conflict directions and missing-evidence combinations have explicit contracts.
 for old in ['supported','contradicted','unknown']:
  for new in ['supported','contradicted','unknown']:
   d=dict(cards=[dict(id='generic',motion_evidence=dict(old_geometry_state=old,state=new),shape_evidence=[],status='supported',display_status='similar',difference_components=[],checked_components=[],difference_spans=[])])
   f=apply(d);conf={old,new}=={'supported','contradicted'};assert bool(f['direction_conflict_guard']['changed'])==conf;checks+=1
 write(U/'CONFLICT_GUARD_CHECK.json',dict(status='PASS',checks=checks,changed_cases=sum(bool(r['changed_cards']) for r in out),changed_cards=sum(len(r['changed_cards']) for r in out),normal_changed=0,rows=out,not_independent_evaluation=True,not_direction_accuracy_improvement=True));event('conflict_safety_checked',checks=checks,changed_cases=sum(bool(r['changed_cards']) for r in out))
if __name__=='__main__':main()
