"""Audit source-level supervision, temporal contact, and all normal speed conditions."""
from local_rank_core import *
from collections import Counter
E=L/'evaluation';c=read(E/'COMPARISON.json');s=read(E/'shadow_components/RESULTS.json')
assert not (L/'CONTRACT_AUDIT.json').exists()
normal_kinds={'full_normal','speed075','speed125'}
assert normal_kinds<={r['kind']for r in c['rows']}
changes=[];order=[]
for r in s['rows']:
 if r['kind']not in normal_kinds:continue
 for mode in ['local_full','local_no_hands']:
  for before,after in zip(r['conditions']['baseline'],r['conditions'][mode]):
   a=[(v['task'],v['state'])for v in before['evidence']];b=[(v['task'],v['state'])for v in after['evidence']]
   if a!=b:changes.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],condition=mode,card=before['card'],before=a,after=b))
for r in c['rows']:
 for mode in ['baseline','rank_extended','local_full','local_no_hands']:
  d=r['conditions'][mode];cards={v['card']:v for v in d['cards']}
  for o in d['order']:
   a,b=[cards[k]['chosen']for k in o['cards']]
   if a is None or b is None:continue
   aa,ab=a['span'];ba,bb=b['span'];overlap=min(ab,bb)-max(aa,ba)
   if overlap==0:
    # Same timestamp is zero elapsed overlap, not evidence of overlapping motion.
    new='forward'if ab==ba else'reverse'if bb==aa else'undefined'
    order.append(dict(seed=r['seed'],clip_id=r['clip_id'],kind=r['kind'],condition=mode,cards=o['cards'],spans=[a['span'],b['span']],saved_state=o['state'],zero_duration_contact_state=new,adopted=False))
# Show why listwise rank values alone cannot be interpreted as existence or completion.
v=torch.tensor([1.,2.,3.,4.],dtype=torch.float64);positive=torch.tensor([False,True,True,False]);loss=lambda q:torch.logsumexp(q,0)-torch.logsumexp(q[positive],0)
errors=[float(abs(loss(v+shift)-loss(v)))for shift in [-100.,-10.,10.,100.]];assert max(errors)<1e-12
bags=read(L/'INPUTS_COMPLETE.json')['bags'];assert all(b['positive']>0 for b in bags)
write(L/'CONTRACT_AUDIT.json',dict(at=stamp(),all_normal_speed_kinds=sorted(normal_kinds),normal_component_changes=changes,zero_duration_endpoint_contacts=order,listwise_existence=dict(shift_invariant_loss=True,max_error=max(errors),all_training_bags_have_positive=True,interpretation='Ranking chooses between candidates of a known-present action. It does not train a calibrated none-of-the-above or whole-action completion output; existing separate presence gate does not turn rank score into that output.'),diagnostic_correction='TARGET_DIAGNOSIS normal-change list used speed_075/speed_125 rather than actual speed075/speed125; full comparison/summary arithmetic was still checked. This supplement audits every intended normal kind without overwriting initial records.',no_new_inference=True,no_threshold_change=True,web_changed=False))
print('All normal-speed changed components',len(changes),'endpoint contacts',len(order))
for r in changes:
 if r['condition']=='local_full':print(r)
for r in order:
 if r['condition']=='local_full'and r['kind']in normal_kinds:print(r)
