"""Direct fix of source phase-to-card decoder. Retains chronological evidence."""
import numpy as np
from binding import bind_occurrences,order_relation

def phase_occurrences(phase,available,ts,guide_specs,cards,threshold):
 phase=np.asarray(phase);out=[]
 for spec in guide_specs:
  card=next(c for c in cards if c['id']==spec['id'])
  if spec.get('parent'):continue # child's features share parent's evidence, not a second action
  columns=sorted({g+1 for group in card['groups'] for g in group});score=phase[:,columns].sum(1)
  pp=np.zeros((len(phase),11));family=spec['families'][0];pp[:,family]=score
  sides=[0 if s=='L' else 1 for s in spec['hands']];valid=np.asarray(available)[:,sides].all(1)
  for e in bind_occurrences(pp,valid,ts,[family],threshold):e['card']=card['id'];out.append(e)
 return sorted(out,key=lambda e:(e['a'],e['b'],e['family']))

def phase_pairs(occ,guide_specs):
 specs=[c for c in guide_specs if c['required'] and not c.get('parent')];out=[]
 for a,b in zip(specs,specs[1:]):
  fa,fb=a['families'][0],b['families'][0]
  if fa!=fb:out.append(dict(cards=[a['id'],b['id']],families=[fa,fb],state=order_relation(occ,fa,fb)))
 return out
