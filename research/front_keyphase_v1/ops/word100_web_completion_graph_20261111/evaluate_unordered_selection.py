"""Exact source-bug ablation: preserve the original peak-run selector, remove chronology filter only."""
from graph import *
from collections import Counter
import numpy as np
O=U.parent;R=O/'word100_candidate_role_learning_20261111';D=O/'word100_action_feedback_demo_20261108'
def main():
 assert not (U/'UNORDERED_SELECTION_RESULTS.json').exists()
 cfg=read(D/'FROZEN.json')['phase'];cards=read(D/'CARDS.json')['words'];specs=read(O/'word100_action_episode_supervision_20261110/GUIDE_FAMILIES.json')['words'];src=read(R/'EVALUATION_INPUTS.json')['rows']
 write(U/'UNORDERED_SELECTION_FROZEN.json',dict(reason='Exact one-variable ablation of verified previous_end filter; run generation, confidence and peak selector unchanged from source evidence.py',threshold=cfg['threshold'],min_ms=cfg['min_evidence_ms'],source='stored phase_evidence.available_runs + stored phase_probability',new_model_inference=False,new_training=False,never_whole_sign_verdict=True))
 out=[]
 for s in src:
  before=read(s['baseline']);phase=np.asarray(before['phase_probability']);word=s['clip_id'].split('_')[2];new=[]
  for card in before['cards']:
   ev=[]
   for e in card['phase_evidence']:
    prob=phase[:,np.array(e['group'])+1].sum(1);runs=e['available_runs'];chosen=max(runs,key=lambda z:float(prob[z[0]:z[1]+1].max())) if runs else None
    ev.append(dict(group=e['group'],span=chosen,available_runs=runs))
   span=[min(e['span'][0] for e in ev),max(e['span'][1] for e in ev)] if ev and all(e['span'] for e in ev) else None
   new.append(dict(id=card['id'],span=span,before_span=card.get('query_indices'),evidence=ev))
  pairs=[];required=[f for f in specs[word] if f['required'] and not f.get('parent')]
  for a,b in zip(required,required[1:]):
   if a['families']==b['families']:continue
   aa=next(c for c in new if c['id']==a['id']);bb=next(c for c in new if c['id']==b['id']);sa,sb=aa['span'],bb['span']
   st='association_unknown' if sa is None or sb is None else 'supported' if sa[1]<sb[0] else 'reversed' if sb[1]<sa[0] else 'ambiguous'
   pairs.append(dict(cards=[a['id'],b['id']],state=st,spans=[sa,sb]))
  out.append(dict(clip_id=s['clip_id'],kind=s['kind'],cards=new,pairs=pairs))
 write(U/'UNORDERED_SELECTION_RESULTS.json',out);summary={k:dict(Counter(p['state'] for r in out if r['kind']==k for p in r['pairs'])) for k in sorted({r['kind'] for r in out})};write(U/'UNORDERED_SELECTION_COMPLETE.json',dict(summary=summary,changed_normal_cards=[dict(clip=r['clip_id'],card=c['id'],before=c['before_span'],after=c['span']) for r in out if r['kind']=='full_normal' for c in r['cards'] if c['span']!=c['before_span']],scope='Original selector chronology filter ablation; alternate occurrences retained, no claim they are uniquely resolved'));print(summary)
if __name__=='__main__':main()
