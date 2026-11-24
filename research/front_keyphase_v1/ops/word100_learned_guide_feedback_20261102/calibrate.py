"""Fit-only range/support calibration; no DEV outcomes are read."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from prepare import U,WORDS,read,save
from engine import LearnedGuideEngine
def main():
 assert not (U/'CALIBRATION_COMPLETE.json').exists()
 manifest=read(U/'INPUT_MANIFEST.json');fit=[r for r in manifest['rows'] if r['role']=='fit'];records=[]
 for condition in ['frozen_projection','projection_finetune']:
  for seed in [42,43,44]:
   engine=LearnedGuideEngine(condition,seed,False);values=defaultdict(list);labels=defaultdict(lambda:defaultdict(list))
   for r in fit:
    a=dict(np.load(r['input']));p=dict(np.load(U/'runs'/f'{condition}_{seed}'/'predictions'/f"{r['clip_id']}.npz"))
    result=engine.compare(p,a,r['word_id'],True)
    for s in result['stages']:values[s['guide_id']].append(s)
    for s in r['stages']:
     lo,hi=s['frames']
     for label in r['labels']:
      if lo<=label['center_frame']<=hi:
       for j in range(7):
        if label['target_mask'][j]:labels[s['canonical_stage_id']][j].append((r['person'],label['y'][j]))
   norms={}
   for gid,rows in values.items():
    supported=[]
    for j,known in labels[gid].items():
     people={r[0] for r in known};fraction=float(np.mean([r[1] for r in known]))
     if len(people)<2 or .2<fraction<.8:continue
     expected=float(fraction>=.8)
     # Only physically descriptive shape differences are exposed initially.
     # Slight hand-height differences outside explicit alternation are not errors.
     guide=next(g for g in engine.guides if g['canonical_stage_id']==gid)
     if j>=5 and not any(t in guide['description'] for t in ['높낮이','위아래','바뀌','낮아','높아']):continue
     distances=[abs(r['attribute_mean'][j]-expected) for r in rows]
     supported.append(dict(target_index=j,expected=expected,fit_label_people=len(people),fit_label_samples=len(known),
       label_fraction=fraction,distance_limit=max(.25,float(np.quantile(distances,.95))+.05)))
    norms[gid]=dict(supported_attributes=supported,flow_limit=float(np.quantile([r['flow_cost'] for r in rows],.95))*1.25+.001,
       fit_clips=len(rows),linguistic_requiredness_confirmed=False)
   name=f'calibration/{condition}_{seed}.json'
   save(name,dict(condition=condition,seed=seed,stages=norms,fit_people=list(range(1,10)),development_used=False,
     numerical_range_is_not_linguistic_correctness=True,threshold_rule='FIT p95 with fixed margin; no DEV score adjustment'))
   records.append(name);print(name,flush=True)
 save('CALIBRATION_COMPLETE.json',dict(status='FIT_ONLY_CALIBRATION',files=records,private_or_dev_inputs_used=False))
if __name__=='__main__':main()
