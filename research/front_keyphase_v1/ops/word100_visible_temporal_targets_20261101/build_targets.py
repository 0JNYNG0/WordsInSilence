"""Compile direct RGB labels to time-local targets; do not infer unknown labels."""
from pathlib import Path
import json,hashlib,numpy as np
from collections import Counter
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
SHAPES=['index_only','open_four','closed_four']
MOTIONS=['up','down','horizontal','still','mixed']
TARGETS=[f'{h}_{s}' for h in 'RL' for s in SHAPES]+['R_higher','L_higher']+[f'{h}_motion_{d}' for h in 'RL' for d in MOTIONS]

def labels(row):
 out={};reason={}
 for h in 'RL':
  shape=row['center_shape'][h];assert len(shape)==3 and all(v in (0,1,None) for v in shape)
  assert sum(v==1 for v in shape)<=1, 'Contradictory exclusive visible shapes'
  for key,v in zip(SHAPES,shape):
   k=f'{h}_{key}';out[k]=v if row['roi_center'][h] is True else None
   reason[k]='visible_center' if out[k] is not None else 'unknown_shape_or_outside_roi'
  direction=row['motion'][h];assert direction in MOTIONS+[None]
  for d in MOTIONS:
   k=f'{h}_motion_{d}';out[k]=int(direction==d) if direction is not None and row['roi_window'][h] is True else None
   reason[k]='visible_whole_13_frame_motion' if out[k] is not None else 'unknown_motion_or_outside_roi'
 relation=row['center_relation'];assert relation in ['R_higher','L_higher','level',None]
 for key in ['R_higher','L_higher']:
  out[key]=int(relation==key) if relation is not None and all(row['roi_center'][h] is True for h in 'RL') else None
  reason[key]='visible_center_relation' if out[key] is not None else 'unknown_bilateral_or_outside_roi'
 return out,reason

def main():
 prep=read(U/'PREP_COMPLETED.json');assert prep['clips']==56 and prep['windows']==168 and prep['protected_sha_verified']==32
 inputs=read(U/'INPUTS.json')['rows'];by={r['alias']:r for r in inputs}
 vi=read(U/'VISUAL_INDEX.json');assert vi['status']=='COMPLETE'
 samples=vi['samples'];assert len(samples)==168
 primary=read(U/'review/LABELS_A.json')['rows']+read(U/'review/LABELS_B.json')['rows']
 secondary=read(U/'review/LABELS_SECONDARY.json')['rows']
 pp={r['sample_id']:r for r in primary};ss={r['sample_id']:r for r in secondary}
 assert len(pp)==len(primary)==168
 assert set(pp)=={r['sample_id'] for r in samples}
 review_selection=read(U/'review/SECONDARY_SELECTION.json')['sample_ids'];assert set(ss)==set(review_selection)
 disagreements=[];known_comparisons=0;known_conflicts=0;rows=[];support={}
 for s in samples:
  a=by[s['alias']];p,reason=labels(pp[s['sample_id']]);q=labels(ss[s['sample_id']])[0] if s['sample_id'] in ss else None
  resolved=p.copy()
  if q is not None:
   for k in TARGETS:
    if p[k] is not None and q[k] is not None:known_comparisons+=1;known_conflicts+=int(p[k]!=q[k])
    if p[k]!=q[k]:
     disagreements.append(dict(sample_id=s['sample_id'],target=k,primary=p[k],secondary=q[k]));resolved[k]=None;reason[k]='review_disagreement_or_unknown'
  values=[0. if resolved[k] is None else float(resolved[k]) for k in TARGETS];mask=[resolved[k] is not None for k in TARGETS]
  rows.append(dict(sample_id=s['sample_id'],alias=s['alias'],clip_id=a['clip_id'],word_id=a['word_id'],person=a['person'],role=a['role'],input=a['input'],input_sha256=a['input_sha256'],frame_ids=s['frame_ids'],center_frame=s['center_frame'],timestamps_us=s['timestamps_us'],center_timestamp_us=s['center_timestamp_us'],y=values,target_mask=mask,reasons=reason,secondary_review=q is not None,primary_note=pp[s['sample_id']].get('notes',''),image=s['sheet'],center_image=s['center_file']))
 gate=read(U/'RUBRIC.json')['class_gate']
 for j,k in enumerate(TARGETS):
  counts={}
  for val,name in [(1,'positive'),(0,'negative')]:
   rr=[r for r in rows if r['role']=='fit' and r['target_mask'][j] and r['y'][j]==val]
   counts[name]=dict(windows=len(rr),clips=len({r['clip_id'] for r in rr}),words=len({r['word_id'] for r in rr}),people=len({r['person'] for r in rr}))
  passed=all(counts[sign][metric]>=gate[f'fit_{sign}_{metric}_min'] for sign in ['positive','negative'] for metric in ['windows','clips','words','people'])
  support[k]=dict(passed=passed,**counts)
 selected=[k for k in TARGETS if support[k]['passed']]
 families=dict(shape=any(k.endswith(tuple(SHAPES)) for k in selected),bilateral=any(k.endswith('higher') for k in selected),motion=any('_motion_' in k for k in selected))
 # Preserve high disagreement as a preparation problem; never lower a criterion after scores.
 quality=known_conflicts/max(known_comparisons,1)<=.05
 status='READY_PROVISIONAL_RGB_FEATURE_STUDY' if all(families.values()) and quality else 'NOT_READY_LABEL_SUPPORT_OR_REVIEW'
 report=dict(status=status,truth='Same-Codex RGB annotations, not expert KSL correctness',targets=TARGETS,selected=selected,support=support,families=families,known_cross_comparisons=known_comparisons,known_cross_conflicts=known_conflicts,known_conflict_rate=known_conflicts/max(known_comparisons,1),disagreements=disagreements,primary_windows=len(pp),secondary_windows=len(ss),mask_count=sum(sum(r['target_mask']) for r in rows),role_counts=dict(Counter(r['role'] for r in rows)),exclusions_not_silent=True)
 write(U/'LABEL_AUDIT.json',report)
 write(U/'DATASET.json',dict(rows=rows,targets=TARGETS,selected_targets=selected,selected_indices=[TARGETS.index(k) for k in selected],pins={str(p):sha(p) for p in [U/'AUTHORIZATION.json',U/'RUBRIC.json',U/'SELECTION.json',U/'INPUTS.json',U/'VISUAL_INDEX.json',U/'review/LABELS_A.json',U/'review/LABELS_B.json',U/'review/LABELS_SECONDARY.json',U/'review/SECONDARY_SELECTION.json']}))
 print(json.dumps({k:report[k] for k in ['status','selected','families','mask_count','known_cross_conflicts','known_cross_comparisons']},ensure_ascii=False))
if __name__=='__main__':main()
