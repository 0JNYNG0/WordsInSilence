"""Compile observed targets without propagating gloss or temporal guesses as truth."""
from pathlib import Path
import sys,json,hashlib,importlib.util
from collections import Counter
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent; P=U.parent/'word100_visible_temporal_targets_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 with Path(p).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def module():
 s=importlib.util.spec_from_file_location('unchanged_label_meanings',P/'build_targets.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def support(rows,targets,role):
 out={}
 for j,k in enumerate(targets):
  counts={}
  for val,name in [(1,'positive'),(0,'negative')]:
   rr=[r for r in rows if r['role']==role and r['target_mask'][j] and r['y'][j]==val]
   counts[name]=dict(windows=len(rr),clips=len({r['clip_id'] for r in rr}),words=len({r['word_id'] for r in rr}),people=len({r['person'] for r in rr}))
  out[k]=counts
 return out
def main():
 assert read(U/'PREP_COMPLETED.json')['clips']==96
 assert read(U/'review/INPUT_NUMERIC_VERIFICATION.json')['status'].startswith('PASS')
 old=read(P/'DATASET.json');targets=old['targets'];selected=old['selected_targets'];rule=read(U/'RUBRIC.json');enc=module()
 inputs=read(U/'INPUTS.json')['rows']; by={r['alias']:r for r in inputs};visual=read(U/'VISUAL_INDEX.json');assert visual['status']=='COMPLETE'
 new_samples=[s for s in visual['samples'] if s['alias'].startswith('N')]
 primary=[]
 for f,count in [('LABELS_A.json',72),('LABELS_B.json',54),('LABELS_C.json',18)]:
  d=read(U/'review'/f);assert len(d['rows'])==count,(f,len(d['rows']));primary+=d['rows']
 secondary=read(U/'review/LABELS_SECONDARY.json')['rows']
 pp={r['sample_id']:r for r in primary};ss={r['sample_id']:r for r in secondary}
 assert len(pp)==len(primary)==144 and set(pp)=={s['sample_id'] for s in new_samples}
 assert len(ss)==len(secondary) and set(ss)==set(read(U/'review/SECONDARY_SELECTION.json')['sample_ids'])
 rows=[dict(r,reused_prior_targets=True,prior_dataset_sha256=sha(P/'DATASET.json')) for r in old['rows'] if r['role'] in ('fit','development')]
 assert len(rows)==144 and all(r['input']==by[r['alias']]['input'] and r['input_sha256']==by[r['alias']]['input_sha256'] for r in rows)
 disagreements=[];conflicts=0;known_pairs=0;categories=Counter();category_conflicts=Counter()
 for s in new_samples:
  a=by[s['alias']];p,reasons=enc.labels(pp[s['sample_id']]);q=enc.labels(ss[s['sample_id']])[0] if s['sample_id'] in ss else None
  resolved=p.copy()
  if q is not None:
   for field,keys in [('motion',['R','L']),('center_shape',['R','L']),('center_relation',[None])]:
    for hand in keys:
     av=pp[s['sample_id']][field] if hand is None else pp[s['sample_id']][field][hand]
     bv=ss[s['sample_id']][field] if hand is None else ss[s['sample_id']][field][hand]
     if field=='center_shape':
      for x,y in zip(av,bv):
       if x is not None and y is not None:categories[field]+=1;category_conflicts[field]+=x!=y
     elif av is not None and bv is not None:categories[field]+=1;category_conflicts[field]+=av!=bv
   for k in targets:
    if p[k] is not None and q[k] is not None:known_pairs+=1;conflicts+=p[k]!=q[k]
    if p[k]!=q[k]:
     disagreements.append(dict(sample_id=s['sample_id'],target=k,primary=p[k],secondary=q[k]));resolved[k]=None;reasons[k]='review_disagreement_or_unknown'
  row=dict(sample_id=s['sample_id'],alias=a['alias'],clip_id=a['clip_id'],word_id=a['word_id'],person=a['person'],role=a['role'],input=a['input'],input_sha256=a['input_sha256'],frame_ids=s['frame_ids'],center_frame=s['center_frame'],timestamps_us=s['timestamps_us'],center_timestamp_us=s['center_timestamp_us'],
    y=[0. if resolved[k] is None else float(resolved[k]) for k in targets],target_mask=[resolved[k] is not None for k in targets],reasons=reasons,secondary_review=q is not None,primary_note=pp[s['sample_id']].get('notes',''),image=s['sheet'],center_image=s['center_file'],reused_prior_targets=False,
    visual_context=pp[s['sample_id']].get('visual_context',{}),semantic_importance=None,phase_truth=None,
    scope='Static labels supervise exact centre; motion labels supervise13-frame window; not all-frame or educational truth')
  rows.append(row)
 assert len(rows)==288 and len({r['sample_id'] for r in rows})==288
 supports={role:support(rows,targets,role) for role in sorted({r['role'] for r in rows})}
 gate=rule['class_gate'];passed={}
 for k in targets:
  passed[k]=all(supports['fit'][k][sign][metric]>=gate[f'fit_{sign}_{metric}_min'] for sign in ('positive','negative') for metric in ('windows','clips','words','people'))
 evaluation_claim={k:all(supports['new_feature_expression_evaluation'][k][s][m]>=n for s in ('positive','negative') for m,n in [('clips',3),('people',2),('words',2)]) for k in targets}
 quality=conflicts/max(known_pairs,1)<=.05
 # Five percent is an inherited technical review trigger, not a linguistic accuracy claim.
 ready=all(passed[k] for k in selected) and quality
 report=dict(status='READY_PROVISIONAL_PHYSICAL_FEATURE_STUDY' if ready else 'NOT_READY_SUPPORT_OR_REVIEW',targets=targets,selected=selected,supports=supports,fit_passed=passed,evaluation_supported_for_broader_target_claim=evaluation_claim,
   known_cross_comparisons=known_pairs,known_cross_conflicts=conflicts,known_conflict_rate=conflicts/max(known_pairs,1),category_compared=dict(categories),category_conflicts=dict(category_conflicts),disagreements=disagreements,
   known_cells=sum(sum(r['target_mask']) for r in rows),unknown_cells=sum(len(r['target_mask'])-sum(r['target_mask']) for r in rows),roles=dict(Counter(r['role'] for r in rows)),primary_new_windows=144,secondary_new_windows=len(ss),reused_windows=144,
   no_hidden_targets_filled=True,truth='Same-Codex RGB research annotations, not independent human/KSL educational truth',official_word_does_not_label_unseen_detail=True,existing_threshold_not_lowered=True)
 pins=[U/'AUTHORIZATION.json',U/'RUBRIC.json',U/'SELECTION.json',U/'INPUTS.json',U/'VISUAL_INDEX.json',U/'review/LABELS_A.json',U/'review/LABELS_B.json',U/'review/LABELS_C.json',U/'review/LABELS_SECONDARY.json',U/'review/SECONDARY_SELECTION.json',P/'DATASET.json']
 write(U/'DATASET.json',dict(rows=rows,targets=targets,selected_targets=selected,selected_indices=[targets.index(k) for k in selected],feature_names=read(U/'INPUTS.json')['feature_names'],pins={str(p):sha(p) for p in pins}))
 write(U/'LABEL_AUDIT.json',report)
 print(json.dumps({k:report[k] for k in ['status','known_cells','unknown_cells','known_cross_conflicts','known_cross_comparisons','category_compared','category_conflicts']},ensure_ascii=False))
if __name__=='__main__':main()
