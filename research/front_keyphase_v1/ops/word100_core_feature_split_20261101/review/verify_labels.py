"""Separate saved-label arithmetic, physical target scope and source checks."""
from pathlib import Path
from collections import Counter
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parents[1];P=U.parent/'word100_visible_temporal_targets_20261101'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=Counter();failures=[]
def check(k,condition,detail):
 checks[k]+=1
 if not condition:failures.append(dict(kind=k,detail=str(detail)))
def encode(row,name):
 hand=name[0]
 if name in ('R_higher','L_higher'):
  value=row['center_relation'];visible=all(row['roi_center'][h] is True for h in 'RL')
  return None if value is None or not visible else int(value==name)
 if '_motion_' in name:
  value=row['motion'][hand]
  return None if value is None or row['roi_window'][hand] is not True else int(value==name.split('_motion_')[1])
 field=name[2:];j=['index_only','open_four','closed_four'].index(field)
 return row['center_shape'][hand][j] if row['roi_center'][hand] is True else None
def main():
 data=read(U/'DATASET.json');audit=read(U/'LABEL_AUDIT.json');rows=data['rows'];names=data['targets'];sel=read(U/'SELECTION.json')
 old={r['sample_id']:r for r in read(P/'DATASET.json')['rows']}
 primary={r['sample_id']:r for f in ['LABELS_A.json','LABELS_B.json','LABELS_C.json'] for r in read(U/'review'/f)['rows']}
 secondary={r['sample_id']:r for r in read(U/'review/LABELS_SECONDARY.json')['rows']}
 visual={r['sample_id']:r for r in read(U/'VISUAL_INDEX.json')['samples']}
 inputs={r['alias']:r for r in read(U/'INPUTS.json')['rows']}
 check('samples',len(rows)==288 and len({r['sample_id'] for r in rows})==288,'all288')
 check('samples',len(primary)==144 and len(secondary)==72,'new144/cross72')
 check('secondary',set(secondary)==set(read(U/'review/SECONDARY_SELECTION.json')['sample_ids']),'exact prereview selection')
 cached={};known=0;missing=0
 for r in rows:
  sid=r['sample_id'];v=visual[sid];a=inputs[r['alias']]
  for key in ['frame_ids','center_frame','timestamps_us','center_timestamp_us']:check('time_alignment',r[key]==v[key],(sid,key))
  check('time_alignment',r['frame_ids']==list(range(r['center_frame']-6,r['center_frame']+7)),sid)
  check('input_identity',r['input']==a['input'] and r['input_sha256']==a['input_sha256'],sid)
  if r['alias'] not in cached:
   check('input_hash',sha(r['input'])==r['input_sha256'],r['input']);cached[r['alias']]=dict(np.load(r['input'],allow_pickle=False))
  x=cached[r['alias']];check('source_time',x['timestamp_us'][r['frame_ids']].tolist()==r['timestamps_us'],sid)
  if r['reused_prior_targets']:
   for key in ['y','target_mask','reasons','frame_ids','center_frame','clip_id','role']:check('prior_unchanged',r[key]==old[sid][key],(sid,key))
  else:
   p=primary[sid];q=secondary.get(sid)
   for hand in 'RL':
    val=p['center_shape'][hand]
    check('shape_schema',len(val)==3 and all(z is None or type(z) is int and z in (0,1) for z in val) and val.count(1)<=1,(sid,hand))
    check('motion_schema',p['motion'][hand] in ('up','down','horizontal','still','mixed',None),(sid,hand))
   for j,name in enumerate(names):
    truth=encode(p,name)
    if q is not None and truth!=encode(q,name):truth=None
    check('label_exact',r['target_mask'][j]==(truth is not None) and r['y'][j]==(0 if truth is None else truth),(sid,name))
   check('no_semantic_invention',r['semantic_importance'] is None and r['phase_truth'] is None,sid)
  for value,mask in zip(r['y'],r['target_mask']):
   check('masked_target',mask or value==0,sid);known+=mask;missing+=not mask
 check('label_totals',known==audit['known_cells'] and missing==audit['unknown_cells'],'totals')
 check('roles',dict(Counter(r['role'] for r in rows))==audit['roles'],'role counts')
 for role,features in audit['supports'].items():
  for j,name in enumerate(names):
   for value,key in [(1,'positive'),(0,'negative')]:
    rr=[r for r in rows if r['role']==role and r['target_mask'][j] and r['y'][j]==value]
    expected=dict(windows=len(rr),clips=len({r['clip_id'] for r in rr}),words=len({r['word_id'] for r in rr}),people=len({r['person'] for r in rr}))
    check('support_counts',expected==features[name][key],(role,name,key))
 fit={r['person'] for r in rows if r['role']=='fit'};dev={r['person'] for r in rows if r['role']=='development'};ev={r['person'] for r in rows if r['role']=='new_feature_expression_evaluation'}
 check('person_split',fit=={1,2,3} and dev=={4} and ev=={7,8},'recorded IDs')
 fw={r['word_id'] for r in rows if r['role']=='fit'};ew={r['word_id'] for r in rows if r['role']=='new_feature_expression_evaluation'}
 check('word_split',not fw&ew,'fit/evaluation words')
 for p,h in {**data['pins'],**sel['protected']}.items():check('protected_hash',sha(p)==h,p)
 result=dict(status='PASS_INPUT_LABEL_ARITHMETIC' if not failures else 'FAIL_INPUT_LABEL_ARITHMETIC',checks=dict(checks),check_count=sum(checks.values()),failures=failures,known=known,unknown=missing,
   reviewer='Root Codex separate explicit decoder, not independent human/KSL ground truth',new_visuals_reviewed_by_agents=True,metadata_does_not_prove_semantic_validity=True)
 with (U/'review/INPUT_LABEL_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False));assert not failures
if __name__=='__main__':main()
