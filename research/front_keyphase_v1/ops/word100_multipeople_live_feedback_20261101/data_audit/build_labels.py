"""Package already-viewed exact-frame physical labels; no predictions or new images."""
from pathlib import Path
import json, hashlib
from collections import Counter
import numpy as np
U=Path(__file__).resolve().parent; OPS=U.parent.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(n,d):
 p=U/n;assert not p.exists(),p;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

sources=read(U/'SOURCES.json'); manual=read(U/'MANUAL_CENTER_REVIEW.json')
visual={r['alias']:r for r in read(U/'VISUAL_INDEX.json')['rows']}
targets=manual['targets'];rows=[];checks=0
for s in sources['rows']:
 assert sha(s['source'])==s['source_sha256'];checks+=1
 assert sha(s['native'])==s['native_sha256'];checks+=1
 with np.load(s['native'],allow_pickle=False) as z:
  ts=z['timestamp_us'];fids=z['frame_id']
 for k,c in enumerate(s['centers']):
  frameids=[int(x) for x in fids[c-6:c+7]];times=[int(x) for x in ts[c-6:c+7]]
  assert len(frameids)==13 and frameids[6]==c and frameids==s['context_frame_ids'][k];checks+=1
  if s['person']<=4:
   old=next(r for r in s['old_labels'] if r['center_frame']==c)
   assert frameids==old['frame_ids'] and times==old['timestamps_us'];checks+=1
   y=old['y'];mask=old['target_mask'];nullable=[v if m else None for v,m in zip(y,mask)]
   reason=old['reasons'];note=old.get('primary_note','');image=old['center_image']
   provenance={'type':'EXACT_PRIOR_CENTER_REUSED_UNCHANGED','dataset':str(OPS/'word100_semantic_feature_cycle_20261101/DATASET.json'),
     'sample_id':old['sample_id'],'old_role':old['role'],'current_new_protocol_role':s['role'],'secondary_review_previously':old.get('secondary_review',False)}
   confidence='previous_direct_RGB_label_reused_no_new_inference'; sheet=None
  else:
   alias=s['visual_alias'];m=manual['rows'][alias];v=visual[alias];win=v['windows'][k]
   assert win['center_frame']==c and win['timestamp_us']==times[6];checks+=1
   assert sha(v['sheet'])==v['sheet_sha256'];checks+=1
   image=win['full_image'];assert sha(image)==win['full_image_sha256'];checks+=1
   nullable=m['y'][k];mask=[x is not None for x in nullable];y=[float(x) if x is not None else 0.0 for x in nullable]
   note=m['notes'][k];reason={t:('UNKNOWN_DIRECT_EVIDENCE_INSUFFICIENT: ' if x is None else 'DIRECT_RGB_PRESENT: ' if x==1 else 'DIRECT_RGB_INCOMPATIBLE: ')+note for t,x in zip(targets,nullable)}
   provenance={'type':'NEW_EXACT_CENTER_RGB_PRIMARY_REVIEW','alias':alias,'sample_id':win['sample_id'],'sheet':v['sheet'],
     'sheet_sha256':v['sheet_sha256'],'body_and_R_L_detail_viewed':True,'reviewer':'Codex semantic_data_audit','fully_blind':False,'independent_human_truth':False}
   confidence='direct_physical_feature_or_explicit_unknown_not_calibrated_probability';sheet=v['sheet']
  assert len(y)==len(mask)==7 and all(x in (0,1,0.0,1.0) for x in y);checks+=1
  assert not (mask[5] and mask[6] and y[5]==y[6]==1);checks+=1
  for op,cl in [(1,3),(2,4)]:assert not(mask[op] and mask[cl] and y[op]==y[cl]==1)
  row={'sample_id':f"{s['clip_id']}_f{c:04}",'clip_id':s['clip_id'],'person':s['person'],'word_id':s['word_id'],'role':s['role'],
   'center_frame':c,'center_timestamp_us':times[6],'frame_ids':frameids,'timestamps_us':times,'y':y,'target_mask':mask,
   'nullable_labels':nullable,'reasons':reason,'primary_note':note,'center_image':image,'center_image_sha256':sha(image),
   'sheet':sheet,'source':s['source'],'source_sha256':s['source_sha256'],'native':s['native'],'native_sha256':s['native_sha256'],
   'roi_box':s['roi_box'],'source_provenance':provenance,'review_confidence':confidence,
   'scope':'Exact center physical attributes only;13 frame IDs are input context, not13 labels; word/guide/model predictions never propagated.',
   'secondary_review':s['person']<=4,'educational_correctness':None}
  rows.append(row)
assert len(rows)==144 and len({(r['clip_id'],r['center_frame']) for r in rows})==144
assert sum(r['role']=='fit' for r in rows)==108 and sum(r['person']<=4 for r in rows)==48
summary={}
for group,rr in [('all',rows),('new96',[r for r in rows if r['person']>4]),('fit108',[r for r in rows if r['role']=='fit']),('dev36',[r for r in rows if r['role']=='development'])]:
 summary[group]={'centers':len(rr),'clips':len({r['clip_id'] for r in rr}),'people':sorted({r['person'] for r in rr}),
  'targets':{t:{'positive':sum(r['target_mask'][j] and r['y'][j]==1 for r in rr),'negative':sum(r['target_mask'][j] and r['y'][j]==0 for r in rr),'unknown':sum(not r['target_mask'][j] for r in rr)} for j,t in enumerate(targets)}}
pins={str(p):sha(p) for p in [U/'SOURCES.json',U/'ANNOTATION_INPUTS.json',U/'MANUAL_CENTER_REVIEW.json',U/'VISUAL_INDEX.json',Path(__file__),OPS/'word100_semantic_feature_cycle_20261101/DATASET.json',OPS/'word100_core_feature_split_20261101/RUBRIC.json']}
save('LABELS.json',{'status':'PRIMARY_PHYSICAL_LABELS_COMPLETE_SECONDARY_REVIEW_PENDING','targets':targets,'rows':rows,'summary':summary,'source_pins':pins,
 'protocol':'New development study FIT01..09 / DEV11..13. Historical head DEV/test reuse explicit; old results unchanged. No claim of fresh final test.',
 'readiness':'Structural/lineage checks passed; root subset secondary RGB review must be recorded before training freeze.',
 'normal_variation_is_not_error_truth':True,'model_predictions_used_for_labels':False,'new_inference':0,'new_training':0})
save('LABELS_READY.json',{'status':'PASS_STRUCTURAL_PRIMARY_RGB_REVIEW_COMPLETE','checks':checks,'failed_checks':0,'label_path':str(U/'LABELS.json'),'label_sha256':sha(U/'LABELS.json'),
 'counts':{'clips':48,'centers':144,'old_reused':48,'new_primary':96,'fit':108,'dev':36},
 'secondary_gate':'PENDING_ROOT_SUBSET_REVIEW','recommended_secondary_review_aliases':['N003','N004','N009','N010','N011','N014','N016','N018','N025','N026','N031','N032'],
 'reason':'Check marginal open-four, transient partly bent fingers, overlap and ROI-null consistency; include all DEV per existing review policy where feasible.',
 'all_unknown_centers_retained':True,'word_label_propagation':False,'new_inference':0,'new_training':0})
print(json.dumps({'label_sha256':sha(U/'LABELS.json'),'checks':checks,'summary':summary},ensure_ascii=False))
