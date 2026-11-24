"""Compile source-component learning/evaluation evidence without new inference."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,copy
import numpy as np
from temporal_controls import controls,preflight

U=Path(__file__).resolve().parent;O=U.parent;Q=O/'word100_predicate_supervision_20261102'
D=O/'word100_action_episode_supervision_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,d):
 with (U/n).open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2)

TYPES={
 'obedience_closed_hand':('right_non_thumb_closed','point'),
 'obedience_lateral_arm_then_lower':('right_elbow_lowers_hand_remains_raised','sequence'),
 'well_palm_left_arm':('right_hand_at_left_arm_projection','point'),
 'well_brush':('right_hand_traverses_left_arm_toward_wrist','sequence'),
 'ancestor_thumbs':('both_visible_thumbs_up_with_closed_finger_masses','point'),
 'ancestor_alternation':('bilateral_height_alternation','sequence'),
 'egypt_tip_meeting':('bilateral_upper_ends_converged','point'),
 'egypt_oblique_pose':('bilateral_oblique_long_finger_axes','point'),
 'egypt_lower_shared':('source_variant_specific_lowering','sequence'),
 'egypt_gather_suffix':('each_hand_all_fingertips_gathered','point'),
}

def main():
 preflight()
 plan=read(U/'PREPARATION_PLAN.json')
 assert sha(plan['source_targets'])==plan['source_targets_sha256']
 assert sha(plan['development_manifest'])==plan['development_manifest_sha256']
 old=read(plan['source_targets']); sup=read(Q/'SUPERVISION.json')
 inputs={r['clip_id']:r for r in sup['sources']}
 fitm={r['clip_id']:r for r in read(D/'REVIEW_MANIFEST.json')['rows']}
 devm={r['clip_id']:r for r in read(D/'DEV_REVIEW_MANIFEST.json')['rows']}
 clauses={r['clause_id']:r for r in read(Q/'CLAUSE_COMPONENT_LINKS.json')['rows']}
 timestamps={};pins={};merged={};claim_counts=Counter();dev_seen=set();review_cells=0
 def pin(p):
  p=str(p)
  if p not in pins:pins[p]=sha(p)
  return pins[p]
 def times(cid,frames):
  if cid not in timestamps:
   inp=inputs[cid]; assert pin(inp['input'])==inp['input_sha256']
   with np.load(inp['input'],allow_pickle=False) as a:timestamps[cid]=a['timestamp_us'].copy()
  return [int(timestamps[cid][i]) for i in frames]
 def add(row):
  r=copy.deepcopy(row);key=(r['clip_id'],r['target'],r['kind'],tuple(r['native_frames']))
  assert r['person'] in (list(range(1,10)) if r['role']=='fit' else [11,12,13])
  assert r['native_timestamps_us']==times(r['clip_id'],r['native_frames'])
  assert (r['y'] in [0,1]) if r['mask'] else (r['y'] is None)
  r['dependency_group']=r['clip_id'];r.setdefault('whole_clause_truth',None);r.setdefault('learner_error',None)
  assert r['whole_clause_truth'] is None and r['learner_error'] is None
  if key in merged:
   prior=merged[key]
   for k in ['y','mask','role','word_id','person','native_timestamps_us']:assert prior[k]==r[k],(key,k)
   prior['provenance']+=r['provenance']
   prior['source_clause_ids']=sorted(set(prior['source_clause_ids']+r['source_clause_ids']))
  else:merged[key]=r
 for r in old['rows']:add(r)
 paths=[U/n for n in ['DEV_BINDINGS_ROOT.json','DEV_BINDINGS_1567.json','DEV_BINDINGS_2523.json',
                      'FIT_THUMB_COUNTERPOINTS.json','FIT_OBLIQUE_COUNTERPOINTS.json']]
 for path in paths:
  pin(path)
  for i,r in enumerate(read(path)['rows']):
   cid=r['clip_id'];role=r['role'];m=(fitm if role=='fit' else devm)[cid]
   assert r['word_id']==m['word_id'] and r['person']==m['person']
   assert pin(m['source'])==m['source_sha256']
   s=r['reviewed_sheet']; assert s['path']==m['sheet'] and pin(s['path'])==s['sha256']==m['sheet_sha256']
   available=set(m['indices']) if role!='fit' else {p['frame'] for p in m['frames']}
   reviewed=r.get('reviewed_frames',r.get('inspected_native_frames',r.get('reviewed_native_frames')))
   assert reviewed is not None and set(reviewed)<=available
   extras=r.get('additional_full_frame_views',[])+r.get('additional_reviewed_images',[])+r.get('reviewed_images',[])
   for e in extras:
    assert pin(e['path'])==e['sha256']
    assert e.get('frame',e.get('native_frame')) in reviewed
   if role!='fit':
    assert cid not in dev_seen;dev_seen.add(cid);review_cells+=len(reviewed)
   for j,c in enumerate(r['claims']):
    typ,kind=TYPES[c['clause_id']]
    if 'target' in c:assert c['target']==typ
    assert (c['kind']=='point')==(kind=='point')
    frames=c['witness_frames'];assert frames and sorted(set(frames))==frames
    assert set(frames)<=set(reviewed)
    if 'witness_timestamp_us' in c:assert c['witness_timestamp_us']==times(cid,frames)
    assert c['support'] in ['supported','visibly_contrary','unknown']
    y={'supported':1,'visibly_contrary':0,'unknown':None}[c['support']]
    clause=clauses[c['clause_id']];assert pin(clause['source']['path'])==clause['source']['sha256']
    origin=dict(path=str(path),pointer=f'/rows/{i}/claims/{j}',sha256=pins[str(path)],
                type='direct_image_component_review',source=clause['source'],text=clause['text'])
    claim_counts[(role,c['support'])]+=1
    groups=[[f] for f in frames] if kind=='point' else [frames]
    for g in groups:add(dict(clip_id=cid,word_id=r['word_id'],person=r['person'],role=role,
     target=typ,kind=kind,native_frames=g,native_timestamps_us=times(cid,g),y=y,mask=y is not None,
     source_clause_ids=[c['clause_id']],provenance=[origin]))
 assert len(dev_seen)==12 and dev_seen==set(devm)
 # Same-definition historic development observations add direct negatives.
 # Their original source-clause ownership is not invented.
 join={'R_closed_four':'right_non_thumb_closed','R_hand_at_L_arm':'right_hand_at_left_arm_projection',
       'bilateral_upper_ends_converged':'bilateral_upper_ends_converged'}
 for i,p in enumerate(sup['points']):
  if p['role']=='fit':continue
  for oldname,new in join.items():
   j=sup['point_targets'].index(oldname)
   if not p['mask'][j]:continue
   add(dict(clip_id=p['clip_id'],word_id=p['word_id'],person=p['person'],role='development',
    target=new,kind='point',native_frames=[p['frame']],native_timestamps_us=[p['timestamp_us']],
    y=int(p['y'][j]),mask=True,source_clause_ids=[],
    provenance=[dict(path=str(Q/'SUPERVISION.json'),pointer=f'/points/{i}',target_index=j,type='existing_same_definition_direct_point')]))
 for i,p in enumerate(sup['windows']):
  if p['role']=='fit':continue
  j=sup['window_targets'].index('right_hand_toward_left_wrist_along_arm')
  if not p['mask'][j]:continue
  add(dict(clip_id=p['clip_id'],word_id=p['word_id'],person=p['person'],role='development',
   target='right_hand_traverses_left_arm_toward_wrist',kind='interval',native_frames=p['frames'],
   native_timestamps_us=p['timestamps_us'],y=int(p['y'][j]),mask=True,source_clause_ids=[],
   provenance=[dict(path=str(Q/'SUPERVISION.json'),pointer=f'/windows/{i}',target_index=j,type='existing_same_definition_direct_interval')]))
 rows=list(merged.values()); synthetic=[]
 for r in rows:
  if r['kind']=='sequence' and r['mask'] and r['y']==1:synthetic+=controls(r)
 coverage={}
 for typ,_ in TYPES.values():
  coverage[typ]={}
  for role in ['fit','development']:
   rr=[r for r in rows if r['target']==typ and r['role']==role]
   cc=[r for r in synthetic if r['target']==typ and r['role']==role]
   counts=lambda a:dict(Counter('unknown' if not r['mask'] else 'positive' if r['y'] else 'negative' for r in a))
   coverage[typ][role]=dict(direct=counts(rr),constructed=counts(cc),
    direct_clips=len({r['clip_id'] for r in rr}),recorded_ids=sorted({r['person'] for r in rr}))
 for p in [Q/'TYPED_TARGETS_v2.json',Q/'SUPERVISION.json',Q/'CLAUSE_COMPONENT_LINKS.json',
           U/'temporal_controls.py',D/'REVIEW_MANIFEST.json',D/'DEV_REVIEW_MANIFEST.json']:pin(p)
 fit=[r for r in rows if r['role']=='fit'];dev=[r for r in rows if r['role']=='development']
 assert {r['clip_id'] for r in fit}.isdisjoint({r['clip_id'] for r in dev})
 save('DIRECT_TARGETS.json',dict(rows=rows,coverage=coverage,pins=pins,
  repeated_development_not_final_holdout=True,source_clause_truth_not_labelled=True,
  existing_targets_preserved=True,whole_goal_complete=False))
 save('TEMPORAL_CONTROLS.json',dict(rows=synthetic,script_sha256=sha(U/'temporal_controls.py'),
  no_new_people=True,natural_learner_error_gold=False,word_id_not_truth=True,
  source_extent_not_shrunk_when_missing=True))
 summary=dict(status='DIRECT_AND_CONSTRUCTED_COMPONENT_INPUTS_COMPILED',fit_records=len(fit),
  dev_records=len(dev),dev_clips=len(dev_seen),dev_review_image_cells=review_cells,
  additional_claims={str(k):v for k,v in claim_counts.items()},constructed_records=len(synthetic),
  coverage=coverage,new_training=0,new_inference=0,whole_goal_complete=False)
 save('PREPARATION_CHECKS.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k!='coverage'},ensure_ascii=False))
if __name__=='__main__':main()
