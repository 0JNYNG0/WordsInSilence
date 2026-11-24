"""Independent numeric rebuild of fresh targets; no builder/model import or inference."""
from pathlib import Path
import json, hashlib, datetime, collections
import numpy as np

U=Path(__file__).resolve().parents[1]
F=U/'fresh'
O=U.parent/'word100_stage_feedback_cycle_20261031'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
features=['R_index_only','L_index_only','R_open_four','L_open_four','R_face_overlap','L_face_overlap','hands_overlap','R_palm_front','L_palm_front']
dirs=['up','down','left','right']
names=features+[s+'_motion_'+d for s in ['R','L'] for d in dirs]
rubric=read(O/'RGB_RUBRIC.json')
assert features==rubric['features']
assert names==read(O/'LEARNING_INPUTS.json')['all_targets']
allowed=set(rubric['motion_values'])
directions={'up':(1,0,0,0),'down':(0,1,0,0),'left':(0,0,1,0),'right':(0,0,0,1),
'up_left':(1,0,1,0),'up_right':(1,0,0,1),'down_left':(0,1,1,0),'down_right':(0,1,0,1),
'still':(0,0,0,0),'mixed':(None,)*4,None:(None,)*4}
inputs=read(F/'INPUTS.json')['rows']
visual={r['alias']:r for r in read(F/'VISUAL_INDEX.json')['rows']}
primary_list=read(F/'RGB_LABELS_A.json')['rows']+read(F/'RGB_LABELS_B.json')['rows']
primary={r['alias']:r for r in primary_list}
secondary={r['alias']:r for r in read(F/'RGB_LABELS_ROOT.json')['rows']}
audit=read(F/'LABEL_AUDIT.json')
audrows={r['alias']:r for r in audit['rows']}
aliases=[f'F{i:03}' for i in range(1,33)]
assert sorted(primary)==sorted(visual)==sorted(audrows)==sorted(r['alias'] for r in inputs)==aliases
assert len(inputs)==len(primary_list)==32
assert sorted(secondary)==[f'F{i:03}' for i in range(1,33,4)]
assert sorted(p.stem for p in (F/'targets').glob('*.npz'))==aliases
pin_checks=[]
for path,expected in audit['pins'].items():
 actual=sha(path);assert actual==expected,('pin',path)
 pin_checks.append(dict(path=path,sha256=actual))
extra_pins={str(p):sha(p) for p in [F/'VISUAL_INDEX.json',U/'quality.py',F/'LABEL_AUDIT.json',Path(__file__)]}
cross=[];details=[];status_counts=collections.Counter();frames_total=0;roi_checks=0
perfeature={n:dict(positive=0,negative=0) for n in names}
center_checks=[];source_pin_checks=0
for row in inputs:
 alias=row['alias'];p=primary[alias];q=secondary.get(alias);v=visual[alias]
 for key in ['input','observation']:
  assert sha(row[key])==row[key+'_sha256'],(alias,key,'hash')
  extra_pins[row[key]]=row[key+'_sha256'];source_pin_checks+=1
 with np.load(row['input'],allow_pickle=False) as z:
  ids=z['frame_id'].copy();ts=z['timestamp_us'].copy();n=len(z['x'])
 with np.load(row['observation'],allow_pickle=False) as z:
  xy=z['xy_px'].copy();observed=z['observed'].copy()
  assert np.array_equal(ids,z['frame_id']) and np.array_equal(ts,z['timestamp_us'])
 assert n==row['frames']==len(ids)==len(ts)
 assert np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0)
 assert len(v['frames'])==13 and [z['ordinal'] for z in v['frames']]==list(range(13))
 index={int(fid):i for i,fid in enumerate(ids)}
 for fr in v['frames']:
  assert fr['seconds']==float(ts[index[fr['frame_id']]]/1e6),(alias,fr['ordinal'],'seconds')
 assert sorted(s['ordinal'] for s in p['static'])==[3,6,9]
 assert sorted((m['start_ordinal'],m['end_ordinal']) for m in p['motion'])==[(0,3),(3,6),(6,9),(9,12)]
 if q:
  assert sorted(s['ordinal'] for s in q['static'])==[3,6,9]
  assert sorted((m['start_ordinal'],m['end_ordinal']) for m in q['motion'])==[(0,3),(3,6),(6,9),(9,12)]
 x0,y0,x1,y1=row['box']
 inside={}
 for side,off in [('R',54),('L',33)]:
  pts=xy[:,off:off+21];valid=observed[:,off:off+21]&np.isfinite(pts).all(axis=2)
  out=(pts[:,:,0]<x0)|(pts[:,:,0]>=x1)|(pts[:,:,1]<y0)|(pts[:,:,1]>=y1)
  inside[side]=valid.any(axis=1)&~(valid&out).any(axis=1)
 expected_y=np.zeros((n,17),np.float32);expected_mask=np.zeros((n,17),bool);entries=[];used=set();centers=[]
 def record(feature,value,other,center,lo,hi,kind):
  global roi_checks
  assert value in [None,0,1] and other in [None,0,1,'not_reviewed']
  original=value
  reason='single_Codex_RGB_draft'
  if other!='not_reviewed':
   cross.append(dict(alias=alias,feature=feature,frame=int(ids[center]),primary=value,secondary=other))
   if value is not None and value==other:reason='same_Codex_cross_agreement'
   else:value=None;reason='cross_unknown_or_disagreement'
  sides=['R','L'] if feature=='hands_overlap' else [feature[0]]
  roi_checks+=(hi-lo+1)*len(sides)
  roi=all(bool(inside[s][lo:hi+1].all()) for s in sides)
  if value is not None and not roi:value=None;reason='outside_or_no_direct_hand_ROI_evidence'
  col=names.index(feature)
  assert (center,col) not in used,(alias,feature,'duplicate slot')
  used.add((center,col))
  if value is not None:expected_y[center,col]=value;expected_mask[center,col]=True
  entries.append(dict(feature=feature,frame=int(ids[center]),kind=kind,initial=original,secondary=other,value=value,status=reason,interval_frame_ids=[int(ids[lo]),int(ids[hi])]))
  status_counts[reason]+=1
 for st in p['static']:
  oi=st['ordinal'];i=index[v['frames'][oi]['frame_id']]
  assert len(st['values'])==9
  qs=next((z for z in q['static'] if z['ordinal']==oi),None) if q else None
  for j,name in enumerate(features):record(name,st['values'][j],qs['values'][j] if qs else 'not_reviewed',i,i,i,'static')
 for mo in p['motion']:
  lo=index[v['frames'][mo['start_ordinal']]['frame_id']];hi=index[v['frames'][mo['end_ordinal']]['frame_id']]
  midpoint=(float(ts[lo]/1e6)+float(ts[hi]/1e6))/2
  distance=np.abs(ts/1e6-midpoint)
  center=int(np.flatnonzero(distance==distance.min())[0])
  assert lo<=center<=hi
  centers.append(center)
  center_checks.append(dict(alias=alias,interval_frame_ids=[int(ids[lo]),int(ids[hi])],center_frame_id=int(ids[center]),original_frame_count=hi-lo+1))
  qm=next((z for z in q['motion'] if (z['start_ordinal'],z['end_ordinal'])==(mo['start_ordinal'],mo['end_ordinal'])),None) if q else None
  for side in ['R','L']:
   assert mo[side] in allowed and (qm is None or qm[side] in allowed)
   for j,d in enumerate(dirs):
    record(side+'_motion_'+d,directions[mo[side]][j],directions[qm[side]][j] if qm else 'not_reviewed',center,lo,hi,'interval')
 target=F/'targets'/f'{alias}.npz'
 with np.load(target,allow_pickle=False) as z:
  assert set(z.files)=={'y','target_mask'}
  assert z['y'].dtype==np.float32 and z['target_mask'].dtype==np.bool_
  assert np.array_equal(z['y'],expected_y),(alias,'y')
  assert np.array_equal(z['target_mask'],expected_mask),(alias,'mask')
 assert entries==audrows[alias]['rows'],(alias,'audit_entries')
 assert expected_mask.sum()==audrows[alias]['known']
 assert set(np.flatnonzero(expected_mask[:,9:].any(axis=1))).issubset(set(centers))
 for side in ['R','L']:
  a,b=names.index(side+'_index_only'),names.index(side+'_open_four')
  assert not np.any(expected_mask[:,a]&expected_mask[:,b]&(expected_y[:,a]==1)&(expected_y[:,b]==1))
 for j,name in enumerate(names):
  perfeature[name]['positive']+=int(np.sum(expected_mask[:,j]&(expected_y[:,j]==1)))
  perfeature[name]['negative']+=int(np.sum(expected_mask[:,j]&(expected_y[:,j]==0)))
 details.append(dict(alias=alias,frames=n,known=int(expected_mask.sum()),static_known=int(expected_mask[:,:9].sum()),motion_known=int(expected_mask[:,9:].sum()),target_sha256=sha(target)))
 frames_total+=n
joint=[x for x in cross if x['primary'] is not None and x['secondary'] is not None]
bad=[x for x in joint if x['primary']!=x['secondary']]
known=sum(x['known'] for x in details)
assert cross==audit['cross'] and bad==audit['disagreements']
assert len(joint)==audit['joint_known'] and len(bad)==audit['known_disagreements']
assert len(bad)/len(joint)==audit['known_disagreement_rate']
assert len(secondary)==audit['cross_reviewed_videos']==8
assert known==audit['known']==981
result=dict(status='PASS_FRESH_TARGET_MASKS',reviewer='Same Codex separate agent /root/rgb_labels_a; independent numeric implementation, not independent human labels',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),no_inference=True,no_relabel=True,scope='All32 fresh evaluation clips; no model prediction/score files opened. Earlier held32 labels/images/arrays not used.',features=names,videos=32,frames=frames_total,expected_annotation_cells=32*59,known=known,static_known=sum(x['static_known'] for x in details),motion_known=sum(x['motion_known'] for x in details),cross_reviewed_videos=8,cross_cells=len(cross),joint_known=len(joint),known_disagreements=len(bad),known_disagreement_rate=len(bad)/len(joint),interval_centers_checked=len(center_checks),original_frame_side_roi_checks=roi_checks,pinned_input_and_observation_files_checked=source_pin_checks,audit_pin_count=len(pin_checks),audit_pins=pin_checks,supplemental_pins=extra_pins,status_counts=dict(status_counts),per_feature=perfeature,rows=details,interval_centers=center_checks,checks=['Independent literal direction truth table, including diagonal components and mixed/null withholding','Secondary disagreement or either unknown masks corresponding component only','Direct observed finite hand points required at every original interval frame; inclusive left/top, exclusive right/bottom; no body-wrist fallback','All target y and target_mask cells equal independent arrays, including zero and false elsewhere','Static ordinal3/6/9 and float-second nearest-original-PTS centers match; first index on equal distance','All1888 audit cells, full cross table, disagreement count and denominator equal independent recomputation','All32 videos and every native frame retained; finite binary targets and exclusive hand-shape invariants','All39 builder audit pins and64 input/observation hashes verified'],limitations=['Numeric label conversion PASS only; no visual-semantic or linguistic correctness certification.','PrimaryB RGB labels were authored by this agent; numeric rebuild is separate from root implementation, not independent-human validation.','Builder audit omits VISUAL_INDEX.json and quality.py direct pins; supplemental hashes of both are captured here for pre-evaluation freeze.'])
out=U/'review/FRESH_LABELS_INDEPENDENT.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ['status','videos','frames','known','static_known','motion_known','joint_known','known_disagreements','interval_centers_checked','audit_pin_count']}))

