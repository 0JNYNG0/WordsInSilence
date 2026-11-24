from pathlib import Path
import json,hashlib,shutil
from datetime import datetime,timezone
U=Path(__file__).resolve().parent;O=U.parent;R=O.parents[2]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
assert R.name=='WordsInSilence'
assert not (U/'FROZEN.json').exists()
P=O/'word100_shared_feature_bundle_20261027';T=O/'word100_temporal_shape_context_20261027'
prior=read(T/'FROZEN.json');oldwords={r['word_id'] for r in prior['rows']}
shared=read(P/'FROZEN.json');index=read(P/'visual_inputs/INDEX.json');sr={r['clip_id']:r for r in shared['rows']}
rows=[]
for r in index['rows']:
 if r['word_id'] in oldwords:continue
 center=next(f for f in r['frames'] if f['ordinal']==8)
 rows.append({'case_id':f'N{len(rows)+1:02d}','source_alias':r['alias'],'group':'new_dense_window',
  'side':'R','source':r['source'],'source_sha256':r['source_sha256'],'clip_id':r['clip_id'],
  'word_id':r['word_id'],'label':r['label'],'recorded_person_id':r['recorded_person_id'],
  'inherited_role':r['role'],'record':sr[r['clip_id']]['record'],'record_sha256':sr[r['clip_id']]['record_sha256'],
  'center_frame_id':center['frame_id'],'center_pts_us':center['pts_us'],'source_sample_ordinal':8,
  'display_only_crop_xyxy':r['display_only_crop_xyxy'],
  'guard_evaluation_role':'same_recorded_ID_as_calibration' if r['recorded_person_id']==1 else 'different_recorded_ID_from_calibration'})
assert len(rows)==18 and len({r['word_id'] for r in rows})==6
for r in rows:assert sha(r['source'])==r['source_sha256']
config={'at':datetime.now(timezone.utc).isoformat(),
 'authorization':'User approved next common temporal/RGB improvement; asked original-frame button audit and preservation of endpoints plus transition interval. No operating replacement or user data.',
 'rows':rows,'dense_offsets_frames':list(range(-6,7)),
 'selection':'All six remaining expressions from previous36-clip bundle; ordinal8 center, all3 recorded IDs, R hand. No new dense outcomes inspected for selection. Not random population sample.',
 'pins':{str(p):sha(p) for p in [P/'FROZEN.json',P/'visual_inputs/INDEX.json',P/'RUBRIC.json',T/'FROZEN.json',T/'COMPARISON.json',T/'temporal_support.py',T/'FINAL_STATUS.json']},
 'features':['index_only','open_four'],
 'rubric_source':str(P/'RUBRIC.json'),
 'temporal_baseline':'Frozen prior C bidirectional state agreement, +/-200000us (+1us rounding), min2 known each direction, all known agree, direct unchanged, no recursive fill.',
 'candidate':'Keep direct and baseline outputs. For baseline temporal estimates only, require image+pose transition evidence be usable and below frozen calibration limits on every adjacent edge between nearest known past and future anchors; otherwise preserve unknown and record transition_or_view_change / insufficient_evidence. No coordinates invented.',
 'rgb_metrics':{'method':'ShiTomasi+LK forward/backward tracks within union target hand hull; compensate global in-plane hand translation/rotation/scale with affine partial fit; p90 remaining residual / palm width.',
  'max_corners':160,'quality_level':0.01,'min_distance_px':3,'lk_window':21,'lk_levels':3,'fb_max_px':1.5,
  'minimum_tracks':8,'hull_dilation_px':8,'affine_ransac_px':2.0,'seed':20261027,
  'palm_width':'distance between observed index and pinky MCP (5,17), median over two frames; min10px else unavailable',
  'track_geometry':'need >=8 accepted tracks and affine estimated with >=50% inliers; mean reprojection does not certify true hand identity or finger shape'},
 'pose_metrics':{'method':'p90 absolute 2D flexion-angle change from same current hand points; no interpolation. Need at least8 of12 PIP/DIP/MCP angle triplets observed in both frames.',
  'triplets':[[0,5,6],[5,6,7],[6,7,8],[0,9,10],[9,10,11],[10,11,12],[0,13,14],[13,14,15],[14,15,16],[0,17,18],[17,18,19],[18,19,20]],
  'minimum_angles':8,'minimum_segment_px':2},
 'calibration':{'sources':[{'unit':str(T),'case_id':'T07','note':'low-rest recordedID01 L hand'},{'unit':str(T),'case_id':'T10','note':'stable open crossing recordedID01 R hand'}],
  'rule':'For each metric separately use maximum valid edge score on these two previously reviewed stable windows. Need >=12 valid edges per metric over24total. No evaluation-label tuning; inadequate calibration stops candidate comparison.',
  'threshold_floor':'none; report excessive rejection rather than search new limits',
  'limits':'Two stable calibration windows only, no population tolerance claim.'},
 'evaluation':{'fresh':'18 new dense windows, IDs02/03 subset reported separately; known original interior states independently masked one at a time. Report same-AI draft agreement/conflict/withheld, actual unknown estimates separately.',
  'diagnostic':'Previous four conflicts and18 old windows may be replayed once with frozen thresholds, never advertised as fresh improvement.',
  'preservation':'All known direct states; no frame deletions; chronological endpoints and uncertain/transition interval provenance retained.',
  'not_truth':'Image drafts are not independently human-verified labels. Image/2D-pose residual indicates change or viewing uncertainty, not linguistic wrongness or a certified shape transition.'},
 'forbidden':['user/mentor material','training','detector downloads','new detector inference','target-word input','operating C/T3F/web/DTW edits','old frozen outputs edits','Git','coefficient search'],
 'completion':'Frame audit + common candidate implemented + frozen public comparison + separate verification + report and synchronized handoff. Do not claim recognition/feedback accuracy improvement.'}
(U/'before').mkdir(exist_ok=True)
for a,b in [(R/'WordsInSilence_PROJECT_CONTEXT.md','WordsInSilence_PROJECT_CONTEXT.md'),(R/'NEW_CHAT_HANDOFF.md','NEW_CHAT_HANDOFF.md'),(O/'STATE.json','STATE.json')]:shutil.copyfile(a,U/'before'/b)
save(U/'FROZEN.json',config)
print(json.dumps({'windows':len(rows),'expressions':[r['label'] for r in rows[::3]],'calibration':'T07/T10 ID01 only','dense_eval_role_counts':{'ID01':6,'ID02/03':12}},ensure_ascii=False))
