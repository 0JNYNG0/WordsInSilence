"""Parent arithmetic/provenance audit; never calls an inference engine."""
from pathlib import Path
import hashlib,json
import numpy as np

X=Path(__file__).resolve().parent; O=X.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
 checks.append(dict(name=name,pass_=bool(value)))
 assert value,name

b=read(X/'SOURCE_DEMO_BROWSER_VERIFIED.json'); f=b['fresh']; d=Path(f['analysis_directory'])
base=read(d/'FEEDBACK.json'); result=read(d/'SOURCE_FEEDBACK.json')
a=dict(np.load(d/'analysis.npz',allow_pickle=False))
check('single_actual_public_job',b['actual_fresh_public_posts_across_attempts']==1 and len(list((X/'public_runtime_runs').iterdir()))==1)
check('source_video_sha',sha(f['source'])==result['source_sha256']==f['source_sha256'])
check('new_feedback_sha',sha(d/'SOURCE_FEEDBACK.json')==f['source_feedback_sha256'])
for k,v in base.items():
 if k!='latency':check('base_field_'+k,result[k]==v)
for k,v in base['latency'].items():
 if k!='total_analysis_seconds':check('old_latency_'+k,result['latency'][k]==v)
check('new_total_accounts_for_bridge',result['latency']['total_analysis_seconds']>=base['latency']['total_analysis_seconds'])
check('no_reference_loop',result['runtime_individual_reference_video_comparisons']==0 and result['learned_prototype_comparisons']==1)
check('no_training_tuning',result['training'] is False and result['tuning'] is False)
check('whole_sign_unknown',result['whole_sign_correctness'] is None and result['sequence_verdict']['whole_sign_correctness'] is None)
ids=np.asarray(a['frame_id']);ts=np.asarray(a['timestamp_us']);path=np.asarray(result['path'])
check('native_axis_increasing',np.all(np.diff(ids)>0) and np.all(np.diff(ts)>0))
check('full_native_frames_preserved',len(ids)==result['frames']==119)
for c in result['source_components']:
 key=c['component_id']; lo,hi=c['canonical_anchor_extent']
 q=np.unique(path[(path[:,1]>=lo)&(path[:,1]<=hi),0]).astype(int)
 q=np.arange(q.min(),q.max()+1)
 check(key+'_native_extent',c['frame_ids']==ids[q].tolist())
 check(key+'_time',c['start_seconds']==float(ts[q[0]]/1e6) and c['end_seconds']==float(ts[q[-1]]/1e6))
 check(key+'_not_whole_correctness',c['whole_sign_correctness'] is None and c['whole_clause_correctness'] is None and c['complete_action_verified'] is False and c['correction_enabled'] is False)
 if 'point_evidence' in c:
  check(key+'_point_count',len(c['point_evidence'])==len(q))
  check(key+'_point_probabilities',all(p['probability']==float(a['relation_probability'][j,2]) for p,j in zip(c['point_evidence'],q)))
  check(key+'_native_points',all(p['frame']==int(ids[j]) and p['seconds']==float(ts[j]/1e6) for p,j in zip(c['point_evidence'],q)))
 if 'old_motion_status' in c:
  m=[v for v in result['motion_observations'] if v['target']==0 and v['frame_ids']==c['frame_ids']]
  check(key+'_same_motion_scope_probability',len(m)==1 and m[0]['probability']==c['probability'] and m[0]['guide_time_binding']==c['guide_time_binding'])

pins={}
for p in [X/'DEMO_FROZEN.json',O/'word100_learned_feedback_integration_20261102/FROZEN.json']:
 pins.update(read(p)['pins'])
for p,h in pins.items():check('frozen_pin_'+str(p),sha(p)==h)
for r in read(X/'PUBLIC_REPLAY_VERIFIED.json')['cases']:
 check('old_result_'+r['word']+'_'+r['kind'],sha(r['source'])==r['source_sha256'])
 check('old_input_'+r['word']+'_'+r['kind'],sha(r['input'])==r['input_sha256'])

report=dict(status='PASS',checks=checks,checks_count=len(checks),failures=0,
 public_job=f['job_id'],fresh_total_analysis_seconds=result['latency']['total_analysis_seconds'],
 latency_is_single_server_analysis_not_click_to_display_distribution=True,
 parent_visual_review=['SOURCE_DEMO_SCREEN_UI_RECHECK.png','SOURCE_DEMO_RESULT_UI_RECHECK.png'],
 visual_result='Two videos and source phrases/native replay/partial-evidence boundaries readable; no hidden white text found.',
 no_inference=True,no_training=True,no_private_reads=True,old_pins_preserved=len(pins),
 whole_goal_complete=False,
 limitations=['Same Codex separate arithmetic and browser verification, not independent human KSL ground truth.',
 'UI estimate color was checked using a memory-only fixture; no model accuracy gain follows.',
 'Source binding is an additive bridge over previously learned features, not newly trained complete-action understanding.'])
(X/'RUNTIME_SAVED_VERIFIED.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:report[k] for k in ['status','checks_count','failures','fresh_total_analysis_seconds','old_pins_preserved']},ensure_ascii=False))
