"""Separate saved-output arithmetic/integrity audit; no model import or inference."""
from common import *
import numpy as np,collections
checks=[]
def ck(name,condition):
 checks.append(dict(name=name,ok=bool(condition)))
def display(d):return [c['display_status'] for c in d['cards']]
def fullsimilar(d):return all(c['display_status']=='similar' for c in d['cards'] if c.get('required',True))
def anydiff(d):return any(c['display_status']=='difference' for c in d['cards'])
def anyunknown(d):return any(c['display_status']=='uncertain' for c in d['cards'] if c.get('required',True))
summary={};changed=[];cache={}
def array(path):
 if path not in cache:
  with np.load(path,allow_pickle=False) as z:cache[path]={k:z[k].copy() for k in z.files}
 return cache[path]
for cohort,manifest,expected in [('historical_raw','REPLAY_RECOVERED.json',60),('new_record_id10','FRESH_COMPLETE.json',20)]:
 m=read(F/manifest);ck(cohort+'_count',len(m['rows'])==expected==m['conditions']==m['neural_calls']);ck(cohort+'_unique_conditions',len({(r['clip_id'],r['kind']) for r in m['rows']})==expected);agg={};seconds=[];added=[]
 for r in m['rows']:
  key=r['clip_id']+'_'+r['kind'];kind=r['kind'];a=agg.setdefault(kind,dict(total=0,errors=0,before_all_similar=0,after_all_similar=0,before_difference=0,after_difference=0,before_uncertain=0,after_uncertain=0,changed=0,top1_correct=0));a['total']+=1
  if r['status']!='OK':a['errors']+=1;ck(key+'_executed',False);continue
  z=array(r['path']);original=array(r['raw_source']);ids=np.array(r['source_indices']);n=len(ids)
  ck(key+'_input_hash',sha(r['path'])==r['sha256']);ck(key+'_raw_hash',sha(r['raw_source'])==r['raw_source_sha256']);ck(key+'_source_frames',np.array_equal(z['source_frame_id'],original['frame_id'][ids]));ck(key+'_frame_count',len(z['frame_id'])==r['frames']==n);ck(key+'_increasing',np.all(np.diff(z['frame_id'])>0) and np.all(np.diff(z['timestamp_us'])>0));ck(key+'_body_unchanged',np.array_equal(z['xy_px'][:,:91],original['xy_px'][ids,:91],equal_nan=True))
  if kind not in ['all_hand_missing','mask_final_event']:ck(key+'_all_coordinates_preserved',np.array_equal(z['xy_px'],original['xy_px'][ids],equal_nan=True))
  else:
   missing=~np.isfinite(z['xy_px'][:,91:133]).any((1,2));ck(key+'_sensor_gap_exists',missing.any());ck(key+'_missing_scores',np.all(z['score'][missing,91:133]==0));ck(key+'_prefix_unchanged',np.array_equal(z['xy_px'][~missing],original['xy_px'][ids[~missing]],equal_nan=True))
  p=Path(r['result']);b=read(p.parent/'before.json');d=read(p);pre=read(p.parent/'before_final_quarantine.json')
  ck(key+'_result_hash',sha(p)==r['result_sha256']);ck(key+'_ranks',b['tcn_top3']==d['tcn_top3']);ck(key+'_dtw',b['path']==d['path']);ck(key+'_out_frames',d['frame_id']==z['frame_id'].tolist());ck(key+'_no_rgb_or_reference_reads',d['reference_video_reads']==0 and d['rgb_model_calls']==0);ck(key+'_no_correctness_claim',d['educational_correctness_validated'] is False and d['correction_validated'] is False)
  ck(key+'_saved_aggregate_before',display(b)==r['before']['display'] and fullsimilar(b)==r['before']['all_required_similar']);ck(key+'_saved_aggregate_after',display(d)==r['after']['display'] and fullsimilar(d)==r['after']['all_required_similar'])
  q=np.asarray(d['observation_guard']['hand_quarantined']);ck(key+'_quarantine_frame_count',len(q)==n)
  for bc,dc,pc in zip(b['cards'],d['cards'],pre['cards']):
   ck(key+'_'+dc['id']+'_window_preserved',bc.get('query_indices')==dc.get('query_indices'));ck(key+'_'+dc['id']+'_stage_preserved',bc['phase_evidence']==dc['phase_evidence']);ck(key+'_'+dc['id']+'_no_wrong_claim',not dc['educational_wrong'])
   if bc['display_status']!=dc['display_status']:ck(key+'_'+dc['id']+'_only_withhold_success',bc['display_status']=='similar' and dc['display_status']=='uncertain')
   span=dc.get('query_indices');sides=[0 if s=='L' else 1 for s in dc['hands']];overlap=bool(span and q[span[0]:span[1]+1,sides].any())
   ck(key+'_'+dc['id']+'_final_quarantine',dc['display_status']=='uncertain' if overlap else dc['display_status']==pc['display_status'])
   ck(key+'_'+dc['id']+'_old_difference_retained',bc['display_status']!='difference' or dc['display_status']=='difference')
  if kind in ['full_normal','speed075','speed125']:ck(key+'_normal_regression',display(b)==display(d))
  if kind in ['all_hand_missing','mask_final_event']:ck(key+'_no_gap_wrong',not anydiff(d) and not fullsimilar(d))
  if kind=='omit_final_event':ck(key+'_no_false_complete',not fullsimilar(d))
  for prefix,out in [('before',b),('after',d)]:
   a[prefix+'_all_similar']+=int(fullsimilar(out));a[prefix+'_difference']+=int(anydiff(out));a[prefix+'_uncertain']+=int(anyunknown(out))
  a['top1_correct']+=int(d['tcn_top3'][0]['word']==r['word_id']);a['changed']+=int(display(b)!=display(d))
  if display(b)!=display(d):changed.append(dict(cohort=cohort,clip_id=r['clip_id'],kind=kind,before=display(b),after=display(d)))
  seconds.append(float(d['compute_seconds']));added.append(float(d['timing']['completion_identity_seconds']))
 summary[cohort]=dict(conditions=expected,by_condition=agg,consumer_seconds=dict(median=float(np.median(seconds)),p95=float(np.percentile(seconds,95)),maximum=max(seconds)),completion_branch_ms=dict(median=float(np.median(added)*1000),p95=float(np.percentile(added,95)*1000)),timing_scope='Consumer.raw only: cached RTMW input -> channels + TCN/feedback + DTW + completion + final guard. Excludes camera, detector, UI, disk saves and startup.')
for name in ['REPLAY_FROZEN.json','FRESH_FROZEN.json']:
 for p,h in read(F/name)['pins'].items():ck('pin:'+p,sha(p)==h)
for r in read(F/'FRESH_REVIEW.json')['rows']:ck('public_source:'+r['clip_id'],sha(r['source'])==r['source_sha256'])
raw=read(F/'FRESH_EXTRACTION_COMPLETE.json');ck('new_extraction_519',raw['frames']==raw['detector_calls']==519);ck('new_raw_normal_only',len(raw['rows'])==4)
rec=read(F/'RECOVERY_AUDIT.json');ck('first_reporting_failure_preserved',(F/'REPLAY_COMPLETE.json').exists() and len(read(F/'REPLAY_COMPLETE.json')['rows'])==60);ck('fresh_logging_recovery_no_rerun',read(F/'FRESH_EVALUATION_RECOVERY.json')['recomputed']==0 and read(F/'FRESH_COMPLETE.json')['committed_reused']==1)
line=read(F/'LINEAGE_AUDIT.json');ck('new_lineage_candidates',len(line['eligible'])==4);ck('new_lineage_no_hits',not any(r['candidate_id_hits'] or r.get('candidate_byte_hash_hits') for r in line['rows']))
write(F/'COMPARISON.json',dict(summary=summary,changed=changed,verification_script_sha256=sha(__file__),training=False,default_web_changed=False,new_independent_educational_accuracy=False))
write(F/'VERIFICATION.json',dict(status='PASS' if all(c['ok'] for c in checks) else 'FAIL',checks=len(checks),failures=[c for c in checks if not c['ok']],scope='Same Codex separately written saved-output arithmetic/integrity audit; not independent human/expert truth; no inference rerun',code_sha256=sha(__file__),comparison_sha256=sha(F/'COMPARISON.json')))
print(json.dumps(read(F/'VERIFICATION.json'),ensure_ascii=False));print(json.dumps(summary,ensure_ascii=False,indent=2))
assert all(c['ok'] for c in checks)
