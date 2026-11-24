"""Saved-artifact arithmetic audit. No model, capture, server or browser imports."""
import json
import hashlib
import importlib.util
from fractions import Fraction
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
close=lambda a,b:bool(np.isclose(a,b,rtol=1e-7,atol=1e-8))
def arrays(p,keys=None):
 with np.load(p,allow_pickle=False) as z:return {k:np.array(z[k]) for k in (keys or z.files)}
result=read(U/'WEB_RESULT.json');frozen=read(U/'WEB_FROZEN.json')
auth={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']}
aliases=['P036','P048','P066','P072']
assert [r['alias'] for r in result['rows']]==aliases
assert len(list((U/'web_runs').iterdir()))==4
assert result['fixed_seed']==42 and result['model_selection_by_ui'] is False
assert result['whole_sign_verdict'] is None and result['server_port']!=8765
assert result['protocol_sha256']==sha(U/'WEB_FROZEN.json')
assert not result['js_errors'] and result['narrow_no_horizontal_page_overflow']
pin_count=0
for group in ['pins','protected']:
 for p,h in frozen[group].items():
  assert sha(p)==h,p
  pin_count+=1

recovery=read(U/'WEB_RECOVERY.json')
original=(U/'web_browser_test.py').read_text(encoding='utf-8')
recovered=(U/'web_browser_recovery.py').read_text(encoding='utf-8')
expected=original.replace('WEB_EXECUTION_COMPLETE.json','WEB_RECOVERY_EXECUTION_COMPLETE.json').replace('web_chrome_profile','web_chrome_recovery_profile').replace('WEB_FIRST_FAILURE.json','WEB_RECOVERY_FIRST_FAILURE.json')
assert expected==recovered
assert recovery['original_launcher_sha256']==sha(U/'web_browser_test.py')
assert recovery['recovery_launcher_sha256']==sha(U/'web_browser_recovery.py')
failure=read(U/'WEB_FIRST_FAILURE.json')
assert failure['error']=='Isolated Chrome exited before debugging became ready'
assert read(U/'WEB_EXECUTION_COMPLETE.json')['status']=='FAILED'
success=read(U/'WEB_RECOVERY_EXECUTION_COMPLETE.json')
assert success['status']=='COMPLETE' and success['server_closed'] and success['chrome_closed'] and success['worker_finished']
assert not (U/'WEB_RECOVERY_FIRST_FAILURE.json').exists()
assert not (U/'WEB_RUNNING.json').exists()

# Pure frozen feature calculation only; no episode/model code is imported.
feature_path=U.parent/'word100_guide_video_feedback_20261028/motion_observer.py'
spec=importlib.util.spec_from_file_location('web_audit_numeric_features',feature_path)
feature_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(feature_module)
COLS=[28,29,36,37,30,31,32,33]+list(range(39,53))
checked=[];pairs_checked=0;output_pins=0;frame_total=0
for saved in result['rows']:
 alias=saved['alias'];d=U/'web_runs'/saved['key'];r=auth[alias]
 complete=read(d/'COMPLETE.json');backend=read(d/'BACKEND_COMPLETE.json');paint=read(d/'PAINT_ACK.json')
 assert saved==complete
 for k,v in backend.items():assert complete[k]==v,(alias,k)
 for name,h in backend['output_sha256'].items():
  assert sha(d/name)==h,(alias,name)
  output_pins+=1
 before=read(d/'PRE_RUN_INPUTS.json')
 assert before['frozen_protocol_sha256']==sha(U/'WEB_FROZEN.json')
 for p,h in before['sha256'].items():assert sha(p)==h,p
 req=read(d/'RUN_REQUEST.json');assert req['requested_ns']>failure['at_ns']
 assert req['alias']==alias and req['mode']=='during_recording' and req['repeat']==1
 assert not (d/'FIRST_FAILURE.json').exists()
 original_input=arrays(r['input'],['frame_id','timestamp_us'])
 obs=arrays(d/'candidate.npz',['xy_px','observed','usable','frame_id','timestamp_us'])
 baseline=arrays(d/'baseline.npz',['frame_id','timestamp_us'])
 inp=arrays(d/'input.npz')
 auto=arrays(d/'automatic.npz')
 trace=read(d/'TRACE.json');sidecar=read(d/'candidate.sidecar.json');payload=read(d/'feedback.json')
 n=len(obs['frame_id']);frame_total+=n
 assert n==r['frames']==backend['frames']==payload['source_frames']
 raw=arrays(d/'raw.npz',['width','height','frame_id','timestamp_us'])
 assert len(raw['width'])==n and (raw['width']==sidecar['width']).all() and (raw['height']==sidecar['height']).all()
 assert sidecar['width']==1920 and sidecar['height']==1080
 assert np.array_equal(raw['frame_id'],obs['frame_id']) and np.array_equal(raw['timestamp_us'],obs['timestamp_us'])
 assignments=arrays(d/'assignment.npz',['sides'])['sides']
 diagnostics=read(d/'ASSIGNMENT_DIAGNOSTICS.json')
 assert assignments.shape==(n,2) and len(diagnostics)==n
 assert [x['frame'] for x in diagnostics]==list(range(n))
 assert [x['timestamp_us'] for x in diagnostics]==obs['timestamp_us'].tolist()
 for a in [obs,baseline,inp]:
  for k in ['frame_id','timestamp_us']:assert np.array_equal(a[k],original_input[k]),(alias,k)
 assert np.array_equal(obs['frame_id'],np.arange(n))
 assert (np.diff(obs['timestamp_us'])>0).all()
 assert payload['frame_id']==obs['frame_id'].tolist()
 assert np.array_equal(payload['seconds'],obs['timestamp_us']/1e6)
 assert sidecar['source_sha256']==r['source_sha256']==sha(r['source'])
 assert sidecar['decoded_frames']==n and sidecar['dropped_non_monotonic_pts']==0
 assert sidecar['hand_assignment_policy']=='frozen_joint_v3'
 assert not sidecar['mirrored'] and sidecar['rotation_deg']==0
 assert before['sha256'][r['input']]==backend['input_before_sha256']==sha(r['input'])
 assert backend['generated_input_sha256']==sha(d/'input.npz')
 xy=obs['xy_px'];b=r['box']
 roi=(xy[:,:,0]>=b[0])&(xy[:,:,0]<b[2])&(xy[:,:,1]>=b[1])&(xy[:,:,1]<b[3])
 raw_valid=obs['observed']&obs['usable']&np.isfinite(xy).all(2)
 scale=np.linalg.norm(xy[:,11].astype(float)-xy[:,12].astype(float),axis=1)
 normalizer=(raw_valid&roi)[:,[11,12]].all(1)&np.isfinite(scale)&(scale>1e-8)
 independent_support=(raw_valid&roi)[:,54]&normalizer
 assert np.array_equal(independent_support,inp['mask'][:,[28,29]].all(1))
 roi_obs={k:v.copy() for k,v in obs.items()};roi_obs['usable'] &= roi
 rebuilt=feature_module.extract_features(roi_obs)
 assert np.array_equal(rebuilt['mask'],inp['mask'])
 assert np.allclose(np.nan_to_num(rebuilt['x']).astype('float32'),inp['x'],atol=1e-6)
 assert inp['x'].shape==(n,69) and np.isfinite(inp['x']).all()
 for name in ['pred','eligible','coverage','geometry']:assert len(auto[name])==n
 assert auto['prob'].shape==(n,3) and np.isfinite(auto['prob']).all()
 assert np.allclose(auto['prob'].sum(1),1,atol=1e-5)
 calculated_elig=[];calculated_cov=[]
 for i in range(n):
  sl=slice(max(0,i-6),min(n,i+7));s=independent_support[sl]
  ok=len(s)>=3 and s.mean()>=.75 and s[:3].any() and s[-3:].any() and independent_support[i]
  calculated_elig.append(bool(ok));calculated_cov.append(float(s.mean()))
 assert np.array_equal(auto['eligible'],calculated_elig)
 assert np.allclose(auto['coverage'],calculated_cov)
 assert (auto['pred'][~auto['eligible']]==-1).all()
 assert np.array_equal(auto['pred'][auto['eligible']],auto['prob'][auto['eligible']].argmax(1))
 assert payload['support']==dict(actual_right_hand_frames=int(independent_support.sum()),total_frames=n,unknown_episode_frames=int((auto['pred']<0).sum()))
 # Independently reconstruct contiguous output runs, not inference or accuracy scoring.
 seen=[]
 for e in payload['episodes']:
  ids=np.arange(e['start_frame'],e['end_frame']+1);seen.extend(ids.tolist())
  value={'up':0,'down':1,'other':2,'unknown':-1}[e['state']]
  assert (auto['pred'][ids]==value).all()
  assert len(ids)==e['frames'] and e['short_proposal']==(len(ids)<3)
  assert close(e['start_s'],obs['timestamp_us'][ids[0]]/1e6) and close(e['end_s'],obs['timestamp_us'][ids[-1]]/1e6)
 assert seen==list(range(n))
 tr=trace['rows'];arrivals=trace['arrivals'];assert len(tr)==len(arrivals)==n
 assert trace['capture_end_ns']==arrivals[-1]['available_ns']==backend['capture_end_ns']
 assert trace['ready_ns']==backend['ready_ns']
 stages=[trace[k] for k in ['capture_end_ns','saved_ns','feature_end_ns','head_end_ns','compare_end_ns','ready_ns']]
 assert stages==sorted(stages)
 tb=Fraction(sidecar['time_base'])
 for i,(rr,aa) in enumerate(zip(tr,arrivals)):
  assert rr['frame_id']==aa['frame_id']==i and rr['timestamp_us']==aa['timestamp_us']==int(obs['timestamp_us'][i])
  assert rr['source_pts']==aa['source_pts']
  assert int(round(float((aa['source_pts']-arrivals[0]['source_pts'])*tb)*1e6))==aa['timestamp_us']
  assert aa['due_ns']==trace['epoch_ns']+aa['timestamp_us']*1000
  assert aa['available_ns']==rr['available_ns'] and aa['due_ns']==rr['due_ns']
  assert rr['process_start_ns']>=aa['available_ns']>=aa['due_ns']
  assert rr['process_start_ns']<=rr['detector_end_ns']<=rr['assignment_end_ns']
  if i:assert rr['process_start_ns']>=tr[i-1]['assignment_end_ns']
  assert close(rr['queue_wait_seconds'],(rr['process_start_ns']-aa['available_ns'])/1e9)
  assert close(rr['detector_seconds'],(rr['detector_end_ns']-rr['process_start_ns'])/1e9)
  assert close(rr['assignment_seconds'],(rr['assignment_end_ns']-rr['detector_end_ns'])/1e9)
 measures={
  'backend_after_end_seconds':(trace['ready_ns']-trace['capture_end_ns'])/1e9,
  'capture_saved_tail_seconds':(trace['saved_ns']-trace['capture_end_ns'])/1e9,
  'feature_seconds':(trace['feature_end_ns']-trace['saved_ns'])/1e9,
  'episode_seconds':(trace['head_end_ns']-trace['feature_end_ns'])/1e9,
  'dtw_three_refs_seconds':(trace['compare_end_ns']-trace['head_end_ns'])/1e9,
  'queue_wait_max_seconds':max(x['queue_wait_seconds'] for x in tr),
  'arrival_lateness_max_seconds':max((x['available_ns']-x['due_ns'])/1e9 for x in arrivals),
  'backend_ready_to_2raf_ack_seconds':(complete['render_ack_received_ns']-trace['ready_ns'])/1e9,
  'capture_end_to_2raf_ack_seconds':(complete['render_ack_received_ns']-trace['capture_end_ns'])/1e9,
  'backend_ready_to_media_and_paint_ack_seconds':(complete['media_and_paint_ack_received_ns']-trace['ready_ns'])/1e9,
  'capture_end_to_media_and_paint_ack_seconds':(complete['media_and_paint_ack_received_ns']-trace['capture_end_ns'])/1e9,
  'paint_ack_to_final_ack_seconds':(complete['media_and_paint_ack_received_ns']-complete['render_ack_received_ns'])/1e9}
 for k,v in measures.items():assert close(complete[k],v),(alias,k,complete[k],v)
 assert complete['render_ack_received_ns']<complete['media_and_paint_ack_received_ns']
 assert backend['remaining_at_capture_end']==sum(x['assignment_end_ns']>trace['capture_end_ns'] for x in tr)
 assert sum(x['assignment_end_ns']<=trace['capture_end_ns'] for x in tr)>0
 assert 0<=backend['maximum_queued_frames']<=n and backend['decode_passes']==1
 assert paint['render_ack_received_ns']==complete['render_ack_received_ns']
 assert paint['browser']==complete['browser_paint']
 bp=complete['browser_paint'];bm=complete['browser_media']
 assert bp['rendered_key']==bm['rendered_key']==saved['key'] and bp['paint_callbacks']==2 and bp['reference_rows']==3
 assert bp['visible_scope'] and bp['visible_caution']
 assert bm['query']['alias']==alias
 for side in ['query','reference']:
  assert bm[side]['ready_state']>=2 and bm[side]['video_width']>0 and bm[side]['loadeddata_or_already_ready']
  assert 0<=bm[side]['receive_to_loadeddata_seconds']<=bm['receive_to_media_and_paint_seconds']+1e-6
 assert bp['receive_to_2raf_seconds']<=bm['receive_to_media_and_paint_seconds']+1e-6
 refs=[x for x in auth.values() if x['role']=='fit' and x['word_id']==r['word_id']]
 assert {x['reference'] for x in payload['pairs']}=={x['alias'] for x in refs} and len(payload['pairs'])==3
 assert bm['reference']['alias'] in {x['alias'] for x in refs}
 assert payload['educational_verdict'] is None and payload['full_clip_correctness'] is None
 assert '부족' in payload['caution'] and '전체 게임 총지연' in payload['timing_scope']
 for p in payload['pairs']:
  rr=auth[p['reference']];ref=arrays(rr['input'],['x','mask','frame_id','timestamp_us'])
  rs=arrays(U/'runs/42/auto'/f"{p['reference']}.npz",['pred'])['pred'];qs=auto['pred']
  m=len(ref['frame_id']);assert p['reference_frames']==p['full_scope_reference_frames']==m
  assert p['full_scope_query_frames']==n and p['reference_scope_retained']
  assert p['reference_frame_ids']==ref['frame_id'].tolist() and np.array_equal(p['reference_seconds'],ref['timestamp_us']/1e6)
  path=np.array(p['path_original_frames'],int);qi,ri=path.T
  assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[n-1,m-1])
  assert all(tuple(x) in [(1,1),(0,1),(1,0)] for x in np.diff(path,axis=0))
  assert np.array_equal(np.unique(qi),np.arange(n)) and np.array_equal(np.unique(ri),np.arange(m))
  assert p['all_query_frames_accounted']==n
  mapping=[int(np.median(qi[ri==j])) for j in range(m)];assert p['query_map']==mapping
  x,y=inp['x'][:,COLS],ref['x'][:,COLS]
  qm=inp['mask'][:,COLS]&np.isfinite(x);rm=ref['mask'][:,COLS]&np.isfinite(y)
  shared=qm[qi]&rm[ri];observed=np.zeros_like(rm);maxima=np.zeros_like(y)
  for k,j in enumerate(ri):
   observed[j]|=shared[k]
   maxima[j]=np.maximum(maxima[j],np.where(shared[k],np.minimum(np.abs(x[qi[k]]-y[j]),1),0))
  expected=int(rm.sum());compared=int(observed.sum());pathmissing=int((rm[ri]&~shared).sum())
  validstate=(qs[qi]>=0)&(rs[ri]>=0);disagree=int(((qs[qi]!=rs[ri])&validstate).sum())
  state_missing=int(((rs[ri]>=0)&(qs[qi]<0)).sum())
  exact={'expected':expected,'compared':compared,'missing':expected-compared,'path_missing':pathmissing,
   'direction_expected_pairs':int((rs[ri]>=0).sum()),'direction_compared_pairs':int(validstate.sum()),
   'direction_missing_pairs':state_missing,'direction_disagreements':disagree,'reference_unavailable_feature_cells':int(rm.size-rm.sum())}
  for k,v in exact.items():assert p[k]==v,(alias,p['reference'],k)
  assert p['reference_unavailable_frames']==np.flatnonzero(~rm.any(1)).tolist()
  assert p['reference_direction_unavailable_frames']==np.flatnonzero(rs<0).tolist()
  assert p['uncompared_reference_frames']==np.flatnonzero((rm&~observed).any(1)).tolist()
  assert p['query_frames_with_missing_expected']==np.unique(qi[(rm[ri]&~shared).any(1)]).tolist()
  cov=compared/max(expected,1);assert close(p['coverage'],cov)
  od=float(maxima.sum()/compared) if compared else None
  sd=disagree/int(validstate.sum()) if validstate.any() else None
  total=od+.25*(sd or 0) if od is not None else None
  for k,v in [('observed_position_shape_distance',od),('observed_direction_distance',sd),('distance',total)]:
   assert p[k] is None if v is None else close(p[k],v),(alias,p['reference'],k)
  complete_evidence=expected>0 and pathmissing==0 and state_missing==0 and rm.any(1).all() and (rs>=0).all()
  assert p['evidence_status']==('complete_relative_to_reference' if complete_evidence else 'incomplete_observation')
  assert p['status']==('comparable' if cov>=.75 and expected else 'insufficient')
  pairs_checked+=1
 seek=next(x for x in result['media_seek_checks'] if x['alias']==alias)
 first=payload['pairs'][0];mid=(len(first['reference_frame_ids'])-1)//2
 assert close(seek['videos'][0]['time'],payload['seconds'][first['query_map'][mid]])
 assert close(seek['videos'][1]['time'],first['reference_seconds'][mid])
 assert all(v['width']>0 and v['ready']>=2 for v in seek['videos'])
 checked.append(dict(alias=alias,frames=n,processed_before_capture_end=n-backend['remaining_at_capture_end'],
  backend_after_end_seconds=complete['backend_after_end_seconds'],
  capture_end_to_2raf_ack_seconds=complete['capture_end_to_2raf_ack_seconds'],
  capture_end_to_media_and_paint_ack_seconds=complete['capture_end_to_media_and_paint_ack_seconds'],
  backend_ready_to_2raf_ack_seconds=complete['backend_ready_to_2raf_ack_seconds'],
  backend_ready_to_media_and_paint_ack_seconds=complete['backend_ready_to_media_and_paint_ack_seconds'],
  actual_hand_support_frames=int(independent_support.sum()),unknown_output_frames=int((auto['pred']<0).sum()),
  raw_detector_fullframe_size=[sidecar['width'],sidecar['height']],raw_frame_and_assignment_records=n,
  references=[dict(alias=p['reference'],frames=p['reference_frames'],coverage=p['coverage'],missing=p['missing'],path_missing=p['path_missing'],evidence_status=p['evidence_status']) for p in payload['pairs']],
  final_ack_two_media_loaded=True,hashes_verified=True))

report=dict(status='PASS_SAVED_WEB_ARTIFACTS_AND_TIMING_ARITHMETIC',
 reviewer='Same-Codex rgb_labels_a verification; this agent implemented web code, so this is separately calculated artifact verification, not independent human or independent-implementer review.',
 no_new_inference=True,no_capture_or_replay=True,no_browser_started=True,
 files={'WEB_RESULT.json':sha(U/'WEB_RESULT.json'),'WEB_FROZEN.json':sha(U/'WEB_FROZEN.json'),'WEB_RECOVERY.json':sha(U/'WEB_RECOVERY.json')},
 counts={'replays':4,'native_frames':frame_total,'reference_pairs':pairs_checked,'frozen_and_protected_pin_checks':pin_count,'per_run_output_hash_checks':output_pins},
 checks=['4 fixed development aliases exactly once; saved requests after initial failure','source/input/observation/native frame and PTS agreement','original PTS reconstruction from recorded source_pts/time_base','ROI after full-frame observation, all69 feature/mask reconstruction and independent actual-wrist support','automatic eligibility/unknown and episode serialization from saved probabilities only','12 complete DTW paths retain every reference and query frame','per-reference expected/compared/missing/coverage/distance/evidence arithmetic','stage and ACK timing arithmetic','two RAF and query/reference loadeddata recorded separately','aligned slider mapping matches saved original-frame timestamps','all frozen/protected/input/output hashes unchanged'],
 recovery={'first_failure_preserved':True,'first_failure_before_debugging':True,'no_prior_replays_supported_by':'Failure control flow occurs before any runOne/start, all4 immutable RUN_REQUEST timestamps are after failure, first execution worker_finished=true; no extra run directories. Historical absence is supported by these artifacts, not inferred solely from current filesystem.','launcher_difference':'Only profile path and first-failure/execution-state filenames; exact string replacement equality verified.','runtime_permission_change':'Root reports second execution required elevated tool permission; this is an environment difference, not a model/data protocol change.','original_launcher_sha256':sha(U/'web_browser_test.py'),'recovery_launcher_sha256':sha(U/'web_browser_recovery.py')},
 rows=checked,
 timing_range_seconds={k:[min(r[k] for r in checked),max(r[k] for r in checked)] for k in ['backend_after_end_seconds','capture_end_to_2raf_ack_seconds','capture_end_to_media_and_paint_ack_seconds','backend_ready_to_2raf_ack_seconds','backend_ready_to_media_and_paint_ack_seconds']},
 limitations=['Root reported report-builder activity during replay/transition time; this was not a rigorously idle or isolated-load benchmark. Exact overlapping wall-clock work is not reconstructed from saved traces.','Four one-off public-file replays with warm application/reference cache; no repeated timing uncertainty estimate.','End is last frame arrival at original PTS, not terminal exposure completion.','Server receipt of browser ACK includes persistence, polling and network scheduling; two RAF is not direct physical display latency.','Both initial media loadeddata are evidenced; later reference changes are not included in first-result latency.','This route excludes old shape-head, SignRep and TCN inference; numbers are not total game latency.','This audit did not visually inspect screenshots and does not certify rendered layout beyond recorded browser checks.','Input masks and path arithmetic do not certify detector accuracy, dense RGB truth, stage boundaries or educational correctness.'])
(U/'review/WEB_RESULT_VERIFICATION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':report['status'],'counts':report['counts'],'timing_range_seconds':report['timing_range_seconds']},ensure_ascii=False))
