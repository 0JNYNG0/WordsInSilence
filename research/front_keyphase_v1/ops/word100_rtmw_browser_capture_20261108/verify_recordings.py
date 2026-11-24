"""Separate saved-evidence audit. No detector or inference import/call."""
from pathlib import Path
import json,hashlib
import numpy as np
import cv2,av
U=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 trial=Path(json.loads((U/'LATEST_TRIAL.json').read_text())['trial'])
 complete=json.loads((trial/'COMPLETE.json').read_text(encoding='utf-8'))
 freeze=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'));checks=[];runs=[]
 def check(name,ok):checks.append({'name':name,'pass':bool(ok)})
 check('hardware_camera_never_opened',complete['hardware_camera_calls']==0)
 check('no_browser_runtime_exceptions',not complete['errors'])
 check('localhost_requests_only',all(r['url'].startswith(('http://127.0.0.1:','blob:http://127.0.0.1:','data:')) for r in complete['requests']))
 for p,h in freeze['protected'].items():check('protected:'+Path(p).name,sha(p)==h)
 for spec in freeze['rows']:check('public_source:'+spec['clip_id'],sha(spec['source'])==spec['source_sha256'])
 for item in complete['results']:
  r=item['result'];sid=r['id'];d=U/'runs'/sid;bt=json.loads((d/'BROWSER_TRACE.json').read_text(encoding='utf-8'))
  receipts=json.loads((d/'RECEIPTS.json').read_text());sm=json.loads((d/'rtmw/SESSION.json').read_text());mp=json.loads((d/'MP_TCN.json').read_text(encoding='utf-8'))
  trace=bt['trace'];ids=np.array([v['frame_id'] for v in trace]);ts=np.array([v['timestamp_us'] for v in trace]);n=len(ts)
  prefix=sid+':'
  with np.load(d/'rtmw/observations.npz') as z:
   check(prefix+'rtmw_frame_id_exact',np.array_equal(z['frame_id'],ids));check(prefix+'rtmw_timestamp_exact',np.array_equal(z['timestamp_us'],ts))
  with np.load(d/'mp_observations.npz') as z:
   check(prefix+'mp_frame_id_exact',np.array_equal(z['frame_id'],ids));check(prefix+'mp_timestamp_exact',np.array_equal(z['timestamp_us'],ts))
  with np.load(d/'pose_analysis_input.npz') as z:
   check(prefix+'reused_id_exact',np.array_equal(z['frame_id'],ids));check(prefix+'reused_timestamp_exact',np.array_equal(z['timestamp_us'],ts))
  check(prefix+'all_frame_requests_accepted',len(receipts)==n and all(x['rtmw_accepted'] and x['mp_accepted'] for x in receipts))
  check(prefix+'same_receipt_ids_and_times',[(x['frame_id'],x['timestamp_us']) for x in receipts]==list(zip(ids,ts)))
  check(prefix+'monotone_clock',n>1 and np.all(np.diff(ts)>0));check(prefix+'no_second_detection',r['second_pass_detector_calls']==0)
  check(prefix+'rtmw_once_per_frame',sm['detector_calls']==n);check(prefix+'mp_once_per_frame',mp['detector_calls']==n)
  check(prefix+'no_silent_drop_or_error',not bt['errors'] and not mp['errors'] and not mp['dropped'] and not sm['errors'] and sm['dropped']==0)
  pts=[];thumbs=[];bgr_hashes=[]
  if (d/'recorded.webm').exists():
   with av.open(str(d/'recorded.webm')) as vid:
    for f in vid.decode(video=0):
     pts.append(round(float(f.pts*f.time_base)*1e6));thumbs.append(cv2.resize(f.to_ndarray(format='rgb24'),(32,18),interpolation=cv2.INTER_AREA));bgr_hashes.append(hashlib.sha256(f.to_ndarray(format='bgr24').tobytes()).hexdigest())
  check(prefix+'recorded_frame_count_matches',len(pts)==n)
  if (d/'PACKETS.json').exists():
   with np.load(d/'rtmw/observations.npz') as z:check(prefix+'recording_pixels_equal_inference_pixels',np.array_equal(z['frame_sha256'],bgr_hashes))
  err=None;mae=None
  if len(pts)==n and n:
   delta=np.array(pts)-pts[0]-ts;err=int(np.abs(delta).max())
   check(prefix+'relative_recorded_pts_within_2ms',err<=freeze['recording_pts_tolerance_us'])
   if len(receipts)==n:
    mae=float(np.abs(np.asarray(thumbs,dtype=float)-np.asarray([x['rgb_thumbnail'] for x in receipts],dtype=float)).mean())
    check(prefix+'recorded_content_matches_submitted_pixels_MAE_lt_6',mae<6)
   else:check(prefix+'recorded_content_matches_submitted_pixels_MAE_lt_6',False)
  else:check(prefix+'relative_recorded_pts_within_2ms',False)
  lat=[(a['completed_s']-a['received_s'])*1000 for a in sm['records']]
  last=float(bt['result_ready_perf_ms']-trace[-1]['capture_perf_ms']) if n else None
  check(prefix+'reported_time_recomputed',last is not None and abs(last-bt['last_frame_to_result_ms'])<.001)
  if r['cancelled']:check(prefix+'cancel_never_marked_complete',not r['input_complete'] and not r['client_input_complete'])
  else:
   check(prefix+'complete_input',r['input_complete'] and r['client_input_complete'])
   check(prefix+'after_last_frame_budget_1s',last is not None and last<=freeze['input_ready_budget_ms'])
  runs.append(dict(id=sid,clip=item['clip_id'],cancelled=r['cancelled'],frames=n,recorded_frames=len(pts),timestamp_error_us=err,
      recording_thumbnail_mae=mae,last_frame_to_tcn_pose_ready_ms=last,finish_request_to_ready_ms=r['client_finish_ms'],stop_total_including_recording_upload_ms=item['client_stop_total_ms'],
      rtmw_latency_p95_ms=float(np.percentile(lat,95)) if lat else None,mp_latency_p95_ms=float(np.percentile([x['lag_ms'] for x in mp['records']],95)),
      tcn_ms=mp['tcn_seconds']*1000,median_gap_ms=float(np.median(np.diff(ts))/1000),max_gap_ms=float(max(np.diff(ts))/1000),
      t3f_status=r['T3F']['model_status'],top3=[x for x in r['T3F'].get('ranking',[])[:3]]))
 out=dict(status='PASS' if all(x['pass'] for x in checks) else 'FAIL',trial=str(trial),checks=checks,runs=runs,
     verification='Same Codex separate arithmetic and recording audit, not independent human annotation or hardware webcam accuracy',
     training=False,private_data=False,whole_feedback_model=False)
 (trial/'VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(dict(status=out['status'],failures=[x['name'] for x in checks if not x['pass']],runs=runs),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
