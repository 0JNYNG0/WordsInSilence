"""Verify saved native results without model inference."""
from pathlib import Path
import json,hashlib
import av,numpy as np
U=Path(__file__).resolve().parent
def main():
 c=json.loads((U/'NATIVE_COMPLETE.json').read_text(encoding='utf-8'));p=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'));scope=json.loads((U/'NATIVE_SCOPE.json').read_text());checks=[];summaries=[]
 def check(k,v):checks.append(dict(name=k,pass_=bool(v)))
 for path,h in p['protected'].items():check('protected:'+path,hashlib.sha256(Path(path).read_bytes()).hexdigest()==h)
 for r in c['runs']:
  d=U/'native_runs'/r['id'];spec=next(x for x in p['rows'] if x['clip_id']==r['clip_id']);n=r['frames'];ids=np.arange(n);times=np.array([x['timestamp_us'] for x in r['records']]);name=r['mode']+':'+r['label']
  with np.load(d/'rtmw/observations.npz') as z:obs={k:z[k] for k in z.files}
  with np.load(d/'pose_analysis_input.npz') as z:
   check(name+':reused_ids_exact',np.array_equal(z['frame_id'],ids));check(name+':reused_timestamps_exact',np.array_equal(z['timestamp_us'],times))
  check(name+':raw_ids_exact',np.array_equal(obs['frame_id'],ids));check(name+':raw_timestamps_exact',np.array_equal(obs['timestamp_us'],times))
  check(name+':no_loss_or_errors',r['input_complete'] and r['frames']==r['rtmw_frames']==r['recorded_frames'])
  source_ts=[];source_hash=[]
  with av.open(spec['source']) as v:
   first=None
   for f in v.decode(video=0):
    if first is None:first=f.pts
    source_ts.append(round(float((f.pts-first)*f.time_base)*1e6));source_hash.append(hashlib.sha256(f.to_ndarray(format='bgr24').tobytes()).hexdigest())
  check(name+':all_original_frames',len(source_ts)==n);check(name+':original_pts_exact',np.array_equal(source_ts,times));check(name+':source_pixels_exact',np.array_equal(source_hash,obs['frame_sha256']))
  pts=[]
  with av.open(str(d/'recorded.mp4')) as v:
   for f in v.decode(video=0):pts.append(round(float(f.pts*f.time_base)*1e6))
  check(name+':video_all_frames',len(pts)==n)
  err=int(np.max(np.abs(np.array(pts)-pts[0]-times))) if len(pts)==n else None
  check(name+':video_pts_within_2ms',err is not None and err<=scope['timestamp_tolerance_us'])
  with np.load(spec['saved_baseline']) as z:
   same_pixels=np.array_equal(z['frame_sha256'],obs['frame_sha256']);same_coords=np.array_equal(z['xy_px'],obs['xy_px'],equal_nan=True)
  check(name+':prior_frozen_rtmw_pixels_same',same_pixels);check(name+':prior_frozen_rtmw_coordinates_same',same_coords)
  check(name+':no_second_detection',r['second_pass_detector_calls']==0)
  check(name+':rtmw_ready_budget_1s',r['last_capture_to_rtmw_ready_ms']<=scope['last_capture_budget_ms'])
  check(name+':all_analysis_ready_budget_1s',r['last_capture_to_analysis_ready_ms']<=scope['last_capture_budget_ms'])
  check(name+':capture_lateness_le_one_frame',r['capture_lateness_max_ms']<=33.334)
  if r['mode']=='RTMW_PLUS_FROZEN_TCN':
   with np.load(d/'mp_observations.npz') as z:
    check(name+':mp_all_ids',np.array_equal(z['frame_id'],ids));check(name+':mp_all_timestamps',np.array_equal(z['timestamp_us'],times))
   mp=json.loads((d/'MP_TCN.json').read_text(encoding='utf-8'));top=mp['T3F'].get('ranking',[])[:3]
  else:top=[]
  s=json.loads((d/'rtmw/SESSION.json').read_text());lat=[(x['completed_s']-x['received_s'])*1000 for x in s['records']]
  summaries.append(dict(id=r['id'],label=r['label'],mode=r['mode'],frames=n,recording_timestamp_error_us=err,coordinates_identical_to_frozen_baseline=same_coords,
   last_capture_to_rtmw_ms=r['last_capture_to_rtmw_ready_ms'],last_capture_to_analysis_ms=r['last_capture_to_analysis_ready_ms'],last_capture_to_all_ms=r['last_capture_to_recording_and_analysis_ms'],rtmw_latency_p95_ms=float(np.percentile(lat,95)),capture_lateness_max_ms=r['capture_lateness_max_ms'],top3=top))
 out=dict(status='PASS' if all(x['pass_'] for x in checks) else 'PARTIAL',checks=checks,runs=summaries,verification='Same Codex separate recording/source/hash/array audit; not independent human or actual hardware webcam test')
 (U/'NATIVE_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(status=out['status'],failures=[x['name'] for x in checks if not x['pass_']],runs=summaries),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
