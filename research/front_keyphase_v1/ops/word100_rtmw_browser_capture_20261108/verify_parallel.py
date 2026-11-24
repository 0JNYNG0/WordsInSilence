from pathlib import Path
import json,hashlib
import av,numpy as np,cv2
U=Path(__file__).resolve().parent
def main():
 now=json.loads((U/'NATIVE_PARALLEL_COMPLETE.json').read_text(encoding='utf-8'));old=json.loads((U/'NATIVE_COMPLETE.json').read_text(encoding='utf-8'));freeze=json.loads((U/'FROZEN.json').read_text(encoding='utf-8'))
 checks=[];rows=[]
 def check(k,v):checks.append(dict(name=k,pass_=bool(v)))
 for p,h in freeze['protected'].items():check('protected:'+p,hashlib.sha256(Path(p).read_bytes()).hexdigest()==h)
 for r in now['runs']:
  prev=next(x for x in old['runs'] if x['clip_id']==r['clip_id'] and x['mode']=='RTMW_PLUS_FROZEN_TCN');d=U/'native_runs'/r['id'];before=U/'native_runs'/prev['id'];name=r['label']
  with np.load(d/'mp_observations.npz') as a,np.load(before/'mp_observations.npz') as b:
   for k in ['xy_px','timestamp_us','frame_id','observed','usable','conf','hand_score']:check(name+':same_MP_'+k,np.array_equal(a[k],b[k],equal_nan=True))
  with np.load(d/'rtmw/observations.npz') as a,np.load(before/'rtmw/observations.npz') as b:
   for k in ['xy_px','score','frame_id','timestamp_us','frame_sha256']:check(name+':same_RTMW_'+k,np.array_equal(a[k],b[k],equal_nan=True) if a[k].dtype.kind!='U' else np.array_equal(a[k],b[k]))
   ts=a['timestamp_us'];ids=a['frame_id']
  with np.load(d/'pose_analysis_input.npz') as a:
   check(name+':reused_times_exact',np.array_equal(a['timestamp_us'],ts));check(name+':reused_ids_exact',np.array_equal(a['frame_id'],ids))
  mp=json.loads((d/'MP_TCN.json').read_text(encoding='utf-8'));prevmp=json.loads((before/'MP_TCN.json').read_text(encoding='utf-8'))
  for m in ['T3F','C']:
   check(name+':same_'+m+'_ranking_and_scores',mp[m]['ranking']==prevmp[m]['ranking'])
   check(name+':same_'+m+'_checkpoint',mp[m]['checkpoint_sha256']==prevmp[m]['checkpoint_sha256'])
  pts=[]
  with av.open(str(d/'recorded.mp4')) as v:
   for f in v.decode(video=0):pts.append(round(float(f.pts*f.time_base)*1000000))
  check(name+':all_video_frames',len(pts)==r['frames']);check(name+':all_original_frames',r['frames']==prev['frames']);check(name+':no_loss',r['input_complete']);check(name+':video_pts_exact',np.array_equal(np.array(pts)-pts[0],ts))
  check(name+':TCN_observations_once',mp['detector_calls']==r['frames']);s=json.loads((d/'rtmw/SESSION.json').read_text());check(name+':RTMW_once',s['detector_calls']==r['frames'])
  check(name+':no_whole_video_second_pass',r['second_pass_detector_calls']==0);check(name+':one_second_budget',r['last_capture_to_recording_and_analysis_ms']<=1000)
  check(name+':one_frame_capture_lateness',r['capture_lateness_max_ms']<=33.334)
  spec=next(x for x in freeze['rows'] if x['clip_id']==r['clip_id']);diffs=[]
  with av.open(spec['source']) as src,av.open(str(d/'recorded.mp4')) as rec:
   for a,b in zip(src.decode(video=0),rec.decode(video=0)):
    aa=cv2.resize(a.to_ndarray(format='rgb24'),(32,18),interpolation=cv2.INTER_AREA).astype(float)
    bb=cv2.resize(b.to_ndarray(format='rgb24'),(32,18),interpolation=cv2.INTER_AREA).astype(float)
    diffs.append(float(np.abs(aa-bb).mean()))
  check(name+':recorded_content_matches_source',len(diffs)==r['frames'] and max(diffs)<6)
  lat=[(x['completed_s']-x['received_s'])*1000 for x in s['records']]
  rows.append(dict(id=r['id'],label=name,frames=r['frames'],last_capture_to_rtmw_ms=r['last_capture_to_rtmw_ready_ms'],last_capture_to_all_ms=r['last_capture_to_recording_and_analysis_ms'],previous_all_ms=prev['last_capture_to_recording_and_analysis_ms'],previous_tcn_ms=prevmp['tcn_seconds']*1000,tcn_ms=mp['tcn_seconds']*1000,capture_lateness_max_ms=r['capture_lateness_max_ms'],mp_compute_p50_ms=float(np.median([x['compute_ms'] for x in mp['records']])),mp_compute_p95_ms=float(np.percentile([x['compute_ms'] for x in mp['records']],95)),rtmw_latency_p95_ms=float(np.percentile(lat,95)),tcn_top1=mp['T3F']['ranking'][0]['label'],recording_thumbnail_max_mae=max(diffs)))
 out=dict(status='PASS' if all(x['pass_'] for x in checks) else 'PARTIAL',checks=checks,runs=rows,verification='Same Codex independent arithmetic from saved outputs. No rerun inference. Not human semantic labels or actual webcam/Unity.');(U/'NATIVE_PARALLEL_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(dict(status=out['status'],failures=[x['name'] for x in checks if not x['pass_']],runs=rows),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
