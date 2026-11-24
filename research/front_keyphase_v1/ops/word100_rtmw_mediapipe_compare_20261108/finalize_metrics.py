from pathlib import Path
import json,hashlib
import numpy as np
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
f=read(U/'FROZEN.json');methods=[];detail=[];delta=[]
for method,label in [('mediapipe','현재 MediaPipe 전체 관측 경로'),('rtmw_gpu','RTMW-l · Intel GPU · 준비 후'),('rtmw_cpu','RTMW-l · CPU · 준비 후')]:
 vals=[];compute=[]
 for cid in f['speed_repeats']['clips']:
  clipvals=[]
  for rep in range(3):
   p=U/'speed'/cid/str(rep)/method;d=read(p/'RUN.json');vals.append(d['wall_seconds']);clipvals.append(d['wall_seconds'])
   if method=='mediapipe':t=read(p/'CAPTURE_TIMING.json')['totals'];compute.append(t['complete_frame_process_seconds']/d['frames']*1000)
   else:compute.append((d['person_detector_seconds']+d['pose_preprocess_infer_postprocess_seconds'])/d['frames']*1000)
  detail.append(dict(method=method,clip_id=cid,values=clipvals,median=float(np.median(clipvals))))
 methods.append(dict(method=method,label=label,n=len(vals),median=float(np.median(vals)),min=min(vals),max=max(vals),frame_compute_ms_median=float(np.median(compute))))
for cid in f['speed_repeats']['clips']:
 a=np.load(U/'speed'/cid/'0/rtmw_gpu/rtmw.npz');b=np.load(U/'speed'/cid/'0/rtmw_cpu/rtmw.npz')
 assert np.array_equal(a['frame_id'],b['frame_id']);d=np.linalg.norm(a['xy_px'][:,91:]-b['xy_px'][:,91:],axis=-1)
 delta.append(dict(clip_id=cid,hand_joint_distance_median_px=float(np.median(d)),hand_joint_distance_p95_px=float(np.quantile(d,.95)),max_px=float(np.max(d)),meaning='CPU/GPU numerical/model-output difference, not accuracy against human truth'))
events=[json.loads(x) for x in (U/'EVENTS.jsonl').read_text(encoding='utf-8').splitlines()];ready=[x for x in events if x['kind']=='rtmw_ready' and x['device']=='gpu'][0]
summary=dict(methods=methods,details=detail,gpu_median=methods[1]['median'],gpu_compile=ready['compile_seconds'],gpu_warm=ready['warmup_seconds'],cpu_gpu_difference=delta,timing_scope='sequential 3 clips x 3 repeats each; no simultaneous benchmark jobs. System load not controlled against other user apps. MP includes clip construction/preflight/probe/save, RTMW excludes persistent compile/warmup and compressed output serialization; neither includes SignRep/TCN/DTW. Per-frame compute is a separate nearer-scope comparison.')
save(U/'SPEED_SUMMARY.json',summary)
# Both methods on identical common-valid triplets, using same body scale/center.
rows=[];pool={'mp':[],'rt':[]}
for row in f['rows']:
 if not row['intervals']:continue
 p=U/'runs'/row['clip_id'];mp=np.load(p/'mediapipe/observations.npz');rt=np.load(p/'rtmw_gpu/rtmw.npz');ts=mp['timestamp_us']/1e6;main=np.zeros(len(ts),bool)
 for lo,hi in row['intervals']:main|=(ts>=lo)&(ts<=hi)
 a=mp['xy_px'][:,[33,54,41,62]];b=rt['xy_px'][:,[91,112,99,120]];valid=mp['usable'][:,[33,54,41,62]]&np.isfinite(b).all(2)
 sc=np.linalg.norm(mp['xy_px'][:,11]-mp['xy_px'][:,12],axis=1);c=(mp['xy_px'][:,11]+mp['xy_px'][:,12])/2;valid &= (sc>20)[:,None]
 mask=valid[2:]&valid[1:-1]&valid[:-2]&main[2:,None]&main[1:-1,None]&main[:-2,None]
 out={'clip_id':row['clip_id'],'common_joint_triplets':int(mask.sum())}
 for name,x in [('mp',a),('rt',b)]:
  q=(x-c[:,None,:])/sc[:,None,None];r=np.linalg.norm(q[2:]-2*q[1:-1]+q[:-2],axis=2)[mask];pool[name].extend(r.tolist());out[name]={'median':float(np.median(r)),'p95':float(np.quantile(r,.95))}
 rows.append(out)
rough={m:dict(median=float(np.median(v)),p95=float(np.quantile(v,.95)),n=len(v)) for m,v in pool.items()}
save(U/'TEMPORAL_PROXY.json',dict(rows=rows,aggregate=rough,meaning='body-normalized second difference of left/right wrists/index tips on shared available main-motion triplets. Both real movement and detector instability cause large values. Not true jitter error, not independent accuracy. Missing MP frames excluded equally and counted separately.'))
print(json.dumps(summary,ensure_ascii=False,indent=2));print('rough',rough)
