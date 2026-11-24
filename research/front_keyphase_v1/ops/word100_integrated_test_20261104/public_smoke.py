"""One integration invocation on the existing named public input, no fitting."""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
U=Path(__file__).resolve().parent;O=U.parent;sys.path.insert(0,str(U))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
pins=read(U/'EXECUTION_FROZEN.json')['pins']
for p,h in pins.items():assert sha(p)==h,p
original=read(O/'word100_local_event_runtime_20261103/PUBLIC_RUN_RESULT.json')
source=read(O/'word100_local_event_runtime_20261103/PUBLIC_INPUT_FROZEN.json')
video=Path(source['source']);assert sha(video)==source['source_sha256']
jid='a07dd7a4bca0488c99c36a3c4bb38f52';d=U/'public_runtime_runs'/jid/'analysis'
assert not d.exists();write(U/'PUBLIC_SMOKE_STARTED.json',dict(job_id=jid,source=str(video),source_sha256=sha(video),training=False))
import integrated_adapter as adapter
try:
 warm=adapter.warmup();print('WARM',warm,flush=True)
 start=time.perf_counter();r=adapter.analyze(video,d,'WORD1133',provenance=dict(source_role='public_runtime_inference',source_sha256=sha(video),word_id='WORD1133',training=False,tuning=False,candidate_selection=False))
 old=read(original['result']);assert r['path']==old['path']
 assert r['overlay']==old['overlay'];assert r['stages']==old['stages']
 assert r['runtime_individual_reference_video_comparisons']==0
 previous=read(U/'public_tcn.json')['tcn']
 for k in ['C','T3F']:assert r['tcn'][k]['ranking']==previous[k]['ranking']
 # Preview detector consumes a single publicly sourced decoded frame; no camera.
 import av,cv2
 with av.open(str(video)) as c:
  frame=next(f for i,f in enumerate(c.decode(video=0)) if i==50)
  im=frame.to_ndarray(format='bgr24');im=cv2.resize(im,(640,360));ok,jpeg=cv2.imencode('.jpg',im);assert ok
  preview=adapter.preview(jpeg.tobytes());assert preview['saved'] is False
 for p,h in pins.items():assert sha(p)==h,p
 write(U/'PUBLIC_SMOKE.json',dict(status='PASS',job_id=jid,result=str(d/'INTEGRATED_FEEDBACK.json'),result_sha256=sha(d/'INTEGRATED_FEEDBACK.json'),source_sha256=sha(video),word_id='WORD1133',warmup=warm,
  native_frames=r['frames'],seconds=r['integrated_analysis_seconds'],old_feedback_and_overlay_exact=True,old_path_exact=True,TCN_rankings_exact=True,public_preview_result=preview,training=0,user_content_access=0,
  limitation='Single existing public integration regression; not new accuracy/generalization evaluation'))
 print('PASS',r['integrated_analysis_seconds'],flush=True)
except BaseException:
 with (U/'PUBLIC_SMOKE_FIRST_FAILURE.txt').open('x',encoding='utf8') as f:f.write(traceback.format_exc())
 raise
finally:adapter.close()
