"""Fit-only real pixel speed variations, through the preserved live pipeline."""
from pathlib import Path
from fractions import Fraction
from datetime import datetime,timezone,timedelta
import json,sys,os,hashlib,shutil,traceback,time
import av,numpy as np
N=Path(__file__).resolve().parent;O=N.parent;P=O/'word100_learned_guide_feedback_20261102';M=O/'word100_learned_motion_order_20261102';V=O/'word100_learned_body_relation_20261102'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def exclusive(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def capacity():
 free=shutil.disk_usage(N).free
 if free<2*1024**3:raise RuntimeError(f'Preserve at least2GiB free; currently {free}. No old files deleted.')
 return free
def clock():return datetime.now(timezone(timedelta(hours=9))).isoformat()
def encode(frames,indices,dest):
 assert not dest.exists();capacity()
 with av.open(str(dest),'w') as out:
  stream=out.add_stream('libx264',rate=30);stream.width=frames[0].shape[1];stream.height=frames[0].shape[0];stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
  for i,k in enumerate(indices):
   f=av.VideoFrame.from_ndarray(frames[int(k)],format='rgb24');f.pts=i;f.time_base=Fraction(1,30)
   for packet in stream.encode(f):out.mux(packet)
  for packet in stream.encode():out.mux(packet)
def main():
 manifest=read(P/'INPUT_MANIFEST.json');rows=[r for r in manifest['rows'] if r['role']=='fit'];assert len(rows)==36 and {r['person'] for r in rows}==set(range(1,10))
 plan=dict(status='FROZEN_BEFORE_NEW_FIT_PIXELS',goal='Jointly improve actual-speed invariance, guide-time boundaries and observed path evidence',
  previous_turn='progress:108 reviewed episodes,six motion-head trainings,12 actual reverse extractions,288 frozen-path comparisons',
  people=list(range(1,10)),words=sorted({r['word_id'] for r in rows}),source_videos=36,conditions=['original','speed_0.8','speed_1.25'],expected_inputs=108,
  no_private_or_dev_inputs=True,models_preserved=['C','T3-F','old shape','old relations','old learned DTW','old motion heads'],
  source_motion_labels=str(M/'MOTION_TARGETS.json'),source_motion_labels_sha256=sha(M/'MOTION_TARGETS.json'),
  masks='Full-frame sensor and body-relative coordinates, no synthetic lower cutoff; missing evidence remains missing',
  storage='Separate unit only; no original/development/frozen result deletion; stop before free space <2GiB',
  after_input='Verify source/PTS mappings and input drift on FIT; freeze retraining and state-order conditions before new DEV predictions',
  pretrained_finetune_status='Prior final SignRep projection rank8 update retained; no new backbone modification during preparation')
 if not (N/'PLAN.json').exists():exclusive(N/'PLAN.json',plan)
 else:assert read(N/'PLAN.json')==plan
 exclusive(N/'RUN_STARTED.json',dict(pid=os.getpid(),at=clock(),free_bytes=capacity())) if not (N/'RUN_STARTED.json').exists() else None
 sys.path.insert(0,str(V));import demo_adapter
 demo_adapter.U=N;demo_adapter.warmup();done=[];start=time.perf_counter()
 for r in rows:
  cid=r['clip_id'];assert sha(r['source'])==r['source_sha256'];maps={};frames=None
  for kind,speed in [('original',1),('speed_0.8',.8),('speed_1.25',1.25)]:
   key=f'{cid}_{kind}';meta=N/'input_records'/f'{key}.json';d=N/'inputs'/key
   if meta.exists():
    entry=read(meta);assert sha(d/'analysis.npz')==entry['analysis_sha256'];done.append(entry);continue
   capacity();indices=np.minimum(np.arange(0,r['frames'],speed).astype(int),r['frames']-1).tolist()
   source=Path(r['source'])
   if kind!='original':
    source=N/'fit_videos'/f'{key}.mp4';source.parent.mkdir(exist_ok=True)
    encoded=source.with_suffix('.json')
    if not encoded.exists():
     if frames is None:
      with av.open(r['source']) as container:frames=[f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
      assert len(frames)==r['frames']
     encode(frames,indices,source);exclusive(encoded,dict(sha256=sha(source),source_indices=indices,source_sha256=r['source_sha256']))
    else:assert read(encoded)['sha256']==sha(source) and read(encoded)['source_indices']==indices
   if (d/'RESULT.json').exists():result=read(d/'RESULT.json')
   else:result=demo_adapter.analyze(source,d,r['word_id'])
   a=np.load(d/'analysis.npz');assert len(a['frame_id'])==len(indices) and np.all(np.diff(a['timestamp_us'])>0)
   assert np.array_equal(a['frame_id'],np.arange(len(indices)))
   entry=dict(clip_id=cid,word_id=r['word_id'],person=r['person'],role='fit',kind=kind,source=r['source'],source_sha256=r['source_sha256'],path=str(source),video_sha256=sha(source),
    source_frame_indices=indices,input=str(d/'analysis.npz'),analysis_sha256=sha(d/'analysis.npz'),result=str(d/'RESULT.json'),
    frames=len(indices),seconds=result['latency']['total_analysis_seconds'],device=result['device'],private_inputs=False)
   exclusive(meta,entry);done.append(entry);write(N/'PREP_PROGRESS.json',dict(pid=os.getpid(),at=clock(),done=len(done),total=108,last=key,free_bytes=capacity(),elapsed_seconds=time.perf_counter()-start))
   print('prepared',len(done),108,key,flush=True)
  del frames
 exclusive(N/'INPUT_MANIFEST.json',dict(status='ALL108_FIT_INPUTS_EXTRACTED',rows=done,source_plan_sha256=sha(N/'PLAN.json'),new_training=0,development_inputs=0))
 write(N/'PREP_PROGRESS.json',dict(status='COMPLETE',done=len(done),total=108,pid=os.getpid(),at=clock(),free_bytes=capacity()))
if __name__=='__main__':
 try:main()
 except BaseException:
  if not (N/'FIRST_FAILURE.txt').exists():(N/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
