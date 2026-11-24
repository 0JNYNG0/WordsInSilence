"""Locked public recorded IDs14/15. No fitting, threshold or candidate selection."""
from pathlib import Path
from fractions import Fraction
import sys,os,time,json,traceback,hashlib,shutil
import numpy as np,av,cv2
H=Path(__file__).resolve().parent;O=H.parent;R=O/'word100_learned_rgb_motion_20261102';Q=O/'word100_learned_motion_path_20261102';N=O/'word100_learned_motion_invariance_20261102';V=O/'word100_learned_body_relation_20261102'
sys.path.insert(0,str(N));from prepare_real_fit import read,sha,exclusive,write,clock

def capacity():
 free=shutil.disk_usage(H).free
 if free<2*1024**3:raise RuntimeError('Preserve2GiB free; no old files deleted')
 return free

def encode(source,dest,mapping,scale_camera=False):
 assert not dest.exists();capacity();j=0;decoded=0
 with av.open(str(source)) as inp,av.open(str(dest),'w') as out:
  stream=None
  for i,frame in enumerate(inp.decode(video=0)):
   decoded+=1
   if j>=len(mapping) or mapping[j]!=i:continue
   rgb=frame.to_ndarray(format='rgb24');h,w=rgb.shape[:2]
   if scale_camera:
    neww=int(w*.9);newh=int(h*.9);small=cv2.resize(rgb,(neww,newh),interpolation=cv2.INTER_AREA)
    color=np.median(rgb[:12,:12].reshape(-1,3),axis=0).astype(np.uint8);canvas=np.empty_like(rgb);canvas[:]=color
    left=(w-neww)//2+int(w*.03);top=(h-newh)//2;assert left>=0 and left+neww<=w and top+newh<=h
    canvas[top:top+newh,left:left+neww]=small;rgb=canvas
   if stream is None:
    stream=out.add_stream('libx264',rate=30);stream.width=w;stream.height=h;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast'}
   while j<len(mapping) and mapping[j]==i:
    f=av.VideoFrame.from_ndarray(rgb,format='rgb24');f.pts=j;f.time_base=Fraction(1,30)
    for packet in stream.encode(f):out.mux(packet)
    j+=1
  assert j==len(mapping) and stream is not None
  for packet in stream.encode():out.mux(packet)
 return decoded

def main():
 assert read(R/'DEVELOPMENT_GATE.json')['status']=='PASS_TO_LOCKED_EVALUATION'
 assert read(R/'SAVED_VERIFICATION.json')['status'].startswith('PASS')
 source=read(N/'LOCKED_REVIEW_MANIFEST.json')['rows'];assert len(source)==8
 frozen=dict(at=clock(),scope='4 supported expressions, recorded IDs14/15,8 sources; no biometric identity claim',
  prior_exposure='Used for old TCN evaluation; not globally untouched. Not used for current motion neural training/tuning/selection.',
  source_manifest_sha256=sha(N/'LOCKED_REVIEW_MANIFEST.json'),physical_targets_sha256=sha(N/'LOCKED_PHYSICAL_TARGETS.json'),
  conditions=['original','speed_0.8','speed_1.25','scale0.9_shift0.03'],inputs=32,
  candidates=['previous_separated_relative','fixed_rgb_residual'],seeds=[42,43,44],deployment_seed=42,
  no_training=True,no_selection=True,no_threshold_changes=True,source_physical_labels='18 previously reviewed physical direction/stable intervals, not independent expert semantic labels',
  acceptance=dict(original_normal_motion_confirmed_at_least27_of30=True,all_normal_variants_confirmed_at_least108_of120=True,
   no_new_opposite_direction_candidates_on_normal=True,old_features_and_DTW_preserved=True),
  further_completion='Even a pass is not whole goal completion. Pose-only gaps, meaningful guide/order contrasts, integrated new-video demo and total latency remain to verify.',
  preserve='C/T3-F, prior demos, sources, locked results. No private/user/mentor media access.',
  pins={str(p):sha(p) for p in [R/'PLAN.json',R/'RGB_NORMALIZATION.json',Q/'CONTRACT.json']+
    [R/'runs'/f'rgb_residual_{s}/model.pt' for s in [42,43,44]]+[Q/'runs'/f'separated_relative_{s}/model.pt' for s in [42,43,44]]})
 if (H/'FROZEN.json').exists():
  previous=read(H/'FROZEN.json');frozen['at']=previous['at'];assert previous==frozen
 else:exclusive(H/'FROZEN.json',frozen)
 if not (H/'RUN_STARTED.json').exists():exclusive(H/'RUN_STARTED.json',dict(pid=os.getpid(),at=clock()))
 sys.path.insert(0,str(V));import demo_adapter
 demo_adapter.U=H;demo_adapter.warmup();rows=[];start=time.perf_counter()
 for r in source:
  path=Path(r['local_source']);assert sha(path)==r['source_sha256'];n=r['frames']
  for kind,speed in [('original',1),('speed_0.8',.8),('speed_1.25',1.25),('scale0.9_shift0.03',1)]:
   key=r['clip_id']+'_'+kind;meta=H/'records'/f'{key}.json';dest=H/'inputs'/key
   if meta.exists():
    record=read(meta);assert sha(record['input'])==record['input_sha256'];rows.append(record);continue
   capacity();mapping=np.arange(0,n,speed).astype(int).tolist();video=path
   if kind!='original':
    video=H/'videos'/f'{key}.mp4';video.parent.mkdir(exist_ok=True);vm=video.with_suffix('.json')
    if not vm.exists():
     decoded=encode(path,video,mapping,kind=='scale0.9_shift0.03');assert decoded==n
     exclusive(vm,dict(source_sha256=r['source_sha256'],video_sha256=sha(video),source_frame_indices=mapping))
    else:assert sha(video)==read(vm)['video_sha256'] and read(vm)['source_frame_indices']==mapping
   result=read(dest/'RESULT.json') if (dest/'RESULT.json').exists() else demo_adapter.analyze(video,dest,r['word_id'])
   a=np.load(dest/'analysis.npz');assert len(a['frame_id'])==len(mapping) and np.all(np.diff(a['timestamp_us'])>0)
   assert np.array_equal(a['frame_id'],np.arange(len(mapping)))
   record=dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],kind=kind,role='locked_task_evaluation',source=str(path),source_sha256=r['source_sha256'],
    video=str(video),video_sha256=sha(video),source_frame_indices=mapping,input=str(dest/'analysis.npz'),input_sha256=sha(dest/'analysis.npz'),result=str(dest/'RESULT.json'),
    seconds=result['latency']['total_analysis_seconds'],device=result['device'])
   exclusive(meta,record);rows.append(record);write(H/'PREP_PROGRESS.json',dict(pid=os.getpid(),done=len(rows),total=32,last=key,at=clock(),seconds=time.perf_counter()-start,free_bytes=capacity()))
   print('prepared locked',len(rows),32,key,flush=True)
 for p,h in frozen['pins'].items():assert sha(p)==h
 exclusive(H/'INPUT_MANIFEST.json',dict(status='ALL32_LOCKED_PUBLIC_INPUTS_READY',rows=rows,no_training=True))
 write(H/'PREP_PROGRESS.json',dict(status='COMPLETE',done=32,total=32,pid=os.getpid(),at=clock()))

if __name__=='__main__':
 try:main()
 except BaseException:
  if not (H/'FIRST_FAILURE.txt').exists():(H/'FIRST_FAILURE.txt').write_text(traceback.format_exc(),encoding='utf8')
  raise
