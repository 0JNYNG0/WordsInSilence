from pathlib import Path
import os,sys,json,time,hashlib,importlib.util,traceback,argparse
os.environ['OPENVINO_TELEMETRY_DISABLED']='1'
os.environ['INTEL_OPENVINO_TELEMETRY_CONSENT']='0'
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent;O=U.parent
sys.path.insert(0,str(U/'vendor'))
import numpy as np
import av,cv2
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def event(**kw):
 kw.update(at=time.time());s=json.dumps(kw,ensure_ascii=False)
 with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as f:f.write(s+'\n')
 print(s,flush=True)
def frames(p):
 with av.open(str(p)) as c:
  stream=c.streams.video[0];base=None
  for i,f in enumerate(c.decode(stream)):
   if base is None:base=f.pts
   yield i,round(float((f.pts-base)*stream.time_base)*1e6),f.to_ndarray(format='bgr24')
def mp_capture(row,d):
 spec=importlib.util.spec_from_file_location('_comparison_original_parallel',O/'word100_integrated_test_20261104/parallel_capture.py')
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 # Change only new-output ownership in this isolated module instance.
 m.HERE=U;d.mkdir(parents=True,exist_ok=False)
 t=time.perf_counter();ob,face=m.capture_parallel(Path(row['source']),d);wall=time.perf_counter()-t
 write(d/'RUN.json',dict(status='COMPLETE',wall_seconds=wall,frames=ob.n_frames,source_sha256=row['source_sha256'],method='unchanged current full capture with joint_v3 and face; no SignRep'))
 return wall
class RTM:
 def __init__(self,device):
  from rtmlib import RTMPose,YOLOX
  import openvino as ov
  core=ov.Core();pose=next((U/'models/rtmw_l_384').rglob('*.onnx'));det=next((U/'models/person_yolox_m').rglob('*.onnx'))
  graph=core.read_model(pose);shape=graph.input(0).partial_shape
  assert str(shape[2])=='384' and str(shape[3])=='288',str(shape)
  self.device=device;t=time.perf_counter()
  self.det=YOLOX(str(det),model_input_size=(640,640),backend='openvino',device=device)
  self.pose=RTMPose(str(pose),model_input_size=(288,384),backend='openvino',device=device)
  self.compile=time.perf_counter()-t
  self.devices={k:[str(x) for x in v.compiled_model.get_property('EXECUTION_DEVICES')] for k,v in [('det',self.det),('pose',self.pose)]}
  assert all(device.upper() in x for ds in self.devices.values() for x in ds)
  t=time.perf_counter()
  for _ in range(3):
   self.det(np.zeros((640,640,3),np.uint8));self.pose(np.zeros((384,288,3),np.uint8),bboxes=[[0,0,288,384]])
  self.warm=time.perf_counter()-t
  event(kind='rtmw_ready',device=device,compile_seconds=self.compile,warmup_seconds=self.warm,graph_shape=str(shape),actual_devices=self.devices)
 def run(self,row,d):
  d.mkdir(parents=True,exist_ok=False);allxy=[];allscores=[];ts=[];ids=[];boxes=[];npeople=[];rgbhash=[];costs=[]
  total=time.perf_counter();last=total
  for i,tus,bgr in frames(row['source']):
   decode=time.perf_counter()-last;t=time.perf_counter();bb=self.det(bgr);td=time.perf_counter()-t
   counts=len(bb);box=np.full(4,np.nan,np.float32);xy=np.full((133,2),np.nan,np.float32);sc=np.full(133,np.nan,np.float32)
   t=time.perf_counter()
   if counts:
    area=(bb[:,2]-bb[:,0])*(bb[:,3]-bb[:,1]);box=np.asarray(bb[int(np.argmax(area))],np.float32)
    # Official exported Normalize.to_rgb=true. Detector remains BGR.
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    out,s=self.pose(rgb,bboxes=[box]);assert out.shape==(1,133,2);xy=out[0].astype(np.float32);sc=s[0].astype(np.float32)
   tp=time.perf_counter()-t
   allxy.append(xy);allscores.append(sc);boxes.append(box);npeople.append(counts);ts.append(tus);ids.append(i);costs.append([decode,td,tp])
   rgbhash.append(hashlib.sha256(bgr.tobytes()).hexdigest())
   last=time.perf_counter()
  wall=time.perf_counter()-total
  xy=np.asarray(allxy);sc=np.asarray(allscores);costs=np.asarray(costs)
  np.savez_compressed(d/'rtmw.npz',xy_px=xy,score=sc,frame_id=np.asarray(ids,np.int64),timestamp_us=np.asarray(ts,np.int64),bbox=np.asarray(boxes),n_people=np.asarray(npeople),timings=costs,frame_sha256=np.asarray(rgbhash),source_width=bgr.shape[1],source_height=bgr.shape[0])
  write(d/'RUN.json',dict(status='COMPLETE',wall_seconds=wall,frames=len(ids),source_sha256=row['source_sha256'],actual_device=self.devices,compile_seconds_excluded=self.compile,warmup_seconds_excluded=self.warm,decode_seconds=float(costs[:,0].sum()),person_detector_seconds=float(costs[:,1].sum()),pose_preprocess_infer_postprocess_seconds=float(costs[:,2].sum()),zero_person_frames=int((np.asarray(npeople)==0).sum()),multiple_person_frames=int((np.asarray(npeople)>1).sum()),color='BGR person detector, RGB pose',threshold_applied_to_saved_coordinates=False,scope='single video native all frames, decode/detection/pose and audit hashing, excludes engine compile and npz serialization'))
  return wall
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['main','speed'],default='main');args=ap.parse_args()
 plan=read(U/'FROZEN.json')
 for p,h in plan['protected'].items():assert sha(p)==h,p
 for p,h in plan['models'].items():assert sha(U/p)==h,p
 if args.mode=='main':
  engine=RTM('gpu')
  for row in plan['rows']:
   assert sha(row['source'])==row['source_sha256'];cid=row['clip_id']
   for method in ['mediapipe','rtmw_gpu']:
    d=U/'runs'/cid/method
    if (d/'RUN.json').exists():event(kind='reuse_completed',clip=cid,method=method);continue
    if d.exists():raise RuntimeError('Inspect partial first: '+str(d))
    event(kind='start',clip=cid,method=method)
    wall=mp_capture(row,d) if method=='mediapipe' else engine.run(row,d)
    event(kind='done',clip=cid,method=method,seconds=wall)
  write(U/'RUN_COMPLETE.json',dict(clips=len(plan['rows']),training=0,at=time.time()))
 else:
  gpu=RTM('gpu');cpu=RTM('cpu')
  for cid in plan['speed_repeats']['clips']:
   row=next(r for r in plan['rows'] if r['clip_id']==cid)
   for rep in range(3):
    order=['mediapipe','rtmw_gpu','rtmw_cpu'] if rep%2==0 else ['rtmw_cpu','rtmw_gpu','mediapipe']
    for method in order:
     d=U/'speed'/cid/str(rep)/method
     if (d/'RUN.json').exists():continue
     event(kind='speed_start',clip=cid,rep=rep,method=method)
     wall=mp_capture(row,d) if method=='mediapipe' else (gpu if method.endswith('gpu') else cpu).run(row,d)
     event(kind='speed_done',clip=cid,rep=rep,method=method,seconds=wall)
  write(U/'SPEED_COMPLETE.json',dict(clips=3,repeats=3,training=0,at=time.time()))
 for p,h in plan['protected'].items():assert sha(p)==h,p
if __name__=='__main__':
 try:main()
 except BaseException:
  p=U/('FIRST_FAILURE_'+str(time.time_ns())+'.txt');p.write_text(traceback.format_exc(),encoding='utf-8');raise
