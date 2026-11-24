import prepare as R
from pathlib import Path
import os,time,traceback,importlib.util,zipfile,shutil
import numpy as np
U=R.U;O=R.O;I=R.I
def mod(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
rule=mod('visible_hand_merge',O/'word100_fullframe_hand_merge_20261020/candidate.py')
Q=mod('visible_shape_evidence',O/'word100_finger_evidence_20261020/candidate.py')
E=mod('visible_angle_evidence',O/'word100_angle_evidence_20261021/candidate.py')
def clean(x):
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 return x
def roi(x,v,b):return v&np.isfinite(x).all(-1)&(x[...,0]>=b[0])&(x[...,0]<b[2])&(x[...,1]>=b[1])&(x[...,1]<b[3])
def fuse(mp_ok,mp_conflict,mp_identity_conflict,y_ok,y_conflict,y_identity_conflict):
 return (mp_ok|y_ok)&~(mp_conflict|y_conflict)&~(mp_identity_conflict|y_identity_conflict)[...,None]
def freeze():
 f=R.read(U/'FROZEN.json');assert R.read(U/'PREPARED_VERIFIED.json')['status']=='PASS899_TRAIN_ONLY_SCHEDULE';R.check(f['pins']);R.check(f['protected']);old=R.read(I/'FROZEN.json');rows=[];pins={}
 for row in f['rows']:
  cid=row['clip_id']
  if cid not in f['selected_ids']:continue
  recp=I/'records'/(cid+'__masked.json');rec=R.read(recp);assert rec['context_sha256']==R.sha(I/'FROZEN.json');R.check(rec['outputs']);pins[str(recp)]=R.sha(recp)
  vp=Path(rec['source_video']) if rec['source_video'] else None
  if vp is None or not vp.exists():
   cached=old['cache'].get(cid);vp=Path(cached['video']) if cached else U/'sources'/(cid+'.mp4')
  if vp.exists():assert R.sha(vp)==row['input']['source']['sha256']
  rows.append(dict(**row,video=str(vp),old_records=str(recp)))
 zero=np.zeros((2,15),bool);one=~zero;idzero=np.zeros(2,bool);idone=~idzero
 assert fuse(zero,zero,idzero,one,zero,idzero).all()
 assert not fuse(one,zero,idzero,one,one,idzero).any()
 assert not fuse(zero,zero,idone,one,zero,idzero).any()
 assert fuse(one,zero,idzero,zero,zero,idzero).all()
 R.write(U/'FUSION_FIXTURES.json',dict(status='PASS',yolo_can_add_when_mp_unassessed=True,shape_conflict_not_overridden=True,identity_conflict_not_overridden=True,mp_evidence_not_lost_when_yolo_absent=True))
 extra=[Path(__file__),U/'model_io.py',Path(rule.__file__),Path(Q.__file__),Path(E.__file__),U/'candidate.py',U/'FROZEN.json',U/'PREPARED_VERIFIED.json',R.ROOT/'Tools/models/hand_landmarker.task',O/'word100_yolo_hand_fourway_20261025/yolo_hand_best.pt',O/'word100_yolo_hand_fourway_20261025/experiment.py']
 pins.update({str(p):R.sha(p) for p in extra});R.write(U/'IMAGE_FROZEN.json',dict(at=R.now(),rows=rows,pins=pins,protected=f['protected'],mp_options=f['mp_options'],
  methods=['existing native coordinates,unchanged','native plus same-frame MP full/crop corroboration','MP then selective YOLO full/crop corroboration if MP cannot corroborate computable native angles'],
  decisions='Only existing arm/physical-hand duplicate rules for matching. Full/crop and native angle comparison separate. No coordinate replacement. An available prediction disagreement blocks fused corroboration,not hidden by choosing another detector. Unrequested and failed and conflicting kept separate. Same-model agreement not anatomical truth.',
  timing='Only current new calls timed. Exact old identical image/model full/crop outputs reused; cached calls counted separately andnotincluded in fresh latency mean. No end-to-endFPS claim.',
  gates='Every rawcoordinate/mask/time/side/scale preserved,all3185targetframes accounted,failures notdropped; source/ROI respected; independent recompute matching/angleevidence before interpretation; normal/crossing public-image inspection. No sign accuracy claim,TCNrun or training here.',training=0,production_changed=False))
 for n in ['records','outputs','sources','starts','images']:(U/n).mkdir()
 print('FROZEN_IMAGE100',sum(len(r['targets']) for r in rows),flush=True)

def detect(lm,rgb,mp,box=None):
 t=time.perf_counter();res=lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb)));h,w=rgb.shape[:2];off=np.asarray(box[:2] if box else [0,0]);out=[]
 for pts,s in zip(res.hand_landmarks,res.handedness):
  d=dict(xy=np.array([[p.x*w,p.y*h] for p in pts])+off,score=float(s[0].score),label=s[0].category_name)
  if box:d['box']=box
  out.append(d)
 return out,(time.perf_counter()-t)*1000

def evidence(full,crops,base,valid,old,sides,pose,pu,w,h,box,side):
 decisions=rule.merge(full,crops,old,sides,pose,pu,w,h);ds=[d for d in decisions if d['side']==side and d['reason']=='confirm_existing_hand'];out=dict(side=side,identity_confirmed=False,pair_available=False,available=np.zeros(15,bool),unresolved=np.zeros(15,bool),full_index=None,crop_index=None,reason='no_unique_current_arm_confirmation',decisions=decisions)
 out['identity_conflict']=any(d['reason'] in ('existing_identity_conflict','unresolved_existing_overlap') and (d['side']==side or any(sides[j]==side for j in d['matched_old'])) for d in decisions)
 if len(ds)!=1:return out
 d=ds[0];a=np.asarray(full[d['index']]['xy'],float);good,scale=rule.context(pose,pu,w,h);matches=[]
 for i,c in enumerate(crops):
  b=np.asarray(c['xy'],float)
  if c['requested_side']!=side or rule.arm(b,pose,good,scale)!=side:continue
  if rule.relation(a,b,side,side,scale).startswith('same_'):matches.append(i)
 out.update(identity_confirmed=True,full_index=d['index'],reason='crop_missing_or_ambiguous')
 if len(matches)!=1:return out
 j=matches[0];b=np.asarray(crops[j]['xy'],float);av=roi(a,rule.inside(a,w,h),box);bv=roi(b,rule.inside(b,w,h),box);aa=Q.compare_shape(base,valid,a,av);bb=Q.compare_shape(base,valid,b,bv)
 out.update(crop_index=j,pair_available=bool(aa['pair_available'] and bb['pair_available']),available=aa['angle_available']&bb['angle_available'],unresolved=aa['angle_unresolved']|bb['angle_unresolved'],reason='paired_shape_assessed',full_shape=aa,crop_shape=bb)
 return out

def run(limit=100):
 f=R.read(U/'IMAGE_FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(U/'RUNNING.json').exists(),'Inspect live execution before resuming';R.write(U/'RUNNING.json',dict(pid=os.getpid(),at=R.now(),stage='image_recheck',limit=limit))
 import cv2,torch,mediapipe as mp
 from mediapipe.tasks.python import vision
 from mediapipe.tasks.python.core.base_options import BaseOptions
 cv2.setNumThreads(1);torch.set_num_threads(4);torch.set_num_interop_threads(1)
 loader=mod('visible_yolo_loader',U/'model_io.py');ym,environment=loader.load_model();exp=mod('visible_frozen_yolo',O/'word100_yolo_hand_fourway_20261025/experiment.py')
 if not(U/'ENVIRONMENT.json').exists():R.write(U/'ENVIRONMENT.json',environment)
 opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R.ROOT/'Tools/models/hand_landmarker.task')),running_mode=vision.RunningMode.IMAGE,**f['mp_options']);zips={};done=0;start=time.perf_counter()
 with vision.HandLandmarker.create_from_options(opts) as lm:
  detect(lm,np.zeros((1080,1920,3),np.uint8),mp);exp.ydetect(ym,np.zeros((1080,1920,3),np.uint8))
  for row in f['rows']:
   cid=row['clip_id'];rp=U/'records'/(cid+'.json')
   if rp.exists():r=R.read(rp);assert r['context_sha256']==R.sha(U/'IMAGE_FROZEN.json');R.check(r['outputs']);continue
   if done>=limit:break
   assert not(U/'starts'/(cid+'.json')).exists(),'Prior partial attempt must be preserved and audited';R.write(U/'starts'/(cid+'.json'),dict(at=R.now()))
   inp=row['input'];R.check(inp['input_pins']);native=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];box=inp['info']['original_roi'];w,h=inp['info']['canvas'];n=len(native['frame_id']);video=Path(row['video'])
   if not video.exists():
    src=inp['source'];arc=src['resolved_archive']
    if arc not in zips:zips[arc]=zipfile.ZipFile(arc,'r')
    zi=zips[arc].getinfo(src['member']);assert zi.CRC==src['crc32'] and zi.file_size==src['bytes']
    with zips[arc].open(zi) as a,video.open('xb') as b:shutil.copyfileobj(a,b,1024*1024)
   assert R.sha(video)==inp['source']['sha256'];oldcache={r['i']:r for r in R.read(row['old_records'])['frames']};cap=cv2.VideoCapture(str(video));assert cap.isOpened();pos=0;records=[]
   out=dict(frame_id=native['frame_id'],timestamp_us=native['timestamp_us'],target=np.zeros(n,bool),requested=np.zeros((n,2),bool),control=np.zeros(n,bool),mp_eligible=np.zeros((n,2,15),bool),mp_unresolved=np.zeros((n,2,15),bool),yolo_eligible=np.zeros((n,2,15),bool),yolo_unresolved=np.zeros((n,2,15),bool),fused_eligible=np.zeros((n,2,15),bool),mp_identity=np.zeros((n,2),bool),yolo_identity=np.zeros((n,2),bool),timing=np.zeros((n,4)),cached=np.zeros((n,4),int))
   out['mp_identity_conflict']=np.zeros((n,2),bool);out['yolo_identity_conflict']=np.zeros((n,2),bool)
   x=native['xy_px'][:,33:].reshape(n,2,21,2);valid=roi(x,native['usable'][:,33:].reshape(n,2,21),box);out['original_angles']=R.C.angles(x,valid);out['original_computable']=np.isfinite(out['original_angles'])
   for item in row['targets']:
    i=item['index'];target=int(native['frame_id'][i]);t=time.perf_counter()
    while pos<=target:ok,bgr=cap.read();assert ok;pos+=1
    assert bgr.shape[:2]==(h,w);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);out['target'][i]=True;out['control'][i]=item['control'];sides_to_check=[s for s in range(2) if valid[i,s].any() and (s in item['sides'] or item['control'])];out['requested'][i,sides_to_check]=True
    cache=oldcache.get(i,{});calls=[]
    def get(kind,side=None,b=None):
     key=kind+('_full' if b is None else '_crop');cached=None
     if key in cache:
      if b is None:cached=cache[key]
      else:
       cc=[d for d in cache[key] if d.get('box')==b];oldboxes=cache.get('boxes' if kind=='mp' else 'yolo_boxes',{})
       if b in list(oldboxes.values()):cached=cc
     col=(0 if kind=='mp' else 2)+(b is not None)
     if cached is not None:
      ds=[dict(d,xy=np.asarray(d['xy'],float)) for d in cached];ms=0;out['cached'][i,col]+=1
     else:
      ds,ms=detect(lm,rgb if b is None else rgb[b[1]:b[3],b[0]:b[2]],mp,b) if kind=='mp' else exp.ydetect(ym,bgr if b is None else bgr[b[1]:b[3],b[0]:b[2]],b)
      out['timing'][i,col]+=ms
     if side is not None:
      for d in ds:d['requested_side']=side
     calls.append(dict(detector=kind,box=b,cached=cached is not None,ms=ms));return ds
    if not sides_to_check:records.append(dict(i=i,frame_id=target,requested=[],control=item['control'],calls=[],mp_full=[],mp_crop=[],mp_checks=[],yolo_full=[],yolo_crop=[],yolo_checks=[]));continue
    full=get('mp');crops=[]
    for s in sides_to_check:
     b=R.C.crop_box(x[i,s],valid[i,s],w,h);crops.extend(get('mp',s,b))
    nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);old_sides=assignment[i,:nn].tolist()
    for s in range(2):
     if native['usable'][i,33+21*s:54+21*s].any() and s not in old_sides:old.append(x[i,s]);old_sides.append(s)
    checks=[];ys=[]
    for s in sides_to_check:
     e=evidence(full,crops,x[i,s],valid[i,s],old,old_sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box,s);checks.append(e);out['mp_identity'][i,s]=e['identity_confirmed'];out['mp_unresolved'][i,s]=e['unresolved'];out['mp_eligible'][i,s]=out['original_computable'][i,s]&e['available']&~e['unresolved']&e['identity_confirmed']
     out['mp_identity_conflict'][i,s]=e['identity_conflict']
     if (out['original_computable'][i,s]&~out['mp_eligible'][i,s]).any():ys.append(s)
    yf=[];yc=[];yd=[]
    if ys:
     yf=get('yolo')
     for s in ys:
      b=R.C.crop_box(x[i,s],valid[i,s],w,h);yc.extend(get('yolo',s,b))
     for s in ys:
      e=evidence(yf,yc,x[i,s],valid[i,s],old,old_sides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box,s);yd.append(e);out['yolo_identity'][i,s]=e['identity_confirmed'];out['yolo_unresolved'][i,s]=e['unresolved'];out['yolo_eligible'][i,s]=out['original_computable'][i,s]&e['available']&~e['unresolved']&e['identity_confirmed']
      out['yolo_identity_conflict'][i,s]=e['identity_conflict']
    out['fused_eligible'][i]=fuse(out['mp_eligible'][i],out['mp_unresolved'][i],out['mp_identity_conflict'][i],out['yolo_eligible'][i],out['yolo_unresolved'][i],out['yolo_identity_conflict'][i])
    rr=dict(i=i,frame_id=target,requested=sides_to_check,control=item['control'],reason=item['reasons'],calls=calls,mp_full=full,mp_crop=crops,mp_checks=checks,yolo_full=yf,yolo_crop=yc,yolo_checks=yd,elapsed_ms=(time.perf_counter()-t)*1000);records.append(clean(rr))
    if item['control']:assert cv2.imwrite(str(U/'images'/f'{cid}_{i:04d}.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,92])
   cap.release();assert out['target'].sum()==len(row['targets']);op=U/'outputs'/(cid+'.npz');np.savez_compressed(op,**out);R.check(inp['input_pins']);R.write(rp,dict(at=R.now(),clip_id=cid,context_sha256=R.sha(U/'IMAGE_FROZEN.json'),source_video=str(video),source_sha256=inp['source']['sha256'],frames=records,outputs={str(op):R.sha(op)},native_input_unchanged=True));done+=1
   ev=dict(at=R.now(),clip_id=cid,newly_done=done,total_done=len(list((U/'records').glob('*.json'))),elapsed_seconds=time.perf_counter()-start,targets=len(row['targets']),mp_eligible=int(out['mp_eligible'].sum()),fused_eligible=int(out['fused_eligible'].sum()))
   with (U/'EVENTS.jsonl').open('a',encoding='utf-8') as ef:ef.write(R.json.dumps(ev)+'\n')
   print('IMAGE_CLIP',ev['total_done'],100,cid,'elapsed',round(ev['elapsed_seconds'],1),flush=True)
 for z in zips.values():z.close()
 R.check(f['pins']);R.check(f['protected']);total=len(list((U/'records').glob('*.json')));R.write(U/('IMAGE_DONE.json' if total==100 else 'IMAGE_BATCH_'+str(total)+'.json'),dict(at=R.now(),clips=total,seconds=time.perf_counter()-start,status='PENDING_INDEPENDENT_VALIDATION'));(U/'RUNNING.json').unlink();print('IMAGE_BATCH_END',total,flush=True)
if __name__=='__main__':
 try:freeze() if R.sys.argv[1]=='freeze' else run(int(R.sys.argv[2]) if len(R.sys.argv)>2 else 100)
 except BaseException:
  p=U/'IMAGE_FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf-8')
  raise
