"""Only previously unrequested, raw-computable public core hand frames."""
import prepare as R
import numpy as np,os,sys,time,json,traceback
from pathlib import Path
N=R.O/'word100_source_separate_multiref_20261026'
H=R.mod('frozen_reference_operations',N/'references.py')
def cached(z,i,b):
 for rr in z:
  r=rr.get(i,{})
  if b is None and 'mp_full' in r:return r['mp_full']
  boxes=list(r.get('boxes',{}).values())+[c['box'] for c in r.get('calls',[]) if c.get('detector','mp')=='mp' and c['box'] is not None]
  if b is not None and b in boxes and 'mp_crop' in r:return [d for d in r['mp_crop'] if d.get('box')==b]
 return None
def run(limit=599):
 assert R.read(R.U/'BASE_VERIFIED.json')['inputs']==899
 f=R.read(R.U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(R.U/'RUNNING.json').exists();R.write(R.U/'RUNNING.json',dict(at=R.now(),pid=os.getpid(),limit=limit));beg=time.perf_counter();done=0
 import cv2,mediapipe as mp
 from mediapipe.tasks.python import vision
 from mediapipe.tasks.python.core.base_options import BaseOptions
 cv2.setNumThreads(1);opt=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R.ROOT/'Tools/models/hand_landmarker.task')),running_mode=vision.RunningMode.IMAGE,**f['mp_options'])
 with vision.HandLandmarker.create_from_options(opt) as lm:
  H.H.detect(lm,np.zeros((1080,1920,3),np.uint8),mp)
  for row in f['rows']:
   cid=row['clip_id'];rp=R.U/'records'/(cid+'.json')
   if rp.exists():rr=R.read(rp);assert rr['context_sha256']==R.sha(R.U/'FROZEN.json');R.check(rr['outputs']);continue
   if done>=limit:break
   sp=R.U/'starts'/(cid+'.json');assert not sp.exists(),'Preserve and inspect partial before recovery';R.write(sp,dict(at=R.now()));src=row['source'];inp=src['input'];n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];base=R.load(row['baseline_input']);w,h=inp['info']['canvas'];box=inp['info']['original_roi'];x=n['xy_px'][:,33:].reshape(-1,2,21,2);v=H.H.roi(x,n['usable'][:,33:].reshape(-1,2,21),box);rr=R.read(row['recent_record']);video=rr['source_video'];assert R.sha(video)==inp['source']['sha256'];caches=[{z['i']:z for z in rr['frames']},{z['i']:z for z in R.read(src['old_records'])['frames']}];cap=cv2.VideoCapture(video);assert cap.isOpened();pos=0;Nframes=len(x);out={k:np.zeros((Nframes,2,15),bool) for k in ['mp_eligible','mp_unresolved','pixel_available','pixel_unresolved','pixel_eligible']}
   for k in ['requested','mp_identity','mp_identity_conflict']:out[k]=np.zeros((Nframes,2),bool)
   out.update(frame_id=n['frame_id'],timestamp_us=n['timestamp_us'],timing=np.zeros((Nframes,2)),cached=np.zeros((Nframes,2),int),target=np.zeros(Nframes,bool));frames=[]
   for t in row['targets']:
    i=t['index'];fid=t['frame_id'];sides=t['sides'];out['target'][i]=True;out['requested'][i,sides]=True
    while pos<=fid:ok,bgr=cap.read();assert ok;pos+=1
    assert bgr.shape[:2]==(h,w);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);calls=[]
    def get(b=None,s=None):
     ds=cached(caches,i,b);j=int(b is not None);reused=ds is not None
     if reused:ds=[dict(d,xy=np.asarray(d['xy'],float)) for d in ds];ms=0;out['cached'][i,j]+=1
     else:ds,ms=H.H.detect(lm,rgb if b is None else rgb[b[1]:b[3],b[0]:b[2]],mp,b);out['timing'][i,j]+=ms
     if s is not None:
      for d in ds:d['requested_side']=s
     calls.append(dict(box=b,cached=reused,ms=ms));return ds
    full=get();crops=[]
    for s in sides:crops.extend(get(t['boxes'][str(s)],s))
    nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);osides=assignment[i,:nn].tolist()
    for s in (0,1):
     if n['usable'][i,33+21*s:54+21*s].any() and s not in osides:old.append(x[i,s]);osides.append(s)
    checks=[]
    for s in sides:
     e=H.H.evidence(full,crops,x[i,s],v[i,s],old,osides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box,s);pa,pu,px=H.mp_guard(e,full,crops,x[i,s],v[i,s],box,base['available'][i,s,:15]);out['mp_identity'][i,s]=e['identity_confirmed'];out['mp_identity_conflict'][i,s]=e['identity_conflict'];out['mp_unresolved'][i,s]=e['unresolved'];out['mp_eligible'][i,s]=base['available'][i,s,:15]&e['available']&~e['unresolved']&e['identity_confirmed'];out['pixel_available'][i,s]=pa;out['pixel_unresolved'][i,s]=pu;out['pixel_eligible'][i,s]=px;checks.append(e)
    frames.append(H.H.clean(dict(i=i,frame_id=fid,requested=sides,calls=calls,mp_full=full,mp_crop=crops,mp_checks=checks)))
   cap.release();op=R.U/'outputs'/(cid+'.npz');np.savez_compressed(op,**out);R.write(rp,dict(at=R.now(),clip_id=cid,source_video=video,source_sha256=inp['source']['sha256'],context_sha256=R.sha(R.U/'FROZEN.json'),frames=frames,outputs={str(op):R.sha(op)},coordinate_changes=False));done+=1;count=len(list((R.U/'records').glob('*.json')))
   with (R.U/'EVENTS.jsonl').open('a',encoding='utf8') as g:g.write(json.dumps(dict(at=R.now(),clip_id=cid,done=count,seconds=time.perf_counter()-beg))+'\n')
   print('CORE_IMAGES',count,len(f['rows']),'seconds',round(time.perf_counter()-beg,1),flush=True)
 count=len(list((R.U/'records').glob('*.json')));R.check(f['pins']);R.check(f['protected']);R.write(R.U/('DONE.json' if count==len(f['rows']) else f'BATCH_{count}.json'),dict(at=R.now(),clips=count,seconds=time.perf_counter()-beg));(R.U/'RUNNING.json').unlink();print('CORE_BATCH_END',count,flush=True)

if __name__=='__main__':
 try:run(int(sys.argv[1]) if len(sys.argv)>1 else 599)
 except BaseException:
  p=R.U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
