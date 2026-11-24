import prepare as R
from pathlib import Path
from collections import defaultdict
import numpy as np,os,time,zipfile,shutil,sys,traceback
U=R.U;V=R.V
H=R.mod('recheck_source_helpers',V/'images.py')
G=R.mod('direct_image_geometry',V/'pixel_guard.py')
def freeze():
 assert R.read(V/'IMAGE_VERIFIED.json')['clips']==100 and R.read(V/'PIXEL_GUARD_VERIFIED.json')['clips']==100
 f=R.read(V/'FROZEN.json');im=R.read(V/'IMAGE_FROZEN.json');R.check(f['pins']);R.check(im['protected']);by=defaultdict(list)
 for r in f['rows']:by[r['word_code']].append(r)
 queries=im['rows'];refs=[];pairs=[];excluded=[];pins={}
 for q in queries:
  choices=sorted(by[q['word_code']],key=lambda r:R.hashlib.sha256(('multiref-visible-v1:'+r['clip_id']).encode()).hexdigest());chosen=[];ids={q['signer_id']};shas={q['input']['source']['sha256']}
  for r in choices:
   if r['signer_id'] in ids:continue
   if r['input']['source']['sha256'] in shas:excluded.append(dict(query=q['clip_id'],candidate=r['clip_id'],reason='identical_source_sha'));continue
   chosen.append(r);ids.add(r['signer_id']);shas.add(r['input']['source']['sha256'])
   if len(chosen)==2:break
  assert len(chosen)==2
  for r in chosen:
   cid=r['clip_id'];rp=R.I/'records'/(cid+'__masked.json');rr=R.read(rp);R.check(rr['outputs']);vp=Path(rr['source_video']) if rr['source_video'] else U/'sources'/(cid+'.mp4')
   if not vp.exists():vp=U/'sources'/(cid+'.mp4')
   if vp.exists():assert R.sha(vp)==r['input']['source']['sha256']
   refs.append(dict(**r,video=str(vp),old_records=str(rp)));pairs.append(dict(query=q['clip_id'],reference=cid,word_code=q['word_code'],label=q['label']));pins[str(rp)]=R.sha(rp);pins.update(r['input']['input_pins']);pins[r['diagnostics']]=R.sha(r['diagnostics'])
  for p in [V/'records'/(q['clip_id']+'.json'),V/'outputs'/(q['clip_id']+'.npz'),V/'pixel_guard'/(q['clip_id']+'.npz')]:pins[str(p)]=R.sha(p)
 assert len(refs)==200 and len({r['clip_id'] for r in refs})==200 and len(pairs)==200
 paths=[Path(__file__),U/'prepare.py',V/'candidate.py',V/'images.py',V/'pixel_guard.py',V/'verify_images.py',V/'IMAGE_FROZEN.json',V/'IMAGE_VERIFIED.json',V/'PIXEL_GUARD_VERIFIED.json',V/'FROZEN.json',V/'PREPARED_VERIFIED.json',Path(H.rule.__file__),Path(H.Q.__file__),R.ROOT/'Tools/models/hand_landmarker.task']
 pins.update({str(p):R.sha(p) for p in paths})
 R.write(U/'FROZEN.json',dict(at=R.now(),queries=queries,rows=refs,pairs=pairs,pins=pins,protected=im['protected'],mp_options=im['mp_options'],source_thresholds=f['thresholds'],source_hash_exclusions=excluded,
 authorization='Active autonomous goal, public TRAIN only. No user/mentor/extra199/200. ExistingC/T3F/E3/operating unchanged.',
 scope='100previous representative query clips plus2different-recorded-ID refs each;200new refs,300unique publicTRAIN clips,100words. Select by predetermined hash,not model results. Reuse all100query image outputs. No recognition/training here.',
 observation='Same frozen visible-hand schedule from TRAIN899 and sameMP fullimage+handcrop settings and originalROI. Entire-hand absence remains separate. References MP only: queryYOLO retained as separate audit,but is not a positive precision source or veto for MP. No freshYOLO requested,previous100showed no extra core-angle support after current-image check.',
 comparison='Keep original fixedDTW raw path/point differences and referencecorex2x23 auditgrid; attach raw availability and per-angleMP current-image corroboration separately. Keep eachref,never fabricate an averaged pose. Unknown remains null,not0; observation insufficiency not semantic error. No inference of allowed deviation/required hands from two refs.',
 validation='Verify200new source outputs/coordinates/selection/cached calls; verify300sidecars and200pairs reference denominator,raw point preservation,local missingness,normal crossing unchanged. Manual/AI image review not21pointtruth. Control missingangle injection must never reduce denominator or create success.',
 training=0,model_inference=0,production_changed=False))
 for n in ['sources','records','outputs','starts']:(U/n).mkdir()
 print('FROZEN_MULTIREf',len(refs),sum(len(r['targets']) for r in refs),flush=True)
def mp_guard(e,full,crops,a,valid,box,original):
 av=np.zeros(15,bool);bad=av.copy()
 if e['identity_confirmed'] and e['pair_available'] and e['full_index'] is not None and e['crop_index'] is not None:
  b=np.asarray(full[e['full_index']]['xy'],float);c=np.asarray(crops[e['crop_index']]['xy'],float);bb=G.mask(b,box);cc=G.mask(c,box);checks=[G.pair(a,valid,b,bb),G.pair(a,valid,c,cc),G.pair(b,bb,c,cc)];av=np.logical_and.reduce([x[0] for x in checks]);bad=np.logical_or.reduce([x[1] for x in checks])
 return av,bad,original&np.asarray(e['available'])&~np.asarray(e['unresolved'])&e['identity_confirmed']&~np.asarray(e['identity_conflict'])&av&~bad
def run(limit=200):
 f=R.read(U/'FROZEN.json');R.check(f['pins']);R.check(f['protected']);assert not(U/'RUNNING.json').exists();R.write(U/'RUNNING.json',dict(at=R.now(),pid=os.getpid(),stage='MP_reference_images'))
 import cv2,mediapipe as mp
 from mediapipe.tasks.python import vision
 from mediapipe.tasks.python.core.base_options import BaseOptions
 cv2.setNumThreads(1);opts=vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(R.ROOT/'Tools/models/hand_landmarker.task')),running_mode=vision.RunningMode.IMAGE,**f['mp_options']);archives={};done=0;beg=time.perf_counter()
 with vision.HandLandmarker.create_from_options(opts) as lm:
  H.detect(lm,np.zeros((1080,1920,3),np.uint8),mp)
  for row in f['rows']:
   cid=row['clip_id'];rp=U/'records'/(cid+'.json')
   if rp.exists():rr=R.read(rp);assert rr['context_sha256']==R.sha(U/'FROZEN.json');R.check(rr['outputs']);continue
   if done>=limit:break
   sp=U/'starts'/(cid+'.json');assert not sp.exists(),'Audit prior partial';R.write(sp,dict(at=R.now()))
   inp=row['input'];R.check(inp['input_pins']);n=R.load(inp['info']['source_arrays']['capture']);raw=R.load(inp['info']['source_arrays']['raw']);assignment=R.load(Path(inp['info']['source_arrays']['capture']).parent/'assignment.npz')['sides'];w,h=inp['info']['canvas'];box=inp['info']['original_roi'];N=len(n['frame_id']);x=n['xy_px'][:,33:].reshape(N,2,21,2);valid=H.roi(x,n['usable'][:,33:].reshape(N,2,21),box);video=Path(row['video'])
   if not video.exists():
    src=inp['source'];arc=src['resolved_archive']
    if arc not in archives:archives[arc]=zipfile.ZipFile(arc,'r')
    z=archives[arc].getinfo(src['member']);assert z.CRC==src['crc32'] and z.file_size==src['bytes']
    with archives[arc].open(z) as aa,video.open('xb') as bb:shutil.copyfileobj(aa,bb,1024*1024)
   assert R.sha(video)==inp['source']['sha256'];cache={z['i']:z for z in R.read(row['old_records'])['frames']};cap=cv2.VideoCapture(str(video));assert cap.isOpened();pos=0;frames=[]
   out=dict(frame_id=n['frame_id'],timestamp_us=n['timestamp_us'],target=np.zeros(N,bool),requested=np.zeros((N,2),bool),control=np.zeros(N,bool),original_angles=R.C.angles(x,valid),timing=np.zeros((N,2)),cached=np.zeros((N,2),int))
   out['original_computable']=np.isfinite(out['original_angles'])
   for k in ['mp_eligible','mp_unresolved','pixel_available','pixel_unresolved','pixel_eligible']:out[k]=np.zeros((N,2,15),bool)
   for k in ['mp_identity','mp_identity_conflict']:out[k]=np.zeros((N,2),bool)
   for item in row['targets']:
    i=item['index'];frame_id=int(n['frame_id'][i])
    while pos<=frame_id:ok,bgr=cap.read();assert ok;pos+=1
    assert bgr.shape[:2]==(h,w);rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);out['target'][i]=True;out['control'][i]=item['control'];sides=[s for s in (0,1) if valid[i,s].any() and (s in item['sides'] or item['control'])];out['requested'][i,sides]=True;calls=[]
    def get(b=None,s=None):
     cc=cache.get(i,{});key='mp_full' if b is None else 'mp_crop';saved=None
     if key in cc:
      if b is None:saved=cc[key]
      elif b in list(cc.get('boxes',{}).values()):saved=[d for d in cc[key] if d.get('box')==b]
     col=int(b is not None)
     if saved is not None:ds=[dict(d,xy=np.asarray(d['xy'],float)) for d in saved];ms=0;out['cached'][i,col]+=1
     else:ds,ms=H.detect(lm,rgb if b is None else rgb[b[1]:b[3],b[0]:b[2]],mp,b);out['timing'][i,col]+=ms
     if s is not None:
      for d in ds:d['requested_side']=s
     calls.append(dict(box=b,cached=saved is not None,ms=ms));return ds
    full=get() if sides else [];crops=[]
    for s in sides:crops.extend(get(R.C.crop_box(x[i,s],valid[i,s],w,h),s))
    nn=int(raw['raw_count'][i]);old=list(raw['hand_xy_px'][i,:nn]);osides=assignment[i,:nn].tolist()
    for s in (0,1):
     if n['usable'][i,33+21*s:54+21*s].any() and s not in osides:old.append(x[i,s]);osides.append(s)
    checks=[]
    for s in sides:
     e=H.evidence(full,crops,x[i,s],valid[i,s],old,osides,raw['pose_xy_px'][i],raw['pose_usable'][i],w,h,box,s);checks.append(e);out['mp_identity'][i,s]=e['identity_confirmed'];out['mp_identity_conflict'][i,s]=e['identity_conflict'];out['mp_unresolved'][i,s]=e['unresolved'];out['mp_eligible'][i,s]=out['original_computable'][i,s]&e['available']&~e['unresolved']&e['identity_confirmed'];av,un,eligible=mp_guard(e,full,crops,x[i,s],valid[i,s],box,out['original_computable'][i,s]);out['pixel_available'][i,s]=av;out['pixel_unresolved'][i,s]=un;out['pixel_eligible'][i,s]=eligible
    frames.append(H.clean(dict(i=i,frame_id=frame_id,requested=sides,control=item['control'],reason=item['reasons'],calls=calls,mp_full=full,mp_crop=crops,mp_checks=checks)))
   cap.release();op=U/'outputs'/(cid+'.npz');np.savez_compressed(op,**out);R.check(inp['input_pins']);R.write(rp,dict(at=R.now(),clip_id=cid,context_sha256=R.sha(U/'FROZEN.json'),source_video=str(video),source_sha256=inp['source']['sha256'],frames=frames,outputs={str(op):R.sha(op)},no_coordinate_changes=True));done+=1
   event=dict(at=R.now(),clip_id=cid,done=len(list((U/'records').glob('*.json'))),elapsed=time.perf_counter()-beg)
   with (U/'EVENTS.jsonl').open('a',encoding='utf8') as ff:ff.write(json.dumps(event)+'\n')
   print('REF_IMAGES',event['done'],200,'seconds',round(event['elapsed'],1),flush=True)
 for z in archives.values():z.close()
 total=len(list((U/'records').glob('*.json')));R.check(f['pins']);R.check(f['protected']);R.write(U/('REFERENCES_DONE.json' if total==200 else f'REFERENCE_BATCH_{total}.json'),dict(at=R.now(),clips=total,seconds=time.perf_counter()-beg));(U/'RUNNING.json').unlink();print('REFERENCE_BATCH_END',total,flush=True)
if __name__=='__main__':
 import json
 try:freeze() if sys.argv[1]=='freeze' else run(int(sys.argv[2]) if len(sys.argv)>2 else 200)
 except BaseException:
  p=U/'FIRST_ERROR.txt'
  if not p.exists():p.write_text(traceback.format_exc(),encoding='utf8')
  raise
