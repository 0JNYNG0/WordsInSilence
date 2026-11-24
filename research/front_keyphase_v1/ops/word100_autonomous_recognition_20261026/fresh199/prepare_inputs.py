"""Evaluation inputs only: reuse exact frozen C detector/boundary/features."""
from common import *
import zipfile,shutil,time,traceback,concurrent.futures,multiprocessing
sys.path.insert(0,str(B));import prepare as body
legacy=body.legacy

def one(row,context_hash):
 cid=row['clip_id'];rp=DATA/'records'/f'{cid}.json'
 if rp.exists():
  r=read(rp);assert r['context_sha256']==context_hash;check(r['pins']);return {'clip_id':cid,'status':'REUSED','seconds':0}
 start=time.perf_counter();d=DATA/'clips'/cid
 assert not d.exists(),'Partial evidence must be reviewed: '+cid
 d.mkdir(parents=True);e=row['video_entries'][0];vp=d/'source.mp4'
 with zipfile.ZipFile(e['archive']) as z:
  zi=z.getinfo(e['member']);assert (zi.CRC,zi.file_size)==(e['crc32'],e['bytes'])
  with z.open(zi) as source,vp.open('xb') as out:shutil.copyfileobj(source,out,1<<20)
 assert vp.stat().st_size==e['bytes'];source_hash=sha(vp)
 ctx=read(U/'EVAL_CONTEXT.json');assert source_hash not in ctx['prior_source_sha256'],'Exact source duplication against train/DEV'
 contract=read(legacy.LAB/'config/contract.json')
 native_rec=legacy.capture(vp,cid,d/'native_capture',contract)
 native=legacy.Observation.load(native_rec['candidate']);w=int(native.sidecar['width']);h=int(native.sidecar['height'])
 cal=body.boundary.calibrate(native,w,h,read(B/'design_data_COEFFICIENT.json'));write(d/'calibration.json',cal,new=True)
 r={'clip_id':cid,'signer_id':row['signer_id'],'label_index':row['label_index'],'canonical_label':row['canonical_label'],'word_code':row['word_code'],'role':'heldout_evaluation_only','context_sha256':context_hash,'source_sha256':source_hash,'source':e,'frames':native.n_frames,'native_capture':native_rec,'calibration':cal,'views':{},'source_file':str(vp),'training':0}
 if cal['status']=='CALIBRATED':
  box=cal['box_source_pixels'];sensor_rec=legacy.capture(vp,cid,d/'sensor_capture',contract,box)
  sensor=legacy.Observation.load(sensor_rec['candidate']);r['sensor_capture']=sensor_rec
  assert native_rec['detector_instance']!=sensor_rec['detector_instance']
  np.testing.assert_array_equal(native.frame_id,sensor.frame_id);np.testing.assert_array_equal(native.timestamp_us,sensor.timestamp_us)
  assert native.sidecar['dropped_non_monotonic_pts']==sensor.sidecar['dropped_non_monotonic_pts']==0
  for v,obs in [('masked',legacy.roi.apply_roi(native,box)),('sensor',legacy.roi.apply_roi(sensor,box))]:
   r['views'][v]=legacy.save_view(obs,d/v,row)
 else:
  r['views']={v:{'observation':None,'channel_file':None,'feature_info':{'reason':'NOT_CALIBRATED:'+cal['reason']}} for v in VIEWS}
 r['seconds']=time.perf_counter()-start;r['pins']={str(p):sha(p) for p in d.rglob('*') if p.is_file()}
 write(rp,r,new=True)
 return {'clip_id':cid,'status':'PREPARED','seconds':r['seconds'],'frames':r['frames'],'boundary':cal['status'],'view_failures':{v:r['views'][v]['feature_info']['reason'] for v in VIEWS}}

def worker_init():
 directory=U/'worker_logs';directory.mkdir(exist_ok=True)
 p=directory/f'{os.getpid()}.log';fd=os.open(str(p),os.O_WRONLY|os.O_CREAT|os.O_EXCL)
 os.dup2(fd,1);os.dup2(fd,2);os.close(fd)
 sys.stdout.reconfigure(encoding='utf8',write_through=True);sys.stderr.reconfigure(encoding='utf8',write_through=True)

def main():
 ctx=read(U/'EVAL_CONTEXT.json');check(ctx['pins']);assert read(U/'PREFLIGHT.json')['status']=='PASS'
 assert not (U/'PREP_RUNNING.json').exists() and not (U/'FIRST_PREP_ERROR.json').exists()
 for p,st in ctx['archive_stat'].items():
  x=Path(p).stat();assert [x.st_size,x.st_mtime_ns]==st
 assert shutil.disk_usage(U).free>10_000_000_000
 ch=sha(U/'EVAL_CONTEXT.json');write(U/'PREP_RUNNING.json',{'pid':os.getpid(),'at':now(),'context_sha256':ch},new=True)
 rows=ctx['rows'];pending=[];done=0;started=time.perf_counter()
 for row in rows:
  rp=DATA/'records'/f"{row['clip_id']}.json"
  if rp.exists():one(row,ch);done+=1
  else:pending.append(row)
 failed=False;futures={};cursor=0
 try:
  with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
   while cursor<len(pending) or futures:
    while not failed and cursor<len(pending) and len(futures)<2:
     row=pending[cursor];cursor+=1;futures[pool.submit(one,row,ch)]=row['clip_id']
    if not futures:break
    finished,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
    for f in finished:
     cid=futures.pop(f)
     try:
      r=f.result();done+=1;event(stage='PREPARING_HELDOUT',completed=done,total=len(rows),seconds=time.perf_counter()-started,last=r)
     except BaseException as exc:
      failed=True
      err={'at':now(),'clip_id':cid,'error':str(exc),'traceback':traceback.format_exc()}
      write(U/'failures'/f'{cid}.json',err,new=True)
      if not (U/'FIRST_PREP_ERROR.json').exists():write(U/'FIRST_PREP_ERROR.json',err,new=True)
      event(stage='FAILED_AWAITING_INFLIGHT',completed=done,total=len(rows),error=err)
   assert not failed,'Preserved first error, no automatic retry'
  assert done==199
  check(ctx['pins']);write(U/'PREP_COMPLETE.json',{'at':now(),'records':done,'seconds':time.perf_counter()-started,'context_sha256':ch,'training':0},new=True)
  event(stage='INPUTS_READY_FOR_INDEPENDENT_VERIFICATION',completed=done,total=199,seconds=time.perf_counter()-started)
 finally:
  lock=U/'PREP_RUNNING.json'
  if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()

if __name__=='__main__':main()
