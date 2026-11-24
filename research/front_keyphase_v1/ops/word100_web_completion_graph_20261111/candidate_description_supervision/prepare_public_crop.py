"""Public FIT sources only, original frame times. No camera or learner capture."""
from pathlib import Path
import sys,json,hashlib,time,traceback
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent;R=W/'framing_boundary_candidate';G=O/'word100_observation_guard_20261108'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
 R.mkdir(exist_ok=True);assert not (R/'CROP_INPUTS_COMPLETE.json').exists()
 src=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];old={r['clip_id']:r for r in read(G/'INPUT_PLAN.json')['rows']};rows=[]
 assert {s['person'] for s in src}=={2,4,5,6,8,9}
 for s in src:
  cid=s['clip_id'];rec=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');assert rec['role']=='fit' and sha(s['source'])==s['source_sha256'] and sha(rec['raw_path'])==rec['raw_sha256']
  z=np.load(rec['raw_path']);b=np.nanmedian(z['xy_px'][:12],0);scale=float(np.linalg.norm(b[5]-b[6]));top=max(0,int(b[0,1]-.7*scale));bottom=min(int(z['source_height']),int(max(b[7,1],b[8,1])+.15*scale));height=bottom-top;width=int(height*4/3);cx=(b[5,0]+b[6,0])/2;left=max(0,min(int(z['source_width'])-width,int(cx-width/2)));crop=[left,top,left+width,bottom]
  assert height>0 and width>0
  cached=None
  if cid in old:
   assert old[cid]['crop']==crop and old[cid]['source_sha256']==s['source_sha256']
   p=Path(old[cid]['cache']) if old[cid]['cache'] else G/'crops'/f'{cid}.npz'
   if p.exists():cached=str(p)
  rows.append(dict(clip_id=cid,person=s['person'],split='probe' if s['person'] in [8,9] else 'train',source=s['source'],source_sha256=s['source_sha256'],raw=rec['raw_path'],raw_sha256=rec['raw_sha256'],crop=crop,reused=cached))
 cfg=dict(reason='Audit found normal real-crop inputs unlike original full-frame inputs. Add the same existing framing policy on public FIT; do not change detector or select crop from labels.',policy='First12 median nose/shoulder/elbow, identical 20261108 prepare.py; resize640x480; all frames',rows=rows,source_clips=24,train_people=[2,4,5,6],probe_people=[8,9],not_independent_final=True,no_camera=True,no_private=True,no_final8=True)
 assert not (R/'CROP_PLAN.json').exists();write(R/'CROP_PLAN.json',cfg)
 import cv2,av
 sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'));from stream_runtime import Engine
 engine=None;out=[]
 for r in rows:
  t=time.perf_counter();raw=np.load(r['raw']);dest=Path(r['reused']) if r['reused'] else R/'crops'/f"{r['clip_id']}.npz"
  if not r['reused']:
   assert not dest.exists();dest.parent.mkdir(exist_ok=True)
   if engine is None:engine=Engine()
   xy=[];sc=[];ts=[];first=None;l,top,right,bottom=r['crop']
   with av.open(r['source']) as c:
    for frame in c.decode(video=0):
     tm=frame.pts*frame.time_base
     if first is None:first=tm
     im=frame.to_ndarray(format='bgr24');im=cv2.resize(im[top:bottom,l:right],(640,480));v=engine.infer(im);xy.append(v['xy']);sc.append(v['score']);ts.append(round(float(tm-first)*1e6))
   assert np.array_equal(ts,raw['timestamp_us']);np.savez_compressed(dest,xy_px=np.stack(xy),score=np.stack(sc),timestamp_us=np.array(ts),frame_id=raw['frame_id'],source_width=640,source_height=480)
  z=np.load(dest);assert np.array_equal(z['timestamp_us'],raw['timestamp_us']) and np.array_equal(z['frame_id'],raw['frame_id']);assert z['xy_px'].shape==(len(raw['frame_id']),133,2)
  out.append(dict(r,path=str(dest),sha256=sha(dest),frames=len(raw['frame_id']),seconds=time.perf_counter()-t));write(R/'CROP_PROGRESS.json',dict(rows=out));print('public crop',r['clip_id'],'reuse' if r['reused'] else 'extracted',len(raw['frame_id']),flush=True)
 write(R/'CROP_INPUTS_COMPLETE.json',dict(rows=out,devices=engine.devices if engine else None,complete=True,all_frame_ids_and_times_preserved=True))
if __name__=='__main__':
 try:main()
 except BaseException:R.mkdir(exist_ok=True);(R/f'CROP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
