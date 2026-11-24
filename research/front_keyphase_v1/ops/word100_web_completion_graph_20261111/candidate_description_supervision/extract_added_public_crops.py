"""Same frozen framing/RTMW policy on newly annotated existing public FIT clips."""
from pathlib import Path
import sys,time,traceback,json,hashlib
import numpy as np
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent;D=W/'additional_source_review'
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n','utf-8')
def main():
 assert not(D/'CROP_INPUTS_COMPLETE.json').exists();sources=read(D/'SOURCE_EPISODES_ADDED.json')['rows'];old=read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'];assert not({s['source_sha256']for s in sources}&{s['source_sha256']for s in old});rows=[]
 for s in sources:
  cid=s['clip_id'];r=read(O/'word100_rtmw_joint_learning_20261108/records'/f'{cid}.json');assert r['role']=='fit' and s['person']in[3,7];assert sha(s['source'])==s['source_sha256'] and sha(r['raw_path'])==r['raw_sha256'];z=np.load(r['raw_path']);b=np.nanmedian(z['xy_px'][:12],0);scale=float(np.linalg.norm(b[5]-b[6]));top=max(0,int(b[0,1]-.7*scale));bottom=min(int(z['source_height']),int(max(b[7,1],b[8,1])+.15*scale));height=bottom-top;width=int(height*4/3);cx=(b[5,0]+b[6,0])/2;left=max(0,min(int(z['source_width'])-width,int(cx-width/2)));assert height>0
  rows.append(dict(clip_id=cid,person=s['person'],split='train',source=s['source'],source_sha256=s['source_sha256'],raw=r['raw_path'],raw_sha256=r['raw_sha256'],crop=[left,top,left+width,bottom]))
 write(D/'CROP_PLAN.json',dict(rows=rows,policy='Identical first12 median nose/shoulder/elbow crop;640x480; no target-informed ROI',private=False,final8=False,all_fit=True,code_sha256=sha(__file__)))
 import av,cv2
 sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'));from stream_runtime import Engine
 engine=Engine();out=[]
 for r in rows:
  t=time.perf_counter();raw=np.load(r['raw']);dest=D/'crops'/f"{r['clip_id']}.npz";assert not dest.exists();dest.parent.mkdir(exist_ok=True);xy=[];sc=[];ts=[];first=None;l,top,right,bottom=r['crop']
  with av.open(r['source']) as c:
   for frame in c.decode(video=0):
    tm=frame.pts*frame.time_base
    if first is None:first=tm
    im=cv2.resize(frame.to_ndarray(format='bgr24')[top:bottom,l:right],(640,480));v=engine.infer(im);xy.append(v['xy']);sc.append(v['score']);ts.append(round(float(tm-first)*1e6))
  assert np.array_equal(ts,raw['timestamp_us']);np.savez_compressed(dest,xy_px=np.stack(xy),score=np.stack(sc),timestamp_us=np.array(ts),frame_id=raw['frame_id'],source_width=640,source_height=480);out.append(dict(r,path=str(dest),sha256=sha(dest),frames=len(ts),seconds=time.perf_counter()-t));write(D/'CROP_PROGRESS.json',dict(rows=out));print('Added FIT crop',r['clip_id'],len(ts),flush=True)
 write(D/'CROP_INPUTS_COMPLETE.json',dict(rows=out,devices=engine.devices,all_frame_ids_and_times_preserved=True,complete=True))
if __name__=='__main__':
 try:main()
 except BaseException:(D/f'CROP_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc(),'utf-8');raise
