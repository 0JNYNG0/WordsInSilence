"""Public source at laptop camera resolution; no physical camera or personal input."""
from audit_public import *
import cv2,av
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
def main():
 torch.set_num_threads(2);cv2.setNumThreads(2);dest=HERE/'public_camera';dest.mkdir(exist_ok=True)
 rows=read(HERE/'PUBLIC_INPUTS.json')['rows'];picked=[r for r in rows if r['id'].endswith('_F')and r['person']in[1,7,11,13,18]]
 # Public manifests, not user framing/measurements, define the same fixed crop.
 write(dest/'CONTRACT.json',dict(persons=[1,7,11,13,18],fit=[1,7],development=[11,13,18],crop=[.25,.05,.75,.72],output=[640,480],resize='Aspect-preserving with black padding',source='Existing public normal clips; no private data',new_holdout=False))
 engine=Engine();net=SparseVisible();out=[]
 for r in picked:
  video=r['video']
  if not video:
   source=next(x for x in read(B/'INPUTS.json')['rows']if x['clip_id']==r['id']);video=source['video']
  dst=dest/(r['id']+'.npz');imgs=[]
  if not dst.exists():
   xy=[];score=[];ts=[];cost=[];bbox=[]
   with av.open(video)as c:
    st=c.streams.video[0];base=None
    for i,fr in enumerate(c.decode(st)):
     if base is None:base=fr.pts
     im=fr.to_ndarray(format='bgr24');h,w=im.shape[:2];im=im[int(h*.05):int(h*.72),int(w*.25):int(w*.75)];h,w=im.shape[:2];ratio=min(640/w,480/h);im=cv2.resize(im,(round(w*ratio),round(h*ratio)));h,w=im.shape[:2];canvas=np.zeros((480,640,3),np.uint8);canvas[(480-h)//2:(480-h)//2+h,(640-w)//2:(640-w)//2+w]=im
     q=engine.infer(canvas);xy.append(q['xy']);score.append(q['score']);bbox.append(q['bbox']);cost.append(q['compute_ms']);ts.append(round(float((fr.pts-base)*st.time_base)*1e6));imgs.append(canvas)
   np.savez_compressed(dst,xy_px=xy,score=score,timestamp_us=np.array(ts,np.int64),frame_id=np.arange(len(ts)),source_width=640,source_height=480,bbox=bbox,compute_ms=cost)
   vw=cv2.VideoWriter(str(dst.with_suffix('.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),30,(640,480))
   for im in imgs:vw.write(im)
   vw.release()
  raw=dict(np.load(dst));q=net.analyze(raw);write(dst.with_suffix('.json'),q)
  entry=dict(r);entry.update(raw=str(dst),video=str(dst.with_suffix('.mp4')),source_video=video,source_sha256=hashlib.sha256(Path(video).read_bytes()).hexdigest(),id=r['id']+'_camera640',frames=len(raw['frame_id']),criteria={c['id']:c['status']for c in q['criteria']},events=len(q['events']))
  out.append(entry);print(json.dumps({k:v for k,v in entry.items()if k not in['video','raw','source_video']},ensure_ascii=False),flush=True)
 write(dest/'INPUTS.json',dict(rows=out,engine=engine.devices))
if __name__=='__main__':main()
