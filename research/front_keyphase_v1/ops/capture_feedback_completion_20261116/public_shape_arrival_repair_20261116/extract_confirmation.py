"""Actual alternative public cameras, fixed selection before evaluation."""
from prepare_public_camera import *
import zipfile

def main():
 torch.set_num_threads(2);cv2.setNumThreads(2);d=HERE/'confirmation_views';d.mkdir(exist_ok=True)
 source={r['clip_id']:r for r in read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json')}
 # Source manifests of newer gratitude examples contain the exact ZIP member.
 chosen=[dict(clip_id=f'NIA_SL_WORD1290_REAL{p:02d}_F',person=p,role='nonfit_new_camera_confirmation')for p in[14,15]]
 write(d/'CONTRACT.json',dict(persons=[14,15],views=['L','R'],source='Synchronized public cameras. No user footage.',labels='Require RGB review, not copied from front camera'))
 engine=Engine();net=SparseVisible();out=[]
 for r in chosen:
  front=source.get(r['clip_id'])
  if front is None:
   archive=Path('W:/WordsInSilence/data/AIHub_103/videos/수어 영상')/('1.Training/[원천]02_real_word_video.zip'if r['person']<10 else'2.Validation/[원천]01_real_word_video.zip')
  else:archive=Path(front['archive'])
  archive=Path(next(x['archive']for x in read(HERE/'confirmation_views/SOURCES.json')['rows']if x['clip']==r['clip_id'][:-1]+'L'))
  with zipfile.ZipFile(archive)as z:
   for view in ['L','R']:
    clip=r['clip_id'][:-1]+view;matches=[x for x in z.namelist()if x.endswith('/'+clip+'.mp4')]
    if len(matches)!=1:raise ValueError((clip,archive,len(matches)))
    video=d/(clip+'.mp4');rawpath=d/(clip+'.npz')
    if not video.exists():video.write_bytes(z.read(matches[0]))
    if not rawpath.exists():
     xy=[];scores=[];ts=[];cost=[];boxes=[]
     with av.open(str(video))as c:
      st=c.streams.video[0];base=None
      for fr in c.decode(st):
       if base is None:base=fr.pts
       im=fr.to_ndarray(format='bgr24');h,w=im.shape[:2];im=cv2.resize(im,(640,360));q=engine.infer(im)
       xy.append(q['xy']);scores.append(q['score']);boxes.append(q['bbox']);cost.append(q['compute_ms']);ts.append(round(float((fr.pts-base)*st.time_base)*1e6))
     np.savez_compressed(rawpath,xy_px=xy,score=scores,timestamp_us=np.array(ts,np.int64),frame_id=np.arange(len(ts)),source_width=640,source_height=360,bbox=boxes,compute_ms=cost)
    raw=dict(np.load(rawpath));q=net.analyze(raw);write(d/(clip+'.json'),q)
    e=dict(id=clip,person=r['person'],role=r['role'],raw=str(rawpath),video=str(video),source_zip=str(archive),source_member=matches[0],source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),criteria={c['id']:c['status']for c in q['criteria']},closed={side:np.flatnonzero((np.array(q['shape_states'][side])==1)&(np.array(q['relation_states'])>0)).tolist()for side in ['R','L']})
    out.append(e);write(d/'INPUTS.json',dict(rows=out));print(clip,e['closed'],flush=True)
if __name__=='__main__':main()
