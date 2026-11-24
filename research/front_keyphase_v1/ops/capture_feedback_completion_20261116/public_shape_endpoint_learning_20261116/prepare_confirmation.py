from form_candidate import *
import zipfile,av
sys.path.insert(0,str(O/'word100_rtmw_stream_reuse_20261108'))
from stream_runtime import Engine
def main(person=16,output_name='confirmation',candidate='form_v2.py'):
 torch.set_num_threads(2);cv2.setNumThreads(2);out=D/output_name;out.mkdir(exist_ok=True)
 inv=next(r for r in read(O/'demo_scope_portability_review_20261113/INVENTORY.json')['targets']['WORD1290']if r['record_id']==person)
 write(out/'FROZEN.json',dict(views=[f'{person}_L',f'{person}_R'],purpose='Unselected camera views for this candidate; person front view is known DEV, not unseen-person performance.',candidate=hashlib.sha256((D/candidate).read_bytes()).hexdigest(),training=False))
 engine=Engine();rows=[]
 with zipfile.ZipFile(inv['archive'])as z:
  for view in['L','R']:
   member=inv['member'].replace('_F.mp4',f'_{view}.mp4');cid=Path(member).stem;video=out/(cid+'.mp4');video.write_bytes(z.read(member));xy=[];ss=[];ts=[];tiles=[]
   with av.open(str(video))as c:
    st=c.streams.video[0];base=None
    for i,fr in enumerate(c.decode(st)):
     if base is None:base=fr.pts
     im=fr.to_ndarray(format='bgr24');im=cv2.resize(im,(640,360));q=engine.infer(im);xy.append(q['xy']);ss.append(q['score']);ts.append(round(float((fr.pts-base)*st.time_base)*1e6))
     if i in[43,48,53,58,63,68,73,78]:
      tile=im[:,140:500].copy();cv2.putText(tile,str(i),(4,22),0,.7,(255,255,255),2);tiles.append(tile)
   path=out/(cid+'.npz');np.savez_compressed(path,xy_px=xy,score=ss,timestamp_us=np.array(ts,np.int64),frame_id=np.arange(len(ts)),source_width=640,source_height=360)
   cv2.imwrite(str(out/(cid+'.jpg')),np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
   rows.append(dict(id=cid,raw=str(path),video=str(video),source_zip=inv['archive'],member=member,source_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),candidate_selection=False,training=False))
 write(out/'INPUTS.json',dict(rows=rows));print('PREPARED',len(rows),flush=True)
if __name__=='__main__':main()
