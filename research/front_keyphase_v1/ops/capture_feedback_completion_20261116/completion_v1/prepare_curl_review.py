from train_shape import *
import cv2,zipfile,io,av
J=O/'word100_rtmw_joint_learning_20261108'
out=V/'curl_review';out.mkdir(exist_ok=True)
rows=read(J/'SOURCES.json');records=[]
for person in [1,4,7]:
 tiles=[]
 for word in ['WORD0678','WORD0071']:
  cid=f'NIA_SL_{word}_REAL{person:02}_F';r=next(x for x in rows if x['clip_id']==cid)
  assert r['role']=='fit'
  with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
  assert hashlib.sha256(data).hexdigest()==r['source_sha256']
  rec=read(J/'records'/f'{cid}.json');raw=dict(np.load(rec['raw_path']))
  frames=[35,45,55,65,75] if word=='WORD0678' else [55,65,75,85,95]
  with av.open(io.BytesIO(data)) as cap:
   for fid,frame in enumerate(cap.decode(video=0)):
    if fid not in frames:continue
    im=frame.to_ndarray(format='bgr24');ids=range(112,133) if word=='WORD0678' else range(91,112)
    pts=raw['xy_px'][fid,ids];pts=pts[np.isfinite(pts).all(1)]
    lo=np.maximum(np.floor(pts.min(0)-35),0).astype(int);hi=np.minimum(np.ceil(pts.max(0)+35),[im.shape[1],im.shape[0]]).astype(int)
    crop=im[lo[1]:hi[1],lo[0]:hi[0]];scale=min(300/crop.shape[1],250/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
    tile=np.full((295,320,3),255,np.uint8);tile[35:35+crop.shape[0],10:10+crop.shape[1]]=crop
    cv2.putText(tile,f'{word} P{person} F{fid}',(4,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(15,25,35),1);tiles.append(tile)
    records.append(dict(clip_id=cid,person=person,word=word,role='fit',frame=fid,hand='R' if word=='WORD0678'else'L',
       raw=rec['raw_path'],channels=rec['channel_path'],source_sha256=r['source_sha256'],archive=r['archive'],member=r['member']))
 cv2.imwrite(str(out/f'p{person:02}.jpg'),np.vstack([np.hstack(tiles[i:i+5])for i in[0,5]]))
write(out/'INPUTS.json',dict(rows=records,truth_not_inferred=True))
print('REVIEW',len(records))
