from train_shape import *
import cv2,zipfile,io,av
J=O/'word100_rtmw_joint_learning_20261108';sources=read(J/'SOURCES.json');tiles=[]
for cid,anchors,side in [('NIA_SL_WORD0071_REAL13_F',[79,82,85],'L'),('NIA_SL_WORD0678_REAL11_F',[42,45,48],'R')]:
 r=next(x for x in sources if x['clip_id']==cid)
 with zipfile.ZipFile(r['archive'])as z:data=z.read(r['member'])
 assert hashlib.sha256(data).hexdigest()==r['source_sha256']
 raw=np.load(read(J/'records'/f'{cid}.json')['raw_path'])
 with av.open(io.BytesIO(data))as cap:
  for fid,fr in enumerate(cap.decode(video=0)):
   if fid not in anchors:continue
   im=fr.to_ndarray(format='bgr24');start=91 if side=='L'else 112;p=raw['xy_px'][fid,start:start+21]
   lo=np.maximum(np.floor(p.min(0)-40),0).astype(int);hi=np.minimum(np.ceil(p.max(0)+40),[im.shape[1],im.shape[0]]).astype(int)
   crop=im[lo[1]:hi[1],lo[0]:hi[0]];scale=min(390/crop.shape[1],320/crop.shape[0]);a=cv2.resize(crop,None,fx=scale,fy=scale);b=a.copy();q=np.rint((p-lo)*scale).astype(int)
   for k in[1,5,9,13,17]:
    for i,j in zip([0]+list(range(k,k+4)),list(range(k,k+4))):cv2.line(b,tuple(q[i]),tuple(q[j]),(0,210,255),1)
   canvas=np.full((370,820,3),255,np.uint8);canvas[40:40+a.shape[0],:a.shape[1]]=a;canvas[40:40+b.shape[0],420:420+b.shape[1]]=b
   cv2.putText(canvas,f'{cid} F{fid} {side}',(7,26),cv2.FONT_HERSHEY_SIMPLEX,.57,(20,30,40),1);tiles.append(canvas)
cv2.imwrite(str(V/'regression_context.jpg'),np.vstack([np.hstack(tiles[i:i+2])for i in range(0,len(tiles),2)]))
