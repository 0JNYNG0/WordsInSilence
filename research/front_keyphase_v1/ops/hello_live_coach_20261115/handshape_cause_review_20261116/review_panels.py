from diagnose import *
def main():
 rows=json.loads((O/'basic_demo_feedback_20261115/local_shape/fresh/EXTRACTED.json').read_text(encoding='utf-8'))['rows'];tiles=[]
 for person,frame in [(16,75),(17,95),(17,102)]:
  row=next(r for r in rows if r['person']==person);z=load(row['raw_path']);cap=cv2.VideoCapture(row['path']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();cap.release();assert ok
  p=z['xy_px'][frame,112:133];lo=np.floor(p.min(0)-[25,25]).astype(int);hi=np.ceil(p.max(0)+[25,25]).astype(int);lo=np.maximum(lo,0);hi=np.minimum(hi,[im.shape[1],im.shape[0]])
  raw=im[lo[1]:hi[1],lo[0]:hi[0]];fac=400/max(raw.shape[:2]);a=cv2.resize(raw,None,fx=fac,fy=fac);b=a.copy();q=np.rint((p-lo)*fac).astype(int)
  for k in [5,9,13,17]:
   chain=[0]+list(range(k,k+4))
   for i,j in zip(chain,chain[1:]):cv2.line(b,tuple(q[i]),tuple(q[j]),(0,180,210),2,cv2.LINE_AA)
  canvas=np.full((450,820,3),255,np.uint8);canvas[40:40+a.shape[0],:a.shape[1]]=a;canvas[40:40+b.shape[0],420:420+b.shape[1]]=b;cv2.putText(canvas,f'public thanks / P{person} frame {frame}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(20,25,40),1);tiles.append(canvas)
 cv2.imwrite(str(V/'model_disagreement_review.jpg'),np.vstack(tiles))
if __name__=='__main__':main()
