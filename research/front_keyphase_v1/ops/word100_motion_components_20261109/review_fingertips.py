from common import *
import cv2
def main():
 rows=[r for r in read(U/'REVIEW.json')['rows'] if r['kind']=='late_hands'];out=[]
 for r in rows:
  z=np.load(r['raw_path']);samples=r['frames'][3:9:2];cap=cv2.VideoCapture(r['source']);canvas=np.full((660,990,3),245,np.uint8)
  for col,i in enumerate(samples):
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
   for row,start in enumerate([91,112]):
    pts=z['xy_px'][i,start:start+21];pts=pts[np.isfinite(pts).all(1)];cx,cy=pts.mean(0);size=max(80,float(np.ptp(pts,axis=0).max())*1.8);h,w=im.shape[:2]
    l,t,rr,b=max(0,int(cx-size/2)),max(0,int(cy-size/2)),min(w,int(cx+size/2)),min(h,int(cy+size/2));tile=cv2.resize(im[t:b,l:rr],(330,330));cv2.rectangle(tile,(0,0),(330,25),(245,245,245),-1);cv2.putText(tile,f"ID{r['person']} f{i} {'L' if row==0 else 'R'}",(5,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1);canvas[row*330:(row+1)*330,col*330:(col+1)*330]=tile
  cap.release();path=U/'review'/f"tips_{r['person']:02d}.jpg";cv2.imwrite(str(path),canvas);out.append(dict(clip_id=r['clip_id'],frames=samples,path=str(path),source_sha256=r['source_sha256']))
 write(U/'FINGERTIP_REVIEW_INPUT.json',dict(rows=out,method='Original public RGB crops; enlargement adds no resolution or hidden evidence; no model predictions overlaid'))
if __name__=='__main__':main()
