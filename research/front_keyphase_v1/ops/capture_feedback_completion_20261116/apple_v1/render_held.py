from apple_io import *
def main():
 for r in read(U/'holdout/EXTRACTED.json')['rows']:
  cap=cv2.VideoCapture(r['path']);a,b=r['source_interval'];ids=np.linspace(max(0,round(a*30)-4),min(r['frames']-1,round(b*30)+6),24).round().astype(int);tiles=[]
  for i in ids:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok;h,w=im.shape[:2];im=cv2.resize(im[int(h*.07):int(h*.76),int(w*.29):int(w*.71)],(250,270));tile=np.full((300,250,3),255,np.uint8);tile[30:]=im;cv2.putText(tile,f'P{r["person"]} F{i}',(4,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(20,30,40),1);tiles.append(tile)
  cv2.imwrite(str(U/'holdout/review'/f'p{r["person"]}_native.jpg'),np.vstack([np.hstack(tiles[i:i+6])for i in range(0,24,6)]));cap.release()
if __name__=='__main__':main()
