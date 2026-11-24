from train_shape import *
import cv2
r=read(V/'holdout/EXTRACTED.json')['rows'][0];cap=cv2.VideoCapture(r['path']);tiles=[]
for fid in range(40,77,2):
 cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
 h,w=im.shape[:2];im=im[int(h*.12):int(h*.6),int(w*.32):int(w*.66)];im=cv2.resize(im,(320,290));t=np.full((320,320,3),255,np.uint8);t[30:]=im
 cv2.putText(t,f'F{fid}',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,40,60),2);tiles.append(t)
while len(tiles)%5:tiles.append(np.full_like(tiles[0],255))
cv2.imwrite(str(V/'holdout/review/dense.jpg'),np.vstack([np.hstack(tiles[i:i+5])for i in range(0,len(tiles),5)]));cap.release()
