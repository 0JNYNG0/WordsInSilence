"""Public-source-only apple lesson preparation. No user footage access."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import cv2
U=Path(__file__).resolve().parent;O=U.parents[1];V=U.parent/'completion_v1'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=[r for r in read(O/'demo_handshape_learning_20261113/INPUTS.json')['rows']if r['word']=='WORD0099']
 for r in read(O/'basic_demo_feedback_20261115/acquisition_motion_followup/confirmation/EXTRACTED.json')['rows']:
  if r['word']=='WORD0099':rows.append(dict(r,video=r['path'],raw=r['raw_path'],channels=r['channel_path'],source_sha256=r['sha256']))
 rows=[r for r in rows if r['person']!=5]
 write(U/'INPUTS.json',dict(rows=rows,excluded_person5='Prior source expression mismatch unresolved; not silently admitted.',development_exposure='People11-15 already used for related model comparisons, not fresh final evaluation.'))
 for r in rows:
  assert sha(r['video'])==r['source_sha256']
  cap=cv2.VideoCapture(r['video']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
  a,b=r['source_interval'];ids=np.unique(np.rint(np.linspace(max(0,a*fps-5),min(n-1,b*fps+9),24)).astype(int));tiles=[]
  for i in ids:
   cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
   h,w=im.shape[:2];crop=im[int(h*.07):int(h*.83),int(w*.28):int(w*.72)];crop=cv2.resize(crop,(250,260))
   tile=np.full((290,250,3),255,np.uint8);tile[30:]=crop;cv2.putText(tile,f'P{r["person"]:02d} F{i}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.57,(20,25,30),1);tiles.append(tile)
  cap.release();dest=U/'review';dest.mkdir(exist_ok=True);cv2.imwrite(str(dest/f'p{r["person"]:02d}.jpg'),np.vstack([np.hstack(tiles[i:i+6])for i in range(0,24,6)]))
 print(json.dumps([dict(person=r['person'],role=r['role'],frames=r['frames'],interval=r['source_interval'])for r in rows]))
if __name__=='__main__':main()
