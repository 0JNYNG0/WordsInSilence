"""Local private diagnostic frames only; never a training manifest."""
from pathlib import Path
import json,cv2,numpy as np,hashlib
N=Path(__file__).resolve().parent
rows=json.loads((N/'DISCOVERY.json').read_text(encoding='utf8'))['rows']
out=N/'diagnostic_only';out.mkdir(exist_ok=True);meta=[]
choices=[[61,84,106,119,121,137],[74,100,125,145,165,180],[70,85,105,123,145,155],[50,60,75,85,115,140],[40,47,53,59,67,79],[45,55,75,90,110,125],[30,40,53,65,75,82],[0,10,35,55,75,100,130,180,220,240]]
for j,(r,ids)in enumerate(zip(rows,choices),1):
 p=Path(r['directory']);video=p/'recorded.mp4';c=cv2.VideoCapture(str(video));fps=c.get(cv2.CAP_PROP_FPS);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[]
 for i in ids:
  c.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=c.read()
  if not ok:continue
  im=cv2.resize(im,(480,360));tile=np.full((395,480,3),248,np.uint8);tile[35:]=im;cv2.putText(tile,f'Attempt {j} / frame {i} / {i/fps:.2f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(25,25,25),1);tiles.append(tile)
 c.release();cols=2 if j==8 else 3
 while len(tiles)%cols:tiles.append(np.full((395,480,3),248,np.uint8))
 sheet=np.vstack([np.hstack(tiles[i:i+cols])for i in range(0,len(tiles),cols)]);dest=out/f'attempt_{j:02d}.jpg';cv2.imwrite(str(dest),sheet,[cv2.IMWRITE_JPEG_QUALITY,92]);meta.append(dict(index=j,video=str(video),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),fps=fps,frames=n,selected_frames=ids,image=str(dest)))
(out/'FRAME_REVIEW_INPUTS.json').write_text(json.dumps(dict(rows=meta,scope='User-requested diagnostic review only',training_use=False,tuning_use=False,candidate_selection_use=False),ensure_ascii=False,indent=2),encoding='utf8')
print('DIAGNOSTIC_CONTACT_SHEETS',len(meta))
