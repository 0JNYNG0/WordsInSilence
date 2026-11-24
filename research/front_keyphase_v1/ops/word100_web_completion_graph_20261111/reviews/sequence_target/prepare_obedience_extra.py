from pathlib import Path
import cv2,json,hashlib,math
from PIL import Image,ImageDraw
H=Path(__file__).resolve().parent
d=json.loads((H/'INPUTS.json').read_text('utf-8'))
for r in d['rows']:
    if r['word_id']!='WORD0739' or r['person']==11:continue
    cap=cv2.VideoCapture(r['source']);raw=[]
    while True:
        ok,bgr=cap.read()
        if not ok:break
        raw.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release();assert len(raw)==r['frame_count']
    for k in range(0,len(raw),20):
        ids=list(range(k,min(len(raw),k+20)));canvas=Image.new('RGB',(1200,math.ceil(len(ids)/4)*300),'white');draw=ImageDraw.Draw(canvas)
        for j,f in enumerate(ids):
            im=Image.fromarray(raw[f]).crop((500,50,1420,1040));im.thumbnail((296,275));x=j%4*300;y=j//4*300
            canvas.paste(im,(x+(300-im.width)//2,y+23));draw.text((x+4,y+4),f'WORD0739 REAL{r["person"]} source f{f}',fill='black')
        p=H/f'{r["clip_id"]}_dense_{k//20:02d}.jpg';canvas.save(p,quality=95)
        r['panels'].append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_frame_ids=ids,source_crop_xyxy=[500,50,1420,1040],reviewed=False))
(H/'INPUTS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
print('created15densepanels')
