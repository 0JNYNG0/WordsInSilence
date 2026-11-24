from pathlib import Path
import cv2,json,hashlib,math
from PIL import Image,ImageDraw
H=Path(__file__).resolve().parent
d=json.loads((H/'INPUTS.json').read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for r in d['rows']:
    if r['person']!=11:continue
    word=r['word_id']
    ids=([65,66,67,68,69,70,71,72,75,76,77,78,79,80,81,82,84,86,88,90] if word=='WORD0739' else [40,42,44,46,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,64])
    box=(600,210,1080,610) if word=='WORD0739' else (650,280,1270,620)
    cap=cv2.VideoCapture(r['source']);canvas=Image.new('RGB',(4*310,math.ceil(len(ids)/4)*250),'white');draw=ImageDraw.Draw(canvas)
    for j,f in enumerate(ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
        im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop(box);im.thumbnail((306,223));x=j%4*310;y=j//4*250
        canvas.paste(im,(x+(310-im.width)//2,y+25));draw.text((x+3,y+5),f'{word} REAL11 source f{f}',fill='black')
    cap.release();p=H/f'{r["clip_id"]}_hand_detail.jpg';canvas.save(p,quality=97)
    r['panels'].append(dict(path=str(p),sha256=sha(p),source_frame_ids=ids,source_crop_xyxy=list(box),reviewed=False))
(H/'INPUTS.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
print('created2detailpanels')
