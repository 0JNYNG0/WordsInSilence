"""Public RGB only review artifacts; no learned outputs or inference."""
from pathlib import Path
import cv2, json, hashlib, math
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
OPS=HERE.parents[2]
SRC=OPS/'word100_rtmw_joint_learning_20261108/SOURCES.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta=json.loads(SRC.read_text('utf-8-sig'))
rows=[]
for s in meta:
    if s['word_id'] not in ['WORD0739','WORD2523'] or s['person'] not in [11,12,13]: continue
    path=Path(s['feedback_source'])
    assert path.exists() and sha(path)==s['source_sha256']
    cap=cv2.VideoCapture(str(path)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    raw=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        raw.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release(); assert len(raw)==n
    rec={k:s[k] for k in ['clip_id','word_id','person','role','archive','member','source_sha256']}
    rec.update(source=str(path),frame_count=n,fps=fps,source_frame_origin=0,panels=[])
    sets=[('overview',sorted(set(range(0,n,4))|{n-1}))]
    if s['person']==11:
        sets += [(f'dense_{i//20:02d}',list(range(i,min(n,i+20)))) for i in range(0,n,20)]
    for name,ids in sets:
        cols=5 if name=='overview' else 4
        cell=(240,260) if name=='overview' else (300,300)
        canvas=Image.new('RGB',(cols*cell[0],math.ceil(len(ids)/cols)*cell[1]),'white'); d=ImageDraw.Draw(canvas)
        for j,f in enumerate(ids):
            im=Image.fromarray(raw[f]).crop((500,50,1420,1040));im.thumbnail((cell[0]-4,cell[1]-25))
            x=(j%cols)*cell[0];y=(j//cols)*cell[1]
            canvas.paste(im,(x+(cell[0]-im.width)//2,y+23)); d.text((x+4,y+5),f'{s["word_id"]} P{s["person"]} f{f}',fill='black')
        dst=HERE/f'{s["clip_id"]}_{name}.jpg';canvas.save(dst,quality=95)
        rec['panels'].append(dict(path=str(dst),sha256=sha(dst),source_frame_ids=ids,source_crop_xyxy=[500,50,1420,1040],reviewed=False))
    rows.append(rec)
(HERE/'INPUTS.json').write_text(json.dumps(dict(metadata=str(SRC),metadata_sha256=sha(SRC),rows=rows),ensure_ascii=False,indent=2)+'\n','utf-8')
print(json.dumps([dict(clip_id=r['clip_id'],frames=r['frame_count'],panels=len(r['panels'])) for r in rows]))
