"""Fixed public DEV RGB samples, chosen before candidate scores, no training labels."""
from pathlib import Path
import json,hashlib,zipfile,io
import av
from PIL import Image,ImageOps,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent;J=O/'word100_rtmw_joint_learning_20261108'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
words=['WORD0003','WORD0071','WORD0500','WORD0678','WORD0949','WORD1139','WORD1290','WORD2551']
assert not (U/'EXTRA_EVAL_RGB.json').exists()
rows=[r for r in read(J/'SOURCES.json') if r['word_id'] in words and r['role']=='development'];assert len(rows)==24
out=[]
for w in words:
 sheet=Image.new('RGB',(1350,1170),'white');draw=ImageDraw.Draw(sheet)
 for pi,r in enumerate(sorted([r for r in rows if r['word_id']==w],key=lambda x:x['person'])):
  ids=[round((r['frames_header']-1)*f) for f in [.35,.5,.65]]
  with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
  assert hashlib.sha256(data).hexdigest()==r['source_sha256']
  with av.open(io.BytesIO(data)) as con:
   for i,frame in enumerate(con.decode(video=0)):
    if i not in ids:continue
    im=frame.to_image();dst=U/'extra_eval_rgb'/f"{r['clip_id']}_{i}.jpg";dst.parent.mkdir(exist_ok=True);im.save(dst,quality=95)
    crop=im.crop((int(im.width*.22),int(im.height*.06),int(im.width*.78),int(im.height*.86)));crop=ImageOps.contain(crop,(448,360))
    k=ids.index(i);x=k*450;y=pi*390;sheet.paste(crop,(x+(450-crop.width)//2,y+25));draw.text((x+5,y+6),f"{w} ID{r['person']:02} frame{i}",fill='black')
    out.append(dict(clip_id=r['clip_id'],person=r['person'],word_id=w,frame=i,source_sha256=r['source_sha256'],image=str(dst),image_sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
  print(r['clip_id'],flush=True)
 dst=U/'extra_eval_review'/f'{w}.jpg';dst.parent.mkdir(exist_ok=True);sheet.save(dst,quality=95)
(U/'EXTRA_EVAL_RGB.json').write_text(json.dumps(dict(scope='24 public historicalDEV clips, fixed35/50/65percent frames, labels pending before predictions',rows=out),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
