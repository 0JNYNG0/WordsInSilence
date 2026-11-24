from pathlib import Path
import json,hashlib
from PIL import Image,ImageOps,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
rows=json.loads((O/'word100_shared_feature_bundle_20261027/DRAFT_LABELS.json').read_text(encoding='utf-8-sig'))['rows']
base={'WORD0739','WORD1133','WORD1567','WORD2523'}
selected=[r for r in rows if r['word_id'] not in base]
words=sorted({r['word_id'] for r in selected});out=[]
for w in words:
 rr=[r for r in selected if r['word_id']==w]
 sheet=Image.new('RGB',(1200,1062),'white');draw=ImageDraw.Draw(sheet)
 for k,r in enumerate(rr):
  src=Path(r['image']);im=Image.open(src).convert('RGB')
  # Preserve the full saved public image, with no model prediction overlay.
  im=ImageOps.contain(im,(398,326));x=(k%3)*400;y=(k//3)*354
  sheet.paste(im,(x+(400-im.width)//2,y+28))
  draw.text((x+6,y+7),f"{w} ID{r['recorded_person_id']:02} frame{r['frame_id']}",fill='black')
  out.append(dict(clip_id=r['clip_id'],frame_id=r['frame_id'],pts_us=r['pts_us'],source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
 dst=U/'review'/f'{w}.jpg';dst.parent.mkdir(exist_ok=True);sheet.save(dst,quality=94)
(U/'REVIEW_INPUTS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'words':words,'images':len(out),'source':'existing public RGB samples only; no model predictions'}))
