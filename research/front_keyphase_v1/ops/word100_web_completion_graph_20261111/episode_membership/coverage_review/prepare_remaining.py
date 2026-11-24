"""Read only the remaining predeclared public FIT recordings for RGB audit."""
from pathlib import Path
import json,hashlib,cv2
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 out=D/'REMAINING_REVIEW_QUEUE.json';assert not out.exists()
 rows=[r for r in read(D/'REVIEW_QUEUE.json')['rows'] if r['person'] in [5,6,8,9]];assert len(rows)==16
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for r in rows:
  assert sha(r['source'])==r['source_sha256'];cap=cv2.VideoCapture(r['source']);fps=cap.get(cv2.CAP_PROP_FPS);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[];ids=[];i=0
  while True:
   ok,f=cap.read()
   if not ok:break
   if i%4==0:
    h,w=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f[int(h*.04):int(h*.9),int(w*.2):int(w*.8)],cv2.COLOR_BGR2RGB));im.thumbnail((230,230));tile=Image.new('RGB',(240,258),'white');tile.paste(im,((240-im.width)//2,24));ImageDraw.Draw(tile).text((5,3),f'{i} / {i/fps:.2f}s',fill='black',font=font);tiles.append(tile);ids.append(i)
   i+=1
  cap.release();assert i==n
  sheet=Image.new('RGB',(1440,((len(tiles)+5)//6)*258+35),'#ddd');ImageDraw.Draw(sheet).text((8,7),r['clip_id']+' | actual RGB every4 frames | UNREVIEWED',fill='black',font=font)
  for j,t in enumerate(tiles):sheet.paste(t,((j%6)*240,(j//6)*258+35))
  p=D/(r['clip_id']+'_overview.jpg');assert not p.exists();sheet.save(p,quality=93)
  r.update(overview=str(p),overview_sha256=sha(p),sampled_source_frames=ids,fps=fps,frames=n,reviewed=False)
 out.write_text(json.dumps(dict(rows=rows,selection='All remaining recorded FIT IDs 05,06,08,09 and all four words; no DEV scores used',training_allowed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print('Prepared 16 public FIT overview sheets; no labels or model execution.')
if __name__=='__main__':main()
