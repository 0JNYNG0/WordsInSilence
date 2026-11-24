from common import *
import zipfile, cv2
from PIL import Image, ImageDraw, ImageFont

targets={'NIA_SL_WORD0071_REAL13_F':(82,'L'), 'NIA_SL_WORD0739_REAL12_F':(61,'R')}
dest=U/'review'/'regression';dest.mkdir(exist_ok=True)
found={}
for ap in sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip')):
 with zipfile.ZipFile(ap) as z:
  for item in z.infolist():
   if Path(item.filename).stem in targets:found[Path(item.filename).stem]=(ap,item.filename)
assert set(found)==set(targets)
rows=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
for cid,(frame,hand) in targets.items():
 p=dest/f'{cid}.mp4';ap,member=found[cid]
 if not p.exists():
  with zipfile.ZipFile(ap) as z:p.write_bytes(z.read(member))
 record=read(J/'records'/f'{cid}.json');assert sha(p)==record['source_sha256']
 raw=np.load(record['raw_path']);cap=cv2.VideoCapture(str(p));sheet=Image.new('RGB',(1260,1080),'white');dr=ImageDraw.Draw(sheet)
 base=91 if hand=='L' else 112
 for col,f in enumerate([frame-2,frame,frame+2]):
  cap.set(1,f);ok,bgr=cap.read();assert ok;im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));pts=raw['xy_px'][f,base:base+21];c=(pts.min(0)+pts.max(0))/2;size=max(max(pts.max(0)-pts.min(0))+80,200);box=tuple(int(v) for v in [c[0]-size/2,c[1]-size/2,c[0]+size/2,c[1]+size/2])
  full=im.copy();full.thumbnail((410,360));sheet.paste(full,(col*420,35))
  crop=im.crop(box).resize((400,310));sheet.paste(crop,(col*420,400))
  d=ImageDraw.Draw(im)
  for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:d.line([tuple(q) for q in pts[[0]+list(range(b,b+4))]],fill=color,width=4)
  sheet.paste(im.crop(box).resize((400,310)),(col*420,730));dr.text((col*420+8,6),f'{cid[7:]} {hand} f{f}',font=font,fill='#12263d')
 cap.release();image=dest/f'{cid}.jpg';sheet.save(image,quality=97)
 rows.append(dict(clip_id=cid,frame=frame,hand=hand,source=str(p),source_sha256=sha(p),image=str(image),archive=str(ap),member=member))
write(U/'REGRESSION_SOURCES.json',dict(rows=rows,selection='All three newly confident errors in two public clips, after predictions, diagnostic only; no relabeling or training'))
print('prepared',len(rows),'public regression panels')
