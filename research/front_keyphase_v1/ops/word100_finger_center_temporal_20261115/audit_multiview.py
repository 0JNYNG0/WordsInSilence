from common import *
import zipfile,cv2
from PIL import Image,ImageDraw,ImageFont
rows=read(U/'REGRESSION_SOURCES.json')['rows'];wanted={r['clip_id'][:-1]+v for r in rows for v in ['L','R']};found={};dest=U/'review'/'regression'
for ap in sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip')):
 with zipfile.ZipFile(ap)as z:
  for item in z.infolist():
   if Path(item.filename).stem in wanted:found[Path(item.filename).stem]=(ap,item.filename,item.CRC)
assert set(found)==wanted
s=importlib.util.spec_from_file_location('audit_stream',O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);engine=m.Engine();font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);output=[]
for r in rows:
 cid=r['clip_id'];sheet=Image.new('RGB',(1260,1080),'white');dr=ImageDraw.Draw(sheet);views={}
 for col,v in enumerate(['F','L','R']):
  p=Path(r['source'])if v=='F'else dest/f'{cid[:-1]+v}.mp4'
  if v!='F':
   ap,member,crc=found[cid[:-1]+v]
   if not p.exists():
    with zipfile.ZipFile(ap)as z:p.write_bytes(z.read(member))
   views[v]=dict(path=str(p),archive=str(ap),member=member,crc=crc,sha256=sha(p))
  cap=cv2.VideoCapture(str(p))
  for row,f in enumerate([r['frame']-2,r['frame'],r['frame']+2]):
   cap.set(1,f);ok,im=cap.read();assert ok;xy=engine.infer(im)['xy'];base=91 if r['hand']=='L' else 112;pts=xy[base:base+21];c=(pts.min(0)+pts.max(0))/2;size=max(max(pts.max(0)-pts.min(0))+90,210);box=tuple(int(v)for v in [c[0]-size/2,c[1]-size/2,c[0]+size/2,c[1]+size/2]);pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).crop(box);pic.thumbnail((410,310));sheet.paste(pic,(col*420,row*360+35));dr.text((col*420+5,row*360+4),f'{cid[7:]} f{f} {v}',font=font,fill='#142c49')
  cap.release()
 p=dest/f'{cid}_multiview.jpg';sheet.save(p,quality=97);output.append(r|dict(extra_views=views,panel=str(p)))
write(U/'REGRESSION_MULTIVIEW_SOURCES.json',dict(rows=output,note='New source audit after predictions; no labels or candidates changed. Side keypoints used only to crop images, never train. Cross-view frame index is nominal, not calibrated sync.'))
print('multiview diagnostic ready')
