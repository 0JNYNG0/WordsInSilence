from pathlib import Path
import json,hashlib,zipfile
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_threeway_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=read(U/'FROZEN.json');r=read(U/'RESULT.json')
for sub in ['source_clips','views','source_frames']:(U/sub).mkdir(exist_ok=True)
(U/'.gitignore').write_text('source_clips/\nsource_frames/\n',encoding='utf-8')
FONT=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
LINKS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def getframe(path,t):
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,im=cap.read();cap.release();assert ok;return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
def mark(im,xy,valid,colors):
 im=im.copy();d=ImageDraw.Draw(im)
 for s in [0,1]:
  for a,b in LINKS:
   if valid[s,a] and valid[s,b]:d.line([tuple(xy[s,a]),tuple(xy[s,b])],fill=colors[s],width=3)
 return im
def tile(im,title,crop=None):
 im=im.crop(crop) if crop else im
 im=ImageOps.contain(im,(450,460));out=Image.new('RGB',(462,510),'#142337');out.paste(im,((462-im.width)//2,45+(460-im.height)//2));ImageDraw.Draw(out).text((10,10),title,font=FONT,fill='white');return out
def writepanel(items,dest):
 out=Image.new('RGB',(462*len(items),510))
 for k,(im,title,crop) in enumerate(items):out.paste(tile(im,title,crop),(462*k,0))
 out.save(dest,quality=96)
reviews=[]
for q in [x for x in r['rows'] if x['old_H'] and not x['new_G']]+[x for x in r['rows'] if (x['clip_index'],x['frame']) in [(0,66),(6,90),(0,75),(1,73)]]:
 m=f['manifest'][q['clip_index']];im=getframe(m['rgb_path'],q['frame']);t=q['frame'];old=np.load(OLD/'comparisons'/f"{m['clip_id']}.npz");new=np.load(U/'outputs'/f"{m['clip_id']}.npz")
 stem=f"{m['clip_id']}_{t:04d}";im.save(U/'source_frames'/f'{stem}.jpg',quality=98)
 before=mark(im,old['H_xy'][t],old['H_valid'][t],['#28e1d4','#ffa733']);after=mark(im,new['G_xy'][t],new['G_valid'][t],['#28e1d4','#ffa733'])
 crop=[650,280,1250,700];dest=U/'views'/f'{stem}.jpg';writepanel([(im,'원본 · 표시용 확대',crop),(before,'기존 통합 H',crop),(after,'새 중복 검사 G',crop)],dest)
 reviews.append({'id':stem,'type':'changed' if q['old_H'] and not q['new_G'] else 'protected_control','clip':m['clip_id'],'frame':t,'label':m['word_label_for_display_only'],'image':str(dest),'source':str(U/'source_frames'/f'{stem}.jpg')})
for n,q in enumerate(r['broad_proxy']['flags']):
 m=next(m for m in f['controls'] if m['clip_id']==q['clip_id']);src=m['source'];dest=U/'source_clips'/f"{m['clip_id']}.mp4"
 if not dest.exists():
  with zipfile.ZipFile(src['archive']) as z:data=z.read(src['member'])
  assert hashlib.sha256(data).hexdigest()==src['sha256'];dest.write_bytes(data)
 assert sha(dest)==src['sha256'];z=np.load(m['path']);t=q['frame_index'];im=getframe(dest,q['frame_id']);xy=z['xy_px'][t,33:].reshape(2,21,2);v=z['usable'][t,33:].reshape(2,21)
 stem=f"PROXY{n+1}_{m['clip_id']}_{t:04d}";im.save(U/'source_frames'/f'{stem}.jpg',quality=98)
 bounds=[int(xy[:,:,0].min())-100,int(xy[:,:,1].min())-100,int(xy[:,:,0].max())+100,int(xy[:,:,1].max())+100]
 bounds=[max(0,bounds[0]),max(0,bounds[1]),min(1920,bounds[2]),min(1080,bounds[3])]
 over=mark(im,xy,v,['#28e1d4','#ffa733']);panel=U/'views'/f'{stem}.jpg'
 writepanel([(im,'원본 전체',None),(im,'접촉 장면 확대',bounds),(over,'기존 좌우 좌표',bounds)],panel)
 items=[]
 for offset in [-2,-1,0,1,2]:
  tt=max(0,min(len(z['frame_id'])-1,t+offset));im2=getframe(dest,int(z['frame_id'][tt]));items.append((im2,f'원본 {tt}프레임',bounds))
 writepanel(items,U/'views'/f'{stem}_context.jpg')
 reviews.append({'id':stem,'type':'broad_proxy','clip':m['clip_id'],'frame':t,'label':m['label_for_display_only'],'image':str(panel),'context':str(U/'views'/f'{stem}_context.jpg'),'source':str(U/'source_frames'/f'{stem}.jpg')})
(U/'VISUAL_ITEMS.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2),encoding='utf-8');print('prepared',len(reviews),'visual review items')

