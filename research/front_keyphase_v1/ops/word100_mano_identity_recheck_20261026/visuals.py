from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;OLD=O/'word100_mano_threeway_20261025'
f=json.loads((U/'FROZEN.json').read_text('utf-8'));r=json.loads((U/'RESULT_V3.json').read_text('utf-8'));idx=json.loads((U/'TRACK_INDEX_V3.json').read_text('utf-8'))
out=U/'views';out.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);big=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',22)
links=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def frame(path,t):
 c=cv2.VideoCapture(str(path));c.set(cv2.CAP_PROP_POS_FRAMES,t);ok,b=c.read();c.release();assert ok;return Image.fromarray(cv2.cvtColor(b,cv2.COLOR_BGR2RGB))
def skeleton(im,xy,color):
 d=ImageDraw.Draw(im)
 for a,b in links:
  if np.isfinite(xy[[a,b]]).all():d.line([tuple(xy[a]),tuple(xy[b])],fill=color,width=3)
 return im
def panel(im,title,subtitle):
 p=Image.new('RGB',(600,520),'#122033');p.paste(im.crop((650,230,1250,650)),(0,55));d=ImageDraw.Draw(p);d.text((14,12),title,font=big,fill='white');d.text((14,480),subtitle,font=font,fill='#d4e4ee');return p
cid=f['manifest'][0]['clip_id'];m=f['manifest'][0];z=np.load(m['prepared']);t=82
raw=np.load(OLD/'raw_predictions/00_0082_1.npz')['xy_px']
panels=[]
for anchor,which in [(74,'이전 장면 → 현재 추적'),(88,'이후 장면 → 현재 역추적')]:
 tr=next(x for x in idx[cid] if x['anchor']==anchor and x['target']==t and x['side']==1);a=np.load(tr['path']);im=frame(m['rgb_path'],t);d=ImageDraw.Draw(im)
 for j in [4,8,12,16,20]:
  xy=a['xy'][j];color='#6fe3cb' if a['final_valid'][j] else '#d85e72'
  if np.isfinite(xy).all():d.ellipse((xy[0]-5,xy[1]-5,xy[0]+5,xy[1]+5),outline=color,width=3);d.text((xy[0]+5,xy[1]),str(j),font=font,fill=color)
 n=int(a['final_valid'][[4,8,12,16,20]].sum());panels.append(panel(im,which,f'{anchor} → {t}프레임 · 추적 조건 통과 손끝 {n}/5'))
im=frame(m['rgb_path'],t);skeleton(im,z['full_xy'][t,0],'#68e0d1');skeleton(im,raw,'#ffbd64')
panels.insert(0,panel(im,'현재 두 후보 · 좌우 충돌','청록: 기존 왼손칸 / 주황: HaMeR 오른손칸'))
sheet=Image.new('RGB',(1824,520),'#0c1725')
for j,p in enumerate(panels):sheet.paste(p,(j*612,0))
sheet.save(out/'tracking82.jpg',quality=94)
# Additional previously-held conflict frames, separate from7newly-blocked additions.
for t in [78,85,86]:
 raw=np.load(OLD/f'raw_predictions/00_{t:04d}_1.npz')['xy_px'];ims=[]
 ims.append(panel(frame(m['rgb_path'],t),f'원본 · {t}프레임','기존부터 새 후보 보류'))
 im=frame(m['rgb_path'],t);skeleton(im,z['full_xy'][t,0],'#68e0d1');skeleton(im,raw,'#ffbd64');ims.append(panel(im,'원본 후보 두 개를 그대로 표시','둘을 함께 채택했다는 뜻이 아님'))
 sh=Image.new('RGB',(1212,520),'#0c1725');sh.paste(ims[0],(0,0));sh.paste(ims[1],(612,0));sh.save(out/f'conflict{t}.jpg',quality=93)
# Full context source frames, not only display crops.
context=Image.new('RGB',(1824,430),'#122033');d=ImageDraw.Draw(context)
for j,t in enumerate([74,82,88]):
 im=frame(m['rgb_path'],t).resize((600,338));context.paste(im,(612*j,55));d.text((612*j+15,15),f'{t}프레임 · 원본 전체',font=big,fill='white')
context.save(out/'context74_82_88.jpg',quality=93)
print('5 comparison figures saved')
