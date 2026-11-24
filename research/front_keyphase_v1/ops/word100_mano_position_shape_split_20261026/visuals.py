from pathlib import Path
import json,cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;H=O/'word100_mano_threeway_20261025';G=O/'word100_mano_duplicate_guard_20261025'
def read(p):return json.loads(Path(p).read_text('utf-8'))
f=read(U/'FROZEN.json');rows=read(U/'EVIDENCE.json');idx=read(U/'TRACK_INDEX.json');out=U/'views';out.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16);big=ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf',21)
links=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def data(r):
 k=r['clip_index'];t=r['frame'];s=r['side'];cid=r['clip']
 if k<7:
  m=f['manifest'][k];z=np.load(m['prepared']);pose=z['pose'][t];hand=z['full_xy'][t,s] if r['source']=='original' else np.load(H/f'raw_predictions/{k:02d}_{t:04d}_{s}.npz')['xy_px'];path=m['rgb_path'];label=m['word_label_for_display_only']
 else:
  m=next(m for m in f['controls'] if m['clip_id']==cid);z=np.load(m['path']);pose=z['xy_px'][t,:33];hand=z['xy_px'][t,33+s*21:54+s*21];path=G/'source_clips'/(cid+'.mp4');label='독재자' if k==7 else '추출'
 cap=cv2.VideoCapture(str(path));cap.set(cv2.CAP_PROP_POS_FRAMES,t);ok,b=cap.read();cap.release();assert ok
 return Image.fromarray(cv2.cvtColor(b,cv2.COLOR_BGR2RGB)),pose,hand,label
def draw_overlay(im,r,pose,hand):
 d=ImageDraw.Draw(im)
 for s,color in [(0,'#52dbe5'),(1,'#ffb34c')]:
  d.line([tuple(pose[j]) for j in [11+s,13+s,15+s]],fill=color,width=4)
 for a,b in links:d.line([tuple(hand[a]),tuple(hand[b])],fill='#ffcf58',width=3)
 for s in (0,1):
  for tr in r['assessment']['regions'][s]['details']:
   item=next(i for i in idx if i['clip']==r['clip'] and i['target']==r['frame'] and i['anchor']==tr['anchor'] and i['side']==s)
   z=np.load(item['path']);color='#64ef9b' if tr['anchor']<r['frame'] else '#e098ff'
   if tr['flow_qualified']:
    for p in z['xy'][z['final_valid']]:d.ellipse((p[0]-3,p[1]-3,p[0]+3,p[1]+3),fill=color)
 return im
new=[r for r in rows if r['previous_strict'] is not None and r['previous_strict']['supported_side']<0 and r['assessment']['identity_proposal']>=0]
assert len(new)==16
items=[]
for page in range(4):
 sheet=Image.new('RGB',(1500,1160),'#111e30');dd=ImageDraw.Draw(sheet)
 for j,r in enumerate(new[page*4:page*4+4]):
  im,p,h,label=data(r);x=(j%2)*750;y=(j//2)*580
  raw=im.crop((400,100,1520,940)).resize((360,270));ov=draw_overlay(im.copy(),r,p,h).crop((400,100,1520,940)).resize((360,270))
  dd.text((x+15,y+15),f"{label} · 촬영자 {f['manifest'][r['clip_index']]['signer_id']} · {r['frame']}프레임",font=big,fill='white')
  sheet.paste(raw,(x+10,y+65));sheet.paste(ov,(x+380,y+65))
  dd.text((x+12,y+345),'왼쪽: 원본 / 오른쪽: 팔·후보·영역 추적 근거',font=font,fill='#d8e4ef')
  s=r['assessment']['identity_proposal'];details=r['assessment']['regions'][s]['details'];text=' / '.join(f"{q['anchor']}→{r['frame']} : {q['surviving_points']}/{q['initial_points']}점" for q in details)
  dd.text((x+12,y+380),'위치·좌우 제안: '+('왼손' if s==0 else '오른손')+' / 손가락 모양 확정 아님',font=font,fill='#78eac4')
  dd.text((x+12,y+415),text,font=font,fill='white')
  dd.text((x+12,y+450),'청록·주황 선: 왼팔·오른팔 / 노랑: HaMeR 후보',font=font,fill='#d8e4ef')
  dd.text((x+12,y+480),'초록·보라 점: 앞뒤 영상에서 추적된 이미지 특징',font=font,fill='#d8e4ef')
 path=out/f'new_support_{page+1}.jpg';sheet.save(path,quality=95);items.append(dict(path=str(path),kind='all_new_coarse_support',records=[{k:r[k] for k in ['clip','frame','side']} for r in new[page*4:page*4+4]]))
for k,t in [(0,82),(7,59),(8,53)]:
 rs=[r for r in rows if r['clip_index']==k and r['frame']==t and (r['source']=='original' or (k==0 and r['side']==1))]
 sheet=Image.new('RGB',(720*len(rs),630),'#111e30');d=ImageDraw.Draw(sheet)
 for j,r in enumerate(rs):
  im,p,h,label=data(r);ov=draw_overlay(im,r,p,h);sheet.paste(ov.crop((500,220,1420,900)).resize((700,518)),(720*j+10,60));a=r['assessment']
  d.text((720*j+12,15),f"{label} {t} · {r['source']} {['왼','오른'][r['side']]}손칸",font=big,fill='white')
  d.text((720*j+12,590),f"팔 {a['arm_side']} / 영역 {a['region_side']} / 좌우제안 {a['identity_proposal']} (-1: 보류)",font=font,fill='#bde8dd')
 path=out/f'control_{k}_{t}.jpg';sheet.save(path,quality=95);items.append(dict(path=str(path),kind='fixed_conflict_or_control'))
(U/'VISUAL_ITEMS.json').write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding='utf-8');print('7 figure sheets saved; crops are display only; algorithm used full images.')
