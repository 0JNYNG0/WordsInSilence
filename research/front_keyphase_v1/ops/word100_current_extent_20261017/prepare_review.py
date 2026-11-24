import json,hashlib,zipfile,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;s=json.loads((U/'SCOPE.json').read_text('utf-8'))
items=[];videos={};folder=U/'public_videos';folder.mkdir(exist_ok=True)
for r in s['records']:
 if r['group']!='new8':continue
 raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate']);ts=raw['timestamp_us']/1e6
 core=np.where(np.logical_or.reduce([(ts>=a)&(ts<=b) for a,b in r['intervals']]))[0]
 def good(side):return obs['usable'][:,33+21*side+np.array([0,5,9,13,17])].all(1)
 for side in (0,1):
  gaps=[]
  for f in core[~good(side)[core]]:
   if not gaps or int(f)!=gaps[-1][-1]+1:gaps.append([])
   gaps[-1].append(int(f))
  if gaps:
   g=sorted(gaps,key=lambda g:(-len(g),g[0]))[0]
   for f in sorted(set((g[0],g[len(g)//2],g[-1]))):items.append({'clip_id':r['clip_id'],'label':r['label'],'side':side,'frame':f,'type':'missing','gap':[g[0],g[-1]],'box':r['box']})
 both=core[(good(0)&good(1))[core]]
 if len(both):
  for side in (0,1):items.append({'clip_id':r['clip_id'],'label':r['label'],'side':side,'frame':int(both[0]),'type':'control','box':r['box']})
 p=folder/f"{r['clip_id']}.mp4"
 if not p.exists():
  with zipfile.ZipFile(r['source']['archive']) as z:
   info=z.getinfo(r['source']['member']);assert info.CRC==r['source']['crc32']
   with z.open(info) as src,p.open('xb') as dst:
    while b:=src.read(1048576):dst.write(b)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==r['source']['sha256'];videos[r['clip_id']]=str(p)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
tiles=[]
for i,x in enumerate(items):
 x['review_id']=f'P{i+1:03}'
 cap=cv2.VideoCapture(videos[x['clip_id']]);cap.set(cv2.CAP_PROP_POS_FRAMES,x['frame']);ok,bgr=cap.read();cap.release();assert ok
 img=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w,h=img.size;img=img.resize((700,round(h*700/w)))
 d=ImageDraw.Draw(img);d.line((0,x['box'][3]*700/w,700,x['box'][3]*700/w),fill='#ffdf00',width=3)
 tile=Image.new('RGB',(700,img.height+65),'white');tile.paste(img,(0,65));d=ImageDraw.Draw(tile)
 d.text((8,4),f"{x['review_id']} {x['label']} / {'왼손 L' if x['side']==0 else '오른손 R'} / f{x['frame']}",font=font,fill='black')
 d.text((8,35),f"{x['type']} | yellow = actual input bottom",font=small,fill='black');tiles.append(tile)
for k in range(0,len(tiles),6):
 part=tiles[k:k+6];im=Image.new('RGB',(1400,part[0].height*math.ceil(len(part)/2)),'#e6ebf0')
 for j,t in enumerate(part):im.paste(t,((j%2)*700,(j//2)*t.height))
 im.save(U/f'review_{k//6+1:02}.jpg',quality=95)
with (U/'REVIEW_SELECTION.json').open('x',encoding='utf-8') as f:json.dump({'items':items,'videos':videos,'predictions_seen':False},f,ensure_ascii=False,indent=2)
print(json.dumps({'items':len(items),'pages':math.ceil(len(items)/6)},ensure_ascii=False))
