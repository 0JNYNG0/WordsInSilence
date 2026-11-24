import json,math,hashlib,sys
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
s=read(U/'SCOPE.json');records={r['clip_id']:r for r in s['records']};out={r['clip_id']:r for r in read(U/'OUTPUT.json')['records']}
videos=read(O/'word100_hand_region_uncertainty_20261017/REVIEW_SELECTION.json')['videos']|read(U/'REVIEW_SELECTION.json')['videos']
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
selection=sys.argv[1] if len(sys.argv)>1 else 'ASSERTION_SELECTION.json'
prefix=sys.argv[2] if len(sys.argv)>2 else 'audit'
items=read(U/selection)['items'];tiles=[];paths={}
for x in items:
 cid=x['clip_id'];r=records[cid];p=r['source_video'] or videos[cid]
 if cid not in paths:assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==r['source']['sha256'];paths[cid]=p
 cap=cv2.VideoCapture(p);cap.set(cv2.CAP_PROP_POS_FRAMES,x['frame']);ok,bgr=cap.read();cap.release();assert ok
 im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));w,h=im.size;sc=700/w;im=im.resize((700,round(h*sc)))
 d=ImageDraw.Draw(im);d.line((0,r['box'][3]*sc,700,r['box'][3]*sc),fill='#ffe100',width=3)
 hand=out[cid]['frames'][x['frame']]['hands'][x['side']]
 for points,col in ((hand['current_extent'].get('hull'),'#46ff99'),(hand['current_overlap'].get('other_hull'),'#ff713d')):
  if points:
   p=[(a*sc,b*sc) for a,b in points];d.line(p+[p[0]],fill=col,width=2)
 tile=Image.new('RGB',(700,im.height+65),'white');tile.paste(im,(0,65));d=ImageDraw.Draw(tile)
 d.text((8,3),f"{x['review_id']} {x['label']} / {'왼손 L' if x['side']==0 else '오른손 R'} / f{x['frame']}",font=font,fill='black')
 d.text((8,34),f"{x['kind']} | green: Pose4 / orange: other Hand / yellow: input edge",font=small,fill='black');tiles.append(tile)
for start in range(0,len(tiles),6):
 part=tiles[start:start+6];im=Image.new('RGB',(1400,part[0].height*math.ceil(len(part)/2)),'#eef2f4')
 for j,p in enumerate(part):im.paste(p,((j%2)*700,(j//2)*p.height))
 im.save(U/f'{prefix}_{start//6+1:02}.jpg',quality=96)
print('audit pages',math.ceil(len(tiles)/6),'items',len(items))
