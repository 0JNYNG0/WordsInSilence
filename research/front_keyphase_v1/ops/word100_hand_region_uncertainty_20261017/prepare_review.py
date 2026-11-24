"""Select and render public review before running visibility labels."""
import json, hashlib, zipfile, math
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
U=Path(__file__).resolve().parent
scope=json.loads((U/'SCOPE.json').read_text('utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
outdir=U/'public_videos';outdir.mkdir(exist_ok=True)
items=[];videos={}
for r in scope['records']:
    if r['group']!='fresh12':continue
    raw=np.load(r['paths']['raw']);obs=np.load(r['paths']['candidate'])
    ts=raw['timestamp_us']/1e6
    core=np.where(np.logical_or.reduce([(ts>=a)&(ts<=b) for a,b in r['intervals']]))[0]
    def good(side):return obs['usable'][:,33+21*side+np.array([0,5,9,13,17])].all(1)
    for side in (0,1):
        gaps=[]
        for frame in core[~good(side)[core]]:
            if not gaps or int(frame)!=gaps[-1][-1]+1:gaps.append([])
            gaps[-1].append(int(frame))
        if gaps:
            longest=sorted(gaps,key=lambda g:(-len(g),g[0]))[0]
            for frame in sorted(set([longest[0],longest[len(longest)//2],longest[-1]])):
                items.append({'clip_id':r['clip_id'],'label':r['label'],'side':side,'frame':frame,'type':'missing_longest','gap':[longest[0],longest[-1]],'box':r['box']})
    both=core[(good(0)&good(1))[core]]
    if len(both):
        for side in (0,1):items.append({'clip_id':r['clip_id'],'label':r['label'],'side':side,'frame':int(both[0]),'type':'observed_control','box':r['box']})
    p=outdir/f"{r['clip_id']}.mp4"
    if not p.exists():
        with zipfile.ZipFile(r['source']['archive']) as z:
            info=z.getinfo(r['source']['member']);assert info.CRC==r['source']['crc32'] and info.file_size==r['source']['bytes']
            with z.open(info) as src,p.open('xb') as dst:
                while block:=src.read(1048576):dst.write(block)
    assert sha(p)==r['source']['sha256'];videos[r['clip_id']]=str(p)
    print('prepared',r['clip_id'],flush=True)
for i,x in enumerate(items):x['review_id']=f'N{i+1:03}'
with (U/'REVIEW_SELECTION.json').open('x',encoding='utf-8') as f:json.dump({'items':items,'videos':videos,'code_predictions_seen':False,'selection':'longest gap start/middle/end per hand plus first both-observed core control'},f,ensure_ascii=False,indent=2)
caps={cid:cv2.VideoCapture(p) for cid,p in videos.items()}
tiles=[]
for x in items:
    cap=caps[x['clip_id']];cap.set(cv2.CAP_PROP_POS_FRAMES,x['frame']);ok,bgr=cap.read();assert ok
    h,w=bgr.shape[:2];image=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((600,round(h*600/w)))
    d=ImageDraw.Draw(image);bottom=x['box'][3]*600/w;d.line((0,bottom,600,bottom),fill='#ffdd00',width=4)
    canvas=Image.new('RGB',(600,image.height+60),'white');canvas.paste(image,(0,60));d=ImageDraw.Draw(canvas)
    d.text((8,4),f"{x['review_id']} {x['label']} · {'왼손 L' if x['side']==0 else '오른손 R'} · f{x['frame']}",font=font,fill='black')
    d.text((8,33),x['type']+' | yellow = actual sensor bottom',font=small,fill='#334455')
    tiles.append(canvas)
pages=[]
for start in range(0,len(tiles),8):
    part=tiles[start:start+8];im=Image.new('RGB',(1200,part[0].height*math.ceil(len(part)/2)), '#e8edf2')
    for j,p in enumerate(part):im.paste(p,((j%2)*600,(j//2)*p.height))
    name=f'review_{start//8+1:02}.jpg';im.save(U/name,quality=94);pages.append(name)
for cap in caps.values():cap.release()
with (U/'REVIEW_PAGES.json').open('x',encoding='utf-8') as f:json.dump({'pages':pages,'items':len(items)},f,indent=2)
print(json.dumps({'items':len(items),'pages':pages},ensure_ascii=False))
