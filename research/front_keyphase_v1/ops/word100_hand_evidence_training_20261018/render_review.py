from common import *
import numpy as np, cv2, zipfile, math
from PIL import Image,ImageDraw,ImageFont
c=read(U/'PREP_CONTEXT.json');assert read(U/'PREP_RESULT.json')['records']==1199
byid={r['clip_id']:r for r in c['records']};out=U/'review';out.mkdir(exist_ok=True);videos=U/'public_review_videos';videos.mkdir(exist_ok=True)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
selection=[];pages=[];tiles=[]
for num,cid in enumerate(c['additional_review_clips']):
 r=byid[cid];paths=r['views']['sensor'];obs=loadnp(paths['observation']);raw=loadnp(paths['raw']);new=loadnp(read(U/'inputs_v2'/f'{cid}.json')['views']['sensor']);ts=obs['timestamp_us']/1e6
 core=np.where(np.logical_or.reduce([(ts>=a)&(ts<=b) for a,b in r['intervals']]))[0];assert len(core)
 good=np.array([obs['usable'][:,33+21*s+np.array([0,5,9,13,17])].all(1) for s in (0,1)]).T
 gaps=[]
 for side in (0,1):
  chunks=[]
  for f in core[~good[core,side]]:
   if not chunks or f!=chunks[-1][-1]+1:chunks.append([])
   chunks[-1].append(int(f))
  gaps.extend((a,side) for a in chunks)
 items=[]
 if gaps:
  gap,side=min(gaps,key=lambda x:(-len(x[0]),x[0][0],x[1]));items=[{'frame':f,'side':side,'kind':'longest_gap'} for f in sorted(set([gap[0],gap[len(gap)//2],gap[-1]]))]
 normals=core[good[core].all(1)]
 if not len(normals):normals=core[good[core].any(1)]
 control=int(normals[0]) if len(normals) else int(core[len(core)//2]);items.append({'frame':control,'side':None,'kind':'control' if len(normals) else 'core_no_observed_control'})
 sel={'clip_id':cid,'label':r['label'],'items':items,'box':r['box']};selection.append(sel)
 p=videos/f'{cid}.mp4'
 if not p.exists():
  with zipfile.ZipFile(r['source']['archive']) as z:
   info=z.getinfo(r['source']['member']);assert info.CRC==r['source']['crc32']
   with z.open(info) as src,p.open('xb') as dst:
    while b:=src.read(1048576):dst.write(b)
 assert sha(p)==r['source']['sha256']
 tile=Image.new('RGB',(1680,310),'white');d=ImageDraw.Draw(tile);d.text((8,3),f'{num+1:02}  {r["label"]}  {cid}',font=font,fill='black')
 cap=cv2.VideoCapture(str(p))
 for j,it in enumerate(items):
  f=it['frame'];cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok,(cid,f)
  im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));sw,sh=im.size;ratio=420/sw;im=im.resize((420,round(sh*ratio)));dr=ImageDraw.Draw(im);dr.line((0,r['box'][3]*ratio,420,r['box'][3]*ratio),fill='yellow',width=2)
  for side,col in [(0,'#18ef40'),(1,'#ff80ef')]:
   for k in range(33+21*side,54+21*side):
    if obs['usable'][f,k]:
     x,y=obs['xy_px'][f,k]*ratio;dr.ellipse((x-1,y-1,x+1,y+1),fill=col)
  tile.paste(im,(420*j,30));d.text((420*j+4,270),f'f{f} {it["kind"]} L{new["full_states"][f,0].tolist()} R{new["full_states"][f,1].tolist()}',font=small,fill='black')
 cap.release();tiles.append(tile)
 if len(tiles)==6 or num+1==len(c['additional_review_clips']):
  page=Image.new('RGB',(1680,310*len(tiles)),'#eeeeee')
  for j,t in enumerate(tiles):page.paste(t,(0,310*j))
  name=out/f'page_{len(pages)+1:02}.jpg';page.save(name,quality=94);pages.append(str(name));tiles=[];print('Review page',len(pages),flush=True)
write(U/'REVIEW_SELECTION.json',{'at':now(),'clips':selection,'pages':pages,'selector_frozen_before_training':True,'model_predictions_seen':False,'legend':'yellow=input bottom; green=left actual usable points; purple=right; perhand[position,extent,intersection,proximity]. Position0observed1partial2interior3boundary4assignment5unknown; others0unavailable1inside/nohint2crossing/hint.'},new=True)
