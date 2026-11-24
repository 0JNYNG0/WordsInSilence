"""Reviewer-only source decoding and saved-array overlay, no model calls."""
from pathlib import Path
import json
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

U = Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text('utf-8'))
plan = read(U/'RAW_CAPTURE_CONTEXT.json')
targets = [('0500',47),('0949',88),('1139',49),('1139',50),('2177',61),('0106',53),('0106',54)]
font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
edges = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
index = []
for word,target in targets:
    cid=f'NIA_SL_WORD{word}_REAL01_F'; mode='crop'
    item=next(v for v in plan['items'] if v['clip_id']==cid)
    arrays=[]
    for path in [U/'raw_capture'/cid/mode/'raw.npz',U/'raw_capture'/cid/mode/'baseline.npz',U/'replay'/cid/mode/'candidate.npz']:
        with np.load(path,allow_pickle=False) as z: arrays.append({k:z[k] for k in z.files})
    raw,base,cand=arrays
    ids=[target-1,target,target+1]; frames={}
    with av.open(item['video_path']) as c:
        for f,frame in enumerate(c.decode(video=0)):
            if f in ids:frames[f]=frame.to_image().crop(item['box'])
            if f>=max(ids):break
    sheet=Image.new('RGB',(1800,1260),'white'); draw=ImageDraw.Draw(sheet)
    draw.text((10,5),f'{item["label"]} / {cid} / L cyan; R magenta',font=font,fill='black')
    for r,f in enumerate(ids):
        for col,title in enumerate(['Source (no overlay)','Baseline anatomical sides','Candidate anatomical sides']):
            im=frames[f].copy();dr=ImageDraw.Draw(im)
            if col:
                ob=base if col==1 else cand
                for start,color,side in [(33,'cyan','L'),(54,'magenta','R')]:
                    xy=ob['xy_px'][f,start:start+21];valid=ob['usable'][f,start:start+21]
                    for j,k in edges:
                        if valid[j] and valid[k] and np.isfinite(xy[[j,k]]).all():dr.line([tuple(xy[j]),tuple(xy[k])],fill=color,width=5)
                    if valid[0] and np.isfinite(xy[0]).all():dr.text(tuple(xy[0]+[8,-25]),side,font=font,fill=color)
            im.thumbnail((595,350));x=600*col;y=70+395*r
            draw.text((x+5,y-30),f'{title} / frame {f}',font=font,fill='black');sheet.paste(im,(x,y))
    dest=U/f'REVIEW_EXTRA_SIDE_{word}_{target}.jpg'
    if dest.exists():raise RuntimeError('Do not overwrite reviewer images')
    sheet.save(dest,quality=94)
    index.append({'word_id':word,'clip_id':cid,'target_frame':target,'image':dest.name})
with (U/'REVIEW_EXTRA_SIDE_INDEX.json').open('x',encoding='utf-8') as h:json.dump(index,h,indent=2)
print('Rendered',len(index),'saved-array review sheets')
