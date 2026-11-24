"""Independent visual sheets from frozen raw observations and existing public video."""
from pathlib import Path
import json,av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;P=U.parent/'word100_hand_assignment_20261016'
def read(p):return json.loads(Path(p).read_text('utf-8'))
plan=read(P/'RAW_CAPTURE_CONTEXT.json')
targets=[('0003','crop',32),('1140','crop',45),('0500','crop',47),('0949','crop',88),('2177','native',64),('2177','crop',78),('1139','native',47),('1139','native',71),('1139','crop',46),('1139','crop',56),('1140','crop',94),('2572','crop',34),('0106','crop',52)]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
index=[]
for word,mode,target in targets:
    cid=f'NIA_SL_WORD{word}_REAL01_F';item=next(v for v in plan['items'] if v['clip_id']==cid)
    obs=[]
    for p in [P/'raw_capture'/cid/mode/'baseline.npz',U/'replay'/cid/mode/'candidate.npz']:
        with np.load(p,allow_pickle=False) as z:obs.append({k:z[k] for k in z.files})
    ids=[target-1,target,target+1];frames={}
    with av.open(item['video_path']) as c:
        for f,fr in enumerate(c.decode(video=0)):
            if f in ids:
                im=fr.to_image()
                if mode=='crop':im=im.crop(item['box'])
                frames[f]=im
            if f>=max(ids):break
    sheet=Image.new('RGB',(1800,1260),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,5),f'{item["label"]} / {mode} / L cyan; R magenta',font=font,fill='black')
    for r,f in enumerate(ids):
        for col,title in enumerate(['Source','Production baseline','Repair candidate']):
            im=frames[f].copy();dr=ImageDraw.Draw(im)
            if col:
                ob=obs[col-1]
                for start,color,side in [(33,'cyan','L'),(54,'magenta','R')]:
                    xy=ob['xy_px'][f,start:start+21];valid=ob['usable'][f,start:start+21]
                    for j,k in edges:
                        if valid[j] and valid[k] and np.isfinite(xy[[j,k]]).all():dr.line([tuple(xy[j]),tuple(xy[k])],fill=color,width=5)
                    if valid[0] and np.isfinite(xy[0]).all():dr.text(tuple(xy[0]+[8,-25]),side,font=font,fill=color)
            im.thumbnail((595,350));x=600*col;y=70+395*r
            draw.text((x+5,y-30),f'{title} / frame {f}',font=font,fill='black');sheet.paste(im,(x,y))
    name=f'REVIEW_IMAGE_{word}_{mode}_{target}.jpg'
    if (U/name).exists():raise RuntimeError('Image already exists')
    sheet.save(U/name,quality=94)
    index.append({'word_id':word,'clip_id':cid,'mode':mode,'target_frame':target,'image':name})
with (U/'REVIEW_IMAGE_INDEX.json').open('x',encoding='utf-8') as h:json.dump(index,h,indent=2)
print('Rendered',len(index),'independent review sheets')
