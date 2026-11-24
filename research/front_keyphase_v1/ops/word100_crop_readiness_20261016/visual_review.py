"""Render saved public frames and landmark overlays for manual inspection."""
from pathlib import Path
import json,sys
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;P=U.parent/'word100_upperbody_preflight_20261014'
p=json.loads((P/'PLAN.json').read_text('utf-8'));a=json.loads((U/'RESULT.json').read_text('utf-8'))
sys.path.insert(0,str(U.parents[1]/'src'))
from wis_lab.core.observation import Observation
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
index=[]
for label in ['잘하다','수영','수어','믿다']:
    item=next(x for x in p['items'] if x['label']==label);r=next(x for x in a['stored_pairs'] if x['label']==label)
    if r['cross_side_closer_frames']:target=r['cross_side_closer_frames'][0]['frame']
    else:target=r['all21_native_inside_but_crop_side_absent']['L'][0]['frame']
    obs={m:Observation.load(str(P/item['clip_id']/(m+'.npz'))) for m in ['native','crop']}
    ids=[max(0,target-1),target,min(obs['native'].n_frames-1,target+1)]
    imgs={}
    with av.open(item['video_path']) as c:
        for f,frame in enumerate(c.decode(video=0)):
            if f in ids:imgs[f]=frame.to_image()
            if f>=max(ids):break
    canvas=Image.new('RGB',(1280,1400),'white');d=ImageDraw.Draw(canvas)
    d.text((16,8),f'{label} / 공개 학습 REAL01 / L=청록, R=자홍 / 관절 추정 비교',font=font,fill='black')
    for row,f in enumerate(ids):
        for col,mode in enumerate(['native','crop']):
            img=imgs[f].copy();dr=ImageDraw.Draw(img);ob=obs[mode];xy=ob.xy_px[f].copy()
            if mode=='crop':xy+=np.array(item['box'][:2])
            dr.rectangle(item['box'],outline='yellow',width=5)
            for side,start,color in [('L',33,'cyan'),('R',54,'magenta')]:
                for j,k in edges:
                    if ob.usable[f,start+j] and ob.usable[f,start+k]:dr.line([tuple(xy[start+j]),tuple(xy[start+k])],fill=color,width=4)
                if ob.usable[f,start]:
                    x,y=xy[start];dr.ellipse([x-8,y-8,x+8,y+8],outline=color,width=4);dr.text((x+12,y-24),side,font=font,fill=color)
            for side,j in [('PL',15),('PR',16)]:
                if ob.usable[f,j]:
                    x,y=xy[j];dr.rectangle([x-6,y-6,x+6,y+6],outline='red',width=3);dr.text((x+10,y+10),side,font=small,fill='red')
            # Same original pixel view in both columns, including boundary context.
            img=img.crop((300,260,1620,1050));img.thumbnail((630,378))
            x=col*640;y=70+row*440
            d.text((x+12,y-27),f'{mode} / frame {f} / {ob.timestamp_us[f]/1e6:.3f}s',font=small,fill='black');canvas.paste(img,(x+5,y))
    name=f'PUBLIC_HAND_REVIEW_{item["word_id"]}.jpg';canvas.save(U/name,quality=91)
    index.append({'label':label,'clip_id':item['clip_id'],'frames':ids,'file':name,'source':item['video_path']})
(U/'VISUAL_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(index,ensure_ascii=False))
