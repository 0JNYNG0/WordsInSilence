"""Side-by-side source, previous, and new assignment; no inference."""
from pathlib import Path
import json,sys
sys.dont_write_bytecode=True
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;P=U.with_name('word100_hand_assignment_20261016')
def read(p):return json.loads(p.read_text('utf-8-sig'))
items={r['clip_id']:r for c in [read(P/'RAW_CAPTURE_CONTEXT.json'),read(U/'BROADENED_CAPTURE_CONTEXT.json')] for r in c['items']}
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
targets=[]
for row in read(U/'REPLAY_RESULT.json')['rows']:
    dest=U/'replay'/row['clip_id']/row['mode']
    changes=read(dest/'changes_vs_previous.json')
    for f in sorted(set(c['frame'] for c in changes)):
        targets.append((row,f))
index=[]
for row,target in targets:
    cid,mode=row['clip_id'],row['mode'];item=items[cid];d=U/'replay'/cid/mode
    with np.load(Path(row['raw_dir'])/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    sides=[np.load(d/n,allow_pickle=False)['sides'] for n in ['previous_assignment.npz','assignment.npz']]
    ids=[f for f in [target-1,target,target+1] if 0<=f<len(raw['raw_count'])];frames={}
    with av.open(item['video_path']) as c:
        for f,fr in enumerate(c.decode(video=0)):
            if f in ids:
                im=fr.to_image()
                if mode=='crop':im=im.crop(item['box'])
                frames[f]=im
            if f>=max(ids):break
    sheet=Image.new('RGB',(1800,1260),'white');draw=ImageDraw.Draw(sheet)
    draw.text((10,5),f'{item["label"]} / {cid} / {mode} / L cyan; R magenta',font=font,fill='black')
    for r,f in enumerate(ids):
        for col,title in enumerate(['Source','Previous v2','New candidate']):
            im=frames[f].copy();dr=ImageDraw.Draw(im)
            if col:
                for i in range(int(raw['raw_count'][f])):
                    side=int(sides[col-1][f,i])
                    if side<0:continue
                    color=['cyan','magenta'][side];xy=raw['hand_xy_px'][f,i];w,h=im.size
                    valid=np.isfinite(xy).all(1)&(xy[:,0]>=-.05*w)&(xy[:,0]<=1.05*w)&(xy[:,1]>=-.05*h)&(xy[:,1]<=1.05*h)
                    for j,k in edges:
                        if valid[j] and valid[k]:dr.line([tuple(xy[j]),tuple(xy[k])],fill=color,width=5)
                    if valid[0]:dr.text(tuple(xy[0]+[8,-25]),['L','R'][side],font=font,fill=color)
            im.thumbnail((595,350));x=600*col;y=70+395*r
            draw.text((x+5,y-30),f'{title} / frame {f}',font=font,fill='black');sheet.paste(im,(x,y))
    name=f'REVIEW_IMAGE_{cid}_{mode}_{target}.jpg'
    assert not (U/name).exists()
    sheet.save(U/name,quality=94)
    index.append(dict(clip_id=cid,label=item['label'],mode=mode,target_frame=target,image=name))
with (U/'REVIEW_IMAGE_INDEX.json').open('x',encoding='utf-8') as f:json.dump(index,f,ensure_ascii=False,indent=2)
print('Rendered',len(index),'changed-frame sheets')
