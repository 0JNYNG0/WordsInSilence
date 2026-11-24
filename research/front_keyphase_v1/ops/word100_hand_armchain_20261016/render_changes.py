"""Render actual source frames plus raw pose chains/assigned hands; no inference."""
from pathlib import Path
import hashlib,json,sys
sys.dont_write_bytecode=True
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
P=U.with_name('word100_hand_assignment_20261016');J=U.with_name('word100_hand_assignment_joint_20261016')
def read(p):return json.loads(p.read_text('utf-8-sig'))
items={r['clip_id']:r for c in [read(P/'RAW_CAPTURE_CONTEXT.json'),read(J/'BROADENED_CAPTURE_CONTEXT.json')] for r in c['items']}
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
def overlay(im,raw,f,sides):
    im=im.copy();d=ImageDraw.Draw(im);w,h=im.size
    pose=raw['pose_xy_px'][f];use=raw['pose_usable'][f]
    for side,chain in enumerate([(11,13,15),(12,14,16)]):
        color=['#00ffcc','#ff6ad5'][side]
        for a,b in zip(chain[:-1],chain[1:]):
            if use[a] and use[b] and np.isfinite(pose[[a,b]]).all():d.line([tuple(pose[a]),tuple(pose[b])],fill=color,width=6)
        for label,j in zip(['S','E','W'],chain):
            p=pose[j]
            if not np.isfinite(p).all():continue
            x,y=map(float,p)
            if not (-.05*w<=x<=1.05*w and -.05*h<=y<=1.05*h):continue
            col=color if use[j] else '#aaaaaa'
            if use[j]:d.ellipse((x-7,y-7,x+7,y+7),fill=col)
            else:d.line((x-6,y-6,x+6,y+6),fill=col,width=3);d.line((x-6,y+6,x+6,y-6),fill=col,width=3)
            d.text((x+10,y+4),['L','R'][side]+label,font=font,fill=col)
    for i in range(int(raw['raw_count'][f])):
        side=int(sides[f,i]);col=['cyan','magenta'][side] if side>=0 else '#ffc04d'
        xy=raw['hand_xy_px'][f,i]
        valid=np.isfinite(xy).all(1)&(xy[:,0]>=-.05*w)&(xy[:,0]<=1.05*w)&(xy[:,1]>=-.05*h)&(xy[:,1]<=1.05*h)
        for a,b in edges:
            if valid[a] and valid[b]:d.line([tuple(xy[a]),tuple(xy[b])],fill=col,width=4)
        if valid[0]:d.text(tuple(xy[0]+[8,-28]),f'raw{i} '+(['L','R'][side] if side>=0 else '?'),font=font,fill=col)
    return im
index=[]
for row in read(U/'REPLAY_RESULT.json')['rows']:
    cid,mode=row['clip_id'],row['mode'];dest=U/'replay'/cid/mode
    changes=read(dest/'changes_vs_previous.json');targets=sorted(set(c['frame'] for c in changes))
    if not targets:continue
    item=items[cid]
    with np.load(Path(row['raw_dir'])/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    sides=[]
    for name in ['previous_assignment.npz','assignment.npz']:
        with np.load(dest/name,allow_pickle=False) as z:sides.append(z['sides'])
    traces=read(dest/'trace.json');needed=sorted({f for t in targets for f in [t-1,t,t+1] if 0<=f<len(raw['raw_count'])})
    frames={}
    with av.open(item['video_path']) as c:
        for f,fr in enumerate(c.decode(video=0)):
            if f in needed:
                im=fr.to_image()
                if mode=='crop':im=im.crop(item['box'])
                frames[f]=im
            if f>=needed[-1]:break
    for target in targets:
        ids=[f for f in [target-1,target,target+1] if f in frames]
        sheet=Image.new('RGB',(1800,1330),'white');d=ImageDraw.Draw(sheet)
        d.text((10,6),f'{item["label"]} / {cid} / {mode} / target {target}',font=font,fill='black')
        d.text((10,34),'L cyan / R magenta / unassigned orange. Pose S shoulder, E elbow, W wrist; gray X unusable.',font=small,fill='black')
        for r,f in enumerate(ids):
            for col,title in enumerate(['Source (actual detector input)','Previous joint v3','New arm-chain candidate']):
                im=frames[f].copy() if not col else overlay(frames[f],raw,f,sides[col-1])
                im.thumbnail((595,350));x=600*col;y=95+410*r
                d.text((x+5,y-26),f'{title} / frame {f}',font=small,fill='black');sheet.paste(im,(x,y))
                if col:
                    d.text((x+5,y+354),'raw sides '+str(sides[col-1][f].tolist()),font=small,fill='black')
                    if col==2:d.text((x+5,y+376),str(traces[f]['reasons'])[:70],font=small,fill='black')
        name=f'ARMCHAIN_IMAGE_{cid}_{mode}_{target}.jpg';assert not (U/name).exists();sheet.save(U/name,quality=94)
        index.append(dict(clip_id=cid,label=item['label'],mode=mode,target_frame=target,source_frame_indices=ids,image=name,
            changes=[c for c in changes if c['frame']==target],image_sha256=hashlib.sha256((U/name).read_bytes()).hexdigest()))
with (U/'VISUAL_IMAGE_INDEX.json').open('x',encoding='utf-8') as f:json.dump(index,f,ensure_ascii=False,indent=2)
print(json.dumps({'changed_frames_rendered':len(index),'new_detection':0,'training':0}))
