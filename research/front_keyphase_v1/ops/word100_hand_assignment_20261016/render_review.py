"""Local public TRAIN visual review, unchanged video frames + detector overlays."""
from pathlib import Path
import json,sys
import av,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parents[1]/'src'))
from wis_lab.core.observation import Observation
def read(p):return json.loads(Path(p).read_text('utf-8'))
plan=read(U/'RAW_CAPTURE_CONTEXT.json');result=read(U/'REPLAY_RESULT.json')
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
targets=[('NIA_SL_WORD1545_REAL01_F','crop',60,'known_swimming'),('NIA_SL_WORD0003_REAL01_F','crop',33,'known_sign_language'),('NIA_SL_WORD1133_REAL01_F','native',56,'known_well_done')]
for row in sorted(result['rows'],key=lambda r:r['counts'].get('dropped',0),reverse=True)[:3]:
    changes=read(U/'replay'/row['clip_id']/row['mode']/'changes.json')
    dropped=[r for r in changes if r['category']=='dropped']
    if dropped:
        targets.append((row['clip_id'],row['mode'],dropped[len(dropped)//2]['frame'],'drop_review'))
extra='--extra' in sys.argv
if extra:
    targets=read(U/'EXTRA_VISUAL_TARGETS.json')
index=[]
for cid,mode,target,tag in targets:
    item=next(v for v in plan['items'] if v['clip_id']==cid)
    obs=[Observation.load(str(U/'raw_capture'/cid/mode/'baseline.npz')),Observation.load(str(U/'replay'/cid/mode/'candidate.npz'))]
    with np.load(U/'raw_capture'/cid/mode/'raw.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    trace=read(U/'replay'/cid/mode/'trace.json')
    ids=range(max(0,target-1),min(obs[0].n_frames,target+2))
    frames={}
    with av.open(item['video_path']) as c:
        for f,frame in enumerate(c.decode(video=0)):
            if f in ids:
                im=frame.to_image()
                if mode=='crop':im=im.crop(item['box'])
                frames[f]=im
            if f>=max(ids):break
    canvas=Image.new('RGB',(1440,1050),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,8),f'{item["label"]} / {mode} / {tag} / L=청록 R=자홍',font=font,fill='black')
    for row,f in enumerate(ids):
        for col,title in enumerate(['raw detector','기존 배정','수정 후보']):
            im=frames[f].copy();dr=ImageDraw.Draw(im)
            if col==0:
                groups=[(raw['hand_xy_px'][f,i],np.ones(21,bool),['orange','lime'][i],f'{i}: '+str(raw['raw_label_name'][f,i])) for i in range(int(raw['raw_count'][f]))]
            else:
                ob=obs[col-1];groups=[(ob.xy_px[f,sl:sl+21],ob.usable[f,sl:sl+21],color,s) for sl,color,s in [(33,'cyan','L'),(54,'magenta','R')]]
            for xy,u,color,s in groups:
                for j,k in edges:
                    if u[j] and u[k] and np.isfinite(xy[[j,k]]).all():dr.line([tuple(xy[j]),tuple(xy[k])],fill=color,width=5)
                if u[0] and np.isfinite(xy[0]).all():dr.text(tuple(xy[0]+[8,-25]),s,font=font,fill=color)
            im.thumbnail((475,270));x=col*480;y=70+row*325
            draw.text((x+8,y-25),f'{title} / frame {f}',font=font,fill='black');canvas.paste(im,(x,y))
            if col==2:draw.text((x+5,y+275),', '.join(trace[f]['reasons']),font=font,fill='black')
    name=f'REVIEW_{cid}_{mode}_{target}.jpg';canvas.save(U/name,quality=91)
    index.append({'clip_id':cid,'label':item['label'],'mode':mode,'frame':target,'tag':tag,'image':name,'trace':trace[target]})
(U/('EXTRA_VISUAL_INDEX.json' if extra else 'VISUAL_REVIEW_INDEX.json')).write_text(json.dumps(index,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps([{k:r[k] for k in ['label','mode','frame','tag','image']} for r in index],ensure_ascii=False))
