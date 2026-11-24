"""Evidence visualisation from existing RGB and saved observations only."""
from pathlib import Path
import json,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
a=json.loads((U/'AUDIT.json').read_text('utf-8'))
rows={r['clip_id']:r for r in a['records']}
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
big=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
caps={};raws={};assigned={}
def tile(cid,v,f,width=480):
    row=rows[cid];info=row['views'][v]
    if cid not in caps:caps[cid]=cv2.VideoCapture(row['video'])
    if (cid,v) not in raws:
        raws[cid,v]=dict(np.load(info['raw_path']));assigned[cid,v]=np.load(info['assignment_path'])['sides']
    raw=raws[cid,v];sides=assigned[cid,v][f];cap=caps[cid]
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
    h,w=bgr.shape[:2];scale=width/w;im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((width,round(h*scale)))
    d=ImageDraw.Draw(im)
    pose=raw['pose_xy_px'][f];pu=raw['pose_usable'][f]
    for s,chain in [(0,[11,13,15]),(1,[12,14,16])]:
        color=['#f9b02b','#00e6ff'][s]
        for x,y in zip(chain,chain[1:]):
            if pu[x] and pu[y]:d.line([tuple(pose[x]*scale),tuple(pose[y]*scale)],fill=color,width=2)
    for i in range(int(raw['raw_count'][f])):
        side=int(sides[i]);color={0:'#ffb229',1:'#00e5ff',-1:'#fb50dd'}[side]
        pts=raw['hand_xy_px'][f,i]*scale
        for x,y in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]:
            d.line([tuple(pts[x]),tuple(pts[y])],fill=color,width=2)
        for x,y in pts:d.ellipse((x-2,y-2,x+2,y+2),fill=color)
        x,y=pts[0];txt=f"{'LR'[raw['raw_label'][f,i]]}>{'LR'[side] if side>=0 else '?'}"
        d.text((x+4,y+5),txt,font=font,fill=color,stroke_width=1,stroke_fill='black')
    out=Image.new('RGB',(width,im.height+55),'#f3f6fa');out.paste(im,(0,30));d=ImageDraw.Draw(out)
    d.text((5,3),f"{row['label']} {row['signer_id']:02} {v} f{f}",font=font,fill='#11263a')
    d.text((5,im.height+32),f"{f/30:.2f}s | raw={int(raw['raw_count'][f])} | assigned={sides[:int(raw['raw_count'][f])].tolist()}",font=font,fill='#11263a')
    return out
def sheet(items,name,cols=4):
    tiles=[tile(*x) for x in items];ww,hh=tiles[0].size
    out=Image.new('RGB',(ww*cols,hh*math.ceil(len(tiles)/cols)+45),'white')
    d=ImageDraw.Draw(out);d.text((10,8),'Orange: anatomical L | Cyan: R | Pink: withheld | text raw label > assigned',font=big,fill='#172a40')
    for k,im in enumerate(tiles):out.paste(im,((k%cols)*ww,(k//cols)*hh+45))
    out.save(U/name,quality=93)
# Every flagged frame, without duplicates across flag reasons.
flag_frames=sorted(set((x['clip_id'],x['view'],f) for x in a['flags'] for f in x['frames']))
pages=[]
for i in range(0,len(flag_frames),16):
    name=f'flags_{i//16+1:02}.jpg';sheet(flag_frames[i:i+16],name);pages.append({'file':name,'frames':flag_frames[i:i+16]})
# Longest sensor gap per word, deterministic length then ID/side tie break.
gap_pages=[]
for word in ['WORD0500','WORD0003','WORD1642','WORD1884']:
    options=[(g['frames_count'],r['clip_id'],side,g['frames']) for r in a['records'] if r['word_code']==word for side in ['L','R'] for g in r['views']['sensor']['gaps'][side]]
    count,cid,side,fs=sorted(options,key=lambda x:(-x[0],x[1],x[2],x[3][0]))[0]
    raw=rows[cid]['views']['sensor'];n=len(np.load(raw['raw_path'])['timestamp_us'])
    # Inspect all missing frames plus two neighbours, paired native/sensor.
    sample=sorted(set([max(0,fs[0]-2),max(0,fs[0]-1)]+fs+[min(n-1,fs[-1]+1),min(n-1,fs[-1]+2)]))
    items=[(cid,v,f) for f in sample for v in ['native','sensor']]
    files=[]
    for i in range(0,len(items),16):
        name=f'{word}_gap_{i//16+1:02}.jpg';sheet(items[i:i+16],name);files.append(name)
    gap_pages.append({'word':word,'clip_id':cid,'side':side,'gap_frames':fs,'reviewed_context_frames':sample,'files':files})
# Four easily legible sequence strips: same signer01, 12 ordered annotation samples.
seq=[]
for word in ['WORD0500','WORD0003','WORD1642','WORD1884']:
    r=next(r for r in a['records'] if r['word_code']==word and r['signer_id']==1)
    core=[x['frame'] for x in r['views']['sensor']['frames']]
    fs=np.linspace(core[0],core[-1],12).round().astype(int).tolist()
    name=f'{word}_sequence.jpg';sheet([(r['clip_id'],'sensor',f) for f in fs],name,cols=4)
    seq.append({'word':word,'clip_id':r['clip_id'],'frames':fs,'file':name})
for c in caps.values():c.release()
with (U/'VISUAL_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump({'flag_pages':pages,'gap_pages':gap_pages,'sequence_pages':seq,'flag_unique_frames':len(flag_frames)},f,ensure_ascii=False,indent=2)
print(json.dumps({'flag_pages':len(pages),'flag_frames':len(flag_frames),'gap_pages':gap_pages,'sequence_pages':seq},ensure_ascii=False))

