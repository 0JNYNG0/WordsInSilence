"""Post-run visual safety audit of EVERY confident core run, sampled endpoints/middle.
This is not a new blinded benchmark and does not retune the candidate.
"""
from pathlib import Path
import json, math
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
p=json.loads((U/'PREDICTIONS.json').read_text('utf-8'))
s=json.loads((U/'SCOPE.json').read_text('utf-8'));rows={r['clip_id']:r for r in s['records']}
items=[];runs=[]
for r in p['records']:
    for side in (0,1):
        groups=[]
        for f in r['frames']:
            state=f['hands'][side]['state']
            if not f['core'] or state not in ('boundary_outside_likely','overlap_likely'):continue
            if not groups or f['frame']!=groups[-1][-1]['frame']+1 or state!=groups[-1][-1]['state']:groups.append([])
            groups[-1].append({'frame':f['frame'],'state':state})
        for g in groups:
            run_id=len(runs)+1
            runs.append({'run_id':run_id,'clip_id':r['clip_id'],'side':side,'state':g[0]['state'],'frames':[f['frame'] for f in g]})
            indices=sorted(set([0,len(g)//2,len(g)-1]))
            for j in indices:items.append({'id':f'A{len(items)+1:03}','run_id':run_id,'clip_id':r['clip_id'],'label':r['label'],'side':side,**g[j]})
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',14)
caps={};tiles=[]
for x in items:
    r=rows[x['clip_id']]
    if x['clip_id'] not in caps:
        path=r['source_video'] or str(U/'public_videos'/f"{r['clip_id']}.mp4")
        caps[x['clip_id']]=cv2.VideoCapture(path)
    cap=caps[x['clip_id']];cap.set(cv2.CAP_PROP_POS_FRAMES,x['frame']);ok,bgr=cap.read();assert ok
    h,w=bgr.shape[:2];im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).resize((600,round(h*600/w)))
    d=ImageDraw.Draw(im);bottom=r['box'][3]*600/w;d.line((0,bottom,600,bottom),fill='#ffdd00',width=4)
    tile=Image.new('RGB',(600,im.height+60),'white');tile.paste(im,(0,60));d=ImageDraw.Draw(tile)
    d.text((8,4),f"{x['id']} {x['label']} {r['signer_id']:02} {'L' if x['side']==0 else 'R'} f{x['frame']}",font=font,fill='black')
    d.text((8,32),x['state'],font=small,fill='#334455');tiles.append(tile)
pages=[]
for start in range(0,len(tiles),8):
    ts=tiles[start:start+8];image=Image.new('RGB',(1200,ts[0].height*math.ceil(len(ts)/2)),'#e7edf3')
    for j,t in enumerate(ts):image.paste(t,((j%2)*600,(j//2)*t.height))
    name=f'assertions_{start//8+1:02}.jpg';image.save(U/name,quality=94);pages.append(name)
for cap in caps.values():cap.release()
with (U/'ASSERTION_SELECTION.json').open('x',encoding='utf-8') as f:json.dump({'runs':runs,'items':items,'pages':pages,'note':'Post-result safety review; no coefficient changes; sampled frames not every frame in each run'},f,ensure_ascii=False,indent=2)
print(json.dumps({'runs':len(runs),'items':len(items),'pages':pages}))
