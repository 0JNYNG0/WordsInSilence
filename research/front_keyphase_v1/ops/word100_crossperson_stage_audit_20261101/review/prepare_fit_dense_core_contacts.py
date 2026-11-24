from pathlib import Path
import av,json
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parents[1];SRC=U.parent/'word100_core_stage_evidence_20261031'
manifest=json.loads((U/'evidence/FIT_CONTACT_MANIFEST.json').read_text(encoding='utf8'));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
rows=[]
for c in manifest['clips']:
    alias=c['alias'];out=U/'evidence'/alias
    lo=min(x['start_bracket'][0] for x in c['core_contact_rows']);hi=max(x['end_bracket'][1] for x in c['core_contact_rows'])
    meta=json.loads((SRC/'media'/f'{alias}.json').read_text(encoding='utf8'))
    with av.open(str(SRC/'media'/f'{alias}.webm')) as container:frames=[x.to_image() for x in container.decode(video=0)]
    for part,start in enumerate(range(lo,hi+1,24),1):
        ids=list(range(start,min(start+24,hi+1)));image=Image.new('RGB',(1200,((len(ids)+3)//4)*300),'white');draw=ImageDraw.Draw(image)
        for j,fid in enumerate(ids):
            x=j%4*300;y=j//4*300
            image.paste(frames[fid].crop((200,10,760,520)).resize((300,273)),(x,y+27));draw.text((x+3,y+3),f'{alias} f{fid} {meta["source_pts_seconds"][fid]:.3f}s',fill='black',font=font)
        target=out/f'dense_core_{part:02d}.jpg';image.save(target,quality=95)
        rows.append(dict(alias=alias,path=str(target),frames=ids,viewed=False))
with (U/'evidence/FIT_DENSE_MANIFEST.json').open('x',encoding='utf8') as f:json.dump(dict(sheets=rows,scope='Only generated, actual view_image inspection recorded separately'),f,ensure_ascii=False,indent=2)
print('generated',len(rows),'sheets',sum(len(x['frames']) for x in rows),'core/bracket frames')
