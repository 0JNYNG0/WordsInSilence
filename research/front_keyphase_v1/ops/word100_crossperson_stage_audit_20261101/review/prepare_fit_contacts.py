"""Read-only video decoding for RGB review; no prediction files or models."""
from pathlib import Path
import av,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parents[1];SRC=U.parent/'word100_core_stage_evidence_20261031'
ALIASES=['P033','P034','P035','P045','P046','P047','P063','P064','P065','P069','P070','P071']
phase=json.loads((SRC/'PHASE_DATA.json').read_text(encoding='utf8'))
clips={c['alias']:c for c in phase['clips']}
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
manifest=[]
for alias in ALIASES:
    metadata=json.loads((SRC/'media'/f'{alias}.json').read_text(encoding='utf8'))
    src=SRC/'media'/f'{alias}.webm';out=U/'evidence'/alias;out.mkdir(exist_ok=True)
    with av.open(str(src)) as container:frames=[f.to_image() for f in container.decode(video=0)]
    assert len(frames)==metadata['frames']
    whole=sorted(set(int(x) for x in np.linspace(0,len(frames)-1,24).round()))
    segments=[s for s in clips[alias]['segments'] if s['role']=='core_candidate']
    rows=[]
    for s in segments:
        a,b=s['core'];ids=sorted(set([s['start_bracket'][0],s['end_bracket'][1]]+[int(x) for x in np.linspace(a,b,5).round()]))
        rows.append((s,ids))
    contact=Image.new('RGB',(7*300,len(rows)*300),'white');draw=ImageDraw.Draw(contact)
    selected=set(whole)
    for ri,(segment,ids) in enumerate(rows):
        selected.update(ids)
        for ci,fid in enumerate(ids):
            image=frames[fid].crop((200,10,760,520)).resize((300,273))
            contact.paste(image,(ci*300,ri*300+27));draw.text((ci*300+3,ri*300+3),f"{segment['segment_id']}  f{fid}  {metadata['source_pts_seconds'][fid]:.3f}s",fill='black',font=font)
    contact.save(out/'core_contact.jpg',quality=95)
    wholeimg=Image.new('RGB',(4*320,6*205),'white');draw=ImageDraw.Draw(wholeimg)
    for index,fid in enumerate(whole):
        x=(index%4)*320;y=(index//4)*205
        wholeimg.paste(frames[fid].resize((320,180)),(x,y+25));draw.text((x+3,y+2),f'{alias} f{fid} {metadata["source_pts_seconds"][fid]:.3f}s',fill='black',font=font)
    wholeimg.save(out/'whole_contact.jpg',quality=95)
    for fid in selected:frames[fid].save(out/f'frame{fid:04d}.jpg',quality=95)
    manifest.append(dict(alias=alias,source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_frame_count=len(frames),whole_contact_frames=whole,core_contact_rows=[dict(segment_id=s['segment_id'],frames=ids,core=s['core'],start_bracket=s['start_bracket'],end_bracket=s['end_bracket']) for s,ids in rows],extracted_unique_frames=sorted(selected),crop=[200,10,760,520],full_frames_saved=True,source_pts_seconds={str(fid):metadata['source_pts_seconds'][fid] for fid in selected},viewed_by_reviewer=False))
with (U/'evidence/FIT_CONTACT_MANIFEST.json').open('x',encoding='utf8') as f:json.dump(dict(clips=manifest,scope='contact extraction only; viewed status is recorded separately after actual view_image'),f,ensure_ascii=False,indent=2)
print('prepared',len(manifest),'clips',sum(len(x['core_contact_rows']) for x in manifest),'cores')
