from pathlib import Path
import json,hashlib
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent;G=O/'word100_web_completion_graph_20261111'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
src=[r for p in [G/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json',G/'candidate_description_supervision/additional_source_review/SOURCE_EPISODES_ADDED.json']for r in read(p)['rows']]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20);out=[]
for person in range(2,10):
    clips=sorted([r for r in src if r['person']==person],key=lambda r:r['word_id']);panels=[]
    for r in clips:
        assert sha(Path(r['source']))==r['source_sha256'];cap=cv2.VideoCapture(r['source'])
        for g in r['groups']:
            points=sorted(set(k for e in r['events'] if e['event_id']in g['event_ids']for k in e['progression_witness_frames']))
            # Explicit review points only. They do NOT automatically receive
            # start/middle/end state labels; RGB review decides those labels.
            ids=[g['onset'][0],points[len(points)//2],g['offset'][1]]
            images=[]
            for fid in ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,frame=cap.read();assert ok
                h,w=frame.shape[:2];img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                # Center crop solely for human-readable review, no model input.
                side=min(h,w);left=(w-side)//2;img=img.crop((left,0,left+side,h)).resize((350,350))
                images.append((fid,img))
            panels.append((r,g,images))
        cap.release()
    sheet=Image.new('RGB',(1080,len(panels)*400),'white');d=ImageDraw.Draw(sheet)
    for row,(r,g,images)in enumerate(panels):
        for col,(fid,img)in enumerate(images):
            x=col*360;y=row*400;sheet.paste(img,(x,y+42));d.text((x+4,y+6),f"{r['word_id']} f{fid} | family{g['family']}",font=font,fill='#172d49')
            out.append(dict(clip_id=r['clip_id'],person=person,split='fit'if person<=7 else'exposed_probe',word_id=r['word_id'],
                group_id=g['group_id'],family=g['family'],review_slot=col,frame=fid,source=r['source'],source_sha256=r['source_sha256'],
                posture_label=None,linguistic_correctness=None))
    p=U/f'review_record{person:02}.jpg';sheet.save(p,quality=92)
with (U/'REVIEW_INPUTS.json').open('x',encoding='utf-8')as f:json.dump(dict(rows=out,source_files=32,review_points=len(out),review_completed=False),f,ensure_ascii=False,indent=2)
print('Source review sheets and',len(out),'unlabeled points prepared.')
