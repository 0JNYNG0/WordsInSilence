"""Local review of already-saved replay frames and public reference videos."""
import json, zipfile, io
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
U=Path(__file__).resolve().parent; LAB=U.parents[1]
audit=json.loads((U/'AUDIT.json').read_text('utf-8'))
config=json.loads((LAB/'config/word100_practice.json').read_text('utf-8'))
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
for group,labels in enumerate([['멀다','공항','별'],['잘하다','믿다','감사']]):
    dest=U/f'CONTACT_{group+1}.jpg'
    assert not dest.exists()
    sheet=Image.new('RGB',(1280,len(labels)*420),'white');draw=ImageDraw.Draw(sheet)
    for ri,label in enumerate(labels):
        row=next(x for x in audit['rows'] if x['label']==label)
        item=next(x for x in config['items'] if x['label']==label)
        draw.text((8,ri*420),f'{label}  |  위: 저장 사용자 촬영 / 아래: AI Hub 참고 영상',fill='black',font=font)
        with zipfile.ZipFile(row['capture']) as z:
            manifest=json.loads(z.read('manifest.json'))
            frames=[f for f in manifest['frames'] if row['profile']['lo']<=f['observation_row']<=row['profile']['hi']]
            for j,frac in enumerate([.2,.4,.6,.8]):
                f=frames[round((len(frames)-1)*frac)]
                im=Image.open(io.BytesIO(z.read(f['image']))).convert('RGB');im.thumbnail((320,180))
                sheet.paste(im,(320*j,ri*420+30))
        cap=cv2.VideoCapture(item['video_path']);n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for j,frac in enumerate([.2,.4,.6,.8]):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round((n-1)*frac));ok,frame=cap.read();assert ok
            im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((320,180))
            sheet.paste(im,(320*j,ri*420+220))
        cap.release()
    sheet.save(dest,quality=90)
    print(dest)
