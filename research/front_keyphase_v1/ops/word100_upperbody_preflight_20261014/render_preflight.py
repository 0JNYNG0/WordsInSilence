"""Local evidence sheet of unchanged public source frames and declared crop."""
from pathlib import Path
import json
import av
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
p=json.loads((U/'PLAN.json').read_text('utf-8'))
labels=['멀다','공항','별','감사']
canvas=Image.new('RGB',(1200,4*510),'white');d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',21)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
for i,label in enumerate(labels):
    r=next(x for x in p['items'] if x['label']==label)
    frames=[];times=[]
    with av.open(r['video_path']) as c:
        s=c.streams.video[0];origin=None
        for frame in c.decode(s):
            if frame.pts is None:continue
            if origin is None:origin=frame.pts
            times.append(float((frame.pts-origin)*s.time_base));frames.append(frame.to_image())
    center=sum(r['core_seconds'])/2
    ids=[0,min(range(len(times)),key=lambda j:abs(times[j]-center)),len(times)-1]
    y=i*510
    d.text((8,y+3),f'{label} / REAL01 학습 참고 / 실제 재추출 입력 범위',fill='black',font=font)
    for j,k in enumerate(ids):
        original=frames[k].copy();mark=ImageDraw.Draw(original);mark.rectangle(r['box'],outline='yellow',width=8)
        original.thumbnail((390,220));canvas.paste(original,(j*400,y+52))
        cropped=frames[k].crop(r['box']);cropped.thumbnail((390,220));canvas.paste(cropped,(j*400,y+285))
        d.text((j*400+8,y+29),f'{["시작","주석 핵심 중간","종료"][j]} · {times[k]:.2f}s',fill='black',font=small)
    d.text((5,y+265),'위: 원본·노란 경계 / 아래: MediaPipe에 실제 입력한 픽셀 범위',fill='black',font=small)
dest=U/'ACTUAL_CROP_CONTACT.jpg';assert not dest.exists();canvas.save(dest,quality=90)
print(dest)
