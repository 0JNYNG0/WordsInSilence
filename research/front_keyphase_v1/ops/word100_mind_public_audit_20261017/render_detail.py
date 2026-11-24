import json,hashlib
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
r=json.loads((U/'AUDIT.json').read_text('utf-8'))
P=U.parent/'word100_body_boundary_20261017'
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',16)
evidence=[]
for rid in ['NIA_SL_WORD0500_REAL06_F','NIA_SL_WORD1884_REAL07_F']:
    row=next(r for r in r['records'] if r['clip_id']==rid)
    rec=json.loads((P/'data/records'/f'{rid}.json').read_text('utf-8'))
    sensor=np.load(rec['views']['sensor']['observation'])
    masked=np.load(rec['views']['masked']['observation'])
    ts=sensor['timestamp_us']/1e6
    a,b=row['annotation']['data'][0]['start'],row['annotation']['data'][0]['end']
    core=(ts>=a)&(ts<=b)
    indices=np.flatnonzero(core)
    if '0500' in rid:sel=np.unique(np.linspace(indices[0],indices[-1],16).astype(int))
    else:sel=np.arange(max(0,indices[-1]-10),min(len(ts),indices[-1]+6))
    sheet=Image.new('RGB',(1600,4*264),'#f6f8fc');d=ImageDraw.Draw(sheet)
    cap=cv2.VideoCapture(row['video'])
    xy=sensor['xy_px'];good=sensor['usable'].astype(bool)
    for k,idx in enumerate(sel):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx));ok,im=cap.read();assert ok
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        x0,y0=k%4*400,k//4*264
        pic=Image.fromarray(im).resize((400,225));sheet.paste(pic,(x0,y0+23))
        bottom=rec['calibration']['box_source_pixels'][3]
        d.line((x0,y0+23+bottom*225/1080,x0+400,y0+23+bottom*225/1080),fill='#ffce00',width=2)
        for side,start,color in [('L',33,'#00ff60'),('R',54,'#ff55ff')]:
            for j in range(start,start+21):
                if good[idx,j]:
                    xx=x0+xy[idx,j,0]*400/1920;yy=y0+23+xy[idx,j,1]*225/1080
                    d.ellipse((xx-2,yy-2,xx+2,yy+2),fill=color)
        l=good[idx,[33,38,42,46,50]].all();rr=good[idx,[54,59,63,67,71]].all()
        ml=masked['usable'][idx,[33,38,42,46,50]].all()
        d.text((x0+4,y0+2),f'frame {idx} / {ts[idx]:.3f}s',font=font,fill='#172a41')
        d.text((x0+4,y0+246),f'sensor L:{int(l)} R:{int(rr)} / masked L:{int(ml)}',font=font,fill='#172a41')
    cap.release();sheet.save(U/(rid+'_DETAIL.jpg'),quality=95)
    frames=[]
    for idx in indices:
        palm=np.array([33,38,42,46,50])
        ml=bool(masked['usable'][idx,palm].all());sl=bool(good[idx,palm].all())
        if ml and not sl:frames.append({'frame':int(idx),'time':float(ts[idx])})
    evidence.append({'clip_id':rid,'sensor_missing_masked_present_left':frames,'detail_frames':sel.tolist()})
(U/'FOLLOWUP_DETAIL.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(evidence,indent=2))

