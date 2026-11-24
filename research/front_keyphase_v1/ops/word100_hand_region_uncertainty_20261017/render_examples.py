import json,hashlib
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent
def read(n):return json.loads((U/n).read_text('utf-8'))
scope=read('SCOPE.json');rows={r['clip_id']:r for r in scope['records']}
pred={r['clip_id']:r for r in read('PREDICTIONS.json')['records']}
fresh=read('REVIEW_SELECTION.json')['videos']
examples=[('NIA_SL_WORD0500_REAL06_F',55,0,'마음에 들다 · 누락 직후'),
 ('NIA_SL_WORD0500_REAL06_F',61,0,'마음에 들다 · 누락 약0.23초'),
 ('NIA_SL_WORD1624_REAL03_F',70,0,'유성구주민지원센터 · 손 경계 걸침'),
 ('NIA_SL_WORD1420_REAL04_F',59,1,'공항시장 · 두 손이 떨어진 장면')]
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',18)
small=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
tiles=[];logs=[]
for cid,frame,side,title in examples:
    r=rows[cid];video=r['source_video'] or fresh[cid]
    assert hashlib.sha256(Path(video).read_bytes()).hexdigest()==r['source']['sha256']
    cap=cv2.VideoCapture(video);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();assert ok
    rgb=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));scale=600/rgb.width
    rgb=rgb.resize((600,round(rgb.height*scale))).convert('RGBA')
    f=pred[cid]['frames'][frame];h=f['hands'][side];e=h['evidence']
    overlay=Image.new('RGBA',rgb.size);d=ImageDraw.Draw(overlay)
    outer=e['outer_bounds'];box=[outer[0][0]*scale,outer[0][1]*scale,outer[1][0]*scale,outer[1][1]*scale]
    d.rectangle(box,fill=(255,100,40,35),outline=(255,100,40,255),width=3)
    polygon=[(x*scale,y*scale) for x,y in e['nominal_hull']]
    d.line(polygon+[polygon[0]],fill=(75,255,170,255),width=3)
    bottom=r['box'][3]*scale;d.line((0,bottom,600,bottom),fill=(255,225,0,255),width=4)
    rgb=Image.alpha_composite(rgb,overlay).convert('RGB')
    canvas=Image.new('RGB',(600,460),'white');canvas.paste(rgb,(0,85));d=ImageDraw.Draw(canvas)
    d.text((10,5),title,font=font,fill='#163652')
    d.text((10,32),f"f{frame} | uncertainty {e['uncertainty_pixels']:.1f}px | missing {e['region_age_seconds']:.3f}s",font=small,fill='black')
    d.text((10,57),f"{f['baseline'][side]['state']} → {h['state']}",font=small,fill='#5b3e31')
    d.text((10,429),'초록: 과거 손 영역 추정 / 주황: 불확실성 범위 / 노랑: 화면 하단',font=small,fill='#334455')
    tiles.append(canvas)
    logs.append({'clip_id':cid,'frame':frame,'side':side,'title':title,'baseline':f['baseline'][side]['state'],'candidate':h['state'],'evidence':e})
sheet=Image.new('RGB',(1200,920),'#e8edf2')
for i,t in enumerate(tiles):sheet.paste(t,((i%2)*600,(i//2)*460))
sheet.save(U/'REGION_EXAMPLES.jpg',quality=95)
with (U/'REGION_EXAMPLES.json').open('x',encoding='utf-8') as f:json.dump({'examples':logs,'note':'Full RGB for verification display only; algorithm uses sensor coordinates. Orange is engineering uncertainty, not calibrated probability or recovered hand.'},f,ensure_ascii=False,indent=2)
print('rendered4')
