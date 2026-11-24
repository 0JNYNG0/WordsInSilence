from pathlib import Path
import json,shutil,hashlib
import numpy as np,cv2,av
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;O=U.parent.parent;WEB=O/'web_tutorial_live_integration_20261113/site';A=WEB/'sign-demos';A.mkdir(exist_ok=True)
FONT='C:/Windows/Fonts/malgun.ttf'
catalog={}
for word,label,revision,geo,frames in [('WORD0099','사과','guided_v2','calibration_seed_v2',[43,50,55,65,70,78]),('WORD1290','감사','baseline','video_only',[47,54,61,68,74,80])]:
 clip='NIA_SL_'+word+'_REAL01';D=U/clip/revision;V=U/clip/'video_only';build=json.loads((D/'BUILD.json').read_text('utf-8'));assert build['frames']>0
 shutil.copy2(D/'motion.glb',A/(word+'.glb'))
 panels=[];cap=cv2.VideoCapture(str(V/(clip+'_F.mp4')))
 for frame in frames:
  cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();assert ok
  src=Image.fromarray(cv2.cvtColor(bgr[:,480:1440],cv2.COLOR_BGR2RGB)).resize((320,360))
  avatar=Image.open(D/f'front_{frame:03d}.png').convert('RGB').resize((320,360));side=Image.open(D/f'side_{frame:03d}.png').convert('RGB').resize((320,360))
  row=Image.new('RGB',(960,400),'white');draw=ImageDraw.Draw(row);font=ImageFont.truetype(FONT,18)
  for i,(im,title)in enumerate([(src,'원본'),(avatar,'캐릭터 정면'),(side,'캐릭터 옆')]):row.paste(im,(320*i,40));draw.text((320*i+12,8),f'{title} · {frame/30:.2f}초',font=font,fill='#20324b')
  panels.append(row)
 cap.release();out=Image.new('RGB',(960,400*len(panels)),'white')
 for i,row in enumerate(panels):out.paste(row,(0,i*400))
 out.save(U/(word+'_COMPARISON.jpg'),quality=91)
 entry=dict(label=label,url='sign-demos/'+word+'.glb',reference='/apple-media/reference.mp4'if word=='WORD0099'else'/basic-media/reference.mp4',preview=frames[1]/30,frames=build['frames'],fps=30,source_clip=clip,
  steps=[dict(label='① 입술 아래 스치기',text='검지를 펴서 입술 아래를 오른쪽으로 스쳐요.',time=43/30),dict(label='② 왼 주먹 닦기',text='왼손은 주먹, 오른손바닥으로 그 위를 위에서 아래로 닦아요.',time=65/30)]if word=='WORD0099'else[dict(label='양손 모양 준비',text='왼손은 손등이 위로 가게 눕히고, 오른손은 옆면을 아래로 세워요.',time=47/30),dict(label='두 번 이어 하기',text='오른손 옆면을 왼 손등에 두 번 대요.',time=54/30)],
  note='캐릭터 체형에 맞춰 옮긴 시범이에요. 손모양은 원본도 함께 살펴보세요.',sha256=hashlib.sha256((A/(word+'.glb')).read_bytes()).hexdigest())
 catalog[word]=entry
(A/'catalog.js').write_text('export const signDemos = '+json.dumps(catalog,ensure_ascii=False,indent=2)+';\n','utf-8')
(U/'PACKAGED.json').write_text(json.dumps(dict(status='PACKAGED_FOR_LOCAL_REVIEW_PENDING_ADMISSION',catalog=catalog),ensure_ascii=False,indent=2),'utf-8')
print('Packaged two local animations and source comparison panels.')
