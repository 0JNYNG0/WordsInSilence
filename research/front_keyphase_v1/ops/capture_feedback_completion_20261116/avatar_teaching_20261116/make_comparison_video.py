from pathlib import Path
import numpy as np,av,cv2
from PIL import Image,ImageDraw,ImageFont
U=Path(__file__).resolve().parent;font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
for word,label,rev,nf in [('WORD0099','사과','guided_v2',116),('WORD1290','감사','baseline',135)]:
 clip='NIA_SL_'+word+'_REAL01';D=U/clip/rev;cap=cv2.VideoCapture(str(U/clip/'video_only'/(clip+'_F.mp4')));out=av.open(str(U/(word+'_SIDE_BY_SIDE.mp4')),'w');st=out.add_stream('libx264',rate=30);st.width=1024;st.height=640;st.pix_fmt='yuv420p';st.options={'crf':'20','preset':'fast'};tiles=[]
 for f in range(nf):
  ok,bgr=cap.read();assert ok;im=Image.new('RGB',(1024,640),'#f5f7fa');source=Image.fromarray(cv2.cvtColor(bgr[:,480:1440],cv2.COLOR_BGR2RGB)).resize((512,576));render=Image.open(D/'frames'/f'{f:04d}.png').convert('RGB');im.paste(source,(0,64));im.paste(render,(512,64));draw=ImageDraw.Draw(im);draw.text((20,18),label+' · 공개 원본',font=font,fill='#20334e');draw.text((532,18),'캐릭터 · '+f'{f/30:.2f}초',font=font,fill='#20334e')
  for packet in st.encode(av.VideoFrame.from_ndarray(np.asarray(im),format='rgb24')):out.mux(packet)
  if f in np.linspace(0,nf-1,16,dtype=int):tiles.append(im.resize((320,200)))
 for packet in st.encode():out.mux(packet)
 out.close();cap.release();sheet=Image.new('RGB',(1280,800),'white')
 for i,t in enumerate(tiles):sheet.paste(t,(i%4*320,i//4*200))
 sheet.save(U/(word+'_FULL_FLOW.jpg'),quality=90)
 print(word,'complete',flush=True)
