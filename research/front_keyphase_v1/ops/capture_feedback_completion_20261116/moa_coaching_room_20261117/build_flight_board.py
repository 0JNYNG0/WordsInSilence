from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent/'qa'
regular='C:/Windows/Fonts/malgun.ttf';bold='C:/Windows/Fonts/malgunbd.ttf'
f=lambda n,b=False:ImageFont.truetype(bold if b else regular,n)
out=Image.new('RGB',(1600,1190),'#f1f4f8');g=ImageDraw.Draw(out)
g.text((44,27),'MOA 비행 · 연기 전환 시안',font=f(36,True),fill='#152942')
g.text((44,83),'살짝 내려와 앞으로 → 위로 급상승 → 남은 연기가 커지며 공간 전환',font=f(22),fill='#455770')
rows=[(4,'01  고개를 숙이며 살짝 내려오기','대화창과 메뉴는 전환 시작과 동시에 숨깁니다.'),(6,'02  화면 앞쪽으로 돌아 나오기','앞으로 가까워지며 타원 곡선을 만듭니다.'),(8,'03  위로 급상승하며 화면 밖으로','아랫부분에서 연기가 나와 지나온 궤적에 남습니다.'),(10,'04  연기 꼬리가 빠르게 커지기','남은 연기가 넓어지며 검푸른 연습 공간으로 바뀝니다.')]
for k,(i,title,desc) in enumerate(rows):
 x=44+(k%2)*780;y=142+(k//2)*500
 g.rounded_rectangle((x,y,x+732,y+476),radius=14,fill='white')
 g.text((x+20,y+17),title,font=f(24,True),fill='#152942')
 im=Image.open(D/f'flight_seq_{i:02d}.png').convert('RGB');im=im.resize((700,426),Image.Resampling.LANCZOS)
 # Preserve the full capture inside a wide visual area.
 im.thumbnail((700,370));out.paste(im,(x+16+(700-im.width)//2,y+58))
 g.text((x+20,y+433),desc,font=f(19),fill='#455770')
g.text((44,1160),'기존 8821 데모에는 미적용 · 별도 시안의 실제 렌더 화면',font=f(18),fill='#66758a')
out.save(D/'MOA_FLIGHT_TRAIL_STORYBOARD.png')
frames=[]
for i in range(2,15):
 im=Image.open(D/f'flight_seq_{i:02d}.png').convert('RGB');im.thumbnail((900,550),Image.Resampling.LANCZOS);frames.append(im)
frames[0].save(D/'MOA_FLIGHT_TRAIL_PREVIEW.gif',save_all=True,append_images=frames[1:],duration=[400]+[260]*11+[1300],loop=0,optimize=False)
print('STORYBOARD_AND_GIF_READY')
