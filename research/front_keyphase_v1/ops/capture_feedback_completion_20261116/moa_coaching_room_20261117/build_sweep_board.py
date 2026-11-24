from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent/'qa'
f=lambda n,b=False:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf'if b else'C:/Windows/Fonts/malgun.ttf',n)
out=Image.new('RGB',(1600,1180),'#f2f5f9');g=ImageDraw.Draw(out)
g.text((44,27),'MOA 상승에 맞춰 바뀌는 공간',font=f(36,True),fill='#152942')
g.text((44,83),'앞으로 다가오기 → 상승 직전 아래에서 시작 → 위로 번지며 공간 전환 → 오른쪽 재등장',font=f(21),fill='#455770')
rows=[(5,'01  앞으로 다가오는 동안은 유지','아직 흰 배경을 유지해 MOA의 움직임을 보여줍니다.'),(8,'02  상승 직전 아래쪽부터 시작','화면 아래에서 네이비색 페이드가 은은하게 시작됩니다.'),(11,'03  아래에서 위로 넓게 번지기','MOA가 올라가는 동안 부드러운 경계가 위로 이동합니다.'),(20,'04  오른쪽에서 다시 날아오기','전환이 끝난 공간으로 MOA가 돌아옵니다.')]
for k,(i,title,desc) in enumerate(rows):
 x=44+(k%2)*780;y=142+(k//2)*495
 g.rounded_rectangle((x,y,x+732,y+470),radius=14,fill='white')
 g.text((x+20,y+17),title,font=f(24,True),fill='#152942')
 im=Image.open(D/f'sweep_seq_{i:02d}.png').convert('RGB');im.thumbnail((700,370),Image.Resampling.LANCZOS);out.paste(im,(x+16+(700-im.width)//2,y+58))
 g.text((x+20,y+428),desc,font=f(19),fill='#455770')
g.text((44,1150),'실제 렌더한 전환 시안 · 전환 중 대화창 숨김 · 기존 데모 미적용',font=f(18),fill='#66758a')
out.save(D/'MOA_BOTTOM_UP_STORYBOARD.png')
frames=[]
for i in range(23):
 im=Image.open(D/f'sweep_seq_{i:02d}.png').convert('RGB');im.thumbnail((840,520),Image.Resampling.LANCZOS);frames.append(im)
frames[0].save(D/'MOA_BOTTOM_UP_PREVIEW.gif',save_all=True,append_images=frames[1:],duration=[200]*22+[1000],loop=0,optimize=False)
print('FADE_BOARD_AND_GIF_READY')

