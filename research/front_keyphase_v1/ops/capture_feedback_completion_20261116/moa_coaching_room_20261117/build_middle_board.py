from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent/'qa'
f=lambda n,b=False:ImageFont.truetype('C:/Windows/Fonts/malgunbd.ttf'if b else'C:/Windows/Fonts/malgun.ttf',n)
out=Image.new('RGB',(1600,1180),'#f2f5f9');g=ImageDraw.Draw(out)
g.text((44,27),'세모형과 ㅗ형 사이의 풍성한 연기',font=f(36,True),fill='#152942')
g.text((44,83),'MOA와 연결되는 위쪽 · 아래로 갈수록 넓어지는 볼륨 · 이전의 몽글몽글한 질감',font=f(21),fill='#455770')
rows=[(8,'01  로봇 아랫부분에서 시작','상승 직전에 MOA 밑에서 효과가 나오기 시작합니다.'),(10,'02  로봇을 따라 위로 올라가기','위쪽 줄기에서 아래쪽 덩어리로 부드럽게 넓어집니다.'),(13,'03  남은 흐름으로 공간 전환','MOA가 빠져나간 뒤 넓어지며 전환을 마무리합니다.'),(20,'04  오른쪽에서 다시 날아오기','전환이 끝난 공간으로 MOA가 돌아옵니다.')]
for k,(i,title,desc) in enumerate(rows):
 x=44+(k%2)*780;y=142+(k//2)*495
 g.rounded_rectangle((x,y,x+732,y+470),radius=14,fill='white')
 g.text((x+20,y+17),title,font=f(24,True),fill='#152942')
 im=Image.open(D/f'middle_seq_{i:02d}.png').convert('RGB');im.thumbnail((700,370),Image.Resampling.LANCZOS);out.paste(im,(x+16+(700-im.width)//2,y+58))
 g.text((x+20,y+428),desc,font=f(19),fill='#455770')
g.text((44,1150),'실제 렌더한 전환 시안 · 전환 중 대화창 숨김 · 기존 데모 미적용',font=f(18),fill='#66758a')
out.save(D/'MOA_VOLUMETRIC_MIDDLE_STORYBOARD.png')
frames=[]
for i in range(23):
 im=Image.open(D/f'middle_seq_{i:02d}.png').convert('RGB');im.thumbnail((840,520),Image.Resampling.LANCZOS);frames.append(im)
frames[0].save(D/'MOA_VOLUMETRIC_MIDDLE_PREVIEW.gif',save_all=True,append_images=frames[1:],duration=[200]*22+[1000],loop=0,optimize=False)
print('FADE_BOARD_AND_GIF_READY')



