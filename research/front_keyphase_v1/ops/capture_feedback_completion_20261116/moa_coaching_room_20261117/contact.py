from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent/'qa'
out=Image.new('RGB',(1200,900),'white');p=ImageDraw.Draw(out)
for i in range(12):
 im=Image.open(D/f'flight_seq_{i:02d}.png');im.thumbnail((290,260));x=(i%4)*300;y=(i//4)*300;out.paste(im,(x,y+25));p.text((x+8,y+5),str(i),fill='black')
out.save(D/'flight_contact.png')
