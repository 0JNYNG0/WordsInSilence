from pathlib import Path
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent
for w,a,b in [('0003',42,69),('0500',53,68),('0678',74,88),('1290',54,76)]:
 c=cv2.VideoCapture(str(U/'public_rgb'/f'NIA_SL_WORD{w}_REAL13_F.mp4'));ims=[];i=0
 while True:
  ok,f=c.read()
  if not ok:break
  if a<=i<=b:
   h,ww=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f[int(h*.15):int(h*.64),int(ww*.31):int(ww*.68)],cv2.COLOR_BGR2RGB));im.thumbnail((300,240));tile=Image.new('RGB',(310,266),'white');tile.paste(im,((310-im.width)//2,24));ImageDraw.Draw(tile).text((5,4),f'{w} frame {i}',fill='black');ims.append(tile)
  i+=1
 c.release();sheet=Image.new('RGB',(1550,266*((len(ims)+4)//5)),'white')
 for j,im in enumerate(ims):sheet.paste(im,(j%5*310,j//5*266))
 sheet.save(U/'public_rgb'/f'dense_{w}.jpg')
