from pathlib import Path
import cv2
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent
for person,w,a,b in [(5,'0003',29,54),(5,'1290',43,64),(14,'0003',63,98),(14,'1290',47,72)]:
 cap=cv2.VideoCapture(str(U/'public_rgb'/f'NIA_SL_WORD{w}_REAL{person:02d}_F.mp4'));imgs=[]
 for f in range(a,b+1):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();assert ok;h,ww=im.shape[:2];im=Image.fromarray(cv2.cvtColor(im[int(h*.12):int(h*.64),int(ww*.28):int(ww*.72)],cv2.COLOR_BGR2RGB));im.thumbnail((294,210));tile=Image.new('RGB',(300,236),'white');tile.paste(im,((300-im.width)//2,24));ImageDraw.Draw(tile).text((3,3),f'{w} ID{person} f{f}',fill='black');imgs.append(tile)
 cap.release();sheet=Image.new('RGB',(1500,236*((len(imgs)+4)//5)),'white')
 for j,im in enumerate(imgs):sheet.paste(im,(j%5*300,j//5*236))
 sheet.save(U/'public_rgb'/f'dense_{w}_{person}.jpg',quality=94)
