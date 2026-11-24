from core import *
import cv2
from PIL import Image,ImageDraw,ImageFont
s=next(r for r in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows'] if r['clip_id']=='NIA_SL_WORD0739_REAL08_F')
assert sha(s['source'])==s['source_sha256'];cap=cv2.VideoCapture(s['source']);frames=list(range(73,85));im=Image.new('RGB',(1600,1050),'#eeeeee');d=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
for k,f in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,a=cap.read();assert ok;h,w=a.shape[:2];p=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB)).crop((int(w*.28),int(h*.08),int(w*.72),int(h*.73)));p.thumbnail((390,310));x=k%4*400;y=k//4*350;im.paste(p,(x+(400-p.width)//2,y));d.text((x+10,y+315),f'frame {f}',font=font,fill='#111111')
cap.release();p=W/'short_action_review/NIA_SL_WORD0739_REAL08_F_forearm.jpg';assert not p.exists();im.save(p,quality=94)
