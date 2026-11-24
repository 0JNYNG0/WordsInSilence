from core import *
import cv2,math
from PIL import Image,ImageDraw,ImageFont

def main():
 folder=W/'short_action_review';folder.mkdir(exist_ok=True);rows=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',22)
 for s in read(U/'episode_membership/coverage_review/SOURCE_EPISODES_DENSE.json')['rows']:
  if s['word_id']!='WORD2523' or s['person'] not in [5,6,8,9]:continue
  assert sha(s['source'])==s['source_sha256'];e=s['events'][0];fs=list(range(e['onset'][0]-2,e['offset'][1]+3));cap=cv2.VideoCapture(s['source']);width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS)
  cw,ch=500,320;im=Image.new('RGB',(cw*4,ch*math.ceil(len(fs)/4)+50),'#eeeeee');draw=ImageDraw.Draw(im);draw.text((12,8),s['clip_id']+' | actual consecutive frames',font=font,fill='#222222');roi=[int(width*.28),int(height*.08),int(width*.72),int(height*.73)]
  for k,f in enumerate(fs):
   cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,frame=cap.read();assert ok;pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).crop(roi);pil.thumbnail((cw-6,ch-32));x=(k%4)*cw+(cw-pil.width)//2;y=(k//4)*ch+50;im.paste(pil,(x,y));draw.text(((k%4)*cw+10,y+ch-28),f'{f}  ({f/fps:.3f}s)',font=font,fill='#111111')
  cap.release();p=folder/(s['clip_id']+'.jpg');assert not p.exists();im.save(p,quality=94);rows.append(dict(clip_id=s['clip_id'],person=s['person'],source=s['source'],source_sha256=s['source_sha256'],path=str(p),sha256=sha(p),frames=fs,crop=roi,original_event=e,viewed=False))
 write(folder/'INPUTS.json',dict(rows=rows,status='PREPARED_NOT_YET_VIEWED'))
 print([r['path'] for r in rows])
if __name__=='__main__':main()
