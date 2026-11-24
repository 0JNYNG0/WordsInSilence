from common_split import *
import zipfile,cv2
from PIL import Image,ImageDraw,ImageFont
targets={'NIA_SL_WORD2748_REAL10':61,'NIA_SL_WORD2748_REAL16':66,'NIA_SL_WORD2551_REAL12':35};found={}
for ap in sorted(Path('W:/WordsInSilence/data/AIHub_103/videos').rglob('*.zip')):
 with zipfile.ZipFile(ap)as z:
  for info in z.infolist():
   cid=Path(info.filename).stem
   if cid[:-2]in targets and cid[-1:]in ['F','L','R']:found[cid]=(ap,info.filename,info.CRC)
out=U/'review'/'multiview';out.mkdir(exist_ok=True);media=U/'media_audit';media.mkdir(exist_ok=True);records=[];font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',20)
for stem,f in targets.items():
 sheet=Image.new('RGB',(1800,1600),'white');d=ImageDraw.Draw(sheet)
 for col,view in enumerate(['F','L','R']):
  cid=stem+'_'+view;ap,member,crc=found[cid]
  with zipfile.ZipFile(ap)as z:b=z.read(member)
  p=media/(cid+'.mp4');p.write_bytes(b);cap=cv2.VideoCapture(str(p));n=int(cap.get(7));fps=cap.get(5);ww=cap.get(3);hh=cap.get(4)
  for row,offset in enumerate([-3,0,3]):
   ff=max(0,min(n-1,f+offset));cap.set(1,ff);ok,im=cap.read();assert ok
   pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));pic=pic.crop((int(ww*.25),int(hh*.08),int(ww*.75),int(hh*.80)));pic.thumbnail((590,490));sheet.paste(pic,(col*600+(600-pic.width)//2,row*530+35));d.text((col*600+5,row*530+3),f'{cid} f{ff} / {fps:g}fps',font=font,fill='black')
  cap.release();records.append(dict(clip_id=cid,path=str(p),sha256=sha(p),archive=str(ap),member=member,crc=crc,frames=n,fps=fps,center_frame=f,post_prediction_audit=True))
 sheet.save(out/(stem+'.jpg'),quality=97)
write(U/'MULTIVIEW_INPUTS.json',dict(rows=records,purpose='Audit source posture truth only; frozen models unchanged. Equal frame index alone is not proof of synchronization; compare neighboring visible motion too.'))
