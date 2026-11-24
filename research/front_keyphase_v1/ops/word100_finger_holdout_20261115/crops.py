from holdout_common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
side_by_word={'WORD0003':'L','WORD0071':'L','WORD0949':'R','WORD2551':'R','WORD1290':'L','WORD2700':'R','WORD2748':'R','WORD2718':'R'}
out=U/'crops';out.mkdir(exist_ok=True);rows=[]
for r in read(U/'FROZEN.json')['sources']:
 rec=U/'raw'/f"{r['clip_id']}.json"
 if not rec.exists():continue
 d=read(rec);raw=np.load(d['raw_path']);n=len(raw['frame_id']);hand=side_by_word[r['word']];ids=np.unique(np.linspace(round(n*.27),min(n-1,round(n*.77)),12).round().astype(int));sheet=Image.new('RGB',(1280,900),'white');dr=ImageDraw.Draw(sheet)
 cap=cv2.VideoCapture(r['path']);start=91 if hand=='L' else 112
 for k,i in enumerate(ids):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(i));ok,im=cap.read();assert ok
  pts=raw['xy_px'][max(0,i-2):min(n,i+3),start:start+21].reshape(-1,2);pts=pts[np.isfinite(pts).all(1)];lo=pts.min(0);hi=pts.max(0);c=(lo+hi)/2;s=max(hi-lo)+65;s=max(s,130);h,w=im.shape[:2];box=[max(0,int(c[0]-s/2)),max(0,int(c[1]-s/2)),min(w,int(c[0]+s/2)),min(h,int(c[1]+s/2))]
  rgb=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));crop=rgb.crop(box);crop.thumbnail((310,240));x=(k%4)*320;y=(k//4)*300;sheet.paste(crop,(x+(320-crop.width)//2,y+50));dr.text((x+5,y+3),f"{k}: {hand} f{i}  {raw['timestamp_us'][i]/1e6:.2f}s",font=font,fill='#18314a')
  rows.append(dict(clip_id=r['clip_id'],word=r['word'],person=r['person'],role=r['role'],hand=hand,index=k,frame=int(i),timestamp_us=int(raw['timestamp_us'][i]),box=box,source_sha256=r['sha256']))
 cap.release();sheet.save(out/f"{r['clip_id']}.jpg",quality=96)
write(U/'CROP_INPUTS.json',dict(rows=rows,predictions_seen=False,selection='Twelve fixed equally-spaced time points27%-77% before finger inference; raw box expands +/-2frames, no pose overlay on truth image'))
print('crops',len(rows))
