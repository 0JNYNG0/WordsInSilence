"""Render all hard changes using verified public RGB; no detection or inference."""
from run import *
import cv2
from PIL import Image,ImageDraw,ImageFont
FONT=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',19)
def main():
 c=read(U/'CONTEXT.json');rows={r['clip_id']:r for r in c['rows']};changes=read(U/'CHANGES.json');(U/'evidence').mkdir()
 out=[]
 for ix,ch in enumerate(changes,1):
  r=rows[ch['clip_id']];video=A/'videos'/f"{ch['clip_id']}.mp4";assert sha(video)==r['source']['sha256']
  raw=load(r['views'][ch['view']]['raw']);new=load(U/'results'/f"{ch['clip_id']}_{ch['view']}.npz");old=load(P/'results'/f"{ch['clip_id']}_{ch['view']}.npz");cap=cv2.VideoCapture(str(video));page=Image.new('RGB',(1800,1320),'#132638');facts=[]
  for j,(delta,plain) in enumerate([(-2,False),(-1,False),(0,True),(0,False),(1,False),(2,False)]):
   f=ch['index']+delta;fid=int(raw['frame_id'][f]);cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
   cv2.line(im,(0,int(raw['height'][f])),(im.shape[1]-1,int(raw['height'][f])),(0,220,255),3)
   text=[]
   for k in range(int(raw['raw_count'][f])):
    s0=int(old['sides'][f,k]);s1=int(new['sides'][f,k]);name=lambda s:'L' if s==0 else 'R' if s==1 else '보류'
    text.append(f"raw{k}: {name(s0)} → {name(s1)}")
    if not plain:
     color=(0,255,80) if delta==0 else (255,190,40)
     for p in raw['hand_xy_px'][f,k]:cv2.circle(im,tuple(p.astype(int)),4,color,-1)
     cv2.circle(im,tuple(raw['hand_xy_px'][f,k,0].astype(int)),12,color,2)
   sh=raw['pose_xy_px'][f,[11,12]];cx=int(sh[:,0].mean());x0=max(0,min(im.shape[1]-1080,cx-540))
   tile=Image.new('RGB',(600,660),'#132638');tile.paste(Image.fromarray(cv2.cvtColor(im[:,x0:x0+1080],cv2.COLOR_BGR2RGB)).resize((600,600)),(0,60));d=ImageDraw.Draw(tile)
   d.text((8,4),f"{ch['word']} f{fid} {'원본' if plain else '저장 관측'}",font=FONT,fill='white');d.text((8,30),' · '.join(text),font=FONT,fill='#c8e7fa');page.paste(tile,((j%3)*600,(j//3)*660))
   facts.append({'index':f,'frame_id':fid,'plain':plain,'old':old['sides'][f].tolist(),'new':new['sides'][f].tolist(),'state':new['state'][f].tolist()})
  cap.release();path=U/'evidence'/f'change_{ix:03}.jpg';page.save(path,quality=96)
  out.append({'change':ch,'source_video':str(video),'source_sha256':sha(video),'image':str(path),'frames':facts})
 write(U/'VISUAL_EVIDENCE.json',out);print('Rendered all hard changes',len(out),flush=True)
if __name__=='__main__':main()
