"""Eight existing public FIT sources, before any new annotation/model scores."""
from core import *
import cv2
from PIL import Image,ImageDraw,ImageFont
D=W/'additional_source_review'
def main():
 D.mkdir(exist_ok=True);assert not(D/'QUEUE.json').exists();src=read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json');ready={r['clip_id']:r for r in read(O/'word100_shared_motion_learning_20261109/READY.json')['rows'] if r['role']=='fit'};out=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
 for r in src:
  if r['role']!='fit' or r['person']not in[3,7] or r['word_id']not in['WORD0739','WORD1133','WORD1567','WORD2523']:continue
  p=Path(r['feedback_source']);assert sha(p)==r['source_sha256'];assert r['clip_id']in ready;c=cv2.VideoCapture(str(p));fps=c.get(cv2.CAP_PROP_FPS);n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[];frames=list(range(0,n,6));i=0
  while True:
   ok,f=c.read()
   if not ok:break
   if i in frames:
    h,w=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f[int(h*.06):int(h*.81),int(w*.25):int(w*.77)],cv2.COLOR_BGR2RGB));im.thumbnail((305,245));t=Image.new('RGB',(320,280),'white');t.paste(im,((320-im.width)//2,30));ImageDraw.Draw(t).text((7,5),f'{i} | {i/fps:.2f}s',fill='black',font=font);tiles.append(t)
   i+=1
  c.release();assert len(tiles)==len(frames) and i==n;im=Image.new('RGB',(1600,35+280*((len(tiles)+4)//5)),'#ddd');ImageDraw.Draw(im).text((8,7),r['clip_id']+' | PUBLIC FIT - UNREVIEWED',font=font,fill='black')
  for j,t in enumerate(tiles):im.paste(t,(j%5*320,35+j//5*280))
  outp=D/(r['clip_id']+'_overview.jpg');im.save(outp,quality=94);out.append(dict(clip_id=r['clip_id'],word_id=r['word_id'],person=r['person'],role='fit',source=str(p),source_sha256=sha(p),frames=n,fps=fps,overview=str(outp),overview_sha256=sha(outp),viewed_frames=frames,reviewed=False,input=ready[r['clip_id']]['path']))
 assert len(out)==8;write(D/'QUEUE.json',dict(at=stamp(),rows=out,new_download=False,no_model_scores_read=True,training_allowed=False,reason='Additional existing FIT performers03/07, preserving persons08/09 probe and DEV/final split. Do not convert observed extent into mandatory linguistic completeness.'));print('Prepared eight existing public FIT sources for RGB annotation')
if __name__=='__main__':main()
