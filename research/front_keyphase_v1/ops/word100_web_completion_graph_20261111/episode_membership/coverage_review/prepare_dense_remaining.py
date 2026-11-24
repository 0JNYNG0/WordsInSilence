"""Denser direct source samples for brief motions; no predicted labels."""
from prepare_remaining import D,read,sha
import json,cv2
from PIL import Image,ImageDraw,ImageFont
RANGES={'0739_05':[50,86],'1133_05':[40,66],'1567_05':[42,90],'2523_05':[58,74],
 '0739_06':[66,98],'1133_06':[38,58],'1567_06':[50,82],'2523_06':[64,82],
 '0739_08':[58,86],'1133_08':[66,94],'1567_08':[58,108],'2523_08':[50,66],
 '0739_09':[50,78],'1133_09':[42,62],'1567_09':[54,90],'2523_09':[62,78]}
def main():
 out=D/'DENSE_REMAINING_QUEUE.json';assert not out.exists();rows=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 for r in read(D/'REMAINING_REVIEW_QUEUE.json')['rows']:
  assert sha(r['source'])==r['source_sha256'];key=r['word_id'][4:]+'_'+str(r['person']).zfill(2);a,b=RANGES[key];wanted=list(range(a,b+1,2));cap=cv2.VideoCapture(r['source']);tiles=[];i=0
  while True:
   ok,f=cap.read()
   if not ok:break
   if i in wanted:
    h,w=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f[int(h*.08):int(h*.78),int(w*.24):int(w*.77)],cv2.COLOR_BGR2RGB));im.thumbnail((310,245));t=Image.new('RGB',(320,275),'white');t.paste(im,((320-im.width)//2,27));ImageDraw.Draw(t).text((6,3),f'frame {i} | {i/r["fps"]:.3f}s',fill='black',font=font);tiles.append(t)
   i+=1
  cap.release();assert len(tiles)==len(wanted);im=Image.new('RGB',(1600,((len(tiles)+4)//5)*275+35),'#ddd');ImageDraw.Draw(im).text((8,7),r['clip_id']+' | source RGB every2 frames | UNREVIEWED',fill='black',font=font)
  for j,t in enumerate(tiles):im.paste(t,((j%5)*320,(j//5)*275+35))
  p=D/(r['clip_id']+'_dense.jpg');assert not p.exists();im.save(p,quality=94);rows.append(dict(clip_id=r['clip_id'],path=str(p),sha256=sha(p),frames=wanted,reviewed=False))
 out.write_text(json.dumps(dict(rows=rows,reason='Pretraining audit found some short-motion probe classes had no directly observed inner samples. Denser RGB source review, not model-derived interpolation.',training_allowed=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print('Prepared16 dense source sheets.')
if __name__=='__main__':main()
