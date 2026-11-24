from prepare_additional_source_review import *
RANGES={'0739_3':[64,94],'0739_7':[48,78],'1133_3':[50,74],'1133_7':[44,70],'1567_3':[38,80],'1567_7':[34,72],'2523_3':[62,82],'2523_7':[40,60]}
out=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
for r in read(D/'QUEUE.json')['rows']:
 key=r['word_id'][4:]+'_'+str(r['person']);a,b=RANGES[key];wanted=list(range(a,b+1,2));c=cv2.VideoCapture(r['source']);tiles=[];i=0
 while True:
  ok,f=c.read()
  if not ok:break
  if i in wanted:
   h,w=f.shape[:2];im=Image.fromarray(cv2.cvtColor(f[int(h*.07):int(h*.79),int(w*.24):int(w*.78)],cv2.COLOR_BGR2RGB));im.thumbnail((305,245));t=Image.new('RGB',(320,280),'white');t.paste(im,((320-im.width)//2,30));ImageDraw.Draw(t).text((7,5),f'frame {i}',fill='black',font=font);tiles.append(t)
  i+=1
 c.release();assert len(tiles)==len(wanted);im=Image.new('RGB',(1600,35+280*((len(tiles)+4)//5)),'#ddd');ImageDraw.Draw(im).text((8,7),r['clip_id']+' | dense RGB - UNREVIEWED',font=font,fill='black')
 for j,t in enumerate(tiles):im.paste(t,(j%5*320,35+j//5*280))
 p=D/(r['clip_id']+'_dense.jpg');assert not p.exists();im.save(p,quality=94);out.append(dict(clip_id=r['clip_id'],path=str(p),sha256=sha(p),frames=wanted,reviewed=False))
write(D/'DENSE_QUEUE.json',dict(rows=out,training_allowed=False,reason='Overview visually inspected; denser onset/end/turn comparison before annotation.'))
print('Eight dense RGB sheets ready')
