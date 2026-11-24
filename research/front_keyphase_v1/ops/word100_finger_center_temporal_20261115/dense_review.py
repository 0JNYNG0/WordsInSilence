from common import *
import cv2
from PIL import Image,ImageDraw,ImageFont
def rr(a,b):return list(range(a,b+1))
select={('WORD2551',2):rr(42,46)+rr(54,58)+rr(68,72),('WORD2551',5):rr(51,55)+rr(68,72)+rr(80,84),('WORD2551',7):rr(25,29)+rr(41,45)+rr(52,56),('WORD2748',2):rr(47,61),('WORD2748',5):rr(55,69),('WORD2748',7):rr(50,64),('WORD2551',10):rr(46,50)+rr(61,65)+rr(80,84),('WORD2551',16):rr(36,40)+rr(49,53)+rr(65,69),('WORD2748',11):rr(63,77),('WORD2748',12):rr(50,64),('WORD1290',14):[55,58,61,64,67,70],('WORD1290',15):[46,49,52,55,58,61]}
assert not(U/'DENSE_REVIEW.json').exists();write(U/'DENSE_SELECTION.json',dict(rows=[dict(word=w,person=p,frames=f)for(w,p),f in select.items()],selection='Source RGB overview chosen before any candidate inference; main transitions, brief raised index and stable open-hand controls; no fabricated frames'))
s=importlib.util.spec_from_file_location('audit_stream',O/'word100_rtmw_stream_reuse_20261108/stream_runtime.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);engine=m.Engine()
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17);out=U/'review'/'dense';out.mkdir(exist_ok=True);records=[]
for r in read(U/'READY_SOURCES.json')['rows']:
 raw=np.load(r['raw_path']);ids=select[r['word'],r['person']];caps={v:cv2.VideoCapture(d['path'])for v,d in r['views'].items()};cache={};hands=['R','L']if r['word']=='WORD1290'else['R']
 for hand in hands:
  base=112 if hand=='R'else 91
  for page in range((len(ids)+4)//5):
   sh=Image.new('RGB',(1600,1450),'white');dr=ImageDraw.Draw(sh)
   for row,f in enumerate(ids[page*5:page*5+5]):
    for col,v in enumerate(['F','R','L','overlay']):
     view='F'if v=='overlay'else v
     if view not in caps:continue
     if(view,f)not in cache:
      caps[view].set(1,f);ok,im=caps[view].read();assert ok
      points=raw['xy_px'][f]if view=='F'else engine.infer(im)['xy'];cache[view,f]=(im,points)
     im,points=cache[view,f];pic=Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));p=points[base:base+21];lo=p.min(0);hi=p.max(0);c=(lo+hi)/2;size=max(max(hi-lo)+80,180);box=[int(c[0]-size/2),int(c[1]-size/2),int(c[0]+size/2),int(c[1]+size/2)]
     if v=='overlay':
      d=ImageDraw.Draw(pic)
      for b,color in [(5,'#2563eb'),(9,'#009999'),(13,'#ed8623'),(17,'#b334be')]:d.line([tuple(q)for q in points[[base]+list(range(base+b,base+b+4))]],fill=color,width=3)
     pic=pic.crop(box);pic.thumbnail((390,255));sh.paste(pic,(400*col+(400-pic.width)//2,290*row+30));dr.text((400*col+4,290*row+2),f'{r["person"]:02} {r["word"]} {hand} f{f} {v}',font=font,fill='#12263d')
    records.append(dict(clip_id=r['clip_id'],hand=hand,frame=f,role=r['role'],image=str(out/f'{r["clip_id"]}_{hand}_{page}.jpg')))
   sh.save(out/f'{r["clip_id"]}_{hand}_{page}.jpg',quality=97)
 for c in caps.values():c.release()
 print(r['clip_id'],'review prepared',flush=True)
write(U/'DENSE_REVIEW.json',dict(rows=records,scope='Side RTMW used for ROI only; state/quality truth from original RGB, all model input features still front-only. No side pose values train the model.'))
