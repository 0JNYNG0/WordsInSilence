"""Target the missing annotation type in already-owned100word FIT, before scoring anything."""
from pathlib import Path
import sys,zipfile,hashlib
import cv2
from PIL import Image,ImageDraw
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent
sys.path.insert(0,str(U))
from graph import read,write,stamp
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 assert not (W/'DISTINCT_SOURCE_INPUTS.json').exists()
 sources=read(O/'word100_rtmw_joint_learning_20261108/SOURCES.json');byclip={r['clip_id']:r for r in sources}
 profiles=read(O/'word100_linguistic_feedback_pipeline_20261027/FEATURE_PROFILES.json')['profiles'];words=['WORD0185','WORD0332'];people=[1,3,5]
 selection=dict(at=stamp(),words=words,people=people,rationale='Existing source descriptions contain explicitly successive different actions: behind-shoulder indication then opposite-hand relation; face-side waving then both hands rising. Select for distinct-action coverage, before model scores. Text-to-current-video variant must still be verified from RGB. These remain the existing100word public FIT.',profiles=[r for r in profiles if r['word_id'] in words],training_allowed=False,probe_or_final_not_accessed=True)
 write(W/'DISTINCT_SOURCE_SELECTION.json',selection);rows=[]
 for word in words:
  for person in people:
   clip=f'NIA_SL_{word}_REAL{person:02}_F';r=byclip[clip];assert r['role']=='fit'
   with zipfile.ZipFile(r['archive']) as z:data=z.read(r['member'])
   h=hashlib.sha256(data).hexdigest();assert h==r['source_sha256']
   dst=W/'public_sources'/f'{clip}.mp4';dst.parent.mkdir(exist_ok=True);assert not dst.exists();dst.write_bytes(data)
   cap=cv2.VideoCapture(str(dst));num=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));frames=list(range(0,num,3));im=Image.new('RGB',(1500,190*((len(frames)+5)//6)),'white');draw=ImageDraw.Draw(im)
   for j,f in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok;hh,ww=bgr.shape[:2];tile=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).crop((int(.25*ww),int(.02*hh),int(.75*ww),int(.9*hh)));tile.thumbnail((246,166));x,y=j%6*250,j//6*190;im.paste(tile,(x+(250-tile.width)//2,y+22));draw.text((x+3,y+3),f'{word} P{person} f{f}',fill='black')
   cap.release();sheet=dst.with_suffix('.jpg');im.save(sheet,quality=94)
   rows.append(dict(clip_id=clip,word_id=word,person=person,role='fit',source=str(dst),source_sha256=h,archive=r['archive'],member=r['member'],frames=num,fps=fps,sheet=str(sheet),sheet_sha256=sha(sheet),sampled_frames=frames,viewed=False,labels=None))
 write(W/'DISTINCT_SOURCE_INPUTS.json',dict(at=stamp(),rows=rows,inference=False,training=False,selection_sha256=sha(W/'DISTINCT_SOURCE_SELECTION.json')))
 print('Prepared6 publicFIT contact sheets; no labels assigned before viewing.')

if __name__=='__main__':main()
