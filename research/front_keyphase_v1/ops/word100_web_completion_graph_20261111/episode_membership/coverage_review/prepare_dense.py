"""Preserve root RGB observations and prepare denser, unlabelled source frames."""
from pathlib import Path
import json,hashlib,cv2
from PIL import Image,ImageDraw,ImageFont
D=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
specs={
 'WORD0739_REAL02':dict(dense=[58,86],notes='Open hand is shown near the face at44..60; closing/lowering toward chest around64..72; forearm becomes upright around76..84; final pose84..96. Movement differs from the later hold.',provisional_events=[{'axis':0,'onset':[60,64],'offset':[72,76]},{'axis':1,'onset':[72,76],'offset':[80,84]}],holds=[[84,96]]),
 'WORD1133_REAL02':dict(dense=[42,66],notes='Right open hand reaches upper opposite arm44..48, travels toward its wrist around48..60, then stays near wrist64..72. Exact physical contact not established.',provisional_events=[{'axis':2,'onset':[44,48],'offset':[60,64]}],holds=[[64,72]]),
 'WORD1567_REAL02':dict(dense=[42,80],notes='Observed right-high at44, left-high52..56, right-high64, left-high76; then mostly held76..100. Three relative-height transitions, not three linguistically required repetitions.',provisional_events=[{'axis':4,'onset':[44,48],'offset':[52,56]},{'axis':4,'onset':[52,56],'offset':[64,68]},{'axis':4,'onset':[64,68],'offset':[72,76]}],holds=[[80,100]]),
 'WORD2523_REAL02':dict(dense=[48,66],notes='Hands form a central roof-like arrangement52..56, separate/lower56..60, hold60..84. Thumb/fingertip gathering not labelled from this overview.',provisional_events=[{'axis':3,'onset':[52,56],'offset':[60,64]}],holds=[[64,80]]),
 'WORD0739_REAL04':dict(dense=[84,118],notes='Open face presentation68..84; hand closes and lowers88..100; forearm rises104..116; pose held116..128. Existing prior fine RGB review must be reconciled before finalising.',provisional_events=[{'axis':0,'onset':[84,88],'offset':[96,100]},{'axis':1,'onset':[100,104],'offset':[112,116]}],holds=[[116,128]]),
 'WORD1133_REAL04':dict(dense=[70,96],notes='Both arms rise68..76; right hand near upper opposite forearm76..80, travels to wrist84..92, holds92..100. Initial approach/adjustment is separate from the sweep.',provisional_events=[{'axis':2,'onset':[76,80],'offset':[88,92]}],holds=[[92,100]]),
 'WORD1567_REAL04':dict(dense=[62,104],notes='Right-high64..68, left-high76, right-high88, left-high100; held100..120. Preserve the group of direction changes rather than choosing a single half-cycle as the whole action.',provisional_events=[{'axis':4,'onset':[68,72],'offset':[76,80]},{'axis':4,'onset':[76,80],'offset':[84,88]},{'axis':4,'onset':[88,92],'offset':[96,100]}],holds=[[104,120]]),
 'WORD2523_REAL04':dict(dense=[82,100],notes='Central arrangement76..88, separation/lowering88..96, held96..104. Exact contact or individual finger correctness is not asserted.',provisional_events=[{'axis':3,'onset':[88,92],'offset':[92,96]}],holds=[[96,104]])}
def main():
 assert not (D/'COARSE_RGB_REVIEW.json').exists();queue=read(D/'REVIEW_QUEUE.json')['rows'];out=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 for name,spec in specs.items():
  cid='NIA_SL_'+name+'_F';r=next(r for r in queue if r['clip_id']==cid);assert sha(r['source'])==r['source_sha256'];assert sha(r['overview'])==r['overview_sha256']
  wanted=list(range(spec['dense'][0],spec['dense'][1]+1,2));cap=cv2.VideoCapture(r['source']);tiles=[];i=0
  while True:
   ok,f=cap.read()
   if not ok:break
   if i in wanted:
    h,w=f.shape[:2];box=[int(w*.24),int(h*.08),int(w*.77),int(h*.78)];crop=f[box[1]:box[3],box[0]:box[2]];im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((310,245));t=Image.new('RGB',(320,275),'white');t.paste(im,((320-im.width)//2,27));ImageDraw.Draw(t).text((6,3),f'frame {i}  | {i/r["fps"]:.3f}s',fill='black',font=font);tiles.append(t)
   i+=1
  cap.release();assert len(tiles)==len(wanted)
  cols=5;im=Image.new('RGB',(1600,((len(tiles)+4)//5)*275+35),'#dddddd');ImageDraw.Draw(im).text((8,7),cid+' | source RGB every2 frames | UNREVIEWED dense sheet',fill='black',font=font)
  for j,t in enumerate(tiles):im.paste(t,((j%cols)*320,(j//cols)*275+35))
  p=D/f'{cid}_dense.jpg';im.save(p,quality=94)
  out.append(dict(clip_id=cid,person=r['person'],source=r['source'],source_sha256=r['source_sha256'],overview=r['overview'],overview_sha256=r['overview_sha256'],overview_viewed=True,reviewer='root Codex actual image inspection; not independent human/KSL expert',**spec,dense_path=str(p),dense_sha256=sha(p),dense_frames=wanted,dense_reviewed=False,admitted_for_training=False))
 write(D/'COARSE_RGB_REVIEW.json',dict(status='EIGHT_PUBLIC_FIT_OVERVIEWS_REVIEWED_DENSE_VERIFICATION_PENDING',rows=out,existing_frames_not_model_predictions=True,labels_are_provisional=True,no_framewise_label_interpolation=True,no_hidden_finger_or_contact_truth=True,training_allowed=False))
 print('Stored8 source RGB reviews and prepared8 dense sheets; not training labels yet.')
if __name__=='__main__':main()
