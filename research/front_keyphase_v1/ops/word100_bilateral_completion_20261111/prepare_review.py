from bilateral_core import *
import cv2,os
from PIL import Image,ImageDraw
from datetime import datetime,timezone,timedelta

def main():
 assert not (U/'REVIEW_READY.json').exists()
 docs=[ROOT/'WordsInSilence_PROJECT_CONTEXT.md',ROOT/'NEW_CHAT_HANDOFF.md',O/'STATE.json',ROOT/'Docs/PROJECT_DIRECTION_HISTORY_KO.txt']
 for p in docs:
  d=U/'before'/p.name;d.parent.mkdir(exist_ok=True);assert not d.exists();d.write_bytes(p.read_bytes())
 prot=read(R/'PROTECTIONS.json');prot.update({str(p):sha(p) for p in R.glob('*.py')});prot.update({str(p):sha(p) for p in R.glob('*.json')})
 for f in ['app.js','index.html','style.css','consumer.py','server.py','CARDS.json']:
  p=D/f;prot[str(p)]=sha(p);b=U/'before_web'/f;b.parent.mkdir(exist_ok=True);b.write_bytes(p.read_bytes())
 write(U/'PROTECTIONS.json',prot)
 write(U/'AUTHORIZATION.json',dict(user='검토한1~4단계모두;부분/완료구분;양손공통;공개검증후통과기능만웹연결',C_T3F_preserved=True,private_training_tuning_selection=False,downloads=False,camera=False,scope='New isolated heads and common binding; only passed research-web functions may change',implementation='Codex',independent_human=False))
 fitanns=read(A/'FIT_LABELS.json')['rows'];evalanns=read(A/'EVALUATION_LABELS.json')['rows']
 focus={f'NIA_SL_WORD{w}_REAL{p:02d}_F' for w in ['1567','2523'] for p in [1,3,5,11,12,13]}
 sources={r['clip_id']:r for r in read(D/'PREPARED.json')['rows']}
 for unit in [A,J]:
  p=unit/'EXTRACTION_COMPLETE.json'
  if p.exists():sources.update({r['clip_id']:r for r in read(p)['rows']})
 chosen=focus|{r['clip_id'] for r in fitanns};rows=[r for r in old.rows() if r['clip_id'] in chosen]
 cache=[];reviews=[]
 for r in rows:
  clip=r['clip_id'];p=U/'public_cache'/f'{clip}.npz';p.parent.mkdir(exist_ok=True)
  np.savez_compressed(p,x=prior.physical(r['x_full']),upper=prior.physical(r['x_upper']),ts=r['timestamp_us'],fid=r['frame_id'])
  cache.append(dict(clip_id=clip,word_id=r['word_id'],person=r['person'],role=r['role'],source_sha256=r['source_sha256'],path=str(p),sha256=sha(p)))
  if clip not in focus:continue
  src=sources[clip];assert sha(src['source'])==src['source_sha256']==r['source_sha256']
  # Inspect full upper body before labels; no model predictions or overlays.
  n=len(r['frame_id']);ids=np.linspace(int(n*.2),int(n*.85),30).round().astype(int)
  im=Image.new('RGB',(6*280,5*230),'white');draw=ImageDraw.Draw(im);cap=cv2.VideoCapture(src['source'])
  for j,k in enumerate(ids):
   f=int(r['frame_id'][k]);cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
   h,w=bgr.shape[:2];rgb=cv2.cvtColor(bgr[:int(h*.83),int(w*.18):int(w*.82)],cv2.COLOR_BGR2RGB);tile=Image.fromarray(rgb);tile.thumbnail((278,200));xx=j%6*280;yy=j//6*230;im.paste(tile,(xx+(280-tile.width)//2,yy+26));draw.text((xx+4,yy+3),f'{clip[7:]} idx{k} frame{f}',fill='black')
  cap.release();sp=U/'review'/f'{clip}.jpg';sp.parent.mkdir(exist_ok=True);im.save(sp,quality=95)
  reviews.append(dict(clip_id=clip,role=r['role'],person=r['person'],source=src['source'],source_sha256=src['source_sha256'],frames=ids.tolist(),sheet=str(sp),sheet_sha256=sha(sp)))
 write(U/'PUBLIC_CACHE.json',dict(rows=cache));write(U/'REVIEW_READY.json',dict(rows=reviews,model_predictions_shown=False,private_data=0,old_completed_inference_repeated=False))
 write(U/'CURRENT_EXECUTION.json',dict(status='LABEL_REVIEW',pid=None))
 state=read(O/'STATE.json');state.update(current_stage=U.name,next_single_action='공개양손정답검토→부분방향/완료분리학습→최종비교→통과한웹기능만연결',last_updated=datetime.now(timezone(timedelta(hours=9))).isoformat());state[U.name]=dict(status='LABEL_REVIEW',unit=str(U),private_data=0);write(O/'STATE.json',state)
 (ROOT/'NEW_CHAT_HANDOFF.md').write_text('# 최신 인수인계\n\n승인1~4단계 실행중. 현재단위 '+str(U)+'\n부분방향/행동완료분리·실제왼손정답보강을먼저진행. 새평가전조건동결,통과기능만웹에연결. 기존C·T3-F/100공통유지,사용자자료금지. 이전완료실험재실행금지.\n','utf-8')
 event('review_ready',public_cache=len(cache),sheets=len(reviews))
if __name__=='__main__':main()
