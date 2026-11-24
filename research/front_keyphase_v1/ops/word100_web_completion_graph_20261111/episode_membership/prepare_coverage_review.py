"""Prepare only named public FIT sources for review; no predicted labels."""
from pathlib import Path
import json,hashlib,collections
import cv2
from PIL import Image,ImageDraw,ImageFont
W=Path(__file__).resolve().parent;U=W.parent;O=U.parent;D=W/'coverage_review'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
 if (D/'REVIEW_QUEUE.json').exists():
  assert read(D/'REVIEW_QUEUE.json')['first_batch_visuals']==0 and (D/'INITIAL_PATH_LOOKUP.json').exists()
 words=['WORD0739','WORD1133','WORD1567','WORD2523'];raw=read(O/'word100_shared_motion_learning_20261109/READY.json')['rows']
 fit=[r for r in raw if r['role']=='fit' and r['word_id'] in words and 1<=r['person']<=9]
 assert len(fit)==36 and len({r['source_sha256'] for r in fit})==36
 ann=read(O/'word100_autonomous_localization_20261111/FIT_LABELS.json')['rows'];reviewed={r['clip_id'] for r in ann if r['role']=='fit'}
 canonical={r['clip_id']:r for r in read(O/'word100_multipeople_live_feedback_20261101/data_audit/SOURCES.json')['rows'] if r['role']=='fit'}
 queue=[];font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for r in sorted(fit,key=lambda r:(r['person'],r['word_id'])):
  old=canonical[r['clip_id']];assert old['source_sha256']==r['source_sha256'];video=Path(old['source'])
  exists=video.is_file();status='existing_reviewed_episode_labels' if r['clip_id'] in reviewed else 'needs_source_RGB_episode_review'
  record=dict(clip_id=r['clip_id'],person=r['person'],word_id=r['word_id'],source=str(video),source_sha256=r['source_sha256'],exists=exists,status=status,split='fit',raw_cache=r['path'],raw_sha256=r['sha256'])
  if r['person'] in (2,4) and status.startswith('needs') and exists:
   assert sha(video)==r['source_sha256'];cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));tiles=[];ids=[];i=0
   while True:
    ok,frame=cap.read()
    if not ok:break
    if i%4==0:
     h,w=frame.shape[:2];crop=frame[int(h*.04):int(h*.9),int(w*.2):int(w*.8)]
     im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((230,230));tile=Image.new('RGB',(240,258),'white');tile.paste(im,((240-im.width)//2,24));ImageDraw.Draw(tile).text((5,3),f'{i} / {i/fps:.2f}s',fill='black',font=font);tiles.append(tile);ids.append(i)
    i+=1
   cap.release();assert i==frames and fps>0
   cols=6;sheet=Image.new('RGB',(cols*240,((len(tiles)+cols-1)//cols)*258+35),'#dddddd');draw=ImageDraw.Draw(sheet);draw.text((8,7),r['clip_id']+' | source RGB every4 frames | NO prediction',font=font,fill='black')
   for j,t in enumerate(tiles):sheet.paste(t,((j%cols)*240,(j//cols)*258+35))
   p=D/f"{r['clip_id']}_overview.jpg";p.parent.mkdir(parents=True,exist_ok=True);sheet.save(p,quality=93)
   record.update(overview=str(p),overview_sha256=sha(p),sampled_source_frames=ids,fps=fps,frames=frames,raw_video_rehashed=True,reviewed=False)
  queue.append(record)
 write(D/'REVIEW_QUEUE.json',dict(rows=queue,source_people=list(range(1,10)),public_fit_videos=36,already_episode_reviewed=sum(r['status']=='existing_reviewed_episode_labels' for r in queue),missing_episode_reviews=sum(r['status']=='needs_source_RGB_episode_review' for r in queue),first_batch_people=[2,4],first_batch_visuals=sum('overview' in r for r in queue),selection='Predeclared remaining FIT people by recorded IDs, all four words. Not selected from new DEV scores or identity appearance.',no_generated_labels=True,no_training=True,no_models=True,no_private=True,remaining_rule='After coarse RGB review, extract dense frames around uncertain changes; do not treat four-frame sample gaps as exact start/end truth.'))
 print('Coverage queue',len(queue),'first batch',sum('overview'in r for r in queue),'missing videos',sum(not r['exists'] for r in queue))
if __name__=='__main__':main()
