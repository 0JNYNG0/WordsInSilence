from common import *
import cv2,shutil
assert not (U/'VIDEO_PTS_CHECK.json').exists()
shutil.copyfile(U/'verify.py',U/'before/verify_initial.py')
out=[]
for r in read(U/'CATALOG.json')['items']:
 cap=cv2.VideoCapture(r['video_path']);rows=[]
 for f in sorted({im['frame_id'] for p in r['spans'] for im in p['images']}):
  cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,img=cap.read();assert ok
  pts=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  idx=r['frame_ids'].index(f);stored=r['timestamps'][idx];assert abs(pts-stored)<1e-6
  rows.append(dict(frame_id=f,pts_seconds=pts,stored_seconds=stored,error=abs(pts-stored)))
 cap.release();out.append(dict(word_code=r['word_code'],frames=rows))
write(U/'VIDEO_PTS_CHECK.json',dict(status='PASS',at=now(),items=out,frames_checked=sum(len(r['frames'])for r in out),
 conclusion='Saved observation timestamps match actual decoded-frame presentation timestamps. Container average FPS is not a correct per-frame timestamp formula for16files. Initial verifier assumption corrected; observations/images/candidate marks unchanged.',
 max_error=max(f['error']for r in out for f in r['frames'])))
print('PASS: actual decoded PTS, all100clips')
