"""Capture-only public replay through the running preview endpoint; no feedback inference."""
from pathlib import Path
import json,hashlib,time,urllib.request,cv2
import numpy as np
A=Path(__file__).resolve().parent;U=A.parent;O=U.parent;BASE='http://127.0.0.1:8812'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def get(url):return json.load(urllib.request.urlopen(BASE+url,timeout=10))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
state=get('/api/state');assert state['ready'] and not state['busy']
rows=[r for r in read(O/'word100_learned_guide_feedback_20261102/INPUT_MANIFEST.json')['rows'] if r['person']==1]
display=read(U/'reference_data.json');results=[]
for row in rows:
 assert row['role']=='fit' and sha(row['source'])==row['source_sha256']
 cap=cv2.VideoCapture(row['source']);fps=cap.get(cv2.CAP_PROP_FPS);frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 variants=[]
 for variant in ['public_viewport','public_viewport_short_blackout']:
  observed=[];last_frame=None
  # A common fixed upper-body viewport simulates the desk camera boundary only in this test.
  # The stored source, full-frame analysis pipeline and feedback input remain untouched.
  for sec in np.arange(0,(frames-1)/fps+1.6,.2):
   fid=min(frames-1,int(round(sec*fps)));cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,im=cap.read();assert ok
   im=im[:int(im.shape[0]*.75),:]
   im=cv2.resize(im,(640,round(im.shape[0]*640/im.shape[1])))
   # 400ms full blank is a controlled sensor-gap probe, not a human-occlusion label.
   if variant.endswith('blackout') and 1.2-1e-6<=sec<1.6-1e-6:im=np.zeros_like(im)
   ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,65]);assert ok
   req=urllib.request.Request(BASE+'/api/preview',data=b.tobytes(),headers={'X-Session-Token':state['token'],'Content-Type':'image/jpeg'})
   t=time.perf_counter()
   with urllib.request.urlopen(req,timeout=10) as response:p=json.load(response)
   observed.append(dict(t_ms=round(sec*1000),source_frame=fid,preview=p,request_seconds=time.perf_counter()-t))
  variants.append(dict(kind=variant,observations=observed))
 cap.release()
 d=display[row['word_id']];end=max(g['frames'][1] for g in d['guides']);main_end_ms=round(d['canonical_times'][end]*1000)
 results.append(dict(word=row['word_id'],source_sha256=row['source_sha256'],person_id=1,main_end_ms=main_end_ms,variants=variants))
 print(row['word_id'],[(v['kind'],len(v['observations']),sum(bool(x['preview']['hands']) for x in v['observations'])) for v in variants],flush=True)
 out=dict(scope='Public capture detector only. Common 75% height viewport + 1.6sec last-frame tail; synthetic blackout400ms. Not full-frame feedback or real webcam accuracy.',rows=results,private_access=0,feedback_inference=0,training=0)
 (A/'PUBLIC_PREVIEW.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
