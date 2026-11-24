from pathlib import Path
import json,hashlib,cv2,numpy as np
from PIL import Image,ImageDraw
U=Path(__file__).resolve().parent;O=U.parent
rd=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={r['clip_id']:r for r in rd(O/'word100_action_feedback_demo_20261108/PREPARED.json')['rows']}
raws={(r['clip_id'],r['kind']):r for r in rd(O/'word100_completion_fullpath_20261110/REPLAY_INPUTS.json')['rows']}
rev={r['clip_id']:r for r in rd(O/'word100_automatic_window_learning_20261110/FULLPATH_COMPLETE.json')['rows'] if r['model']=='learned_core_42' and r['kind']=='reversed_core_auto_order'}
changes=[r for r in rd(U/'CHANGED_FEEDBACK.json')['rows'] if r['mode']=='combined' and r['seed']==42];out=[]
for r in changes:
 clip=r['clip_id'];kind=r['kind'];src=sources[clip];assert sha(src['source'])==src['source_sha256'];d=rd(r['result']);normal=raws[clip,'full_normal'];raw=np.load(normal['path']);ids=raw['frame_id'];ix=np.arange(len(ids))
 if kind=='omit_final_event':ix=np.asarray(raws[clip,kind]['source_indices'])
 elif kind=='reversed_core_auto_order':
  lo,hi=rev[clip]['reversed_source_frame_span'];k=np.flatnonzero((ids>=lo)&(ids<=hi));ix[k]=ix[k[::-1]]
 else:continue
 highlights=sorted(set(j for c in d['cards'] if c['display_status']=='difference' for a,b in c.get('highlight_spans',[]) for j in np.linspace(a,b,4).round().astype(int)))
 observed=sorted(set(np.clip(highlights+[j-3 for j in highlights]+[j+3 for j in highlights],0,len(ix)-1).tolist()))
 if len(observed)>24:observed=np.asarray(observed)[np.linspace(0,len(observed)-1,24).round().astype(int)].tolist()
 canvas=Image.new('RGB',(6*240,((len(observed)+5)//6)*190),'white');dr=ImageDraw.Draw(canvas);cap=cv2.VideoCapture(src['source'])
 for j,f in enumerate(observed):
  sourceframe=int(ids[ix[f]]);cap.set(cv2.CAP_PROP_POS_FRAMES,sourceframe);ok,img=cap.read();assert ok
  h,w=img.shape[:2];img=img[:int(h*.77),int(w*.15):int(w*.85)];im=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));im.thumbnail((238,160));xx=(j%6)*240;yy=(j//6)*190;canvas.paste(im,(xx+(240-im.width)//2,yy+25));dr.text((xx+3,yy+3),f'{clip[7:]} t{f} src{sourceframe}',fill='black')
 cap.release();p=U/'change_review'/f'{clip}_{kind}.jpg';p.parent.mkdir(exist_ok=True);canvas.save(p,quality=92);out.append(dict(clip_id=clip,kind=kind,sheet=str(p),sheet_sha256=sha(p),query_frames=observed,source_frames=[int(ids[ix[f]]) for f in observed],source=src['source'],source_sha256=src['source_sha256']))
(U/'CHANGE_REVIEW_INPUTS.json').write_text(json.dumps(dict(rows=out,scope='Post-score audit of actual source frame order and correction spans, not new training labels'),ensure_ascii=False,indent=2),'utf-8')
print(len(out))
