"""Explicit public-source-only baseline audit; never read user captures."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;G=HERE.parent;O=G.parent;V=G/'completion_v1';B=O/'basic_demo_feedback_20261115'
sys.path.insert(0,str(V))
from observed_gap import SparseVisible,adapter,torch
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():
 rows=[];seen=set()
 for r in read(B/'INPUTS.json')['rows']:
  if r.get('word')=='WORD1290' or 'WORD1290' in r['clip_id']:
   rows.append(dict(id=r['clip_id'],person=r['person'],role=r['role'],raw=r.get('raw',r.get('raw_path')),video=r.get('video')))
 sys.path.insert(0,str(G));from diagnose_terminal import sources as prior
 for r in prior():rows.append(dict(id=r['clip_id'],person=r['person'],role='known_development',raw=r['raw_path'],video=r.get('video',r.get('path'))))
 for p in (G/'avatar_teaching_20261116/NIA_SL_WORD1290_REAL01/video_only').glob('NIA_SL_WORD1290_REAL01_*.npz'):
  rows.append(dict(id=p.stem,person=1,role='fit_view_diagnostic',raw=str(p),video=str(p.with_suffix('.mp4'))))
 result=[]
 for r in rows:
  if r['id'] in seen:continue
  seen.add(r['id']);assert r['raw'] and 'Temp' not in r['raw'] and 'user_' not in r['raw'];result.append(r)
 return result
def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);HERE.mkdir(exist_ok=True)
 rows=sources();write(HERE/'PUBLIC_INPUTS.json',dict(rows=rows,private_data=False))
 net=SparseVisible();summary=[]
 (HERE/'baseline').mkdir(exist_ok=True)
 for r in rows:
  raw=dict(np.load(r['raw']));q=net.analyze(raw);write(HERE/'baseline'/f'{r["id"]}.json',q)
  ch=adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
  s=np.array(q['relation_states']);m=net.motion.sequence(ch)
  np.savez_compressed(HERE/'baseline'/f'{r["id"]}.npz',features=net.last_features,static=s,motion=m,available=ch['point_available'],body=ch['body_relative_xy'],ts=ch['timestamp_us'])
  item=dict(id=r['id'],role=r['role'],events=len(q['events']),criteria={c['id']:c['status']for c in q['criteria']},seconds=q['seconds'])
  summary.append(item);print(json.dumps(item,ensure_ascii=False),flush=True)
 write(HERE/'BASELINE.json',dict(rows=summary))
if __name__=='__main__':main()
