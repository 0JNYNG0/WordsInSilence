from pathlib import Path
import sys,json,time
import numpy as np
D=Path(__file__).resolve().parent;G=D.parent;O=G.parent;N=G/'frontal_scope_recovery_20261117'
sys.path.insert(0,str(N))
from frontal_candidate import FrontalLesson,adapter,torch
read=lambda p:json.loads(Path(p).read_text('utf-8-sig'))
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sources():
 rows=read(N/'INPUTS.json')['rows']
 for r in rows:
  assert r['id'].endswith('_F') and 'Temp' not in r['raw'] and 'user_' not in r['raw']
 return rows
def channel(raw):return adapter.convert(raw['xy_px'],raw['score'],raw['timestamp_us'],raw['frame_id'],int(raw['source_width']),int(raw['source_height']))
def main():
 torch.set_num_threads(2);net=FrontalLesson();out=[]
 for r in sources():
  if r['word']!='WORD1290' or r['person'] not in [1,2]:continue
  raw=dict(np.load(r['raw']));q=net.analyze(raw);ch=channel(raw);s=np.array(q['relation_states']);m=net.motion.sequence(ch)
  z=dict(id=r['id'],events=q['events'],evidence=q.get('movement_evidence'),static=s.tolist(),motion=m.tolist())
  write(D/'probe'/f'{r["id"]}.json',z)
  print(r['id'],json.dumps(q['events']),flush=True)
  print('states',[(i,int(s[i]),int(m[i]))for i in range(len(s))if s[i]>0 or m[i]>0],flush=True)
  out.append(z)
 write(D/'probe/FIT_BASELINE.json',out)
if __name__=='__main__':main()
