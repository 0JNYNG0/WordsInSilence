"""Connection regression, not a new generalization/accuracy experiment."""
from pathlib import Path
import json,time,copy
import numpy as np
import engine
U=Path(__file__).resolve().parent;P=U.parent/'word100_public_g4_20261114'
read=lambda p:json.loads(p.read_text('utf-8-sig'))
def plain(x):return json.loads(json.dumps(x,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
def semantic(r):
 # Floating-point scores/probabilities remain in both saved evidence files;
 # this gate checks all decisions, selected spans and localization, not logits.
 if isinstance(r,dict):return {k:semantic(v) for k,v in r.items() if k not in ['clip_id','kind','runtime','score','probability','probabilities']}
 if isinstance(r,list):
  # Evidence bags are keyed by card; their JSON storage order is not chronology.
  if r and all(isinstance(v,dict) and 'card' in v for v in r):r=sorted(r,key=lambda v:v['card'])
  return [semantic(v) for v in r]
 if isinstance(r,float):return round(r,5)
 return r
engine.torch.set_num_threads(4);engine.torch.set_num_interop_threads(1)
models=engine.initialize();reports=[];(U/'equivalence').mkdir(exist_ok=True)
rows=read(P/'CONFLICT_REGRESSION.json')['rows'];sources={r['clip_id']:r for r in read(P/'EXTRACTED.json')['rows']}
labels=read(P/'LABELS_FROZEN.json')['actions']
for group in ['public_normal','public_continuous','public_splice']:
 for expected in rows[group]:
  if expected['condition']!='current_g3' or expected['kind']=='reversed_reviewed_main':continue
  cid=expected['clip_id'];kind=expected['kind'];sr=sources[cid];path=U/'equivalence'/f'{cid}__{kind}.json'
  if path.exists():
   cached=read(path);actual=cached['result'];a,b=semantic(actual),semantic(expected)
   ck=dict(cached['check'],equal=a==b,different_keys=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)])
   reports.append(ck);continue
  ch=dict(np.load(sr['channel_path']));raw=dict(np.load(sr['raw_path']));x=ch['tcn280'];n=len(x);ids=np.arange(n)
  if kind=='prefix_before_goal':n=next(a['partial_end'] for a in labels if a['clip_id']==cid and a['card']=='close_chest')+1;ids=np.arange(n)
  if kind=='continuous_full_reverse':ids=ids[::-1].copy()
  rate={'speed075':.75,'speed125':1.25}.get(kind,1)
  ch['tcn280']=x[ids].copy();ch['timestamp_us']=np.round(ch['timestamp_us'][:n]/rate).astype(np.int64);ch['frame_id']=ch['frame_id'][ids];ch['point_available']=ch['point_available'][ids];raw['xy_px']=raw['xy_px'][ids]
  if kind=='all_hands_missing':ch['tcn280'][:,18:186]=0;ch['tcn280'][:,195:279]=0
  actual=plain(engine.infer(models,ch,raw,sr['word']))
  a,b=semantic(actual),semantic(expected);equal=a==b
  check=dict(clip_id=cid,kind=kind,equal=equal,seconds=actual['runtime']['seconds'],windows=actual['runtime']['candidate_windows'],different_keys=[k for k in set(a)|set(b) if a.get(k)!=b.get(k)])
  path.write_text(json.dumps(dict(check=check,result=actual),ensure_ascii=False,indent=2),encoding='utf-8');reports.append(check)
  print(cid,kind,equal,round(check['seconds'],3),check['different_keys'],flush=True)
out=dict(passed=all(r['equal'] for r in reports),rows=reports,scope='Frozen G4 replay equality; exposed public sources; not unseen accuracy',training=False)
(U/'EQUIVALENCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
assert out['passed'], 'Saved output decisions differ; inspect before live connection'
