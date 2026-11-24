from pathlib import Path
import json,collections,statistics
U=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text('utf-8'))
def good(d,t,confidence=False):return bool(d and d['eligible'] and d['prediction']==t and (not confidence or d['confidence']>=.65))
def metrics(rs,mode):
 counts=collections.Counter(total=len(rs))
 for r in rs:
  t=r['truth']['direction'];v=r['manual'] if mode=='manual' else r['automatic'][mode]
  counts['matched']+=mode=='manual' or v['matched']
  for k in ['before','raw_new','after']:
   d=v.get(k);counts[k]+=good(d,t);counts[k+'_confident']+=good(d,t,True)
   counts[k+'_hold']+=bool(d and d['prediction']==1 and d['eligible'])
 return dict(counts)
dirs=read(U/'DIRECTION_RESULTS.json');holds=read(U/'HOLD_RESULTS.json');full=read(U/'FULLPATH_COMPLETE.json')['rows'];comp=read(U/'COMPLETION_RESULTS.json');result=dict(direction={},holds={},fullpath={},completion={},probe={})
for seed in [42,43,44]:
 for group in ['additional_replay','reserved_id12','reserved_id13']:
  kinds=sorted({r['kind'] for r in dirs if r['group']==group})
  for kind in kinds:
   key=f'{seed}/{group}/{kind}';rs=[r for r in dirs if r['seed']==seed and r['group']==group and r['kind']==kind];result['direction'][key]={mode:metrics(rs,mode) for mode in ['manual','baseline','combined']}
   hs=[r for r in holds if r['seed']==seed and r['group']==group and r['kind']==kind];result['holds'][key]=dict(total=len(hs),**{k:sum(good(r[k],1) for r in hs) for k in ['before','raw_new','after']},false_rescues=[dict(clip=r['clip_id'],span=r['truth']['span'],prediction=r['after']['prediction'],confidence=r['after']['confidence']) for r in hs if r['after']['rescue_used']])
 for mode in ['motion_only','combined']:
  for kind in sorted({r['kind'] for r in full}):
   rs=[r for r in full if r['seed']==seed and r['mode']==mode and r['kind']==kind];key=f'{seed}/{mode}/{kind}';result['fullpath'][key]=dict(total=len(rs),before_warnings=sum(r['before']['warning'] for r in rs),interval_warnings=sum(r['interval_only']['warning'] for r in rs),after_warnings=sum(r['after']['warning'] for r in rs),before_similar=sum(r['before']['similar_no_warning'] for r in rs),after_similar=sum(r['after']['similar_no_warning'] for r in rs),changed=[dict(clip=r['clip_id'],before=r['before'],interval_only=r['interval_only'],after=r['after']) for r in rs if r['before']!=r['after']],added_warning=[r['clip_id'] for r in rs if not r['before']['warning'] and r['after']['warning']])
  times=[r['seconds'] for r in full if r['seed']==seed and r['mode']==mode];result['fullpath'][f'{seed}/{mode}/timing']=dict(n=len(times),median_seconds=statistics.median(times),p95_seconds=sorted(times)[int(.95*(len(times)-1))],scope='only added local movement feedback; excludes extraction/frozen feature encoder/TCN/DTW/UI')
 cs=[r for r in comp if r['seed']==seed];result['completion'][str(seed)]={timing:dict(collections.Counter(r['guard']['state'] for r in cs if r['offset_relation']==timing)) for timing in ['early','end_in_range','late','unmatched']}
prs=read(U/'PROBE.json')['rows'];result['probe']=dict(total=len(prs),correct=sum(r['pred']==r['label'] for r in prs),scope='fine supervision held out records03/07; pretraining not held out')
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({k:v for k,v in result.items() if k in ['direction','holds']},ensure_ascii=False,indent=2))
print('PRIMARY FULLPATH',json.dumps({k:v for k,v in result['fullpath'].items() if k.startswith('42/')},ensure_ascii=False,indent=2))
print('COMPLETION',json.dumps(result['completion'],ensure_ascii=False));print('PROBE',result['probe'])
