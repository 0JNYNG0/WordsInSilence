from pathlib import Path
import json,collections,statistics
U=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text('utf-8'))
def good(d,t):return bool(d and d['eligible'] and d['prediction']==t)
dr=read(U/'DIRECTION_RESULTS.json');ev=read(U/'EVENT_RESULTS.json');fr=read(U/'FULLPATH_COMPLETE.json')['rows'];hold=read(U/'HOLD_RESULTS.json');result=dict(direction={},events={},feedback={},timing={},holds={})
for seed in [42,43,44]:
 for group in sorted({r['group'] for r in dr}):
  for kind in sorted({r['kind'] for r in dr if r['group']==group}):
   key=f'{seed}/{group}/{kind}';rs=[r for r in dr if r['seed']==seed and r['group']==group and r['kind']==kind];d=dict(total=len(rs),manual=sum(good(r['manual']['after'],r['truth']['direction']) for r in rs))
   for name,source,decision in [('baseline','baseline','before'),('motion_only','baseline','after'),('localization_only','union','before'),('combined','union','after'),('new_standalone','localized','after')]:
    d[name]=sum(good(r['automatic'][source].get(decision),r['truth']['direction']) for r in rs);d[name+'_matched']=sum(r['automatic'][source]['matched'] for r in rs)
   result['direction'][key]=d;es=[r for r in ev if r['seed']==seed and r['group']==group and r['kind']==kind];result['events'][key]={k:sum(r[k] for r in es) for k in ['baseline_events','new_events','union_events','truth_events','baseline_matched','new_matched','union_matched','known_nonmotion_baseline','known_nonmotion_new']}
   hs=[r for r in hold if r['seed']==seed and r['group']==group and r['kind']==kind];result['holds'][key]={k:sum(r[k] for r in hs) for k in ['frames','observed','predicted_moving','predicted_hold','predicted_background']}
 for mode in ['localization_only','combined']:
  for kind in sorted({r['kind'] for r in fr}):
   key=f'{seed}/{mode}/{kind}';rs=[r for r in fr if r['seed']==seed and r['mode']==mode and r['kind']==kind];d=dict(total=len(rs))
   for src in ['baseline','motion_only','after']:
    d[src+'_warnings']=sum(r[src]['warning'] for r in rs);d[src+'_similar']=sum(r[src]['similar_no_warning'] for r in rs)
   d['new_warning_clips']=[r['clip_id'] for r in rs if r['after']['warning'] and not r['baseline']['warning']];d['changed']=[dict(clip_id=r['clip_id'],before=r['baseline'],after=r['after']) for r in rs if r['baseline']!=r['after']];result['feedback'][key]=d
  ts=[r['localizer_seconds']+r['binding_seconds'] for r in fr if r['seed']==seed and r['mode']==mode];result['timing'][f'{seed}/{mode}']=dict(n=len(ts),median_seconds=statistics.median(ts),p95_seconds=sorted(ts)[int(.95*(len(ts)-1))],scope='Added localizer + guide binding only, excludes RTMW/100word encoder/TCN/DTW/UI')
(U/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk.startswith('42/')} for k,v in result.items() if k!='holds'},ensure_ascii=False,indent=2))
