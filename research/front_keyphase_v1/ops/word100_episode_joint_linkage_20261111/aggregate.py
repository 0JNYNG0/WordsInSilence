from pathlib import Path
import json,statistics,hashlib
from collections import Counter
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
def gooddir(d,t):return bool(d and d['eligible'] and d['prediction']==t['direction'])
def goodfam(f,t):return t.get('family') is None or bool(f and f['pred']==t['family'])
cfg=read(U/'FROZEN.json');fr=read(U/'FULLPATH_COMPLETE.json')['rows'];groups=[]
for seed in cfg['seeds']:
 for mode in ['interval_only','role_only','combined']:
  ss=[r for r in fr if r['seed']==seed and r['mode']==mode]
  for kind in sorted({r['kind'] for r in ss}):
   rs=[r for r in ss if r['kind']==kind];groups.append(dict(seed=seed,mode=mode,kind=kind,n=len(rs),before_warning=sum(r['before']['warning'] for r in rs),after_warning=sum(r['after']['warning'] for r in rs),before_similar=sum(r['before']['similar_no_warning'] for r in rs),after_similar=sum(r['after']['similar_no_warning'] for r in rs),changed=sum(r['before']!=r['after'] for r in rs)))
diagnostics=[];causes=[]
for source,path in [('historical',U/'HISTORICAL_DIAGNOSTIC.json'),('additional',U/'ADDITIONAL_DIAGNOSTIC.json')]:
 dd=read(path)
 for seed in cfg['seeds']:
  for kind in sorted({r['kind'] for r in dd}):
   for scope in ['all','four_focus','six_other'] if source=='additional' else ['all']:
    rs=[r for r in dd if r['seed']==seed and r['kind']==kind and (scope=='all' or (r['truth'].get('family') is not None)==(scope=='four_focus'))]
    if not rs:continue
    for mode in ['baseline','interval_only','role_only','combined','selected_only']:
     cs=Counter()
     for r in rs:
      t=r['truth'];a=r['automatic'][mode];cs['matched']+=a['matched'];cs['direction_correct']+=gooddir(a['direction'],t);cs['both_correct']+=gooddir(a['direction'],t) and goodfam(a['family'],t);cs['boundaries_in_range']+=a['both_boundaries_in_range'];cs['manual_direction_correct']+=gooddir(r['manual_direction'],t);cs['manual_both_correct']+=gooddir(r['manual_direction'],t) and goodfam(r['manual_old_family'] if mode in ['baseline','interval_only'] else r['manual_new_family'],t)
     diagnostics.append(dict(source=source,seed=seed,kind=kind,scope=scope,mode=mode,n=len(rs),**cs))
    if scope=='all':
     for r in rs:
      t=r['truth'];a=r['automatic']['combined'];tags=[]
      if not gooddir(r['manual_direction'],t):tags.append('observation_or_direction_judge')
      if t.get('family') is not None and not goodfam(r['manual_new_family'],t):tags.append('action_role')
      if not a['matched'] or not a['both_boundaries_in_range']:tags.append('interval_linkage')
      if a['matched'] and gooddir(r['manual_direction'],t) and not gooddir(a['direction'],t):tags.append('interval_changes_direction')
      if a['matched'] and goodfam(r['manual_new_family'],t) and not goodfam(a['family'],t):tags.append('interval_changes_role')
      if tags:causes.append(dict(source=source,seed=seed,clip_id=r['clip_id'],kind=r['kind'],truth=t,causes=tags))
holds=read(U/'HOLD_DIAGNOSTIC.json');hr=[]
for seed in cfg['seeds']:
 rs=[r for r in holds if r['seed']==seed];hr.append(dict(seed=seed,n=len(rs),observed=sum(r['coverage']>=.8 for r in rs),false_selected=sum(bool(r['false_selected']) for r in rs)))
probe=read(U/'ANNOTATION_HOLDOUT.json');pq=Counter();pp=Counter()
for r in probe['rows']:
 for a,b in zip(r['quality_truth'],r['quality_pred']):pq['n']+=1;pq['correct']+=a==b
 for a,b in zip(r['pair_truth'],r['pair_pred']):pp['n']+=1;pp['correct']+=a==b
lat={}
for mode in ['interval_only','role_only','combined']:
 v=sorted(r['seconds'] for r in fr if r['mode']==mode);lat[mode]=dict(n=len(v),median_seconds=statistics.median(v),p95_seconds=v[int(.95*(len(v)-1))],max_seconds=max(v),scope='New observation/linkage and supplement on cached features; excludes RTMW, frozen feature encoder, TCN/DTW, browser, live recording.')
result=dict(status='EVALUATED_PENDING_INDEPENDENT_ARITHMETIC',data=read(U/'DATA_AUDIT.json'),training=[dict(run=str(s),**{k:v for k,v in read(U/'runs'/str(s)/'COMPLETE.json').items() if k!='history'}) for s in ['annotation_probe',42,43,44]],groups=groups,diagnostics=diagnostics,hold=hr,annotation_probe=dict(quality=dict(pq),pair=dict(pp),scope=probe['scope']),latency=lat,prior_result=str(O/'word100_action_episode_supervision_20261110/RESULT.json'),no_default_change=True,private_data=0,additional_full_feedback=False,not_independent_expert=True)
write(U/'RESULT.json',result);write(U/'JOINT_CAUSES.json',causes)
print(json.dumps(dict(primary_groups=[r for r in groups if r['seed']==42],primary_additional=[r for r in diagnostics if r['seed']==42 and r['source']=='additional' and r['scope']=='all'],latency=lat),ensure_ascii=False,indent=2))
