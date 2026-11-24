from pathlib import Path
import json,statistics
from collections import Counter
U=Path(__file__).resolve().parent;V=U/'proposal_adapter'
def rd(p):return json.loads(Path(p).read_text('utf-8-sig'))
def wr(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n','utf-8')
fr=rd(V/'FULLPATH_COMPLETE.json')['rows'];di=rd(V/'DIAGNOSTIC.json');groups=[];components=[]
for seed in [42,43,44]:
 for mode in ['interval_only','role_only','combined']:
  for kind in sorted({r['kind'] for r in fr}):
   rs=[r for r in fr if (r['seed'],r['mode'],r['kind'])==(seed,mode,kind)];groups.append(dict(seed=seed,mode=mode,kind=kind,n=len(rs),before_warning=sum(r['before']['warning'] for r in rs),after_warning=sum(r['after']['warning'] for r in rs),before_similar=sum(r['before']['similar_no_warning'] for r in rs),after_similar=sum(r['after']['similar_no_warning'] for r in rs),changed=sum(r['before']!=r['after'] for r in rs)))
 for group in ['additional_replay','reserved_id12']:
  for kind in ['normal','whole_reverse']:
   rs=[r for r in di if (r['seed'],r['group'],r['kind'])==(seed,group,kind)]
   for mode in ['baseline','interval_only','role_only','combined','selected_only']:
    count=Counter()
    for r in rs:
     a=r['automatic'][mode];t=r['truth'];d=a['direction'];f=a['family'];ok=bool(d and d['eligible'] and d['prediction']==t['direction']);count['matched']+=a['matched'];count['direction']+=ok;count['both']+=ok and (t.get('family') is None or bool(f and f['pred']==t['family']));count['exact_bounds']+=a['both_boundaries_in_range'];md=r['manual_direction'];count['manual_direction']+=bool(md and md['eligible'] and md['prediction']==t['direction'])
    components.append(dict(seed=seed,group=group,kind=kind,mode=mode,n=len(rs),**count))
hr=rd(V/'HOLD_DIAGNOSTIC.json');holds=[dict(seed=s,group=g,n=len(rs),false_selected=sum(bool(r['false_selected']) for r in rs)) for s in [42,43,44] for g in ['additional_replay','reserved_id12'] if (rs:=[r for r in hr if r['seed']==s and r['group']==g])]
lat={}
for mode in ['interval_only','role_only','combined']:
 v=sorted(r['seconds'] for r in fr if r['mode']==mode);lat[mode]=dict(median_seconds=statistics.median(v),p95_seconds=v[int(.95*(len(v)-1))],scope='New linkage+supplement only, excludes RTMW/frozen encoder/TCN/DTW/web.')
fit=rd(V/'FIT_PROPOSAL_AUDIT.json')['rows'];held=rd(V/'ANNOTATION_HOLDOUT.json')['rows'];actual=dict(parent_correct=sum(r['actual_correct'] for r in fit if r['person'] in [3,7]),n=sum(r['actual_observed'] for r in fit if r['person'] in [3,7]),new_correct=sum(a==b for r in held for a,b,s in zip(r['targets'],r['pred'],r['source']) if s=='actual_proposal'),scope='Fine annotation heldout, encoder already exposed to FIT; not independent unseen-person accuracy.')
result=dict(status='EVALUATED_NO_DEFAULT_REPLACEMENT',groups=groups,components=components,holds=holds,latency=lat,annotation_holdout=actual,common100_encoder_preserved=True)
wr(V/'RESULT.json',result);print(json.dumps(dict(primary=[r for r in components if r['seed']==42],changed=[r for r in groups if r['changed']],holds=holds,annotation_holdout=actual,latency=lat),ensure_ascii=False,indent=2))
