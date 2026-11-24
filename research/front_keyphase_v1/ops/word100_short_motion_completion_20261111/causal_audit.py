from pathlib import Path
import json,collections
U=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text('utf-8'))
def good(d,t):return bool(d and d['eligible'] and d['prediction']==t)
ds=read(U/'DIRECTION_RESULTS.json');hs=read(U/'HOLD_RESULTS.json');cs=read(U/'COMPLETION_RESULTS.json');out={}
for group in ['additional_replay','reserved_id12','reserved_id13']:
 rows=[r for r in ds if r['seed']==42 and r['group']==group and r['kind']=='normal'];counts=collections.Counter();cases=[]
 for r in rows:
  t=r['truth']['direction'];manual=good(r['manual']['after'],t);a=r['automatic']['combined'];auto=good(a.get('after'),t);state='both_correct' if manual and auto else 'interval_or_context_gap' if manual else 'both_unresolved' if not auto else 'auto_only_correct';counts[state]+=1;cases.append(dict(clip=r['clip_id'],truth=r['truth'],manual_correct=manual,auto_correct=auto,matched=a['matched'],cause=state))
 out[group]=dict(total=len(rows),counts=dict(counts),cases=cases)
out['hold_failure_review']=dict(cases=[dict(clip=r['clip_id'],kind=r['kind'],span=r['truth']['span'],trajectory=r['raw_new']['geometry']['trajectory']) for r in hs if r['seed']==42 and r['after']['rescue_used']],interpretation='Both frozen hold labels show small endpoint drift in the extracted trajectory; RGB review does not establish a distinct extra sign action. Could include hold-boundary ambiguity and tracking drift. Kept as failures, no postscore relabeling.',limitation='Cannot assign all drift to extraction from 2D plots alone. Not expert review.')
out['completion']=dict(early_selected=[dict(clip=r['clip_id'],kind=r['kind'],span=[r['event']['a'],r['event']['b']],state=r['guard']['state']) for r in cs if r['seed']==42 and r['offset_relation']=='early'],interpretation='Local tail check did not confidently mark continues_same_direction on any of six early-ending matched proposals. It is not an accurate semantic completion detector. semantic_complete=False is a contract, not an empirical success score.')
out['next_design_evidence']=dict(local_classifier_gain='manual9:6to8; automatic9:4to4 relative to prior interval candidate',interval_bottleneck='4 of9 manual correct but auto fails; 1 both unresolved',no_threshold_tuning=True,no_adoption=True)
(U/'CAUSAL_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf-8');print(json.dumps({k:v['counts'] for k,v in out.items() if 'counts' in v},ensure_ascii=False))
