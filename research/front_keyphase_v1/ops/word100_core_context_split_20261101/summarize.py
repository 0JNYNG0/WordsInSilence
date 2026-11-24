from collections import defaultdict,Counter
import json
import numpy as np
from run import U,OLD,REL,read,write,sha,load
def main():
 rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};by=defaultdict(list);pairs=[];cases=[]
 for row in read(U/'RESULT_MANIFEST.json')['results']:
  assert sha(row['file'])==row['sha256'];d=read(row['file']);old=read(d['old_result']);a,b=d['query'],d['reference'];rawq=load(d['query_array']);original=load(rows[a]['input']);ref=load(rows[b]['input'])
  prev=read(REL/'controls'/(d['name']+'.json')) if d['kind']=='control' else None
  indices=prev['metadata'].get('original_indices',list(range(len(rawq['x'])))) if prev else list(range(len(rawq['x'])))
  olds={s['segment_id']:s for s in old['anchors']};nr={x['segment_id']:x for x in d.get('numerical_raw',[])};ns={x['segment_id']:x for x in d.get('numerical_selected',[])}
  evidence=[]
  if prev:
   oldnum={x['segment_id']:x for x in old['numerical_correspondence']};mean={x['segment_id']:x for x in prev['free_order']['anchors']}
   for s in d['anchors']:
    sid=s['segment_id'];ma=mean[sid];lo,hi=s['reference_core'];observed=s['selected'] is not None and s['selected']['core_supported'];fully=s['state']=='supported_core_candidate'
    evidence.append(dict(segment_id=sid,mean_hit=lo<=indices[ma['query_center']]<=hi,context_hit=oldnum[sid]['center_in_original_core'],raw_core=nr[sid],selected=ns[sid],core_observed=observed,supported_unambiguous=fully,context_complete=bool(s['selected'] and s['selected']['context']['complete']),conflict=s.get('old_context_disjoint',False),tie_resolved=s.get('context_tie_resolved',False),state=s['state']))
   by[d['control']].append(dict(name=d['name'],query=a,role=d['role'],stages=evidence,relations=d['relations']))
  else:pairs.append(dict(query=a,reference=b,role=d['role'],states=dict(Counter(x['state'] for x in d['anchors'])),conflicts=sum(x.get('old_context_disjoint',False) for x in d['anchors']),seconds_incremental=d['seconds_incremental_core_split'],old_seconds_saved=d['old_context_seconds_saved']))
  ui=[]
  for s in d['anchors']:
   prior=olds[s['segment_id']];best=s['selected'];oldspan=prior['best']['query_core'] if prior.get('best') else None
   if best:
    ctx=best['context'];path=ctx['phases']['entry']['path']+best['core_path']+ctx['phases']['exit']['path']
    packed=dict(cost=best['core_cost'],query_core=best['query_core'],query_window=ctx['query_context'],path=path,core_supported=best['core_supported'],context_support={k:v['supported'] for k,v in ctx['phases'].items()},phases=dict(core=best['groups'],**{k:v['groups'] for k,v in ctx['phases'].items()}))
   else:ctx=None;packed=None
   numerical=ns.get(s['segment_id'])
   if numerical:numerical=dict(numerical,center_in_original_core=numerical['center_hit'],path_mean_original_frame_error=numerical['path_mean_error'])
   ui.append(dict(segment_id=s['segment_id'],reference_core=s['reference_core'],reference_context=ctx['reference_context'] if ctx else s['reference_core'],old_center=(oldspan[0]+oldspan[1])//2 if oldspan else None,old_window=oldspan,state=s['state'],best=packed,second_disjoint_gap=None,numerical=numerical,feedback=s['feedback'],reference_guidance=s['reference_guidance'],raw_core=s['raw']['query_core'] if s['raw'] else None,tie_count=len(s['core_ties']),context_tie_resolved=s.get('context_tie_resolved',False),core_context_disagreement=s.get('old_context_disjoint',False),diagnostic_hold_reasons=s.get('diagnostic_hold_reasons',[]),path_is_concatenated_independent_parts=True))
  cases.append(dict(id=d['name'],title=f"{rows[a]['label']} {a} / "+(d.get('control') or f'참고 {b}'),kind=d['kind'],query=a,reference=b,role=d['role'],control=d.get('control'),query_pts=(rawq['timestamp_us']/1e6).tolist(),original_query_pts=(original['timestamp_us']/1e6).tolist(),reference_pts=(ref['timestamp_us']/1e6).tolist(),original_indices=indices,anchors=ui,relations=d['relations'],outside_core_candidates=d['outside_core_candidates']))
 metrics={}
 for name,ds in by.items():
  stages=[s for d in ds for s in d['stages']]
  metrics[name]=dict(cases=len(ds),stages=len(stages),mean_hits=sum(s['mean_hit'] for s in stages),context_hits=sum(s['context_hit'] for s in stages),raw_core_hits=sum(s['raw_core']['center_hit'] for s in stages),selected_hits=sum(s['selected']['center_hit'] for s in stages),raw_to_selected_losses=sum(s['raw_core']['center_hit'] and not s['selected']['center_hit'] for s in stages),raw_to_selected_gains=sum(not s['raw_core']['center_hit'] and s['selected']['center_hit'] for s in stages),supported_hits=sum(s['selected']['center_hit'] and s['supported_unambiguous'] for s in stages),context_complete_hits=sum(s['selected']['center_hit'] and s['supported_unambiguous'] and s['context_complete'] for s in stages),conflicts=sum(s['conflict'] for s in stages),states=dict(Counter(s['state'] for s in stages)),mean_core_recall=float(np.mean([s['selected']['recall'] for s in stages])),mean_core_precision=float(np.mean([s['selected']['precision'] for s in stages])),tie_resolved=sum(s['tie_resolved'] for s in stages),relations=dict(Counter(r['state'] for d in ds for r in d['relations'])))
 times=defaultdict(float);oldtimes=defaultdict(float)
 for p in pairs:
  if p['role']!='fit':times[p['query']]+=p['seconds_incremental'];oldtimes[p['query']]+=p['old_seconds_saved']
 gates=dict(identity_all45=metrics['identity']['selected_hits']==45,slow_all45=metrics['speed_075']['selected_hits']==45,swap_not_worse_than_mean40=metrics['swap_adjacent_core_blocks']['selected_hits']>=40,no_context_override_damage=all(v['raw_to_selected_losses']==0 for v in metrics.values()),right_missing_no_supported=metrics['right_hand_unobserved']['states'].get('supported_core_candidate',0)==0,semantic_correctness_not_tested=True)
 summary=dict(status='COMPLETE_PENDING_VERIFICATION',controls=metrics,gates=gates,person_pairs=len(pairs),development_states=dict(sum((Counter(p['states']) for p in pairs if p['role']!='fit'),Counter())),incremental_seconds_by_dev_query=dict(times),previous_context_seconds_by_dev_query=dict(oldtimes),runtime_note='core-only+anchoredcontext measured now; previouscontextcached; not fullpipeline or combined measuredruntime',word_specific_branches=False,validated_expressions=4,all_word_generalization_verified=False,neural_training=0,production_adopted=False)
 write(U/'SUMMARY.json',summary);write(U/'CONTROL_EVALUATION.json',dict(controls=by));write(U/'PAIR_EVALUATION.json',dict(pairs=pairs));write(U/'REPORT_DATA.json',dict(summary=summary,cases=cases));print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
