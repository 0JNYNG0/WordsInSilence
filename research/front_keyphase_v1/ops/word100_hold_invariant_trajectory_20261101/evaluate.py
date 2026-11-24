"""Fixed geometry diagnostics; no old model score or educational accuracy."""
from common import *
from collections import defaultdict,Counter

def describe(values):
 return dict(n=len(values),minimum=min(values) if values else None,median=float(np.median(values)) if values else None,
  mean=float(np.mean(values)) if values else None,maximum=max(values) if values else None)
def collapsed(values):return [x for i,x in enumerate(values) if not i or x!=values[i-1]]
def main():
 checkpins();t=read(U/'TASKS.json');m=read(U/'RESULT_MANIFEST.json');eps=t['numerical_tolerance']
 rows=[];checks=Counter();failures=[];groups=defaultdict(list)
 def check(key,ok,item):
  checks[key]+=1
  if not ok:failures.append(dict(check=key,id=item))
 for rec in m['results']:
  assert sha(rec['path'])==rec['sha256'];d=read(rec['path']);o=d['comparison'];c=o['candidate'];b=o['baseline']
  support={axis:dict(possible_cells=o[axis]['possible_cells'],observed_cells=o[axis]['observed_cells'],
   possible_frames=o[axis]['possible_frames'],fully_observed_frames=o[axis]['fully_observed_frames']) for axis in ['query','reference']}
  complete=all(x['observed_cells']==x['possible_cells'] for x in support.values())
  check('full_scope',b['full_scope'],d['id']);check('complete_exact_masks',c['complete']==complete,d['id'])
  check('missing_null_not_agreement',complete or(c['distance'] is None and not c['path']),d['id'])
  if complete:check('minmax_le_saved_path_max',c['distance']<=b['complete_link_max']+eps,d['id'])
  row=dict(id=d['id'],kind=d['kind'],query=d['query'],reference=d['reference'],word_id=d['word_id'],role=d['role'],
   parent_id=d['parent']['segment_id'],component=d['component'],control=d.get('control'),
   candidate_distance=c['distance'],baseline_mean=b['mean'],baseline_max=b['max'],baseline_complete_max=b['complete_link_max'],
   complete=complete,optimal_path_ambiguous=c['optimal_path_ambiguous'],support=support,seconds=d['elapsed_seconds'],result=rec['path'],
   semantic_verdict=None,normal_speed_or_hold_error=None)
  if d['kind']=='control':
   edit=d['edit'];mapping=edit['original_indices'];refids=list(range(d['reference_range'][0],d['reference_range'][1]+1))
   same_scope=collapsed(mapping)==refids
   row.update(exact_source_scope_with_only_duplicates=same_scope,
    edited_query_outside_parent=len(edit['relevant_changed_outside_scope']),
    intended_edit_in_parent=bool(edit['intended_changed_reference_frames'] or edit['removed_original_frames_in_reference_scope']),
    removed_original_frames=len(edit['removed_original_frames_in_reference_scope']),
    change_scope_preserved=not edit['relevant_changed_outside_scope'])
   # Same-index reference coupling is used only to check a mathematical bound,
   # never supplied to either comparison algorithm.
   if complete and same_scope:
    q=np.asarray(d['query_plot']['values']);r=np.asarray(d['reference_plot']['values']);ri=np.asarray(mapping)-refids[0]
    bound=float(np.max(np.abs(q-r[ri])))
    row['actual_same_index_coordinate_perturbation_bound']=bound
    if d['control'] in ['identity','speed_075','hold_insert5']:
     check('pure_timing_exact_values',bound==0,d['id'])
     check('pure_timing_distance_zero',c['distance']<=eps,d['id'])
    if d['control']=='small_common_coordinate_drift':
     check('jitter_extracted_geometry_bound',c['distance']<=bound+eps,d['id'])
   if complete:row['positive_geometry_distance']=c['distance']>eps
  rows.append(row);groups[(row['kind'],row['role'],row['component'],row['control'])].append(row)
 summary=[]
 for (kind,role,comp,ctrl),rr in sorted(groups.items(),key=lambda x:str(x[0])):
  cc=[x for x in rr if x['complete']]
  summary.append(dict(kind=kind,role=role,component=comp,control=ctrl,actual_records=len(rr),complete=len(cc),unknown=len(rr)-len(cc),
   old_mean_on_complete=describe([x['baseline_mean'] for x in cc]),old_max_on_complete=describe([x['baseline_max'] for x in cc]),
   new_bottleneck_on_complete=describe([x['candidate_distance'] for x in cc]),
   positive_geometry=sum(x.get('positive_geometry_distance',False) for x in cc),
   exact_source_scope=sum(x.get('exact_source_scope_with_only_duplicates',False) for x in rr),
   exact_source_scope_complete=sum(x.get('exact_source_scope_with_only_duplicates',False) for x in cc),
   changed_parent_scope_incomplete=sum(x.get('edited_query_outside_parent',0)>0 for x in rr),
   timing_ms=describe([x['seconds']*1000 for x in rr])))
 controls=[]
 for name in sorted({x['control'] for x in rows if x['kind']=='control'}):
  rr=[x for x in rows if x['control']==name];cc=[x for x in rr if x['complete']];missing=[x for x in m['missing_component_opportunities'] if x['control']==name]
  eligible=[x for x in cc if x['exact_source_scope_with_only_duplicates']]
  controls.append(dict(control=name,component_denominator=len(rr)+len(missing),computed=len(rr),missing_control=len(missing),
   complete=len(cc),unknown=len(rr)-len(cc),positive_distance=sum(x['candidate_distance']>eps for x in cc),
   zero_distance=sum(x['candidate_distance']<=eps for x in cc),exact_source_scope_complete=len(eligible),
   exact_source_scope_zero=sum(x['candidate_distance']<=eps for x in eligible),
   crop_changed_scope=sum(not x['exact_source_scope_with_only_duplicates'] for x in rr),
   relevant_edit_outside_crop=sum(x['edited_query_outside_parent']>0 for x in rr),
   new_bottleneck=describe([x['candidate_distance'] for x in cc]),old_mean=describe([x['baseline_mean'] for x in cc]),old_max=describe([x['baseline_max'] for x in cc])))
 nat=[x for x in rows if x['kind']=='natural'];cn=[x for x in nat if x['complete']]
 result=dict(status='COMPLETE_FIXED_GEOMETRY_RESEARCH_DIAGNOSTIC',checks=dict(checks),technical_failure_count=len(failures),technical_failures=failures,
  natural=dict(pairs=36,public_clips=16,public_words=4,parent_instances=117,components=585,complete=len(cn),unknown=585-len(cn),
   per_role={role:dict(opportunities=sum(x['role']==role for x in nat),complete=sum(x['role']==role for x in cn)) for role in ['fit','development']},
   mean_summary=describe([x['baseline_mean'] for x in cn]),old_max_summary=describe([x['baseline_max'] for x in cn]),new_summary=describe([x['candidate_distance'] for x in cn])),
  controls=controls,summary=summary,rows=rows,missing=m['missing_component_opportunities'],
  timing=dict(geometry_compute_seconds=m['geometry_compute_seconds'],compute_wall_seconds=m['elapsed_compute_seconds'],
   pure_component_median_ms=m['pure_component_median_seconds']*1000,pure_component_max_ms=m['pure_component_max_seconds']*1000,
   new_control_baseline_seconds=sum(x['elapsed_seconds'] for x in m['new_control_baselines']),end_to_end_user_latency_measured=False),
  decision=dict(technical_checks_pass=not failures,original_global_paths_changed=False,production_adopted=False,
   educational_accuracy=None,recognition_accuracy=None,old_stage_score_recomputed=False,all_words_generalization_proven=False,
   observed_geometry_coverage_not_semantic_correctness=True,no_new_training=True),
  limits=['Minmax <= same-path max is mathematical optimization, not correspondence accuracy.',
   'Means and maxima are different aggregations, not before/after accuracy.',
   'Controls rearrange saved features, not new RGB performances or MediaPipe reruns.',
   'speed125 removes samples; straight deletion/symmetric reversal may be indistinguishable from normal variation.',
   'Parent crops still depend on the existing baseline. Edited frames outside a crop are retained in the ledger, not counted as correct agreement.',
   'Complete observations and small geometry distance do not authorize educational correction.',
   'Different references are retained individually. No post-score threshold, reference or word selection.',
   'Geometry computed once because inputs are seed independent; existing3seed shape/readout records remain unchanged.'])
 new(U/'EVALUATION.json',result)
 print('natural',result['natural']);print('technical_failures',len(failures))
 for x in controls:print({k:x[k] for k in ['control','component_denominator','complete','unknown','zero_distance','positive_distance','exact_source_scope_complete','relevant_edit_outside_crop']})

if __name__=='__main__':main()
