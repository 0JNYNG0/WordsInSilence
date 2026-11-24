from common import *
import time,os,traceback

def freeze():
 t=read(U/'TASKS.json');pins=t['pins'].copy()
 for f in ['PLAN_KO.txt','AUTHORIZATION.json','TASKS.json','common.py','prepare.py','run.py','evaluate.py','trajectory.py','selftest.py','SELFTEST.json',
  'review/SOURCE_INVENTORY.json','review/EVALUATION_PLAN.json','review/METHOD_PLAN.json','review/PREEXECUTION_REVIEW.json']:
  pins[str(U/f)]=sha(U/f)
 assert read(U/'SELFTEST.json')['status'].startswith('PASS')
 new(U/'FROZEN.json',dict(status='FROZEN_ONE_ADDITIVE_TRAJECTORY_CANDIDATE',pins=pins,protected=t['protected'],
  public_clips=16,natural_pairs=36,parent_instances=117,components=5,control_case_denominator=160,
  old_control_paths_reused=94,new_control_baselines=64,geometry_seed_independent=True,
  old_model_scores_recalculated=False,old_global_paths_changed=False,html_explicitly_authorized=True,
  training=0,model_inference=0))
 print('frozen',len(pins),flush=True)

def edit_info(c,q,base,qr,rr):
 n=len(q['x']);bn=len(base['x']);meta=c['metadata'];mapping=np.asarray(meta.get('original_indices',list(range(n))),int)
 assert len(mapping)==n and mapping.min()>=0 and mapping.max()<bn
 source_values=base['x'][mapping];source_mask=base['mask'][mapping]
 point_value_changed=np.any((q['x']!=source_values)&q['mask']&source_mask,axis=1)
 mask_changed=np.any(q['mask']!=source_mask,axis=1)
 changed=set(meta.get('changed_query_indices',[]))|set(meta.get('changed_frames',[]))|set(np.flatnonzero(point_value_changed|mask_changed).tolist())
 if c['control'] in ['swap_adjacent_core_blocks','reverse_first_parent']:
  changed.update(np.flatnonzero(mapping!=np.arange(n)).tolist())
 if c['control']=='hold_insert5':changed.update(meta['changed_query_indices'])
 removed=set(meta.get('omitted_original_frames',[]))|set(meta.get('removed_original_frames',[]))
 lost=set(range(bn))-set(mapping.tolist());removed.update(lost)
 in_q={i for i in changed if qr[0]<=i<=qr[1]};expected={int(mapping[i]) for i in changed if rr[0]<=mapping[i]<=rr[1]}
 missing_expected={i for i in removed if rr[0]<=i<=rr[1]}
 expected_query={i for i in changed if rr[0]<=mapping[i]<=rr[1]}
 return dict(original_indices=mapping[qr[0]:qr[1]+1].tolist(),changed_query_indices=sorted(changed),
  changed_query_inside_scope=sorted(in_q),changed_query_outside_scope=sorted(changed-in_q),
  relevant_changed_query_indices=sorted(expected_query),relevant_changed_outside_scope=sorted(expected_query-in_q),
  intended_changed_reference_frames=sorted(expected),removed_original_frames_in_reference_scope=sorted(missing_expected),
  source_point_values_changed=int(point_value_changed.sum()),source_masks_changed=int(mask_changed.sum()),
  original_index_is_evaluation_only=True,actual_linguistic_error=None)

def execute():
 checkpins();import trajectory
 tasks=read(U/'TASKS.json');new(U/'RUNNING.json',dict(pid=os.getpid(),started=time.time()))
 update(U/'CURRENT_EXECUTION.json',dict(status='RUNNING_FIXED_TRAJECTORY_COMPARISON',active=os.getpid()))
 raw={a:array(r['input']) for a,r in tasks['rows'].items()};geos={a:components(r) for a,r in raw.items()}
 records=[];baselines=[];timing=[];fullstart=time.perf_counter()
 def one(name,kind,qa,ra,role,word,parent,path,qgeo,rgeo,source_q,source_r,extra):
  local,qr,rr,original=crop(path,parent['core'])
  for component in COMPONENTS:
   q=subset(qgeo[component],*qr);r=subset(rgeo[component],*rr)
   st=time.perf_counter();out=trajectory.compare(q,r,local.tolist());elapsed=time.perf_counter()-st
   assert out['baseline']['full_scope']
   for ax,a in [('query',q),('reference',r)]:
    assert out[ax]['possible_frames']==len(a['frame_id']) and out[ax]['possible_cells']==a['mask'].size
   item=dict(id=name+'__'+component,kind=kind,query=qa,reference=ra,role=role,word_id=word,
    parent=parent,component=component,query_range=qr,reference_range=rr,baseline_full_path=original.tolist(),
    source_query=source_q,source_reference=source_r,comparison=out,elapsed_seconds=elapsed,
    query_plot=dict(values=q['values'].tolist(),mask=q['mask'].tolist()),reference_plot=dict(values=r['values'].tolist(),mask=r['mask'].tolist()),
    frozen_sha256=sha(U/'FROZEN.json'),**extra)
   p=U/'results'/(item['id']+'.json.gz');new(p,item)
   records.append(dict(id=item['id'],path=str(p),sha256=sha(p),kind=kind,query=qa,reference=ra,role=role,word_id=word,component=component,
    control=extra.get('control'),parent_id=parent['segment_id']))
   timing.append(elapsed)
  update(U/'STATUS.json',dict(geometry_records=len(records),last=name,new_control_baselines=len(baselines)))
 for event in tasks['natural']:
  qa,ra=event['query'],event['reference'];path=read(event['baseline'])['path'];parent=event['parent']
  one('pair_'+qa+'_'+ra+'_'+parent['segment_id'],'natural',qa,ra,event['role'],event['word_id'],parent,path,geos[qa],geos[ra],
   tasks['rows'][qa]['input'],tasks['rows'][ra]['input'],dict(baseline_origin='saved_global',baseline_source=event['baseline'],
   seed_aliases=[42,43,44],source_readouts=event['source_readouts'],original_global_path_replaced=False))
 print('natural',len(records),flush=True)
 for ci,c in enumerate(tasks['controls']):
  qa=c['alias'];qraw=array(c['array']);qgeo=components(qraw)
  if c['baseline']:
   path=read(c['baseline'])['path'];bs=c['baseline']
  else:
   st=time.perf_counter();path=legacy.dtw(legacy.alignment_cost(legacy.features(qraw),legacy.features(raw[qa]))).tolist();elapsed=time.perf_counter()-st
   p=U/'new_control_baselines'/(c['id']+'.json');new(p,dict(path=path,control=c['control'],source_query=c['array'],source_reference=tasks['rows'][qa]['input'],
    elapsed_seconds=elapsed,algorithm=str(REL/'relational.py'),algorithm_sha256=sha(REL/'relational.py'),new_input_only=True,
    true_index_used=False,frozen_sha256=sha(U/'FROZEN.json')))
   bs=str(p);baselines.append(dict(path=bs,sha256=sha(p),elapsed_seconds=elapsed))
  for parent in c['parents']:
   _,qr,rr,_=crop(path,parent['core']);edit=edit_info(c,qraw,raw[qa],qr,rr)
   one('control_'+c['id']+'_'+parent['segment_id'],'control',qa,qa,c['role'],c['word_id'],parent,path,qgeo,geos[qa],c['array'],tasks['rows'][qa]['input'],
    dict(control=c['control'],control_id=c['id'],baseline_origin=c['baseline_origin'],baseline_source=bs,
    edit=edit,seed_aliases=[],technical_control_not_real_ksl_error=True))
  if ci%20==0 or ci+1==len(tasks['controls']):print('controls',ci+1,len(tasks['controls']),'geometry',len(records),flush=True)
 missing=[]
 for c in tasks['missing_controls']:
  for parent in c['parents']:
   for component in COMPONENTS:missing.append(dict(alias=c['alias'],role=c['role'],control=c['control'],parent_id=parent['segment_id'],component=component,status='NO_SAVED_CONTROL'))
 assert len(baselines)==64 and len(records)==2825 and len(missing)==10
 checkpins()
 new(U/'RESULT_MANIFEST.json',dict(status='COMPLETE_PENDING_VERIFICATION',results=records,missing_component_opportunities=missing,
  new_control_baselines=baselines,geometry_record_count=len(records),missing_control_cases=tasks['missing_controls'],
  elapsed_compute_seconds=time.perf_counter()-fullstart,geometry_compute_seconds=sum(timing),
  pure_component_median_seconds=float(np.median(timing)),pure_component_max_seconds=max(timing),
  end_to_end_user_latency_measured=False,new_model_inference=0,new_training=0,old_global_paths_replaced=0))
 (U/'RUNNING.json').unlink();update(U/'CURRENT_EXECUTION.json',dict(status='COMPARISON_COMPLETE_PENDING_VERIFICATION',active=None))

if __name__=='__main__':
 try:{'freeze':freeze,'run':execute}[sys.argv[1]]()
 except Exception:
  p=U/'FIRST_EXECUTION_FAILURE.json'
  if not p.exists():new(p,dict(error=traceback.format_exc(),argv=sys.argv,first_failure_preserved=True))
  raise
