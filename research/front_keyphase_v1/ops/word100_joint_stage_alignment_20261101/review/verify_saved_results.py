"""Independent source/path arithmetic. No aligner/model import or path search."""
from pathlib import Path
from collections import Counter,defaultdict
import json,hashlib,math,statistics,datetime
import numpy as np
U=Path(__file__).resolve().parent.parent;OPS=U.parent;OLD=OPS/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
failures=[];checks=0
def check(name,v):
 global checks
 checks+=1
 if not bool(v):failures.append(name)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8)
def same(a,b):
 if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 if isinstance(a,(float,int)) and not isinstance(a,bool) and isinstance(b,(float,int)) and not isinstance(b,bool):return near(a,b)
 return a==b
FAM={'shape':['L_shape','R_shape','learned3','L_axis','R_axis'],'relations':['L_location','R_location','L_arm','R_arm','L_to_R_forearm','R_to_L_forearm','hands_relation'],'movement':['L_displacement','R_displacement','L_direction','R_direction']}
tasks=read(U/'TASKS.json');frozen=read(U/'FROZEN.json');manifest=read(U/'RESULT_MANIFEST.json');ev=read(U/'EVALUATION.json')
for p,h in frozen['pins'].items():check('pin:'+p,sha(p)==h)
for p,h in tasks['protected'].items():check('protected:'+p,sha(p)==h)
check('all534_jobs',len(manifest['results'])==534 and len({r['path'] for r in manifest['results']})==534)
check('fixed_scope',len(tasks['pairs'])==36 and len(tasks['controls'])==142 and tasks['seeds']==[42,43,44] and frozen['candidate_count']==1 and frozen['reference_full_frames_per_seed']==4380 and frozen['reference_certain_frames_per_seed']==1080)
check('evaluation_scope',len(ev['pairs'])==216 and len(ev['controls'])==552 and ev['new_training']==0 and ev['new_model_inference']==0 and ev['query_annotations_runtime'] is False)
physical=read(OPS/'word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json');physical={x['alias']:x for x in physical['clips']}
info=read(OLD/'LEARNING_INPUTS.json');rows={r['alias']:r for r in info['rows']};ix={r['alias']:i for i,r in enumerate(info['rows'])}
prob={s:np.load(OLD/'runs'/f'pose_plus_signrep_{s}'/'probabilities.npy',allow_pickle=False,mmap_mode='r') for s in tasks['seeds']}
old3=[info['targets'].index(t) for t in ['R_index_only','R_open_four','L_open_four']]
cache={};controlcache={}
for alias in physical:
 row=rows[alias]
 with np.load(row['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
 a['names']=row['feature_names'];n=len(a['x']);a['prob']={s:np.asarray(prob[s][ix[alias],:n])[:,old3] for s in tasks['seeds']};cache[alias]=a
 check('input_seal:'+alias,sha(row['input'])==row['input_sha256']==frozen['pins'][row['input']])

def source_groups(a,p):
 x=np.asarray(a['x'],float);mask=a['mask']&np.isfinite(x);names=a['names'];n=len(x);g={}
 def cols(*ns,scale=1):
  ids=[names.index(s) for s in ns];m=mask[:,ids];return np.where(m,x[:,ids]/scale,0),m
 for hand,other in [('L','R'),('R','L')]:
  angles=[name for name in names if name.startswith(hand+'_') and name.endswith('_angle') and 'elbow' not in name]
  g[hand+'_shape']=cols(*angles);g[hand+'_axis']=cols(hand+'_palm_axis_x',hand+'_palm_axis_y',scale=2)
  g[hand+'_location']=cols(*[hand+'_'+part+'_'+axis for part in ['hand_wrist','palm_center'] for axis in ['x','y']])
  wrist,wm=cols(hand+'_hand_wrist_x',hand+'_hand_wrist_y');pw,pm=cols(hand+'_pose_wrist_x',hand+'_pose_wrist_y');el,em=cols(hand+'_elbow_x',hand+'_elbow_y')
  vec=pw-el;length=np.sqrt((vec**2).sum(1));ok=pm.all(1)&em.all(1)&(length>1e-8)
  axis=np.zeros_like(vec);axis[ok]=vec[ok]/length[ok,None]/2
  base,bm=cols(hand+'_elbow_x',hand+'_elbow_y',hand+'_elbow_angle')
  g[hand+'_arm']=(np.column_stack([base,axis,np.where(ok,length,0)]),np.column_stack([bm,ok,ok,ok]))
  ow,owm=cols(other+'_pose_wrist_x',other+'_pose_wrist_y');oe,oem=cols(other+'_elbow_x',other+'_elbow_y');v=ow-oe;d=wrist-oe;norm=(v*v).sum(1);good=wm.all(1)&owm.all(1)&oem.all(1)&(norm>1e-8)
  along=np.zeros(n);cross=np.zeros(n);along[good]=(d[good]*v[good]).sum(1)/norm[good];cross[good]=(v[good,0]*d[good,1]-v[good,1]*d[good,0])/norm[good]
  g[hand+'_to_'+other+'_forearm']=(np.column_stack([along,cross]),np.column_stack([good,good]))
  delta=np.zeros_like(wrist);valid=np.zeros_like(wm)
  for i in range(n):
   left=max(0,i-1);right=min(n-1,i+1)
   if right>left and wm[left:right+1].all() and int(a['frame_id'][right])-int(a['frame_id'][left])==right-left:delta[i]=wrist[right]-wrist[left];valid[i]=True
  mag=np.sqrt((delta**2).sum(1));direc=np.zeros_like(delta);nonzero=mag>0;direc[nonzero]=delta[nonzero]/mag[nonzero,None]/2
  g[hand+'_displacement']=(delta,valid.copy());g[hand+'_direction']=(direc,valid.copy())
 support=[]
 for hand in ['R','R','L']:
  fingers=[i for i,nm in enumerate(names) if nm.startswith(hand+'_') and '_angle' in nm and 'elbow' not in nm];w=[names.index(hand+'_hand_wrist_'+axis) for axis in ['x','y']]
  support.append(a['mask'][:,w].all(1)&(a['mask'][:,fingers].sum(1)>=10))
 sm=np.stack(support,1)&np.isfinite(p);g['learned3']=(np.where(sm,p,0),sm)
 g['hands_relation']=cols('hands_wrist_delta_x','hands_wrist_delta_y','hands_wrist_distance')
 return g

def control_array(task):
 name=task['name']
 if name in controlcache:return controlcache[name]
 base=cache[task['alias']]
 if task['array']:
  with np.load(task['array'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
  old=read(task['baseline']);check('control_metadata:'+name,old['metadata']==task['metadata'] and old['array_sha256']==sha(task['array']))
 else:
  a={k:base[k].copy() for k in ['x','mask','frame_id','timestamp_us']};noise=.002*np.sin(np.arange(len(a['x']))*2*np.pi/8)
  for j,nm in enumerate(base['names']):
   if nm in [side+'_'+part+'_'+axis for side in ['L','R'] for part in ['hand_wrist','pose_wrist','elbow','index_tip','palm_center'] for axis in ['x','y']]:a['x'][:,j]+=np.where(a['mask'][:,j],noise,0)
 a['names']=base['names'];ids=np.array(task['metadata'].get('original_indices',list(range(len(a['x'])))),int)
 check('control_sourceindices:'+name,len(ids)==len(a['x']) and (ids>=0).all() and (ids<len(base['x'])).all())
 if 'omitted_original_frames' in task['metadata']:check('fiveframe_omission:'+name,sorted(set(range(len(base['x'])))-set(ids))==task['metadata']['omitted_original_frames'])
 if 'kept_original_ids' in task['metadata']:check('resampling_count:'+name,len(set(ids))==task['metadata']['kept_original_ids'])
 if 'changed_frames' in task['metadata']:
  changed=np.not_equal(a['x'],base['x'])&a['mask']&base['mask'];check('shape_changed_cells:'+name,int(changed.sum())==task['metadata']['actual_observed_changed_cells'])
 a['source_indices']=ids;controlcache[name]=a;return a

job_data={};fullpaths=0;total_frame_scope=0;groupcost_records=0
expected_jobs={(kind,(t['query']+'_'+t['reference'] if kind=='pair' else t['name'])+f'_{seed}') for kind,ls in [('pair',tasks['pairs']),('control',tasks['controls'])] for t in ls for seed in tasks['seeds']}
check('exact_job_inventory',{(r['kind'],r['name']) for r in manifest['results']}==expected_jobs)
for rec in manifest['results']:
 check('result_seal:'+rec['name'],sha(rec['path'])==rec['sha256']);d=read(rec['path']);job_data[rec['path']]=d;t=d['task'];seed=d['seed'];name=d['name']
 check('frozen_result_identity:'+name,d['frozen_sha256']==sha(U/'FROZEN.json') and d['kind']==rec['kind'] and name==rec['name'])
 if d['kind']=='pair':q=cache[t['query']];r=cache[t['reference']];qp=q['prob'][seed];rp=r['prob'][seed]
 else:q=control_array(t);r=cache[t['alias']];qp=r['prob'][seed][q['source_indices']];rp=r['prob'][seed]
 n,m=len(q['x']),len(r['x']);path=d['path'];weights=d['path_weights'];qi=np.array([v[0] for v in path],int);ri=np.array([v[1] for v in path],int)
 scope=bool(path) and set(qi)==set(range(n)) and set(ri)==set(range(m));check('full_native_scope:'+name,scope);fullpaths+=scope;total_frame_scope+=n+m
 check('frame_and_pts_trace:'+name,d['query_frames']==n and d['reference_frames']==m and d['query_frame_ids']==q['frame_id'].tolist() and d['reference_frame_ids']==r['frame_id'].tolist() and d['query_timestamp_us']==q['timestamp_us'].tolist() and d['reference_timestamp_us']==r['timestamp_us'].tolist())
 check('path_endpoints_unique:'+name,path[0]==[0,0] and path[-1]==[n-1,m-1] and len({tuple(v) for v in path})==len(path))
 steps=np.diff(np.asarray(path),axis=0);expectedweights=[2]+[2 if tuple(v)==(1,1) else 1 for v in steps]
 check('path_steps_weights:'+name,all(tuple(v) in [(1,1),(1,0),(0,1)] for v in steps) and weights==expectedweights and sum(weights)==d['path_weight_sum']==n+m)
 run=0;last=None;maxrun=0
 for step in steps:
  v=tuple(step);run=run+1 if v==last and v!=(1,1) else 0 if v==(1,1) else 1;maxrun=max(maxrun,run);last=v
 check('same_axis_run:'+name,maxrun<=2)
 check('runtime_blind_and_no_verdict:'+name,d['word_or_phase_used'] is False and d['query_physical_labels_used'] is False and d['new_model_inference'] is False and d['new_inference']==0 and d['whole_sign_verdict'] is None and d['semantic_order_verified'] is False and d['global_order_forced'] is True)
 check('tie_claim_scope:'+name,d['optimal_path_ambiguous']==(d['optimal_path_count_capped']>1) and d['alternate_paths_fully_enumerated'] is False)
 qg=source_groups(q,qp);rg=source_groups(r,rp);famvals={};w=np.array(weights)
 for family,names in FAM.items():
  costs=[]
  for gname in names:
   x,qm=qg[gname];y,rm=rg[gname];shared=qm[qi]&rm[ri];missing=rm[ri]&~qm[qi];den=rm[ri].sum(1);num=(np.minimum(np.abs(x[qi]-y[ri]),1)*shared).sum(1)+missing.sum(1);values=np.full(len(path),.5);np.divide(num,den,out=values,where=den>0);costs.append(values)
   seen=np.zeros_like(rm)
   for i,j in zip(qi,ri):seen[j]|=qm[i]&rm[j]
   expected={'reference_possible_cells':int(rm.size),'reference_known_cells':int(rm.sum()),'reference_unknown_cells':int(rm.size-rm.sum()),'compared_distinct_reference_cells':int(seen.sum()),'reference_coverage':float(seen.sum()/rm.sum()) if rm.sum() else None,'query_possible_cells':int(qm.size),'query_known_cells':int(qm.sum()),'shared_path_cells':int(shared.sum()),'known_reference_missing_query_path_cells':int(missing.sum()),'absence_is_zero_truth':False}
   check('group_support:'+name+gname,same(expected,d['group_support'][gname]));groupcost_records+=1
  famvals[family]=float((np.mean(costs,axis=0)*w).sum()/(n+m))
 check('family_weighted_cost:'+name,same(famvals,d['family_costs']) and near(sum(famvals.values())/3,d['normalized_cost']) and near(d['normalized_cost']*(n+m),d['weighted_cost_sum']) and d['family_groups']==FAM and all(near(v,1/3) for v in d['family_weights'].values()))

def phase_calculation(qa,ra,paths):
 q=physical[qa];r=physical[ra];states=Counter();events=[]
 for event in r['regions']:
  if event['region_type']!='observed_physical_core':continue
  a,b=event['core'];mapping=defaultdict(list)
  for qi,ri in paths.get(event['source_segment_id'],paths.get('*',[])):
   if a<=ri<=b:mapping[ri].append(qi)
  count=Counter();trace=[]
  for ri in range(a,b+1):
   if not mapping[ri]:count['unmapped']+=1;trace.append([ri,None,'unmapped',1]);continue
   for qi in mapping[ri]:
    hits=[h for h in q['regions'] if h['outer'][0]<=qi<=h['outer'][1]];known=[h for h in hits if h['region_type']=='observed_physical_core' and h['core'][0]<=qi<=h['core'][1]]
    st=('same_observed_kind' if known[0]['kind']==event['kind'] else 'different_observed_kind') if len(known)==1 else 'boundary_or_uncertain' if hits else 'unreviewed_or_uncertain'
    weight=1/len(mapping[ri]);count[st]+=weight;trace.append([ri,qi,st,weight])
  states.update(count);events.append({'event':event['id'],'parent':event['source_segment_id'],'reference_kind':event['kind'],'reference_core':event['core'],'expected':b-a+1,'weighted_states':dict(count),'trace':trace})
 return dict(reference_full_frames=r['native_frame_count'],expected_certain_frames=sum(v['expected'] for v in events),states=dict(states),events=events,query_labels_only_for_audit=True)

summary={};byword={};fullrows=[]
for row in ev['pairs']:
 src=job_data.get(row['source']) or read(row['source']);check('audit_source_sha:'+row['source'],sha(row['source'])==row['source_sha256'])
 paths={a['segment_id']:a['selected']['core_path'] if a.get('selected') else [] for a in src['anchors']} if row['method']=='core_first' else {'*':src['path']}
 calc=phase_calculation(row['query'],row['reference'],paths);check('phase_audit:'+row['source'],same(calc,row['audit']))
 for target,key in [(summary,f"{row['role']}/{row['method']}/{row['seed']}"),(byword,f"{row['role']}/{row['word_id']}/{row['method']}/{row['seed']}")]:
  g=target.setdefault(key,dict(pairs=0,full_reference_frames=0,certain_reference_frames=0,states=Counter(),ambiguous_paths=0));g['pairs']+=1;g['full_reference_frames']+=calc['reference_full_frames'];g['certain_reference_frames']+=calc['expected_certain_frames'];g['states'].update(calc['states']);g['ambiguous_paths']+=int(src.get('optimal_path_ambiguous',False)) if row['method']=='joint' else 0
 for k in ['full_scope','ambiguous','family_costs','group_support']:
  if row['method']=='joint' and k in ['family_costs','group_support']:check('evaluation_copy:'+row['source']+k,same(src[k],row[k]))
 if row['method']=='joint':fullrows.append(row['full_scope'])
for table,actual in [(summary,ev['summary']),(byword,ev['byword'])]:
 for k,g in table.items():g['same_kind_percent']=100*g['states']['same_observed_kind']/g['certain_reference_frames'];g['states']=dict(g['states']);check('aggregate:'+k,same(g,actual[k]))
 check('aggregate_keys',set(table)==set(actual))
for method,seed in [('global_dtw',None),('core_first',None),('relational_global',None)]+[('joint',s) for s in tasks['seeds']]:
 selected=[v for k,v in summary.items() if k.endswith(f'/{method}/{seed}')];check('fixed4380_1080:'+method+str(seed),sum(v['full_reference_frames'] for v in selected)==4380 and sum(v['certain_reference_frames'] for v in selected)==1080 and sum(v['pairs'] for v in selected)==36)

ctasks={c['name']:c for c in tasks['controls']};controlsum={}
for row in ev['controls']:
 task=ctasks[row['name']];source=U/'results'/f"control_{row['name']}_{row['seed']}.json" if row['method']=='joint' else Path(task['baseline']);d=job_data.get(str(source)) or read(source);path=d['path'];n=d['query_frames'];m=d['reference_frames'];ids=task['metadata'].get('original_indices',list(range(n)));missing=sorted(set(range(m))-set(ids));by=defaultdict(list)
 for qi,ri in path:by[ri].append(qi)
 vals=[];exacts=[];present=[];deleted=[]
 for ri in range(m):
  errors=[abs(ids[qi]-ri) for qi in by[ri]];mean=statistics.mean(errors) if errors else 0;vals.append(mean);exacts.append(sum(v==0 for v in errors)/len(errors) if errors else 0)
  if errors:(deleted if ri in missing else present).append(mean)
 calc=dict(reference_frames=m,query_frames=n,expected_reference_frames=m,present_original_ids=len(set(ids)),deleted_original_ids=missing,mean_original_frame_error=statistics.mean(vals) if path else None,exact_original_frame_fraction=statistics.mean(exacts) if path else 0.,present_original_frame_error=statistics.mean(present) if present else None,deleted_frames_mean_error=statistics.mean(deleted) if deleted else None,full_path=bool(path),query_scope_retained=n if path else 0,forced_monotonic_order=True,actual_erroneous_sign_video=False)
 changed=task['metadata'].get('changed_frames',[])
 if changed:calc.update(changed_frames=changed,changed_original_identity_retention=statistics.mean(sum(ids[qi]==ri for qi in by[ri])/len(by[ri]) if by[ri] else 0 for ri in changed),changed_cells=task['metadata']['actual_observed_changed_cells'])
 check('control_metrics:'+row['name']+str(row['seed']),same(calc,row['audit']))
 key=f"{row['control']}/{row['method']}/{row['seed']}";g=controlsum.setdefault(key,dict(cases=0,errors=[],exact=[],fullpaths=0,changed_identity=[],deleted_expected=0));g['cases']+=1;g['errors'].append(calc['mean_original_frame_error']);g['exact'].append(calc['exact_original_frame_fraction']);g['fullpaths']+=calc['full_path'];g['deleted_expected']+=len(missing)
 if changed:g['changed_identity'].append(calc['changed_original_identity_retention'])
for key,g in controlsum.items():g['mean_original_frame_error']=statistics.mean(v for v in g['errors'] if v is not None) if any(v is not None for v in g['errors']) else None;g['exact_original_frame_percent']=statistics.mean(g['exact'])*100;check('control_summary:'+key,same(g,ev['control_summary'][key]))
scores=[summary[f'development/joint/{s}']['same_kind_percent'] for s in tasks['seeds']];baselines=[summary[f'development/{m}/None']['same_kind_percent'] for m in ['global_dtw','relational_global']];reg=[];speed=[]
for word in sorted({p['word_id'] for p in tasks['pairs']}):
 base=max(byword[f'development/{word}/{m}/None']['same_kind_percent'] for m in ['global_dtw','relational_global'])
 for s in tasks['seeds']:reg.append(dict(word=word,seed=s,pp=byword[f'development/{word}/joint/{s}']['same_kind_percent']-base))
for kind in ['speed_075','speed_125']:
 base=controlsum[f'{kind}/saved_relational_global/None']['mean_original_frame_error']
 for s in tasks['seeds']:speed.append(dict(case=kind,seed=s,increase=controlsum[f'{kind}/joint/{s}']['mean_original_frame_error']-base))
gates={'mean_gain_at_least5pp_vs_both':all(statistics.mean(scores)>=x+5 for x in baselines),'every_seed_above_both':all(x>max(baselines) for x in scores),'no_word_regression_over5pp':all(x['pp']>=-5 for x in reg),'normal_speed_error_increase_at_most1frame':all(x['increase']<=1 for x in speed),'all_source_scope_preserved':all(g['fullpaths']==g['cases'] for g in controlsum.values()) and all(fullrows),'identity_error_zero':all(controlsum[f'identity/joint/{s}']['mean_original_frame_error']<1e-8 for s in tasks['seeds'])}
decision=dict(clear_development_improvement=all(gates.values()),gates=gates,dev_scores=scores,baselines=baselines,word_regressions=reg,speed_changes=speed,adoption=False,educational_accuracy=None);check('decision_preregistered',same(decision,ev['decision']))
result=dict(status='PASS_INDEPENDENT_ALL_STORED_RESULTS_AND_EVALUATION' if not failures else 'FAIL_INDEPENDENT_RESULT_AUDIT',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='Codex semantic_data_audit',checks=checks,failure_count=len(failures),failures=failures,frozen_pin_count=len(frozen['pins']),protected_pin_count=len(tasks['protected']),all_candidate_results=len(manifest['results']),full_scope_paths=fullpaths,total_query_plus_reference_frame_scope=total_frame_scope,group_cost_support_records=groupcost_records,pair_evaluation_rows=len(ev['pairs']),control_evaluation_rows=len(ev['controls']),independent_summary=summary,independent_decision=decision,static_code_review={'aligner_no_io_or_label_api':True,'runtime_does_not_read_PHYSICAL':True,'query_control_origin_map_removed_before_align':True,'physical_only_post_output_evaluation':True,'no_model_training_or_inference_invoked':True,'no_word_or_reference_or_seed_selection_after_scores':True},artifact_pins={str(U/n):sha(U/n) for n in ['FROZEN.json','TASKS.json','joint_alignment.py','run.py','evaluate.py','RESULT_MANIFEST.json','EVALUATION.json']},audit_code_sha256=sha(__file__),limits=['No model, feature extractor, cost matrix search, or dynamic-programming solver imported or executed.','Path costs/support were independently recomputed only on every stored path from sealed observations and old probabilities.','Optimality and tied-path count are covered by prior synthetic code review; this audit checks saved path legality and numeric cost, not a new path search.','Same physical kind is not repeated occurrence or educational correctness.','Technical edits operate on saved features and probability source-index rearrangement, not new RGB observations.','Some left-angle controls change zero observed cells; they are no-op cases, not positive error-detection successes.','All source ranges are retained by forced fullscope DTW; omission detection is not thereby established.'],new_model_inference=0,new_training=0,new_path_search=0,old_artifacts_modified=False)
with (U/'review/RESULT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'status':result['status'],'checks':checks,'failures':failures[:30],'decision':decision,'fullpaths':fullpaths,'pins':len(frozen['pins'])}))
