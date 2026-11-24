"""Read saved results only. No model, candidate import, cost-matrix search or DP.

Run only after root confirms the 822-path execution and evaluation completed.
Independent feature definitions reuse this reviewer's earlier arithmetic audit.
"""
from pathlib import Path
from collections import Counter,defaultdict
import ast,json,hashlib,math,statistics,datetime
import numpy as np
U=Path(__file__).resolve().parent.parent;OPS=U.parent;OLD=OPS/'word100_joint_stage_alignment_20261101';STAGE=OPS/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0;failures=[]
def check(label,value):
 global checks
 checks+=1
 if not bool(value):failures.append(label)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8)
def equal(a,b):
 if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):return near(a,b)
 return a==b
def arr_eq(a,b):return np.asarray(a).shape==np.asarray(b).shape and np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-9,atol=1e-8,equal_nan=True)
FAM={'shape':['L_shape','R_shape','learned3','L_axis','R_axis'],'relations':['L_location','R_location','L_arm','R_arm','L_to_R_forearm','R_to_L_forearm','hands_relation'],'movement':['L_displacement','R_displacement','L_direction','R_direction']}

def axis_calc(x,xm,y,ym,own_indices,other_indices):
 n,d=x.shape;matched=np.zeros((n,d),int);missing=np.zeros((n,d),int);aligned=np.zeros(n,int);sums=np.zeros((n,d));peak=np.full((n,d),-np.inf)
 shared=xm[own_indices]&ym[other_indices];absdiff=np.minimum(np.abs(x[own_indices]-y[other_indices]),1)
 np.add.at(aligned,own_indices,1);np.add.at(matched,own_indices,shared);np.add.at(missing,own_indices,xm[own_indices]&~ym[other_indices]);np.add.at(sums,own_indices,np.where(shared,absdiff,0))
 np.maximum.at(peak,own_indices,np.where(shared,absdiff,-np.inf));support=matched>0;means=np.full((n,d),np.nan);np.divide(sums,matched,out=means,where=support);peak[~support]=np.nan;uncompared=xm&~support;mixed=support&(missing>0)
 status=np.full((n,d),'NOT_COMPARABLE',object);status[support]='COMPARED';status[mixed]='PARTIAL_SHARED_SUPPORT'
 reason=np.full((n,d),'OWN_UNOBSERVED',object);reason[xm]='COUNTERPART_UNOBSERVED';reason[xm&(aligned[:,None]==0)]='NO_ALIGNED_COUNTERPART';reason[support]='SHARED_OBSERVATION';reason[mixed]='SHARED_AND_MISSING_COUNTERPART_LINKS'
 counts=dict(possible=n*d,own_observed=int(xm.sum()),own_unobserved=int((~xm).sum()),matched_support=int(support.sum()),observed_but_not_comparable=int(uncompared.sum()),counterpart_missing_link_cells=int(missing.sum()),mixed_support_cells=int(mixed.sum()))
 idx={'own_unobserved':[np.flatnonzero(~xm[:,j]).tolist() for j in range(d)],'counterpart_unobserved_any_link':[np.flatnonzero(missing[:,j]>0).tolist() for j in range(d)],'observed_without_shared_counterpart':[np.flatnonzero(uncompared[:,j]).tolist() for j in range(d)],'not_comparable':[np.flatnonzero(~support[:,j]).tolist() for j in range(d)]}
 return dict(own=xm,support=support,means=means,peaks=peak,matched=matched,missing=missing,aligned=aligned,uncompared=uncompared,mixed=mixed,status=status,reason=reason,counts=counts,indices=idx)

def local(a,frames):
 own=a['own'][frames];support=a['support'][frames];means=a['means'][frames];peaks=a['peaks'][frames]
 return dict(possible=int(own.size),own_observed=int(own.sum()),shared=int(support.sum()),own_known_uncompared=int((own&~support).sum()),own_unknown=int((~own).sum()),mixed_support_cells=int(a['mixed'][frames].sum()),missing_counterpart_link_cells=int(a['missing'][frames].sum()),residual_mean_conditions_on_shared_links=True,shared_does_not_imply_all_links_observed=True,mean_residual_sum=float(np.nansum(means)),peak_residual_sum=float(np.nansum(peaks)),supported_mean=float(np.nanmean(means)) if support.any() else None,supported_peak_mean=float(np.nanmean(peaks)) if support.any() else None)

def control_metrics(d,t):
 path=d['path'];n,m=d['query_frames'],d['reference_frames'];ids=t['metadata'].get('original_indices',list(range(n)));by=defaultdict(list)
 for q,r in path:by[r].append(q)
 absent=sorted(set(range(m))-set(ids));allerr=[];exact=[];present=[];deleted=[]
 for r in range(m):
  errors=[abs(ids[q]-r) for q in by[r]];v=statistics.mean(errors) if errors else 0;allerr.append(v);exact.append(sum(x==0 for x in errors)/len(errors) if errors else 0)
  if errors:(deleted if r in absent else present).append(v)
 out=dict(reference_frames=m,query_frames=n,expected_reference_frames=m,present_original_ids=len(set(ids)),deleted_original_ids=absent,mean_original_frame_error=statistics.mean(allerr) if path else None,exact_original_frame_fraction=statistics.mean(exact) if path else 0.,present_original_frame_error=statistics.mean(present) if present else None,deleted_frames_mean_error=statistics.mean(deleted) if deleted else None,full_path=bool(path),query_scope_retained=n if path else 0,forced_monotonic_order=True,actual_erroneous_sign_video=False)
 changed=t['metadata'].get('changed_frames',[])
 if changed:out.update(changed_frames=changed,changed_original_identity_retention=statistics.mean(sum(ids[q]==r for q in by[r])/len(by[r]) if by[r] else 0 for r in changed),changed_cells=t['metadata']['actual_observed_changed_cells'])
 return out

def main():
 for f in ['FROZEN.json','RESULT_MANIFEST.json','EVALUATION.json','AXIS_ACCOUNTING.json']:
  if not (U/f).exists():raise RuntimeError('Await completed root execution/evaluation: '+f)
 t=read(U/'TASKS.json');frozen=read(U/'FROZEN.json');manifest=read(U/'RESULT_MANIFEST.json');ev=read(U/'EVALUATION.json');account=read(U/'AXIS_ACCOUNTING.json');prior=read(OLD/'EVALUATION.json');oldmanifest=read(OLD/'RESULT_MANIFEST.json')
 for section in ['pins','protected']:
  for p,h in frozen[section].items():check(section+':'+p,sha(p)==h)
 for rec in oldmanifest['results']:check('old_result_unchanged:'+rec['name'],sha(rec['path'])==rec['sha256'])
 check('822_fixed_jobs',len(manifest['results'])==822 and len({r['path'] for r in manifest['results']})==822 and t['total']==822 and len(t['pairs'])==36 and len(t['controls'])==238)
 expected={(kind,(p['query']+'_'+p['reference'] if kind=='pair' else p['name'])+f'_{s}') for kind,ls in [('pair',t['pairs']),('control',t['controls'])] for p in ls for s in [42,43,44]}
 check('exact_inventory',{(r['kind'],r['name']) for r in manifest['results']}==expected)
 ext=read(U/'review/EXTENSION_ELIGIBILITY.json');check('extension32_inventory',t['extended_rows']==ext['rows'] and len(ext['rows'])==32 and {r['alias'] for r in ext['rows']}=={p['alias'] for p in t['controls'] if p['scope']=='expanded32'})
 check('old_pair_rows_verbatim',ev['pairs'][:len(prior['pairs'])]==prior['pairs'])
 check('old_control_rows_verbatim',[{k:v for k,v in r.items() if k!='scope'} for r in ev['controls'][:len(prior['controls'])]]==prior['controls'])
 info=read(STAGE/'LEARNING_INPUTS.json');row_by={r['alias']:r for r in info['rows']};idx={r['alias']:i for i,r in enumerate(info['rows'])};prob={s:np.load(STAGE/'runs'/f'pose_plus_signrep_{s}'/'probabilities.npy',mmap_mode='r',allow_pickle=False) for s in [42,43,44]};base={}
 aliases={p[k] for p in t['pairs'] for k in ['query','reference']}|{p['alias'] for p in t['controls']}
 for alias in aliases:
  row=row_by[alias]
  with np.load(row['input'],allow_pickle=False) as z:a={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
  a['names']=row['feature_names'];a['probs']={s:np.asarray(prob[s][idx[alias],:len(a['x'])])[:,:3] for s in [42,43,44]};base[alias]=a
  check('sealed_input:'+alias,sha(row['input'])==row['input_sha256']==frozen['pins'][row['input']])
 # Import ONLY our earlier independent pure formulas, not its audit driver or candidate.
 independent=OLD/'review/verify_saved_results.py';tree=ast.parse(independent.read_text(encoding='utf-8'));scope={'np':np,'Counter':Counter,'defaultdict':defaultdict}
 phys=read(OPS/'word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json');scope['physical']={r['alias']:r for r in phys['clips']}
 defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['source_groups','phase_calculation']];exec(compile(ast.Module(body=defs,type_ignores=[]),'<independent_review_formulas>','exec'),scope)
 def controlled(task):
  b=base[task['alias']]
  if task['array']:
   with np.load(task['array'],allow_pickle=False) as z:q={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
  else:q={k:b[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
  q['names']=b['names'];n=len(q['x']);ids=np.array(task['metadata'].get('original_indices',list(range(n))),int)
  if task['scope']=='expanded32':
   check('fixed_center:'+task['name'],not task['metadata']['changed_frames'] or task['metadata']['changed_frames']==list(range(n//2-2,n//2+3)))
   if task['control']=='right_hand_unobserved':q['mask'][:,[28,29,34,35,36,37]+list(range(39,59))+[60,65,66,67,68]]=False
   if task['control']=='short_shape_change_5f':
    ix=np.ix_(task['metadata']['changed_frames'],list(range(39,53)));before=q['x'][ix].copy();q['x'][ix]=np.where(before<.5,1.,0.);check('actual_changes:'+task['name'],int((q['mask'][ix]&(q['x'][ix]!=before)).sum())==task['metadata']['actual_observed_changed_cells'])
  elif task['control']=='small_common_coordinate_drift':
   jitter=.002*np.sin(np.arange(n)*2*np.pi/8)
   for side in ['L','R']:
    for part in ['hand_wrist','pose_wrist','elbow','index_tip','palm_center']:
     for axis in ['x','y']:
      j=q['names'].index(f'{side}_{part}_{axis}');q['x'][:,j]+=np.where(q['mask'][:,j],jitter,0)
  check('source_indices:'+task['name'],len(ids)==n and (ids>=0).all() and (ids<len(b['x'])).all());return q,ids
 accounting={(r['name'],r['group'],r['side']):r for r in account['rows']};newpairs={};newcontrols={};shortcalc={};representatives={};axis_count=0;cell_count=0;null_count=0;mixed_count=0;full_count=0
 for rec in manifest['results']:
  check('result_sha:'+rec['name'],sha(rec['path'])==rec['sha256']);d=read(rec['path']);task=d['task'];seed=d['seed'];name=d['name'];e=d['evidence'];path=d['path'];w=np.asarray(d['path_weights']);qi=np.array([p[0] for p in path],int);ri=np.array([p[1] for p in path],int)
  check('frozen_result:'+name,d['frozen_sha256']==sha(U/'FROZEN.json') and d['name']==rec['name'] and d['kind']==rec['kind'])
  if d['kind']=='pair':q=base[task['query']];r=base[task['reference']];qp=q['probs'][seed]
  else:q,ids=controlled(task);r=base[task['alias']];qp=r['probs'][seed][ids]
  n,m=len(q['x']),len(r['x']);full=bool(path) and set(qi)==set(range(n)) and set(ri)==set(range(m));full_count+=full
  check('full_frame_trace:'+name,full and d['query_frames']==n and d['reference_frames']==m and d['query_frame_ids']==q['frame_id'].tolist() and d['reference_frame_ids']==r['frame_id'].tolist() and d['query_timestamp_us']==q['timestamp_us'].tolist() and d['reference_timestamp_us']==r['timestamp_us'].tolist())
  steps=np.diff(np.asarray(path),axis=0);expectedw=[2]+[2 if tuple(s)==(1,1) else 1 for s in steps];run=0;last=None;maxrun=0
  for step in steps:
   v=tuple(step);run=run+1 if v==last and v!=(1,1) else 0 if v==(1,1) else 1;maxrun=max(maxrun,run);last=v
  check('path_contract:'+name,path[0]==[0,0] and path[-1]==[n-1,m-1] and len({tuple(p) for p in path})==len(path) and all(tuple(s) in [(1,0),(0,1),(1,1)] for s in steps) and d['path_weights']==expectedw and d['path_weight_sum']==int(w.sum())==n+m and d['non_diagonal_moves']==sum(tuple(s)!=(1,1) for s in steps) and maxrun<=2)
  check('no_semantic_claim:'+name,d['query_labels_used'] is False and d['word_or_phase_used'] is False and d['source_alignment_indices_used'] is False and d['new_model_inference'] is False and d['new_inference']==0 and all(d[k] is None for k in ['scalar_agreement','semantic_verdict','whole_sign_verdict']) and e['semantic_verdict'] is None and e['scalar_agreement'] is None and e['missing_values_imputed'] is False)
  qg=scope['source_groups'](q,qp);rg=scope['source_groups'](r,r['probs'][seed]);familycost={};axes_for_short={};any_shared=False
  for family,names in FAM.items():
   costs=[]
   for group in names:
    x,xm=qg[group];y,ym=rg[group];shared=xm[qi]&ym[ri];values=np.where(shared,np.minimum(np.abs(x[qi]-y[ri]),1),0).sum(1)/x.shape[1];costs.append(values)
    seen=np.zeros_like(ym)
    for i,j in zip(qi,ri):seen[j]|=xm[i]&ym[j]
    support=dict(reference_possible_cells=int(ym.size),reference_known_cells=int(ym.sum()),reference_unknown_cells=int(ym.size-ym.sum()),compared_distinct_reference_cells=int(seen.sum()),reference_coverage=float(seen.sum()/ym.sum()) if ym.sum() else None,query_possible_cells=int(xm.size),query_known_cells=int(xm.sum()),shared_path_cells=int(shared.sum()),known_reference_missing_query_path_cells=int((ym[ri]&~xm[qi]).sum()),absence_is_zero_truth=False)
    any_shared|=bool(shared.any())
    check('group_support:'+name+group,equal(support,d['group_support'][group]) and e['groups'][group]['status']==('LOCAL_RESIDUALS_ONLY' if shared.any() else 'UNASSESSABLE') and e['groups'][group]['semantic_verdict'] is None)
    for side,own,om,other,ot,oi,oj,source in [('query',x,xm,y,ym,qi,ri,q),('reference',y,ym,x,xm,ri,qi,r)]:
     a=axis_calc(own,om,other,ot,oi,oj);saved=e['groups'][group][side];lab=name+'/'+group+'/'+side;axis_count+=1;cell_count+=own.size;null_count+=int((~a['support']).sum());mixed_count+=int(a['mixed'].sum())
     fields={'own_observed':'own','matched_support':'support','matched_link_count':'matched','aligned_link_count':'aligned','missing_counterpart':'uncompared','missing_counterpart_link_count':'missing','mixed_counterpart_missing':'mixed'}
     check('axis_arrays:'+lab,all(np.array_equal(np.asarray(saved[k]),a[v]) for k,v in fields.items()) and arr_eq(saved['mean_residual'],a['means']) and arr_eq(saved['max_residual'],a['peaks']))
     check('axis_null_status_provenance:'+lab,np.array_equal(np.asarray(saved['cell_status'],object),a['status']) and np.array_equal(np.asarray(saved['cell_reason'],object),a['reason']) and equal(saved['counts'],a['counts']) and saved['missing_indices_by_dimension']==a['indices'] and saved['frame_id']==source['frame_id'].tolist() and saved['timestamp_us']==source['timestamp_us'].tolist() and saved['possible_shape']==list(own.shape) and saved['all_opportunities_retained'] is True and len(saved['dimension_labels'])==own.shape[1] and saved['semantic_verdict'] is None and saved['raw_residual_is_automatic_error'] is False)
     calc=local(a,list(range(len(own))));expect=dict(name=name,group=group,side=side,**calc);check('accounting:'+lab,equal(expect,accounting[(name,group,side)]))
     if group in ['L_shape','R_shape']:axes_for_short[(group,side)]=a
     for typ,mask in [('shared',a['support']),('missing',a['uncompared']),('partial',a['mixed'])]:
      if typ not in representatives and mask.any():
       fi,di=np.argwhere(mask)[0];representatives[typ]={'result':rec['path'],'group':group,'side':side,'frame_ordinal':int(fi),'dimension':int(di),'mean':float(a['means'][fi,di]) if np.isfinite(a['means'][fi,di]) else None,'max':float(a['peaks'][fi,di]) if np.isfinite(a['peaks'][fi,di]) else None,'shared_links':int(a['matched'][fi,di]),'missing_links':int(a['missing'][fi,di]),'status':str(a['status'][fi,di]),'recomputed_from_raw':True}
   familycost[family]=float((np.mean(costs,axis=0)*w).sum()/(n+m))
  check('all_cost_arithmetic:'+name,equal(familycost,d['family_costs']) and near(statistics.mean(familycost.values()),d['normalized_cost']) and near(d['normalized_cost']*(n+m),d['weighted_cost_sum']) and near(d['representative_weighted_cost_sum'],d['weighted_cost_sum']))
  check('no_support_no_agreement:'+name,e['status']==('LOCAL_RESIDUALS_ONLY' if any_shared else 'UNASSESSABLE') and e['all_query_and_reference_opportunities_retained'] is True and e['query_frame_ids']==q['frame_id'].tolist() and e['reference_frame_ids']==r['frame_id'].tolist())
  if d['kind']=='pair':newpairs[(task['query'],task['reference'],seed)]=scope['phase_calculation'](task['query'],task['reference'],{'*':path})
  else:
   newcontrols[(task['name'],seed)]=control_metrics(d,task)
   if task['control'] in ['short_shape_change_5f','left_short_shape_change_5f']:
    group='L_shape' if task['control'].startswith('left_') else 'R_shape';frames=task['metadata']['changed_frames'];item=dict(alias=task['alias'],seed=seed,scope=task['scope'],control=task['control'],changed_frames=frames,actual_observed_changes=task['metadata']['actual_observed_changed_cells'],new_query=local(axes_for_short[(group,'query')],frames),new_reference=local(axes_for_short[(group,'reference')],frames))
    if task['scope']=='previous16':
     oldpath=read(OLD/'results'/f"control_{task['name']}_{seed}.json")['path'];oq=np.array([v[0] for v in oldpath],int);orr=np.array([v[1] for v in oldpath],int);x,xm=qg[group];y,ym=rg[group]
     item['old_path_query']=local(axis_calc(x,xm,y,ym,oq,orr),frames);item['old_path_reference']=local(axis_calc(y,ym,x,xm,orr,oq),frames)
    shortcalc[(task['alias'],seed,task['scope'],task['control'])]=item
 check('all_axis_rows',axis_count==822*16*2==len(account['rows'])==len(accounting))
 for item in ev['shorts']:check('short_dual_axis:'+str((item['alias'],item['seed'],item['scope'],item['control'])),equal(shortcalc[(item['alias'],item['seed'],item['scope'],item['control'])],item))
 check('all_short_records',len(shortcalc)==len(ev['shorts'])==192 and ev['shorts']==account['short_change_diagnosis'])
 summary={};byword={};cs={}
 for row in ev['pairs']:
  calc=newpairs[(row['query'],row['reference'],row['seed'])] if row['method']=='split' else row['audit'];check('stage_audit:'+row['source'],equal(calc,row['audit']))
  for target,key in [(summary,f"{row['role']}/{row['method']}/{row['seed']}"),(byword,f"{row['role']}/{row['word_id']}/{row['method']}/{row['seed']}")]:
   g=target.setdefault(key,dict(pairs=0,full_reference_frames=0,certain_reference_frames=0,states=Counter(),ambiguous_paths=0));g['pairs']+=1;g['full_reference_frames']+=calc['reference_full_frames'];g['certain_reference_frames']+=calc['expected_certain_frames'];g['states'].update(calc['states']);g['ambiguous_paths']+=bool(row.get('ambiguous',False))
 for table,saved in [(summary,ev['summary']),(byword,ev['byword'])]:
  for key,g in table.items():g['same_kind_percent']=100*g['states']['same_observed_kind']/g['certain_reference_frames'];g['states']=dict(g['states']);check('aggregate:'+key,equal(g,saved[key]))
 for seed in [42,43,44]:check('4380_1080:'+str(seed),sum(summary[f'{role}/split/{seed}']['full_reference_frames'] for role in ['fit','development'])==4380 and sum(summary[f'{role}/split/{seed}']['certain_reference_frames'] for role in ['fit','development'])==1080)
 for row in ev['controls']:
  calc=newcontrols[(row['name'],row['seed'])] if row['method']=='split' else row['audit'];check('control_metric:'+row['name']+str(row['seed']),equal(calc,row['audit']));key=f"{row['scope']}/{row['control']}/{row['method']}/{row['seed']}";g=cs.setdefault(key,dict(cases=0,errors=[],fullpaths=0,exact=[],ambiguous=0));g['cases']+=1;g['errors'].append(calc['mean_original_frame_error']);g['exact'].append(calc['exact_original_frame_fraction']);g['fullpaths']+=calc['full_path'];g['ambiguous']+=bool(row.get('ambiguous',False))
 for key,g in cs.items():g['mean_original_frame_error']=statistics.mean(g['errors']);g['exact_original_frame_percent']=100*statistics.mean(g['exact']);check('control_aggregate:'+key,equal(g,ev['control_summary'][key]))
 scores=[summary[f'development/split/{s}']['same_kind_percent'] for s in [42,43,44]];baselines=[summary[f'development/{m}/None']['same_kind_percent'] for m in ['global_dtw','relational_global']];reg=[];speed=[]
 for word in sorted({p['word_id'] for p in t['pairs']}):
  best=max(byword[f'development/{word}/{m}/None']['same_kind_percent'] for m in ['global_dtw','relational_global'])
  for seed in [42,43,44]:reg.append(dict(word=word,seed=seed,pp=byword[f'development/{word}/split/{seed}']['same_kind_percent']-best))
 for kind in ['speed_075','speed_125']:
  baseline=cs[f'previous16/{kind}/saved_relational_global/None']['mean_original_frame_error']
  for seed in [42,43,44]:speed.append(dict(control=kind,seed=seed,increase=cs[f'previous16/{kind}/split/{seed}']['mean_original_frame_error']-baseline))
 technical=dict(full_scope_preserved=full_count==822,all_axis_opportunities_accounted=axis_count==26304,identity_exact=all(cs[f'{scope}/identity/split/{seed}']['mean_original_frame_error']==0 for scope in ['previous16','expanded32'] for seed in [42,43,44]),right_mask_same_clip_exact=all(cs[f'{scope}/right_hand_unobserved/split/{seed}']['mean_original_frame_error']==0 for scope in ['previous16','expanded32'] for seed in [42,43,44]),short_changed_cells_accounted=all(v['new_query']['own_observed']==v['actual_observed_changes'] for v in shortcalc.values()),normal_speed_error_increase_at_most1=all(x['increase']<=1 for x in speed))
 broad=dict(mean_gain_at_least5pp_vs_both=all(statistics.mean(scores)>=b+5 for b in baselines),every_seed_above_both=all(s>max(baselines) for s in scores),no_word_regression_over5pp=all(x['pp']>=-5 for x in reg));decision=dict(technical_gates=technical,broad_gates=broad,technical_pass=all(technical.values()),clear_development_improvement=all(technical.values()) and all(broad.values()),dev_scores=scores,baselines=baselines,word_regressions=reg,speed_changes=speed,adopted=False,unresolved_missing_island=True,educational_accuracy=None);check('all_original_gates',equal(decision,ev['decision']))
 out=dict(status='PASS_INDEPENDENT_STORED_SPLIT_RESULTS' if not failures else 'FAIL_INDEPENDENT_SPLIT_RESULT_REVIEW',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,failure_count=len(failures),failures=failures,frozen_pins=len(frozen['pins']),protected_pins=len(frozen['protected']),old534_results_unchanged=True,paths=822,full_scope_paths=full_count,axis_group_records=axis_count,axis_cells=cell_count,not_comparable_cells=null_count,partial_shared_support_cells=mixed_count,raw_representatives=representatives,independent_decision=decision,artifact_pins={str(U/n):sha(U/n) for n in ['FROZEN.json','TASKS.json','RESULT_MANIFEST.json','EVALUATION.json','AXIS_ACCOUNTING.json','split_alignment.py','run.py','evaluate.py']},verifier_sha256=sha(__file__),independent_source_formula_pin={str(independent):sha(independent)},new_model_inference=0,new_path_search=0,new_training=0,limits=['Every path cost and both-axis readout recomputed from raw sealed numeric arrays and saved old3 probabilities; no DP executed.','Numerical optimum/tie selection is reviewed through separate synthetic selftests, not re-solved here.','Non-comparable is not successful matching; partial means condition only on shared links and retained missing-link counts.','Expanded32 only technical controls; samekind remains historical4word scope, not educational truth.','Original old method rows reused verbatim; no historical score retuning.'])
 with (U/'review/RESULT_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps({'status':out['status'],'checks':checks,'failures':failures[:30],'paths':full_count,'axes':axis_count,'cells':cell_count,'decision':decision}))

if __name__=='__main__':main()
