"""Hash, exact-path and evidence-accounting audit; no readout/model/DP replay."""
import pathlib,json,gzip,hashlib,datetime,ast,numpy as np
U=pathlib.Path(__file__).resolve().parent.parent;OPS=U.parent
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
checks=0;failures=[]
def check(label,v):
 global checks
 checks+=1
 if not bool(v):failures.append(label)
contract=read(U/'READOUT_ONLY_CONTRACT.json');result=read(U/'READOUT_ONLY_RESULT.json');tasks=read(U/'TASKS.json');ev=read(U/'EVALUATION.json')
for p,h in contract['pins'].items():check('contract_pin:'+p,sha(p)==h)
check('one108_inventory',result['record_count']==len(result['records'])==108 and len({r['path'] for r in result['records']})==108)
pairmap={(p['query'],p['reference']):p['baselines']['global_dtw'] for p in tasks['pairs']}
check('36x3_exact_requests',{(r['query'],r['reference'],r['seed']) for r in result['records']}=={(q,r,s) for q,r in pairmap for s in [42,43,44]})
check('failed_candidate_unchanged',ev['decision']['adopted'] is False and ev['decision']['clear_development_improvement'] is False and result['failed_new_candidate_still_rejected'] is True)
check('no_new_score_or_search',all(result[k]==0 for k in ['new_path_search','new_model_inference','new_training','new_score_calculation']) and result['production_change'] is False and result['existing_development_stage_score_reused']==ev['summary']['development/global_dtw/None']['same_kind_percent'])
code=(U/'readout_only.py').read_text(encoding='utf-8');tree=ast.parse(code);api=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compare_existing_path');calls=[ast.unparse(n.func) for n in ast.walk(api) if isinstance(n,ast.Call)]
check('api_only_fixed_path_readout',all(n not in calls for n in ['candidate.align','candidate.solve','candidate.cost']) and 'candidate.readout' in calls and 'candidate.features' in calls)
check('runtime_search_disabled_before_loop',code.index('candidate.align=forbidden;candidate.solve=forbidden;candidate.cost=forbidden')<code.index("for task in t['pairs']"))
info=read(OPS/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json');rows={r['alias']:r for r in info['rows']};axes={}
for alias in {x for p in pairmap for x in p}:
 with np.load(rows[alias]['input'],allow_pickle=False) as z:axes[alias]=(z['frame_id'].tolist(),z['timestamp_us'].tolist())
total_cells=matched_cells=uncompared_cells=unknown_cells=partial_cells=0
group_dims={'L_shape':14,'R_shape':14,'learned3':3,'L_axis':2,'R_axis':2,'L_location':4,'R_location':4,'L_arm':6,'R_arm':6,'L_to_R_forearm':2,'R_to_L_forearm':2,'hands_relation':3,'L_displacement':2,'R_displacement':2,'L_direction':2,'R_direction':2}
for row in result['records']:
 check('gzip_sha:'+row['path'],sha(row['path'])==row['sha256'])
 with gzip.open(row['path'],'rt',encoding='utf-8') as f:d=json.load(f)
 source=pairmap[(row['query'],row['reference'])];original=read(source);path=original['path'];qids,qpts=axes[row['query']];rids,rpts=axes[row['reference']]
 check('exact_source_path:'+row['path'],row['source']==d['source']==source and row['source_sha256']==d['source_sha256']==sha(source) and d['path']==path and row['path_cells']==len(path) and d['contract_sha256']==sha(U/'READOUT_ONLY_CONTRACT.json'))
 check('full_endpoints_scope:'+row['path'],path[0]==[0,0] and path[-1]==[len(qids)-1,len(rids)-1] and {p[0] for p in path}==set(range(len(qids))) and {p[1] for p in path}==set(range(len(rids))) and all((b[0]-a[0],b[1]-a[1]) in [(0,1),(1,0),(1,1)] for a,b in zip(path,path[1:])))
 e=d['evidence'];check('payload_no_newpath_claim:'+row['path'],d['new_alignment'] is False and d['new_model_inference'] is False and d['whole_sign_verdict'] is None and e['semantic_verdict'] is None and e['scalar_agreement'] is None and e['missing_values_imputed'] is False and set(e['groups'])==set(group_dims))
 for group,dim in group_dims.items():
  for side,ids,pts in [('query',qids,qpts),('reference',rids,rpts)]:
   a=e['groups'][group][side];own=np.array(a['own_observed'],bool);shared=np.array(a['matched_support'],bool);missing=np.array(a['missing_counterpart'],bool);links=np.array(a['missing_counterpart_link_count'],int);mixed=np.array(a['mixed_counterpart_missing'],bool);means=np.array(a['mean_residual'],float);peaks=np.array(a['max_residual'],float)
   expected=dict(possible=len(ids)*dim,own_observed=int(own.sum()),own_unobserved=int((~own).sum()),matched_support=int(shared.sum()),observed_but_not_comparable=int((own&~shared).sum()),counterpart_missing_link_cells=int(links.sum()),mixed_support_cells=int((shared&(links>0)).sum()))
   check('axis_counts:'+row['path']+group+side,own.shape==(len(ids),dim) and a['possible_shape']==[len(ids),dim] and a['frame_id']==ids and a['timestamp_us']==pts and a['counts']==expected==row['group_counts'][group][side] and np.array_equal(missing,own&~shared) and np.array_equal(mixed,shared&(links>0)) and not (shared&~own).any())
   check('axis_null_and_no_false_normal:'+row['path']+group+side,np.array_equal(np.isfinite(means),shared) and np.array_equal(np.isfinite(peaks),shared) and a['semantic_verdict'] is None and a['raw_residual_is_automatic_error'] is False and expected['matched_support']+expected['observed_but_not_comparable']+expected['own_unobserved']==expected['possible'])
   total_cells+=expected['possible'];matched_cells+=expected['matched_support'];uncompared_cells+=expected['observed_but_not_comparable'];unknown_cells+=expected['own_unobserved'];partial_cells+=expected['mixed_support_cells']
out=dict(status='PASS_EXISTING_PATH_ADDITIVE_READOUT_COMPATIBILITY' if not failures else 'FAIL_READOUT_ONLY_VERIFICATION',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='Codex semantic_data_audit',checks=checks,failures=failures,records=108,unique_original_global_paths=36,original_paths_bytewise_array_equal=True,all_full_native_axes=True,axis_cells=total_cells,shared_cells=matched_cells,own_observed_without_shared_counterpart=uncompared_cells,own_unobserved_cells=unknown_cells,partial_shared_support_cells=partial_cells,raw_readout_arithmetic_repeated=False,basis_for_no_repeat='Exact same frozen readout code already verified against raw arrays over822 paths; this review checks all108 actual gzip payloads, original path identities, fullscope and serialized mask/null/count accounting.',api_calls_reviewed=calls,new_alignment=0,new_model_inference=0,new_training=0,new_score_calculation=0,failed822candidate_still_rejected=True,production_adopted=False,limits=['Compatibility evidence only, not a second alignment candidate or performance improvement.','Existing global samekind score was reused, not recalculated.','Local residual maximum is not an educational error verdict.','Partial cells preserve missing-link counts; mean is conditional on shared links.'],pins={str(U/n):sha(U/n) for n in ['READOUT_ONLY_CONTRACT.json','READOUT_ONLY_RESULT.json','readout_only.py','split_alignment.py','FROZEN.json','EVALUATION.json','review/RESULT_VERIFICATION_FINAL.json']},audit_script_sha256=sha(__file__))
with (U/'review/READOUT_ONLY_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':out['status'],'checks':checks,'failures':failures,'records':108,'cells':total_cells,'shared':matched_cells,'missing':uncompared_cells,'unknown':unknown_cells,'partial':partial_cells}))
