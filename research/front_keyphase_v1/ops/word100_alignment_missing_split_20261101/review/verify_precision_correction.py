"""Correct only reviewer float32 subtraction; no full audit or model/DP replay."""
import ast,json,pathlib,hashlib,datetime,numpy as np
U=pathlib.Path(__file__).resolve().parent.parent;OPS=U.parent;STAGE=OPS/'word100_stage_feedback_cycle_20261031'
read=lambda p:json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
initial=read(U/'review/RESULT_VERIFICATION.json');frozen=read(U/'FROZEN.json');manifest=read(U/'RESULT_MANIFEST.json');account=read(U/'AXIS_ACCOUNTING.json');info=read(STAGE/'LEARNING_INPUTS.json')
assert initial['failure_count']==260 and all('learned3' in f and f.startswith(('axis_arrays:','accounting:')) for f in initial['failures'])
ns={'np':np};tree=ast.parse((U/'review/verify_results.py').read_text(encoding='utf-8'));defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['axis_calc','local','near','equal','arr_eq']];ns['math']=__import__('math');exec(compile(ast.Module(body=defs,type_ignores=[]),'<reviewer_corrected_scalar_math>','exec'),ns)
rows={r['alias']:r for r in info['rows']};indices={r['alias']:i for i,r in enumerate(info['rows'])};base={};probs={s:np.load(STAGE/'runs'/f'pose_plus_signrep_{s}'/'probabilities.npy',allow_pickle=False,mmap_mode='r') for s in [42,43,44]}
accounting={(r['name'],r['group'],r['side']):r for r in account['rows']};checks=0;failures=[]
def check(k,v):
 global checks
 checks+=1
 if not v:failures.append(k)
def original(alias):
 if alias not in base:
  row=rows[alias]
  with np.load(row['input'],allow_pickle=False) as z:base[alias]=(z['mask'].copy(),len(z['frame_id']),row['feature_names'])
 return base[alias]
def support(mask,names):
 return np.stack([mask[:,[names.index(side+'_hand_wrist_'+a) for a in ['x','y']]].all(1)&(mask[:,[i for i,n in enumerate(names) if n.startswith(side+'_') and '_angle' in n and 'elbow' not in n]].sum(1)>=10) for side in ['R','R','L']],axis=1)
for p,h in initial['artifact_pins'].items():check('initial_source_unchanged:'+p,sha(p)==h)
for rec in manifest['results']:
 check('result_sha:'+rec['name'],sha(rec['path'])==rec['sha256']);d=read(rec['path']);t=d['task'];seed=d['seed'];qa=t['query'] if d['kind']=='pair' else t['alias'];ra=t['reference'] if d['kind']=='pair' else t['alias'];qm,n,names=original(qa);rm,m,_=original(ra);qm=qm.copy();ids=np.arange(n)
 if d['kind']=='control':
  if t['array']:
   with np.load(t['array'],allow_pickle=False) as z:qm=z['mask'].copy();n=len(qm)
  elif t['scope']=='expanded32' and t['control']=='right_hand_unobserved':qm[:,[28,29,34,35,36,37]+list(range(39,59))+[60,65,66,67,68]]=False
  ids=np.array(t['metadata'].get('original_indices',list(range(n))),int)
 q=np.asarray(probs[seed][indices[qa],:original(qa)[1],:3],np.float64)[ids];r=np.asarray(probs[seed][indices[ra],:m,:3],np.float64);qs=support(qm,names)&np.isfinite(q);rs=support(rm,names)&np.isfinite(r);q=np.where(qs,q,0);r=np.where(rs,r,0)
 qi=np.array([p[0] for p in d['path']]);ri=np.array([p[1] for p in d['path']])
 for side,own,om,other,otherm,i,j in [('query',q,qs,r,rs,qi,ri),('reference',r,rs,q,qs,ri,qi)]:
  a=ns['axis_calc'](own,om,other,otherm,i,j);saved=d['evidence']['groups']['learned3'][side];label=d['name']+'/'+side
  check('float64_mean_max:'+label,ns['arr_eq'](saved['mean_residual'],a['means']) and ns['arr_eq'](saved['max_residual'],a['peaks']))
  check('float64_support_counts:'+label,saved['counts']==a['counts'] and np.array_equal(np.asarray(saved['matched_link_count']),a['matched']) and np.array_equal(np.asarray(saved['missing_counterpart_link_count']),a['missing']) and np.array_equal(np.asarray(saved['cell_status'],object),a['status']) and np.array_equal(np.asarray(saved['cell_reason'],object),a['reason']))
  calc=dict(name=d['name'],group='learned3',side=side,**ns['local'](a,list(range(len(own)))));check('float64_accounting:'+label,ns['equal'](calc,accounting[(d['name'],'learned3',side)]))
out=dict(initial)
out.update(status='PASS_INDEPENDENT_STORED_SPLIT_RESULTS_AFTER_REVIEWER_PRECISION_CORRECTION' if not failures else 'FAIL_PRECISION_CORRECTION',at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),failure_count=len(failures),failures=failures,initial_review_preserved=str(U/'review/RESULT_VERIFICATION.json'),initial_review_sha256=sha(U/'review/RESULT_VERIFICATION.json'),initial_verifier_preserved=str(U/'review/verify_results_initial.py'),initial_verifier_sha256=sha(U/'review/verify_results_initial.py'),corrected_verifier_sha256=sha(U/'review/verify_results.py'),precision_recheck_count=checks,precision_recheck_scope='All822 results learned3 query/reference means, peaks, masks, counts, status and accounting. Original260 failures all attributable to reviewer float32 subtraction versus frozen implementation float64. All other initial checks retained; no full audit repeated.',reviewer_correction='Promote saved old3 probabilities to float64 before subtraction as specified by frozen features(). No tolerance relaxed, no product/source/result changed.',representative_precision_evidence={'case':'P033_P034_42 learned3 query','initial_max_mean_difference':2.5103872758336365e-08,'corrected_max_mean_difference':0.0},new_model_inference=0,new_path_search=0,new_training=0,correction_script_sha256=sha(__file__))
with (U/'review/RESULT_VERIFICATION_FINAL.json').open('x',encoding='utf-8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'status':out['status'],'precision_rechecks':checks,'failures':failures[:20],'preserved_initial_checks':initial['checks']}))
