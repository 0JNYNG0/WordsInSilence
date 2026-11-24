"""Saved new-public-upload/guard integration audit. No producer imports or inference."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import hashlib,importlib.util,json
import numpy as np

U=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def nz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def arrhash(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(str(a.shape).encode()+a.dtype.str.encode()+a.tobytes()).hexdigest()

def support(qm,rm,p,bounds,shape):
 p=np.asarray(p,int);qs=np.unique(p[:,0]);rr=np.arange(bounds[0],bounds[1]+1)
 rq=np.asarray([qm[p[p[:,1]==j,0]].any(0)&rm[j] for j in rr])
 need=np.asarray([rm[p[p[:,0]==i,1]].any(0) for i in qs]);good=need&qm[qs]
 dre=rm[rr].sum(0);dro=rq.sum(0);dqe=need.sum(0);dqo=good.sum(0)
 re,ro,qe,qo=[int(x.sum()) for x in [dre,dro,dqe,dqo]]
 rc=ro/re if re else None;qc=qo/qe if qe else None
 dc=np.minimum(dro/np.maximum(dre,1),dqo/np.maximum(dqe,1))
 coverage=min(rc,qc) if rc is not None and qc is not None else None
 if shape and re and qe:coverage=float(min(dc))
 return dict(reference_expected=re,reference_observed=ro,query_expected=qe,query_observed=qo,
  reference_coverage=rc,query_coverage=qc,coverage=coverage,dimension_reference_expected=dre.tolist(),
  dimension_reference_observed=dro.tolist(),dimension_query_expected=dqe.tolist(),dimension_query_observed=dqo.tolist(),
  dimension_coverage=dc.tolist(),query_indices=qs.tolist(),query_expected_mask=need.tolist(),query_observed_mask=good.tolist(),
  possible_query_cells=int(need.size),possible_reference_cells=int(rm[rr].size),expected_from_reference_only=True)

def run():
 output=U/'review/GUARD_LIVE_VERIFICATION.json'
 if output.exists():raise FileExistsError(output)
 checks=0;failures=[];categories=Counter();pins={}
 def check(category,name,value):
  nonlocal checks
  checks+=1;categories[category]+=1
  if not bool(value):failures.append(category+':'+name)
 def pin(p):pins[str(Path(p).resolve())]=sha(p)
 # This is the independent saved arithmetic module, NOT the engine or guard.
 arithmetic_path=U/'review/verify_feedback_evaluation.py'
 spec=importlib.util.spec_from_file_location('independent_saved_math',arithmetic_path)
 math=importlib.util.module_from_spec(spec);spec.loader.exec_module(math)
 pin(arithmetic_path)
 ui=read(U/'GUARD_LIVE_UI_VERIFICATION.json');upload=read(U/'GUARD_PUBLIC_UPLOAD_STARTED.json');intent=read(U/'GUARD_PUBLIC_UPLOAD_INTENT.json')
 jobs=list((U/'sessions').glob('*/'+ui['job_id']))
 if len(jobs)!=1:raise ValueError('Unique reported raw job required')
 job=jobs[0];session=job.parent;folder=job/'adapter'
 complete=read(folder/'COMPLETE.json');started=read(folder/'STARTED.json');request=read(job/'request.json')
 timing=read(job/'REQUEST_TIMING.json');startup=read(session/'STARTUP.json')
 base=read(job/'FEEDBACK.json');final=read(job/'FINAL_FEEDBACK.json');server=read(job/'server_result.json');audit=read(job/'EVIDENCE_GUARD.json')
 bank=read(U/'REFERENCE_BANK.json');public=read(U/'PUBLIC_INPUTS.json')
 row=next(r for r in public['rows'] if r['clip_id']==ui['public_source']);source=Path(upload['source']);submitted=job/'submission.mp4'
 for name in ['GUARD_LIVE_UI_VERIFICATION.json','GUARD_PUBLIC_UPLOAD_STARTED.json','GUARD_PUBLIC_UPLOAD_INTENT.json','PUBLIC_INPUTS.json','REFERENCE_BANK.json','GUARD_FROZEN.json','engine.py','evidence_guard.py','live_engine.py','server.py','adapter.py','app.js','index.html','check_guard_live_browser.py']:
  p=U/name
  if p.exists():pin(p)
 for p in [Path(__file__),source,submitted,job/'request.json',job/'REQUEST_TIMING.json',session/'STARTUP.json',session/'FIRST_REQUEST_TIMING.json',folder/'COMPLETE.json',folder/'STARTED.json',job/'FEEDBACK.json',job/'FINAL_FEEDBACK.json',job/'server_result.json',job/'EVIDENCE_GUARD.json']:pin(p)
 check('source','new public Egypt REAL13',row['person']==13 and row['word_id']=='WORD2523')
 check('source','source/upload/adapter/UI exact SHA',all(h==row['source_sha256'] for h in [sha(source),sha(submitted),upload['sha256'],started['source_sha256'],complete['source_sha256'],ui['source_sha256']]))
 check('source','byte size and accepted identity',submitted.stat().st_size==request['bytes']==source.stat().st_size and timing['job_id']==upload['job_id']==ui['job_id'])
 check('source','distinct previous raw upload',upload['sha256']!=read(U/'PUBLIC_UPLOAD_STARTED.json')['sha256'])
 check('source','target remains fixed',base['word']==final['word']==request['word']==row['word_id'])
 check('source','no user training or bank addition',request['training_allowed'] is False and request['reference_bank_addition_allowed'] is False and final['training_used_submission'] is False)
 check('UI','saved nine checks and no JS/camera',ui['status']=='PASS' and len(ui['checks'])==9 and not ui['js_errors'] and ui['camera_access'] is False and ui['user_data_read'] is False)
 check('UI','exact HTML and JS source',ui['html_sha256']==sha(U/'index.html') and ui['js_sha256']==sha(U/'app.js'))
 check('UI','server response is final guarded JSON',server==final and ui['latency']==server['latency'] and ui['request_timing']==timing)
 check('runtime','complete with no recorded pipeline failure',timing['status']=='COMPLETE' and not (job/'SERVER_FIRST_FAILURE.txt').exists() and not (folder/'FIRST_FAILURE.txt').exists())
 check('runtime','startup before actual first request',datetime.fromisoformat(startup['ready_at'])<datetime.fromisoformat(request['created_at']) and startup['status']=='READY' and timing['is_first_actual_analysis_request'])
 check('runtime','first request timing exact and startup separate',read(session/'FIRST_REQUEST_TIMING.json')==timing and timing['startup_seconds_included'] is False and timing['separately_recorded_startup_seconds']==startup['total_startup_seconds'])
 check('runtime','fresh video predictions with initialized runtimes',started['fresh_video'] and not started['cached_predictions_used'] and not complete['cached_predictions_used'] and not complete['runtime_reuse']['predictions_cached'] and complete['runtime_reuse']['signrep_initialized_before_video'] and complete['runtime_reuse']['heads_initialized_before_video'])
 check('runtime','GPU route retained',complete['device']==startup['warmup']['device'] and complete['device']['actual']==['GPU.0'] and complete['device']['precision']=='f16')
 check('runtime','no real-video warmup',startup['warmup']['real_video_inference']==0 and startup['warmup']['synthetic_rgb_windows']==1)
 for name,h in complete['artifacts'].items():check('adapter_pin',name,sha(folder/name)==h);pin(folder/name)
 a=nz(folder/'analysis.npz');n=complete['frames'];names=complete['feature_names'];a['feature_names']=names
 check('adapter','full arrays and monotonic native axes',a['x'].shape==a['mask'].shape==(n,69) and a['rgb'].shape==(n,768) and a['probabilities'].shape==(n,7) and np.all(np.diff(a['frame_id'])>0) and np.all(np.diff(a['timestamp_us'])>0))
 for file in ['raw.npz','observations.npz','face.npz']:
  z=nz(folder/file);check('adapter',file+' exact axes',np.array_equal(z['frame_id'],a['frame_id']) and np.array_equal(z['timestamp_us'],a['timestamp_us']))
 centers=np.asarray([i for i in range(6,n-6) if np.array_equal(a['frame_id'][i-6:i+7],np.arange(a['frame_id'][i]-6,a['frame_id'][i]+7))])
 known=np.zeros((n,7),bool);known[:,:3]=True;known[centers,3:]=True
 check('adapter','exact native3 and center4 availability',np.array_equal(np.isfinite(a['probabilities']),known) and np.array_equal(a['posterior_available'],known) and np.array_equal(a['new_four_center_indices'],centers) and complete['new_four_predicted_frames']==len(centers))
 check('adapter','posterior range',np.all((a['probabilities'][known]>=0)&(a['probabilities'][known]<=1)))
 check('adapter','maximum duration/frame contract',n<=1200 and a['timestamp_us'][-1]-a['timestamp_us'][0]<=20_000_000)
 check('latency','adapter durations retained',all(final['latency'][k]==v==base['latency'][k] for k,v in complete['latency'].items()))
 check('latency','separate base guard and HTTP',timing['analysis_seconds']>=final['latency']['total_analysis_seconds']>=base['latency']['total_analysis_seconds']>=complete['latency']['total_adapter_seconds'] and final['latency']['evidence_guard_seconds']>=0)
 check('guard','compact response and full local audit','guard_audit' not in final and final['evidence_guard']['aggregate_changes']==audit['aggregate_changes'] and final['evidence_guard']['full_audit_saved_locally'])
 check('guard','frozen producer/source contract',audit['frozen_contract_sha256']==sha(U/'GUARD_FROZEN.json') and audit['engine_sha256']==sha(U/'engine.py') and audit['new_inference']==0)
 refs={r['clip_id']:r for r in bank['rows'] if r['word_id']==final['word']};base_refs={r['id']:r for r in base['references']}
 check('guard','nine FIT people and exact reference selection',len(base_refs)==9 and set(base_refs)==set(refs)==set(audit['pairs']) and {r['person'] for r in refs.values()}==set(range(1,10)) and final['selected_whole_references']==base['selected_whole_references']==audit['selected_whole_references'])
 for k,v in base.items():
  if k not in ['stages','references','latency']:check('preservation','base top '+k,final[k]==v)
 qc=math.channels(a,names);family_support={};auxiliary_paths=0
 for rid,row in refs.items():
  r=nz(row['input']);r['probabilities']=nz(row['probability_file'])['probabilities'];r['feature_names']=names
  pin(row['input']);pin(row['probability_file']);p=audit['pairs'][rid];bp=np.asarray(base_refs[rid]['path']);ap=np.asarray(p['auxiliary_path']);auxiliary_paths+=1
  math.verify_pair(base_refs[rid],a,r,row,names,rid)
  check('guard_path',rid,np.array_equal(ap[0],[0,0]) and np.array_equal(ap[-1],[n-1,len(r['x'])-1]) and all(tuple(v) in [(0,1),(1,0),(1,1)] for v in np.diff(ap,axis=0)))
  rc=math.channels(r,names);scope=math.scopes(row,rc)
  # Reproduce the deterministic source-only cache key; no optimization.
  h=hashlib.sha256(b'arm-path-guard-v1'+audit['engine_sha256'].encode()+audit['frozen_contract_sha256'].encode())
  cols=[names.index(s+'_'+part+'_'+d) for s in ['L','R'] for part in ['pose_wrist','elbow'] for d in ['x','y']]
  for seq in [a,r]:
   for value in [seq['x'][:,cols],seq['mask'][:,cols],seq['frame_id'],seq['timestamp_us']]:
    value=np.ascontiguousarray(value);h.update(str(value.shape).encode());h.update(value.dtype.str.encode());h.update(value.tobytes())
  check('guard_input',rid,p['arm_path_key']==h.hexdigest())
  for i,stage in enumerate(row['stages']):
   lo,hi=stage['frames'];saved=p['stages'][i];b=bp[(bp[:,1]>=lo)&(bp[:,1]<=hi)];extra=ap[(ap[:,1]>=lo)&(ap[:,1]<=hi)];union=np.unique(np.vstack([b,extra]),axis=0)
   check('guard_links',rid+str(i),saved['original_links']==b.tolist() and saved['auxiliary_links']==extra.tolist() and saved['union_links']==union.tolist() and saved['reference_bounds']==[lo,hi])
   check('guard_time',rid+str(i),saved['query_frame_ids']==a['frame_id'][np.unique(union[:,0])].tolist() and saved['query_timestamp_us']==a['timestamp_us'][np.unique(union[:,0])].tolist())
   arm=support(qc['arm_path'][1],rc['arm_path'][1]&scope['arm_path'],extra,[lo,hi],False)
   check('guard_support',rid+str(i)+'auxiliary arm',arm==saved['auxiliary_arm_support'])
   for family in math.F:
    v=saved['families'][family];rm=rc[family][1]&scope[family]
    for label,links in [('original',b),('auxiliary',extra),('union',union)]:
     want=support(qc[family][1],rm,links,[lo,hi],family.endswith('_shape'))
     for k,value in want.items():check('guard_support',rid+str(i)+family+label+k,math.equal(value,v[label][k]))
    required=bool(scope[family][lo:hi+1].any());union_ok=v['union']['coverage'] is not None and v['union']['coverage']>=.65;arm_ok=arm['coverage'] is not None and arm['coverage']>=.65
    reasons=[]
    if required and not union_ok:reasons.append('UNION_QUERY_OR_REFERENCE_FEATURE_SUPPORT_BELOW_0_65')
    if required and not arm_ok:reasons.append('AUXILIARY_POSE_CORRESPONDENCE_SUPPORT_BELOW_0_65')
    check('guard_decision',rid+str(i)+family,v['sufficient']==(required and not reasons) and v['comparison_required']==required and v['reasons']==reasons)
    family_support[(rid,i,family)]=v
 changes=Counter()
 def check_stage(old,new,label,rids,index):
  changed=[]
  for f,(before,after) in zip(math.F,zip(old['features'],new['features'])):
   for k,value in before.items():
    if k!='status':check('unchanged_feature',label+f+k,after[k]==value)
   records=[family_support[(rid,index,f)] for rid in rids];sufficient=all(v['sufficient'] for v in records)
   expected='INSUFFICIENT_EVIDENCE' if before['status']=='SIMILAR_REFERENCE_RANGE' and not sufficient else before['status']
   check('status',label+f,after['status']==expected and after['guard_support']['sufficient']==sufficient and after['guard_support']['union_coverage']==[v['union']['coverage'] for v in records])
   if expected!=before['status']:changed.append(before['name']);changes['feature_'+label.split(':')[0]]+=1
  expected='INSUFFICIENT_EVIDENCE' if old['status']=='SIMILAR_REFERENCE_RANGE' and changed else old['status']
  check('status',label+'stage',new['status']==expected and new['guard_changed_features']==changed)
  if expected!=old['status']:changes['stage_'+label.split(':')[0]]+=1
  for k,v in old.items():
   if k not in ['features','status','message']:check('unchanged_stage',label+k,new[k]==v)
  if expected==old['status']:check('unchanged_stage',label+'message',new['message']==old['message'])
 for i,(old,new) in enumerate(zip(base['stages'],final['stages'])):check_stage(old,new,'aggregate:'+str(i),base['selected_whole_references'],i)
 for old,new in zip(base['references'],final['references']):
  for k,v in old.items():
   if k!='stages':check('unchanged_reference',old['id']+k,new[k]==v)
  for i,(before,after) in enumerate(zip(old['stages'],new['stages'])):check_stage(before,after,'individual:'+old['id']+str(i),[old['id']],i)
 check('base_saved_arithmetic','all nine original metrics',not math.failures)
 for p,h in pins.items():check('end_pin',Path(p).name,sha(p)==h)
 result=dict(status='PASS_SAVED_GUARD_RAW_INTEGRATION_WITH_LIMITS' if not failures and not math.failures else 'FAIL',
  checks=checks+math.checks,own_checks=checks,saved_base_arithmetic_checks=math.checks,failures=failures+math.failures,categories=dict(categories),source_pins=pins,
  job_id=ui['job_id'],job_directory=str(job.resolve()),public_clip=ui['public_source'],source_sha256=row['source_sha256'],frames=n,
  duration_seconds=float((a['timestamp_us'][-1]-a['timestamp_us'][0])/1e6),probabilities_array_sha256=arrhash(a['probabilities']),
  protected_baseline_native3_cells=int(known[:,:3].sum()),new4_center_cells=int(known[:,3:].sum()),new4_boundary_unknown_cells=int((~known[:,3:]).sum()),
  source_model_route=complete['probability_route'],device=complete['device'],runtime_reuse=complete['runtime_reuse'],
  saved_primary_paths_checked=9,saved_auxiliary_paths_checked=auxiliary_paths,original_selection=base['selected_whole_references'],guard_changes=dict(changes),
  aggregate_status_before=[s['status'] for s in base['stages']],aggregate_status_after=[s['status'] for s in final['stages']],
  latency=dict(startup_seconds=startup['total_startup_seconds'],warmup_seconds=startup['warmup_seconds'],adapter_seconds=complete['latency']['total_adapter_seconds'],
    base_analysis_seconds=base['latency']['total_analysis_seconds'],guard_seconds=final['latency']['evidence_guard_seconds'],final_analysis_seconds=final['latency']['total_analysis_seconds'],
    server_processing_seconds=timing['analysis_seconds'],upload_seconds=timing['upload_seconds'],queue_seconds=timing['queue_seconds'],startup_included=False),
  claims=['This is one additional public raw clip and an independent saved arithmetic audit; not a user-camera trial or educational accuracy test.',
   'Saved paths were inspected, not recomputed. Original posterior/metric/path/reference-selection values were preserved by the guard.',
   'Auxiliary arm correspondence remains inferred; support union is not same-phase ground truth.',
   'No matched speedup claim: this source differs from both earlier cold and warm raw uploads.',
   'UI PASS9 is the saved root browser test; no browser replay was performed by this verifier.'],
  producer_imports=0,new_model_forwards=0,new_DTWs=0,new_video_decodes=0,new_server_starts=0,new_camera_access=0)
 with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
 print(json.dumps(dict(status=result['status'],checks=result['checks'],failures=len(result['failures']),path=str(output),sha256=sha(output))))
 if result['failures']:raise AssertionError(result['failures'][:20])

if __name__=='__main__':run()
