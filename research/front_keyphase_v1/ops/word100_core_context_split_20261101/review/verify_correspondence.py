"""Stored-result audit; execute only after root completion notification.
No import or execution of producer/search/DP/model code.
"""
import hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_context_sequence_match_20261031'
REL=U.parent/'word100_relational_sequence_compare_20261031'
PHASE=U.parent/'word100_core_stage_evidence_20261031'
checks=0;failures=[];pins={};hashes={};cache={};maskcache={}
def sha(p):
 p=str(p)
 if p not in hashes:hashes[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
 return hashes[p]
def read(p):pins[str(p)]=sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
 p=str(p)
 if p not in cache:
  pins[p]=sha(p)
  with np.load(p,allow_pickle=False) as z:cache[p]={k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
 return cache[p]
def check(key,cond,**info):
 global checks
 checks+=1
 if not bool(cond):failures.append({'check':key,**info})
def eq(a,b,p):
 if isinstance(a,dict):
  check('keys',isinstance(b,dict) and set(a)==set(b),at=p)
  if isinstance(b,dict):
   for k,v in a.items():
    if k in b:eq(v,b[k],p+'/'+k)
 elif isinstance(a,list):
  check('length',isinstance(b,list) and len(a)==len(b),at=p)
  if isinstance(b,list):
   for i,(x,y) in enumerate(zip(a,b)):eq(x,y,p+f'/{i}')
 elif isinstance(a,float):check('float',isinstance(b,(float,int)) and np.isclose(a,b,atol=1e-12,rtol=0),at=p,expected=a,saved=b)
 else:check('value',a==b,at=p,expected=a,saved=b)

def feature_masks(a):
 x=a['x'].astype(float);m=a['mask']&np.isfinite(x);out={}
 for hand,o in [('L',0),('R',28)]:
  for name,ids in [('shape',range(o+11,o+25)),('location',[o,o+1,o+8,o+9]),('axis',[o+25,o+26])]:out[hand+'_'+name]=m[:,list(ids)]
  delta=x[:,o+2:o+4]-x[:,o+4:o+6];ok=m[:,o+2:o+6].all(1)&(np.linalg.norm(delta,axis=1)>1e-8)
  out[hand+'_arm']=np.c_[m[:,[o+4,o+5,o+10]],np.tile(ok[:,None],(1,3))]
  other=28 if o==0 else 0;v=x[:,other+2:other+4]-x[:,other+4:other+6]
  good=m[:,o:o+2].all(1)&m[:,other+2:other+6].all(1)&((v*v).sum(1)>1e-8)
  out[hand+'_to_'+('R' if o==0 else 'L')+'_forearm']=np.tile(good[:,None],(1,2))
  out[hand+'_direction']=m[:,[63,64] if o==0 else [67,68]]
 out['hands_relation']=m[:,[56,57,58]]
 return out

def verify_evidence(saved,qmask,rmask,ids,path,tag):
 ids=np.asarray(ids,dtype=int);pp=np.asarray(path,dtype=int).reshape(-1,2)
 check('all13groups_preserved',set(saved)==set(rmask),at=tag)
 for g,rm in rmask.items():
  qm=qmask[g];expected=int(rm[ids].sum());possible=int(len(ids)*rm.shape[1]);v=saved[g]
  eq(expected,v['expected'],tag+'/'+g+'/expected');eq(possible,v['reference_possible'],tag+'/'+g+'/possible')
  eq(possible-expected,v['reference_unobserved'],tag+'/'+g+'/unobserved')
  if not len(pp):
   eq(0,v['compared'],tag+'/'+g+'/compared');eq(expected,v['unaligned_expected_cells'],tag+'/'+g+'/unaligned')
   eq(False,v['sufficient'],tag+'/'+g+'/sufficient');eq(None,v['path_missing_cells'],tag+'/'+g+'/missing')
   continue
  local=pp[np.isin(pp[:,1],ids)];comp=np.zeros_like(rm);missing=0;sharedquery=set()
  for i,j in local:
   shared=qm[i]&rm[j];comp[j]|=shared;missing+=int((rm[j]&~qm[i]).sum())
   if shared.any():sharedquery.add(int(i))
  cnt=int(comp[ids].sum());coverage=cnt/expected if expected else None
  sufficient=bool(expected and coverage>=.75 and missing==0 and rm[ids].any(1).sum()>=2 and len(sharedquery)>=2)
  for k,value in [('compared',cnt),('coverage',coverage),('path_missing_cells',missing),('distinct_shared_query_frames',len(sharedquery)),('sufficient',sufficient)]:eq(value,v[k],tag+'/'+g+'/'+k)

def numerical(anchor,indices,key):
 lo,hi=anchor['reference_core'];expected=set(range(lo,hi+1));present=set(indices.tolist());p=anchor[key]
 out={'segment_id':anchor['segment_id'],'state':anchor['state'],'expected_original_frames':len(expected),
  'present_unique_original_frames':len(present&expected),'missing_original_frames':sorted(expected-present)}
 if p is None:out.update(center_hit=False,center_original=None,recall=0.,precision=0.,path_mean_error=None)
 else:
  a,b=p['query_core'];got=set(indices[a:b+1].tolist());center=int(indices[(a+b)//2]);pp=np.asarray(p['core_path'],int)
  check('core_path_full_reference',set(pp[:,1])==expected,segment=anchor['segment_id'],candidate=key)
  check('core_path_full_selected_query',set(pp[:,0])==set(range(a,b+1)),segment=anchor['segment_id'],candidate=key)
  err=[abs(int(indices[i])-int(j)) for i,j in pp]
  out.update(center_hit=center in expected,center_original=center,recall=len(got&expected)/len(expected),precision=len(got&expected)/len(got),path_mean_error=float(np.mean(err)))
 return out

required=['RESULT_MANIFEST.json','SUMMARY.json','REPORT_DATA.json','CONTROL_EVALUATION.json','PAIR_EVALUATION.json','FROZEN.json']
assert all((U/x).exists() for x in required),'Root completion notification and all summaries required'
manifest=read(U/'RESULT_MANIFEST.json');frozen=read(U/'FROZEN.json');oldmanifest=read(OLD/'RESULT_MANIFEST.json')
check('162_complete_unique_jobs',len(manifest['results'])==162 and len({x['name'] for x in manifest['results']})==162)
check('same_prior_job_set',{x['name'] for x in manifest['results']}=={x['name'] for x in oldmanifest['results']})
for p,h in frozen['pins'].items():check('frozen_pin',sha(p)==h,path=p)
rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']};phases={c['alias']:c for c in read(PHASE/'PHASE_DATA.json')['clips']}
original={a:load(r['input']) for a,r in rows.items()};masks={a:feature_masks(v) for a,v in original.items()}
report=read(U/'REPORT_DATA.json');reportcases={c['id']:c for c in report['cases']}
by=defaultdict(list);pairs=[];stage_rows=[];none_counts=Counter();feedback_count=0;guidance_count=0
for item in manifest['results']:
 d=read(item['file']);name=d['name'];a,b=d['query'],d['reference'];kind=d['kind']
 check('result_hash',sha(item['file'])==item['sha256'],name=name);check('frozen_result',d['frozen_sha256']==sha(U/'FROZEN.json'),name=name)
 old=read(d['old_result']);check('old_result_sha',sha(d['old_result'])==d['old_result_sha256'],name=name)
 q=load(d['query_array']);qm=feature_masks(q);rm=masks[b]
 check('query_array_sha',sha(d['query_array'])==d['query_array_sha256'],name=name)
 check('scope',d['query_frames']==len(q['x']) and d['reference_frames']==len(original[b]['x']) and d['original_scope_preserved'] is True,name=name)
 check('no_semantic_or_forced_order',d['semantic_correctness'] is False and d['forced_interstage_order'] is False and d['word_or_person_used_for_search'] is False,name=name)
 prev=read(REL/'controls'/f'{name}.json') if kind=='control' else None
 indices=np.asarray(prev['metadata'].get('original_indices',list(range(len(q['x'])))) if prev else range(len(q['x'])),int)
 check('index_map_bounds',len(indices)==len(q['x']) and np.all(indices>=0) and np.all(indices<len(original[a]['x'])),name=name)
 if prev:
  check('control_source_pin',prev['array_sha256']==d['query_array_sha256'] and prev['source_input_sha256']==rows[a]['input_sha256'],name=name)
  if d['control'] in ['identity','speed_075','speed_125','omit_short_core_5f','swap_adjacent_core_blocks']:
   check('known_index_xmask_preserved',np.array_equal(q['x'],original[a]['x'][indices]) and np.array_equal(q['mask'],original[a]['mask'][indices]),name=name)
 cores=[s for s in phases[b]['segments'] if s['role']=='core_candidate'];sm={s['segment_id']:s for s in cores}
 check('all_core_stage_denominator_retained',[s['segment_id'] for s in cores]==[s['segment_id'] for s in d['anchors']],name=name)
 oldanchors={s['segment_id']:s for s in old['anchors']};mean={s['segment_id']:s for s in prev['free_order']['anchors']} if prev else {}
 raw=[];selected=[];evidence=[];occupied=set();ui={x['segment_id']:x for x in reportcases[name]['anchors']}
 for an in d['anchors']:
  sid=an['segment_id'];lo,hi=sm[sid]['core'];eq([lo,hi],an['reference_core'],name+'/'+sid+'/core')
  for key in ['raw','selected']:
   candidate=an[key]
   if candidate is None:none_counts[key]+=1
   else:
    verify_evidence(candidate['groups'],qm,rm,range(lo,hi+1),candidate['core_path'],name+'/'+sid+'/'+key+'/core')
    vals=[v for v in candidate['groups'].values() if v['expected']]
    eq(bool(vals) and all(v['sufficient'] for v in vals),candidate['core_supported'],name+'/'+sid+'/'+key+'/support')
    ctx=candidate['context'];t=original[b]['timestamp_us']/1e6
    ra=int(np.searchsorted(t,t[lo]-.2,'left'));rb=min(len(t)-1,int(np.searchsorted(t,t[hi]+.2,'right'))-1)
    eq([ra,rb],ctx['reference_context'],name+'/'+sid+'/'+key+'/context_reference')
    check('context_not_moving_core',ctx['core_not_moved'] is True,name=name,segment=sid,candidate=key)
    for phase,ids in [('entry',range(ra,lo)),('exit',range(hi+1,rb+1))]:
     cp=ctx['phases'][phase];verify_evidence(cp['groups'],qm,rm,ids,cp['path'],name+'/'+sid+'/'+key+'/'+phase)
  if prev:
   nr=numerical(an,indices,'raw');ns=numerical(an,indices,'selected');raw.append(nr);selected.append(ns)
   olda=oldanchors[sid];op=olda['best'];oldcenter=int(indices[sum(op['query_core'])//2]) if op else None
   context_hit=oldcenter is not None and lo<=oldcenter<=hi
   oldnum=next(z for z in old['numerical_correspondence'] if z['segment_id']==sid)
   eq(context_hit,oldnum['center_in_original_core'],name+'/'+sid+'/previous_context_hit')
   mean_hit=lo<=int(indices[mean[sid]['query_center']])<=hi
   observed=an['selected'] is not None and an['selected']['core_supported'];fully=an['state']=='supported_core_candidate'
   ev={'segment_id':sid,'mean_hit':mean_hit,'context_hit':context_hit,'raw_core':nr,'selected':ns,'core_observed':observed,
      'supported_unambiguous':fully,'context_complete':bool(an['selected'] and an['selected']['context']['complete']),
      'conflict':an.get('old_context_disjoint',False),'tie_resolved':an.get('context_tie_resolved',False),'state':an['state']}
   evidence.append(ev);stage_rows.append({'name':name,**ev})
  if an['selected']:occupied.update(range(an['selected']['query_core'][0],an['selected']['query_core'][1]+1))
  fb=an['feedback'];check('evidence_only_feedback',fb['kind']=='research_evidence_only' and fb['sign_correct'] is None and fb['sign_error'] is None and fb['correction_instruction'] is None and fb['source_guidance_is_target_not_query_observation'] is True,name=name,segment=sid)
  check('feedback_code_message_count',len(fb['messages'])==len(an['feedback_codes']),name=name,segment=sid);feedback_count+=1
  expected_guide={'description':sm[sid].get('description_ko'),'source_links':sm[sid].get('source_links',[]),'query_label_inferred':False}
  eq(expected_guide,an['reference_guidance'],name+'/'+sid+'/reference_only_guidance');guidance_count+=1
  eq(fb,ui[sid]['feedback'],name+'/'+sid+'/report_feedback');eq(expected_guide,ui[sid]['reference_guidance'],name+'/'+sid+'/report_guidance')
  check('UI_independent_path_parts_label',ui[sid]['path_is_concatenated_independent_parts'] is True,name=name,segment=sid)
  eq(an['raw']['query_core'] if an['raw'] else None,ui[sid]['raw_core'],name+'/'+sid+'/report_raw_core')
  eq(an['selected']['query_core'] if an['selected'] else None,ui[sid]['best']['query_core'] if ui[sid]['best'] else None,name+'/'+sid+'/report_selected_core')
 for relation in d['relations']:check('no_educational_order_verdict',relation['educational_verdict'] is None,name=name)
 eq([i for i in range(len(q['x'])) if i not in occupied],d['outside_core_candidates'],name+'/outside')
 check('outside_not_normal_or_discarded',d['outside_is_not_normal_or_discarded'] is True,name=name)
 if prev:
  eq(raw,d['numerical_raw'],name+'/numerical_raw');eq(selected,d['numerical_selected'],name+'/numerical_selected')
  by[d['control']].append({'name':name,'query':a,'role':d['role'],'stages':evidence,'relations':d['relations']})
 else:
  pairs.append({'query':a,'reference':b,'role':d['role'],'states':dict(Counter(x['state'] for x in d['anchors'])),
    'conflicts':sum(x.get('old_context_disjoint',False) for x in d['anchors']),
    'seconds_incremental':d['seconds_incremental_core_split'],'old_seconds_saved':d['old_context_seconds_saved']})
 for k,expected in [('original_indices',indices.tolist()),('query_pts',(q['timestamp_us']/1e6).tolist()),('original_query_pts',(original[a]['timestamp_us']/1e6).tolist()),('reference_pts',(original[b]['timestamp_us']/1e6).tolist()),('outside_core_candidates',d['outside_core_candidates'])]:eq(expected,reportcases[name][k],name+'/report/'+k)

metrics={}
for control,ds in by.items():
 ss=[s for d in ds for s in d['stages']];n=len(ss)
 metrics[control]={'cases':len(ds),'stages':n,'mean_hits':sum(s['mean_hit'] for s in ss),'context_hits':sum(s['context_hit'] for s in ss),
 'raw_core_hits':sum(s['raw_core']['center_hit'] for s in ss),'selected_hits':sum(s['selected']['center_hit'] for s in ss),
 'raw_to_selected_losses':sum(s['raw_core']['center_hit'] and not s['selected']['center_hit'] for s in ss),
 'raw_to_selected_gains':sum(not s['raw_core']['center_hit'] and s['selected']['center_hit'] for s in ss),
 'supported_hits':sum(s['selected']['center_hit'] and s['supported_unambiguous'] for s in ss),
 'context_complete_hits':sum(s['selected']['center_hit'] and s['supported_unambiguous'] and s['context_complete'] for s in ss),
 'conflicts':sum(s['conflict'] for s in ss),'states':dict(Counter(s['state'] for s in ss)),
 'mean_core_recall':sum(s['selected']['recall'] for s in ss)/n,'mean_core_precision':sum(s['selected']['precision'] for s in ss)/n,
 'tie_resolved':sum(s['tie_resolved'] for s in ss),'relations':dict(Counter(x['state'] for d in ds for x in d['relations']))}
summary=read(U/'SUMMARY.json');eq(metrics,summary['controls'],'summary/controls')
eq(dict(by),read(U/'CONTROL_EVALUATION.json')['controls'],'control_evaluation');eq(pairs,read(U/'PAIR_EVALUATION.json')['pairs'],'pair_evaluation')
check('fixed_counts',len(pairs)==36 and sum(len(ds) for ds in by.values())==126 and len(stage_rows)==358)
for name,v in metrics.items():check('same_denominator_even_none',v['stages']==(43 if name=='swap_adjacent_core_blocks' else 45),control=name)
times=defaultdict(float);oldtimes=defaultdict(float);states=Counter()
for p in pairs:
 if p['role']!='fit':times[p['query']]+=p['seconds_incremental'];oldtimes[p['query']]+=p['old_seconds_saved'];states.update(p['states'])
eq(dict(times),summary['incremental_seconds_by_dev_query'],'summary/new_times');eq(dict(oldtimes),summary['previous_context_seconds_by_dev_query'],'summary/old_times');eq(dict(states),summary['development_states'],'summary/dev_states')
gates={'identity_all45':metrics['identity']['selected_hits']==45,'slow_all45':metrics['speed_075']['selected_hits']==45,
 'swap_not_worse_than_mean40':metrics['swap_adjacent_core_blocks']['selected_hits']>=40,
 'no_context_override_damage':all(v['raw_to_selected_losses']==0 for v in metrics.values()),
 'right_missing_no_supported':metrics['right_hand_unobserved']['states'].get('supported_core_candidate',0)==0,'semantic_correctness_not_tested':True}
eq(gates,summary['gates'],'summary/gates');eq(summary,report['summary'],'report/summary')
check('no_production_or_allword_claim',summary['neural_training']==0 and summary['production_adopted'] is False and summary['all_word_generalization_verified'] is False)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items());check('all_read_pins_unchanged',unchanged)
out={'status':'PASS_SAVED_CORRESPONDENCE_MASKS_AND_FEEDBACK_SCOPE' if not failures else 'FAIL_SAVED_CORRESPONDENCE_MASKS_AND_FEEDBACK_SCOPE',
 'reviewer':'Codex agent A; independent stored set/mask arithmetic, same family not independent human KSL truth',
 'checks':checks,'failures':failures,'results':len(manifest['results']),'control_stage_denominator':len(stage_rows),
 'candidate_none_counts_all_results':{k:none_counts[k] for k in ['raw','selected']},'reference_guidance_checked':guidance_count,'feedback_checked':feedback_count,
 'metrics_recomputed':metrics,'gates_recomputed':gates,'control_stages':stage_rows,'pins':pins,'all_read_artifacts_unchanged':unchanged,
 'raw_to_selected_losses_all_control_stages':sum(v['raw_to_selected_losses'] for v in metrics.values()),
 'raw_to_selected_gains_all_control_stages':sum(v['raw_to_selected_gains'] for v in metrics.values()),
 'first_validation_attempt':'CORRESPONDENCE_FIRST_FAILURE.json preserves8 schema-key mismatches because raw_to_selected_losses/gains were added to producer summary after validator preparation. Existing value/mask checks passed; validator now independently computes the new fields and stricter per-stage/state gates. No producer/results changed.',
 'execution':{'new_searches':0,'DP_calls':0,'model_calls':0,'old_files_written':0},
 'limitations':['Original-index hits/recall/precision apply only to reused numerical controls; not educational or semantic sign correctness.',
 'Raw, selected, supported and context-complete counts keep the same denominator and express different evidence conditions. None candidates remain zero-hit/zero-recall/zero-precision cases.',
 'No actual raw/selected None candidate occurred in these162 outputs. The full358control-stage denominator was verified; None behavior is a read-code contract, not an exercised outcome in this dataset.',
 'Reference expected masks are reconstructed from saved inputs, including unobserved and unaligned expected cells. Numeric0 is not a fist, rest or support.',
 'Reference descriptions/source links are checked against the reference segment only. They are not labels inferred for the query.',
 'Different-person pairs have no known exact original-frame target; no correspondence accuracy is claimed for them.',
 'Global optimal paths/costs are handled by the separate algorithm verifier; no search or DP was rerun here.',
 'Saved timing separates new incremental computation from previously cached context time; this is not a full game/video latency benchmark.']}
(U/'review'/'CORRESPONDENCE_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':checks,'failures':failures[:10],'failure_count':len(failures),'results':out['results'],'metrics':metrics},ensure_ascii=False))
if failures:raise SystemExit(1)
