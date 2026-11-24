"""Read-only saved-index/denominator verification; never imports search/model code.
Execute only after root reports that all162 results and summaries are complete.
"""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_relational_sequence_compare_20261031'
PHASE=U.parent/'word100_core_stage_evidence_20261031'
checks=0;failures=[];pins={};hash_cache={}
def sha(p):
    p=str(p)
    if p not in hash_cache:hash_cache[p]=hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return hash_cache[p]
def read(p):
    pins[str(p)]=sha(p);return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    pins[str(p)]=sha(p)
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ['x','mask','frame_id','timestamp_us']}
def check(check_key,condition,**detail):
    global checks
    checks+=1
    if not bool(condition):failures.append({'check':check_key,**detail})
def equal(a,b,where):
    if isinstance(a,dict):
        check('dict_keys',isinstance(b,dict) and set(a)==set(b),where=where)
        if not isinstance(b,dict):return
        for k,v in a.items():
            if k in b:equal(v,b[k],where+'/'+k)
    elif isinstance(a,list):
        check('list_length',isinstance(b,list) and len(a)==len(b),where=where)
        if isinstance(b,list):
            for i,(x,y) in enumerate(zip(a,b)):equal(x,y,where+f'/{i}')
    elif isinstance(a,float):check('float',isinstance(b,(int,float)) and np.isclose(a,b,atol=1e-12,rtol=0),where=where,expected=a,saved=b)
    else:check('scalar',a==b,where=where,expected=a,saved=b)

needed=['RESULT_MANIFEST.json','SUMMARY.json','CONTROL_EVALUATION.json','PAIR_EVALUATION.json','REPORT_DATA.json','FROZEN.json']
assert all((U/n).exists() for n in needed),'Root completion/summary required; never partial validation'
manifest=read(U/'RESULT_MANIFEST.json');frozen=read(U/'FROZEN.json')
check('manifest_count_162',len(manifest['results'])==162)
check('manifest_unique_names',len({x['name'] for x in manifest['results']})==162)
check('manifest_frozen_sha',manifest['frozen_sha256']==sha(U/'FROZEN.json'))
for p,h in frozen['pins'].items():check('frozen_pin',sha(p)==h,path=p)
rows={r['alias']:r for r in read(U/'AUTHORIZATION.json')['rows']}
phase={r['alias']:r for r in read(PHASE/'PHASE_DATA.json')['clips']}
arrays={a:load(r['input']) for a,r in rows.items()}
for a,r in rows.items():check('original_input_hash',sha(r['input'])==r['input_sha256'],alias=a)
expected_names={p.stem for p in (OLD/'controls').glob('*.json')}|{p.stem for p in (OLD/'pairs').glob('B_same_path_*.json')}
check('exact_job_set',set(x['name'] for x in manifest['results'])==expected_names)
controls=defaultdict(list);pairs=[];correspondence_rows=[];case_checks=[];report_cases={c['id']:c for c in read(U/'REPORT_DATA.json')['cases']}

for item in manifest['results']:
    p=Path(item['file']);d=read(p);name=item['name'];a=d['query'];b=d['reference']
    check('result_hash',sha(p)==item['sha256'],name=name)
    check('result_name_kind',d['name']==name and d['kind']==item['kind'],name=name)
    check('result_frozen_sha',d['frozen_sha256']==sha(U/'FROZEN.json'),name=name)
    old=read(d['baseline_path']);check('baseline_sha',sha(d['baseline_path'])==d['baseline_sha256'],name=name)
    baseline_anchors={z['segment_id']:z for z in old['free_order']['anchors']}
    cores=[s for s in phase[b]['segments'] if s['role']=='core_candidate']
    check('all_reference_core_candidates_retained',[s['segment_id'] for s in cores]==[s['segment_id'] for s in d['anchors']],name=name)
    q=load(OLD/'controls'/f'{name}.npz') if d['kind']=='control' else arrays[a]
    qpath=OLD/'controls'/f'{name}.npz' if d['kind']=='control' else rows[a]['input']
    check('query_array_sha',sha(qpath)==d['query_array_sha256'],name=name)
    check('query_reference_fullscope',d['query_frames']==len(q['x']) and d['reference_frames']==len(arrays[b]['x']) and d['query_search_scope']==len(q['x']) and d['reference_scope']==len(arrays[b]['x']),name=name)
    indices=np.asarray(old.get('metadata',{}).get('original_indices',list(range(d['query_frames']))),dtype=int)
    check('original_index_length_bounds',len(indices)==len(q['x']) and np.all(indices>=0) and np.all(indices<len(arrays[a]['x'])),name=name)
    if d['kind']=='control' and d['control'] in ['identity','speed_075','speed_125','omit_short_core_5f','swap_adjacent_core_blocks']:
        check('saved_transform_x_matches_original_indices',np.array_equal(q['x'],arrays[a]['x'][indices]),name=name)
        check('saved_transform_mask_matches_original_indices',np.array_equal(q['mask'],arrays[a]['mask'][indices]),name=name)
    covered=set();computed=[]
    for anchor,s in zip(d['anchors'],cores):
        sid=s['segment_id'];rlo,rhi=s['core'];expected=set(range(rlo,rhi+1));best=anchor['best']
        check('reference_core_preserved',anchor['reference_core']==s['core'],name=name,segment=sid)
        present=len(set(indices.tolist())&expected);samples=int(np.isin(indices,list(expected)).sum())
        if best is None:
            corr={'segment_id':sid,'center_original':None,'center_in_original_core':False,'recall':0.,'precision':None,
                  'original_core_frames':len(expected),'original_core_frames_present':present,'query_samples_from_original_core':samples,
                  'path_mean_original_frame_error':None,'path_exact_fraction':None,'state':anchor['state']}
        else:
            lo,hi=best['query_core'];wl,wh=best['query_window'];ra,rb=anchor['reference_context'];pp=np.asarray(best['path'],dtype=int)
            check('path_bounds',pp.ndim==2 and pp.shape[1]==2 and np.all((pp[:,0]>=0)&(pp[:,0]<len(q['x']))) and np.all((pp[:,1]>=0)&(pp[:,1]<len(arrays[b]['x']))),name=name,segment=sid)
            check('path_complete_selected_window_and_reference_context',set(pp[:,0])==set(range(wl,wh+1)) and set(pp[:,1])==set(range(ra,rb+1)),name=name,segment=sid)
            part=pp[(pp[:,1]>=rlo)&(pp[:,1]<=rhi)]
            check('query_core_from_path',len(part)>0 and [int(part[:,0].min()),int(part[:,0].max())]==[lo,hi],name=name,segment=sid)
            check('all_original_core_reference_frames_in_path',set(part[:,1])==expected,name=name,segment=sid)
            pred=set(indices[lo:hi+1].tolist());hit=len(pred&expected);center=(lo+hi)//2
            err=np.asarray([abs(int(indices[i])-int(j)) for i,j in part],dtype=float)
            corr={'segment_id':sid,'center_original':int(indices[center]),'center_in_original_core':int(indices[center]) in expected,
                  'recall':hit/len(expected),'precision':hit/len(pred),'original_core_frames':len(expected),
                  'original_core_frames_present':present,'query_samples_from_original_core':samples,
                  'path_mean_original_frame_error':float(err.mean()),'path_exact_fraction':float((err==0).mean()),'state':anchor['state']}
            covered.update(range(lo,hi+1))
        if d['kind']=='control':
            olda=baseline_anchors[sid];oldcenter=int(indices[olda['query_center']])
            audit={**corr,'old_center':olda['query_center'],'old_center_original':oldcenter,
              'old_center_in_original_core':oldcenter in expected,'old_supported':olda['supported'],'old_unique':olda['unique']}
            computed.append(audit);correspondence_rows.append({'name':name,**audit})
    check('outside_core_frames_preserved',d['outside_core_candidates']==[i for i in range(len(q['x'])) if i not in covered] and d['outside_is_not_normal_or_discarded'] is True,name=name)
    if d['kind']=='control':
        raw=[{k:v for k,v in c.items() if not k.startswith('old_')} for c in computed]
        equal(raw,d['numerical_correspondence'],name+'/numerical_correspondence')
        record={'name':name,'alias':a,'role':rows[a]['role'],'stages':computed,
          'new_states':dict(Counter(z['state'] for z in d['anchors'])),
          'old_reverse':sum(z['state']=='reverse' for z in old['free_order']['relations']),
          'new_reverse':sum(z['state']=='reverse_candidate' for z in d['relations']),
          'new_overlaps':sum(z['state']=='overlapping_occurrences' for z in d['relations']),'seconds':d['seconds']}
        controls[d['control']].append(record)
    else:
        pairs.append({'query':a,'reference':b,'role':rows[a]['role'],'seconds':d['seconds'],
          'states':dict(Counter(z['state'] for z in d['anchors'])),'relations':d['relations'],'old_relations':old['free_order']['relations']})
    rc=report_cases[name]
    equal(indices.tolist(),rc['original_indices'],name+'/report_original_indices')
    equal((q['timestamp_us']/1e6).tolist(),rc['query_pts'],name+'/report_query_pts')
    equal((arrays[a]['timestamp_us']/1e6).tolist(),rc['original_query_pts'],name+'/report_original_pts')
    equal((arrays[b]['timestamp_us']/1e6).tolist(),rc['reference_pts'],name+'/report_ref_pts')
    case_checks.append({'name':name,'kind':d['kind'],'query_frames':len(q['x']),'reference_frames':len(arrays[b]['x']),
                        'core_denominator':len(cores),'outside_frames_retained':len(d['outside_core_candidates'])})

metrics={}
for control,records in controls.items():
    ss=[s for r in records for s in r['stages']];den=len(ss)
    metrics[control]={'cases':len(records),'stages':den,
      'old_center_hits':sum(s['old_center_in_original_core'] for s in ss),
      'new_center_hits':sum(s['center_in_original_core'] for s in ss),
      'new_mean_core_recall':sum(s['recall'] for s in ss)/den,
      'new_mean_core_precision':sum(s['precision'] if s['precision'] is not None else 0 for s in ss)/den,
      'old_supported_hits':sum(s['old_center_in_original_core'] and s['old_supported'] for s in ss),
      'new_fully_supported_hits':sum(s['center_in_original_core'] and s['state']=='supported_candidate' for s in ss),
      'new_states':dict(Counter(s['state'] for s in ss)),
      'old_reverse_relations':sum(r['old_reverse'] for r in records),
      'new_reverse_relations':sum(r['new_reverse'] for r in records),
      'new_overlaps':sum(r['new_overlaps'] for r in records)}
saved=read(U/'SUMMARY.json');equal(metrics,saved['controls'],'summary/controls')
equal(dict(controls),read(U/'CONTROL_EVALUATION.json')['controls'],'control_evaluation')
equal(pairs,read(U/'PAIR_EVALUATION.json')['pairs'],'pair_evaluation')
check('person_pair_count_36',len(pairs)==36 and saved['person_pairs']==36)
check('control_count_126',sum(len(v) for v in controls.values())==126)
times=defaultdict(float);states=Counter()
for p in pairs:
    if p['role']!='fit':times[p['query']]+=p['seconds'];states.update(p['states'])
equal(dict(times),saved['development_pair_seconds_by_query'],'summary/dev_times')
equal(dict(states),saved['new_person_pair_states'],'summary/dev_states')
raw_gate=all(metrics[k]['new_center_hits']>=metrics[k]['old_center_hits'] and metrics[k]['new_reverse_relations']<=metrics[k]['old_reverse_relations'] for k in ['identity','speed_075'])
supported_gate=all(metrics[k]['new_fully_supported_hits']>=metrics[k]['old_supported_hits'] for k in ['identity','speed_075'])
equal({'raw_same_source_no_regression':raw_gate,'supported_correct_coverage_no_regression':supported_gate,'not_enough_for_semantic_accuracy':True},saved['gates'],'summary/gates')
check('saved_results_and_inputs_unchanged',all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in pins.items()))
targeted=[]
for anchor in report_cases['P063_identity']['anchors']:
    if anchor['segment_id'] not in ['P063_s05','P063_s06']:continue
    best=anchor['best']
    targeted.append({'segment_id':anchor['segment_id'],'reference_core':anchor['reference_core'],
      'reference_context':anchor['reference_context'],'new_query_core':best['query_core'],
      'new_query_window':best['query_window'],'new_cost':best['cost'],'new_state':anchor['state'],
      'core_supported':best['core_supported'],'context_support':best['context_support'],
      'old_query_center':anchor['old_center'],'old_query_window':anchor['old_window'],
      'interpretation':'Known numerical occurrence correspondence recovered and supported; not hidden by withholding. No linguistic correctness claim.'})
out={'status':'PASS_SAVED_CORRESPONDENCE_AND_FIXED_DENOMINATORS' if not failures else 'FAIL_SAVED_CORRESPONDENCE_AND_FIXED_DENOMINATORS',
 'author':'Codex agent A; separate scalar/set arithmetic from producer, same model family',
 'checks':checks,'failures':failures,'result_count':len(manifest['results']),'control_cases':sum(len(v) for v in controls.values()),
 'person_pairs':len(pairs),'control_core_denominator':len(correspondence_rows),'metrics_recomputed':metrics,'gates_recomputed':{'raw':raw_gate,'supported':supported_gate},
 'case_scope':case_checks,'correspondence':correspondence_rows,'input_and_result_pins':pins,
 'P063_identity_specific_recovery':targeted,
 'material_report_findings':[
   'identity raw original-core center hits44->45/45 and macro recall/precision1.0; fully supported hits44->36/45, so support non-regression gate fails.',
   'speed075 raw45->45/45 but fully supported45->36/45. Raw numerical correspondence and supported coverage must remain separate.',
   'swap-adjacent-block controls raw center hits40->34/43 and reported reverse relations14->2, with3overlaps. This is a major adverse/context-break result, not an overall improvement.',
   'All45 stages in right-hand-unobserved controls are observation_insufficient despite45 raw center hits; a raw numerical location is not sufficient motion evidence.'
 ],
 'validator_runtime_recovery':'CORRESPONDENCE_FIRST_FAILURE.json preserves helper argument-name collision; only independent validator helper renamed before successful run.',
 'execution':{'search_calls':0,'DTW_calls':0,'model_calls':0,'training':0,'old_files_written':0},
 'interpretation_limits':[
   'Center hits, recall and precision measure known original-index correspondence on synthetic saved-feature controls; not sign-language accuracy, semantic errors or obligatory stage correctness.',
   'All original core frames remain in the recall denominator, including deliberately omitted or fast-resampling-dropped frames. Duplicated query samples are separately counted.',
   'No-candidate/unsupported stages remain in the common stage denominator; missing precision is zero only for the published macro average, not a semantic false label.',
   'Old supported hits use the old static-template support flag; new fully supported hits also require context/ambiguity rules. These are different evidence conditions, reported separately.',
   'For different-person pairs, original-frame identity is not ground truth and no center-hit accuracy was computed.',
   'This audit verifies stored correspondence and denominator arithmetic; it does not rerun search or prove globally optimal candidates.'],
 'all_read_artifacts_unchanged':True}
(U/'review'/'CORRESPONDENCE_VERIFICATION.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':out['status'],'checks':checks,'failure_count':len(failures),'first_failures':failures[:5],
 'results':len(manifest['results']),'control_core_denominator':len(correspondence_rows),'metrics':metrics},ensure_ascii=False))
if failures:raise SystemExit(1)
