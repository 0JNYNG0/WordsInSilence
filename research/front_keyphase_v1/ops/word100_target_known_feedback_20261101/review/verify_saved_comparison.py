"""Independent saved-artifact arithmetic audit; no implementation import/inference."""
import json, hashlib, datetime, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

U=Path(__file__).resolve().parent.parent
OPS=U.parent
OLD=OPS/'word100_stage_feedback_cycle_20261031'
TARGETS=['R_index_only','R_open_four','L_open_four']
SEEDS=[42,43,44]
METHODS=['global_dtw','core_first']
AGREE='ESTIMATED_FEATURE_AGREEMENT'
DIFFER='ESTIMATED_FEATURE_DIFFERENCE_CANDIDATE'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
failures=[]
check_count=0
def check(name, condition):
    global check_count
    check_count+=1
    if not bool(condition):failures.append(name)
def near(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-8)
def counter_equal(a,b):return all(near(a.get(k,0),b.get(k,0)) for k in set(a)|set(b))
def state(q,r,qs,rs):
    if not rs:return 'REFERENCE_UNSUPPORTED'
    if not qs:return 'QUERY_UNSUPPORTED'
    if not math.isfinite(q) or not math.isfinite(r):return 'OUTPUT_UNAVAILABLE'
    if (0.2<q<0.8) or (0.2<r<0.8):return 'LOW_CONFIDENCE'
    if abs(q-r)>=0.6:return DIFFER
    return AGREE if (q>=0.5)==(r>=0.5) else 'UNCERTAIN_FEATURE_DIFFERENCE'

bank=read(U/'REFERENCE_BANK.json')
tasks=read(U/'COMPARATOR_TASKS.json')
frozen=read(U/'COMPARATOR_EXECUTION.json')
output=read(U/'COMPARISON_RESULTS.json')
phase=read(U/'REFERENCE_PHASES.json')['clips']
physical=read(OPS/'word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json')
physical={c['alias']:c for c in physical['clips']}
for path,digest in frozen['pins'].items():check('execution_pin:'+path,sha(path)==digest)
for path,digest in bank['protected'].items():check('production_protected:'+path,sha(path)==digest)
for path,digest in bank['source_pins'].items():check('bank_source:'+path,sha(path)==digest)
check('execution_identity',output['execution_sha256']==sha(U/'COMPARATOR_EXECUTION.json'))
check('code_identity',output['code_sha256']==sha(U/'comparator.py'))
check('all72_paths_sealed',len(tasks['saved_path_pins'])==72 and all(frozen['pins'].get(k)==v for k,v in tasks['saved_path_pins'].items()))
check('fixed_methods',frozen['methods']==METHODS)
check('fixed_seeds',frozen['seeds']==SEEDS)
check('no_training_or_inference',output['new_inference']==0 and output['new_training']==0)
check('no_education_or_fullsign',output['educational_accuracy'] is None and output['whole_sign_correctness'] is None)
check('bank_denominators',len(bank['words'])==20 and len(bank['development_queries'])==20 and output['bank_query_denominator']==20 and output['phase_supported_development_queries']==4 and output['phase_unsupported_development_queries']==16)
check('all20_actual_routes',len(output['routing_results'])==20)
for route,request in zip(output['routing_results'],bank['development_queries']):
    word=next(w for w in bank['words'] if w['word_id']==request['practice_word_id'])
    expected=[{'reference_alias':r['legacy_alias'],'status':'LOCAL_ESTIMATED_EVIDENCE' if r['phase_reference_available'] else 'PHASE_COMPARISON_UNSUPPORTED'} for r in word['references']]
    check('actual_route:'+route['practice_word_id'],route['practice_word_id']==request['practice_word_id'] and route['query']==(request['query']['legacy_alias'] or request['query']['alias']) and route['reference_statuses']==expected and route['whole_sign_verdict'] is None and route['has_phase_local_evidence']==any(r['phase_reference_available'] for r in word['references']))

info=read(OLD/'LEARNING_INPUTS.json');rows={r['alias']:r for r in info['rows']};rowindex={r['alias']:i for i,r in enumerate(info['rows'])}
indices=[info['targets'].index(t) for t in TARGETS]
prob={s:np.load(OLD/'runs'/f'pose_plus_signrep_{s}'/'probabilities.npy',mmap_mode='r',allow_pickle=False) for s in SEEDS}
cache={}
for alias in {t['query_alias'] for t in tasks['tasks']}:
    row=rows[alias]
    check('input_seal:'+alias,frozen['pins'].get(row['input'])==row['input_sha256']==sha(row['input']))
    with np.load(row['input'],allow_pickle=False) as z:a={k:z[k] for k in ['x','mask','frame_id','timestamp_us']}
    n=len(a['frame_id']);names=row['feature_names'];support=[]
    check('native_axis:'+alias,np.array_equal(a['frame_id'],np.arange(n)) and np.all(np.diff(a['timestamp_us'])>0))
    check('observed_finite:'+alias,np.isfinite(a['x'][a['mask']]).all())
    for target in TARGETS:
        side=target[0];wrist=[names.index(side+'_hand_wrist_'+axis) for axis in ['x','y']]
        fingers=[j for j,name in enumerate(names) if name.startswith(side+'_') and '_angle' in name and 'elbow' not in name]
        support.append(a['mask'][:,wrist].all(axis=1)&(a['mask'][:,fingers].sum(axis=1)>=10))
    a['support']=np.stack(support,axis=1);a['names']=names
    a['probs']={s:np.asarray(prob[s][rowindex[alias],:n])[:,indices] for s in SEEDS}
    cache[alias]=a
for alias,refphase in phase.items():
    check('fit_only_ref:'+alias,rows[alias]['role']=='fit' and physical[alias]['role']=='fit')
    check('source_regions_exact:'+alias,refphase['regions']==physical[alias]['regions'])

task_map={t['query_alias']:t for t in tasks['tasks']}
check('all16_tasks',len(output['results'])==len(task_map)==16 and {r['query'] for r in output['results']}==set(task_map))
groups={};perword={};projection_expected=[];allpairs=set();parents=set();children=0
for result in output['results']:
    qa=result['query'];q=cache[qa];task=task_map[qa];role=task['role'];word=next(w for w in bank['words'] if w['word_id']==task['practice_word_id'])
    check('explicit_target:'+qa,result['practice_word_id']==task['practice_word_id'] and result['test_role']==role)
    check('all3_refs:'+qa,[r['reference_alias'] for r in result['references']]==[r['legacy_alias'] for r in word['references']])
    check('guide_no_override:'+qa,result['guide_can_override_observation'] is False and result['mandatory_linguistic_rules_enabled'] is False and result['whole_sign_verdict'] is None)
    for ref in result['references']:
        ra=ref['reference_alias']
        if qa==ra:
            check('self_excluded:'+qa,ref['status']=='SELF_REFERENCE_EXCLUDED');continue
        check('local_ref:'+qa+'/'+ra,ref['status']=='LOCAL_ESTIMATED_EVIDENCE')
        r=cache[ra];refphase=phase[ra];pair=(qa,ra);allpairs.add(pair)
        pids=[p['segment_id'] for p in refphase['parents']]
        check('all_parent_ids:'+str(pair),ref['original_parent_ids']==pids)
        parents.update((qa,ra,p) for p in pids)
        check('all_original_regions:'+str(pair),ref['reference_original_regions']==refphase['regions'])
        observed=[x for x in refphase['regions'] if x['region_type']=='observed_physical_core']
        children+=len(observed)
        for path,digest in ref['saved_path_sources'].items():check('result_path_seal:'+path,digest==frozen['pins'].get(path))
        glob=read(OPS/'word100_relational_sequence_compare_20261031/pairs'/f'B_same_path_{qa}_{ra}.json')['path']
        core=read(OPS/'word100_core_context_split_20261101/results'/f'B_same_path_{qa}_{ra}.json')
        core={a['segment_id']:a for a in core['anchors']}
        check('both_methods:'+str(pair),list(ref['methods'])==METHODS)
        for method,methodout in ref['methods'].items():
            stages=methodout['stages'];order=methodout['projected_order'];spans=[];ambigs=[]
            check('all_child_events:'+str(pair)+method,[s['reference_event'] for s in stages]==observed)
            for stage,event in zip(stages,observed):
                parent=stage['parent_segment_id'];start,end=event['core'];expected=end-start+1
                anchor=core[parent];sourcepath=glob if method=='global_dtw' else (anchor['selected']['core_path'] if anchor.get('selected') else [])
                path=[p for p in sourcepath if start<=p[1]<=end]
                label=f'{qa}/{ra}/{method}/{event["id"]}'
                check('exact_saved_child_path:'+label,stage['saved_path']==path)
                check('no_truth_or_mandatory:'+label,stage['query_physical_annotation_used'] is False and stage['mandatory_stage'] is False and stage['unique_physical_correspondence_claim'] is False)
                by=defaultdict(list)
                for qi,ri in path:by[ri].append(qi)
                qframes=sorted({p[0] for p in path});spans.append([min(qframes),max(qframes)] if qframes else None)
                ambiguity=False if method=='global_dtw' else bool(anchor.get('disjoint_core_tie') or 'ambig' in str(anchor['state']).lower() or len(anchor.get('core_ties',[]))>1)
                ambigs.append(ambiguity);check('ambiguity_retained:'+label,stage['saved_candidate_ambiguity']==ambiguity)
                check('seed_order:'+label,[s['seed'] for s in stage['seeds']]==SEEDS)
                for saved in stage['seeds']:
                    seed=saved['seed'];name=f'{label}/{seed}';counts={t:Counter() for t in TARGETS}
                    check('nativeframe_denominator:'+name,[f['reference_frame'] for f in saved['frames']]==list(range(start,end+1)))
                    check('target_order:'+name,saved['target_order']==TARGETS)
                    for frame in saved['frames']:
                        ri=frame['reference_frame'];links=frame['links'];qis=by[ri]
                        check('reference_trace:'+name+str(ri),frame['reference_pts_us']==int(r['timestamp_us'][ri]) and frame['reference_probabilities']==[float(v) for v in r['probs'][seed][ri]] and frame['reference_input_support']==r['support'][ri].tolist())
                        check('query_path_links:'+name+str(ri),[l['query_frame'] for l in links]==qis)
                        for link,qi in zip(links,qis):
                            states=[state(float(q['probs'][seed][qi,j]),float(r['probs'][seed][ri,j]),bool(q['support'][qi,j]),bool(r['support'][ri,j])) for j in range(3)]
                            weight=1/len(qis)
                            check('query_trace:'+name+str(ri)+'/'+str(qi),link['query_pts_us']==int(q['timestamp_us'][qi]) and link['query_probabilities']==[float(v) for v in q['probs'][seed][qi]] and link['query_input_support']==q['support'][qi].tolist() and link['states']==states and near(link['weight_per_feature'],weight))
                            for target,st in zip(TARGETS,states):counts[target][st]+=weight
                        if not links:
                            for j,target in enumerate(TARGETS):counts[target]['NO_SAVED_CORRESPONDENCE' if r['support'][ri,j] else 'REFERENCE_UNSUPPORTED']+=1
                    key=f'{role}/{method}/{seed}'
                    g=groups.setdefault(key,dict(pair_ids=set(),parent_ids=set(),child_events=0,expected_feature_frame_cells=0,confidence_qualified_weight=0.,input_supported_weight=0.,agreement_weight=0.,difference_candidate_weight=0.,retained_despite_global_gate_weight=0.,candidate_child_events=0))
                    g['pair_ids'].add(pair);g['parent_ids'].add((qa,ra,parent));g['child_events']+=1;candidate=False
                    for j,target in enumerate(TARGETS):
                        count=counts[target];f=saved['features'][target];usable=count[AGREE]+count[DIFFER];known=usable+count['LOW_CONFIDENCE']+count['UNCERTAIN_FEATURE_DIFFERENCE'];gate=bool(q['support'][:,j].all() and r['support'][:,j].all() and np.isfinite(q['probs'][seed][:,j]).all() and np.isfinite(r['probs'][seed][:,j]).all())
                        check('all_weights:'+name+target,near(sum(count.values()),expected) and counter_equal(f['weighted_states'],count))
                        check('summary_values:'+name+target,f['expected_reference_frames']==expected and near(f['confidence_qualified_weight'],usable) and near(f['input_supported_and_finite_weight'],known) and near(f['input_coverage'],known/expected) and near(f['confidence_coverage'],usable/expected) and f['global_complete_input_gate_counterfactual']==gate and near(f['local_qualified_weight_retained_despite_global_gate'],0 if gate else usable) and f['contains_difference_candidate']==(count[DIFFER]>0) and f['educational_warning'] is None)
                        for k,v in [('expected_feature_frame_cells',expected),('confidence_qualified_weight',usable),('input_supported_weight',known),('agreement_weight',count[AGREE]),('difference_candidate_weight',count[DIFFER]),('retained_despite_global_gate_weight',0 if gate else usable)]:g[k]+=v
                        candidate|=count[DIFFER]>0
                    g['candidate_child_events']+=candidate
                for relation in stage['measured_relations']:
                    feature=relation['feature'];qj=q['names'].index(feature);rj=r['names'].index(feature);known=0.
                    check('relation_denominator:'+label+feature,[f['reference_frame'] for f in relation['frames']]==list(range(start,end+1)) and relation['expected_reference_frames']==expected and relation['linguistic_tolerance'] is None and relation['difference_warning'] is None)
                    for frame in relation['frames']:
                        ri=frame['reference_frame'];rv=float(r['x'][ri,rj]);rok=bool(r['mask'][ri,rj]) and math.isfinite(rv)
                        check('relation_reference:'+label+str(ri)+feature,frame['reference_value']==(rv if rok else None) and [l['query_frame'] for l in frame['links']]==by[ri])
                        for link in frame['links']:
                            qi=link['query_frame'];qv=float(q['x'][qi,qj]);qok=bool(q['mask'][qi,qj]) and math.isfinite(qv);weight=1/len(by[ri]);delta=qv-rv if rok and qok else None
                            check('relation_query:'+label+str(ri)+feature,link['query_value']==(qv if qok else None) and link['raw_difference']==delta and near(link['weight'],weight))
                            if rok and qok:known+=weight
                    check('relation_known:'+label+feature,near(relation['jointly_observed_weight'],known))
                counts=Counter();trace=[]
                for ri in range(start,end+1):
                    if not by[ri]:counts['unmapped']+=1;trace.append(dict(reference_frame=ri,query_frame=None,status='unmapped',weight=1));continue
                    for qi in by[ri]:
                        hits=[h for h in physical[qa]['regions'] if h['outer'][0]<=qi<=h['outer'][1]];st='unreviewed_or_uncertain'
                        if len(hits)==1:
                            h=hits[0];certain=h.get('region_type')=='observed_physical_core' and h.get('core') and h['core'][0]<=qi<=h['core'][1] and h.get('kind') is not None
                            st=('same_observed_kind' if h['kind']==event['kind'] else 'different_observed_kind') if certain else 'boundary_or_uncertain'
                        w=1/len(by[ri]);counts[st]+=w;trace.append(dict(reference_frame=ri,query_frame=qi,status=st,weight=w))
                projection_expected.append((qa,ra,role,method,event['id'],expected,dict(counts),trace))
            check('order_event_count:'+str(pair)+method,len(order['events'])==len(stages) and order['all_event_instances_retained'] is True and order['required_repetition_count'] is None and order['independent_correct_order_claim'] is False and order['method_forces_monotonic_order']==(method=='global_dtw'))
            check('order_spans:'+str(pair)+method,[e['projected_query_span'] for e in order['events']]==spans)
            for i,relation in enumerate(order['relations']):
                a,b=spans[i:i+2];st='UNMAPPED_EVENT' if a is None or b is None else ('FORWARD_DISJOINT' if a[1]<b[0] else 'REVERSE_DISJOINT' if b[1]<a[0] else 'OVERLAPPING_PROJECTIONS');ambig=ambigs[i] or ambigs[i+1]
                check('order_relation:'+str(pair)+method+str(i),relation['selected_projection_relation']==st and relation['status']==('AMBIGUOUS_CANDIDATE_ORDER' if ambig else st) and relation['source_candidate_ambiguity']==ambig)

for key,g in groups.items():
    g['pairs']=len(g.pop('pair_ids'));g['original_parent_instances']=len(g.pop('parent_ids'))
    g['confidence_coverage']=g['confidence_qualified_weight']/g['expected_feature_frame_cells'];g['input_coverage']=g['input_supported_weight']/g['expected_feature_frame_cells']
    check('aggregate:'+key,counter_equal(g,output['summary'][key]))
check('summary_keys',set(groups)==set(output['summary']))
check('pair_parent_child_denominators',len(allpairs)==36 and len(parents)==117 and children==144)
for method in METHODS:
    for seed in SEEDS:check('cell_denominator:'+method+str(seed),sum(v['expected_feature_frame_cells'] for k,v in groups.items() if k.endswith(f'/{method}/{seed}'))==3240)
projection=read(U/'PROJECTION_AUDIT.json')
check('projection_count',len(projection['events'])==len(projection_expected)==288)
projection_groups=defaultdict(Counter)
for saved,exp in zip(projection['events'],projection_expected):
    qa,ra,role,method,eventid,expected,counts,trace=exp
    check('posthoc_projection:'+qa+ra+method+eventid,(saved['query'],saved['reference'],saved['role'],saved['method'],saved['reference_event'],saved['expected_reference_frames'])==exp[:6] and counter_equal(saved['weighted_states'],counts) and saved['trace']==trace and saved['query_annotations_used_for_runtime'] is False and saved['same_kind_does_not_validate_shape_or_semantics'] is True)
    projection_groups[role+'/'+method].update(counts)
controls=read(U/'COMPARATOR_CONTROLS.json')
check('controls_only_technical',controls['status']=='PASS_TECHNICAL_ONLY' and controls['actual_learner_errors']==0 and controls['new_inference']==0 and controls['automatic_alignment_on_transformed_input_tested'] is False)
check('all_control_cases',all(c['pass'] is not False for c in controls['cases']) and controls['executed']==sum(c['pass'] is not None for c in controls['cases']) and controls['unsupported']==sum(c['pass'] is None for c in controls['cases']))
result={'status':'PASS_INDEPENDENT_STORED_COMPARISON_ARITHMETIC_AND_PROTECTED_PINS' if not failures else 'FAIL_INDEPENDENT_STORED_COMPARISON_AUDIT','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reviewer':'Codex semantic_data_audit','check_count':check_count,'failure_count':len(failures),'failures':failures,'execution_pin_count':len(frozen['pins']),'protected_pin_count':len(bank['protected']),'pair_count':len(allpairs),'original_parent_instances':len(parents),'child_instances_per_method_seed':children,'reference_child_frames_per_method_seed':1080,'feature_cells_per_method_seed':3240,'all_methods_seeds_feature_cells':19440,'independent_summary':groups,'independent_projection_audit_summary':dict(projection_groups),'controls_executed':controls['executed'],'controls_unsupported':controls['unsupported'],'artifact_pins':{str(U/n):sha(U/n) for n in ['COMPARATOR_EXECUTION.json','comparator.py','COMPARISON_RESULTS.json','PROJECTION_AUDIT.json','COMPARATOR_CONTROLS.json']},'audit_code_sha256':sha(__file__),'limitations':['No model, feature extractor, alignment search, or training code was imported or executed.','Stored automatic path choice is conditional and ambiguity metadata remains; forward global DTW does not establish correct semantic order.','Local estimated feature agreement is not educational correctness, and same observed physical kind does not validate shape or linguistic semantics.','Historical public development is not fresh participant or actual learner-error validation.','Only 4 of 20 bank words have physical phase scope; absent phases are not passing comparisons.'],'new_training':0,'new_model_inference':0,'old_score_overwrite':False}
with (U/'review/COMPARATOR_RESULTS_VERIFICATION.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({'status':result['status'],'checks':check_count,'failures':failures[:20],'summary':groups,'projection':dict(projection_groups),'controls':(controls['executed'],controls['unsupported'])},ensure_ascii=True))
