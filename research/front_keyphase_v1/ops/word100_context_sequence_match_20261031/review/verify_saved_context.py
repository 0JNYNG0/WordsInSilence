"""Saved-path verification only: no evaluator import, search, DTW or model call."""
from pathlib import Path
from collections import Counter
import datetime, hashlib, importlib.util, json, math
import numpy as np

U=Path(__file__).resolve().parents[1]
OLD=U.parent/'word100_relational_sequence_compare_20261031'
PHASE=U.parent/'word100_core_stage_evidence_20261031'
ORACLE=OLD/'review/verify_saved_relational.py'
spec=importlib.util.spec_from_file_location('prior_independent_arithmetic',ORACLE)
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
GROUPS=oracle.GROUPS
counts=Counter();failures=[];largest_error=0.;result_rows=[]

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(label,actual,expected=True,category='arithmetic'):
    global largest_error
    counts[category]+=1
    if isinstance(expected,(float,np.floating)):
        good=isinstance(actual,(float,int)) and math.isfinite(actual) and math.isclose(actual,expected,rel_tol=3e-9,abs_tol=3e-11)
        if isinstance(actual,(float,int)) and math.isfinite(actual):largest_error=max(largest_error,abs(actual-expected))
    else:good=actual==expected
    if not good:failures.append(dict(check=label,actual=actual,expected=expected,category=category))

def fields(label,actual,expected,category):
    check(label+'/field_set',set(actual)==set(expected),category=category)
    for k,v in expected.items():check(label+'/'+k,actual.get(k),v,category)

def cell_cost(q,r,i,j):
    group_costs=[]
    for group in GROUPS:
        x,xok=q[group];y,yok=r[group]
        dimensions=[k for k,valid in enumerate(yok[j]) if valid]
        if dimensions:
            total=math.fsum(min(abs(float(x[i,k])-float(y[j,k])),1.) if xok[i,k] else 1. for k in dimensions)
            group_costs.append(total/len(dimensions))
    return math.fsum(group_costs)/len(group_costs) if group_costs else 0.

def verify_pack(label,p,q,r,core,context):
    path=np.asarray(p['path']);n=len(q[GROUPS[0]][0]);m=len(r[GROUPS[0]][0])
    check(label+'/integer_path',bool(np.issubdtype(path.dtype,np.integer)),category='path')
    # Explicit shape/index checks precede all indexing.
    check(label+'/path_two_columns',bool(path.ndim==2 and path.shape[1]==2),category='path')
    if path.ndim!=2 or path.shape[1]!=2 or not len(path):raise ValueError('Malformed saved path '+label)
    path=path.astype(int)
    check(label+'/indices_in_bounds',bool(((path[:,0]>=0)&(path[:,0]<n)&(path[:,1]>=0)&(path[:,1]<m)).all()),category='path')
    lo,hi=p['query_window'];ra,rb=context;a,b=core
    check(label+'/endpoints',[path[0].tolist(),path[-1].tolist()],[[lo,ra],[hi,rb]],'path')
    check(label+'/every_selected_query',sorted(set(path[:,0].tolist())),list(range(lo,hi+1)),'path')
    check(label+'/every_reference_template',sorted(set(path[:,1].tolist())),list(range(ra,rb+1)),'path')
    check(label+'/window_bounds',0<=lo<=hi<n,category='path')
    nq=hi-lo+1;nr=rb-ra+1
    check(label+'/window_ratio',max(2,math.ceil(.5*nr))<=nq<=math.floor(2*nr),category='path')
    steps=np.diff(path,axis=0).tolist();run=0;prev=None;maxrun=0;weights=[2]
    legal=True
    for step in steps:
        pair=tuple(step);legal &= pair in ((1,1),(1,0),(0,1))
        if pair==(1,1):run=0;prev=None;weights.append(2)
        else:
            run=run+1 if pair==prev else 1;prev=pair;maxrun=max(maxrun,run);weights.append(1)
    check(label+'/unit_monotone_steps',legal,category='path')
    check(label+'/max_axis_run',maxrun<=2,category='path')
    check(label+'/symmetric_weight_sum',sum(weights),nq+nr,'path')
    objective=math.fsum(w*cell_cost(q,r,int(i),int(j)) for w,(i,j) in zip(weights,path))/(nq+nr)
    check(label+'/normalized_saved_path_cost',p['cost'],objective,'cost')
    coreq=sorted({int(i) for i,j in path if a<=j<=b})
    check(label+'/recovered_core_span',p['query_core'],[coreq[0],coreq[-1]],'path')
    independent={}
    for phase,ids in [('entry',range(ra,a)),('core',range(a,b+1)),('exit',range(b+1,rb+1))]:
        if not len(ids):
            check(label+'/'+phase,p['phases'][phase],None,'support');independent[phase]=None
        else:
            independent[phase]=oracle.summary(q,r,path,ids)
            check(label+'/'+phase+'/group_set',set(p['phases'][phase])==set(GROUPS),category='support')
            for group in GROUPS:fields(label+'/'+phase+'/'+group,p['phases'][phase][group],independent[phase][group],'support')
    def supported(phase):
        available=[] if independent[phase] is None else [v for v in independent[phase].values() if v['expected']>0]
        return bool(available) and all(v['sufficient'] for v in available)
    core_support=supported('core');context_support={k:supported(k) for k in ['entry','exit']}
    check(label+'/core_supported',p['core_supported'],core_support,'support')
    check(label+'/context_support',p['context_support'],context_support,'support')
    return core_support,context_support,maxrun

def main():
    manifest=read(U/'RESULT_MANIFEST.json');status=read(U/'STATUS.json');freeze=read(U/'FROZEN.json')
    check('completed_manifest',manifest['status'],'EXECUTION_COMPLETE_PENDING_VERIFICATION','provenance')
    check('complete_progress',status['done'],162,'provenance');check('total_progress',status['total'],162,'provenance')
    check('no_active_lock',not (U/'RUNNING.json').exists(),category='provenance')
    frozen_sha=sha(U/'FROZEN.json');manifest_sha=sha(U/'RESULT_MANIFEST.json')
    check('manifest_frozen_link',manifest['frozen_sha256'],frozen_sha,'provenance')
    for p,h in freeze['pins'].items():check('frozen_pin/'+p,sha(p),h,'provenance')
    rows={x['alias']:x for x in read(U/'AUTHORIZATION.json')['rows']}
    oldrows={x['alias']:x for x in read(OLD/'AUTHORIZATION.json')['rows']}
    clips={x['alias']:x for x in read(PHASE/'PHASE_DATA.json')['clips']}
    arrays={a:oracle.load_array(r['input']) for a,r in rows.items()}
    feature={a:oracle.independent_features(arrays[a],oldrows[a]['feature_names']) for a in rows}
    for a,row in rows.items():
        check(a+'/raw_sha',sha(row['input']),row['input_sha256'],'provenance')
        check(a+'/all_native_frame_ids',arrays[a]['frame_id'].tolist(),list(range(row['frames'])),'scope')
        check(a+'/strict_native_pts',bool((np.diff(arrays[a]['timestamp_us'])>0).all()),category='scope')
    expected_jobs={p.stem:('control',p) for p in (OLD/'controls').glob('*.json')}
    expected_jobs.update({p.stem:('person_pair',p) for p in (OLD/'pairs').glob('B_same_path_*.json')})
    check('manifest_jobs',sorted(x['name'] for x in manifest['results']),sorted(expected_jobs),'scope')
    check('job_count',len(manifest['results']),162,'scope')
    check('job_kind_counts',dict(Counter(x['kind'] for x in manifest['results'])),{'control':126,'person_pair':36},'scope')
    check('all_result_files_manifested',sorted(p.stem for p in (U/'results').glob('*.json')),sorted(expected_jobs),'scope')
    state_counts=Counter();relation_counts=Counter();total_paths=0;total_anchors=0
    for number,item in enumerate(manifest['results'],1):
        name=item['name'];d=read(item['file']);kind,baseline_path=expected_jobs[name]
        check(name+'/result_hash',sha(item['file']),item['sha256'],'provenance')
        check(name+'/baseline_path',str(Path(d['baseline_path'])),str(baseline_path),'provenance')
        check(name+'/baseline_hash',sha(baseline_path),d['baseline_sha256'],'provenance')
        check(name+'/frozen_link',d['frozen_sha256'],frozen_sha,'provenance')
        check(name+'/kind',d['kind'],kind,'scope');check(name+'/name',d['name'],name,'scope')
        baseline=read(baseline_path);a=d['query'];b=d['reference'];r=feature[b]
        if kind=='control':
            rawpath=baseline_path.with_suffix('.npz');raw=oracle.load_array(rawpath)
            check(name+'/control_reference_is_self',a,b,'scope')
            check(name+'/control_identity',d['control'],baseline['control'],'scope')
            check(name+'/control_sha',sha(rawpath),baseline['array_sha256'],'provenance')
            q=oracle.independent_features(raw,oldrows[a]['feature_names'])
        else:
            rawpath=Path(rows[a]['input']);raw=arrays[a];q=feature[a]
            check(name+'/pair_aliases',[a,b],[baseline['query'],baseline['reference']],'scope')
            check(name+'/same_expression',rows[a]['word_id'],rows[b]['word_id'],'scope')
            check(name+'/different_person',rows[a]['person']!=rows[b]['person'],category='scope')
            check(name+'/reference_is_fit',rows[b]['role'],'fit','scope')
        check(name+'/query_array_hash',d['query_array_sha256'],sha(rawpath),'provenance')
        n=len(raw['x']);m=len(arrays[b]['x']);pts=arrays[b]['timestamp_us']/1e6
        for field,value in [('query_frames',n),('reference_frames',m),('query_search_scope',n),('reference_scope',m),('semantic_correctness',False),('forced_interstage_order',False),('original_query_frames_preserved_in_global_baseline',True),('outside_is_not_normal_or_discarded',True)]:check(name+'/'+field,d[field],value,'scope')
        check(name+'/finite_nonnegative_time',math.isfinite(d['seconds']) and d['seconds']>=0,category='scope')
        segments=[s for s in clips[b]['segments'] if s['role']=='core_candidate']
        check(name+'/fixed_core_targets',[s['segment_id'] for s in d['anchors']],[s['segment_id'] for s in segments],'scope')
        states=[];covered=set();paths_in_result=0
        for segment,anchor in zip(segments,d['anchors']):
            label=name+'/'+segment['segment_id'];core=segment['core'];aa,bb=core
            requested=[float(pts[aa]-.2),float(pts[bb]+.2)]
            # Independent scalar selection from fixed reference original PTS.
            ref_ids=[i for i,t in enumerate(pts) if requested[0]<=t<=requested[1]]
            context=[ref_ids[0],ref_ids[-1]];ra,rb=context;nr=rb-ra+1
            check(label+'/core',anchor['reference_core'],core,'scope')
            check(label+'/context',anchor['reference_context'],context,'scope')
            lo=max(2,math.ceil(nr*.5));hi=min(n,math.floor(2*nr));eligible=sum(n-length+1 for length in range(lo,hi+1))
            if anchor['best'] is None:
                check(label+'/no_candidate_count',eligible,0,'path');check(label+'/no_candidate_state',anchor['state'],'no_duration_candidate','state');check(label+'/no_alternatives',anchor['alternatives'],[],'state');states.append(anchor['state']);continue
            check(label+'/enumerated_window_count',anchor['searched_windows'],eligible,'path')
            check(label+'/requested_context',anchor['requested_context_seconds'],requested,'scope')
            check(label+'/available_context',anchor['available_context_seconds'],[float(pts[ra]),float(pts[rb])],'scope')
            edge=[requested[0]<pts[0],requested[1]>pts[-1]]
            check(label+'/clipped_flags',anchor['clipped_at_video_edge'],edge,'scope')
            best=anchor['best'];core_ok,ctx_ok,_=verify_pack(label+'/best',best,q,r,core,context);paths_in_result+=1
            span=best['query_core'];covered.update(range(span[0],span[1]+1))
            check(label+'/at_most_one_saved_disjoint_alternative',len(anchor['alternatives'])<=1,category='ties')
            for alt_index,alt in enumerate(anchor['alternatives']):
                verify_pack(label+'/alternative'+str(alt_index),alt,q,r,core,context);paths_in_result+=1
                other=alt['query_core'];check(label+'/core_disjoint',other[1]<span[0] or other[0]>span[1],category='ties')
                check(label+'/best_cost_no_larger',best['cost']<=alt['cost']+3e-11,category='ties')
            gap=anchor['alternatives'][0]['cost']-best['cost'] if anchor['alternatives'] else None
            check(label+'/disjoint_gap',anchor['second_disjoint_gap'],gap,'ties')
            ties=anchor['tied_core_spans']
            check(label+'/best_span_in_ties',span in ties,category='ties')
            check(label+'/unique_tie_spans',len(ties),len(set(tuple(x) for x in ties)),'ties')
            check(label+'/tie_span_bounds',all(0<=x[0]<=x[1]<n for x in ties),category='ties')
            check(label+'/ties_no_more_than_windows',len(ties)<=eligible,category='ties')
            disjoint=any(x[1]<span[0] or x[0]>span[1] for x in ties)
            check(label+'/disjoint_tie_field',anchor['disjoint_exact_tie'],disjoint,'ties')
            check(label+'/disjoint_tie_alternative_witness',disjoint,gap is not None and gap<=1e-9,'ties')
            check(label+'/internal_path_ties_limitation',anchor['within_window_path_ties_enumerated'],False,'ties')
            context_ok=all(ctx_ok.values()) and not any(edge)
            expected_state='observation_insufficient' if not core_ok else 'context_incomplete' if not context_ok else 'ambiguous_occurrence' if disjoint else 'supported_candidate'
            check(label+'/derived_state',anchor['state'],expected_state,'state');states.append(expected_state)
        relations=[]
        for left,right in zip(d['anchors'],d['anchors'][1:]):
            if any(x['best'] is None or x['state'] in ['observation_insufficient','context_incomplete'] for x in [left,right]):state='insufficient'
            elif any(x['state']=='ambiguous_occurrence' for x in [left,right]):state='ambiguous'
            else:
                x=left['best']['query_core'];y=right['best']['query_core'];state='forward_candidate' if x[1]<y[0] else 'reverse_candidate' if y[1]<x[0] else 'overlapping_occurrences'
            relations.append(dict(first=left['segment_id'],second=right['segment_id'],state=state))
        check(name+'/derived_relations',d['relations'],relations,'state')
        check(name+'/outside_candidates_preserved',d['outside_core_candidates'],sorted(set(range(n))-covered),'scope')
        state_counts.update(states);relation_counts.update(x['state'] for x in relations);total_paths+=paths_in_result;total_anchors+=len(d['anchors'])
        result_rows.append(dict(name=name,kind=kind,anchors=len(d['anchors']),saved_paths=paths_in_result,states=dict(Counter(states))))
        if number%20==0:print('verified',number,'/',len(manifest['results']),'failures',len(failures),flush=True)
    # Recheck pins after the read-only audit so concurrent changes cannot silently pass.
    for p,h in freeze['pins'].items():check('final_frozen_pin/'+p,sha(p),h,'provenance')
    check('manifest_unchanged_during_audit',sha(U/'RESULT_MANIFEST.json'),manifest_sha,'provenance')
    output=dict(status='PASS' if not failures else 'FAIL',reviewer='Codex rgb_labels_b; independent saved arithmetic, same model family',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=sum(counts.values()),checks_by_category=dict(counts),failures=len(failures),failure_details=failures,max_float_error=largest_error,results_checked=len(result_rows),anchors_checked=total_anchors,saved_best_and_alternative_paths_checked=total_paths,states=dict(state_counts),relations=dict(relation_counts),frozen_sha256=frozen_sha,manifest_sha256=manifest_sha,verifier_sha256=sha(__file__),independent_feature_and_mask_oracle_sha256=sha(ORACLE),no_project_evaluator_import=True,no_search_or_DTW_run=True,no_model_inference_or_training=True,original_index_and_summary_audit='separate agent A artifact',limitations=['Objective is independently reconstructed scalar arithmetic along saved paths; global/window optimality is not recomputed.','Tie-span bounds, disjoint witnesses and state arithmetic are checked, but exhaustive retrieval of every equally optimal window/path is not independently rerun.','Feature/mask arithmetic uses the earlier independent name-based review helper, not production relational.features/observed_summary.','Observed support and exact numeric correspondence are not real event identity, normal signing, required-stage completeness or educational accuracy.','Test count is arithmetic assertions over saved artifacts, not independent accuracy examples.'],result_checks=result_rows)
    with (U/'review/SAVED_VERIFICATION.json').open('x',encoding='utf8') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:output[k] for k in ['status','checks','failures','results_checked','anchors_checked','saved_best_and_alternative_paths_checked','max_float_error']}),flush=True)
    return 0 if not failures else 1

if __name__=='__main__':raise SystemExit(main())
