"""Synthetic split policy tests only. No public arrays, model or old result search."""
from pathlib import Path
from contextlib import contextmanager
import datetime, hashlib, json, math, sys
import numpy as np
U=Path(__file__).resolve().parents[1];sys.path.insert(0,str(U))
import split_match as sm
checks=[];cases=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(name,value):checks.append(dict(name=name,pass_=bool(value)))
def data(values,mask=None):
    values=np.asarray(values,float).reshape(-1,1)
    return {'toy':(values,np.ones_like(values,bool) if mask is None else np.asarray(mask,bool).reshape(-1,1))}
def summary(q,r,path,ids):
    # Scalar reference-cell arithmetic independent of production observed_summary.
    x,xok=q['toy'];y,yok=r['toy'];wanted=set(int(v) for v in ids)
    pairs=[(int(i),int(j)) for i,j in path if int(j) in wanted]
    maxima={};missing=0;shared_query=set()
    for i,j in pairs:
        if yok[j,0]:
            if xok[i,0]:maxima[j]=max(maxima.get(j,0.),min(abs(float(x[i,0]-y[j,0])),1.));shared_query.add(i)
            else:missing+=1
    expected=sum(bool(yok[j,0]) for j in wanted);count=len(maxima);qids=sorted(set(i for i,j in pairs))
    return {'toy':dict(expected=expected,compared=count,reference_possible=len(wanted),reference_unobserved=len(wanted)-expected,path_missing_cells=missing,coverage=count/expected if expected else None,mean_difference=math.fsum(maxima.values())/count if count else None,peak_difference=max(maxima.values()) if count else None,sufficient=bool(expected and count/expected>=.75 and missing==0 and expected>=2 and len(shared_query)>=2),distinct_shared_query_frames=len(shared_query),query_unobserved_frames=[i for i in qids if not xok[i,0]])}
def old(span,state='supported_candidate'):
    return {'anchors':[dict(segment_id='arbitrary',state=state,best={'query_core':span})]}
def run_case(name,q,r,core,previous=None,qpts=None,rpts=None):
    qn=len(q['toy'][0]);rn=len(r['toy'][0])
    if qpts is None:qpts=np.arange(qn)*100000
    if rpts is None:rpts=np.arange(rn)*100000
    result=sm.match(q,r,[dict(role='core_candidate',core=core,segment_id='arbitrary')],qpts,rpts,previous or old(core),['toy'],summary)
    a=result['anchors'][0];cases.append(dict(name=name,result=result))
    check(name+'/no_semantic_verdict',result['semantic_correctness'] is False)
    if a['selected']:
        check(name+'/selection_is_core_minimum',a['selected']['core_cost']<=a['minimum_core_cost']+sm.EPS)
        check(name+'/selected_in_tie_set',any(t['query_core']==a['selected']['query_core'] for t in a['core_ties']))
        check(name+'/core_path_ends_fixed',a['selected']['core_path'][0]==[a['selected']['query_core'][0],core[0]] and a['selected']['core_path'][-1]==[a['selected']['query_core'][1],core[1]])
        for t in a['core_ties']:
            for phase in t['context']['phases'].values():
                if phase['path']:
                    p=np.asarray(phase['path']);check(name+'/context_keeps_all_points',set(p[:,0])==set(range(phase['query_range'][0],phase['query_range'][1]+1)) and set(p[:,1])==set(range(phase['reference_range'][0],phase['reference_range'][1]+1)))
        check(name+'/core_not_moved_flags',all(t['context']['core_not_moved'] for t in a['core_ties']))
    f=sm.feedback(a);check(name+'/feedback_never_sign_correct_or_error',f['sign_correct'] is None and f['sign_error'] is None and f['correction_instruction'] is None)
    return a,result

@contextmanager
def policy_inputs(ranked,complete=True,context_costs=None):
    saved=(sm.windows,sm.recover,sm.anchored_context)
    sm.windows=lambda cost:list(ranked)
    def recover(cost,start,length):
        m=cost.shape[1]
        # Synthetic legal monotone witnesses used only to isolate selection policy.
        if length>=m:return np.column_stack([np.arange(start,start+length),np.floor(np.arange(length)*m/length).astype(int)])
        return np.column_stack([start+np.floor(np.arange(m)*length/m).astype(int),np.arange(m)])
    sm.recover=recover
    def context(q,r,c,cs,qs,*args):
        start=qs[0];ok=complete.get(start,True) if isinstance(complete,dict) else complete
        return dict(complete=ok,mean_cost=(context_costs or {}).get(start,0.),core_not_moved=True,phases={})
    sm.anchored_context=context
    try:yield
    finally:sm.windows,sm.recover,sm.anchored_context=saved

def main():
    before=sha(U/'split_match.py')
    rv=np.full(30,.99);rv[4:6]=[.45,.54];rv[6:10]=[.11,.73,.29,.82];rv[10:12]=[.37,.63]
    r=data(rv)
    a,_=run_case('unique_core_ignores_old_disjoint',data(rv),r,[6,9],old([18,21]))
    check('unique_core_exact',a['selected']['query_core']==[6,9] and len(a['core_ties'])==1 and not a['context_tie_resolved'])
    check('old_disjoint_is_hold',a['old_context_disjoint'] and 'core_context_localization_disagreement' in a['diagnostic_hold_reasons'])
    qv=np.full(30,.99);qv[6:10]=rv[6:10];qv[4:6]=[.95,.96];qv[10:12]=[.97,.98];qv[16:18]=rv[4:6];qv[18:22]=rv[6:10];qv[22:24]=rv[10:12]
    a,_=run_case('two_exact_cores_unique_context',data(qv),r,[6,9])
    check('two_exact_saved',len(a['core_ties'])==2 and all(t['core_cost']==0 for t in a['core_ties']))
    check('context_selects_second_exact_core',a['raw']['query_core']==[6,9] and a['selected']['query_core']==[18,21] and a['context_tie_resolved'])
    mask=np.ones(30,bool);mask[4:6]=False
    a,_=run_case('one_tie_context_missing',data(qv,mask),r,[6,9])
    check('all_context_support_required',not a['context_tie_resolved'] and a['selected']['query_core']==a['raw']['query_core'] and a['state']=='ambiguous_occurrence')
    qsame=qv.copy();qsame[4:6]=rv[4:6];qsame[10:12]=rv[10:12]
    a,_=run_case('equal_core_equal_context',data(qsame),r,[6,9])
    check('equal_context_stays_ambiguous',not a['context_tie_resolved'] and a['state']=='ambiguous_occurrence')
    a,_=run_case('all_query_missing',data(rv,np.zeros(30,bool)),r,[6,9])
    check('query_missing_retains_expected',a['selected']['groups']['toy']['expected']==4 and a['selected']['groups']['toy']['compared']==0 and a['state']=='observation_insufficient')
    a,_=run_case('all_reference_missing',data(rv),data(rv,np.zeros(30,bool)),[6,9])
    check('zero_reference_cost_not_observed',a['minimum_core_cost']==0 and not a['selected']['core_supported'] and a['state']=='observation_insufficient')
    a,_=run_case('overlap_boundary_disagreement',data(rv),r,[6,9],old([5,9]))
    check('overlap_not_silently_equal',not a['old_context_disjoint'] and not a['old_context_span_equal'] and 'core_context_boundary_difference' in a['diagnostic_hold_reasons'])
    a,_=run_case('old_unsupported',data(rv),r,[6,9],old([6,9],'context_incomplete'))
    check('old_support_separate_hold','previous_context_unsupported' in a['diagnostic_hold_reasons'] and a['state']=='supported_core_candidate')
    edge=np.array([.11,.73,.29,.82,.37,.63,.99,.98])
    a,_=run_case('video_entry_clipped',data(edge),data(edge),[0,3])
    check('edge_retains_core_and_holds_context',a['selected']['core_supported'] and not a['selected']['context']['complete'] and 'context_incomplete' in a['diagnostic_hold_reasons'])
    a,_=run_case('too_short_query_no_candidate',data([.11]),r,[6,9])
    check('no_candidate_target_kept',a['selected'] is None and a['raw'] is None and a['state']=='no_candidate')
    # More than two consecutive axial steps would be necessary in the entry.
    qq=data(np.zeros(20));rr=data(np.zeros(12));qpts=np.array(list(np.linspace(.101,.299,10))+list(np.arange(10)*.1+.3))*1e6;rpts=np.arange(12)*100000
    ctx=sm.anchored_context(qq,rr,sm.frame_cost(qq,rr,['toy']),[3,4],[10,11],qpts,rpts,['toy'],summary)
    ent=ctx['phases']['entry'];check('infeasible_fully_observed_context_not_false_missing',ent['path']==[] and ent['groups']['toy']['path_missing_cells'] is None and ent['groups']['toy']['unaligned_expected_cells']==ent['groups']['toy']['expected'] and not ent['supported'])
    q=data(np.zeros(40));r=data(np.zeros(20));core=[3,12]
    with policy_inputs([(0.,10,20),(sm.EPS/2,5,10),(sm.EPS/2,25,10)]):
        a,_=run_case('bridging_representative_policy',q,r,core)
        check('any_pair_disjoint_policy',a['disjoint_core_tie'] and a['state']=='ambiguous_occurrence')
    with policy_inputs([(0.,5,10),(2*sm.EPS,25,10)],context_costs={5:1.,25:0.}):
        a,_=run_case('higher_core_cost_excluded_policy',q,r,core)
        check('context_never_expands_core_ties',len(a['core_ties'])==1 and a['selected']['query_core']==[5,14])
    with policy_inputs([(0.,5,10),(sm.EPS/2,25,10)],context_costs={5:.5,25:0.}):
        a,_=run_case('within_eps_core_tolerance_policy',q,r,core)
        check('numeric_tolerance_explicit',a['selected']['query_core']==[25,34] and a['selected']['core_cost']>a['minimum_core_cost'] and a['selected']['core_cost']<=a['minimum_core_cost']+sm.EPS)
    with policy_inputs([(0.,5,10),(0.,25,10)],complete={5:False,25:True},context_costs={5:.5,25:0.}):
        a,_=run_case('unsupported_one_candidate_policy',q,r,core)
        check('cannot_drop_unsupported_tie_to_resolve',a['selected_tie_index']==0 and not a['context_tie_resolved'] and a['state']=='ambiguous_occurrence')
    # Altering the old combined-context candidate must not enter either score/selection.
    one,_=run_case('old_context_invariance_a',data(qv),data(rv),[6,9],old([0,3]))
    two,_=run_case('old_context_invariance_b',data(qv),data(rv),[6,9],old([18,21],'observation_insufficient'))
    check('old_candidate_no_selection_leak',one['raw']==two['raw'] and one['selected']==two['selected'] and one['core_ties']==two['core_ties'] and one['state']==two['state'])
    # Segment IDs are output identity only; no matching branch can depend on them.
    q=data(rv);r=data(rv);pts=np.arange(30)*100000
    d=sm.match(q,r,[dict(role='core_candidate',core=[6,9],segment_id='different_word_person_alias')],pts,pts,{'anchors':[dict(segment_id='different_word_person_alias',state='supported_candidate',best={'query_core':[6,9]})]},['toy'],summary)
    check('segment_name_no_branch',d['anchors'][0]['selected']['query_core']==[6,9] and d['anchors'][0]['minimum_core_cost']==0)
    changed_segment=dict(role='core_candidate',core=[6,9],segment_id='different_word_person_alias',word_id='UNSEEN_WORD',person=999,alias='UNSEEN_ALIAS',description_ko='이 설명을 정답으로 복사하지 않음',source_links=[{'text':'synthetic changed source text'}])
    changed_old={'anchors':[dict(segment_id='different_word_person_alias',state='supported_candidate',best={'query_core':[6,9]},word_id='OTHER_WORD',source_text='another arbitrary source')]}
    changed=sm.match(q,r,[changed_segment],pts,pts,changed_old,['toy'],summary)
    check('word_person_alias_source_metadata_invariance',d==changed)
    check('source_unchanged',sha(U/'split_match.py')==before)
    failures=[x['name'] for x in checks if not x['pass_']]
    result=dict(status='PASS_SYNTHETIC_SPLIT_POLICY' if not failures else 'FAIL_SYNTHETIC_SPLIT_POLICY',author='Codex agent B; independent synthetic inputs and scalar summary, same model family',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=len(checks),failures=len(failures),failure_details=failures,public_arrays_read=False,public_candidate_searches=0,model_calls=0,synthetic_cases=len(cases),source_sha256=before,script_sha256=sha(__file__),checks_detail=[{'name':x['name'],'pass':x['pass_']} for x in checks],cases=cases,limitations=['Toy feature traces and explicit policy fixtures only; no public candidate performance estimate.','The bridging/tolerance/missing-policy fixtures stub ranked candidates and contexts to isolate selection logic; they do not test DP optimality.','Real DP/window code is exercised by the unmocked synthetic trace cases. Previous independent weighted-DP oracle remains separate.','One recovered path per window does not enumerate within-window path ties.','EPS-equal is numerical tolerance, not mathematically exact equality or calibrated linguistic equivalence.'])
    with (U/'review/SYNTHETIC_CHECK.json').open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ['status','checks','failures','failure_details','synthetic_cases']}),flush=True)
    return 0 if not failures else 1
if __name__=='__main__':raise SystemExit(main())
