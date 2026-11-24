"""Independent synthetic downstream checks; no observer, torch or data inference."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('engine_under_independent_review',U/'engine.py')
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
T=['R_open_four','R_index_only']

def obs(alias,values):
    p=np.array(values,dtype=float);n=len(p)
    return dict(alias=alias,prob=p,support=np.ones_like(p,dtype=bool),frame_ids=np.arange(n),times_us=np.arange(n)*33333,sample_indices=np.arange(n),intervals=[[0,1.]])

def refs():
    rr=[]
    for name in ['r1','r2','r3']:
        r=obs(name,[[.9,.1],[.9,.1],[.9,.1]])
        r.update(anchors={T[0]:[1]},reference_to_query=[0,1,2]);rr.append(r)
    return rr

def digest(value):
    if isinstance(value,np.ndarray):return [str(value.dtype),value.shape,value.tobytes().hex()]
    if isinstance(value,dict):return {k:digest(v) for k,v in value.items()}
    if isinstance(value,list):return [digest(v) for v in value]
    return value

def main(output):
    checks=[]
    def check(ok,name,actual=None):checks.append(dict(name=name,pass_contract=bool(ok),actual=actual))
    clause=dict(id='c',feature=T[0],source_span='synthetic',source_url='synthetic://fixture',link_status='partial_supported',expected=1)
    q=obs('q',[[.9,.1],[.9,.1],[.9,.1]]);rr=refs()
    before=digest([q,rr])
    base=E.compare_clause(clause,rr,q,T)
    check(base['status']=='DESCRIBED_FEATURE_OBSERVED' and base['expected_anchor_count']==3,'known_positive_fragment_only')
    check(base['educational_verdict'] is None and base['linguistic_tolerance'] is None,'no_semantic_verdict_or_tolerance')
    for status in ['rejected','unverified','unsupported','pending_unrecognized','',None]:
        result=E.compare_clause(dict(clause,link_status=status),rr,q,T)
        check(result['status']=='SOURCE_LINK_WITHHELD','source_allowlist/'+str(status))
    check(E.compare_clause(dict(clause,feature='unsupported_fist'),rr,q,T)['status']=='SOURCE_LINK_WITHHELD','unsupported_shape_not_substituted')
    for value in [np.nan,np.inf,-np.inf,-.1,1.1]:
        missing=copy.deepcopy(q);missing['prob'][1,0]=value
        result=E.compare_clause(clause,rr,missing,T)
        serializable=True
        try:json.dumps(result,allow_nan=False)
        except (TypeError,ValueError):serializable=False
        check(serializable and result['references'][0]['cells'][0]['query_probability'] is None,'invalid_probability_serializes_null/'+str(value))
        check(result['status']=='INSUFFICIENT_LINKED_EVIDENCE' and result['expected_anchor_count']==3 and result['observed_anchor_count']==0,'invalid_probability_cannot_pass/'+str(value))
    masked=copy.deepcopy(q);masked['support'][:]=False
    result=E.compare_clause(clause,rr,masked,T)
    check(result['expected_anchor_count']==base['expected_anchor_count'] and result['observed_anchor_count']==0,'query_missing_preserves_reference_denominator')
    check(all(r['counts']['UNOBSERVED']==1 for r in result['references']),'query_missing_not_opposite')
    negative=copy.deepcopy(q);negative['prob'][1,0]=.1
    result=E.compare_clause(clause,rr,negative,T)
    check(result['status']=='GUIDE_RELEVANT_DIFFERENCE_CANDIDATE' and result['educational_verdict'] is None,'negative_open_is_only_attribute_difference')
    check('fist' not in json.dumps(result).lower() and '주먹' not in E.explanation(result),'negative_open_not_named_fist')
    uncertain=copy.deepcopy(q);uncertain['prob'][1,0]=.5
    result=E.compare_clause(clause,rr,uncertain,T)
    check(result['status']=='INSUFFICIENT_LINKED_EVIDENCE' and result['references'][0]['counts']['UNCERTAIN_MODEL']==1,'uncertain_not_false')
    for prob,expected,want in [(.8,1,'SUPPORTED_STATE'),(.2,1,'OPPOSITE_STATE'),(.8,0,'OPPOSITE_STATE'),(.2,0,'SUPPORTED_STATE')]:
        actual=E.classify_probability(prob,True,expected);check(actual==want,'symmetric_confidence_boundary/'+str(prob)+'/'+str(expected),actual)
    hidden_refs=copy.deepcopy(rr)
    for r in hidden_refs:r['support'][:]=False
    result=E.compare_clause(clause,hidden_refs,q,T)
    check(result['expected_anchor_count']==3 and result['observed_anchor_count']==0 and all(r['counts']['REFERENCE_UNOBSERVED']==1 for r in result['references']),'missing_reference_anchors_retained')
    conflict_refs=copy.deepcopy(rr)
    for r in conflict_refs:r['prob'][1,0]=.1
    result=E.compare_clause(clause,conflict_refs,q,T)
    check(result['expected_anchor_count']==3 and result['observed_anchor_count']==0 and all(r['counts']['REFERENCE_OBSERVER_CONFLICT']==1 for r in result['references']),'reference_label_observer_conflict_explicit')
    no_alignment=copy.deepcopy(rr)
    for r in no_alignment:r['sample_indices']=np.array([],dtype=int)
    result=E.compare_clause(clause,no_alignment,q,T)
    check(result['expected_anchor_count']==3 and all(r['counts']['ALIGNMENT_UNAVAILABLE']==1 for r in result['references']),'unmappable_anchor_not_removed')
    separated=copy.deepcopy(rr)
    for i,r in enumerate(separated):r['reference_to_query']=[i,i,i]
    result=E.compare_clause(clause,separated,q,T)
    check(result['status']=='INSUFFICIENT_LINKED_EVIDENCE' and not result['coherent_query_frame_consensus'],'no_consensus_from_separate_query_times')
    duplicate=copy.deepcopy(rr)
    for r in duplicate:r['alias']='one_person'
    result=E.compare_clause(clause,duplicate,q,T)
    check(result['status']=='INSUFFICIENT_LINKED_EVIDENCE','repeated_reference_not_multiple_people')
    label=np.array([[1],[0],[1]],float);mask=np.array([[True],[True],[False]])
    anchors=E.select_anchor_indices(dict(timestamps_us=q['times_us'],intervals=[[0.,1.]]),0,label,mask)
    check(anchors.tolist()==[0],'anchors_only_known_positive_not_missing_or_negative')
    names=[f'R_{finger}_{joint}_angle' for finger in ['index','middle','ring','little'] for joint in ['mcp','pip','dip']]
    a=dict(mask=np.ones((3,12),bool));a['mask'][1,:3]=False;old=np.ones((3,2),bool)
    supported=E.feature_support(a,names,T,old)
    check(not supported[1].any() and supported[0].all() and old.all(),'required_index_masks_and_no_input_mutation')
    order=dict(id='o',before=T[0],after=T[1],link_status='proxy_only')
    forward=obs('forward',[[.9,.1],[.9,.1],[.1,.1],[.1,.9],[.1,.9],[.1,.1]])
    reverse=obs('reverse',[[.1,.9],[.1,.9],[.1,.1],[.9,.1],[.9,.1],[.1,.1]])
    check(E.check_order(order,forward,T)['status']=='DESCRIBED_ORDER_PATTERN_OBSERVED','original_forward_order')
    reversed_result=E.check_order(order,reverse,T)
    check(reversed_result['status']=='REVERSE_ORDER_CANDIDATE' and reversed_result['educational_verdict'] is None,'same_state_set_reversed_time_not_forward')
    for value in [np.nan,np.inf,1.1]:
        invalid=copy.deepcopy(forward);invalid['prob'][:2,0]=value
        result=E.check_order(order,invalid,T)
        check(not result['before_events'] and result['status']=='ORDER_UNCONFIRMED','invalid_prob_not_temporal_event/'+str(value))
    absent=copy.deepcopy(forward);absent['support'][:,1]=False
    result=E.check_order(order,absent,T)
    check(result['status']=='ORDER_UNCONFIRMED' and result['mandatory_omission_verdict'] is None,'missing_after_not_omission')
    broken=copy.deepcopy(forward);broken['support'][1,0]=False
    check(not E.positive_runs(broken,T[0],T),'masked_gap_not_filled')
    broken=copy.deepcopy(forward);broken['frame_ids'][1:]+=1
    check(not E.positive_runs(broken,T[0],T),'nonconsecutive_frames_not_continuous_event')
    outside=copy.deepcopy(forward);outside['intervals']=[[.09,.2]]
    check(not E.positive_runs(outside,T[0],T),'events_outside_supplied_token_interval_excluded')
    order_withheld=E.check_order(dict(order,after='unsupported_contact'),forward,T)
    check(order_withheld['status']=='ORDER_NOT_OBSERVABLE','no_contact_proxy_substitution')
    contract=dict(word_id='synthetic_word',label='synthetic label',clauses=[clause],orders=[])
    actual=E.run_expression(contract,rr,q,T)
    changed=copy.deepcopy(contract);changed['label']='contradictory source text';changed['clauses'][0]['source_span']='opposite wording'
    other=E.run_expression(changed,rr,q,T)
    check(digest([q,rr])==before,'text_changes_do_not_mutate_observations_or_references')
    check(actual['clauses'][0]['references']==other['clauses'][0]['references'],'text_changes_do_not_change_physical_evidence')
    check(actual['whole_sign_verdict'] is None and actual['linguistically_acceptable_variation'] is None,'whole_sign_and_tolerance_withheld')
    failures=[x for x in checks if not x['pass_contract']]
    result=dict(status='PASS' if not failures else 'FAIL',checks=len(checks),failures=failures,cases=checks,engine_sha256=hashlib.sha256((U/'engine.py').read_bytes()).hexdigest(),verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_model_inference=0,training=0,source_video_decodes=0,imports='Only engine.py and NumPy; no model/checkpoint/project inference imports.',limitations=['A forward-order result proves an existential two-event witness, not exclusive full-sequence order or exact repetition count.','Known positive reference anchors do not establish all source phases or a linguistic correctness/tolerance boundary.'])
    with (U/'review'/output).open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],checks=len(checks),failures=failures),ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='ENGINE_REVIEW_TESTS.json');args=parser.parse_args();main(args.output)
