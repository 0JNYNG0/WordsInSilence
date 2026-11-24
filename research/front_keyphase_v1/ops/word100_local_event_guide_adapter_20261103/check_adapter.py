"""Synthetic structural tests only; no real prediction reads or model imports."""
from pathlib import Path
import copy
import json
import hashlib
import numpy as np
from adapter import attach,value_sha

U=Path(__file__).resolve().parent

def fixture(target=1):
    n=12;ids=np.arange(n,dtype=np.int64);ts=ids*33333
    identity=dict(clip_id='synthetic',word_id='WORD2523',source_sha256='a'*64,input_sha256='b'*64)
    p=np.full((n,5),.1,np.float32);p[5,target]=.9
    pred=dict(probabilities=p,frame_id=ids.copy(),timestamp_us=ts.copy(),
              motion_anchor=np.ones(n,bool),evidence_available=np.ones(n,bool),rgb_available=np.ones(n,bool))
    prepared=dict(ids=ids,ts=ts,mask=np.ones((n,73),bool),available=np.ones(n,bool),
                  motion_anchor=np.ones(n,bool),evidence_available=np.ones(n,bool),
                  geometry_valid=np.ones(146,np.float32),input_identity=identity.copy())
    base=dict(word_id='WORD2523',path=np.column_stack([ids,ids]).tolist(),
              path_axes=['query_native_index','learned_canonical_index'],
              stages=[dict(guide_id='test_stage',frame_ids=[3,4,5,6,7,8],description='test',
                  neural_stage_agreement=.7,image_time_coverage=1.,unique_query_frames=6,
                  features=[dict(kind='shape',value=.123)],motions=[dict(probability=.321)])],
              motion_observations=[dict(target=target,probability=.234,frame_ids=[3,4,5,6,7,8])],
              source_components=[dict(component_id='existing',point_evidence=[dict(value=.456)])],
              partial_order=dict(value=.654,unknown=True),sequence_verdict=dict(status='old',whole_sign_correctness=None))
    proposals=[dict(target_index=target,guide_id='test_stage',indices=[3,8],native_frame_ids=list(range(3,9)),
                    timestamp_us=ts[3:9].tolist(),source_constraint=None)]
    envelope=dict(**identity,AH_prediction_sha256='c'*64,native_frame_ids=ids.tolist(),timestamp_us=ts.tolist(),
                  base_value_sha256=value_sha(base),proposals_value_sha256=value_sha(proposals))
    support={target:dict(nominal_native_frames=list(range(3,9)),outside_nominal=[0,1,2,9,10,11],direct_truth=False)}
    return base,envelope,prepared,pred,proposals,support

def main():
    output=U/'STRUCTURAL_CHECKS_v2.json'
    if output.exists():raise FileExistsError('Preserve completed structural checks')
    checks=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)));assert value,name
    def expect_error(name,items):
        try:attach(*items)
        except (ValueError,AssertionError):check(name,True)
        else:check(name,False)
    a=fixture();before=copy.deepcopy(a[0]);out=attach(*a);o=out['local_event_feedback']['observations'][0]
    check('one_additive_key_and_all_prior_values_preserved',set(out)==set(before)|{'local_event_feedback'} and all(out[k]==v for k,v in before.items()))
    check('input_object_not_mutated',a[0]==before)
    check('high_fixed_anchor_observed',o['status']=='LOCAL_MOVEMENT_EVIDENCE_OBSERVED' and o['maximum_anchor_native_frame']==5)
    check('all_nominal_points_and_times_preserved',len(o['native_point_estimates'])==6 and [r['timestamp_us'] for r in o['native_point_estimates']]==a[4][0]['timestamp_us'])
    check('source_unrelated_target_does_not_become_clause_truth',o['source_binding']=='REFERENCE_OBSERVATION_ONLY' and o['source_constraint'] is None and o['whole_clause_truth'] is None)
    check('completion_error_repeat_unknown',o['complete_action'] is None and o['educational_error'] is None and o['required_repetitions'] is None and o['correction_enabled'] is False)
    a=fixture();a[2]['motion_anchor'][7]=False;a[2]['evidence_available'][7]=False;a[3]['motion_anchor'][7]=False;a[3]['evidence_available'][7]=False
    a[2]['available'][7]=False;a[2]['mask'][7]=False;a[3]['rgb_available'][7]=False
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('high_anchor_survives_unrelated_gap',o['status']=='LOCAL_MOVEMENT_EVIDENCE_OBSERVED' and not o['scope_evidence_complete'] and o['unavailable_native_frames']==[7])
    missing_point=next(r for r in o['native_point_estimates'] if r['frame_id']==7)
    check('nonanchor_probability_null_raw_preserved',missing_point['probability'] is None and np.isclose(missing_point['raw_model_probability'],.1))
    a[3]['probabilities'][:]=.1
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('low_with_gap_is_insufficient',o['status']=='INSUFFICIENT_OBSERVATION' and o['educational_error'] is None)
    a=fixture();a[3]['probabilities'][:]=.1
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('low_complete_is_unconfirmed_not_error',o['status']=='LOCAL_MOVEMENT_NOT_CONFIRMED' and not o['correction_enabled'])
    a=fixture();a[3]['probabilities'][:]=.1;a[3]['probabilities'][0,1]=.999
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('outside_scope_high_does_not_expand_proposal',o['status']=='LOCAL_MOVEMENT_NOT_CONFIRMED' and o['proposal']['indices']==[3,8])
    a=fixture();a[0]['stages'][0]['neural_stage_agreement']=.2;a[1]['base_value_sha256']=value_sha(a[0])
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('high_motion_cannot_promote_stage_binding',o['status']=='LOCAL_MOVEMENT_EVIDENCE_OBSERVED' and o['guide_time_binding']['status']=='UNCONFIRMED')
    check('guard_uses_original_guide_scope',o['guide_time_binding']['guard_scope_native_frames']==[3,4,5,6,7,8] and o['guide_time_binding']['source_applicability_verified'] is None)
    a=fixture();a[2]['mask'][:]=False
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('rgb_only_is_estimate_not_missing_shape_repair',o['status']=='LOCAL_MOVEMENT_EVIDENCE_OBSERVED' and o['maximum_anchor_evidence_mode']=='IMAGE_TIME_ESTIMATE')
    a=fixture();a[1]['source_sha256']='d'*64;expect_error('same_axis_wrong_source_rejected',a)
    a=fixture();a[1]['input_sha256']='e'*64;expect_error('same_axis_wrong_cache_rejected',a)
    a=fixture();a[3]['timestamp_us'][5]+=1;expect_error('timestamp_mismatch_rejected',a)
    a=fixture();a[3]['motion_anchor'][5]=False;expect_error('availability_mismatch_rejected',a)
    a=fixture();a[4][0]['indices']=[2,8];expect_error('proposal_mutation_rejected',a)
    a=fixture();a[0]['word_id']='WORD1133';expect_error('wrong_target_word_rejected',a)
    a=fixture();a[4][0]['source_constraint']=dict(clause_id='egypt_lower_shared',entry_id='NIKL_legacy:5990',component_id='lowering',condition_relation='RELATED_ONLY');a[1]['proposals_value_sha256']=value_sha(a[4]);expect_error('separation_not_lowering_source',a)
    a=fixture(0);a[4][0]['source_constraint']=dict(clause_id='well_brush',entry_id='NIKL_legacy:41',component_id='right_hand_traverses_left_arm_toward_wrist',condition_relation='RELATED_ONLY');a[1]['proposals_value_sha256']=value_sha(a[4]);expect_error('same_gloss_wrong_variant_rejected',a)
    a=fixture();a[0]['local_event_feedback']={};a[1]['base_value_sha256']=value_sha(a[0]);expect_error('duplicate_attachment_rejected',a)
    a=fixture(0)
    for v in [a[0],a[1],a[2]['input_identity']]:v['word_id']='WORD1133'
    a[4][0]['source_constraint']=dict(word_id='WORD1133',clause_id='well_brush',entry_id='NIKL_legacy:9543',component_id='right_hand_traverses_left_arm_toward_wrist',condition_relation='RELATED_ONLY')
    a[1]['base_value_sha256']=value_sha(a[0]);a[1]['proposals_value_sha256']=value_sha(a[4]);a[3]['probabilities'][:]=.1
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('source_guard_support_without_high_not_observed',o['guide_time_binding']['status']=='SUPPORTED' and o['source_binding']=='SOURCE_ASSOCIATION_ONLY_MOVEMENT_UNCONFIRMED' and o['source_constraint']['condition_relation']=='RELATED_ONLY')
    a[2]['motion_anchor'][7]=False;a[2]['evidence_available'][7]=False;a[3]['motion_anchor'][7]=False;a[3]['evidence_available'][7]=False
    a[2]['available'][7]=False;a[2]['mask'][7]=False;a[3]['rgb_available'][7]=False
    o=attach(*a)['local_event_feedback']['observations'][0]
    check('source_low_gap_not_observed',o['source_binding']=='SOURCE_ASSOCIATION_ONLY_INSUFFICIENT_OBSERVATION' and o['source_variant_applicability_verified'] is None)
    result=dict(status='PASS',checks=checks,checks_count=len(checks),real_saved_predictions_read=0,
        model_loads=0,model_forwards=0,new_DTW=0,source_media_reads=0,
        scope='Synthetic identity, unknown, fixed-scope and deep old-output preservation; not movement accuracy.',
        code_pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [U/'adapter.py',Path(__file__)]})
    with output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'PASS','checks':len(checks)}))
if __name__=='__main__':main()
