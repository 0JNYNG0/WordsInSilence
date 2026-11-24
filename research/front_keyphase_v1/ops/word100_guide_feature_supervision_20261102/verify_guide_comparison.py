"""Independent saved comparison audit plus synthetic pure-formatter contracts.

No neural inference, labels, reference video reads, or old accuracy re-scoring.
The saved-output reconstruction below does not call GuideComparator.attach.
"""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import hashlib,json,math
import numpy as np

U=Path(__file__).resolve().parent;O=U.parent
A=O/'word100_partial_order_live_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def expected_point(p):
    if p['image_available'] is False or p['probability'] is None:return 'UNOBSERVED',None
    value=p['probability'];g=p['joint_support']
    if .2 < value < .8:return 'UNCERTAIN',None
    direction='compatible' if value>=.8 else 'counter'
    if g<.5:return 'IMAGE_TIME_ESTIMATE',direction
    return ('MODEL_COMPATIBLE' if value>=.8 else 'MODEL_COUNTER_EVIDENCE'),direction

def expected_summary(states):
    c=Counter(states);positive=c['MODEL_COMPATIBLE'];negative=c['MODEL_COUNTER_EVIDENCE'];n=len(states)
    if n==0 or c['UNOBSERVED']==n:s='INSUFFICIENT_OBSERVATION'
    elif positive and negative:s='MIXED_OBSERVATIONS'
    elif positive:s='COMPATIBLE_EVIDENCE_PRESENT'
    elif negative:s='COUNTER_EVIDENCE_TIME_UNCONFIRMED'
    else:s='UNCERTAIN_OR_ESTIMATED_ONLY'
    return dict(status=s,total_points=n,counts=dict(c),observed_model_points=positive+negative,
                coverage=(positive+negative)/n if n else 0.,denominator_includes_missing=True,
                whole_window_confirmed=False,has_unresolved_points=(positive+negative<n),short_counterpoints_preserved=True)

def fixture_tests(contract):
    # Pure synthetic formatter calls are distinct from saved-public verification.
    from guide_comparison import GuideComparator
    comparator=GuideComparator();cases=[];limitations=[]
    c=next(c for c in contract['constraints'] if c['component_id']=='right_non_thumb_closed')
    def fixture(values=(.9,.1,.5),geometry=(1.,1.,1.),available=(True,True,True)):
        points=[dict(frame=i,seconds=i/30,probability=v,joint_support=g,image_available=a,status='SAVED_STATUS')
                for i,v,g,a in zip([10,11,12],values,geometry,available)]
        comp=dict(component_id=c['component_id'],source_clauses=[dict(id=c['clause_id'],text=c['text'],
                  url=c['url'],source=deepcopy(c['source']))],frame_ids=[10,11,12],status='POINT_EVIDENCE_AVAILABLE',point_evidence=points)
        return dict(word_id=c['word_id'],source_components=[comp],preserved={'shape':[.1,.2],'sequence_verdict':'old'})
    def run(base):return comparator.attach(base,[10,11,12],[333333,366667,400000])
    # Use exact rounded native timestamps in point records.
    def timed(base):
        for p,t in zip(base['source_components'][0]['point_evidence'],[333333,366667,400000]):p['seconds']=t/1e6
        return base
    good=timed(fixture());saved=deepcopy(good);out=run(good);item=out['guide_feature_comparison']['items'][0]
    assert good==saved and all(out[k]==v for k,v in saved.items());cases.append('old_and_input_fields_unchanged')
    assert item['summary']['status']=='MIXED_OBSERVATIONS' and item['summary']['total_points']==3
    assert [p['observation'] for p in item['native_points']]==['MODEL_COMPATIBLE','MODEL_COUNTER_EVIDENCE','UNCERTAIN']
    assert all(p['guide_violation'] is None for p in item['native_points']);cases.append('single_frame_counter_retained_not_guide_error')
    x=run(timed(fixture((.8,.2,None),(.49,.49,1.),(True,True,False))))['guide_feature_comparison']['items'][0]
    assert [p['observation'] for p in x['native_points']]==['IMAGE_TIME_ESTIMATE','IMAGE_TIME_ESTIMATE','UNOBSERVED']
    assert [p['model_direction'] for p in x['native_points']]==['compatible','counter',None]
    assert x['summary']['coverage']==0 and x['summary']['total_points']==3;cases.append('both_low_geometry_directions_and_missing_denominator')
    # Nearby unknown point is not inherited from its positive neighbours.
    x=run(timed(fixture((.9,.5,.9))))['guide_feature_comparison']['items'][0]
    assert x['native_points'][1]['observation']=='UNCERTAIN' and not x['summary']['whole_window_confirmed'];cases.append('no_temporal_fill')
    mutations={
      'nan_seconds':lambda b:b['source_components'][0]['point_evidence'][0].update(seconds=float('nan')),
      'wrong_seconds':lambda b:b['source_components'][0]['point_evidence'][0].update(seconds=.35),
      'float_point_frame':lambda b:b['source_components'][0]['point_evidence'][0].update(frame=10.0),
      'float_component_frame':lambda b:b['source_components'][0].update(frame_ids=[10.,11,12]),
      'string_availability':lambda b:b['source_components'][0]['point_evidence'][0].update(image_available='False'),
      'boolean_probability':lambda b:b['source_components'][0]['point_evidence'][0].update(probability=True),
      'boolean_geometry':lambda b:b['source_components'][0]['point_evidence'][0].update(joint_support=True),
      'source_variant_pointer':lambda b:b['source_components'][0]['source_clauses'][0]['source'].update(json_pointer='/cards/6/candidates/1'),
      'source_phrase':lambda b:b['source_components'][0]['source_clauses'][0].update(text='other form'),
      'source_url':lambda b:b['source_components'][0]['source_clauses'][0].update(url='https://example.invalid/other'),
      'duplicate_component':lambda b:b['source_components'].append(deepcopy(b['source_components'][0])),
      'omitted_native_point':lambda b:b['source_components'][0]['point_evidence'].pop(1),
      'omitted_native_extent':lambda b:b['source_components'][0].update(frame_ids=[10,12]),
    }
    for name,mutate in mutations.items():
        b=deepcopy(good);mutate(b)
        try:run(b)
        except (ValueError,TypeError):cases.append('reject_'+name)
        else:raise AssertionError('Accepted malformed fixture: '+name)
    for name,ff,tt in [('fractional_frame',[10.2,11,12],[333333,366667,400000]),
                       ('fractional_timestamp',[10,11,12],[333333.1,366667,400000]),
                       ('boolean_axis',[True,11,12],[333333,366667,400000])]:
        try:comparator.attach(good,ff,tt)
        except ValueError:cases.append('reject_'+name)
        else:raise AssertionError('Accepted axis fixture: '+name)
    duplicate=deepcopy(good);duplicate['source_components'][0]['source_clauses'].append(deepcopy(duplicate['source_components'][0]['source_clauses'][0]))
    try:run(duplicate)
    except ValueError:cases.append('reject_duplicate_clause_in_component')
    else:limitations.append('Same-clause duplication within one component is not rejected by current helper; the fixed16 inputs were independently checked unique. Harden before accepting arbitrary inputs.')
    return cases,limitations

def main():
    dst=U/'COMPARATOR_CODE_REVIEW.json';assert not dst.exists()
    verified=U/'SAVED_PUBLIC_COMPARISON_VERIFIED.json';assert not verified.exists()
    contract=read(U/'COMPARISON_CONTRACT.json');checks=0
    for path,digest in contract['pins'].items():assert sha(path)==digest;checks+=1
    assert contract['point_thresholds']==dict(compatible_min=.8,counter_max=.2,joint_support_min=.5)
    assert {c['component_id'] for c in contract['constraints'] if c['condition_relation']=='NECESSARY_PARTIAL'}=={
        'right_non_thumb_closed','both_visible_thumbs_up_with_closed_finger_masses'}
    assert all(c['condition_relation']=='RELATED_ONLY' for c in contract['constraints'] if c['component_id'] not in {
        'right_non_thumb_closed','both_visible_thumbs_up_with_closed_finger_masses'})
    assert all(c['expectation_is_training_label'] is False and c['correction_enabled'] is False for c in contract['constraints']);checks+=4
    cases,limits=fixture_tests(contract)
    manifest=read(A/'SAVED_CASES_MANIFEST.json');saved=read(U/'SAVED_PUBLIC_COMPARISON.json');frozen=read(U/'SAVED_ATTACHMENT_FROZEN.json')
    assert sha(U/'COMPARISON_CONTRACT.json')==saved['contract_sha256']==frozen['contract_sha256']
    assert sha(U/'guide_comparison.py')==saved['code_sha256']==frozen['code_sha256']
    assert sha(U/'attach_saved_comparisons.py')==frozen['driver_sha256']
    assert sha(A/'SAVED_CASES_MANIFEST.json')==saved['manifest_sha256']==frozen['manifest_sha256'];checks+=4
    expected_cases={(w,k) for w,d in manifest['cases'].items() for k in d}
    assert len(saved['rows'])==16 and {(r['word_id'],r['condition']) for r in saved['rows']}==expected_cases;checks+=1
    aggregate=Counter();summaries=Counter();report_rows=[];point_total=0
    for row in saved['rows']:
        w,k=row['word_id'],row['condition'];entry=manifest['cases'][w][k]
        assert row['base_path']==entry['result'] and row['arrays_path']==entry['analysis']
        assert row['base_sha256']==entry['result_sha256']==sha(row['base_path'])
        assert row['arrays_sha256']==entry['analysis_sha256']==sha(row['arrays_path'])
        assert sha(row['result'])==row['result_sha256'];checks+=4
        base=read(row['base_path']);out=read(row['result'])
        assert base['word_id']==out['word_id']==entry['word_id']==w
        assert base['source_sha256']==entry['source_sha256']==entry['video_sha256']
        assert set(out)==set(base)|{'guide_feature_comparison'}
        assert all(out[key]==value for key,value in base.items());checks+=4
        with np.load(row['arrays_path'],allow_pickle=False) as z:
            ff=z['frame_id'];tt=z['timestamp_us']
            assert row['native_axis_sha256']==hashlib.sha256(ff.tobytes()+tt.tobytes()).hexdigest()
            frames=ff.tolist();times=tt.tolist()
        assert frames==entry['native_frame_ids'] and [t/1e6 for t in times]==entry['native_times_seconds'];checks+=2
        position=dict(zip(frames,range(len(frames))))
        addon=out['guide_feature_comparison'];items=addon['items']
        cs=[c for c in contract['constraints'] if c['word_id']==w]
        assert len(items)==len(cs)==row['items'] and addon['contract_sha256']==saved['contract_sha256']
        assert addon['new_model_inference']==addon['new_training']==addon['individual_reference_comparisons_added']==0
        assert addon['educational_corrections_enabled'] is False and addon['accuracy_gain_established'] is False
        states=Counter();item_summaries=[];unbound=[]
        for item,c in zip(items,cs):
            for key,ck in [('clause_id','clause_id'),('component_id','component_id'),('source_text','text'),
                           ('source_url','url'),('source_entry','entry_id'),('source','source'),
                           ('condition_relation','condition_relation'),('expected_component_value','expected_component_value'),('unsupported','unsupported')]:
                assert item[key]==c[ck];checks+=1
            assert item['correction_enabled'] is False and item['whole_clause_correctness'] is None and item['whole_sign_correctness'] is None
            assert item['applicability']=='SOURCE_VARIANT_AND_LOCAL_PHASE_NOT_CONFIRMED'
            assert item['expectation_is_observation'] is False and item['expectation_is_training_label'] is False;checks+=3
            matches=[b for b in base['source_components'] if b['component_id']==c['component_id'] and any(s['id']==c['clause_id'] for s in b['source_clauses'])]
            assert len(matches)<=1
            if matches:
                b=matches[0];cl=[s for s in b['source_clauses'] if s['id']==c['clause_id']]
                assert len(cl)==1 and all(cl[0][key]==c[ck] for key,ck in [('text','text'),('url','url'),('source','source')])
                ids=b['frame_ids'];assert ids==frames[position[ids[0]]:position[ids[-1]]+1]==item['frame_ids']
                assert item['start_seconds']==times[position[ids[0]]]/1e6 and item['end_seconds']==times[position[ids[-1]]]/1e6
                assert item['time_binding']=='PROPOSED_DTW';checks+=4
                if c['runtime_observer']=='point':
                    points=item['native_points'];assert len(points)==len(b['point_evidence'])==len(ids)
                    local=[]
                    for old,new,f in zip(b['point_evidence'],points,ids):
                        expected,direction=expected_point(old)
                        assert new['observation']==expected and new['model_direction']==direction
                        assert new['frame']==old['frame']==f and new['seconds']==old['seconds']==times[position[f]]/1e6
                        assert all(new[key]==old[key] for key in ['probability','joint_support','image_available'])
                        assert new['guide_violation'] is None and new['time_binding']=='PROPOSED_DTW'
                        assert new['original_status']==old['status'] and new['evidence_origin']=='FROZEN_NEURAL_OBSERVER_NOT_MANUAL_TRUTH'
                        local.append(expected);checks+=5
                    assert item['summary']==expected_summary(local);states.update(local);point_total+=len(local);checks+=1
                elif c['runtime_observer']=='interval':
                    ok=b['status']=='RELATED_MOVEMENT_OBSERVED' and b.get('guide_time_binding')=='SUPPORTED'
                    es=dict(status='RELATED_MOVEMENT_EVIDENCE_PRESENT' if ok else 'INSUFFICIENT_MOVEMENT_OR_TIME_EVIDENCE',
                            total_native_frames=len(ids),exact_observer_scope=b.get('evidence_mode'),
                            prior_guide_time_binding=b.get('guide_time_binding'),probability=b.get('probability'),image_coverage=b.get('image_coverage'),
                            complete_action_verified=False,negative_inferred_from_low_probability=False)
                    assert item['summary']==es and item['native_points']==[];checks+=1
                else:assert item['summary']==dict(status='NO_BOUND_RUNTIME_OBSERVER',total_points=0) and item['native_points']==[];unbound.append(c['clause_id'])
            else:
                assert item['frame_ids']==[] and item['native_points']==[] and item['time_binding']=='UNBOUND'
                assert item['summary']==dict(status='NO_BOUND_RUNTIME_OBSERVER',total_points=0);unbound.append(c['clause_id']);checks+=2
            summaries[item['summary']['status']]+=1
            item_summaries.append(dict(clause_id=item['clause_id'],component_id=item['component_id'],summary=item['summary']['status'],
                                      frames=len(item['frame_ids']),time_binding=item['time_binding'],correction_enabled=False))
        assert dict(states)==row['point_states'] and item_summaries==row['component_summaries'];checks+=2
        assert addon['comparison_seconds']==row['comparison_seconds'] and math.isfinite(row['comparison_seconds']) and row['comparison_seconds']>=0
        aggregate.update(states);report_rows.append(dict(word_id=w,condition=k,points=sum(states.values()),point_states=dict(states),unbound_clauses=unbound))
    assert dict(aggregate)==saved['point_states'] and point_total==sum(aggregate.values())
    assert np.median([r['comparison_seconds'] for r in saved['rows']])==saved['comparison_seconds_median'];checks+=2
    paths=[U/'COMPARISON_CONTRACT.json',U/'guide_comparison.py',U/'guide_comparison_initial.py',U/'attach_saved_comparisons.py',
           U/'SAVED_ATTACHMENT_FROZEN.json',U/'SAVED_PUBLIC_COMPARISON.json',A/'SAVED_CASES_MANIFEST.json',Path(__file__)]
    review=dict(status='PASS_FIXED_16_ADDITIVE_SCOPE_WITH_GENERAL_INPUT_LIMITS',checks=checks,synthetic_checks=cases,
                initial_findings_corrected=['NaN seconds accepted by abs comparison','Fractional axes silently int-truncated',
                                           'Float point frame equality','Nonboolean availability','Exact source URL mismatch'],
                remaining_general_input_limits=limits+['Driver does not independently assert base word ID/source identity against manifest before attach; all16 actual saved bindings were independently verified here. Future arbitrary inputs must fail closed.'],
                semantics=['NECESSARY_PARTIAL is a conditional one-way physical implication only; RELATED_ONLY is not mandatory.',
                           'DTW local applicability remains unconfirmed. Low confidence does not make a linguistic error.',
                           'Short contrary points and unavailable/estimated points remain in the native denominator.',
                           'No-bound observers remain explicit, including already annotated but not runtime-enabled thumb predicates.',
                           'This is comparison formatting of frozen model outputs, not new training or a demonstrated accuracy increase.'],
                model_inference=0,new_training=0,old_scores_recomputed=False,whole_goal_complete=False,
                pins={str(p):sha(p) for p in paths})
    dst.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    report=dict(status='PASS_SAVED_16_COMPARISONS_RECONSTRUCTED_WITHOUT_MODEL_EXECUTION',checks=checks,cases=16,
                original_native_points=point_total,point_states=dict(aggregate),component_summary_counts=dict(summaries),rows=report_rows,
                code_review_sha256=sha(dst),comparison_report_sha256=sha(U/'SAVED_PUBLIC_COMPARISON.json'),
                original_fields_equal=True,source_variant_and_word_identity_checked=True,missing_denominator_preserved=True,
                new_model_inference=0,new_training=0,old_accuracy_rescoring=0,educational_error_accuracy=False,
                end_to_end_latency_measured=False,whole_goal_complete=False)
    verified.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'status':report['status'],'checks':checks,'synthetic_checks':len(cases),'point_states':dict(aggregate),'total':point_total},ensure_ascii=False))

if __name__=='__main__':main()
