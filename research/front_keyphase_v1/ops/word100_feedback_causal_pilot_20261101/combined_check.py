"""Four fixed geometry sensitivity cells + semantic consumer technical controls.

Alternate time is a competing repetition, NOT corrected alignment ground truth.
Truth-based corruption examples are technical controls, never new model scores.
"""
from pathlib import Path
import json, hashlib, itertools, time
import numpy as np
import action_evidence as ae
U=Path(__file__).resolve().parent;O=U.parent
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,v):
    with (U/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def selftest():
    checks=[]
    def ck(name,v):
        checks.append(dict(name=name,pass_=bool(v)));assert v,name
    # Independent per-attribute outcomes; differences and unknowns cannot disappear.
    for a,b in itertools.product((None,0,1),repeat=2):
        x=ae.compare_attribute({'R_open_four':a},{'R_open_four':b},'R_open_four')
        ck(f'pair_{a}_{b}',x['raw_attribute_difference']==(None if a is None or b is None else a!=b))
        ck(f'never_whole_shape_{a}_{b}',x['whole_handshape_equal'] is None)
    x=ae.compare_attribute({'R_open_four':1,'R_closed_four':1,'L_open_four':1},
                           {'R_open_four':0,'R_closed_four':1,'L_open_four':1},'R_open_four')
    ck('contradiction_retains_difference',x['inconsistent_relevant_group'] and x['raw_attribute_difference'])
    ck('other_hand_unchanged',not ae.compare_attribute({'R_open_four':1,'R_closed_four':1,'L_open_four':1},{'L_open_four':1},'L_open_four')['inconsistent_relevant_group'])
    ck('mutual_height_conflict','height' in ae.conflicts({'R_higher':1,'L_higher':1}))
    states=[{'R_open_four':1,'R_closed_four':0},{'R_open_four':0,'R_closed_four':1}]
    # Public confirmed open/closed endpoints motivate this constructed cross-link test.
    table=[]
    for bad_feature,bad_link in itertools.product((False,True),repeat=2):
        actual=states[int(bad_link)]; predicted=states[1-int(bad_link)] if bad_feature else actual
        comp=ae.compare_attribute(predicted,states[0],'R_open_four')
        ck(f'factorial_{bad_feature}_{bad_link}',comp['raw_attribute_difference']==(bad_feature!=bad_link))
        table.append(dict(feature_values_deliberately_flipped=bad_feature,deliberately_wrong_endpoint=bad_link,
             comparison=comp,hidden_errors=bad_feature and bad_link,technical_control_only=True))
    for name,vals in [('normal_speed',[1,1,0,0]),('slower',[1,1,1,0,0,0]),('short_change',[1,0,1]),
                      ('unobserved',[1,None,1]),('crossing',[0,1,0]),('lowering',[1,0,None]),('reverse_order',[0,1])]:
        ids=list(range(len(vals)));s=ae.sequence_evidence(vals,ids)
        ck(name+'_samples_preserved',[x['value'] for x in s['samples']]==vals)
        ck(name+'_no_filled_states',s['inferred_missing_states']==[])
        ck(name+'_not_mandatory_repeats',s['obligatory_repeat_count'] is None)
    # All seven ternary attribute combinations: contradiction metadata cannot erase a cell.
    targets=['R_index_only','R_open_four','L_open_four','R_closed_four','L_closed_four','R_higher','L_higher']
    truth={t:0 for t in targets}
    preserved=0
    for values in itertools.product((None,0,1),repeat=7):
        q=dict(zip(targets,values))
        for t,v in q.items():
            x=ae.compare_attribute(q,truth,t)
            assert x['query']==v and x['raw_attribute_difference']==(None if v is None else bool(v))
            preserved+=1
    return dict(status='PASS_TECHNICAL_SEMANTIC_INVARIANTS',checks=checks,
         exhaustive_cell_preservations=preserved,factorial_corruption_table=table,
         natural_performance=False,does_not_establish_educational_correctness=True)
def main():
    assert not (U/'COMBINED_CHECK.json').exists()
    case=read(O/'word100_action_feedback_casebook_20261101/INPUTS.json')
    paths=[U/'P065_f48_recovered.npz',U/'RGB_REVIEW.json',Path(case['rows']['P065']['input']),Path(case['rows']['P066']['input']),
           U/'ONE_RECOVERY_RESULT.json',Path(__file__),U/'action_evidence.py']
    write('FROZEN_COMBINED_CHECK.json',dict(scope='same two stage hypotheses; old/new actual geometry; no stage oracle and no new classifier scores',
         reference=['P065',48],query_alternatives=[['P066',74],['P066',98]],geometry_conditions=['original','reobserved_current_image'],
         success='preserve all original supported values, unknown denominator, raw differences; extra geometry is not added classifier/education accuracy',
         new_head_predictions=False,features=case['rows']['P065']['feature_names'],pins={str(p):sha(p) for p in paths}))
    q=arrays(case['rows']['P066']['input']);r=arrays(case['rows']['P065']['input']);n=arrays(U/'P065_f48_recovered.npz')
    rows=[];tick=time.perf_counter()
    for version,rv,rm in [('original',r['x'][48],r['mask'][48]),('reobserved_current_image',n['x'][0],n['mask'][0])]:
        for frame in [74,98]:
            qv,qm=q['x'][frame],q['mask'][frame];both=qm&rm
            vals={name:dict(query=float(qv[j]) if qm[j] else None,reference=float(rv[j]) if rm[j] else None,
                       absolute_difference=float(abs(qv[j]-rv[j])) if both[j] else None)
                  for j,name in enumerate(case['rows']['P065']['feature_names'])}
            def delta(v,m):return float(v[9]-v[37]) if m[[8,9,36,37]].all() else None
            rows.append(dict(input_version=version,query='P066',query_frame=frame,reference='P065',reference_frame=48,
                 baseline_path_endpoint=frame==74,alternative_is_corrected_time=False,
                 opportunities=69,shared=int(both.sum()),not_jointly_observed=int((~both).sum()),fields=vals,
                 query_palm_left_y_minus_right_y=delta(qv,qm),reference_palm_left_y_minus_right_y=delta(rv,rm),
                 unique_correct_stage=None,whole_sign_judgement=None,learned_outputs_unchanged=True))
    for frame in [74,98]:
        a=next(x for x in rows if x['query_frame']==frame and x['input_version']=='original')
        b=next(x for x in rows if x['query_frame']==frame and x['input_version']!='original')
        for key,v in a['fields'].items():
            if v['absolute_difference'] is not None:assert v==b['fields'][key]
        assert b['shared']>=a['shared'] and a['opportunities']==b['opportunities']
    tech=selftest()
    write('COMBINED_CHECK.json',dict(status='COMPLETE_GEOMETRY_SENSITIVITY_NOT_CORRECTED_STAGE_TRIAL',rows=rows,
        four_cells_executed=True,confirmed_feature_by_confirmed_stage_trial='CLOSED_NOT_IDENTIFIABLE',
        technical=tech,comparison_seconds=time.perf_counter()-tick,landmark_or_educational_accuracy=None,
        new_training=0,new_classifier_inference=0,new_dtw_paths=0))
    print(json.dumps([dict(version=x['input_version'],q=x['query_frame'],shared=x['shared'],unknown=x['not_jointly_observed'],
          q_delta=x['query_palm_left_y_minus_right_y'],r_delta=x['reference_palm_left_y_minus_right_y']) for x in rows]))
if __name__=='__main__':main()
