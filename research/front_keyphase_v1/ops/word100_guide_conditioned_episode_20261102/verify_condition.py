"""Saved outputs only: separate arithmetic, preservation and functional gates."""
from condition_model import *
from verify_episode import metric
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def functional(report):
    normal=[q for q in report['sequences'] if q['kind'] in NORMAL]
    rev=[q for q in report['sequences'] if q['kind']=='actual_reverse_main' and q['precedences']]
    rest=[p for p in report['points'] if p['kind'] in NORMAL and p['y']==8]
    return dict(normal_order=sum(q['status']==SUPPORTED for q in normal),normal_false_order=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in normal),
      local_reverse_detected=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in rev),local_reverse_false_supported=sum(q['status']==SUPPORTED for q in rev),
      normal_rest_correct=sum(p['pred']==8 for p in rest),normal_rest_known=len(rest),
      normal_action_correct=report['normal_action_metrics']['correct'],normal_action_known=report['normal_action_metrics']['known'],
      normal_action_macro=report['normal_action_metrics']['macro_recall'],head_median_seconds=report['head_median_seconds'])

def main():
    assert not (E/'VERIFIED.json').exists();ev=read(E/'EVALUATION.json');plan=read(E/'FROZEN.json')
    meta={r['key']:r for r in read(B/'EVALUATION_INPUTS.json')['rows']};reports={};arrays=0;effects={}
    for summary in ev['reports']:
        name=summary['condition'];r=read(E/'evaluations'/f'{name}.json');reports[name]=r
        for metric_name,pts in [('normal_metrics',[p for p in r['points'] if p['kind'] in NORMAL]),('normal_action_metrics',[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8])]:
            actual=metric(pts);stored=r[metric_name]
            assert actual['known']==stored['known'] and actual['correct']==stored['correct'] and actual['groups']==stored['groups'] and abs(actual['macro_recall']-stored['macro_recall'])<1e-12
        for sq,fb in zip(r['sequences'],r['feedback']):
            a=np.load(E/'predictions'/name/f"{sq['key']}.npz");b=np.load(T/'predictions/current_42'/f"{sq['key']}.npz")
            assert np.array_equal(a['current_probability'],b['probability']);n=len(a['probability'])
            assert a['probability'].shape==(n,10) and np.isfinite(a['probability']).all() and np.allclose(a['probability'].sum(1),1,atol=1e-6)
            assert np.array_equal(a['frame_id'],np.arange(n)) and sq['comparison_frames']==n and sq['comparison_frame_range']==fb['full_native_frame_range']==[0,n-1]
            assert sq['whole_sign_correctness'] is None and fb['whole_sign_correctness'] is None
            base=read(meta[sq['key']]['base_path']);assert len(fb['guide_time_evidence'])==len(base['stages'])
            for stage,prior in zip(fb['guide_time_evidence'],base['stages']):
                for k in ['features','relations','frame_ids','source_phrases','start_seconds','end_seconds']:assert stage[k]==prior.get(k,[])
            assert not set(sq['rest_observed_frames'])&set(sq['approach_return_observed_frames'])
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and fb['status']=='OBSERVATION_GAP_NOT_CERTIFIED'
            arrays+=1
        effects[name]=functional(r)
    assert arrays==352
    r=reports[plan['primary']];normal=[q for q in r['sequences'] if q['kind'] in NORMAL];by={q['key']:q for q in normal};ok={q['key'] for q in normal if q['status']==SUPPORTED}
    before={q['key'] for q in read(A/'evaluations/temporal_42.json')['sequences'] if q['kind'] in NORMAL and q['status']==SUPPORTED}
    gains=sorted(ok-before);rejected={k:dict(core_unconfirmed=by[k]['previous_reviewed_core_unconfirmed'],action_errors=by[k]['evaluation_action_errors']) for k in gains if by[k]['previous_reviewed_core_unconfirmed'] or by[k]['evaluation_action_errors']}
    cuts=[q for q in r['feedback'] if q['kind']=='action_cut' and q['word']=='WORD1133'];assert len(cuts)==1
    m=r['normal_action_metrics'];g=plan['gate'];f=effects[plan['primary']]
    gates=dict(action_macro=m['macro_recall']>=g['normal_action_macro_min'],each_action_recall=all(v['correct']/v['known']>=g['normal_family_min_recall'] for v in m['groups'].values()),
      normal_order=len(ok)>=g['normal_supported_order_min'],previous_order_preserved=not(before-ok),normal_false_order=f['normal_false_order']==0,
      local_reverse=f['local_reverse_detected']>=g['local_reverse_detected_min'],reverse_false_support=f['local_reverse_false_supported']==0,
      movement_cut_guard=all(not q['movement_confirmed'] and q['status']!='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' and not q['combined_supported'] for q in cuts),
      image_gap_guard=True,old_observations_preserved=True,no_gain_by_core_exclusion=not rejected)
    diffs={}
    for factor in ['guide','context']:
        pairs=[(tag(False,c),tag(True,c)) for c in [False,True]] if factor=='guide' else [(tag(g,False),tag(g,True)) for g in [False,True]]
        diffs[factor]=[dict(before=a,after=b,difference={k:effects[b][k]-effects[a][k] for k in effects[a] if k!='head_median_seconds'}) for a,b in pairs]
    for p,h in ev['pins'].items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in read(D/'DEV_ACTION_TARGETS.json')['pins'].items():assert sha(p)==h
    out=dict(at=clock(),status='PASS_TO_STABILITY_AND_ISOLATED_RUNTIME' if all(gates.values()) else 'FACTORS_TESTED_NOT_READY',primary=plan['primary'],gates=gates,
      conditions=effects,paired_factor_differences=diffs,normal_action=m,normal_order_before=sorted(before),normal_order_after=sorted(ok),gained=gains,lost=sorted(before-ok),gains_requiring_caution=rejected,
      image_gap_cases=sum(bool(q['image_gap_frames']) for q in r['sequences']),verified_observer_arrays=arrays,
      normal_combined_supported=[q['key'] for q in r['feedback'] if q['kind'] in NORMAL and q['combined_supported']],
      inference='Fixed one-seed factorial on exposed development, not independent generalization or proof of a unique cause. No best model chosen from four conditions.',
      end_to_end_seconds=None,old_accuracy_not_rescored=True,verification='Same Codex separate saved-output arithmetic/regression, not independent expert validation',whole_goal_complete=False)
    exclusive(E/'VERIFIED.json',out);print(out['status'],gates);print(effects)
if __name__=='__main__':main()
