"""Separate saved-output arithmetic and regression audit; no model execution."""
from episode_model import *
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def metric(points):
    groups={}
    for p in points:
        g=groups.setdefault(str(p['y']),dict(known=0,correct=0));g['known']+=1;g['correct']+=int(p['y']==p['pred'])
    return dict(known=len(points),correct=sum(g['correct'] for g in groups.values()),groups=groups,macro_recall=sum(g['correct']/g['known'] for g in groups.values())/len(groups))

def main():
    assert not (D/'VERIFIED.json').exists();ev=read(D/'EVALUATION.json');plan=read(D/'FROZEN.json');meta={r['key']:r for r in read(B/'EVALUATION_INPUTS.json')['rows']};reports={};arrays=0
    for summary in ev['reports']:
        tag=summary['condition'];r=read(D/'evaluations'/f'{tag}.json');reports[tag]=r
        for name,pts in [('normal_metrics',[p for p in r['points'] if p['kind'] in NORMAL]),('normal_action_metrics',[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8])]:
            actual=metric(pts);stored=r[name];assert actual['known']==stored['known'] and actual['correct']==stored['correct'] and actual['groups']==stored['groups'];assert abs(actual['macro_recall']-stored['macro_recall'])<1e-12
        for sq,fb in zip(r['sequences'],r['feedback']):
            a=np.load(D/'predictions'/tag/f"{sq['key']}.npz");b=np.load(T/'predictions/current_42'/f"{sq['key']}.npz");assert np.array_equal(a['current_probability'],b['probability'])
            n=len(a['probability']);assert a['probability'].shape==(n,9) and np.isfinite(a['probability']).all();assert np.allclose(a['probability'].sum(1),1,atol=1e-6)
            assert np.array_equal(a['frame_id'],np.arange(n)) and sq['comparison_frames']==n and sq['comparison_frame_range']==fb['full_native_frame_range']==[0,n-1]
            assert sq['whole_sign_correctness'] is None and fb['whole_sign_correctness'] is None
            base=read(meta[sq['key']]['base_path']);assert len(fb['guide_time_evidence'])==len(base['stages'])
            for stage,prior in zip(fb['guide_time_evidence'],base['stages']):
                for k in ['features','relations','frame_ids','source_phrases','start_seconds','end_seconds']:assert stage[k]==prior.get(k,[])
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and fb['status']=='OBSERVATION_GAP_NOT_CERTIFIED'
            arrays+=1
    assert arrays==528
    r=reports[plan['primary']];normal=[q for q in r['sequences'] if q['kind'] in NORMAL];by={q['key']:q for q in normal}
    ok={q['key'] for q in normal if q['status']==SUPPORTED};bad=[q['key'] for q in normal if q['status']=='PHYSICAL_ORDER_CONFLICT']
    prior=read(A/'evaluations/temporal_42.json');before={q['key'] for q in prior['sequences'] if q['kind'] in NORMAL and q['status']==SUPPORTED}
    gains=sorted(ok-before);rejected={k:dict(core_unconfirmed=by[k]['previous_reviewed_core_unconfirmed'],action_errors=by[k]['evaluation_action_errors']) for k in gains if by[k]['previous_reviewed_core_unconfirmed'] or by[k]['evaluation_action_errors']}
    reverse=[q for q in r['sequences'] if q['kind']=='actual_reverse_main' and q['precedences']];detected=[q['key'] for q in reverse if q['status']=='PHYSICAL_ORDER_CONFLICT'];false=[q['key'] for q in reverse if q['status']==SUPPORTED]
    cuts=[q for q in r['feedback'] if q['kind']=='action_cut' and q['word']=='WORD1133'];assert len(cuts)==1
    m=r['normal_action_metrics'];g=plan['gate']
    gates=dict(action_macro=m['macro_recall']>=g['normal_action_macro_min'],each_action_recall=all(v['correct']/v['known']>=g['normal_family_min_recall'] for v in m['groups'].values()),
      normal_order=len(ok)>=g['normal_supported_order_min'],previous_order_preserved=not(before-ok),normal_false_order=not bad,local_reverse=len(detected)>=g['local_reverse_detected_min'],reverse_false_support=not false,
      movement_cut_guard=all(not q['movement_confirmed'] and q['status']!='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' and not q['combined_supported'] for q in cuts),
      image_gap_guard=True,old_observations_preserved=True,no_gain_by_core_exclusion=not rejected)
    for p,h in ev['pins'].items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in read(D/'DEV_ACTION_TARGETS.json')['pins'].items():assert sha(p)==h
    result=dict(at=clock(),status='PASS_TO_ISOLATED_RUNTIME' if all(gates.values()) else 'ACTION_COMPONENT_ONLY_NOT_READY',primary=plan['primary'],gates=gates,normal_action=m,
      normal_order_before=sorted(before),normal_order_after=sorted(ok),gained=gains,lost=sorted(before-ok),gains_requiring_caution=rejected,normal_false_order=bad,local_reverse_detected=detected,local_reverse_false_supported=false,
      image_gap_cases=sum(bool(q['image_gap_frames']) for q in r['sequences']),verified_observer_arrays=arrays,normal_combined_supported=[q['key'] for q in r['feedback'] if q['kind'] in NORMAL and q['combined_supported']],
      head_median_seconds=r['head_median_seconds'],end_to_end_seconds=None,old_accuracy_not_rescored=True,verification='Same Codex separate arithmetic/regression code, not independent expert validation',whole_goal_complete=False)
    exclusive(D/'VERIFIED.json',result);print(result['status'],gates);print('action',m['correct'],'/',m['known'],'order',len(before),'->',len(ok),'local reverse',len(detected),'unaccepted gains',len(rejected))
if __name__=='__main__':main()
