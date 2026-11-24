"""Saved-output arithmetic and invariance only, no rerun of model inference."""
from sequence_model import *
from verify_episode import metric
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def main():
    assert not (J/'VERIFIED.json').exists();ev=read(J/'EVALUATION.json');plan=read(J/'FROZEN.json');summaries={};reports={};arrays=0
    metadata={r['key']:r for r in read(B/'EVALUATION_INPUTS.json')['rows']}
    for desc in ev['reports']:
        name=desc['condition'];r=read(J/'evaluations'/f'{name}.json');reports[name]=r
        normal=[p for p in r['points'] if p['kind'] in NORMAL]
        raw=[p for p in r['raw_points'] if p['kind'] in NORMAL and p['y']<8]
        roles=[p for p in r['role_points'] if p['kind'] in NORMAL]
        for k,pts in [('normal_metrics',normal),('normal_action_metrics',[p for p in normal if p['y']<8]),('raw_normal_action_metrics',raw),('normal_role_metrics',roles)]:
            m=metric(pts);saved=r[k]
            assert m['known']==saved['known'] and m['correct']==saved['correct'] and m['groups']==saved['groups'] and abs(m['macro_recall']-saved['macro_recall'])<1e-12
        for sq,fb in zip(r['sequences'],r['feedback']):
            a=np.load(J/'predictions'/name/f"{sq['key']}.npz");b=np.load(T/'predictions/current_42'/f"{sq['key']}.npz");n=len(a['probability'])
            assert np.array_equal(a['current_probability'],b['probability'])
            assert np.array_equal(a['role_probability'][:,2],b['probability'][:,9])
            for k,w in [('probability',9),('raw_action_probability',9),('role_probability',3),('activity_probability',2)]:
                assert a[k].shape==(n,w) and np.isfinite(a[k]).all() and np.allclose(a[k].sum(1),1,atol=1e-6)
            conflicts=np.flatnonzero((a['current_probability'][:,9]>=.8)&(a['activity_probability'][:,1]>=.8)&a['image_available']).tolist()
            assert conflicts==sq['rest_action_conflict_frames']
            changed=set(sq['context_frames']+sq['rest_observed_frames']);kept=[i for i in range(n) if i not in changed]
            assert not changed.intersection(conflicts) and np.array_equal(a['probability'][kept],a['raw_action_probability'][kept])
            assert not changed.intersection(sq['role_uncertain_frames'])
            assert np.array_equal(a['frame_id'],np.arange(n)) and sq['comparison_frames']==n and sq['comparison_frame_range']==fb['full_native_frame_range']==[0,n-1]
            assert sq['whole_sign_correctness'] is None and fb['whole_sign_correctness'] is None
            base=read(metadata[sq['key']]['base_path']);assert len(fb['guide_time_evidence'])==len(base['stages'])
            for new,old in zip(fb['guide_time_evidence'],base['stages']):
                for key in ['features','relations','frame_ids','source_phrases','start_seconds','end_seconds']:assert new[key]==old.get(key,[])
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and fb['status']=='OBSERVATION_GAP_NOT_CERTIFIED'
            if conflicts:assert sq['status'] in ['REST_ACTION_OBSERVATION_CONFLICT','OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED']
            arrays+=1
        ns=[q for q in r['sequences'] if q['kind'] in NORMAL];rev=[q for q in r['sequences'] if q['kind']=='actual_reverse_main' and q['precedences']]
        effective=[p for p in normal if p['y']<8];assert [(p['key'],p['frame']) for p in raw]==[(p['key'],p['frame']) for p in effective]
        rest=[p for p in roles if p['y']==2]
        summaries[name]=dict(effective_action=r['normal_action_metrics'],raw_action=r['raw_normal_action_metrics'],role=r['normal_role_metrics'],
            normal_rest_known=len(rest),normal_rest_correct=sum(p['pred']==2 for p in rest),normal_order=sum(q['status']==SUPPORTED for q in ns),
            normal_false_order=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in ns),local_reverse=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in rev),
            reverse_false_supported=sum(q['status']==SUPPORTED for q in rev),normal_conflict_cases=sum(bool(q['rest_action_conflict_frames']) for q in ns),
            correct_actions_lost_by_context=[dict(key=a['key'],frame=a['frame'],truth=a['y']) for a,b in zip(raw,effective) if a['pred']==a['y'] and b['pred']!=b['y']],head_median_seconds=r['head_median_seconds'])
    assert arrays==176
    r=reports[plan['primary']];s=summaries[plan['primary']];normal=[q for q in r['sequences'] if q['kind'] in NORMAL];by={q['key']:q for q in normal};ok={q['key'] for q in normal if q['status']==SUPPORTED}
    old=read(A/'evaluations/temporal_42.json');before={q['key'] for q in old['sequences'] if q['kind'] in NORMAL and q['status']==SUPPORTED}
    gains=sorted(ok-before);rejected={k:dict(core_unconfirmed=by[k]['previous_reviewed_core_unconfirmed'],action_errors=by[k]['evaluation_action_errors']) for k in gains if by[k]['previous_reviewed_core_unconfirmed'] or by[k]['evaluation_action_errors']}
    cuts=[q for q in r['feedback'] if q['kind']=='action_cut' and q['word']=='WORD1133'];assert len(cuts)==1;g=plan['gate'];m=s['effective_action']
    gates=dict(action_macro=m['macro_recall']>=g['normal_action_macro_min'],each_action_recall=all(v['correct']/v['known']>=g['normal_family_min_recall'] for v in m['groups'].values()),
        normal_order=len(ok)>=g['normal_supported_order_min'],previous_order_preserved=not(before-ok),normal_false_order=s['normal_false_order']==0,
        local_reverse=s['local_reverse']>=g['local_reverse_detected_min'],reverse_false_support=s['reverse_false_supported']==0,
        movement_cut_guard=all(not q['movement_confirmed'] and q['status']!='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' and not q['combined_supported'] for q in cuts),
        image_gap_guard=True,old_observations_preserved=True,no_gain_by_core_exclusion=not rejected,
        rest_observation_preserved=True,normal_rest_preserved=s['normal_rest_correct']>=96,no_correct_action_loss_by_context=not s['correct_actions_lost_by_context'])
    for p,h in ev['pins'].items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in plan['fit_cache_pins'].items():assert sha(p)==h
    original=torch.load(D/'runs/episode_42/model.pt',map_location='cpu',weights_only=True)
    encoder=[k for k in original if any(k.startswith(n+'.') for n in ['embed','rgb','context','delta'])]
    for name in reports:
        candidate=torch.load(J/'runs'/name/'model.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(original[k],candidate['actor.'+k]) for k in encoder)
    result=dict(at=clock(),status='PASS_TO_STABILITY_AND_ISOLATED_RUNTIME' if all(gates.values()) else 'WITNESS_SEQUENCE_COMPONENT_NOT_READY',primary=plan['primary'],gates=gates,conditions=summaries,
        normal_order_before=sorted(before),normal_order_after=sorted(ok),gained=gains,lost=sorted(before-ok),gains_requiring_caution=rejected,
        exact_frozen_encoder_tensors_per_condition=len(encoder),old_observer_arrays_preserved=arrays,native_time_and_original_evidence_preserved=True,
        end_to_end_seconds=None,verification='Same Codex separate saved-output arithmetic and tensor checks, not independent KSL expert annotations.',whole_goal_complete=False)
    exclusive(J/'VERIFIED.json',result);print(result['status'],gates)
    print({k:dict(action=v['effective_action']['correct'],raw=v['raw_action']['correct'],order=v['normal_order'],rest=v['normal_rest_correct'],reverse=v['local_reverse'],context_losses=len(v['correct_actions_lost_by_context'])) for k,v in summaries.items()})
if __name__=='__main__':main()
