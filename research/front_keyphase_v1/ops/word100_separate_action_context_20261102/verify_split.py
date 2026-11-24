"""Separate arithmetic/regression; no model inference."""
from model import *
from verify_episode import metric
NORMAL=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest']
SUPPORTED='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def main():
    assert not (F/'VERIFIED.json').exists();ev=read(F/'EVALUATION.json');plan=read(F/'FROZEN.json');reports={};summaries={};arrays=0
    meta={r['key']:r for r in read(B/'EVALUATION_INPUTS.json')['rows']}
    for summary in ev['reports']:
        name=summary['condition'];r=read(F/'evaluations'/f'{name}.json');reports[name]=r
        for key,pts in [('normal_metrics',[p for p in r['points'] if p['kind'] in NORMAL]),('normal_action_metrics',[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8]),('raw_normal_action_metrics',[p for p in r['raw_points'] if p['kind'] in NORMAL and p['y']<8]),('normal_role_metrics',[p for p in r['role_points'] if p['kind'] in NORMAL])]:
            actual=metric(pts);stored=r[key];assert actual['known']==stored['known'] and actual['correct']==stored['correct'] and actual['groups']==stored['groups'] and abs(actual['macro_recall']-stored['macro_recall'])<1e-12
        for sq,fb in zip(r['sequences'],r['feedback']):
            a=np.load(F/'predictions'/name/f"{sq['key']}.npz");b=np.load(T/'predictions/current_42'/f"{sq['key']}.npz")
            assert np.array_equal(a['current_probability'],b['probability']);n=len(a['probability'])
            for k,width in [('probability',9),('raw_action_probability',9),('role_probability',3)]:assert a[k].shape==(n,width) and np.isfinite(a[k]).all() and np.allclose(a[k].sum(1),1,atol=1e-6)
            changed=set(sq['context_frames']+sq['rest_observed_frames']);preserved=[i for i in range(n) if i not in changed]
            assert np.array_equal(a['probability'][preserved],a['raw_action_probability'][preserved])
            assert not set(sq['role_uncertain_frames'])&changed
            assert np.array_equal(a['frame_id'],np.arange(n)) and sq['comparison_frames']==n and sq['comparison_frame_range']==fb['full_native_frame_range']==[0,n-1]
            assert sq['whole_sign_correctness'] is None and fb['whole_sign_correctness'] is None
            base=read(meta[sq['key']]['base_path']);assert len(fb['guide_time_evidence'])==len(base['stages'])
            for stage,prior in zip(fb['guide_time_evidence'],base['stages']):
                for k in ['features','relations','frame_ids','source_phrases','start_seconds','end_seconds']:assert stage[k]==prior.get(k,[])
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' and fb['status']=='OBSERVATION_GAP_NOT_CERTIFIED'
            arrays+=1
        normal=[q for q in r['sequences'] if q['kind'] in NORMAL];reverse=[q for q in r['sequences'] if q['kind']=='actual_reverse_main' and q['precedences']]
        qualified=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8];raw=[p for p in r['raw_points'] if p['kind'] in NORMAL and p['y']<8]
        summaries[name]=dict(effective_action=r['normal_action_metrics'],raw_action=r['raw_normal_action_metrics'],role=r['normal_role_metrics'],
          normal_order=sum(q['status']==SUPPORTED for q in normal),normal_false_order=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in normal),
          local_reverse=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in reverse),reverse_false_supported=sum(q['status']==SUPPORTED for q in reverse),
          correct_actions_lost_by_role=[dict(key=a['key'],frame=a['frame'],truth=a['y']) for a,b in zip(raw,qualified) if a['pred']==a['y'] and b['pred']!=b['y']],head_median_seconds=r['head_median_seconds'])
    assert arrays==176
    r=reports[plan['primary']];s=summaries[plan['primary']];normal=[q for q in r['sequences'] if q['kind'] in NORMAL];by={q['key']:q for q in normal};ok={q['key'] for q in normal if q['status']==SUPPORTED}
    old=read(A/'evaluations/temporal_42.json');before={q['key'] for q in old['sequences'] if q['kind'] in NORMAL and q['status']==SUPPORTED}
    gains=sorted(ok-before);rejected={k:dict(core_unconfirmed=by[k]['previous_reviewed_core_unconfirmed'],action_errors=by[k]['evaluation_action_errors']) for k in gains if by[k]['previous_reviewed_core_unconfirmed'] or by[k]['evaluation_action_errors']}
    cuts=[q for q in r['feedback'] if q['kind']=='action_cut' and q['word']=='WORD1133'];assert len(cuts)==1;g=plan['gate'];m=s['effective_action']
    gates=dict(action_macro=m['macro_recall']>=g['normal_action_macro_min'],each_action_recall=all(v['correct']/v['known']>=g['normal_family_min_recall'] for v in m['groups'].values()),
      normal_order=len(ok)>=g['normal_supported_order_min'],previous_order_preserved=not(before-ok),normal_false_order=s['normal_false_order']==0,
      local_reverse=s['local_reverse']>=g['local_reverse_detected_min'],reverse_false_support=s['reverse_false_supported']==0,
      movement_cut_guard=all(not q['movement_confirmed'] and q['status']!='SUPPORTED_REVIEWED_ORDER_AND_MOVEMENT' and not q['combined_supported'] for q in cuts),
      image_gap_guard=True,old_observations_preserved=True,no_gain_by_core_exclusion=not rejected)
    for p,h in ev['pins'].items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in read(D/'DEV_ACTION_TARGETS.json')['pins'].items():assert sha(p)==h
    original=torch.load(D/'runs/episode_42/model.pt',map_location='cpu',weights_only=True);candidate=torch.load(F/'runs/frozen_encoder_42/model.pt',map_location='cpu',weights_only=True)
    encoder=[k for k in original if any(k.startswith(n+'.') for n in ['embed','rgb','context','delta'])]
    assert encoder and all(torch.equal(original[k],candidate['actor.'+k]) for k in encoder)
    out=dict(at=clock(),status='PASS_TO_STABILITY_AND_ISOLATED_RUNTIME' if all(gates.values()) else 'SEPARATE_CONTEXT_COMPONENT_NOT_READY',primary=plan['primary'],gates=gates,conditions=summaries,
      normal_order_before=sorted(before),normal_order_after=sorted(ok),gained=gains,lost=sorted(before-ok),gains_requiring_caution=rejected,
      exact_frozen_encoder_tensors=len(encoder),old_observer_arrays_preserved=arrays,native_time_and_original_evidence_preserved=True,
      new_observation_unknown_is_not_success=True,end_to_end_seconds=None,
      verification='Same Codex separate saved-output arithmetic and tensor invariance; not independent KSL expert truth',whole_goal_complete=False)
    exclusive(F/'VERIFIED.json',out);print(out['status'],gates);print({k:dict(effective=v['effective_action']['correct'],raw=v['raw_action']['correct'],order=v['normal_order'],reverse=v['local_reverse'],role_losses=len(v['correct_actions_lost_by_role'])) for k,v in summaries.items()})
if __name__=='__main__':main()
