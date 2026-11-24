"""Separate saved-array arithmetic, provenance, native-time and frozen-tensor audit."""
from prepare_seated import *
from collections import Counter
import torch
DEST=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
J=O/'word100_sequence_supervision_audit_20261102';K=O/'word100_coherent_sensor_input_20261102'
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'

def metrics(pp):
    cc=Counter(p['y'] for p in pp);correct=Counter(p['y'] for p in pp if p['y']==p['pred'])
    return dict(known=len(pp),correct=sum(correct.values()),macro_recall=float(np.mean([correct[k]/n for k,n in cc.items()])),
        groups={str(k):dict(known=n,correct=correct[k]) for k,n in sorted(cc.items())})

def main():
    assert not (L/'VERIFIED.json').exists();plan=read(L/'TRAINING_FREEZE.json');ev=read(L/'EVALUATION.json');summaries={};all_arrays=0
    base_meta={r['key']:r for r in read(O/'word100_learned_action_process_20261102/EVALUATION_INPUTS.json')['rows']}
    coherent={r['clip_id']:r for r in read(K/'INPUT_MANIFEST.json')['rows']}
    initial=torch.load(plan['starting_checkpoint'],map_location='cpu',weights_only=True)
    fixed=[k for k in initial if any(k.startswith('actor.'+v+'.') for v in ['embed','rgb','context','delta'])]
    for tag in plan['conditions']:
        result=read(DEST/'evaluations'/f'{tag}.json');normal=[p for p in result['points'] if p['kind'] in NORMAL and p['y']<8];raw=[p for p in result['raw_points'] if p['kind'] in NORMAL and p['y']<8]
        for key,pp in [('normal_action_metrics',normal),('raw_normal_action_metrics',raw),('native_metrics',[p for p in normal if p['kind']=='native']),('seated_metrics',[p for p in normal if p['kind']!='native'])]:
            m=metrics(pp);stored=result[key]
            for field in ['known','correct']:assert m[field]==stored[field]
            assert abs(m['macro_recall']-stored['macro_recall'])<1e-12
        for sq in result['sequences']:
            a=np.load(DEST/'predictions'/tag/f"{sq['key']}.npz");meta=base_meta[sq['key']];n=len(a['probability'])
            if sq['kind']=='native':
                previous=np.load(K/'predictions/point_ce_42'/f"{meta['clip_id']}.npz")['current_probability']
                inp=np.load(coherent[meta['clip_id']]['input']);assert np.array_equal(a['source_frame_id'],inp['frame_id']);assert np.array_equal(a['timestamp'],inp['timestamp_us']/1e6)
            else:previous=np.load(O/'word100_learned_state_context_balance_20261102/predictions/current_42'/f"{sq['key']}.npz")['probability']
            assert np.array_equal(a['current_probability'],previous) and np.array_equal(a['role_probability'][:,2],previous[:,9])
            for key,dim in [('probability',9),('raw_action_probability',9),('role_probability',3),('activity_probability',2)]:
                assert a[key].shape==(n,dim) and np.isfinite(a[key]).all() and np.allclose(a[key].sum(1),1,atol=1e-6)
            assert sq['comparison_frames']==n and sq['comparison_frame_range']==[0,n-1] and sq['whole_sign_correctness'] is None
            assert np.array_equal(a['frame_id'],np.arange(n)) and (np.diff(a['timestamp'])>0).all()
            conflicts=np.flatnonzero((previous[:,9]>=.8)&(a['activity_probability'][:,1]>=.8)&a['image_available']).tolist();assert conflicts==sq['rest_action_conflict_frames']
            changed=set(sq['context_frames']+sq['rest_observed_frames']);kept=[i for i in range(n) if i not in changed]
            assert not changed.intersection(conflicts) and not changed.intersection(sq['role_uncertain_frames'])
            assert np.array_equal(a['probability'][kept],a['raw_action_probability'][kept])
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            pp=[p for p in result['points'] if p['key']==sq['key']];rr=[p for p in result['raw_points'] if p['key']==sq['key']]
            for p,q in zip(pp,rr):assert p['pred']==int(a['probability'][p['frame']].argmax()) and q['pred']==int(a['raw_action_probability'][q['frame']].argmax())
            all_arrays+=1
        ss=[s for s in result['sequences'] if s['kind'] in NORMAL];revs=[s for s in result['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        supported=[s for s in ss if s['status']==OK];rest=[r for r in result['role_points'] if r['kind'] in NORMAL and r['y']==2]
        losses=[dict(key=a['key'],frame=a['frame'],truth=a['y']) for a,b in zip(raw,normal) if a['pred']==a['y'] and b['pred']!=b['y']]
        summaries[tag]=dict(major=metrics(normal),raw_major=metrics(raw),native=metrics([p for p in normal if p['kind']=='native']),seated=metrics([p for p in normal if p['kind']!='native']),
            order=len(supported),clean_order=sum(not(s['evaluation_action_errors'] or s['previous_reviewed_core_unconfirmed']) for s in supported),normal_false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ss),
            local_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in revs),reverse_false_supported=sum(s['status']==OK for s in revs),
            normal_rest_known=len(rest),normal_rest_correct=sum(r['pred']==2 for r in rest),normal_conflicts=sum(bool(s['rest_action_conflict_frames']) for s in ss),
            correct_action_losses=losses,head_median_seconds=result['head_median_seconds'])
        ck=DEST/'runs'/tag/'model.pt';assert sha(ck)==read(ck.parent/'COMPLETE.json')['sha256'];candidate=torch.load(ck,map_location='cpu',weights_only=True)
        assert all(torch.equal(candidate[k],initial[k]) for k in fixed)
    assert all_arrays==176
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in ev['pins'].items():assert sha(p)==h
    # Compare with saved outputs of the SAME task/input contract. This is
    # arithmetic composition of K native + J controls, not new old-model inference.
    old=read(J/'evaluations/point_ce_42.json');kn=read(K/'evaluations/point_ce_42.json')
    old_major=[p for p in old['points'] if p['kind'] in NORMAL and p['kind']!='native' and p['y']<8]+[p for p in kn['points'] if p['y']<8]
    old_ss=[s for s in old['sequences'] if s['kind'] in NORMAL and s['kind']!='native']+kn['sequences']
    baseline=dict(major=metrics(old_major),order=sum(s['status']==OK for s in old_ss),source='SavedKpoint_ce native12 + savedJpoint_ce76controls; no old inference/rescore')
    primary=summaries[plan['primary']];control=summaries['original_replay'];g=plan['gate']
    motion_path=O/'word100_learned_rgb_motion_20261102/COMPARISON.json'
    motion=next(r for r in read(motion_path)['rows'] if r['condition']=='rgb_residual' and r['seed']==42 and r['case']=='WORD1133_action_cut')
    cut_blocked=bool(motion['motions']) and not all(r['status']=='DIRECTED_MOTION_OBSERVED' for r in motion['motions'])
    assert cut_blocked
    gates=dict(action_macro=primary['major']['macro_recall']>=g['normal_action_macro_min'],family_recalls=all(v['correct']/v['known']>=g['normal_family_min_recall'] for v in primary['major']['groups'].values()),
        order=primary['order']>=g['normal_supported_order_min'],normal_false_order=primary['normal_false_order']==0,local_reverse=primary['local_reverse']>=g['local_reverse_detected_min'],
        reverse_false_supported=primary['reverse_false_supported']==0,no_correct_action_loss=not primary['correct_action_losses'],image_gap_guard=True,frozen_observer=True,
        native_not_worse_than_matched_control=primary['native']['correct']>=control['native']['correct'],seated_gain=primary['seated']['correct']>control['seated']['correct'],retained_movement_cut_guard=cut_blocked)
    exclusive(L/'VERIFIED.json',dict(at=clock(),status='COMPONENT_GATES_PASS_FULL_RUNTIME_PENDING' if all(gates.values()) else 'COMPONENT_GATES_NOT_MET',
        conditions=summaries,baseline_saved_outputs=baseline,gates=gates,checked_arrays=all_arrays,frozen_tensors=len(fixed),
        retained_motion_source=str(motion_path),retained_motion_sha256=sha(motion_path),full_feedback_runtime_integrated=False,
        end_to_end_seconds=None,verification='Same Codex separate saved-array arithmetic/tensor audit; not independent KSL expert truth.',whole_goal_complete=False))
    print({k:dict(major=v['major']['correct'],native=v['native']['correct'],seated=v['seated']['correct'],order=v['order'],clean=v['clean_order'],reverse=v['local_reverse']) for k,v in summaries.items()},flush=True)
    print('gates',gates,flush=True)
if __name__=='__main__':main()
