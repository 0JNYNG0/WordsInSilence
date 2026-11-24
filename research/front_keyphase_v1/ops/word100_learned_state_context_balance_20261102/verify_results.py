"""Saved arithmetic/behavior checks, distinct from independent KSL truth."""
from state_model import *
from sequence_full import observe

def behavior():
    graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];g=graph['WORD0739'];a,b,c=g['groups']
    def call(ids,g=g,missing=()):
        n=len(ids);p=np.zeros((n,10));p[np.arange(n),ids]=1;av=np.ones(n,bool);av[list(missing)]=False
        return observe(p,av,g,np.arange(n)/30)
    assert call([9,9,a,a,b,b,c,c,9,9])['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    assert call([9,9,c,c,b,b,a,a,9,9])['status']=='PHYSICAL_ORDER_CONFLICT'
    absent=call([9,9,a,a,b,b,c,c,9,9],missing=[4,5]);assert absent['comparison_frames']==10 and absent['image_gap_frames']==[4,5] and absent['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    assert call([a,a,b,b,a,c,c])['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    assert call([a,a,b,b,a,a,c,c])['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    assert call([a,a,c,c])['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    cyc=graph['WORD1567'];x,y=cyc['groups'];assert call([x,x,y,y,x,x,y,y],cyc)['status']=='CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED'
    return dict(ordered=True,reversed=True,rest_retained=True,gap_axis_preserved=True,short_contrary_evidence_held=True,interleaving_held=True,missing_state_held=True,cycles_not_error=True)

def main():
    assert not (T/'SAVED_VERIFICATION.json').exists();tests=behavior();e=read(T/'EVALUATION.json');plan=read(T/'FROZEN_TRAINING.json');reports={}
    for r in e['reports']:
        full=read(T/'evaluations'/f"{r['condition']}.json");reports[r['condition']]=full
        for kind,m in full['metrics'].items():
            for label,check in [('action',lambda p:p['y']<9),('rest',lambda p:p['y']==9)]:
                pts=[p for p in full['points'] if p['kind']==kind and check(p)];counts={}
                for p in pts:
                    z=counts.setdefault(str(p['y']),[0,0]);z[0]+=1;z[1]+=int(p['y']==p['pred'])
                correct=sum(z[1] for z in counts.values());macro=sum(z[1]/z[0] for z in counts.values())/len(counts)
                assert m[label]['known']==len(pts) and m[label]['correct']==correct and abs(m[label]['macro_recall']-macro)<1e-12
                assert m==r['metrics'][kind]
        assert len(full['sequences'])==88
        for seq in full['sequences']:
            assert seq['comparison_frame_range']==[0,seq['comparison_frames']-1] and seq['whole_sign_correctness'] is None
            if seq['image_gap_frames']:assert seq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
        assert full['anchor_changed']==0
    oldr=reports['old_phase'];mainr=reports[plan['primary']];normal=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'];reverse=['actual_reverse','actual_reverse_main']
    def m(r,k):return metrics([p for p in r['points'] if p['kind'] in k and p['y']<9])
    before=m(oldr,normal);after=m(mainr,normal);br=m(oldr,reverse);ar=m(mainr,reverse)
    drops={g:after['groups'][g]['correct']/after['groups'][g]['known']-v['correct']/v['known'] for g,v in before['groups'].items()}
    rest=mainr['metrics']['native']['rest'];action=[p for p in mainr['points'] if p['kind'] in normal and p['y']<9];as_rest=sum(p['pred']==9 for p in action)/len(action)
    normal_seqs=[s for s in mainr['sequences'] if s['kind'] in normal];conflicts=[s['key'] for s in normal_seqs if s['status']=='PHYSICAL_ORDER_CONFLICT']
    main_reversed=[s for s in mainr['sequences'] if s['kind']=='actual_reverse_main'];false_supported=[s['key'] for s in main_reversed if s['precedences'] and s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    observed=[s['key'] for s in normal_seqs if s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'];old_observed=[s['key'] for s in oldr['sequences'] if s['kind'] in normal and s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    coarse_before=[s['key'] for s in read(S/'evaluations/old_phase.json')['sequences'] if s['kind'] in normal and s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    reverse_detected=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in main_reversed if s['precedences'])
    gates=dict(normal_macro=after['macro_recall']>=before['macro_recall'],per_state=min(drops.values())>=-.05,reverse_gain=ar['macro_recall']-br['macro_recall']>=.03,
      native_rest=rest['macro_recall']>=.9,action_not_rest=as_rest<=.05,anchor=mainr['anchor_changed']==0,normal_order=not conflicts,main_reverse=not false_supported,
      # Avoid calling pervasive abstention an improved working sequence system.
      normal_confirmed_not_reduced=len(observed)>=max(len(old_observed),len(coarse_before)),local_order_contrasts_detected=reverse_detected>=2)
    inputs=read(T/'EVALUATION_INPUTS.json')['rows'];preserved=all(sha(p)==h for p,h in plan['pins'].items()) and all(sha(r['input'])==r['input_sha256'] and sha(r['base_path'])==r['base_sha256'] for r in inputs);assert preserved
    summary=dict(at=clock(),status='PASS_TO_RUNTIME' if all(gates.values()) else 'RETAIN_OLD_DEMO_DIAGNOSE_REMAINING',gates=gates,
      normal_before=before,normal_after=after,reversed_before=br,reversed_after=ar,normal_group_changes=drops,native_rest=rest,normal_action_as_rest_rate=as_rest,
      normal_order_conflicts=conflicts,main_reverse_false_supported=false_supported,normal_confirmed_before=old_observed,normal_confirmed_coarse_before=coarse_before,normal_confirmed_after=observed,
      main_reversed_sequences=[{k:s[k] for k in ['key','status','missing_groups','precedences']} for s in main_reversed],
      behavior_tests=tests,head_median_seconds=mainr['head_median_seconds'],new_comparisons=528,reused_saved_comparisons=176,old_arrays_preserved=preserved,
      independent_expert=False,same_codex_arithmetic=True,whole_goal_complete=False)
    exclusive(T/'SAVED_VERIFICATION.json',summary);print(summary['status']);print('normal',before['correct'],before['known'],'->',after['correct'],after['known']);print('reverse',br['correct'],br['known'],'->',ar['correct'],ar['known']);print(gates)

if __name__=='__main__':main()
