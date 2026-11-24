"""Separate saved-output arithmetic and preservation audit; no inference."""
from role_model import *
from role_sequence import behavior
from evaluate_roles import NORMAL
def metric(points):
    per={}
    for p in points:
        c=per.setdefault(str(p['y']),dict(known=0,correct=0));c['known']+=1;c['correct']+=int(p['y']==p['pred'])
    return dict(known=len(points),correct=sum(v['correct'] for v in per.values()),macro_recall=sum(v['correct']/v['known'] for v in per.values())/len(per),groups=per)
def main():
    assert not (A/'SAVED_VERIFICATION.json').exists();e=read(A/'EVALUATION.json');plan=read(A/'FROZEN_TRAINING.json');reports={};count=0
    for rr in e['reports']:
        r=read(A/'evaluations'/f"{rr['condition']}.json");reports[rr['condition']]=r
        for kind,m in r['metrics'].items():
            computed=metric([p for p in r['points'] if p['kind']==kind]);assert computed['known']==m['known'] and computed['correct']==m['correct'] and abs(computed['macro_recall']-m['macro_recall'])<1e-12 and computed['groups']==m['groups']
        assert len(r['sequences'])==88
        for seq in r['sequences']:
            saved=np.load(A/'predictions'/rr['condition']/f"{seq['key']}.npz");old=np.load(T/'predictions/current_42'/f"{seq['key']}.npz")
            assert np.array_equal(saved['current_probability'],old['probability']);n=len(saved['frame_id']);count+=1
            assert seq['comparison_frame_range']==[0,n-1] and seq['comparison_frames']==n and seq['whole_sign_correctness'] is None
            if seq['image_gap_frames']:assert seq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    r=reports[plan['primary']];normal=metric([p for p in r['points'] if p['kind'] in NORMAL]);core=normal['groups']['1']
    ss=[s for s in r['sequences'] if s['kind'] in NORMAL];ok=[s['key'] for s in ss if s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'];bad=[s['key'] for s in ss if s['status']=='PHYSICAL_ORDER_CONFLICT']
    rev=[s for s in r['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']];det=[s['key'] for s in rev if s['status']=='PHYSICAL_ORDER_CONFLICT'];false=[s['key'] for s in rev if s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    gaps=[s for s in r['sequences'] if s['image_gap_frames']];assert all(s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED' for s in gaps)
    old=read(T/'evaluations/current_42.json');old_ok=[s['key'] for s in old['sequences'] if s['kind'] in NORMAL and s['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED']
    gates=dict(normal_role_macro=normal['macro_recall']>=.85,normal_presented_recall=core['correct']/core['known']>=.95,normal_usable_order=len(ok)>=21,
      normal_false_order=not bad,local_reverse_detection=len(det)>=2,local_reverse_false_support=not false,all_image_gaps_held=True,raw_states_unchanged=True)
    for p,h in read(A/'INPUT_VERIFICATION.json')['input_pins'].items():assert sha(p)==h
    for p,h in read(A/'EVALUATION_INPUTS.json')['pins'].items():assert sha(p)==h
    for p,h in plan['pins'].items():assert sha(p)==h
    out=dict(at=clock(),status='PASS_TO_RUNTIME' if all(gates.values()) else 'ROLE_COMPONENT_ONLY_REMAINING_SEQUENCE_LIMITS',gates=gates,
      primary=plan['primary'],normal_role=normal,normal_core_recall=core['correct']/core['known'],normal_usable_order_before=old_ok,normal_usable_order_after=ok,
      old_coarse_window_supported=21,normal_false_conflicts=bad,local_reverse_detected=det,local_reverse_false_supported=false,
      local_reverse_sequences=[{k:s[k] for k in ['key','status','missing_groups','precedences','role_core_witnesses']} for s in rev],
      image_gap_cases=len(gaps),verified_arrays=count,head_median_seconds=r['head_median_seconds'],end_to_end_seconds=None,
      preservation=True,behavior=behavior(),independent_expert=False,same_codex_saved_arithmetic=True,whole_goal_complete=False)
    exclusive(A/'SAVED_VERIFICATION.json',out);print(out['status'],gates);print('normal usable',len(old_ok),'->',len(ok),'role',normal['correct'],'/',normal['known'],'reverse',len(det),'/',len(rev))
if __name__=='__main__':main()
