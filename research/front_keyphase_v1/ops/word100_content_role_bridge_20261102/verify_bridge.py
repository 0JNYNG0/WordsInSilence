"""Independent arithmetic over saved arrays, no neural inference."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
from collections import Counter
import numpy as np,torch
V=Path(__file__).resolve().parent;O=V.parent;L=O/'word100_seated_fit_transfer_20261102'
OUT=Path('W:/WordsInSilence_derived/word100_content_role_bridge_20261102');PRIOR=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'};OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def main():
    assert not (V/'VERIFIED.json').exists();ev=read(V/'EVALUATION.json');plan=read(V/'FROZEN.json');parent=read(L/'VERIFIED.json')['conditions']['seated_variation']
    initial=torch.load(PRIOR/'runs/seated_variation/model.pt',map_location='cpu',weights_only=True);fixed=[k for k in initial if k.startswith('actor.')];out={};checked=0;reports={}
    for tag in plan['conditions']:
        r=read(OUT/'evaluations'/f'{tag}.json');reports[tag]=r
        for sq in r['sequences']:
            a=np.load(OUT/'predictions'/tag/f"{sq['key']}.npz");old=np.load(PRIOR/'predictions/seated_variation'/f"{sq['key']}.npz");n=len(a['content_probability'])
            assert np.array_equal(a['content_probability'],old['raw_action_probability'])
            for k in ['current_probability','image_available','frame_id','source_frame_id','timestamp']:assert np.array_equal(a[k],old[k]),k
            for k,w in [('content_probability',9),('role_probability',3),('activity_probability',2)]:assert a[k].shape==(n,w) and np.isfinite(a[k]).all() and np.allclose(a[k].sum(1),1,atol=1e-6)
            assert np.array_equal(a['role_probability'][:,2],old['current_probability'][:,9])
            assert np.flatnonzero(~a['eligible']).tolist()==sq['process_excluded_frames']
            assert np.flatnonzero(a['image_available']&(a['current_probability'][:,9]>=.8)).tolist()==sq['rest_observed_frames']
            assert sq['comparison_frames']==n and sq['comparison_frame_range']==[0,n-1] and sq['whole_sign_correctness'] is None
            if not a['image_available'].all():assert sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            for p in [p for p in r['points'] if p['key']==sq['key']]:
                assert p['pred']==int(a['content_probability'][p['frame']].argmax()) and p['eligible']==bool(a['eligible'][p['frame']])
            checked+=1
        major=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8];ns=[s for s in r['sequences'] if s['kind'] in NORMAL];reverse=[s for s in r['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        losses=[p for p in major if p['y']==p['pred'] and not p['eligible']];assert losses==r['main_eligibility_losses']
        content_correct=sum(p['y']==p['pred'] for p in major);assert content_correct==r['raw_major']['correct']==589 and len(major)==615
        rest=[p for p in r['role_points'] if p['kind'] in NORMAL and p['y']==2];ss=[s for s in ns if s['status']==OK]
        metrics=dict(raw_correct=content_correct,known_major=len(major),eligibility_losses=len(losses),main_eligible_correct=content_correct-len(losses),
            normal_order=len(ss),normal_clean_order=sum(not(s['evaluation_content_errors'] or s['evaluation_qualified_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ss),
            false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in reverse),short_reverse_false_supported=sum(s['status']==OK for s in reverse),
            rest_correct=sum(p['pred']==2 for p in rest),rest_known=len(rest),missing_role_event_cases=sum(any(not w['observed'] for w in s['role_main_witnesses']) for s in ns),
            head_median_seconds=r['head_median_seconds'])
        assert metrics['normal_order']==r['normal_order'] and metrics['normal_clean_order']==r['normal_clean_order'];out[tag]=metrics
        ck=OUT/'runs'/tag/'model.pt';assert sha(ck)==read(ck.parent/'COMPLETE.json')['sha256'];m=torch.load(ck,map_location='cpu',weights_only=True)
        assert all(torch.equal(m[k],initial[k]) for k in fixed)
    assert checked==176
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in ev['pins'].items():assert sha(p)==h
    p=out['content_connected'];c=out['role_only']
    gates=dict(raw_exact=True,role_loss_below_parent=p['eligibility_losses']<14,role_loss_below_matched_control=p['eligibility_losses']<c['eligibility_losses'],
        normal_order_at_least_parent=p['normal_order']>=6,normal_clean_order_at_least_parent=p['normal_clean_order']>=1,
        short_reverse=p['short_reverse']>=2,reverse_false_supported=p['short_reverse_false_supported']==0,normal_false_order=p['false_order']==0,
        image_gap_guard=True,normal_rest=p['rest_correct']>=96,whole_major_events_not_worse=p['missing_role_event_cases']<=c['missing_role_event_cases'])
    save(V/'VERIFIED.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='ROLE_COMPONENT_IMPROVEMENT_FULL_FUNCTION_PENDING' if all(gates.values()) else 'ROLE_BRIDGE_GATES_NOT_MET',
        conditions=out,component_gates=gates,raw_observation_equal_arrays=checked,frozen_actor_tensors=len(fixed),parent_role_losses=14,
        schema_change_not_accuracy_gain=True,end_to_end_seconds=None,whole_goal_complete=False,verification='Same Codex separate saved-output audit; not independent KSL expert truth.'))
    print(out);print(gates)
if __name__=='__main__':main()
