"""Separate arithmetic over saved predictions; no model or decoder invocation."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone,timedelta
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102')
OLD=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'FROZEN.json');ev=read(U/'EVALUATION.json');protocol=read(U/'EVALUATION_PROTOCOL.json')
    before=torch.load(plan['initial'],map_location='cpu',weights_only=True);results={};checked=0;gapok=True
    for tag in plan['conditions']:
        report=read(OUT/'evaluations'/f'{tag}.json');allpoints={}
        for typ in ['points','raw_points','role_points']:
            allpoints[typ]={}
            for p in report[typ]:allpoints[typ].setdefault(p['key'],[]).append(p)
        for sq in report['sequences']:
            a=np.load(OUT/'predictions'/tag/f"{sq['key']}.npz");old=np.load(OLD/'predictions/seated_variation'/f"{sq['key']}.npz")
            pr=a['raw_action_probability'];act=a['activity_probability'];av=a['image_available'];n=len(pr)
            for k in ['current_probability','image_available','frame_id','source_frame_id','timestamp']:assert np.array_equal(a[k],old[k]),(tag,sq['key'],k)
            for k,w in [('raw_action_probability',9),('probability',9),('activity_probability',2),('role_probability',3)]:
                assert a[k].shape==(n,w) and np.isfinite(a[k]).all() and np.allclose(a[k].sum(1),1,atol=1e-6)
            rest=a['current_probability'][:,9];role=np.column_stack([(1-rest)*act[:,0],(1-rest)*act[:,1],rest]);assert np.array_equal(role,a['role_probability'])
            clash=(rest>=.8)&(act[:,1]>=.8)&av;used=role.copy();used[clash]=1/3
            exclude=(used.argmax(1)!=1)&(used.max(1)>=.8)&av
            effective=pr.copy();effective[exclude]=0;effective[exclude,8]=1;assert np.array_equal(effective,a['probability'])
            assert np.flatnonzero(clash).tolist()==sq['rest_action_conflict_frames']
            assert sq['comparison_frames']==n and sq['comparison_frame_range']==[0,n-1] and sq['whole_sign_correctness'] is None
            if not av.all():gapok &= sq['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            assert sq['context_frames']==np.flatnonzero(exclude&(used.argmax(1)==0)).tolist()
            graph=protocol['graphs'][sq['word']];qt=effective.argmax(1);qc=effective.max(1)
            for state in sq['states']:
                yes=(qt==state['group'])&(qc>=.8)&av;stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
                assert state['frames']==np.flatnonzero(stable).tolist() and state['isolated_candidate_frames']==np.flatnonzero(yes&~stable).tolist()
            if sq['status']==OK:
                by={s['group']:s for s in sq['states']}
                for f,t in graph['precedences']:
                    aa=by[f]['frames'];bb=by[t]['frames'];assert aa and bb and max(aa)<min(bb)
                    assert not any(i>min(bb) for i in by[f]['isolated_candidate_frames'])
                    assert not any(i<max(aa) for i in by[t]['isolated_candidate_frames'])
                assert all(w['observed'] for w in sq['role_main_witnesses']) and not clash.any()
            for typ,pp in [('points',effective),('raw_points',pr),('role_points',role)]:
                for p in allpoints[typ].get(sq['key'],[]):
                    assert p['pred']==int(pp[p['frame']].argmax()) and p['source_frame']==int(a['source_frame_id'][p['frame']])
            checked+=1
        raw=[p for p in report['raw_points'] if p['kind'] in NORMAL and p['y']<8];pts=[p for p in report['points'] if p['kind'] in NORMAL and p['y']<8]
        ns=[s for s in report['sequences'] if s['kind'] in NORMAL];rs=[s for s in report['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        rest=[p for p in report['role_points'] if p['kind'] in NORMAL and p['y']==2]
        z=dict(raw_correct=sum(p['y']==p['pred'] for p in raw),major_known=len(raw),used_correct=sum(p['y']==p['pred'] for p in pts),
            family_used={str(g):sum(p['y']==g and p['pred']==g for p in pts) for g in range(8)},
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['raw_evaluation_action_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rs),reverse_false_pass=sum(s['status']==OK for s in rs),
            false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),rest_known=len(rest),rest_exact=sum(p['pred']==2 for p in rest),
            rest_false_main=sum(p['pred']==1 for p in rest),normal_cases=len(ns),head_median_seconds=report['head_median_seconds'])
        assert z==report['summary'];assert len(raw)==615 and len(rest)==107 and len(ns)==28 and len(rs)==3
        results[tag]=z
        ck=OUT/'runs'/tag/'model.pt';assert sha(ck)==read(ck.parent/'COMPLETE.json')['sha256'];state=torch.load(ck,map_location='cpu',weights_only=True)
        fixed=[k for k in before if k.startswith(('actor.embed.','actor.rgb.')) or k=='actor.guide_vectors']
        assert all(torch.equal(state[k],before[k]) for k in fixed)
        for k in ['actor.context.weight_ih_l0','actor.delta.weight']:assert (not torch.equal(state[k],before[k]))==tag.endswith('adapt')
    assert checked==352==ev['comparisons']
    for p,h in plan['pins'].items():assert sha(p)==h
    for p,h in protocol['pins'].items():assert sha(p)==h
    p=results['new_adapt'];controls=[results[t] for t in plan['conditions'] if t!='new_adapt']
    gates=dict(raw_at_least589=p['raw_correct']>=589,used_at_least582=p['used_correct']>=582,
        family_not_worse=all(p['family_used'][g]>=v for g,v in plan['baseline_family_used'].items()),
        normal_order_above6_and_all_controls=p['normal_order']>max([6]+[c['normal_order'] for c in controls]),
        short_reverse_at_least2_and_controls=p['short_reverse']>=max([2]+[c['short_reverse'] for c in controls]),
        clean_order_at_least1_and_controls=p['clean_order']>=max([1]+[c['clean_order'] for c in controls]),
        normal_false_order0=p['false_order']==0,reversed_false_pass0=p['reverse_false_pass']==0,rest_false_main0=p['rest_false_main']==0,missing_never_supported=gapok)
    assert set(gates)==set(plan['gates'])
    out=dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='COMPONENT_GATES_MET_END_TO_END_PENDING' if all(gates.values()) else 'FACTORIAL_GATES_NOT_MET',
        conditions=results,gates=gates,arrays_checked=checked,old_models_rerun=False,source_and_protected_inputs_unchanged=True,
        full_latency_seconds=None,whole_goal_complete=False,verification='Same Codex separate arithmetic, not independent expert-labelled KSL feedback correctness.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))

if __name__=='__main__':main()
