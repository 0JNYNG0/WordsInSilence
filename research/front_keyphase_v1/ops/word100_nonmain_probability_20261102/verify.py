"""Independent categorical arithmetic over immutable source arrays."""
from pathlib import Path
import json,hashlib,numpy as np
U=Path(__file__).resolve().parent
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'};OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'FROZEN.json');result={};checked=0;gapheld=True;permodel=[]
    for spec in plan['models']:
        report=read(U/'comparisons'/f"{spec['tag']}.json");points=report['points']
        for c in report['cases']:
            a=np.load(c['source_array']);raw=a[spec['content_key']];av=a['image_available'];activity=a['activity_probability'];n=len(raw)
            role=a['role_probability'];s=c['sequence'];rest=a['current_probability'][:,9]
            expected=np.column_stack([(1-rest)*activity[:,0],(1-rest)*activity[:,1],rest]);assert np.array_equal(role,expected)
            total=role[:,0]+role[:,2];assert np.allclose(total,1-role[:,1],atol=2e-7)
            conflict=(rest>=.8)&(activity[:,1]>=.8)&av
            outside=(total>=.8)&av&~conflict;eligible=~outside
            assert np.array_equal(eligible,c['eligible']) and np.array_equal(total,c['nonmain_probability'])
            assert s['process_excluded_frames']==np.flatnonzero(outside).tolist()
            assert s['nonmain_subtype_uncertain_frames']==np.flatnonzero(outside&(role[:,[0,2]].max(1)<.8)).tolist()
            assert s['rest_observed_frames']==np.flatnonzero(av&(rest>=.8)).tolist()
            assert s['comparison_frames']==n and s['comparison_frame_range']==[0,n-1] and s['whole_sign_correctness'] is None
            if not av.all():gapheld &= s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            qual=raw.copy();qual[outside]=0;qual[outside,8]=1;top=qual.argmax(1);confidence=qual.max(1)
            for st in s['states']:
                yes=(top==st['group'])&(confidence>=.8)&av;stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
                assert st['frames']==np.flatnonzero(stable).tolist() and st['isolated_candidate_frames']==np.flatnonzero(yes&~stable).tolist()
            if s['status']==OK:
                by={v['group']:v for v in s['states']}
                for edge in s['precedences']:
                    # Runtime serializes precedence entries as evidence objects.
                    if isinstance(edge,dict):f,t=edge['before'],edge['after']
                    else:f,t=edge
                    aa=by[f]['frames'];bb=by[t]['frames'];assert aa and bb and max(aa)<min(bb)
                assert all(w['observed'] for w in s['role_main_witnesses'])
            for p in [p for p in points if p['key']==s['key']]:
                assert p['raw_pred']==int(raw[p['frame']].argmax()) and p['eligible']==bool(eligible[p['frame']])
            checked+=1
        main=[p for p in points if p['kind'] in NORMAL and p['y']<8];ns=[c['sequence'] for c in report['cases'] if c['sequence']['kind'] in NORMAL]
        rev=[c['sequence'] for c in report['cases'] if c['sequence']['kind']=='actual_reverse_main' and c['sequence']['precedences']]
        z=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in main),known_major=len(main),used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in main),
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['evaluation_content_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==OK for s in rev),false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),
            subtype_uncertain_nonmain_frames=sum(len(s['nonmain_subtype_uncertain_frames']) for s in ns))
        assert z==report['metrics'];assert len(main)==615 and len(ns)==28 and len(rev)==3
        b=spec['baseline'];permodel.append(dict(no_new_main_loss=z['used_correct']>=b['used_correct'],no_order_loss=z['normal_order']>=b['normal_order'],
            no_reverse_loss=z['short_reverse']>=b['short_reverse'],no_false_order=z['false_order']==0,no_false_pass=z['reverse_false_pass']==0,gain=z['normal_order']>b['normal_order']))
        assert z['raw_correct']==b['raw_correct'];result[spec['tag']]=z
    assert checked==440
    for p,h in plan['pins'].items():assert sha(p)==h
    gates=dict(raw_time_observations_exact=True,no_additional_known_main_exclusions_each_model=all(r['no_new_main_loss'] for r in permodel),
        no_order_regression_each_model=all(r['no_order_loss'] for r in permodel),no_short_reverse_regression_each_model=all(r['no_reverse_loss'] for r in permodel),
        normal_false_order0_each_model=all(r['no_false_order'] for r in permodel),reverse_false_pass0_each_model=all(r['no_false_pass'] for r in permodel),
        missing_held_all=gapheld,normal_order_gain_somewhere=any(r['gain'] for r in permodel))
    assert set(gates)==set(plan['gates'])
    out=dict(status='COMMON_CALCULATION_GATES_MET' if all(gates.values()) else 'COMMON_CALCULATION_REGRESSION_DETECTED',conditions=result,gates=gates,
        cases_checked=checked,old_scores_unchanged=True,neural_inference=0,web_adoption=False,whole_goal_complete=False,
        limitation='Same Codex separate arithmetic. New interpretation only; does not validate a formerly failed model or all KSL feedback.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
