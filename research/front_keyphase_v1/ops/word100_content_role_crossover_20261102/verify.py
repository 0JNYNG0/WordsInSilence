from pathlib import Path
import json,hashlib,numpy as np
U=Path(__file__).resolve().parent
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'};OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'FROZEN.json');results={};count=0;gaps=True
    for tag,(whichcontent,whichrole) in plan['conditions'].items():
        r=read(U/'comparisons'/f'{tag}.json')
        for case in r['cases']:
            s=case['sequence'];sources={k:np.load(Path(v['directory'])/(s['key']+'.npz')) for k,v in plan['sources'].items()}
            for k in ['current_probability','image_available','frame_id','source_frame_id','timestamp']:assert np.array_equal(sources['old'][k],sources['new'][k])
            raw=sources[whichcontent][plan['sources'][whichcontent]['content_key']];activity=sources[whichrole]['activity_probability'];a=sources['old'];n=len(raw);av=a['image_available'];rest=a['current_probability'][:,9]
            role=np.column_stack([(1-rest)*activity[:,0],(1-rest)*activity[:,1],rest]);assert np.array_equal(role,case['role_probability'])
            conflict=(rest>=.8)&(activity[:,1]>=.8)&av;eligible=~((role[:,0]+role[:,2]>=.8)&av&~conflict)
            assert np.array_equal(eligible,case['eligible']) and s['process_excluded_frames']==np.flatnonzero(~eligible).tolist()
            assert s['comparison_frames']==n and s['comparison_frame_range']==[0,n-1] and s['whole_sign_correctness'] is None
            if not av.all():gaps &= s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            qual=raw.copy();qual[~eligible]=0;qual[~eligible,8]=1;top=qual.argmax(1);conf=qual.max(1)
            by={}
            for state in s['states']:
                yes=(top==state['group'])&(conf>=.8)&av;stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
                assert state['frames']==np.flatnonzero(stable).tolist() and state['isolated_candidate_frames']==np.flatnonzero(yes&~stable).tolist();by[state['group']]=state
            if s['status']==OK:
                for e in s['precedences']:
                    aa=by[e['before']]['frames'];bb=by[e['after']]['frames'];assert aa and bb and max(aa)<min(bb)
                assert all(w['observed'] for w in s['role_main_witnesses'])
            for p in [p for p in r['points'] if p['key']==s['key']]:assert p['raw_pred']==int(raw[p['frame']].argmax()) and p['eligible']==bool(eligible[p['frame']])
            count+=1
        main=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8];ns=[c['sequence'] for c in r['cases'] if c['sequence']['kind'] in NORMAL]
        rev=[c['sequence'] for c in r['cases'] if c['sequence']['kind']=='actual_reverse_main' and c['sequence']['precedences']]
        z=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in main),used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in main),
            family_used={str(g):sum(p['y']==g and p['raw_pred']==g and p['eligible'] for p in main) for g in range(8)},
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['evaluation_content_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==OK for s in rev),false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns))
        assert z==r['metrics'] and len(main)==615 and len(ns)==28 and len(rev)==3;results[tag]=z
    for p,h in plan['pins'].items():assert sha(p)==h
    p=results[plan['primary']]
    gates=dict(raw_at_least589=p['raw_correct']>=589,used_at_least582=p['used_correct']>=582,
        family_not_worse=all(p['family_used'][g]>=v for g,v in plan['family_baseline'].items()),normal_order_at_least19=p['normal_order']>=19,
        clean_order_at_least5=p['clean_order']>=5,short_reverse_at_least2=p['short_reverse']>=2,normal_false_order0=p['false_order']==0,
        reverse_false_pass0=p['reverse_false_pass']==0,missing_held=gaps,raw_role_time_exact=True)
    assert set(gates)==set(plan['gates']) and count==176
    out=dict(status='CROSSOVER_COMPONENT_GATES_MET' if all(gates.values()) else 'CROSSOVER_GATES_NOT_MET',conditions=results,gates=gates,cases_checked=count,
        model_inference=0,training=0,whole_goal_complete=False,scope='Saved-output component attribution. Not an implemented live joint network or independent KSL expert evaluation.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
