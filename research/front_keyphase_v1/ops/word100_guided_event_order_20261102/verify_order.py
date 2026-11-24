"""Independent saved-array arithmetic. Does not call a model or decoder."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_guided_event_order_20261102')
OLD=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
INIT=Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102/run/model.pt')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def soft_order(pr,occ,av,a,b):
    wa=pr[:,a].astype(float)*occ[:,a]*av;wb=pr[:,b].astype(float)*occ[:,b]*av
    f=float(np.dot(wb,np.r_[0,np.cumsum(wa)[:-1]]));r=float(np.dot(wa,np.r_[0,np.cumsum(wb)[:-1]]))
    return (f+1e-8)/(f+r+2e-8),f+r

def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'FROZEN.json');ev=read(U/'EVALUATION.json');protocol=read(U/'EVALUATION_PROTOCOL.json')
    base=torch.load(INIT,map_location='cpu',weights_only=True);fixed=[k for k in base if k.startswith('actor.')];checked=0;result={};guide_keys=None
    for tag in plan['conditions']:
        report=read(OUT/'evaluations'/f'{tag}.json')
        for s in report['sequences']:
            a=np.load(OUT/'predictions'/tag/f"{s['key']}.npz");old=np.load(OLD/'predictions/seated_variation'/f"{s['key']}.npz")
            pr=a['content_probability'];oc=a['occurrence_probability'];act=a['activity_probability'];av=a['image_available'];n=len(pr)
            assert np.array_equal(pr,old['raw_action_probability'])
            for k in ['current_probability','image_available','frame_id','source_frame_id','timestamp']:assert np.array_equal(a[k],old[k])
            assert oc.shape==(n,8) and np.isfinite(oc).all() and (oc>=0).all() and (oc<=1).all()
            for k,width in [('content_probability',9),('activity_probability',2),('role_probability',3)]:
                assert a[k].shape==(n,width) and np.isfinite(a[k]).all() and np.allclose(a[k].sum(1),1,atol=1e-6)
            top=pr.argmax(1);selected=act[:,1].copy();ii=np.flatnonzero(top<8);selected[ii]=oc[ii,top[ii]]
            rest=a['current_probability'][:,9];role=np.column_stack([(1-rest)*(1-selected),(1-rest)*selected,rest])
            assert np.array_equal(role,a['role_probability'])
            conflict=(rest>=.8)&(selected>=.8)&av;used=role.copy();used[conflict]=1/3
            excluded=(used.argmax(1)!=1)&(used.max(1)>=.8)&av
            assert np.array_equal(~excluded,a['eligible']) and np.flatnonzero(excluded).tolist()==s['process_excluded_frames']
            assert np.flatnonzero(conflict).tolist()==s['rest_action_conflict_frames']
            assert s['comparison_frames']==n and s['comparison_frame_range']==[0,n-1] and s['whole_sign_correctness'] is None
            if not av.all():assert s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            graph=protocol['graphs'][s['word']];qualified=pr.copy();qualified[excluded]=0;qualified[excluded,8]=1
            qt=qualified.argmax(1);qc=qualified.max(1)
            for state in s['states']:
                chosen=(qt==state['group'])&(qc>=.8)&av
                keep=(chosen&np.r_[False,chosen[:-1]])|(chosen&np.r_[chosen[1:],False])
                assert np.flatnonzero(keep).tolist()==state['frames']
                assert np.flatnonzero(chosen&~keep).tolist()==state['isolated_candidate_frames']
            for edge in s['learned_order']:
                p,mass=soft_order(pr,oc,av,edge['before'],edge['after'])
                swapped,_=soft_order(pr,oc,av,edge['after'],edge['before'])
                assert abs(p-edge['forward_probability'])<2e-6 and np.isclose(mass,edge['pair_evidence_mass'],rtol=1e-5,atol=1e-6)
                assert abs(p+swapped-1)<1e-10
            if s['status']==OK:
                assert s['physical_order_status']==OK and all(e['forward_probability']>=.8 for e in s['learned_order'])
                by={z['group']:z for z in s['states']}
                for first,second in graph['precedences']:
                    aa=by[first]['frames'];bb=by[second]['frames'];assert aa and bb and max(aa)<min(bb)
                    assert not any(f>min(bb) for f in by[first]['isolated_candidate_frames'])
                    assert not any(f<max(aa) for f in by[second]['isolated_candidate_frames'])
                assert all(z['observed'] for z in s['role_main_witnesses']) and not conflict.any()
            for p in [p for p in report['points'] if p['key']==s['key']]:
                assert p['pred']==int(top[p['frame']]) and p['eligible']==bool(a['eligible'][p['frame']])
            checked+=1
        major=[p for p in report['points'] if p['kind'] in NORMAL and p['y']<8];ns=[s for s in report['sequences'] if s['kind'] in NORMAL]
        assert len(major)==615 and sum(p['y']==p['pred'] for p in major)==589
        lost=[p for p in major if p['y']==p['pred'] and not p['eligible']];assert lost==report['main_eligibility_losses']
        families={str(g):sum(p['y']==g and p['pred']==g and p['eligible'] for p in major) for g in range(8)}
        ordered=[s for s in ns if s['precedences']];assert len(ordered)==21
        supported=[s for s in ordered if s['status']==OK]
        reverse=[s for s in report['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']];assert len(reverse)==3
        rest=[p for p in report['role_points'] if p['kind'] in NORMAL and p['y']==2]
        z=dict(raw_correct=589,major_known=615,eligibility_losses=len(lost),main_eligible_correct=589-len(lost),family_eligible=families,
            normal_order=len(supported),normal_clean_order=sum(not(s['evaluation_content_errors'] or s['evaluation_qualified_errors'] or s['previous_reviewed_core_unconfirmed']) for s in supported),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in reverse),short_reverse_false_supported=sum(s['status']==OK for s in reverse),
            false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),rest_known=len(rest),rest_exact=sum(p['pred']==2 for p in rest),normal_nonmain_false_main=sum(p['pred']==1 for p in rest),
            missing_role_event_cases=sum(any(not e['observed'] for e in s['role_main_witnesses']) for s in ns),head_median_seconds=report['head_median_seconds'])
        assert z['normal_order']==report['normal_order'];result[tag]=z
        ck=OUT/'runs'/tag/'model.pt';assert sha(ck)==read(ck.parent/'COMPLETE.json')['sha256'];m=torch.load(ck,map_location='cpu',weights_only=True)
        assert all(torch.equal(m[k],base[k]) for k in fixed)
        if guide_keys is None:guide_keys=m['family_guide_keys']
        else:assert torch.equal(m['family_guide_keys'],guide_keys)
        assert read(ck.parent/'COMPLETE.json')['changes']['family_guide_keys']==0
    assert checked==176 and ev['comparisons']==176
    for p,h in plan['pins'].items():assert sha(p)==h
    p=result['directed_order'];c=result['occurrence_only']
    gates=dict(raw_observation_exact=True,major_eligible_at_least582=p['main_eligible_correct']>=582 and p['main_eligible_correct']>=c['main_eligible_correct'],
        family_eligible_not_worse=all(p['family_eligible'][g]>=n for g,n in plan['baseline_family_eligible'].items()),
        normal_order_above6_and_matched_control=p['normal_order']>6 and p['normal_order']>c['normal_order'],
        normal_clean_at_least1_and_control=p['normal_clean_order']>=max(1,c['normal_clean_order']),
        short_reverse_at_least2_and_control=p['short_reverse']>=max(2,c['short_reverse']),reversed_false_supported0=p['short_reverse_false_supported']==0,
        normal_false_order0=p['false_order']==0,image_gap_never_certified=True,normal_nonmain_false_main0=p['normal_nonmain_false_main']==0,missing_major_families0=p['missing_role_event_cases']==0)
    assert set(gates)==set(plan['gates'])
    save(U/'VERIFIED.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='ORDER_COMPONENT_IMPROVED_FULL_FUNCTION_PENDING' if all(gates.values()) else 'GUIDED_ORDER_GATES_NOT_MET',
        conditions=result,gates=gates,arrays_checked=checked,frozen_actor_tensors=len(fixed),guide_keys_unchanged=True,
        physical_gain_claimed=False,end_to_end_seconds=None,whole_goal_complete=False,verification='Same Codex separate saved-array arithmetic, not independent KSL expert labels.'))
    print(result);print(gates)

if __name__=='__main__':main()
