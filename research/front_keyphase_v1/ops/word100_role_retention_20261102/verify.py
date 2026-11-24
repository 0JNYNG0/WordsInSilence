"""Separate saved-array arithmetic; no model forward pass or threshold search."""
from pathlib import Path
import json,hashlib,numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_role_retention_20261102')
BASE=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102')
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (U/'VERIFIED.json').exists();plan=read(U/'TRAINING_FREEZE.json');proto=read(U/'EVALUATION_PROTOCOL.json')
    assert all(proto['behavior'].values())
    base=torch.load(plan['source_checkpoint'],map_location='cpu',weights_only=True)
    results={};count=0;gaps={};fixed_fields=['current_probability','image_available','frame_id','source_frame_id','timestamp']
    for tag in plan['conditions']:
        report=read(OUT/'evaluations'/f'{tag}.json');gaps[tag]=True
        model=torch.load(OUT/'runs'/tag/'model.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v,model[k]) for k,v in base.items() if k.startswith('actor.') or k=='blank')
        assert sha(OUT/'runs'/tag/'model.pt')==read(OUT/'runs'/tag/'COMPLETE.json')['sha256']
        assert len(report['sequences'])==88 and len({s['key'] for s in report['sequences']})==88
        for s in report['sequences']:
            a=np.load(OUT/'predictions'/tag/(s['key']+'.npz'));b=np.load(BASE/'predictions/new_adapt'/(s['key']+'.npz'))
            assert np.array_equal(a['content_probability'],b['raw_action_probability'])
            for k in fixed_fields:assert np.array_equal(a[k],b[k]),(tag,s['key'],k)
            raw=a['content_probability'];act=a['activity_probability'];av=a['image_available'];rest=a['current_probability'][:,9];n=len(raw)
            assert np.isfinite(act).all() and np.allclose(act.sum(1),1,atol=1e-6)
            role=np.column_stack([(1-rest)*act[:,0],(1-rest)*act[:,1],rest]);assert np.array_equal(role,a['role_probability'])
            conflict=(rest>=.8)&(act[:,1]>=.8)&av;eligible=~((role[:,0]+role[:,2]>=.8)&av&~conflict)
            assert np.array_equal(eligible,a['eligible']) and s['process_excluded_frames']==np.flatnonzero(~eligible).tolist()
            assert s['comparison_frames']==n and s['comparison_frame_range']==[0,n-1] and s['whole_sign_correctness'] is None
            assert s['rest_action_conflict_frames']==np.flatnonzero(conflict).tolist()
            if not av.all():gaps[tag] &= s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
            qualified=raw.copy();qualified[~eligible]=0;qualified[~eligible,8]=1;top=qualified.argmax(1);confidence=qualified.max(1);by={}
            for state in s['states']:
                yes=(top==state['group'])&(confidence>=.8)&av;stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
                assert state['frames']==np.flatnonzero(stable).tolist() and state['isolated_candidate_frames']==np.flatnonzero(yes&~stable).tolist();by[state['group']]=state
            for w in s['role_main_witnesses']:
                yes=(raw.argmax(1)==w['group'])&(raw.max(1)>=.8)&(role[:,1]>=.8)&av&~conflict
                stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False]);assert w['frames']==np.flatnonzero(stable).tolist() and w['observed']==bool(stable.any())
            if s['status']==OK:
                assert all(w['observed'] for w in s['role_main_witnesses']) and not conflict.any()
                for e in s['precedences']:
                    aa=by[e['before']]['frames'];bb=by[e['after']]['frames'];assert aa and bb and max(aa)<min(bb)
            pts=[p for p in report['points'] if p['key']==s['key']]
            for p in pts:
                f=p['frame'];assert p['raw_pred']==int(raw[f].argmax()) and p['eligible']==bool(eligible[f]) and p['source_frame']==int(a['source_frame_id'][f])
            assert s['evaluation_content_errors']==[p['frame'] for p in pts if p['y']<8 and p['y']!=p['raw_pred']]
            count+=1
        pts=[p for p in report['points'] if p['kind'] in NORMAL and p['y']<8];ns=[s for s in report['sequences'] if s['kind'] in NORMAL]
        rev=[s for s in report['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
        z=dict(raw_correct=sum(p['y']==p['raw_pred'] for p in pts),used_correct=sum(p['y']==p['raw_pred'] and p['eligible'] for p in pts),
            family_used={str(g):sum(p['y']==g and p['raw_pred']==g and p['eligible'] for p in pts) for g in range(8)},
            normal_order=sum(s['status']==OK and bool(s['precedences']) for s in ns),
            clean_order=sum(s['status']==OK and bool(s['precedences']) and not(s['evaluation_action_errors'] or s['evaluation_content_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ns),
            short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in rev),reverse_false_pass=sum(s['status']==OK for s in rev),false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns))
        assert len(pts)==615 and len(ns)==28 and len(rev)==3 and all(z[k]==report['metrics'][k] for k in z)
        results[tag]=z
    for p,h in {**plan['pins'],**proto['pins']}.items():assert sha(p)==h
    p=results[plan['primary']];c=results['new_labels_only']
    gates=dict(raw_exact=all(z['raw_correct']==590 for z in results.values()),used_at_least583_and_control=p['used_correct']>=max(583,c['used_correct']),
        family_not_worse=all(p['family_used'][g]>=max(v,c['family_used'][g]) for g,v in plan['family_baseline'].items()),
        normal_order_at_least19_and_control=p['normal_order']>=max(19,c['normal_order']),clean_order_at_least5_and_control=p['clean_order']>=max(5,c['clean_order']),
        short_reverse_at_least2_and_control=p['short_reverse']>=max(2,c['short_reverse']),normal_false_order0=p['false_order']==0,reverse_false_pass0=p['reverse_false_pass']==0,
        missing_held=all(gaps.values()))
    assert set(gates)==set(plan['gates']) and count==176
    out=dict(status='RETENTION_COMPONENT_GATES_MET' if all(gates.values()) else 'RETENTION_GATES_NOT_MET',conditions=results,gates=gates,cases_checked=count,
        actor_time_images_preserved=True,model_inference=0,whole_goal_complete=False,scope='Separate Codex arithmetic and tensor verification, not independent expert truth or end-to-end feedback validation.')
    with (U/'VERIFIED.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
