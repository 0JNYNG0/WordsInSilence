"""Independent saved-array and label/input audit. No neural inference."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,hashlib
import numpy as np,torch
U=Path(__file__).resolve().parent;O=U.parent
OUT=Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102')
PARENT=Path('W:/WordsInSilence_derived/word100_seated_fit_transfer_20261102')
CONTROL=O/'word100_content_role_bridge_20261102'
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)

def main():
    assert not (U/'VERIFIED.json').exists()
    ev=read(U/'EVALUATION.json');r=read(OUT/'evaluation.json');plan=read(U/'FROZEN.json');data=read(U/'FLOW_INPUTS.json')
    train=read(U/'TRAINING_RESULT.json');control=read(CONTROL/'VERIFIED.json')['conditions']['role_only'];checked=0
    fit=[];persons={'fit':set(),'dev':set()}
    for key,meta in data['items'].items():
        assert sha(meta['path'])==meta['sha256'];a=np.load(meta['path']);p=a['path'];n=len(a['features']);split=key.split('_')[0]
        persons[split].add(meta['person']);assert np.array_equal(np.unique(p[:,0]),np.arange(n))
        assert np.array_equal(np.unique(p[:,1]),np.arange(p[:,1].max()+1)) and np.isfinite(a['features']).all()
        assert np.array_equal(a['query_frame_id'],np.arange(n)) and np.array_equal(a['features'][:,5]>0,a['image_available'])
        if split=='fit':fit.append(a['features'][a['image_available'],:5])
    assert persons['fit']==set(range(1,10)) and persons['dev']=={11,12,13}
    f=np.concatenate(fit);sd=f.std(0);sd=np.where(sd>1e-6,sd,1.)
    assert np.array_equal(f.mean(0),np.asarray(data['mean'],dtype=np.float32)) and np.array_equal(sd,np.asarray(data['std'],dtype=np.float32))
    for s in r['sequences']:
        a=np.load(OUT/'predictions'/f"{s['key']}.npz");b=np.load(PARENT/'predictions/seated_variation'/f"{s['key']}.npz");n=len(a['content_probability'])
        assert np.array_equal(a['content_probability'],b['raw_action_probability'])
        for k in ['current_probability','image_available','frame_id','source_frame_id','timestamp']:assert np.array_equal(a[k],b[k]),k
        assert np.array_equal(a['role_probability'][:,2],b['current_probability'][:,9])
        for key,width in [('content_probability',9),('role_probability',3),('activity_probability',2)]:
            assert a[key].shape==(n,width) and np.isfinite(a[key]).all() and np.allclose(a[key].sum(1),1,atol=1e-6)
        assert np.flatnonzero(~a['eligible']).tolist()==s['process_excluded_frames']
        assert s['comparison_frame_range']==[0,n-1] and s['comparison_frames']==n and s['whole_sign_correctness'] is None
        if not a['image_available'].all():assert s['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
        for point in [p for p in r['points'] if p['key']==s['key']]:
            assert point['pred']==int(a['content_probability'][point['frame']].argmax()) and point['eligible']==bool(a['eligible'][point['frame']])
        checked+=1
    assert checked==88
    major=[p for p in r['points'] if p['kind'] in NORMAL and p['y']<8];ns=[s for s in r['sequences'] if s['kind'] in NORMAL]
    reverse=[s for s in r['sequences'] if s['kind']=='actual_reverse_main' and s['precedences']]
    rest=[p for p in r['role_points'] if p['kind'] in NORMAL and p['y']==2]
    raw=sum(p['y']==p['pred'] for p in major);loss=sum(p['y']==p['pred'] and not p['eligible'] for p in major)
    assert raw==589 and len(major)==615 and loss==ev['main_eligibility_losses']
    ss=[s for s in ns if s['status']==OK]
    result=dict(raw_correct=raw,known_major=615,eligibility_losses=loss,main_eligible_correct=raw-loss,normal_order=len(ss),
        normal_clean_order=sum(not(s['evaluation_content_errors'] or s['evaluation_qualified_errors'] or s['previous_reviewed_core_unconfirmed']) for s in ss),
        short_reverse=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in reverse),short_reverse_false_supported=sum(s['status']==OK for s in reverse),
        false_order=sum(s['status']=='PHYSICAL_ORDER_CONFLICT' for s in ns),rest_correct=sum(p['pred']==2 for p in rest),rest_known=len(rest),
        missing_role_event_cases=sum(any(not w['observed'] for w in s['role_main_witnesses']) for s in ns),head_median_seconds=r['head_median_seconds'])
    assert result['normal_order']==ev['normal_order']
    ck=OUT/'run/model.pt';assert sha(ck)==train['sha256']
    a=torch.load(ck,map_location='cpu',weights_only=True);b=torch.load(PARENT/'runs/seated_variation/model.pt',map_location='cpu',weights_only=True)
    fixed=[k for k in b if k.startswith('actor.')];assert all(torch.equal(a[k],b[k]) for k in fixed)
    for path,h in plan['pins'].items():assert sha(path)==h
    gates=dict(raw_exact=True,main_exclusions_below_control9=loss<control['eligibility_losses'],normal_order_above_control6=len(ss)>control['normal_order'],
        normal_clean_order_at_least1=result['normal_clean_order']>=1,short_reverse_min2=result['short_reverse']>=2,
        reverse_false_supported0=result['short_reverse_false_supported']==0,normal_false_order0=result['false_order']==0,normal_rest_min96=result['rest_correct']>=96,
        image_gap_held=True,whole_event_missing0=result['missing_role_event_cases']==0)
    assert set(gates)==set(plan['gate'])
    save(U/'VERIFIED.json',dict(at=datetime.now(timezone(timedelta(hours=9))).isoformat(),status='FLOW_ROLE_COMPONENT_IMPROVED_FULL_FUNCTION_PENDING' if all(gates.values()) else 'FLOW_ROLE_GATES_NOT_MET',
        candidate=result,matched_saved_control=control,gates=gates,arrays_checked=checked,flow_inputs_checked=len(data['items']),fixed_actor_tensors=len(fixed),
        raw_physical_gain_claimed=False,end_to_end_seconds=None,whole_goal_complete=False,
        verification='Same Codex separate saved-array arithmetic; not independent KSL expert labels.'))
    print(result);print(gates)

if __name__=='__main__':main()
