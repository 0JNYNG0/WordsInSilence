"""Separate verification of saved arithmetic, provenance and preservation.
No training or model inference. Same Codex reviewer, not an independent expert.
"""
from pathlib import Path
import json,hashlib,collections
import numpy as np
import torch
from state_model import attach_component,features
from goal_contract import summarize_component
U=Path(__file__).resolve().parent;O=U.parent
read=lambda p:json.loads(p.read_text('utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(U/'TRAINING_FROZEN.json');meta=read(U/'LEARNING_INPUTS.json');tr=read(U/'TRAINING_COMPLETE.json');ev=read(U/'EVALUATION.json');dense=read(U/'DENSE_REPLAY.json')
checks=0
for p,h in {**cfg['protected'],**cfg['pins'],**meta['pins']}.items():assert sha(Path(p))==h;checks+=1
assert tr['training_frozen_sha256']==sha(U/'TRAINING_FROZEN.json')
assert ev['training_complete_sha256']==sha(U/'TRAINING_COMPLETE.json')
assert dense['plan_sha256']==sha(U/'DENSE_REPLAY_FROZEN.json');checks+=3
assert len(tr['rows'])==6 and len({(t['kind'],t['seed'])for t in tr['rows']})==6
index={r['index']:r for r in meta['rows']}
for t in tr['rows']:
    assert t['steps']==800 and t['log'][-1]['step']==800 and t['weights_changed']
    for key in ['model','predictions']:assert sha(Path(t[key]))==t[key+'_sha256'];checks+=1
    assert {index[i]['person']for i in t['fit_indices']}==set(range(2,8))
    assert len(t['fit_indices'])==len(set(t['fit_indices']))
    for i in t['fit_indices']:assert index[i]['split']=='fit'and index[i]['label']is not None and index[i]['available'];checks+=1
    state=torch.load(t['model'],map_location='cpu',weights_only=True);assert all(torch.isfinite(a).all()for a in state.values());checks+=1
fitsha={r['source_sha256']for r in meta['rows']if r['split']=='fit'};prsha={r['source_sha256']for r in meta['rows']if r['split']=='exposed_probe'}
assert len(fitsha)==24 and len(prsha)==8 and not fitsha&prsha;checks+=3
names={'forearm':['wrist_above_elbow','forearm_across','wrist_below_elbow'],'arm_region':['upper_arm_side','wrist_side'],
       'separation':['hands_near','hands_apart'],'height':['active_hand_higher','opposite_hand_higher']}
for p in ev['predictions']:
    probabilities=[]
    for t in tr['rows']:
        if t['kind']!=p['kind']:continue
        z=np.load(t['predictions']);j=int(np.where(z[p['domain']+'_indices']==p['index'])[0][0]);probabilities.append(z[p['domain']][j])
    votes=[int(a.argmax())for a in probabilities];sure=p['available']and len(set(votes))==1 and all(float(a.max())>=.8 for a in probabilities)
    actual=names[p['domain']][votes[0]]if sure else None
    assert p['decision']==actual and p['truth']==index[p['index']]['label'];checks+=2
for m in ev['metrics']:
    rr=[r for r in ev['predictions']if(r['kind'],r['split'],r['domain'])==(m['kind'],m['split'],m['domain'])and r['truth']is not None]
    assert m['n']==len(rr)and m['correct']==sum(r['decision']==r['truth']for r in rr)
    assert m['unknown']==sum(r['decision']is None for r in rr)
    assert m['wrong']==m['n']-m['correct']-m['unknown'];checks+=4
# Independent interpretation of saved states: no temporal decision function reuse.
for r in ev['temporal']+dense['rows']:
    states=r['states'];available=r['available'];path=r['path'];observed=[(i,s)for i,s in enumerate(states)if s is not None]
    ordered=any(i<j and a==path[0]and b==path[1]for i,a in observed for j,b in observed)
    goal=any(s==path[-1]for i,s in observed)
    if not observed:expected='observation_insufficient'
    elif ordered:expected='ordered_state_change_observed'
    elif not available[-1]or states[-1]is None:expected='observation_insufficient'
    elif goal:expected='goal_pose_only'if all(s==path[-1]for i,s in observed)else'order_not_supported'
    elif states[-1]==path[0]:expected='observed_before_goal'
    else:expected='unresolved_state'
    out=r.get('actual',r.get('result'));assert out['status']==expected
    assert out['goal_observed']==goal and out['ordered_transition_observed']==ordered
    assert out['semantic_completion']is None and out['educational_error']is None and out['comparison_retained'];checks+=5
    assert r['correct']==(out['status']==r['expected']);checks+=1
for s in dense['summary']:
    rr=[r for r in dense['rows']if(r['split'],r['domain'],r['variant'])==(s['split'],s['domain'],s['variant'])]
    assert len(rr)==s['n']and s['correct']==sum(r['correct']for r in rr);checks+=2
old=read(O/'word100_action_completion_binding_20261114/FULLPATH_COMPARISON.json')
normal_kept=error_kept=0
for r in old['rows']:
    a=r['conditions']['fixed_rank'];z=attach_component(a,[{'status':'observation_insufficient'}])
    z.pop('goal_state_diagnostics');z.pop('goal_state_diagnostics_display_enabled')
    assert json.dumps(z,sort_keys=True)==json.dumps(a,sort_keys=True);checks+=1
    normal_kept+=r['kind']in ['full_normal','speed075','speed125']
    error_kept+=r['kind']in ['right_closed_hand_main','main_reversed','closed_hand_and_reversed']
# A directly visible goal is not the full action; semantic error is never implied.
for status in ['ordered_state_change_observed','observed_before_goal','goal_pose_only','order_not_supported','observation_insufficient']:
    for visible in [True,False]:
        result=dict(status=status,goal_observed=status in ['ordered_state_change_observed','goal_pose_only'],ordered_transition_observed=status=='ordered_state_change_observed')
        a=summarize_component(result,[visible],[None],'arm_region')
        assert a['semantic_completion']is None and a['educational_error']is None and not a['display_enabled'];checks+=3
assert ev['preserved_existing_conditions']==246 and ev['invariance_checks']==240
components={d:all(s['correct']==s['n']for s in dense['summary']if s['split']=='exposed_probe'and s['domain']==d)and ev['observable_state_probe_pass'][d]for d in names}
known=sum(r['label']is not None for r in meta['rows']);unique_frames=len({(r['clip_id'],r['frame'])for r in meta['rows']})
result=dict(passed=True,checks=checks,source_files=32,fit_files=24,exposed_probe_files=8,reviewed_domain_hand_items=len(meta['rows']),
    reviewed_unique_frames=unique_frames,known_state_items=known,uncertain_items=len(meta['rows'])-known,fit_available_items=len(tr['rows'][0]['fit_indices']),
    training_runs=6,protected_files=len(cfg['protected']),preserved_conditions=246,normal_speed_conditions=normal_kept,shape_direction_error_conditions=error_kept,
    dense_conditions=len(dense['rows']),dense_probe_conditions=sum(r['split']=='exposed_probe'for r in dense['rows']),
    component_research_pass=components,overall_goal_completion_validated=False,web_changed=False,C_T3F_changed=False,common100_changed=False,
    private_used=False,final8_opened=False,G2='REWORK',
    verification='root Codex separate saved arithmetic/provenance/contracts. Not independent KSL expert; no new inference.',
    limitations=read(U/'LIMITATION_AUDIT.json'),pins={str(U/p):sha(U/p)for p in ['EVALUATION.json','DENSE_REPLAY.json','TRAINING_COMPLETE.json','GUIDE_STATE_BINDINGS.json','COACHING_CONTRACT.json']})
with (U/'RESULTS_VERIFIED.json').open('x',encoding='utf-8')as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({k:result[k]for k in ['passed','checks','known_state_items','reviewed_unique_frames','fit_available_items','component_research_pass','normal_speed_conditions','shape_direction_error_conditions','dense_conditions','dense_probe_conditions']},ensure_ascii=False))
