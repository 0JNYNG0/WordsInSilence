"""Separate certainty of nonperformance from its preparation/rest subtype."""
from pathlib import Path
import sys,json,hashlib,numpy as np
U=Path(__file__).resolve().parent;O=U.parent
T=O/'word100_event_temporal_finetune_20261102'
sys.path.insert(0,str(T))
from experiment import eval_rows,previous,hierarchy,read,sha,exclusive,clock
OUT=Path('W:/WordsInSilence_derived/word100_event_temporal_finetune_20261102')
FLOW=Path('W:/WordsInSilence_derived/word100_learned_flow_role_20261102')
OK='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
NORMAL={'native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'}

def observe(content,activity,row,graph):
    raw=content.copy();role,conflict=hierarchy(activity,row['current'],row['available'])
    nonmain=role[:,0]+role[:,2]
    excluded=(nonmain>=.8)&row['available']&~conflict
    # Internal event-eligibility sentinel only. Never publish this as a
    # physically observed resting hand or delete the native frame.
    qualified=raw.copy();qualified[excluded]=0;qualified[excluded,8]=1
    q=previous.observe(qualified,row,graph)
    top=raw.argmax(1);conf=raw.max(1);main=(role[:,1]>=.8)&row['available']&~conflict
    witnesses=[]
    for g in graph['groups']:
        yes=(top==g)&(conf>=.8)&main
        stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
        witnesses.append(dict(group=g,frames=np.flatnonzero(stable).tolist(),observed=bool(stable.any())))
    if not all(w['observed'] for w in witnesses) and q['status'] in [OK,'CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED']:
        q['status']='ACTION_ROLE_OR_CONTENT_UNCONFIRMED'
    if conflict.any() and q['status']!='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED':q['status']='REST_ACTION_OBSERVATION_CONFLICT'
    q.update(role_main_witnesses=witnesses,process_excluded_frames=np.flatnonzero(excluded).tolist(),
        nonmain_subtype_uncertain_frames=np.flatnonzero(excluded&(role[:,[0,2]].max(1)<.8)).tolist(),
        process_role_uncertain_frames=np.flatnonzero((np.maximum(role[:,1],nonmain)<.8)|~row['available']|conflict).tolist(),
        rest_observed_frames=np.flatnonzero(row['available']&(row['current'][:,9]>=.8)).tolist(),
        rest_action_conflict_frames=np.flatnonzero(conflict).tolist(),
        evaluation_content_errors=[int(f) for f in np.flatnonzero((row['episode']>=0)&(row['episode']<8)) if top[f]!=row['episode'][f]],
        raw_observations_preserved=True,
        scope='Nonmain is the union of preparation/return and observed-rest roles. Unknown subtype is not uncertain main participation. Raw hand/action observations, all frames and missingness remain separate; no full-sign correctness.')
    assert np.array_equal(raw,content)
    return dict(content_probability=raw,role_probability=role,nonmain_probability=nonmain,eligible=~excluded,sequence=q)

def behavior():
    graph=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    def call(ids,act,rest=None,gap=()):
        n=len(ids);r=np.zeros((n,10));r[:,0]=1
        if rest is not None:r[:,0]=1-np.asarray(rest);r[:,9]=rest
        row=dict(current=r,available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.array(ids))
        row['available'][list(gap)]=False
        a=np.column_stack([1-np.asarray(act),act]);c=np.eye(9)[ids]
        z=observe(c,a,row,graph);assert np.array_equal(c,z['content_probability']) and z['sequence']['comparison_frames']==n
        return z
    # Main probability zero, nonmain probability one, while each subtype<.8.
    q=call([6,0,0,1,1,2,2],[0,1,1,1,1,1,1],rest=[.75,0,0,0,0,0,0])
    assert q['sequence']['status']==OK and q['sequence']['nonmain_subtype_uncertain_frames']==[0]
    assert not q['sequence']['rest_observed_frames'] and not q['eligible'][0]
    q2=call([6,0,0,1,1,2,2],[0,1,1,1,1,1,1],rest=[.25,0,0,0,0,0,0])
    assert np.array_equal(q['eligible'],q2['eligible']) and q2['sequence']['status']==OK
    assert call([0,0,1,1,0,2,2],[1,1,1,1,.5,1,1])['sequence']['status']!=OK
    assert call([0,0,1,1,0,2,2],[1]*7)['sequence']['status']!=OK
    assert call([2,2,1,1,0,0],[1]*6)['sequence']['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,2,2],[1]*4)['sequence']['status']!=OK
    assert call([0,0,1,1,2,2],[0]*6)['sequence']['status']!=OK
    assert call([0,0,1,1,2,2],[1]*6,gap=[3])['sequence']['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    q=call([0,0,1,1,2,2],[1]*6,rest=[0,0,1,1,0,0]);assert q['sequence']['status']=='REST_ACTION_OBSERVATION_CONFLICT' and q['eligible'][2:4].all()
    return dict(nonmain_subtype_reallocation_invariant=True,nonmain_not_observed_rest=True,raw_native_frames_preserved=True,
        short_clear_and_uncertain_contrary_kept=True,reverse_detected=True,missing_family_held=True,all_nonmain_not_complete=True,image_gap_held=True,strong_rest_main_conflict_held=True)
