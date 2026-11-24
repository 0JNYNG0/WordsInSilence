"""Bounded temporal evidence support, never a sign-correctness classifier.

Works on image-reviewed discrete states for the feasibility experiment.
Does not read the desired word, fill landmark coordinates, or train a model.
"""
MAX_US=200000
ROUNDING_US=1
MIN_SUPPORT=2

def support(states, times_us, target, mode):
    assert len(states)==len(times_us) and mode in ('A','B','C')
    assert all(x in (0,1,None) for x in states)
    assert all(b>a for a,b in zip(times_us,times_us[1:]))
    current=states[target]
    if current is not None:
        return {'value':current,'kind':'direct','support_indices':[target]}
    if mode=='A':return {'value':None,'kind':'unknown','reason':'no_current_evidence','support_indices':[]}
    past=[i for i in range(target) if times_us[target]-times_us[i] <= MAX_US+ROUNDING_US and states[i] is not None]
    if len(past)<MIN_SUPPORT:return {'value':None,'kind':'unknown','reason':'past_insufficient','support_indices':past}
    if len({states[i] for i in past})!=1:return {'value':None,'kind':'unknown','reason':'past_conflict','support_indices':past}
    pv=states[past[-1]]
    if mode=='B':return {'value':pv,'kind':'temporal_estimate','reason':'past_agreement','support_indices':past}
    future=[i for i in range(target+1,len(states)) if times_us[i]-times_us[target] <= MAX_US+ROUNDING_US and states[i] is not None]
    if len(future)<MIN_SUPPORT:return {'value':None,'kind':'unknown','reason':'future_insufficient','support_indices':past+future}
    if len({states[i] for i in future})!=1 or states[future[0]]!=pv:
        return {'value':None,'kind':'unknown','reason':'direction_conflict','support_indices':past+future}
    return {'value':pv,'kind':'temporal_estimate','reason':'bidirectional_agreement','support_indices':past+future}

def selfcheck():
    t=[i*33333 for i in range(9)]
    tests=[]
    def chk(name, assertion):
        assert assertion,name; tests.append(name)
    a=[1,1,None,1,1]
    chk('stable_gap_C',support(a,t[:5],2,'C')['value']==1)
    chk('stable_zero_not_unknown',support([0,0,None,0,0],t[:5],2,'C')['value']==0)
    chk('one_anchor_insufficient',support([None,1,None,1,None],t[:5],2,'C')['value'] is None)
    chk('change_blocks_C',support([1,1,None,0,0],t[:5],2,'C')['value'] is None)
    chk('causal_can_estimate_before_unseen_change',support([1,1,None,0,0],t[:5],2,'B')['value']==1)
    chk('direct_pulse_preserved',support([1,1,0,1,1],t[:5],2,'C')['value']==0)
    chk('hidden_pulse_cannot_be_inferred_from_endpoints',support([1,1,None,1,1],t[:5],2,'C')['value']==1)
    chk('long_gap_rejected',support([1,1,None,1,1],[0,33333,600000,900000,933333],2,'C')['value'] is None)
    chk('all_unknown_preserved',support([None]*5,t[:5],2,'C')['value'] is None)
    chk('A_never_fills',support(a,t[:5],2,'A')['value'] is None)
    # Future must not affect past-only output.
    chk('B_causal',support([1,1,None,1,1],t[:5],2,'B')==support([1,1,None,0,0],t[:5],2,'B'))
    return tests

if __name__=='__main__':print(selfcheck())
