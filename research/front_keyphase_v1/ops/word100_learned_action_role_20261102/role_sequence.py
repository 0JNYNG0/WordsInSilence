"""Interpret role without erasing observations, time, or uncertain short events."""
from role_model import *
from sequence_full import observe as old_observe
def observe(current,role,available,graph,times):
    assert current.shape==(len(role),10) and role.shape[1]==3
    original=current.copy();rp=role.argmax(1);rc=role.max(1);interpreted=current.copy()
    nonkey=(rp!=1)&(rc>=.8)&available
    # Only a separate interpretation for precedence: original coordinates,
    # shape/motion results and native times are returned unchanged elsewhere.
    interpreted[nonkey]=0;interpreted[nonkey,9]=1
    out=old_observe(interpreted,available,graph,times)
    top=current.argmax(1);conf=current.max(1);core=(rp==1)&(rc>=.8)&available
    witnessed=[]
    for g in graph['groups']:
        yes=(top==g)&(conf>=.8)&core
        stable=(yes&np.r_[False,yes[:-1]])|(yes&np.r_[yes[1:],False])
        witnessed.append(dict(group=g,frames=np.flatnonzero(stable).tolist(),observed=bool(stable.any())))
    if not all(w['observed'] for w in witnessed) and out['status'] in ['SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED','CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED']:
        out['status']='ACTION_ROLE_OR_CONFIGURATIONS_UNCONFIRMED'
    assert np.array_equal(current,original)
    out.update(role_core_witnesses=witnessed,travelling_frames=np.flatnonzero(nonkey&(rp==0)).tolist(),role_rest_frames=np.flatnonzero(nonkey&(rp==2)).tolist(),
      role_uncertain_frames=np.flatnonzero((rc<.8)|~available).tolist(),raw_configurations=old_observe(current,available,graph,times)['states'],
      observation_outputs_unchanged=True,scope='Whole native time. Only sufficiently supported role changes interpretation of configuration precedence; travelling remains movement evidence. Unknown role/short contrary events remain. Physical process only; not full linguistic correctness.')
    return out

def behavior():
    graphs=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];g=graphs['WORD0739'];a,b,c=g['groups']
    def call(ids,roles,missing=(),gg=g):
        p=np.eye(10)[ids];r=np.eye(3)[roles];av=np.ones(len(ids),bool);av[list(missing)]=False;before=p.copy()
        v=observe(p,r,av,gg,np.arange(len(ids))/30);assert np.array_equal(p,before) and v['comparison_frames']==len(ids);return v
    assert call([a,a,c,c,b,b,c,c],[1,1,0,0,1,1,1,1])['status']=='SUPPORTED_PHYSICAL_STATES_AND_ORDER_OBSERVED'
    assert call([c,c,b,b,a,a],[1]*6)['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([a,a,b,b,a,c,c],[1]*7)['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    v=call([a,a,b,b,c,c],[1]*6,[2,3]);assert v['image_gap_frames']==[2,3] and v['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    assert call([a,a,c,c],[1]*4)['status']=='IMPORTANT_STATE_OR_ORDER_UNCONFIRMED'
    gg=graphs['WORD1567'];x,y=gg['groups'];assert call([x,x,y,y,x,x,y,y],[1]*8,gg=gg)['status']=='CONFIGURATIONS_OBSERVED_ORDER_RULE_UNSUPPORTED'
    return dict(transit_not_completed_config=True,opposite_order_detected=True,single_short_contrary_retained=True,image_gap_axis_retained=True,missing_config_held=True,continuous_crossing_retained=True)
