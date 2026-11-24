"""Physical content and process eligibility are different outputs."""
from bridge_model import *
import evaluate_witness as old
OK=old.SUPPORTED

def observe(content,activity,row,graph):
    before=content.copy();q,legacy_qualified,role=old.observe(content,activity,row,graph)
    excluded=np.zeros(len(content),bool);excluded[q['context_frames']+q['rest_observed_frames']]=True
    top=content.argmax(1);conf=content.max(1);available=row['available']
    q['evaluation_qualified_errors']=q.pop('evaluation_action_errors')
    q['evaluation_content_errors']=[int(f) for f in np.flatnonzero((row['episode']>=0)&(row['episode']<8)) if top[f]!=row['episode'][f]]
    q['process_excluded_frames']=np.flatnonzero(excluded).tolist()
    q['content_while_nonmain_frames']=np.flatnonzero(excluded&available&(conf>=.8)&(top<8)).tolist()
    # The old internal eligibility sentinel must never be presented as a
    # physically observed resting hand. Keep genuine old observer rest separate.
    q['rest_observed_frames']=np.flatnonzero(available&(row['current'][:,9]>=.8)).tolist()
    q['scope']='Raw action configuration and process eligibility kept separately. Context/return can coexist with a hand configuration. Exclusion does not mean absence, resting shape or linguistic error.'
    q['previous_order_policy_unchanged']=True
    assert np.array_equal(content,before)
    return dict(content_probability=content.copy(),role_probability=role,activity_probability=activity,
        eligible=~excluded,sequence=q)

def behavior():
    g=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    def call(ids,roles,unknown=(),gap=()):
        n=len(ids);cur=np.zeros((n,10));cur[:,0]=1;r=dict(current=cur,available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.asarray(ids))
        r['available'][list(gap)]=False;pr=np.eye(9)[ids];act=np.eye(2)[roles];act[list(unknown)]=.5
        z=observe(pr,act,r,g);assert np.array_equal(z['content_probability'],pr);assert z['sequence']['comparison_frames']==n
        return z
    q=call([0,0,1,1,2,2,1,1],[1]*6+[0,0]);assert q['sequence']['status']==OK
    assert q['sequence']['content_while_nonmain_frames']==[6,7] and not q['sequence']['rest_observed_frames']
    assert call([0,0,1,1,0,2,2],[1]*7,unknown=[4])['sequence']['status']!=OK
    assert call([2,2,1,1,0,0],[1]*6)['sequence']['status']=='PHYSICAL_ORDER_CONFLICT'
    assert call([0,0,2,2],[1]*4)['sequence']['status']!=OK
    assert call([0,0,1,1,2,2],[0]*6)['sequence']['status']!=OK
    assert call([0,0,1,1,2,2],[1]*6,gap=[3])['sequence']['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    return dict(raw_configuration_never_overwritten=True,return_configuration_not_rest=True,short_uncertain_contrary_kept=True,reverse_detected=True,
        missing_event_not_certified=True,all_nonmain_not_complete=True,image_gap_held=True,all_native_frames_kept=True)
