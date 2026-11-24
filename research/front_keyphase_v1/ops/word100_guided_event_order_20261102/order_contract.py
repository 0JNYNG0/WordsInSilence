"""Occurrence interpretation retains every native time and short contrary cue."""
from order_model import *
import evaluate_split as prior
OK=prior.SUPPORTED

def observe(content,activity,occurrence,row,graph):
    before=content.copy();top=content.argmax(1);n=len(top);available=row['available'];rest=row['current'][:,9]
    selected=activity[:,1].copy();hand=top<8
    selected[hand]=occurrence[np.arange(n)[hand],top[hand]]
    role=np.column_stack([(1-rest)*(1-selected),(1-rest)*selected,rest])
    conflict=(rest>=.8)&(selected>=.8)&available
    used=role.copy();used[conflict]=1/3
    q,_=prior.observe(content,used,row,graph)
    excluded=np.zeros(n,bool);excluded[q['context_frames']+q['rest_observed_frames']]=True
    if conflict.any() and q['status']!='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED':q['status']='REST_ACTION_OBSERVATION_CONFLICT'
    learned=[]
    for a,b in graph['precedences']:
        with torch.no_grad():
            p,mass=directed_probability(torch.as_tensor(content),torch.as_tensor(occurrence),torch.as_tensor(available),a,b)
        learned.append(dict(before=a,after=b,forward_probability=float(p),pair_evidence_mass=float(mass)))
    # The soft order readout can withhold a claim; it cannot overrule missing,
    # overlapping or short contradictory physical evidence into a pass.
    physical_status=q['status']
    if q['status']==OK and any(e['forward_probability']<.8 for e in learned):q['status']='LEARNED_ORDER_UNCONFIRMED'
    elif q['status']=='PHYSICAL_ORDER_CONFLICT':
        conflict_edges={(e['before'],e['after']) for e in q['precedences'] if e['status']=='PHYSICAL_ORDER_CONFLICT'}
        if not any((e['before'],e['after']) in conflict_edges and e['forward_probability']<=.2 for e in learned):
            q['status']='ORDER_MODEL_EVIDENCE_DISAGREEMENT'
    q['evaluation_qualified_errors']=q.pop('evaluation_action_errors')
    q['evaluation_content_errors']=[int(f) for f in np.flatnonzero((row['episode']>=0)&(row['episode']<8)) if top[f]!=row['episode'][f]]
    q.update(process_excluded_frames=np.flatnonzero(excluded).tolist(),rest_observed_frames=np.flatnonzero(available&(rest>=.8)).tolist(),
        rest_action_conflict_frames=np.flatnonzero(conflict).tolist(),content_while_nonmain_frames=np.flatnonzero(excluded&available&(content.max(1)>=.8)&(top<8)).tolist(),
        learned_order=learned,physical_order_status=physical_status,prior_strict_evidence_guard_preserved=True,
        scope='Observed physical content, learned guide-family occurrence relevance and directed-order question are separate. Soft order never certifies away missing or short contrary evidence.')
    assert np.array_equal(content,before)
    return dict(content_probability=content.copy(),role_probability=role,eligible=~excluded,sequence=q)

def behavior():
    g=dict(groups=[0,1,2],precedences=[[0,1],[0,2],[1,2]])
    def call(ids,nonmain=(),uncertain=(),gap=()):
        n=len(ids);pr=np.eye(9)[ids].astype(np.float32);occ=np.ones((n,8),np.float32);occ[list(nonmain)]=0;occ[list(uncertain)]=.5
        cur=np.zeros((n,10),np.float32);cur[:,0]=1
        r=dict(current=cur,available=np.ones(n,bool),times=np.arange(n)/30,y=np.ones(n,int),episode=np.asarray(ids))
        r['available'][list(gap)]=False
        z=observe(pr,np.tile([0,1],(n,1)).astype(np.float32),occ,r,g)
        assert np.array_equal(z['content_probability'],pr) and z['sequence']['comparison_frames']==n
        return z['sequence']
    assert call([0,0,1,1,2,2])['status']==OK
    assert call([2,2,1,1,0,0])['status']=='PHYSICAL_ORDER_CONFLICT'
    q=call([0]*40+[1]*40+[0]+[2]*40,uncertain=[80])
    assert q['learned_order'][0]['forward_probability']>.95 and q['status']!=OK
    assert call([0,0,1,1,2,2,1,1],nonmain=[6,7])['status']==OK
    assert call([0,0,2,2])['status']!=OK
    assert call([0,0,1,1,2,2],gap=[3])['status']=='OBSERVATION_GAP_SEQUENCE_NOT_CERTIFIED'
    assert call([0,0,1,1,2,2],nonmain=range(6))['status']!=OK
    return dict(native_axis_and_content_preserved=True,ordered=True,reversed=True,high_mean_does_not_hide_short_contrary=True,
        observed_return_not_false_order=True,missing_occurrence_held=True,image_gap_held=True,all_nonmain_not_complete=True)
