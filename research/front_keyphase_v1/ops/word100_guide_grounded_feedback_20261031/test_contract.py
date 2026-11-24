"""Adversarial software controls; not recordings of correct/incorrect KSL."""
from engine import *
from pathlib import Path
import json
U=Path(__file__).resolve().parent

def main():
    targets=['R_index_only','R_open_four','L_open_four','R_motion_up','R_motion_down','R_motion_left']
    n=12
    q=dict(alias='control',prob=np.full((n,6),.95),support=np.ones((n,6),bool),frame_ids=np.arange(n),times_us=np.arange(n)*33333,sample_indices=np.arange(n),intervals=[[0,.4]])
    refs=[dict(copy.deepcopy(q),alias='ref'+str(i),anchors={'R_index_only':[3,4]},reference_to_query=np.arange(n)) for i in range(3)]
    clause=dict(id='shape',feature='R_index_only',expected=1,link_status='partial_supported',source_span='synthetic test source',source_url=None)
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    baseline=compare_clause(clause,refs,q,targets)
    check('matching linked feature can be observed',baseline['status']=='DESCRIBED_FEATURE_OBSERVED')
    check('reference anchor denominator six',baseline['expected_anchor_count']==6)
    altered=copy.deepcopy(q);altered['prob'][3:5,0]=.05
    difference=compare_clause(clause,refs,altered,targets)
    check('fully observed opposite state yields relevant difference candidate',difference['status']=='GUIDE_RELEVANT_DIFFERENCE_CANDIDATE')
    check('physical mismatch never becomes KSL error',difference['educational_verdict'] is None)
    unknown=copy.deepcopy(altered);unknown['support'][3:5,0]=False
    hidden=compare_clause(clause,refs,unknown,targets)
    check('occluded opposite predictions cannot trigger correction',hidden['status']=='INSUFFICIENT_LINKED_EVIDENCE')
    check('missing never shrinks denominator',hidden['expected_anchor_count']==difference['expected_anchor_count']==6)
    check('all hidden not passed',hidden['observed_anchor_count']==0)
    uncertain=copy.deepcopy(q);uncertain['prob'][3:5,0]=.5
    check('model uncertainty is not false or true',compare_clause(clause,refs,uncertain,targets)['status']=='INSUFFICIENT_LINKED_EVIDENCE')
    for state in ['rejected','unverified','unsupported']:
        c=dict(clause,link_status=state)
        check('source '+state+' is withheld',compare_clause(c,refs,altered,targets)['status']=='SOURCE_LINK_WITHHELD')
    irrelevant=copy.deepcopy(q);irrelevant['prob'][:,5]=.01
    check('unrelated motion cannot create shape correction',compare_clause(clause,refs,irrelevant,targets)==baseline)
    broken=copy.deepcopy(refs);broken[0]['support'][:,0]=False;broken[1]['support'][:,0]=False
    re=compare_clause(clause,broken,q,targets)
    check('reference missing cannot be erased',re['expected_anchor_count']==6 and re['status']=='INSUFFICIENT_LINKED_EVIDENCE')
    contradiction=copy.deepcopy(refs)
    for ref in contradiction:ref['prob'][:,0]=.01
    conflict=compare_clause(clause,contradiction,q,targets)
    check('reference model conflicting with RGB anchor is not query fault',conflict['status']=='INSUFFICIENT_LINKED_EVIDENCE' and all(r['counts']['REFERENCE_OBSERVER_CONFLICT']==2 for r in conflict['references']))
    disjoint=copy.deepcopy(refs)
    for i,ref in enumerate(disjoint):ref['reference_to_query']=np.minimum(np.arange(n)+i*3,n-1)
    all_opposite=copy.deepcopy(q);all_opposite['prob'][:,0]=.01
    check('different times from refs cannot be stitched into consensus',compare_clause(clause,disjoint,all_opposite,targets)['status']=='INSUFFICIENT_LINKED_EVIDENCE')
    order=dict(id='order',before='R_motion_up',after='R_motion_down',link_status='proxy_only',source_span='raise then lower',source_url=None)
    motion=copy.deepcopy(q);motion['prob'][:]=.05;motion['prob'][2:4,3]=.95;motion['prob'][7:9,4]=.95
    forward=check_order(order,motion,targets)
    check('explicit up before down supported',forward['status']=='DESCRIBED_ORDER_PATTERN_OBSERVED')
    reverse=copy.deepcopy(motion);reverse['prob']=motion['prob'][::-1].copy()
    check('reverse order distinguished without DTW',check_order(order,reverse,targets)['status']=='REVERSE_ORDER_CANDIDATE')
    for factor in [.75,1.25]:
        speed=copy.deepcopy(motion);speed['times_us']=np.rint(speed['times_us']/factor).astype('int64');speed['intervals']=[[0,.4/factor]]
        check(f'normal time scaling {factor} retains order',check_order(order,speed,targets)['status']==forward['status'])
    repetition=copy.deepcopy(motion);repetition['prob'][9:11,3]=.95
    check('extra observed repetition not automatic error',check_order(order,repetition,targets)['status']==forward['status'])
    missing_motion=copy.deepcopy(motion);missing_motion['support'][7:9,4]=False
    missing_order=check_order(order,missing_motion,targets)
    check('missing stage evidence not mandatory omission',missing_order['status']=='ORDER_UNCONFIRMED' and missing_order['mandatory_omission_verdict'] is None)
    jump=copy.deepcopy(motion);jump['frame_ids'][3:]+=1
    check('nonconsecutive frames cannot form stable event',not positive_runs(jump,'R_motion_up',targets))
    transient=copy.deepcopy(motion);transient['prob'][3,3]=.05
    check('single noisy positive not stable event',not positive_runs(transient,'R_motion_up',targets))
    no_semantics=[baseline,difference,hidden,forward,missing_order]
    check('all linguistic judgments stay unassigned',all(x['educational_verdict'] is None for x in no_semantics))
    before=q['prob'].tobytes()+q['support'].tobytes()
    compare_clause(dict(clause,source_span='wrong text cannot edit observations'),refs,q,targets)
    check('guide cannot modify observation output',before==q['prob'].tobytes()+q['support'].tobytes())
    info=json.loads((U.parent/'word100_stage_feedback_cycle_20261031/LEARNING_INPUTS.json').read_text(encoding='utf-8-sig'))
    names=info['rows'][0]['feature_names'];a=dict(mask=np.ones((4,len(names)),bool));old=np.ones((4,len(info['targets'])),bool)
    for j in ['mcp','pip','dip']:a['mask'][:,names.index('R_index_'+j+'_angle')]=False
    s=feature_support(a,names,info['targets'],old)
    check('missing index cannot pass generic remaining angles',not s[:,0].any() and not s[:,1].any())
    check('missing finger does not erase hand movement',s[:,info['targets'].index('R_motion_up')].all())
    for value in [np.nan,np.inf,-np.inf,-.1,1.1]:
        bad=copy.deepcopy(q);bad['prob'][3:5,0]=value
        c=compare_clause(clause,refs,bad,targets)
        json.dumps(c,allow_nan=False)
        check(f'invalid probability {value} not a fact or JSON NaN',c['status']=='INSUFFICIENT_LINKED_EVIDENCE')
        bm=copy.deepcopy(motion);bm['prob'][:,3]=value
        check(f'invalid order probability {value} not stable event',not positive_runs(bm,'R_motion_up',targets))
    check('unrecognized source link cannot permit comparison',compare_clause(dict(clause,link_status='pending_unrecognized'),refs,q,targets)['status']=='SOURCE_LINK_WITHHELD')
    out=dict(status='PASS',checks=len(checks),items=checks,scope='Synthetic observable evidence/ordering controls only; not human sign errors or linguistic tolerance labels')
    with (U/'CONTROLS.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2)
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
