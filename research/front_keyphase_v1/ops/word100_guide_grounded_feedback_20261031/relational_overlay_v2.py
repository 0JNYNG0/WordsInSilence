"""Frame-local relational audit. Never let one conflict erase other times."""
import copy

def annotate(clause):
    out=copy.deepcopy(clause)
    initial=out.get('initial_status',out['status'])
    opposite={int(c['query_frame']) for c in out.get('coherent_query_frame_consensus',[]) if c['state']=='OPPOSITE_STATE'}
    supplement=out.get('body_relation_evidence',{})
    conflicts={int(c['query_frame']) for c in supplement.get('conflict_consensus',[]) if len(set(c.get('references',[])))>=2}
    conflicts &= opposite
    remaining=opposite-conflicts
    out['initial_status']=initial
    out['conflicted_candidate_frames']=sorted(conflicts)
    out['unresolved_candidate_frames']=sorted(remaining)
    if remaining: out['status']='GUIDE_RELEVANT_DIFFERENCE_CANDIDATE'
    elif conflicts: out['status']='OBSERVER_RELATION_CONFLICT'
    else: out['status']=initial
    # Neither type of unresolved discrepancy is an educational correction.
    out['educational_verdict']=None
    out['linguistic_tolerance']=None
    return out

def selftest():
    base={'status':'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE','coherent_query_frame_consensus':[{'query_frame':10,'state':'OPPOSITE_STATE'},{'query_frame':20,'state':'OPPOSITE_STATE'}],'references':[{'reference':'a','cells':[1,2,3]}]}
    before=copy.deepcopy(base);n=0
    for frames,expected_remaining,expected_status in [([], [10,20],'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE'),([10],[20],'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE'),([10,20],[],'OBSERVER_RELATION_CONFLICT'),([99],[10,20],'GUIDE_RELEVANT_DIFFERENCE_CANDIDATE')]:
        x=copy.deepcopy(base);x['body_relation_evidence']={'conflict_consensus':[{'query_frame':f,'references':['a','b']} for f in frames]};y=annotate(x)
        assert y['unresolved_candidate_frames']==expected_remaining;n+=1
        assert y['status']==expected_status;n+=1
        assert y['references']==x['references'];n+=1
        assert y['educational_verdict'] is None;n+=1
    x=copy.deepcopy(base);x['body_relation_evidence']={'conflict_consensus':[{'query_frame':10,'references':['a','a']}]}
    assert annotate(x)['conflicted_candidate_frames']==[];n+=1
    assert base==before;n+=1
    return n

if __name__=='__main__':
    from pathlib import Path
    import json,hashlib
    u=Path(__file__).resolve().parent
    checks=selftest();saved=json.loads((u/'FINAL_RESULT.json').read_text(encoding='utf8'));out=copy.deepcopy(saved)
    states=[]
    for row in out['rows']:
        for i,c in enumerate(row['clauses']):
            revised=annotate(c);assert revised['status']==c['status']
            row['clauses'][i]=revised;states.append(revised['status'])
    out['delivery_audit']={'status':'PER_FRAME_CONFLICT_PRESERVATION_VERIFIED','changed_clause_statuses':0,'controls_passed':checks,'initial_results_preserved':True,'input_sha256':hashlib.sha256((u/'FINAL_RESULT.json').read_bytes()).hexdigest()}
    with (u/'DELIVERY_RESULT.json').open('x',encoding='utf8') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False)
    print('PASS',checks,'controls; original 41 statuses unchanged')
