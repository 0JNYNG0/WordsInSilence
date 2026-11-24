"""New causal diagnostic using saved predictions; no model rerun or selection."""
from collections import Counter
from phase_model import *
from sequence_evidence import observe

def main():
    assert not (S/'SEQUENCE_DIAGNOSIS.json').exists();rows=read(S/'EVALUATION_INPUTS.json')['rows'];graph=read(S/'LEARNED_STATE_GRAPH.json')['graphs'];normal=['native','reencoded_normal','speed_0.8','speed_1.25','trim_initial_rest'];result=[]
    for name in ['old_phase','phase_readout_42','temporal_readout_42']:
        e=read(S/'evaluations'/f'{name}.json');diagnostics=[]
        for r in rows:
            if r['kind'] not in ['actual_reverse_main','action_cut']:continue
            a=np.load(r['axis_path']);p=np.load(S/'predictions'/name/f"{r['key']}.npz");y=labels_for(r['clip_id'],r['mapping']);ix=np.flatnonzero(y>=0);scope=np.arange(ix.min(),ix.max()+1)
            automatic=next(q for q in e['sequences'] if q['key']==r['key']);oracle=observe(p['probability'],p['image_available'],scope,graph[r['word_id']],a['timestamp_us']/1e6)
            diagnostics.append(dict(key=r['key'],kind=r['kind'],auto_scope=[min(r['scope']),max(r['scope'])],reviewed_point_scope=[int(ix.min()),int(ix.max())],
                auto_status=automatic['status'],known_scope_status=oracle['status'],known_scope_is_runtime=False,
                state_points=[dict(frame=int(i),source_frame=int(r['mapping'][i]),target=int(y[i]),predicted=int(p['probability'][i].argmax()),probability=p['probability'][i].tolist()) for i in ix],
                oracle_evidence=oracle))
        nseq=[q for q in e['sequences'] if q['kind'] in normal]
        result.append(dict(condition=name,normal_sequence_count=len(nseq),normal_order_conflict=sum(q['status']=='PHYSICAL_ORDER_CONFLICT' for q in nseq),diagnostics=diagnostics))
    old=read(S/'evaluations/old_phase.json');direct=read(S/'evaluations/phase_readout_42.json');temporal=read(S/'evaluations/temporal_readout_42.json');maps={name:{(p['key'],p['frame']):p for p in e['points']} for name,e in [('old',old),('frame',direct),('context',temporal)]};regress=[]
    for key,p in maps['context'].items():
        if p['kind'] in normal and p['pred']!=p['y']:
            regress.append(dict(key=p['key'],frame=p['frame'],source_frame=p['source_frame'],target=p['y'],predictions={name:dict(pred=v[key]['pred'],confidence=max(v[key]['probability'])) for name,v in maps.items()}))
    out=dict(at=clock(),rows=result,normal_context_errors=regress,
       findings=[
        'Normal sparse state accuracy improves overall, but the temporal candidate labels one proximal-arm state as distal across multiple transformations of the same source moment. These are correlated variants, not independent failed people.',
        'For actual main-range reversal, the temporal candidate misses the returning proximal-arm state even when a reviewed window is supplied. Boundary improvement alone cannot solve that classification error.',
        'For the separate-hands main reversal, changing only the diagnostic window reveals the reverse order. Coarse bounds/repeated-state aggregation also affect the conclusion.',
        'Full-clip reversed FIT data does not establish local-reorder generalization. Directly reviewed physical configurations and learned temporal context need joint protection.',
        'All28normal sequence cases have zero physical-order-conflict warnings, but that does not excuse state misclassification or certify whole-sign correctness.'
       ],
       next_hypotheses_not_proven=[
        'Include actual local-main reorders in FIT supervision to reduce shortcuts from global sequence direction.',
        'Learn visible hands-down/background independently so main-state comparison is not limited to DTW-chosen bounds.',
        'Preserve confident present-state evidence while context helps uncertain frames; evaluate with the same fixed general policy, no word-specific branches.'
       ],
       no_new_inference=True,no_weight_or_threshold_changes=True,independent_expert_truth=False,not_production_feedback=True)
    exclusive(S/'SEQUENCE_DIAGNOSIS.json',out);print('Saved diagnosis; current demo preserved; primary not adopted')

if __name__=='__main__':main()
