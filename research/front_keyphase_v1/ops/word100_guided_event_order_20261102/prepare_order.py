"""Source-witness order supervision; no full-video truth from DTW."""
from pathlib import Path
import sys
U=Path(__file__).resolve().parent
sys.path.insert(0,str(U.parent/'word100_learned_flow_role_20261102'))
from flow_model import *
import evaluate_episode
from collections import Counter

def main():
    assert not (U/'ORDER_SUPERVISION.json').exists()
    rr=get_rows('fit');graphs=evaluate_episode.graphs();items=[];counts=Counter();pins={}
    for p in [D/'ACTION_TARGETS.json',E/'CONTEXT_TARGETS.json',FLOW_UNIT/'FLOW_INPUTS.json',U/'prepare_order.py',
        U.parent/'word100_guide_event_contract_20261102/GRAPH_SCOPE_AUDIT.json']:
        pins[str(p)]=sha(p)
    for r in rr:
        assert r['person'] in range(1,10);pins[r['cache']]=r['cache_sha256']
        for a,b in graphs[r['word_id']]['precedences']:
            ia=np.flatnonzero(r['episode']==a);ib=np.flatnonzero(r['episode']==b)
            if len(ia) and len(ib) and max(ia)<min(ib):target=1;reason='reviewed_A_occurrence_precedes_reviewed_B'
            elif len(ia) and len(ib) and max(ib)<min(ia):target=0;reason='reviewed_B_occurrence_precedes_reviewed_A'
            else:target=-1;reason='witness_missing_or_interleaved_not_forced_to_binary'
            # Only reviewed non-main points are negatives for main-occurrence
            # relevance. Another physical configuration is not assumed absent.
            context=np.flatnonzero((r['role']>=0)&(r['role']!=1))
            assert not set(context)&(set(ia)|set(ib))
            for swapped in [False,True]:
                first,second=(b,a) if swapped else (a,b)
                fa,fb=(ib,ia) if swapped else (ia,ib)
                t=(1-target if swapped and target>=0 else target)
                known={int(f):0 for f in context};known.update({int(f):1 for f in fa});known.update({int(f):2 for f in fb})
                record=dict(flow_key=r['flow_key'],clip_id=r['clip_id'],word_id=r['word_id'],recorded_id=r['person'],kind=r['kind'],
                    asked_before=first,asked_after=second,swapped_query=swapped,target=t,source_target=target,reason=reason,
                    first_witnesses=fa.tolist(),second_witnesses=fb.tolist(),reviewed_nonmain=context.tolist(),
                    occurrence_targets=[dict(frame=f,label=known[f]) for f in sorted(known)],frames=len(r['x']),
                    source_first_witnesses=np.asarray(r['mapping'])[fa].tolist(),source_second_witnesses=np.asarray(r['mapping'])[fb].tolist())
                items.append(record);counts[(r['kind'],t)]+=1
            if r['kind'] in ['original','speed_0.8','speed_1.25','pose_only_gap3','seated_variation']:
                assert target==1,(r['clip_id'],r['kind'],a,b)
    for i in range(0,len(items),2):
        a,b=items[i:i+2]
        assert a['flow_key']==b['flow_key'] and a['first_witnesses']==b['second_witnesses']
        assert (a['target']+b['target']==1) if a['target']>=0 else b['target']==-1
    for p,h in pins.items():assert sha(p)==h
    out=dict(at=clock(),status='PUBLIC_FIT_ORDER_SUPERVISION_PREPARED',rows=items,
        recorded_ids=sorted({r['recorded_id'] for r in items}),input_videos_and_variants=len(rr),
        known_queries=sum(r['target']>=0 for r in items),unknown_queries=sum(r['target']<0 for r in items),
        counts=[dict(kind=k[0],target=k[1],queries=v) for k,v in sorted(counts.items())],pins=pins,
        frame_target_names={'0':'source-reviewed nonmain for occurrence relevance, NOT resting hand shape','1':'source-reviewed queried first action witness','2':'source-reviewed queried second action witness'},
        query_counterexample='Same video gets A-before-B and B-before-A queries with opposite known labels. An edit artifact or word identity alone cannot solve this paired objective.',
        negative_scope='Timing-transformed reviewed action occurrences, not expert-labelled real learner errors.',
        unknown_scope='Unlabelled frames/other action families/interleaved witnesses not assigned new absence, transition or binary truth.',
        source_scope='Sparse FIT visual witnesses and sourced partial guide connections. No dense action boundaries, language acceptability, mandatory hold or repeat count claimed.',
        next_model='Shared guide-pair-conditioned temporal occurrence network. Raw physical actor fixed; supervised queried occurrence relevance plus relative-order loss. DTW flow is an input hint, never the label.',
        no_dev_labels_or_private=True,new_training=0,new_model_inference=0,whole_goal_complete=False)
    exclusive(U/'ORDER_SUPERVISION.json',out)
    print('queries',len(items),'known',out['known_queries'],'unknown',out['unknown_queries']);print(out['counts'])

if __name__=='__main__':main()
