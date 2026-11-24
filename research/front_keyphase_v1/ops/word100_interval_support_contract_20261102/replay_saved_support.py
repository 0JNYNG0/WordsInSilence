"""Public FIT saved-output structural replay, never model accuracy evaluation."""
from pathlib import Path
import json
import hashlib
from relation_support import relation_support, inclusive_indices

U=Path(__file__).resolve().parent
AC=U.parent/'word100_process_contract_alignment_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=U/'PUBLIC_SAVED_SUPPORT_REPLAY.json'
    if out.exists():
        assert out.read_bytes() == (U/'before_compact'/out.name).read_bytes(), 'Only preserved initial result may be replaced'
    verified=read(AC/'EVALUATION_VERIFIED.json')
    assert verified['evaluation_sha256']==sha(AC/'EVALUATION.json')
    results=[];checks=0
    for r in read(AC/'EVALUATION.json')['rows']:
        if r['role']!='fit' or r['origin']!='runtime_proposal':continue
        support=relation_support(r['point_probabilities'],r['image_available'])
        assert not support['budget_exceeded'];checks+=1
        n=len(r['axis']);candidate_axes=[]
        for c in support['candidates']:
            ii=inclusive_indices(c['context_envelope_range'])
            assert len(ii)>=2 and min(ii)>=0 and max(ii)<n;checks+=1
            assert all(r['axis'][i]['image_available'] for i in ii);checks+=1
            assert all(r['point_probabilities'][i][0]>=.8 for i in inclusive_indices(c['start_support_range']));checks+=1
            assert all(r['point_probabilities'][i][1]>=.8 for i in inclusive_indices(c['end_support_range']));checks+=1
            candidate_axes.append(dict(
                group_id=c['group_id'],setup_run_id=c['setup_run_id'],arrival_run_id=c['arrival_run_id'],
                envelope_source_frame_ids=[r['axis'][i]['source_frame'] for i in ii],
                envelope_source_timestamps_us=[r['axis'][i]['source_timestamp_us'] for i in ii],
                end_support_source_frames=[r['axis'][i]['source_frame'] for i in inclusive_indices(c['end_support_range'])],
                no_single_event_truth=True))
        # Every saved old capped pair must remain represented. Structural
        # fixtures separately check uncapped enumeration. No scoring here.
        for old in r['qualification']['interval_candidates']:
            assert any(list(inclusive_indices(c['former_first_only_range']))==old['output_indices'] for c in support['candidates']);checks+=1
        results.append(dict(case_id=r['case_id'],point_input_sha256=hashlib.sha256(json.dumps(r['point_probabilities']).encode()).hexdigest(),
            rows=n,unavailable_indices=[i for bounds in support['unavailable_ranges'] for i in inclusive_indices(bounds)],
            setup_runs=sum(len(g['setup_relation_run_ranges']) for g in support['contexts']),
            arrival_runs=sum(len(g['arrival_relation_run_ranges']) for g in support['contexts']),
            candidate_count=len(support['candidates']),old_capped_count=len(r['qualification']['interval_candidates']),
            candidates=candidate_axes,old_predictions_changed=False))
    assert len(results)==45
    before=read(U/'before_compact'/'PUBLIC_SAVED_SUPPORT_REPLAY.json')
    assert results==before['rows'];checks+=1
    report=dict(status='PASS_PUBLIC_FIT_STRUCTURAL_REPLAY_NOT_PERFORMANCE',rows=results,
        checks=checks,input_queries=45,maximum_candidates=max(r['candidate_count'] for r in results),
        total_candidates=sum(r['candidate_count'] for r in results),
        no_new_probabilities=True,model_inference=0,training=0,development_used=0,
        adoption=False,accuracy_gain_claim=False,
        compact_representation_restores_initial_rows_exactly=True,
        max_pairs=256,over_budget_cases=0,
        runtime_budget_warning='All relation runs survive. If pair count exceeds max_pairs, no pair is materialized; status is UNRESOLVED_BUDGET_EXCEEDED. No model scoring is authorized by this structural replay.',
        pins={str(p):sha(p) for p in [AC/'EVALUATION.json',AC/'EVALUATION_VERIFIED.json',U/'relation_support.py',Path(__file__)]})
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],report['checks'],report['total_candidates'],report['maximum_candidates'])


if __name__=='__main__':main()
