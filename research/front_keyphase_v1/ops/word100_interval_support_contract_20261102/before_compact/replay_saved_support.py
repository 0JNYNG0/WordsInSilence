"""Public FIT saved-output structural replay, never model accuracy evaluation."""
from pathlib import Path
import json
import hashlib
from relation_support import relation_support

U=Path(__file__).resolve().parent
AC=U.parent/'word100_process_contract_alignment_20261102'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out=U/'PUBLIC_SAVED_SUPPORT_REPLAY.json'
    assert not out.exists()
    verified=read(AC/'EVALUATION_VERIFIED.json')
    assert verified['evaluation_sha256']==sha(AC/'EVALUATION.json')
    results=[];checks=0
    for r in read(AC/'EVALUATION.json')['rows']:
        if r['role']!='fit' or r['origin']!='runtime_proposal':continue
        support=relation_support(r['point_probabilities'],r['image_available'])
        n=len(r['axis']);candidate_axes=[]
        for c in support['candidates']:
            ii=c['context_envelope_indices']
            assert len(ii)>=2 and min(ii)>=0 and max(ii)<n;checks+=1
            assert all(r['axis'][i]['image_available'] for i in ii);checks+=1
            assert all(r['point_probabilities'][i][0]>=.8 for i in c['start_support_indices']);checks+=1
            assert all(r['point_probabilities'][i][1]>=.8 for i in c['end_support_indices']);checks+=1
            candidate_axes.append(dict(
                group_id=c['group_id'],setup_run_id=c['setup_run_id'],arrival_run_id=c['arrival_run_id'],
                envelope_source_frame_ids=[r['axis'][i]['source_frame'] for i in ii],
                envelope_source_timestamps_us=[r['axis'][i]['source_timestamp_us'] for i in ii],
                end_support_source_frames=[r['axis'][i]['source_frame'] for i in c['end_support_indices']],
                no_single_event_truth=True))
        # Every earlier uncapped pair must remain represented, not just the
        # first eight older candidates. None are reclassified or re-scored.
        for old in r['qualification']['interval_candidates']:
            assert any(c['former_first_only_indices']==old['output_indices'] for c in support['candidates']);checks+=1
        results.append(dict(case_id=r['case_id'],point_input_sha256=hashlib.sha256(json.dumps(r['point_probabilities']).encode()).hexdigest(),
            rows=n,unavailable_indices=support['unavailable_indices'],
            setup_runs=sum(len(g['setup_relation_runs']) for g in support['contexts']),
            arrival_runs=sum(len(g['arrival_relation_runs']) for g in support['contexts']),
            candidate_count=len(support['candidates']),old_capped_count=len(r['qualification']['interval_candidates']),
            candidates=candidate_axes,old_predictions_changed=False))
    assert len(results)==45
    report=dict(status='PASS_PUBLIC_FIT_STRUCTURAL_REPLAY_NOT_PERFORMANCE',rows=results,
        checks=checks,input_queries=45,maximum_candidates=max(r['candidate_count'] for r in results),
        total_candidates=sum(r['candidate_count'] for r in results),
        no_new_probabilities=True,model_inference=0,training=0,development_used=0,
        adoption=False,accuracy_gain_claim=False,
        runtime_budget_warning='No silent cap. Pathological alternating probabilities can create quadratic candidate count; a reviewed runtime budget is required before scoring any candidates.',
        pins={str(p):sha(p) for p in [AC/'EVALUATION.json',AC/'EVALUATION_VERIFIED.json',U/'relation_support.py',Path(__file__)]})
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'],report['checks'],report['total_candidates'],report['maximum_candidates'])


if __name__=='__main__':main()
