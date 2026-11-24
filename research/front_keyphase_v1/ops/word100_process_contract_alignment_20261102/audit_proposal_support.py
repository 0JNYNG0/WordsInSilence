"""FIT-only saved structural audit. No network, score, threshold or label changes.

The alternative below is an unscored envelope of the *same* terminal relation
run. Containing reviewed frames is not a correct event, exact boundary, or
accuracy claim. This audit must not be used as an alternative evaluation score.
"""
from pathlib import Path
import json
import hashlib
from collections import Counter

U = Path(__file__).resolve().parent
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    out = U / 'PROPOSAL_SUPPORT_AUDIT.json'
    assert not out.exists()
    e = read(U / 'EVALUATION.json')
    verified = read(U / 'EVALUATION_VERIFIED.json')
    assert verified['evaluation_sha256'] == sha(U / 'EVALUATION.json')
    labels = read(U / 'NOMINAL_PROCESS_TARGETS.json')['rows']
    direct = {r['case_id']: r for r in labels
              if r['binding_kind'] == 'explicit_direct_chain'
              and r['observed_process_evidence'] == 1}
    rows = []
    for r in e['rows']:
        if r['role'] != 'fit' or r['origin'] != 'runtime_proposal':
            continue
        # Only already directly reviewed FIT chains. No DEV scope optimization.
        truth = direct.get(r['case_id'])
        if truth is None:
            continue
        frames = [a['source_frame'] for a in r['axis']]
        assert len(frames) == len(set(frames))
        witness = set(truth['source_frames'])
        assert witness.issubset(frames)
        points = r['point_probabilities']
        available = r['image_available']
        old = r['qualification']['interval_candidates']
        candidates = []
        for c in old:
            start, end = c['start'], c['end']
            tail = end
            # Same confident terminal relation run; stop at availability loss.
            while tail + 1 < len(frames) and available[tail+1] and points[tail+1][1] >= .8:
                tail += 1
            assert all(available[i] for i in range(start, tail+1))
            inside = set(frames[start:end+1])
            envelope = set(frames[start:tail+1])
            candidates.append(dict(
                saved_start=frames[start], saved_end=frames[end],
                same_terminal_run_tail=frames[tail],
                old_contains_reviewed_chain=witness.issubset(inside),
                unscored_envelope_contains_reviewed_chain=witness.issubset(envelope),
                appended_frames=tail-end, exact_span_truth=None,
                saved_local_probability=c['connected_process_probability'],
                unscored_envelope_probability=None))
        rows.append(dict(case_id=r['case_id'], physical_query=r['physical_query'],
            known_direct_chain_source_frames=truth['source_frames'],
            known_direct_chain_not_exact_event_boundary=True,
            native_scope_contains_chain=True,
            incumbent_contains_chain=any(c['old_contains_reviewed_chain'] for c in candidates),
            unscored_run_envelope_contains_chain=any(c['unscored_envelope_contains_reviewed_chain'] for c in candidates),
            no_point_candidate=not candidates,
            saved_confirmed=r['qualification']['qualified_display']==1,
            candidates=candidates))
    assert len(rows) == 44
    counts = Counter()
    for r in rows:
        for k in ('incumbent_contains_chain', 'unscored_run_envelope_contains_chain', 'no_point_candidate', 'saved_confirmed'):
            counts[k] += int(r[k])
    result = dict(status='FIT_ONLY_SAVED_PROPOSAL_SCOPE_DIAGNOSIS_NOT_NEW_EVALUATION',
        rows=rows, counts=dict(counts), directly_reviewed_fit_chains=len(rows),
        alternative_model_inference=0, training=0, development_rows_used=0,
        alternative_accuracy_claim=False, adoption=False,
        warning='An envelope can contain extra holds or repetitions, and may worsen false acceptance. No new local probability or success is inferred from containing a known chain.',
        next_question='Do learned relation runs retain the actual process while the first-terminal-only selection discards it? Any replacement must also reject incomplete/uncertain observations and validate temporal localization.',
        pins={str(p):sha(p) for p in [U/'EVALUATION.json',U/'EVALUATION_VERIFIED.json',U/'NOMINAL_PROCESS_TARGETS.json',Path(__file__)]})
    with out.open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'FIT':len(rows),**counts},ensure_ascii=False))


if __name__ == '__main__':
    main()
