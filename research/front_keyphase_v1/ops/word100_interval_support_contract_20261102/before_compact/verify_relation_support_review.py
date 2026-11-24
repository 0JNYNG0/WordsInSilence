"""Independent structural review; no models, labels, alternative scores or DEV."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np
from relation_support import relation_support

U = Path(__file__).resolve().parent
AC = U.parent / 'word100_process_contract_alignment_20261102'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def oracle(p, available):
    """Runs by transition boundaries; ordered pair support by enumeration."""
    contexts = []
    at = 0
    while at < len(available):
        if not available[at]:
            at += 1
            continue
        end = at + 1
        while end < len(available) and available[end]:
            end += 1
        runs = []
        for channel in (0, 1):
            flags = [bool(p[i][channel] >= .8) for i in range(at, end)]
            boundaries = [k for k, flag in enumerate(flags) if flag and (k == 0 or not flags[k-1])]
            channel_runs = []
            for start in boundaries:
                stop = start + 1
                while stop < len(flags) and flags[stop]:
                    stop += 1
                channel_runs.append(list(range(at + start, at + stop)))
            runs.append(channel_runs)
        contexts.append(dict(group_id=len(contexts), real_indices=list(range(at, end)),
                             setup_relation_runs=runs[0], arrival_relation_runs=runs[1]))
        at = end
    candidates = []
    for g in contexts:
        for sid, start in enumerate(g['setup_relation_runs']):
            for tid, arrival in enumerate(g['arrival_relation_runs']):
                pairs = [(s, t) for s in start for t in arrival if s < t]
                if not pairs:
                    continue
                ss = sorted({s for s, t in pairs})
                tt = sorted({t for s, t in pairs})
                candidates.append(dict(group_id=g['group_id'], setup_run_id=sid, arrival_run_id=tid,
                                       start_support_indices=ss, end_support_indices=tt,
                                       context_envelope_indices=list(range(min(ss), max(tt)+1)),
                                       former_first_only_indices=list(range(min(ss), min(tt)+1))))
    return contexts, sorted(candidates, key=lambda x:(min(x['end_support_indices']), min(x['start_support_indices'])))


def main():
    out = U / 'CODE_REVIEW.json'
    assert not out.exists()
    checks = 0
    cases = 0
    # Exhaustive five-state timelines: unavailable, neither, setup, arrival, both.
    for n in range(1, 6):
        for states in itertools.product(range(5), repeat=n):
            available = [s != 0 for s in states]
            p = [[.8 if s in (2, 4) else 0, .8 if s in (3, 4) else 0] for s in states]
            expected_contexts, expected_candidates = oracle(p, available)
            actual = relation_support(p, available)
            assert actual['contexts'] == expected_contexts
            assert len(actual['candidates']) == len(expected_candidates)
            assert actual['unavailable_indices'] == [i for i, a in enumerate(available) if not a]
            checks += 3
            for c, e in zip(actual['candidates'], expected_candidates):
                assert {k:c[k] for k in e} == e
                assert all(c[k] is None for k in ('selected_start', 'selected_end', 'process_probability',
                                                'directly_observed_process', 'inferred_continuity',
                                                'whole_word_correctness', 'learner_error'))
                checks += 2
            cases += 1
    # Review stored FIT replay independently; no probabilities generated/changed.
    stored = read(U / 'PUBLIC_SAVED_SUPPORT_REPLAY.json')
    structural = read(U / 'STRUCTURAL_CHECKS.json')
    for name, digest in structural['sha256'].items():
        assert sha(U / name) == digest
        checks += 1
    for path, digest in stored['pins'].items():
        assert sha(path) == digest
        checks += 1
    rows = [r for r in read(AC / 'EVALUATION.json')['rows']
            if r['role'] == 'fit' and r['origin'] == 'runtime_proposal']
    assert len(rows) == len(stored['rows']) == 45
    by_id = {r['case_id']:r for r in stored['rows']}
    total = 0
    max_candidates = 0
    old_preserved = 0
    for r in rows:
        saved = by_id[r['case_id']]
        contexts, candidates = oracle(r['point_probabilities'], r['image_available'])
        assert saved['point_input_sha256'] == hashlib.sha256(json.dumps(r['point_probabilities']).encode()).hexdigest()
        assert saved['rows'] == len(r['axis'])
        assert saved['candidate_count'] == len(candidates)
        assert saved['setup_runs'] == sum(len(g['setup_relation_runs']) for g in contexts)
        assert saved['arrival_runs'] == sum(len(g['arrival_relation_runs']) for g in contexts)
        assert saved['unavailable_indices'] == [i for i,a in enumerate(r['image_available']) if not a]
        assert saved['old_capped_count'] == len(r['qualification']['interval_candidates'])
        assert saved['old_predictions_changed'] is False
        checks += 8
        expected_axes = []
        for c in candidates:
            indices = c['context_envelope_indices']
            assert all(r['axis'][i]['image_available'] for i in indices)
            assert all(a < b for a,b in zip([r['axis'][i]['source_timestamp_us'] for i in indices],
                                           [r['axis'][i]['source_timestamp_us'] for i in indices][1:]))
            expected_axes.append(dict(group_id=c['group_id'], setup_run_id=c['setup_run_id'],
                                      arrival_run_id=c['arrival_run_id'],
                                      envelope_source_frame_ids=[r['axis'][i]['source_frame'] for i in indices],
                                      envelope_source_timestamps_us=[r['axis'][i]['source_timestamp_us'] for i in indices],
                                      end_support_source_frames=[r['axis'][i]['source_frame'] for i in c['end_support_indices']],
                                      no_single_event_truth=True))
            checks += 2
        assert expected_axes == saved['candidates']
        checks += 1
        for old in r['qualification']['interval_candidates']:
            assert any(c['former_first_only_indices'] == old['output_indices'] for c in candidates)
            old_preserved += 1
            checks += 1
        total += len(candidates)
        max_candidates = max(max_candidates, len(candidates))
    assert total == stored['total_candidates'] == 75
    assert max_candidates == stored['maximum_candidates'] == 3
    checks += 2
    report = dict(
        status='PASS_STRUCTURAL_PRESERVATION_NOT_READY_FOR_MODEL_SCORING_OR_RUNTIME',
        reviewer='Codex predicate_contract_review; separate code and saved-value review, not expert sign labels',
        checks=checks, exhaustive_synthetic_timelines=cases,
        saved_fit_queries=45, saved_candidates=total, saved_maximum_candidates=max_candidates,
        old_capped_candidates_preserved=old_preserved,
        code_changes_required_for_current_scope=False, source_code_modified=False,
        first_code_sha256=sha(U/'relation_support.py'),
        findings=[
            'Transition-boundary and ordered-pair enumeration independently preserve every available threshold-supported relation run and every eligible start-before-arrival pairing in the tested timelines.',
            'No two-frame minimum beyond two distinct times, no dwell requirement, no run exclusivity and no silent candidate cap. Missing images break connection; relation evidence stays in contexts even when no process pair exists.',
            'context_envelope is an unscored support hull that may include holds, returns and several repetitions. First/last support is not an exact action boundary, a complete-process label, or educational correctness.',
            'Saved public replay covers only FIT45 already-produced point arrays. All previously saved capped intervals remain represented; no uncapped old-model output was available or independently scored.',
            'Future endpoint selection must enforce selected_start < selected_end within the same availability group; support lists alone are not an unconstrained Cartesian product.',
            'Candidate count is O(setup_runs * arrival_runs). Materializing each envelope and former interval may require O(N * setup_runs * arrival_runs) total time/memory, up to cubic in N for alternating runs. FIT maximum3 is not a production bound.'
        ],
        required_before_scoring_or_online_use=[
            'Compact ranges or lazy iteration plus an explicit, reviewed execution budget/overflow state. No hidden truncation and no accepted/completed claim on overflow.',
            'Separate source-native time/PTS binding and input-mask validation at the caller; this component accepts row order only and cannot detect a missing camera timestamp or invalid availability values silently cast to bool.',
            'A new frozen scoring/supervision/evaluation contract with negative partial processes and runtime-matched scope. Preserving a bigger range alone cannot fix unchanged global confidence failures.'
        ],
        model_inference=0, training=0, development_used=0, old_result_rescoring=0,
        performance_gain_claim=False, runtime_adoption=False, whole_goal_complete=False,
        pins={str(p):sha(p) for p in [U/'SCOPE.json',U/'relation_support.py',U/'check_relation_support.py',
                                   U/'STRUCTURAL_CHECKS.json',U/'replay_saved_support.py',
                                   U/'PUBLIC_SAVED_SUPPORT_REPLAY.json',Path(__file__)]})
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(report['status'], 'checks',checks,'synthetic',cases,'FIT',len(rows),'candidates',total)


if __name__ == '__main__':
    main()
