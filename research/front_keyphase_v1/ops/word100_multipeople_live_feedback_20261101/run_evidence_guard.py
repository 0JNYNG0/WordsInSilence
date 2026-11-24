"""One fixed development replay of the separate guard, never baseline rerun."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import evidence_guard as G

U = Path(__file__).resolve().parent
PINS = {}


def read(p):
    p = Path(p).resolve()
    PINS[str(p)] = G.sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def load(p):
    p = Path(p).resolve()
    PINS[str(p)] = G.sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def save(p, value):
    p = Path(p)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return dict(path=str(p), sha256=G.sha(p))


def signature(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def arrays_hash(a):
    import hashlib
    h = hashlib.sha256()
    for k in sorted(a):
        if isinstance(a[k], np.ndarray):
            h.update(k.encode())
            G._array_hash(h, a[k])
    return h.hexdigest()


def assert_copy_only(old, new):
    """Remove additive guard fields; only documented status/message can differ."""
    def walk(a, b):
        if isinstance(a, dict):
            for k, v in a.items():
                if k == 'status' and v == G.SIMILAR and b[k] == G.INSUFFICIENT:
                    continue
                if k == 'message' and b.get('guard_previous_message') == v:
                    continue
                walk(v, b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                walk(x, y)
        else:
            assert a == b, (a, b)
    walk(old, new)


def run():
    # Collect every source and finalized implementation pin before any new path.
    for target in ['GUARD_EXECUTION.json', 'GUARD_REPORT.json', 'guard_arm_paths', 'guard_pairs', 'guard_cases']:
        if (U / target).exists():
            raise FileExistsError('Preserve existing execution: ' + target)
    frozen = read(U / 'GUARD_FROZEN.json')
    for p, h in frozen['source_pins'].items():
        assert G.sha(p) == h, ('Frozen input changed', p)
        PINS[p] = h
    for name in ['evidence_guard.py', 'guard_selftest.py', 'run_evidence_guard.py', 'GUARD_SELFTEST_FINAL.json']:
        PINS[str(U / name)] = G.sha(U / name)
    selftest = read(U / 'GUARD_SELFTEST_FINAL.json')
    assert selftest['status'] == 'PASS' and selftest['code_sha256'] == G.sha(U / 'evidence_guard.py')
    bank = read(U / 'REFERENCE_BANK.json')
    tasks = read(U / 'EVALUATION_TASKS.json')['tasks']
    evaluation = read(U / 'FEEDBACK_EVALUATION.json')
    case_pins = {v['id']: v for v in evaluation['case_index']}
    rows = {r['clip_id']: r for r in bank['rows']}
    refs = {}
    for rid, row in rows.items():
        a = load(row['input'])
        a['probabilities'] = load(row['probability_file'])['probabilities']
        a['feature_names'] = bank['feature_names']
        refs[rid] = a
    inputs, cases, comparisons = {}, {}, {}
    for task in tasks:
        tid = task['id']
        p = U / 'evaluation_cases' / (tid + '.json')
        assert G.sha(p) == case_pins[tid]['sha256']
        case = read(p)
        cases[tid] = case
        if case['modified_input']:
            source = case['modified_input']
            assert G.sha(source['path']) == source['sha256']
            a = load(source['path'])
        else:
            assert G.sha(task['source_input']) == task['source_input_sha256']
            assert G.sha(task['source_probabilities']) == task['source_probabilities_sha256']
            a = load(task['source_input'])
            a['probabilities'] = load(task['source_probabilities'])['probabilities']
        a['feature_names'] = bank['feature_names']
        inputs[tid] = a
        comparisons[tid] = []
        for ptr in case['comparisons']:
            assert G.sha(ptr['path']) == ptr['sha256']
            comparisons[tid].append(read(ptr['path']))
    source_signatures = {k: arrays_hash(v) for k, v in inputs.items()}
    ref_signatures = {k: arrays_hash(v) for k, v in refs.items()}
    save(U / 'GUARD_EXECUTION.json', dict(status='PINNED_BEFORE_PUBLIC_AUXILIARY_PATH_EXECUTION',
        created_utc=datetime.now(timezone.utc).isoformat(), source_pins=PINS,
        observer_calls=0, baseline_path_runs=0, frozen_candidate_count=1,
        cases=len(tasks), original_pairs=sum(len(v) for v in comparisons.values()),
        modes=['all9', 'subset3'], synthetic_checks=selftest['checks'],
        authorization='Root authorized one fixed diagnostic; no live attachment by this script.'))
    (U / 'guard_pairs').mkdir()
    (U / 'guard_cases').mkdir()
    cache = G.ArmPathCache(U / 'guard_arm_paths')
    tally = defaultdict(lambda: {'stages': Counter(), 'features': Counter(), 'cases': 0,
                                'stage_opportunities': 0, 'feature_opportunities': 0})
    changed = []
    case_index = []
    pair_index = []
    start = time.perf_counter()
    for task in tasks:
        tid = task['id']
        c = cases[tid]
        q = inputs[tid]
        mode_outputs = {}
        pair_pointers = {}
        for mode in ['all9', 'subset3']:
            available = task['references'] if mode == 'all9' else task['subset_references']
            rawrefs = [p for p in comparisons[tid] if p['id'] in available]
            base = dict(word=task['target_word'], stages=deepcopy(c['modes'][mode]['stages']),
                        references=deepcopy(rawrefs), selected_whole_references=c['modes'][mode]['selected_whole_references'])
            old_signature = signature(base)
            guarded = G.guard_result(q, rows, refs, base, cache=cache)
            assert signature(base) == old_signature
            assert_copy_only(base, guarded)
            assert base['references'] == guarded['references'], 'raw saved pair values changed'
            assert base['selected_whole_references'] == guarded['selected_whole_references']
            assert arrays_hash(q) == source_signatures[tid]
            ga = guarded['guard_audit']
            if mode == 'all9':
                rawptr = {p['reference']: p for p in c['comparisons']}
                for rid, pa in ga['pairs'].items():
                    value = dict(task_id=tid, query=task['query'], reference_id=rid,
                                 source_pair=rawptr[rid], guard=pa)
                    ptr = save(U / 'guard_pairs' / (tid + '__' + rid + '.json'), value)
                    pair_pointers[rid] = ptr
                    pair_index.append(dict(task_id=tid, reference_id=rid, **ptr))
            else:
                for rid, pa in ga['pairs'].items():
                    # Same helper, same audit; no mode-specific path adaptation.
                    stored = json.loads(Path(pair_pointers[rid]['path']).read_text(encoding='utf-8'))['guard']
                    assert stored == pa
            key = (task['kind'], mode)
            group = tally[key]
            group['cases'] += 1
            for si, (old, new) in enumerate(zip(base['stages'], guarded['stages'])):
                transition = old['status'] + ' -> ' + new['status']
                group['stages'][transition] += 1
                group['stage_opportunities'] += 1
                if old['status'] != new['status']:
                    changed.append(dict(task_id=tid, kind=task['kind'], mode=mode, stage_index=si + 1,
                                        canonical_stage_id=old.get('canonical_stage_id'),
                                        before=old['status'], after=new['status'],
                                        original_query_range_seconds=old.get('query_range_seconds'),
                                        selected_whole_references=base['selected_whole_references'],
                                        features=new['guard_changed_features']))
                for f, nf in zip(old['features'], new['features']):
                    group['features'][f['status'] + ' -> ' + nf['status']] += 1
                    group['feature_opportunities'] += 1
            # Compact reference pointers preserve all immutable large source data.
            mode_outputs[mode] = dict(selected_whole_references=guarded['selected_whole_references'],
                                     stages=guarded['stages'], guard_pair_sources={rid: pair_pointers[rid] for rid in available},
                                     guard_audit={k: v for k, v in ga.items() if k != 'pairs'})
        output = dict(task_id=tid, query=task['query'], kind=task['kind'],
                      original_case=dict(path=case_pins[tid]['path'], sha256=case_pins[tid]['sha256']),
                      modified_input=c['modified_input'], modes=mode_outputs,
                      original_values_preserved=True, no_live_attachment=True,
                      educational_correctness_ground_truth=None)
        ptr = save(U / 'guard_cases' / (tid + '.json'), output)
        case_index.append(dict(task_id=tid, kind=task['kind'], **ptr))
        print(json.dumps(dict(completed=len(case_index), cases=96, new_arm_paths=len(cache.records), cached_requests=sum(x['cache_hit'] for x in cache.requests))), flush=True)
    assert len(case_index) == 96 and len(pair_index) == 864
    assert all(arrays_hash(refs[k]) == v for k, v in ref_signatures.items())
    assert all(G.sha(p) == h for p, h in PINS.items())
    summary = {}
    for (kind, mode), g in tally.items():
        summary.setdefault(kind, {})[mode] = {k: dict(v) if isinstance(v, Counter) else v for k, v in g.items()}
    remaining_missing_similar = {}
    for mode in ['all9', 'subset3']:
        g = summary['main_R_hand_missing'][mode]['stages']
        remaining_missing_similar[mode] = g.get(G.SIMILAR + ' -> ' + G.SIMILAR, 0)
    no_new_difference = all(not tr.endswith(' -> DIFFERENCE_CANDIDATE') or tr.startswith('DIFFERENCE_CANDIDATE ->')
                            for g in summary.values() for m in g.values() for tr in m['stages'])
    report = dict(status='COMPLETE_FIXED_DEVELOPMENT_GUARD_DIAGNOSTIC',
        contract_sha256=G.sha(U / 'GUARD_FROZEN.json'), execution_sha256=G.sha(U / 'GUARD_EXECUTION.json'),
        code_sha256=G.sha(U / 'evidence_guard.py'), runner_sha256=G.sha(__file__),
        cases=96, modes=2, original_pair_paths=864, original_pair_path_reruns=0,
        new_unique_arm_paths=len(cache.records), arm_path_requests=len(cache.requests),
        arm_path_cache_hits=sum(x['cache_hit'] for x in cache.requests),
        aggregate_stage_opportunities=sum(g['stage_opportunities'] for g in tally.values()),
        aggregate_feature_opportunities=sum(g['feature_opportunities'] for g in tally.values()),
        new_observer_calls=0, new_training=0, video_reads=0, all_source_pins_unchanged=True,
        input_arrays_unchanged=True, reference_arrays_unchanged=True, original_reference_selection_unchanged=True,
        original_metrics_and_paths_unchanged=True, summary=summary, changed_stage_records=changed,
        acceptance=dict(no_missing_control_similar_remaining=all(v == 0 for v in remaining_missing_similar.values()),
                        remaining_missing_similar=remaining_missing_similar,
                        no_new_difference_candidate=no_new_difference,
                        normal_additional_abstention_requires_review=True),
        adopted=False, attached_to_live=False,
        interpretation='Fixed development reuse: conservative observability safeguard, not education accuracy or true stage matching. All normal additional abstentions are enumerated.',
        case_index=case_index, pair_index=pair_index,
        arm_path_index=[dict(key=k, source=v['saved']) for k, v in cache.records.items()],
        arm_path_request_ledger=cache.requests, seconds=time.perf_counter() - start)
    save(U / 'GUARD_REPORT.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'cases', 'new_unique_arm_paths', 'aggregate_stage_opportunities', 'acceptance', 'seconds']}))


if __name__ == '__main__':
    run()
