"""Read saved 96-condition results; add details and verify total preservation."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from copy import deepcopy
import numpy as np

U = Path(__file__).resolve().parent.parent
OLD = U.parent / 'word100_multipeople_live_feedback_20261101'
spec = importlib.util.spec_from_file_location('detail_enrichment', U / 'detail_enrichment.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
PINS = {}
CHECKS = 0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    p = Path(p).resolve()
    PINS[str(p)] = sha(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))


def load(p):
    p = Path(p).resolve()
    PINS[str(p)] = sha(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def check(v, name):
    global CHECKS
    CHECKS += 1
    if not v:
        raise AssertionError(name)


def preserve(a, b):
    if isinstance(a, dict):
        for k, v in a.items():
            check(k in b, 'original key retained')
            preserve(v, b[k])
    elif isinstance(a, list):
        check(len(a) == len(b), 'original list denominator')
        for v, w in zip(a, b):
            preserve(v, w)
    else:
        check(a == b, 'exact original value retained')


def signature(a):
    h = hashlib.sha256()
    for k in sorted(a):
        if isinstance(a[k], np.ndarray):
            arr = np.ascontiguousarray(a[k])
            h.update(k.encode() + str(arr.shape).encode() + arr.dtype.str.encode() + arr.tobytes())
    return h.hexdigest()


def write(p, data):
    if p.exists():
        raise FileExistsError(p)
    content = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    if p.suffix == '.gz':
        p.write_bytes(gzip.compress(content, mtime=0))
    else:
        p.write_bytes(content)
    return dict(path=str(p), sha256=sha(p))


def main():
    for name in ['DETAIL_EXECUTION.json', 'DETAIL_REPORT.json', 'detail_cases']:
        if (U / 'warning_audit' / name).exists():
            raise FileExistsError('Preserve prior run: ' + name)
    frozen = read(U / 'FROZEN.json')
    for p, h in frozen['source_pins'].items():
        check(sha(p) == h, 'frozen source pin')
    for p in [U / 'detail_enrichment.py', Path(__file__), U / 'warning_audit/detail_selftest.py']:
        PINS[str(p.resolve())] = sha(p)
    st = read(U / 'warning_audit/DETAIL_SELFTEST.json')
    check(st['status'] == 'PASS' and st['code_sha256'] == sha(U / 'detail_enrichment.py'), 'final synthetic code pin')
    bank = read(OLD / 'REFERENCE_BANK.json')
    rows = {r['clip_id']: r for r in bank['rows']}
    normal = read(OLD / 'NORMAL_RANGES.json')['ranges']
    subset = read(OLD / 'THREE_REFERENCE_RANGES.json')['ranges']
    tasks = read(OLD / 'EVALUATION_TASKS.json')['tasks']
    guard = read(OLD / 'GUARD_REPORT.json')
    original = read(OLD / 'FEEDBACK_EVALUATION.json')
    names = bank['feature_names']
    refs = {}
    for rid, row in rows.items():
        a = load(row['input'])
        a['probabilities'] = load(row['probability_file'])['probabilities']
        a['feature_names'] = names
        refs[rid] = a
    queries, cases, guarded, pairs = {}, {}, {}, {}
    gp = {v['task_id']: v for v in guard['case_index']}
    cp = {v['id']: v for v in original['case_index']}
    for task in tasks:
        tid = task['id']
        check(sha(cp[tid]['path']) == cp[tid]['sha256'], 'baseline case source pin')
        check(sha(gp[tid]['path']) == gp[tid]['sha256'], 'guard case source pin')
        c = read(cp[tid]['path'])
        cases[tid] = c
        guarded[tid] = read(gp[tid]['path'])
        if c['modified_input']:
            check(sha(c['modified_input']['path']) == c['modified_input']['sha256'], 'condition input pin')
            a = load(c['modified_input']['path'])
        else:
            a = load(task['source_input'])
            a['probabilities'] = load(task['source_probabilities'])['probabilities']
        a['feature_names'] = names
        if task['kind'] not in ['same_expression', 'different_target_expression']:
            edit = c['edit_ledger']
            a['original_source_index'] = np.asarray(edit['source_indices'])
            a['original_source_frame_id'] = np.asarray(edit['source_frame_ids'])
            a['original_source_timestamp_us'] = np.asarray(edit['source_timestamp_us'])
        queries[tid] = a
        pairs[tid] = []
        for ptr in c['comparisons']:
            check(sha(ptr['path']) == ptr['sha256'], 'saved pair pin')
            pairs[tid].append(read(ptr['path']))
    write(U / 'warning_audit/DETAIL_EXECUTION.json', dict(status='PINNED_BEFORE_SAVED_DATA_ENRICHMENT',
        source_pins=PINS, contract='FROZEN warning copy-only clause', source_cases=96,
        source_pairs=864, modes=['all9', 'subset3'], new_alignment=0, new_inference=0,
        fixed_numeric_cutoffs='Existing per-mode FIT p90 and peak numbers, instantaneous residual annotation only.'))
    target = U / 'warning_audit/detail_cases'
    target.mkdir()
    refsignatures = {k: signature(v) for k, v in refs.items()}
    case_index = []
    warnings = []
    totals = dict(aggregate_stages=0, aggregate_features=0, reference_stage_families=0,
                  trace_query_samples=0, trace_query_dimension_opportunities=0,
                  fully_unshared_query_samples=0, p90_episodes=0, peak_episodes=0,
                  single_sample_peak_episodes=0, original_peak_preserved=0)
    start = time.perf_counter()
    for task in tasks:
        tid = task['id']
        a = queries[tid]
        before = signature(a)
        outputs = {}
        for mode, ranges in [('all9', normal), ('subset3', subset)]:
            gm = guarded[tid]['modes'][mode]
            available = task['references'] if mode == 'all9' else task['subset_references']
            base = dict(word=task['target_word'], stages=gm['stages'], selected_whole_references=gm['selected_whole_references'],
                        references=[p for p in pairs[tid] if p['id'] in available],
                        guard_audit=gm['guard_audit'])
            raw = deepcopy(base)
            out = D.enrich(a, rows, refs, base, ranges)
            preserve(raw, out)
            check(raw == base, 'caller result immutable')
            check(signature(a) == before, 'input arrays immutable')
            totals['aggregate_stages'] += len(out['stages'])
            totals['aggregate_features'] += sum(len(s['features']) for s in out['stages'])
            for comp in out['references']:
                for si, stage in enumerate(comp['stages']):
                    for family, d in stage['detail_enrichment']['families'].items():
                        totals['reference_stage_families'] += 1
                        totals['trace_query_samples'] += d['query_frame_opportunities']
                        totals['trace_query_dimension_opportunities'] += d['query_dimension_opportunities']
                        totals['fully_unshared_query_samples'] += d['fully_unshared_query_samples']
                        totals['original_peak_preserved'] += int(d['original_peak_preserved_in_trace'])
                        for name in ['p90', 'peak']:
                            crit = d['exceedances'][name + '_number_exceedance']
                            episodes = crit['episodes']
                            totals[name + '_episodes'] += len(episodes)
                            flattened = [i for ep in episodes for i in ep['query_indices']]
                            expected = [v['query']['index'] for v in d['trace'] if crit['threshold'] is not None and
                                        v['max_abs_residual'] is not None and v['max_abs_residual'] > crit['threshold']]
                            check(flattened == expected, 'all instantaneous exceedance samples retained once')
                            for ep in episodes:
                                check(ep['sample_count'] == len(ep['query_indices']), 'episode sample count')
                                check(ep['source_pts_span_us'] == ep['end']['timestamp_us'] - ep['start']['timestamp_us'], 'episode source time span')
                                if name == 'peak' and ep['singleton']:
                                    totals['single_sample_peak_episodes'] += 1
                        if task['kind'] == 'same_expression' and mode == 'all9' and comp['id'] in out['selected_whole_references']:
                            f = next(f for f in out['stages'][si]['features'] if f['name'] == D._engine().LABELS[family])
                            if f['status'] == 'DIFFERENCE_CANDIDATE':
                                warnings.append(dict(task_id=tid, query=task['query'], stage_index=si + 1,
                                    canonical_stage_id=stage['canonical_stage_id'], family=family, reference_id=comp['id'],
                                    original_status=out['stages'][si]['status'], original_peak=stage['families'][family]['peak_distance'],
                                    original_peak_at=stage['families'][family]['peak_at'],
                                    exceedances=d['exceedances'], interpretation=d['interpretation']))
            outputs[mode] = out
        record = dict(task_id=tid, query=task['query'], kind=task['kind'],
                      source_case=cp[tid], source_guard_case=gp[tid],
                      source_pairs=cases[tid]['comparisons'], modes=outputs,
                      source_axis='condition-native plus original_source fields for stored technical transformations',
                      original_information_preserved=True)
        ptr = write(target / (tid + '.json.gz'), record)
        case_index.append(dict(task_id=tid, kind=task['kind'], **ptr))
        print(json.dumps(dict(completed=len(case_index), total=96, copy_checks=CHECKS)), flush=True)
    check(len(case_index) == 96 and len(warnings) == 14, 'full96 and7warning2reference denominator')
    check(totals['aggregate_stages'] == 624 and totals['aggregate_features'] == 4368, 'both modes complete')
    for rid, before in refsignatures.items():
        check(signature(refs[rid]) == before, 'reference arrays unchanged')
    for p, h in PINS.items():
        check(sha(p) == h, 'source and code pin unchanged')
    report = dict(status='PASS_COPY_ONLY_DETAIL_ENRICHMENT', checks=CHECKS, cases=96,
        source_pairs=864, mode_pair_views=1152, totals=totals, case_index=case_index,
        original_seven_warning_details=warnings, original_warning_stages=7,
        status_changes=0, original_numeric_value_changes=0, new_DTW=0, observer_calls=0,
        new_training=0, new_RGB=0, user_inputs=0, live_attachment=False,
        all_source_pins_unchanged=True, all_input_arrays_unchanged=True,
        educational_accuracy_gain_claim=False, false_warning_removed_claim=False,
        limitation='Instantaneous comparison to stored p90 cutoff is an explanation marker, not reclassification; any-coordinate runs may contain changes in which coordinate exceeds, retained in the full per-dimension trace. No explicit crossing ground truth in original96.',
        code_sha256=sha(U / 'detail_enrichment.py'), execution_sha256=sha(U / 'warning_audit/DETAIL_EXECUTION.json'),
        runner_sha256=sha(__file__), seconds=time.perf_counter() - start)
    write(U / 'warning_audit/DETAIL_REPORT.json', report)
    print(json.dumps(dict(status=report['status'], checks=CHECKS, totals=totals, seconds=report['seconds'])))


if __name__ == '__main__':
    main()
