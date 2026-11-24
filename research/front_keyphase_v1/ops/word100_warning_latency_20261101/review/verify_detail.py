"""Independent saved detail traces/episodes: no producer, model, DP or image reads."""
from pathlib import Path
from collections import defaultdict, Counter
import gzip
import hashlib
import importlib.util
import json
import traceback
import numpy as np

U = Path(__file__).resolve().parents[1]
O = U.parent / 'word100_multipeople_live_feedback_20261101'
spec = importlib.util.spec_from_file_location('_independent_detail_masks', O / 'review/verify_guard_saved.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
F = A.FAMILIES
checks = 0
failures = []
pins = {}


def check(ok, name):
    global checks
    checks += 1
    if not bool(ok):
        failures.append(name)
        if len(failures) >= 20:
            raise AssertionError(repr(failures))


def pin(p, expected=None):
    p = str(Path(p).resolve())
    h = A.sha(p)
    check(expected is None or h == expected, 'pin:' + p)
    check(p not in pins or pins[p] == h, 'stable:' + p)
    pins[p] = h


def read(p, expected=None):
    pin(p, expected)
    return A.read(p)


def load(p):
    pin(p)
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in ['x', 'mask', 'frame_id', 'timestamp_us']}


def channel_values(a, names):
    x = np.asarray(a['x'], np.float64)
    p = np.asarray(a['probabilities'], np.float64)
    v = {'R_shape': np.nan_to_num(p[:, [0, 1, 3]]), 'L_shape': np.nan_to_num(p[:, [2, 4]])}
    for s in ['L', 'R']:
        v[s + '_angles'] = x[:, [i for i, n in enumerate(names) if n.startswith(s + '_') and '_angle' in n and 'elbow' not in n]]
    v['hand_relation'] = x[:, [names.index(n) for n in ['hands_wrist_delta_x', 'hands_wrist_delta_y', 'hands_wrist_distance']]]
    v['arm_path'] = x[:, [names.index(s + '_' + site + '_' + d) for s in ['L', 'R'] for site in ['pose_wrist', 'elbow'] for d in ['x', 'y']]]
    v['hand_axis'] = x[:, [names.index(s + '_palm_axis_' + d) for s in ['L', 'R'] for d in ['x', 'y']]]
    return v


def axis(a, i):
    out = dict(index=int(i), frame_id=int(a['frame_id'][i]), timestamp_us=int(a['timestamp_us'][i]))
    for key in ['original_source_index', 'original_source_frame_id', 'original_source_timestamp_us']:
        if key in a:
            out[key] = int(a[key][i])
    return out


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k != 'detail_enrichment'}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def raw_trace(a, r, qv, rv, qm, rm, links):
    groups = defaultdict(set)
    for qi, ri in links:
        groups[int(qi)].add(int(ri))
    trace = []
    for qi, ris in sorted(groups.items()):
        ris = sorted(ris)
        cells = [(ri, d, abs(float(qv[qi, d]) - float(rv[ri, d]))) for ri in ris
                 for d in range(qm.shape[1]) if qm[qi, d] and rm[ri, d]]
        per = [max([v for _, dd, v in cells if dd == d], default=None) for d in range(qm.shape[1])]
        peak = max([v for v in per if v is not None], default=None)
        trace.append(dict(query=axis(a, qi), reference_indices=ris, own_observed=qm[qi].tolist(),
            expected_from_reference=[bool(any(rm[ri, d] for ri in ris)) for d in range(qm.shape[1])],
            shared_observed=[v is not None for v in per], per_dimension_max_abs_residual=per,
            max_abs_residual=peak, max_witnesses=[dict(query=axis(a, qi), reference=axis(r, ri),
                coordinate=d, abs_residual=v) for ri, d, v in cells if v == peak]))
    return trace


def episodes(a, qv, qm, trace, limit):
    data = {v['query']['index']: v for v in trace}
    selected = [i for i, v in data.items() if limit is not None and v['max_abs_residual'] is not None and v['max_abs_residual'] > limit]
    runs = []
    for i in selected:
        if runs and i == runs[-1][-1] + 1 and a['frame_id'][i] == a['frame_id'][runs[-1][-1]] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    def context(i):
        if i < 0 or i >= len(qv):
            return None
        return dict(query=axis(a, i), belongs_to_this_stage_correspondence=i in data,
            own_values=[float(qv[i, d]) if qm[i, d] else None for d in range(qm.shape[1])], own_observed=qm[i].tolist(),
            paired_max_abs_residual=data[i]['max_abs_residual'] if i in data else None)
    result = []
    for run in runs:
        peak = max(data[i]['max_abs_residual'] for i in run)
        result.append(dict(start=axis(a, run[0]), end=axis(a, run[-1]), query_indices=run,
            sample_count=len(run), source_pts_span_us=int(a['timestamp_us'][run[-1]] - a['timestamp_us'][run[0]]),
            singleton=len(run) == 1, singleton_meaning='One observed sample, not absence or zero-duration action' if len(run) == 1 else None,
            dimensions=sorted({d for i in run for d, v in enumerate(data[i]['per_dimension_max_abs_residual']) if v is not None and v > limit}),
            max_abs_residual=peak, max_witnesses=[w for i in run if data[i]['max_abs_residual'] == peak for w in data[i]['max_witnesses']],
            context_before=context(run[0]-1), context_after=context(run[-1]+1)))
    return selected, result


def run():
    report = read(U / 'warning_audit/DETAIL_REPORT.json')
    execution = read(U / 'warning_audit/DETAIL_EXECUTION.json', report['execution_sha256'])
    pin(U / 'detail_enrichment.py', report['code_sha256'])
    pin(U / 'warning_audit/run_detail_enrichment.py', report['runner_sha256'])
    for p, h in execution['source_pins'].items():
        pin(p, h)
    bank = read(O / 'REFERENCE_BANK.json')
    rows = {r['clip_id']: r for r in bank['rows']}
    names = bank['feature_names']
    refs = {}
    for rid, row in rows.items():
        a = load(row['input'])
        pin(row['probability_file'], row['probability_sha256'])
        with np.load(row['probability_file'], allow_pickle=False) as z:
            a['probabilities'] = z['probabilities']
        refs[rid] = a
    rm = {k: A.independent_masks(a, names) for k, a in refs.items()}
    rv = {k: channel_values(a, names) for k, a in refs.items()}
    scope = {k: A.independent_scope(rows[k], rm[k]) for k in rows}
    tasks = read(O / 'EVALUATION_TASKS.json')['tasks']
    base_cases = {r['id']: r for r in read(O / 'FEEDBACK_EVALUATION.json')['case_index']}
    guard_cases = {r['task_id']: r for r in read(O / 'GUARD_REPORT.json')['case_index']}
    detailed = {r['task_id']: r for r in report['case_index']}
    ranges = {'all9': read(O / 'NORMAL_RANGES.json')['ranges'], 'subset3': read(O / 'THREE_REFERENCE_RANGES.json')['ranges']}
    check(len(tasks) == len(detailed) == 96 and set(detailed) == {t['id'] for t in tasks}, 'all96_cases')
    totals = Counter()
    warnings = []
    pair_views = 0
    for task in tasks:
        tid = task['id']
        c = read(base_cases[tid]['path'], base_cases[tid]['sha256'])
        g = read(guard_cases[tid]['path'], guard_cases[tid]['sha256'])
        dp = detailed[tid]
        pin(dp['path'], dp['sha256'])
        d = json.loads(gzip.decompress(Path(dp['path']).read_bytes()))
        check(d['source_case'] == base_cases[tid] and d['source_guard_case'] == guard_cases[tid] and d['source_pairs'] == c['comparisons'], tid + ':provenance')
        if c['modified_input']:
            a = load(c['modified_input']['path'])
            with np.load(c['modified_input']['path'], allow_pickle=False) as z:
                a['probabilities'] = z['probabilities']
            for k, source in [('original_source_index', 'source_indices'), ('original_source_frame_id', 'source_frame_ids'), ('original_source_timestamp_us', 'source_timestamp_us')]:
                a[k] = np.asarray(c['edit_ledger'][source])
        else:
            a = load(task['source_input'])
            pin(task['source_probabilities'])
            with np.load(task['source_probabilities'], allow_pickle=False) as z:
                a['probabilities'] = z['probabilities']
        qm, qv = A.independent_masks(a, names), channel_values(a, names)
        originals = {ptr['reference']: read(ptr['path'], ptr['sha256']) for ptr in c['comparisons']}
        traces = {}
        for mode in ['all9', 'subset3']:
            out = d['modes'][mode]
            available = task['references'] if mode == 'all9' else task['subset_references']
            base = dict(word=task['target_word'], stages=g['modes'][mode]['stages'],
                selected_whole_references=g['modes'][mode]['selected_whole_references'],
                references=[v for k, v in originals.items() if k in available], guard_audit=g['modes'][mode]['guard_audit'])
            check(clean(out) == base, tid + ':' + mode + ':every_original_value_exact')
            totals['aggregate_stages'] += len(out['stages'])
            totals['aggregate_features'] += sum(len(s['features']) for s in out['stages'])
            for ci, pair in enumerate(out['references']):
                rid = pair['id']
                pair_views += 1
                for si, stage in enumerate(pair['stages']):
                    bounds = rows[rid]['stages'][si]['frames']
                    links = [v for v in pair['path'] if bounds[0] <= v[1] <= bounds[1]]
                    for j, family in enumerate(F):
                        detail = stage['detail_enrichment']['families'][family]
                        raw = stage['families'][family]
                        key = (rid, si, family)
                        rmask = rm[rid][family] & scope[rid][family]
                        if key not in traces:
                            traces[key] = raw_trace(a, refs[rid], qv[family], rv[rid][family], qm[family], rmask, links)
                        expected = traces[key]
                        tag = f'{tid}:{mode}:{rid}:{si}:{family}'
                        check(detail['trace'] == expected, tag + ':all_raw_cells_masks_axes_max_witnesses')
                        dims = qm[family].shape[1]
                        owncounts = dict(query_frame_opportunities=len(expected), query_dimension_opportunities=len(expected)*dims,
                            reference_frame_opportunities=bounds[1]-bounds[0]+1,
                            reference_dimension_opportunities=(bounds[1]-bounds[0]+1)*dims,
                            reference_original_expected=int(rm[rid][family][bounds[0]:bounds[1]+1].sum()),
                            reference_scoped_expected=int(rmask[bounds[0]:bounds[1]+1].sum()),
                            shared_query_dimension_cells=sum(sum(v['shared_observed']) for v in expected),
                            fully_unshared_query_samples=sum(v['max_abs_residual'] is None for v in expected))
                        owncounts['reference_scope_excluded'] = owncounts['reference_original_expected']-owncounts['reference_scoped_expected']
                        check(all(detail[k] == v for k, v in owncounts.items()), tag + ':denominators')
                        peak = max([v['max_abs_residual'] for v in expected if v['max_abs_residual'] is not None], default=None)
                        check(peak == raw['peak_distance'] == detail['original_peak_distance'] and
                              detail['original_peak_at'] == raw['peak_at'], tag + ':original_peak_and_short_event_preserved')
                        totals['reference_stage_families'] += 1
                        totals['trace_query_samples'] += len(expected)
                        totals['trace_query_dimension_opportunities'] += len(expected)*dims
                        totals['fully_unshared_query_samples'] += owncounts['fully_unshared_query_samples']
                        totals['original_peak_preserved'] += 1
                        for threshold in ['p90', 'peak']:
                            limit = ranges[mode][family][threshold]
                            selected, episodes_expected = episodes(a, qv[family], qm[family], expected, limit)
                            actual = detail['exceedances'][threshold + '_number_exceedance']
                            check(actual['threshold'] == limit and actual['above_query_samples'] == len(selected) and
                                  actual['episodes'] == episodes_expected and actual['comparison'] == 'strict greater than',
                                  tag + ':' + threshold + ':all_episodes_with_singletons_gaps_context')
                            totals[threshold + '_episodes'] += len(episodes_expected)
                            if threshold == 'peak':
                                totals['single_sample_peak_episodes'] += sum(v['singleton'] for v in episodes_expected)
                        if task['kind'] == 'same_expression' and mode == 'all9' and rid in out['selected_whole_references'] and out['stages'][si]['features'][j]['status'] == 'DIFFERENCE_CANDIDATE':
                            warnings.append(dict(task_id=tid, query=task['query'], stage_index=si+1,
                                canonical_stage_id=stage['canonical_stage_id'], family=family, reference_id=rid,
                                original_status=out['stages'][si]['status'], original_peak=raw['peak_distance'], original_peak_at=raw['peak_at'],
                                exceedances=detail['exceedances'], interpretation=detail['interpretation']))
            for si, stage in enumerate(out['stages']):
                for j, feature in enumerate(stage['features']):
                    wanted = [dict(reference_id=rid, json_pointer=f'/references/{next(i for i,p in enumerate(out["references"]) if p["id"]==rid)}/stages/{si}/detail_enrichment/families/{F[j]}') for rid in out['selected_whole_references']]
                    check(feature['detail_enrichment']['references'] == wanted, f'{tid}:{mode}:{si}:{F[j]}:selected2_detail_pointers')
        if int(tid[1:]) % 24 == 0:
            print(json.dumps(dict(cases=int(tid[1:]), checks=checks, failures=len(failures))), flush=True)
    check(dict(totals) == report['totals'], 'every_aggregate_total')
    check(pair_views == 1152 == report['mode_pair_views'], 'mode_pair_views')
    check(warnings == report['original_seven_warning_details'] and len(warnings) == 14, 'all7_warning2reference_records')
    check(len({(w['task_id'], w['stage_index']) for w in warnings}) == 7, 'original7_unique_warning_stages')
    for p, h in list(pins.items()):
        check(A.sha(p) == h, 'end_pin:' + p)
    return dict(source_pins=pins, producer_report_sha256=A.sha(U/'warning_audit/DETAIL_REPORT.json'),
        source_execution_pins=len(execution['source_pins']), cases=96, mode_pair_views=pair_views,
        independently_recomputed_totals=dict(totals), original_warning_stages=7, warning_reference_records=14,
        original_status_or_numeric_changes=0, threshold_changes=0,
        execution=dict(producer_imports=0, models=0, GPU=0, DP=0, RGB_decode=0, training=0),
        limits=['Copy-only explanation preserves all seven warnings; it does not remove false positives or establish educational correctness.',
            'A duration episode means any shared coordinate exceeds a saved numeric cutoff; changing coordinates within that episode are retained per dimension.',
            'P90-number exceedance is an instantaneous raw comparison, not recomputed stageP90 or a new classifier.',
            'Single-sample spans are0 between first/last timestamp and still contain one observation. No short peak or unknown gap is removed.',
            'For synthetic resampling, timestamp_us/span use the condition-native clock; original_source_timestamp_us is separately retained. Neither is a certified action duration.',
            'Existing96 conditions have no explicit crossing ground truth; this verifies data/left-right/path preservation only.'])


if __name__ == '__main__':
    out = U/'review/DETAIL_VERIFICATION.json'
    if out.exists():
        raise FileExistsError('Preserve first audit')
    report = {}
    try:
        report = run()
    except BaseException:
        p = U/'review/DETAIL_VERIFICATION_FIRST_FAILURE.txt'
        if not p.exists():
            p.write_text(traceback.format_exc(), encoding='utf8')
        failures.append(traceback.format_exc())
    report.update(status='PASS_INDEPENDENT_DETAIL_AUDIT' if not failures else 'FAIL_PRESERVED',
        checks=checks, failure_count=len(failures), failures=failures, verifier_sha256=A.sha(__file__))
    with out.open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: report.get(k) for k in ['status','checks','failure_count','independently_recomputed_totals']}))
    raise SystemExit(0 if not failures else 1)
