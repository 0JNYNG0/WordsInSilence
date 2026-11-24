"""Independent stored-record arithmetic only; no project imports or model calls."""
from pathlib import Path
import ast
import hashlib
import json
import math
import statistics

U = Path(__file__).resolve().parent.parent
read = lambda name: json.loads((U / name).read_text(encoding='utf-8-sig'))
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
checks = []


def check(name, ok, detail=None):
    checks.append(dict(check=name, passed=bool(ok), detail=detail))


def close(actual, expected):
    return isinstance(actual, (float, int)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11)


def main():
    source_names = ['CPU_RUNTIME_RESULT.json', 'CPU_TIMING_FREEZE.json', 'cpu_benchmark.py', 'LEARNING_INPUTS.json']
    pins = {name: digest(U / name) for name in source_names}
    result, frozen, info = (read(name) for name in source_names if name.endswith('.json'))
    check('frozen_aliases', frozen['aliases'] == ['P005', 'P043', 'P079'])
    check('frozen_repeats', frozen['repeats'] == 2)
    check('frozen_seed', frozen['seed'] == 42)
    check('frozen_no_gpu_inference', frozen['no_gpu_inference'] is True)
    check('frozen_no_accuracy_rescore', frozen['accuracy_not_rescored'] is True)
    check('result_no_gpu_inference', result['no_gpu_inference'] is True)
    conditions = ['legacy_dtw', 'pose_capacity_matched']
    expected_keys = {(alias, rep, cond) for alias in frozen['aliases'] for rep in [1, 2] for cond in conditions}
    keys = [(r['alias'], r['repeat'], r['condition']) for r in result['rows']]
    check('exact_manifest_6_captures_12_route_rows', len(keys) == len(set(keys)) == 12 and set(keys) == expected_keys)
    original = {r['alias']: r for r in info['rows']}
    capture_components = {}
    response_hashes = {}
    for row in result['rows']:
        name = '/'.join(map(str, [row['alias'], row['repeat'], row['condition']]))
        values = [row[f] for f in ['detector_seconds', 'features_seconds', 'head_dtw_serialize_seconds']]
        check(name + '/nonnegative_finite_components', all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values))
        check(name + '/component_sum', close(row['total_seconds'], math.fsum(values)), dict(actual=row['total_seconds'], independent=math.fsum(values)))
        check(name + '/source_identity', row['source_sha256'] == original[row['alias']]['source_sha256'])
        check(name + '/held_person', original[row['alias']]['role'] == 'held_person')
        capture_key = (row['alias'], row['repeat'])
        components = (row['detector_seconds'], row['features_seconds'])
        if capture_key in capture_components:
            check(name + '/shared_capture_feature_components', components == capture_components[capture_key])
        capture_components[capture_key] = components
        response_path = U / 'cpu_runtime' / f"{row['alias']}_{row['repeat']}" / f"{row['condition']}.json"
        response = json.loads(response_path.read_text(encoding='utf8'))
        response_hashes[str(response_path)] = digest(response_path)
        check(name + '/saved_response_identity', response['query'] == row['alias'] and response['condition'] == row['condition'])
        expected_refs = {r['alias'] for r in info['rows'] if r['role'] == 'fit' and r['word_id'] == original[row['alias']]['word_id']}
        check(name + '/three_fit_references', len(response['pairs']) == 3 and {p['reference'] for p in response['pairs']} == expected_refs)
        check(name + '/not_educational_verdict', response['comparison_is_not_correctness'] is True and all(p['educational_verdict'] is None for p in response['pairs']))
    independent_summaries = []
    for condition in conditions:
        samples = sorted(r['total_seconds'] for r in result['rows'] if r['condition'] == condition)
        position = .95 * (len(samples) - 1)
        low, high = math.floor(position), math.ceil(position)
        p95 = samples[low] + (samples[high] - samples[low]) * (position - low)
        expected = dict(condition=condition, n=len(samples), median=statistics.median(samples), p95=p95, range=[samples[0], samples[-1]])
        saved = next(s for s in result['summary'] if s['condition'] == condition)
        check(condition + '/n6', saved['n'] == expected['n'] == 6)
        for field in ['median', 'p95']:
            check(condition + '/' + field, close(saved[field], expected[field]), dict(actual=saved[field], independent=expected[field]))
        check(condition + '/range', all(close(a, e) for a, e in zip(saved['range'], expected['range'])))
        independent_summaries.append(expected)
    check('separate_setup_seconds', close(result['setup_seconds'], 5.098945899982937))
    # Static call-path inspection only; importing/executing the benchmark is prohibited here.
    tree = ast.parse((U / 'cpu_benchmark.py').read_text(encoding='utf-8-sig'))
    call_names = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    check('no_signrep_compile_or_encode_call_in_cpu_script', not ({'compile_gpu', 'encode'} & call_names))
    check('actual_capture_call_present', any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'capture' for node in ast.walk(tree)))
    check('capture_full_frame_box_none', any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'capture' and any(kw.arg == 'box' and isinstance(kw.value, ast.Constant) and kw.value.value is None for kw in node.keywords) for node in ast.walk(tree)))
    check('same_pose_head_checkpoint_named', 'runs/pose_capacity_matched_42/head.pt' in (U / 'cpu_benchmark.py').read_text(encoding='utf-8-sig'))
    report_source = (U / 'build_report.py').read_text(encoding='utf-8-sig')
    report_revisions = {
        'final_verification_link': 'review/RESULTS_VERIFICATION_FINAL.json' in report_source,
        'physical_sample_endpoint_wording': '같은 물리 상태로 관측한 인접144쌍에서 두 시점의 예측 상태가 달랐던 수' in report_source,
        'crossing_claim_limited': '모든 정상 교차의 의미 보존이나 교육적 허용성을 별도 정답으로 검증한 것은 아닙니다' in report_source,
        'alignment_rounding': '3.34~3.35ms' in report_source,
    }
    for name, ok in report_revisions.items():
        check('report_revision/' + name, ok)
    for name, expected in pins.items():
        check('file_unchanged_during_audit/' + name, digest(U / name) == expected)
    failures = [c for c in checks if not c['passed']]
    output = dict(status='PASS' if not failures else 'FAIL', checks=len(checks), failures=failures, source_sha256=pins,
                  verifier_sha256=digest(__file__), independent_summary=independent_summaries,
                  setup_seconds=result['setup_seconds'], setup_included_in_route_totals=False,
                  capture_count=6, route_rows=12, samples_per_condition=6,
                  model_calls_during_verification=0, benchmark_imported_or_executed=False,
                  cpu_route_has_signrep_inference=False, saved_response_sha256=response_hashes,
                  report_revisions_confirmed=report_revisions,
                  scope='Full original-public-video MediaPipe capture plus fixed pose head and 3-reference DTW; isolated CPU trial. Detector/features are shared between the two reported route component sums. This is not CPU-only SignRep or live-browser end-to-end timing.',
                  limits='The 95th percentile is linearly interpolated from six recorded samples, not a production tail guarantee. Setup is reported separately. This audit checks stored arithmetic and code path, not machine-wide GPU process telemetry.')
    path = U / 'review' / 'CPU_RUNTIME_VERIFICATION.json'
    with path.open('x', encoding='utf8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status=output['status'], checks=len(checks), failures=len(failures), independent_summary=independent_summaries, setup_seconds=result['setup_seconds'], output=str(path)), ensure_ascii=False))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
