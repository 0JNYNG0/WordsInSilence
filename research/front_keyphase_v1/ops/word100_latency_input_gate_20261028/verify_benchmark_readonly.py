"""Independent saved-array audit; no torch, model loading, or model forward."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import numpy as np

U = Path(__file__).resolve().parent
P = U.parent / 'word100_shubert_feasibility_20261028'
checks = []


def check(name, ok, evidence=None):
    checks.append({'check': name, 'pass': bool(ok), 'evidence': evidence})


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Bad comparison array')
    n_a = np.sqrt(np.einsum('ij,ij->i', a, a))
    n_b = np.sqrt(np.einsum('ij,ij->i', b, b))
    good = np.logical_and(n_a > 1e-12, n_b > 1e-12)
    delta = a - b
    cosine = np.einsum('ij,ij->i', a[good], b[good]) / (n_a[good] * n_b[good])
    relative = np.sqrt(np.einsum('ij,ij->i', delta[good], delta[good])) / n_a[good]
    return {
        'rows': a.shape[0], 'zero_norm_rows': int(np.count_nonzero(~good)),
        'max_abs': float(np.absolute(delta).max()),
        'cosine_min': float(np.min(cosine)) if good.any() else None,
        'cosine_p05': float(np.percentile(cosine, 5)) if good.any() else None,
        'cosine_median': float(np.percentile(cosine, 50)) if good.any() else None,
        'relative_l2_median': float(np.percentile(relative, 50)) if good.any() else None,
        'relative_l2_p95': float(np.percentile(relative, 95)) if good.any() else None,
        'relative_l2_max': float(np.max(relative)) if good.any() else None,
    }


def same_metrics(actual, saved):
    return set(actual) == set(saved) and all(
        saved[k] is None if v is None else saved[k] is not None and math.isclose(v, saved[k], rel_tol=1e-10, abs_tol=1e-10)
        for k, v in actual.items()
    )


def main():
    result = json.loads((U / 'BENCH_RESULT.json').read_text(encoding='utf-8'))
    frozen = json.loads((U / 'FROZEN.json').read_text(encoding='utf-8'))
    aliases, variants = ['V01', 'V07', 'V12'], ['fp32', 'int8_linear']
    check('fixed scope', frozen['fixed_aliases'] == aliases and frozen['threads'] == 4 and frozen['batch_size'] == 8 and frozen['repeats'] == 2)
    check('complete status', result['status'] == 'COMPLETE_CPU_SPEED_AND_NUMERICAL_AUDIT_NOT_SIGN_ACCURACY')
    check('frozen hash', sha(U / 'FROZEN.json') == result['frozen_sha256'])
    check('executed script hash', sha(U / 'benchmark_dynamic_int8.py') == result['script_sha256'])
    for path, expected in frozen['source_pins'].items():
        check('frozen source ' + Path(path).name, sha(path) == expected)
    for kind, audit in result['quantization'].items():
        check('actual quantization ' + kind, audit['quantized_linear_count'] == audit['original_linear_count'] > 0 and audit['weight_dtype'] == 'torch.qint8')
        check('quantized module names ' + kind, len(set(audit['module_names'])) == audit['quantized_linear_count'])
    expected_order = [(a, v, r) for r in range(2) for a in (aliases if r == 0 else aliases[::-1]) for v in (variants if r == 0 else variants[::-1])]
    actual_order = [(r['alias'], r['variant'], r['repeat']) for r in result['rows']]
    check('complete AB BA order', actual_order == expected_order)
    arrays, rows = {}, {}
    for row in result['rows']:
        key = row['alias'], row['variant'], row['repeat']
        rows[key] = row
        check('array hash ' + str(key), sha(row['array']) == row['sha256'])
        with np.load(row['array'], allow_pickle=False) as z:
            arrays[key] = {k: z[k].copy() for k in z.files}
        n = row['frames']
        expected_shapes = {'face': (n, 384), 'left_hand': (n, 384), 'right_hand': (n, 384), 'body_posture': (n, 14), 'last_ffn': (n, 768)}
        check('shape finite ' + str(key), set(arrays[key]) == set(expected_shapes) and all(v.shape == expected_shapes[k] and np.isfinite(v).all() for k, v in arrays[key].items()))
        with np.load(P / 'features' / row['alias'] / 'streams.npz', allow_pickle=False) as old:
            check('same body and original T ' + str(key), np.array_equal(arrays[key]['body_posture'], old['body_posture']) and n == len(old['face']))
        t = row['timing']
        components = sum(t[k] for k in ('face_seconds', 'left_hand_seconds', 'right_hand_seconds', 'body_read_seconds', 'shubert_seconds'))
        check('positive timings and total ' + str(key), all(math.isfinite(v) and v >= 0 for v in t.values()) and 0 <= t['cached_rgb_to_features_seconds'] - components < 1)
    for row in result['differences']:
        key = row['alias'], row['repeat']
        a, b = arrays[key[0], 'fp32', key[1]], arrays[key[0], 'int8_linear', key[1]]
        for field, reported in row['fields'].items():
            check('INT8 drift arithmetic ' + str(key) + '/' + field, same_metrics(metrics(a[field], b[field]), reported))
    for row in result['repeat_differences']:
        key = row['alias'], row['variant']
        a, b = arrays[*key, 0], arrays[*key, 1]
        for field, reported in row['fields'].items():
            check('repeat arithmetic ' + str(key) + '/' + field, same_metrics(metrics(a[field], b[field]), reported))
    for row in result['previous_cache_differences']:
        a = arrays[row['alias'], 'fp32', 0]
        with np.load(P / 'features' / row['alias'] / 'streams.npz', allow_pickle=False) as z:
            old = {k: z[k].copy() for k in z.files}
        old['last_ffn'] = np.load(P / 'features' / row['alias'] / 'last_ffn.npy', allow_pickle=False)
        for field, reported in row['fields'].items():
            check('old cache arithmetic ' + row['alias'] + '/' + field, same_metrics(metrics(a[field], old[field]), reported))
    summary = {}
    for variant in variants:
        times = [r['timing']['cached_rgb_to_features_seconds'] for r in result['rows'] if r['variant'] == variant]
        summary[variant] = {'min': min(times), 'median': statistics.median(times), 'max': max(times), 'n': len(times)}
        check('summary arithmetic ' + variant, summary[variant] == result['summary'][variant])
    speedup = summary['fp32']['median'] / summary['int8_linear']['median']
    check('pooled speed ratio', math.isclose(speedup, result['median_speedup'], rel_tol=1e-12))
    per_clip = []
    for alias in aliases:
        elapsed = {v: [rows[alias, v, r]['timing']['cached_rgb_to_features_seconds'] for r in range(2)] for v in variants}
        a, b = statistics.median(elapsed['fp32']), statistics.median(elapsed['int8_linear'])
        per_clip.append({'alias': alias, 'seconds': elapsed, 'fp32_median': a, 'int8_median': b, 'median_cost_reduction_fraction': 1 - b / a, 'paired_speedup_by_repeat': [x / y for x, y in zip(elapsed['fp32'], elapsed['int8_linear'])]})
    for path, expected in result['protected_pins'].items():
        check('protected hash ' + str(Path(path).relative_to(P)), sha(path) == expected)
    check('no product/training promotion', result['training_runs'] == 0 and result['private_access'] is False and result['operating_changes'] == 0 and result['face_removed'] is False and result['fps_reduced'] is False)
    report = {
        'at': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Codex latency_quality_reviewer; independent saved-array arithmetic audit',
        'status': 'PASS' if all(c['pass'] for c in checks) else 'FAIL',
        'checks': len(checks), 'failed': [c for c in checks if not c['pass']],
        'per_clip': per_clip, 'pooled_summary': summary, 'pooled_speedup': speedup,
        'all_details': checks,
        'new_model_forward_calls': 0, 'training_runs': 0, 'private_access': False,
        'meaning': 'Arithmetic, model/input preservation, and bounded execution verified; no semantic accuracy or real user wait validation.',
        'result_sha256': sha(U / 'BENCH_RESULT.json'), 'verifier_sha256': sha(__file__),
    }
    target = U / 'BENCH_INDEPENDENT_VERIFY.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'failed', 'per_clip', 'pooled_summary')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
