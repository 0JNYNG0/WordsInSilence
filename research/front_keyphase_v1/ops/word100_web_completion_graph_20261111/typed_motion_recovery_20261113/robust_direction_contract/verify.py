"""Finite saved-matrix and formula checks. No models or held-out sources."""
from pathlib import Path
import json
import hashlib
import numpy as np
from trend import summarize

U = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dest = U / 'VERIFICATION.json'
    assert not dest.exists()
    result = json.loads((U / 'RESULTS.json').read_text('utf-8'))
    checks = 0

    def check(condition):
        nonlocal checks
        assert bool(condition)
        checks += 1

    for p, digest in {**result['pins'], **result['source_pins']}.items():
        check(sha(p) == digest)
    check(sha(result['matrix_path']) == result['matrix_sha256'])
    rows = result['rows']
    check(len(rows) == 1871 and all(r['person'] in range(1, 8) and r['split'] == 'train' for r in rows))
    with np.load(result['matrix_path'], allow_pickle=False) as z:
        matrix = {k: z[k].copy() for k in z.files}
    values = matrix['values']
    eligible = matrix['eligible']
    check(np.array_equal(values, [r['values'] for r in rows]))
    check(np.array_equal(eligible, [r['eligible'] for r in rows]))
    # Independent scalar implementation of the definition on every base row.
    manual = []
    for v in values:
        slopes = [(v[j]-v[i])*8/(j-i) for i in range(9) for j in range(i+1, 9)]
        ordered = sorted(slopes)
        manual.append((ordered[17]+ordered[18])/2)
    check(np.allclose(manual, [r['stats']['trend'] for r in rows], atol=1e-12))
    check(np.allclose(values[:, -1]-values[:, 0], [r['stats']['net'] for r in rows], atol=1e-12))
    for slope in (-2., 0., 2.):
        toy = 3.+np.linspace(0., 1., 9)*slope
        check(abs(float(summarize(toy)['trend'])-slope) < 1e-12)
        check(abs(float(summarize(toy[::-1])['trend'])+slope) < 1e-12)
    for j, (pos, multiplier) in enumerate(matrix['noise_controls']):
        perturbed = values.copy()
        perturbed[:, int(pos)] += multiplier * float(matrix['fit_scale'])
        stats = summarize(perturbed)
        for name in ('net', 'trend', 'positive_path', 'negative_path', 'range', 'adjacent_sign_switches'):
            check(np.array_equal(stats[name], matrix['noise_'+name][:, j]))
        check(np.allclose(summarize(perturbed[:, ::-1])['trend'], -stats['trend'], atol=1e-12))
    for j, (start, length) in enumerate(matrix['block_controls']):
        start, length = int(start), int(length)
        perturbed = values.copy()
        perturbed[:, start:start+length] = values[:, start:start+length][:, ::-1]
        stats = summarize(perturbed)
        for name in ('net', 'trend', 'positive_path', 'negative_path', 'range', 'adjacent_sign_switches'):
            check(np.array_equal(stats[name], matrix['block_'+name][:, j]))
        check(np.array_equal(perturbed[:, :start], values[:, :start]) and np.array_equal(perturbed[:, start+length:], values[:, start+length:]))
        check(np.allclose(summarize(perturbed[:, ::-1])['trend'], -stats['trend'], atol=1e-12))
    primary = np.asarray([r['cohort'] == 'primary' and r['target'] != 1 for r in rows]) & eligible
    gold = np.asarray([0 if r['direction_gold'] is None else r['direction_gold']-1 for r in rows])
    check(int(primary.sum()) == 452)
    actual_net = int((matrix['noise_net'][primary]*gold[primary, None] > 0).sum())
    actual_trend = int((matrix['noise_trend'][primary]*gold[primary, None] > 0).sum())
    saved_noise = [r for r in result['noise_summary'] if r['cohort'] == 'primary_direction']
    check(actual_net == sum(r['net_label_sign_compatible'] for r in saved_noise))
    check(actual_trend == sum(r['trend_label_sign_compatible'] for r in saved_noise))
    check(sum(r['eligible'] for r in saved_noise) == int(primary.sum())*54 == 24408)
    block_total = 0
    trend_retained = 0
    net_retained = 0
    for j, (start, length) in enumerate(matrix['block_controls']):
        start, length = int(start), int(length)
        forward = (np.diff(values[:, start:start+length], axis=1)*gold[:, None] > 0).all(1) & primary
        block_total += int(forward.sum())
        trend_retained += int((forward & (matrix['block_trend'][:, j]*gold > 0)).sum())
        net_retained += int((forward & (matrix['block_net'][:, j]*gold > 0)).sum())
    saved_block = [r for r in result['block_summary'] if r['cohort'] == 'primary_direction']
    check(block_total == sum(r['strictly_forward_block_reversed'] for r in saved_block))
    check(trend_retained == sum(r['reversal_visible_but_trend_keeps_reference'] for r in saved_block))
    check(net_retained == sum(r['reversal_visible_but_net_keeps_reference'] for r in saved_block))
    check(int(eligible.sum())*54 == result['total_noise_controls'] == 101034)
    check(int(eligible.sum())*21 == result['total_block_controls'] == 39291)
    check(all(r['direction_gold'] is None and r['target'] is None for r in rows if r['cohort'] == 'partial'))
    out = dict(status='PASS', checks=checks, mathematical_controls_not_semantic_truth=True,
        input_rows=1871, eligible=int(eligible.sum()), unavailable=int((~eligible).sum()),
        primary_direction_noise=dict(total=24408, net_sign_preserved=actual_net, trend_sign_preserved=actual_trend),
        local_reversal=dict(cases_with_explicit_opposing_steps=block_total, trend_keeps_global_sign=trend_retained, net_keeps_global_sign=net_retained),
        no_models=True, no_training=True, no_DEV=True, no_threshold=True,
        review='Implementation self-check of formula and saved arrays; root separately reviews conclusions. Not independent human truth.',
        pins={str(p): sha(p) for p in [Path(__file__), U / 'RESULTS.json', U / 'trend.py', U / 'PLAN_FROZEN.json']})
    with dest.open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
