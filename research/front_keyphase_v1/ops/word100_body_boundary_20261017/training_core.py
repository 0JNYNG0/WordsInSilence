"""Frozen comparison arithmetic; no detector, model, or training execution on import."""
from collections import Counter
import hashlib
import math

import numpy as np

SEEDS = (20261013, 20261014, 20261015)
ARMS = ('A', 'B', 'C')
VIEWS = ('masked', 'sensor')
FAILURE_CE = math.log(100)
MODEL = dict(in_channels=280, width=32, dilations=[1, 2, 4, 8],
             kernel_size=3, dropout=0.1, n_classes=100)
TRAIN = dict(batch_size=32, epochs_max=80, early_stopping_patience=12,
             lr=0.001, weight_decay=0.001, gradient_clip_norm=1.0,
             cpu_threads=4, wall_safety_seconds_per_arm=3600,
             strict_ce_improvement=1e-6)


def schedule(cid, epoch, arm):
    assert epoch >= 1 and arm in ARMS
    time_parity = int(hashlib.sha256(cid.encode()).hexdigest(), 16) % 2
    temporal = 'x_context' if (epoch + time_parity) % 2 == 1 else 'x_whole'
    if arm == 'A':
        spatial = 'native'
    elif arm == 'B':
        spatial = 'masked'
    else:
        parity = int(hashlib.sha256(('sensor_view:' + cid).encode()).hexdigest(), 16) % 2
        spatial = 'sensor' if ((epoch - 1) // 2 + parity) % 2 == 1 else 'masked'
    return spatial, temporal


def validate_channels(x):
    assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == 280
    assert len(x) >= 1 and np.isfinite(x).all()
    mask = x[:, 186:279]
    assert np.logical_or(mask == 0, mask == 1).all()
    assert np.all(x[:, :102].reshape(-1, 51, 2)[mask[:, :51] == 0] == 0)
    assert np.all(x[:, 102:186].reshape(-1, 42, 2)[mask[:, 51:] == 0] == 0)
    assert x[0, 279] == 0 and np.all(x[1:, 279] > 0)


def padded_arrays(xs):
    assert xs and all(len(x) > 0 for x in xs)
    n = max(map(len, xs))
    data = np.zeros((len(xs), n, 280), dtype=np.float32)
    pad = np.zeros((len(xs), n), dtype=np.float32)
    for i, x in enumerate(xs):
        data[i, :len(x)] = x
        # Even an all-missing real frame is a real time step.
        pad[i, :len(x)] = 1
    return data, pad


def prediction(row, logits=None, reason=None):
    out = {k: row[k] for k in ('clip_id', 'label_index', 'signer_id')}
    if logits is None:
        out.update(status='INPUT_UNAVAILABLE', reason=reason, logits=None,
                   prediction=None, top5=[], correct=False, top5_correct=False,
                   cross_entropy=FAILURE_CE)
        return out
    a = np.asarray(logits, dtype=np.float64)
    assert a.shape == (100,) and np.isfinite(a).all()
    # Deterministic lower-class-index tie handling for all saved-score arithmetic.
    order = np.argsort(-a, kind='stable')
    maximum = float(a.max())
    ce = maximum + math.log(float(np.exp(a - maximum).sum())) - float(a[row['label_index']])
    top5 = order[:5].tolist()
    out.update(status='OK', logits=a.tolist(), prediction=int(order[0]), top5=top5,
               correct=int(order[0]) == row['label_index'],
               top5_correct=row['label_index'] in top5, cross_entropy=ce)
    return out


def metrics(rows, require300=True):
    assert len({r['clip_id'] for r in rows}) == len(rows)
    if require300:
        assert len(rows) == 300
        assert Counter(r['signer_id'] for r in rows) == {11: 100, 12: 100, 13: 100}
        assert Counter(r['label_index'] for r in rows) == {i: 3 for i in range(100)}
    n = len(rows)
    return dict(total=n, top1_correct=sum(r['correct'] for r in rows),
                top5_correct=sum(r['top5_correct'] for r in rows),
                top1_percent=100 * sum(r['correct'] for r in rows) / n if n else None,
                top5_percent=100 * sum(r['top5_correct'] for r in rows) / n if n else None,
                mean_ce=sum(r['cross_entropy'] for r in rows) / n if n else None,
                input_failures=sum(r['status'] != 'OK' for r in rows))


def detailed_metrics(rows, groups=None):
    result = metrics(rows)
    result['by_signer'] = {str(s): metrics([r for r in rows if r['signer_id'] == s], False)
                           for s in (11, 12, 13)}
    result['by_class'] = {str(c): metrics([r for r in rows if r['label_index'] == c], False)
                          for c in range(100)}
    if groups is not None:
        assert set(groups) == {r['clip_id'] for r in rows}
        result['by_observation_pattern'] = {
            g: metrics([r for r in rows if groups[r['clip_id']] == g], False)
            for g in sorted(set(groups.values()))}
    return result


def paired(before, after):
    aa = {r['clip_id']: r for r in before}
    bb = {r['clip_id']: r for r in after}
    assert set(aa) == set(bb)
    for cid in aa:
        assert aa[cid]['label_index'] == bb[cid]['label_index']
        assert aa[cid]['signer_id'] == bb[cid]['signer_id']
    gain = sorted(c for c in aa if not aa[c]['correct'] and bb[c]['correct'])
    loss = sorted(c for c in aa if aa[c]['correct'] and not bb[c]['correct'])
    return dict(improved=gain, regressed=loss, net_correct=len(gain)-len(loss),
                percentage_point_difference=100 * (len(gain)-len(loss)) / len(aa))


def aggregate(run_metrics):
    assert len(run_metrics) == 3
    result = {}
    for k in ('top1_correct', 'top5_correct', 'top1_percent', 'top5_percent', 'input_failures'):
        values = [float(r[k]) for r in run_metrics]
        result[k] = dict(mean=sum(values)/3, minimum=min(values), maximum=max(values), values=values)
    result['by_signer'] = {str(s): {
        'top1_correct_mean': sum(r['by_signer'][str(s)]['top1_correct'] for r in run_metrics)/3,
        'top1_percent_mean': sum(r['by_signer'][str(s)]['top1_percent'] for r in run_metrics)/3
    } for s in (11, 12, 13)}
    return result


def progression_gate(arms, m0):
    """Counts avoid a binary-rounding false failure at exactly1/3percentage points."""
    c = arms['C']['sensor']; b = arms['B']['sensor']; cm = arms['C']['masked']
    cvals = c['top1_correct']['values']; bvals = b['top1_correct']['values']
    checks = dict(
        C_sensor_mean_exceeds_B=sum(cvals) > sum(bvals),
        C_sensor_mean_exceeds_M0=sum(cvals) > 3*m0['sensor']['top1_correct'],
        C_sensor_better_than_B_at_least_two_seeds=sum(x > y for x, y in zip(cvals, bvals)) >= 2,
        C_sensor_failure_no_increase=c['input_failures']['maximum'] <= min(
            m0['sensor']['input_failures'], b['input_failures']['minimum']),
        original_pipeline_mean_regression_at_most_three_cases=sum(cm['top1_correct']['values']) >=
            3*(m0['native']['top1_correct']-3))
    for s in (11, 12, 13):
        for newview, oldview in (('masked', 'native'), ('sensor', 'sensor')):
            checks[f'{newview}_signer{s}_mean_regression_at_most_three_cases'] = (
                arms['C'][newview]['by_signer'][str(s)]['top1_correct_mean'] + 1e-12 >=
                m0[oldview]['by_signer'][str(s)]['top1_correct'] - 3)
    return dict(pass_to_next_review=all(checks.values()), checks=checks,
                automatic_adoption=False, generalization_proven=False,
                note='Development engineering gate; no best-seed choice, no final blind test.')
