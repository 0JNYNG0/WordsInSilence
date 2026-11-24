"""Saved-only palm-head arithmetic and invariant audit; no model construction.

Loading a checkpoint here reads parameters only. No training, feature rebuild,
or model forward is performed. Partial annotations are not direction gold.
"""
from pathlib import Path
import json
import hashlib
import datetime
from collections import Counter
import numpy as np
import torch

U = Path(__file__).resolve().parent
H = U.parent
G = H.parent
checks = 0
failures = []


def rd(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ck(name, condition, detail=None):
    global checks
    checks += 1
    if not bool(condition):
        failures.append(dict(check=name, detail=detail))


def pins(items, label):
    for path, digest in items.items():
        ck(label + ':' + path, sha(path) == digest)


def softmax(x):
    x = np.asarray(x, float)
    exp = np.exp(x - x.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def verdict(prob, desired, eligible=True):
    if not eligible:
        return 'observation_unknown'
    if desired == 1:
        return 'supported' if prob[1] >= .65 and max(prob[0], prob[2]) < .65 else 'uncertain'
    opposite = 2 - desired
    if prob[desired] >= .65 and prob[opposite] < .65 and prob[1] < .65:
        return 'supported'
    if prob[opposite] >= .65 and prob[desired] < .35 and prob[1] < .65:
        return 'opposite_direction'
    if prob[1] >= .65:
        return 'movement_not_established'
    return 'uncertain'


def card_status(card):
    states = [e['state'] for e in card['evidence']]
    if card['span'] is None:
        return 'association_unknown'
    if 'difference' in states or 'opposite_direction' in states:
        return 'difference_evidence'
    if states and all(v == 'supported' for v in states):
        return 'supported_features_only'
    return 'insufficient_evidence'


def key(row):
    return row['seed'], row['clip_id'], row['kind']


def primary_metrics(rows, logits, eligible):
    result = []
    p = softmax(logits)
    chosen = logits.argmax(1)
    for split in ('train', 'probe'):
        for target in (0, 1, 2):
            indices = [i for i, r in enumerate(rows) if r['split'] == split and r['target'] == target]
            valid = [i for i in indices if eligible[i]]
            result.append(dict(split=split, target=target, total=len(indices), eligible=len(valid),
                unavailable=len(indices) - len(valid),
                correct_argmax=int(sum(chosen[i] == target for i in valid)),
                predicted_classes=dict(Counter(str(int(chosen[i])) for i in valid)),
                existing_verdict_states=dict(Counter(verdict(p[i], target) for i in valid))))
    return result


def partial_metrics(rows, g, logits, eligible):
    chosen = logits.argmax(1)
    p = softmax(logits)
    valid = np.flatnonzero(eligible)
    d = g[:, 40]
    # These are invariance/observed-coordinate comparisons, NOT gold direction.
    conflict = ((chosen == 0) & (d > 0)) | ((chosen == 2) & (d < 0))
    return dict(total=len(rows), eligible=len(valid), unavailable=len(rows) - len(valid),
        argmax_progression=int(np.isin(chosen[valid], [0, 2]).sum()),
        argmax_hold=int((chosen[valid] == 1).sum()),
        explicit_direction_mass_support=int((np.maximum(p[valid, 0], p[valid, 2]) >= .65).sum()),
        coordinate_opposed_argmax=int(conflict[valid].sum()),
        no_direction_gold=True, no_completion_gold=True,
        meaning='Progression annotation only; coordinate sign agreement is a model contract, not semantic direction accuracy.')


def static_metrics(rows, logits, eligible):
    chosen = logits.argmax(1)
    p = softmax(logits)
    result = []
    for split in ('train', 'probe'):
        indices = [i for i, r in enumerate(rows) if r['split'] == split]
        valid = [i for i in indices if eligible[i]]
        result.append(dict(split=split, total=len(indices), eligible=len(valid), unavailable=len(indices) - len(valid),
            argmax_hold=int((chosen[valid] == 1).sum()),
            argmax_progression=int(np.isin(chosen[valid], [0, 2]).sum()),
            explicit_hold_support=int((p[valid, 1] >= .65).sum()),
            source='Mathematically constant measured geometry; not independent human stillness/error data.'))
    return result


def main():
    dest = U / 'VERIFICATION.json'
    assert not dest.exists(), 'Completed verification preserved'
    # Files must already exist; this script never waits by rerunning an evaluator.
    meta = rd(U / 'INPUTS_COMPLETE.json')
    prep_review = rd(U / 'INPUT_PREPARATION_REVIEW.json')
    cfg = rd(U / 'PREPARATION_CONTRACT.json')
    contract = rd(U / 'TRAINING_CONTRACT.json')
    complete = rd(U / 'TRAINING_COMPLETE.json')
    ef = rd(U / 'evaluation/FROZEN.json')
    result = rd(U / 'evaluation/RESULTS.json')
    previous = rd(H / 'evaluation/RESULTS.json')
    pins(meta['pins'], 'input pin')
    pins(contract['pins'], 'training pin')
    pins(complete['pins'], 'completed training pin')
    pins(ef['pins'], 'evaluation pin')
    pins(result['source_pins'], 'runtime source pin')
    ck('input NPZ hash', sha(meta['path']) == meta['sha256'])
    ck('preparation separately checked', prep_review['status'] == 'PASS')
    ck('one architecture three fixed runs', cfg['architecture_conditions'] == 1 and cfg['seeds'] == [42, 43, 44] and cfg['steps'] == 800)
    ck('all seeds retained', sorted(r['seed'] for r in complete['runs']) == [42, 43, 44])
    with np.load(meta['path'], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    primary_rows = meta['primary_rows']
    pairs = {}
    for i, r in enumerate(primary_rows):
        pair = tuple(r[k] for k in ('clip_id', 'view', 'origin', 'a', 'b', 'axis', 'moving', 'anchor'))
        pairs.setdefault(pair, {})[r['reverse']] = i
    run_summary = []
    scales = {}
    for run in complete['runs']:
        seed = run['seed']
        pins({run[k]: run[k + '_sha256'] for k in ('model', 'predictions')}, 'run pin')
        state = torch.load(run['model'], map_location='cpu', weights_only=True)
        scale = float(np.logaddexp(0., float(state['raw_scale'])))
        scales[seed] = scale
        ck('scale positive ' + str(seed), scale > 0 and np.isclose(scale, run['scale'], rtol=1e-6))
        ck('norm buffers exact ' + str(seed), np.array_equal(state['gm'].numpy(), arrays['mean']) and np.array_equal(state['gs'].numpy(), arrays['std']) and float(state['delta_scale']) == float(arrays['delta_scale']))
        ck('no common100/context parameters ' + str(seed), all(k in ('raw_scale', 'gm', 'gs', 'delta_scale') or k.startswith(('motion_conv.', 'motion_gru.', 'motion_head.')) for k in state))
        ck('fixed final step ' + str(seed), run['log'][-1]['step'] == 800 and run['partial_direction_labels'] is False)
        with np.load(run['predictions'], allow_pickle=False) as z:
            lo = {k: z[k].copy() for k in z.files}
        for collection in ('primary', 'partial', 'static'):
            g = arrays[collection + '_geometry']
            logits = lo[collection + '_logits']
            ck('finite saved logits ' + str((seed, collection)), logits.shape == (len(g), 3) and np.isfinite(logits).all())
            expected = scale * g[:, 40].astype(float) / float(arrays['delta_scale'])
            error = max(float(abs(logits[:, 0] + expected).max()), float(abs(logits[:, 2] - expected).max()))
            ck('uncentered signed logit arithmetic ' + str((seed, collection)), error < 3e-5, error)
            ck('direction scores antisymmetric ' + str((seed, collection)), np.array_equal(logits[:, 0], -logits[:, 2]))
        symmetry_error = 0.
        for pair, indices in pairs.items():
            ck('both fixed reversals ' + str((seed, pair)), set(indices) == {False, True})
            if set(indices) != {False, True}:
                continue
            i, j = indices[False], indices[True]
            ck('same availability on reversal ' + str((seed, pair)), arrays['primary_eligible'][i] == arrays['primary_eligible'][j])
            if arrays['primary_eligible'][i]:
                symmetry_error = max(symmetry_error, float(abs(lo['primary_logits'][i] - lo['primary_logits'][j, [2, 1, 0]]).max()))
        ck('saved reversal equivariance ' + str(seed), symmetry_error < 2e-4, symmetry_error)
        run_summary.append(dict(seed=seed, scale=scale, paired_primary_intervals=len(pairs), maximum_reversal_logit_error=symmetry_error,
            primary=primary_metrics(primary_rows, lo['primary_logits'], arrays['primary_eligible']),
            partial=partial_metrics(meta['partial_rows'], arrays['partial_geometry'], lo['partial_logits'], arrays['partial_eligible']),
            static=static_metrics(meta['static_rows'], lo['static_logits'], arrays['static_eligible'])))
    prior_rows = {key(r): r for r in previous['rows']}
    rows = {key(r): r for r in result['rows']}
    source_pred = {(*key(r), r['card']): r for r in previous['direction_predictions'] if r['condition'] == 'reencoded_projected_geometry'}
    pred = {(*key(r), r['card']): r for r in result['predictions']}
    ck('same246rows82scenarios', len(rows) == len(result['rows']) == 246 and set(rows) == set(prior_rows))
    ck('same72axis2events', len(pred) == len(result['predictions']) == 72 and set(pred) == set(source_pred))
    preserved = 0
    changed = []
    used = set()
    for k, r in rows.items():
        before = prior_rows[k]['conditions']
        after = r['conditions']
        ck('same four conditions ' + str(k), set(after) == set(ef['conditions']))
        for name in ('baseline', 'local_old', 'reencoded_projected_geometry'):
            ck('saved condition exact ' + str((k, name)), after[name] == before[name])
        ck('card count exact ' + str(k), len(after['palm_progression']) == len(before['local_old']))
        for card, oc in zip(after['palm_progression'], before['local_old']):
            ck('unchanged selected extent/card ' + str((k, card['card'])), {f: v for f, v in card.items() if f not in ('evidence', 'status')} == {f: v for f, v in oc.items() if f not in ('evidence', 'status')})
            ck('event count exact ' + str((k, card['card'])), len(card['evidence']) == len(oc['evidence']))
            for e, oe in zip(card['evidence'], oc['evidence']):
                if e['task'] != 'direction' or e.get('axis') != 2:
                    ck('nonaxis2 event exact ' + str((k, card['card'], preserved)), e == oe)
                    preserved += 1
                    continue
                pk = (*k, card['card'])
                p, old_p = pred[pk], source_pred[pk]
                used.add(pk)
                ck('same interval/source/desired ' + str(pk), p['span'] == e['span'] == oe['span'] == old_p['span'] and p['desired'] == e['desired'] == oe['desired'] and p['source'] == old_p['source'] and p['source_sha256'] == old_p['source_sha256'])
                ck('other evidence fields exact ' + str(pk), {f: v for f, v in e.items() if f not in ('probabilities', 'state', 'model_scope')} == {f: v for f, v in oe.items() if f not in ('probabilities', 'state', 'model_scope')})
                if p['eligible']:
                    g = np.asarray(p['g'], float)
                    logits = np.asarray(p['logits'], float)
                    expected = scales[k[0]] * g[40] / float(arrays['delta_scale'])
                    ck('runtime raw direction arithmetic ' + str(pk), abs(logits[0] + expected) < 3e-5 and abs(logits[2] - expected) < 3e-5)
                    ck('runtime softmax ' + str(pk), np.allclose(softmax(logits), p['probabilities'], atol=1e-6))
                else:
                    ck('unavailable not interpreted ' + str(pk), p['logits'] is None and not np.asarray(p['g']).any() and p['probabilities'] == [0., 0., 0.])
                ck('existing verdict rule ' + str(pk), verdict(p['probabilities'], p['desired'], p['eligible']) == p['state'] == e['state'] and e['probabilities'] == p['probabilities'])
                if e['state'] != oe['state']:
                    changed.append(dict(seed=k[0], clip_id=k[1], kind=k[2], card=card['card'], old=oe['state'], new=e['state'], eligible=p['eligible'], delta=p['g'][40]))
            ck('card state arithmetic ' + str((k, card['card'])), card['status'] == card_status(card))
    ck('all predictions consumed', used == set(pred))
    summary = []
    for seed in (42, 43, 44):
        for condition in result['rows'][0]['conditions']:
            for kind in sorted({r['kind'] for r in result['rows']}):
                events = [e for r in result['rows'] if r['seed'] == seed and r['kind'] == kind for c in r['conditions'][condition] for e in c['evidence'] if e['task'] == 'direction' and e.get('axis') == 2]
                summary.append(dict(seed=seed, condition=condition, kind=kind, n=len(events), states=dict(Counter(e['state'] for e in events))))
    ck('all condition denominators/summary exact', summary == result['summary'])
    lookup = {(r['seed'], r['condition'], r['kind']): r for r in summary}
    comparisons = []
    for seed in (42, 43, 44):
        for kind, state_name in [('full_normal', 'supported'), ('speed075', 'supported'), ('speed125', 'supported'), ('main_reversed', 'opposite_direction'), ('closed_hand_and_reversed', 'opposite_direction')]:
            nr = lookup[seed, 'palm_progression', kind]
            vals = {name: lookup[seed, name, kind]['states'].get(state_name, 0) for name in ('baseline', 'local_old', 'reencoded_projected_geometry', 'palm_progression')}
            comparisons.append(dict(seed=seed, kind=kind, counted_state=state_name, n=nr['n'], counts=vals,
                no_decrease_vs_local_old=vals['palm_progression'] >= vals['local_old'],
                no_decrease_vs_reencoded=vals['palm_progression'] >= vals['reencoded_projected_geometry']))
    protections = rd(G / 'PROTECTIONS.json')
    ck('protected477', len(protections) == 477)
    pins(protections, 'protected file')
    ck('old models/private/final8/web unchanged', complete['old_models_changed'] is False and all(complete[k] is False for k in ('web_changed', 'private_access', 'final8')))
    report = dict(at=datetime.datetime.now().astimezone().isoformat(), status='PASS' if not failures else 'FAIL', checks=checks,
        failures=failures, runs=run_summary, comparisons=comparisons, runtime_summary=summary, changed_events=changed,
        runtime_rows=len(rows), axis2_events=len(pred), preserved_other_events=preserved, protected_files=len(protections),
        no_model_forward=True, no_training=True, no_feature_rebuild=True, full_G2_pass=False, adopted=False,
        reviewer='Codex saved arithmetic check; same agent authored input/core, root separately reviewed code and trained/evaluated. Not independent human semantic gold.',
        limitations=['Partial rows have progression labels only: no direction accuracy or completion claim.',
            'Probe08/09 and development fixtures are exposed upstream development data, not independent final generalization.',
            'Legacy closed-hand intervention changes palm/MCP position, so its direction comparison is not pure handshape invariance.',
            'Static measured geometry controls do not prove performance on natural human holds or novice errors.',
            '2D palm projection and reversal score constraints do not establish contact, full trajectory, action association or completion.'],
        pins={str(p): sha(p) for p in [Path(__file__), U / 'TRAINING_COMPLETE.json', U / 'evaluation/RESULTS.json', U / 'INPUT_PREPARATION_REVIEW.json']})
    # Serialize fully before opening the final artifact, so a report-writing
    # failure cannot masquerade as a completed verification.
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    with dest.open('x', encoding='utf-8') as f:
        f.write(serialized + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'checks', 'failures', 'runtime_rows', 'axis2_events', 'preserved_other_events', 'protected_files')}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
