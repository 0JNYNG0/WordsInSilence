"""Independent arithmetic audit of saved development predictions; never runs a model.

Run only after EVALUATION_COMPLETE.json. This file intentionally imports none of
the training/evaluation/feature code. Torch is used solely to compare weights.
PASS means that evidence and arithmetic agree, not that a candidate passed its
scientific selection gate or that educational feedback accuracy was established.
"""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import sys
import time
import traceback
import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
B = O / 'word100_shared_motion_learning_20261109'
S = O / 'word100_gru_semantic_finetune_20261109'
AUX = O / 'demo_handshape_learning_20261113'
C7 = O / 'word100_web_completion_graph_20261111/typed_motion_recovery_20261113/body_context_generalization/c7_local_cause_recovery_20261113'
FINE = ('normal', 'upper', 'slow', 'fast', 'fine_missing_2', 'fine_missing_12')
LEGACY = ('normal', 'upper', 'slow', 'fast', 'missing_2', 'missing_long', 'reversed_main')
DENOM = dict(expanded_shape=133, normal_order=12, original_shape_labels=194,
             reviewed_shape=228, semantic_stage=84)
CHECKS = Counter()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message, category='assertions'):
    CHECKS[category] += 1
    if not value:
        raise AssertionError(message)


def same(actual, expected, where):
    """Recursive comparisons without importing the original summarizer."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), where + ': keys')
        for key in expected:
            same(actual[key], expected[key], where + '/' + str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), where + ': length')
        for index, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, where + '/' + str(index))
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=2e-8, abs_tol=1e-9), where + ': numeric mismatch')
    else:
        require(actual == expected, where + ': value mismatch')


def hash_group(mapping, label):
    for path, wanted in mapping.items():
        require(sha(path) == wanted, label + ': ' + str(path), 'hashes')
    return len(mapping)


def finger_metrics(rows):
    # Derive confusion using counting, and precision/recall denominators directly.
    cm = [[0 for _ in range(3)] for _ in range(3)]
    accepted = accepted_correct = correct = 0
    for row in rows:
        t, p = row['target'], row['pred']
        cm[t][p] += 1
        correct += int(t == p)
        if row['accepted']:
            accepted += 1
            accepted_correct += int(t == p)
    f1 = []
    for state in range(3):
        actual = sum(cm[state])
        predicted = sum(cm[j][state] for j in range(3))
        f1.append(2.0 * cm[state][state] / (actual + predicted) if actual + predicted else None)
    present = [x for x in f1 if x is not None]
    return dict(total=len(rows), correct=correct, accuracy=correct / len(rows) if rows else None,
                macro_f1=sum(present) / len(present) if present else None, state_f1=f1,
                confusion=cm, accepted=accepted, accepted_correct=accepted_correct,
                withheld=len(rows) - accepted)


def legacy_metrics(rows):
    totals = {}
    for task in sorted({r['task'] for r in rows}):
        rs = [r for r in rows if r['task'] == task]
        accepted = [r for r in rs if r['eligible'] and r['confident']]
        totals[task] = dict(n=len(rs), raw_correct=sum(r['pred'] == r['target'] for r in rs),
                            accepted=len(accepted), accepted_correct=sum(r['pred'] == r['target'] for r in accepted))
    return totals


def annotation_targets(metadata):
    labels = read(U / 'annotation_review/ORDINARY_LABELS.json')['rows']
    labels += read(U / 'annotation_review/EXTERNAL_LABELS.json')['rows']
    for row in read(AUX / 'ADMISSION.json')['rows']:
        labels.append(dict(clip_id=row['clip_id'], frame=row['frame'], side='R',
                           y=[{1: 0, 0: 2, -1: -1}[v] for v in row['finger_states']],
                           source_sha256=row['source_sha256']))
    targets = defaultdict(dict)
    for row in labels:
        cid = row['clip_id']
        if cid not in metadata:
            continue
        meta = metadata[cid]
        require(row['source_sha256'] == meta['source_sha256'], 'annotation source ' + cid)
        if 'role' in row:
            require(row['role'] == meta['role'] and row['person'] == meta['person'], 'annotation role ' + cid)
        side = 0 if row['side'] == 'L' else 1
        key = (int(row['frame']), side)
        require(key not in targets[cid], 'duplicate annotation anchor ' + cid)
        values = list(row['y'])
        if meta['origin'] == 'selected_extra' and meta['word_id'] in ('WORD2633', 'WORD2635'):
            values[0] = -1  # Explicit pre-training admission correction, not inferred truth.
        targets[cid][key] = values
    return targets, len(labels)


def read_truth():
    ready = read(U / 'DATA_READY.json')
    metadata = {r['clip_id']: r for r in ready['rows']}
    require(len(metadata) == len(ready['rows']) == 1273, 'cache uniqueness/count')
    count = Counter((r['origin'], r['role']) for r in metadata.values())
    require(count == Counter({('common100', 'fit'): 899, ('common100', 'development'): 300,
                              ('selected_extra', 'fit'): 56, ('selected_extra', 'development'): 18}), 'split sizes')
    for role, allowed in [('fit', set(range(1, 10))), ('development', {11, 12, 13})]:
        people = {r['person'] for r in metadata.values() if r['role'] == role}
        require(people == allowed, role + ' public record IDs')
    fit_sha = {r['source_sha256'] for r in metadata.values() if r['role'] == 'fit'}
    dev_sha = {r['source_sha256'] for r in metadata.values() if r['role'] == 'development'}
    require(not fit_sha & dev_sha, 'source SHA leakage')
    annotations, label_count = annotation_targets(metadata)
    rows, coverage = {}, defaultdict(Counter)
    for cid, meta in metadata.items():
        require(sha(meta['path']) == meta['sha256'], 'cache hash ' + cid, 'hashes')
        with np.load(meta['path'], allow_pickle=False) as z:
            needed = ['x_full', 'x_upper', 'hand', 'fy', 'frame_id', 'timestamp_us', 'stage_gold', 'y']
            needed += [k for k in ('direct_ym', 'old_ym', 'old_y') if k in z]
            r = dict(meta, **{k: z[k].copy() for k in needed})
        n = len(r['frame_id'])
        require((np.diff(r['frame_id']) > 0).all() and (np.diff(r['timestamp_us']) > 0).all(), 'native time ' + cid)
        require(r['fy'].shape == (n, 2, 5) and set(np.unique(r['fy'])).issubset({-1, 0, 1, 2}), 'fy shape/states ' + cid)
        for key in ['x_full', 'x_upper', 'hand']:
            require(np.isfinite(r[key]).all(), 'finite cache ' + cid + '/' + key)
        expected = np.full((n, 2, 5), -1, dtype=np.int64)
        index = {int(f): i for i, f in enumerate(r['frame_id'])}
        for (frame, side), values in annotations.get(cid, {}).items():
            require(frame in index, 'annotation frame ' + cid)
            expected[index[frame], side] = values
        require(np.array_equal(expected, r['fy']), 'known AND unknown annotation cells ' + cid)
        known = int((r['fy'] >= 0).sum())
        require(known == meta['finger_known'], 'known cell count ' + cid)
        group = meta['origin'] + '/' + meta['role']
        coverage[group]['clips'] += 1
        coverage[group]['known'] += known
        for s in range(3):
            coverage[group]['state_' + str(s)] += int((r['fy'] == s).sum())
        coverage[group]['left'] += int((r['fy'][:, 0] >= 0).sum())
        coverage[group]['right'] += int((r['fy'][:, 1] >= 0).sum())
        if meta['role'] == 'development' and meta['origin'] == 'common100':
            rows[cid] = r
    require(coverage['common100/development']['known'] == 389, 'ordinary fine denominator')
    require({v['class_index'] for v in rows.values()} == set(range(100)), '100 ordinary classes')
    require(all(n == 3 for n in Counter(v['word_id'] for v in rows.values()).values()), '3 development records per word')
    return rows, dict(coverage), label_count


def observed_variant(r, variant):
    """Independently construct observation indices and masks; no network call."""
    x = r['x_upper' if variant == 'upper' else 'x_full'].copy()
    h = r['hand'].copy()
    n = len(x)
    source = np.arange(n)
    if variant == 'upper':
        for side, start in enumerate((195, 216)):
            mask = x[:, start:start + 21] > .5
            valid_basis = mask[:, [0, 5, 9, 13, 17]].all(axis=1)
            h[:, side, :42] *= np.repeat(mask, 2, axis=1)
            h[:, side, 42:] *= mask
            h[:, side] *= valid_basis[:, None]
    gold = np.flatnonzero(r['stage_gold'] >= 0)
    if variant == 'reversed_main' and len(gold):
        first = min([int(gold[0])] + [int(p['a']) for p in r['pairs']])
        last = max([int(gold[-1])] + [int(p['b']) for p in r['pairs']])
        source[first:last + 1] = np.arange(last, first - 1, -1)
        x, h = x[source], h[source]
    if variant in ('slow', 'fast'):
        count = round(n * (1.25 if variant == 'slow' else .75))
        source = np.rint(np.linspace(0, n - 1, count)).astype(np.int64)
        x, h = x[source], h[source]
    if variant.startswith('fine_missing_') or variant in ('missing_2', 'missing_long'):
        if variant.startswith('fine_missing_'):
            center = int(np.flatnonzero((r['fy'] >= 0).any(axis=(1, 2)))[0])
            gap = int(variant.rsplit('_', 1)[1])
        else:
            center = int(np.median(gold)) if len(gold) else n // 2
            gap = 2 if variant == 'missing_2' else n
        first, last = max(0, center - gap // 2), min(n, max(0, center - gap // 2) + gap)
        for a, b in ((18, 102), (102, 186), (195, 237), (237, 279)):
            x[first:last, a:b] = 0
        h[first:last] = 0
    return x, h, source


def coarse_available(x, feature):
    visible = x[:, 186:237] > .5
    if feature in (5, 6):
        points = [9, 30]
    elif feature in (2, 4):
        points = [9] + list(range(14, 30))
    else:
        points = [30] + list(range(35, 51))
    return visible[:, points].all(axis=1)


def identity(row):
    return (row['variant'], row['task'], row['clip_id'], row.get('frame', -1), row.get('feature', -1), row['target'], bool(row['eligible']))


def expected_records(rows):
    groups = {int(s): group for group, values in enumerate(read(S / 'FROZEN.json')['semantic_stage_groups']) for s in values}
    fine, legacy = {}, Counter()
    for cid, r in rows.items():
        anchors = np.argwhere(r['fy'] >= 0)
        if not len(anchors):
            require(r['word'] < 0, 'feedback clip absent from fine evaluation loop ' + cid)
            continue
        for variant in tuple(dict.fromkeys(FINE + LEGACY)):
            if variant == 'reversed_main' and r['word'] < 0:
                continue
            x, h, source = observed_variant(r, variant)
            nearest = {int(i): int(np.abs(source - i).argmin()) for i in range(len(r['frame_id']))}
            if variant in FINE:
                for i, side, finger in anchors:
                    at = nearest[int(i)]
                    joints = [0] + list(range(1 + int(finger) * 4, 5 + int(finger) * 4))
                    eligible = bool((h[at, side, 42:][joints] > .5).all())
                    key = (variant, cid, int(r['frame_id'][i]), 'L' if side == 0 else 'R', int(finger))
                    fine[key] = dict(target=int(r['fy'][i, side, finger]), person=r['person'], word=r['word_id'],
                                     pts_us=int(r['timestamp_us'][i]), eligible=eligible)
            if variant not in LEGACY or r['word'] < 0:
                continue
            for task, mask, target in [('reviewed_shape', r['direct_ym'], r['y']),
                                       ('original_shape_labels', r['old_ym'], r['old_y'])]:
                for i, feature in np.argwhere(mask > 0):
                    row = dict(variant=variant, task=task, clip_id=cid, frame=int(r['frame_id'][i]),
                               feature=int(feature), target=int(target[i, feature]),
                               eligible=bool(coarse_available(x, int(feature))[nearest[int(i)]]))
                    legacy[identity(row)] += 1
            for i in np.flatnonzero(r['stage_gold'] >= 0):
                at = nearest[int(i)]
                row = dict(variant=variant, task='semantic_stage', clip_id=cid, frame=int(r['frame_id'][i]),
                           target=groups[int(r['stage_gold'][i])],
                           eligible=bool(min(x[at, 187:189]) > .5 and sum(x[at, 186:237]) >= 5))
                legacy[identity(row)] += 1
            available = np.stack([coarse_available(x, f) for f in range(7)], axis=1)
            for pair in r['orders']:
                a, b = sorted((nearest[int(pair['a'])], nearest[int(pair['b'])]))
                row = dict(variant=variant, task='normal_order', clip_id=cid,
                           target=int(variant != 'reversed_main'), eligible=bool(available[a:b + 1].any()))
                legacy[identity(row)] += 1
    for row in read(B / 'EXTRA_EVAL_LABELS.json')['rows']:
        r = rows[row['clip_id']]
        at = int(np.flatnonzero(r['frame_id'] == row['frame'])[0])
        for feature, target in enumerate(row['targets']):
            if target is not None:
                legacy[identity(dict(variant='normal', task='expanded_shape', clip_id=row['clip_id'],
                                     frame=row['frame'], feature=feature, target=target,
                                     eligible=bool(coarse_available(r['x_full'], feature)[at])))] += 1
    require(len(fine) == len(FINE) * 389, 'all fine variants fixed denominator')
    return fine, legacy


def check_words(words, rows, label):
    require(len(words) == 300 and len({r['clip_id'] for r in words}) == 300, label + ' word denominator')
    require({r['clip_id'] for r in words} == set(rows), label + ' word cohort')
    for row in words:
        require(row['target'] == rows[row['clip_id']]['class_index'] and 0 <= row['pred'] < 100, label + ' word target/range')
    return dict(correct=sum(r['target'] == r['pred'] for r in words), total=300)


def check_legacy(records, expected, label):
    require(Counter(identity(r) for r in records) == expected, label + ' all legacy truth/identity/mask cells')
    for row in records:
        if row['task'] in ('expanded_shape', 'original_shape_labels', 'reviewed_shape'):
            p = row['probability']
            require(math.isfinite(p) and 0 <= p <= 1 and row['pred'] == int(p >= .5), label + ' coarse probability')
            require(row['confident'] == (p <= .35 or p >= .65), label + ' coarse confidence')
        elif row['task'] == 'normal_order':
            require(row['pred'] in (0, 1), label + ' order binary output')
    out = {v: legacy_metrics([r for r in records if r['variant'] == v]) for v in LEGACY}
    for v, tasks in out.items():
        wanted = DENOM if v == 'normal' else {k: n for k, n in DENOM.items() if k != 'expanded_shape'}
        same({k: t['n'] for k, t in tasks.items()}, wanted, label + '/' + v + '/denominators')
    return out


def check_fingers(records, expected, name):
    keys = [(r['variant'], r['clip_id'], r['frame'], r['side'], r['finger']) for r in records]
    require(len(keys) == len(set(keys)) and set(keys) == set(expected), name + ' fine known-only cohort')
    for key, row in zip(keys, records):
        for field, value in expected[key].items():
            require(row[field] == value, name + '/fine/' + field)
        probabilities = np.asarray(row['probability'], dtype=np.float64)
        require(probabilities.shape == (3,) and np.isfinite(probabilities).all() and
                (probabilities >= 0).all() and (probabilities <= 1).all() and
                abs(float(probabilities.sum()) - 1) < 2e-6, name + ' three state probabilities')
        require(row['pred'] == int(probabilities.argmax()), name + ' fine argmax')
        require(row['accepted'] == bool(row['eligible'] and probabilities[row['pred']] >= .85), name + ' acceptance mask')
    normal = [r for r in records if r['variant'] == 'normal']
    return dict(finger_by_variant={v: finger_metrics([r for r in records if r['variant'] == v]) for v in FINE},
                finger_by_hand={s: finger_metrics([r for r in normal if r['side'] == s]) for s in ('L', 'R')},
                finger_by_record={str(p): finger_metrics([r for r in normal if r['person'] == p]) for p in (11, 12, 13)},
                finger_by_word={w: finger_metrics([r for r in normal if r['word'] == w]) for w in sorted({r['word'] for r in normal})},
                finger_by_finger={str(f): finger_metrics([r for r in normal if r['finger'] == f]) for f in range(5)})


def compare_weights(cfg):
    import torch  # No network architecture is imported or instantiated.
    result = []
    expected_seen = {r['clip_id'] for r in read(U / 'DATA_READY.json')['rows']
                     if r['role'] == 'fit' and r['origin'] == 'common100'}
    for seed in cfg['seeds']:
        before = torch.load(B / f'runs/shared100_{seed}/model.pt', map_location='cpu', weights_only=True)
        for condition in cfg['conditions']:
            name = f'{condition}_{seed}'
            directory = U / 'runs' / name
            note = read(directory / 'COMPLETE.json')
            checkpoint = directory / 'model.pt'
            require(note['checkpoint_sha256'] == sha(checkpoint), name + ' completed model hash', 'hashes')
            require(note['condition'] == condition and note['seed'] == seed and note['development_training'] is False,
                    name + ' run identity')
            require(note['shared100_seen'] == 899 and [v['epoch'] for v in note['history']] == list(range(1, 13)), name + ' training coverage/epochs')
            after = torch.load(checkpoint, map_location='cpu', weights_only=True)
            require(set(before).issubset(after) and all(k.startswith(('finger', 'extra')) for k in set(after) - set(before)), name + ' architecture keys')
            frozen = 0
            for key, tensor in before.items():
                require(after[key].shape == tensor.shape and torch.isfinite(after[key]).all().item(), name + ' tensor ' + key)
                if not key.startswith(('base.input', 'base.temporal', 'base.context')):
                    require(torch.equal(after[key], tensor), name + ' frozen legacy weight ' + key)
                    frozen += 1
            deltas = {}
            for prefix, reported in note['deltas'].items():
                squared = sum(float((after[k].double() - v.double()).square().sum()) for k, v in before.items() if k.startswith(prefix))
                delta = math.sqrt(squared)
                require(math.isclose(delta, reported, rel_tol=2e-6, abs_tol=1e-10), name + ' independent weight delta ' + prefix)
                deltas[prefix] = delta
            require(all(deltas[p] > 0 for p in ('base.input', 'base.temporal', 'base.context')), name + ' genuine base fine-tuning')
            last = torch.load(directory / 'last.pt', map_location='cpu', weights_only=True)
            require(last['frozen_sha256'] == sha(U / 'FROZEN.json') and last['epoch'] == 12, name + ' resume freeze/epoch')
            require(set(last['seen']) == expected_seen and len(expected_seen) == 899, name + ' resume coverage')
            require(set(after) == set(last['model']), name + ' resume model keys')
            require(all(torch.equal(after[k], last['model'][k]) for k in after), name + ' last/complete weight consistency')
            same(last['history'], note['history'], name + '/history')
            result.append(dict(model=name, frozen_old_tensors=frozen, independent_deltas=deltas, complete_resume_consistent=True))
    return result


def candidate_gates(models, baseline, cfg):
    retention, gates = [], []
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            now, old = models[f'{condition}_{seed}'], baseline[seed]
            fail = []
            for variant, tasks in now['legacy'].items():
                for task, q in tasks.items():
                    b = old['legacy'][variant][task]
                    require(q['n'] == b['n'], 'paired legacy denominator')
                    if q['raw_correct'] < b['raw_correct']:
                        fail.append(dict(variant=variant, task=task, before=b['raw_correct'], after=q['raw_correct'], n=q['n']))
            retention.append(dict(condition=condition, seed=seed, legacy_failures=fail,
                                  word100_before=old['word100']['correct'], word100_after=now['word100']['correct'],
                                  full_preservation=not fail and now['word100']['correct'] >= old['word100']['correct']))
    for seed in cfg['seeds']:
        a, c = models[f'A_{seed}'], models[f'C_{seed}']
        normal_a, normal_c = a['finger_by_variant']['normal'], c['finger_by_variant']['normal']
        gain = normal_c['macro_f1'] >= normal_a['macro_f1'] + .01
        hands = all(c['finger_by_hand'][s]['accuracy'] >= a['finger_by_hand'][s]['accuracy'] for s in ('L', 'R'))
        states = all((z or 0) >= (y or 0) for y, z in zip(normal_a['state_f1'], normal_c['state_f1']))
        retain = next(v for v in retention if v['condition'] == 'C' and v['seed'] == seed)
        fail = [dict(v, baseline_n=v['n']) for v in retain['legacy_failures']]
        words = c['word100']['correct'] >= baseline[seed]['word100']['correct']
        gates.append(dict(seed=seed, consistent_fine_gain=gain, hands_preserved=hands, states_preserved=states,
                          legacy_failures=fail, word100_preserved=words,
                          pass_candidate=bool(gain and hands and states and not fail and words)))
    return retention, gates


def main():
    require((U / 'EVALUATION_COMPLETE.json').is_file(), 'Evaluation incomplete; do not run this verifier yet.')
    cfg = read(U / 'FROZEN.json')
    completion = read(U / 'EVALUATION_COMPLETE.json')
    require(cfg['conditions'] == ['A', 'B', 'C'] and cfg['seeds'] == [42, 43, 44], 'frozen paired nine design')
    require(cfg['production_changed'] is False and cfg['training_no_private'] is True, 'scope metadata')
    protection = dict(training_pins=hash_group(cfg['pins'], 'training pins'),
                      production_pins=hash_group(cfg['protected'], 'protected old production'),
                      evaluation_pins=hash_group(read(U / 'EVAL_FROZEN.json')['pins'], 'evaluation pins'))
    c7pins = read(C7 / 'TRAINING_FROZEN.json')['protected_pins']
    require(len(c7pins) == 477, 'separate existing graph protection denominator')
    protection['existing_graph_477'] = hash_group(c7pins, 'existing graph protection')
    rows, coverage, annotations = read_truth()
    expected_fine, expected_legacy = expected_records(rows)
    baseline = {}
    for seed in cfg['seeds']:
        old = read(B / f'scores/shared100_{seed}.json')
        raw = read(B / f'scores/shared100_{seed}_records.json')
        metrics = check_legacy(raw, expected_legacy, 'baseline_' + str(seed))
        for variant, tasks in metrics.items():
            for task, values in tasks.items():
                same({k: old['by_variant'][variant][task][k] for k in values}, values, f'baseline/{seed}/{variant}/{task}')
        word = check_words(read(B / f'scores/shared100_{seed}_words.json'), rows, 'baseline_' + str(seed))
        same({k: old['aux_word'][k] for k in word}, word, 'baseline auxiliary words')
        baseline[seed] = dict(legacy=metrics, word100=word)
    models, raw_hashes, latency_status = {}, {}, {}
    for condition in cfg['conditions']:
        for seed in cfg['seeds']:
            name = f'{condition}_{seed}'
            paths = {k: U / 'scores' / f'{name}_{k}.json' for k in ('fingers', 'legacy', 'words')}
            raw_hashes[name] = {k: sha(p) for k, p in paths.items()}
            saved = read(U / 'scores' / f'{name}.json')
            derived = check_fingers(read(paths['fingers']), expected_fine, name)
            derived['legacy'] = check_legacy(read(paths['legacy']), expected_legacy, name)
            derived['word100'] = check_words(read(paths['words']), rows, name)
            for key, value in derived.items():
                same(saved[key], value, name + '/' + key)
            same(completion['models'][name], saved, name + '/completion consistency')
            latency = saved['latency']
            require(latency['clips'] == 36 and 0 < latency['p50_seconds'] <= latency['p95_seconds'], name + ' latency scope')
            # A separately saved raw vector enables independent percentile arithmetic.
            latency_path = U / 'scores' / f'{name}_latency.json'
            if latency_path.exists():
                raw_latency = read(latency_path)
                if isinstance(raw_latency, dict):
                    if 'clip_ids' in raw_latency:
                        expected_clips = {cid for cid, r in rows.items() if (r['fy'] >= 0).any()}
                        require(len(raw_latency['clip_ids']) == len(set(raw_latency['clip_ids'])) == 36 and
                                set(raw_latency['clip_ids']) == expected_clips, name + ' latency clip cohort')
                    raw_latency = raw_latency.get('rows', raw_latency.get('seconds', raw_latency.get('values')))
                require(isinstance(raw_latency, list) and len(raw_latency) == 36, name + ' latency raw count')
                seconds = [float(v['seconds'] if isinstance(v, dict) else v) for v in raw_latency]
                require(all(math.isfinite(v) and v > 0 for v in seconds), name + ' latency raw finite')
                sorted_seconds = sorted(seconds)
                median = (sorted_seconds[17] + sorted_seconds[18]) / 2
                position = .95 * (len(seconds) - 1)
                lo = math.floor(position)
                p95 = sorted_seconds[lo] + (position - lo) * (sorted_seconds[lo + 1] - sorted_seconds[lo])
                same(latency['p50_seconds'], median, name + '/latency median')
                same(latency['p95_seconds'], p95, name + '/latency p95')
                raw_hashes[name]['latency'] = sha(latency_path)
                latency_status[name] = 'saved raw 36 timings independently recomputed; no new measurement'
            else:
                latency_status[name] = 'summary count/range only; raw timings absent, percentiles not independently verified'
            models[name] = derived
    weights = compare_weights(cfg)
    retention, gates = candidate_gates(models, baseline, cfg)
    same(completion['retention_all_conditions'], retention, 'all nine retention gates')
    same(completion['gates'], gates, 'paired C versus A selection gates')
    passed = all(g['pass_candidate'] for g in gates)
    same(completion['all_pass'], passed, 'selection all seeds')
    require(completion['no_production_promotion'] is True, 'no promotion')
    return dict(status='PASS_SAVED_RESULT_ARITHMETIC', verification_actor='separate Codex agent, not independent human KSL truth',
                inference_calls=0, training_runs=0, source_scope='public cached FIT01-09 / historical DEV11-13 only',
                frozen_sha256=sha(U / 'FROZEN.json'), evaluation_sha256=sha(U / 'EVALUATION_COMPLETE.json'),
                checks=dict(CHECKS), protection=protection, annotation_rows=annotations, coverage=coverage,
                per_variant_known_finger_cells=389, ordinary_fine_states=['extended', 'bent', 'folded'],
                per_variant_denominators=DENOM, baselines=baseline, models=models,
                paired_selection_pass=passed, selection_gates=gates, retention_all_conditions=retention,
                weight_changes=weights, saved_score_hashes=raw_hashes, latency_verification=latency_status,
                limitations=['Fine task is physical finger state in ordinary signs, not numeric-word recognition or educational correctness.',
                             'Historical development cohort is reused; 6 variants share the same 389 known cells, not 2334 independent examples.',
                             'Unknown (-1) labels are excluded from truth; unavailable but known cells remain in the raw-score denominator and cannot be accepted.',
                             'Order metric is the historical stage-index-order proxy; it does not establish complete movement direction or repetition feedback.',
                             'Semantic-stage confidence and order confidence are checked as saved flags unless full per-frame probabilities exist; no inference is repeated.',
                             'Latency is cached-coordinate neural processing, excluding video extraction, network, UI and end-to-end user waiting.'])


if __name__ == '__main__':
    # Failing early is intentional if the training/evaluation process is unfinished.
    try:
        result = main()
    except BaseException:
        result = dict(status='FAIL_SAVED_RESULT_VERIFICATION', checks=dict(CHECKS), error=traceback.format_exc(),
                      inference_calls=0, training_runs=0)
        target = U / f'VERIFICATION_FAILURE_{time.time_ns()}.json'
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(result['error'], file=sys.stderr)
        raise
    else:
        target = U / 'RESULT_VERIFIED.json'
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(json.dumps(dict(status=result['status'], paired_selection_pass=result['paired_selection_pass'],
                              checks=result['checks'], report=str(target)), ensure_ascii=False))
