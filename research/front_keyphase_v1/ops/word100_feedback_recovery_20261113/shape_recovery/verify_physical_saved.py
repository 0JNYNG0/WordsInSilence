"""Independent saved-output arithmetic; no neural forward or training.

Read completed 18 physical wrist-window runs only. Never imports implementation
code. Checkpoint reading verifies tensors/metadata, not prediction regeneration.
"""
from pathlib import Path
import collections
import datetime
import hashlib
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
U = HERE.parent / 'motion_recovery'
EXPECTED_FROZEN = '9f0398eda9d84c54ab688ea8efa7149493d7048c4720e82d66c43051f83ca236'
EXPECTED_ADOPTION = 'b68d15b00d38ae0cb3cf5fd389854c5d5edd88c5f55fe4c010d0598df1bd5ad4'
PEOPLE = [2, 3, 4, 5, 6, 7]
SEEDS = [42, 43, 44]
VIEWS = ['full', 'upper', 'actual_crop']
OUTPUT = HERE / 'MOTION_SAVED_RESULT_VERIFIED.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stamp():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()


def main():
    # No partial-run polling or partial score publication.
    assert (U / 'PHYSICAL_TRAINING_COMPLETE.json').exists(), 'Await all18 completed runs.'
    assert not OUTPUT.exists(), 'Preserve the completed independent verification.'
    checks = []
    pins = {}

    def require(condition, name):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None:
            require(digest == expected, 'SHA: ' + str(path))
        pins[str(path)] = digest
        return digest

    pin(U / 'PHYSICAL_FROZEN.json', EXPECTED_FROZEN)
    pin(U / 'PHYSICAL_ADOPTION_CONTRACT.json', EXPECTED_ADOPTION)
    cfg = read(U / 'PHYSICAL_FROZEN.json')
    adoption = read(U / 'PHYSICAL_ADOPTION_CONTRACT.json')
    pretrain = read(U / 'PHYSICAL_PRETRAIN_VERIFICATION.json')
    pin(U / 'PHYSICAL_PRETRAIN_VERIFICATION.json')
    require(pretrain['status'] == 'PASS' and pretrain['frozen_sha256'] == EXPECTED_FROZEN,
            'Root pretraining gate tied to frozen SHA')
    for path, digest in {**cfg['pins'], **pretrain['additional_pins']}.items():
        pin(path, digest)
    require(cfg['seeds'] == SEEDS and [x['person'] for x in cfg['folds']] == PEOPLE,
            'Exactly six prespecified folds and three seeds')
    require(cfg['steps'] == 1200 and cfg['task'] == 'event.window_contains_wrist_translation',
            'Fixed task and training budget')
    require(adoption['known_source_cells'] == 228 and adoption['all_source_cells'] == 230,
            'Frozen source denominators')

    # Verify ancestor preserved files, not ancestor performance on a different task.
    for path in cfg['pins']:
        if Path(path).name == 'FROZEN.json':
            ancestor = read(path)
            for k in ('pins', 'protected_pins'):
                for p, h in ancestor.get(k, {}).items():
                    pin(p, h)

    labels = read(U / 'PHYSICAL_SOURCE_LABELS.json')['rows']
    rows = read(U / 'PHYSICAL_INDEX.json')['rows']
    with np.load(U / 'PHYSICAL_INPUTS.npz', allow_pickle=False) as f:
        z = {k: f[k].copy() for k in f.files}
    source = {r['key']: r for r in labels}
    require(len(labels) == len(source) == 115, '115 unique native source windows')
    require(len(rows) == 1380 and z['x'].shape == (1380, 20), '230 cells x six observation variations')
    require(len({r['clip_id'] for r in labels}) == 24 and
            len({r['source_sha256'] for r in labels}) == 24, '24 distinct source videos')
    require({r['person'] for r in labels} == set(PEOPLE), 'Only six public FIT record IDs')
    require(len({r['word_id'] for r in labels}) == 4, 'Four source expressions')
    truth_counts = collections.Counter(s for r in labels for s in [r['right'], r['left']])
    require(dict(truth_counts) == {'M': 142, 'H': 86, 'U': 2}, 'M142 H86 U2 unchanged')
    for r in labels:
        lo, hi = r['native_window']
        require([lo, hi] == [max(0, r['frame'] - 4), r['frame'] + 4], 'Native bounds: ' + r['key'])
        if 'H' in [r['right'], r['left']]:
            require(r['reviewed_unique_frames'] == list(range(lo, hi + 1)),
                    'All actual native times reviewed for hold: ' + r['key'])
    lookup = {}
    for i, r in enumerate(rows):
        key = (r['source_key'], r['view'], bool(r['reverse']))
        require(key not in lookup, 'Unique source/view/order: ' + str(i))
        lookup[key] = i
        src = source[r['source_key'][:-2]]
        target = src['right' if r['side'] == 'R' else 'left']
        require(r['label'] == target and z['y'][i] == {'U': -1, 'H': 0, 'M': 1}[target],
                'Truth alignment: ' + str(i))
        require(r['source_sha256'] == src['source_sha256'] and r['person'] == src['person'] == z['person'][i],
                'Source/record alignment: ' + str(i))
        require(sorted(r['source_frames']) == list(range(src['native_window'][0], src['native_window'][1] + 1)),
                'Actual window set: ' + str(i))
        require(np.flatnonzero(z['exists'][i]).tolist() == r['positions'], 'Boundary existence mask: ' + str(i))
        valid = z['valid'][i] & z['exists'][i] & np.isfinite(z['q'][i]).all(1)
        actual = np.flatnonzero(z['exists'][i]); seen = np.flatnonzero(valid)
        longest = current = 0
        for j in actual:
            current = 0 if valid[j] else current + 1
            longest = max(longest, current)
        eligible = len(seen) >= 4 and len(seen) / len(actual) >= .75 and valid[actual[0]] and valid[actual[-1]] and longest <= 2
        require(bool(z['eligible'][i]) == bool(eligible), 'Independent eligibility: ' + str(i))
        q = z['q'][i, valid].astype(float)
        maximum = float(np.sqrt(((q[:, None] - q[None, :]) ** 2).sum(-1)).max()) if len(q) else 0.
        require(np.isclose(maximum, z['x'][i, 11], atol=3e-6), 'Independent rule feature: ' + str(i))

    person = z['person']; y = z['y']; eligible = z['eligible'].astype(bool)
    view = np.array([r['view'] for r in rows]); reverse = np.array([bool(r['reverse']) for r in rows])
    hand = np.array([r['side'] for r in rows]); source_key = np.array([r['source_key'] for r in rows])
    word = np.array([r['word_id'] for r in rows]); kind = np.array([r['kind'] for r in rows])
    for v in VIEWS:
        for rev in (False, True):
            take = (view == v) & (reverse == rev)
            require(int(take.sum()) == 230 and len(set(source_key[take])) == 230, 'Source denominator ' + str((v, rev)))
            require(int((take & (y >= 0)).sum()) == 228, 'Known denominator ' + str((v, rev)))

    fold_audit = []
    fold_rules = {}
    for fold in cfg['folds']:
        pid = fold['person']
        weights = np.load(fold['weights'], allow_pickle=False)['weights']
        metadata = read(fold['normalization_rule']); fit = person != pid
        expected = np.zeros(len(rows), np.float64)
        for value in (0, 1):
            for side in ('R', 'L'):
                for rid in PEOPLE:
                    if rid == pid:
                        continue
                    take = (person == rid) & (hand == side) & (y == value) & eligible
                    sources = set(source_key[take]); require(bool(sources), 'Training stratum available')
                    for skey in sources:
                        selected = take & (source_key == skey)
                        expected[selected] = 1. / (20 * len(sources) * int(selected.sum()))
        require(np.allclose(weights, expected, atol=1e-9, rtol=1e-6), 'Balanced fold weights ' + str(pid))
        tr = weights > 0; te = ~fit
        require(not weights[te | (y < 0) | ~eligible].any(), 'No held/U/unobserved weight ' + str(pid))
        require(set(r['source_sha256'] for i, r in enumerate(rows) if tr[i]).isdisjoint(
                r['source_sha256'] for i, r in enumerate(rows) if te[i]), 'Source SHA split ' + str(pid))
        a = z['x'][tr].astype(float); norm = metadata['normalization']
        require(np.allclose(a.mean(0), norm['mean'], atol=1e-10) and
                np.allclose(np.maximum(a.std(0), 1e-4), norm['std'], atol=1e-10), 'FIT-only normalization ' + str(pid))
        score = z['x'][:, 11]; unique = np.unique(score[tr]).astype(float)
        thresholds = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2, np.nextafter(unique[-1], np.inf)]
        accuracies = np.array([float(weights @ ((score > t) == y)) for t in thresholds])
        optimal = np.flatnonzero(np.isclose(accuracies, accuracies.max(), atol=1e-10, rtol=0))
        threshold = float(thresholds[optimal[len(optimal) // 2]])
        require(threshold == metadata['rule']['threshold'], 'FIT-only optimal rule threshold ' + str(pid))
        require(metadata['training_record_ids'] == [p for p in PEOPLE if p != pid] and metadata['held_record_id'] == pid,
                'Fold record contract ' + str(pid))
        fold_rules[pid] = threshold
        fold_audit.append(dict(held_id=pid, trained_rows=int(tr.sum()), weights_sum=float(weights.sum()), rule_threshold=threshold))

    completion = read(U / 'PHYSICAL_TRAINING_COMPLETE.json'); pin(U / 'PHYSICAL_TRAINING_COMPLETE.json')
    require(completion['frozen_sha256'] == EXPECTED_FROZEN and len(completion['runs']) == 18, '18-run completion contract')
    runs = {(r['person'], r['seed']): r for r in completion['runs']}
    require(set(runs) == {(p, s) for p in PEOPLE for s in SEEDS}, 'Exactly one run per held ID and seed')
    run_dirs = {p.name for p in (U / 'physical_runs').iterdir() if p.is_dir()}
    require(run_dirs == {f'{p}_{s}' for p in PEOPLE for s in SEEDS}, 'No additional model run directories')
    logits = {seed: np.full(len(rows), np.nan) for seed in SEEDS}
    rules = {seed: np.full(len(rows), -1, dtype=np.int8) for seed in SEEDS}
    writes = {seed: np.zeros(len(rows), np.uint8) for seed in SEEDS}
    model_checks = []
    import torch  # deserialize tensors only; no model object/forward is constructed
    expected_shapes = {'0.weight': (16, 20), '0.bias': (16,), '2.weight': (1, 16), '2.bias': (1,)}
    for (pid, seed), r in sorted(runs.items()):
        directory = U / 'physical_runs' / f'{pid}_{seed}'
        complete = read(directory / 'COMPLETE.json'); started = read(directory / 'STARTED.json')
        require(complete == r, 'Run completion agreement ' + str((pid, seed)))
        require(started['person'] == pid and started['seed'] == seed and started['frozen_sha256'] == EXPECTED_FROZEN,
                'Run start contract ' + str((pid, seed)))
        pin(directory / 'COMPLETE.json'); pin(directory / 'STARTED.json')
        require(r['frozen_sha256'] == EXPECTED_FROZEN and math.isfinite(r['final_training_loss']) and r['training_seconds'] > 0,
                'Finite completed run ' + str((pid, seed)))
        pin(r['model'], r['model_sha256']); pin(r['prediction'], r['prediction_sha256'])
        ck = torch.load(r['model'], map_location='cpu', weights_only=True)
        require(ck['task'] == cfg['task'] and ck['person'] == pid and ck['seed'] == seed, 'Checkpoint identity ' + str((pid, seed)))
        state = ck['state_dict']
        require({k: tuple(v.shape) for k, v in state.items()} == expected_shapes and
                sum(v.numel() for v in state.values()) == 353 and all(torch.isfinite(v).all().item() for v in state.values()),
                'Checkpoint architecture and finite weights ' + str((pid, seed)))
        norm = read(U / 'physical_folds' / str(pid) / 'NORMALIZATION_RULE.json')['normalization']
        require(np.array_equal(ck['mean'].numpy(), np.asarray(norm['mean'], np.float32)) and
                np.array_equal(ck['std'].numpy(), np.asarray(norm['std'], np.float32)), 'Checkpoint normalization ' + str((pid, seed)))
        with np.load(r['prediction'], allow_pickle=False) as f:
            pred = {k: f[k].copy() for k in f.files}
        take = person == pid
        require(pred['logits'].shape == (1380,) and np.isfinite(pred['logits']).all(), 'Saved logits shape/finiteness ' + str((pid, seed)))
        require(np.array_equal(pred['held'], take) and np.array_equal(pred['eligible'], eligible), 'Saved evaluation masks ' + str((pid, seed)))
        expected_rule = z['x'][:, 11] > fold_rules[pid]
        require(np.array_equal(pred['rule_prediction'], expected_rule), 'Saved simple rule matches FIT threshold ' + str((pid, seed)))
        logits[seed][take] = pred['logits'][take]
        rules[seed][take] = pred['rule_prediction'][take]
        writes[seed][take] += 1
        model_checks.append(dict(person=pid, seed=seed, model_sha256=r['model_sha256'], prediction_sha256=r['prediction_sha256']))
    for seed in SEEDS:
        require(np.isfinite(logits[seed]).all() and (writes[seed] == 1).all(), 'Each cell evaluated once by its held-ID model ' + str(seed))

    def score(take, pred):
        known = take & (y >= 0); available = known & eligible
        correct = available & (pred == y)
        cm = [[int((available & (y == t) & (pred == p)).sum()) for p in (0, 1)] for t in (0, 1)]
        f1 = []
        for cls in (0, 1):
            tp = int((available & (y == cls) & (pred == cls)).sum())
            fp = int((available & (y != cls) & (pred == cls)).sum())
            # Abstentions are unresolved positives, never removed from known truth.
            fn = int((known & (y == cls) & (~eligible | (pred != cls))).sum())
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)
        n = int(known.sum()); observed = int(available.sum()); good = int(correct.sum())
        unknown = take & (y < 0)
        return dict(all_targets=int(take.sum()), known=n, truth_U=int(unknown.sum()), correct=good,
                    wrong_observed=observed-good, unresolved=n-observed, observed_known=observed,
                    accuracy_all_known=good/n if n else None, accuracy_observed=good/observed if observed else None,
                    observation_rate=observed/n if n else None, confusion_HM=cm, F1_H=f1[0], F1_M=f1[1],
                    truth_U_outputs={'H': int((unknown & eligible & (pred == 0)).sum()),
                                     'M': int((unknown & eligible & (pred == 1)).sum()),
                                     'U': int((unknown & ~eligible).sum())})

    strata = {'all': np.ones(len(rows), bool), 'M': y == 1, 'H': y == 0, 'R': hand == 'R', 'L': hand == 'L'}
    for side in ('R', 'L'):
        for label, value in [('M', 1), ('H', 0)]:
            strata[side+'/'+label] = (hand == side) & (y == value)
    cross = (word == 'WORD1567') & np.char.startswith(kind.astype(str), 'main_')
    short_keys = set(adoption['short_change']['source_hand_keys'])
    require(len(short_keys) == 6 and len({k[:-2] for k in short_keys}) == 4, 'Fixed short-change6 cells/4windows')
    short = np.isin(source_key, list(short_keys))
    require(int(short.sum()) == 36 and (y[short] == 1).all(), 'Short movement labels retained in all six conditions')
    gates = []; summaries = []; diagnostics = []; short_cells = []; primary = []
    for seed in SEEDS:
        neural = (logits[seed] >= 0).astype(np.int8); rule = rules[seed]
        for v in VIEWS:
            for rev in (False, True):
                condition = (view == v) & (reverse == rev)
                for name, st in strata.items():
                    take = condition & st; a = score(take, neural); b = score(take, rule)
                    item = dict(seed=seed, view=v, reverse=rev, group='preservation', stratum=name, neural=a, rule=b)
                    summaries.append(item)
                    if a['known']:
                        gates.append(dict(seed=seed, view=v, reverse=rev, group='preservation', stratum=name,
                                          known=a['known'], neural=a['correct'], rule=b['correct'], pass_=a['correct'] >= b['correct']))
                    if v == 'full' and not rev and name == 'all':
                        primary.append(item)
                for name in ('all', 'R', 'L'):
                    a = score(condition & cross & strata[name], neural); b = score(condition & cross & strata[name], rule)
                    summaries.append(dict(seed=seed, view=v, reverse=rev, group='cross_cycle', stratum=name, neural=a, rule=b))
                    if a['known']:
                        gates.append(dict(seed=seed, view=v, reverse=rev, group='cross_cycle', stratum=name,
                                          known=a['known'], neural=a['correct'], rule=b['correct'], pass_=a['correct'] >= b['correct']))
                a = score(condition & short, neural); b = score(condition & short, rule)
                require(a['known'] == 6, 'Short-change denominator6')
                summaries.append(dict(seed=seed, view=v, reverse=rev, group='short_change', stratum='all', neural=a, rule=b))
                gates.append(dict(seed=seed, view=v, reverse=rev, group='short_change', stratum='all', known=6,
                                  neural=a['correct'], rule=b['correct'], pass_=a['correct'] >= b['correct']))
                for i in np.flatnonzero(condition & short):
                    short_cells.append(dict(seed=seed, view=v, reverse=rev, source_hand_key=str(source_key[i]), truth='M',
                                            eligible=bool(eligible[i]), neural=('M' if neural[i] else 'H') if eligible[i] else 'U',
                                            rule=('M' if rule[i] else 'H') if eligible[i] else 'U'))
                for group, values in [('record_id', person), ('expression', word), ('source_position', kind)]:
                    for val in sorted(set(values.tolist())):
                        take = condition & (values == val)
                        diagnostics.append(dict(seed=seed, view=v, reverse=rev, group=group, value=val,
                                                neural=score(take, neural), rule=score(take, rule)))
    failures = [g for g in gates if not g['pass_']]
    gains = [g for g in gates if g['neural'] > g['rule']]
    decision = ('FAIL_NOT_ADOPTED' if failures else
                'PASS_CANDIDATE_FOR_LINKING' if gains else 'NO_DEMONSTRATED_NEURAL_ADVANTAGE')
    result = dict(status='PASS', verification_meaning='Saved input/output/hash arithmetic verified; this status does not mean model adoption.',
                  at=stamp(), reviewer='Separate Codex subagent saved-result verifier; not independent human ground truth',
                  verifier_sha256=sha(__file__), frozen_sha256=EXPECTED_FROZEN, adoption_contract_sha256=EXPECTED_ADOPTION,
                  checks=len(checks), all_checks=checks, runs=18, source_windows=115, source_hand_cells=230, known_source_cells=228,
                  unknown_truth_cells=2, run_audit=model_checks, fold_audit=fold_audit, primary=primary,
                  gate_checks=len(gates), gate_failures=failures, strict_gain_gate_count=len(gains), decision=decision,
                  summaries=summaries, detailed_diagnostics=diagnostics, short_change_cells=short_cells,
                  limitations=['All six record IDs were exposed in earlier research; this is new-head held-record diagnostic, not untouched final-user evaluation.',
                               'Physical wrist displacement within a short window is not clause membership, central-frame motion, exact onset, completion or sign correctness.',
                               'Source annotations are Codex RGB judgments; no independent Korean Sign Language expert validation is implied.',
                               'Transforms share the same source videos. They are not additional people or independent examples.',
                               'Actual crop can lose observation coverage. Unknown eligibility remains unresolved within all known denominators.',
                               'No old C6/C9 different-task score was subtracted. No existing web/recognition/feedback model is promoted.',
                               'Checkpoint and saved logits hashes were verified; model predictions were not regenerated.'],
                  pins=pins, model_forward_calls=0, training_runs_started=0, production_promotion=False)
    with OUTPUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status=result['status'], decision=decision, checks=len(checks), gate_failures=len(failures),
                          primary=[dict(seed=x['seed'], neural=x['neural']['correct'], rule=x['rule']['correct'], known=x['neural']['known']) for x in primary],
                          output=str(OUTPUT)), ensure_ascii=False))


if __name__ == '__main__':
    main()
