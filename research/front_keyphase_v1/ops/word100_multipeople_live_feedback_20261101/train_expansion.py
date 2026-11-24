"""Fixed matched-three versus expanded-nine WindowHead7 development comparison.

No extraction, download, backbone update, or production write. `prepare` reads
the approved public bank/labels only. `run` also requires root TRAIN_READY.json.
Every run is final-epoch only and refuses existing/partial run directories.
`infer_candidate(arrays)` uses expanded seed42 only after the saved adoption gate;
it returns [T,7], with NaN at unsupported temporal centers (no padding).
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.dont_write_bytecode = True
U = Path(__file__).resolve().parent
O = U.parent
OLD = O/'word100_stage_feedback_cycle_20261031'
SEM = O/'word100_semantic_feature_cycle_20261101'
TARGETS = ('R_index_only', 'R_open_four', 'L_open_four', 'R_closed_four',
           'L_closed_four', 'R_higher', 'L_higher')
CONDITIONS = ('matched_three_people', 'expanded_nine_people')
SEEDS = (42, 43, 44)
EPOCHS = 160
_CANDIDATES = {}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(U):
        raise ValueError('Research-unit output only')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def save_array(path, value):
    with Path(path).open('xb') as f:
        np.save(f, value, allow_pickle=False)


def array_fingerprint(arrays):
    """Content identity, not filenames or target words, gates saved-center reuse."""
    h = hashlib.sha256()
    for name in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us'):
        v = np.ascontiguousarray(arrays[name])
        h.update(name.encode()); h.update(v.dtype.str.encode()); h.update(str(v.shape).encode())
        h.update(v.tobytes())
    return h.hexdigest()


def comparison():
    name = '_multipeople_frozen_window_comparison'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SEM/'comparison.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def utc():
    return datetime.now(timezone.utc).isoformat()


def check_pins(pins):
    for p, expected in pins.items():
        if sha(p) != expected:
            raise ValueError('Changed pinned input: ' + p)


def contract():
    f = read(U/'FROZEN.json'); k = f['learning']
    expected = dict(targets=list(TARGETS), conditions=list(CONDITIONS), seeds=list(SEEDS),
                    epochs=EPOCHS, lr=.001, weight_decay=.01, threshold=.5)
    for name, value in expected.items():
        if k.get(name) != value:
            raise ValueError('Unexpected frozen learning field: ' + name)
    if f['fit_person_ids'] != list(range(1, 10)) or f['development_person_ids'] != [11, 12, 13]:
        raise ValueError('Frozen person split differs')
    return f


def assemble_dataset():
    """Original180 FIT rows verbatim +72 expansion +36 common development."""
    f = contract()
    bank = read(U/'PUBLIC_INPUTS.json')
    labels = read(U/'data_audit/LABELS.json')
    original = read(SEM/'DATASET.json')
    if tuple(original['selected_targets']) != TARGETS or tuple(labels['targets']) != TARGETS:
        raise ValueError('Label target order mismatch')
    if bank['feature_names'] != original['feature_names']:
        raise ValueError('69 feature order mismatch')
    source = {r['clip_id']: r for r in bank['rows']}
    if len(source) != 48 or len(labels['rows']) != 144:
        raise ValueError('Expected48 public clips and144 center labels')
    if bank['frozen_sha256'] != sha(U/'FROZEN.json'):
        raise ValueError('Public bank uses another freeze')
    rows = []
    for r in original['rows']:
        if r['role'] == 'fit':
            q = dict(r, sample_id='old:' + r['sample_id'], cohort='original_fit')
            if r['person'] not in [1, 2, 3]:
                raise ValueError('Original FIT person differs')
            rows.append(q)
    if len(rows) != 180:
        raise ValueError('Original180 FIT must remain intact')
    seen = set(); label_counts = Counter()
    for r in labels['rows']:
        b = source[r['clip_id']]
        if (r['person'] != b['person'] or r['word_id'] != b['word_id']
                or r['role'] != b['role'] or r['word_id'] not in f['word_ids']):
            raise ValueError('Public label/input identity mismatch')
        key = (r['clip_id'], int(r['center_frame']))
        if key in seen:
            raise ValueError('Duplicate labeled center')
        seen.add(key); label_counts[r['clip_id']] += 1
        y, m = r['y'], r['target_mask']
        if (len(y) != 7 or len(m) != 7 or not all(isinstance(v, bool) for v in m)
                or any(v not in [0, 1] for v, known in zip(y, m) if known)):
            raise ValueError('Explicit7 binary labels and boolean unknown mask required')
        if r['person'] <= 3:
            continue  # Exact old target examples already included, no double exposure.
        role = 'development' if r['person'] in [11, 12, 13] else 'fit'
        if role != r['role'] or (role == 'fit' and r['person'] not in range(4, 10)):
            raise ValueError('Unexpected extra person')
        rows.append(dict(r, sample_id=f"new:{r['clip_id']}:{r['center_frame']}",
                         input=b['input'], input_sha256=b['input_sha256'],
                         cohort='development' if role == 'development' else 'expansion'))
    if set(label_counts.values()) != {3} or len(label_counts) != 48:
        raise ValueError('Every bank clip must have three fixed labeled centers')
    if Counter(r['cohort'] for r in rows) != {'original_fit': 180, 'expansion': 72, 'development': 36}:
        raise ValueError('Fixed180/72/36 denominator differs')
    if Counter((r['word_id'], r['person']) for r in rows if r['cohort'] != 'original_fit') != {
            (w, p): 3 for w in f['word_ids'] for p in list(range(4, 10))+[11, 12, 13]}:
        raise ValueError('Target-expression/person coverage differs')
    fit_people = {r['person'] for r in rows if r['role'] == 'fit'}
    dev_people = {r['person'] for r in rows if r['role'] == 'development'}
    if fit_people & dev_people:
        raise ValueError('Person leakage')
    clips = {}
    for r in rows:
        clips.setdefault(r['clip_id'], set()).add(r['role'])
    if any(len(v) != 1 for v in clips.values()):
        raise ValueError('Clip leakage')
    pins = {str(p): sha(p) for p in [U/'FROZEN.json', U/'PUBLIC_INPUTS.json',
            U/'data_audit/LABELS.json', SEM/'DATASET.json', SEM/'comparison.py', Path(__file__)]}
    pins.update({r['input']: r['input_sha256'] for r in rows})
    return dict(rows=rows, targets=list(TARGETS), feature_names=original['feature_names'],
                source_pins=pins, historical_exposure=f['historical_exposure'])


def load_arrays(info):
    c = comparison(); raw = {}; windows = []
    for row in info['rows']:
        key = row['clip_id']
        if key not in raw:
            if sha(row['input']) != row['input_sha256']:
                raise ValueError('Public input changed: ' + key)
            with np.load(row['input'], allow_pickle=False) as z:
                a = {k: z[k] for k in ('x', 'mask', 'rgb', 'frame_id', 'timestamp_us')}
            c._validate_raw(a); raw[key] = a
        a = raw[key]; ix = c._indices(row, a)
        windows.append({k: a[k][ix] for k in ('x', 'mask', 'rgb')})
    return raw, windows


def fit_indices(rows, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown condition')
    return [i for i, r in enumerate(rows) if r['role'] == 'fit' and
            (condition == CONDITIONS[1] or r['cohort'] == 'original_fit')]


def fit_statistics(windows, rows, indices):
    if not indices or any(rows[i]['role'] != 'fit' for i in indices):
        raise ValueError('Only FIT windows may fit normalization')
    x = np.concatenate([windows[i]['x'] for i in indices])
    m = np.concatenate([windows[i]['mask'] for i in indices])
    z = np.concatenate([windows[i]['rgb'] for i in indices])
    pm = [float(x[m[:, j], j].mean()) if m[:, j].any() else 0. for j in range(69)]
    ps = [max(float(x[m[:, j], j].std()), .001) if m[:, j].any() else 1. for j in range(69)]
    return dict(pose_mean=pm, pose_std=ps, rgb_mean=z.mean(0).tolist(),
                rgb_std=np.maximum(z.std(0), .001).tolist(),
                scope='Only fixed FIT13-frame windows; overlapping frame occurrences retained',
                fit_sample_ids=[rows[i]['sample_id'] for i in indices])


def label_arrays(rows):
    y = np.asarray([r['y'] for r in rows], np.float32)
    m = np.asarray([r['target_mask'] for r in rows], bool)
    if y.shape != (len(rows), 7) or m.shape != y.shape or not np.isin(y[m], [0, 1]).all():
        raise ValueError('Invalid known target labels')
    return y, m


def loss_weights(y, mask, rows, indices):
    """FIT-class balance and equal known target exposure within each clip."""
    if not indices or any(rows[i]['role'] != 'fit' for i in indices):
        raise ValueError('FIT only class weighting')
    yy, mm = y[indices], mask[indices]
    counts = np.stack([((yy == k) & mm).sum(0) for k in (0, 1)])
    if (counts <= 0).any():
        raise ValueError('Both FIT classes required for each of the frozen seven targets')
    weights = (counts.sum(0)[None] / (2*counts)).astype(np.float32)
    events = np.zeros_like(yy, np.float32)
    for clip in {rows[i]['clip_id'] for i in indices}:
        ix = [j for j, i in enumerate(indices) if rows[i]['clip_id'] == clip]
        events[ix] = 1/np.maximum(mm[ix].sum(0), 1)
    return weights, events, counts


def masked_loss(logits, y, mask, class_weights, event_weights):
    # Same loss normalization as the previous completed WindowHead comparison.
    import torch
    safe = torch.where(mask, y, torch.zeros_like(y))
    raw = torch.nn.functional.binary_cross_entropy_with_logits(logits, safe, reduction='none')
    w = torch.where(safe > .5, class_weights[1], class_weights[0])*event_weights
    return torch.where(mask, raw*w, torch.zeros_like(raw)).sum()/event_weights[mask].sum().clamp_min(1)


def metrics(probabilities, rows):
    y, mask = label_arrays(rows); p = np.asarray(probabilities)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Finite probabilities required on all evaluated centers')
    pred = p >= .5; items = []
    for j, name in enumerate(TARGETS):
        known = mask[:, j]; pos = known & (y[:, j] == 1); neg = known & (y[:, j] == 0)
        correct = pred[:, j] == y[:, j]
        pa = float(correct[pos].mean()) if pos.any() else None
        na = float(correct[neg].mean()) if neg.any() else None
        items.append(dict(target=name, possible_cells=len(rows), known=int(known.sum()),
                          unknown=int((~known).sum()), positive=int(pos.sum()), negative=int(neg.sum()),
                          correct=int(correct[known].sum()),
                          accuracy=float(correct[known].mean()) if known.any() else None,
                          positive_accuracy=pa, negative_accuracy=na,
                          balanced_accuracy=(pa+na)/2 if pa is not None and na is not None else None))
    values = [q['accuracy'] for q in items if q['accuracy'] is not None]
    old = [q['accuracy'] for q in items[:3] if q['accuracy'] is not None]
    return dict(rows=len(rows), possible_cells=len(rows)*7, known_cells=int(mask.sum()),
                unknown_cells=int((~mask).sum()), per_target=items,
                macro_supported_target_accuracy=float(np.mean(values)) if values else None,
                old_three_macro_accuracy=float(np.mean(old)) if old else None,
                score_scope='Direct binary attributes at labeled centers; not whole handshape or educational accuracy')


def breakdown(p, rows):
    out = {}
    for key in ('word_id', 'person'):
        out[key] = {}
        for value in sorted({r[key] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r[key] == value]
            out[key][str(value)] = metrics(p[ix], [rows[i] for i in ix])
    return out


def prepare():
    for name in ('TRAIN_DATASET.json', 'TRAIN_PREPARED.json', 'training_prepared'):
        if (U/name).exists():
            raise FileExistsError(U/name)
    info = assemble_dataset(); _, windows = load_arrays(info)
    y, mask = label_arrays(info['rows']); conditions = {}
    for name in CONDITIONS:
        ids = fit_indices(info['rows'], name)
        if len(ids) != (180 if name == CONDITIONS[0] else 252):
            raise ValueError('FIT count mismatch')
        stats = fit_statistics(windows, info['rows'], ids)
        weights, event, counts = loss_weights(y, mask, info['rows'], ids)
        path = U/'training_prepared'/f'{name}_norm.json'; save(path, stats)
        conditions[name] = dict(fit_rows=len(ids), fit_clips=len({info['rows'][i]['clip_id'] for i in ids}),
            fit_people=sorted({info['rows'][i]['person'] for i in ids}), fit_sample_ids=stats['fit_sample_ids'],
            normalization=str(path), normalization_sha256=sha(path), class_counts_zero_one=counts.tolist(),
            class_weights_zero_one=weights.tolist(), equal_clip_target_event_sum=float(event[mask[ids]].sum()))
    protected = {}
    paths = [OLD/'NORMALIZATION.json', OLD/'LEARNING_INPUTS.json', OLD/'learning.py',
             SEM/'NORMALIZATION.json', SEM/'FROZEN.json']
    for seed in SEEDS:
        paths += [OLD/'runs'/f'pose_plus_signrep_{seed}'/n for n in ('head.pt', 'RUN.json')]
        paths += [SEM/'runs'/f'protected_shape_split_features_{seed}'/n for n in ('head.pt', 'COMPLETE.json')]
    protected.update({str(p): sha(p) for p in paths})
    save(U/'TRAIN_DATASET.json', info)
    save(U/'TRAIN_PREPARED.json', dict(status='PREPARED_NO_TRAINING_OR_MODEL_INFERENCE',
        dataset_sha256=sha(U/'TRAIN_DATASET.json'), frozen_sha256=sha(U/'FROZEN.json'),
        conditions=conditions, development_rows=36, targets=list(TARGETS),
        protected_pins=protected, source_pins=info['source_pins'],
        loss='FIT class-balanced masked BCE; equal per clip/target exposure; global known event denominator',
        optimizer='AdamW .001/.01, full FIT batch, gradient norm clip1, final epoch160',
        validation_instruction='Root must provide DATASET_VALIDATION.json PASS and explicit TRAIN_READY.json'))


def validate_ready():
    contract(); ready = read(U/'TRAIN_READY.json'); prepared = read(U/'TRAIN_PREPARED.json')
    validation = read(U/'DATASET_VALIDATION.json')
    if (not str(ready.get('status', '')).startswith('PASS') or ready.get('allow_training') is not True
            or not str(validation.get('status', '')).startswith('PASS')):
        raise ValueError('Explicit root approval and passing dataset validation required')
    required = {'frozen_sha256': U/'FROZEN.json', 'dataset_sha256': U/'TRAIN_DATASET.json',
                'prepared_sha256': U/'TRAIN_PREPARED.json', 'code_sha256': Path(__file__),
                'labels_sha256': U/'data_audit/LABELS.json', 'public_inputs_sha256': U/'PUBLIC_INPUTS.json',
                'validation_sha256': U/'DATASET_VALIDATION.json'}
    for key, path in required.items():
        if ready.get(key) != sha(path):
            raise ValueError('TRAIN_READY missing/changed pin: ' + key)
    if prepared['dataset_sha256'] != sha(U/'TRAIN_DATASET.json') or prepared['frozen_sha256'] != sha(U/'FROZEN.json'):
        raise ValueError('Prepared freeze changed')
    check_pins(prepared['source_pins']); check_pins(prepared['protected_pins'])
    for info in prepared['conditions'].values():
        if sha(info['normalization']) != info['normalization_sha256']:
            raise ValueError('Prepared normalization changed')
    return prepared


def normalized_windows(windows, stats):
    import torch
    norms = [comparison()._normalize(w, stats) for w in windows]
    return torch.from_numpy(np.stack([w[0] for w in norms])), torch.from_numpy(np.stack([w[1] for w in norms]))


def protected_baseline(raw, rows, seed, destination):
    """Twelve new full clips once/seed; retain all posteriors for bank reuse."""
    import torch
    c = comparison(); old = c.load_frozen(seed)
    directory = SEM/'runs'/f'protected_shape_split_features_{seed}'
    manifest = read(directory/'COMPLETE.json')
    if (sha(directory/'head.pt') != manifest['checkpoint_sha256']
            or sha(SEM/'NORMALIZATION.json') != manifest['normalization_sha256']):
        raise ValueError('Protected new4 source changed')
    newer = c.WindowHead(4)
    newer.load_state_dict(torch.load(directory/'head.pt', map_location='cpu', weights_only=True), strict=True)
    newer.eval(); newer.requires_grad_(False)
    old_stats = read(OLD/'NORMALIZATION.json'); new_stats = read(SEM/'NORMALIZATION.json')
    old_names = read(OLD/'LEARNING_INPUTS.json')['targets']; cols = [old_names.index(t) for t in TARGETS[:3]]
    out = np.empty((len(rows), 7), np.float32); full_outputs = []; total_centers = 0
    with torch.inference_mode():
        for clip in sorted({r['clip_id'] for r in rows}):
            a = raw[clip]; xx, zz = c._normalize(a, old_stats)
            pred = old(torch.from_numpy(xx[None]), torch.from_numpy(zz[None]), torch.ones(1, len(xx))).sigmoid().numpy()[0]
            nx, nz = c._normalize(a, new_stats)
            probabilities = np.full((len(xx), 7), np.nan, np.float32)
            probabilities[:, :3] = pred[:, cols]
            centers = valid_centers(a['frame_id']); total_centers += len(centers)
            for start in range(0, len(centers), 64):
                ix = centers[start:start+64]
                wx = torch.from_numpy(np.stack([nx[i-6:i+7] for i in ix]))
                wz = torch.from_numpy(np.stack([nz[i-6:i+7] for i in ix]))
                probabilities[ix, 3:] = newer(wx, wz).sigmoid().numpy()
            for i, r in enumerate(rows):
                if r['clip_id'] == clip:
                    center = c._indices(r, a)[6]; out[i] = probabilities[center]
            folder = U/'baseline_probabilities' if seed == 42 else destination/'full'
            folder.mkdir(parents=True, exist_ok=True); path = folder/f'{clip}.npz'
            fingerprint = array_fingerprint(a)
            with path.open('xb') as f:
                np.savez_compressed(f, probabilities=probabilities, frame_id=a['frame_id'],
                    timestamp_us=a['timestamp_us'], posterior_available=np.isfinite(probabilities),
                    targets=np.asarray(TARGETS), seed=np.asarray(seed), array_fingerprint=np.asarray(fingerprint))
            full_outputs.append(dict(clip_id=clip, path=str(path), sha256=sha(path),
                                     array_fingerprint=fingerprint, new4_centers=len(centers)))
    return out, full_outputs, total_centers


def run():
    import torch
    prepared = validate_ready()
    if (U/'trainruns').exists() or (U/'TRAINING_REPORT.json').exists() or (U/'TRAIN_STARTED.json').exists():
        raise FileExistsError('Completed or partial training must not be rerun')
    info = read(U/'TRAIN_DATASET.json'); rows = info['rows']; raw, windows = load_arrays(info)
    y, mask = label_arrays(rows); dev = [i for i, r in enumerate(rows) if r['role'] == 'development']
    dr = [rows[i] for i in dev]
    if len(dev) != 36 or len({r['clip_id'] for r in dr}) != 12:
        raise ValueError('Expected36 centers/12 development clips')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    save(U/'TRAIN_STARTED.json', dict(started=utc(), ready_sha256=sha(U/'TRAIN_READY.json'),
                                    code_sha256=sha(__file__), epoch_selection=160))
    results = []
    try:
        for condition in CONDITIONS:
            fit = fit_indices(rows, condition); stats = read(prepared['conditions'][condition]['normalization'])
            xx, zz = normalized_windows(windows, stats)
            weights, events, counts = loss_weights(y, mask, rows, fit)
            fy, fm = torch.from_numpy(y[fit]), torch.from_numpy(mask[fit])
            cw, ew = torch.from_numpy(weights), torch.from_numpy(events)
            for seed in SEEDS:
                dest = U/'trainruns'/f'{condition}_{seed}'; dest.mkdir(parents=True, exist_ok=False)
                save(dest/'norm.json', stats)
                torch.manual_seed(seed); np.random.seed(seed)
                model = comparison().WindowHead(7)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                tic = time.perf_counter(); losses = []
                with (dest/'EXPOSURES.jsonl').open('x', encoding='utf8') as exposure:
                    for epoch in range(1, EPOCHS+1):
                        model.train(); optimizer.zero_grad()
                        loss = masked_loss(model(xx[fit], zz[fit]), fy, fm, cw, ew)
                        if not torch.isfinite(loss):
                            raise FloatingPointError('Nonfinite training loss')
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                        optimizer.step(); losses.append(float(loss.detach()))
                        exposure.write(json.dumps(dict(epoch=epoch, loss=losses[-1], fit_rows=len(fit),
                                                       fit_sample_ids=stats['fit_sample_ids'], known_target_cells=int(fm.sum())))+'\n')
                        exposure.flush()
                        if epoch % 40 == 0:
                            print(json.dumps(dict(condition=condition, seed=seed, epoch=epoch, loss=losses[-1])), flush=True)
                model.eval()
                with torch.inference_mode():
                    p = model(xx[dev], zz[dev]).sigmoid().numpy()
                with (dest/'model.pt').open('xb') as f:
                    torch.save(model.state_dict(), f)
                save_array(dest/'probs.npy', p)
                result = dict(condition=condition, seed=seed, epochs=EPOCHS, selected_epoch=EPOCHS,
                    fit_rows=len(fit), fit_people=prepared['conditions'][condition]['fit_people'],
                    fit_sample_ids=stats['fit_sample_ids'], development_sample_ids=[r['sample_id'] for r in dr],
                    trainable_parameters=sum(v.numel() for v in model.parameters()),
                    final_loss=losses[-1], losses=losses, seconds=time.perf_counter()-tic,
                    class_counts_zero_one=counts.tolist(), metrics=metrics(p, dr), breakdown=breakdown(p, dr),
                    checkpoint=str(dest/'model.pt'), checkpoint_sha256=sha(dest/'model.pt'),
                    normalization_sha256=sha(dest/'norm.json'), probabilities_sha256=sha(dest/'probs.npy'),
                    exposures_sha256=sha(dest/'EXPOSURES.jsonl'), code_sha256=sha(__file__),
                    development_center_cache=[dict(clip_id=r['clip_id'], center_frame=r['center_frame'],
                        center_timestamp_us=r['center_timestamp_us'], probability_row=j,
                        array_fingerprint=array_fingerprint(raw[r['clip_id']])) for j,r in enumerate(dr)],
                    ready_sha256=sha(U/'TRAIN_READY.json'), dataset_sha256=sha(U/'TRAIN_DATASET.json'))
                save(dest/'COMPLETE.json', dict(status='COMPLETE_FIXED_FINAL_EPOCH', **result)); results.append(result)
        baseline = []
        for seed in SEEDS:
            dest = U/'trainruns'/f'protected_baseline_{seed}'; dest.mkdir(exist_ok=False)
            tic = time.perf_counter(); p, full_outputs, total_centers = protected_baseline(raw, dr, seed, dest)
            save_array(dest/'probs.npy', p)
            b = dict(seed=seed, seconds=time.perf_counter()-tic, development_sample_ids=[r['sample_id'] for r in dr],
                     new_fullclip_old3_calls=12, new4_window_centers=total_centers, scored_labeled_centers=36,
                     original_completed_predictions_recomputed=0, full_outputs=full_outputs,
                     metrics=metrics(p, dr), breakdown=breakdown(p, dr), probabilities_sha256=sha(dest/'probs.npy'))
            save(dest/'COMPLETE.json', dict(status='COMPLETE_NEW_DEVELOPMENT_BASELINE', **b)); baseline.append(b)
        report = make_report(results, baseline, dr)
        check_pins(prepared['source_pins']); check_pins(prepared['protected_pins'])
        save(U/'TRAINING_REPORT.json', report)
        print(json.dumps(dict(status=report['status'], adoption=report['adoption'])), flush=True)
    except BaseException:
        failure = U/'TRAIN_FIRST_FAILURE.txt'
        if not failure.exists():
            failure.write_text(traceback.format_exc(), encoding='utf8')
        raise


def make_report(results, baseline, dev_rows):
    def distribution(xs):
        if any(v is None for v in xs):
            return dict(values=xs, mean=None, std=None, minimum=None, maximum=None)
        return dict(values=xs, mean=float(np.mean(xs)), std=float(np.std(xs)),
                    minimum=float(np.min(xs)), maximum=float(np.max(xs)))
    aggregates = {}
    for condition in CONDITIONS + ('protected_baseline',):
        rr = baseline if condition == 'protected_baseline' else [r for r in results if r['condition'] == condition]
        rr = sorted(rr, key=lambda q: q['seed'])
        if [r['seed'] for r in rr] != list(SEEDS):
            raise ValueError('All three fixed seeds required')
        aggregates[condition] = dict(
            macro_supported_target_accuracy=distribution([r['metrics']['macro_supported_target_accuracy'] for r in rr]),
            old_three_macro_accuracy=distribution([r['metrics']['old_three_macro_accuracy'] for r in rr]),
            per_target=[dict(target=name,
                accuracy=distribution([r['metrics']['per_target'][j]['accuracy'] for r in rr]),
                balanced_accuracy=distribution([r['metrics']['per_target'][j]['balanced_accuracy'] for r in rr]))
                for j, name in enumerate(TARGETS)])
    a, b, old = (aggregates[k] for k in CONDITIONS+('protected_baseline',))
    gain = b['macro_supported_target_accuracy']['mean'] - a['macro_supported_target_accuracy']['mean']
    old_delta = b['old_three_macro_accuracy']['mean'] - old['old_three_macro_accuracy']['mean']
    adopt = gain > 0 and old_delta >= 0
    return dict(status='COMPLETE_DEVELOPMENT_COMPARISON', targets=list(TARGETS), seeds=list(SEEDS),
        results=results, protected_baseline=baseline, aggregates=aggregates,
        development_sample_ids=[r['sample_id'] for r in dev_rows],
        adoption=dict(allowed=bool(adopt), chosen_seed=42 if adopt else None,
            chosen_condition=CONDITIONS[1] if adopt else 'protected_baseline',
            expanded_minus_matched_macro_accuracy=gain, expanded_minus_baseline_old3_macro_accuracy=old_delta,
            rule='Strict positive three-seed mean supported-target macro accuracy gain and nonnegative old3 macro versus protected baseline; seed42 fixed before scores',
            model_path=str(U/'trainruns'/f'{CONDITIONS[1]}_42/model.pt') if adopt else None),
        training_ready_sha256=sha(U/'TRAIN_READY.json'), code_sha256=sha(__file__),
        historical_exposure=read(U/'FROZEN.json')['historical_exposure'],
        limitations=['Development comparison, not globally unseen people or final linguistic accuracy',
                    'Unknown target cells remain in reported denominators; unlabeled cells are not negative',
                    'One-class targets have no balanced accuracy; primary frozen gate uses ordinary target accuracy',
                    'The matched6 runs compare data exposure; protected baseline is a different system/history',
                    'No best-seed, epoch, threshold or model search; no backbone update'])


def valid_centers(ids):
    ids = np.asarray(ids)
    return [i for i in range(6, len(ids)-6) if np.array_equal(ids[i-6:i+7], np.arange(ids[i]-6, ids[i]+7))]


def _predict_windows(model, arrays, stats, cached_centers=None):
    import torch
    c = comparison(); c._validate_raw(arrays)
    xx, zz = c._normalize(arrays, stats); ids = np.asarray(arrays['frame_id'])
    centers = valid_centers(ids)
    out = np.full((len(ids), 7), np.nan, np.float32)
    cached_centers = cached_centers or {}
    for index, values in cached_centers.items():
        values = np.asarray(values)
        if index not in centers or values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError('Invalid exact saved-center prediction')
        out[index] = values
    centers = [i for i in centers if i not in cached_centers]
    with torch.inference_mode():
        for start in range(0, len(centers), 64):
            ix = centers[start:start+64]
            x = torch.from_numpy(np.stack([xx[i-6:i+7] for i in ix]))
            z = torch.from_numpy(np.stack([zz[i-6:i+7] for i in ix]))
            out[ix] = model(x, z).sigmoid().numpy()
    return out


def infer_candidate(arrays):
    """Adopted expanded seed42 only. No target word or cached video predictions."""
    import torch
    report = read(U/'TRAINING_REPORT.json')
    if report['adoption']['allowed'] is not True or report['adoption']['chosen_seed'] != 42:
        raise ValueError('Expanded candidate not adopted; caller must use protected baseline')
    directory = U/'trainruns'/f'{CONDITIONS[1]}_42'
    manifest = read(directory/'COMPLETE.json')
    if (manifest['selected_epoch'] != 160 or manifest['condition'] != CONDITIONS[1]
            or manifest['seed'] != 42 or report['adoption']['model_path'] != str(directory/'model.pt')):
        raise ValueError('Candidate route changed')
    key = (sha(directory/'model.pt'), sha(directory/'norm.json'))
    if key != (manifest['checkpoint_sha256'], manifest['normalization_sha256']):
        raise ValueError('Candidate checkpoint/normalization changed')
    if key not in _CANDIDATES:
        model = comparison().WindowHead(7)
        model.load_state_dict(torch.load(directory/'model.pt', map_location='cpu', weights_only=True), strict=True)
        model.eval(); model.requires_grad_(False)
        if sha(directory/'probs.npy') != manifest['probabilities_sha256']:
            raise ValueError('Saved development center probabilities changed')
        cache = {}; probabilities = np.load(directory/'probs.npy', allow_pickle=False)
        for row in manifest['development_center_cache']:
            cache.setdefault(row['array_fingerprint'], []).append((row, probabilities[row['probability_row']]))
        _CANDIDATES[key] = (model, read(directory/'norm.json'), cache)
    model, stats, cache = _CANDIDATES[key]; saved = {}
    for row, probabilities in cache.get(array_fingerprint(arrays), []):
        ix = np.flatnonzero(np.asarray(arrays['frame_id']) == row['center_frame'])
        if len(ix) != 1 or int(arrays['timestamp_us'][ix[0]]) != row['center_timestamp_us']:
            raise ValueError('Exact center cache source identity mismatch')
        saved[int(ix[0])] = probabilities
    return _predict_windows(model, arrays, stats, saved)


def selftest():
    """Synthetic only: no real checkpoint, label, video or public prediction."""
    import torch
    torch.set_num_threads(1); torch.manual_seed(602)
    checks = []
    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)
    def reject(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name); return
        raise AssertionError(name)
    rows = [dict(role='fit', cohort='original_fit', sample_id=str(i), clip_id=str(i//2),
                 y=[i%2]*7, target_mask=[True]*7) for i in range(4)]
    rows[0]['target_mask'][0] = False
    y, m = label_arrays(rows); weights, events, counts = loss_weights(y, m, rows, [0, 1, 2, 3])
    check('known class counts exclude unknown', counts[:, 0].tolist() == [1, 2])
    check('class balance counts exact', np.allclose(weights[:, 0], [1.5, .75]))
    for clip in ('0', '1'):
        ix = [i for i, r in enumerate(rows) if r['clip_id'] == clip]
        check('equal clip target event mass ' + clip, np.allclose((events[ix]*m[ix]).sum(0), 1))
    logits = torch.zeros(4, 7, requires_grad=True); ty = torch.from_numpy(y); tm = torch.from_numpy(m)
    cw = torch.from_numpy(weights); ew = torch.from_numpy(events)
    first = masked_loss(logits, ty, tm, cw, ew)
    changed = ty.clone(); changed[~tm] = float('nan')
    second = masked_loss(logits, changed, tm, cw, ew)
    ga = torch.autograd.grad(first, logits, retain_graph=True)[0]
    gb = torch.autograd.grad(second, logits)[0]
    check('unknown label value never changes loss', torch.equal(first, second))
    check('unknown gradient exactly zero', (ga[~tm] == 0).all().item())
    check('unknown gradients invariant', torch.equal(ga, gb))
    manual = np.log(2)*sum(float(weights[int(y[i,j]), j]*events[i,j]) for i in range(4) for j in range(7) if m[i,j])/float(events[m].sum())
    check('loss denominator is original known clip-target event mass', abs(first.item()-manual) < 1e-7)
    all_missing = torch.zeros_like(tm)
    loss = masked_loss(logits, changed, all_missing, cw, ew)
    check('all unknown no training signal', loss.item() == 0)
    reject('class missing cannot be learned silently', lambda: loss_weights(np.zeros_like(y), m, rows, list(range(4))))
    devrow = dict(rows[0], role='development')
    reject('DEV cannot fit class weights', lambda: loss_weights(y, m, [devrow]+rows[1:], [0, 1, 2, 3]))
    windows = [dict(x=np.full((13, 69), i, np.float32), mask=np.ones((13, 69), bool),
                    rgb=np.full((13, 768), i, np.float32)) for i in range(4)]
    stats = fit_statistics(windows, rows, [0, 1]); windows[3]['x'][:] = 1e6; windows[3]['rgb'][:] = -1e6
    check('non-FIT-index extreme cannot change normalization', stats == fit_statistics(windows, rows, [0, 1]))
    reject('DEV cannot fit normalization', lambda: fit_statistics(windows, [devrow]+rows[1:], [0, 1]))
    check('FIT mean correct', stats['pose_mean'] == [.5]*69 and stats['rgb_mean'] == [.5]*768)
    p = np.full((4, 7), .5); result = metrics(p, rows)
    check('all opportunities retained in metric', result['possible_cells'] == 28 and result['known_cells'] == 27 and result['unknown_cells'] == 1)
    check('threshold .5 inclusive', result['per_target'][0]['positive_accuracy'] == 1)
    check('positive/negative balanced accuracy', result['per_target'][0]['balanced_accuracy'] == .5)
    single = [dict(r, y=[0]*7) for r in rows]
    check('one sign balanced accuracy null', all(r['balanced_accuracy'] is None for r in metrics(p, single)['per_target']))
    model = comparison().WindowHead(7); model.eval()
    a = dict(x=np.zeros((20, 69), np.float32), mask=np.ones((20, 69), bool), rgb=np.zeros((20, 768), np.float32),
             frame_id=np.arange(20), timestamp_us=np.arange(20)*33333)
    prob = _predict_windows(model, a, stats)
    check('dense only exact13frame centers', np.isfinite(prob).all(1).tolist() == [False]*6+[True]*8+[False]*6)
    calls = []
    hook = model.register_forward_pre_hook(lambda module, args: calls.append(len(args[0])))
    cached = {6: np.linspace(.1, .7, 7, dtype=np.float32)}
    reused = _predict_windows(model, a, stats, cached)
    hook.remove()
    check('saved exact center is reused without reinference', np.array_equal(reused[6], cached[6]) and sum(calls) == 7)
    before_fingerprint = array_fingerprint(a)
    copy = {k: v.copy() for k,v in a.items()}
    check('array fingerprint ignores object identity', array_fingerprint(copy) == before_fingerprint)
    copy['mask'][0, 0] = False
    check('array fingerprint includes observation mask', array_fingerprint(copy) != before_fingerprint)
    a['frame_id'][10:] += 1; gap = _predict_windows(model, a, stats)
    check('no crossing dropped frame window', not np.isfinite(gap).any())
    x = torch.randn(4, 13, 138); z = torch.randn(4, 13, 768)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loss = masked_loss(model(x, z), ty, tm, cw, ew); loss.backward(); opt.step()
    check('synthetic observed training updates head', any(not torch.equal(v, model.state_dict()[k]) for k,v in before.items()))
    check('head7 output dimensions', model(x,z).shape == (4,7))
    return dict(status='PASS', checks=len(checks), results=checks, code_sha256=sha(__file__),
                public_inference=0, real_training=0, original_artifact_mutations=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'selftest'])
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            prepare()
        elif args.action == 'run':
            run()
        else:
            print(json.dumps(selftest(), ensure_ascii=False, indent=2))
    except BaseException:
        if args.action != 'selftest':
            failure = U/('TRAIN_PREP_FIRST_FAILURE.txt' if args.action == 'prepare' else 'TRAIN_FIRST_FAILURE.txt')
            if not failure.exists():
                failure.write_text(traceback.format_exc(), encoding='utf8')
        raise
