'''Local frozen ST-GCN feature-transfer screening runner (research screening only).

Two arms ('pretrained', 'random') freeze an STGCNEncoder. Each arm fits a zero-initialised
weighted linear head on cached 256-d features with exactly one fixed LBFGS step.
Phase 'train' never reads evaluation arrays or baseline scores. Phase 'evaluate'
verifies every pin in FROZEN_HEADS.json before scoring. This tool makes no adoption
decision, no new-person generalisation claim and no pipeline FPS claim.
'''
import argparse
import hashlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261005
NUM_CLASSES = 93
FEAT_DIM = 256
TOTAL_WEIGHT = 21120
STD_FLOOR = 1e-6
L2 = 1e-4
FROZEN_LBFGS = {'lr': 1.0, 'max_iter': 500, 'max_eval': 625, 'history_size': 20,
                'tolerance_grad': 1e-7, 'tolerance_change': 1e-9, 'line_search_fn': 'strong_wolfe'}
ARMS = ('pretrained', 'random')
TRAIN_GROUPS = ('base', 'pair_context')
EVAL_GROUPS = ('annotation_dev', 'whole_dev', 'public', 'mp', 'ksl36')
EXPECTED_SIZES = {'base': 2122, 'pair_context': 224, 'annotation_dev': 699, 'whole_dev': 74,
                  'public': 32, 'mp': 14, 'ksl36': 36}
EXPECTED_BASE_ZERO_WEIGHT = 19
EXPECTED_KSL_EXPOSED = 2
HAND_IDX = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20]
SELECT = {
    'native67': [0, 16, 15, 5, 2, 6, 3] + [25 + i for i in HAND_IDX] + [46 + i for i in HAND_IDX],
    'mp75': [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53,
             54, 58, 59, 62, 63, 66, 67, 70, 71, 74],
}
# Raw array keys and 21-point hand blocks (left block first in both layouts).
LAYOUT = {
    'native67': {'n': 67, 'xy': 'xy_native', 'valid': 'coordinate_present', 'ts': 'timestamp_nominal_us',
                 'left': (25, 46), 'right': (46, 67)},
    'mp75': {'n': 75, 'xy': 'xy_px', 'valid': 'usable', 'ts': 'timestamp_us',
             'left': (33, 54), 'right': (54, 75)},
}
SHOULDER_POS = (3, 4)  # selection positions of native 5/2 and mp 11/12
TRAIN_FILES = {'log': 'train_log.txt', 'frozen': 'FROZEN_HEADS.json'}
for _a in ARMS:
    TRAIN_FILES['features_' + _a] = 'train_features_%s.npz' % _a
    TRAIN_FILES['head_' + _a] = 'head_%s.npz' % _a
    TRAIN_FILES['summary_' + _a] = 'train_summary_%s.json' % _a
EVAL_FILES = {'log': 'eval_log.txt', 'scores': 'eval_scores.npz', 'metrics': 'eval_metrics.json'}


def recipe():
    return {
        'seed': SEED, 'num_classes': NUM_CLASSES, 'feature_dim': FEAT_DIM, 'total_weight': TOTAL_WEIGHT,
        'std_floor': STD_FLOOR, 'l2_W_only': L2, 'head_init': 'zeros', 'lbfgs': dict(FROZEN_LBFGS),
        'lbfgs_step_calls': 1, 'arms': list(ARMS),
        'loss': 'sum((count/21120)*cross_entropy(logits,y,none)) + 0.5*l2*sum(W**2); no bias reg',
        'standardization': 'float64 weighted mean and population std over positive-weight train rows',
        'selection': {k: list(v) for k, v in SELECT.items()}, 'shoulder_positions': list(SHOULDER_POS),
        'hand_blocks': {k: {'left': list(v['left']), 'right': list(v['right'])} for k, v in LAYOUT.items()},
        'normalization': 'origin=per-frame shoulder midpoint; scale=clip median shoulder distance; '
                         'invalid joints and frames without both shoulders zeroed; output [1,2,T,27] float32',
    }


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, 'rb') as f:
        return sha256_bytes(f.read())


def array_sha(x):
    x = np.ascontiguousarray(x)
    return sha256_bytes(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes())


def read_pinned(path, expected, what):
    with open(path, 'rb') as f:
        data = f.read()
    got = sha256_bytes(data)
    if got != str(expected).lower():
        raise RuntimeError('%s sha256 mismatch for %s: expected %s got %s' % (what, path, expected, got))
    return data


def write_json_x(path, obj):
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(obj, f, indent=1, sort_keys=True, allow_nan=False)
    return sha256_file(path)


def write_npz_x(path, **arrays):
    with open(path, 'xb') as f:
        np.savez(f, **arrays)
    return sha256_file(path)


def refuse_existing(paths):
    found = [p for p in paths if os.path.exists(p)]
    if found:
        raise FileExistsError('refusing to overwrite existing outputs (first attempt preserved): %s' % found)


class Log:
    def __init__(self, path):
        self.f = open(path, 'x', encoding='utf-8')

    def __call__(self, msg):
        line = '%s %s' % (time.strftime('%Y-%m-%dT%H:%M:%S'), msg)
        print(line, flush=True)
        print(line, file=self.f, flush=True)


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _need(ok, where, msg):
    if not ok:
        raise ValueError('%s: %s' % (where, msg))


def row_where(i, r):
    return 'row %d id=%r group=%r' % (i, r.get('id'), r.get('group'))


def validate_manifest(m, strict=True):
    '''Structural validation only (baseline score values are parsed in evaluate).'''
    classes, rows = m.get('classes'), m.get('rows')
    if (not isinstance(classes, list) or len(classes) != NUM_CLASSES or len(set(classes)) != NUM_CLASSES
            or not all(isinstance(c, str) for c in classes)):
        raise ValueError('manifest classes must be %d distinct strings' % NUM_CLASSES)
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest rows must be a non-empty list')
    cidx = {c: i for i, c in enumerate(classes)}
    seen = set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            raise ValueError('row %d is not an object' % i)
        w = row_where(i, r)
        g = r.get('group')
        _need(g in TRAIN_GROUPS + EVAL_GROUPS, w, 'bad group')
        _need(isinstance(r.get('id'), str) and r.get('id') != '', w, 'bad id')
        _need(r.get('label') in cidx and _is_int(r.get('y')) and r.get('y') == cidx.get(r.get('label')),
              w, 'label/y mismatch')
        _need(isinstance(r.get('role'), str) and isinstance(r.get('condition'), str), w, 'role/condition not strings')
        _need(r.get('signer_id') is None or _is_int(r.get('signer_id')), w, 'bad signer_id')
        _need(isinstance(r.get('path'), str) and os.path.isabs(r.get('path')), w, 'path must be absolute')
        h = r.get('sha256')
        _need(isinstance(h, str) and len(h) == 64 and all(ch in '0123456789abcdef' for ch in h.lower()), w, 'bad sha256')
        _need(r.get('layout') in LAYOUT, w, 'bad layout')
        st, en = r.get('start'), r.get('end_exclusive')
        _need(_is_int(st) and _is_int(en) and 0 <= st < en, w, 'bad start/end_exclusive')
        _need(r.get('mask_hand') in (None, 'left', 'right'), w, 'bad mask_hand')
        _need(_is_int(r.get('weight_count')) and r.get('weight_count') >= 0, w, 'bad weight_count')
        _need(isinstance(r.get('exposed'), bool), w, 'exposed must be bool')
        if g in EVAL_GROUPS:
            _need('baseline_scores' in r, w, 'evaluation row lacks baseline_scores')
        else:
            _need(r.get('baseline_scores') is None, w, 'baseline_scores only allowed on evaluation rows')
        _need((g, r['id'], r['condition']) not in seen, w, 'duplicate (group,id,condition)')
        seen.add((g, r['id'], r['condition']))
    train = [r for r in rows if r['group'] in TRAIN_GROUPS]
    agg = np.zeros(NUM_CLASSES)
    for r in train:
        agg[r['y']] += r['weight_count']
    if (agg <= 0).any():
        raise ValueError('train labels without positive aggregate weight: %s'
                         % [classes[c] for c in np.flatnonzero(agg <= 0)])
    if strict:
        sizes = {g: sum(1 for r in rows if r['group'] == g) for g in EXPECTED_SIZES}
        if sizes != EXPECTED_SIZES:
            raise ValueError('group sizes %s differ from expected %s' % (sizes, EXPECTED_SIZES))
        total = sum(r['weight_count'] for r in train)
        if total != TOTAL_WEIGHT:
            raise ValueError('train weight total %d != %d' % (total, TOTAL_WEIGHT))
        nz = sum(1 for r in rows if r['group'] == 'base' and r['weight_count'] == 0)
        if nz != EXPECTED_BASE_ZERO_WEIGHT:
            raise ValueError('base zero-weight rows %d != %d' % (nz, EXPECTED_BASE_ZERO_WEIGHT))
        ne = sum(1 for r in rows if r['group'] == 'ksl36' and r['exposed'])
        if ne != EXPECTED_KSL_EXPOSED:
            raise ValueError('ksl36 exposed rows %d != %d' % (ne, EXPECTED_KSL_EXPOSED))
        for i, r in enumerate(rows):
            s = r['signer_id']
            if r['group'] in TRAIN_GROUPS and s is not None and not 1 <= s <= 9:
                raise ValueError('%s: training signer outside 1..9' % row_where(i, r))
            if r['group'] in ('annotation_dev', 'whole_dev') and s not in (11, 12, 13):
                raise ValueError('%s: native eval signer outside 11..13' % row_where(i, r))
    return classes, rows


def load_clip_arrays(row):
    '''Verify SHA on the exact bytes parsed, validate arrays, slice exactly, apply hand mask.'''
    with open(row['path'], 'rb') as f:
        raw = f.read()
    src_sha = sha256_bytes(raw)
    if src_sha != row['sha256'].lower():
        raise ValueError('source sha256 mismatch: expected %s got %s' % (row['sha256'], src_sha))
    spec = LAYOUT[row['layout']]
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        missing = [k for k in (spec['xy'], spec['valid'], spec['ts']) if k not in z.files]
        if missing:
            raise ValueError('missing arrays %s' % missing)
        xy, valid, ts = z[spec['xy']], z[spec['valid']], z[spec['ts']]
    if xy.ndim != 3 or xy.shape[1:] != (spec['n'], 2) or not np.issubdtype(xy.dtype, np.floating):
        raise ValueError('bad %s shape/dtype %s %s' % (spec['xy'], xy.shape, xy.dtype))
    T = xy.shape[0]
    if valid.shape != (T, spec['n']):
        raise ValueError('bad %s shape %s' % (spec['valid'], valid.shape))
    if valid.dtype != np.bool_:
        if not np.issubdtype(valid.dtype, np.integer) or not np.isin(valid, (0, 1)).all():
            raise ValueError('validity mask must be bool or 0/1 integers')
        valid = valid.astype(bool)
    is_float_ts = np.issubdtype(ts.dtype, np.floating)
    if ts.shape != (T,) or not (is_float_ts or np.issubdtype(ts.dtype, np.integer)):
        raise ValueError('bad %s shape/dtype %s %s' % (spec['ts'], ts.shape, ts.dtype))
    st, en = row['start'], row['end_exclusive']
    if en > T:
        raise ValueError('interval [%d,%d) exceeds %d frames' % (st, en, T))
    xy, valid, ts = xy[st:en], valid[st:en].copy(), ts[st:en]
    tsd = ts.astype(np.float64) if is_float_ts else ts.astype(np.int64)
    if not np.isfinite(tsd.astype(np.float64)).all() or (tsd.size > 1 and not (np.diff(tsd) > 0).all()):
        raise ValueError('timestamps not finite and strictly increasing inside interval')
    if row['mask_hand'] is not None:
        a, b = spec[row['mask_hand']]
        valid[:, a:b] = False
    return xy, valid, ts, src_sha


def normalize_clip(xy, valid, layout):
    sel = SELECT[layout]
    xs = np.asarray(xy, dtype=np.float64)[:, sel, :]
    vs = np.asarray(valid, dtype=bool)[:, sel]
    T = xs.shape[0]
    if T < 1:
        raise ValueError('empty clip')
    bad = vs & ~np.isfinite(xs).all(axis=2)
    if bad.any():
        f, j = np.argwhere(bad)[0]
        raise ValueError('present coordinate is non-finite at frame %d joint %d' % (f, sel[j]))
    li, ri = SHOULDER_POS
    both = vs[:, li] & vs[:, ri]
    if not both.any():
        raise ValueError('no frame with both shoulders valid; scale undefined')
    safe = np.where(vs[:, :, None], xs, 0.0)
    scale = float(np.median(np.linalg.norm(safe[both, li] - safe[both, ri], axis=1)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('invalid shoulder scale %r' % scale)
    origin = 0.5 * (safe[:, li] + safe[:, ri])
    keep = vs & both[:, None]
    norm = np.where(keep[:, :, None], (safe - origin[:, None, :]) / scale, 0.0)
    out = np.ascontiguousarray(norm.transpose(2, 0, 1)[None], dtype=np.float32)
    if not np.isfinite(out).all():
        raise ValueError('non-finite normalized output')
    return out, {'frame_count': int(T), 'zero_frame_count': int((~both).sum()), 'scale': scale}


def preprocess_row(row, index=-1):
    try:
        xy, valid, _ts, src_sha = load_clip_arrays(row)
        x, diag = normalize_clip(xy, valid, row['layout'])
        if diag['frame_count'] != row['end_exclusive'] - row['start']:
            raise ValueError('interval length changed')
    except Exception as e:
        raise RuntimeError('input failure at %s: %s' % (row_where(index, row), e)) from e
    diag.update({'source_sha256': src_sha, 'input_array_sha256': array_sha(x), 'layout': row['layout'],
                 'start': row['start'], 'end_exclusive': row['end_exclusive'], 'mask_hand': row['mask_hand']})
    return x, diag


def preprocess_rows(rows, indices, log):
    '''Preprocess in order; exact duplicate source/slice/mask keys reuse one normalized input.'''
    inputs, diags, cache = {}, [], {}
    for k, i in enumerate(indices):
        r = rows[i]
        key = (r['sha256'].lower(), r['layout'], r['start'], r['end_exclusive'], r['mask_hand'])
        if key in cache:
            d = dict(cache[key], reused_duplicate=True)
        else:
            x, d = preprocess_row(r, i)
            d['reused_duplicate'] = False
            inputs[d['input_array_sha256']] = x
            cache[key] = d
        diags.append(dict(d, row_index=i, id=r['id']))
        if (k + 1) % 100 == 0 or k + 1 == len(indices):
            log('preprocessed %d/%d rows' % (k + 1, len(indices)))
    return inputs, diags


def graph_A(tensors):
    cands = sorted(k for k, v in tensors.items() if k.split('.')[-1] == 'A' and tuple(v.shape) == (3, 27, 27))
    if not cands:
        raise ValueError('graph A (3,27,27) not found in encoder tensors')
    A = tensors[cands[0]]
    if any(not torch.equal(A, tensors[k]) for k in cands[1:]):
        raise ValueError('conflicting graph A tensors: %s' % cands)
    return A.detach().clone()


def build_encoder(arm, tensors):
    import stgcn_core
    A = graph_A(tensors)
    if arm == 'random':
        torch.manual_seed(SEED)
        enc = stgcn_core.STGCNEncoder(A)
    elif arm == 'pretrained':
        enc = stgcn_core.STGCNEncoder(A)
    else:
        raise ValueError('unknown arm %r' % arm)
    keys = set(enc.state_dict().keys())
    if len(keys) != 170:
        raise RuntimeError('encoder state has %d keys, expected 170' % len(keys))
    if arm == 'pretrained':
        extra = set(tensors) - keys
        if any(k.split('.')[-1] != 'A' for k in extra):
            raise RuntimeError('unexpected non-encoder tensors: %s' % sorted(extra))
        enc.load_state_dict({k: tensors[k] for k in tensors if k in keys}, strict=True)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def state_hash(module):
    '''Deterministic hash over every state entry (parameters and BN buffers).'''
    h = hashlib.sha256()
    for k, v in sorted(module.state_dict().items()):
        t = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def encode(enc, x):
    if enc.training:
        raise RuntimeError('encoder left eval mode')
    with torch.no_grad():
        out = enc(torch.from_numpy(x))
    if tuple(out.shape) != (1, FEAT_DIM) or not bool(torch.isfinite(out).all()):
        raise RuntimeError('bad encoder output shape %s or non-finite values' % (tuple(out.shape),))
    return out[0].to(torch.float32).numpy().copy()


def extract_features(enc, diags, inputs, log, arm):
    feats = np.zeros((len(diags), FEAT_DIM), np.float32)
    cache = {}
    for k, d in enumerate(diags):
        h = d['input_array_sha256']
        if h not in cache:
            try:
                cache[h] = encode(enc, inputs[h])
            except Exception as e:
                raise RuntimeError('%s encoder failure at row %d id=%r: %s' % (arm, d['row_index'], d['id'], e)) from e
        feats[k] = cache[h]
        if (k + 1) % 100 == 0 or k + 1 == len(diags):
            log('%s features %d/%d rows (%d unique forwards)' % (arm, k + 1, len(diags), len(cache)))
    return feats


def weighted_stats(feats, counts, total=None):
    '''Weighted mean / population std; zero-weight rows are dropped before any arithmetic.'''
    f = np.asarray(feats, np.float64)
    c = np.asarray(counts, np.float64)
    if f.ndim != 2 or c.shape != (f.shape[0],) or (c < 0).any():
        raise ValueError('bad features/counts')
    keep = c > 0
    tot = float(c[keep].sum()) if total is None else float(total)
    if tot <= 0 or not math.isclose(float(c[keep].sum()), tot):
        raise ValueError('positive counts must sum to total %r' % tot)
    p, fk = c[keep] / tot, f[keep]
    if not np.isfinite(fk).all():
        raise ValueError('non-finite features on positive-weight rows')
    mean = p @ fk
    std = np.sqrt(p @ (fk - mean) ** 2)
    return mean, np.maximum(std, STD_FLOOR)


def standardize(feats, mean, std):
    return (np.asarray(feats, np.float64) - mean) / std


def fit_head(Z, y, weights, num_classes=NUM_CLASSES, optimizer_config=None):
    '''Zero-init weighted linear head, one LBFGS step(closure). optimizer_config is for unit tests only.'''
    cfg = dict(FROZEN_LBFGS)
    if optimizer_config is not None:
        cfg.update(optimizer_config)
    w = np.asarray(weights, np.float64)
    keep = w > 0
    if (w < 0).any() or not keep.any():
        raise ValueError('row weights must be nonnegative with at least one positive')
    X = torch.from_numpy(np.ascontiguousarray(np.asarray(Z, np.float64)[keep]))
    Y = torch.from_numpy(np.ascontiguousarray(np.asarray(y, np.int64)[keep]))
    P = torch.from_numpy(np.ascontiguousarray(w[keep]))
    if not bool(torch.isfinite(X).all()):
        raise ValueError('non-finite standardized training features')
    W = torch.zeros(num_classes, X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(num_classes, dtype=torch.float64, requires_grad=True)

    def objective():
        logits = X @ W.T + b
        return (P * F.cross_entropy(logits, Y, reduction='none')).sum() + 0.5 * L2 * (W ** 2).sum()

    opt = torch.optim.LBFGS([W, b], **cfg)
    calls = [0]

    def closure():
        opt.zero_grad()
        loss = objective()
        loss.backward()
        calls[0] += 1
        return loss

    with torch.no_grad():
        initial = float(objective())
    opt.step(closure)
    st = opt.state[W]
    opt.zero_grad()
    final = objective()
    final.backward()
    grads = torch.cat([W.grad.reshape(-1), b.grad.reshape(-1)])
    n_iter, n_eval = int(st.get('n_iter', 0)), int(st.get('func_evals', 0))
    report = {
        'initial_train_loss': initial, 'final_train_loss': float(final.detach()),
        'final_grad_finite': bool(torch.isfinite(grads).all()), 'final_grad_norm': float(grads.norm()),
        'final_grad_max_abs': float(grads.abs().max()), 'closure_calls': calls[0], 'lbfgs_n_iter': n_iter,
        'lbfgs_func_evals': n_eval, 'hit_max_iter': n_iter >= cfg['max_iter'],
        'hit_max_eval': n_eval >= cfg['max_eval'], 'n_rows_in_loss': int(keep.sum()),
        'optimizer_config': cfg, 'step_calls': 1,
        'note': 'limits and final gradient reported; convergence is not inferred from step return'}
    Wn, bn = W.detach().numpy().copy(), b.detach().numpy().copy()
    if not (np.isfinite(Wn).all() and np.isfinite(bn).all() and report['final_grad_finite']
            and math.isfinite(report['final_train_loss'])):
        raise RuntimeError('non-finite head, loss or gradient after fit: %s' % report)
    return Wn, bn, report


def predict_scores(feats, head):
    return standardize(feats, head['mean'], head['std']) @ head['W'].T + head['b']


def top1(scores, num_classes=NUM_CLASSES):
    s = np.asarray(scores, np.float64)
    if s.ndim != 2 or s.shape[1] != num_classes or not np.isfinite(s).all():
        raise ValueError('scores must be finite [N,%d]' % num_classes)
    return np.argmax(s, axis=1)


def summarize(pred, y, classes):
    pred, y = np.asarray(pred), np.asarray(y)
    n = int(y.size)
    if n == 0:
        return {'n': 0}
    correct = pred == y
    per = {}
    for c in sorted(set(y.tolist())):
        m = y == c
        per[classes[c]] = {'n': int(m.sum()), 'correct': int(correct[m].sum()),
                           'recall_pct': float(correct[m].sum()) / int(m.sum()) * 100.0}
    return {'n': n, 'correct': int(correct.sum()), 'accuracy_pct': float(correct.sum()) / n * 100.0,
            'macro_recall_gt_present_pct': float(np.mean([v['recall_pct'] for v in per.values()])),
            'n_gt_present_classes': len(per), 'per_class': per}


def compare(pa, pb, y, row_index, ids):
    y = np.asarray(y)
    ca, cb = np.asarray(pa) == y, np.asarray(pb) == y
    n = int(y.size)

    def ident(m):
        return [[int(row_index[j]), str(ids[j])] for j in np.flatnonzero(m)]
    return {'n': n, 'a_correct_b_wrong': ident(ca & ~cb), 'regressions_b_correct_a_wrong': ident(cb & ~ca),
            'pp_delta_a_minus_b': (float(ca.sum()) - float(cb.sum())) / n * 100.0 if n else 0.0}


def build_subsets(meta):
    g, c, s, e, lay = meta['groups'], meta['conditions'], meta['signer'], meta['exposed'], meta['layouts']
    subs = {}

    def add(name, mask):
        idx = np.flatnonzero(mask)
        if idx.size:
            subs[name] = idx
    for grp in EVAL_GROUPS:
        add('group=' + grp, g == grp)
    for grp in ('public', 'mp'):
        for cond in sorted(set(c[g == grp].tolist())):
            add('group=%s|condition=%s' % (grp, cond), (g == grp) & (c == cond))
    for grp in EVAL_GROUPS:
        nat = (g == grp) & (lay == 'native67') & (s >= 0)
        for sid in sorted(set(s[nat].tolist())):
            add('group=%s|native_signer=%d' % (grp, sid), nat & (s == sid))
    add('group=ksl36|primary_unexposed', (g == 'ksl36') & ~e)
    add('group=ksl36|exposed', (g == 'ksl36') & e)
    return subs


def evaluate_metrics(scores, y, meta, classes):
    preds = {name: top1(s) for name, s in scores.items()}
    out = {}
    for sname, idx in build_subsets(meta).items():
        ent = {'n': int(idx.size), 'by_source': {}, 'comparisons': {}}
        for src, p in preds.items():
            ent['by_source'][src] = summarize(p[idx], y[idx], classes)
        for a, b in (('pretrained', 'baseline'), ('random', 'baseline'), ('pretrained', 'random')):
            ent['comparisons'][a + '_vs_' + b] = compare(preds[a][idx], preds[b][idx], y[idx],
                                                       meta['row_index'][idx], meta['ids'][idx])
        out[sname] = ent
    return out


def parse_baseline(rows, idx):
    out = np.zeros((len(idx), NUM_CLASSES), np.float64)
    for k, i in enumerate(idx):
        bs = rows[i].get('baseline_scores')
        if not (isinstance(bs, list) and len(bs) == NUM_CLASSES and all(_is_num(v) for v in bs)):
            raise ValueError('%s: baseline_scores must be %d finite numbers' % (row_where(i, rows[i]), NUM_CLASSES))
        out[k] = bs
    return out


def load_head(data, classes):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        h = {k: z[k] for k in ('W', 'b', 'mean', 'std', 'classes')}
    if (h['W'].shape != (NUM_CLASSES, FEAT_DIM) or h['b'].shape != (NUM_CLASSES,)
            or h['mean'].shape != (FEAT_DIM,) or h['std'].shape != (FEAT_DIM,)):
        raise ValueError('bad head shapes')
    if h['classes'].tolist() != list(classes):
        raise ValueError('head class order differs from manifest')
    if not all(np.isfinite(h[k]).all() for k in ('W', 'b', 'mean', 'std')) or (h['std'] < STD_FLOOR).any():
        raise ValueError('non-finite head or std below floor')
    return h


def setup_runtime(threads):
    torch.set_num_threads(int(threads))
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def load_pinned_inputs(args):
    mbytes = read_pinned(args.manifest, args.manifest_sha, 'manifest')
    read_pinned(args.zip, args.zip_sha, 'zip')
    read_pinned(args.metadata, args.metadata_sha, 'metadata')
    pins = {'manifest_sha256': args.manifest_sha.lower(), 'zip_sha256': args.zip_sha.lower(),
            'metadata_sha256': args.metadata_sha.lower()}
    return pins, json.loads(mbytes.decode('utf-8'))


def code_shas():
    import stgcn_core
    return {'runner_sha256': sha256_file(os.path.abspath(__file__)),
            'core_sha256': sha256_file(os.path.abspath(stgcn_core.__file__))}


def read_tensors(args):
    import stgcn_core
    tensors, audit = stgcn_core.read_encoder_tensors(args.zip, args.metadata, args.zip_sha)
    return tensors, json.loads(json.dumps(audit, default=str))


def run_train(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    paths = {k: os.path.join(args.out, v) for k, v in TRAIN_FILES.items()}
    refuse_existing(list(paths.values()) + [os.path.join(args.out, v) for v in EVAL_FILES.values()])
    log = Log(paths['log'])
    try:
        pins, manifest = load_pinned_inputs(args)
        classes, rows = validate_manifest(manifest, strict=True)
        setup_runtime(args.threads)
        code = code_shas()
        log('pins verified %s code %s threads %d' % (pins, code, args.threads))
        tensors, audit = read_tensors(args)
        tidx = [i for i, r in enumerate(rows) if r['group'] in TRAIN_GROUPS]
        counts = np.array([rows[i]['weight_count'] for i in tidx], np.int64)
        y = np.array([rows[i]['y'] for i in tidx], np.int64)
        inputs, diags = preprocess_rows(rows, tidx, log)
        rw = counts.astype(np.float64) / TOTAL_WEIGHT
        arms = {}
        for arm in ARMS:
            enc = build_encoder(arm, tensors)
            h0 = state_hash(enc)
            feats = extract_features(enc, diags, inputs, log, arm)
            h1 = state_hash(enc)
            mean, std = weighted_stats(feats, counts, TOTAL_WEIGHT)
            W, b, report = fit_head(standardize(feats, mean, std), y, rw)
            h2 = state_hash(enc)
            if not h0 == h1 == h2:
                raise RuntimeError('%s encoder state changed during train phase' % arm)
            log('%s head fit report %s' % (arm, json.dumps(report, sort_keys=True)))
            feat_sha = write_npz_x(paths['features_' + arm], features=feats, row_index=np.array(tidx, np.int64),
                                   ids=np.array([rows[i]['id'] for i in tidx]),
                                   groups=np.array([rows[i]['group'] for i in tidx]), y=y, weight_count=counts,
                                   input_array_sha256=np.array([d['input_array_sha256'] for d in diags]))
            head_sha = write_npz_x(paths['head_' + arm], W=W, b=b, mean=mean, std=std,
                                   classes=np.array(classes), arm=np.array(arm))
            summary = {'arm': arm, 'pins': pins, 'code': code, 'recipe': recipe(), 'threads': args.threads,
                       'encoder_state_sha256_before': h0, 'encoder_state_sha256_after_features': h1,
                       'encoder_state_sha256_after_fit': h2, 'optimizer_report': report,
                       'n_train_rows': len(tidx), 'n_positive_weight_rows': int((counts > 0).sum()),
                       'features_sha256': feat_sha, 'head_sha256': head_sha, 'encoder_audit': audit,
                       'preprocess_diagnostics': diags}
            sum_sha = write_json_x(paths['summary_' + arm], summary)
            arms[arm] = {'head_file': TRAIN_FILES['head_' + arm], 'head_sha256': head_sha,
                         'features_file': TRAIN_FILES['features_' + arm], 'features_sha256': feat_sha,
                         'summary_file': TRAIN_FILES['summary_' + arm], 'summary_sha256': sum_sha,
                         'encoder_state_sha256': h0}
            log('%s saved head %s features %s summary %s' % (arm, head_sha, feat_sha, sum_sha))
        frozen = {'arm_names': list(ARMS), 'arms': arms, 'inputs': pins, 'code': code, 'recipe': recipe(),
                  'classes': classes, 'n_train_rows': len(tidx), 'threads': args.threads,
                  'note': 'written only after both head fits completed; no evaluation data touched'}
        fsha = write_json_x(paths['frozen'], frozen)
        log('FROZEN_HEADS.json sha256=%s elapsed %.1fs' % (fsha, time.time() - t0))
    except BaseException as e:
        log('ABORTED: %s: %s' % (type(e).__name__, e))
        raise


def run_evaluate(args):
    t0 = time.time()
    paths = {k: os.path.join(args.out, v) for k, v in EVAL_FILES.items()}
    refuse_existing(list(paths.values()))
    missing = [v for v in TRAIN_FILES.values() if not os.path.exists(os.path.join(args.out, v))]
    if missing:
        raise FileNotFoundError('train outputs missing; train phase must complete first: %s' % missing)
    log = Log(paths['log'])
    try:
        pins, manifest = load_pinned_inputs(args)
        classes, rows = validate_manifest(manifest, strict=True)
        setup_runtime(args.threads)
        code = code_shas()
        with open(os.path.join(args.out, TRAIN_FILES['frozen']), 'rb') as f:
            fbytes = f.read()
        frozen_sha, frozen = sha256_bytes(fbytes), json.loads(fbytes.decode('utf-8'))
        for key, want in (('inputs', pins), ('code', code), ('recipe', recipe()), ('classes', classes),
                          ('arm_names', list(ARMS))):
            if frozen.get(key) != want:
                raise RuntimeError('FROZEN_HEADS pin mismatch for %s' % key)
        heads = {}
        for arm in ARMS:
            ent = frozen['arms'][arm]
            for kind in ('features', 'summary', 'head'):
                name = ent[kind + '_file']
                if os.path.basename(name) != name:
                    raise RuntimeError('unexpected path in FROZEN_HEADS: %r' % name)
                data = read_pinned(os.path.join(args.out, name), ent[kind + '_sha256'], '%s %s' % (arm, kind))
                if kind == 'head':
                    heads[arm] = load_head(data, classes)
        tensors, _audit = read_tensors(args)
        encs = {}
        for arm in ARMS:
            enc = build_encoder(arm, tensors)
            h = state_hash(enc)
            if h != frozen['arms'][arm]['encoder_state_sha256']:
                raise RuntimeError('%s reconstructed encoder hash differs from frozen hash' % arm)
            encs[arm] = (enc, h)
        log('all frozen pins verified (FROZEN_HEADS sha256=%s)' % frozen_sha)
        eidx = [i for i, r in enumerate(rows) if r['group'] in EVAL_GROUPS]
        baseline = parse_baseline(rows, eidx)
        n = len(eidx)
        scores = {arm: np.zeros((n, NUM_CLASSES), np.float64) for arm in ARMS}
        diags = []
        for k, i in enumerate(eidx):
            x, d = preprocess_row(rows[i], i)
            try:
                for arm in ARMS:
                    s = predict_scores(encode(encs[arm][0], x)[None], heads[arm])[0]
                    if not np.isfinite(s).all():
                        raise RuntimeError('non-finite %s scores' % arm)
                    scores[arm][k] = s
            except Exception as e:
                raise RuntimeError('prediction failure at %s: %s' % (row_where(i, rows[i]), e)) from e
            diags.append(d)
            if (k + 1) % 100 == 0 or k + 1 == n:
                log('evaluated %d/%d rows' % (k + 1, n))
        for arm in ARMS:
            if state_hash(encs[arm][0]) != encs[arm][1]:
                raise RuntimeError('%s encoder state changed during evaluation' % arm)

        def col(key):
            return [rows[i][key] for i in eidx]
        y = np.array(col('y'), np.int64)
        meta = {'row_index': np.array(eidx, np.int64), 'ids': np.array(col('id')), 'groups': np.array(col('group')),
                'conditions': np.array(col('condition')), 'layouts': np.array(col('layout')),
                'signer': np.array([-1 if s is None else s for s in col('signer_id')], np.int64),
                'exposed': np.array(col('exposed'), bool)}
        all_scores = {'pretrained': scores['pretrained'], 'random': scores['random'], 'baseline': baseline}
        subsets = evaluate_metrics(all_scores, y, meta, classes)
        scores_sha = write_npz_x(paths['scores'], labels=np.array(col('label')), y=y, classes=np.array(classes),
                                 scores_pretrained=scores['pretrained'], scores_random=scores['random'],
                                 scores_baseline=baseline,
                                 input_array_sha256=np.array([d['input_array_sha256'] for d in diags]),
                                 source_sha256=np.array([d['source_sha256'] for d in diags]),
                                 frame_count=np.array([d['frame_count'] for d in diags], np.int64),
                                 zero_frame_count=np.array([d['zero_frame_count'] for d in diags], np.int64),
                                 scale=np.array([d['scale'] for d in diags], np.float64), **meta)
        metrics = {'phase': 'evaluate', 'pins': pins, 'code': code, 'frozen_heads_sha256': frozen_sha,
                   'encoder_state_sha256': {a: encs[a][1] for a in ARMS}, 'n_eval_rows': n,
                   'scores_npz_sha256': scores_sha, 'elapsed_seconds': time.time() - t0, 'threads': args.threads,
                   'classes': classes, 'tie_break': 'argmax lowest index over all 93 classes', 'subsets': subsets,
                   'notes': ['screening metrics only; no adoption decision is made',
                             'no new-person generalisation claim; no pipeline FPS claim',
                             'accuracy is top-1 correctness, never softmax confidence',
                             'overlapping datasets are never summed; no output-label restriction',
                             'target18 partial metrics not computed (no label mapping supplied)',
                             'standardization uses saved train statistics only']}
        msha = write_json_x(paths['metrics'], metrics)
        log('eval outputs written scores=%s metrics=%s elapsed %.1fs' % (scores_sha, msha, time.time() - t0))
    except BaseException as e:
        log('ABORTED: %s: %s' % (type(e).__name__, e))
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description='Frozen ST-GCN feature-transfer screening runner (local CPU).')
    ap.add_argument('--phase', required=True, choices=('train', 'evaluate'))
    for name in ('--manifest', '--manifest-sha', '--zip', '--zip-sha', '--metadata', '--metadata-sha', '--out'):
        ap.add_argument(name, required=True)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error('--threads must be >= 1')
    (run_train if args.phase == 'train' else run_evaluate)(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
