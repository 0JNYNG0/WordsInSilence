'''Paired fine-tuning experiment: head_only (control) vs full_encoder (candidate).

Fixed recipe, three phases (preflight, train, evaluate). Every mismatch fails closed.
Outputs are created exclusively and are never overwritten.
'''
import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

SEED = 20261005
THREADS = 2
STEPS = 660
PER_GROUP = 16
BATCH = 2 * PER_GROUP
HEAD_LR = 1e-4
ENC_LR = 1e-5
BETAS = (0.9, 0.999)
EPS = 1e-8
L2 = 1e-4
LOG_EVERY = 25
ARMS = ('head_only', 'full_encoder')
CONTROL, CANDIDATE = ARMS
PIN_NAMES = ('manifest', 'zip', 'metadata', 'head', 'train_features', 'schedule',
             'prior_scores', 'prior_core', 'prior_runner', 'plan')
TRAIN_SIZES = {'base': 2122, 'pair_context': 224}
EVAL_SIZES = {'annotation_dev': 699, 'whole_dev': 74, 'public': 32, 'mp': 14, 'ksl36': 36}
N_TRAIN = 2346
N_EVAL = 855
N_CLASSES = 93
FEAT_DIM = 256
N_TENSORS = 170
EXPOSURES = STEPS * BATCH
FEAT_TOL = {'atol': 1e-5, 'rtol': 1e-5}
SCORE_TOL = {'atol': 1e-4, 'rtol': 1e-5}
CE_TOL = {'atol': 1e-4, 'rtol': 1e-5}
N_SPACED = 8
NORM_ROWS = 32
FIXED_PARAM_NAMES = ('A',)
META_KEYS = ('row_index', 'ids', 'groups', 'conditions', 'layouts', 'signer', 'exposed')
DIAG_KEYS = ('input_array_sha256', 'source_sha256', 'frame_count', 'zero_frame_count', 'scale')
EVAL_ARMS = ('head_only', 'full_encoder', 'baseline', 'frozen_parent')
RECIPE = {'seed': SEED, 'threads': THREADS, 'steps': STEPS, 'batch_base_context': [PER_GROUP, PER_GROUP],
          'head_lr': HEAD_LR, 'encoder_lr_candidate_only': ENC_LR, 'betas': list(BETAS), 'eps': EPS,
          'weight_decay': 0.0, 'amsgrad': False, 'l2_coef': L2,
          'loss': 'mean 93-way CE over 32 clips + 0.5*l2_coef*sum(W^2)', 'head_dtype': 'float64',
          'encoder_dtype': 'float32', 'encoder_mode': 'eval', 'clipping': None, 'scheduler': None,
          'exposures': EXPOSURES, 'checkpoint': 'final only'}


class FailClosed(RuntimeError):
    pass


def require(cond, msg):
    if not cond:
        raise FailClosed(msg)


# ---------------- hashing / exclusive IO ----------------
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return jsonable(o.detach().cpu().numpy())
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.str_):
        return str(o)
    return o


def normalize(obj):
    return json.loads(json.dumps(jsonable(obj), sort_keys=True, default=str))


def write_json_x(path, obj):
    data = json.dumps(jsonable(obj), indent=1, sort_keys=True, default=str).encode('utf-8')
    with open(path, 'xb') as f:
        f.write(data)
    return sha256_bytes(data)


def write_npz_x(path, arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    data = buf.getvalue()
    with open(path, 'xb') as f:
        f.write(data)
    return sha256_bytes(data)


def ensure_absent(paths):
    for p in paths:
        require(not os.path.exists(p), 'refusing to overwrite existing output: ' + str(p))


def read_verified(path, sha):
    with open(path, 'rb') as f:
        data = f.read()
    require(sha256_bytes(data) == sha, 'sha256 mismatch: ' + str(path))
    return data


def npz_from_bytes(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def batch_sha(batches):
    return sha256_bytes(np.ascontiguousarray(batches, dtype=np.int64).tobytes())


def input_digest(cache):
    return sha256_bytes(','.join(str(h) for h in cache['input_array_sha256'].tolist()).encode('utf-8'))


class Logger:
    def __init__(self, path):
        self.f = open(path, 'x', encoding='utf-8')

    def __call__(self, msg):
        line = time.strftime('%Y-%m-%d %H:%M:%S ') + str(msg)
        print(line, flush=True)
        print(line, file=self.f, flush=True)

    def close(self):
        self.f.close()


# ---------------- pins / prior modules ----------------
def resolve_pins(config, base_dir):
    pins = config.get('pins')
    require(isinstance(pins, dict), 'config.pins missing')
    require(set(pins) == set(PIN_NAMES), 'pins must be exactly: ' + ','.join(PIN_NAMES))
    out = {}
    for name in PIN_NAMES:
        p = pins[name]
        require(isinstance(p, dict) and isinstance(p.get('path'), str) and isinstance(p.get('sha256'), str),
                'bad pin entry: ' + name)
        path = p['path'] if os.path.isabs(p['path']) else os.path.join(base_dir, p['path'])
        out[name] = {'path': os.path.abspath(path), 'sha256': p['sha256'].lower()}
    return out


def verify_pins(pins):
    for name in sorted(pins):
        p = pins[name]
        require(os.path.isfile(p['path']), 'pinned file missing: ' + name)
        require(sha256_file(p['path']) == p['sha256'], 'pin sha256 mismatch: ' + name)
    return True


def load_prior(prior_dir, pins):
    prior_dir = os.path.realpath(prior_dir)
    require(os.path.isdir(prior_dir), 'prior-dir missing')
    for mod in ('transfer_runner', 'stgcn_core'):
        require(mod not in sys.modules, mod + ' already imported from elsewhere')
    sys.path.insert(0, prior_dir)
    import transfer_runner as R
    import stgcn_core as C
    for mod, name in ((R, 'prior_runner'), (C, 'prior_core')):
        f = os.path.realpath(mod.__file__)
        require(os.path.normcase(os.path.dirname(f)) == os.path.normcase(prior_dir), name + ' not loaded from prior-dir')
        require(sha256_file(f) == pins[name]['sha256'], name + ' module file differs from pin')
    return R, C


class Ctx:
    def __init__(self, args):
        self.out = os.path.abspath(args.out)
        cfg_path = os.path.abspath(args.config)
        with open(cfg_path, 'rb') as f:
            raw = f.read()
        self.config_sha = sha256_bytes(raw)
        self.config = json.loads(raw.decode('utf-8'))
        require(isinstance(self.config, dict), 'config must be a JSON object')
        self.pins = resolve_pins(self.config, os.path.dirname(cfg_path))
        verify_pins(self.pins)
        self.R, self.C = load_prior(args.prior_dir, self.pins)
        self.code = {'paired_finetune': sha256_file(os.path.abspath(__file__)),
                     'transfer_runner': sha256_file(self.R.__file__),
                     'stgcn_core': sha256_file(self.C.__file__)}
        self.R.setup_runtime(THREADS)
        require(torch.get_num_threads() == THREADS, 'torch thread count is not %d' % THREADS)

    def path(self, name):
        return os.path.join(self.out, name)

    def blob(self, name):
        p = self.pins[name]
        return self.R.read_pinned(p['path'], p['sha256'], name)

    def identity(self):
        return normalize({'config_sha256': self.config_sha, 'code_sha256': self.code, 'pins': self.pins})


# ---------------- input validation ----------------
def validate_head(h, classes, n_classes=N_CLASSES, dim=FEAT_DIM):
    require(all(k in h for k in ('W', 'b', 'mean', 'std', 'classes')), 'head keys missing')
    shapes = {'W': (n_classes, dim), 'b': (n_classes,), 'mean': (dim,), 'std': (dim,)}
    for k, s in shapes.items():
        a = np.asarray(h[k])
        require(a.dtype == np.float64 and a.shape == s, 'head %s must be float64 %s' % (k, s))
        require(np.isfinite(a).all(), 'head %s is nonfinite' % k)
    require((np.asarray(h['std']) > 0).all(), 'head std must be positive')
    require([str(c) for c in np.asarray(h['classes']).tolist()] == [str(c) for c in classes],
            'head classes differ from manifest')
    return {k: np.asarray(h[k]) for k in ('W', 'b', 'mean', 'std', 'classes')}


def load_common(ctx):
    manifest = json.loads(ctx.blob('manifest').decode('utf-8'))
    classes, rows = ctx.R.validate_manifest(manifest, strict=True)
    classes = [str(c) for c in classes]
    require(len(classes) == N_CLASSES and len(set(classes)) == N_CLASSES, 'classes must be 93 unique')
    keys = [(r['group'], r['id'], r['condition']) for r in rows]
    require(len(set(keys)) == len(keys), 'duplicate (group,id,condition)')
    head = validate_head(npz_from_bytes(ctx.blob('head')), classes)
    return classes, rows, head


def group_indices(rows, group):
    return [i for i, r in enumerate(rows) if r['group'] == group]


def validate_cache(cache, rows, classes):
    need = ('features', 'row_index', 'ids', 'groups', 'y', 'weight_count', 'input_array_sha256')
    require(all(k in cache for k in need), 'train_features keys missing')
    f = cache['features']
    require(f.dtype == np.float32 and f.shape == (N_TRAIN, FEAT_DIM) and np.isfinite(f).all(), 'bad cached features')
    ri = cache['row_index'].astype(np.int64)
    require(ri.shape == (N_TRAIN,) and len(set(ri.tolist())) == N_TRAIN, 'bad cache row_index')
    positions = {}
    for g, n in TRAIN_SIZES.items():
        man = group_indices(rows, g)
        require(len(man) == n, 'manifest size of group %s != %d' % (g, n))
        pos = np.flatnonzero(cache['groups'] == g)
        require(ri[pos].tolist() == man, 'cache order differs from manifest order for ' + g)
        positions[g] = pos
    require(sum(len(p) for p in positions.values()) == N_TRAIN, 'cache contains non-training rows')
    for k, i in enumerate(ri.tolist()):
        r = rows[i]
        y = int(r['y'])
        require(r['group'] in TRAIN_SIZES, 'cache row is not a training row: %d' % k)
        require(str(cache['ids'][k]) == str(r['id']), 'cache id mismatch at %d' % k)
        require(0 <= y < N_CLASSES and int(cache['y'][k]) == y and classes[y] == str(r['label']), 'label mismatch at %d' % k)
        require(int(cache['weight_count'][k]) == int(r['weight_count']), 'weight_count mismatch at %d' % k)
    return ri, positions


def map_schedule(base_idx, ctx_idx, base_pos, ctx_pos, steps=STEPS, per_group=PER_GROUP):
    '''Map group-local schedule indices to cache positions: 16 base then 16 context per step.'''
    out = []
    for name, idx, pos in (('base', base_idx, base_pos), ('context', ctx_idx, ctx_pos)):
        idx = np.asarray(idx)
        require(np.issubdtype(idx.dtype, np.integer), name + ' schedule must be integer')
        require(idx.shape == (steps, per_group), name + ' schedule shape must be %dx%d' % (steps, per_group))
        require(int(idx.min()) >= 0 and int(idx.max()) < len(pos), name + ' schedule index out of bounds')
        out.append(np.asarray(pos, dtype=np.int64)[idx])
    return np.concatenate(out, axis=1)


def exposure_counts(batches, n_rows):
    return np.bincount(np.asarray(batches, dtype=np.int64).ravel(), minlength=n_rows)


def validate_counts(counts, weight_count):
    wc = np.asarray(weight_count, dtype=np.int64)
    require(counts.shape == wc.shape, 'exposure count shape mismatch')
    bad = np.flatnonzero(counts != wc)
    require(bad.size == 0, 'schedule exposure != weight_count at %d rows, first %s' % (bad.size, bad[:5].tolist()))
    return int(counts.sum())


def load_training(ctx, rows, classes):
    cache = npz_from_bytes(ctx.blob('train_features'))
    ri, positions = validate_cache(cache, rows, classes)
    sched = npz_from_bytes(ctx.blob('schedule'))
    require('base_indices' in sched and 'context_indices' in sched, 'schedule keys missing')
    batches = map_schedule(sched['base_indices'], sched['context_indices'], positions['base'], positions['pair_context'])
    counts = exposure_counts(batches, N_TRAIN)
    total = validate_counts(counts, cache['weight_count'])
    require(total == EXPOSURES, 'total exposure %d != %d' % (total, EXPOSURES))
    return cache, ri, batches, counts


def preprocess_train(R, rows, cache, ri):
    arrays = []
    for k, i in enumerate(ri.tolist()):
        require(rows[i]['group'] in TRAIN_SIZES, 'refusing to preprocess non-training row')
        x, diag = R.preprocess_row(rows[i], index=i)
        require(isinstance(x, np.ndarray) and x.dtype == np.float32 and x.ndim == 4 and x.shape[:2] == (1, 2)
                and x.shape[2] > 0 and x.shape[3] == 27, 'bad preprocessed array at cache row %d' % k)
        require(R.array_sha(x) == diag['input_array_sha256'] == str(cache['input_array_sha256'][k]),
                'input array hash mismatch at cache row %d' % k)
        arrays.append(x)
    return arrays


def read_tensors(ctx):
    tensors, audit = ctx.C.read_encoder_tensors(ctx.pins['zip']['path'], ctx.pins['metadata']['path'],
                                                ctx.pins['zip']['sha256'])
    require(isinstance(tensors, dict) and len(tensors) == N_TENSORS, 'expected %d encoder tensors' % N_TENSORS)
    return tensors, audit


def clone_tensors(tensors):
    return {k: (v.detach().clone() if torch.is_tensor(v) else np.array(v, copy=True)) for k, v in tensors.items()}


# ---------------- model helpers ----------------
def is_fixed(name):
    return name.split('.')[-1] in FIXED_PARAM_NAMES


def set_trainable(module, flag):
    '''flag=True: every nn.Parameter trainable (graph A excluded if registered as one). Always eval().'''
    for name, p in module.named_parameters():
        p.requires_grad_(bool(flag) and not is_fixed(name))
    module.eval()
    require(not any(m.training for m in module.modules()), 'module not in eval mode')
    return sorted(n for n, p in module.named_parameters() if p.requires_grad)


def state_snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def state_diff(before, module):
    after = module.state_dict()
    require(set(before) == set(after), 'state keys changed')
    diff = {}
    for k in sorted(before):
        a, b = before[k], after[k]
        require(a.dtype == b.dtype and a.shape == b.shape, 'dtype/shape changed: ' + k)
        diff[k] = 0.0 if torch.equal(a, b) else float((b.double() - a.double()).abs().max())
    return diff


def states_equal(m1, m2):
    s1, s2 = m1.state_dict(), m2.state_dict()
    return set(s1) == set(s2) and all(s1[k].dtype == s2[k].dtype and torch.equal(s1[k], s2[k]) for k in s1)


def state_arrays(module):
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}


def load_state_arrays(module, arrays):
    sd = module.state_dict()
    require(set(arrays) == set(sd), 'state key set mismatch')
    new = {}
    for k, v in sd.items():
        a = np.asarray(arrays[k])
        ref = v.detach().cpu().numpy()
        require(a.dtype == ref.dtype and a.shape == ref.shape, 'dtype/shape mismatch: ' + k)
        new[k] = torch.from_numpy(np.array(a, copy=True))
    module.load_state_dict(new, strict=True)
    module.eval()
    return module


def head_params(head):
    W = torch.from_numpy(np.array(head['W'], dtype=np.float64, copy=True)).requires_grad_(True)
    b = torch.from_numpy(np.array(head['b'], dtype=np.float64, copy=True)).requires_grad_(True)
    mean = torch.from_numpy(np.array(head['mean'], dtype=np.float64, copy=True))
    std = torch.from_numpy(np.array(head['std'], dtype=np.float64, copy=True))
    return W, b, mean, std


def head_logits(feat, W, b, mean, std):
    z = (feat.to(torch.float64) - mean) / std
    return z @ W.t() + b


def require_finite(tensors, what):
    for t in tensors:
        require(bool(torch.isfinite(t).all()), 'nonfinite ' + what)


def np_log_softmax(s):
    s = np.asarray(s, dtype=np.float64)
    z = s - s.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def np_ce(scores, y):
    lsm = np_log_softmax(scores)
    return float(-lsm[np.arange(len(y)), np.asarray(y, dtype=np.int64)].mean()), lsm


def step_gradients(batch, labels, W, b, mean, std, feature_fn):
    '''One clip at a time: CE/n backward immediately (no retained graph), then L2 once.'''
    n = len(batch)
    require(n > 0, 'empty batch')
    ce_sum = 0.0
    for pos in batch:
        pos = int(pos)
        logits = head_logits(feature_fn(pos), W, b, mean, std)
        require(logits.dim() == 2 and logits.shape[0] == 1, 'feature_fn must return exactly one clip')
        ce = F.cross_entropy(logits, torch.tensor([int(labels[pos])], dtype=torch.long))
        (ce / n).backward()
        ce_sum += float(ce.detach())
    reg = 0.5 * L2 * W.square().sum()
    reg.backward()
    return ce_sum / n, float(reg.detach())


def gradient_probe(encoder, head, x, y, candidate):
    '''Single-clip backward without optimizer step; reports which encoder parameters receive gradient.'''
    set_trainable(encoder, candidate)
    before = state_snapshot(encoder)
    named = dict(encoder.named_parameters())
    for p in named.values():
        p.grad = None
    W, b, mean, std = head_params(head)
    logits = head_logits(encoder(torch.from_numpy(x)), W, b, mean, std)
    ce = F.cross_entropy(logits, torch.tensor([int(y)], dtype=torch.long))
    ce.backward()
    reached = sorted(k for k, p in named.items() if p.grad is not None and bool((p.grad != 0).any()))
    res = {'candidate': bool(candidate), 'ce': float(ce.detach()), 'n_parameters': len(named),
           'reached_nonzero': len(reached), 'reached_keys': reached,
           'grad_none_keys': sorted(k for k, p in named.items() if p.grad is None),
           'head_grad_nonzero': bool((W.grad != 0).any()),
           'state_unchanged': all(v == 0.0 for v in state_diff(before, encoder).values())}
    for p in named.values():
        p.grad = None
    set_trainable(encoder, False)
    return res


def descriptive_norms(encoder, arrays, W, b, mean, std):
    with torch.no_grad():
        feats = torch.cat([encoder(torch.from_numpy(a)) for a in arrays]).to(torch.float64)
        logits = head_logits(feats, W, b, mean, std)
        return {'rows': len(arrays), 'feature_l2_mean': float(feats.norm(dim=1).mean()),
                'logit_l2_mean': float(logits.norm(dim=1).mean()), 'W_fro': float(W.norm()),
                'b_l2': float(b.norm()), 'note': 'descriptive only; not used for any selection'}


def train_arm(arm, encoder, head, batches, labels, feature_fn, log, n_rows):
    '''Train one arm from the given encoder/head. Returns (W, b, exposure_counts, summary).'''
    require(arm in ARMS, 'unknown arm ' + str(arm))
    candidate = arm == CANDIDATE
    torch.manual_seed(SEED)
    trainable = set_trainable(encoder, candidate)
    named = dict(encoder.named_parameters())
    require(all(p.dtype == torch.float32 for p in named.values()), 'encoder parameters must be float32')
    enc_params = [named[k] for k in trainable]
    require(candidate == bool(enc_params), 'trainable encoder set inconsistent with arm')
    W, b, mean, std = head_params(head)
    W0, b0 = W.detach().clone(), b.detach().clone()
    groups = [{'params': [W, b], 'lr': HEAD_LR}]
    if candidate:
        groups.append({'params': enc_params, 'lr': ENC_LR})
    opt = torch.optim.Adam(groups, betas=BETAS, eps=EPS, weight_decay=0.0, amsgrad=False)
    before = state_snapshot(encoder)
    batches = np.asarray(batches, dtype=np.int64)
    counts = np.zeros(n_rows, dtype=np.int64)
    per_step, grad_none = [], None
    t0 = time.time()
    for s in range(len(batches)):
        batch = batches[s]
        opt.zero_grad(set_to_none=True)
        ce, reg = step_gradients(batch, labels, W, b, mean, std, feature_fn)
        loss = ce + reg
        require(np.isfinite(loss), 'nonfinite loss at step %d' % (s + 1))
        if grad_none is None:
            grad_none = [k for k in trainable if named[k].grad is None]
        if not candidate:
            require(all(p.grad is None for p in named.values()), 'control encoder received a gradient')
        gn = {}
        for gname, ps in (('head', [W, b]), ('encoder', enc_params)):
            grads = [p.grad for p in ps if p.grad is not None]
            require_finite(grads, '%s gradient at step %d' % (gname, s + 1))
            gn[gname] = float(sum(float(g.double().square().sum()) for g in grads) ** 0.5)
        opt.step()
        require_finite([W, b] + enc_params, 'weights after step %d' % (s + 1))
        counts += np.bincount(batch, minlength=n_rows)
        per_step.append({'step': s + 1, 'loss': loss, 'ce': ce, 'l2': reg,
                         'grad_norm_head': gn['head'], 'grad_norm_encoder': gn['encoder']})
        if s == 0 or (s + 1) % LOG_EVERY == 0 or s + 1 == len(batches):
            log('%s step %d/%d loss %.6f ce %.6f l2 %.6e gnorm_head %.4e gnorm_enc %.4e %.1fs' % (
                arm, s + 1, len(batches), loss, ce, reg, gn['head'], gn['encoder'], time.time() - t0))
    diff = state_diff(before, encoder)
    tset = set(trainable)
    buffer_keys = sorted(k for k in diff if k not in named)
    changed = {k: v for k, v in diff.items() if v != 0.0}
    require(all(k in tset for k in changed), 'non-trainable state changed: %s' % sorted(set(changed) - tset))
    require(candidate or not changed, 'control encoder changed')
    W, b = W.detach(), b.detach()
    summary = {'arm': arm, 'steps': len(batches), 'batch_size': int(batches.shape[1]),
               'trainable_encoder_keys': trainable,
               'frozen_parameter_keys': sorted(k for k in named if k not in tset),
               'grad_none_keys_step1': grad_none or [],
               'changed_keys': changed, 'unchanged_keys': sorted(k for k, v in diff.items() if v == 0.0),
               'buffer_keys': buffer_keys, 'buffers_unchanged': all(diff[k] == 0.0 for k in buffer_keys),
               'control_encoder_equal': (not candidate) and not changed,
               'exposure_total': int(counts.sum()), 'unique_rows_exposed': int((counts > 0).sum()),
               'head_max_abs_change': {'W': float((W - W0).abs().max()), 'b': float((b - b0).abs().max())},
               'optimizer': {'type': 'Adam', 'head_lr': HEAD_LR, 'encoder_lr': ENC_LR if candidate else None,
                             'betas': list(BETAS), 'eps': EPS, 'weight_decay': 0.0, 'amsgrad': False},
               'final_loss': per_step[-1]['loss'] if per_step else None,
               'seconds': time.time() - t0, 'per_step': per_step}
    return W, b, counts, summary


# ---------------- phases ----------------
def phase_preflight(ctx):
    R = ctx.R
    out = ctx.path('PREFLIGHT.json')
    ensure_absent([out])
    classes, rows, head = load_common(ctx)
    cache, ri, batches, counts = load_training(ctx, rows, classes)
    checks = {'schedule': {'shape': list(batches.shape), 'exposures': int(counts.sum()),
                           'rows_exposed': int((counts > 0).sum()), 'max_row_exposure': int(counts.max())}}
    arrays = preprocess_train(R, rows, cache, ri)
    checks['preprocessed_train_rows'] = len(arrays)
    tensors, audit = read_tensors(ctx)
    encs = {arm: R.build_encoder('pretrained', clone_tensors(tensors)) for arm in ARMS}
    h0 = R.state_hash(encs[CONTROL])
    require(R.state_hash(encs[CANDIDATE]) == h0 and states_equal(encs[CONTROL], encs[CANDIDATE]),
            'initial arm encoder states differ')
    enc = encs[CONTROL]
    spaced = np.linspace(0, N_TRAIN - 1, N_SPACED).round().astype(np.int64)
    probe = sorted(set(batches[0].tolist()) | set(spaced.tolist()))
    live, worst = {}, 0.0
    for k in probe:
        f = np.asarray(R.encode(enc, arrays[k]), dtype=np.float64).reshape(-1)
        c = cache['features'][k].astype(np.float64)
        require(f.shape == c.shape, 'feature shape mismatch at row %d' % k)
        worst = max(worst, float(np.abs(f - c).max()))
        require(np.allclose(f, c, **FEAT_TOL), 'live feature differs from cache at row %d' % k)
        live[k] = f
    checks['features'] = {'rows': probe, 'max_abs_diff': worst}
    b0 = batches[0]
    y0 = cache['y'][b0].astype(np.int64)
    s_live = np.asarray(R.predict_scores(np.stack([live[int(k)] for k in b0]), head), dtype=np.float64)
    s_cache = np.asarray(R.predict_scores(cache['features'][b0].astype(np.float64), head), dtype=np.float64)
    W, b, mean, std = head_params(head)
    with torch.no_grad():
        s_torch = head_logits(torch.from_numpy(cache['features'][b0]), W, b, mean, std).numpy()
    ce_live, _ = np_ce(s_live, y0)
    ce_cache, l_cache = np_ce(s_cache, y0)
    ce_torch, l_torch = np_ce(s_torch, y0)
    require(np.allclose(s_live, s_cache, **SCORE_TOL), 'step0 scores live vs cache differ')
    require(np.allclose(l_torch, l_cache, **SCORE_TOL), 'torch head differs from R.predict_scores')
    require(np.isclose(ce_live, ce_cache, **CE_TOL) and np.isclose(ce_torch, ce_cache, **CE_TOL), 'step0 CE differs')
    checks['step0'] = {'ce_live': ce_live, 'ce_cache': ce_cache, 'ce_torch': ce_torch,
                       'max_abs_score_diff_live_cache': float(np.abs(s_live - s_cache).max()),
                       'max_abs_logsoftmax_diff_torch_R': float(np.abs(l_torch - l_cache).max())}
    for k, t in zip(('W', 'b', 'mean', 'std'), (W, b, mean, std)):
        require(t.dtype == torch.float64 and np.array_equal(t.detach().numpy(), head[k]), 'head tensor not exact: ' + k)
    keys_c = set_trainable(encs[CANDIDATE], True)
    expect = sorted(n for n, _ in encs[CANDIDATE].named_parameters() if not is_fixed(n))
    require(keys_c == expect and len(keys_c) > 0, 'candidate must have all parameters trainable')
    require(set_trainable(encs[CONTROL], False) == [], 'control must have no trainable encoder parameters')
    checks['trainable'] = {'candidate_keys': keys_c, 'n_candidate': len(keys_c), 'control_keys': [],
                           'fixed_parameters': sorted(n for n, _ in encs[CANDIDATE].named_parameters() if is_fixed(n))}
    y_first = int(cache['y'][0])
    gp_c = gradient_probe(encs[CANDIDATE], head, arrays[0], y_first, True)
    gp_h = gradient_probe(encs[CONTROL], head, arrays[0], y_first, False)
    require(gp_c['reached_nonzero'] > 0 and gp_c['head_grad_nonzero'] and gp_c['state_unchanged'], 'candidate gradient probe failed')
    require(gp_h['reached_nonzero'] == 0 and gp_h['head_grad_nonzero'] and gp_h['state_unchanged'], 'control gradient probe failed')
    checks['gradient_probe'] = {'candidate': gp_c, 'control': gp_h}
    require(R.state_hash(encs[CONTROL]) == h0 and R.state_hash(encs[CANDIDATE]) == h0, 'state changed during preflight')
    record = {'pass': True, 'phase': 'preflight', 'identity': ctx.identity(), 'recipe': RECIPE,
              'tolerances': {'features': FEAT_TOL, 'scores': SCORE_TOL, 'ce': CE_TOL},
              'initial_state_hash': h0, 'encoder_audit': normalize(audit),
              'schedule_sha256': batch_sha(batches), 'train_inputs_digest': input_digest(cache),
              'checks': checks, 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
    write_json_x(out, record)
    print('PREFLIGHT PASS: ' + out)


def load_preflight(ctx):
    p = ctx.path('PREFLIGHT.json')
    require(os.path.isfile(p), 'PREFLIGHT.json missing; run --phase preflight first')
    with open(p, 'rb') as f:
        data = f.read()
    pre = json.loads(data.decode('utf-8'))
    require(pre.get('pass') is True, 'preflight did not pass')
    require(pre.get('identity') == ctx.identity(), 'config/code/pins changed since preflight')
    require(pre.get('recipe') == normalize(RECIPE), 'recipe changed since preflight')
    pre['_sha256'] = sha256_bytes(data)
    return pre


def arm_files(arm):
    return {'encoder_state': arm + '_encoder_state.npz', 'head': arm + '_head.npz',
            'training': arm + '_training.json', 'log': 'train_log_' + arm + '.txt'}


def phase_train(ctx):
    R = ctx.R
    pre = load_preflight(ctx)
    frozen_path = ctx.path('FROZEN_MODELS.json')
    ensure_absent([frozen_path] + [ctx.path(f) for arm in ARMS for f in arm_files(arm).values()])
    classes, rows, head = load_common(ctx)
    cache, ri, batches, counts = load_training(ctx, rows, classes)
    require(batch_sha(batches) == pre['schedule_sha256'], 'schedule differs from preflight')
    require(input_digest(cache) == pre['train_inputs_digest'], 'train inputs differ from preflight')
    arrays = preprocess_train(R, rows, cache, ri)
    tensors, _ = read_tensors(ctx)
    labels = cache['y'].astype(np.int64)
    cached = torch.from_numpy(np.ascontiguousarray(cache['features']))
    frozen_arms = {}
    for arm in ARMS:
        files = arm_files(arm)
        log = Logger(ctx.path(files['log']))
        try:
            enc = R.build_encoder('pretrained', clone_tensors(tensors))
            require(R.state_hash(enc) == pre['initial_state_hash'], 'initial encoder differs from preflight')
            norms_before = descriptive_norms(enc, arrays[:NORM_ROWS], *head_params(head))
            if arm == CANDIDATE:
                feature_fn = lambda pos, e=enc: e(torch.from_numpy(arrays[pos]))
            else:
                feature_fn = lambda pos: cached[pos:pos + 1]
            log('%s: start %d steps x %d clips, expected exposures %d' % (arm, len(batches), BATCH, EXPOSURES))
            W, b, used, summary = train_arm(arm, enc, head, batches, labels, feature_fn, log, N_TRAIN)
            require(np.array_equal(used, counts) and int(used.sum()) == EXPOSURES, 'actual exposure differs from schedule')
            set_trainable(enc, False)
            _, _, mean_t, std_t = head_params(head)
            norms_after = descriptive_norms(enc, arrays[:NORM_ROWS], W, b, mean_t, std_t)
            state = state_arrays(enc)
            state_hash = R.state_hash(enc)
            if arm == CONTROL:
                require(state_hash == pre['initial_state_hash'], 'control encoder state hash changed')
            sha_state = write_npz_x(ctx.path(files['encoder_state']), state)
            sha_head = write_npz_x(ctx.path(files['head']), {'W': W.numpy(), 'b': b.numpy(), 'mean': head['mean'],
                                                             'std': head['std'], 'classes': head['classes']})
            summary.update({'identity': ctx.identity(), 'preflight_sha256': pre['_sha256'], 'recipe': RECIPE,
                            'state_dtypes': {k: str(v.dtype) for k, v in state.items()},
                            'initial_state_hash': pre['initial_state_hash'], 'final_state_hash': state_hash,
                            'expected_exposures': EXPOSURES, 'norms_first32_before': norms_before,
                            'norms_first32_after': norms_after,
                            'files': {'encoder_state': sha_state, 'head': sha_head}})
            sha_train = write_json_x(ctx.path(files['training']), summary)
            frozen_arms[arm] = {'state_hash': state_hash,
                                'encoder_state': {'file': files['encoder_state'], 'sha256': sha_state},
                                'head': {'file': files['head'], 'sha256': sha_head},
                                'training': {'file': files['training'], 'sha256': sha_train},
                                'log': {'file': files['log']}}
            log('%s: final checkpoint frozen (state %s)' % (arm, state_hash))
        except BaseException:
            log('FAILED: ' + traceback.format_exc())
            raise
        finally:
            log.close()
    frozen = {'complete': True, 'identity': ctx.identity(), 'classes': classes, 'recipe': RECIPE,
              'preflight_sha256': pre['_sha256'], 'initial_state_hash': pre['initial_state_hash'],
              'schedule_sha256': pre['schedule_sha256'], 'train_inputs_digest': pre['train_inputs_digest'],
              'arms': frozen_arms}
    write_json_x(frozen_path, frozen)
    print('FROZEN: ' + frozen_path)


def validate_prior(prior, rows, classes):
    need = ('scores_pretrained', 'scores_baseline', 'y', 'classes') + META_KEYS + DIAG_KEYS
    require(all(k in prior for k in need), 'prior_scores keys missing')
    require([str(c) for c in prior['classes'].tolist()] == classes, 'prior classes differ')
    eval_idx = [i for i, r in enumerate(rows) if r['group'] not in TRAIN_SIZES]
    sizes = {}
    for i in eval_idx:
        sizes[rows[i]['group']] = sizes.get(rows[i]['group'], 0) + 1
    require(sizes == EVAL_SIZES, 'evaluation group sizes differ: %s' % sizes)
    order = [int(v) for v in prior['row_index'].tolist()]
    require(len(order) == N_EVAL and len(set(order)) == N_EVAL and sorted(order) == eval_idx,
            'prior row_index differs from evaluation rows')
    for k, i in enumerate(order):
        r = rows[i]
        require(str(prior['ids'][k]) == str(r['id']) and str(prior['groups'][k]) == str(r['group'])
                and str(prior['conditions'][k]) == str(r['condition']) and int(prior['y'][k]) == int(r['y']),
                'prior identity mismatch at %d' % k)
    return order


def check_scores(s, name, n=N_EVAL):
    s = np.asarray(s, dtype=np.float64)
    require(s.shape == (n, N_CLASSES) and np.isfinite(s).all(), 'bad scores: ' + name)
    return s


def validated_subset_indices(sel, n):
    """Validate a subset selector given as a 1D integer index array.

    Requirements: 1D, integer dtype (bool rejected), every index in [0, n),
    no duplicates. Order is preserved and empty selectors are accepted.
    Returns a new int64 array.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError('n must be an integer, got ' + repr(type(n)))
    n = int(n)
    if n < 0:
        raise ValueError('n must be non-negative, got ' + str(n))
    arr = np.asarray(sel)
    if arr.ndim != 1:
        raise ValueError('subset indices must be 1D, got shape ' + str(arr.shape))
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError('subset indices must have integer dtype (bool not allowed), got ' + str(arr.dtype))
    if arr.size:
        lo = int(arr.min())
        hi = int(arr.max())
        if lo < 0 or hi >= n:
            raise ValueError('subset indices out of range [0, ' + str(n) + '): min=' + str(lo) + ' max=' + str(hi))
        if np.unique(arr).size != arr.size:
            raise ValueError('subset indices contain duplicates')
    return arr.astype(np.int64, copy=True)


def whole_argmax(scores):
    return np.argmax(scores, axis=1).astype(np.int64)  # first (lowest) index wins ties


def phase_evaluate(ctx):
    R = ctx.R
    out_scores, out_metrics = ctx.path('eval_scores.npz'), ctx.path('eval_metrics.json')
    ensure_absent([out_scores, out_metrics])
    fpath = ctx.path('FROZEN_MODELS.json')
    require(os.path.isfile(fpath), 'FROZEN_MODELS.json missing; train both arms first')
    with open(fpath, 'rb') as f:
        fraw = f.read()
    frozen = json.loads(fraw.decode('utf-8'))
    require(frozen.get('complete') is True and frozen.get('identity') == ctx.identity(), 'frozen identity mismatch')
    require(sorted(frozen.get('arms', {})) == sorted(ARMS), 'frozen arms incomplete')
    require(sha256_file(ctx.path('PREFLIGHT.json')) == frozen['preflight_sha256'], 'PREFLIGHT.json changed')
    classes, rows, head = load_common(ctx)
    require(frozen['classes'] == classes, 'classes differ from frozen')
    tensors, _ = read_tensors(ctx)
    encs, heads = {}, {}
    for arm in ARMS:
        rec = frozen['arms'][arm]
        for kind in ('encoder_state', 'head', 'training'):
            require(sha256_file(ctx.path(rec[kind]['file'])) == rec[kind]['sha256'], 'frozen hash mismatch: %s %s' % (arm, kind))
        enc = R.build_encoder('pretrained', clone_tensors(tensors))
        st = rec['encoder_state']
        load_state_arrays(enc, npz_from_bytes(read_verified(ctx.path(st['file']), st['sha256'])))
        set_trainable(enc, False)
        require(R.state_hash(enc) == rec['state_hash'], 'loaded state hash mismatch: ' + arm)
        hd = rec['head']
        h = validate_head(npz_from_bytes(read_verified(ctx.path(hd['file']), hd['sha256'])), classes)
        require(np.array_equal(h['mean'], head['mean']) and np.array_equal(h['std'], head['std']), 'standardization changed: ' + arm)
        encs[arm], heads[arm] = enc, h
    require(frozen['arms'][CONTROL]['state_hash'] == frozen['initial_state_hash'], 'control encoder is not the frozen parent')
    prior = npz_from_bytes(ctx.blob('prior_scores'))
    order = validate_prior(prior, rows, classes)
    feats = {arm: [] for arm in ARMS}
    for k, i in enumerate(order):
        x, diag = R.preprocess_row(rows[i], index=i)
        require(R.array_sha(x) == diag['input_array_sha256'] == str(prior['input_array_sha256'][k]), 'eval input hash mismatch at %d' % k)
        require(str(diag['source_sha256']) == str(prior['source_sha256'][k])
                and int(diag['frame_count']) == int(prior['frame_count'][k])
                and int(diag['zero_frame_count']) == int(prior['zero_frame_count'][k])
                and np.isclose(float(diag['scale']), float(prior['scale'][k]), rtol=1e-6, atol=0.0),
                'eval diagnostics mismatch at %d' % k)
        for arm in ARMS:
            feats[arm].append(np.asarray(R.encode(encs[arm], x), dtype=np.float64).reshape(-1))
    scores = {arm: check_scores(R.predict_scores(np.stack(feats[arm]), heads[arm]), arm) for arm in ARMS}
    scores['baseline'] = check_scores(prior['scores_baseline'], 'baseline')
    scores['frozen_parent'] = check_scores(prior['scores_pretrained'], 'frozen_parent')
    rescore = check_scores(R.predict_scores(np.stack(feats[CONTROL]), head), 'parent_rescore')
    l_new, l_old = np_log_softmax(rescore), np_log_softmax(scores['frozen_parent'])
    parent_diff = float(np.abs(l_new - l_old).max())
    require(np.allclose(l_new, l_old, **SCORE_TOL), 'frozen parent scores not reproduced')
    y = prior['y'].astype(np.int64)
    out = {'scores_' + a: scores[a] for a in EVAL_ARMS}
    out.update({'y': y, 'classes': np.asarray(classes)})
    for k in META_KEYS + DIAG_KEYS:
        out[k] = prior[k]
    sha_scores = write_npz_x(out_scores, out)
    preds = {a: whole_argmax(scores[a]) for a in EVAL_ARMS}
    meta = {k: prior[k] for k in META_KEYS}
    subsets = R.build_subsets(meta)
    per = {}
    for name in sorted(subsets):
        sel = validated_subset_indices(subsets[name], N_EVAL)
        n = int(sel.size)
        entry = {'n': n}
        if n:
            entry['metrics'] = {a: R.summarize(preds[a][sel], y[sel], classes) for a in EVAL_ARMS}
            entry['paired'] = {'full_encoder_vs_' + o: R.compare(preds['full_encoder'][sel], preds[o][sel], y[sel],
                                                                 meta['row_index'][sel], meta['ids'][sel])
                               for o in ('head_only', 'baseline', 'frozen_parent')}
        per[str(name)] = entry
    report = {'identity': ctx.identity(), 'frozen_models_sha256': sha256_bytes(fraw), 'eval_scores_sha256': sha_scores,
              'n_rows': N_EVAL, 'arms': list(EVAL_ARMS), 'prediction': 'whole-93 argmax, lowest index on ties',
              'frozen_parent_reproduction_max_abs_logsoftmax_diff': parent_diff, 'subsets': per,
              'aggregation': 'none; subsets overlap so no aggregate is reported',
              'decision': 'none; metrics reported only, no automatic adoption'}
    write_json_x(out_metrics, report)
    print('EVALUATED: ' + out_metrics)


def record_failure(out, phase, exc):
    try:
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, 'FAILURE_%s_%d.json' % (phase, time.time_ns()))
        write_json_x(path, {'pass': False, 'phase': phase, 'error_type': type(exc).__name__,
                            'error': str(exc), 'traceback': traceback.format_exc()})
        print('FAILED (%s): %s; details in %s' % (phase, exc, path), file=sys.stderr)
    except Exception as e2:
        print('could not record failure: %r' % (e2,), file=sys.stderr)


PHASES = {'preflight': phase_preflight, 'train': phase_train, 'evaluate': phase_evaluate}


def main(argv=None):
    ap = argparse.ArgumentParser(description='Paired head_only vs full_encoder fine-tuning (fixed recipe).')
    ap.add_argument('--phase', required=True, choices=sorted(PHASES))
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prior-dir', required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    try:
        PHASES[args.phase](Ctx(args))
    except BaseException as exc:
        record_failure(args.out, args.phase, exc)
        raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
