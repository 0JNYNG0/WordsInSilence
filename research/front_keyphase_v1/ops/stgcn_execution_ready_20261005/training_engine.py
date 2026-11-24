'''Small generic CPU PyTorch training engine for a research classifier.

Contracts
- model: externally built torch.nn.Module; forward(x[B,6,2,10,27]) returns
  float32 logits[B,num_classes]; all parameters are trainable float32 CPU tensors.
- dataset: dict(x=float32[N,6,2,10,27], y=int64[N], weight=float64[N] >= 0 (finite, positive sum),
  ids=list[str] unique, origins=list[str]) plus optional bool primary[N] on
  validation data. Source tensors are never modified; samples are cloned.
- seeding: epoch_seed(seed, epoch) = (seed + epoch) mod 2**32 with 1-based
  epoch. At the START of every training epoch Python random, NumPy's global RNG
  and torch's CPU default generator are seeded with it (augmentation, dropout),
  and a separate torch.Generator seeded with the same value draws the weighted
  sample indices, so sampling never depends on how much RNG augmentation or
  dropout consume. No RNG state crosses epochs, so resuming at an epoch
  boundary reproduces an uninterrupted run without serializing RNG objects.
- learning rate: base_lr * 0.5 * (1 + cos(pi * (epoch - 1) / schedule_epochs)),
  identical in select and refit modes.
- checkpoints: in-memory dicts of cloned tensors and plain Python values. They
  must be treated as read-only (best and last share tensors when they coincide).
  Nothing is read from or written to disk, and only the CPU is used.
'''
import copy
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = 'generic_cpu_v1'
CHECKPOINT_FORMAT = 'generic_cpu_v1_checkpoint'
SAMPLE_SHAPE = (6, 2, 10, 27)
CONFIG_KEYS = frozenset({
    'num_classes', 'batch_size', 'max_epochs', 'min_epochs', 'patience',
    'base_lr', 'schedule_epochs', 'seed', 'num_samples', 'schema'})
_CONTEXT_KEYS = ('cfg_sha256', 'mode', 'fixed_epochs', 'train_sha256', 'val_sha256')


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(cfg, key, lo=None, hi=None):
    value = cfg[key]
    if not _is_int(value):
        raise TypeError(f'config {key} must be an int, got {type(value).__name__}')
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ValueError(f'config {key}={value} outside [{lo}, {hi}]')
    return value


def validate_config(cfg):
    '''Strictly validate cfg and return a normalized plain copy.'''
    if not isinstance(cfg, dict):
        raise TypeError('config must be a dict')
    keys = set(cfg)
    if keys != CONFIG_KEYS:
        missing = sorted(map(repr, CONFIG_KEYS - keys))
        extra = sorted(map(repr, keys - CONFIG_KEYS))
        raise ValueError(f'config keys mismatch: missing={missing} extra={extra}')
    if not isinstance(cfg['schema'], str) or cfg['schema'] != SCHEMA:
        raise ValueError(f'config schema must be {SCHEMA!r}')
    _require_int(cfg, 'num_classes', 2)
    batch_size = _require_int(cfg, 'batch_size', 2)
    max_epochs = _require_int(cfg, 'max_epochs', 1)
    _require_int(cfg, 'min_epochs', 1, max_epochs)
    _require_int(cfg, 'patience', 1)
    _require_int(cfg, 'schedule_epochs', max_epochs)
    _require_int(cfg, 'seed')
    if _require_int(cfg, 'num_samples', 1) % batch_size:
        raise ValueError('config num_samples must be divisible by batch_size')
    base_lr = cfg['base_lr']
    if isinstance(base_lr, bool) or not isinstance(base_lr, (int, float)):
        raise TypeError('config base_lr must be a number')
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError('config base_lr must be positive and finite')
    out = dict(cfg)
    out['base_lr'] = float(base_lr)
    return out


def _config_hash(cfg):
    text = json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def learning_rate(epoch, cfg):
    '''Single-cycle cosine LR for 1-based epoch; epochs past schedule_epochs are rejected.'''
    cfg = validate_config(cfg)
    total = cfg['schedule_epochs']
    if not _is_int(epoch) or not 1 <= epoch <= total:
        raise ValueError(f'epoch must be an int in [1, {total}], got {epoch!r}')
    return cfg['base_lr'] * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / total))


def epoch_seed(seed, epoch):
    '''Seed for 1-based epoch: (seed + epoch) mod 2**32.'''
    return (seed + epoch) % (2 ** 32)


def _seed_globals(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.default_generator.manual_seed(seed)  # CPU generator only; no device enumeration


def _check_tensor(name, value, dtype, shape):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor')
    if value.dtype != dtype:
        raise TypeError(f'{name} must be {dtype}, got {value.dtype}')
    if value.device.type != 'cpu' or value.layout != torch.strided:
        raise ValueError(f'{name} must be a dense CPU tensor')
    if value.requires_grad:
        raise ValueError(f'{name} must not require grad')
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f'{name} shape {tuple(value.shape)} != {tuple(shape)}')
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} contains non-finite values')


def _check_str_list(name, value, n):
    if type(value) is not list or len(value) != n or not all(isinstance(s, str) for s in value):
        raise TypeError(f'{name} must be a list of {n} str')


def validate_dataset(data, num_classes, validation=False):
    '''Strictly validate a dataset dict without modifying it; returns the row count.'''
    if not _is_int(num_classes) or num_classes < 2:
        raise ValueError('num_classes must be an int >= 2')
    if not isinstance(data, dict):
        raise TypeError('dataset must be a dict')
    required = {'x', 'y', 'weight', 'ids', 'origins'}
    allowed = (required | {'primary'}) if validation else required
    keys = set(data)
    if not required <= keys or not keys <= allowed:
        raise ValueError(f'dataset keys {sorted(map(str, keys))} invalid; allowed {sorted(allowed)}')
    x = data['x']
    if not isinstance(x, torch.Tensor) or x.dim() != 5:
        raise TypeError('x must be a 5-D tensor [N,6,2,10,27]')
    n = x.shape[0]
    if n < 1:
        raise ValueError('dataset must be nonempty')
    _check_tensor('x', x, torch.float32, (n,) + SAMPLE_SHAPE)
    _check_tensor('y', data['y'], torch.int64, (n,))
    if int(data['y'].min()) < 0 or int(data['y'].max()) >= num_classes:
        raise ValueError('labels outside [0, num_classes)')
    _check_tensor('weight', data['weight'], torch.float64, (n,))
    if (not bool((data['weight'] >= 0).all()) or not math.isfinite(float(data['weight'].sum()))
            or not float(data['weight'].sum()) > 0):
        raise ValueError('weights must be finite and nonnegative with a positive finite sum')
    _check_str_list('ids', data['ids'], n)
    _check_str_list('origins', data['origins'], n)
    if len(set(data['ids'])) != n:
        raise ValueError('row ids must be unique')
    if 'primary' in data:
        _check_tensor('primary', data['primary'], torch.bool, (n,))
        if not bool(data['primary'].any()):
            raise ValueError('primary selection must be nonempty')
    return n


def _dataset_signature(data):
    digest = hashlib.sha256()
    for key in ('x', 'y', 'weight', 'primary'):
        if key in data:
            value = data[key]
            digest.update(f'{key}|{value.dtype}|{tuple(value.shape)}|'.encode('utf-8'))
            digest.update(value.detach().contiguous().numpy().tobytes())
    for key in ('ids', 'origins'):
        digest.update(f'{key}|'.encode('utf-8'))
        digest.update(json.dumps(data[key]).encode('utf-8'))
    return digest.hexdigest()


def _check_model(model):
    if not isinstance(model, torch.nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    if not list(model.parameters()):
        raise ValueError('model has no parameters')
    for name, param in model.named_parameters():
        if param.dtype != torch.float32 or param.device.type != 'cpu' or not param.requires_grad:
            raise ValueError(f'parameter {name} must be a trainable float32 CPU tensor')
    for name, buffer in model.named_buffers():
        if buffer.device.type != 'cpu':
            raise ValueError(f'buffer {name} must be on CPU')


def _check_logits(logits, batch, num_classes):
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32
            or tuple(logits.shape) != (batch, num_classes)):
        raise ValueError(f'model must return float32 logits of shape {(batch, num_classes)}')


def _clone_state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _evaluate(model, data, cfg):
    modes = [(module, module.training) for module in model.modules()]
    x, y = data['x'], data['y']
    n = y.shape[0]
    primary = data['primary'] if 'primary' in data else torch.ones(n, dtype=torch.bool)
    batch_size, num_classes = cfg['batch_size'], cfg['num_classes']
    chunks = []
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                xb = x[start:start + batch_size].clone()  # a final batch of 1 is fine in eval
                logits = model(xb)
                _check_logits(logits, xb.shape[0], num_classes)
                chunks.append(logits.detach().clone())
    finally:
        for module, was_training in modes:
            module.training = was_training
    logits = torch.cat(chunks).double()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('non-finite validation logits')
    ce = F.cross_entropy(logits, y, reduction='none')
    hit = logits.argmax(dim=1) == y
    labels, hits = y[primary], hit[primary]
    recalls = [float(hits[labels == c].double().mean()) for c in torch.unique(labels).tolist()]
    count, correct, all_correct = int(primary.sum()), int(hits.sum()), int(hit.sum())
    return {
        'primary_count': count,
        'primary_correct': correct,
        'primary_accuracy': correct / count,
        'primary_macro_recall': sum(recalls) / len(recalls),
        'primary_ce': float(ce[primary].mean()),
        'all_count': n,
        'all_correct': all_correct,
        'all_top1': all_correct / n,
    }


def evaluate(model, data, cfg):
    '''Evaluate without augmentation in eval mode; restores every module's mode.'''
    cfg = validate_config(cfg)
    _check_model(model)
    validate_dataset(data, cfg['num_classes'], validation=True)
    return _evaluate(model, data, cfg)


def _augment_one(augment, sample):
    out = augment(sample)
    if (not isinstance(out, torch.Tensor) or out.dtype != torch.float32
            or out.device.type != 'cpu' or tuple(out.shape) != SAMPLE_SHAPE):
        raise ValueError(f'augment must return a float32 CPU tensor of shape {SAMPLE_SHAPE}')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('augment returned non-finite values')
    return out.detach()


def _train_epoch(model, optimizer, train, cfg, epoch, augment):
    seed = epoch_seed(cfg['seed'], epoch)
    _seed_globals(seed)
    lr = learning_rate(epoch, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    sampler = torch.Generator(device='cpu')
    sampler.manual_seed(seed)
    weight = train['weight']
    draws = torch.multinomial(weight / weight.sum(), cfg['num_samples'],
                              replacement=True, generator=sampler)
    x, y = train['x'], train['y']
    batch_size, num_classes = cfg['batch_size'], cfg['num_classes']
    model.train()
    ce_sum, correct, updates, checked, missing = 0.0, 0, 0, 0, 0
    for start in range(0, draws.shape[0], batch_size):
        rows = draws[start:start + batch_size]
        samples = []
        for row in rows.tolist():  # sequential draw order, one call per occurrence
            sample = x[row].clone()
            samples.append(sample if augment is None else _augment_one(augment, sample))
        xb, yb = torch.stack(samples), y[rows]
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        _check_logits(logits, batch_size, num_classes)
        loss = F.cross_entropy(logits, yb)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'non-finite training loss at epoch {epoch}, update {updates + 1}')
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                missing += 1
            elif not bool(torch.isfinite(param.grad).all()):
                raise FloatingPointError(f'non-finite gradient for {name} at epoch {epoch}, update {updates + 1}')
            else:
                checked += 1
        optimizer.step()
        updates += 1
        with torch.no_grad():
            detached = logits.detach().double()
            ce_sum += float(F.cross_entropy(detached, yb, reduction='sum'))
            correct += int((detached.argmax(dim=1) == yb).sum())
    count = draws.shape[0]
    return {
        'epoch': epoch, 'epoch_seed': seed, 'lr': lr, 'updates': updates,
        'train_count': count, 'train_correct': correct,
        'train_accuracy': correct / count, 'train_ce': ce_sum / count,
        'sample_counts': torch.bincount(draws, minlength=x.shape[0]).tolist(),
        'sample_sha256': hashlib.sha256(draws.numpy().astype('<i8').tobytes()).hexdigest(),
        'grad_finite_checks': checked, 'grad_missing': missing,
    }


def _selection_key(metrics):
    return (-metrics['primary_correct'], metrics['primary_ce'])


def _check_resume(resume, ctx, cfg, model, final_epoch):
    '''Validate a resume checkpoint without touching the model; returns (best, stale).'''
    if not isinstance(resume, dict) or resume.get('format') != CHECKPOINT_FORMAT or resume.get('kind') != 'last':
        raise ValueError('resume must be a last_checkpoint produced by run_training')
    saved_cfg = resume.get('cfg')
    if not isinstance(saved_cfg, dict) or saved_cfg != cfg or _config_hash(saved_cfg) != resume.get('cfg_sha256'):
        raise ValueError('resume cfg differs from this run or from its recorded sha256')
    for key in _CONTEXT_KEYS:
        if resume.get(key) != ctx[key]:
            raise ValueError(f'resume {key} does not match this run')
    done = resume.get('completed_epoch')
    if not _is_int(done) or not 1 <= done < final_epoch or resume.get('stop_reason') is not None:
        raise ValueError('resume checkpoint is not an unfinished epoch boundary of this run')
    history = resume.get('history')
    epochs = [r.get('epoch') for r in history if isinstance(r, dict)] if isinstance(history, list) else None
    if epochs != list(range(1, done + 1)):
        raise ValueError('resume history does not match completed_epoch')
    state, current = resume.get('model_state'), model.state_dict()
    if not isinstance(state, dict) or set(state) != set(current):
        raise ValueError('resume model_state keys do not match the model')
    for key, value in current.items():
        saved = state[key]
        if (not isinstance(saved, torch.Tensor) or saved.shape != value.shape
                or saved.dtype != value.dtype or saved.device.type != 'cpu'):
            raise ValueError(f'resume model_state[{key!r}] is incompatible with the model')
    optimizer_state = resume.get('optimizer_state')
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {'state', 'param_groups'}:
        raise ValueError('resume optimizer_state is malformed')
    best, best_epoch, stale = resume.get('best_snapshot'), resume.get('best_epoch'), resume.get('stale')
    if ctx['mode'] == 'refit':
        if best is not None or best_epoch is not None or stale is not None:
            raise ValueError('refit checkpoints carry no best snapshot or stale counter')
        return None, 0
    if (not isinstance(best, dict) or best.get('kind') != 'best' or not _is_int(best_epoch)
            or not 1 <= best_epoch <= done or best.get('completed_epoch') != best_epoch
            or best.get('best_metrics') != resume.get('best_metrics')
            or any(best.get(key) != ctx[key] for key in _CONTEXT_KEYS)):
        raise ValueError('resume best snapshot is inconsistent')
    if stale != done - best_epoch:
        raise ValueError('resume stale counter is inconsistent with best_epoch')
    return best, stale


def run_training(model, train, val, cfg, augment=None, *, mode='select',
                 fixed_epochs=None, on_epoch=None, resume=None):
    '''Train model in place with a fresh Adam optimizer.

    mode='select' evaluates val after every epoch and keeps the best snapshot by
    (higher primary_correct, lower primary_ce, earlier epoch); macro recall and
    all-val metrics are logged only. mode='refit' trains exactly fixed_epochs
    epochs without evaluation or early stopping. on_epoch(last_checkpoint,
    record) runs after each epoch and its exceptions propagate. resume accepts a
    last_checkpoint of the identical run contract and continues after it.
    '''
    cfg = validate_config(cfg)
    _check_model(model)
    num_classes = cfg['num_classes']
    validate_dataset(train, num_classes)
    if mode == 'select':
        if fixed_epochs is not None:
            raise ValueError('fixed_epochs must be None in select mode')
        if val is None:
            raise ValueError('select mode requires validation data')
        validate_dataset(val, num_classes, validation=True)
        overlap = set(train['origins']) & set(val['origins'])
        if overlap:
            raise ValueError(f'{len(overlap)} origins shared by fit and validation, e.g. {sorted(overlap)[:3]}')
        final_epoch = cfg['max_epochs']
    elif mode == 'refit':
        if val is not None:
            raise ValueError('refit mode requires val=None')
        if not _is_int(fixed_epochs) or not 1 <= fixed_epochs <= cfg['max_epochs']:
            raise ValueError('refit mode requires an int fixed_epochs in [1, max_epochs]')
        final_epoch = fixed_epochs
    else:
        raise ValueError(f'unknown mode {mode!r}')
    if augment is not None and not callable(augment):
        raise TypeError('augment must be callable or None')
    if on_epoch is not None and not callable(on_epoch):
        raise TypeError('on_epoch must be callable or None')
    ctx = {
        'format': CHECKPOINT_FORMAT, 'mode': mode, 'fixed_epochs': fixed_epochs,
        'cfg_sha256': _config_hash(cfg),
        'train_sha256': _dataset_signature(train),
        'val_sha256': None if val is None else _dataset_signature(val),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate(1, cfg), weight_decay=0.0)
    history, best, stale, start = [], None, 0, 1
    if resume is not None:
        best, stale = _check_resume(resume, ctx, cfg, model, final_epoch)
        optimizer.load_state_dict(copy.deepcopy(resume['optimizer_state']))
        model.load_state_dict(resume['model_state'])
        history = copy.deepcopy(resume['history'])
        start = resume['completed_epoch'] + 1

    stop_reason, last = None, None
    for epoch in range(start, final_epoch + 1):
        started = time.perf_counter()
        record = _train_epoch(model, optimizer, train, cfg, epoch, augment)
        record.update(val=None, is_best=False, stale=None)
        if mode == 'select':
            metrics = _evaluate(model, val, cfg)
            record['val'] = metrics
            if best is None or _selection_key(metrics) < _selection_key(best['best_metrics']):
                record['is_best'] = True
                stale = 0
            else:
                stale += 1
            record['stale'] = stale
            if epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
                stop_reason = 'patience'
            elif epoch == final_epoch:
                stop_reason = 'budget_limited_improving' if record['is_best'] else 'max_epochs'
        elif epoch == final_epoch:
            stop_reason = 'fixed_epochs'
        record['elapsed_seconds'] = time.perf_counter() - started
        history.append(record)
        snapshot = dict(
            ctx, cfg=dict(cfg), completed_epoch=epoch,
            model_state=_clone_state(model),
            optimizer_state=copy.deepcopy(optimizer.state_dict()),
            history=copy.deepcopy(history), stale=record['stale'],
            stop_reason=stop_reason, best_snapshot=None)
        if record['is_best']:
            best = dict(snapshot, kind='best', best_epoch=epoch,
                        best_metrics=copy.deepcopy(record['val']))
        last = dict(snapshot, kind='last',
                    best_epoch=None if best is None else best['best_epoch'],
                    best_metrics=None if best is None else copy.deepcopy(best['best_metrics']),
                    best_snapshot=best)
        if on_epoch is not None:
            on_epoch(last, copy.deepcopy(record))
        if stop_reason is not None:
            break
    return {
        'history': copy.deepcopy(history),
        'best_epoch': None if best is None else best['best_epoch'],
        'best_checkpoint': best,
        'last_checkpoint': last,
        'stop_reason': stop_reason,
    }
