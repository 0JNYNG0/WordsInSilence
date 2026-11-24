"""Minimal select-then-refit lifecycle around the generic training engine.

Files are only read back if this module wrote them inside output_dir and their
SHA-256 matches a companion <name>.meta.json bound to the caller's context hash
and run hash. The hashes detect corruption, interrupted writes and mismatched
runs; they are not a signature against anyone who can write to output_dir.
"""
import collections
import hashlib
import io
import json
import os
import time

import numpy as np
import torch

from training_engine import check_window_valid, run_training

INIT_SELECT = 'INIT_SELECT.json'
INIT_REFIT = 'INIT_REFIT.json'
SELECT_LAST = 'select_last.pt'
SELECT_BEST = 'select_best.pt'
SELECT_RESULT = 'SELECT_RESULT.json'
REFIT_LAST = 'refit_last.pt'
FINAL_MODEL = 'final_model.npz'
FROZEN_MODEL = 'FROZEN_MODEL.json'
REFIT_RESULT = 'REFIT_RESULT.json'
PROGRESS = 'progress.json'
_MAX_EPOCHS_KEY = 'max_epochs'  # cfg key holding the select-phase epoch cap
_RESULT_KEYS = ('history', 'best_epoch', 'best_checkpoint', 'last_checkpoint', 'stop_reason')
_SCALAR_TYPES = (str, int, float, bool, type(None))
_DICT_TYPES = (dict, collections.OrderedDict)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _json_bytes(obj):
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False).encode('utf-8')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path, data):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _write_signed(path, data, binding):
    _atomic_write(path, data)
    meta = {'file': os.path.basename(path), 'sha256': _sha(data), 'size': len(data), **binding}
    _atomic_write(path + '.meta.json', _json_bytes(meta))
    return meta['sha256']


def _read_signed(path, context_sha, run_sha=None):
    """Return the bytes of a file this module wrote, after verifying its companion."""
    with open(path + '.meta.json', 'rb') as handle:
        meta = json.loads(handle.read())
    if meta.get('file') != os.path.basename(path) or meta.get('context_sha256') != context_sha:
        raise ValueError(f'{path}: companion metadata does not match this context')
    if run_sha is not None and meta.get('run_sha256') != run_sha:
        raise ValueError(f'{path}: written for a different configuration or dataset')
    with open(path, 'rb') as handle:
        data = handle.read()
    if _sha(data) != meta.get('sha256'):
        raise ValueError(f'{path}: SHA-256 mismatch')
    return data


def context_sha256(context):
    """Hash of a plain-JSON, string-keyed context dict (never modified)."""
    if not isinstance(context, dict):
        raise TypeError('context must be a dict')
    text = _canon(context)
    if json.loads(text) != context:
        raise TypeError('context must be plain JSON: string keys, lists, finite numbers')
    return _sha(text.encode('utf-8'))


def _update_tensor(digest, name, tensor):
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    digest.update(_canon([name, array.dtype.str, list(array.shape)]).encode('utf-8'))
    digest.update(array.tobytes())


def state_digest(state):
    """SHA-256 over name, dtype, shape and contiguous bytes of every tensor."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if not isinstance(state[name], torch.Tensor):
            raise TypeError(f'state entry {name!r} is not a tensor')
        _update_tensor(digest, name, state[name])
    return digest.hexdigest()


def _dataset_digest(data):
    digest = hashlib.sha256()
    for key in sorted(data):
        value = data[key]
        if isinstance(value, torch.Tensor):
            _update_tensor(digest, key, value)
        elif isinstance(value, list) and all(isinstance(v, str) for v in value):
            digest.update(_canon([key, 'str_list', value]).encode('utf-8'))
        else:
            raise TypeError(f'unsupported dataset field {key!r}')
    return digest.hexdigest()


def _check_safe(obj, where='checkpoint'):
    # Only types torch.load(weights_only=True) rebuilds without extra allowlists.
    if isinstance(obj, torch.Tensor):
        if obj.device.type != 'cpu':
            raise ValueError(f'{where}: tensor is not on CPU')
    elif type(obj) in (list, tuple):
        for i, item in enumerate(obj):
            _check_safe(item, f'{where}[{i}]')
    elif type(obj) in _DICT_TYPES:
        for key, item in obj.items():
            if type(key) not in (str, int):
                raise TypeError(f'{where}: unsupported key {key!r}')
            _check_safe(item, f'{where}[{key!r}]')
    elif type(obj) not in _SCALAR_TYPES:
        raise TypeError(f'{where}: unsupported type {type(obj).__name__}')


def _save_checkpoint(path, checkpoint, binding):
    _check_safe(checkpoint)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    return _write_signed(path, buffer.getvalue(), binding)


def _prev_path(path):
    return path + '.prev'


def _try_read(path, context_sha, run_sha):
    try:
        return _read_signed(path, context_sha, run_sha)
    except (OSError, ValueError, AttributeError):
        return None


def _read_last(path, context_sha, run_sha):
    """Verified bytes of an own *_last.pt, else of its verified previous generation."""
    data = _try_read(path, context_sha, run_sha)
    if data is None:
        data = _try_read(_prev_path(path), context_sha, run_sha)
        if data is not None:
            print(f'recovery: {os.path.basename(path)} failed verification; '
                  'using its verified previous generation', flush=True)
    return data


def _save_last(path, checkpoint, binding):
    # Copy a verified current pair to <name>.prev (payload, then metadata) before
    # replacing it, so one valid pair always exists. A corrupt current never
    # overwrites the backup.
    current = _try_read(path, binding['context_sha256'], binding['run_sha256'])
    if current is not None:
        _write_signed(_prev_path(path), current, binding)
    return _save_checkpoint(path, checkpoint, binding)


def _resume_state(path, resume, context_sha, run_sha):
    prev = _prev_path(path)
    candidates = (path, path + '.meta.json', prev, prev + '.meta.json')
    if not resume or not any(os.path.exists(p) for p in candidates):
        return None
    data = _read_last(path, context_sha, run_sha)  # verified before deserializing
    if data is None:
        raise ValueError(f'{path}: neither the checkpoint nor its previous generation verifies')
    return torch.load(io.BytesIO(data), weights_only=True, map_location='cpu')


def _check_finite_state(state, what):
    for name, tensor in state.items():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{what}: non-finite values in {name}')


def _fresh_model(factory):
    model = factory()
    if not isinstance(model, torch.nn.Module):
        raise TypeError('factory must return a torch.nn.Module')
    state = model.state_dict()
    if any(t.device.type != 'cpu' for t in state.values()):
        raise ValueError('factory must return a CPU model')
    return model, state_digest(state)


def _check_cfgs(cfg, refit_cfg):
    if not isinstance(cfg, dict) or not isinstance(refit_cfg, dict):
        raise TypeError('cfg and refit_cfg must be dicts')
    if set(cfg) != set(refit_cfg) or any(cfg[k] != refit_cfg[k] for k in cfg if k != 'num_samples'):
        raise ValueError('refit_cfg must equal cfg except for num_samples')


def _index(data, name):
    ids, origins = data.get('ids'), data.get('origins')
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or len(set(ids)) != len(ids):
        raise ValueError(f'{name}: ids must be a list of unique strings')
    if not isinstance(origins, list) or len(origins) != len(ids):
        raise ValueError(f'{name}: origins must be a list matching ids')
    for key in ('x', 'y', 'weight'):
        value = data.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != len(ids):
            raise ValueError(f'{name}: {key} must be a tensor with one row per id')
    return {sample_id: row for row, sample_id in enumerate(ids)}


def _check_splits(train, val, all_train):
    train_ix, val_ix = _index(train, 'train'), _index(val, 'val')
    all_ix = _index(all_train, 'all_train')
    if set(train_ix) & set(val_ix):
        raise ValueError('train and val share ids')
    if set(train_ix) | set(val_ix) != set(all_ix):
        raise ValueError('all_train ids must be exactly the union of train and val ids')
    if set(train['origins']) & set(val['origins']):
        raise ValueError('train and val share origins')
    for data in (train, val):
        if data['x'].dtype != all_train['x'].dtype or data['weight'].dtype != all_train['weight'].dtype:
            raise ValueError('train/val and all_train dtypes differ')
    has_mask = 'window_valid' in all_train
    for name, data in (('train', train), ('val', val)):
        if ('window_valid' in data) != has_mask:
            raise ValueError(f'{name} and all_train disagree on window_valid presence')
    if has_mask:
        for name, data in (('all_train', all_train), ('train', train), ('val', val)):
            check_window_valid(data['window_valid'], len(data['ids']), f'{name} window_valid')
    for name, data, index, same_weight in (('train', train, train_ix, True), ('val', val, val_ix, False)):
        for sample_id, row in index.items():
            other = all_ix[sample_id]
            same = (data['origins'][row] == all_train['origins'][other]
                    and int(data['y'][row]) == int(all_train['y'][other])
                    and torch.equal(data['x'][row], all_train['x'][other])
                    and (not has_mask or torch.equal(data['window_valid'][row],
                                                     all_train['window_valid'][other])))
            if same_weight:
                same = same and torch.equal(data['weight'][row], all_train['weight'][other])
            if not same:
                raise ValueError(f'{name} id {sample_id!r} differs from all_train')


def _unpack(result):
    if isinstance(result, dict):
        return {key: result[key] for key in _RESULT_KEYS}
    if isinstance(result, tuple) and len(result) == len(_RESULT_KEYS):
        return dict(zip(_RESULT_KEYS, result))
    raise TypeError('unexpected run_training result')


def _selected_epoch(value):
    if type(value) is not int or value < 1:
        raise ValueError(f'invalid selected epoch {value!r}')
    return value


def _terminal_checkpoint(state, mode, cfg, final_epoch):
    """Return a verified checkpoint if it is terminal for this phase, None if unfinished."""
    if state is None:
        return None
    if not isinstance(state, dict):
        raise ValueError(f'{mode} checkpoint is not a dict')
    done = state.get('completed_epoch')
    if type(done) is not int or done < 1:
        raise ValueError(f'{mode} checkpoint has invalid completed_epoch {done!r}')
    if state.get('stop_reason') is None and (final_epoch is None or done < final_epoch):
        return None  # unfinished: the engine validates and resumes it
    if state.get('mode') != mode or state.get('cfg') != cfg:
        raise ValueError(f'terminal {mode} checkpoint was written for a different mode or cfg')
    if final_epoch is not None and done > final_epoch:
        raise ValueError(f'terminal {mode} checkpoint ran past epoch {final_epoch}')
    if not isinstance(state.get('history'), list) or not isinstance(state.get('model_state'), dict):
        raise ValueError(f'terminal {mode} checkpoint lacks history or model_state')
    return state


def _select_from_terminal(state):
    best_epoch = _selected_epoch(state.get('best_epoch'))
    snapshot = state.get('best_snapshot')
    if (best_epoch > state['completed_epoch'] or not isinstance(snapshot, dict)
            or snapshot.get('completed_epoch') != best_epoch
            or not isinstance(snapshot.get('model_state'), dict)):
        raise ValueError('terminal select checkpoint has an invalid best snapshot')
    return {'history': state['history'], 'best_epoch': best_epoch, 'best_checkpoint': snapshot,
            'last_checkpoint': state, 'stop_reason': state.get('stop_reason')}


def _epoch_writer(phase, output_dir, last_name, binding):
    started = time.monotonic()

    def on_epoch(checkpoint, record):
        _save_last(os.path.join(output_dir, last_name), checkpoint, binding)
        report = {'phase': phase, 'epoch': checkpoint['completed_epoch'],
                  'elapsed_s': round(time.monotonic() - started, 3)}
        _atomic_write(os.path.join(output_dir, PROGRESS), _json_bytes(
            {**report, 'record': record, 'history': checkpoint['history'], **binding}))
        scalars = {k: v for k, v in record.items() if type(v) in _SCALAR_TYPES}
        print(_canon({**scalars, **report}), flush=True)

    return on_epoch


def _ensure_init(path, payload, binding):
    if os.path.exists(path):
        stored = json.loads(_read_signed(path, binding['context_sha256'], binding['run_sha256']))
        if stored != payload:
            raise ValueError(f'{path}: initial model digest differs from the recorded run')
    else:
        _write_signed(path, _json_bytes(payload), binding)


def _verified_final(output_dir, context):
    ctx_sha = context_sha256(context)
    frozen_bytes = _read_signed(os.path.join(output_dir, FROZEN_MODEL), ctx_sha)
    frozen = json.loads(frozen_bytes)
    if frozen.get('context_sha256') != ctx_sha or frozen.get('context') != context:
        raise ValueError(f'{FROZEN_MODEL} was produced for a different context')
    run_sha = frozen.get('run_sha256')
    if not isinstance(run_sha, str):
        raise ValueError(f'{FROZEN_MODEL} lacks a run hash')
    model_bytes = _read_signed(os.path.join(output_dir, FINAL_MODEL), ctx_sha, run_sha)
    if _sha(model_bytes) != frozen.get('final_model_sha256'):
        raise ValueError(f'{FINAL_MODEL} does not match {FROZEN_MODEL}')
    return frozen, _sha(frozen_bytes), model_bytes


def _completed(output_dir, context, binding):
    def at(name):
        return os.path.join(output_dir, name)

    ctx_sha, run_sha = binding['context_sha256'], binding['run_sha256']
    result = json.loads(_read_signed(at(REFIT_RESULT), ctx_sha, run_sha))
    selected = json.loads(_read_signed(at(SELECT_RESULT), ctx_sha, run_sha))
    frozen, frozen_sha, _ = _verified_final(output_dir, context)
    if (frozen['run_sha256'] != run_sha or frozen_sha != result.get('frozen_sha256')
            or frozen.get('final_model_sha256') != result.get('final_model_sha256')
            or not result.get('selected_epoch') == selected.get('best_epoch') == frozen.get('selected_epoch')):
        raise ValueError(f'{output_dir}: finalized outputs are inconsistent; refusing to reuse them')
    return {'output_dir': output_dir, 'selected_epoch': frozen['selected_epoch'],
            'select_result': at(SELECT_RESULT), 'refit_result': at(REFIT_RESULT),
            'frozen_model': at(FROZEN_MODEL), 'final_model': at(FINAL_MODEL), 'frozen': frozen}


def run_experiment(factory, train, val, all_train, cfg, refit_cfg, output_dir, context,
                   augment=None, resume=False):
    """Select an epoch count on train/val, then refit a fresh model on all_train.

    factory() must return a newly constructed CPU model with an identical initial
    state on every call. cfg and refit_cfg must be equal except num_samples.
    all_train must hold exactly the train and val samples (val weights may differ).
    context is a plain-JSON dict of caller pins (hashes, optional classes and
    inference_shape). Returns output paths, the selected epoch and frozen metadata.
    """
    def at(name):
        return os.path.join(output_dir, name)

    ctx_sha = context_sha256(context)
    _check_cfgs(cfg, refit_cfg)
    if val is None:
        raise ValueError('selection requires a validation split')
    _check_splits(train, val, all_train)
    digests = {name: _dataset_digest(d) for name, d in (('train', train), ('val', val), ('all_train', all_train))}
    run_sha = _sha(_canon({'context_sha256': ctx_sha, 'cfg': cfg, 'refit_cfg': refit_cfg,
                           'data': digests}).encode('utf-8'))
    binding = {'context_sha256': ctx_sha, 'run_sha256': run_sha}

    os.makedirs(output_dir, exist_ok=True)
    if not resume and os.listdir(output_dir):
        raise FileExistsError(f'{output_dir} is not empty; pass resume=True to continue that run')
    if resume and os.path.exists(at(REFIT_RESULT)):
        return _completed(output_dir, context, binding)
    if resume and any(os.path.exists(at(name)) for name in (FROZEN_MODEL, FINAL_MODEL)):
        raise RuntimeError(f'{output_dir}: {FROZEN_MODEL}/{FINAL_MODEL} exist without {REFIT_RESULT}; '
                           'finalization was interrupted and needs manual review')

    select_model, initial = _fresh_model(factory)
    probe, probe_digest = _fresh_model(factory)
    if probe is select_model or probe_digest != initial:
        raise ValueError('factory must return a new model with an identical initial state on every call')
    del probe
    _ensure_init(at(INIT_SELECT), {'state_digest': initial, **binding}, binding)
    _ensure_init(at(INIT_REFIT), {'state_digest': probe_digest, **binding}, binding)

    if resume and os.path.exists(at(SELECT_RESULT)):
        selected = json.loads(_read_signed(at(SELECT_RESULT), ctx_sha, run_sha))
        if _sha(_read_signed(at(SELECT_BEST), ctx_sha, run_sha)) != selected.get('best_checkpoint_sha256'):
            raise ValueError(f'{SELECT_BEST} does not match {SELECT_RESULT}')
        best_epoch = _selected_epoch(selected.get('best_epoch'))
    else:
        resume_state = _resume_state(at(SELECT_LAST), resume, ctx_sha, run_sha)
        cap = cfg.get(_MAX_EPOCHS_KEY)
        terminal = _terminal_checkpoint(resume_state, 'select', cfg, cap if type(cap) is int else None)
        if terminal is not None:
            print('recovery: select checkpoint is terminal; finalizing without training', flush=True)
            out = _select_from_terminal(terminal)
        else:
            out = _unpack(run_training(
                select_model, train, val, cfg, augment, mode='select',
                on_epoch=_epoch_writer('select', output_dir, SELECT_LAST, binding),
                resume=resume_state))
        best_epoch = _selected_epoch(out['best_epoch'])
        if out['best_checkpoint'].get('completed_epoch') != best_epoch:
            raise ValueError('best checkpoint does not belong to the selected epoch')
        best_sha = _save_checkpoint(at(SELECT_BEST), out['best_checkpoint'], binding)
        _write_signed(at(SELECT_RESULT), _json_bytes({
            'best_epoch': best_epoch, 'stop_reason': out['stop_reason'], 'history': out['history'],
            'best_checkpoint_sha256': best_sha, **binding}), binding)
    del select_model

    refit_model, refit_digest = _fresh_model(factory)
    if refit_digest != initial:
        raise ValueError('fresh refit model differs from the recorded initial state')
    resume_state = _resume_state(at(REFIT_LAST), resume, ctx_sha, run_sha)
    terminal = _terminal_checkpoint(resume_state, 'refit', refit_cfg, best_epoch)
    if terminal is not None:
        print('recovery: refit checkpoint is terminal; finalizing without training', flush=True)
        refit_model.load_state_dict(terminal['model_state'], strict=True)
        out = {'history': terminal['history'], 'last_checkpoint': terminal,
               'stop_reason': terminal.get('stop_reason')}
    else:
        out = _unpack(run_training(
            refit_model, all_train, None, refit_cfg, augment, mode='refit', fixed_epochs=best_epoch,
            on_epoch=_epoch_writer('refit', output_dir, REFIT_LAST, binding),
            resume=resume_state))
    last = out['last_checkpoint']
    if last.get('completed_epoch') != best_epoch:
        raise ValueError('refit did not complete the selected number of epochs')
    state = {name: t.detach().cpu().contiguous() for name, t in refit_model.state_dict().items()}
    if state_digest(state) != state_digest(last['model_state']):
        raise ValueError('refit model state differs from its last checkpoint')
    _check_finite_state(state, 'refit model')

    names = list(state)
    buffer = io.BytesIO()
    np.savez(buffer, *[state[name].numpy() for name in names])
    model_sha = _write_signed(at(FINAL_MODEL), buffer.getvalue(), binding)
    frozen = {'final_model_file': FINAL_MODEL, 'final_model_sha256': model_sha,
              'state_digest': state_digest(state), 'state_names': names,
              'selected_epoch': best_epoch, 'context': context, **binding}
    for key in ('classes', 'inference_shape'):
        if key in context:
            frozen[key] = context[key]
    frozen_sha = _write_signed(at(FROZEN_MODEL), _json_bytes(frozen), binding)
    refit_last = _read_last(at(REFIT_LAST), ctx_sha, run_sha)
    if refit_last is None:
        raise ValueError(f'{REFIT_LAST} and its previous generation fail verification')
    refit_last_sha = _sha(refit_last)
    _write_signed(at(REFIT_RESULT), _json_bytes({
        'selected_epoch': best_epoch, 'stop_reason': out['stop_reason'], 'history': out['history'],
        'refit_last_sha256': refit_last_sha, 'final_model_sha256': model_sha,
        'frozen_sha256': frozen_sha, **binding}), binding)
    return _completed(output_dir, context, binding)


def validate_frozen(output_dir, context):
    """Return FROZEN_MODEL.json after checking the context and final model hashes."""
    return _verified_final(output_dir, context)[0]


def load_final_state(model, output_dir, context):
    """Strictly load the verified frozen weights into a caller-built model."""
    frozen, _, model_bytes = _verified_final(output_dir, context)
    names = frozen['state_names']
    with np.load(io.BytesIO(model_bytes), allow_pickle=False) as npz:
        if sorted(npz.files) != sorted(f'arr_{i}' for i in range(len(names))):
            raise ValueError(f'{FINAL_MODEL} entries do not match state_names')
        state = {name: torch.from_numpy(np.array(npz[f'arr_{i}'], copy=True)) for i, name in enumerate(names)}
    if state_digest(state) != frozen['state_digest']:
        raise ValueError(f'{FINAL_MODEL} state digest mismatch')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def score_batches(model, x, batch_size, window_valid=None):
    """Return float32 numpy logits [N, C]; model mode is restored afterwards.

    If window_valid (dense CPU bool [N,6], every row with a True) is given, each
    batch is scored as model(batch, window_valid=<matching rows>); otherwise
    model(batch) exactly as before.
    """
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float32 or x.device.type != 'cpu' \
            or x.ndim < 1 or x.shape[0] < 1:
        raise ValueError('x must be a non-empty CPU float32 tensor')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('x must be finite')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if window_valid is not None:
        check_window_valid(window_valid, x.shape[0])
    was_training = model.training
    outputs = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = x[start:start + batch_size]
                if window_valid is None:
                    out = model(batch)
                else:
                    out = model(batch, window_valid=window_valid[start:start + batch_size].clone())
                if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != batch.shape[0]:
                    raise ValueError('model must return logits [batch, classes]')
                outputs.append(out.detach().to('cpu', torch.float32))
    finally:
        model.train(was_training)
    logits = torch.cat(outputs).numpy()
    if not np.isfinite(logits).all():
        raise ValueError('model produced non-finite logits')
    return logits
