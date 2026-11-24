"""Small, deterministic distance-input adaptation for a masked temporal classifier."""
from __future__ import annotations

import copy
import math
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAIRS = ((4, 8), (4, 12), (4, 16), (4, 20), (0, 4),
         (0, 8), (0, 12), (0, 16), (0, 20))
N_VALUES = 18
EXTRA_WIDTH = 36


def _require(ok, message, exc=ValueError):
    if not ok:
        raise exc(message)


def _binary(value, shape, name):
    _require(isinstance(value, np.ndarray), name + ' must be an ndarray', TypeError)
    _require(value.shape == shape, name + ' has an incompatible shape')
    _require(value.dtype.kind in 'biuf', name + ' must be numeric or bool', TypeError)
    _require(bool(np.isfinite(value).all()) and bool(((value == 0) | (value == 1)).all()),
             name + ' must contain only 0 or 1')
    return value.astype(bool, copy=False)


def hand_distances(q, present):
    """Return left9/right9 distances followed by their validity flags, [T,36]."""
    _require(isinstance(q, np.ndarray) and q.dtype.kind == 'f',
             'q must be a floating ndarray', TypeError)
    _require(q.ndim == 4 and q.shape[0] > 0 and q.shape[1:] == (2, 21, 2),
             'q must have shape [T,2,21,2], T>0')
    valid = _binary(present, q.shape[:-1], 'present')
    _require(bool(np.isfinite(q[valid]).all()), 'observed coordinates must be finite')
    # Never subtract absent NaNs, even for pairs that will subsequently be masked.
    points = np.where(valid[..., None], q, 0.0).astype(np.float64)
    values = np.zeros((q.shape[0], 2, len(PAIRS)), dtype=np.float64)
    masks = np.zeros(values.shape, dtype=bool)
    for j, (a, b) in enumerate(PAIRS):
        good = valid[:, :, a] & valid[:, :, b]
        delta = points[:, :, a] - points[:, :, b]
        values[:, :, j] = np.where(good, np.hypot(delta[..., 0], delta[..., 1]), 0.0)
        masks[:, :, j] = good
    with np.errstate(over='ignore', invalid='ignore'):
        out = np.concatenate((values.reshape(-1, N_VALUES),
                              masks.reshape(-1, N_VALUES)), axis=1).astype(np.float32)
    _require(bool(np.isfinite(out).all()), 'distance exceeds finite float32 range')
    return out


def _extra(array, name='extra', require_float32=False):
    _require(isinstance(array, np.ndarray) and array.dtype.kind == 'f',
             name + ' must be a floating ndarray', TypeError)
    if require_float32:
        _require(array.dtype == np.float32, name + ' must be float32', TypeError)
    _require(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] == EXTRA_WIDTH,
             name + ' must have shape [T,36], T>0')
    _require(bool(np.isfinite(array).all()), name + ' must be finite')
    mask = _binary(array[:, N_VALUES:], (len(array), N_VALUES), name + ' mask')
    _require(bool((array[:, :N_VALUES] >= 0).all()), name + ' distances must be nonnegative')
    _require(bool((array[:, :N_VALUES][~mask] == 0).all()), name + ' invalid values must be zero')
    return mask


def _labels(labels, n, name, classes=None):
    arr = np.asarray(labels)
    _require(arr.ndim == 1 and len(arr) == n and arr.dtype.kind in 'iu',
             name + ' must be one integer per clip')
    if classes is not None:
        _require(bool(((arr >= 0) & (arr < classes)).all()), name + ' outside class range')
    return arr.astype(np.int64, copy=False)


def fit_scale(extras, labels):
    """Class-balanced RMS of per-clip valid-frame mean squares; no centering."""
    _require(isinstance(extras, (list, tuple)) and len(extras) > 0, 'extras must be nonempty')
    ys = _labels(labels, len(extras), 'labels')
    frequency = Counter(int(y) for y in ys)
    numerator = np.zeros(N_VALUES, dtype=np.float64)
    denominator = np.zeros(N_VALUES, dtype=np.float64)
    for clip, y in zip(extras, ys):
        mask = _extra(clip)
        count = mask.sum(axis=0)
        seen = count > 0
        values = clip[:, :N_VALUES].astype(np.float64)
        means = (values * values).sum(axis=0) / np.maximum(count, 1)
        weight = 1.0 / frequency[int(y)]
        numerator[seen] += weight * means[seen]
        denominator[seen] += weight
    rms = np.sqrt(numerator / np.where(denominator > 0, denominator, 1.0))
    rms[(denominator == 0) | (rms == 0)] = 1.0
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        out = rms.astype(np.float32)
    _require(bool(np.isfinite(out).all()) and bool((out > 0).all()), 'invalid RMS scale')
    return out


def _validate_parent(parent):
    _require(isinstance(parent, nn.Module), 'parent must be a torch module', TypeError)
    proj, blocks, head = (getattr(parent, key, None) for key in ('proj', 'blocks', 'head'))
    _require(isinstance(proj, nn.Conv1d), 'parent.proj must be Conv1d')
    _require(proj.kernel_size == (1,) and proj.stride == (1,) and proj.padding == (0,)
             and proj.dilation == (1,) and proj.groups == 1,
             'parent.proj must be an ungrouped pointwise Conv1d')
    _require(isinstance(blocks, nn.ModuleList) and len(blocks) > 0, 'parent.blocks must be nonempty')
    _require(isinstance(head, nn.Linear) and head.in_features == proj.out_channels,
             'parent.head must be compatible Linear')
    for name, tensor in list(parent.named_parameters()) + list(parent.named_buffers()):
        _require(tensor.device.type == 'cpu', 'parent must be CPU: ' + name)
        _require(not tensor.is_complex(), 'complex parent state: ' + name)
        if tensor.is_floating_point():
            _require(tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()),
                     'parent floating state must be finite float32: ' + name)
    return proj.in_channels, proj.out_channels, head.out_features


class GeometryTCN(nn.Module):
    def __init__(self, parent, scale, enabled=True):
        super().__init__()
        self.in_channels, self.hidden, self.n_classes = _validate_parent(parent)
        _require(type(enabled) is bool, 'enabled must be bool', TypeError)
        arr = np.asarray(scale)
        _require(arr.shape == (N_VALUES,) and arr.dtype.kind == 'f'
                 and bool(np.isfinite(arr).all()) and bool((arr > 0).all()),
                 'scale must contain 18 positive finite floating values')
        scale32 = arr.astype(np.float32)
        _require(bool(np.isfinite(scale32).all()) and bool((scale32 > 0).all()),
                 'scale must be representable as positive float32')
        # Deepcopy, zeros and as_tensor do not draw random numbers.
        self.base = copy.deepcopy(parent)
        self.enabled = enabled
        self.register_buffer('extra_scale', torch.from_numpy(scale32.copy()))
        self.extra_weight = nn.Parameter(torch.zeros(self.hidden, EXTRA_WIDTH, 1,
                                                     dtype=torch.float32, device='cpu'))
        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.base.blocks[-1].parameters():
            p.requires_grad_(True)
        for p in self.base.head.parameters():
            p.requires_grad_(True)
        self.eval()

    @property
    def trainable_names(self):
        return [name for name, value in self.named_parameters() if value.requires_grad]

    @property
    def trainable_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, x, extra, pad):
        for name, tensor in (('x', x), ('extra', extra), ('pad', pad)):
            _require(isinstance(tensor, torch.Tensor) and tensor.device.type == 'cpu',
                     name + ' must be a CPU tensor', TypeError)
            _require(tensor.dtype == torch.float32, name + ' must be float32', TypeError)
            _require(bool(torch.isfinite(tensor).all()), name + ' must be finite')
        _require(x.ndim == 3 and min(x.shape[:2]) > 0 and x.shape[2] == self.in_channels,
                 'x must have shape [B,T,C]')
        _require(extra.shape == (*x.shape[:2], EXTRA_WIDTH) and pad.shape == x.shape[:2],
                 'extra/pad shapes must match x')
        _require(bool(((pad == 0) | (pad == 1)).all()) and bool((pad.sum(dim=1) > 0).all()),
                 'pad must be binary with a real frame per clip')
        em = extra[:, :, N_VALUES:]
        _require(bool(((em == 0) | (em == 1)).all()), 'extra validity must be binary')
        values = extra[:, :, :N_VALUES]
        _require(bool((values >= 0).all()) and bool((values[em == 0] == 0).all()),
                 'distances must be nonnegative, invalid values zero')
        _require(not any(module.training for module in self.base.modules()),
                 'all base modules must remain in eval mode')
        mask = pad.unsqueeze(1)
        h = self.base.proj(x.transpose(1, 2) * mask) * mask
        features = torch.cat((values / self.extra_scale, em), dim=2)
        _require(bool(torch.isfinite(features).all()), 'scaled extra is nonfinite')
        if not self.enabled:
            features = torch.zeros_like(features)
        h = h + F.conv1d(features.transpose(1, 2), self.extra_weight) * mask
        for block in self.base.blocks:
            h = block(h, mask)
        pooled = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return self.base.head(pooled)


def _clips(xs, extras, channels, name):
    _require(isinstance(xs, (list, tuple)) and len(xs) > 0, name + ' must be a nonempty clip list')
    _require(isinstance(extras, (list, tuple)) and len(extras) == len(xs), name + ' extras length mismatch')
    for i, (x, e) in enumerate(zip(xs, extras)):
        _require(isinstance(x, np.ndarray) and x.dtype == np.float32,
                 name + ' clips must be float32 ndarrays', TypeError)
        _require(x.ndim == 2 and x.shape[0] > 0 and x.shape[1] == channels
                 and bool(np.isfinite(x).all()), name + ' incompatible/nonfinite clip')
        _extra(e, name + ' extra ' + str(i), require_float32=True)
        _require(len(e) == len(x), name + ' extra time mismatch')


def _pad(xs, extras):
    length = max(len(x) for x in xs)
    x = np.zeros((len(xs), length, xs[0].shape[1]), dtype=np.float32)
    e = np.zeros((len(xs), length, EXTRA_WIDTH), dtype=np.float32)
    mask = np.zeros((len(xs), length), dtype=np.float32)
    for i, (clip, extra) in enumerate(zip(xs, extras)):
        t = len(clip)
        x[i, :t], e[i, :t], mask[i, :t] = clip, extra, 1.0
    return torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(mask)


def _schedule(indices, count, name):
    _require(isinstance(indices, np.ndarray) and indices.dtype.kind in 'iu',
             name + ' must be an integer ndarray', TypeError)
    _require(indices.ndim == 2 and min(indices.shape) > 0, name + ' must have shape [steps,half]')
    _require(bool(((indices >= 0) & (indices < count)).all()), name + ' outside data range')
    return indices


def _state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _same_state(module, before):
    after = module.state_dict()
    return list(after) == list(before) and all(torch.equal(before[k], after[k]) for k in before)


def train_pair(parent, base_x, base_y, context_x, context_y, teacher_logits,
               base_indices, context_indices, extras_base, extras_context, scale,
               settings, progress=None):
    """Train two fixed arms without evaluation, RNG draws, I/O or model selection."""
    channels, _, classes = _validate_parent(parent)
    _clips(base_x, extras_base, channels, 'base')
    _clips(context_x, extras_context, channels, 'context')
    by = _labels(base_y, len(base_x), 'base_y', classes)
    cy = _labels(context_y, len(context_x), 'context_y', classes)
    _require(np.asarray(base_y).dtype == np.int64 and np.asarray(context_y).dtype == np.int64,
             'training labels must be int64')
    bi = _schedule(base_indices, len(base_x), 'base_indices')
    ci = _schedule(context_indices, len(context_x), 'context_indices')
    _require(bi.shape == ci.shape, 'schedule shapes differ')
    _require(isinstance(teacher_logits, np.ndarray) and teacher_logits.dtype == np.float64
             and teacher_logits.shape == (len(base_x), classes)
             and bool(np.isfinite(teacher_logits).all()), 'teacher must be finite float64 [Nbase,K]')
    _require(isinstance(settings, dict), 'settings must be a dict', TypeError)
    cfg = {}
    for key in ('lr', 'weight_decay', 'anchor_coefficient', 'output_coefficient'):
        value = settings.get(key)
        _require(isinstance(value, (int, float, np.integer, np.floating))
                 and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value)),
                 'missing/nonfinite setting ' + key)
        cfg[key] = float(value)
        _require(cfg[key] > 0 if key == 'lr' else cfg[key] >= 0, 'invalid setting ' + key)
    _require(progress is None or callable(progress), 'progress must be callable')
    parent_before = _state(parent)
    parent_modes = [(name, m.training) for name, m in parent.named_modules()]
    parent_grads = [(name, p.requires_grad) for name, p in parent.named_parameters()]
    rng_before = torch.random.get_rng_state().clone()
    results = {}
    steps, half = bi.shape
    for arm, enabled in (('control', False), ('treatment', True)):
        model = GeometryTCN(parent, scale, enabled=enabled)
        model.eval()
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        names = [name for name, _ in trainable]
        params = [p for _, p in trainable]
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        buffers = {name: b.detach().clone() for name, b in model.named_buffers()}
        optimizer = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
                                     betas=(0.9, 0.999), eps=1e-8)
        losses = np.zeros((steps, 4), dtype=np.float64)
        with torch.enable_grad():
            for step in range(steps):
                br, cr = bi[step], ci[step]
                xs = [base_x[int(i)] for i in br] + [context_x[int(i)] for i in cr]
                es = [extras_base[int(i)] for i in br] + [extras_context[int(i)] for i in cr]
                x, extra, pad = _pad(xs, es)
                y = torch.from_numpy(np.concatenate((by[br], cy[cr])))
                logits = model(x, extra, pad)
                _require(bool(torch.isfinite(logits).all()), arm + ' nonfinite logits', RuntimeError)
                ce = F.cross_entropy(logits, y)
                square = sum((p - initial[name]).square().sum() for name, p in trainable)
                anchor = square * (0.5 * cfg['anchor_coefficient'])
                target = torch.from_numpy(np.array(teacher_logits[br], copy=True))
                delta = logits[:half].to(torch.float64) - target
                delta = delta - delta.mean(dim=1, keepdim=True)
                retention = delta.square().mean() * (0.5 * cfg['output_coefficient'])
                total = ce.double() + anchor.double() + retention
                terms = (ce, anchor, retention, total)
                _require(all(bool(torch.isfinite(term)) for term in terms),
                         arm + ' nonfinite loss', RuntimeError)
                losses[step] = [float(term.detach()) for term in terms]
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                for name, p in trainable:
                    _require(p.grad is not None and bool(torch.isfinite(p.grad).all()),
                             arm + ' missing/nonfinite gradient: ' + name, RuntimeError)
                if not enabled:
                    _require(not bool(model.extra_weight.grad.any()), 'control branch has nonzero gradient', RuntimeError)
                optimizer.step()
                _require(all(bool(torch.isfinite(p).all()) for p in params),
                         arm + ' nonfinite updated parameter', RuntimeError)
                if progress is not None and ((step + 1) % 100 == 0 or step + 1 == steps):
                    progress(arm, step + 1, steps, losses[step].tolist())
        frozen_ok = all(torch.equal(initial[name], p) for name, p in model.named_parameters()
                        if not p.requires_grad)
        buffers_ok = all(torch.equal(buffers[name], b) for name, b in model.named_buffers())
        _require(frozen_ok and buffers_ok, arm + ' frozen state changed', RuntimeError)
        nonzero = bool(torch.count_nonzero(model.extra_weight).item())
        _require(enabled or not nonzero, 'disabled control branch changed', RuntimeError)
        results[arm] = {'state_dict': _state(model), 'losses': losses,
                        'trainable_names': names, 'trainable_count': model.trainable_count,
                        'updates': int(steps), 'frozen_parameters_unchanged': frozen_ok,
                        'buffers_unchanged': buffers_ok, 'extra_weight_nonzero': nonzero}
    _require(_same_state(parent, parent_before) and parent_modes == [(n, m.training) for n, m in parent.named_modules()]
             and parent_grads == [(n, p.requires_grad) for n, p in parent.named_parameters()],
             'source parent changed', RuntimeError)
    _require(torch.equal(rng_before, torch.random.get_rng_state()), 'training consumed CPU RNG', RuntimeError)
    return results


def predict(model, xs, extras, batch_size=32):
    _require(isinstance(model, GeometryTCN), 'model must be GeometryTCN', TypeError)
    _require(not model.training and not any(m.training for m in model.base.modules()), 'predict requires eval mode')
    _require(isinstance(batch_size, (int, np.integer)) and not isinstance(batch_size, (bool, np.bool_))
             and batch_size > 0, 'batch_size must be a positive integer')
    _clips(xs, extras, model.in_channels, 'prediction')
    output = []
    with torch.inference_mode():
        for start in range(0, len(xs), int(batch_size)):
            x, extra, pad = _pad(xs[start:start + batch_size], extras[start:start + batch_size])
            scores = model(x, extra, pad)
            _require(scores.shape == (len(x), model.n_classes) and bool(torch.isfinite(scores).all()),
                     'prediction has invalid shape or nonfinite values', RuntimeError)
            output.append(scores.to(torch.float64).numpy().copy())
    return np.concatenate(output, axis=0)
