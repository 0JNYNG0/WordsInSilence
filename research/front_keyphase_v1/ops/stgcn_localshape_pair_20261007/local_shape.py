'''Wrist-relative local hand-shape XY channels for a 49-node graph model.

Channel layout after expansion: [global_x, global_y, local_x, local_y].
Only conv_encoder.data_bn and the first graph conv are widened. The adjacency,
remaining encoder, temporal modules, classifier and losses are untouched.
'''
import copy

import numpy as np
import torch
import torch.nn as nn

NUM_NODES = 49
OLD_C = 2
NEW_C = 4
CONV_OUT = 192
HAND_BASES = (('left', 7), ('right', 28))
HAND_SIZE = 21
MCP_OFFSETS = (5, 9, 13, 17)
MIN_SCALE = 1e-6
NUM_WINDOWS = 6
WINDOW_LEN = 10
_BN_DEFAULTS = (('weight', 1.0), ('bias', 0.0), ('running_mean', 0.0), ('running_var', 1.0))
_CONV_ATTRS = ('out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups',
               'padding_mode', 'transposed', 'output_padding', 'training')


# ------------------------------------------------------------------ helpers
def _storage_ptr(t):
    try:
        return t.untyped_storage().data_ptr()
    except AttributeError:  # torch < 2.0
        return t.storage().data_ptr()


def _module_name(root, target):
    for name, mod in root.named_modules():
        if mod is target:
            return name
    raise ValueError('module not reachable from model')


def _shape(t):
    return [int(s) for s in t.shape]


def _copy_extra_attrs(old, new):
    new_dict = vars(new)
    for key, val in vars(old).items():
        if key not in new_dict and not key.startswith('_'):
            setattr(new, key, copy.deepcopy(val))


# ------------------------------------------------------------------ 1) model expansion
def _locate(model):
    enc = getattr(model, 'conv_encoder', None)
    if not isinstance(enc, nn.Module):
        raise ValueError('model.conv_encoder must be an nn.Module')
    bn = getattr(enc, 'data_bn', None)
    if type(bn) is not nn.BatchNorm1d or bn.num_features != OLD_C * NUM_NODES:
        raise ValueError('conv_encoder.data_bn must be nn.BatchNorm1d(98)')
    adj = getattr(enc, 'A', None)
    if not isinstance(adj, torch.Tensor) or tuple(adj.shape) != (3, NUM_NODES, NUM_NODES):
        raise ValueError('conv_encoder.A must be a [3,49,49] tensor')
    try:
        block = enc.st_gcn_networks[0]
    except Exception:
        raise ValueError('conv_encoder.st_gcn_networks[0] missing') from None
    conv = getattr(getattr(block, 'gcn', None), 'conv', None)
    if type(conv) is not nn.Conv2d:
        raise ValueError('st_gcn_networks[0].gcn.conv must be nn.Conv2d')
    if (conv.in_channels != OLD_C or conv.out_channels != CONV_OUT
            or tuple(conv.kernel_size) != (1, 1) or conv.groups != 1 or conv.bias is None):
        raise ValueError('first conv must be Conv2d(2,192,kernel_size=1,bias=True,groups=1)')
    res = getattr(block, 'residual', False)
    if res is True or (isinstance(res, nn.Module) and len(list(res.parameters())) > 0):
        raise ValueError('first block residual must be disabled (residual=False)')
    return enc, bn, block, conv


def _validate_model(model):
    if not isinstance(model, nn.Module):
        raise TypeError('model must be an nn.Module')
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        raise ValueError('wrapped/bound models are not supported; pass the bare module')
    for name, mod in model.named_modules():
        if 'forward' in vars(mod):
            raise ValueError('instance-bound forward on module %r' % (name or '<root>'))
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    tensors += [(k, v) for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)]
    for key, t in tensors:
        if t.device.type != 'cpu':
            raise ValueError('non-CPU tensor: %s' % key)
        if t.is_floating_point() or t.is_complex():
            if t.dtype != torch.float32:
                raise ValueError('non-float32 floating tensor: %s' % key)
            if not bool(torch.isfinite(t.detach()).all()):
                raise ValueError('non-finite state: %s' % key)
    enc, bn, block, conv = _locate(model)
    n = OLD_C * NUM_NODES
    if bn.affine:
        if bn.weight is None or bn.bias is None or _shape(bn.weight) != [n] or _shape(bn.bias) != [n]:
            raise ValueError('data_bn affine parameters must have shape [98]')
    if bn.track_running_stats:
        for buf in (bn.running_mean, bn.running_var):
            if buf is None or _shape(buf) != [n]:
                raise ValueError('data_bn running stats must have shape [98]')
        if bn.num_batches_tracked is None:
            raise ValueError('data_bn.num_batches_tracked missing')
        if bool((bn.running_var < 0).any()):
            raise ValueError('data_bn.running_var must be non-negative')
    if _shape(conv.weight) != [CONV_OUT, OLD_C, 1, 1] or _shape(conv.bias) != [CONV_OUT]:
        raise ValueError('first conv parameter shapes incompatible')
    return enc, bn, block, conv


def _slot_index():
    j = torch.arange(NUM_NODES).repeat_interleave(OLD_C)
    c = torch.arange(OLD_C).repeat(NUM_NODES)
    return j * OLD_C + c, j * NEW_C + c  # old[2*j+c] -> new[4*j+c]


def _build_bn(old):
    ref = old.weight if old.affine else old.running_mean
    kw = {} if ref is None else {'device': ref.device, 'dtype': ref.dtype}
    new = nn.BatchNorm1d(NEW_C * NUM_NODES, eps=old.eps, momentum=old.momentum, affine=old.affine,
                         track_running_stats=old.track_running_stats, **kw)
    old_i, new_i = _slot_index()
    with torch.no_grad():
        if old.affine:
            new.weight.fill_(1.0)
            new.bias.zero_()
            new.weight[new_i] = old.weight.detach()[old_i]
            new.bias[new_i] = old.bias.detach()[old_i]
        if old.track_running_stats:
            new.running_mean.zero_()
            new.running_var.fill_(1.0)
            new.running_mean[new_i] = old.running_mean[old_i]
            new.running_var[new_i] = old.running_var[old_i]
            new.num_batches_tracked.copy_(old.num_batches_tracked)
    if old.affine:
        new.weight.requires_grad_(old.weight.requires_grad)
        new.bias.requires_grad_(old.bias.requires_grad)
    new.train(old.training)
    _copy_extra_attrs(old, new)
    return new


def _build_conv(old):
    w = old.weight
    new = nn.Conv2d(NEW_C, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                    padding=old.padding, dilation=old.dilation, groups=old.groups, bias=True,
                    padding_mode=old.padding_mode, device=w.device, dtype=w.dtype)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :OLD_C].copy_(w.detach())
        new.bias.copy_(old.bias.detach())
    new.weight.requires_grad_(w.requires_grad)
    new.bias.requires_grad_(old.bias.requires_grad)
    new.train(old.training)
    _copy_extra_attrs(old, new)
    for key in _CONV_ATTRS:
        if getattr(new, key) != getattr(old, key):
            raise RuntimeError('conv attribute mismatch: %s' % key)
    return new


def _check_mapped(key, old, new, bn_p, conv_p):
    if key == conv_p + 'weight':
        ok = (_shape(new) == [CONV_OUT, NEW_C, 1, 1] and torch.equal(new[:, :OLD_C], old)
              and not bool(new[:, OLD_C:].any()))
    else:
        default = dict(_BN_DEFAULTS)[key[len(bn_p):]]
        ok = new.numel() == NEW_C * NUM_NODES
        if ok:
            n = new.reshape(NUM_NODES, NEW_C)
            ok = (torch.equal(n[:, :OLD_C], old.reshape(NUM_NODES, OLD_C))
                  and bool((n[:, OLD_C:] == default).all()))
    if not ok:
        raise RuntimeError('mapping self-check failed for ' + key)


def _report(old_sd, new_sd, bn_p, conv_p, old_ptrs):
    rules = {bn_p + k: 'node-major old[2*j+c] -> new[4*j+c] (c=0,1); local slots 2,3 = %g' % v
             for k, v in _BN_DEFAULTS}
    rules[conv_p + 'weight'] = 'W[:, :2] = old W; W[:, 2:] = 0'
    rules = {k: v for k, v in rules.items() if k in old_sd}
    missing = sorted(set(old_sd) - set(new_sd))
    added = sorted(set(new_sd) - set(old_sd))
    expected_added = []
    if missing:
        raise RuntimeError('state keys lost: %s' % missing)
    if added != expected_added:
        raise RuntimeError('unexpected new state keys: %s' % added)
    exact, mapped, shape_changes, added_elements = [], {}, {}, {}
    for key, ov in old_sd.items():
        nv = new_sd[key]
        if not isinstance(ov, torch.Tensor):
            exact.append(key)  # non-tensor extra state carried by deepcopy
            continue
        if not isinstance(nv, torch.Tensor) or nv.dtype != ov.dtype or nv.device != ov.device:
            raise RuntimeError('type/dtype/device changed for ' + key)
        if nv.numel() > 0 and _storage_ptr(nv) in old_ptrs:
            raise RuntimeError('storage shared with source model: ' + key)
        if key in rules:
            _check_mapped(key, ov, nv, bn_p, conv_p)
            mapped[key] = {'old_shape': _shape(ov), 'new_shape': _shape(nv), 'rule': rules[key]}
            if _shape(ov) != _shape(nv):
                shape_changes[key] = {'old': _shape(ov), 'new': _shape(nv)}
                added_elements[key] = int(nv.numel() - ov.numel())
        elif nv.shape == ov.shape and torch.equal(nv, ov):
            exact.append(key)
        else:
            raise RuntimeError('unexpected change in ' + key)
    return {
        'exact': sorted(exact),
        'mapped': mapped,
        'shape_changes': shape_changes,
        'added_keys': added,
        'expected_added_keys': expected_added,
        'added_parameter_elements': added_elements,
        'missing_keys': missing,
        'all_original_keys_retained': not missing,
        'num_old_keys': len(old_sd),
        'num_new_keys': len(new_sd),
        'initial_output': 'mathematically preserved; compare numerically (atol=rtol=1e-5)',
    }


def expand_input4(model):
    '''Return (new_model, report) with a 4-channel input path.

    Only conv_encoder.data_bn (98 -> 196) and st_gcn_networks[0].gcn.conv
    (in_channels 2 -> 4) are replaced, on a deep copy. All validation happens
    before copying; the source model is never mutated and the global CPU RNG
    state is preserved (fork_rng around copy + construction).
    '''
    _validate_model(model)
    old_sd = model.state_dict()
    old_ptrs = set()
    for t in list(old_sd.values()) + list(model.parameters()) + list(model.buffers()):
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            old_ptrs.add(_storage_ptr(t))
    with torch.random.fork_rng(devices=[]):
        new_model = copy.deepcopy(model)
        enc, bn, block, conv = _locate(new_model)
        bn_prefix = _module_name(new_model, bn) + '.'
        conv_prefix = _module_name(new_model, conv) + '.'
        new_bn = _build_bn(bn)
        new_conv = _build_conv(conv)
        enc.data_bn = new_bn
        block.gcn.conv = new_conv
    report = _report(old_sd, new_model.state_dict(), bn_prefix, conv_prefix, old_ptrs)
    return new_model, report


# ------------------------------------------------------------------ 2) local coordinates
def _validate_raw(xy, present):
    if not isinstance(xy, np.ndarray) or not np.issubdtype(xy.dtype, np.floating):
        raise TypeError('xy must be a floating numpy array [T,49,2]')
    if xy.ndim != 3 or xy.shape[1:] != (NUM_NODES, 2):
        raise ValueError('xy must have shape [T,49,2], got %s' % (xy.shape,))
    if not isinstance(present, np.ndarray) or present.dtype != np.bool_:
        raise TypeError('present must be a bool numpy array [T,49]')
    if present.shape != xy.shape[:2]:
        raise ValueError('present shape must be [T,49] matching xy')
    if not np.isfinite(xy[present]).all():
        raise ValueError('observed coordinates must be finite')


def local_coordinates(xy, present):
    '''Clip-level wrist-relative, hand-scale-normalised coordinates.

    xy: float [T,49,2] (missing may be NaN); present: bool [T,49].
    Body nodes 0..6 -> 0. Per hand (left 7..27, right 28..48, wrist local 0,
    MCP local 5,9,13,17) the scale is the median of ALL observed wrist-MCP
    distances pooled over all frames. Returns (float32 [T,49,2], diag).
    valid_local_points counts (frame, hand point) pairs including the wrist.
    '''
    _validate_raw(xy, present)
    num_frames = xy.shape[0]
    pres = present.copy()
    pts = np.where(pres[..., None], xy.astype(np.float64), 0.0)
    local = np.zeros((num_frames, NUM_NODES, 2), dtype=np.float64)
    diag = {'num_frames': int(num_frames), 'body_local': 'zeros (nodes 0..6)',
            'min_scale': MIN_SCALE}
    for name, base in HAND_BASES:
        wrist_ok = pres[:, base]
        dists = []
        for off in MCP_OFFSETS:
            ok = wrist_ok & pres[:, base + off]
            dists.append(np.linalg.norm(pts[ok, base + off] - pts[ok, base], axis=-1))
        pooled = np.concatenate(dists)
        n_pairs = int(pooled.size)
        scale = None
        if n_pairs == 0:
            status = 'no_valid_pairs'
        else:
            med = float(np.median(pooled))
            if not np.isfinite(med):
                status = 'nonfinite_scale'
            elif med <= MIN_SCALE:
                status = 'degenerate_scale'
            else:
                status = 'ok'
                scale = med
        n_points = 0
        if scale is not None:
            hand = slice(base, base + HAND_SIZE)
            valid = pres[:, hand] & wrist_ok[:, None]
            rel = (pts[:, hand] - pts[:, base:base + 1]) / scale
            local[:, hand] = np.where(valid[..., None], rel, 0.0)
            n_points = int(valid.sum())
        diag[name] = {'scale': scale, 'count_valid_pairs': n_pairs,
                      'valid_local_points': n_points, 'status': status}
    out = local.astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError('non-finite local coordinates (overflow?)')
    return out, diag


# ------------------------------------------------------------------ 3) append local channels
def _validate_indices(frame_indices, num_frames, length):
    if isinstance(frame_indices, torch.Tensor):
        if frame_indices.dtype == torch.bool or frame_indices.is_floating_point() or frame_indices.is_complex():
            raise TypeError('frame_indices must be integers')
        arr = frame_indices.detach().cpu().numpy()
    else:
        if isinstance(frame_indices, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in frame_indices):
            raise TypeError('frame_indices must be integers, not bools')
        arr = np.asarray(frame_indices)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise TypeError('frame_indices must be integers')
    if arr.ndim != 1 or arr.shape[0] != length:
        raise ValueError('frame_indices must be 1-D of length %d' % length)
    arr = arr.astype(np.int64)
    if np.any(arr < -1) or np.any(arr >= num_frames):
        raise ValueError('frame_indices must be -1 (padding) or in [0, T-1]')
    valid = arr[arr >= 0]
    if valid.size > 1 and np.any(np.diff(valid) < 0):
        raise ValueError('valid frame_indices must be chronological')
    return arr


def append_local(global_windows, raw_xy49, raw_present49, frame_indices):
    '''Concatenate local channels to existing global windows -> [1,6,4,10,49].

    Local coordinates are computed on the full raw clip, then gathered with
    the SAME frame_indices used by the existing global preparation (-1 -> 0).
    The two global channels are copied bit-exactly.
    '''
    if not isinstance(global_windows, torch.Tensor):
        raise TypeError('global_windows must be a torch tensor')
    if global_windows.dtype != torch.float32 or global_windows.device.type != 'cpu':
        raise ValueError('global_windows must be CPU float32')
    expected = (1, NUM_WINDOWS, OLD_C, WINDOW_LEN, NUM_NODES)
    if tuple(global_windows.shape) != expected:
        raise ValueError('global_windows must have shape %s' % (expected,))
    local, ldiag = local_coordinates(raw_xy49, raw_present49)
    n = NUM_WINDOWS * WINDOW_LEN
    idx = _validate_indices(frame_indices, local.shape[0], n)
    gathered = np.zeros((n, NUM_NODES, 2), dtype=np.float32)
    valid = idx >= 0
    gathered[valid] = local[idx[valid]]
    loc = torch.from_numpy(gathered).reshape(1, NUM_WINDOWS, WINDOW_LEN, NUM_NODES, 2)
    loc = loc.permute(0, 1, 4, 2, 3)
    out = torch.cat([global_windows, loc], dim=2)
    g_bits = out[:, :, :OLD_C].contiguous().view(torch.int32)
    if not torch.equal(g_bits, global_windows.contiguous().view(torch.int32)):
        raise RuntimeError('global channels not preserved bit-exactly')
    diag = {'local': ldiag, 'num_indices': n, 'num_valid_indices': int(valid.sum()),
            'num_padding': int((~valid).sum()),
            'layout': '[1,6,4,10,49]: [global_x, global_y, local_x, local_y]'}
    return out, diag


# ------------------------------------------------------------------ 4) paired augmentation
def augment_paired(sample, transform_fn):
    '''Apply ONE caller transform jointly to global and local XY.

    sample [W,4,T,49] -> pack [2, W*T, 98] (global vertices then local),
    call transform_fn exactly once, split and repack. RNG consumption equals a
    single global-only call for transforms whose draws do not depend on V.
    Local zeros stay zero only for linear transforms without translation.
    '''
    if not isinstance(sample, torch.Tensor) or not sample.is_floating_point():
        raise TypeError('sample must be a floating torch tensor')
    if sample.dim() != 4 or sample.shape[1] != NEW_C or sample.shape[3] != NUM_NODES:
        raise ValueError('sample must have shape [W,4,T,49]')
    if not bool(torch.isfinite(sample).all()):
        raise ValueError('sample must be finite')
    if not callable(transform_fn):
        raise TypeError('transform_fn must be callable')
    w, _, t, v = sample.shape
    g = sample[:, :OLD_C].permute(1, 0, 2, 3).reshape(2, w * t, v)
    l = sample[:, OLD_C:].permute(1, 0, 2, 3).reshape(2, w * t, v)
    packed = torch.cat([g, l], dim=2).contiguous()  # fresh tensor; sample never exposed
    out = transform_fn(packed)
    if not isinstance(out, torch.Tensor):
        raise TypeError('transform_fn must return a tensor')
    if tuple(out.shape) != (2, w * t, 2 * v) or out.dtype != sample.dtype or out.device != sample.device:
        raise ValueError('transform_fn output shape/dtype/device mismatch')
    if not bool(torch.isfinite(out).all()):
        raise ValueError('transform_fn output must be finite')
    og = out[:, :, :v].reshape(2, w, t, v).permute(1, 0, 2, 3)
    ol = out[:, :, v:].reshape(2, w, t, v).permute(1, 0, 2, 3)
    return torch.cat([og, ol], dim=1).contiguous()
