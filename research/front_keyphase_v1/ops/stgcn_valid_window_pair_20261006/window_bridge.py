'''window_bridge.py: explicit per-window validity metadata helpers (standalone).

window_valid_from_indices derives bool masks ONLY from frame-index metadata
(-1 = synthetic padding, >=0 = real frame); coordinate values are never read.
bind_window_forward gives an existing module instance a mask-aware forward
without adding parameters, buffers, modules or name prefixes. Persist such
models via state_dict(); the bound instance itself is not meant to be pickled.
'''
import types

import numpy as np
import torch
from torch import nn

NUM_WINDOWS = 6
WINDOW_LEN = 10
SEQ_LEN = NUM_WINDOWS * WINDOW_LEN
CHILD_NAMES = ('conv_encoder', 'agg', 'final_bn', 'final_fc')
_TAG = '_window_bridge_forward_fn'


def _as_int64_array(indices):
    '''Copy indices into a fresh int64 ndarray, rejecting bool/float/non-int input.'''
    if isinstance(indices, torch.Tensor):
        if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
            raise TypeError(f'indices tensor must have an integer dtype, got {indices.dtype}')
        return indices.detach().to('cpu').numpy().astype(np.int64, copy=True)
    if isinstance(indices, np.ndarray):
        if indices.dtype == np.bool_ or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f'indices array must have an integer dtype, got {indices.dtype}')
        if (np.issubdtype(indices.dtype, np.unsignedinteger) and indices.size
                and int(indices.max()) > np.iinfo(np.int64).max):
            raise ValueError('indices exceed the int64 range')
        return indices.astype(np.int64, copy=True)
    try:
        obj = np.array(indices, dtype=object)
    except ValueError as exc:
        raise ValueError(f'indices must be a rectangular sequence: {exc}') from None
    for value in obj.reshape(-1):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f'indices must be integers, got {type(value).__name__}')
    return obj.astype(np.int64)


def window_valid_from_indices(indices):
    '''Return CPU bool [6] for [60] input or [N,6] for [N,60] input.

    A window is valid iff any of its 10 positions holds a real frame index (>=0).
    Raises on bool/float input, values < -1, wrong shapes or all-padding rows.
    The input is never modified.
    '''
    arr = _as_int64_array(indices)
    if arr.ndim not in (1, 2) or arr.shape[-1] != SEQ_LEN or (arr.ndim == 2 and arr.shape[0] < 1):
        raise ValueError(f'indices must have shape [{SEQ_LEN}] or [N,{SEQ_LEN}] with N>=1, got {arr.shape}')
    if bool((arr < -1).any()):
        raise ValueError('indices must be -1 (padding) or >= 0 (real frame)')
    valid = (arr.reshape(-1, NUM_WINDOWS, WINDOW_LEN) >= 0).any(axis=2)
    if not bool(valid.any(axis=1).all()):
        raise ValueError('every sample needs at least one real frame (all-padding row found)')
    out = torch.from_numpy(np.ascontiguousarray(valid)).clone()
    return out[0].clone() if arr.ndim == 1 else out


def bind_window_forward(model, forward_fn):
    '''Attach forward(self, x, window_valid) to an existing module instance.

    The new forward calls forward_fn(self.conv_encoder, self.agg, self.final_bn,
    self.final_fc, x, window_valid). Children keep their registered names, no
    state is added, and training flags are untouched. Re-binding the same
    forward_fn is a no-op; binding over any other instance-level forward raises.
    '''
    if not isinstance(model, nn.Module):
        raise TypeError('model must be an existing torch.nn.Module instance')
    if isinstance(forward_fn, nn.Module) or not callable(forward_fn):
        raise TypeError('forward_fn must be a plain callable, not an nn.Module (it would hide state)')
    for name in CHILD_NAMES:
        if not isinstance(model._modules.get(name), nn.Module):
            raise ValueError(f'model lacks a registered child module {name!r}')
    existing = model.__dict__.get('forward')
    if existing is not None:
        if (isinstance(existing, types.MethodType) and existing.__self__ is model
                and getattr(existing.__func__, _TAG, None) is forward_fn):
            return model
        raise RuntimeError('model already has an instance-level forward; refusing to rebind')
    keys = list(model.state_dict().keys())

    def forward(self, x, window_valid):
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError('x must be a batched tensor')
        if (not isinstance(window_valid, torch.Tensor) or window_valid.dtype != torch.bool
                or tuple(window_valid.shape) != (x.shape[0], NUM_WINDOWS)):
            raise ValueError(f'window_valid must be a bool tensor of shape {(x.shape[0], NUM_WINDOWS)}')
        if not bool(window_valid.any(dim=1).all()):
            raise ValueError('every window_valid row must contain at least one True window')
        return forward_fn(self.conv_encoder, self.agg, self.final_bn, self.final_fc, x, window_valid)

    setattr(forward, _TAG, forward_fn)
    object.__setattr__(model, 'forward', types.MethodType(forward, model))
    if list(model.state_dict().keys()) != keys:
        raise RuntimeError('binding changed state_dict keys')  # defensive; should be unreachable
    return model
