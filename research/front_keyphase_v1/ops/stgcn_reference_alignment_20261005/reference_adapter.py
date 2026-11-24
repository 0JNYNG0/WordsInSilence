'''Verification adapter around pinned, unmodified OpenHands release sources.

Unit contract only: pin-verified isolated import of three official files,
release DPC_RNN_Finetuner construction with caller-supplied tensors, and
release isolated-sign input windowing. No training, downloads, pickle or
torch.load, device access, or logging of user data.
'''
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import math
import numbers
import os
import re
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn

RELEASE_COMMIT = 'fc4599d2c6a9e68d002bb2e1832a835e9b8b512d'
PINS_FILENAME = 'SOURCE_PINS.json'
ST_GCN = 'openhands/models/ssl/st_gcn.py'
DPC_RNN = 'openhands/models/ssl/dpc_rnn.py'
POSE_TRANSFORMS = 'openhands/datasets/pose_transforms.py'
# Import order matters: dpc_rnn uses `from .st_gcn import ...`.
_MODULE_SUFFIX = {
    ST_GCN: 'models.ssl.st_gcn',
    DPC_RNN: 'models.ssl.dpc_rnn',
    POSE_TRANSFORMS: 'datasets.pose_transforms',
}
_SUBPACKAGES = ('models', 'models.ssl', 'datasets')
_SHA256 = re.compile('[0-9a-f]{64}')
_BACKSLASH = chr(92)

ENCODER_KEY_COUNT = 170
GRU_KEYS = ('weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0')
HIDDEN_DIM = 256
_MODEL_CHILDREN = {'conv_encoder', 'agg', 'final_bn', 'final_fc'}

_HAND = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)
LAYOUTS = {
    'mp75': (75, (0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46, 49, 50, 53, 54,
                  58, 59, 62, 63, 66, 67, 70, 71, 74)),
    'native67': (67, (0, 16, 15, 5, 2, 6, 3) + tuple(25 + i for i in _HAND)
                 + tuple(46 + i for i in _HAND)),
}
NORMALIZE_PRESET = 'shoulder_mediapipe_holistic_minimal_27'
SHOULDER_JOINTS = (3, 4)  # preset reference points in the 27-joint layout
TARGET_FRAMES, NUM_WINDOWS, WINDOW_FRAMES = 60, 6, 10


def _sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def _array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f'{array.dtype.str}{array.shape}'.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(f'{key}|{tensor.dtype}|{tuple(tensor.shape)}|'.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _openhands_modules():
    return {name for name in sys.modules if name.split('.')[0] == 'openhands'}


# ---------------------------------------------------------------- source pins

def _resolve_inside(root, rel):
    '''Resolve a pinned relative POSIX path; refuse traversal, absolute paths and escapes.'''
    if (not isinstance(rel, str) or not rel or rel.startswith('/')
            or _BACKSLASH in rel or ':' in rel
            or any(part in ('', '.', '..') for part in rel.split('/'))):
        raise ValueError(f'unsafe pinned path: {rel!r}')
    try:
        path = root.joinpath(*rel.split('/')).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f'pinned path not found: {rel!r}') from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'pinned path escapes reference_root or is not a file: {rel!r}')
    return path


def _verified_sources(root, pins):
    if not isinstance(pins, Mapping):
        raise ValueError("pins must be a mapping")
    top_keys = set(pins.keys())
    if not {"full_commit", "files"} <= top_keys:
        raise ValueError("pins must have exactly keys full_commit/files, got %r" % sorted(map(str, top_keys)))
    full_commit = pins["full_commit"]
    if not isinstance(full_commit, str) or full_commit != RELEASE_COMMIT:
        raise ValueError("full_commit %r does not match RELEASE_COMMIT" % (full_commit,))
    files = pins["files"]
    if not isinstance(files, (list, tuple)) or not files:
        raise ValueError("files must be a non-empty list")

    try:
        root_path = Path(os.fspath(root)).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError("reference root does not exist: %r" % (root,)) from exc
    if not root_path.is_dir():
        raise ValueError("reference root is not a directory: %s" % root_path)

    entry_keys = {"upstream_path", "local_path", "url", "sha256", "bytes"}

    def _check_upstream(rel):
        if not isinstance(rel, str) or not rel:
            raise ValueError("upstream_path must be a non-empty string")
        if "\\" in rel or ":" in rel or "\x00" in rel or rel.startswith("/"):
            raise ValueError("unsafe upstream_path %r" % rel)
        for part in rel.split("/"):
            if part in ("", ".", "..") or part != part.strip():
                raise ValueError("unsafe upstream_path %r" % rel)

    # Phase 1: validate the entire manifest structure.
    seen_upstream = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise ValueError("files[%d] must be a mapping" % index)
        keys = set(entry.keys())
        if not entry_keys <= keys:
            raise ValueError("files[%d] keys must be exactly %s, got %r" % (index, sorted(entry_keys), sorted(map(str, keys))))
        upstream = entry["upstream_path"]
        _check_upstream(upstream)
        if upstream in seen_upstream:
            raise ValueError("duplicate upstream_path %r" % upstream)
        seen_upstream.add(upstream)

        sha = entry["sha256"]
        if not isinstance(sha, str) or sha != sha.lower() or not _SHA256.fullmatch(sha):
            raise ValueError("files[%d] sha256 must be 64 lowercase hex" % index)
        size = entry["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("files[%d] bytes must be a non-negative int" % index)

        url = entry["url"]
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError("files[%d] url must be https" % index)
        if any(ch in url for ch in "?#\\ \t\r\n"):
            raise ValueError("files[%d] url contains forbidden characters" % index)
        suffix = "/" + RELEASE_COMMIT + "/" + upstream
        if not url.endswith("/" + upstream) or not url.endswith(suffix) or len(url) <= len("https://") + len(suffix):
            raise ValueError("files[%d] url must end with /<RELEASE_COMMIT>/%s" % (index, upstream))
        if RELEASE_COMMIT not in url[len("https://"):].split("/"):
            raise ValueError("files[%d] url lacks exact release commit segment" % index)

        local = entry["local_path"]
        if not isinstance(local, str) or not local or "\x00" in local:
            raise ValueError("files[%d] local_path must be a non-empty string" % index)

    missing = [key for key in _MODULE_SUFFIX if key not in seen_upstream]
    if missing:
        raise ValueError("pins missing required import targets: %r" % missing)

    # Phase 2: resolve every local path inside root.
    resolved = []
    seen_local = set()
    for index, entry in enumerate(files):
        local = entry["local_path"]
        try:
            if os.path.isabs(local):
                native = Path(local)
                if ".." in native.parts or ".." in local.replace("\\", "/").split("/"):
                    raise ValueError("files[%d] local_path contains '..'" % index)
                path = native.resolve(strict=True)
                try:
                    path.relative_to(root_path)
                except ValueError:
                    raise ValueError("files[%d] local_path escapes reference root" % index) from None
            else:
                path = Path(_resolve_inside(root_path, local))
        except OSError as exc:
            raise ValueError("files[%d] local_path cannot be resolved: %r" % (index, local)) from exc
        if not path.is_file():
            raise ValueError("files[%d] local_path is not a file: %s" % (index, path))
        key = os.path.normcase(str(path))
        if key in seen_local:
            raise ValueError("duplicate resolved local_path %s" % path)
        seen_local.add(key)
        upstream = entry["upstream_path"]
        if upstream in _MODULE_SUFFIX:
            try:
                expected = Path(_resolve_inside(root_path, upstream))
            except OSError as exc:
                raise ValueError("import target %r missing under root" % upstream) from exc
            if os.path.normcase(str(expected)) != key:
                raise ValueError("import target %r must resolve to root/upstream_path" % upstream)
        resolved.append((upstream, path, entry))

    # Phase 3: read and hash every entry before returning anything.
    sources = {}
    for upstream, path, entry in resolved:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise ValueError("cannot read %s" % path) from exc
        if len(data) != entry["bytes"]:
            raise ValueError("%s: size %d != pinned %d" % (upstream, len(data), entry["bytes"]))
        digest = _sha256_hex(data)
        if digest != entry["sha256"]:
            raise ValueError("%s: sha256 %s != pinned %s" % (upstream, digest, entry["sha256"]))
        sources[upstream] = (data, path, entry)
    return sources



def _exec_verified(name, data, path):
    spec = importlib.machinery.ModuleSpec(name, None, origin=str(path))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def load_reference(reference_root, pins_sha):
    '''Verify SOURCE_PINS.json and all pinned sources, then import the three files.

    Returns (dpc_module, transforms_module, source_audit).
    '''
    if not isinstance(pins_sha, str) or not _SHA256.fullmatch(pins_sha.lower()):
        raise ValueError('pins_sha is required: 64-hex SHA-256 of SOURCE_PINS.json')
    root = Path(reference_root).resolve(strict=True)
    pins_bytes = _resolve_inside(root, PINS_FILENAME).read_bytes()
    pins_digest = _sha256_hex(pins_bytes)
    if pins_digest != pins_sha.lower():
        raise ValueError('SOURCE_PINS.json does not match the trusted pins_sha')
    sources = _verified_sources(root, json.loads(pins_bytes.decode('utf-8')))

    # All pinned bytes are verified above; only now is code executed, and exactly
    # those verified bytes are compiled (no re-read between check and use).
    package = f'_openhands_ref_{uuid.uuid4().hex}'
    before = _openhands_modules()
    try:
        for name in (package,) + tuple(f'{package}.{sub}' for sub in _SUBPACKAGES):
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            sys.modules[name] = importlib.util.module_from_spec(spec)
        modules = {}
        for upstream in (ST_GCN, DPC_RNN, POSE_TRANSFORMS):
            data, path, _ = sources[upstream]
            modules[upstream] = _exec_verified(f'{package}.{_MODULE_SUFFIX[upstream]}', data, path)
        leaked = _openhands_modules() - before
        if leaked:
            raise ImportError(f'official sources pulled in the openhands package: {sorted(leaked)}')
    except BaseException:
        for name in [n for n in sys.modules if n == package or n.startswith(package + '.')]:
            del sys.modules[name]
        raise

    audit = {
        'full_commit': RELEASE_COMMIT,
        'pins_sha256': pins_digest,
        'synthetic_package': package,
        'openhands_package_imported': False,
        'files': [
            {
                'upstream_path': upstream,
                'local_path': sources[upstream][2]['local_path'],
                'url': sources[upstream][2]['url'],
                'sha256': sources[upstream][2]['sha256'],
                'bytes': sources[upstream][2]['bytes'],
                'module': modules[upstream].__name__,
            }
            for upstream in (ST_GCN, DPC_RNN, POSE_TRANSFORMS)
        ],
    }
    return modules[DPC_RNN], modules[POSE_TRANSFORMS], audit


# ---------------------------------------------------------------------- model

def _check_structure(model, num_classes):
    extra = sorted({key.split('.', 1)[0] for key in model.state_dict()} - _MODEL_CHILDREN)
    if extra:
        raise ValueError(f'unexpected stateful submodules (e.g. DPC prediction network): {extra}')
    if not isinstance(getattr(model, 'conv_encoder', None), nn.Module):
        raise ValueError('official model has no conv_encoder module')
    agg = getattr(model, 'agg', None)
    if not (isinstance(agg, nn.GRU) and agg.input_size == HIDDEN_DIM and agg.hidden_size == HIDDEN_DIM
            and agg.num_layers == 1 and agg.batch_first and not agg.bidirectional and agg.bias):
        raise ValueError('agg is not GRU(256, 256, batch_first=True)')
    if set(agg.state_dict()) != set(GRU_KEYS):
        raise ValueError('agg state keys differ from single-layer GRU contract')
    bn = getattr(model, 'final_bn', None)
    if not (isinstance(bn, nn.BatchNorm1d) and bn.num_features == HIDDEN_DIM and bn.track_running_stats):
        raise ValueError('final_bn is not BatchNorm1d(256)')
    fc = getattr(model, 'final_fc', None)
    if not (isinstance(fc, nn.Sequential) and len(fc) == 2
            and isinstance(fc[0], nn.Dropout) and fc[0].p == 0.5
            and isinstance(fc[1], nn.Linear) and fc[1].in_features == HIDDEN_DIM
            and fc[1].out_features == num_classes):
        raise ValueError('final_fc is not Sequential(Dropout(0.5), Linear(256, num_classes))')


def _validate_state(label, source, target, expected_count):
    if not isinstance(source, Mapping):
        raise TypeError(f'{label} state must be a mapping of tensors')
    if len(target) != expected_count:
        raise ValueError(f'official {label} exposes {len(target)} state keys, expected {expected_count}')
    source_keys, target_keys = set(source), set(target)
    if source_keys != target_keys:
        raise ValueError(
            f'{label} state keys mismatch: missing={sorted(target_keys - source_keys)[:5]} '
            f'unexpected={sorted(source_keys - target_keys, key=str)[:5]}')
    for key, reference in target.items():
        value = source[key]
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError(f'{label}.{key} is not a dense tensor')
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise ValueError(f'{label}.{key} expected {reference.dtype}{tuple(reference.shape)}, '
                             f'got {value.dtype}{tuple(value.shape)}')
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError(f'{label}.{key} contains nonfinite values')


def _head_state(model):
    return {k: v for k, v in model.state_dict().items() if k.startswith(('final_bn.', 'final_fc.'))}


def make_reference_model(dpc_module, encoder_state, gru_state, num_classes=93, seed=20261005):
    '''Build the release DPC_RNN_Finetuner; load caller tensors into conv_encoder and agg only.

    final_bn/final_fc keep the official fresh initialisation. Returns (model in eval mode, audit).
    '''
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError('num_classes must be a positive int')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an int')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = dpc_module.DPC_RNN_Finetuner(
            num_class=num_classes, pred_steps=2, in_channels=2, hidden_channels=64,
            hidden_dim=HIDDEN_DIM, dropout=0.5,
            graph_args={'layout': 'mediapipe-27', 'strategy': 'spatial'},
            edge_importance_weighting=True)
    _check_structure(model, num_classes)

    encoder_target = model.conv_encoder.state_dict()
    _validate_state('conv_encoder', encoder_state, encoder_target, ENCODER_KEY_COUNT)
    _validate_state('agg', gru_state, model.agg.state_dict(), len(GRU_KEYS))

    # The checkpoint graph must equal the graph the official code just built.
    graph_keys = sorted(k for k in encoder_target if k.rsplit('.', 1)[-1] == 'A')
    if not graph_keys:
        raise ValueError('official conv_encoder has no graph buffer A')
    for key in graph_keys:
        if not torch.equal(encoder_state[key].detach().cpu(), encoder_target[key].detach().cpu()):
            raise ValueError(f'checkpoint graph {key} differs from official mediapipe-27 spatial graph')

    encoder_digest, gru_digest = _state_digest(encoder_state), _state_digest(gru_state)
    head_digest = _state_digest(_head_state(model))
    frozen = sorted(name for name, p in model.named_parameters() if not p.requires_grad)

    model.conv_encoder.load_state_dict(dict(encoder_state), strict=True)
    model.agg.load_state_dict(dict(gru_state), strict=True)
    for module, source in ((model.conv_encoder, encoder_state), (model.agg, gru_state)):
        for key, value in module.state_dict().items():
            if not torch.equal(value.detach().cpu(), source[key].detach().cpu()):
                raise RuntimeError(f'loaded tensor {key} does not equal caller tensor')
    if _state_digest(_head_state(model)) != head_digest:
        raise RuntimeError('new head changed during encoder/GRU load')
    bn = model.final_bn
    if not (torch.all(bn.running_mean == 0) and torch.all(bn.running_var == 1)
            and int(bn.num_batches_tracked) == 0):
        raise RuntimeError('final_bn statistics are not freshly initialised')

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.eval()
    audit = {
        'seed': seed,
        'num_classes': num_classes,
        'encoder_keys': len(encoder_target),
        'encoder_state_sha256': encoder_digest,
        'gru_keys': list(GRU_KEYS),
        'gru_state_sha256': gru_digest,
        'graph_keys_exact_match': graph_keys,
        'new_head_sha256': head_digest,
        'new_head_loaded_from_caller': False,
        'final_bn_fresh_statistics': True,
        'frozen_by_official_init': frozen,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'mode': 'eval',
    }
    return model, audit


# --------------------------------------------------------------------- inputs

def _require_dimension(name, value):
    if value is None:
        raise ValueError(f'BLOCKER: image {name} is missing; supply it from verified source metadata '
                         '(it is never inferred from coordinates)')
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f'BLOCKER: image {name} must be a real number, got {type(value).__name__}')
    as32 = np.float32(value)
    if not math.isfinite(float(value)) or not np.isfinite(as32) or as32 <= 0:
        raise ValueError(f'BLOCKER: image {name} must be finite and positive, got {value!r}')
    return as32


def prepare_windows(xy_pixels, present, layout, width, height, transforms_module):
    '''Pixel landmarks -> release-equivalent input tensor [1, 6, 2, 10, 27].

    Steps: zero absent joints, scale by width/height, select 27 joints, official
    clip-level CenterAndScaleNormalize over all T, then prepend zeros / keep
    first 60 frames, then split into 6 contiguous 10-frame windows.
    '''
    if layout not in LAYOUTS:
        raise ValueError(f'unsupported layout {layout!r}; expected one of {sorted(LAYOUTS)}')
    num_joints, joint_index = LAYOUTS[layout]
    w = _require_dimension('width', width)
    h = _require_dimension('height', height)

    xy = np.asarray(xy_pixels)
    if xy.dtype.kind not in 'iuf' or xy.ndim != 3 or xy.shape[1:] != (num_joints, 2):
        raise ValueError(f'{layout} expects numeric xy of shape (T, {num_joints}, 2), got {xy.dtype}{xy.shape}')
    original_frames = xy.shape[0]
    if original_frames == 0:
        raise ValueError('sequence must be nonempty')
    mask = np.asarray(present)
    if mask.dtype != np.bool_ or mask.shape != xy.shape[:2]:
        raise ValueError(f'present must be bool of shape {xy.shape[:2]}, got {mask.dtype}{mask.shape}')

    frames = xy.astype(np.float32)  # always a fresh copy; caller arrays are untouched
    bad = mask & ~np.isfinite(frames).all(axis=-1)
    if bad.any():
        t, v = (int(i) for i in np.argwhere(bad)[0])
        raise ValueError(f'present joint has nonfinite coordinate at frame {t}, joint {v}')
    frames[~mask] = 0.0
    frames /= np.array([w, h], dtype=np.float32)
    if not np.isfinite(frames).all():
        raise ValueError('nonfinite value after resolution scaling')

    selected = frames[:, list(joint_index), :]  # T, 27, 2
    x = torch.from_numpy(np.ascontiguousarray(selected.transpose(2, 0, 1)))  # C, T, V
    left, right = SHOULDER_JOINTS
    shoulder = float(torch.linalg.vector_norm(x[:, :, left] - x[:, :, right], dim=0).mean())
    if not math.isfinite(shoulder) or shoulder <= 0:
        raise ValueError(f'ineligible clip: mean shoulder distance {shoulder!r} (must be finite and > 0)')

    transform = transforms_module.CenterAndScaleNormalize(
        reference_points_preset=NORMALIZE_PRESET, frame_level=False)
    normalized = transform({'frames': x.clone()})['frames']
    if (not isinstance(normalized, torch.Tensor) or normalized.shape != x.shape
            or normalized.dtype != torch.float32):
        raise ValueError('official normalization returned an unexpected tensor')
    if not bool(torch.isfinite(normalized).all()):
        raise ValueError('official normalization produced nonfinite values')

    padded = max(TARGET_FRAMES - original_frames, 0)
    truncated = max(original_frames - TARGET_FRAMES, 0)
    if padded:
        normalized = torch.cat([normalized.new_zeros(2, padded, normalized.shape[2]), normalized], dim=1)
    clip = normalized[:, :TARGET_FRAMES]
    windows = (clip.reshape(2, NUM_WINDOWS, WINDOW_FRAMES, clip.shape[2])
               .permute(1, 0, 2, 3).unsqueeze(0).contiguous())

    diagnostics = {
        'layout': layout,
        'original_frames': original_frames,
        'padded_frames': padded,
        'truncated_frames': truncated,
        'output_shape': list(windows.shape),
        'width': float(width),
        'height': float(height),
        'source_missing_count': int((~mask).sum()),
        'selected_missing_count': int((~mask[:, list(joint_index)]).sum()),
        'shoulder_mean_distance': shoulder,
        'input_xy_sha256': _array_digest(xy),
        'present_sha256': _array_digest(mask),
        'output_sha256': _array_digest(windows.numpy()),
    }
    return windows, diagnostics
