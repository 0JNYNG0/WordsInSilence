# SPDX-License-Identifier: Apache-2.0
# stgcn_core.py - generic ST-GCN encoder (encoder-only) and a safe tensor-byte checkpoint loader.
#
# Source attribution: the module structure and state_dict key layout follow the public
# AI4Bharat/OpenHands STGCN encoder (https://github.com/AI4Bharat/OpenHands,
# openhands/models/encoder/graph/st_gcn.py), licensed under the Apache License 2.0.
# That code derives from the public ST-GCN reference (Yan, Xiong and Lin, AAAI 2018).
# This file is an independent re-implementation written only from the public
# architecture description. Licensed under the Apache License, Version 2.0:
# http://www.apache.org/licenses/LICENSE-2.0
#
# Scope: encoder-only transfer. No GRU aggregator and no future predictor (DPC heads)
# are built or loaded here. This is not the original DPC fine-tuning procedure.
'''Generic ST-GCN encoder and safe checkpoint tensor loader.

Model input: float tensor of shape (N, 2, T, 27). Encoder output: (N, 256).

The loader never unpickles anything. It reads raw storage bytes from a torch ZIP
archive (nested inside an outer ZIP) and rebuilds tensors from a separately
prevalidated JSON metadata file using bounded numpy copies and torch.as_strided.
'''
from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import zipfile
import zlib
from typing import Any, NoReturn

import numpy as np
import torch
from torch import nn

__all__ = [
    'NUM_SUBSETS',
    'NUM_JOINTS',
    'IN_CHANNELS',
    'OUT_FEATURES',
    'BLOCK_SPECS',
    'ENCODER_PREFIX',
    'EXCLUDED_PREFIXES',
    'validate_adjacency',
    'ConvTemporalGraphical',
    'STGCNBlock',
    'STGCNEncoder',
    'STGCNClassifier',
    'CheckpointValidationError',
    'read_encoder_tensors',
    'load_encoder_checkpoint',
]

NUM_SUBSETS = 3
NUM_JOINTS = 27
IN_CHANNELS = 2
OUT_FEATURES = 256
TEMPORAL_KERNEL_SIZE = 9

# (in_channels, out_channels, temporal_stride, residual_enabled)
BLOCK_SPECS: tuple[tuple[int, int, int, bool], ...] = (
    (2, 64, 1, False),
    (64, 64, 1, True),
    (64, 64, 1, True),
    (64, 64, 1, True),
    (64, 128, 2, True),
    (128, 128, 1, True),
    (128, 128, 1, True),
    (128, 256, 2, True),
    (256, 256, 1, True),
    (256, 256, 1, True),
)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def validate_adjacency(A: Any) -> torch.Tensor:
    '''Validate an externally supplied adjacency stack and return a float32 copy.

    Requirements: torch.Tensor or numpy.ndarray, real-valued, shape (3, 27, 27),
    all values finite.
    '''
    if isinstance(A, torch.Tensor):
        t = A.detach().to(device='cpu').clone()
    elif isinstance(A, np.ndarray):
        if A.dtype.kind not in 'biuf':
            raise TypeError(f'A must be a real numeric array, got dtype {A.dtype}')
        t = torch.from_numpy(np.array(A, dtype=np.float64, copy=True))
    else:
        raise TypeError('A must be a torch.Tensor or numpy.ndarray')
    if t.is_complex():
        raise TypeError('A must be real-valued')
    expected = (NUM_SUBSETS, NUM_JOINTS, NUM_JOINTS)
    if tuple(t.shape) != expected:
        raise ValueError(f'A must have shape {expected}, got {tuple(t.shape)}')
    t = t.to(torch.float64)
    if not bool(torch.isfinite(t).all()):
        raise ValueError('A contains non-finite values')
    return t.to(torch.float32).contiguous()


def _zero_residual(x: torch.Tensor) -> int:
    '''Residual for the first block: contributes zero and holds no state.'''
    return 0


def _identity_residual(x: torch.Tensor) -> torch.Tensor:
    '''Residual for equal-channel, stride-1 blocks: identity, no state.'''
    return x


class ConvTemporalGraphical(nn.Module):
    '''Spatial graph convolution: 1x1 conv to K*C channels, then aggregation.

    The aggregation is mathematically equivalent to
    einsum(nkctv,kvw->nctw) and is implemented as a sum over the static
    k = 0, 1, 2 of matmul(x[:, k], A[k]). The reference_einsum flag switches
    to the einsum form and exists only for exact-equivalence testing.
    '''

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = NUM_SUBSETS, reference_einsum: bool = False) -> None:
        super().__init__()
        if kernel_size != NUM_SUBSETS:
            raise ValueError(f'kernel_size must be {NUM_SUBSETS}')
        self.kernel_size = kernel_size
        self.reference_einsum = bool(reference_einsum)
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        n, kc, t, v = x.shape
        x = x.view(n, NUM_SUBSETS, kc // NUM_SUBSETS, t, v)
        if self.reference_einsum:
            return torch.einsum('nkctv,kvw->nctw', x, A)
        out = torch.matmul(x[:, 0], A[0])
        out = out + torch.matmul(x[:, 1], A[1])
        out = out + torch.matmul(x[:, 2], A[2])
        return out


class STGCNBlock(nn.Module):
    '''One ST-GCN block: gcn -> tcn, plus residual, then ReLU.'''

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 residual: bool = True, reference_einsum: bool = False) -> None:
        super().__init__()
        pad = (TEMPORAL_KERNEL_SIZE - 1) // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, NUM_SUBSETS, reference_einsum)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, (TEMPORAL_KERNEL_SIZE, 1),
                      stride=(stride, 1), padding=(pad, 0), bias=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.0),
        )
        if not residual:
            self.residual = _zero_residual
            self.residual_kind = 'zero'
        elif in_channels == out_channels and stride == 1:
            self.residual = _identity_residual
            self.residual_kind = 'identity'
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
            self.residual_kind = 'conv'
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x)


class STGCNEncoder(nn.Module):
    '''ST-GCN encoder with OpenHands-compatible state keys.

    State keys: A, data_bn.*, st_gcn_networks.{0..9}.(gcn.conv|tcn.{0,2,3}|residual.{0,1}).*,
    edge_importance.{0..9}. Input (N, 2, T, 27); output (N, 256).
    '''

    def __init__(self, A: Any, *, reference_einsum: bool = False) -> None:
        super().__init__()
        a = validate_adjacency(A)
        self.register_buffer('A', a)
        self.data_bn = nn.BatchNorm1d(IN_CHANNELS * NUM_JOINTS)
        self.st_gcn_networks = nn.ModuleList([
            STGCNBlock(cin, cout, stride, residual=res, reference_einsum=reference_einsum)
            for (cin, cout, stride, res) in BLOCK_SPECS
        ])
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(NUM_SUBSETS, NUM_JOINTS, NUM_JOINTS))
            for _ in range(len(BLOCK_SPECS))
        ])
        self.n_out_features = OUT_FEATURES
        self.set_reference_einsum(reference_einsum)

    def set_reference_einsum(self, flag: bool) -> None:
        '''Switch all blocks between matmul (default) and reference einsum aggregation.'''
        for block in self.st_gcn_networks:
            block.gcn.reference_einsum = bool(flag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise ValueError('input must be a 4-D tensor (N, 2, T, 27)')
        n, c, t, v = x.shape
        if c != IN_CHANNELS or v != NUM_JOINTS or t < 1 or n < 1:
            raise ValueError(f'input must have shape (N, {IN_CHANNELS}, T, {NUM_JOINTS}), got {tuple(x.shape)}')
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        for block, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = block(x, self.A * importance)
        return x.mean(dim=(2, 3))


class STGCNClassifier(nn.Module):
    '''Optional classifier: STGCNEncoder followed by Linear(256, num_classes).

    To load pretrained encoder weights, pass classifier.encoder to
    load_encoder_checkpoint.
    '''

    def __init__(self, A: Any, num_classes: int, *, reference_einsum: bool = False) -> None:
        super().__init__()
        if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 1:
            raise ValueError('num_classes must be a positive int')
        self.encoder = STGCNEncoder(A, reference_einsum=reference_einsum)
        self.classifier = nn.Linear(OUT_FEATURES, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ---------------------------------------------------------------------------
# Safe tensor-byte loader
# ---------------------------------------------------------------------------

class CheckpointValidationError(ValueError):
    '''Raised when an archive, metadata file or tensor fails validation.'''


ENCODER_PREFIX = 'model.conv_encoder.'
# Non-encoder checkpoint entries that are intentionally never loaded.
EXCLUDED_PREFIXES: tuple[tuple[str, str], ...] = (
    ('agg', 'model.agg.'),
    ('network_pred', 'model.network_pred.'),
)

_DTYPES: dict[str, tuple[np.dtype, torch.dtype]] = {
    'Float': (np.dtype(np.float32), torch.float32),
    'Long': (np.dtype(np.int64), torch.int64),
}
_REQUIRED_FIELDS = ('key', 'shape', 'stride', 'storage_offset', 'dtype', 'storage_key', 'storage_bytes')
_STORAGE_KEY_RE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}')
_MAX_INDEX = 2 ** 62


def _fail(msg: str) -> NoReturn:
    raise CheckpointValidationError(msg)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _preview(items: Any, limit: int = 20) -> str:
    items = [str(i) for i in items]
    s = ', '.join(items[:limit])
    if len(items) > limit:
        s += f', ... (+{len(items) - limit} more)'
    return '[' + s + ']'


def _normalize_sha256(value: Any) -> str:
    if not isinstance(value, str):
        _fail('expected_sha256 must be a hex string')
    s = value.strip().lower()
    if len(s) != 64 or any(ch not in '0123456789abcdef' for ch in s):
        _fail('expected_sha256 must be 64 hexadecimal characters')
    return s


def _read_file_bounded(path: Any, limit: int, what: str) -> bytes:
    p = os.fspath(path)
    if not os.path.isfile(p):
        _fail(f'{what} not found or not a regular file: {p}')
    size = os.path.getsize(p)
    if size > limit:
        _fail(f'{what} too large: {size} bytes > limit {limit}')
    with open(p, 'rb') as fh:
        data = fh.read(limit + 1)
    if len(data) > limit:
        _fail(f'{what} exceeds limit {limit} bytes')
    return data


def _reject_json_constant(name: str) -> NoReturn:
    _fail(f'metadata JSON contains non-standard constant {name!r}')


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            _fail(f'metadata JSON contains duplicate object key {k!r}')
        out[k] = v
    return out


def _check_extent(key: str, shape: list[int], stride: list[int], offset: int, storage_numel: int) -> None:
    numel = math.prod(shape)
    if numel == 0:
        if offset > storage_numel:
            _fail(f'tensor {key!r}: storage_offset {offset} beyond storage numel {storage_numel}')
        return
    last = offset + sum((d - 1) * s for d, s in zip(shape, stride))
    if last >= storage_numel:
        _fail(f'tensor {key!r}: view exceeds storage (last index {last} >= storage numel {storage_numel})')


def _validate_metadata(obj: Any, *, max_tensors: int, max_ndim: int, max_storage_bytes: int) -> list[dict[str, Any]]:
    if not isinstance(obj, dict):
        _fail('metadata root must be a JSON object')
    tensors = obj.get('tensors')
    if not isinstance(tensors, list):
        _fail('metadata must contain a tensors list')
    if not tensors:
        _fail('metadata tensors list is empty')
    if len(tensors) > max_tensors:
        _fail(f'metadata lists {len(tensors)} tensors > limit {max_tensors}')
    seen: set[str] = set()
    storage_info: dict[str, tuple[str, int]] = {}
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(tensors):
        where = f'metadata tensors[{idx}]'
        if not isinstance(raw, dict):
            _fail(f'{where} must be an object')
        missing = [f for f in _REQUIRED_FIELDS if f not in raw]
        if missing:
            _fail(f'{where} missing fields {missing}')
        key = raw['key']
        if not isinstance(key, str) or not key or len(key) > 1024:
            _fail(f'{where} has invalid key')
        if key in seen:
            _fail(f'duplicate tensor key {key!r}')
        seen.add(key)
        shape = raw['shape']
        stride = raw['stride']
        if not isinstance(shape, list) or not all(_is_int(d) and 0 <= d <= _MAX_INDEX for d in shape):
            _fail(f'{where} ({key!r}) has invalid shape')
        if len(shape) > max_ndim:
            _fail(f'{where} ({key!r}) has too many dimensions')
        if (not isinstance(stride, list) or len(stride) != len(shape)
                or not all(_is_int(s) and 1 <= s <= _MAX_INDEX for s in stride)):
            _fail(f'{where} ({key!r}) has invalid stride (must be positive ints, one per dim)')
        offset = raw['storage_offset']
        if not _is_int(offset) or offset < 0 or offset > _MAX_INDEX:
            _fail(f'{where} ({key!r}) has invalid storage_offset')
        dtype = raw['dtype']
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            _fail(f'{where} ({key!r}) has unsupported dtype {dtype!r}; allowed: Float, Long')
        skey = raw['storage_key']
        if not isinstance(skey, str) or _STORAGE_KEY_RE.fullmatch(skey) is None:
            _fail(f'{where} ({key!r}) has invalid storage_key')
        nbytes = raw['storage_bytes']
        if not _is_int(nbytes) or nbytes < 0 or nbytes > max_storage_bytes:
            _fail(f'{where} ({key!r}) has invalid storage_bytes')
        itemsize = _DTYPES[dtype][0].itemsize
        if nbytes % itemsize:
            _fail(f'{where} ({key!r}) storage_bytes {nbytes} not a multiple of {itemsize}')
        prev = storage_info.get(skey)
        if prev is None:
            storage_info[skey] = (dtype, nbytes)
        elif prev != (dtype, nbytes):
            _fail(f'storage {skey!r} described inconsistently: {prev} vs {(dtype, nbytes)}')
        _check_extent(key, shape, stride, offset, nbytes // itemsize)
        out.append({
            'key': key,
            'shape': list(shape),
            'stride': list(stride),
            'storage_offset': offset,
            'dtype': dtype,
            'storage_key': skey,
            'storage_bytes': nbytes,
        })
    return out


def _read_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int, what: str) -> bytes:
    if info.flag_bits & 0x1:
        _fail(f'{what} is encrypted')
    if info.file_size > limit:
        _fail(f'{what} too large: {info.file_size} bytes > limit {limit}')
    try:
        with zf.open(info, 'r') as fh:
            data = fh.read(limit + 1)
    except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, OSError, EOFError) as exc:
        raise CheckpointValidationError(f'failed to read {what}: {exc}') from exc
    if len(data) != info.file_size:
        _fail(f'{what}: read {len(data)} bytes, header says {info.file_size}')
    if (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        _fail(f'{what}: CRC mismatch')
    return data


def _split_storage_name(name: str) -> tuple[str, str] | None:
    parts = name.split('/')
    if len(parts) == 2 and parts[0] == 'data' and parts[1]:
        return '', parts[1]
    if len(parts) == 3 and parts[0] and parts[1] == 'data' and parts[2]:
        return parts[0], parts[2]
    return None


def _bytes_to_tensor(raw: bytes, dtype_name: str, byteorder: str) -> torch.Tensor:
    np_dtype, torch_dtype = _DTYPES[dtype_name]
    src = np_dtype.newbyteorder('<' if byteorder == 'little' else '>')
    if len(raw) % src.itemsize:
        _fail('storage byte length is not a multiple of the element size')
    if len(raw) == 0:
        arr = np.empty((0,), dtype=np_dtype)
    else:
        arr = np.frombuffer(raw, dtype=src).astype(np_dtype, copy=True)
    t = torch.from_numpy(arr)
    if t.dtype != torch_dtype:
        _fail('internal dtype conversion error')
    return t


def read_encoder_tensors(
    outer_zip_path: Any,
    metadata_json_path: Any,
    expected_sha256: str,
    *,
    encoder_prefix: str = ENCODER_PREFIX,
    sha256_target: str = 'outer_zip',
    max_outer_bytes: int = 2 ** 31,
    max_inner_bytes: int = 2 ** 31,
    max_storage_bytes: int = 2 ** 30,
    max_total_storage_bytes: int = 2 ** 31,
    max_metadata_bytes: int = 64 * 2 ** 20,
    max_tensors: int = 100000,
    max_ndim: int = 8,
    max_inner_members: int = 200000,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    '''Read encoder tensors (prefix stripped) without any unpickling.

    expected_sha256 is checked against the outer ZIP file bytes by default
    (sha256_target=outer_zip) or against the inner .ckpt bytes
    (sha256_target=inner_ckpt). Returns (tensors, json_safe_audit).
    Source files are only opened read-only and are never modified.
    '''
    if sha256_target not in ('outer_zip', 'inner_ckpt'):
        raise ValueError('sha256_target must be outer_zip or inner_ckpt')
    if not isinstance(encoder_prefix, str) or not encoder_prefix:
        raise ValueError('encoder_prefix must be a non-empty string')
    expected = _normalize_sha256(expected_sha256)

    outer_bytes = _read_file_bounded(outer_zip_path, max_outer_bytes, 'outer ZIP')
    outer_sha = hashlib.sha256(outer_bytes).hexdigest()
    if sha256_target == 'outer_zip' and not hmac.compare_digest(outer_sha, expected):
        _fail(f'outer ZIP SHA-256 mismatch: expected {expected}, got {outer_sha}')

    meta_bytes = _read_file_bounded(metadata_json_path, max_metadata_bytes, 'metadata JSON')
    meta_sha = hashlib.sha256(meta_bytes).hexdigest()
    try:
        meta_obj = json.loads(meta_bytes.decode('utf-8'), object_pairs_hook=_no_duplicate_pairs,
                              parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CheckpointValidationError(f'metadata JSON is invalid: {exc}') from exc
    entries = _validate_metadata(meta_obj, max_tensors=max_tensors, max_ndim=max_ndim,
                                 max_storage_bytes=max_storage_bytes)

    try:
        outer_zf = zipfile.ZipFile(io.BytesIO(outer_bytes), 'r')
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError) as exc:
        raise CheckpointValidationError(f'outer file is not a valid ZIP: {exc}') from exc
    with outer_zf:
        outer_infos = outer_zf.infolist()
        ckpt_infos = [i for i in outer_infos if not i.is_dir() and i.filename.lower().endswith('.ckpt')]
        if len(ckpt_infos) != 1:
            _fail(f'outer ZIP must contain exactly one .ckpt member, found {len(ckpt_infos)}: '
                  f'{_preview(i.filename for i in ckpt_infos)}')
        ckpt_info = ckpt_infos[0]
        other_outer = sorted(i.filename for i in outer_infos if i is not ckpt_info)
        inner_bytes = _read_member(outer_zf, ckpt_info, max_inner_bytes, f'outer member {ckpt_info.filename!r}')
    inner_sha = hashlib.sha256(inner_bytes).hexdigest()
    if sha256_target == 'inner_ckpt' and not hmac.compare_digest(inner_sha, expected):
        _fail(f'inner .ckpt SHA-256 mismatch: expected {expected}, got {inner_sha}')

    try:
        inner_zf = zipfile.ZipFile(io.BytesIO(inner_bytes), 'r')
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, ValueError, EOFError) as exc:
        raise CheckpointValidationError(f'inner .ckpt is not a valid torch ZIP archive: {exc}') from exc
    with inner_zf:
        infos = inner_zf.infolist()
        if len(infos) > max_inner_members:
            _fail(f'inner archive has too many members: {len(infos)}')
        names = [i.filename for i in infos]
        if len(set(names)) != len(names):
            _fail('inner archive contains duplicate member names')
        total_size = sum(i.file_size for i in infos)
        if total_size > max_inner_bytes:
            _fail(f'inner archive total uncompressed size {total_size} > limit {max_inner_bytes}')
        for i in infos:
            if i.flag_bits & 0x1:
                _fail(f'inner member {i.filename!r} is encrypted')
        try:
            bad = inner_zf.testzip()
        except (zipfile.BadZipFile, zlib.error, OSError, EOFError, RuntimeError, NotImplementedError) as exc:
            raise CheckpointValidationError(f'inner archive CRC check failed: {exc}') from exc
        if bad is not None:
            _fail(f'inner archive CRC check failed for member {bad!r}')
        by_name = {i.filename: i for i in infos}

        candidates: dict[str, dict[str, zipfile.ZipInfo]] = {}
        for i in infos:
            if i.is_dir():
                continue
            split = _split_storage_name(i.filename)
            if split is None:
                continue
            pfx, sk = split
            candidates.setdefault(sk, {})[pfx] = i

        storage_meta: dict[str, tuple[str, int]] = {}
        for e in entries:
            storage_meta.setdefault(e['storage_key'], (e['dtype'], e['storage_bytes']))
        prefixes: set[str] = set()
        for sk in sorted(storage_meta):
            found = candidates.get(sk, {})
            if len(found) != 1:
                _fail(f'storage key {sk!r} must exist under exactly one archive prefix as '
                      f'<prefix>/data/{sk}; found prefixes {sorted(found)}')
            prefixes.update(found)
        if len(prefixes) != 1:
            _fail(f'storages span multiple archive prefixes: {sorted(prefixes)}')
        prefix = next(iter(prefixes))
        storage_infos = {sk: candidates[sk][prefix] for sk in storage_meta}
        for sk, (_dt, nb) in storage_meta.items():
            actual = storage_infos[sk].file_size
            if actual != nb:
                _fail(f'storage {sk!r}: metadata storage_bytes {nb} != actual {actual}')

        bo_name = f'{prefix}/byteorder' if prefix else 'byteorder'
        if bo_name in by_name:
            raw_bo = _read_member(inner_zf, by_name[bo_name], 16, 'byteorder record')
            try:
                byteorder = raw_bo.decode('ascii').strip()
            except UnicodeDecodeError as exc:
                raise CheckpointValidationError('byteorder record is not ASCII') from exc
            if byteorder not in ('little', 'big'):
                _fail(f'unsupported byteorder {byteorder!r}')
            byteorder_source = 'archive'
        else:
            byteorder = 'little'
            byteorder_source = 'default'

        enc_entries: list[tuple[str, dict[str, Any]]] = []
        excluded: dict[str, list[str]] = {name: [] for name, _ in EXCLUDED_PREFIXES}
        excluded['other'] = []
        for e in entries:
            k = e['key']
            if k.startswith(encoder_prefix):
                stripped = k[len(encoder_prefix):]
                if not stripped:
                    _fail(f'tensor key {k!r} is empty after stripping prefix')
                enc_entries.append((stripped, e))
            else:
                cat = 'other'
                for name, p in EXCLUDED_PREFIXES:
                    if k.startswith(p):
                        cat = name
                        break
                excluded[cat].append(k)
        if not enc_entries:
            _fail(f'no tensors with prefix {encoder_prefix!r} found in metadata')

        needed = sorted({e['storage_key'] for _, e in enc_entries})
        total_needed = sum(storage_meta[sk][1] for sk in needed)
        if total_needed > max_total_storage_bytes:
            _fail(f'encoder storages total {total_needed} bytes > limit {max_total_storage_bytes}')
        bases: dict[str, torch.Tensor] = {}
        for sk in needed:
            dt, _nb = storage_meta[sk]
            raw = _read_member(inner_zf, storage_infos[sk], max_storage_bytes, f'storage {sk!r}')
            bases[sk] = _bytes_to_tensor(raw, dt, byteorder)

    tensors: dict[str, torch.Tensor] = {}
    for stripped, e in enc_entries:
        base = bases[e['storage_key']]
        _np_dt, t_dt = _DTYPES[e['dtype']]
        if base.dtype != t_dt:
            _fail(f'tensor {e["key"]!r}: storage dtype mismatch') if False else None
        shape = tuple(e['shape'])
        stride = tuple(e['stride'])
        if math.prod(shape) == 0:
            t = torch.empty(shape, dtype=t_dt)
        else:
            _check_extent(e['key'], list(shape), list(stride), e['storage_offset'], int(base.numel()))
            t = torch.as_strided(base, size=shape, stride=stride, storage_offset=e['storage_offset'])
            t = t.clone(memory_format=torch.contiguous_format)
        if t.is_floating_point() and not bool(torch.isfinite(t).all()):
            _fail(f'tensor {e["key"]!r} contains non-finite values')
        tensors[stripped] = t

    audit: dict[str, Any] = {
        'loader': 'stgcn_core.read_encoder_tensors',
        'unsafe_deserialization_used': False,
        'outer_zip_path': os.path.abspath(os.fspath(outer_zip_path)),
        'outer_zip_bytes': len(outer_bytes),
        'outer_zip_sha256': outer_sha,
        'metadata_json_path': os.path.abspath(os.fspath(metadata_json_path)),
        'metadata_sha256': meta_sha,
        'sha256_target': sha256_target,
        'sha256_verified': True,
        'inner_member': ckpt_info.filename,
        'inner_ckpt_bytes': len(inner_bytes),
        'inner_ckpt_sha256': inner_sha,
        'inner_crc32': f'{ckpt_info.CRC:08x}',
        'other_outer_members_ignored': other_outer,
        'inner_members_crc_checked': len(infos),
        'archive_prefix': prefix,
        'byteorder': byteorder,
        'byteorder_source': byteorder_source,
        'num_metadata_tensors': len(entries),
        'num_metadata_storages': len(storage_meta),
        'encoder_prefix': encoder_prefix,
        'num_encoder_tensors': len(tensors),
        'encoder_keys': sorted(tensors),
        'excluded_keys': {k: sorted(v) for k, v in excluded.items()},
        'num_excluded_keys': sum(len(v) for v in excluded.values()),
        'storages_read': len(needed),
        'storage_bytes_read': total_needed,
    }
    return tensors, audit


def load_encoder_checkpoint(
    encoder: nn.Module,
    outer_zip_path: Any,
    metadata_json_path: Any,
    expected_sha256: str,
    *,
    adjacency_policy: str = 'checkpoint',
    adjacency_atol: float = 1e-6,
    **reader_kwargs: Any,
) -> dict[str, Any]:
    '''Strictly load encoder weights from a validated archive into encoder.

    The key set and every shape/dtype must match encoder.state_dict() exactly;
    otherwise CheckpointValidationError is raised and the encoder is unchanged.
    adjacency_policy:
      checkpoint   - the checkpoint A buffer replaces the current A (default);
      require_match - raise if checkpoint A differs from current A by > adjacency_atol;
      keep_current - keep the externally supplied A (still a strict full load).
    Returns a JSON-safe audit dict.
    '''
    if not isinstance(encoder, nn.Module):
        raise TypeError('encoder must be an nn.Module')
    if adjacency_policy not in ('checkpoint', 'require_match', 'keep_current'):
        raise ValueError('adjacency_policy must be checkpoint, require_match or keep_current')
    tensors, audit = read_encoder_tensors(outer_zip_path, metadata_json_path, expected_sha256, **reader_kwargs)

    target = encoder.state_dict()
    expected_keys = set(target)
    got_keys = set(tensors)
    missing = sorted(expected_keys - got_keys)
    unexpected = sorted(got_keys - expected_keys)
    if missing or unexpected:
        _fail(f'encoder state key mismatch; missing={_preview(missing)} unexpected={_preview(unexpected)}')
    mismatched = []
    for k in sorted(expected_keys):
        tv = target[k]
        cv = tensors[k]
        if tuple(tv.shape) != tuple(cv.shape) or tv.dtype != cv.dtype:
            mismatched.append(f'{k}: expected {tuple(tv.shape)}/{tv.dtype}, got {tuple(cv.shape)}/{cv.dtype}')
    if mismatched:
        _fail(f'encoder shape/dtype mismatch: {_preview(mismatched)}')

    adjacency_audit: dict[str, Any] = {'policy': adjacency_policy, 'present': 'A' in tensors}
    if 'A' in tensors:
        cur = target['A'].detach().to(device='cpu', dtype=torch.float64)
        ck = tensors['A'].to(torch.float64)
        diff = float((cur - ck).abs().max()) if cur.numel() else 0.0
        adjacency_audit['max_abs_diff_vs_current'] = diff
        if adjacency_policy == 'require_match' and not diff <= adjacency_atol:
            _fail(f'checkpoint A differs from current A (max abs diff {diff} > {adjacency_atol})')
        if adjacency_policy == 'keep_current':
            tensors = dict(tensors)
            tensors['A'] = target['A'].detach().to('cpu').clone()
        adjacency_audit['source_of_loaded_A'] = 'current' if adjacency_policy == 'keep_current' else 'checkpoint'

    snapshot = {k: v.detach().clone() for k, v in target.items()}
    try:
        result = encoder.load_state_dict(tensors, strict=True)
    except Exception:
        encoder.load_state_dict(snapshot, strict=True)
        raise
    if result.missing_keys or result.unexpected_keys:
        encoder.load_state_dict(snapshot, strict=True)
        _fail('strict load reported incompatible keys')

    audit.update({
        'load_function': 'stgcn_core.load_encoder_checkpoint',
        'strict_load': True,
        'missing_keys': [],
        'unexpected_keys': [],
        'loaded_tensor_count': len(tensors),
        'loaded_numel': int(sum(t.numel() for t in tensors.values())),
        'adjacency': adjacency_audit,
    })
    return audit


def _selftest() -> int:
    torch.manual_seed(0)
    a = torch.rand(NUM_SUBSETS, NUM_JOINTS, NUM_JOINTS)
    a = a / a.sum(dim=1, keepdim=True)
    enc = STGCNEncoder(a).eval()
    with torch.no_grad():
        y = enc(torch.randn(2, IN_CHANNELS, 16, NUM_JOINTS))
    ok = tuple(y.shape) == (2, OUT_FEATURES) and bool(torch.isfinite(y).all())
    print(json.dumps({'selftest_forward_ok': ok, 'output_shape': list(y.shape)}))
    return 0 if ok else 1


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv[1:]:
        raise SystemExit(_selftest())
    print('usage: python stgcn_core.py --selftest')
