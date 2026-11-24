"""Opt-in frozen T3-F versus preserved C, on one identical C input tensor.

Inference only: no requested word, training, calibration or recording persistence.
"""
from functools import lru_cache
import hashlib
import sys
import threading
import time

import numpy as np

from ..core.observation import sha256_file
from . import word100, word100_candidate as C

KEY = 'word100_tcn_t3f_v1'
PIN = 'b0faa0df5055204295c60b78c596a0b490cd63ef14f30225582a85ea39855846'
SEED, EPOCH = 20261013, 69
UNIT = C.LAB / 'ops/word100_frozen_c_temporal_20261019'
CHECKPOINT = UNIT / 'T3F_20261013_best.pt'
CONTEXT_PIN = '75415a14d95d8d93a6ba5152ddd7f88f2ed32c31c8f64dc1274df95acccdc308'
SOURCE_PINS = {
    'temporal.py': 'bfaa4b594dbb5573495641db8faa49682b6a5c43b7a2e381d1c7c0864ee6bb30',
    'frozen_temporal.py': '096d11435aed4315a37e7bb054c0074eb809e5b14734e6c41196d59dd1b94c96',
}
_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _load():
    import torch
    with C._LOCK:
        base = C._load()
    if sha256_file(CHECKPOINT) != PIN:
        raise ValueError('T3F_CHECKPOINT_HASH_MISMATCH')
    for name, pin in SOURCE_PINS.items():
        if sha256_file(UNIT / name) != pin:
            raise ValueError('T3F_SOURCE_PIN_MISMATCH: ' + name)
    # Load the exact evaluated implementation without retaining a generic import alias.
    phase = C._module('wis_web_t3f_phase', UNIT / 'temporal.py')
    previous = sys.modules.get('temporal')
    try:
        sys.modules['temporal'] = phase
        frozen = C._module('wis_web_t3f_frozen', UNIT / 'frozen_temporal.py')
    finally:
        if previous is None:
            sys.modules.pop('temporal', None)
        else:
            sys.modules['temporal'] = previous
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    d = word100.config()
    expected = dict(n_classes=100, in_channels=280, width=32,
                    dilations=[1, 2, 4, 8], kernel_size=3, dropout=0.1)
    if not (ck['classes'] == d['labels'] and ck['word_codes'] == d['word_codes']
            and ck['arm'] == 'T3F' and ck['seed'] == SEED and ck['epoch'] == EPOCH
            and ck['model_config'] == expected and ck['in_channels'] == 280
            and ck['base_checkpoint_sha256'] == C.PIN and ck['context_sha256'] == CONTEXT_PIN):
        raise ValueError('T3F_CHECKPOINT_CONTRACT_MISMATCH')
    model = frozen.FrozenCTemporal(**expected)
    model.load_state_dict(ck['state_dict'], strict=True)
    model.eval()
    model.assert_frozen()
    if sum(p.numel() for p in model.parameters()) != 47236:
        raise ValueError('T3F_PARAMETER_COUNT_MISMATCH')
    shared = model.frozen_state()
    if shared.keys() != base.state_dict().keys() or any(
            not torch.equal(v, base.state_dict()[k]) for k, v in shared.items()):
        raise ValueError('T3F_FROZEN_C_STATE_MISMATCH')
    return model


def identity():
    error = None
    try:
        with _LOCK:
            _load()
        status = 'READY'
    except Exception as exc:
        status, error = 'MODEL_UNAVAILABLE', str(exc)
    d = word100.config()
    out = dict(model_key=KEY, model_id=KEY, display_name='시간 순서 보완 T3-F',
        checkpoint_sha256=PIN, model_status=status, fixed_seed=SEED, selected_epoch=EPOCH,
        baseline_model_key=C.KEY, baseline_checkpoint_sha256=C.PIN,
        trained_class_count=100, class_labels=d['labels'], class_word_codes=d['word_codes'],
        feature_schema='original_C_280_frozen_C_soft_temporal3', preprocessing_pipeline=C.PIPELINE,
        preprocessing_pins=dict(C._PINS), temporal_source_pins=dict(SOURCE_PINS),
        training_context_sha256=CONTEXT_PIN, opted_in=True, no_fallback_to_default=True,
        inference_only=True, data_use='evaluation_only_never_training_tuning_or_model_selection')
    if error:
        out['reason'] = error
    return out


def _empty(key, pin, epoch):
    return dict(schema='wis_word100_recognition_v1', model_key=key, checkpoint_sha256=pin,
        candidate_count=100, ranking=[], accepted=None, execution_assessment='UNCONFIGURED',
        is_a_verdict=False, fixed_seed=SEED, selected_epoch=epoch,
        preprocessing_pipeline=C.PIPELINE, no_fallback_to_default=True)


def unavailable(reason):
    return tuple(dict(out, model_status='CANDIDATE_OBSERVATION_UNAVAILABLE', reason=reason)
                 for out in (_empty(KEY, PIN, EPOCH), _empty(C.KEY, C.PIN, C.EPOCH)))


def recognise_pair(obs):
    """One channel build, same tensor and padding for both independent frozen forwards."""
    readings = [_empty(KEY, PIN, EPOCH), _empty(C.KEY, C.PIN, C.EPOCH)]
    try:
        import torch
        clip, reason, support = C.build_candidate_channels(obs)
        for out in readings:
            out['support'] = support
        if clip is None:
            return tuple(dict(out, model_status='INSUFFICIENT_OBSERVATION', reason=reason) for out in readings)
        values = np.ascontiguousarray(clip['x'], dtype='<f4')
        if values.ndim != 2 or values.shape[1] != 280 or not np.isfinite(values).all():
            raise ValueError('INVALID_C_CHANNELS')
        input_hash = hashlib.sha256(values.tobytes()).hexdigest()
        x = torch.from_numpy(values[None])
        pad = torch.ones(x.shape[:2], dtype=torch.float32)
        # Failures are separate: an unavailable T3-F never gets labelled with C scores.
        for out, load, lock in ((readings[0], _load, _LOCK), (readings[1], C._load, C._LOCK)):
            out.update(input_sha256=input_hash, input_shape=list(values.shape),
                       input_encoding='C-contiguous little-endian float32; padding=all_real')
            start = time.perf_counter()
            try:
                with lock:
                    model = load()
                    with torch.inference_mode():
                        logits = model(x, pad)[0].cpu().numpy()
                if logits.shape != (100,) or not np.isfinite(logits).all():
                    raise ValueError('INVALID_MODEL_LOGITS')
                out.update(model_status='READY', ranking=word100.rank_logits(logits), logits=logits.tolist(),
                           score_meaning='uncalibrated model score, not execution correctness')
            except Exception as exc:
                out.update(model_status='MODEL_ERROR', reason=str(exc), error_type=type(exc).__name__)
            out['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 2)
    except Exception as exc:
        for out in readings:
            out.update(model_status='MODEL_ERROR', reason=str(exc), error_type=type(exc).__name__)
    return tuple(readings)
