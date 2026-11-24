"""Opt-in frozen C pipeline; never replaces M0 or learns from web captures.

Visible webcam pixels are the sensor boundary. Offline rest-wrist calibration is
not available in a webcam and is deliberately not estimated here. Detector and
anatomical assignment start fresh for each recorded take, including empty-hand
frames. This module has no camera, download, training, or persistence entrypoint.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np

from ..cli import ROOT
from ..core.observation import Observation, empty_frames, sha256_file
from . import word100

KEY = 'word100_tcn_roi_c_v1'
PIN = '283a54bd9866c97959f9d615ebc38d126ee25688821960b208cf9412389eeaab'
SEED = 20261013
EPOCH = 77
PIPELINE = 'frozen_joint_v3_visible_roi_active_span_280_v1'
LAB = Path(ROOT)
REPO = LAB.parents[1]
UNIT = LAB / 'ops/word100_body_boundary_20261017'
CHECKPOINT = UNIT / 'training_C_20261013_best.pt'
TRAIN_CONTEXT_PIN = '71a6a18eb07d0d510a76992106bceeeecea67438a875956815cad3df380d82b5'
_LOCK = threading.Lock()
_ARRAYS = ('xy_px', 'timestamp_us', 'frame_id', 'observed', 'usable', 'conf', 'hand_score')
_PINS = {
    'research/front_keyphase_v1/ops/word100_fullframe_roi_pair_20261016/roi_adapter.py': 'ac70f9ad67bdaa867c3871c0ef7eefa337527af32ba797ba7771836197a38a5e',
    'research/front_keyphase_v1/ops/word100_hand_assignment_20261016/capture_raw.py': '16179df03a1d17cd114689a39bc9376c1f8b42b2c5731097c03e809413adc3af',
    'research/front_keyphase_v1/ops/word100_hand_assignment_joint_20261016/assignment.py': '400491fa1a360d30e5d85c0cc214ada9258055da2bb2c203a90739696884a5c6',
    'research/front_keyphase_v1/ops/word100_hand_assignment_joint_20261016/SCOPE.json': '47bcb654199c117bd0f17a71640bb191c6310a5f2eefa6fcdc1b7528e4a219de',
    'research/front_keyphase_v1/ops/word100_hand_assignment_joint_20261016/POINT_CONTRACT.json': 'e85c222cd0e59bd960ba7f867f1d241ec6fa6b03bcd9ab7f2180c2ca2f6cfcbd',
    'research/front_keyphase_v1/src/wis_lab/capture/landmarker.py': '138f12c6cbe12c94e07abf1751409f1411e1c98b812c4d6cea5c33df9c9158e8',
    'research/front_keyphase_v1/src/wis_lab/core/observation.py': '88adc956aa2b9e2b34817ecbafe3fc7fe9ce70cd464d15804edd9cd5b89defff',
    'research/front_keyphase_v1/config/contract.json': 'bf64d3f9549f4ac53c41e899d90fbe08ca9eda1d2df2c857cd9472b666119e7e',
    'research/sign_model_pilot_v1/src/pilot/data.py': '5daa52beee310572c4298c74702fdabf08f0931ce38cff9f20201b341f1aa81d',
    'research/sign_model_pilot_v1/src/pilot/web_input.py': '727f6d77c9384b5b9c9d7b5aef35ce6239d4392f01e0b107c9a2fe46b569d07f',
    'research/sign_model_pilot_v1/src/pilot/model_masked_v1.py': 'fb8a963a76395fa6c6ac2281f84c968e672b4940a47c4ab932856f9c0d5f1828',
    'Tools/models/hand_landmarker.task': 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
    'Tools/models/pose_landmarker_lite.task': '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a',
}


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@lru_cache(maxsize=1)
def _frozen():
    for relative, expected in _PINS.items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError('C_PREPROCESSING_PIN_MISMATCH: ' + relative)
    if sha256_file(UNIT / 'training_CONTEXT.json') != TRAIN_CONTEXT_PIN:
        raise RuntimeError('C_TRAINING_CONTEXT_PIN_MISMATCH')
    roi = _module('wis_candidate_c_frozen_roi', LAB / 'ops/word100_fullframe_roi_pair_20261016/roi_adapter.py')
    joint = _module('wis_candidate_c_frozen_joint', LAB / 'ops/word100_hand_assignment_joint_20261016/assignment.py')
    return roi, joint


def preprocessing_identity():
    _, joint = _frozen()
    return dict(pipeline=PIPELINE, pins=dict(_PINS), assignment_sha256=joint.HandAssigner.identity(),
                training_context_sha256=TRAIN_CONTEXT_PIN,
                boundary='all75 points: 0<=x<width and 0<=y<height; visible webcam only',
                temporal='frozen active_span; >=4 span and >=4 actually observed hand frames; internal gaps retained',
                prediction='association hints only; never fills missing observations',
                detector_state='fresh per recorded take; every frame including zero hands')


@lru_cache(maxsize=1)
def _load():
    _frozen()
    if sha256_file(CHECKPOINT) != PIN:
        raise RuntimeError('C_CHECKPOINT_HASH_MISMATCH')
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    ck = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    d = word100.config()
    expected_config = dict(width=32, dilations=[1, 2, 4, 8], kernel_size=3, dropout=0.1)
    if not (ck['classes'] == d['labels'] and ck['word_codes'] == d['word_codes']
            and ck['in_channels'] == 280 and ck['config'] == expected_config
            and ck['arm'] == 'C' and ck['seed'] == SEED and ck['epoch'] == EPOCH
            and ck['training_context_sha256'] == TRAIN_CONTEXT_PIN):
        raise RuntimeError('C_CHECKPOINT_CONTRACT_MISMATCH')
    model = MaskedSignTCN(n_classes=100, in_channels=280, **expected_config)
    model.load_state_dict(ck['state_dict'], strict=True)
    if sum(p.numel() for p in model.parameters()) != 37636:
        raise RuntimeError('C_PARAMETER_COUNT_MISMATCH')
    model.eval()
    return model


def identity():
    d = word100.config()
    error = None
    try:
        with _LOCK:
            _load()
        status = 'READY'
    except Exception as exc:
        status, error = 'MODEL_UNAVAILABLE', str(exc)
    result = dict(model_key=KEY, model_id=KEY, display_name='보완 후보 C',
                  checkpoint_sha256=PIN, model_status=status, fixed_seed=SEED, selected_epoch=EPOCH,
                  trained_class_count=100, class_labels=d['labels'], class_word_codes=d['word_codes'],
                  feature_schema='pilot.data.build_channels:280', preprocessing_pipeline=PIPELINE,
                  preprocessing_pins=dict(_PINS), training_context_sha256=TRAIN_CONTEXT_PIN,
                  opted_in=True, no_fallback_to_default=True,
                  data_use='evaluation_only_never_training_tuning_or_model_selection')
    if error:
        result['reason'] = error
    return result


def finalize_observation(obs, width=None, height=None):
    """Strict visible-pixel mask and metadata; preserves every frame and timestamp."""
    roi, joint = _frozen()
    width = width if width is not None else obs.sidecar.get('frame_width', obs.sidecar.get('width'))
    height = height if height is not None else obs.sidecar.get('frame_height', obs.sidecar.get('height'))
    if (not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer))
            or width <= 0 or height <= 0):
        raise ValueError('C_VISIBLE_FRAME_DIMENSIONS_REQUIRED')
    out = roi.apply_roi(obs, (0, 0, int(width), int(height)))
    out.sidecar.update(frame_width=int(width), frame_height=int(height),
                       hand_assignment_policy='frozen_joint_v3', assignment_sha256=joint.HandAssigner.identity(),
                       candidate_pipeline=PIPELINE, model_key=KEY, checkpoint_sha256=PIN,
                       detector_state='fresh_per_recorded_take', missing_policy='no_interpolation_or_prediction',
                       training_context_sha256=TRAIN_CONTEXT_PIN)
    return out


def build_candidate_channels(obs):
    """Frozen training whole-view input, with strict mask BEFORE active-span gates."""
    roi, _ = _frozen()
    from pilot.web_input import to_native_like, active_span
    if (obs.sidecar.get('candidate_pipeline') != PIPELINE
            or obs.sidecar.get('hand_assignment_policy') != 'frozen_joint_v3'):
        raise ValueError('C_OBSERVATION_PIPELINE_REQUIRED')
    masked = finalize_observation(obs)
    _, present = to_native_like(masked.xy_px, masked.usable)
    span = active_span(present)
    if span is None:
        return None, 'NO_USABLE_HAND_FRAME', dict(span_frames=0, observed_hand_frames=0, observed_hand_fraction=0.0)
    lo, hi = map(int, span)
    n = hi - lo + 1
    observed = int(present[lo:hi + 1, 25:67].any(axis=1).sum())
    stats = dict(span_frames=n, observed_hand_frames=observed,
                 observed_hand_fraction=round(observed / n, 4), active_span_first_index=lo, active_span_last_index=hi)
    if n < 4:
        return None, 'TOO_FEW_USABLE_FRAMES', stats
    if observed < 4:
        return None, 'TOO_FEW_OBSERVED_HAND_FRAMES', stats
    sliced = Observation(**{key: getattr(masked, key)[lo:hi + 1].copy() for key in _ARRAYS}, sidecar=dict(masked.sidecar))
    return roi.build_roi_channels(sliced), None, stats


def recognise(obs):
    """Blind all100 ranking. There is deliberately no requested-word argument."""
    out = dict(schema='wis_word100_recognition_v1', model_key=KEY, checkpoint_sha256=PIN,
               candidate_count=100, ranking=[], accepted=None, execution_assessment='UNCONFIGURED',
               is_a_verdict=False, fixed_seed=SEED, selected_epoch=EPOCH,
               preprocessing_pipeline=PIPELINE, no_fallback_to_default=True)
    start = time.perf_counter()
    try:
        import torch
        clip, reason, support = build_candidate_channels(obs)
        out['support'] = support
        if clip is None:
            out.update(model_status='INSUFFICIENT_OBSERVATION', reason=reason)
        else:
            with _LOCK:
                model = _load()
                x = torch.from_numpy(clip['x'][None].astype(np.float32))
                pad = torch.ones(x.shape[:2], dtype=torch.float32)
                with torch.inference_mode():
                    logits = model(x, pad)[0].cpu().numpy()
            if logits.shape != (100,) or not np.isfinite(logits).all():
                raise ValueError('C_INVALID_LOGITS')
            out.update(model_status='READY', ranking=word100.rank_logits(logits), logits=logits.tolist(),
                       score_meaning='uncalibrated model score, not execution correctness')
    except Exception as exc:
        out.update(model_status='MODEL_ERROR', reason=str(exc), error_type=type(exc).__name__)
    out['elapsed_ms'] = round((time.perf_counter() - start) * 1000, 2)
    return out


def _capture_type():
    return _module('wis_candidate_c_frozen_capture', LAB / 'ops/word100_hand_assignment_20261016/capture_raw.py').RawCaptureLandmarker


class CandidateLandmarker:
    """One take only. Construct before recording starts; never process preview frames."""

    def __init__(self, contract):
        _, joint = _frozen()
        frozen_contract = json.loads((LAB / 'config/contract.json').read_text('utf-8'))
        if contract != frozen_contract:
            raise ValueError('C_CAPTURE_CONTRACT_MISMATCH')
        self.contract = contract
        self.assigner = joint.HandAssigner()
        self.raw = _capture_type()(contract)
        self.frame_count = 0
        self.last_timestamp_us = None
        self.dimensions = None
        self.failed = False
        self.closed = False
        self._extractor_id = hashlib.sha256((self.raw.extractor_id() + PIPELINE + joint.HandAssigner.identity()).encode()).hexdigest()

    def extractor_id(self):
        return self._extractor_id

    def process(self, rgb, timestamp_us, out, row):
        if self.closed or self.failed:
            raise RuntimeError('C_CAPTURE_CLOSED_OR_FAILED')
        try:
            if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
                raise ValueError('C_EXPECTED_UINT8_RGB')
            h, w = map(int, rgb.shape[:2])
            if w <= 0 or h <= 0 or (self.dimensions is not None and self.dimensions != (w, h)):
                raise ValueError('C_FRAME_DIMENSIONS_CHANGED')
            ts = int(timestamp_us)
            if ts < 0 or (self.last_timestamp_us is not None and ts <= self.last_timestamp_us):
                raise ValueError('C_NON_MONOTONIC_RECORDING_TIMESTAMP')
            if self.raw.raw_rows:
                raise RuntimeError('C_UNCONSUMED_RAW_FRAME')
            out['timestamp_us'][row] = ts
            self.raw.process(rgb, ts, out, row)
            if len(self.raw.raw_rows) != 1:
                raise RuntimeError('C_EXPECTED_ONE_RAW_FRAME')
            raw = self.raw.raw_rows.pop()
            self._associate_frame(raw, out, row, ts, w, h)
            single = Observation(**{key: out[key][row:row + 1].copy() for key in _ARRAYS})
            masked = finalize_observation(single, w, h)
            for key in _ARRAYS:
                out[key][row] = getattr(masked, key)[0]
            self.dimensions, self.last_timestamp_us = (w, h), ts
            self.frame_count += 1
        except Exception:
            self.failed = True
            self.raw.raw_rows.clear()
            self.raw._raw_frame = None
            # Prevent a partial baseline hand result from becoming a C observation.
            blank = empty_frames(1)
            for key in ('xy_px', 'observed', 'usable', 'conf', 'hand_score'):
                out[key][row] = blank[key][0]
            raise

    def _associate_frame(self, raw, out, row, ts, w, h):
        n = int(raw['raw_count'])
        chosen, _ = self.assigner.assign(raw['hand_xy_px'][:n], raw['raw_label'][:n], raw['raw_score'][:n],
                                        raw['pose_xy_px'], raw['pose_usable'], ts, w, h)
        for key in ('xy_px', 'conf'):
            out[key][row, 33:] = np.nan
        out['observed'][row, 33:] = False
        out['usable'][row, 33:] = False
        out['hand_score'][row] = np.nan
        margin = self.contract['usable_rule']['bounds_margin_ratio']
        threshold = self.contract['usable_rule']['hand_score_threshold']
        for det, side in chosen.items():
            if side is None:
                continue
            sl = slice(33 + 21 * side, 54 + 21 * side)
            pts = raw['hand_xy_px'][det].astype(np.float32)
            score = float(raw['raw_score'][det])
            out['xy_px'][row, sl] = pts
            out['observed'][row, sl] = True
            out['hand_score'][row, side] = score
            out['usable'][row, sl] = (np.isfinite(pts).all(1) & (pts[:, 0] >= -margin * w)
                                    & (pts[:, 0] <= w * (1 + margin)) & (pts[:, 1] >= -margin * h)
                                    & (pts[:, 1] <= h * (1 + margin)) & (score >= threshold))

    def finalize_observation(self, obs):
        if self.failed or self.dimensions is None or obs.n_frames != self.frame_count:
            raise ValueError('C_INCOMPLETE_CAPTURE')
        if int(obs.timestamp_us[-1]) != self.last_timestamp_us:
            raise ValueError('C_CAPTURE_TIMESTAMP_MISMATCH')
        obs.sidecar = dict(obs.sidecar, extractor_id=self.extractor_id())
        return finalize_observation(obs, *self.dimensions)

    def close(self):
        if not self.closed:
            self.closed = True
            self.raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
