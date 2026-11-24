"""Past-only adapter for the existing shared Conv+GRU action observer.

The model observes generic attributes and movement directions. It has not been
trained to recognize SEN0354 stages or certify a whole greeting. This module
does not load the word TCN, inspect references, save input, or train a model.
"""
from pathlib import Path
from collections import deque
import hashlib
import copy
import importlib.util
import json
import time

import numpy as np
import torch


U = Path(__file__).resolve().parent
O = U.parent
CHECKPOINT = O / 'word100_broad_factorized_learning_20261110/runs/coupled_42/model.pt'
CHECKPOINT_SHA256 = '80aba3b28006270308834bf5ba88277d7b26c8dce01c11fc962ce5c21b8cea17'
MODEL_ID = 'shared100_conv_bigru_coupled42_20261110'
ROLE = '기존 100단어 공통 시간 합성곱+GRU의 손 특성과 구간 방향 관측'


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_model():
    digest = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise RuntimeError('Expected frozen action checkpoint has changed')
    rc = _module('hello_action_requirement', O / 'word100_guide_requirement_learning_20261111/requirement_core.py')
    net = rc.frozen_observer()
    net.eval()
    net.requires_grad_(False)
    adapter = _module('hello_action_rtmw_adapter', O / 'word100_rtmw_shared_input_20261108/adapter.py')
    return rc, net, adapter, digest


class ActionObserver:
    """One encoder call per received update, with a three-observation delay.

    Raw history is bounded. Its channels are rebuilt on every update, so scale
    estimates never include future frames or a precomputed reference clip.
    Movement endpoints must still be present in that history. Dropped endpoints
    and unobserved intervals are not silently filled or treated as completed.
    """
    def __init__(self, max_frames=320, max_age_us=12_000_000, lag_frames=3,
                 max_gap_us=300_000, min_direction_samples=3, candidate_path=None,
                 vertical_candidate_path=None, enable_vertical=True):
        self.max_frames = int(max_frames)
        self.max_age_us = int(max_age_us)
        self.lag_frames = int(lag_frames)
        self.max_gap_us = int(max_gap_us)
        self.min_direction_samples = int(min_direction_samples)
        if self.max_frames < 7 or self.max_age_us <= 0 or self.lag_frames < 0 or self.max_gap_us <= 0:
            raise ValueError('Invalid action history configuration')
        if self.min_direction_samples not in (3, 4):
            raise ValueError('Direction requires three or four genuine observations')
        self.rc, self.net, self.adapter, self.checkpoint_sha256 = _load_model()
        admission=U/'action_integration/ADMITTED.json'
        if candidate_path is None and admission.exists():
            selected=json.loads(admission.read_text(encoding='utf-8'))
            if selected.get('status')!='public_regression_passed':raise RuntimeError('Action head not admitted')
            candidate_path=Path(selected['checkpoint'])
            if hashlib.sha256(candidate_path.read_bytes()).hexdigest()!=selected['sha256']:raise RuntimeError('Admitted action head changed')
        self.head_sha256=None
        if candidate_path is not None:
            candidate_path=Path(candidate_path)
            candidate=torch.load(candidate_path,map_location='cpu',weights_only=True)
            if candidate['base_sha256']!=CHECKPOINT_SHA256:raise RuntimeError('Action head base mismatch')
            self.net.motion_head.load_state_dict(candidate['state'])
            self.net.eval();self.net.requires_grad_(False)
            self.head_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        self.base = self.net.shared.base
        self.vertical_head=None;self.vertical_head_sha256=None
        vertical_admission=U/'seated_diagnosis_20261116/VERTICAL_ADMITTED.json'
        if enable_vertical and vertical_candidate_path is None and vertical_admission.exists():
            selected=json.loads(vertical_admission.read_text(encoding='utf-8'))
            if selected.get('status')!='public_regression_passed' or selected.get('axes')!=[0]:
                raise RuntimeError('Vertical direction head not admitted')
            vertical_candidate_path=Path(selected['checkpoint'])
            if hashlib.sha256(vertical_candidate_path.read_bytes()).hexdigest()!=selected['sha256']:
                raise RuntimeError('Admitted vertical direction head changed')
        if enable_vertical and vertical_candidate_path is not None:
            vertical_candidate_path=Path(vertical_candidate_path)
            ck=torch.load(vertical_candidate_path,map_location='cpu',weights_only=True)
            if ck['base_sha256']!=CHECKPOINT_SHA256:raise RuntimeError('Vertical direction base mismatch')
            self.vertical_head=copy.deepcopy(self.net.motion_head)
            self.vertical_head.load_state_dict(ck['state']);self.vertical_head.eval();self.vertical_head.requires_grad_(False)
            self.vertical_head_sha256=hashlib.sha256(vertical_candidate_path.read_bytes()).hexdigest()
        self.provenance = dict(
            model_id=MODEL_ID, checkpoint=str(CHECKPOINT),
            checkpoint_sha256=self.checkpoint_sha256,
            parameter_count=sum(p.numel() for p in self.net.parameters()),
            parameters_frozen=all(not p.requires_grad for p in self.net.parameters()),
            target_word_input=False, word_tcn_loaded=False, base_trained_this_session=False,
            direction_head_finetuned=bool(self.head_sha256),
            normalization='received_raw_prefix_only',
            whole_word_correct=None, greeting_stage_training=False,
            prediction_scope='generic_attributes_and_segment_direction',
            supplemental_motion_head_sha256=self.head_sha256,
            supplemental_source_scope='one approved public reference plus previous public FIT replay'if self.head_sha256 else None,
            vertical_motion_head_sha256=self.vertical_head_sha256,
            vertical_motion_scope='axis0 only; original forearm/other axes preserved; public view/wait augmentations'if self.vertical_head is not None else None,
            probability_is_accuracy=False,
            minimum_direction_observations=self.min_direction_samples,
            sparse_direction_policy='3 genuine distinct-time points, all observed, >=33333us; 9 interpolated features are not 9 observations',
        )
        self.reset()

    def reset(self):
        self.frames = deque()
        self.last_ts = None
        self.dimensions = None
        self.next_frame_id = 0
        self.calls = 0
        self._clear_cache()

    def _clear_cache(self):
        self.channels = None
        self.x = None
        self.h = None
        self.assessed_timestamp_us = None

    def _result(self, status, message, started, reason=None, movements=None,
                attributes=None, history_reset=None):
        lag = None if self.assessed_timestamp_us is None or self.last_ts is None else (self.last_ts - self.assessed_timestamp_us) / 1000.
        return dict(status=status, message=message, reason=reason,
                    assessed_timestamp_us=self.assessed_timestamp_us,
                    observation_lag_ms=lag, compute_ms=(time.perf_counter()-started)*1000.,
                    model_id=MODEL_ID, role=ROLE, calls=self.calls,
                    prefix_frames=len(self.frames), history_reset=history_reset,
                    movements=movements or {}, attributes=attributes,
                    provenance=dict(self.provenance), whole_word_correct=None)

    def update(self, xy, score, ts, width, height):
        started = time.perf_counter()
        self._clear_cache()  # An invalid new observation must never reuse old evidence.
        try:
            xy = np.asarray(xy, np.float32)
            score = np.asarray(score, np.float32)
            ts, width, height = int(ts), int(width), int(height)
            if xy.shape != (133, 2) or score.shape != (133,) or width <= 0 or height <= 0:
                raise ValueError('Invalid RTMW input')
            if self.last_ts is not None and ts <= self.last_ts:
                raise ValueError('Nonmonotonic action timestamp')
        except (ValueError, TypeError, OverflowError):
            return self._result('uncertain', '새 관측을 확인하기 어려워요.', started, 'invalid_input')

        reset_reason = None
        if self.last_ts is not None and ts-self.last_ts > self.max_gap_us:
            reset_reason = 'observation_gap'
        elif self.dimensions is not None and self.dimensions != (width, height):
            reset_reason = 'source_dimensions_changed'
        if reset_reason:
            self.reset()
        self.dimensions = (width, height)
        self.last_ts = ts
        self.frames.append((xy.copy(), score.copy(), ts, self.next_frame_id))
        self.next_frame_id += 1
        while len(self.frames) > self.max_frames or ts-self.frames[0][2] > self.max_age_us:
            self.frames.popleft()
        # Match the existing hand observer's seven received observations and
        # assess the same latest-minus-three image, not the newest raw image.
        if len(self.frames) < max(7, self.lag_frames+4):
            return self._result('pending', '앞뒤 움직임을 조금 더 보고 있어요.', started,
                                'insufficient_history', history_reset=reset_reason)
        frame_rows = list(self.frames)
        assessed_index = len(frame_rows)-1-self.lag_frames
        assessed_ts = frame_rows[assessed_index][2]
        if ts-assessed_ts > self.max_gap_us:
            return self._result('uncertain', '움직임 관측이 늦어 다시 확인하고 있어요.', started,
                                'assessment_too_old', history_reset=reset_reason)
        try:
            raw_xy = np.stack([r[0] for r in frame_rows])
            raw_score = np.stack([r[1] for r in frame_rows])
            stamps = np.asarray([r[2] for r in frame_rows], np.int64)
            ids = np.asarray([r[3] for r in frame_rows], np.int64)
            channels = self.adapter.convert(raw_xy, raw_score, stamps, ids, width, height)
            x = channels['tcn280'].astype(np.float32, copy=True)
            pose = self.rc.prior.shared.normalize(x)
            if not np.isfinite(pose).all():
                raise ValueError('Nonfinite normalized observation')
            with torch.inference_mode():
                self.calls += 1
                h = torch.nn.functional.gelu(self.base.input(torch.from_numpy(pose[None])))
                h = h.transpose(1, 2)
                for layer in self.base.temporal:
                    h = layer(h)
                h, _ = self.base.context(h.transpose(1, 2))
                probabilities = self.base.attributes(h).sigmoid()[0].numpy()
            if not np.isfinite(probabilities).all() or not bool(torch.isfinite(h).all()):
                raise ValueError('Nonfinite action output')
            self.channels, self.x, self.h = channels, x, h
            self.assessed_timestamp_us = int(assessed_ts)
            available = self.rc.prior.shared.available(x)[assessed_index].astype(bool)
            attr = dict(names=list(self.rc.FEATURE_NAMES),
                        probabilities=[float(p) if ok else None for p, ok in zip(probabilities[assessed_index], available)],
                        available=available.tolist())
            start_ts = int(stamps[max(0, assessed_index-9)])
            moves = dict(
                along_opposite_forearm=self.movement(start_ts, assessed_ts, 2, 'R', 'L'),
                right_wrist_height=self.movement(start_ts, assessed_ts, 0, 'R', 'L'),
                left_wrist_height=self.movement(start_ts, assessed_ts, 0, 'L', 'R'),
            )
            observed = any(m['state'] >= 0 for m in moves.values())
            return self._result('observed' if observed else 'uncertain',
                                '학습한 동작 모델로 움직임을 확인하고 있어요.' if observed else '움직임 방향은 조금 더 확인할게요.',
                                started, None if observed else 'direction_not_established',
                                movements=moves, attributes=attr, history_reset=reset_reason)
        except (ValueError, RuntimeError, FloatingPointError, self.adapter.PilotDataError) as exc:
            self._clear_cache()
            return self._result('uncertain', '몸과 팔의 움직임을 충분히 확인하기 어려워요.',
                                started, 'action_observation_unavailable:'+type(exc).__name__,
                                history_reset=reset_reason)

    def _geometry(self, stamps, a, b, axis, moving, anchor, min_samples):
        """Keep the original four-point policy, with an explicit three-point option.

        Only a window of exactly three fully observed points takes this branch.
        The nine interpolated positions are model features, never new evidence.
        No source frame, word identity or expected direction affects eligibility.
        """
        duration = int(stamps[b]-stamps[a])
        if b-a+1 != 3 or min_samples != 3:
            g, eligible, evidence = self.rc.prior.geometry(self.x, stamps, a, b, axis, moving, anchor)
            evidence = dict(evidence, actual_observed_samples=int(evidence['observed']),
                            window_samples=b-a+1, duration_us=duration, three_point_window=False,
                            interpolated_feature_samples=9 if eligible else 0, synthesized_observations=0)
            return g, eligible, evidence
        values, good = self.rc.prior.wrist_series(self.x, self.rc.prior.motion.KINDS[axis], moving, anchor)
        values, good = values[a:b+1], good[a:b+1]
        tt = np.asarray(stamps[a:b+1], np.float64)
        eligible = bool(good.all() and np.isfinite(values).all() and np.all(np.diff(tt) > 0) and duration >= 33333)
        evidence = dict(coverage=float(good.mean()), observed=int(good.sum()),
                        actual_observed_samples=int(good.sum()), window_samples=3,
                        duration_us=duration, three_point_window=True,
                        interpolated_feature_samples=9 if eligible else 0, synthesized_observations=0,
                        sampling_policy='three_fully_observed_distinct_times_min33333us')
        if not eligible:
            return np.zeros(58, np.float32), False, evidence
        sampled = np.stack([np.interp(np.linspace(tt[0], tt[-1], 9), tt, values[:, j]) for j in range(4)], 1)
        delta = sampled[-1]-sampled[0]
        g = np.r_[(sampled-sampled[0]).ravel(), sampled[0], delta,
                  np.abs(np.diff(sampled, axis=0)).mean(0), np.ptp(sampled, axis=0),
                  np.eye(5)[axis], float(good.mean())].astype(np.float32)
        evidence.update(signed_change=float(delta[0]), trajectory=sampled[:, 0].tolist(),
                        body_wrist_only=True, contact_verified=False)
        return g, True, evidence

    def movement(self, start_us, end_us, axis, moving='R', anchor='L', min_samples=None):
        """Classify an observed interval: 0 decrease, 1 hold, 2 increase.

        A class is an observed direction, not a correct/incorrect sign verdict.
        This method reuses the latest prefix latent and does not re-run its GRU.
        """
        result = dict(eligible=False, state=-1, probabilities=None,
                      start_us=start_us, end_us=end_us, axis=axis,
                      moving=moving, anchor=anchor, reason=None, evidence=None)
        if self.channels is None or self.h is None or self.assessed_timestamp_us is None:
            result['reason'] = 'no_current_prefix'
            return result
        try:
            start_us, end_us, axis = int(start_us), int(end_us), int(axis)
            minimum = self.min_direction_samples if min_samples is None else int(min_samples)
            result.update(start_us=start_us, end_us=end_us, axis=axis)
            if moving not in ('L', 'R') or anchor not in ('L', 'R') or moving == anchor or axis not in range(5) or minimum not in (3, 4):
                raise ValueError('Invalid movement query')
            if end_us < start_us or end_us > self.assessed_timestamp_us:
                result['reason'] = 'interval_not_yet_assessed'
                return result
            stamps = self.channels['timestamp_us']
            aa, bb = np.flatnonzero(stamps == start_us), np.flatnonzero(stamps == end_us)
            if len(aa) != 1 or len(bb) != 1:
                result['reason'] = 'endpoint_outside_received_history'
                return result
            a, b = int(aa[0]), int(bb[0])
            g, eligible, evidence = self._geometry(stamps, a, b, axis, moving, anchor, minimum)
            result['evidence'] = evidence
            if not eligible:
                result['reason'] = 'insufficient_observed_geometry'
                return result
            with torch.inference_mode():
                p = self._direction_logits(g,a,b,axis).softmax(-1)[0].numpy()
            if not np.isfinite(p).all():
                raise ValueError('Nonfinite direction output')
            best = int(p.argmax())
            result.update(eligible=True, probabilities=p.tolist(),
                          state=best if p[best] >= .65 else -1,
                          reason=None if p[best] >= .65 else 'model_uncertain')
        except (ValueError, TypeError, RuntimeError, FloatingPointError, OverflowError) as exc:
            result['reason'] = 'movement_unavailable:'+type(exc).__name__
        return result

    def _direction_logits(self,g,a,b,axis):
        """Task-specific vertical readout; never mutate shared network heads."""
        geom=torch.from_numpy(g[None])
        if axis!=0 or self.vertical_head is None:
            return self.net.movement(self.h,geom,torch.tensor([[0,a,b]],dtype=torch.long))
        z=(geom-self.net.gm)/self.net.gs
        seq=self.net.motion_conv(z[:,:36].reshape(-1,9,4).transpose(1,2)).transpose(1,2)
        _,hn=self.net.motion_gru(seq)
        context=torch.cat([self.h[:,a],self.h[:,b],self.h[:,b]-self.h[:,a]],-1)
        feature=torch.cat([hn[0],hn[1],z,context],-1)
        return self.vertical_head(feature)
