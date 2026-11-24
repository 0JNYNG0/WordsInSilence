"""Causal palm-track side assignment - an ISOLATED candidate.

The operating Landmarker is not modified. Prediction links detections across frames and NEVER
produces coordinates: only which slot a detection is written into can change.
"""
from __future__ import annotations

import json, pathlib, sys
import numpy as np

L = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.capture.landmarker import Landmarker                       # noqa: E402
from wis_lab.core.observation import POSE_L_WRIST, POSE_R_WRIST         # noqa: E402

PALM = (0, 5, 9, 13, 17)          # wrist + the four MCPs
_P = json.loads((pathlib.Path(__file__).resolve().parent / 'FROZEN_PLAN.json')
                .read_text(encoding='utf-8'))['coefficients_fixed_now']
TRACK_GATE = _P['TRACK_GATE']['value']
RATIO = _P['RATIO']['value']
INIT_GATE = _P['INIT_GATE']['value']
MAX_GAP_S = _P['MAX_GAP_S']['value']
MAX_SPEED = _P['MAX_SPEED']['value']


class Track:
    __slots__ = ('palm', 't', 'vel', 'prev_palm', 'prev_t')

    def __init__(self, palm, t):
        self.palm, self.t = palm, t
        self.vel = np.zeros(2, np.float64)
        self.prev_palm, self.prev_t = None, None

    def predict(self, t, sw):
        dt = t - self.t
        step = self.vel * dt
        n = float(np.linalg.norm(step))
        cap = MAX_SPEED * sw * max(dt, 0.0)
        if n > cap > 0:
            step = step * (cap / n)
        return self.palm + step

    def update(self, palm, t, sw):
        dt = t - self.t
        if dt > 0:
            v = (palm.mean(0) - self.palm.mean(0)) / dt
            n = float(np.linalg.norm(v))
            cap = MAX_SPEED * sw
            self.vel = v * (cap / n) if n > cap > 0 else v
        self.prev_palm, self.prev_t = self.palm, self.t
        self.palm, self.t = palm, t


class TemporalLandmarker(Landmarker):
    """Only _assign_sides changes. Everything else is the operating extractor."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.tracks = {}                       # side -> Track
        self._t_s = 0.0
        self.stats = {'matched': 0, 'initialised': 0, 'unassigned_ambiguous': 0,
                      'unassigned_no_evidence': 0, 'tracks_expired': 0,
                      'handedness_agreed': 0, 'handedness_compared': 0}
        # diagnostic only - recording never changes a decision
        self.reasons = {'refused_track_gate': 0, 'refused_ratio': 0, 'refused_init_gate': 0,
                        'refused_side_taken_this_frame': 0, 'refused_live_track_blocks_init': 0,
                        'refused_no_pose_reference': 0}
        self.frame_log = []

    def process(self, rgb, timestamp_us, out, row):
        self._t_s = timestamp_us / 1e6         # the REAL elapsed time, not a frame index
        return super().process(rgb, timestamp_us, out, row)

    # ---- the candidate ---------------------------------------------------
    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        n = len(hand_res.hand_landmarks)
        scores = [float(hand_res.handedness[i][0].score) for i in range(n)]
        labels = [hand_res.handedness[i][0].category_name for i in range(n)]
        t = self._t_s

        if not (pose_usable[11] and pose_usable[12]):
            return super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        sw = float(np.linalg.norm(pose_xy[11] - pose_xy[12]))
        if not np.isfinite(sw) or sw <= 1e-6:
            return super()._assign_sides(hand_res, pose_xy, pose_usable, w, h)

        for s in list(self.tracks):                       # long observation break
            if t - self.tracks[s].t > MAX_GAP_S:
                del self.tracks[s]
                self.stats['tracks_expired'] += 1

        palms = [np.array([[hand_res.hand_landmarks[i][j].x * w,
                            hand_res.hand_landmarks[i][j].y * h] for j in PALM], np.float64)
                 for i in range(n)]

        cost = {}
        for i in range(n):
            for s, tr in self.tracks.items():
                cost[(i, s)] = float(np.linalg.norm(palms[i] - tr.predict(t, sw), axis=1).mean() / sw)

        out, taken_h, taken_s = {}, set(), set()
        for (i, s) in sorted(cost, key=lambda k: cost[k]):
            if i in taken_h or s in taken_s or cost[(i, s)] > TRACK_GATE:
                continue
            other = 1 - s
            rival = cost.get((i, other))
            if rival is not None and cost[(i, s)] > RATIO * rival:
                self.stats['unassigned_ambiguous'] += 1       # not clearly better: refuse
                self.reasons['refused_ratio'] += 1
                taken_h.add(i)
                out[i] = (None, scores[i])
                continue
            out[i] = (s, scores[i])
            taken_h.add(i); taken_s.add(s)
            self.tracks[s].update(palms[i], t, sw)
            self.stats['matched'] += 1
            self.stats['handedness_compared'] += 1
            if (0 if labels[i] == 'Left' else 1) == s:
                self.stats['handedness_agreed'] += 1

        ref = {s: pose_xy[idx] for s, idx in ((0, POSE_L_WRIST), (1, POSE_R_WRIST))
               if pose_usable[idx]}
        for i in range(n):                                    # bootstrap / re-connection
            if i in out:
                continue
            if not ref:
                out[i] = (None, scores[i]); self.stats['unassigned_no_evidence'] += 1
                self.reasons['refused_no_pose_reference'] += 1; continue
            wrist = palms[i][0]
            near = min(ref, key=lambda s: float(np.linalg.norm(wrist - ref[s])))
            d = float(np.linalg.norm(wrist - ref[near]) / sw)
            if near in taken_s or near in self.tracks or d > INIT_GATE:
                out[i] = (None, scores[i]); self.stats['unassigned_no_evidence'] += 1
                if near in taken_s:
                    self.reasons['refused_side_taken_this_frame'] += 1
                elif near in self.tracks:
                    self.reasons['refused_live_track_blocks_init'] += 1
                else:
                    self.reasons['refused_init_gate'] += 1
                if cost:
                    best = min((cost[(i, s)] for s in (0, 1) if (i, s) in cost), default=None)
                    if best is not None and best > TRACK_GATE:
                        self.reasons['refused_track_gate'] += 1
                continue
            out[i] = (near, scores[i])
            taken_s.add(near)
            self.tracks[near] = Track(palms[i], t)
            self.stats['initialised'] += 1
        self.frame_log.append({'t': round(t, 4),
                               'assigned': {('L' if s == 0 else 'R'): i
                                            for i, (s, _) in out.items() if s is not None},
                               'palms': [pp.tolist() for pp in palms]})
        return out
