"""Recovery variant - ONE change against the preserved failed candidate.

A live track on a side no longer blocks re-initialisation by itself. A detection may take the side
over and reset the track, but ONLY when it meets the initialisation evidence that was already
specified (nearest usable pose wrist AND within INIT_GATE). Ambiguous detections are still refused,
one side never receives two detections, and no coefficient was re-searched.

The original candidate file is untouched; this subclasses it.
"""
import pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'temporal_assignment_candidate_20260926'))
from temporal_landmarker import (TemporalLandmarker, Track, PALM,          # noqa: E402
                                 TRACK_GATE, RATIO, INIT_GATE, MAX_GAP_S, MAX_SPEED)
from wis_lab.core.observation import POSE_L_WRIST, POSE_R_WRIST            # noqa: E402


class RecoveryLandmarker(TemporalLandmarker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.stats['recovered_takeover'] = 0
        self.reasons['refused_init_gate_after_unblock'] = 0

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        n = len(hand_res.hand_landmarks)
        scores = [float(hand_res.handedness[i][0].score) for i in range(n)]
        labels = [hand_res.handedness[i][0].category_name for i in range(n)]
        t = self._t_s

        if not (pose_usable[11] and pose_usable[12]):
            return super(TemporalLandmarker, self)._assign_sides(hand_res, pose_xy, pose_usable, w, h)
        sw = float(np.linalg.norm(pose_xy[11] - pose_xy[12]))
        if not np.isfinite(sw) or sw <= 1e-6:
            return super(TemporalLandmarker, self)._assign_sides(hand_res, pose_xy, pose_usable, w, h)

        for s in list(self.tracks):
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
                self.stats['unassigned_ambiguous'] += 1
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
        for i in range(n):
            if i in out:
                continue
            if not ref:
                out[i] = (None, scores[i]); self.stats['unassigned_no_evidence'] += 1
                self.reasons['refused_no_pose_reference'] += 1; continue
            wrist = palms[i][0]
            near = min(ref, key=lambda s: float(np.linalg.norm(wrist - ref[s])))
            d = float(np.linalg.norm(wrist - ref[near]) / sw)
            # ---- THE ONE CHANGE: a live track no longer blocks on its own -------------
            if near in taken_s or d > INIT_GATE:
                out[i] = (None, scores[i]); self.stats['unassigned_no_evidence'] += 1
                if near in taken_s:
                    self.reasons['refused_side_taken_this_frame'] += 1
                else:
                    self.reasons['refused_init_gate'] += 1
                    if near in self.tracks:
                        self.reasons['refused_init_gate_after_unblock'] += 1
                continue
            if near in self.tracks:
                self.stats['recovered_takeover'] += 1
            out[i] = (near, scores[i])
            taken_s.add(near)
            self.tracks[near] = Track(palms[i], t)          # takeover resets the track
            self.stats['initialised'] += 1
        self.frame_log.append({'t': round(t, 4),
                               'assigned': {('L' if s == 0 else 'R'): i
                                            for i, (s, _) in out.items() if s is not None},
                               'palms': [pp.tolist() for pp in palms]})
        return out
