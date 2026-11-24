"""Start-provisional variant - ONE change against the FROZEN arm-consistency candidate.

When both pose wrists are unusable (shoulders usable) and NO live track exists, a detection that
matched no track may be written to the side its MediaPipe handedness label proposes, but only if
its wrist lies on that side's half of the body, the side is free, and no other unmatched detection
proposes the same side. It never creates or updates a track, so a wrong start label cannot
propagate. No new coefficient. The arm candidate file is untouched; this subclasses it.
"""
import pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / 'arm_consistency_candidate_20260926'))
from arm_landmarker import (ArmLandmarker, Track, PALM, TRACK_GATE, RATIO,      # noqa: E402
                            INIT_GATE, MAX_GAP_S, TemporalLandmarker, POSE_L_WRIST, POSE_R_WRIST)


class StartLandmarker(ArmLandmarker):
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

        # ---- THE ONE CHANGE: drop pairs that both pose wrists contradict -----------------
        if pose_usable[POSE_L_WRIST] and pose_usable[POSE_R_WRIST]:
            pw = {0: pose_xy[POSE_L_WRIST], 1: pose_xy[POSE_R_WRIST]}
            for (i, s) in list(cost):
                d_own = float(np.linalg.norm(palms[i][0] - pw[s]) / sw)
                d_oth = float(np.linalg.norm(palms[i][0] - pw[1 - s]) / sw)
                if d_oth <= INIT_GATE < d_own:
                    del cost[(i, s)]
                    self.reasons['pair_removed_arm_contradiction'] += 1
                    self.removed_log.append((round(t, 4), i, s))
        # ----------------------------------------------------------------------------------

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
        pending = []
        for i in range(n):
            if i in out:
                continue
            if not ref:
                pending.append(i)                            # decided below, never forced
                continue
            wrist = palms[i][0]
            near = min(ref, key=lambda s: float(np.linalg.norm(wrist - ref[s])))
            d = float(np.linalg.norm(wrist - ref[near]) / sw)
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
            self.tracks[near] = Track(palms[i], t)
            self.stats['initialised'] += 1
        # ---- THE ONE CHANGE: provisional start when no pose wrist and no live track ----
        if pending:
            mid = (pose_xy[11] + pose_xy[12]) / 2.0
            prop = {i: (0 if labels[i] == 'Left' else 1) for i in pending}
            for i in pending:
                s = prop[i]
                axis = pose_xy[11 if s == 0 else 12] - mid
                half_ok = float((palms[i][0] - mid) @ axis) > 0.0
                clash = any(prop[j] == s for j in pending if j != i)
                if self.tracks or s in taken_s or clash or not half_ok:
                    out[i] = (None, scores[i]); self.stats['unassigned_no_evidence'] += 1
                    self.reasons['refused_no_pose_reference'] += 1
                    key = ('provisional_refused_live_track' if self.tracks else
                           'provisional_refused_side_taken' if s in taken_s else
                           'provisional_refused_same_label' if clash else
                           'provisional_refused_body_half')
                    self.reasons[key] = self.reasons.get(key, 0) + 1
                    continue
                out[i] = (s, scores[i])                      # output only: NO track is created
                taken_s.add(s)
                self.stats['provisional_assigned'] = self.stats.get('provisional_assigned', 0) + 1
        # ----------------------------------------------------------------------------------
        self.frame_log.append({'t': round(t, 4),
                               'assigned': {('L' if s == 0 else 'R'): i
                                            for i, (s, _) in out.items() if s is not None},
                               'palms': [pp.tolist() for pp in palms]})
        return out
