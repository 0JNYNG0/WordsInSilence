"""Research-only, causal anatomical hand association. Never generates observations.

Distances use shoulder widths. Present raw detections are the only output source;
track prediction is an association hint, not interpolation or a reconstructed hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import hashlib
import json
import numpy as np

PALM = [0, 5, 9, 13, 17]
PARAMS = json.loads((Path(__file__).with_name('SCOPE.json')).read_text('utf-8'))['parameters']
POINT_RULE = json.loads((Path(__file__).with_name('POINT_CONTRACT.json')).read_text('utf-8'))


@dataclass
class Track:
    palm: np.ndarray
    time: float
    velocity: np.ndarray


class HandAssigner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tracks = {}
        self.last_time = None
        self.scale = None
        self.scale_time = None
        self.pending_conflicts = {}

    @staticmethod
    def identity():
        return hashlib.sha256(Path(__file__).read_bytes() +
            json.dumps([PARAMS, POINT_RULE], sort_keys=True).encode()).hexdigest()

    def assign(self, hands, labels, scores, pose, pose_usable, timestamp_us, width, height):
        """Return detection->side (0=L/1=R/None), plus auditable evidence.

        Input hands [N,21,2] uses non-mirrored source pixels. labels are raw
        MediaPipe Left=0/Right=1 per the existing measured input convention.
        Call once for EVERY frame including N=0. New clip needs a new object/reset.
        """
        hands = np.asarray(hands, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)
        pose_usable = np.asarray(pose_usable, dtype=bool)
        n = len(hands)
        if n > 2 or hands.shape != (n,21,2) or len(labels) != n or len(scores) != n:
            raise ValueError('Expected at most two 21-point detections with label/score')
        t = int(timestamp_us) / 1e6
        time_reset = self.last_time is not None and t <= self.last_time
        if time_reset:
            self.reset()
        self.last_time = t
        expired = [s for s,tr in self.tracks.items() if t-tr.time > PARAMS['max_gap_s']]
        for s in expired:
            del self.tracks[s]
        if pose_usable[[11,12]].all() and np.isfinite(pose[[11,12]]).all():
            scale = float(np.linalg.norm(pose[11]-pose[12]))
            if scale > 1e-6:
                self.scale, self.scale_time = scale, t
        sw = self.scale if self.scale_time is not None and t-self.scale_time <= PARAMS['max_gap_s'] else None
        palms = hands[:,PALM]
        valid = []
        mx = width * POINT_RULE['bounds_margin_ratio']
        my = height * POINT_RULE['bounds_margin_ratio']
        for i in range(n):
            # Same float32/bounds convention as the observation's usable mask.
            p = palms[i].astype(np.float32)
            valid.append(bool(np.isfinite(hands[i]).all() and np.isfinite(scores[i])
                and scores[i] >= POINT_RULE['hand_score_threshold']
                and ((p[:,0]>=-mx)&(p[:,0]<=width+mx)&(p[:,1]>=-my)&(p[:,1]<=height+my)).all()))
        close_palms = False
        if n == 2 and all(valid):
            sizes = [float(np.median(np.linalg.norm(p[1:]-p[0],axis=1))) for p in palms]
            if min(sizes)>1e-6 and np.linalg.norm(palms[0]-palms[1],axis=1).mean() <= PARAMS['duplicate_palm_fraction']*min(sizes):
                close_palms = True

        pc = np.full((n,2),np.inf)
        tc = np.full((n,2),np.inf)
        if sw is not None:
            for i in range(n):
                if not valid[i]:
                    continue
                for s,idx in enumerate([15,16]):
                    if pose_usable[idx] and np.isfinite(pose[idx]).all():
                        pc[i,s] = float(np.linalg.norm(palms[i,0]-pose[idx])/sw)
                    if s in self.tracks:
                        tr = self.tracks[s]
                        prediction = tr.palm + tr.velocity*(t-tr.time)
                        tc[i,s] = float(np.linalg.norm(palms[i]-prediction,axis=1).mean()/sw)

        def clear_side(cost,i,s,gate):
            return bool(cost[i,s] <= gate and (not np.isfinite(cost[i,1-s])
                or cost[i,s] < PARAMS['ambiguity_ratio']*cost[i,1-s]))

        def clear(cost,i,s,gate,excluded=()):
            if cost[i,s]>gate:
                return False
            rivals = [cost[i,1-s]] + [cost[j,s] for j in range(n) if j!=i and j not in excluded]
            return all(not np.isfinite(r) or cost[i,s] < PARAMS['ambiguity_ratio']*r for r in rivals)

        # A second detector hypothesis for the same wrist is not evidence for a
        # second anatomical hand. Distinct temporal identities prevent merging
        # real crossing hands merely because current pose wrists coincide.
        pose_sides = [[s for s in [0,1] if clear_side(pc,i,s,PARAMS['pose_gate'])] for i in range(n)]
        track_sides = [[s for s in [0,1] if clear(tc,i,s,PARAMS['track_gate'])] for i in range(n)]
        distinct_tracks = n == 2 and all(track_sides) and track_sides[0] != track_sides[1]
        distinct_pose = n == 2 and all(pose_sides) and pose_sides[0] != pose_sides[1]
        same_pose = n == 2 and all(pose_sides) and pose_sides[0] == pose_sides[1]
        suppressed = set()
        competition = None
        duplicates = []
        if same_pose and not distinct_tracks:
            side = pose_sides[0][0]
            # Current pose proximity preserves the original supported choice.
            # Handedness score is ONLY an exact-distance tie-break, not a metric
            # of finger-landmark quality. Coordinate key avoids index bias.
            winner = min(range(n), key=lambda i:(pc[i,side],-float(scores[i]),tuple(hands[i].ravel())))
            suppressed = {i for i in range(n) if i != winner}
            competition = {'side':side,'winner_raw_index':winner,'rule':'nearest_current_pose_wrist'}
        elif close_palms and not (distinct_tracks or distinct_pose):
            # Coincident geometry without independent side evidence is unresolved.
            duplicates = [0,1]
            suppressed = {0,1}

        previous_pending = self.pending_conflicts
        self.pending_conflicts = {}
        corrections = {}
        evidence = {}
        for i in range(n):
            if not valid[i] or i in suppressed:
                continue
            track_s = [s for s in [0,1] if clear(tc,i,s,PARAMS['track_gate'],suppressed)]
            pose_s = pose_sides[i]
            lab = int(labels[i]) if scores[i]>=PARAMS['label_min_score'] and labels[i] in [0,1] else None
            if track_s and pose_s and track_s[0]!=pose_s[0] and lab==pose_s[0]:
                old_side, target = track_s[0], pose_s[0]
                key = (old_side,target)
                prev = previous_pending.get(key)
                continuous = prev is not None and sw is not None and np.linalg.norm(palms[i]-prev['palm'],axis=1).mean()/sw <= PARAMS['track_gate']
                count = prev['count']+1 if continuous else 1
                self.pending_conflicts[key] = {'count':count,'palm':palms[i].copy()}
                if count >= PARAMS['conflict_confirmation_frames']:
                    evidence[i,target] = (6.0,'confirmed_current_reinitialization')
                    corrections[i] = old_side
                # A contradictory frame neither updates the wrong track nor flips immediately.
                continue
            # A clear short temporal link has priority over a one-frame pose-side flip.
            # Otherwise current pose can reinitialize a live stale track (no lockout).
            for s in [0,1]:
                track_ok = s in track_s
                pose_ok = s in pose_s
                if track_ok:
                    value = 4 + (1-tc[i,s]/PARAMS['track_gate'])
                    reason = 'temporal_and_pose' if pose_ok else 'temporal'
                elif track_s:
                    continue
                elif pose_ok:
                    value = 2 + (1-pc[i,s]/PARAMS['pose_gate'])
                    reason = 'pose_and_label' if lab==s else 'bounded_pose_over_label' if lab is not None else 'bounded_pose'
                elif not pose_s and lab==s:
                    value = 1
                    reason = 'label_without_close_pose'
                else:
                    continue
                evidence[i,s] = (float(value),reason)

        # Enumerate whole-frame assignments, including absence. Never force 2 hands.
        possibilities = []
        for sides in product([None,0,1], repeat=n):
            used = [s for s in sides if s is not None]
            if len(used)!=len(set(used)) or any((i,s) not in evidence for i,s in enumerate(sides) if s is not None):
                continue
            total = sum(evidence[i,s][0] for i,s in enumerate(sides) if s is not None)
            possibilities.append((total,sides))
        best_score = max(v for v,_ in possibilities)
        winners = [sides for value,sides in possibilities if abs(value-best_score)<=1e-9]
        result = {}
        reasons = []
        for i in range(n):
            choices = {v[i] for v in winners}
            side = next(iter(choices)) if len(choices)==1 else None
            result[i] = side
            reason = evidence[i,side][1] if side is not None else ('unresolved_coincident_palms' if i in duplicates else 'same_side_alternative' if i in suppressed else 'ambiguous_or_no_evidence')
            reasons.append(reason)
            if side is None:
                continue
        old_tracks = dict(self.tracks)
        # Apply invalidations before commits, including simultaneous correction of both sides.
        for i,old_side in corrections.items():
            if result[i] is not None:
                self.tracks.pop(old_side,None)
        for i,side in result.items():
            if side is None:
                continue
            reason = reasons[i]
            # Only observed current palm updates the identity state, never prediction.
            velocity = np.zeros(2,dtype=np.float64)
            old = old_tracks.get(side)
            if reason.startswith('temporal') and old is not None and t>old.time and sw is not None:
                velocity = (palms[i].mean(0)-old.palm.mean(0))/(t-old.time)
                speed = float(np.linalg.norm(velocity))
                cap = PARAMS['max_speed_shoulders_s']*sw
                if speed>cap:
                    velocity *= cap/speed
            self.tracks[side] = Track(palms[i].copy(),t,velocity)
        # Missing detections output no coordinates; unmatched tracks expire in real time.
        def serial(a):
            return [[float(v) if np.isfinite(v) else None for v in row] for row in a]
        return result, {'time_reset':time_reset,'expired_sides':expired,'scale':sw,
            'pose_distance':serial(pc),'track_distance':serial(tc),'reasons':reasons,
            'assigned_sides':[result[i] for i in range(n)],'detected_hands':n,
            'same_side_competition':competition,'close_palms':close_palms,
            'missing_sides':[s for s in [0,1] if s not in result.values()],
            'coordinates_generated':False}
