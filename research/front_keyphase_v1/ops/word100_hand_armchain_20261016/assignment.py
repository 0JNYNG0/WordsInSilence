"""Research-only arm-chain support for unresolved real hand detections.

The frozen joint assigner runs unmodified. Additional associations never feed
its hand tracks; only independently observed S/E/W + matching raw hand seed an
arm anchor. A projected wrist is an association hypothesis, never output data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

U = Path(__file__).resolve().parent
BASE = U.with_name('word100_hand_assignment_joint_20261016')
_spec = importlib.util.spec_from_file_location('wis_frozen_joint_for_armchain', BASE/'assignment.py')
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)
PARAMS = json.loads((U/'SCOPE.json').read_text('utf-8'))['parameters']
POINT_RULE = json.loads((U/'POINT_CONTRACT.json').read_text('utf-8'))
PALM = _base.PALM
if PARAMS != _base.PARAMS or POINT_RULE != _base.POINT_RULE:
    raise ValueError('Arm support must preserve the frozen base gates and point rules')


@dataclass
class ArmAnchor:
    shoulder: np.ndarray
    elbow: np.ndarray
    wrist: np.ndarray
    scale: float
    time: float
    frame: int


class HandAssigner(_base.HandAssigner):
    def reset(self):
        super().reset()
        self.arm_anchors = {}

    @staticmethod
    def identity():
        files = [U/'assignment.py', U/'SCOPE.json', U/'POINT_CONTRACT.json',
                 BASE/'assignment.py', BASE/'SCOPE.json', BASE/'POINT_CONTRACT.json']
        return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()

    def assign(self, hands, labels, scores, pose, pose_usable, timestamp_us, width, height):
        old_tracks = dict(self.tracks)
        result, trace = super().assign(hands, labels, scores, pose, pose_usable,
                                      timestamp_us, width, height)
        original = dict(result)
        original_reasons = list(trace['reasons'])
        if trace['time_reset']:
            old_tracks = {}
        hands = np.asarray(hands, dtype=np.float64)
        pose = np.asarray(pose, dtype=np.float64)
        usable = np.asarray(pose_usable, dtype=bool)
        t = int(timestamp_us)/1e6
        n = len(hands)
        ratio = PARAMS['ambiguity_ratio']
        gate = PARAMS['pose_gate']
        horizon = PARAMS['bootstrap_max_gap_s']
        max_frames = PARAMS['bootstrap_max_frame_gap']
        mx, my = width*POINT_RULE['bounds_margin_ratio'], height*POINT_RULE['bounds_margin_ratio']

        def good_pose(indices):
            pts = pose[indices].astype(np.float32)
            return bool(usable[indices].all() and np.isfinite(pts).all()
                and ((pts[:,0] >= -mx) & (pts[:,0] <= width+mx)
                     & (pts[:,1] >= -my) & (pts[:,1] <= height+my)).all())

        valid = []
        for i in range(n):
            palm = hands[i,PALM].astype(np.float32)
            valid.append(bool(np.isfinite(hands[i]).all() and np.isfinite(scores[i])
                and scores[i] >= POINT_RULE['hand_score_threshold']
                and ((palm[:,0] >= -mx) & (palm[:,0] <= width+mx)
                     & (palm[:,1] >= -my) & (palm[:,1] <= height+my)).all()))
        sw = float(np.linalg.norm(pose[11]-pose[12])) if good_pose([11,12]) else None
        if sw is not None and sw <= 1e-6:
            sw = None
        expired = [s for s,a in self.arm_anchors.items()
                   if t-a.time > horizon+1e-12 or self.frame-a.frame > max_frames]
        for s in expired:
            del self.arm_anchors[s]
        predictions, prediction_info = {}, []
        for side, anchor in self.arm_anchors.items():
            shoulder, elbow = 11+side, 13+side
            if sw is None or not good_pose([shoulder, elbow]):
                continue
            scale_ratio = sw/anchor.scale
            if not ratio <= scale_ratio <= 1/ratio:
                continue
            upper = (pose[elbow]-pose[shoulder])/sw
            previous_upper = (anchor.elbow-anchor.shoulder)/anchor.scale
            if np.linalg.norm(upper) <= 1e-6 or np.linalg.norm(upper-previous_upper) > gate:
                continue
            # Preserve the last actually observed forearm offset, translated
            # with the current elbow; never extend the upper arm as a forearm.
            wrist = pose[elbow] + (anchor.wrist-anchor.elbow)*scale_ratio
            predictions[side] = wrist
            prediction_info.append(dict(side=side, wrist_hint=wrist.tolist(),
                anchor_time=anchor.time, age_s=t-anchor.time,
                frame_gap=self.frame-anchor.frame, source='observed_pose_and_raw'))

        ac = np.full((n,2), np.inf)
        pc = np.array([[np.inf if v is None else v for v in row]
                       for row in trace['pose_distance']], dtype=float).reshape(n,2)
        tc = np.array([[np.inf if v is None else v for v in row]
                       for row in trace['track_distance']], dtype=float).reshape(n,2)
        if sw is not None:
            for i in range(n):
                if valid[i]:
                    for side,wrist in predictions.items():
                        ac[i,side] = np.linalg.norm(hands[i,0]-wrist)/sw

        def beats(value, opposite):
            return not np.isfinite(opposite) or value < ratio*opposite

        occupied = {s for s in original.values() if s is not None}
        recoveries, rejected = [], []
        for i in range(n):
            # Only unassigned detections that were neither duplicate-suppressed
            # nor withheld for a current pose/track identity conflict qualify.
            if original[i] is not None or not valid[i] or trace['close_palms']:
                continue
            if original_reasons[i] != 'ambiguous_or_no_evidence':
                continue
            side = int(labels[i]) if labels[i] in [0,1] else None
            if side is None or side in occupied or side not in predictions:
                continue
            why = []
            # Own tracked raw hand is mandatory; missing opposite evidence is
            # allowed, but is never treated as positive evidence of a side.
            if ac[i,side] > gate or tc[i,side] > PARAMS['track_gate']:
                why.append('own_arm_or_hand_track_outside_gate')
            if not beats(ac[i,side], ac[i,1-side]):
                why.append('opposite_arm_not_excluded')
            if any(valid[j] and not beats(ac[i,side],ac[j,side]) for j in range(n) if j != i):
                why.append('raw_competition_ambiguous')
            if not beats(tc[i,side],tc[i,1-side]):
                why.append('opposite_hand_track_not_excluded')
            if np.isfinite(pc[i,1-side]) and not beats(ac[i,side],pc[i,1-side]):
                why.append('opposite_current_pose_not_excluded')
            if np.isfinite(pc[i,side]) and pc[i,side] > gate:
                why.append('own_current_pose_disagrees')
            # Do not bypass the baseline's two-frame conflict confirmation.
            if self.pending_conflicts:
                why.append('pending_identity_conflict')
            other = old_tracks.get(1-side)
            if other is not None and sw is not None and 0 <= t-other.time <= horizon+1e-12:
                own = old_tracks.get(side)
                if own is None:
                    why.append('no_own_observed_track')
                else:
                    own_d = np.linalg.norm(hands[i,PALM]-own.palm,axis=1).mean()/sw
                    other_d = np.linalg.norm(hands[i,PALM]-other.palm,axis=1).mean()/sw
                    if not beats(own_d,other_d):
                        why.append('opposite_observed_hand_not_excluded')
            if why:
                rejected.append(dict(raw_index=i,side=side,reasons=why))
                continue
            recoveries.append(dict(raw_index=i,side=side,arm_distance=float(ac[i,side]),
                hand_track_distance=float(tc[i,side]),anchor_time=self.arm_anchors[side].time,
                age_s=t-self.arm_anchors[side].time,frame_gap=self.frame-self.arm_anchors[side].frame))
        for entry in recoveries:
            i,side = entry['raw_index'],entry['side']
            if sum(r['side'] == side for r in recoveries) == 1:
                result[i] = side
                trace['reasons'][i] = 'arm_chain_supported_recovery'

        seeded = []
        if sw is not None:
            for i,side in original.items():
                if side is None or not valid[i] or not good_pose([11+side,13+side,15+side]):
                    continue
                # Independent current pose match, including exclusion of the
                # other pose wrist and raw hypotheses. Recovery never seeds.
                if pc[i,side] > gate or not beats(pc[i,side],pc[i,1-side]):
                    continue
                if any(valid[j] and not beats(pc[i,side],pc[j,side]) for j in range(n) if j != i):
                    continue
                s,e,w = pose[11+side],pose[13+side],pose[15+side]
                if min(np.linalg.norm(e-s),np.linalg.norm(w-e)) <= 1e-6:
                    continue
                self.arm_anchors[side] = ArmAnchor(s.copy(),e.copy(),w.copy(),sw,t,self.frame)
                seeded.append(side)
        trace['baseline_assigned_sides'] = [original[i] for i in range(n)]
        trace['baseline_reasons'] = original_reasons
        trace['arm_chain'] = dict(predictions=prediction_info,
            distances=[[None if not np.isfinite(v) else float(v) for v in row] for row in ac],
            recoveries=[r for r in recoveries if result[r['raw_index']] == r['side']],
            rejected=rejected,seeded_sides=seeded,expired_sides=expired,
            baseline_tracks_modified=False,coordinates_generated=False)
        trace['assigned_sides'] = [result[i] for i in range(n)]
        trace['missing_sides'] = [s for s in [0,1] if s not in result.values()]
        return result, trace
