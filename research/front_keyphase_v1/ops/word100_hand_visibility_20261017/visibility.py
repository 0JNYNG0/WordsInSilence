"""Causal diagnostic sidecar. Never edits landmarks, masks, assignment or scores.

No word, signer, classifier, full-source image or future frame is an input.
Labels express evidence, not certified physical causes or action correctness.
"""
from dataclasses import dataclass
import numpy as np

PARAMS = {
    'pose_confidence': 0.5,
    'history_seconds': 0.5,
    'boundary_shoulders': 0.15,
    'overlap_shoulders': 0.35,
    'history_displacement_shoulders': 0.4,
    'arm_segment_min_shoulders': 0.1,
    'arm_segment_max_shoulders': 1.5,
}
PALM = np.array([0, 5, 9, 13, 17])


@dataclass
class HandHistory:
    palm: np.ndarray
    time: float


class HandVisibility:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = {}
        self.last_time = None
        self.dimensions = None

    def step(self, xy, usable, pose_conf, timestamp_us, width, height,
             assigned=(False, False), unassigned_raw=False):
        xy = np.asarray(xy)
        usable = np.asarray(usable, dtype=bool)
        pose_conf = np.asarray(pose_conf)
        if xy.shape != (75, 2) or usable.shape != (75,) or pose_conf.shape != (33,):
            raise ValueError('Expected current observed 75-point arrays and 33 pose confidence values')
        if width <= 0 or height <= 0:
            raise ValueError('Positive actual input dimensions required')
        t = int(timestamp_us) / 1e6
        if ((self.last_time is not None and t <= self.last_time)
                or self.dimensions not in (None, (width, height))):
            self.reset()
        self.last_time, self.dimensions = t, (width, height)
        self.history = {s: h for s, h in self.history.items() if t - h.time <= PARAMS['history_seconds']}

        def inside(p, margin=0.):
            return bool(np.isfinite(p).all() and margin <= p[0] <= width - margin
                        and margin <= p[1] <= height - margin)

        def trusted_pose(i, require_inside=True):
            return bool(np.isfinite(xy[i]).all() and np.isfinite(pose_conf[i])
                        and pose_conf[i] >= PARAMS['pose_confidence']
                        and (not require_inside or (usable[i] and inside(xy[i]))))

        observed, palms = [], []
        for side in (0, 1):
            indices = 33 + side * 21 + PALM
            ok = bool(usable[indices].all() and np.isfinite(xy[indices]).all())
            observed.append(ok)
            palms.append(xy[indices].mean(0) if ok else None)
        shoulder_ok = trusted_pose(11) and trusted_pose(12)
        scale = float(np.linalg.norm(xy[11] - xy[12])) if shoulder_ok else 0.
        shoulder_ok = shoulder_ok and scale > 1e-6
        output = []
        for side, (shoulder, elbow, wrist) in enumerate(((11, 13, 15), (12, 14, 16))):
            row = {'side': side, 'state': 'unknown', 'reason': 'insufficient_evidence',
                   'evidence': {}, 'coordinate_repair': False}
            if observed[side]:
                row.update(state='observed', reason='existing_palm5_usable')
            elif assigned[side]:
                row.update(state='partial_observation', reason='assigned_hand_palm5_not_usable')
            elif unassigned_raw:
                row.update(state='assignment_uncertain', reason='withheld_raw_not_promoted')
            elif shoulder_ok and trusted_pose(shoulder) and trusted_pose(elbow) and trusted_pose(wrist, False):
                p, e = xy[wrist], xy[elbow]
                upper = float(np.linalg.norm(xy[shoulder] - e) / scale)
                lower = float(np.linalg.norm(e - p) / scale)
                row['evidence'].update(upper_arm_shoulders=upper, forearm_shoulders=lower)
                lengths_ok = all(PARAMS['arm_segment_min_shoulders'] <= z <= PARAMS['arm_segment_max_shoulders'] for z in (upper, lower))
                if lengths_ok:
                    distances = np.array([p[0], width - p[0], p[1], height - p[1]]) / scale
                    edge = int(np.argmin(distances))
                    normals = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
                    outward = float(np.dot(p - e, normals[edge]) / scale)
                    row['evidence'].update(nearest_edge=['left', 'right', 'top', 'bottom'][edge],
                                           edge_distance_shoulders=float(distances[edge]), outward_forearm_shoulders=outward)
                    if distances[edge] <= PARAMS['boundary_shoulders'] and outward > PARAMS['boundary_shoulders']:
                        row.update(state='boundary_outside_likely', reason='pose_wrist_near_or_beyond_edge_with_outward_forearm')
                    elif inside(p, PARAMS['boundary_shoulders'] * scale):
                        row.update(state='inside_missing', reason='interior_pose_wrist_but_missing_hand')
                        history = self.history.get(side)
                        if observed[1 - side] and history is not None:
                            close = float(np.linalg.norm(p - palms[1 - side]) / scale)
                            linked = float(np.linalg.norm(p - history.palm) / scale)
                            row['evidence'].update(other_palm_distance_shoulders=close,
                                                   history_distance_shoulders=linked,
                                                   history_age_seconds=t - history.time)
                            if close <= PARAMS['overlap_shoulders'] and linked <= PARAMS['history_displacement_shoulders']:
                                row.update(state='overlap_likely', reason='interior_wrist_close_to_other_palm_and_own_recent_track')
                else:
                    row['reason'] = 'arm_geometry_uncertain'
            output.append(row)
        # Only actual usable hand observations refresh identity evidence.
        for side in (0, 1):
            if observed[side]:
                self.history[side] = HandHistory(palms[side].copy(), t)
        return output
