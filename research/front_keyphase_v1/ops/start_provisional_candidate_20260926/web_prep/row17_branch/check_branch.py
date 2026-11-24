"""Branch check for 감사합니다 take 1 (tri_a0da37edf083) row 17 - no camera, no MediaPipe inference.
The frozen StartLandmarker._assign_sides is called directly with the STORED pose row and the STORED
left-hand detection. The SECOND detection in cases B/C is SYNTHETIC; it does not describe the real,
unstored detection - it only exercises which branch fires. The same-label refusal does not depend on
its position (it is tested with two different placements)."""
import json, pathlib, sys, types
import numpy as np
H = pathlib.Path(__file__).resolve().parent
L = H.parents[3]
sys.path[:0] = [str(L), str(L / 'src')]
from wis_lab.core.observation import Observation
from wis_lab.service.shadow_assignment import StartLandmarker, candidate_sha256

R = L / 'runs/ses_2bb3148c29e8/trials/tri_a0da37edf083'
p = Observation.load(str(R) + '.npz')
side = json.loads((R.parent / 'tri_a0da37edf083.sidecar.json').read_text(encoding='utf-8'))
W, Hh = side['width'], side['height']
row = 17
pose_xy, pose_us = p.xy_px[row].astype(np.float64), p.usable[row].copy()
A = p.xy_px[row][33:54]                                 # the stored left-hand detection


def det(points, label, score=0.99):
    lms = [types.SimpleNamespace(x=float(x) / W, y=float(y) / Hh) for x, y in points]
    return lms, [types.SimpleNamespace(score=score, category_name=label)]


def run(dets):
    lm = StartLandmarker.__new__(StartLandmarker)        # no models loaded: the method needs none
    lm.tracks, lm._t_s, lm.frame_log = {}, 0.567, []   # take start: tracks empty (reset)
    lm.stats = {k: 0 for k in ('matched', 'initialised', 'unassigned_ambiguous', 'unassigned_no_evidence',
                               'tracks_expired', 'handedness_agreed', 'handedness_compared', 'recovered_takeover')}
    lm.reasons = {k: 0 for k in ('refused_track_gate', 'refused_ratio', 'refused_init_gate', 'refused_side_taken_this_frame',
                                 'refused_live_track_blocks_init', 'refused_no_pose_reference',
                                 'refused_init_gate_after_unblock', 'pair_removed_arm_contradiction')}
    lm.removed_log = []
    res = types.SimpleNamespace(hand_landmarks=[d[0] for d in dets], handedness=[d[1] for d in dets])
    out = StartLandmarker._assign_sides(lm, res, pose_xy.copy(), pose_us.copy(), W, Hh)
    return {'sides': {i: ('LR'[s] if s is not None else None) for i, (s, _) in out.items()},
            'reasons': {k: v for k, v in lm.reasons.items() if v}}


shift_right = A + np.array([-300.0, 0.0])               # synthetic placements, NOT the real detection
shift_left = A + np.array([+60.0, -40.0])
cases = {
    'A_only_stored_left_hand_labelled_Left': run([det(A, 'Left')]),
    'B1_plus_synthetic_Left_label_placement1': run([det(A, 'Left'), det(shift_right, 'Left')]),
    'B2_plus_synthetic_Left_label_placement2': run([det(A, 'Left'), det(shift_left, 'Left')]),
    'C_plus_synthetic_Right_label_on_right_half': run([det(A, 'Left'), det(shift_right, 'Right')]),
}
out = {'candidate_sha256': candidate_sha256(), 'stored_row': row,
       'stored_facts': {'shoulders_usable': [bool(pose_us[11]), bool(pose_us[12])],
                        'pose_wrists_usable': [bool(pose_us[15]), bool(pose_us[16])],
                        'half_dot': float((pose_xy[33] - (pose_xy[11] + pose_xy[12]) / 2) @ (pose_xy[11] - (pose_xy[11] + pose_xy[12]) / 2))},
       'cases': cases}
(H / 'BRANCH_CHECK.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=1))
