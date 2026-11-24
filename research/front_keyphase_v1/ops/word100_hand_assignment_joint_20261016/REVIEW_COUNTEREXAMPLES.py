"""Synthetic causal review only; never imports datasets or writes implementation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

UNIT = Path(__file__).resolve().parent


def hand(x, y):
    return np.array([[x + (j % 4) * 3, y + (j // 4) * 3] for j in range(21)], dtype=float)


def run():
    spec = importlib.util.spec_from_file_location('joint_review', UNIT / 'assignment.py')
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    pose = np.zeros((33, 2))
    usable = np.zeros(33, bool)
    pose[11], pose[12] = [400, 400], [500, 400]
    usable[[11, 12]] = True
    cases = []
    for label, other_time, other_frame, other_x, velocity_x in [
        ('stale_opposite_prediction_overshoot', .7, -9, 560, 1000),
        ('recent_opposite_prediction_overshoot', 1., 0, 560, 1200),
        ('stale_opposite_both_hypotheses_far', .7, -9, 900, 0),
    ]:
        tracker = m.HandAssigner()
        tracker.frame = 1
        tracker.last_time = 1.033333
        tracker.tracks = {
            1: m.Track(hand(500, 500)[m.PALM], 1., np.zeros(2), False, 0),
            0: m.Track(hand(other_x, 500)[m.PALM], other_time, np.array([velocity_x, 0.]), True, other_frame),
        }
        assigned, trace = tracker.assign(np.stack([hand(550, 500)]), [1], [.7],
                                          pose, usable, 1066667, 1000, 1000)
        cases.append({'name': label, 'expected': 'abstain: opposite hypothesis contradicts or is too old',
                      'passed': assigned[0] is None,
                      'assigned': assigned, 'trace': trace})
    return {'assignment_sha256': hashlib.sha256((UNIT / 'assignment.py').read_bytes()).hexdigest(),
            'cases': cases}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
