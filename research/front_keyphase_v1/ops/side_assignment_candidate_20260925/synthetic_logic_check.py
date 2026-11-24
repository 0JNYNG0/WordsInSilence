"""SYNTHETIC coordinate check of the assignment logic. NOT video verification.

This exercises _assign_sides with hand-made coordinates. It says what the code does on constructed
geometry. It is NOT evidence about real recordings and NOT evidence of generalisation - the real
crossed-hands verification gap stays open.

Run: py -3.13 ops/side_assignment_candidate_20260925/synthetic_logic_check.py
"""
import json, pathlib, sys
import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
sys.path[:0] = [str(L), str(L / 'src')]
sys.path.insert(0, str(H))
from wis_lab.capture.landmarker import Landmarker           # noqa: E402
from run_candidate import MutualNearest                     # noqa: E402
from wis_lab.core.observation import POSE_L_WRIST, POSE_R_WRIST   # noqa: E402

W, Hh = 1000, 800
MEASURED = {}


class FakeHands:
    """Just enough of the MediaPipe result for _assign_sides: wrist x/y and a handedness label."""
    def __init__(self, wrists, labels):
        self.hand_landmarks = [[type('P', (), {'x': x / W, 'y': y / Hh})()] for x, y in wrists]
        self.handedness = [[type('C', (), {'score': 0.9, 'category_name': lab})()] for lab in labels]


def assign(cls, hand_wrists, pose_wrists, labels=None):
    """pose_wrists: dict side->xy or None to mark unusable."""
    lm = object.__new__(cls)
    lm.contract = {}
    lm.handedness_disagreements = lm.handedness_comparisons = lm.unassigned_hands = 0
    xy = np.zeros((33, 2), np.float32)
    usable = np.zeros(33, bool)
    for side, idx in ((0, POSE_L_WRIST), (1, POSE_R_WRIST)):
        if pose_wrists.get(side) is not None:
            xy[idx] = pose_wrists[side]
            usable[idx] = True
    res = FakeHands(hand_wrists, labels or ['Left', 'Right'][:len(hand_wrists)])
    out = cls._assign_sides(lm, res, xy, usable, W, Hh)
    return {i: (None if s is None else ('L' if s == 0 else 'R')) for i, (s, _) in out.items()}


def check_crossed_hands_keep_both():
    """Arms crossed: the subject-left hand sits LEFT of the subject-right hand in the image, and
    the pose wrists are crossed the same way. Each hand is still nearest its own wrist."""
    hands = [(300, 400), (700, 400)]                 # hand0 left in image, hand1 right in image
    pose = {0: (310, 410), 1: (690, 410)}            # pose L left, pose R right - crossed
    base = assign(Landmarker, hands, pose)
    cand = assign(MutualNearest, hands, pose)
    assert base == {0: 'L', 1: 'R'}, base
    assert cand == base, (base, cand)
    MEASURED['crossed_hands'] = {'baseline': base, 'candidate': cand}


def check_duplicate_on_one_hand_is_unassigned():
    """Two detections on the SAME hand, far from the idle side's wrist."""
    hands = [(700, 400), (712, 406)]
    pose = {0: (300, 700), 1: (695, 402)}            # left wrist idle and far away
    base = assign(Landmarker, hands, pose)
    cand = assign(MutualNearest, hands, pose)
    assert set(base.values()) == {'L', 'R'}, base
    assert sorted(v for v in cand.values() if v) == ['R'], cand
    assert None in cand.values(), cand
    MEASURED['duplicate'] = {'baseline': base, 'candidate': cand}


def check_two_normal_hands_untouched():
    hands = [(700, 400), (300, 400)]
    pose = {0: (705, 405), 1: (295, 405)}
    assert assign(Landmarker, hands, pose) == assign(MutualNearest, hands, pose)
    MEASURED['two_normal_hands'] = assign(MutualNearest, hands, pose)


def check_one_pose_wrist_is_out_of_scope():
    """Only one usable wrist: that wrist IS the nearest, so the candidate must not fire."""
    hands = [(300, 400)]
    pose = {0: (700, 400), 1: None}
    base = assign(Landmarker, hands, pose)
    cand = assign(MutualNearest, hands, pose)
    assert base == {0: 'L'} and cand == base, (base, cand)
    MEASURED['one_pose_wrist_out_of_scope'] = cand


def check_far_but_own_nearest_is_kept():
    """A hand far from its own wrist is NOT removed when that wrist is still its nearest."""
    hands = [(900, 200)]
    pose = {0: (700, 400), 1: (100, 400)}
    cand = assign(MutualNearest, hands, pose)
    assert cand == {0: 'L'}, cand
    MEASURED['far_but_nearest_kept'] = cand


CHECKS = [('crossed hands keep both', check_crossed_hands_keep_both),
          ('duplicate on one hand is unassigned', check_duplicate_on_one_hand_is_unassigned),
          ('two normal hands untouched', check_two_normal_hands_untouched),
          ('one pose wrist is out of scope', check_one_pose_wrist_is_out_of_scope),
          ('far but own-nearest is kept', check_far_but_own_nearest_is_kept)]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn(); print(f'PASS  {name}')
        except AssertionError as exc:
            failed += 1; print(f'FAIL  {name}: {exc}')
    print(f'\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed')
    out = {'schema': 'wis_side_assignment_synthetic_logic_v1', 'at_kst': '2026-09-25',
           'STATUS': 'SYNTHETIC COORDINATES ONLY. This is NOT real-video verification and NOT '
                     'evidence of generalisation. The crossed-hands verification gap on real '
                     'recordings remains OPEN.',
           'checks_passed': len(CHECKS) - failed, 'checks_total': len(CHECKS),
           'measured': MEASURED}
    (H / 'SYNTHETIC_LOGIC.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
