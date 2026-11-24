"""Per-hand evidence availability; projected overlap is not verified depth.

No word, person, target shape, or duration is an input. An unavailable opposite
hand is not evidence that this hand is invisible. Never reconstruct hidden joints.
"""
import cv2
import numpy as np


def hand_evidence(ch, hand):
    p = ch['body_relative_xy']
    av = ch['point_available'] & np.isfinite(p).all(-1)
    a, b = (91, 112) if hand == 'L' else (112, 91)
    own_missing = ~av[:, a:a + 21].all(1)
    other_missing = ~av[:, b:b + 21].all(1)
    overlap = np.zeros(len(p), dtype=bool)
    # Preserve the previous projected-overlap test whenever both are observed.
    # Missing opposite-hand fingers cannot be turned into an active-hand mask.
    ids = np.array([6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]) + a
    for i in np.flatnonzero(~own_missing & ~other_missing):
        hull = cv2.convexHull(p[i, b:b + 21].astype('float32'))
        overlap[i] = sum(cv2.pointPolygonTest(hull, tuple(map(float, q)), False) >= 0
                         for q in p[i, ids]) >= 6
    return dict(own_missing=own_missing, opposite_missing=other_missing,
                projected_overlap=overlap, uncertain=own_missing | overlap)


def hand_occlusion(ch, hand):
    return hand_evidence(ch, hand)['uncertain']
