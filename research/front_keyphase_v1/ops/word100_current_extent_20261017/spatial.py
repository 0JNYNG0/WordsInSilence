"""Current-view pose extent cues; diagnostic only, never fills hand landmarks.

Pose wrist/index/pinky/thumb are coarse estimates, NOT recovered Hand21 points.
Crossing and 2-D intersection are evidence, not visibility/occlusion truth.
All previous separated fields are retained verbatim.
"""
from copy import deepcopy
import numpy as np
import cv2

PARAMS = {'pose_confidence': .5, 'relative_area_numerical_epsilon': 1e-9}
MISSING = ('wrist_interior', 'wrist_boundary_or_outside')

def hull(points):
    return cv2.convexHull(np.asarray(points, np.float32)).reshape(-1, 2)

def extend_evidence(previous, xy, usable, pose_conf, width, height):
    xy = np.asarray(xy); usable = np.asarray(usable, bool); conf = np.asarray(pose_conf)
    if xy.shape != (75, 2) or usable.shape != (75,) or conf.shape != (33,):
        raise ValueError('Expected current-view75/33 arrays')
    if width <= 0 or height <= 0 or [r['side'] for r in previous] != [0, 1]:
        raise ValueError('Invalid dimensions or side ordering')
    result = deepcopy(previous)
    scale2 = float(np.sum((xy[11] - xy[12]) ** 2))
    for side, row in enumerate(result):
        ext = {'status': 'not_assessed', 'source': 'current_pose_wrist_pinky_index_thumb',
               'whole_hand_visibility_proven': False, 'points_are_estimated': True,
               'beyond_frame_is_extrapolation': True, 'crossed_edges': []}
        ov = {'status': 'not_assessed', 'occlusion_cause_confirmed': False,
              'no_intersection_does_not_exclude_overlap': True}
        row['current_extent'] = ext; row['current_overlap'] = ov
        # Don't override actual hand observations or unresolved assignment.
        if row['position']['state'] not in MISSING:
            continue
        ids = np.array([15, 17, 19, 21]) + side
        pts = xy[ids]
        if (not np.isfinite(pts).all() or not np.isfinite(conf[ids]).all()
                or not np.all(conf[ids] >= PARAMS['pose_confidence'])
                or not np.isfinite(scale2) or scale2 <= 0):
            ext['status'] = 'coarse_pose_unavailable'; ov['status'] = 'coarse_pose_unavailable'
            continue
        # Current pose can extrapolate outside the input image. Keep that
        # separate from usable/observed hand points; never promote masks.
        p = hull(pts)
        if len(p) < 3 or cv2.contourArea(p) / scale2 <= PARAMS['relative_area_numerical_epsilon']:
            ext['status'] = 'degenerate_pose_region'; ov['status'] = 'coarse_pose_unavailable'
            continue
        low, high = pts.min(0), pts.max(0)
        edges = [name for name, yes in zip(('left','top','right','bottom'),
                 (low[0] < 0, low[1] < 0, high[0] > width, high[1] > height)) if yes]
        ext.update(status='estimated_extent_crosses_frame' if edges else 'estimated_extent_inside_frame',
                   crossed_edges=edges, point_indices=ids.tolist(), hull=p.tolist(),
                   minimum_pose_confidence=float(conf[ids].min()))
        if edges:
            ov['status'] = 'boundary_competes_with_overlap'; continue
        other = 1 - side
        if previous[other]['shape']['state'] != 'usable_observed_points':
            ov['status'] = 'other_hand_not_observed'; continue
        handids = np.arange(33 + other*21, 54 + other*21)
        valid = usable[handids] & np.isfinite(xy[handids]).all(1)
        valid &= (xy[handids] >= 0).all(1) & (xy[handids,0] <= width) & (xy[handids,1] <= height)
        hp = xy[handids][valid]
        if len(hp) < 3:
            ov['status'] = 'other_hand_region_unavailable'; continue
        q = hull(hp)
        if len(q) < 3 or cv2.contourArea(q)/scale2 <= PARAMS['relative_area_numerical_epsilon']:
            ov['status'] = 'other_hand_region_unavailable'; continue
        area = float(cv2.intersectConvexConvex(p,q)[0])
        ov.update(status='current_2d_intersection_suspected' if area/scale2 > PARAMS['relative_area_numerical_epsilon'] else 'no_current_2d_intersection',
                  intersection_pixels2=area, other_observed_point_count=len(hp), other_hull=q.tolist())
    return result
