"""Research-only current-image detections. Never rewrites a baseline hand."""
import numpy as np

PALM = [0, 5, 9, 13, 17]
PARAMS = dict(pose_gate=.25, pose_margin=.10, duplicate_gate=.16,
              crop_half_shoulder=.60, detector_confidence=.5, num_hands=2)


def inside(points, w, h):
    p = np.asarray(points)
    return np.isfinite(p).all(-1) & (p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h)


def arm_context(pose, usable, w, h):
    pose, usable = np.asarray(pose), np.asarray(usable)
    good = usable & inside(pose, w, h)
    scale = float(np.linalg.norm(pose[11]-pose[12])) if good[11] and good[12] else 0.
    return good, scale


def crops(pose, usable, base_sides, w, h):
    good, scale = arm_context(pose, usable, w, h)
    out = {}
    if scale <= 1:
        return out
    for side, (elbow, wrist) in enumerate([(13, 15), (14, 16)]):
        if side in base_sides or not good[[elbow, wrist]].all():
            continue
        x, y = pose[wrist]
        d = scale*PARAMS['crop_half_shoulder']
        box = [max(0, int(np.floor(x-d))), max(0, int(np.floor(y-d))),
               min(w, int(np.ceil(x+d))), min(h, int(np.ceil(y+d)))]
        if min(box[2]-box[0], box[3]-box[1]) >= 32:
            out[side] = box
    return out


def decide(detections, base_hands, base_sides, pose, usable, w, h):
    """Side from current arm only; label/word/history/hidden observations not inputs."""
    good, scale = arm_context(pose, usable, w, h)
    checks = []
    for i, det in enumerate(detections):
        points = np.asarray(det['xy'])
        item = dict(index=i, accepted=False, side=-1, reason='invalid_geometry')
        if points.shape != (21, 2) or not inside(points[PALM], w, h).all() or scale <= 1:
            checks.append(item)
            continue
        center = points[PALM].mean(0)
        dist = []
        for elbow, wrist in [(13, 15), (14, 16)]:
            dist.append(float(np.linalg.norm(points[0]-pose[wrist])/scale)
                        if good[[elbow, wrist]].all() else float('inf'))
        side = int(np.argmin(dist))
        item.update(side=side, center=center.tolist(),
                    arm_distances=[None if not np.isfinite(x) else x for x in dist])
        if dist[side] > PARAMS['pose_gate'] or dist[1-side]-dist[side] < PARAMS['pose_margin']:
            item['reason'] = 'arm_uncertain'
        elif side in base_sides:
            item['reason'] = 'preserve_existing_side'
        elif any(np.isfinite(b[PALM]).all() and
                 np.linalg.norm(np.asarray(b)[PALM].mean(0)-center) < PARAMS['duplicate_gate']*scale
                 for b in base_hands):
            item['reason'] = 'duplicate_existing_detection'
        elif 'box' in det and not inside(points[PALM]-np.array(det['box'][:2]),
                                        det['box'][2]-det['box'][0], det['box'][3]-det['box'][1]).all():
            item['reason'] = 'crop_boundary_uncertain'
        else:
            item.update(accepted=True, reason='current_image_and_arm')
        checks.append(item)
    for side in (0, 1):
        same = [x for x in checks if x['accepted'] and x['side'] == side]
        if len(same) > 1:
            for x in same:
                x.update(accepted=False, reason='multiple_candidates_same_side')
    accepted = [x for x in checks if x['accepted']]
    if len(accepted) == 2 and np.linalg.norm(np.array(accepted[0]['center'])-accepted[1]['center']) < PARAMS['duplicate_gate']*scale:
        for x in accepted:
            x.update(accepted=False, reason='new_candidates_overlap_uncertain')
    return checks
