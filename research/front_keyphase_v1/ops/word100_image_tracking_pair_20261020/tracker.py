"""Offline causal image-only position support; never creates hand landmarks."""
import cv2
import numpy as np

PARAMS = dict(image_width=640, max_age_s=.20, max_points=32, min_points=6,
              quality=.01, min_distance=3, win=21, levels=3, fb_px=1.5,
              lk_error=20., keep_fraction=.60, residual_shoulder=.10,
              pose_gate=.25, pose_margin=.10, match_gate=.12,
              ambiguity_ratio=.5, other_hand_gate=.16)
PALM = [0, 5, 9, 13, 17]


def inside(p, w, h):
    p = np.asarray(p)
    return np.isfinite(p).all(-1) & (p[..., 0] >= 0) & (p[..., 0] < w) & (p[..., 1] >= 0) & (p[..., 1] < h)


def optical_step(prev, gray, points, center, shoulder):
    """No future frames. Coordinates are in fixed 640-wide sensor pixels."""
    p = np.asarray(points, np.float32).reshape(-1, 1, 2)
    args = dict(winSize=(PARAMS['win'],)*2, maxLevel=PARAMS['levels'],
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    q, st, err = cv2.calcOpticalFlowPyrLK(prev, gray, p, None, **args)
    if q is None:
        return None, dict(reason='lk_missing')
    back, bs, _ = cv2.calcOpticalFlowPyrLK(gray, prev, q, None, **args)
    if back is None:
        return None, dict(reason='backward_missing')
    p, q, back = p[:, 0], q[:, 0], back[:, 0]
    fb = np.linalg.norm(back-p, axis=1)
    good = ((st[:, 0] == 1) & (bs[:, 0] == 1) & np.isfinite(err[:, 0])
            & (err[:, 0] <= PARAMS['lk_error']) & (fb <= PARAMS['fb_px'])
            & inside(q, gray.shape[1], gray.shape[0]))
    info = dict(points_before=len(p), points_after=int(good.sum()),
                previous_points=p.tolist(), forward=q.tolist(), backward=back.tolist(),
                forward_status=st[:, 0].tolist(), backward_status=bs[:, 0].tolist(),
                errors=err[:, 0].tolist(), previous_center=np.asarray(center).tolist(),
                shoulder=float(shoulder))
    if good.sum() < PARAMS['min_points'] or good.mean() < PARAMS['keep_fraction']:
        return None, dict(info, reason='flow_quality')
    delta = q[good]-p[good]
    shift = np.median(delta, axis=0)
    spread = float(np.quantile(np.linalg.norm(delta-shift, axis=1), .9))
    new = np.asarray(center)+shift
    info.update(spread=spread, shift=shift.tolist(), proposed_center=new.tolist())
    if spread > PARAMS['residual_shoulder']*shoulder or not inside(new, gray.shape[1], gray.shape[0]):
        return None, dict(info, reason='incoherent_or_outside')
    return dict(points=q[good], center=new), dict(info, reason='flow_pass')


class Tracker:
    def __init__(self):
        self.prev = None
        self.tracks = {}
        self.last_time = None

    def step(self, gray, hands, baseline, pose, usable, time_s, suppressed=()):
        if self.last_time is not None and time_s <= self.last_time:
            raise ValueError('strict causal timestamps required')
        self.last_time = time_s
        w, h = gray.shape[1], gray.shape[0]
        hands, pose = np.asarray(hands), np.asarray(pose)
        scale = float(np.linalg.norm(pose[11]-pose[12])) if usable[11] and usable[12] else 0.
        palms = [np.mean(x[PALM], axis=0) for x in hands]
        valid = [bool(inside(x[PALM], w, h).all()) and i not in suppressed for i,x in enumerate(hands)]
        anchors = [-1]*len(hands)
        if scale > 1:
            for i, hand in enumerate(hands):
                if not valid[i]:
                    continue
                distances = []
                for side, (elbow, wrist) in enumerate([(13,15),(14,16)]):
                    distances.append(float(np.linalg.norm(hand[0]-pose[wrist])/scale)
                        if usable[elbow] and usable[wrist] and inside(pose[[elbow,wrist]],w,h).all() else float('inf'))
                side = int(np.argmin(distances))
                if distances[side] <= PARAMS['pose_gate'] and distances[1-side]-distances[side] >= PARAMS['pose_margin']:
                    anchors[i] = side
            for side in (0,1):
                inds = [i for i,s in enumerate(anchors) if s==side]
                if len(inds)>1:
                    for i in inds: anchors[i]=-1
        output = dict(estimate=[None,None], proposal=list(map(int,baseline)), anchors=anchors,
                      seed=[False,False], reasons=['no_seed','no_seed'], traces={},
                      observed_finger_points_added=0)
        predictions = {}
        for side, tr in list(self.tracks.items()):
            age=time_s-tr['anchor_time']
            if age > PARAMS['max_age_s']+1e-8 or scale <= 1 or self.prev is None:
                output['reasons'][side]='expired_or_no_body_scale'
                del self.tracks[side]; continue
            new, info=optical_step(self.prev,gray,tr['points'],tr['center'],scale)
            info.update(anchor_time=tr['anchor_time'], age=age)
            output['traces'][str(side)]=info
            if new is None:
                output['reasons'][side]=info['reason']
                del self.tracks[side];continue
            # A visible other hand near the prediction can occlude/capture the tracker.
            clash=any(valid[i] and anchors[i]==1-side and np.linalg.norm(palms[i]-new['center']) < PARAMS['other_hand_gate']*scale
                      for i in range(len(hands)))
            if clash:
                output['reasons'][side]='other_hand_overlap_uncertain'
                info['decision']='other_hand_overlap_uncertain'
                del self.tracks[side];continue
            predictions[side]=new
            self.tracks[side]={**tr,**new}
            output['estimate'][side]=new['center'].tolist()
            output['reasons'][side]='image_position_estimate'
            info['decision']='image_position_estimate'
        # Match only current observations; unique image trajectory AND current arm.
        cost=np.full((len(hands),2),np.inf)
        for i in range(len(hands)):
            for side,new in predictions.items():
                if valid[i]:
                    cost[i,side]=np.linalg.norm(palms[i]-new['center'])/scale
        for i in range(len(hands)):
            if not valid[i]:continue
            side=int(np.argmin(cost[i]))
            rivals=[cost[i,1-side]]+[cost[j,side] for j in range(len(hands)) if j!=i]
            if (cost[i,side]<=PARAMS['match_gate']
                and all(cost[i,side] < PARAMS['ambiguity_ratio']*r for r in rivals)
                and anchors[i]==side and baseline[i]!=side
                and side not in [baseline[j] for j in range(len(hands)) if j!=i]):
                output['proposal'][i]=side
        # Never silently duplicate named slots.
        if len([s for s in output['proposal'] if s>=0]) != len(set(s for s in output['proposal'] if s>=0)):
            output['proposal']=list(map(int,baseline))
        # Seed on actual, currently arm-supported hand pixels, not raw labels/history.
        for i,side in enumerate(anchors):
            if side<0:continue
            mask=np.zeros_like(gray)
            poly=cv2.convexHull(hands[i,PALM].astype(np.float32)).astype(np.int32)
            cv2.fillConvexPoly(mask,poly,255)
            mask=cv2.erode(mask,np.ones((3,3),np.uint8))
            pts=cv2.goodFeaturesToTrack(gray,maxCorners=PARAMS['max_points'],
                qualityLevel=PARAMS['quality'],minDistance=PARAMS['min_distance'],mask=mask)
            if pts is not None and len(pts)>=PARAMS['min_points']:
                self.tracks[side]=dict(points=pts[:,0],center=palms[i].copy(),anchor_time=time_s)
                output['seed'][side]=True
        self.prev=gray.copy()
        return output
