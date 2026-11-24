"""Causal missing-hand region evidence; no coordinates/masks are repaired.

Engineering uncertainty envelope, NOT a calibrated probability/confidence band.
Only actual current-view full-hand observations refresh region history.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'word100_hand_visibility_20261017'))
from visibility import HandVisibility, PARAMS as BASE_PARAMS

PARAMS = {'history_seconds': BASE_PARAMS['history_seconds'],
          'base_uncertainty_shoulders': 0.05,
          'unknown_deformation_hand_diameters_per_second': 1.0,
          'max_uncertainty_hand_diameters': 0.5}

def intersection(a, b):
    return float(cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))[0])

class HandRegion:
    def __init__(self): self.reset()
    def reset(self):
        self.base = HandVisibility()
        self.tracks = [[], []]
        self.last_time = None
        self.size = None

    def step(self, xy, usable, pose_conf, timestamp_us, width, height,
             assigned=(False,False), unassigned_raw=False):
        xy = np.asarray(xy)
        usable = np.asarray(usable, dtype=bool)
        conf = np.asarray(pose_conf)
        t = int(timestamp_us)/1e6
        if ((self.last_time is not None and t <= self.last_time)
                or self.size not in (None,(width,height))): self.reset()
        self.last_time, self.size = t, (width,height)
        base = self.base.step(xy,usable,conf,timestamp_us,width,height,assigned,unassigned_raw)
        scale = float(np.linalg.norm(xy[11]-xy[12]))
        def in_frame(points):
            return bool(np.isfinite(points).all() and np.all(points>=0)
                        and np.all(points[:,0]<=width) and np.all(points[:,1]<=height))
        def anchor_ok(side):
            indices = [11,12,13+side,15+side]
            if not np.isfinite(scale) or scale<=1e-6: return False
            if not all(np.isfinite(xy[j]).all() and np.isfinite(conf[j]) and conf[j]>=BASE_PARAMS['pose_confidence'] for j in indices): return False
            if not all(usable[j] and in_frame(xy[j:j+1]) for j in indices[:3]): return False
            lengths=[np.linalg.norm(xy[11+side]-xy[13+side])/scale,
                     np.linalg.norm(xy[13+side]-xy[15+side])/scale]
            return all(BASE_PARAMS['arm_segment_min_shoulders']<=v<=BASE_PARAMS['arm_segment_max_shoulders'] for v in lengths)
        current = []
        for side in (0,1):
            points = xy[33+21*side:54+21*side]
            good = bool(usable[33+21*side:54+21*side].all() and in_frame(points) and anchor_ok(side))
            current.append(points.copy() if good else None)
            self.tracks[side] = [h for h in self.tracks[side] if t-h['t']<=PARAMS['history_seconds']]

        output = []
        for side in (0,1):
            old = base[side]
            row = {**old, 'evidence':dict(old['evidence']), 'baseline_state':old['state'],
                   'region_status':'unavailable', 'coordinate_repair':False}
            track = self.tracks[side]
            # Never reinterpret observed/partial/withheld detections as missing.
            if old['state'] not in ('inside_missing','overlap_likely','boundary_outside_likely') or not track or not anchor_ok(side):
                output.append(row); continue
            last = track[-1]
            age = t-last['t']
            # Pose wrist transports the last actually observed hand region only
            # for diagnostic evidence; these points are never output as landmarks.
            offsets = last['points']-last['anchor']
            center_points = offsets+xy[15+side]
            hull = cv2.convexHull(center_points.astype(np.float32)).reshape(-1,2)
            diameter = float(np.linalg.norm(np.ptp(last['points'],axis=0)))
            if len(hull)<3 or diameter<=1e-6 or cv2.contourArea(hull)<=0:
                output.append(row); continue
            residual_speed = 0.
            if len(track)>1:
                earlier=track[-2]
                dt=last['t']-earlier['t']
                residual_speed=float(np.max(np.linalg.norm(offsets-(earlier['points']-earlier['anchor']),axis=1))/dt)
            u = PARAMS['base_uncertainty_shoulders']*scale + age*(diameter*PARAMS['unknown_deformation_hand_diameters_per_second']+residual_speed)
            low,high=hull.min(0),hull.max(0)
            outer_low,outer_high=low-u,high+u
            row['region_status']='estimated_envelope'
            row['evidence'].update(region_age_seconds=age, uncertainty_pixels=u,
                uncertainty_shoulders=u/scale, observed_hand_diameter_pixels=diameter,
                residual_speed_pixels_per_second=residual_speed,
                nominal_hull=hull.tolist(), outer_bounds=[outer_low.tolist(),outer_high.tolist()],
                last_actual_frame_time=last['t'])
            reliable = u<=PARAMS['max_uncertainty_hand_diameters']*diameter
            # A boundary crossing that remains true under every permitted shift.
            certain_cross = bool(low[0]+u<0 or low[1]+u<0 or high[0]-u>width or high[1]-u>height)
            possible_cross = bool(outer_low[0]<0 or outer_low[1]<0 or outer_high[0]>width or outer_high[1]>height)
            if not reliable:
                row.update(state='region_uncertain',reason='motion_or_age_envelope_too_wide')
            elif certain_cross:
                row.update(state='boundary_outside_likely',reason='hand_region_crosses_boundary_across_envelope')
            elif possible_cross:
                row.update(state='boundary_uncertain',reason='hand_extent_may_cross_boundary')
            else:
                row.update(state='inside_missing',reason='estimated_hand_envelope_inside_frame')
                if current[1-side] is not None:
                    other=cv2.convexHull(current[1-side].astype(np.float32)).reshape(-1,2)
                    areas=([intersection(hull+np.array([dx,dy]),other) for dx in (-u,u) for dy in (-u,u)]
                           if len(other)>=3 and cv2.contourArea(other)>0 else [0.,0.,0.,0.])
                    row['evidence']['overlap_areas_pixels2_at_envelope_corners']=areas
                    if min(areas)>0:
                        row.update(state='overlap_likely',reason='hand_regions_intersect_across_translation_envelope')
            output.append(row)
        for side in (0,1):
            if current[side] is not None:
                self.tracks[side].append({'t':t,'points':current[side], 'anchor':xy[15+side].copy()})
                self.tracks[side]=self.tracks[side][-2:]
        return output
