"""Research-only anatomical vicinity evidence, separate from correctness feedback.

Regions are approximate surface patches, not certified anatomical/linguistic borders.
2D projection cannot prove touch. Face landmarks behind a hand remain predictions.
No word ID, target sign, user identity, or classifier output enters these functions.
"""
from __future__ import annotations

import cv2
import numpy as np

REGIONS = ('forehead', 'cheek_R', 'cheek_L', 'mouth', 'chin', 'nose', 'neck', 'chest')
# MediaPipe topology IDs. R/L are subject-anatomical for unmirrored source frames.
PATCHES = ((10, 151, 108, 337), (50, 101, 116, 123), (280, 330, 345, 352),
           (13, 14, 61, 291), (152, 175, 199), (1, 4, 5))
# Ellipse radii in face-width / face-height units; descriptive candidates, not pass thresholds.
RADII = np.array(((.28, .16), (.20, .17), (.20, .17), (.23, .12), (.20, .13), (.14, .14)))
TIPS = np.array((4, 8, 12, 16, 20))


def region_geometry(face_xy, pose_xy, pose_ok):
    centers = np.full((8, 2), np.nan)
    axes = np.eye(2)
    radii = np.full((8, 2), np.nan)
    valid = np.zeros(8, dtype=bool)
    if np.isfinite(face_xy).all():
        # Eye-corner horizontal axis follows roll, not image horizontal.
        ex = face_xy[263] - face_xy[33]
        w = np.linalg.norm(face_xy[454] - face_xy[234])
        h = np.linalg.norm(face_xy[152] - face_xy[10])
        ex_norm = np.linalg.norm(ex)
        if w > 8 and h > 8 and ex_norm > 4:
            ex = ex / ex_norm
            ey = np.array((-ex[1], ex[0]))
            if np.dot(ey, face_xy[152] - face_xy[10]) < 0:
                ey = -ey
            axes = np.stack((ex, ey))
            centers[:6] = [face_xy[list(p)].mean(axis=0) for p in PATCHES]
            radii[:6] = RADII * (w, h)
            # Highly oblique/collapsed geometry isn't a reliable front-face region map.
            a = np.linalg.norm(face_xy[1] - face_xy[234])
            b = np.linalg.norm(face_xy[1] - face_xy[454])
            valid[:6] = abs(a-b)/max(a+b, 1) < .45 and .35 < w/h < 1.5
    if pose_ok[11] and pose_ok[12]:
        shoulders = pose_xy[[11, 12]]
        origin = shoulders.mean(axis=0)
        width = np.linalg.norm(shoulders[0] - shoulders[1])
        down = axes[1]
        if width > 8:
            centers[6] = origin - down * width * .12
            centers[7] = origin + down * width * .35
            radii[6] = (width*.16, width*.20)
            radii[7] = (width*.45, width*.38)
            valid[6:] = True
    return centers, axes, radii, valid


def relations(observation, face):
    if not np.array_equal(observation.frame_id, face['frame_id']):
        raise ValueError("Relations require identical frame IDs")
    n = observation.n_frames
    distances = np.full((n, 2, 5, 8), np.nan, dtype=np.float32)
    centers_all = np.full((n, 8, 2), np.nan, dtype=np.float32)
    region_valid = np.zeros((n, 8), bool)
    occluded = np.zeros((n, 8), bool)
    labels = np.full((n, 2, 5), -1, dtype=np.int16)
    comparable = np.zeros_like(distances, dtype=bool)
    for t in range(n):
        centers, axes, radii, valid = region_geometry(face['xy_px'][t], observation.xy_px[t], observation.usable[t])
        centers_all[t] = centers
        region_valid[t] = valid
        # A visible hand projected over a face patch is an occlusion warning, not touch proof.
        for base in (33, 54):
            pts = observation.xy_px[t, base:base+21]
            ok = observation.usable[t, base:base+21] & np.isfinite(pts).all(axis=1)
            if ok.sum() >= 3:
                hull = cv2.convexHull(pts[ok].astype(np.float32))
                for r in range(6):
                    if valid[r]:
                        margin = min(radii[r]) * .25
                        occluded[t, r] |= cv2.pointPolygonTest(hull, tuple(centers[r].astype(float)), True) >= -margin
        for side, base in enumerate((33, 54)):
            for k, tip in enumerate(TIPS):
                p = base + tip
                if not observation.usable[t, p] or not np.isfinite(observation.xy_px[t, p]).all():
                    continue
                delta = (observation.xy_px[t, p] - centers) @ axes.T
                with np.errstate(invalid='ignore', divide='ignore'):
                    d = np.linalg.norm(delta / radii, axis=1)
                distances[t, side, k, valid] = d[valid]
                comparable[t, side, k] = valid & ~occluded[t]
                ranking = np.argsort(np.where(valid, d, np.inf))
                best, second = ranking[:2]
                if valid[best] and d[best] <= 1 and d[second] - d[best] >= .2:
                    labels[t, side, k] = best
    return dict(distances=distances, comparable=comparable, labels=labels,
                region_centers=centers_all, region_valid=region_valid,
                projected_hand_overlap=occluded,
                frame_id=observation.frame_id.copy(), timestamp_us=observation.timestamp_us.copy())


def compare_path(reference, query, ref_rows, query_rows):
    """Coverage denominator depends on reference, never shrinks when query hand disappears.

    This is a diagnostic; does not label a sign correct or wrong. Repeated DTW matches
    are collapsed by reference frame to avoid artificial coverage inflation.
    """
    a = np.asarray(ref_rows, dtype=int)
    b = np.asarray(query_rows, dtype=int)
    expected = reference['comparable'][a]
    both = expected & query['comparable'][b]
    den, num, diffs = 0, 0, []
    for rid in np.unique(a):
        at = np.flatnonzero(a == rid)
        needed = expected[at[0]]
        supported = both[at].all(axis=0) & needed
        den += int(needed.sum())
        num += int(supported.sum())
        if supported.any():
            delta = np.abs(reference['distances'][a[at]] - query['distances'][b[at]])
            diffs.extend(delta[:, supported].mean(axis=0).tolist())
    return dict(reference_cells=den, compared_cells=num, coverage=num/den if den else None,
                mean_distance_difference=float(np.mean(diffs)) if diffs else None,
                status='INSUFFICIENT' if not num else ('PARTIAL' if num < den else 'OBSERVED'),
                correctness=None, contact_confirmed=False)
