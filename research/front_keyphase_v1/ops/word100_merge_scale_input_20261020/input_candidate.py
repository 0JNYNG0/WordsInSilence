"""Research input-only three-way contrast; original models and observations read-only."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
ROOT = O.parents[2]
sys.dont_write_bytecode = True
sys.path[:0] = [str(O.parent / "src"), str(ROOT / "research/sign_model_pilot_v1/src")]
from wis_lab.core.observation import Observation
from pilot import data as D
from pilot.web_input import to_native_like

spec = importlib.util.spec_from_file_location("merge_scale_roi", O / "word100_fullframe_roi_pair_20261016/roi_adapter.py")
roi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roi)

CONDITIONS = ("full_before", "merge_fixed_scale", "merge_recomputed_scale")
PALM = (0, 5, 9, 13, 17)
CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
          (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
TRIPLES = [(c[i-1], c[i], c[i+1]) for c in CHAINS for i in (1, 2, 3)]


def numeric(channels):
    return {k: v.copy() for k, v in channels.items() if isinstance(v, np.ndarray)}


def three_conditions(before, merged):
    a = roi.build_roi_channels(before)
    c = roi.build_roi_channels(merged)
    b = numeric(c)
    xy, present = D.to_common(*to_native_like(merged.xy_px, merged.usable))
    local = np.zeros((merged.n_frames, 42, 2), np.float32)
    valid = np.zeros((merged.n_frames, 42), np.float32)
    for side, sl in enumerate(D.HAND_SLICES):
        scale = a["hand_scales"]["left" if side == 0 else "right"]
        if not np.isfinite(scale):
            continue
        for j in range(21):
            good = present[:, sl.start] & present[:, sl.start + j]
            local[good, side*21+j] = (xy[good, sl.start+j] - xy[good, sl.start]) / scale
            valid[good, side*21+j] = 1
    b["x"][:, 102:186] = local.reshape(merged.n_frames, 84)
    b["x"][:, 237:279] = valid
    b["hand_valid"] = valid
    return dict(zip(CONDITIONS, (numeric(a), b, numeric(c)))), a["hand_scales"], c["hand_scales"]


def angles(xy, usable):
    result = np.full((*xy.shape[:-2], 15), np.nan, np.float32)
    available = np.zeros(result.shape, bool)
    for k, (a, b, c) in enumerate(TRIPLES):
        u, v = xy[..., a, :] - xy[..., b, :], xy[..., c, :] - xy[..., b, :]
        nu, nv = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        ok = usable[..., [a, b, c]].all(-1) & (nu > 1e-6) & (nv > 1e-6)
        cosine = np.clip((u*v).sum(-1) / np.where(ok, nu*nv, 1), -1, 1)
        result[..., k] = np.where(ok, np.degrees(np.arccos(cosine)), np.nan)
        available[..., k] = ok
    return result, available


def evidence(before, merged, records):
    """Availability is not anatomical accuracy. Do not feed diagnostics to 280 channels."""
    t = merged.n_frames
    p = merged.xy_px[:, 33:].reshape(t, 2, 21, 2)
    v = merged.usable[:, 33:].reshape(t, 2, 21)
    old = before.usable[:, 33:].reshape(t, 2, 21)
    origin = np.where(old, 1, np.where(v, 2, 0)).astype(np.uint8)
    av, ok = angles(p, v)
    palm_ok = v[..., PALM].all(-1)
    palm = np.where(palm_ok[..., None], np.mean(p[..., PALM, :], axis=-2), np.nan)
    crop_angles = np.full_like(av, np.nan)
    crop_available = np.zeros_like(ok)
    discrepancy = np.full_like(av, np.nan)
    for i, frame in enumerate(records["frames"]):
        for decision in frame["decisions"]:
            if not decision["accepted"]:
                continue
            s = decision["side"]
            q = np.array(frame["crop"][decision["crop_index"]]["xy"], np.float32)
            # ROI/point agreement already constrains merged; use identical point mask.
            ca, cv = angles(q, v[i, s])
            crop_angles[i, s] = ca
            crop_available[i, s] = cv
            discrepancy[i, s] = np.where(ok[i, s] & cv, abs(av[i, s] - ca), np.nan)
    return dict(point_origin=origin, wrist_available=v[..., 0], palm_available=palm_ok,
                palm_xy_px=palm.astype(np.float32), bend_degrees=av, bend_available=ok,
                crop_bend_degrees=crop_angles, crop_bend_available=crop_available,
                two_scale_angle_difference_deg=discrepancy,
                core=np.array([e["core"] for e in records["frames"]]),
                frame_id=merged.frame_id.copy(), timestamp_us=merged.timestamp_us.copy())
