"""Common input for both the learned classifier and the DTW baseline.

Reads the prepared pilot NPZ files read-only and turns each clip into the channel layout fixed in
model_contract.json. Every rule here comes from that contract; nothing is inferred from results.

Two facts were verified against the arrays rather than assumed:
  * native slice [25,46) is the LEFT hand and [46,67) the RIGHT - the slice-0 wrist sits nearer
    BODY_25 joint 7 (LWrist) in 1281/1281 checked frames for the first, 0/1281 for the second;
  * the subject faces a non-mirrored camera - BODY_25 LShoulder(5) is right of RShoulder(2) on
    screen in 40/40 clips checked.

`coordinate_present` means "finite and not (0,0)". The source confidence column is 1 everywhere,
including absent lower-body coordinates, so it is NOT used as observation quality anywhere here.
"""
from __future__ import annotations

import json
import os

import numpy as np

# --- contract: OpenPose BODY_25 -> the common 9-point upper body -------------
COMMON_BODY_ORDER = ("nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist", "left_hip", "right_hip")
OPENPOSE_BODY25_INDICES = (0, 5, 2, 6, 3, 7, 4, 12, 9)
NATIVE_HAND_SLICES = ((25, 46), (46, 67))        # (left, right), verified above
HAND_MCP = (5, 9, 13, 17)                        # index/middle/ring/pinky MCP
N_BODY, N_HAND, N_POINTS = 9, 21, 9 + 21 + 21    # 51
BODY = slice(0, 9)
LEFT_HAND = slice(9, 30)
RIGHT_HAND = slice(30, 51)
HAND_SLICES = (LEFT_HAND, RIGHT_HAND)
# indices inside the common 9-point body
L_SHOULDER, R_SHOULDER = 1, 2

CLASSES = ("ALDA", "GAMSAHAMNIDA", "JOTA", "MOREUDA", "NA")
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

# channels, in the order the model receives them (contract: 280 total)
CH_BODY_XY = N_POINTS * 2          # 102
CH_HAND_XY = N_HAND * 2 * 2        # 84
CH_BODY_VALID = N_POINTS           # 51
CH_HAND_VALID = N_HAND * 2         # 42
CH_DT = 1
TOTAL_CHANNELS = CH_BODY_XY + CH_HAND_XY + CH_BODY_VALID + CH_HAND_VALID + CH_DT


class PilotDataError(Exception):
    pass


def to_common(xy_native: np.ndarray, present_native: np.ndarray):
    """[T,67,*] native -> [T,51,*] common order: 9 body, 21 left hand, 21 right hand."""
    body = OPENPOSE_BODY25_INDICES
    lh, rh = NATIVE_HAND_SLICES
    xy = np.concatenate([xy_native[:, body], xy_native[:, lh[0]:lh[1]],
                         xy_native[:, rh[0]:rh[1]]], axis=1)
    pr = np.concatenate([present_native[:, body], present_native[:, lh[0]:lh[1]],
                         present_native[:, rh[0]:rh[1]]], axis=1)
    return xy.astype(np.float32), pr.astype(bool)


def body_scale(xy: np.ndarray, present: np.ndarray) -> float:
    """Median shoulder separation over frames where both shoulders are present."""
    ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    if not ok.any():
        raise PilotDataError("no frame has both shoulders present; clip has no usable body scale")
    d = np.linalg.norm(xy[ok, L_SHOULDER] - xy[ok, R_SHOULDER], axis=1)
    s = float(np.median(d))
    if not np.isfinite(s) or s <= 1e-6:
        raise PilotDataError(f"degenerate body scale {s}")
    return s


def hand_scale(xy: np.ndarray, present: np.ndarray, sl: slice) -> float:
    """Per-hand clip median wrist-to-MCP distance. NaN when it cannot be measured."""
    base = sl.start
    lens = []
    for t in range(xy.shape[0]):
        if not present[t, base]:
            continue
        for m in HAND_MCP:
            if present[t, base + m]:
                lens.append(float(np.linalg.norm(xy[t, base + m] - xy[t, base])))
    if not lens:
        return float("nan")
    s = float(np.median(lens))
    return s if s > 1e-6 else float("nan")


def build_channels(xy_native, present_native, timestamp_us) -> dict:
    """One clip -> the contract channel layout, plus the masks that say what is real.

    Invalid entries are marked in the masks and only then filled with zero. The fill happens
    AFTER normalisation, never before, so a missing joint cannot masquerade as the origin.
    """
    xy, present = to_common(xy_native, present_native)
    t_len = xy.shape[0]

    bs = body_scale(xy, present)
    shoulders_ok = present[:, L_SHOULDER] & present[:, R_SHOULDER]
    origin = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0      # per-frame midpoint

    body_rel = np.zeros((t_len, N_POINTS, 2), dtype=np.float32)
    body_valid = np.zeros((t_len, N_POINTS), dtype=np.float32)
    ok = present & shoulders_ok[:, None]                         # needs a frame origin too
    body_rel[ok] = (xy[ok] - np.repeat(origin[:, None, :], N_POINTS, axis=1)[ok]) / bs
    body_valid[ok] = 1.0

    hand_rel = np.zeros((t_len, N_HAND * 2, 2), dtype=np.float32)
    hand_valid = np.zeros((t_len, N_HAND * 2), dtype=np.float32)
    hand_scales = {}
    for hi, sl in enumerate(HAND_SLICES):
        hs = hand_scale(xy, present, sl)
        hand_scales["left" if hi == 0 else "right"] = hs
        if not np.isfinite(hs):
            continue                                             # whole hand stays invalid
        base, out = sl.start, hi * N_HAND
        wrist_ok = present[:, base]
        for j in range(N_HAND):
            good = wrist_ok & present[:, base + j]
            hand_rel[good, out + j] = (xy[good, base + j] - xy[good, base]) / hs
            hand_valid[good, out + j] = 1.0

    dt = np.zeros((t_len, 1), dtype=np.float32)
    if t_len > 1:
        dt[1:, 0] = np.diff(timestamp_us).astype(np.float32) / 1e6

    x = np.concatenate([body_rel.reshape(t_len, -1), hand_rel.reshape(t_len, -1),
                        body_valid, hand_valid, dt], axis=1).astype(np.float32)
    if x.shape[1] != TOTAL_CHANNELS:
        raise PilotDataError(f"channel count {x.shape[1]} != contract {TOTAL_CHANNELS}")
    if not np.isfinite(x).all():
        raise PilotDataError("non-finite value survived normalisation")
    return {"x": x, "body_valid": body_valid, "hand_valid": hand_valid,
            "body_scale": bs, "hand_scales": hand_scales, "frames": t_len}


def load_manifest(prep_dir: str) -> list:
    with open(os.path.join(prep_dir, "pilot_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_clip(prep_dir: str, item: dict) -> dict:
    z = np.load(os.path.join(prep_dir, "pilot_npz", os.path.basename(item["npz"])))
    out = build_channels(z["xy_native"], z["coordinate_present"], z["timestamp_nominal_us"])
    out["label"] = CLASS_TO_INDEX[item["word_id"]]
    out["word_id"] = item["word_id"]
    out["sample_id"] = item["sample_id"]
    out["signer_id"] = item["signer_id"]
    out["sequence_id"] = item["original_sequence_id"]
    out["split"] = item["split_role"]
    return out


def load_split(prep_dir: str, split: str) -> list:
    """Eligible clips of one split role. reserved_test is loadable but never fed to a model here."""
    items = [x for x in load_manifest(prep_dir)
             if x["source_only_pilot_eligible"] and x["split_role"] == split]
    return [load_clip(prep_dir, it) for it in items]


def collate(clips: list):
    """Pad to the longest clip in the batch and return an explicit padding mask."""
    t_max = max(c["frames"] for c in clips)
    n = len(clips)
    x = np.zeros((n, t_max, TOTAL_CHANNELS), dtype=np.float32)
    pad = np.zeros((n, t_max), dtype=np.float32)      # 1 = real frame
    y = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(clips):
        t = c["frames"]
        x[i, :t] = c["x"]
        pad[i, :t] = 1.0
        y[i] = c["label"]
    return x, pad, y
