"""Stage 0 helpers: native frame decimation and a pinned, role-checked loader. No model here.

decimate() keeps frames offset, offset+stride, ... of the stored native arrays. Coordinates, the
presence mask and the timestamps are copied together and unchanged: nothing is interpolated,
filled or appended. Frame 0 is kept when offset is 0. An odd-length clip keeps its last frame
under stride 2 and an even-length clip does not - the rule is the slice, nothing more.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

NATIVE_FIELDS = ("xy_native", "coordinate_present", "timestamp_nominal_us")
NOMINAL_STEP_US = 1e6 / 30.0
RESERVED_SIGNERS = (14, 15, 16)


class LoaderRefused(Exception):
    """A row the stage-0 contract does not allow to be opened."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def decimate(xy, present, ts, stride: int = 2, offset: int = 0):
    """-> (xy, present, ts) copies of rows offset::stride. The inputs are never modified."""
    if type(stride) is not int or stride < 1:
        raise ValueError(f"stride must be an int >= 1, got {stride!r}")
    if type(offset) is not int or not 0 <= offset < stride:
        raise ValueError(f"offset must be an int in [0, stride), got {offset!r}")
    xy, present, ts = np.asarray(xy), np.asarray(present), np.asarray(ts)
    n = len(ts)
    if n == 0:
        raise ValueError("empty clip")
    if xy.shape[0] != n or present.shape[0] != n:
        raise ValueError(f"frame counts differ: xy {xy.shape[0]}, present {present.shape[0]}, ts {n}")
    keep = slice(offset, None, stride)
    return xy[keep].copy(), present[keep].copy(), ts[keep].copy()


def kept_intervals_are_sums(ts, ts_kept, stride: int = 2, offset: int = 0) -> bool:
    """Every kept interval equals the sum of the `stride` original intervals it spans."""
    ts = np.asarray(ts, dtype=np.int64)
    ts_kept = np.asarray(ts_kept, dtype=np.int64)
    return bool(np.array_equal(ts_kept, ts[offset::stride]))


def nominal_dt_doubles(ts, ts_kept, tol_us: float = 1.0) -> bool:
    """On the 30 Hz nominal clock: every original step ~33333 us, every kept step twice that."""
    d0 = np.diff(np.asarray(ts, dtype=np.float64))
    d1 = np.diff(np.asarray(ts_kept, dtype=np.float64))
    if len(d0) == 0:
        return len(d1) == 0
    ok0 = bool(np.all(np.abs(d0 - NOMINAL_STEP_US) <= tol_us))
    ok1 = bool(np.all(np.abs(d1 - 2 * NOMINAL_STEP_US) <= 2 * tol_us)) if len(d1) else True
    return ok0 and ok1


def recording_id(sample_id: str) -> str:
    """Samples cut from one recording share this prefix (same rule as the supcon preflight)."""
    return re.sub(r"_a\d\d_\d\d\d$", "", sample_id)


def load_native(row: dict, allowed_roles, allowed_signers, allowed_root) -> dict:
    """Open ONE metadata row's native npz after role, signer, location and hash checks.

    Returns the three native arrays only. The reserved signers are refused whatever the role says.
    """
    role, signer = row.get("role"), row.get("signer_id")
    if role not in allowed_roles:
        raise LoaderRefused(f"{row.get('id')}: role {role!r} not allowed")
    if signer in RESERVED_SIGNERS or signer not in allowed_signers:
        raise LoaderRefused(f"{row.get('id')}: signer {signer!r} not allowed")
    path = Path(row["path"]).resolve()
    root = Path(allowed_root).resolve()
    if os.path.commonpath([str(path), str(root)]) != str(root):
        raise LoaderRefused(f"{row.get('id')}: {path} is outside {root}")
    got = sha256_file(path)
    if got != row["sha256"]:
        raise LoaderRefused(f"{row.get('id')}: sha256 mismatch {got}")
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in NATIVE_FIELDS if k not in z.files]
        if missing:
            raise LoaderRefused(f"{row.get('id')}: npz lacks {missing}")
        return {"xy": z["xy_native"].copy(), "present": z["coordinate_present"].copy(),
                "ts": z["timestamp_nominal_us"].copy(), "sha256": got}


def web_gate_information(present_native, min_frames: int, min_hand_frames: int) -> dict:
    """Would the web adapter's frame / observed-hand minimums reject this clip? Information only.

    Counted over the whole clip (annotation clips are already cut); native hand slots 25..66.
    """
    present_native = np.asarray(present_native, dtype=bool)
    frames = int(present_native.shape[0])
    hand_frames = int(present_native[:, 25:67].any(axis=1).sum())
    return {"frames": frames, "observed_hand_frames": hand_frames,
            "under_web_minimums": bool(frames < min_frames or hand_frames < min_hand_frames)}
