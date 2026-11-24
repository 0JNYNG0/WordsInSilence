"""Raw 2D observation container: the one schema shared by clips and webcam.

Contract: config/contract.json -> observation (wis_observation_2d_v1)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "wis_observation_2d_v1"
N_POINTS = 75
POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
HAND_SLICES = (LEFT_HAND_SLICE, RIGHT_HAND_SLICE)
POSE_L_WRIST, POSE_R_WRIST = 15, 16
POSE_L_SHOULDER, POSE_R_SHOULDER = 11, 12


@dataclass
class Observation:
    xy_px: np.ndarray          # float32 [T,75,2], NaN where missing
    timestamp_us: np.ndarray   # int64 [T]
    frame_id: np.ndarray       # int64 [T]
    observed: np.ndarray       # bool [T,75]
    usable: np.ndarray         # bool [T,75]
    conf: np.ndarray           # float32 [T,75], NaN for hand joints
    hand_score: np.ndarray     # float32 [T,2]
    sidecar: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.xy_px.shape[0])

    def validate(self) -> None:
        t = self.n_frames
        assert self.xy_px.shape == (t, N_POINTS, 2), self.xy_px.shape
        assert self.xy_px.dtype == np.float32
        assert self.timestamp_us.shape == (t,) and self.timestamp_us.dtype == np.int64
        assert self.frame_id.shape == (t,) and self.frame_id.dtype == np.int64
        assert self.observed.shape == (t, N_POINTS) and self.observed.dtype == np.bool_
        assert self.usable.shape == (t, N_POINTS) and self.usable.dtype == np.bool_
        assert self.conf.shape == (t, N_POINTS) and self.conf.dtype == np.float32
        assert self.hand_score.shape == (t, 2) and self.hand_score.dtype == np.float32
        # missing must be NaN, never a valid-looking (0,0)
        assert not np.isfinite(self.xy_px[~self.observed]).any(), "missing points must be NaN"
        # usable implies observed
        assert not (self.usable & ~self.observed).any(), "usable without observed"
        if t > 1:
            assert (np.diff(self.timestamp_us) > 0).all(), "timestamps must be strictly increasing"

    def save(self, npz_path: str) -> str:
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        self.validate()
        np.savez_compressed(
            npz_path,
            schema=np.array(SCHEMA),
            xy_px=self.xy_px,
            timestamp_us=self.timestamp_us,
            frame_id=self.frame_id,
            observed=self.observed,
            usable=self.usable,
            conf=self.conf,
            hand_score=self.hand_score,
        )
        side = os.path.splitext(npz_path)[0] + ".sidecar.json"
        sidecar = dict(self.sidecar)
        sidecar["observation_sha256"] = sha256_file(npz_path)
        with open(side, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(sidecar, fh, ensure_ascii=False, indent=2, sort_keys=True)
        self.sidecar = sidecar
        return sidecar["observation_sha256"]

    @classmethod
    def load(cls, npz_path: str) -> "Observation":
        with np.load(npz_path, allow_pickle=False) as z:
            assert str(z["schema"]) == SCHEMA, f"unexpected schema {z['schema']}"
            obs = cls(
                xy_px=z["xy_px"], timestamp_us=z["timestamp_us"], frame_id=z["frame_id"],
                observed=z["observed"], usable=z["usable"], conf=z["conf"],
                hand_score=z["hand_score"],
            )
        side = os.path.splitext(npz_path)[0] + ".sidecar.json"
        if os.path.exists(side):
            with open(side, encoding="utf-8") as fh:
                obs.sidecar = json.load(fh)
        obs.validate()
        return obs

    def quality_summary(self) -> dict:
        t = max(self.n_frames, 1)
        def rate(sl):
            return float(self.usable[:, sl].any(axis=1).sum()) / t
        gaps = np.diff(self.timestamp_us) if self.n_frames > 1 else np.array([0], dtype=np.int64)
        any_hand = (self.usable[:, LEFT_HAND_SLICE].any(axis=1)
                    | self.usable[:, RIGHT_HAND_SLICE].any(axis=1))
        idx = np.flatnonzero(any_hand)
        # A clip starts and ends with the hands resting out of view. Those frames are a real
        # absence, not a detector failure, so the span-local rate is reported next to the raw one.
        if idx.size:
            first, last = int(idx[0]), int(idx[-1])
            span_rate = float(any_hand[first:last + 1].mean())
        else:
            first, last, span_rate = -1, -1, 0.0
        return {
            "active_span_first_frame": first,
            "active_span_last_frame": last,
            "lead_in_frames": first if first >= 0 else self.n_frames,
            "lead_out_frames": (self.n_frames - 1 - last) if last >= 0 else self.n_frames,
            "hand_rate_in_active_span": span_rate,
            "frames": self.n_frames,
            "duration_ms": float((self.timestamp_us[-1] - self.timestamp_us[0]) / 1000.0) if self.n_frames > 1 else 0.0,
            "pose_frame_rate": rate(POSE_SLICE),
            "left_hand_frame_rate": rate(LEFT_HAND_SLICE),
            "right_hand_frame_rate": rate(RIGHT_HAND_SLICE),
            "any_hand_frame_rate": float(
                (self.usable[:, LEFT_HAND_SLICE].any(axis=1) | self.usable[:, RIGHT_HAND_SLICE].any(axis=1)).sum()
            ) / t,
            "both_hands_frame_rate": float(
                (self.usable[:, LEFT_HAND_SLICE].any(axis=1) & self.usable[:, RIGHT_HAND_SLICE].any(axis=1)).sum()
            ) / t,
            "shoulder_frame_rate": float(
                (self.usable[:, POSE_L_SHOULDER] & self.usable[:, POSE_R_SHOULDER]).sum()
            ) / t,
            "median_frame_gap_ms": float(np.median(gaps) / 1000.0),
            "max_frame_gap_ms": float(np.max(gaps) / 1000.0),
        }


def empty_frames(t: int) -> dict:
    """Pre-allocated arrays for one clip/session of t frames."""
    return {
        "xy_px": np.full((t, N_POINTS, 2), np.nan, dtype=np.float32),
        "timestamp_us": np.zeros(t, dtype=np.int64),
        "frame_id": np.zeros(t, dtype=np.int64),
        "observed": np.zeros((t, N_POINTS), dtype=np.bool_),
        "usable": np.zeros((t, N_POINTS), dtype=np.bool_),
        "conf": np.full((t, N_POINTS), np.nan, dtype=np.float32),
        "hand_score": np.full((t, 2), np.nan, dtype=np.float32),
    }


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
