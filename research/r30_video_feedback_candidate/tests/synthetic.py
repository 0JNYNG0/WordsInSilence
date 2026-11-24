"""Synthetic observations, capture manifests and references. Nothing here is a real recording.

Every landmark is generated from a formula, so these tests say what the CODE does and never make a
claim about a participant, a sign or an accuracy. The geometry is deliberately plain: straight
finger chains with a per-frame curl, so the bend_* angles are finite and change over time.
"""
from __future__ import annotations

import json
import zipfile

import numpy as np

from wis_lab.core import features as F
from wis_lab.core.observation import (
    LEFT_HAND_SLICE, N_POINTS, Observation, RIGHT_HAND_SLICE, empty_frames,
)

CONTRACT = {
    "contract_id": "synthetic_contract_v1",
    "sampling_A": {"budget": 32},
    "dtw": {"band": "ratio 0.2", "min_sampled_frames": 8},
    "cost": {"one_sided_penalty": 1.5, "one_sided_mode": "flat", "hand_participation": False},
}
WIDTH, HEIGHT = 640, 480
FINGER_DIRECTIONS = (-1.25, -0.45, -0.12, 0.20, 0.52)     # radians, in the image plane


def hand_points(cx: float, cy: float, curl: float, size: float = 46.0) -> np.ndarray:
    """21 landmarks: a wrist plus five straight chains, each joint turning by `curl`."""
    pts = np.zeros((21, 2), dtype=np.float64)
    pts[0] = (cx, cy)
    for direction, chain in zip(FINGER_DIRECTIONS, F.FINGER_CHAINS):
        angle = direction
        p = np.array([cx, cy], dtype=np.float64)
        length = size * 0.55                                 # wrist -> MCP
        for i in range(1, 5):
            p = p + np.array([np.cos(angle), np.sin(angle)]) * length
            pts[chain[i]] = p
            angle += curl
            length = size * 0.30
    return pts


def pose_points(x: float, y: float) -> np.ndarray:
    """33 pose landmarks. Only the ones the features read are placed meaningfully."""
    pts = np.zeros((33, 2), dtype=np.float64)
    for i in range(33):
        pts[i] = (x + 10 + i, y - 40 + (i % 5))
    pts[0] = (x, y - 90)                       # nose
    pts[9], pts[10] = (x - 8, y - 70), (x + 8, y - 70)
    pts[11], pts[12] = (x - 60, y), (x + 60, y)            # shoulders
    pts[13], pts[14] = (x - 80, y + 60), (x + 80, y + 60)  # elbows
    pts[15], pts[16] = (x - 90, y + 120), (x + 90, y + 120)  # wrists
    pts[23], pts[24] = (x - 40, y + 160), (x + 40, y + 160)  # hips
    return pts


def make_observation(n_frames: int = 60, *, hand_span=(5, 54), curl0: float = 0.10,
                     curl1: float = 0.60, frame_id_base: int = 7000, frame_step: int = 2,
                     body_x: float = 320.0, body_y: float = 200.0,
                     left_missing_rows=(), right_missing: bool = False,
                     sha256: str = "0" * 64) -> Observation:
    """A complete, valid Observation. frame_id is NOT the row index, on purpose."""
    a = empty_frames(n_frames)
    lo, hi = hand_span
    left_missing = set(left_missing_rows)
    for t in range(n_frames):
        a["frame_id"][t] = frame_id_base + t * frame_step
        a["timestamp_us"][t] = 1_000_000 + t * 33_333
        a["xy_px"][t, 0:33] = pose_points(body_x, body_y)
        a["observed"][t, 0:33] = True
        a["usable"][t, 0:33] = True
        a["conf"][t, 0:33] = 0.9
        if not (lo <= t <= hi):
            continue
        curl = curl0 + (curl1 - curl0) * (t - lo) / max(hi - lo, 1)
        if t not in left_missing:
            a["xy_px"][t, LEFT_HAND_SLICE] = hand_points(body_x - 90, body_y + 120, curl)
            a["observed"][t, LEFT_HAND_SLICE] = True
            a["usable"][t, LEFT_HAND_SLICE] = True
            a["hand_score"][t, 0] = 0.95
        if not right_missing:
            a["xy_px"][t, RIGHT_HAND_SLICE] = hand_points(body_x + 90, body_y + 120, curl * 0.8)
            a["observed"][t, RIGHT_HAND_SLICE] = True
            a["usable"][t, RIGHT_HAND_SLICE] = True
            a["hand_score"][t, 1] = 0.93
    obs = Observation(**a)
    obs.sidecar = {"observation_sha256": sha256, "width": WIDTH, "height": HEIGHT,
                   "clock_id": "synthetic_clock"}
    obs.validate()
    return obs


def normalized_points(obs, row: int) -> list:
    out = []
    for point, usable in zip(obs.xy_px[row], obs.usable[row]):
        out.append([float(point[0]) / WIDTH, float(point[1]) / HEIGHT]
                   if usable and np.isfinite(point).all() else None)
    return out


def make_manifest(obs, rows, *, status: str = "AVAILABLE", session_id: str = "ses_synthetic",
                  trial_id: str = "tri_synthetic", overrides=None) -> dict:
    """A capture manifest naming EXACTLY the given observation rows, in the persisted shape.

    `overrides` maps a position in `rows` to a dict of fields to corrupt, so identity rejection can
    be tested without reaching into the saved file.
    """
    overrides = overrides or {}
    frames = []
    for n, row in enumerate(rows):
        entry = {
            "frame_id": int(obs.frame_id[row]), "timestamp_us": int(obs.timestamp_us[row]),
            "width": WIDTH, "height": HEIGHT, "source_width": WIDTH, "source_height": HEIGHT,
            "observation_row": int(row), "image": f"frames/{n:04d}.jpg",
            "sha256": f"{n:064d}", "points_xy_normalized": normalized_points(obs, row),
        }
        entry.update(overrides.get(n, {}))
        frames.append(entry)
    return {
        "schema": "wis_local_capture_review_v1", "opted_in": True, "status": status,
        "frame_count": len(frames), "limit_reached": False, "encode_failures": 0,
        "encode_total_ms": 0.0, "max_sample_fps": 10, "audio": False, "training_use": False,
        "session_id": session_id, "trial_id": trial_id,
        "observation_sha256": obs.sidecar["observation_sha256"],
        "clock_id": "synthetic_clock", "mirrored": False, "frames": frames,
    }


def write_capture_zip(path, manifest: dict) -> str:
    """The same container persist_capture writes, built without touching cv2 or a camera."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for entry in manifest["frames"]:
            archive.writestr(entry["image"], b"synthetic-not-a-photo")
        archive.writestr("manifest.json",
                         json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    return str(path)


def make_reference(word_id: str = "synthetic_word", clip_id: str = "SYN_word_01", **kwargs) -> dict:
    """A prepared reference sample, built through the real build_reference path."""
    from wis_lab.core.recognize import build_reference
    obs = make_observation(**{"n_frames": 48, "hand_span": (4, 43), "curl0": 0.35,
                              "curl1": 0.85, "frame_id_base": 100, "body_x": 300.0,
                              "sha256": "1" * 64, **kwargs})
    return build_reference(obs, clip_id, word_id, int(CONTRACT["sampling_A"]["budget"]))


def uniform_query_rows(obs) -> list[int]:
    """The observation rows the CLASSIFIER's uniform-time sample selects. A different list."""
    seq = F.sequence_features(obs, trim=True)
    sample = F.uniform_time_sample(seq, int(CONTRACT["sampling_A"]["budget"]))
    lo = int(seq["span"][0])
    return [lo + int(i) for i in sample["picked_rows"]]


assert N_POINTS == 75        # the generators above assume the 33 + 21 + 21 layout
