"""Single owner of the MediaPipe models. Used identically by clips and webcam.

Loaded once per process; VIDEO running mode; monotonic int-ms timestamps.
"""
from __future__ import annotations

import os

import numpy as np
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..core.observation import (
    LEFT_HAND_SLICE, POSE_L_WRIST, POSE_R_WRIST, RIGHT_HAND_SLICE,
    sha256_file, sha256_json,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DEFAULT_HAND_MODEL = os.path.join(PROJECT_ROOT, "Tools", "models", "hand_landmarker.task")
DEFAULT_POSE_MODEL = os.path.join(PROJECT_ROOT, "Tools", "models", "pose_landmarker_lite.task")

EXTRACTOR_CODE_VERSION = "wis_extract_v1.1"


class Landmarker:
    def __init__(self, contract: dict, hand_model: str = DEFAULT_HAND_MODEL,
                 pose_model: str = DEFAULT_POSE_MODEL):
        self.contract = contract
        self.hand_model_path = os.path.abspath(hand_model)
        self.pose_model_path = os.path.abspath(pose_model)
        for p in (self.hand_model_path, self.pose_model_path):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        rule = contract["usable_rule"]
        self.pose_vis_thr = float(rule["pose_visibility_threshold"])
        self.hand_score_thr = float(rule["hand_score_threshold"])
        self.margin = float(rule["bounds_margin_ratio"])

        self.hand = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=2))
        self.pose = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.pose_model_path),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=1))
        self._last_ms = -1
        self.handedness_disagreements = 0
        self.handedness_comparisons = 0
        self.unassigned_hands = 0

    # ---- identity -------------------------------------------------------
    def extractor_id(self) -> str:
        payload = {
            "code": EXTRACTOR_CODE_VERSION,
            "hand_model_sha256": sha256_file(self.hand_model_path),
            "pose_model_sha256": sha256_file(self.pose_model_path),
            "usable_rule": self.contract["usable_rule"],
            "hand_assignment": self.contract["hand_assignment"]["policy"],
            "point_layout": self.contract["observation"]["point_layout"],
            "models": self.contract["models"],
        }
        return sha256_json(payload)[:16]

    def close(self) -> None:
        for m in (self.hand, self.pose):
            try:
                m.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- per frame ------------------------------------------------------
    def next_timestamp_ms(self, timestamp_us: int) -> int:
        """MediaPipe VIDEO mode needs strictly increasing int ms; original us is kept by the caller."""
        ms = int(timestamp_us // 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms

    def process(self, rgb: np.ndarray, timestamp_us: int, out: dict, row: int) -> None:
        """Fill row `row` of the pre-allocated arrays in `out` from one RGB frame."""
        h, w = rgb.shape[:2]
        ms = self.next_timestamp_ms(timestamp_us)
        image = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        pose_res = self.pose.detect_for_video(image, ms)
        hand_res = self.hand.detect_for_video(image, ms)

        xy = out["xy_px"][row]
        observed = out["observed"][row]
        usable = out["usable"][row]
        conf = out["conf"][row]

        if pose_res.pose_landmarks:
            for i, lm in enumerate(pose_res.pose_landmarks[0][:33]):
                xy[i] = (lm.x * w, lm.y * h)
                observed[i] = True
                conf[i] = lm.visibility
                usable[i] = self._in_bounds(xy[i], w, h) and lm.visibility >= self.pose_vis_thr

        if hand_res.hand_landmarks:
            sides = self._assign_sides(hand_res, xy, usable, w, h)
            for hand_idx, (side, score) in sides.items():
                if side is None:
                    self.unassigned_hands += 1
                    continue
                sl = LEFT_HAND_SLICE if side == 0 else RIGHT_HAND_SLICE
                base = sl.start
                out["hand_score"][row, side] = score
                for j, lm in enumerate(hand_res.hand_landmarks[hand_idx][:21]):
                    p = base + j
                    xy[p] = (lm.x * w, lm.y * h)
                    observed[p] = True
                    usable[p] = self._in_bounds(xy[p], w, h) and score >= self.hand_score_thr

    def _in_bounds(self, pt, w: int, h: int) -> bool:
        if not np.isfinite(pt).all():
            return False
        mx, my = self.margin * w, self.margin * h
        return (-mx <= pt[0] <= w + mx) and (-my <= pt[1] <= h + my)

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w: int, h: int) -> dict:
        """Bind each detected hand to an anatomical side.

        Primary: nearest usable pose wrist (the pose model labels sides anatomically).
        Fallback: the raw MediaPipe handedness label - see the measurement note below.
        """
        n = len(hand_res.hand_landmarks)
        wrists = np.array([[hand_res.hand_landmarks[i][0].x * w,
                            hand_res.hand_landmarks[i][0].y * h] for i in range(n)])
        scores = [float(hand_res.handedness[i][0].score) for i in range(n)]
        labels = [hand_res.handedness[i][0].category_name for i in range(n)]
        # MediaPipe label -> anatomical side index (0=left, 1=right), used as fallback only.
        # Measured, not assumed: tests/diag_handedness.py scores the RAW label at 0.991 against
        # pose anatomy on these non-mirrored clips, the swapped reading at 0.009.
        mp_side = [0 if lab == "Left" else 1 for lab in labels]

        pose_ref = {}
        for side, idx in ((0, POSE_L_WRIST), (1, POSE_R_WRIST)):
            if pose_usable[idx]:
                pose_ref[side] = pose_xy[idx]

        out = {}
        if pose_ref:
            # greedy nearest assignment over (hand, side) pairs
            pairs = sorted(
                ((float(np.linalg.norm(wrists[i] - pose_ref[s])), i, s)
                 for i in range(n) for s in pose_ref),
                key=lambda x: x[0])
            taken_hand, taken_side = set(), set()
            for _dist, i, s in pairs:
                if i in taken_hand or s in taken_side:
                    continue
                out[i] = (s, scores[i])
                taken_hand.add(i)
                taken_side.add(s)
                self.handedness_comparisons += 1
                if mp_side[i] != s:
                    self.handedness_disagreements += 1
            for i in range(n):
                if i not in out:
                    out[i] = (None, scores[i])  # ambiguous: do not force a side
        else:
            taken_side = set()
            for i in range(n):
                s = mp_side[i]
                out[i] = ((None, scores[i]) if s in taken_side else (s, scores[i]))
                taken_side.add(s)
        return out
