"""Opt-in SHADOW hand-side assignment for a paired web test. Off unless a session asks for it.

One MediaPipe call per frame, exactly as before. Its result is assigned twice:
  primary row - the OPERATING rule (Landmarker._assign_sides), the only thing recorded/scored live
  shadow row  - the FROZEN start-provisional candidate, saved beside it for offline comparison
Same frame, same detections, same timestamp - only the side assignment differs. The candidate file
is imported unchanged; nothing here edits its rule or coefficients.

Take boundary: reset_assignment_state() clears ONLY the candidate's tracks (the offline evaluation
starts every clip with no track). MediaPipe and its session-continuous clock (_last_ms) are never
touched - that is the separate, already-fixed time-base rule.
"""
from __future__ import annotations

import hashlib
import os
import sys

from ..capture.landmarker import Landmarker
from ..core.observation import LEFT_HAND_SLICE, RIGHT_HAND_SLICE

_HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "ops",
                                             "start_provisional_candidate_20260926"))
CANDIDATE_FILE = os.path.join(CANDIDATE_DIR, "start_landmarker.py")
CANDIDATE_SHA256 = "c3095f12b5ad2b8e"            # prefix of the frozen, compared file
CANDIDATE_ID = "start_provisional_candidate_20260926"

if CANDIDATE_DIR not in sys.path:
    sys.path.insert(0, CANDIDATE_DIR)
from start_landmarker import StartLandmarker  # noqa: E402


def candidate_sha256() -> str:
    with open(CANDIDATE_FILE, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class ShadowLandmarker(StartLandmarker):
    """A Landmarker whose primary output is bit-for-bit the operating one."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        got = candidate_sha256()
        if not got.startswith(CANDIDATE_SHA256):
            raise RuntimeError(f"candidate file changed: {got[:16]} != {CANDIDATE_SHA256}")
        self._shadow_sides = {}

    def reset_assignment_state(self) -> None:
        """Candidate tracks only. Never MediaPipe, never _last_ms."""
        self.tracks = {}
        self.frame_log = []

    def _assign_sides(self, hand_res, pose_xy, pose_usable, w, h):
        self._last_hand_res = hand_res
        # candidate first (it updates its own tracks), on copies so it cannot touch the row
        self._shadow_sides = StartLandmarker._assign_sides(
            self, hand_res, pose_xy.copy(), pose_usable.copy(), w, h)
        return Landmarker._assign_sides(self, hand_res, pose_xy, pose_usable, w, h)

    def process(self, rgb, timestamp_us, out, row, shadow_out=None, shadow_row=0):
        self._shadow_sides = {}
        self._last_hand_res = None
        super().process(rgb, timestamp_us, out, row)          # sets the candidate clock, then operating
        if shadow_out is None:
            return
        for key in ("xy_px", "observed", "usable", "conf"):  # pose part: identical by construction
            shadow_out[key][shadow_row][:33] = out[key][row][:33]
        hand_res = self._last_hand_res
        if hand_res is None:
            return
        h, w = rgb.shape[:2]
        xy, observed, usable = (shadow_out["xy_px"][shadow_row], shadow_out["observed"][shadow_row],
                                shadow_out["usable"][shadow_row])
        # the same hand-writing rule as Landmarker.process, applied to the candidate's sides
        for hand_idx, (side, score) in self._shadow_sides.items():
            if side is None:
                continue
            base = (LEFT_HAND_SLICE if side == 0 else RIGHT_HAND_SLICE).start
            shadow_out["hand_score"][shadow_row, side] = score
            for j, lm in enumerate(hand_res.hand_landmarks[hand_idx][:21]):
                p = base + j
                xy[p] = (lm.x * w, lm.y * h)
                observed[p] = True
                usable[p] = self._in_bounds(xy[p], w, h) and score >= self.hand_score_thr
