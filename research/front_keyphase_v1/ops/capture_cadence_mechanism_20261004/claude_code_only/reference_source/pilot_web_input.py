"""MediaPipe web recordings -> the same 280-channel common input the pilot model was trained on.

Read-only against research/front_keyphase_v1. Nothing in the web recogniser is modified.

The point of this module is to MEASURE the extractor gap, not to paper over it. Mapping the joints
makes the two sources the same SHAPE; it does not make them the same MEASUREMENT, and the contract
says exactly that. Any result produced through here is evidence about transfer, never proof of
compatibility.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .data import CLASS_TO_INDEX, build_channels

# contract: mediapipe_pose33_indices for the common 9-point body, and the web hand slices
MEDIAPIPE_POSE33_INDICES = (0, 11, 12, 13, 14, 15, 16, 23, 24)
WEB_HAND_SLICES = ((33, 54), (54, 75))
WEB_N_POINTS = 75


def to_native_like(xy_px: np.ndarray, usable: np.ndarray):
    """Web [T,75,*] -> the 67-slot native layout build_channels() expects.

    Slot order matches the OpenPose pack: BODY_25 slots 0-24, left hand 25-45, right hand 46-66.
    Only the nine body slots the contract names are filled; the rest stay absent, which the
    validity masks record honestly rather than hiding.
    """
    t = xy_px.shape[0]
    xy = np.zeros((t, 67, 2), dtype=np.float32)
    present = np.zeros((t, 67), dtype=bool)

    # BODY_25 destination slots for the common nine, in the same order as the common body list
    body25_slots = (0, 5, 2, 6, 3, 7, 4, 12, 9)
    for dst, src in zip(body25_slots, MEDIAPIPE_POSE33_INDICES):
        xy[:, dst] = xy_px[:, src]
        present[:, dst] = usable[:, src]

    for (d0, d1), (s0, s1) in zip(((25, 46), (46, 67)), WEB_HAND_SLICES):
        xy[:, d0:d1] = xy_px[:, s0:s1]
        present[:, d0:d1] = usable[:, s0:s1]

    xy[~present] = 0.0          # absent slots are marked absent, not given a plausible position
    return xy, present


def active_span(present: np.ndarray):
    """First..last frame with any usable hand point - the same trimming the web path applies."""
    any_hand = present[:, 25:67].any(axis=1)
    idx = np.flatnonzero(any_hand)
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def load_web_clip(npz_path: str, trim: bool = True) -> dict:
    """One web trial NPZ -> the pilot's channel layout.

    `usable` is the web recogniser's own validity flag. A lost hand stays lost: it is never
    rewritten as an inactive hand or as a legitimate hold.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        xy_px = z["xy_px"].astype(np.float32)
        usable = z["usable"].astype(bool)
        ts = z["timestamp_us"].astype(np.int64)
    if xy_px.shape[1] != WEB_N_POINTS:
        raise ValueError(f"expected 75 web points, got {xy_px.shape[1]}")

    xy, present = to_native_like(xy_px, usable)
    if trim:
        span = active_span(present)
        if span is None:
            raise ValueError("no frame has a usable hand")
        lo, hi = span
        xy, present, ts = xy[lo:hi + 1], present[lo:hi + 1], ts[lo:hi + 1]
    out = build_channels(xy, present, ts)
    out["source"] = "mediapipe_web"
    return out


def collect(session_dir: str, word_ids, correct_only_rows=None) -> list:
    """Trials of the requested words from one web session directory."""
    rows = [json.loads(l) for l in
            open(os.path.join(session_dir, "decisions.jsonl"), encoding="utf-8") if l.strip()]
    out = []
    for i, r in enumerate(rows):
        if correct_only_rows is not None and i not in correct_only_rows:
            continue
        if r.get("is_sign_attempt") is False or r.get("requested_word_id") not in word_ids:
            continue
        if r.get("observation_status") != "OK":
            continue
        p = os.path.join(session_dir, "trials", r["trial_id"] + ".npz")
        if not os.path.exists(p):
            continue
        try:
            c = load_web_clip(p)
        except Exception as exc:                      # noqa: BLE001 - recorded, not hidden
            out.append({"trial_id": r["trial_id"], "word_id": r["requested_word_id"],
                        "error": f"{type(exc).__name__}: {exc}"})
            continue
        c["trial_id"] = r["trial_id"]
        c["word_id"] = r["requested_word_id"]
        # the requested word is the exercise label, used only to score the transfer afterwards.
        # It is never fed to the model - collate() needs the field, predict() never sees it.
        c["label"] = CLASS_TO_INDEX[r["requested_word_id"]]
        c["sample_id"] = r["trial_id"]
        c["attempt_label"] = r.get("attempt_label")
        out.append(c)
    return out
