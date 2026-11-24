"""Clip decoding that preserves real PTS. VFR is never replaced by frame_index/fps."""
from __future__ import annotations

import os
from fractions import Fraction

import av
import numpy as np

from ..core.observation import Observation, empty_frames, sha256_file
from .landmarker import Landmarker


def probe(path: str) -> dict:
    """Container/stream facts only. No MediaPipe, no full decode of pixel data."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        n = 0
        first_pts = None
        last_pts = None
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                if first_pts is None:
                    first_pts = frame.pts
                last_pts = frame.pts
                n += 1
        duration_ms = (float((last_pts - first_pts) * tb) * 1000.0) if n > 1 else 0.0
        return {
            "frame_count": n,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "time_base": f"{tb.numerator}/{tb.denominator}",
            "duration_ms": round(duration_ms, 3),
            "avg_rate": (f"{stream.average_rate.numerator}/{stream.average_rate.denominator}"
                         if stream.average_rate else None),
            "codec": stream.codec_context.name,
            "pix_fmt": stream.codec_context.pix_fmt,
        }


def extract(path: str, lm: Landmarker, clip_id: str) -> Observation:
    """Decode the clip and produce one wis_observation_2d_v1."""
    meta = probe(path)
    out = empty_frames(meta["frame_count"])
    dropped_non_monotonic = 0
    row = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        origin_pts = None
        last_us = -1
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                if origin_pts is None:
                    origin_pts = frame.pts
                ts_us = int(round(float((frame.pts - origin_pts) * tb) * 1_000_000))
                if ts_us <= last_us:
                    dropped_non_monotonic += 1
                    continue
                last_us = ts_us
                out["timestamp_us"][row] = ts_us
                out["frame_id"][row] = row + dropped_non_monotonic
                lm.process(frame.to_ndarray(format="rgb24"), ts_us, out, row)
                row += 1

    for key in out:
        out[key] = out[key][:row]

    obs = Observation(**out)
    obs.sidecar = {
        "schema": "wis_observation_2d_v1",
        "origin": "clip",
        "clip_id": clip_id,
        "source_path": path.replace("\\", "/"),
        "source_sha256": sha256_file(path),
        "width": meta["width"],
        "height": meta["height"],
        "time_base": meta["time_base"],
        "decoded_frames": row,
        "dropped_non_monotonic_pts": dropped_non_monotonic,
        "mirrored": False,
        "rotation_deg": 0,
        "extractor_id": lm.extractor_id(),
        "hand_assignment_policy": lm.contract["hand_assignment"]["policy"],
        "handedness_label_disagreements": lm.handedness_disagreements,
        "handedness_label_comparisons": lm.handedness_comparisons,
        "unassigned_hand_detections": lm.unassigned_hands,
        "clock": "video PTS * time_base, microseconds from first decoded frame",
    }
    obs.sidecar["quality"] = obs.quality_summary()
    return obs
