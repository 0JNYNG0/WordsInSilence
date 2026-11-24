"""One registered source frame, by exact identity, for the DTW step the comparison panel selects.

On demand only. It decodes a registered clip just far enough to reach one frame, numbering frames
exactly as capture/video_source.extract did when the observation was made, and refuses rather
than substitutes: an unknown clip, a hash that disagrees, a frame the observation never kept or a
timestamp that does not match all end as an explicit error, never as a neighbouring frame.

The identity it returns links a picture to an observation row. It is not evidence that the
keypoints, the alignment or the sign are correct. No Landmarker, recogniser, VideoCapture or
camera is touched here, nothing is cached, and no filesystem path is placed in an error.
"""
from __future__ import annotations

import os
import re
from fractions import Fraction
from urllib.parse import quote

import numpy as np

from ..core.observation import Observation, sha256_file

SCHEMA = "wis_scene_evidence_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class EvidenceError(Exception):
    """An explicit refusal. reason is a fixed code; it never carries a path."""

    def __init__(self, status: int, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def _is_sha(value) -> bool:
    return isinstance(value, str) and bool(_SHA.match(value))


def source_identity(clip: dict) -> dict:
    """The registered identity of the clip a replay ACTUALLY read. Never the predicted or the
    comparison word: those are about what the input was compared with, not what it is."""
    return {"schema": SCHEMA, "clip_id": clip.get("clip_id"), "word_id": clip.get("word_id"),
            "source_sha256": clip.get("sha256"),
            "observation_sha256": clip.get("observation_sha256"),
            "selected_by": "observation_id",
            "note": "source linkage only; not a statement that keypoints or the sign are correct"}


def find_clip(clips, clip_id: str) -> dict:
    hits = [c for c in (clips or []) if isinstance(c, dict) and c.get("clip_id") == clip_id]
    if not hits:
        raise EvidenceError(404, "NO_SUCH_CLIP")
    if len(hits) > 1:
        raise EvidenceError(409, "CLIP_ID_NOT_UNIQUE")
    return hits[0]


def _open_container(path: str):
    import av                      # the PyAV already used by capture/video_source
    return av.open(path)


def decode_frame(path: str, frame_id: int, timestamp_us: int, *, opener=None) -> np.ndarray:
    """RGB pixels of the frame video_source.extract numbered frame_id, or an EvidenceError.

    Same rule as extract: frames without a PTS are skipped and NOT numbered; the origin is the
    first valid PTS; every valid-PTS frame is numbered, including a non-monotonic one that extract
    then dropped (frame_id = row + dropped_non_monotonic). A dropped frame was never observed, so
    it is refused, and so is a frame whose recomputed time differs from the observation's.
    """
    container = (opener or _open_container)(path)
    try:
        stream = container.streams.video[0]
        tb = stream.time_base or Fraction(1, 1000)
        origin_pts = None
        last_us = -1
        numbered = 0
        for packet in container.demux(stream):
            for frame in packet.decode():
                if frame.pts is None:
                    continue
                if origin_pts is None:
                    origin_pts = frame.pts
                ts_us = int(round(float((frame.pts - origin_pts) * tb) * 1_000_000))
                index, numbered = numbered, numbered + 1
                kept = ts_us > last_us
                if kept:
                    last_us = ts_us
                if index < frame_id:
                    continue
                if not kept:
                    raise EvidenceError(409, "FRAME_DROPPED_AS_NON_MONOTONIC")
                if ts_us != timestamp_us:
                    raise EvidenceError(409, "TIMESTAMP_MISMATCH")
                return frame.to_ndarray(format="rgb24")
        raise EvidenceError(404, "FRAME_NOT_DECODED")
    finally:
        container.close()


def encode_jpeg(rgb: np.ndarray) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise EvidenceError(409, "ENCODE_FAILED")
    return buf.tobytes()


def evidence_headers(clip_id: str, frame_id: int, timestamp_us: int, source_sha256: str,
                     observation_sha256: str) -> dict:
    # clip_id is percent-encoded so a non-ASCII id still fits an HTTP header; the client decodes it
    return {"X-Evidence-Schema": SCHEMA,
            "X-Evidence-Clip-Id": quote(clip_id, safe=""),
            "X-Evidence-Frame-Id": str(int(frame_id)),
            "X-Evidence-Timestamp-Us": str(int(timestamp_us)),
            "X-Evidence-Source-Sha256": source_sha256,
            "X-Evidence-Observation-Sha256": observation_sha256,
            "X-Evidence-Mirrored": "false",
            "X-Evidence-Scope": "source-linkage-only",
            "Cache-Control": "no-store"}


def _registered_files(clip: dict, root: str):
    src, npz = clip.get("absolute_path"), clip.get("observation_npz")
    if not isinstance(src, str) or not src or not isinstance(npz, str) or not npz:
        raise EvidenceError(409, "CLIP_HAS_NO_REGISTERED_FILES")
    if not os.path.isabs(src):
        src = os.path.join(root, src)
    return src, os.path.join(root, npz)


def reference_frame(clips, root: str, clip_id, frame_id, source_sha256, observation_sha256, *,
                    opener=None, encode=None) -> dict:
    """{"jpeg": bytes, "headers": {...}, "timestamp_us": int} or EvidenceError."""
    if not isinstance(clip_id, str) or not clip_id:
        raise EvidenceError(400, "INVALID_CLIP_ID")
    if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
        raise EvidenceError(400, "INVALID_FRAME_ID")
    frame_id = int(frame_id)
    if not _is_sha(source_sha256) or not _is_sha(observation_sha256):
        raise EvidenceError(400, "INVALID_HASH")
    clip = find_clip(clips, clip_id)
    reg_src, reg_obs = clip.get("sha256"), clip.get("observation_sha256")
    if not _is_sha(reg_src) or not _is_sha(reg_obs):
        raise EvidenceError(409, "CLIP_HAS_NO_REGISTERED_HASHES")
    if reg_src != source_sha256 or reg_obs != observation_sha256:
        raise EvidenceError(409, "REQUEST_HASH_DIFFERS_FROM_REGISTERED")
    src_path, npz_path = _registered_files(clip, root)
    if not os.path.isfile(src_path) or not os.path.isfile(npz_path):
        raise EvidenceError(404, "REGISTERED_FILE_MISSING")
    if sha256_file(src_path) != reg_src:
        raise EvidenceError(409, "SOURCE_HASH_MISMATCH")
    if sha256_file(npz_path) != reg_obs:
        raise EvidenceError(409, "OBSERVATION_HASH_MISMATCH")
    try:
        obs = Observation.load(npz_path)
    except Exception:                                     # noqa: BLE001 - refused, not raised
        raise EvidenceError(409, "OBSERVATION_UNREADABLE") from None
    side = obs.sidecar or {}
    if not side:
        raise EvidenceError(409, "OBSERVATION_SIDECAR_MISSING")
    if (side.get("observation_sha256") != reg_obs or side.get("clip_id") != clip_id
            or side.get("source_sha256") != reg_src or side.get("mirrored") is not False):
        raise EvidenceError(409, "OBSERVATION_SIDECAR_MISMATCH")
    ids, times = np.asarray(obs.frame_id), np.asarray(obs.timestamp_us)
    if ids.ndim != 1 or times.shape != ids.shape:
        raise EvidenceError(409, "OBSERVATION_UNREADABLE")
    if ids.size and (ids < 0).any():
        raise EvidenceError(409, "INVALID_REGISTERED_FRAME_IDS")
    if np.unique(ids).size != ids.size:
        raise EvidenceError(409, "DUPLICATE_REGISTERED_FRAME_IDS")
    hit = np.flatnonzero(ids == frame_id)
    if hit.size != 1:
        # bounded by the observation: a frame it never kept is not decoded at all
        raise EvidenceError(404, "FRAME_NOT_IN_OBSERVATION")
    timestamp_us = int(times[hit[0]])
    if timestamp_us < 0:
        raise EvidenceError(409, "INVALID_REGISTERED_TIMESTAMP")
    try:
        rgb = decode_frame(src_path, frame_id, timestamp_us, opener=opener)
        jpeg = (encode or encode_jpeg)(rgb)
    except EvidenceError:
        raise
    except Exception:                                     # noqa: BLE001 - refused, not raised
        raise EvidenceError(409, "DECODE_FAILED") from None
    return {"jpeg": jpeg, "timestamp_us": timestamp_us,
            "headers": evidence_headers(clip_id, frame_id, timestamp_us, reg_src, reg_obs)}
