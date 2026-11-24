"""Research-only detector/model-view separation; no detector or model execution.

The detector/side-assignment observation is read-only. The ROI is in original
source pixels, while source_origin maps a cropped detector image to that space.
Only retained points are translated; absent points remain explicitly absent.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.dont_write_bytecode = True
_UNIT = Path(__file__).resolve().parent
_LAB = _UNIT.parents[1]
for _path in (_LAB / "src", _LAB.parents[1] / "research/sign_model_pilot_v1/src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wis_lab.core.observation import HAND_SLICES, Observation
from pilot.data import build_channels
from pilot.web_input import to_native_like


_SAFE_PROVENANCE = (
    "schema", "extractor_id", "candidate_code_sha256", "hand_assignment_policy",
    "source", "source_id", "clip_id", "recording_id", "fps", "frame_width",
    "frame_height", "video_width", "video_height", "width", "height",
)
_MASK_SCHEMA = "source_pixel_pointwise_roi_v1"


def _geometry(box, source_origin):
    box_array = np.asarray(box, dtype=np.float64)
    origin_array = np.asarray(source_origin, dtype=np.float64)
    if box_array.shape != (4,) or not np.isfinite(box_array).all():
        raise ValueError("box must contain four finite source-pixel coordinates")
    if origin_array.shape != (2,) or not np.isfinite(origin_array).all():
        raise ValueError("source_origin must contain two finite coordinates")
    if not (box_array[2] > box_array[0] and box_array[3] > box_array[1]):
        raise ValueError("box must have positive width and height")
    return box_array, origin_array


def _inside(xy, box):
    x0, y0, x1, y1 = box
    return (np.isfinite(xy).all(axis=-1)
            & (xy[..., 0] >= x0) & (xy[..., 0] < x1)
            & (xy[..., 1] >= y0) & (xy[..., 1] < y1))


def apply_roi(obs: Observation, box, source_origin=(0, 0)) -> Observation:
    """Return an independent, pointwise-masked Observation in source pixels.

    observed = old observed AND finite source coordinate AND strict ROI bounds.
    usable = old usable AND new observed. No confidence threshold is changed.
    All 75 points are treated alike; partial hands are not silently completed.
    Frame IDs/timestamps are copied exactly and no temporal trimming occurs.
    """
    obs.validate()
    box_array, origin_array = _geometry(box, source_origin)
    source_xy = obs.xy_px.astype(np.float64) + origin_array
    observed = obs.observed & _inside(source_xy, box_array)
    usable = obs.usable & observed
    xy = source_xy.astype(np.float32)
    xy[~observed] = np.nan
    conf = obs.conf.copy()
    conf[~observed] = np.nan
    hand_score = obs.hand_score.copy()
    for side, hand_slice in enumerate(HAND_SLICES):
        hand_score[~observed[:, hand_slice].any(axis=1), side] = np.nan

    # Do not carry raw arrays, detector quality aggregates, original observation
    # hashes, or normalisation statistics through a model-view sidecar.
    sidecar = {key: obs.sidecar[key] for key in _SAFE_PROVENANCE
               if key in obs.sidecar and isinstance(obs.sidecar[key], (str, int, float, bool, type(None)))}
    sidecar["roi_mask"] = {
        "schema": _MASK_SCHEMA,
        "box_source_pixels": box_array.tolist(),
        "source_origin": origin_array.tolist(),
        "coordinate_space": "source_pixels",
        "coordinate_transform": "input_xy_plus_source_origin_only",
        "boundary_rule": "x0<=x<x1 and y0<=y<y1",
        "stage": "after_detector_and_assignment_before_normalisation_and_features",
        "frame_policy": "all_frames_and_original_timestamps_preserved",
        "outside_policy": "observed_false_usable_false_xy_conf_nan",
    }
    out = Observation(
        xy_px=xy, timestamp_us=obs.timestamp_us.copy(), frame_id=obs.frame_id.copy(),
        observed=observed, usable=usable, conf=conf, hand_score=hand_score, sidecar=sidecar,
    )
    out.validate()
    return out


def build_roi_channels(obs: Observation) -> dict:
    """Prepare the unchanged 280-channel contract from an already masked view.

    No model is loaded or called. Body/hand scales are computed only after the
    pointwise ROI mask. Missing wrists invalidate their local hand channels;
    remaining usable hand points can still have body-relative channels. The
    existing explicit error for no usable shoulder scale is preserved.
    """
    obs.validate()
    metadata = obs.sidecar.get("roi_mask", {})
    if metadata.get("schema") != _MASK_SCHEMA or metadata.get("coordinate_space") != "source_pixels":
        raise ValueError("build_roi_channels requires the output of apply_roi")
    box, _ = _geometry(metadata["box_source_pixels"], (0, 0))
    if (obs.observed & ~_inside(obs.xy_px, box)).any():
        raise ValueError("masked observation contains an observed point outside its ROI")
    native_xy, present = to_native_like(obs.xy_px, obs.usable)
    result = build_channels(native_xy, present, obs.timestamp_us)
    result["timestamp_us"] = obs.timestamp_us.copy()
    result["frame_id"] = obs.frame_id.copy()
    result["source"] = "research_roi_masked_observation"
    return result
