"""Observation side input for a FUTURE temporal attribute head, not a trained head.

No labels are produced by this adapter. Native observed/usable/finite point
availability is sensor evidence, not target availability or verified handshape.
Anatomical L/R slots are retained; their identities are not independently proven.
Stored SHuBERT outputs already mix copied RGB into temporal context. Appending
these indicators does NOT repair that contamination or make training ready.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

UNIT = Path(__file__).resolve().parent
PUBLIC_INPUT_ROOT = UNIT.parent / "word100_shubert_feasibility_20261028" / "inputs"
HAND_OFFSETS = (33, 54)
BODY_IDS = (0, 11, 12, 13, 14, 15, 16)
BODY_REQUIRED = (0, 2, 11, 12, 13, 14, 15, 16)
SOURCE_NONE, SOURCE_CURRENT, SOURCE_PREVIOUS = 0, 1, 2
HAND_ORDER = ["anatomical_left", "anatomical_right"]
OBSERVATION_FEATURE_DIM = 108
LIMITATIONS = (
    "Sensor masks are not semantic labels or independent RGB annotation.",
    "Saved anatomical left/right slots are retained without verifying identity.",
    "Stored SHuBERT temporal features may already contain stale RGB context; "
    "these side inputs neither repair the backbone nor establish training readiness.",
    "Finger geometry availability is not verified handshape correctness.",
)


def _array(mapping, key, shape, kind):
    if key not in mapping:
        raise ValueError(f"Missing {key}")
    value = np.asarray(mapping[key])
    if value.shape != shape:
        raise ValueError(f"{key}: shape {value.shape}, expected {shape}")
    if kind == "bool":
        valid_dtype = value.dtype == np.dtype(bool)
    elif kind == "int":
        valid_dtype = np.issubdtype(value.dtype, np.integer)
    else:
        valid_dtype = np.issubdtype(value.dtype, np.floating)
    if not valid_dtype:
        raise ValueError(f"{key}: expected {kind} dtype, got {value.dtype}")
    return value


def _source_origins(kind, origin, timestamp_us, name):
    """Validate exact saved previous-crop policy, including never-observed -1."""
    if not np.isin(kind, (0, 1, 2)).all():
        raise ValueError(f"{name}: unknown source_kind")
    n = len(timestamp_us)
    if kind.shape != origin.shape:
        raise ValueError(f"{name}: origin shape mismatch")
    if kind.ndim == 1:
        kind, origin = kind[:, None], origin[:, None]
    index = np.broadcast_to(np.arange(n)[:, None], kind.shape)
    current = kind == SOURCE_CURRENT
    last_current = np.maximum.accumulate(np.where(current, index, -1), axis=0)
    expected_kind = np.where(current, 1, np.where(last_current >= 0, 2, 0))
    if not np.array_equal(kind, expected_kind):
        raise ValueError(f"{name}: blank/previous state disagrees with saved fallback policy")
    if not np.array_equal(origin, last_current):
        raise ValueError(f"{name}: origin is not the current/latest observed index or -1")
    has_origin = origin >= 0
    origin_time = np.full(origin.shape, -1, dtype=np.int64)
    origin_time[has_origin] = timestamp_us[origin[has_origin]]
    age = np.full(origin.shape, np.nan, dtype=np.float64)
    now = np.broadcast_to(timestamp_us[:, None], origin.shape)
    age[has_origin] = (now[has_origin] - origin_time[has_origin]) / 1_000_000.0
    return origin_time, age


def build_observation_input(native, provenance, image_size):
    """Validate one UNPADDED saved clip and construct per-hand numeric inputs.

    source_kind means RGB crop status; native point validity is kept separately.
    A partial hand can have current wrist/finger points while its RGB crop is a
    previous copy. Invalid coordinates become zero only in conjunction with
    explicit point masks. Unknown elapsed time remains NaN in the semantic
    record, and becomes (0, known=False) in the finite numeric model input.
    Time units are seconds; coordinates are full-image normalized x,y.
    """
    time = np.asarray(native.get("timestamp_us"))
    if time.ndim != 1 or len(time) == 0:
        raise ValueError("timestamp_us must be a nonempty one-dimensional sequence")
    n = len(time)
    time = _array(native, "timestamp_us", (n,), "int")
    frame_id = _array(native, "frame_id", (n,), "int")
    if not np.array_equal(frame_id, np.arange(n)):
        raise ValueError("Saved full-sequence frame_id must be contiguous from zero")
    if not np.issubdtype(time.dtype, np.signedinteger):
        raise ValueError("timestamp_us must use signed integer microseconds")
    if np.any(time < 0) or np.any(np.diff(time) <= 0):
        raise ValueError("timestamp_us must be nonnegative and strictly increasing")
    xy = _array(native, "xy_px", (n, 75, 2), "float")
    observed = _array(native, "observed", (n, 75), "bool")
    usable = _array(native, "usable", (n, 75), "bool")
    for key, expected in (("timestamp_us", time), ("frame_id", frame_id)):
        found = _array(provenance, key, (n,), "int")
        if not np.array_equal(found, expected):
            raise ValueError(f"Native/provenance {key} mismatch")
    decode_time = _array(provenance, "decode_pts_us", (n,), "int")
    if np.any(np.abs(decode_time.astype(np.float64) - time) > 1):
        raise ValueError("Decode/native timestamps differ by more than 1 microsecond")
    size = np.asarray(image_size, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("image_size must contain positive finite width,height")
    kinds = _array(provenance, "source_kind", (n, 3), "int")
    origins = _array(provenance, "origin_frame_index", (n, 3), "int")
    origin_time, age = _source_origins(kinds, origins, time, "RGB")
    point_valid = observed & usable & np.isfinite(xy).all(axis=-1)
    hand_valid = np.stack([point_valid[:, o:o + 21] for o in HAND_OFFSETS], axis=1)
    hand_record = _array(provenance, "hand_point_valid", (n, 2, 21), "bool")
    if not np.array_equal(hand_valid, hand_record):
        raise ValueError("Saved hand_point_valid disagrees with native masks/coordinates")
    if not np.array_equal(kinds[:, :2] == SOURCE_CURRENT, hand_valid.all(axis=-1)):
        raise ValueError("Current RGB crops must agree with all-21-point saved gate")
    body_valid = point_valid[:, BODY_IDS]
    if not np.array_equal(_array(provenance, "body_point_valid", (n, 7), "bool"), body_valid):
        raise ValueError("Saved body_point_valid disagrees with native masks/coordinates")
    body_kind = _array(provenance, "body_source_kind", (n,), "int")
    body_origin = _array(provenance, "body_origin_frame_index", (n,), "int")
    body_origin_time, body_age = _source_origins(body_kind, body_origin, time, "Body14")
    body_age = body_age[:, 0]
    body_age_known = np.isfinite(body_age)
    face_age = age[:, 2]
    face_age_known = np.isfinite(face_age)
    if not np.array_equal(body_kind == SOURCE_CURRENT, point_valid[:, BODY_REQUIRED].all(axis=-1)):
        raise ValueError("Body14 current source disagrees with saved required-point gate")
    # Copy current coordinates only. Previous RGB never creates observed joints.
    finite_xy = np.where(point_valid[..., None], xy, 0) / size
    hand_xy = np.stack([finite_xy[:, o:o + 21] for o in HAND_OFFSETS], axis=1).astype(np.float32)
    finger_valid = hand_valid[:, :, 1:].reshape(n, 2, 5, 4).all(axis=-1) & hand_valid[:, :, :1]
    dt = np.zeros(n, dtype=np.float64)
    dt[1:] = np.diff(time) / 1_000_000.0
    relative = (time - time[0]) / 1_000_000.0
    hand_age = age[:, :2].copy()
    age_known = np.isfinite(hand_age)
    body_xy = finite_xy[:, BODY_IDS].astype(np.float32)
    fields = [
        ("rgb_source_one_hot_none_current_previous", np.eye(3)[kinds[:, :2]]),
        ("rgb_observation_age_s", np.where(age_known, hand_age, 0)[..., None]),
        ("rgb_observation_age_known", age_known[..., None]),
        ("relative_time_and_delta_s", np.broadcast_to(np.stack((relative, dt), axis=-1)[:, None], (n, 2, 2))),
        ("current_hand_xy_normalized", hand_xy.reshape(n, 2, 42)),
        ("current_hand_point_valid", hand_valid),
        ("finger_geometry_valid_thumb_index_middle_ring_little", finger_valid),
        ("current_body_xy_normalized", np.broadcast_to(body_xy.reshape(n, 1, 14), (n, 2, 14))),
        ("current_body_point_valid", np.broadcast_to(body_valid[:, None], (n, 2, 7))),
        ("current_hand_wrist_valid", hand_valid[:, :, :1]),
        ("current_pose_wrist_valid", point_valid[:, (15, 16), None]),
        ("face_rgb_source_one_hot_none_current_previous", np.broadcast_to(np.eye(3)[kinds[:, 2]][:, None], (n, 2, 3))),
        ("face_rgb_observation_age_s", np.broadcast_to(np.where(face_age_known, face_age, 0)[:, None, None], (n, 2, 1))),
        ("face_rgb_observation_age_known", np.broadcast_to(face_age_known[:, None, None], (n, 2, 1))),
        ("body14_source_one_hot_none_current_previous", np.broadcast_to(np.eye(3)[body_kind][:, None], (n, 2, 3))),
        ("body14_observation_age_s", np.broadcast_to(np.where(body_age_known, body_age, 0)[:, None, None], (n, 2, 1))),
        ("body14_observation_age_known", np.broadcast_to(body_age_known[:, None, None], (n, 2, 1))),
    ]
    columns, offset = [], 0
    for name, values in fields:
        columns.append({"name": name, "start": offset, "stop": offset + values.shape[-1]})
        offset += values.shape[-1]
    numeric = np.concatenate([value for _, value in fields], axis=-1).astype(np.float32)
    if numeric.shape != (n, 2, OBSERVATION_FEATURE_DIM) or not np.isfinite(numeric).all():
        raise ValueError("Numeric side input must be finite")
    return {
        "schema": "wis_observation_head_side_input_v2",
        "hand_order": HAND_ORDER.copy(),
        "source_kind_legend": {0: "none", 1: "current", 2: "previous_copy_not_current_observation"},
        "identity_status": "saved_slot_assignment_not_independently_verified",
        "frame_id": frame_id.copy(), "timestamp_us": time.copy(),
        "frame_delta_s": dt, "relative_time_s": relative,
        "hand_source_kind": kinds[:, :2].copy(),
        "hand_origin_frame_index": origins[:, :2].copy(),
        "hand_origin_timestamp_us": origin_time[:, :2].copy(),
        "hand_rgb_age_s": hand_age, "hand_rgb_age_known": age_known,
        "hand_rgb_current": (kinds[:, :2] == SOURCE_CURRENT),
        "hand_rgb_available": (kinds[:, :2] != SOURCE_NONE),
        "hand_point_observed": np.stack([observed[:, o:o + 21] for o in HAND_OFFSETS], axis=1),
        "hand_point_usable": np.stack([usable[:, o:o + 21] for o in HAND_OFFSETS], axis=1),
        "hand_point_valid": hand_valid, "hand_xy_normalized": hand_xy,
        "finger_geometry_valid": finger_valid,
        "hand_wrist_valid": hand_valid[:, :, 0],
        "pose_wrist_valid": point_valid[:, (15, 16)],
        "body_point_valid": body_valid, "body_xy_normalized": body_xy,
        "body14_source_kind": body_kind.copy(),
        "body14_origin_frame_index": body_origin.copy(),
        "body14_origin_timestamp_us": body_origin_time[:, 0].copy(),
        "body14_age_s": body_age.copy(), "body14_age_known": body_age_known,
        "face_source_kind": kinds[:, 2].copy(),
        "face_origin_frame_index": origins[:, 2].copy(),
        "face_origin_timestamp_us": origin_time[:, 2].copy(),
        "face_rgb_age_s": face_age.copy(), "face_rgb_age_known": face_age_known,
        "sequence_padding_mask": np.zeros(n, dtype=bool),
        "observation_features": numeric, "feature_columns": columns,
        "targets": None, "target_availability": None,
        "limitations": list(LIMITATIONS),
    }


def append_saved_context(observation, saved_context_features):
    """Join existing frozen features without inference or any decontamination claim."""
    context = np.asarray(saved_context_features)
    numeric = observation["observation_features"]
    if context.ndim != 2 or context.shape[0] != numeric.shape[0] or context.shape[1] == 0:
        raise ValueError("Saved context must be [same T, positive D]")
    if not np.issubdtype(context.dtype, np.floating) or not np.isfinite(context).all():
        raise ValueError("Saved context must be finite floating-point features")
    repeated = np.broadcast_to(context[:, None], (context.shape[0], 2, context.shape[1]))
    return np.concatenate((numeric, repeated), axis=-1)


def pad_observation_batch(clips):
    """True sequence_padding_mask means beyond clip length, never a missing hand."""
    if not clips:
        raise ValueError("At least one clip is required")
    lengths = []
    for clip in clips:
        ids = np.asarray(clip.get("frame_id"))
        if ids.ndim != 1 or not len(ids):
            raise ValueError("Batch frame_id must be nonempty one-dimensional integers")
        length = len(ids)
        ids = _array(clip, "frame_id", (length,), "int")
        if not np.array_equal(ids, np.arange(length)):
            raise ValueError("Batch frame_id must be contiguous and ordered from zero")
        if clip.get("schema") != "wis_observation_head_side_input_v2" or clip.get("hand_order") != HAND_ORDER:
            raise ValueError("Batch requires v2 schema with anatomical left/right hand order")
        time = _array(clip, "timestamp_us", (length,), "int")
        if not np.issubdtype(time.dtype, np.signedinteger) or np.any(time < 0) or np.any(np.diff(time) <= 0):
            raise ValueError("Batch timestamps must be signed, nonnegative, and strictly increasing")
        values = _array(clip, "observation_features", (length, 2, OBSERVATION_FEATURE_DIM), "float")
        if values.dtype != np.dtype(np.float32) or not np.isfinite(values).all():
            raise ValueError("Batch observation_features must be finite float32")
        kinds = _array(clip, "hand_source_kind", (length, 2), "int")
        if not np.isin(kinds, (0, 1, 2)).all():
            raise ValueError("Batch has unknown hand source_kind")
        _array(clip, "hand_point_valid", (length, 2, 21), "bool")
        pad = _array(clip, "sequence_padding_mask", (length,), "bool")
        if pad.any():
            raise ValueError("Batch accepts unpadded clips only")
        if not clip.get("feature_columns") or clip["feature_columns"] != clips[0].get("feature_columns"):
            raise ValueError("All clips must have identical feature columns")
        lengths.append(length)
    lengths = np.asarray(lengths, dtype=np.int64)
    width = OBSERVATION_FEATURE_DIM
    n, t = len(clips), int(lengths.max())
    features = np.zeros((n, t, 2, width), dtype=np.float32)
    padding = np.ones((n, t), dtype=bool)
    source = np.zeros((n, t, 2), dtype=np.uint8)
    valid = np.zeros((n, t, 2, 21), dtype=bool)
    for i, (clip, length) in enumerate(zip(clips, lengths)):
        features[i, :length] = clip["observation_features"]
        padding[i, :length] = False
        source[i, :length] = clip["hand_source_kind"]
        valid[i, :length] = clip["hand_point_valid"]
    return {"observation_features": features, "sequence_padding_mask": padding,
            "hand_source_kind": source, "hand_point_valid": valid, "lengths": lengths,
            "hand_missing_rgb_mask": (source == SOURCE_NONE) & ~padding[..., None]}


def partial_label_cross_entropy(logits, targets, target_confirmed, sequence_padding_mask, reduction="mean"):
    """Item-masked CE; sensor availability is deliberately NOT a loss gate.

    logits [B,T,2,K,C]; target class and independent confirmed-label mask [B,T,2,K].
    Confirmed RGB labels can supervise missing sensor inputs. Unconfirmed items
    and sequence padding get exactly zero gradient; other hands/items survive.
    Only selected logits are evaluated, so NaN/invalid ignored entries cannot
    contaminate loss. Synthetic labels used in tests are never public labels.
    """
    import torch
    import torch.nn.functional as F
    if logits.ndim != 5 or logits.shape[2] != 2 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [B,T,2,K,C], C>=2")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating-point")
    if targets.shape != logits.shape[:-1] or target_confirmed.shape != targets.shape:
        raise ValueError("Targets and target_confirmed must match [B,T,2,K]")
    if targets.dtype != torch.long or target_confirmed.dtype != torch.bool:
        raise ValueError("Targets must be int64; target_confirmed must be bool")
    if sequence_padding_mask.shape != logits.shape[:2] or sequence_padding_mask.dtype != torch.bool:
        raise ValueError("sequence_padding_mask must be bool[B,T]")
    if any(x.device != logits.device for x in (targets, target_confirmed, sequence_padding_mask)):
        raise ValueError("All loss tensors must use the same device")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError("reduction must be mean, sum, or none")
    selected = target_confirmed & ~sequence_padding_mask[:, :, None, None]
    selected_targets = targets[selected]
    if torch.any((selected_targets < 0) | (selected_targets >= logits.shape[-1])).item():
        raise ValueError("Confirmed, nonpadding targets must be valid class indices")
    selected_logits = logits[selected]
    if not torch.isfinite(selected_logits).all().item():
        raise ValueError("Confirmed logits must be finite")
    count = int(selected_targets.numel())
    # Empty indexed sum remains connected to logits, producing zero gradients.
    losses = F.cross_entropy(selected_logits, selected_targets, reduction="none") if count else selected_logits.sum(dim=-1)
    per_item = torch.zeros_like(targets, dtype=logits.dtype).masked_scatter(selected, losses)
    loss = per_item if reduction == "none" else losses.sum()
    if reduction == "mean" and count:
        loss = loss / count
    return {"loss": loss, "per_item_loss": per_item, "effective_target_mask": selected,
            "supervised_items": count}


def load_public_saved_input(alias):
    """Read only the fixed public12 saved arrays; verify their existing DONE pins."""
    if alias not in {f"V{i:02d}" for i in range(1, 13)}:
        raise ValueError("Only fixed public aliases V01..V12 are accepted")
    directory = PUBLIC_INPUT_ROOT / alias
    done_path = directory / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8-sig"))
    manifest = json.loads((PUBLIC_INPUT_ROOT / "MANIFEST.json").read_text(encoding="utf-8-sig"))
    row = next(r for r in manifest["rows"] if r["alias"] == alias)
    if hashlib.sha256(done_path.read_bytes()).hexdigest() != row["done_sha256"]:
        raise ValueError(f"{alias}: DONE hash mismatch")
    inputs = {}
    for filename in ("native_observation.npz", "provenance.npz"):
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != done["output_pins"][str(path)]:
            raise ValueError(f"{alias}: {filename} hash mismatch")
        with np.load(path, allow_pickle=False) as saved:
            inputs[filename] = {key: saved[key].copy() for key in saved.files}
    result = build_observation_input(inputs["native_observation.npz"], inputs["provenance.npz"], done["image_size"])
    if len(result["frame_id"]) != done["frames"]:
        raise ValueError(f"{alias}: frame count mismatch")
    result["public_alias"] = alias
    return result


def join_public_saved_context(observation):
    """Pin-check existing public FFN and provenance, then join without inference."""
    alias = observation.get("public_alias")
    if alias not in {f"V{i:02d}" for i in range(1, 13)}:
        raise ValueError("Only fixed public aliases V01..V12 are accepted")
    directory = PUBLIC_INPUT_ROOT.parent / "features" / alias
    done = json.loads((directory / "DONE.json").read_text(encoding="utf-8-sig"))
    records = json.loads((PUBLIC_INPUT_ROOT.parent / "PUBLIC_FEATURE_RESULT.json").read_text(encoding="utf-8-sig"))
    recorded = next(row for row in records["rows"] if row["alias"] == alias)
    if done["output_pins"] != recorded["output_pins"]:
        raise ValueError(f"{alias}: feature DONE/result pin mismatch")
    for filename in ("provenance.npz", "last_ffn.npy"):
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != done["output_pins"][filename]:
            raise ValueError(f"{alias}: stored feature {filename} hash mismatch")
    with np.load(directory / "provenance.npz", allow_pickle=False) as saved:
        for key in ("frame_id", "timestamp_us"):
            if not np.array_equal(saved[key], observation[key]):
                raise ValueError(f"{alias}: feature/input {key} mismatch")
        for key, observed_key in (("source_kind", "hand_source_kind"),
                                  ("origin_frame_index", "hand_origin_frame_index")):
            if not np.array_equal(saved[key][:, :2], observation[observed_key]):
                raise ValueError(f"{alias}: feature/input {key} mismatch")
        for key, observed_key, column in (("source_kind", "face_source_kind", 2),
                                          ("origin_frame_index", "face_origin_frame_index", 2),
                                          ("body_source_kind", "body14_source_kind", None),
                                          ("body_origin_frame_index", "body14_origin_frame_index", None)):
            value = saved[key] if column is None else saved[key][:, column]
            if not np.array_equal(value, observation[observed_key]):
                raise ValueError(f"{alias}: feature/input {observed_key} mismatch")
    context = np.load(directory / "last_ffn.npy", allow_pickle=False)
    if context.shape != (len(observation["frame_id"]), 768):
        raise ValueError(f"{alias}: expected saved [T,768] SHuBERT FFN")
    return append_saved_context(observation, context)
