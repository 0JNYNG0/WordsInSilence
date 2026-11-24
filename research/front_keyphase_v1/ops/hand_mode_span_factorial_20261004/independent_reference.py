"""Independent reference for the two saved-observation cross conditions.

No candidate import, model construction, inference, video, or extraction at import.
"""
from pathlib import Path
import hashlib
import numpy as np

FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256((a.dtype.str + "|" + "x".join(map(str, a.shape)) + "|").encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_observation(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z["obs_"+key].copy() for key in FIELDS}


def optional_scale(value):
    finite = bool(np.isfinite(value))
    return {"value": float(value) if finite else None, "shape": [],
            "nonfinite_count": int(not finite),
            "status": "finite" if finite else "nonfinite_serialized_null_missing"}


def prepare(obs, span, mapping, build_channels, collate):
    """Official mapping/features with an explicit inclusive diagnostic crop.

    Never calls active_span, never changes observations or invents missing hands.
    """
    lo, hi = map(int, span)
    need(0 <= lo <= hi < len(obs["frame_id"]), "Explicit span outside saved observations")
    native, present = mapping(np.asarray(obs["xy_px"], dtype=np.float32), np.asarray(obs["usable"], dtype=bool))
    xy = native[lo:hi+1]
    pr = present[lo:hi+1]
    ts = np.asarray(obs["timestamp_us"], dtype=np.int64)[lo:hi+1]
    feature = build_channels(xy, pr, ts)
    x, pad, dummy = collate([dict(feature, label=0)])
    need(x.dtype == np.float32 and x.shape == (1, hi-lo+1, 280), "Cross feature shape/dtype")
    need(np.isfinite(x).all() and np.all(pad == 1) and np.all(dummy == 0), "Nonfinite features or padding/label contract")
    record = {"span": [lo, hi], "frame_id": obs["frame_id"][lo:hi+1].tolist(), "timestamp_us": ts.tolist(),
              "present_shape": list(pr.shape), "present_count": int(pr.sum()),
              "body_scale": optional_scale(feature["body_scale"]),
              "hand_scales": {key: optional_scale(value) for key, value in feature["hand_scales"].items()},
              "x_sha256": array_hash(x), "x_shape": list(x.shape), "x_dtype": str(x.dtype),
              "pad_sha256": array_hash(pad)}
    return (x, pad), record


def rank(logits, classes, label):
    logits = np.asarray(logits, dtype=np.float64)
    need(logits.shape == (93,) and np.isfinite(logits).all(), "Full93 finite logits required")
    truth = classes.index(label)
    order = np.lexsort((np.arange(93), -logits))
    rival = int(order[order != truth][0])
    return {"scope": "full93", "true_label": label, "top1": classes[int(order[0])],
            "top1_index": int(order[0]), "correct": bool(order[0] == truth),
            "true_rank": int(np.flatnonzero(order == truth)[0]) + 1,
            "best_other": classes[rival], "true_vs_best_other_margin": float(logits[truth]-logits[rival])}


def state_digest(state):
    h = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(f"|{value.dtype}|{tuple(value.shape)}|".encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def contrasts(values):
    """C-minus-B observation and long-minus-short span effects, descriptive only."""
    return {"span_long_minus_short_B": values["BB"]-values["BC"],
            "span_long_minus_short_C": values["CB"]-values["CC"],
            "observation_C_minus_B_long": values["CB"]-values["BB"],
            "observation_C_minus_B_short": values["CC"]-values["BC"],
            "interaction_span_C_minus_B": (values["CB"]-values["CC"])-(values["BB"]-values["BC"])}


def pattern(b_short_correct, c_long_correct):
    return {(True, False): "B observation correct and C wrong at both spans",
            (False, True): "Long span correct and short span wrong for both observations",
            (True, True): "Only C-short fails; joint observation/span effect",
            (False, False): "Only B-long succeeds; joint observation/span effect"}[(b_short_correct, c_long_correct)]
