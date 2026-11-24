"""Independent verification core, not a candidate implementation.

Import is read-only and performs no model/MediaPipe/video calls. Official mapping,
span and channel functions are loaded only by ReferenceScorer after parent approval.
The result-schema adapter will call this module after the main runner has finished.
"""
from pathlib import Path
import hashlib
import sys
import numpy as np

FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    header = value.dtype.str + "|" + "x".join(map(str, value.shape)) + "|"
    return hashlib.sha256(header.encode("ascii") + value.tobytes()).hexdigest()


def byte_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def state_digest(state):
    """Documented name/dtype/shape/byte encoding; no candidate helper import."""
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(f"|{tensor.dtype}|{tuple(tensor.shape)}|".encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def read_observation(path, expected_sha=None):
    if expected_sha is not None:
        require(sha256(path) == expected_sha, "Observation hash changed")
    with np.load(path, allow_pickle=False) as z:
        prefix = "obs_" if "obs_xy_px" in z else ""
        obs = {k: z[prefix + k].copy() for k in FIELDS}
    return obs


def validate_observation(obs, n):
    expected = {"xy_px": (np.dtype("float32"), (n, 75, 2)),
                "timestamp_us": (np.dtype("int64"), (n,)), "frame_id": (np.dtype("int64"), (n,)),
                "observed": (np.dtype("bool"), (n, 75)), "usable": (np.dtype("bool"), (n, 75)),
                "conf": (np.dtype("float32"), (n, 75)), "hand_score": (np.dtype("float32"), (n, 2))}
    for k, (dtype, shape) in expected.items():
        require(obs[k].dtype == dtype and obs[k].shape == shape, "Observation schema: " + k)
    require(np.array_equal(obs["frame_id"], np.arange(n)), "Not all original frame IDs")
    require(obs["timestamp_us"][0] == 0 and (np.diff(obs["timestamp_us"]) > 0).all(), "Original PTS invalid")
    require(not np.any(obs["usable"] & ~obs["observed"]), "Usable outside observed")
    require(np.isfinite(obs["xy_px"][obs["usable"]]).all(), "Usable coordinates non-finite")
    require(not np.isfinite(obs["xy_px"][~obs["observed"]]).any(), "Absent points contain finite values")


def compare_raw(b, c, cache):
    require(all(byte_equal(b[k], cache[k]) for k in FIELDS), "Extracted B differs from baseline cache")
    for k in ("xy_px", "observed", "usable", "conf"):
        require(byte_equal(b[k][:, :33], c[k][:, :33]), "B/C pose differs: " + k)
    for k in ("timestamp_us", "frame_id"):
        require(byte_equal(b[k], c[k]), "B/C timing differs: " + k)


class ReferenceScorer:
    """One fixed 832 model, at most 36 explicit forwards; never a training call."""

    def __init__(self, config):
        import torch
        sys.path.insert(0, str(Path(config["workspace_root"]) / "research/sign_model_pilot_v1/src"))
        from pilot.web_input import to_native_like, active_span
        from pilot.data import build_channels, collate
        from pilot.model_masked_v1 import MaskedSignTCN
        self.torch = torch
        self.mapping, self.span = to_native_like, active_span
        self.channels, self.collate = build_channels, collate
        ck = config["scoring"]["checkpoint_path"]
        require(sha256(ck) == config["scoring"]["checkpoint_sha256"], "Checkpoint pin")
        payload = torch.load(ck, map_location="cpu", weights_only=True)
        self.classes = list(payload["classes"])
        require(len(self.classes) == len(set(self.classes)) == payload["n_classes"] == 93, "Class contract")
        torch.set_num_threads(config["scoring"]["cpu_threads"])
        self.model = MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                                  kernel_size=payload["kernel_size"], dropout=payload["dropout"], n_classes=93)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.before = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.payload_state_sha256 = state_digest(payload["state_dict"])
        require(state_digest(self.before) == self.payload_state_sha256, "Loaded model is not checkpoint payload")
        self.forwards = 0

    def prepare(self, obs):
        native, present = self.mapping(np.asarray(obs["xy_px"], dtype=np.float32),
                                       np.asarray(obs["usable"], dtype=bool))
        span = self.span(present)
        if span is None:
            return None, {"status": "UNAVAILABLE_NO_ACTIVE_HANDS", "span": None}
        lo, hi = span
        timestamps = np.asarray(obs["timestamp_us"], dtype=np.int64)[lo:hi+1]
        feat = self.channels(native[lo:hi+1], present[lo:hi+1], timestamps)
        x, pad, labels = self.collate([dict(feat, label=0)])
        require(np.isfinite(x).all() and np.all(pad == 1) and np.all(labels == 0), "Prepared input invalid")
        scales = {k: float(v) if np.isfinite(v) else None for k, v in feat["hand_scales"].items()}
        detail = {"status": "AVAILABLE", "span": [int(lo), int(hi)],
                  "frame_ids": obs["frame_id"][lo:hi+1].tolist(), "timestamp_us": timestamps.tolist(),
                  "observed_hand_frames": int(present[lo:hi+1, 25:67].any(axis=1).sum()),
                  "x_sha256": array_hash(x), "x_shape": list(x.shape), "x_dtype": str(x.dtype),
                  "frames": int(feat["frames"]), "body_scale": float(feat["body_scale"]),
                  "hand_scales": scales, "body_valid_sum": float(feat["body_valid"].sum()),
                  "hand_valid_sum": float(feat["hand_valid"].sum())}
        return (x, pad), detail

    def score(self, prepared, label):
        require(prepared is not None, "No forward permitted for unavailable input")
        require(self.forwards < 36, "Independent forward budget exceeded")
        x, pad = prepared
        with self.torch.inference_mode():
            logits = self.model(self.torch.from_numpy(x), self.torch.from_numpy(pad))[0].numpy().astype(np.float64)
        self.forwards += 1
        require(logits.shape == (93,) and np.isfinite(logits).all(), "Non-finite or wrong-size scores")
        # Labels affect only score reporting after the forward above.
        truth = self.classes.index(label)
        order = np.lexsort((np.arange(93), -logits))
        rival = int(order[order != truth][0])
        return logits, {"scope": "full93", "true_label": label, "top1": self.classes[int(order[0])],
                        "top1_index": int(order[0]), "correct": int(order[0]) == truth,
                        "true_rank": int(np.flatnonzero(order == truth)[0]) + 1,
                        "best_other": self.classes[rival],
                        "true_vs_best_other_margin": float(logits[truth] - logits[rival])}

    def finalize(self, available_c):
        require(self.forwards == 18 + available_c, "Actual independent forward count")
        require(all(self.torch.equal(self.before[k], v) for k, v in self.model.state_dict().items()),
                "Model state changed during independent verification")
        require(state_digest(self.model.state_dict()) == self.payload_state_sha256, "Final model state digest")
        forbidden = [k for k in sys.modules if k == "av" or k == "mediapipe" or k.startswith("mediapipe.")]
        require(not forbidden, "Decoder or MediaPipe unexpectedly loaded")
        return {"model_loads": 1, "tcn_forwards": self.forwards, "full_model_state_unchanged": True,
                "payload_and_before_and_after_state_sha256": self.payload_state_sha256,
                "video_decodes": 0, "mediapipe_calls": 0, "camera_calls": 0, "training_runs": 0}
