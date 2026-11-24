"""handshape_probe: bounded OFFLINE feasibility probe (NOT a KSL recognizer).

Runs the official MediaPipe GestureRecognizer (IMAGE mode, canned 7 gestures + None)
once on each of exactly 5 pre-selected decoded frames per public clip and records the
raw per-hand outputs plus a descriptive shoulder/nose-relative wrist position computed
from cached 2D pose. Everything emitted is a machine suggestion for human review, never
ground truth, never accuracy, never a sign-correctness verdict.

CLI: python handshape_probe.py --config CONFIG.json --out-dir NEW_DIR
Exit codes: 0 complete, 1 failed after output dir creation (FAILED.json), 2 preflight refused.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import sys
from fractions import Fraction
from importlib import metadata

import numpy as np

TOOL_VERSION = "0.1.0"
FIXED_POLICY = {
    "running_mode": "IMAGE",
    "num_hands": 2,
    "min_hand_detection_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "canned_score_threshold": 0.0,
    "canned_max_results": -1,
    "delegate": "CPU",
    "expected_clips": 18,
    "samples_per_clip": 5,
    "timestamp_tolerance_us": 2,
}
NOSE, SH_A, SH_B = 0, 11, 12
N_POINTS = 75
CELL_W, CELL_H = 368, 324
COLORS = [(255, 70, 70), (70, 220, 255), (255, 210, 0), (150, 255, 120)]
HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
CLIP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
WORD_RE = re.compile(r"^[\x21-\x7e][\x20-\x7e]{0,63}$")
EXTS = {
    "video_path": (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"),
    "npz_path": (".npz",),
    "sidecar_path": (".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv"),
    "model_path": (".task",),
    "config": (".json",),
}
WIN_DEVICES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
               *(f"LPT{i}" for i in range(1, 10))}
TOP_KEYS = ("root", "model_path", "model_sha256", "model_url", "inputs", "protected_files", "policy")
INPUT_KEYS = ("clip_id", "word_id", "display_name", "video_path", "video_sha256", "npz_path",
              "sha256", "sidecar_path", "sidecar_sha256", "width", "height", "samples")
NPZ_KEYS = ("xy_px", "usable", "observed", "frame_id", "timestamp_us", "hand_score")
SCIENTIFIC_LIMITS = [
    "Bounded offline feasibility probe only: not a KSL recognizer, not a sign-correctness assessor, not a production feature.",
    "Canned classes are 7 common gestures plus None; many KSL handshapes have no canned class. There is no dictionary-to-gesture truth mapping, so counts are NOT accuracy.",
    "Open_Palm is not KSL handshape 0900; Pointing_Up does not cover all 0100 orientations; Victory is not the whole dynamic sign 'jada'.",
    "Category 'None' means a hand WAS detected and the classifier returned None; it is different from 'no hand detected'.",
    "Scores are raw model outputs as emitted (not all 8 classes necessarily present, not renormalized); a finite score is not calibrated reliability. No threshold, vote, or smoothing was applied.",
    "Handedness is the model's raw image-based suggestion, not anatomical side; detections were not matched to cached left/right tracks.",
    "Relative positions use cached 2D pose only as descriptive context (units: clip-median shoulder width, image axes x right, y down). No depth, contact, direction, or correct-execution verdict. Normalized coordinates inside/outside [0,1] do not imply quality.",
    "Only 5 selected frames per clip: this is not continuous review. Different classes across frames of a dynamic sign are transitions, not errors.",
    "Missing values are null with a reason, never zero. The model never receives word_id or any label.",
]


class ProbeError(Exception):
    """Explicit, expected failure (validation, hash, binding, decode, integrity)."""


# ----------------------------------------------------------------------------- helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def _inside(child, parent):
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:
        return False


def _is_int(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))


def _f(v):
    if v is None or isinstance(v, (bool, np.bool_)):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _ascii(s):
    return str(s).encode("ascii", "replace").decode("ascii")


def dumps_strict(obj):
    return json.dumps(obj, allow_nan=False, ensure_ascii=False, indent=2)


def _reject_constant(name):
    raise ValueError(f"non-finite JSON constant not allowed: {name}")


def _hex(v, key):
    if not isinstance(v, str) or not HEX64.fullmatch(v):
        raise ProbeError(f"{key}: must be a 64-hex sha256")
    return v.lower()


def check_local_file(path, key, exts=None):
    if not isinstance(path, str) or not path:
        raise ProbeError(f"{key}: path must be a non-empty string")
    if "://" in path or path.startswith(("\\\\", "//")):
        raise ProbeError(f"{key}: URI/UNC/device paths are not allowed: {path!r}")
    if not os.path.isabs(path):
        raise ProbeError(f"{key}: path must be absolute: {path!r}")
    if os.path.basename(path).split(".")[0].upper() in WIN_DEVICES:
        raise ProbeError(f"{key}: device name not allowed: {path!r}")
    if exts and not path.lower().endswith(exts):
        raise ProbeError(f"{key}: filename extension not in {exts}: {path!r}")
    try:
        st = os.lstat(path)
    except OSError as e:
        raise ProbeError(f"{key}: missing file {path!r}: {e}") from None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise ProbeError(f"{key}: not a regular non-symlink local file: {path!r}")
    return path


# ----------------------------------------------------------------------------- preflight
def validate_config(cfg, expected_clips):
    """Schema/policy/path validation. Returns {normpath: {path, sha256}} integrity map."""
    if not isinstance(cfg, dict):
        raise ProbeError("config must be a JSON object")
    missing = [k for k in TOP_KEYS if k not in cfg]
    if missing:
        raise ProbeError(f"config missing keys: {missing}")
    pol = cfg["policy"]
    if (not isinstance(pol, dict) or set(pol) != set(FIXED_POLICY) or any(
            pol[k] != v or isinstance(pol[k], bool) != isinstance(v, bool) for k, v in FIXED_POLICY.items())):
        raise ProbeError("policy must equal the fixed policy exactly: " + json.dumps(FIXED_POLICY))
    root = cfg["root"]
    if not isinstance(root, str) or not os.path.isabs(root) or not os.path.isdir(root):
        raise ProbeError("root must be an existing absolute directory")
    expected = {}

    def claim(path, digest, key):
        n, d = _norm(path), _hex(digest, key)
        if n in expected and expected[n]["sha256"] != d:
            raise ProbeError(f"conflicting sha256 declared for {path!r}")
        expected[n] = {"path": path, "sha256": d}

    check_local_file(cfg["model_path"], "model_path", EXTS["model_path"])
    claim(cfg["model_path"], cfg["model_sha256"], "model_sha256")
    if not isinstance(cfg["model_url"], str) or not cfg["model_url"].startswith("https://"):
        raise ProbeError("model_url must be an https URL (recorded only; never downloaded)")
    inputs = cfg["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ProbeError("inputs must be a non-empty list")
    if expected_clips is not None and len(inputs) != expected_clips:
        raise ProbeError(f"expected {expected_clips} clips, got {len(inputs)}")
    seen = set()
    for inp in inputs:
        if not isinstance(inp, dict) or [k for k in INPUT_KEYS if k not in inp]:
            raise ProbeError(f"input entry missing keys (need {INPUT_KEYS})")
        cid = inp["clip_id"]
        if not isinstance(cid, str) or not CLIP_RE.fullmatch(cid) or ".." in cid:
            raise ProbeError(f"invalid clip_id {cid!r} (allowed [A-Za-z0-9_.-], no '..')")
        if cid in seen:
            raise ProbeError(f"duplicate clip_id {cid}")
        seen.add(cid)
        if not isinstance(inp["word_id"], str) or not WORD_RE.fullmatch(inp["word_id"]):
            raise ProbeError(f"{cid}: word_id must be printable ASCII")
        if not isinstance(inp["display_name"], str):
            raise ProbeError(f"{cid}: display_name must be a string")
        for k in ("width", "height"):
            if not _is_int(inp[k]) or inp[k] <= 0:
                raise ProbeError(f"{cid}: {k} must be a positive int")
        for key, hkey in (("video_path", "video_sha256"), ("npz_path", "sha256"), ("sidecar_path", "sidecar_sha256")):
            check_local_file(inp[key], f"{cid}.{key}", EXTS[key])
            if not _inside(_norm(inp[key]), _norm(root)):
                raise ProbeError(f"{cid}.{key} must be under root")
            claim(inp[key], inp[hkey], f"{cid}.{hkey}")
        samples = inp["samples"]
        if not isinstance(samples, list) or len(samples) != FIXED_POLICY["samples_per_clip"]:
            raise ProbeError(f"{cid}: exactly {FIXED_POLICY['samples_per_clip']} samples required")
        for s in samples:
            if not isinstance(s, dict) or not all(_is_int(s.get(k)) and s.get(k) >= 0
                                                  for k in ("row", "frame_id", "timestamp_us")):
                raise ProbeError(f"{cid}: each sample needs non-negative int row, frame_id, timestamp_us")
        if len({s["row"] for s in samples}) != len(samples) or len({s["frame_id"] for s in samples}) != len(samples):
            raise ProbeError(f"{cid}: sample rows and frame_ids must be unique")
    prot = cfg["protected_files"]
    if not isinstance(prot, dict):
        raise ProbeError("protected_files must be a {path: sha256} object")
    for path, digest in prot.items():
        check_local_file(path, "protected_files")
        claim(path, digest, f"protected_files[{path}]")
    return expected


def rehash(expected):
    bad = []
    for _, e in sorted(expected.items()):
        try:
            got = sha256_file(e["path"])
        except OSError as ex:
            got = f"unreadable: {ex}"
        if got != e["sha256"]:
            bad.append({"path": e["path"], "expected": e["sha256"], "actual": got})
    return bad


def load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            missing = [k for k in NPZ_KEYS if k not in z.files]
            if missing:
                raise ProbeError(f"{path}: npz missing arrays {missing}")
            a = {k: np.array(z[k]) for k in NPZ_KEYS}
    except ProbeError:
        raise
    except Exception as e:  # noqa: BLE001 - surface any load failure explicitly
        raise ProbeError(f"cannot load npz {path!r}: {e}") from None
    T = a["frame_id"].shape[0] if a["frame_id"].ndim == 1 else -1
    shapes = {"xy_px": (T, N_POINTS, 2), "usable": (T, N_POINTS), "observed": (T, N_POINTS),
              "frame_id": (T,), "timestamp_us": (T,), "hand_score": (T, 2)}
    for k, shp in shapes.items():
        if T < 1 or a[k].shape != shp:
            raise ProbeError(f"{path}: {k} shape {a[k].shape} != expected {shp}")
    for v in a.values():
        v.flags.writeable = False
    return a


def bind_rows(npz, samples, clip_id):
    """Exact row binding: npz frame_id/timestamp_us at row must equal the sample's."""
    T = npz["frame_id"].shape[0]
    for s in samples:
        r = s["row"]
        if not 0 <= r < T:
            raise ProbeError(f"{clip_id}: row {r} out of range [0,{T})")
        fid, ts = _f(npz["frame_id"][r]), _f(npz["timestamp_us"][r])
        if fid is None or fid != s["frame_id"] or ts is None or ts != s["timestamp_us"]:
            raise ProbeError(f"{clip_id}: row {r} mismatch: npz frame_id={npz['frame_id'][r]} "
                             f"timestamp_us={npz['timestamp_us'][r]} vs config {s['frame_id']}/{s['timestamp_us']}")
    return sorted((dict(s) for s in samples), key=lambda s: s["frame_id"])


# ----------------------------------------------------------------------------- frames
def decode_selected_frames(video_path, ordinals):
    """Sequential PyAV decode; returns {ordinal: (rgb uint8 HxWx3, pts_us relative to first decoded frame)}."""
    import av  # lazy import
    wanted, out, first = set(ordinals), {}, None
    last = max(wanted)
    with av.open(video_path, mode="r") as container:
        if not container.streams.video:
            raise ProbeError(f"{video_path}: no video stream")
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if frame.pts is None:
                raise ProbeError(f"{video_path}: decoded frame {i} has no PTS")
            tb = frame.time_base if frame.time_base is not None else stream.time_base
            us = Fraction(frame.pts) * Fraction(tb) * 1_000_000
            first = us if first is None else first
            if i in wanted:
                out[i] = (np.ascontiguousarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8), us - first)
            if i >= last:
                break
    if wanted - set(out):
        raise ProbeError(f"{video_path}: ordinals not decoded: {sorted(wanted - set(out))}")
    return out


def verify_frames(samples, decoded, width, height, tol_us):
    for s in samples:
        if s["frame_id"] not in decoded:
            raise ProbeError(f"frame {s['frame_id']} not decoded")
        rgb, rel = decoded[s["frame_id"]]
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (height, width, 3):
            raise ProbeError(f"frame {s['frame_id']}: image {getattr(rgb, 'shape', None)} != {(height, width, 3)} uint8")
        diff = abs(Fraction(rel) - Fraction(s["timestamp_us"]))
        if diff > tol_us:
            raise ProbeError(f"frame {s['frame_id']}: PTS {float(Fraction(rel)):.3f}us vs npz {s['timestamp_us']}us (>{tol_us}us)")


# ----------------------------------------------------------------------------- body context
def _pt(xy_row, usable_row, i):
    if not bool(usable_row[i]):
        return None
    x, y = _f(xy_row[i][0]), _f(xy_row[i][1])
    return None if x is None or y is None else [x, y]


def clip_shoulder_width(xy, usable):
    """Median shoulder width (px) over rows where both shoulders are usable AND finite."""
    widths = []
    for t in range(xy.shape[0]):
        a, b = _pt(xy[t], usable[t], SH_A), _pt(xy[t], usable[t], SH_B)
        if a is not None and b is not None:
            w = math.hypot(b[0] - a[0], b[1] - a[1])
            if math.isfinite(w) and w > 0:
                widths.append(w)
    if not widths:
        return None, 0, "no_row_with_both_shoulders_usable_and_finite"
    return float(np.median(widths)), len(widths), None


def frame_anchors(xy_row, usable_row):
    a, b, nose = _pt(xy_row, usable_row, SH_A), _pt(xy_row, usable_row, SH_B), _pt(xy_row, usable_row, NOSE)
    mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] if a is not None and b is not None else None
    return {"pose_11_px": a, "pose_12_px": b, "nose_px": nose, "shoulder_mid_px": mid}


def relative_positions(wrist_px, anchors, shoulder_width, width_reason):
    out = {"shoulder_position": None, "shoulder_position_reason": None,
           "nose_position": None, "nose_position_reason": None,
           "units": "clip_median_shoulder_width; image axes x right, y down"}
    reason = None
    if wrist_px is None or _f(wrist_px[0]) is None or _f(wrist_px[1]) is None:
        reason = "wrist_landmark_not_finite"
    elif shoulder_width is None:
        reason = width_reason or "shoulder_width_unavailable"
    if reason:
        out["shoulder_position_reason"] = out["nose_position_reason"] = reason
        return out
    for key, akey, miss in (("shoulder_position", "shoulder_mid_px", "shoulders_not_both_usable_in_frame"),
                            ("nose_position", "nose_px", "nose_not_usable_in_frame")):
        a = anchors.get(akey)
        if a is None:
            out[key + "_reason"] = miss
        else:
            out[key] = [_f((wrist_px[0] - a[0]) / shoulder_width), _f((wrist_px[1] - a[1]) / shoulder_width)]
    return out


# ----------------------------------------------------------------------------- model output
def _cats(seq):
    out = []
    for c in (seq or []):
        name, disp, idx = getattr(c, "category_name", None), getattr(c, "display_name", None), getattr(c, "index", None)
        out.append({"category_name": None if name is None else str(name),
                    "display_name": disp if isinstance(disp, str) and disp else None,
                    "score": _f(getattr(c, "score", None)),
                    "index": int(idx) if _is_int(idx) else None})
    return out


def top_by_score(cats):
    """Max finite score; ties keep the first returned category. No threshold."""
    best = None
    for c in cats:
        if c["score"] is not None and (best is None or c["score"] > best["score"]):
            best = c
    return best


def _lms(seq):
    return [[_f(getattr(p, "x", None)), _f(getattr(p, "y", None)), _f(getattr(p, "z", None))] for p in (seq or [])]


def parse_result(result, width, height, anchors, shoulder_width, width_reason):
    names = ("gestures", "handedness", "hand_landmarks", "hand_world_landmarks")
    lists = [list(getattr(result, k, None) or []) for k in names]
    n = len(lists[2])
    if any(len(x) != n for x in lists):
        raise ProbeError(f"inconsistent per-hand list lengths {dict(zip(names, map(len, lists)))}")
    hands = []
    for i in range(n):
        g, hd, lm, wl = _cats(lists[0][i]), _cats(lists[1][i]), _lms(lists[2][i]), _lms(lists[3][i])
        if len(lm) != 21:
            raise ProbeError(f"detection {i}: expected 21 landmarks, got {len(lm)}")
        top = top_by_score(g)
        wrist = [lm[0][0] * width, lm[0][1] * height] if lm[0][0] is not None and lm[0][1] is not None else None
        hands.append({
            "detector_index": i,
            "raw_handedness": hd,
            "raw_handedness_top": top_by_score(hd),
            "gesture_categories": g,
            "top_category": top,
            "top_category_reason": None if top else ("no_categories_returned" if not g else "no_finite_scores"),
            "wrist_px": wrist,
            "landmarks_normalized": lm,
            "world_landmarks": wl,
            "body_relative": relative_positions(wrist, anchors, shoulder_width, width_reason),
        })
    return {"detection_status": "hands_detected" if n else "no_hand_detected", "n_hands": n, "hands": hands}


def summarize(frames):
    c = {"frames_total": len(frames), "frames_no_hand": 0, "frames_one_hand": 0, "frames_two_hands": 0,
         "frames_more_than_two_hands": 0, "hand_detections": 0, "detections_without_top_category": 0}
    tops = {}
    for fr in frames:
        n = len(fr["hands"])
        c[{0: "frames_no_hand", 1: "frames_one_hand", 2: "frames_two_hands"}.get(n, "frames_more_than_two_hands")] += 1
        c["hand_detections"] += n
        for h in fr["hands"]:
            t = h["top_category"]
            if t is None or t["category_name"] is None:
                c["detections_without_top_category"] += 1
            else:
                tops[t["category_name"]] = tops.get(t["category_name"], 0) + 1
    c["top_category_counts"] = dict(sorted(tops.items()))
    c["denominators"] = "frame counts over frames_total; category counts over hand_detections; not accuracy"
    return c


def infer_clip(clip, npz, samples, decoded, recognizer):
    """One recognizer call per selected frame, image only (never word_id/labels), on a copy."""
    sw, nrows, reason = clip_shoulder_width(npz["xy_px"], npz["usable"])
    frames = []
    for s in samples:
        rgb, rel = decoded[s["frame_id"]]
        anc = frame_anchors(npz["xy_px"][s["row"]], npz["usable"][s["row"]])
        res = recognizer(np.array(rgb, dtype=np.uint8, order="C", copy=True))
        parsed = parse_result(res, clip["width"], clip["height"], anc, sw, reason)
        frames.append({"row": s["row"], "frame_id": s["frame_id"], "timestamp_us": s["timestamp_us"],
                       "decoded_pts_rel_us": _f(float(Fraction(rel))), "body_anchors_px": anc, **parsed})
    return {"shoulder_width_px_median": sw, "shoulder_width_rows": nrows, "shoulder_width_reason": reason}, frames


def make_mediapipe_recognizer(model_path):
    import mediapipe as mp  # lazy: only when real inference runs
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.components.processors import ClassifierOptions
    from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions, RunningMode
    p = FIXED_POLICY
    opts = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=RunningMode.IMAGE, num_hands=p["num_hands"],
        min_hand_detection_confidence=p["min_hand_detection_confidence"],
        min_hand_presence_confidence=p["min_hand_presence_confidence"],
        min_tracking_confidence=p["min_tracking_confidence"],
        canned_gesture_classifier_options=ClassifierOptions(
            score_threshold=p["canned_score_threshold"], max_results=p["canned_max_results"]))
    rec = GestureRecognizer.create_from_options(opts)

    def recognize(rgb):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb, dtype=np.uint8))
        return rec.recognize(img)
    return recognize, rec.close


# ----------------------------------------------------------------------------- outputs
def prepare_out_dir(out_dir, protected):
    out = os.path.abspath(out_dir)
    if os.path.lexists(out):
        raise ProbeError(f"--out-dir must not exist: {out}")
    n = _norm(out)
    if any(p == n or _inside(p, n) for p in protected):
        raise ProbeError(f"--out-dir collides with a protected/input path: {out}")
    try:
        os.mkdir(out)
    except OSError as e:
        raise ProbeError(f"cannot create --out-dir exclusively: {e}") from None
    os.mkdir(os.path.join(out, "sheets"))
    return out


def safe_write(out, rel, data, protected):
    path = os.path.abspath(os.path.join(out, rel))
    n = _norm(path)
    if n == _norm(out) or not _inside(n, _norm(out)) or n in protected:
        raise ProbeError(f"refusing to write outside output dir or onto protected path: {path}")
    with open(path, "xb") as fh:  # never overwrite
        fh.write(data)
    return path


def _xy(v, reason, short=False):
    if v is not None and v[0] is not None and v[1] is not None:
        return f"({v[0]:+.2f},{v[1]:+.2f})"
    return "null" if short else f"null ({reason})"


def render_sheet(clip, frames, rgbs):
    from PIL import Image, ImageDraw, ImageFont  # lazy
    font = ImageFont.load_default()
    pad, title_h, cap_h, txt_h, line_h = 8, 34, 30, 110, 12
    n = len(frames)
    W = n * CELL_W + (n + 1) * pad
    y_top = title_h + cap_h
    y_bot = y_top + CELL_H + pad
    sheet = Image.new("RGB", (W, y_bot + CELL_H + txt_h + pad), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 5), _ascii(f"clip_id={clip['clip_id']}  word_id={clip['word_id']}"), fill=(255, 255, 255), font=font)
    d.text((pad, 19), "top: unmarked frames | bottom: raw MediaPipe detections (machine output, NOT truth); "
           "+ = cached pose anchors; sh = (wrist - shoulder mid) / clip shoulder width", fill=(190, 190, 190), font=font)
    for c, (fr, rgb) in enumerate(zip(frames, rgbs)):
        x0 = pad + c * (CELL_W + pad)
        d.text((x0, title_h + 2), _ascii(f"frame_id={fr['frame_id']} row={fr['row']}"), fill=(255, 255, 255), font=font)
        d.text((x0, title_h + 15), _ascii(f"t_us={fr['timestamp_us']} hands={fr['n_hands']}"), fill=(255, 255, 255), font=font)
        h, w = rgb.shape[:2]
        s = min(CELL_W / w, CELL_H / h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        img = Image.fromarray(np.ascontiguousarray(rgb)).resize((nw, nh), Image.Resampling.BILINEAR)
        ox, oyt, oyb = x0 + (CELL_W - nw) // 2, y_top + (CELL_H - nh) // 2, y_bot + (CELL_H - nh) // 2
        sheet.paste(img, (ox, oyt))
        sheet.paste(img, (ox, oyb))
        sx, sy = nw / w, nh / h
        for key, col in (("pose_11_px", (255, 255, 255)), ("pose_12_px", (255, 255, 255)),
                         ("nose_px", (255, 140, 0)), ("shoulder_mid_px", (180, 180, 180))):
            p = fr["body_anchors_px"].get(key)
            if p is not None:
                X, Y = ox + p[0] * sx, oyb + p[1] * sy
                d.line([(X - 4, Y), (X + 4, Y)], fill=col, width=2)
                d.line([(X, Y - 4), (X, Y + 4)], fill=col, width=2)
        lines = []
        for hand in fr["hands"]:
            i, col = hand["detector_index"], COLORS[hand["detector_index"] % len(COLORS)]
            pts = [p for p in hand["landmarks_normalized"] if p[0] is not None and p[1] is not None]
            if pts:
                cx = lambda v: min(max(ox + v * nw, ox), ox + nw - 1)  # noqa: E731
                cy = lambda v: min(max(oyb + v * nh, oyb), oyb + nh - 1)  # noqa: E731
                bx1, bx2 = cx(min(p[0] for p in pts)), cx(max(p[0] for p in pts))
                by1, by2 = cy(min(p[1] for p in pts)), cy(max(p[1] for p in pts))
                d.rectangle([bx1, by1, bx2, by2], outline=col, width=2)
                d.text((bx1, max(oyb, by1 - 12)), f"#{i}", fill=col, font=font)
            if hand["wrist_px"] is not None:
                wx, wy = ox + hand["wrist_px"][0] * sx, oyb + hand["wrist_px"][1] * sy
                if ox <= wx <= ox + nw and oyb <= wy <= oyb + nh:
                    d.ellipse([wx - 3, wy - 3, wx + 3, wy + 3], fill=col)
            t, hd, br = hand["top_category"], hand["raw_handedness_top"], hand["body_relative"]
            top = f"{t['category_name']} {t['score']:.2f}" if t else "null"
            hds = f"{hd['category_name']}" if hd else "null"
            lines.append((col, f"#{i} top={top} raw_hand={hds}"))
            lines.append((col, f"   sh={_xy(br['shoulder_position'], None, True)} "
                               f"nose={_xy(br['nose_position'], None, True)}"))
        if not fr["hands"]:
            lines.append(((220, 220, 220), "no hand detected"))
        ty = y_bot + CELL_H + 4
        for col, txt in lines[:8]:
            d.text((x0, ty), _ascii(txt)[:60], fill=col, font=font)
            ty += line_h
    buf = io.BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()


def _counts_md(c):
    tops = ", ".join(f"{k}: {v}" for k, v in c["top_category_counts"].items()) or "(none)"
    return (f"- frames: {c['frames_total']} (no hand detected: {c['frames_no_hand']}, one hand: {c['frames_one_hand']}, "
            f"two hands: {c['frames_two_hands']})\n- hand detections (separate denominator): {c['hand_detections']}; "
            f"without top category: {c['detections_without_top_category']}\n- top-category counts over detections: {tops}")


def _hand_md(h):
    t, hd, br = h["top_category"], h["raw_handedness_top"], h["body_relative"]
    top = f"{t['category_name']} {t['score']:.3f}" if t else f"null ({h['top_category_reason']})"
    hds = f"{hd['category_name']} {hd['score']:.2f}" if hd else "null"
    return (f"#{h['detector_index']} {top}; raw_handedness {hds}; shoulder_rel "
            f"{_xy(br['shoulder_position'], br['shoulder_position_reason'])}; nose_rel "
            f"{_xy(br['nose_position'], br['nose_position_reason'])}")


def render_markdown(r):
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    L = ["# Handshape probe result (descriptive, unvalidated)", "",
         f"- technical_status: **{r['technical_status']}** - {r['technical_status_detail']}",
         f"- scientific_status: {r['scientific_status']}",
         f"- tool {r['tool_version']}; code_sha256 `{r['code_sha256']}`; config_sha256 `{r['config_sha256']}`; "
         f"model_sha256 `{r['model']['sha256']}`", "", "## Scientific limits",
         *[f"- {s}" for s in r["scientific_limits"]], "", "## Totals", _counts_md(r["totals"]), "", "## Clips"]
    for c in r["clips"]:
        sw = "null (" + str(c["shoulder_width_reason"]) + ")" if c["shoulder_width_px_median"] is None \
            else f"{c['shoulder_width_px_median']:.1f} px over {c['shoulder_width_rows']} rows"
        L += ["", f"### {c['clip_id']} (word_id {c['word_id']}) {esc(c['display_name'])}",
              f"- review sheet: [{c['review_sheet']}]({c['review_sheet']})", f"- clip median shoulder width: {sw}",
              _counts_md(c["counts"]), "", "| frame_id | row | timestamp_us | hands | raw detections |", "|---|---|---|---|---|"]
        for f in c["frames"]:
            det = "; ".join(_hand_md(h) for h in f["hands"]) or "no hand detected"
            L.append(f"| {f['frame_id']} | {f['row']} | {f['timestamp_us']} | {f['n_hands']} | {esc(det)} |")
    return "\n".join(L) + "\n"


def versions():
    v = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for dist in ("mediapipe", "av", "Pillow"):
        try:
            v[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            v[dist] = None
    return v


# ----------------------------------------------------------------------------- runner
def run(config_path, out_dir, recognizer_factory=None, frame_source=None,
        expected_clips=FIXED_POLICY["expected_clips"]):
    recognizer_factory = recognizer_factory or make_mediapipe_recognizer
    frame_source = frame_source or decode_selected_frames
    config_path = os.path.abspath(config_path)
    try:
        check_local_file(config_path, "config", EXTS["config"])
        with open(config_path, "rb") as fh:
            raw = fh.read()
        cfg = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        expected = validate_config(cfg, expected_clips)
        protected = set(expected) | {_norm(config_path)}
        out = prepare_out_dir(out_dir, protected)
    except (ProbeError, ValueError, OSError) as e:
        print(f"PREFLIGHT REFUSED (nothing written): {e}", file=sys.stderr)
        return 2
    try:
        before = rehash(expected)
        if before:
            raise ProbeError(f"sha256 mismatch before run: {before}")
        prepared = []
        for inp in cfg["inputs"]:
            npz = load_npz(inp["npz_path"])
            prepared.append((inp, npz, bind_rows(npz, inp["samples"], inp["clip_id"])))
        recognizer, close = recognizer_factory(cfg["model_path"])
        clips = []
        try:
            for inp, npz, samples in prepared:
                decoded = frame_source(inp["video_path"], [s["frame_id"] for s in samples])
                verify_frames(samples, decoded, inp["width"], inp["height"], FIXED_POLICY["timestamp_tolerance_us"])
                ctx, frames = infer_clip(inp, npz, samples, decoded, recognizer)
                sheet_rel = f"sheets/review_{inp['clip_id']}.png"
                safe_write(out, sheet_rel, render_sheet(inp, frames, [decoded[s["frame_id"]][0] for s in samples]), protected)
                del decoded  # selected RGB lives only in memory
                clips.append({"clip_id": inp["clip_id"], "word_id": inp["word_id"], "display_name": inp["display_name"],
                              "video_sha256": inp["video_sha256"].lower(), "npz_sha256": inp["sha256"].lower(),
                              "sidecar_sha256": inp["sidecar_sha256"].lower(), "width": inp["width"],
                              "height": inp["height"], **ctx, "counts": summarize(frames),
                              "review_sheet": sheet_rel, "frames": frames})
        finally:
            close()
        after = rehash(expected)
        if after:
            raise ProbeError(f"sha256 mismatch after run: {after}")
        result = {
            "tool": "handshape_probe", "tool_version": TOOL_VERSION, "technical_status": "complete",
            "technical_status_detail": "every configured frame decoded, row/PTS-bound and inferred exactly once; "
                                       "pre/post sha256 of model, sources, sidecars, npz and protected files matched",
            "scientific_status": "descriptive_only_unvalidated_no_accuracy",
            "scientific_limits": SCIENTIFIC_LIMITS, "versions": versions(),
            "code_sha256": sha256_file(os.path.abspath(__file__)),
            "config_path": config_path, "config_sha256": hashlib.sha256(raw).hexdigest(), "config": cfg,
            "model": {"path": cfg["model_path"], "sha256": cfg["model_sha256"].lower(), "url": cfg["model_url"]},
            "fixed_policy": FIXED_POLICY,
            "integrity": {"files_checked": len(expected), "pre_run_mismatches": before, "post_run_mismatches": after},
            "totals": summarize([f for c in clips for f in c["frames"]]),
            "clips": clips,
        }
        safe_write(out, "result.pending.json", dumps_strict(result).encode("utf-8"), protected)
        safe_write(out, "RESULT.pending.md", render_markdown(result).encode("utf-8"), protected)
        final = rehash(expected)
        if final:
            raise ProbeError(f"sha256 mismatch after writing outputs: {final}")
        os.rename(os.path.join(out, "result.pending.json"), os.path.join(out, "result.json"))
        os.rename(os.path.join(out, "RESULT.pending.md"), os.path.join(out, "RESULT.md"))
        print(f"complete: {os.path.join(out, 'result.json')}")
        return 0
    except Exception as e:  # noqa: BLE001 - any failure is reported, never as success
        failure = {"technical_status": "failed", "error_type": type(e).__name__, "error": str(e),
                   "note": "No successful result. Any *.pending.* or sheet files are not a valid report."}
        try:
            safe_write(out, "FAILED.json", dumps_strict(failure).encode("utf-8"), protected)
        except Exception:  # noqa: BLE001
            pass
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bounded offline MediaPipe GestureRecognizer handshape probe.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return run(a.config, a.out_dir)


if __name__ == "__main__":
    sys.exit(main())
