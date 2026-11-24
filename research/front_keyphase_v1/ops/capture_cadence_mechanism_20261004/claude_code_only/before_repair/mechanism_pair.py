"""Frozen observation mechanism diagnostic: common-mask and pose/hand donor hybrids of arms A/B.

Oracle diagnostic only. The hybrids cannot be built from B alone, A is not anatomical ground truth,
and mixing pose and hand blocks from different arms can be physically inconsistent.

Default invocation validates every gate without importing torch/pilot or writing anything.
`--run --test-receipt PATH` rebuilds the 8 original inputs (zero forwards), reuses their stored
full93 scores, and performs exactly 16 new forwards (4 variants x 4 windows).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

UNIT_NAME = "capture_cadence_mechanism_20261004"
CONFIG_SCHEMA = "wis_capture_cadence_mechanism_config_v1"
RESUME_SCHEMA = "wis_explicit_user_resume_v1"
RECEIPT_SCHEMA = "wis_mechanism_test_receipt_v1"
RESULT_SCHEMA = "wis_capture_cadence_mechanism_result_v1"
KST = dt.timezone(dt.timedelta(hours=9))

VARIANTS = ("common_A", "common_B", "pose_A_hands_B", "pose_B_hands_A")
DONORS = {
    "common_A": {"pose": "A", "hands": "A", "mask": "A.usable & B.usable"},
    "common_B": {"pose": "B", "hands": "B", "mask": "A.usable & B.usable"},
    "pose_A_hands_B": {"pose": "A", "hands": "B", "mask": None},
    "pose_B_hands_A": {"pose": "B", "hands": "A", "mask": None},
}
N_POINTS, N_CLASSES, N_CHANNELS = 75, 93, 280
HANDS = ((33, 54), (54, 75))                   # left, right; pose is [0,33)
OBS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")
DECODED = ("timestamp_us", "pts", "rgb_sha256", "rgb_shape", "processed")
NPZ_KEYS = tuple("obs_" + k for k in OBS) + tuple("decoded_" + k for k in ("frame_id",) + DECODED) \
    + ("selected_frame_id",)
ARM_TAG = {"A": "A_full_then_even", "B": "B_even_rgb_only"}
SINGLE_ROLES = ("source_result", "source_file_hash_manifest", "source_independent_result",
                "source_independent_artifact_check", "source_prepared_config", "frozen_checkpoint",
                "pilot_data", "pilot_mapping", "pilot_model", "observation_schema", "frozen_plan")
BASE_KEYS = ("x_sha256", "x_shape", "x_dtype", "frames", "body_scale", "hand_scales",
             "body_valid_sum", "hand_valid_sum")
SCORE_KEYS = ("true_label", "top1", "top1_index", "correct", "true_rank", "best_other",
              "true_vs_best_other_margin")
# pilot.data channel layout: body xy 102, hand xy 84, body valid 51, hand valid 42, dt 1
BLOCKS = {"body_xy": (0, 102), "hand_xy": (102, 186), "body_valid": (186, 237),
          "hand_valid": (237, 279), "dt": (279, 280)}
VALID_CHANNELS = slice(186, 279)
FORBIDDEN_MODULES = ("av", "mediapipe", "cv2")
RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
LIMITS = (
    "Oracle diagnostic hybrids: they need both arms and cannot be deployed from B alone.",
    "A is not declared anatomical ground truth.",
    "Mixing pose and hand blocks from different arms can be physically inconsistent.",
    "Common masking also changes per-condition normalisation (body/hand scales); remaining or "
    "diminished differences do not identify a unique physical cause.",
    "No adoption, accuracy-improvement, web-FPS, new-person or sign-correctness claim.",
    "Self-run output; not independently verified.",
)


class Fail(Exception):
    def __init__(self, stage, msg):
        super().__init__(f"{stage}: {msg}")
        self.stage = stage


def need(cond, stage, msg):
    if not cond:
        raise Fail(stage, msg)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    """(object, sha256 of the exact bytes parsed). UTF-8-SIG accepts PowerShell BOM."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw.decode("utf-8-sig")), sha256_bytes(raw)


def typed_sha256(a):
    a = np.asarray(a)
    h = hashlib.sha256((a.dtype.str + "|" + "x".join(str(d) for d in a.shape) + "|").encode("ascii"))
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def jnum(v):
    v = float(v)
    return None if math.isnan(v) else v


def same(a, b):
    """Exact equality of shape, dtype and values; NaN equals NaN for float arrays."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))


def same_path(a, b):
    return os.path.normcase(os.path.normpath(str(a))) == os.path.normcase(os.path.normpath(str(b)))


def local_path(p, root, stage="path"):
    """Absolute local path, no URL/UNC/device/stream/'..', no link redirection, inside root."""
    need(isinstance(p, str) and p and "\x00" not in p, stage, f"not a path string: {p!r}")
    need("://" not in p and not p.startswith(("\\\\", "//")), stage, f"URL/UNC/device refused: {p}")
    path = Path(p)
    need(path.is_absolute(), stage, f"not absolute: {p}")
    need(":" not in p[2:], stage, f"stream/drive syntax refused: {p}")
    need(".." not in path.parts, stage, f"parent escape refused: {p}")
    need(not any(part.split(".")[0].strip().upper() in RESERVED for part in path.parts[1:]),
         stage, f"device name refused: {p}")
    norm = os.path.normpath(p)
    real = os.path.realpath(norm)
    need(same_path(real, norm), stage, f"link redirection refused: {p} -> {real}")
    if root is not None:
        r = os.path.normcase(os.path.realpath(str(root)))
        try:
            inside = os.path.commonpath([os.path.normcase(real), r]) == r
        except ValueError:
            inside = False
        need(inside, stage, f"outside {root}: {p}")
    return Path(norm)


def parse_aware(s, stage):
    try:
        t = dt.datetime.fromisoformat(s)
    except (TypeError, ValueError):
        raise Fail(stage, f"not an ISO datetime: {s!r}") from None
    need(t.tzinfo is not None and t.utcoffset() is not None, stage, f"timezone required: {s}")
    return t


# ---------------------------------------------------------------- gates

def check_resume(resume_path, cfg, cfg_sha, unit, now):
    p = local_path(str(resume_path), unit, "resume")
    need(same_path(p, unit / "RESUME_AUTHORIZATION.json"), "resume",
         "must be this unit's RESUME_AUTHORIZATION.json")
    doc, sha = read_json(p)
    need(isinstance(doc, dict), "resume", "not an object")
    need(doc.get("schema") == RESUME_SCHEMA, "resume", "schema")
    need(doc.get("unit") == UNIT_NAME and unit.name == UNIT_NAME, "resume", "unit")
    need(doc.get("prepared_config_sha256") == cfg_sha, "resume", "prepared_config_sha256 != config bytes")
    need(doc.get("allow_after_original_deadline") is True, "resume", "allow_after_original_deadline")
    need(doc.get("supersedes_deadline_only") == cfg.get("deadline_kst"), "resume",
         "supersedes_deadline_only != config.deadline_kst")
    for k in ("camera", "training", "operational_change"):
        need(doc.get(k) is False, "resume", f"{k} must be false")
    need(isinstance(doc.get("user_request"), str) and doc["user_request"].strip(), "resume",
         "user_request empty")
    rec = parse_aware(doc.get("recorded_kst"), "resume")
    need(rec <= now, "resume", "recorded_kst is in the future")
    need(rec.astimezone(KST).date() == now.astimezone(KST).date(), "resume",
         "recorded_kst is not on the current KST date")
    return {"path": str(p), "sha256": sha, "content": doc}


def deadline_gate(cfg, cfg_sha, unit, resume_path, now):
    deadline = parse_aware(cfg.get("deadline_kst"), "deadline")
    if resume_path is None:
        need(now <= deadline, "deadline",
             f"original deadline {cfg['deadline_kst']} passed; --resume-authorization required")
        return None
    return check_resume(resume_path, cfg, cfg_sha, unit, now)


def check_pins(cfg, root):
    pins, inv = cfg.get("pins"), cfg.get("pin_inventory")
    need(isinstance(pins, dict) and isinstance(inv, list), "pins", "pins/pin_inventory missing")
    need(len(inv) == len(pins) and {e["path"]: e["sha256"] for e in inv} == pins, "pins",
         "pin_inventory does not agree exactly with pins")
    roles = {}
    for e in inv:
        need(HEX64.match(str(e["sha256"])), "pins", f"bad sha {e['path']}")
        p = local_path(e["path"], root, "pins")
        need(p.is_file(), "pins", f"missing {p}")
        need(p.stat().st_size == e["bytes"], "pins", f"size mismatch {e['role']} {p}")
        need(sha256_file(p) == e["sha256"], "pins", f"hash mismatch {e['role']} {p}")
        roles.setdefault(e["role"], []).append(e["path"])
    for r in SINGLE_ROLES:
        need(len(roles.get(r, [])) == 1, "pins", f"role {r} must be pinned exactly once")
    raw8 = []
    for clip in cfg["clips"]:
        for arm in "AB":
            rec = clip["arms"][arm]
            need(rec["raw_npz"] in pins and rec["meta_json"] in pins, "pins", f"arm {arm} files unpinned")
            raw8.append(rec["raw_npz"])
    need(len(set(raw8)) == 8 and sorted(cfg["isolation"]["allowed_raw_npz"]) == sorted(raw8), "pins",
         "allowed_raw_npz must be exactly the 8 raw files")
    m = cfg["model"]
    need(roles["frozen_checkpoint"] == [m["checkpoint_path"]]
         and pins[m["checkpoint_path"]] == m["checkpoint_sha256"], "pins", "checkpoint pin")
    src = roles["source_result"][0]
    need(pins[src] == cfg["source_result_sha256"], "pins", "source_result_sha256 != pin")
    need(same_path(Path(src).parent, local_path(cfg["source_result_dir"], root, "pins")), "pins",
         "source result pin outside source_result_dir")
    return {r: v[0] for r, v in roles.items() if len(v) == 1}


def check_source(cfg, roles, classes):
    src, sha = read_json(roles["source_result"])
    need(sha == cfg["source_result_sha256"], "source", "source RESULT bytes changed")
    need(src.get("status") == "COMPLETE", "source", "source status != COMPLETE")
    sc = src["scoring"]
    need(sc["classes"] == classes, "source", "class order differs from config")
    wins = sc["windows"]
    need(len(wins) == 4, "source", "source must have exactly 4 windows")
    files = {c["clip_id"]: c["files"] for c in src["clips"]}
    for clip, w in zip(cfg["clips"], wins):
        for k in ("clip_id", "label", "start_us", "end_us_exclusive", "window_frame_ids",
                  "window_timestamp_us"):
            need(w[k] == clip[k], "source", f"{clip['clip_id']} {k} differs")
        rows = clip["expected_window_rows"]
        need(isinstance(rows, int) and rows > 0 and len(clip["window_frame_ids"]) == rows
             and len(clip["window_timestamp_us"]) == rows, "source", "window row counts")
        need(clip["label"] in classes, "source", "label not in classes")
        need(clip["start_us"] < clip["end_us_exclusive"], "source", "empty window")
        for arm in "AB":
            rec = clip["arms"][arm]
            need(rec["source_arm"] == arm, "source", "source_arm")
            base = rec["baseline_input"]
            need(base["x_shape"] == [1, rows, N_CHANNELS] and base["frames"] == rows
                 and base["x_dtype"] == "float32", "source", "baseline input shape")
            for k in BASE_KEYS:
                need(w[arm]["input"][k] == base[k], "source", f"{clip['clip_id']} {arm} input {k}")
            lg = rec["baseline_logits93"]
            need(isinstance(lg, list) and len(lg) == N_CLASSES and w[arm]["logits93"] == lg
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in lg),
                 "source", f"{clip['clip_id']} {arm} logits93")
            for k in SCORE_KEYS:
                need(w[arm]["score"][k] == rec["baseline_score"][k], "source",
                     f"{clip['clip_id']} {arm} score {k}")
            f = files[clip["clip_id"]][arm]
            need(f["npz"] == Path(rec["raw_npz"]).name and f["meta"] == Path(rec["meta_json"]).name,
                 "source", f"{clip['clip_id']} {arm} file leaves")
    return sha


# ---------------------------------------------------------------- observations

def load_arm(rec, arm, clip_id, pins):
    meta, _ = read_json(rec["meta_json"])
    need(meta.get("npz_sha256") == pins[rec["raw_npz"]] and meta.get("clip_id") == clip_id
         and meta.get("arm") == arm and meta.get("tag") == ARM_TAG[arm], "raw", f"{clip_id} {arm} meta")
    with open(rec["raw_npz"], "rb") as fh:
        raw = fh.read()
    need(sha256_bytes(raw) == pins[rec["raw_npz"]], "raw", f"{clip_id} {arm} npz bytes changed")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(NPZ_KEYS) <= set(z.files), "raw", f"{clip_id} {arm} missing keys")
        d = {k: z[k] for k in NPZ_KEYS}
    n = d["obs_frame_id"].shape[0]
    need(d["obs_xy_px"].shape == (n, N_POINTS, 2) and d["obs_xy_px"].dtype == np.float32
         and d["obs_frame_id"].dtype == np.int64 and d["obs_timestamp_us"].dtype == np.int64
         and d["obs_timestamp_us"].shape == (n,)
         and d["obs_observed"].shape == (n, N_POINTS) and d["obs_observed"].dtype == bool
         and d["obs_usable"].shape == (n, N_POINTS) and d["obs_usable"].dtype == bool
         and d["obs_conf"].shape == (n, N_POINTS) and d["obs_conf"].dtype.kind == "f"
         and d["obs_hand_score"].shape == (n, 2) and d["obs_hand_score"].dtype.kind == "f",
         "raw", f"{clip_id} {arm} observation layout")
    need(n > 0 and (np.diff(d["obs_frame_id"]) > 0).all(), "raw", f"{clip_id} {arm} frame ids not increasing")
    return meta, d


def select(d, arm, clip_id):
    """A: keep even stored frame IDs. B: already even. No renumbering, no resampling."""
    fid = d["obs_frame_id"]
    if arm == "B":
        need((fid % 2 == 0).all(), "select", f"{clip_id} B contains odd frame ids")
    keep = fid % 2 == 0
    need(np.array_equal(fid[keep], d["selected_frame_id"]), "select", f"{clip_id} {arm} selected_frame_id")
    obs = {k: d["obs_" + k][keep].copy() for k in OBS}
    pos = {int(f): i for i, f in enumerate(d["decoded_frame_id"])}
    need(len(pos) == len(d["decoded_frame_id"]), "select", f"{clip_id} {arm} duplicate decoded ids")
    need(all(int(f) in pos for f in obs["frame_id"]), "select", f"{clip_id} {arm} undecoded selected frame")
    idx = np.array([pos[int(f)] for f in obs["frame_id"]], dtype=np.int64)
    dec = {k: d["decoded_" + k][idx] for k in DECODED}
    need(dec["processed"].all(), "select", f"{clip_id} {arm} selected frame not processed")
    need(np.array_equal(dec["timestamp_us"], obs["timestamp_us"]), "select",
         f"{clip_id} {arm} decoded vs observation timestamps")
    return obs, dec


def load_clip(clip, pins):
    cid = clip["clip_id"]
    (ma, ra), (mb, rb) = (load_arm(clip["arms"][arm], arm, cid, pins) for arm in "AB")
    need(ma.get("time_base") == mb.get("time_base"), "select", f"{cid} time_base differs")
    oa, da = select(ra, "A", cid)
    ob, db = select(rb, "B", cid)
    need(np.array_equal(oa["frame_id"], ob["frame_id"]) and np.array_equal(oa["timestamp_us"], ob["timestamp_us"]),
         "select", f"{cid} selected frame_id/timestamp_us differ between arms")
    for k in DECODED:
        need(same(da[k], db[k]), "select", f"{cid} decoded {k} differs between arms")
    ts = oa["timestamp_us"]
    win = (ts >= clip["start_us"]) & (ts < clip["end_us_exclusive"])      # fixed half-open window
    a = {k: v[win] for k, v in oa.items()}
    b = {k: v[win] for k, v in ob.items()}
    need(a["frame_id"].tolist() == clip["window_frame_ids"]
         and a["timestamp_us"].tolist() == clip["window_timestamp_us"]
         and int(win.sum()) == clip["expected_window_rows"], "window", f"{cid} window rows differ from frozen")
    return {"A": a, "B": b, "decoded_pts": da["pts"][win]}


def _common(src, mask):
    v = {k: src[k].copy() for k in OBS}
    v["xy_px"][~mask] = np.nan
    v["conf"][~mask] = np.nan
    v["observed"] = mask.copy()
    v["usable"] = mask.copy()
    for h, (s0, s1) in enumerate(HANDS):
        v["hand_score"][~mask[:, s0:s1].any(axis=1), h] = np.nan
    return v


def _swap(pose, hands):
    v = {k: pose[k].copy() for k in OBS}
    for k in ("xy_px", "observed", "usable", "conf"):
        v[k][:, HANDS[0][0]:HANDS[1][1]] = hands[k][:, HANDS[0][0]:HANDS[1][1]]
    v["hand_score"] = hands["hand_score"].copy()
    return v


def make_variants(a, b):
    """The four fixed variants. Inputs are never mutated."""
    need(np.array_equal(a["frame_id"], b["frame_id"]) and np.array_equal(a["timestamp_us"], b["timestamp_us"]),
         "variants", "arms not aligned")
    mask = a["usable"] & b["usable"]
    vs = {"common_A": _common(a, mask), "common_B": _common(b, mask),
          "pose_A_hands_B": _swap(a, b), "pose_B_hands_A": _swap(b, a)}
    return vs, mask


def check_common_raw(a, b, vs, mask):
    for name, src in (("common_A", a), ("common_B", b)):
        v = vs[name]
        need(np.isfinite(src["xy_px"][mask]).all(), "common", f"{name}: non-finite xy at shared usable point")
        need(np.array_equal(v["xy_px"][mask], src["xy_px"][mask]) and same(v["conf"][mask], src["conf"][mask]),
             "common", f"{name}: shared coordinates/conf not exact")
        need(np.isnan(v["xy_px"][~mask]).all() and np.isnan(v["conf"][~mask]).all(), "common",
             f"{name}: values outside the common mask")
    need(np.array_equal(vs["common_A"]["usable"], vs["common_B"]["usable"]), "common", "common masks differ")
    for name, v in vs.items():
        need(np.array_equal(v["frame_id"], a["frame_id"]) and np.array_equal(v["timestamp_us"], a["timestamp_us"]),
             "common", f"{name}: frame ids/timestamps changed")


def validate(config_path, resume_path=None, now=None):
    """All gates that need neither torch nor the pilot modules. Writes nothing."""
    now = now or dt.datetime.now(KST)
    need(now.tzinfo is not None, "clock", "aware now required")
    cfg_path = Path(os.path.abspath(config_path))
    cfg, cfg_sha = read_json(cfg_path)
    need(cfg.get("schema") == CONFIG_SCHEMA, "config", "schema")
    root = local_path(cfg["workspace_root"], None, "config")
    front = local_path(cfg["front_root"], root, "config")
    unit = local_path(cfg["unit_dir"], front, "config")
    need(unit.name == UNIT_NAME, "config", f"unit dir must be {UNIT_NAME}")
    need(same_path(local_path(str(cfg_path), unit, "config").parent, unit), "config",
         "config must sit in unit_dir")
    out = local_path(cfg["output_dir"], unit, "config")
    need(same_path(out, unit / "result"), "config", "output_dir must be unit_dir/result")
    need(not os.path.lexists(out), "config", f"output exists, refusing overwrite: {out}")
    counts = cfg.get("prepared_execution_counts")
    need(isinstance(counts, dict) and all(v == 0 for v in counts.values()), "config",
         "prepared execution counts must be zero")
    need(tuple(cfg.get("variants", ())) == VARIANTS, "config", "variants")
    pre = cfg["preprocessing"]
    need(pre["new_forward_calls"] == 16 and pre["baseline_forwards"] == 0
         and pre["baseline_input_rechecks"] == 8, "config", "execution plan")
    need(isinstance(cfg.get("clips"), list) and len(cfg["clips"]) == 4
         and len({c["clip_id"] for c in cfg["clips"]}) == 4, "config", "exactly 4 unique clips")
    auth = deadline_gate(cfg, cfg_sha, unit, resume_path, now)
    m = cfg["model"]
    classes = m["classes"]
    need(isinstance(classes, list) and len(classes) == N_CLASSES == m["n_classes"]
         and len(set(classes)) == N_CLASSES and all(isinstance(c, str) for c in classes),
         "config", "class order must be 93 unique strings")
    roles = check_pins(cfg, root)
    source_sha = check_source(cfg, roles, classes)
    clips, variants, masks = [], [], []
    for c in cfg["clips"]:
        cl = load_clip(c, cfg["pins"])
        vs, mask = make_variants(cl["A"], cl["B"])
        check_common_raw(cl["A"], cl["B"], vs, mask)
        clips.append(cl)
        variants.append(vs)
        masks.append(mask)
    return {"cfg": cfg, "config_path": cfg_path, "config_sha": cfg_sha, "root": root, "unit": unit,
            "out": out, "pins": dict(cfg["pins"]), "roles": roles, "classes": classes,
            "source_sha": source_sha, "clips": clips, "variants": variants, "masks": masks,
            "auth": auth, "now": now}


# ---------------------------------------------------------------- scoring

def score_logits(logits, classes, label):
    """Full93. The label enters only here, after the forward."""
    arr = np.asarray(logits, dtype=np.float64)
    t = classes.index(label)
    top = int(np.argmax(arr))
    other = arr.copy()
    other[t] = -np.inf
    bo = int(np.argmax(other))
    return {"true_label": label, "top1": classes[top], "top1_index": top, "correct": top == t,
            "true_rank": int(1 + (arr > arr[t]).sum()), "best_other": classes[bo],
            "true_vs_best_other_margin": float(arr[t] - arr[bo])}


def input_record(inp):
    x = inp["x"]
    return {"x_sha256": typed_sha256(x), "x_shape": list(x.shape), "x_dtype": x.dtype.name,
            "frames": int(inp["frames"]), "body_scale": jnum(inp["body_scale"]),
            "hand_scales": {k: jnum(v) for k, v in inp["hand_scales"].items()},
            "body_valid_sum": float(np.sum(inp["body_valid"])),
            "hand_valid_sum": float(np.sum(inp["hand_valid"]))}


def input_hashes(inp):
    return {"pad_mask_sha256": typed_sha256(inp["pad"]),
            "body_valid_sha256": typed_sha256(inp["body_valid"]),
            "hand_valid_sha256": typed_sha256(inp["hand_valid"]),
            "x_valid_channels_sha256": typed_sha256(inp["x"][..., VALID_CHANNELS])}


def block_changes(x, ref):
    need(x.shape == ref.shape, "blocks", "shape differs")
    out = {}
    for name, (a, b) in BLOCKS.items():
        d = np.abs(x[..., a:b].astype(np.float64) - ref[..., a:b].astype(np.float64))
        out[name] = {"changed_values": int((d != 0).sum()), "max_abs_diff": float(d.max())}
    return out


def count_tests(path):
    return len(re.findall(r"^\s+def test_\w+\(", Path(path).read_text(encoding="utf-8"), re.M))


def check_receipt(receipt_path, root, code_dir):
    need(receipt_path, "receipt", "--run requires explicit --test-receipt PATH")
    p = local_path(str(receipt_path), root, "receipt")
    doc, sha = read_json(p)
    runner, test = code_dir / "mechanism_pair.py", code_dir / "test_mechanism.py"
    expect = {"schema": RECEIPT_SCHEMA, "status": "PASS", "runner_sha256": sha256_file(runner),
              "test_sha256": sha256_file(test), "tests_run": count_tests(test),
              "failures": 0, "errors": 0, "skipped": 0}
    need(expect["tests_run"] > 0, "receipt", "no tests found")
    for k, v in expect.items():
        need(isinstance(doc, dict) and doc.get(k) == v, "receipt", f"{k}: {doc.get(k)!r} != {v!r}")
    return {"path": str(p), "sha256": sha, "content": doc, "runner": str(runner), "test": str(test)}


# ---------------------------------------------------------------- official backend (run mode only)

class TorchBackend:
    """Pinned pilot preprocessing + MaskedSignTCN. Imported lazily; nothing else from pilot."""

    def __init__(self, roles, pins, model_cfg):
        files = {r: Path(roles[r]) for r in ("pilot_data", "pilot_mapping", "pilot_model")}
        need(files["pilot_data"].name == "data.py" and files["pilot_mapping"].name == "web_input.py"
             and files["pilot_model"].name == "model_masked_v1.py", "backend", "pilot module names")
        pkg = files["pilot_data"].parent
        need(pkg.name == "pilot" and all(same_path(f.parent, pkg) for f in files.values()), "backend",
             "pilot modules must share one pilot package")
        need("pilot" not in sys.modules, "backend", "pilot already imported")
        self.files = files
        self.pins = {str(files[r]): pins[roles[r]] for r in files}     # keyed as used below
        sys.path.insert(0, str(pkg.parent))
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.web_input = importlib.import_module("pilot.web_input")
        self.data = importlib.import_module("pilot.data")
        mm = importlib.import_module("pilot.model_masked_v1")
        self._check_modules({"pilot_mapping": self.web_input, "pilot_data": self.data, "pilot_model": mm})
        need(self.data.TOTAL_CHANNELS == N_CHANNELS, "backend", "channel count")
        with open(model_cfg["checkpoint_path"], "rb") as fh:
            raw = fh.read()
        need(sha256_bytes(raw) == model_cfg["checkpoint_sha256"], "backend", "checkpoint bytes changed")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        need(list(payload["classes"]) == model_cfg["classes"] and payload["n_classes"] == N_CLASSES,
             "backend", "checkpoint classes/n_classes")
        self.constructor = {k: payload[k] for k in ("width", "dilations", "kernel_size", "dropout", "n_classes")}
        self.model = mm.MaskedSignTCN(in_channels=self.data.TOTAL_CHANNELS, width=payload["width"],
                                      dilations=tuple(payload["dilations"]), kernel_size=payload["kernel_size"],
                                      dropout=payload["dropout"], n_classes=payload["n_classes"])
        self.model.load_state_dict(payload["state_dict"], strict=True)
        self.model.eval()
        self.classes = list(payload["classes"])
        self.payload_digest = {k: typed_sha256(v.detach().cpu().numpy()) for k, v in payload["state_dict"].items()}
        need(self.state_digest() == self.payload_digest, "backend", "loaded state differs from payload")
        self.checkpoint_sha = model_cfg["checkpoint_sha256"]

    def _check_modules(self, mods):
        for role, mod in mods.items():
            need(same_path(os.path.realpath(mod.__file__), self.files[role]), "backend",
                 f"{role} imported from {mod.__file__}")
            need(sha256_file(mod.__file__) == self.pins[str(self.files[role])], "backend", f"{role} hash")

    def build(self, obs):
        xy, present = self.web_input.to_native_like(obs["xy_px"], obs["usable"])
        ch = self.data.build_channels(xy, present, obs["timestamp_us"])
        ch["label"] = 0                                   # collate needs the field; never informative
        x, pad, _ = self.data.collate([ch])
        return {"x": x, "pad": pad, "body_valid": ch["body_valid"], "hand_valid": ch["hand_valid"],
                "body_scale": ch["body_scale"], "hand_scales": ch["hand_scales"], "frames": ch["frames"]}

    def forward(self, x, pad):
        with self.torch.inference_mode():
            out = self.model(self.torch.from_numpy(x), self.torch.from_numpy(pad))
        need(tuple(out.shape) == (1, N_CLASSES), "forward", f"logit shape {tuple(out.shape)}")
        return [float(v) for v in out[0].numpy()]

    def state_digest(self):
        return {k: typed_sha256(v.detach().cpu().numpy()) for k, v in self.model.state_dict().items()}

    def audit(self):
        loaded = [m for m in FORBIDDEN_MODULES if m in sys.modules]
        need(not loaded, "runtime", f"forbidden modules imported: {loaded}")
        return {"python": sys.version, "numpy": np.__version__, "torch": self.torch.__version__,
                "torch_threads": self.torch.get_num_threads(), "model_training": self.model.training,
                "grad_mode": "inference_mode", "constructor": self.constructor,
                "checkpoint_sha256": self.checkpoint_sha, "state_tensors": len(self.payload_digest),
                "modules": {r: {"path": str(f), "sha256": self.pins[str(f)]} for r, f in self.files.items()},
                "forbidden_modules_loaded": loaded}

    def final_audit(self):
        need(not self.model.training, "post", "model left eval mode")
        need(self.state_digest() == self.payload_digest, "post", "model state differs from checkpoint")
        for role, f in self.files.items():
            need(sha256_file(f) == self.pins[str(f)], "post", f"{role} changed")
        return {"model_training": self.model.training, "state_equals_checkpoint": True}


# ---------------------------------------------------------------- run

def _write_json(path, obj):
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, allow_nan=False)


def _save_npz(path, arrays):
    with open(path, "xb") as fh:
        np.savez(fh, **arrays)


def _summary_md(result):
    lines = ["# Capture cadence mechanism diagnostic (self-run, not independently verified)", ""]
    lines += [f"- {s}" for s in LIMITS]
    lines += ["", f"Execution counts: `{json.dumps(result['execution_counts'])}`", ""]
    for w in result["windows"]:
        lines += [f"## clip {w['clip_index']} `{w['clip_id']}` label `{w['label']}` "
                  f"[{w['start_us']}, {w['end_us_exclusive']}) rows={w['rows']} "
                  f"common_true={w['common_mask_true']}/{w['common_mask_size']}", "",
                  "| condition | status | pose | hands | top1 | correct | true_rank | margin | body_scale | hand L | hand R |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in w["records"]:
            s, i = r["score"], r["input"]
            lines.append(f"| {r['condition']} | {r['status']} | {r['donors']['pose']} | {r['donors']['hands']} "
                         f"| {s['top1']} | {s['correct']} | {s['true_rank']} | {s['true_vs_best_other_margin']:.6g} "
                         f"| {i['body_scale']} | {i['hand_scales'].get('left')} | {i['hand_scales'].get('right')} |")
        lines.append("")
    return "\n".join(lines)


def run(ctx, receipt_path, backend=None, code_dir=None):
    code_dir = Path(code_dir or Path(__file__).resolve().parent)
    receipt = check_receipt(receipt_path, ctx["root"], code_dir)
    cfg, out, classes = ctx["cfg"], ctx["out"], ctx["classes"]
    try:
        os.mkdir(out)
    except FileExistsError:
        raise Fail("output", f"output exists, refusing overwrite: {out}") from None
    counts = {"baseline_input_rechecks": 0, "baseline_forwards": 0, "new_input_builds": 0,
              "new_forward_calls": 0}
    stage = "start"
    try:
        _write_json(out / "RUN_STARTED.json", {
            "unit": UNIT_NAME, "started_kst": dt.datetime.now(KST).isoformat(),
            "config_sha256": ctx["config_sha"], "runner_sha256": receipt["content"]["runner_sha256"],
            "test_sha256": receipt["content"]["test_sha256"], "test_receipt_sha256": receipt["sha256"],
            "resume_authorization": ctx["auth"], "planned_new_forward_calls": 16, "baseline_forwards": 0})
        stage = "runtime"
        be = backend if backend is not None else TorchBackend(ctx["roles"], ctx["pins"], cfg["model"])
        need(list(be.classes) == classes, stage, "model class order differs from config")
        runtime = be.audit()
        _write_json(out / "RUNTIME.json", runtime)

        stage = "baseline_inputs"                          # rebuild only; zero baseline forwards
        base = {}
        for ci, clip in enumerate(ctx["clips"]):
            for arm in "AB":
                inp = be.build({k: clip[arm][k].copy() for k in OBS})
                counts["baseline_input_rechecks"] += 1
                rec, exp = input_record(inp), cfg["clips"][ci]["arms"][arm]["baseline_input"]
                for k in BASE_KEYS:
                    need(rec[k] == exp[k], stage, f"clip {ci} {arm} {k}: rebuilt {rec[k]!r} != prior {exp[k]!r}")
                base[ci, arm] = inp

        stage = "new_inputs"                               # all 16 built and validated before any forward
        new = {}
        for ci, clip in enumerate(ctx["clips"]):
            rows = cfg["clips"][ci]["expected_window_rows"]
            for v in VARIANTS:
                obs = ctx["variants"][ci][v]
                arrays = dict(obs)
                if DONORS[v]["mask"]:
                    arrays["common_mask"] = ctx["masks"][ci]
                _save_npz(out / f"clip{ci}_{v}.npz", arrays)
                inp = be.build({k: obs[k].copy() for k in OBS})
                counts["new_input_builds"] += 1
                x = inp["x"]
                need(x.shape == (1, rows, N_CHANNELS) and x.dtype == np.float32 and np.isfinite(x).all()
                     and inp["frames"] == rows and inp["pad"].shape == (1, rows) and (inp["pad"] == 1).all(),
                     stage, f"clip {ci} {v} input invalid")
                new[ci, v] = inp
            a, b = new[ci, "common_A"], new[ci, "common_B"]
            need(same(a["body_valid"], b["body_valid"]) and same(a["hand_valid"], b["hand_valid"])
                 and same(a["x"][..., VALID_CHANNELS], b["x"][..., VALID_CHANNELS]), stage,
                 f"clip {ci} common_A/common_B final validity channels differ")

        stage = "forward"
        state0 = be.state_digest()
        logits = {}
        for ci in range(len(ctx["clips"])):
            for v in VARIANTS:
                inp = new[ci, v]
                lg = [float(z) for z in be.forward(inp["x"].copy(), inp["pad"].copy())]
                counts["new_forward_calls"] += 1
                need(len(lg) == N_CLASSES and all(math.isfinite(z) for z in lg), stage, f"clip {ci} {v} logits")
                logits[ci, v] = lg
        need(be.state_digest() == state0, stage, "model state changed during inference")
        need(counts["new_forward_calls"] == 16 and len(logits) == 16, stage, "forward count != 16")

        stage = "score"                                    # labels are read only from here on
        windows, n_records = [], 0
        for ci, cc in enumerate(cfg["clips"]):
            mask = ctx["masks"][ci]
            records = []
            for arm in "AB":
                inp, src = base[ci, arm], cc["arms"][arm]
                records.append({"condition": arm, "status": "REUSED", "donors": {"pose": arm, "hands": arm},
                                "logits93": src["baseline_logits93"],
                                "score": {k: src["baseline_score"][k] for k in SCORE_KEYS},
                                "input": input_record(inp), "hashes": input_hashes(inp)})
            for v in VARIANTS:
                inp = new[ci, v]
                records.append({"condition": v, "status": "INFERRED", "donors": DONORS[v],
                                "logits93": logits[ci, v], "score": score_logits(logits[ci, v], classes, cc["label"]),
                                "input": input_record(inp), "hashes": input_hashes(inp),
                                "block_changes_vs_A": block_changes(inp["x"], base[ci, "A"]["x"]),
                                "block_changes_vs_B": block_changes(inp["x"], base[ci, "B"]["x"])})
            n_records += len(records)
            windows.append({"clip_index": ci, "clip_id": cc["clip_id"], "label": cc["label"],
                            "start_us": cc["start_us"], "end_us_exclusive": cc["end_us_exclusive"],
                            "rows": cc["expected_window_rows"], "window_frame_ids": cc["window_frame_ids"],
                            "window_timestamp_us": cc["window_timestamp_us"],
                            "decoded_pts_sha256": typed_sha256(ctx["clips"][ci]["decoded_pts"]),
                            "common_mask_sha256": typed_sha256(mask), "common_mask_true": int(mask.sum()),
                            "common_mask_size": int(mask.size), "records": records})
        need(n_records == 24, stage, "score record count != 24")

        stage = "post_checks"
        need(sha256_file(ctx["config_path"]) == ctx["config_sha"], stage, "config changed")
        for p, h in ctx["pins"].items():
            need(sha256_file(p) == h, stage, f"pinned file changed: {p}")
        need(sha256_file(receipt["runner"]) == receipt["content"]["runner_sha256"]
             and sha256_file(receipt["test"]) == receipt["content"]["test_sha256"], stage, "code changed")
        if ctx["auth"]:
            need(sha256_file(ctx["auth"]["path"]) == ctx["auth"]["sha256"], stage, "resume authorization changed")
        post = be.final_audit()

        stage = "write"
        result = {
            "schema": RESULT_SCHEMA, "status": "RUN_COMPLETE_NOT_INDEPENDENTLY_VERIFIED", "unit": UNIT_NAME,
            "config": {"path": str(ctx["config_path"]), "sha256": ctx["config_sha"],
                       "deadline_kst": cfg["deadline_kst"]},
            "resume_authorization": ctx["auth"], "test_receipt": receipt,
            "source_result_sha256": ctx["source_sha"], "pins": ctx["pins"], "classes": classes,
            "execution_counts": counts, "donor_mapping": DONORS, "runtime": runtime, "post_audit": post,
            "score_record_count": n_records, "windows": windows, "interpretation_limits": list(LIMITS)}
        _write_json(out / "RESULT.json", result)
        with open(out / "SUMMARY.md", "x", encoding="utf-8") as fh:
            fh.write(_summary_md(result))
        hashes = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                  for p in sorted(out.iterdir()) if p.is_file()}
        _write_json(out / "FILE_HASHES.json", hashes)
        return result
    except BaseException as exc:
        try:
            _write_json(out / "FAILURE.json", {
                "status": "TECHNICAL_FAILURE", "stage": stage, "error_type": type(exc).__name__,
                "error": str(exc), "execution_counts": counts,
                "note": "Not an accuracy result. No other condition was retried."})
        except OSError:
            pass
        raise


def main(argv=None):
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(here / "PREPARED_CONFIG.json"))
    ap.add_argument("--resume-authorization")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--test-receipt")
    a = ap.parse_args(argv)
    try:
        need(same_path(Path(os.path.abspath(a.config)).parent, here), "cli", "runner must sit beside the config")
        ctx = validate(a.config, a.resume_authorization)
        if not a.run:
            print(json.dumps({"status": "VALIDATION_PASS", "config_sha256": ctx["config_sha"],
                              "resume_authorization": ctx["auth"] and ctx["auth"]["sha256"],
                              "clips": [{"clip_id": c["clip_id"], "rows": c["expected_window_rows"],
                                         "common_mask_true": int(m.sum())}
                                        for c, m in zip(ctx["cfg"]["clips"], ctx["masks"])],
                              "note": "no model import, no writes; add --run --test-receipt PATH"}, indent=1))
            return 0
        res = run(ctx, a.test_receipt, code_dir=here)
        print(json.dumps({"status": res["status"], "execution_counts": res["execution_counts"]}, indent=1))
        return 0
    except Exception as exc:                              # noqa: BLE001 - reported, exit nonzero
        print(f"FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
