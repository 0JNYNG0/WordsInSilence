"""VIDEO tracking-state lifetime pair (see FROZEN_PLAN.md).

A_fresh  : new hand+pose VIDEO landmarker per clip (closed at clip end).
B_session: one hand+pose VIDEO landmarker per signer group, reused across clips.

Both arms receive the same immutable RGB from ONE streaming decode per clip and the
same absolute timestamp (group offset + original relative PTS). Observations keep the
original relative PTS / decoded frame id. No even selection, no active span, no
interpolation, no training, no camera.

CLI
  python state_lifetime_pair.py --config CFG                  preflight only (hashes/config)
  python state_lifetime_pair.py --config CFG --check-bindings print pinned helper signatures
  python state_lifetime_pair.py --config CFG --run            real run (needs TEST_RECEIPT.json PASS)
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "wis_state_lifetime_pair_v1"
RESULT_SCHEMA = "wis_state_lifetime_pair_result_v1"
SCORES_SCHEMA = "wis_state_lifetime_pair_scores_v1"
ARMS = ("A_fresh", "B_session")
ARM_SHORT = {"A_fresh": "A", "B_session": "B"}
GAP_US = 1_000_000
OBS_FIELD_COUNT = 7
N_CLASSES = 93
TOLERANCE = 1e-5
CPU_THREADS = 2
OWN_CODE = ("state_lifetime_pair.py", "test_state_lifetime.py", "independent_verify.py")
RECEIPT_BOUND = ("state_lifetime_pair.py", "test_state_lifetime.py")
RECEIPT_NAME = "TEST_RECEIPT.json"
COUNTER_KEYS = ("decoded_frames", "process_calls", "landmarkers", "internal_tasks",
                "model_loads", "main_forwards")
FROZEN_GROUPS = [
    ["NIA_SL_SEN0254_REAL01_F", "NIA_SL_SEN0354_REAL01_F",
     "NIA_SL_SEN0356_REAL01_F", "NIA_SL_SEN0408_REAL01_F"],
    ["NIA_SL_SEN0356_REAL02_F", "NIA_SL_SEN0408_REAL02_F"],
]
FROZEN_BUDGETS = {"decoded_frames": 853, "process_calls": 1706, "landmarkers": 8,
                  "internal_tasks": 16, "model_loads": 1, "main_forwards": 8,
                  "independent_forwards": 8}
LIMITATIONS = [
    "Separate public videos of a known training signer stitched with a fixed 1,000,000us "
    "diagnostic gap; not a continuous camera stream, not webcam, not generalization.",
    "Per-frame process timing is offline diagnostic only; not real-camera throughput.",
    "Observation change alone is not an accuracy improvement nor a reset justification.",
]

# ---------------------------------------------------------------------------
# Pinned-interface assumptions. The pinned sources were NOT readable by the author;
# every name/shape below is UNVERIFIED. Check with --check-bindings + source review.
# All of them are confined to PinnedRuntime.
# ---------------------------------------------------------------------------
DECODED_FRAME_ATTRS = ("frame_id", "timestamp_us", "rgb")  # items of cp.strict_frames(...)
HIP_ROW_TASK_MS_KEYS = ("hand_ms", "pose_ms")              # hip Instr.frames row keys
HIP_TASKS_ATTR = "tasks"                                   # Instr attr holding TaskProxy objs
WINDOW_KEYS = ("x", "pad", "body_scale", "hand_scales")    # cp.prepare_window mapping keys


class StageError(RuntimeError):
    def __init__(self, stage, msg):
        super().__init__(f"[{stage}] {msg}")
        self.stage = stage


class TechnicalGateError(StageError):
    pass


def require(cond, stage, msg):
    if not cond:
        raise StageError(stage, msg)


# ---------------------------------------------------------------------------
# hashing / json
# ---------------------------------------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return jsonable(dataclasses.asdict(o))
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def write_json_new(path, obj):
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(jsonable(obj), fh, ensure_ascii=False, indent=1, allow_nan=True)


def write_npz_new(path, arrays):
    with open(path, "xb") as fh:
        np.savez(fh, **arrays)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# config / pins / receipt
# ---------------------------------------------------------------------------
def signer_of(clip_id):
    parts = clip_id.split("_")
    if len(parts) < 4 or not parts[3].startswith("REAL"):
        raise StageError("config", f"cannot derive signer from {clip_id}")
    return parts[3]


def validate_config(cfg, frozen=True):
    st = "config"
    require(cfg.get("schema") == SCHEMA, st, "schema mismatch")
    clips = cfg["clips"]
    ids = [c["clip_id"] for c in clips]
    require(len(set(ids)) == len(ids), st, "duplicate clip ids")
    flat = [cid for g in cfg["groups"] for cid in g]
    require(sorted(flat) == sorted(ids) and len(flat) == len(set(flat)), st,
            "groups must cover every clip exactly once")
    signers = []
    for g in cfg["groups"]:
        require(len(g) >= 1, st, "empty group")
        s = {signer_of(c) for c in g}
        require(len(s) == 1, st, f"group mixes signers {s}")
        signers.append(s.pop())
    require(len(set(signers)) == len(signers), st, "one session per signer")
    require(cfg["gap_us"] == GAP_US, st, "gap_us must be 1000000")
    require(tuple(cfg["arms"]) == ARMS, st, "arms mismatch")
    pins = cfg["pins"]
    for c in clips:
        require(pins.get(c["path"]) == c["sha256"], st, f"clip pin mismatch {c['clip_id']}")
        require(int(c["expected_frames_from_prior"]) > 0, st, "expected frames")
    for key in (cfg["parent_cadence"]["module_path"], cfg["parent_cadence"]["config_path"],
                cfg["instrument_module"], cfg["independent_score_reference"],
                cfg["independent_raw_reference"], cfg["scoring"]["checkpoint_path"]):
        require(key in pins, st, f"not pinned: {key}")
    sc = cfg["scoring"]
    require(pins[sc["checkpoint_path"]] == sc["checkpoint_sha256"], st, "checkpoint pin")
    require(sc["scope"] == "full93", st, "scope must be full93")
    require(float(sc["max_abs_logit_tolerance"]) == TOLERANCE, st, "tolerance must be 1e-5")
    require(int(sc["cpu_threads"]) == CPU_THREADS, st, "cpu_threads must be 2")
    scored = [c for c in clips if c.get("score")]
    for c in scored:
        s = c["score"]
        require(isinstance(s["label"], str) and s["label"], st, "label")
        require(int(s["start_us"]) < int(s["end_us_exclusive"]), st, "window order")
    pos = {cid: (gi, p) for gi, g in enumerate(cfg["groups"]) for p, cid in enumerate(g)}
    n_frames = sum(int(c["expected_frames_from_prior"]) for c in clips)
    n_lm = len(clips) + len(cfg["groups"])
    derived = {"decoded_frames": n_frames, "process_calls": 2 * n_frames,
               "landmarkers": n_lm, "internal_tasks": 2 * n_lm, "model_loads": 1,
               "main_forwards": 2 * len(scored), "independent_forwards": 2 * len(scored)}
    require(cfg["budgets"] == derived, st, f"budgets {cfg['budgets']} != derived {derived}")
    if frozen:
        require(cfg["groups"] == FROZEN_GROUPS, st, "groups/order differ from FROZEN_PLAN")
        require(derived == FROZEN_BUDGETS, st, "budgets differ from FROZEN_PLAN")
    return {
        "n_clips": len(clips), "n_groups": len(cfg["groups"]), "signers": signers,
        "scored": [c["clip_id"] for c in scored],
        "history_applies": [c["clip_id"] for c in scored if pos[c["clip_id"]][1] > 0],
        "controls_group_first": [g[0] for g in cfg["groups"]],
        "budgets": derived,
    }


def hash_pins(cfg):
    actual, bad = {}, []
    for path, exp in cfg["pins"].items():
        try:
            actual[path] = sha256_file(path)
        except OSError as e:
            actual[path] = f"ERROR:{type(e).__name__}"
        if actual[path] != exp:
            bad.append(path)
    return {"actual": actual, "mismatches": bad, "ok": not bad}


def own_code_hashes(here=HERE):
    return {n: sha256_file(os.path.join(here, n)) for n in OWN_CODE}


def preflight(cfg_path, frozen=True):
    cfg = load_json(cfg_path)
    summary = validate_config(cfg, frozen=frozen)
    return cfg, {
        "config_path": os.path.abspath(cfg_path),
        "config_sha256": sha256_file(cfg_path),
        "own_code_sha256": own_code_hashes(),
        "pins": hash_pins(cfg),
        "summary": summary,
    }


def check_receipt(cfg_path, receipt_path=None, here=HERE):
    receipt_path = receipt_path or os.path.join(here, RECEIPT_NAME)
    st = "receipt"
    require(os.path.isfile(receipt_path), st, f"missing {receipt_path}")
    rec = load_json(receipt_path)
    require(rec.get("status") == "PASS", st, "receipt status is not PASS")
    require(int(rec.get("tests_run", 0)) > 0, st, "tests_run must be > 0")
    require(int(rec.get("failures", 1)) == 0 and int(rec.get("errors", 1)) == 0, st,
            "receipt has failures/errors")
    for name in RECEIPT_BOUND:
        require(rec.get("code_sha256", {}).get(name) == sha256_file(os.path.join(here, name)),
                st, f"receipt not bound to current {name}")
    require(rec.get("config_sha256") == sha256_file(cfg_path), st,
            "receipt not bound to this config")
    return rec


def import_pinned(mod_name, path, pins):
    st = "import"
    require(path in pins, st, f"not pinned: {path}")
    require(sha256_file(path) == pins[path], st, f"pin mismatch before import: {path}")
    require(mod_name not in sys.modules, st, f"module name already loaded: {mod_name}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    require(sha256_file(path) == pins[path], st, f"pin mismatch after import: {path}")
    return mod


# ---------------------------------------------------------------------------
# pinned runtime binding (run only; heavy imports happen here)
# ---------------------------------------------------------------------------
class PinnedRuntime:
    """Adapter over approved pinned helpers. Never calls cp main/run/check_deadline,
    cp.extract_arm, hip.gate_diag or hip timing helpers."""

    def __init__(self, cfg):
        pins = cfg["pins"]
        pc = cfg["parent_cadence"]
        self.cp = import_pinned("_wis_slp_cadence_pair", pc["module_path"], pins)
        require(sha256_file(pc["config_path"]) == pins[pc["config_path"]], "import",
                "parent config pin")
        oldcfg = load_json(pc["config_path"])
        self.plan = self.cp.validate_config(oldcfg, self.cp.DEFAULT_ROOTS)
        self.deps = self.cp.load_runtime_deps(self.plan, self.cp.DEFAULT_ROOTS, self.cp.sha256_file)
        self.hip = import_pinned("_wis_slp_hand_image_pair", cfg["instrument_module"], pins)
        fac = self.deps.landmarker_factory
        self.lm_mod = sys.modules[fac.__module__]
        oc = self.cp.observation_cls
        self.observation_cls = oc if isinstance(oc, type) else oc()
        self.obs_fields = tuple(f.name for f in dataclasses.fields(self.observation_cls))
        require(len(self.obs_fields) == OBS_FIELD_COUNT, "bind",
                f"expected {OBS_FIELD_COUNT} Observation fields, got {self.obs_fields}")
        require({"timestamp_us", "frame_id"} <= set(self.obs_fields), "bind",
                "Observation lacks timestamp_us/frame_id")
        self.ckpt = cfg["scoring"]["checkpoint_path"]
        self.ckpt_sha = cfg["scoring"]["checkpoint_sha256"]
        self.task_ms_keys = HIP_ROW_TASK_MS_KEYS

    def signatures(self):
        def sig(f):
            try:
                return str(inspect.signature(f))
            except (TypeError, ValueError) as e:
                return f"<unavailable: {e}>"
        names = {
            "cp.validate_config": self.cp.validate_config,
            "cp.load_runtime_deps": self.cp.load_runtime_deps,
            "cp.open_decoder": self.cp.open_decoder, "cp.strict_frames": self.cp.strict_frames,
            "cp.empty_frames": self.cp.empty_frames, "cp.prepare_window": self.cp.prepare_window,
            "cp.compute_metrics": self.cp.compute_metrics,
            "cp.rank_against_label": self.cp.rank_against_label,
            "deps.landmarker_factory": self.deps.landmarker_factory,
            "deps.load_model": self.deps.load_model, "deps.forward": self.deps.forward,
            "deps.state_hash": self.deps.state_hash,
            "hip.make_factory": self.hip.make_factory, "hip.Instr": self.hip.Instr,
        }
        return {"signatures": {k: sig(v) for k, v in names.items()},
                "observation_cls": f"{self.observation_cls.__module__}.{self.observation_cls.__qualname__}",
                "obs_fields": list(self.obs_fields), "lm_mod": self.lm_mod.__name__,
                "assumptions": {"DECODED_FRAME_ATTRS": DECODED_FRAME_ATTRS,
                                "HIP_ROW_TASK_MS_KEYS": HIP_ROW_TASK_MS_KEYS,
                                "HIP_TASKS_ATTR": HIP_TASKS_ATTR, "WINDOW_KEYS": WINDOW_KEYS}}

    def iter_clip_frames(self, clip):
        dec = self.cp.open_decoder(clip["path"])
        try:
            for fr in self.cp.strict_frames(dec):
                yield tuple(getattr(fr, a) for a in DECODED_FRAME_ATTRS)
        finally:
            close = getattr(dec, "close", None)
            if callable(close):
                close()

    def new_landmarker(self, tag):
        sink = self.hip.Instr()
        factory = self.hip.make_factory(self.deps.landmarker_factory, self.lm_mod, tag, "VIDEO", sink)
        return factory(), sink

    def tasks_of(self, h):
        return list(getattr(h.instr, HIP_TASKS_ATTR))

    def process(self, h, rgb, absolute_us):
        return h.lm.process(rgb, absolute_us)

    def task_ms(self, row):
        return int(row[self.task_ms_keys[0]]), int(row[self.task_ms_keys[1]])

    def close(self, h):
        err = None
        try:
            h.lm.close()  # production close swallows task errors -> inspect close_rec below
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        recs = []
        for t in self.tasks_of(h):
            cr = getattr(t, "close_rec", None)
            recs.append({"task_obj_id": id(t), "ok": bool(cr is not None and cr.ok),
                         "close_rec": None if cr is None else
                         {k: repr(v) for k, v in vars(cr).items()}})
        return err, recs

    def load_model(self):
        import torch
        torch.set_num_threads(CPU_THREADS)
        require(sha256_file(self.ckpt) == self.ckpt_sha, "score", "checkpoint pin")
        model, payload = self.deps.load_model(self.ckpt)
        model.eval()
        require(not model.training, "score", "model not in eval")
        return model, list(payload["classes"])

    def state_hash(self, model):
        return self.deps.state_hash(model)

    def prepare_window(self, obs, start_us, end_us_exclusive):
        res = self.cp.prepare_window(obs, start_us, end_us_exclusive)
        return {k: res[k] for k in WINDOW_KEYS}

    def forward(self, model, x, pad):
        import torch
        with torch.no_grad():
            t = self.deps.forward(model, x, pad)
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    def rank(self, logits, classes, label):
        return self.cp.rank_against_label(logits, classes, label)

    def clip_metrics(self, obs):
        return {"compute_metrics": self.cp.compute_metrics(obs),
                "empty_frames": self.cp.empty_frames(obs)}


# ---------------------------------------------------------------------------
# landmarker registry (retains every LM/task reference so ids are not recycled)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(eq=False)
class LMHandle:
    key: str
    arm: str
    group: int
    signer: str
    tag: str
    lm: object
    instr: object
    lm_obj_id: int
    clips: list = dataclasses.field(default_factory=list)
    process_calls: int = 0
    created_seq: int = 0
    closed_seq: int | None = None
    close_error: str | None = None
    close_records: list = dataclasses.field(default_factory=list)
    task_refs: list = dataclasses.field(default_factory=list)


class Registry:
    def __init__(self, rt, run):
        self.rt, self.run = rt, run
        self.handles, self.events, self._seq = [], [], 0

    def _event(self, kind, h, clip_id=None):
        self._seq += 1
        last = getattr(h.lm, "_last_ms", None)  # read-only diagnostic; never written
        self.events.append({"seq": self._seq, "event": kind, "key": h.key, "clip_id": clip_id,
                            "last_ms_seen": last if isinstance(last, (int, type(None))) else repr(last)})
        return self._seq

    def create(self, arm, group, signer, tag):
        lm, instr = self.rt.new_landmarker(tag)
        h = LMHandle(key=f"LM{len(self.handles):02d}_{tag}", arm=arm, group=group,
                     signer=signer, tag=tag, lm=lm, instr=instr, lm_obj_id=id(lm))
        self.handles.append(h)
        self.run.counters["landmarkers"] += 1
        h.created_seq = self._event("create", h)
        return h

    def clip_event(self, kind, clip_id, handles):
        for h in handles:
            if kind == "clip_start":
                h.clips.append(clip_id)
            self._event(kind, h, clip_id)

    def close(self, h):
        if h.closed_seq is not None:
            return
        try:
            h.task_refs = self.rt.tasks_of(h)
            h.close_error, h.close_records = self.rt.close(h)
        except Exception as e:  # recorded; require_all_closed_ok() turns it into a failure
            h.close_error = f"{type(e).__name__}: {e}"
        h.closed_seq = self._event("close", h)

    def close_all_open(self):
        for h in self.handles:
            self.close(h)

    def require_all_closed_ok(self):
        bad = [h.key for h in self.handles
               if h.closed_seq is None or h.close_error or len(h.task_refs) != 2
               or len(h.close_records) != 2 or not all(r["ok"] for r in h.close_records)]
        require(not bad, "close", f"landmarkers not cleanly closed: {bad}")

    def n_tasks(self):
        return sum(len(h.task_refs) for h in self.handles)

    def to_json(self):
        return {"landmarkers": [{
            "key": h.key, "arm": h.arm, "group": h.group, "signer": h.signer, "tag": h.tag,
            "lm_obj_id": h.lm_obj_id, "clips": h.clips, "process_calls": h.process_calls,
            "created_seq": h.created_seq, "closed_seq": h.closed_seq,
            "close_error": h.close_error, "close_records": h.close_records,
            "task_obj_ids": [id(t) for t in h.task_refs],
            "close_ok": bool(h.closed_seq is not None and not h.close_error
                             and len(h.close_records) == 2
                             and all(r["ok"] for r in h.close_records)),
        } for h in self.handles], "events": self.events}


# ---------------------------------------------------------------------------
# observations
# ---------------------------------------------------------------------------
def relabel(out, relative_us, frame_id):
    """Observation keeps original relative PTS + decoded id (MediaPipe got absolute)."""
    if not (dataclasses.is_dataclass(out) and not isinstance(out, type)):
        raise StageError("observation", f"process() returned non-dataclass {type(out).__name__}")
    return dataclasses.replace(out, timestamp_us=relative_us, frame_id=frame_id)


def validate_obs(rt, ob, fields):
    require(isinstance(ob, rt.observation_cls), "observation",
            f"unexpected observation type {type(ob).__name__}")
    require(tuple(f.name for f in dataclasses.fields(ob)) == tuple(fields), "observation",
            "observation fields changed")
    for f in fields:
        require(np.asarray(getattr(ob, f)).dtype != object, "observation", f"object dtype in {f}")


def stack_obs(obs, fields):
    arrays = {}
    for f in fields:
        vals = [np.asarray(getattr(o, f)) for o in obs]
        require(len({(v.shape, v.dtype.str) for v in vals}) == 1, "observation",
                f"inconsistent shape/dtype in field {f}")
        arrays[f"obs_{f}"] = np.stack(vals)
    return arrays


def obs_bitexact_diff(obs_a, obs_b, fields):
    if len(obs_a) != len(obs_b):
        return [{"local": None, "field": "<length>", "A": len(obs_a), "B": len(obs_b)}]
    mism = []
    for i, (a, b) in enumerate(zip(obs_a, obs_b)):
        for f in fields:
            va, vb = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
            if va.dtype != vb.dtype or va.shape != vb.shape or va.tobytes() != vb.tobytes():
                mism.append({"local": i, "field": f})
    return mism


def field_diff(arr_a, arr_b):
    """A/B difference summary for one stacked field (first axis = frames)."""
    n = len(arr_a)
    rec = {"n_frames": n}
    if n == 0:
        return rec
    ba = np.ascontiguousarray(arr_a).reshape(n, -1).view(np.uint8)
    bb = np.ascontiguousarray(arr_b).reshape(n, -1).view(np.uint8)
    rec["frames_bitwise_differ"] = int((ba != bb).any(axis=1).sum())
    if np.issubdtype(arr_a.dtype, np.floating):
        na, nb = np.isnan(arr_a), np.isnan(arr_b)
        rec["nan_mask_differ_frames"] = int((na != nb).reshape(n, -1).any(axis=1).sum())
        both = ~na & ~nb
        d = np.abs(arr_a.astype(np.float64) - arr_b.astype(np.float64))[both]
        rec["max_abs_diff_finite"] = float(d.max()) if d.size else 0.0
    elif arr_a.dtype == np.bool_:
        rec["true_A"], rec["true_B"] = int(arr_a.sum()), int(arr_b.sum())
        rec["elements_differ"] = int((arr_a != arr_b).sum())
    return rec


def half_open(ids, ts, start_us, end_us_exclusive):
    return [(int(i), int(t)) for i, t in zip(ids, ts) if start_us <= t < end_us_exclusive]


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
class RunState:
    def __init__(self):
        self.stage = "init"
        self.counters = {k: 0 for k in COUNTER_KEYS}
        self.registry = None
        self.pairs = {}
        self.gates = {}


@dataclasses.dataclass
class ClipPair:
    clip_id: str
    group: int
    position: int
    signer: str
    offset_us: int
    frames: list
    obs: dict
    diag: dict
    lm_keys: dict
    instr_slices: dict
    first_absolute_us: int = 0
    last_absolute_us: int = 0


def extract_clip_pair(rt, run, clip, group, position, signer, offset_us, A, B):
    fields = rt.obs_fields
    handles = {"A_fresh": A, "B_session": B}
    starts = {arm: len(h.instr.frames) for arm, h in handles.items()}
    frames, obs = [], {arm: [] for arm in ARMS}
    prev_id = prev_pts = None
    for frame_id, pts_us, rgb in rt.iter_clip_frames(clip):
        run.counters["decoded_frames"] += 1
        frame_id, rel_us = int(frame_id), int(pts_us)
        if not frames and position == 0:
            require(rel_us == 0, "clock", f"group first clip must start at relative 0, got {rel_us}")
        if prev_id is not None:
            require(frame_id > prev_id and rel_us > prev_pts, "clock",
                    f"non-increasing id/pts at {frame_id}/{rel_us}")
        abs_us = offset_us + rel_us
        img = np.array(rgb, copy=True, order="C")
        img.setflags(write=False)
        d0 = sha256_array(img)
        rec = {"local_row": len(frames), "frame_id": frame_id, "pts": int(pts_us),
               "relative_us": rel_us, "absolute_us": abs_us, "rgb_sha256": d0,
               "rgb_shape": list(img.shape)}
        for arm in ARMS:
            h = handles[arm]
            n0 = len(h.instr.frames)
            t0 = time.perf_counter_ns()
            out = rt.process(h, img, abs_us)
            dt = time.perf_counter_ns() - t0
            run.counters["process_calls"] += 1
            h.process_calls += 1
            require(sha256_array(img) == d0, "rgb_integrity", f"RGB changed during {arm} {frame_id}")
            require(len(h.instr.frames) == n0 + 1, "instrument",
                    f"{arm} expected one instrument row per process")
            hand_ms, pose_ms = rt.task_ms(h.instr.frames[-1])
            require(hand_ms == abs_us // 1000 and pose_ms == abs_us // 1000, "clock",
                    f"{arm} task ms hand={hand_ms} pose={pose_ms} != {abs_us // 1000} (clamp?)")
            ob = relabel(out, rel_us, frame_id)
            validate_obs(rt, ob, fields)
            obs[arm].append(ob)
            rec[arm] = {"hand_ms": hand_ms, "pose_ms": pose_ms, "process_ns": dt}
        frames.append(rec)
        prev_id, prev_pts = frame_id, rel_us
    require(len(frames) == int(clip["expected_frames_from_prior"]), "frame_count",
            f"{clip['clip_id']} decoded {len(frames)} != {clip['expected_frames_from_prior']}")
    diag, slices = {}, {}
    for arm, h in handles.items():
        s, e = starts[arm], len(h.instr.frames)
        require(e - s == len(frames), "instrument", f"{arm} slice length")
        slices[arm] = [s, e]
        diag[arm] = [{"clip_id": clip["clip_id"], "signer": signer, "group": group, "arm": arm,
                      "offset_us": offset_us, "local_row": k, "frame_id": frames[k]["frame_id"],
                      "relative_us": frames[k]["relative_us"],
                      "absolute_us": frames[k]["absolute_us"], "row": row}
                     for k, row in enumerate(h.instr.frames[s:e])]
    return ClipPair(clip_id=clip["clip_id"], group=group, position=position, signer=signer,
                    offset_us=offset_us, frames=frames, obs=obs, diag=diag,
                    lm_keys={arm: h.key for arm, h in handles.items()}, instr_slices=slices,
                    first_absolute_us=frames[0]["absolute_us"],
                    last_absolute_us=frames[-1]["absolute_us"])


def persist_clip(out_dir, pair, fields):
    for arm in ARMS:
        arrays = stack_obs(pair.obs[arm], fields)
        fr = pair.frames
        arrays.update({
            "decoded_frame_id": np.array([r["frame_id"] for r in fr], dtype=np.int64),
            "timestamp_us": np.array([r["relative_us"] for r in fr], dtype=np.int64),
            "absolute_us": np.array([r["absolute_us"] for r in fr], dtype=np.int64),
            "pts": np.array([r["pts"] for r in fr], dtype=np.int64),
            "rgb_sha256": np.array([r["rgb_sha256"] for r in fr], dtype="<U64"),
            "rgb_shape": np.array([r["rgb_shape"] for r in fr], dtype=np.int64),
            "hand_task_ms": np.array([r[arm]["hand_ms"] for r in fr], dtype=np.int64),
            "pose_task_ms": np.array([r[arm]["pose_ms"] for r in fr], dtype=np.int64),
            "process_ns": np.array([r[arm]["process_ns"] for r in fr], dtype=np.int64),
            "offset_us": np.array(pair.offset_us, dtype=np.int64),
        })
        stem = f"{pair.clip_id}__{arm}"
        write_npz_new(os.path.join(out_dir, "clips", stem + ".npz"), arrays)
        write_json_new(os.path.join(out_dir, "diag", stem + ".json"), {
            "clip_id": pair.clip_id, "arm": arm, "lm_key": pair.lm_keys[arm],
            "group": pair.group, "signer": pair.signer, "position": pair.position,
            "offset_us": pair.offset_us, "instr_slice": pair.instr_slices[arm],
            "rows": pair.diag[arm]})


def extract_all(cfg, rt, out_dir, run):
    clips = {c["clip_id"]: c for c in cfg["clips"]}
    reg = run.registry
    for gi, group in enumerate(cfg["groups"]):
        signer = signer_of(group[0])
        B = None
        try:
            B = reg.create("B_session", gi, signer, f"B_session_{signer}")
            prev_last = None
            for pos, cid in enumerate(group):
                offset = 0 if pos == 0 else prev_last + int(cfg["gap_us"])
                run.stage = f"extract:{cid}"
                A = reg.create("A_fresh", gi, signer, f"A_fresh_{cid}")
                try:
                    reg.clip_event("clip_start", cid, (A, B))
                    pair = extract_clip_pair(rt, run, clips[cid], gi, pos, signer, offset, A, B)
                    reg.clip_event("clip_end", cid, (A, B))
                finally:
                    reg.close(A)
                run.pairs[cid] = pair
                persist_clip(out_dir, pair, rt.obs_fields)
                prev_last = pair.last_absolute_us
                if pos == 0:
                    run.stage = f"first_clip_parity:{cid}"
                    mism = obs_bitexact_diff(pair.obs["A_fresh"], pair.obs["B_session"], rt.obs_fields)
                    run.gates[cid] = {"first_clip_parity_ok": not mism,
                                      "n_mismatch": len(mism), "mismatches": mism[:200]}
                    if mism:
                        raise TechnicalGateError("first_clip_parity",
                                                 f"{cid}: {len(mism)} field mismatches")
        finally:
            if B is not None:
                reg.close(B)
    run.stage = "close"
    reg.require_all_closed_ok()
    run.counters["internal_tasks"] = reg.n_tasks()


# ---------------------------------------------------------------------------
# scoring / comparison
# ---------------------------------------------------------------------------
def own_rank(logits, classes, label):
    li = classes.index(label)
    return {"top1": classes[int(np.argmax(logits))],
            "label_rank": 1 + int((logits > logits[li]).sum()),
            "label_logit": float(logits[li])}


def score_all(cfg, rt, run, out_dir):
    run.stage = "score:load_model"
    model, classes = rt.load_model()
    run.counters["model_loads"] += 1
    require(len(classes) == N_CLASSES and len(set(classes)) == N_CLASSES, "score",
            f"expected {N_CLASSES} unique classes from payload")
    state_before = rt.state_hash(model)
    pos = {cid: (gi, p) for gi, g in enumerate(cfg["groups"]) for p, cid in enumerate(g)}
    windows, inputs = [], {}
    for clip in cfg["clips"]:
        if not clip.get("score"):
            continue
        cid, s = clip["clip_id"], clip["score"]
        label, start, end = s["label"], int(s["start_us"]), int(s["end_us_exclusive"])
        require(label in classes, "score", f"label {label} not in payload classes")
        run.stage = f"score:{cid}"
        pair = run.pairs[cid]
        ids = [r["frame_id"] for r in pair.frames]
        ts = [r["relative_us"] for r in pair.frames]
        win = half_open(ids, ts, start, end)
        require(len(win) > 0, "score", f"{cid} empty window")
        gi, p = pos[cid]
        entry = {"clip_id": cid, "label": label, "start_us": start, "end_us_exclusive": end,
                 "window_frame_ids": [i for i, _ in win],
                 "window_timestamp_us": [t for _, t in win],
                 "group": gi, "position": p, "signer": pair.signer,
                 "history_applies": p > 0}
        for arm in ARMS:
            prep = rt.prepare_window(pair.obs[arm], start, end)
            x, pad = np.asarray(prep["x"]), np.asarray(prep["pad"])
            logits = np.asarray(rt.forward(model, x, pad), dtype=np.float64).reshape(-1)
            run.counters["main_forwards"] += 1
            require(logits.shape == (N_CLASSES,) and np.isfinite(logits).all(), "score",
                    f"{cid}/{arm} logits shape/finite")
            short = ARM_SHORT[arm]
            inputs[f"{cid}__{short}__x"] = x
            inputs[f"{cid}__{short}__pad"] = pad
            entry[short] = {
                "input": {"x_sha256": sha256_array(x), "x_shape": list(x.shape),
                          "x_dtype": x.dtype.str, "pad_sha256": sha256_array(pad),
                          "pad_shape": list(pad.shape), "pad_dtype": pad.dtype.str,
                          "body_scale": prep["body_scale"], "hand_scales": prep["hand_scales"]},
                "logits93": [float(v) for v in logits],
                "score": rt.rank(logits, classes, label),
                **own_rank(logits, classes, label),
            }
        windows.append(entry)
    state_after = rt.state_hash(model)
    require(state_before == state_after, "score", "model state changed during scoring")
    scores = {"schema": SCORES_SCHEMA,
              "scoring": {"scope": "full93", "classes": classes,
                          "checkpoint_sha256": cfg["scoring"]["checkpoint_sha256"],
                          "cpu_threads": CPU_THREADS, "tolerance": TOLERANCE,
                          "frame_selection": cfg["scoring"]["frame_selection"],
                          "model_state_before": state_before, "model_state_after": state_after},
              "windows": windows}
    write_json_new(os.path.join(out_dir, "SCORES.json"), scores)
    write_npz_new(os.path.join(out_dir, "SCORE_INPUTS.npz"), inputs)
    return scores


def compare(scores, run, fields):
    rows = []
    for w in scores["windows"]:
        a, b, lab = w["A"], w["B"], w["label"]
        ac, bc = a["top1"] == lab, b["top1"] == lab
        trans = ("both_correct" if ac and bc else "both_wrong" if not ac and not bc
                 else "B_new_wrong_regression_vs_fresh" if ac else "B_new_correct_improvement_vs_fresh")
        pair = run.pairs[w["clip_id"]]
        local = [k for k, r in enumerate(pair.frames) if r["frame_id"] in set(w["window_frame_ids"])]
        sa = stack_obs(pair.obs["A_fresh"], fields)
        sb = stack_obs(pair.obs["B_session"], fields)
        rows.append({
            "clip_id": w["clip_id"], "label": lab, "history_applies": w["history_applies"],
            "A_top1": a["top1"], "B_top1": b["top1"], "A_correct": ac, "B_correct": bc,
            "transition": trans, "A_label_rank": a["label_rank"], "B_label_rank": b["label_rank"],
            "label_logit_delta_B_minus_A": b["label_logit"] - a["label_logit"],
            "max_abs_logit_diff": float(np.max(np.abs(np.array(a["logits93"]) - np.array(b["logits93"])))),
            "x_identical": a["input"]["x_sha256"] == b["input"]["x_sha256"],
            "pad_identical": a["input"]["pad_sha256"] == b["input"]["pad_sha256"],
            "window_obs_diff": {f: field_diff(sa[f"obs_{f}"][local], sb[f"obs_{f}"][local]) for f in fields},
        })

    def summary(sel):
        keys = ("both_correct", "both_wrong", "B_new_wrong_regression_vs_fresh",
                "B_new_correct_improvement_vs_fresh")
        c = {k: sum(r["transition"] == k for r in sel) for k in keys}
        c.update({"denominator": len(sel), "A_correct": sum(r["A_correct"] for r in sel),
                  "B_correct": sum(r["B_correct"] for r in sel)})
        return c

    hist = [r for r in rows if r["history_applies"]]
    s_all, s_hist = summary(rows), summary(hist)
    notes = list(LIMITATIONS)
    if s_all["B_new_wrong_regression_vs_fresh"]:
        notes.append("B new error(s): limited evidence that fixed prior-clip history contributed; "
                     "pose/hand sole cause and real web cause not established.")
    if s_all["B_new_correct_improvement_vs_fresh"]:
        notes.append("A new error(s): evidence against 'reset is always better'; B is not auto-promoted.")
    if not (s_all["B_new_wrong_regression_vs_fresh"] or s_all["B_new_correct_improvement_vs_fresh"]):
        notes.append("No answer flips: state-lifetime effect on recognition errors not observed "
                     "under this condition (score deltas alone are not counted as improvement).")
    return {"windows": rows, "summary_all": s_all, "summary_history_only": s_hist,
            "controls_group_first_scored": [r["clip_id"] for r in rows if not r["history_applies"]],
            "interpretation": notes}


def clip_metrics(rt, run, fields):
    out = {}
    for cid, pair in run.pairs.items():
        sa = stack_obs(pair.obs["A_fresh"], fields)
        sb = stack_obs(pair.obs["B_session"], fields)
        timing = {}
        for arm in ARMS:
            ns = np.array([r[arm]["process_ns"] for r in pair.frames], dtype=np.float64)
            timing[arm] = {"mean_ms": float(ns.mean() / 1e6), "median_ms": float(np.median(ns) / 1e6),
                           "note": "offline diagnostic; not real-camera throughput"}
        out[cid] = {"group": pair.group, "position": pair.position, "offset_us": pair.offset_us,
                    "n_frames": len(pair.frames), "first_absolute_us": pair.first_absolute_us,
                    "last_absolute_us": pair.last_absolute_us,
                    "A_fresh": rt.clip_metrics(pair.obs["A_fresh"]),
                    "B_session": rt.clip_metrics(pair.obs["B_session"]),
                    "clip_obs_diff": {f: field_diff(sa[f"obs_{f}"], sb[f"obs_{f}"]) for f in fields},
                    "timing": timing}
    return out


def run_pipeline(cfg, rt, out_dir, run):
    """Extraction -> gates -> 8 forwards -> comparison -> metrics. Writes artifacts into
    out_dir (must already exist and be empty). Returns RESULT dict (not yet written)."""
    os.makedirs(os.path.join(out_dir, "clips"))
    os.makedirs(os.path.join(out_dir, "diag"))
    run.registry = Registry(rt, run)
    try:
        extract_all(cfg, rt, out_dir, run)
    finally:
        run.registry.close_all_open()
        write_json_new(os.path.join(out_dir, "LM_REGISTRY.json"), run.registry.to_json())
    scores = score_all(cfg, rt, run, out_dir)
    run.stage = "compare"
    write_json_new(os.path.join(out_dir, "COMPARISON.json"), compare(scores, run, rt.obs_fields))
    run.stage = "metrics"
    write_json_new(os.path.join(out_dir, "CLIP_METRICS.json"), clip_metrics(rt, run, rt.obs_fields))
    run.stage = "budget"
    budgets = {k: cfg["budgets"][k] for k in COUNTER_KEYS}
    require(run.counters == budgets, "budget", f"counters {run.counters} != budgets {budgets}")
    return {"schema": RESULT_SCHEMA, "status": "PASS", "counters": run.counters,
            "budgets": cfg["budgets"], "obs_fields": list(rt.obs_fields),
            "gates": run.gates,
            "clips": {cid: {"group": p.group, "position": p.position, "signer": p.signer,
                            "offset_us": p.offset_us, "n_frames": len(p.frames),
                            "first_absolute_us": p.first_absolute_us,
                            "last_absolute_us": p.last_absolute_us, "lm_keys": p.lm_keys}
                      for cid, p in run.pairs.items()},
            "limitations": LIMITATIONS}


def write_result_manifest(out_dir):
    files = {}
    for root, _, names in os.walk(out_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, out_dir).replace("\\", "/")
            if rel != "RESULT_MANIFEST.json":
                files[rel] = sha256_file(p)
    write_json_new(os.path.join(out_dir, "RESULT_MANIFEST.json"), {"files": files})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def cli_run(cfg_path):
    cfg, pre = preflight(cfg_path, frozen=True)
    receipt = check_receipt(cfg_path)
    require(pre["pins"]["ok"], "preflight", f"pin mismatches: {pre['pins']['mismatches']}")
    out_dir = cfg["output_dir"]
    require(not os.path.exists(out_dir), "output", f"output dir exists (no overwrite): {out_dir}")
    os.makedirs(out_dir, exist_ok=False)
    run = RunState()
    try:
        run.stage = "bind"
        rt = PinnedRuntime(cfg)
        write_json_new(os.path.join(out_dir, "RUN_MANIFEST.json"), {
            "schema": RESULT_SCHEMA, "preflight": pre, "receipt": receipt,
            "bindings": rt.signatures(), "started_unix": time.time(),
            "python": sys.version, "argv": sys.argv})
        result = run_pipeline(cfg, rt, out_dir, run)
        run.stage = "post_pins"
        post = {"config_sha256": sha256_file(cfg_path), "own_code_sha256": own_code_hashes(),
                "pins": hash_pins(cfg)}
        require(post["pins"]["ok"] and post["config_sha256"] == pre["config_sha256"]
                and post["own_code_sha256"] == pre["own_code_sha256"], "post_pins",
                "pins/config/own code changed during run")
        result["post_hashes"] = post
        write_json_new(os.path.join(out_dir, "RESULT.json"), result)
        write_result_manifest(out_dir)
        print(json.dumps({"status": "PASS", "out_dir": out_dir, "counters": run.counters}))
        return 0
    except Exception as e:
        stage = e.stage if isinstance(e, StageError) else run.stage
        fail = {"schema": RESULT_SCHEMA, "status": "FAIL", "stage": stage,
                "exception_type": type(e).__name__, "exception": str(e),
                "traceback": traceback.format_exc(), "counters": run.counters,
                "gates": run.gates,
                "registry": run.registry.to_json() if run.registry else None}
        try:
            fail["post_pins"] = hash_pins(cfg)
        except Exception as e2:
            fail["post_pins"] = f"ERROR {type(e2).__name__}: {e2}"
        write_json_new(os.path.join(out_dir, "FAILURE.json"), fail)
        write_result_manifest(out_dir)
        print(json.dumps({"status": "FAIL", "stage": stage, "exception": str(e)}))
        return 2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--check-bindings", action="store_true")
    a = ap.parse_args(argv)
    if a.run:
        return cli_run(a.config)
    cfg, pre = preflight(a.config, frozen=True)
    if a.check_bindings:
        require(pre["pins"]["ok"], "preflight", f"pin mismatches: {pre['pins']['mismatches']}")
        print(json.dumps(PinnedRuntime(cfg).signatures(), ensure_ascii=False, indent=1))
        return 0
    print(json.dumps(pre, ensure_ascii=False, indent=1))
    return 0 if pre["pins"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
