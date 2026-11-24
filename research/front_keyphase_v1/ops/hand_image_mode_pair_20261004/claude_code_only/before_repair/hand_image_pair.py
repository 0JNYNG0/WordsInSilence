"""Hand VIDEO vs IMAGE fixed pair: B_video (pose VIDEO + hand VIDEO) vs C_hand_image (pose VIDEO + hand IMAGE).

Thin wrapper over the verified capture_cadence_pair_20261004/cadence_pair.py: validate_config(oldcfg),
load_runtime_deps, extract_arm(arm=ARM_B) twice per clip (B then C), persist_arm, window_rows, prepare_window,
deps.load_model/forward/state_hash, rank_against_label. The original Landmarker.process/_assign_sides run
unchanged; only per-instance proxies (no global monkeypatch) swap the hand task to IMAGE and observe results.

Default: hash-only preflight (no parent import, no decode, no task/model creation). --run executes.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "wis_hand_image_mode_pair_config_v1"
ARMS = (("B_video", "VIDEO"), ("C_hand_image", "IMAGE"))
REQUIRED = ("schema", "unit_dir", "output_dir", "workspace_root", "front_root", "parent_cadence",
            "mechanism_evidence", "clips", "pins", "pin_inventory", "extraction", "scoring", "budgets",
            "isolation", "baseline_parity", "diagnostics")
PC_KEYS = ("module_path", "config_path", "result_path", "runtime_path")
CLIP_KEYS = ("clip_id", "path", "sha256", "expected_frames_from_prior", "expected_selected_even_frames", "score",
             "baseline_b_npz", "baseline_b_meta", "baseline_b_npz_sha256", "baseline_b_meta_sha256")
EX_KEYS = ("contract_path", "hand_model_path", "pose_model_path", "hand_running_mode_by_arm")
CP_COUNTERS = ("decoder_opened", "decoded_frames", "landmarker_constructed", "landmarker_closed",
               "landmarker_process_calls", "model_loads", "tcn_forwards")
TASK_COUNTERS = ("pose_video_tasks", "hand_video_tasks", "hand_image_tasks",
                 "pose_video_calls", "hand_video_calls", "hand_image_calls")
BUDGET_KEYS = CP_COUNTERS + TASK_COUNTERS
HAND_OPT_FIELDS = ("num_hands", "min_hand_detection_confidence", "min_hand_presence_confidence",
                   "min_tracking_confidence")
POSE_FIELDS = ("xy_px", "observed", "usable", "conf")
SRC = Path(__file__).resolve()
TEST_SRC = SRC.with_name("test_hand_image.py")


class GateFail(Exception):
    pass


def need(cond, msg):
    if not cond:
        raise GateFail(msg)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _n(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def bitexact(a, b) -> bool:
    """dtype/shape/raw bytes (NaN payload included). Not cadence exact_equal, which only compares NaN positions."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _rgb_sha(rgb) -> str:
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


# ---------------------------------------------------------------- preflight (hash only)
def _paths(v):
    if isinstance(v, str):
        return [v]
    if isinstance(v, dict):
        return [p for x in v.values() for p in _paths(x)]
    if isinstance(v, list):
        return [p for x in v for p in _paths(x)]
    return []


def check_pins(cfg) -> dict:
    pins = {_n(p): s for p, s in cfg["pins"].items()}
    bad = {p: ("missing" if not os.path.isfile(p) else "sha256") for p, s in pins.items()
           if not os.path.isfile(p) or sha256_file(p) != s}
    need(not bad, f"pin mismatch: {bad}")
    return pins


def preflight(cfg) -> dict:
    missing = [k for k in REQUIRED if k not in cfg]
    need(not missing, f"config missing keys: {missing}")
    need(cfg["schema"] == SCHEMA, f"schema {cfg['schema']!r} != {SCHEMA}")
    pc, ex, sc = cfg["parent_cadence"], cfg["extraction"], cfg["scoring"]
    need(all(k in pc for k in PC_KEYS), "parent_cadence keys")
    need(all(k in ex for k in EX_KEYS), "extraction keys")
    need("checkpoint_path" in sc and "checkpoint_sha256" in sc, "scoring keys")
    need(ex["hand_running_mode_by_arm"] == {"B": "VIDEO", "C": "IMAGE"}, "hand_running_mode_by_arm must be B VIDEO / C IMAGE")
    for k in ("isolation", "baseline_parity", "diagnostics"):
        need(isinstance(cfg[k], dict), f"{k} must be an object")
    pins = check_pins(cfg)
    refs = [pc[k] for k in PC_KEYS] + _paths(cfg["mechanism_evidence"]) + [ex[k] for k in EX_KEYS[:3]]
    refs.append(sc["checkpoint_path"])
    need(pins.get(_n(sc["checkpoint_path"])) == sc["checkpoint_sha256"], "checkpoint sha != pin")
    ids = []
    for c in cfg["clips"]:
        need(all(k in c for k in CLIP_KEYS), f"clip keys: {sorted(set(CLIP_KEYS) - set(c))}")
        ids.append(c["clip_id"])
        for pk, sk in (("path", "sha256"), ("baseline_b_npz", "baseline_b_npz_sha256"),
                       ("baseline_b_meta", "baseline_b_meta_sha256")):
            refs.append(c[pk])
            need(pins.get(_n(c[pk])) == c[sk], f"{c['clip_id']} {pk} sha != pin")
    need(ids and len(set(ids)) == len(ids), "clip ids empty or duplicated")
    unpinned = sorted({r for r in refs if _n(r) not in pins})
    need(not unpinned, f"referenced but unpinned: {unpinned}")
    inv = set()
    for e in cfg["pin_inventory"]:
        need(all(k in e for k in ("path", "role", "bytes", "sha256")), "pin_inventory entry keys")
        need(pins.get(_n(e["path"])) == e["sha256"], f"inventory sha != pin: {e['path']}")
        need(os.path.getsize(e["path"]) == e["bytes"], f"inventory bytes mismatch: {e['path']}")
        inv.add(_n(e["path"]))
    need(inv == set(pins), "pin_inventory must list exactly the pins")
    b = cfg["budgets"]
    need(set(b) == set(BUDGET_KEYS) and all(isinstance(v, int) for v in b.values()),
         f"budgets must be ints for exactly {BUDGET_KEYS}")
    out, unit = Path(cfg["output_dir"]).resolve(), Path(cfg["unit_dir"]).resolve()
    need(unit in out.parents, "output_dir must be inside unit_dir")
    need(not out.exists(), f"output_dir exists (fresh only): {out}")
    need(not any(Path(p) == out or out in Path(p).parents for p in pins), "a pinned file lies under output_dir")
    return {"status": "PREFLIGHT_OK_HASH_ONLY", "pins_checked": len(pins), "clips": ids}


# ---------------------------------------------------------------- per-instance instrumentation
class TaskProxy:
    """Times detect, returns the task's own result object, records its own close (Landmarker.close swallows)."""

    def __init__(self, task, kind, mode, ins):
        self.task, self.kind, self.mode, self.ins = task, kind, mode, ins
        self.calls, self.close_rec = 0, None
        ins.tasks.append(self)

    def detect_for_video(self, image, ms):
        t0 = time.perf_counter_ns()
        res = self.task.detect(image) if self.mode == "IMAGE" else self.task.detect_for_video(image, ms)
        t1 = time.perf_counter_ns()
        self.calls += 1
        if self.ins.cur is None:
            self.ins.errors.append(f"{self.kind} detect outside process")
        else:
            self.ins.cur[self.kind] = {"api": "detect" if self.mode == "IMAGE" else "detect_for_video",
                                       "mode": self.mode, "landmarker_ms": int(ms),
                                       "timestamp_ms_passed": None if self.mode == "IMAGE" else int(ms),
                                       "t0_ns": t0, "t1_ns": t1, "detect_ns": t1 - t0, "_res": res}
        return res

    def close(self):
        self.close_rec = {"attempted": True, "ok": False, "error": None}
        try:
            self.task.close()
            self.close_rec["ok"] = True
        except Exception as exc:
            self.close_rec["error"] = f"{type(exc).__name__}: {exc}"
            self.ins.errors.append(f"close {self.kind}_{self.mode}: {self.close_rec['error']}")
            raise

    def record(self) -> dict:
        return {"task": f"{self.kind}_{self.mode}", "calls": self.calls,
                "close": self.close_rec or {"attempted": False, "ok": False, "error": None}}


class Instr:
    def __init__(self, tag, mode):
        self.tag, self.mode = tag, mode
        self.tasks, self.frames, self.errors, self.cur, self.provenance = [], [], [], None, None


def _hand_raw(res) -> list:
    hl, wl, hd = res.hand_landmarks or [], res.hand_world_landmarks or [], res.handedness or []
    return [{"index": i,
             "normalized_xyz": [[float(p.x), float(p.y), float(p.z)] for p in hl[i]],
             "world_xyz": [[float(p.x), float(p.y), float(p.z)] for p in wl[i]] if i < len(wl) else [],
             "handedness": [{"category_name": c.category_name, "score": float(c.score),
                             "index": None if getattr(c, "index", None) is None else int(c.index)}
                            for c in (hd[i] if i < len(hd) else [])]}
            for i in range(len(hl))]


def _assign_diag(a, raw) -> dict:
    sides, ref, c0, c1, w, h = a
    wr = [[hd["normalized_xyz"][0][0] * w, hd["normalized_xyz"][0][1] * h] for hd in raw]
    return {"index_to_side": [{"index": int(i), "side": s, "score": float(sc)} for i, (s, sc) in sides.items()],
            "pose_wrist_ref_px": {str(s): p for s, p in ref.items()},
            "distances_px": [{"index": i, "side": s, "px": math.dist(wr[i], ref[s])} for i in range(len(wr)) for s in ref],
            "fallback_raw_handedness": not ref,
            "handedness_comparisons_delta": c1[0] - c0[0], "handedness_disagreements_delta": c1[1] - c0[1]}


def instrument(lm, ins, lm_mod) -> None:
    """Instance attributes shadow the class methods for this object only; originals are called unchanged."""
    orig_process, orig_assign = lm.process, lm._assign_sides
    lw, rw = lm_mod.POSE_L_WRIST, lm_mod.POSE_R_WRIST

    def assign(hand_res, pose_xy, pose_usable, w, h):
        c0 = (lm.handedness_comparisons, lm.handedness_disagreements)
        sides = orig_assign(hand_res, pose_xy, pose_usable, w, h)
        ref = {s: [float(v) for v in pose_xy[k]] for s, k in ((0, lw), (1, rw)) if pose_usable[k]}
        ins.cur["_assign"] = (sides, ref, c0, (lm.handedness_comparisons, lm.handedness_disagreements), w, h)
        return sides

    def process(rgb, timestamp_us, out, row):
        before, u0 = _rgb_sha(rgb), lm.unassigned_hands
        ins.cur = cur = {"row": int(row), "timestamp_us": int(timestamp_us)}
        t0 = time.perf_counter_ns()
        try:
            ret = orig_process(rgb, timestamp_us, out, row)
        finally:
            t1 = time.perf_counter_ns()
            ins.cur = None
        # serialization happens after timing
        pose, hand = cur.pop("pose", None), cur.pop("hand", None)
        if pose is None or hand is None:
            ins.errors.append(f"row {row}: pose/hand detect missing")
        raw = _hand_raw(hand.pop("_res")) if hand else []
        if pose:
            pose.pop("_res")
        a = cur.pop("_assign", None)
        cur.update(pose=pose, hand=hand, hand_api=hand and hand["api"], pose_ms=pose and pose["landmarker_ms"],
                   n_hands=len(raw), hand_raw=raw, assignment=_assign_diag(a, raw) if a else None,
                   unassigned_delta=lm.unassigned_hands - u0, process_ns=t1 - t0,
                   rgb_sha256_before=before, rgb_sha256_after=_rgb_sha(rgb))
        ins.frames.append(cur)
        return ret

    lm.process, lm._assign_sides = process, assign


def provenance(lm, mode, opts) -> dict:
    p = {"base_extractor_id": lm.extractor_id(), "base_extractor_id_includes_running_mode": False,
         "pose_running_mode": "VIDEO", "hand_running_mode": mode,
         "hand_api": "detect" if mode == "IMAGE" else "detect_for_video",
         "wrapper_sha256": sha256_file(SRC),
         "hand_task": {"path": lm.hand_model_path, "sha256": sha256_file(lm.hand_model_path)},
         "pose_task": {"path": lm.pose_model_path, "sha256": sha256_file(lm.pose_model_path)},
         "hand_options": {f: getattr(opts, f, None) for f in HAND_OPT_FIELDS},
         "hand_options_tracking_semantics_not_applied": ["min_tracking_confidence"] if mode == "IMAGE" else [],
         "pose_options": {"running_mode": "VIDEO", "num_poses": 1}}
    p["experiment_extractor_id"] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]
    return p


def make_factory(base_factory, lm_mod, tag, mode, sink):
    def factory():
        lm = base_factory()
        ins = Instr(tag, mode)
        sink.append(ins)
        lm.pose = TaskProxy(lm.pose, "pose", "VIDEO", ins)
        lm.hand = TaskProxy(lm.hand, "hand", "VIDEO", ins)
        v = lm_mod.mp_vision
        kw = dict(base_options=lm_mod.mp_python.BaseOptions(model_asset_path=lm.hand_model_path), num_hands=2)
        if mode == "IMAGE":
            try:
                lm.hand.close()                                   # unused VIDEO hand: created, process0, closed
                opts = v.HandLandmarkerOptions(running_mode=v.RunningMode.IMAGE, **kw)
                lm.hand = TaskProxy(v.HandLandmarker.create_from_options(opts), "hand", "IMAGE", ins)
            except BaseException as exc:
                ins.errors.append(f"C setup: {type(exc).__name__}: {exc}")
                for t in ins.tasks:
                    if t.close_rec is None:
                        try:
                            t.close()
                        except Exception:                          # noqa: BLE001 - recorded in close_rec
                            pass
                raise
        else:
            opts = v.HandLandmarkerOptions(running_mode=v.RunningMode.VIDEO, **kw)   # record-only, no task
        instrument(lm, ins, lm_mod)
        ins.provenance = provenance(lm, mode, opts)
        return lm
    return factory


# ---------------------------------------------------------------- gates
def _js(x, jsonable=lambda v: v):
    return json.loads(json.dumps(jsonable(x), allow_nan=False, sort_keys=True))


def gate_b_baseline(r, npz: dict, meta: dict, obs_fields, jsonable=lambda v: v) -> list:
    bad = [f for f in obs_fields if f"obs_{f}" not in npz or not bitexact(r["stored_obs"][f], npz[f"obs_{f}"])]
    bad += [f"selected:{f}" for f in obs_fields if f"obs_{f}" not in npz or not bitexact(r["selected_obs"][f], npz[f"obs_{f}"])]
    bad += [f"decoded_{k}" for k, v in r["decoded_meta"].items() if f"decoded_{k}" not in npz or not bitexact(v, npz[f"decoded_{k}"])]
    if "selected_frame_id" not in npz or not bitexact(r["selected_meta"]["frame_id"], npz["selected_frame_id"]):
        bad.append("selected_frame_id")
    bad += [f"meta:{k}" for k in ("counts", "time_base") if _js(r[k], jsonable) != _js(meta.get(k), jsonable)]
    return bad


def gate_bc(rb, rc, obs_fields, decode_fields) -> list:
    bad = [f"selected_meta:{k}" for k in decode_fields if not bitexact(rb["selected_meta"][k], rc["selected_meta"][k])]
    bad += [f"decoded_meta:{k}" for k in rb["decoded_meta"] if not bitexact(rb["decoded_meta"][k], rc["decoded_meta"].get(k))]
    bad += [f"pose:{f}" for f in POSE_FIELDS if not bitexact(rb["selected_obs"][f][:, :33], rc["selected_obs"][f][:, :33])]
    bad += [f"obs:{f}" for f in ("timestamp_us", "frame_id") if f in obs_fields
            and not bitexact(rb["selected_obs"][f], rc["selected_obs"][f])]
    return bad


def gate_diag(ins, r, n_selected) -> list:
    fr, bad = ins.frames, list(ins.errors)
    if [f["row"] for f in fr] != list(range(n_selected)):
        bad.append("process rows != 0..n_selected-1")
    if [f["timestamp_us"] for f in fr] != np.asarray(r["selected_obs"]["timestamp_us"]).tolist():
        bad.append("process timestamps != selected timestamps")
    if any(f["rgb_sha256_before"] != f["rgb_sha256_after"] for f in fr):
        bad.append("rgb changed during process")
    for t in ins.tasks:
        want = 0 if (t.kind == "hand" and t.mode != ins.mode) else n_selected
        if t.calls != want:
            bad.append(f"{t.kind}_{t.mode} calls {t.calls} != {want}")
        if not (t.close_rec and t.close_rec["ok"]):
            bad.append(f"{t.kind}_{t.mode} not closed ok")
    return bad


def task_counts(instrs) -> Counter:
    c = Counter()
    for ins in instrs:
        for t in ins.tasks:
            k = f"{t.kind}_{t.mode.lower()}"
            c[k + "_tasks"] += 1
            c[k + "_calls"] += t.calls
            c[k + "_closed_ok"] += bool(t.close_rec and t.close_rec["ok"])
    return c


def check_budgets(counters, instrs, budgets, keys=BUDGET_KEYS) -> list:
    tc = task_counts(instrs)
    actual = {**{k: counters.get(k, 0) for k in CP_COUNTERS}, **{k: tc[k] for k in TASK_COUNTERS}}
    bad = [f"{k}: {actual[k]} != {budgets[k]}" for k in keys if actual[k] != budgets[k]]
    bad += [f"{k}_closed_ok {tc[k + '_closed_ok']} != {tc[k + '_tasks']}" for k in ("pose_video", "hand_video", "hand_image")
            if tc[k + "_closed_ok"] != tc[k + "_tasks"]]
    bad += [f"{k} != 0" for k in ("camera_calls", "training_runs", "downloads") if counters.get(k, 0)]
    return bad


# ---------------------------------------------------------------- reporting helpers
def hand_stats(sel, idx, slices) -> dict:
    out = {"observed_points": int(np.asarray(sel["observed"])[idx].astype(bool).sum()),
           "usable_points": int(np.asarray(sel["usable"])[idx].astype(bool).sum())}
    for name, sl in slices:
        o, u = sel["observed"][idx, sl].astype(bool), sel["usable"][idx, sl].astype(bool)
        out[name] = {"rows": int(len(idx)), "all21_missing": int((~o.any(1)).sum()), "all21_usable": int(u.all(1).sum()),
                     "observed_points": int(o.sum()), "usable_points": int(u.sum())}
    return out


def coord_diff(sb, sc, idx, sl):
    xb, xc = sb["xy_px"][idx, sl].astype(np.float64), sc["xy_px"][idx, sl].astype(np.float64)
    m = (sb["usable"][idx, sl].astype(bool) & sc["usable"][idx, sl].astype(bool)
         & np.isfinite(xb).all(-1) & np.isfinite(xc).all(-1))
    if not m.any():
        return None
    d = np.linalg.norm((xb - xc)[m], axis=-1)
    return {"common_usable_points": int(m.sum()), "mean_px": float(d.mean()), "max_px": float(d.max())}


def _sides(frame):
    a = frame["assignment"]
    return sorted(e["side"] for e in a["index_to_side"] if e["side"] is not None) if a else []


def compare_obs(sb, sc, idx, slices, fb, fcc) -> dict:
    return {"B_video": hand_stats(sb, idx, slices), "C_hand_image": hand_stats(sc, idx, slices),
            "coord_diff_B_vs_C": {name: coord_diff(sb, sc, idx, sl) for name, sl in slices},
            "detected_hands": {"B_video": int(sum(fb[i]["n_hands"] for i in idx)),
                               "C_hand_image": int(sum(fcc[i]["n_hands"] for i in idx))},
            "assigned_sides_changed_rows": int(sum(_sides(fb[i]) != _sides(fcc[i]) for i in idx)),
            "note": "more detections are not more correct coordinates"}


def pct(v) -> dict:
    v = np.asarray(v, dtype=np.float64)
    if not v.size:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": int(v.size), "p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def timing(instrs) -> dict:
    out = {"scope": "offline B->C order on this machine; not a hardware FPS or speed-up claim"}
    for tag, _ in ARMS:
        ii = [i for i in instrs if i.tag == tag]
        steady = [f for i in ii for f in i.frames[1:]]
        first = [i.frames[0] for i in ii if i.frames]
        out[tag] = {"first_call_per_landmarker": [{"pose_ns": f["pose"]["detect_ns"], "hand_ns": f["hand"]["detect_ns"],
                                                   "process_ns": f["process_ns"]} for f in first],
                    "steady": {k: pct([f[k]["detect_ns"] for f in steady]) for k in ("pose", "hand")}}
        out[tag]["steady"]["process"] = pct([f["process_ns"] for f in steady])
    return out


def direction(b, c) -> str:
    return {(True, True): "both_correct", (False, False): "both_wrong",
            (False, True): "C_recovers", (True, False): "C_regresses"}[(bool(b), bool(c))]


def decide(pairs) -> dict:
    imp = sum(1 for b, c in pairs if not b and c)
    wor = sum(1 for b, c in pairs if b and not c)
    if imp and wor:
        code = "MIXED_STOP"
    elif wor:
        code = "WORSENED_STOP"
    elif imp and all(c for _, c in pairs):
        code = "LIMITED_RECOVERY_FIXED_DEV_SAMPLE"
    elif imp:
        code = "PARTIAL_RECOVERY_STOP"
    else:
        code = "NO_CHANGE_STOP"
    return {"code": code, "recovered": imp, "regressed": wor, "C_correct": sum(1 for _, c in pairs if c),
            "B_correct": sum(1 for b, _ in pairs if b), "windows": len(pairs),
            "scope": "fixed development sample only; no threshold/coefficient/auto-mode/promotion"}


# ---------------------------------------------------------------- run
def import_parent(cfg, pins):
    p = cfg["parent_cadence"]["module_path"]
    need(sha256_file(p) == pins[_n(p)], "parent module changed")
    need("cadence_pair" not in sys.modules, "cadence_pair already imported")
    spec = importlib.util.spec_from_file_location("cadence_pair", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cadence_pair"] = mod
    spec.loader.exec_module(mod)
    return mod


def run(cfg, cfg_sha, test_receipt=None):
    pre = preflight(cfg)
    pins = {_n(p): s for p, s in cfg["pins"].items()}
    cp = import_parent(cfg, pins)
    out = Path(cfg["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    w = cp.ResultWriter(out)
    counters, instrs, gates, stage, budgets = cp.new_counters(), [], {}, "setup", cfg["budgets"]
    try:
        pc = cfg["parent_cadence"]
        plan = cp.validate_config(json.loads(Path(pc["config_path"]).read_text("utf-8")), cp.DEFAULT_ROOTS)
        need([c["clip_id"] for c in plan["clips"]] == [c["clip_id"] for c in cfg["clips"]], "clip order != old plan")
        for oc, nc in zip(plan["clips"], cfg["clips"]):
            need(_n(oc["path"]) == _n(nc["path"]) and oc["n_frames"] == nc["expected_frames_from_prior"]
                 and oc["n_selected"] == nc["expected_selected_even_frames"], f"{nc['clip_id']} != old plan clip")
        for k, role in (("contract_path", "observation_extraction_contract"), ("hand_model_path", "mediapipe_hand_model"),
                        ("pose_model_path", "mediapipe_pose_model")):
            need(_n(cfg["extraction"][k]) == _n(plan["role_paths"][role]), f"extraction.{k} != old plan {role}")
        need(_n(cfg["scoring"]["checkpoint_path"]) == _n(plan["role_paths"]["frozen_tcn_checkpoint"]), "checkpoint path")
        plan_sha = hashlib.sha256(json.dumps(cp.jsonable(plan), sort_keys=True, default=str).encode()).hexdigest()
        w.json("RUN_RECEIPT.json", {
            "config_sha256": cfg_sha, "old_plan_sha256": plan_sha, "wrapper_sha256": sha256_file(SRC),
            "test_sha256": sha256_file(TEST_SRC) if TEST_SRC.exists() else None,
            "test_receipt": test_receipt and {"path": str(test_receipt), "sha256": sha256_file(test_receipt)},
            "preflight": pre, "isolation": cfg["isolation"], "baseline_parity": cfg["baseline_parity"],
            "diagnostics": cfg["diagnostics"]})
        deps = cp.load_runtime_deps(plan, cp.DEFAULT_ROOTS, cp.sha256_file)
        lm_mod = sys.modules["wis_lab.capture.landmarker"]
        slices = (("left", lm_mod.LEFT_HAND_SLICE), ("right", lm_mod.RIGHT_HAND_SLICE))
        registry, res, seq = [], {}, 0
        for oc, nc in zip(plan["clips"], cfg["clips"]):
            cid = oc["clip_id"]
            stage = f"extract:{cid}"
            arms = {}
            for tag, mode in ARMS:
                n0 = len(instrs)
                d = dataclasses.replace(deps, landmarker_factory=make_factory(deps.landmarker_factory, lm_mod, tag, mode, instrs))
                r = cp.extract_arm(path=oc["path"], clip_id=cid, arm=cp.ARM_B, n_frames=oc["n_frames"],
                                   n_selected=oc["n_selected"], deps=d, counters=counters, registry=registry)
                need(len(instrs) == n0 + 1, f"{cid} {tag}: expected exactly one Landmarker")
                ins = instrs[-1]
                seq += 1
                art = cp.persist_arm(w, seq, tag, r)
                w.json(f"{seq:02d}_{cid}_{tag}.diag.json", {
                    "clip_id": cid, "tag": tag, "internal_selection_arm": r["arm"], "hand_running_mode": mode,
                    "provenance": ins.provenance, "tasks": [t.record() for t in ins.tasks], "errors": ins.errors,
                    "frames": ins.frames, "rgb_content_stored": False})
                gates[f"{cid}:{tag}:diag"] = gate_diag(ins, r, oc["n_selected"])
                arms[tag] = {"r": r, "ins": ins, "artifacts": art}
            with np.load(nc["baseline_b_npz"], allow_pickle=False) as z:
                npz = {k: z[k] for k in z.files}
            meta = json.loads(Path(nc["baseline_b_meta"]).read_text("utf-8"))
            rb, rc = arms["B_video"]["r"], arms["C_hand_image"]["r"]
            gates[f"{cid}:B_baseline"] = gate_b_baseline(rb, npz, meta, cp.OBS_FIELDS, cp.jsonable)
            gates[f"{cid}:BC"] = gate_bc(rb, rc, cp.OBS_FIELDS, cp.DECODE_FIELDS)
            failed = {k: v for k, v in gates.items() if v}
            need(not failed, f"extraction gates failed: {failed}")
            res[cid] = arms
        stage = "budgets_extraction"
        bad = check_budgets(counters, instrs, budgets, [k for k in BUDGET_KEYS if k not in ("model_loads", "tcn_forwards")])
        gates["budgets_extraction"] = bad
        need(not bad, f"extraction budgets: {bad}")

        stage = "inputs"
        old = json.loads(Path(pc["result_path"]).read_text("utf-8"))["scoring"]
        need(len(old["windows"]) * 2 == budgets["tcn_forwards"], "window count != tcn budget / 2")
        wins = []
        for ow in old["windows"]:
            cid = ow["clip_id"]
            need(cid in res, f"window clip {cid} not extracted")
            sb, sc = res[cid]["B_video"]["r"]["selected_obs"], res[cid]["C_hand_image"]["r"]["selected_obs"]
            idx = np.asarray(cp.window_rows(sb["timestamp_us"], ow["start_us"], ow["end_us_exclusive"]))
            need(np.asarray(sb["frame_id"])[idx].tolist() == list(ow["window_frame_ids"])
                 and np.asarray(sb["timestamp_us"])[idx].tolist() == list(ow["window_timestamp_us"]), f"{cid} window rows")
            pb, pcw = cp.prepare_window(sb, idx, deps), cp.prepare_window(sc, idx, deps)
            need(pb["info"]["x_sha256"] == ow["B"]["input"]["x_sha256"], f"{cid} B input hash != old B")
            wins.append((ow, idx, pb, pcw))
        stage = "scoring"
        model, classes, info = deps.load_model()
        counters["model_loads"] += 1
        need(list(classes) == list(old["classes"]) and len(classes) == 93, "classes != old classes / 93")
        s0 = deps.state_hash(model)
        need(s0 == info["payload_state_sha256"], "state != payload")
        logits = {}
        for tag, k in (("B_video", 0), ("C_hand_image", 1)):            # all B forwards + B parity before any C
            for ow, _idx, pb, pcw in wins:
                p = (pb, pcw)[k]
                lg = deps.forward(model, p["x"], p["pad"])
                counters["tcn_forwards"] += 1
                logits[(ow["clip_id"], tag)] = [float(v) for v in lg]
                if tag == "B_video":
                    need(logits[(ow["clip_id"], tag)] == [float(v) for v in ow["B"]["logits93"]], f"{ow['clip_id']} B logits != old B")
        need(deps.state_hash(model) == s0, "model state changed")
        windows, pairs = [], []
        for ow, idx, pb, pcw in wins:
            cid = ow["clip_id"]
            sb = {t: cp.rank_against_label(logits[(cid, t)], classes, ow["label"]) for t, _ in ARMS}
            pairs.append((sb["B_video"]["correct"], sb["C_hand_image"]["correct"]))
            a = res[cid]
            windows.append({
                "clip_id": cid, "label": ow["label"], "start_us": ow["start_us"], "end_us_exclusive": ow["end_us_exclusive"],
                "rows": int(idx.size), "frame_ids": np.asarray(a["B_video"]["r"]["selected_obs"]["frame_id"])[idx].tolist(),
                "timestamp_us": np.asarray(a["B_video"]["r"]["selected_obs"]["timestamp_us"])[idx].tolist(),
                "B_video": {"input": pb["info"], "logits93": logits[(cid, "B_video")], "score": sb["B_video"]},
                "C_hand_image": {"input": pcw["info"], "logits93": logits[(cid, "C_hand_image")], "score": sb["C_hand_image"]},
                "direction": direction(*pairs[-1]),
                "margin_delta_C_minus_B": sb["C_hand_image"]["true_vs_best_other_margin"] - sb["B_video"]["true_vs_best_other_margin"],
                "observation": compare_obs(a["B_video"]["r"]["selected_obs"], a["C_hand_image"]["r"]["selected_obs"], idx,
                                           slices, a["B_video"]["ins"].frames, a["C_hand_image"]["ins"].frames)})
        stage = "final_gates"
        bad = check_budgets(counters, instrs, budgets)
        gates["budgets_final"] = bad
        need(not bad, f"final budgets: {bad}")
        check_pins(cfg)
        clips = {}
        for cid, a in res.items():
            sb, sc = a["B_video"]["r"]["selected_obs"], a["C_hand_image"]["r"]["selected_obs"]
            clips[cid] = {"artifacts": {t: a[t]["artifacts"] for t, _ in ARMS},
                          "provenance": {t: a[t]["ins"].provenance for t, _ in ARMS},
                          "scored": cid in {x["clip_id"] for x in windows},
                          "observation_all_rows": compare_obs(sb, sc, np.arange(len(sb["timestamp_us"])), slices,
                                                              a["B_video"]["ins"].frames, a["C_hand_image"]["ins"].frames)}
        result = {"status": "COMPLETED_SELF_REPORTED_PENDING_INDEPENDENT_REVIEW", "decision": decide(pairs),
                  "windows": windows, "clips": clips, "timing": timing(instrs), "counters": counters,
                  "task_counts": dict(task_counts(instrs)), "gates": gates, "model": info, "versions": deps.versions,
                  "import_audit": deps.import_audit, "config_sha256": cfg_sha, "old_plan_sha256": plan_sha,
                  "wrapper_sha256": sha256_file(SRC),
                  "notes": ["base_extractor_id excludes running mode; use experiment_extractor_id",
                            "assignment algorithm unchanged; side assignment may differ because detections differ",
                            "out-of-model clips: observation diagnostics only, not scored"]}
        w.json("RESULT.json", result)
        d = result["decision"]
        lines = ["# Hand VIDEO vs IMAGE pair", "", f"- status: {result['status']}", f"- decision: {d['code']} "
                 f"(B {d['B_correct']}/{d['windows']}, C {d['C_correct']}/{d['windows']})", f"- scope: {d['scope']}", "",
                 "| clip | label | B top1 | B rank | B margin | C top1 | C rank | C margin | direction |",
                 "|---|---|---|---:|---:|---|---:|---:|---|"]
        for x in windows:
            b, c = x["B_video"]["score"], x["C_hand_image"]["score"]
            lines.append(f"| {x['clip_id']} | {x['label']} | {b['top1']} | {b['true_rank']} | {b['true_vs_best_other_margin']:+.4f} "
                         f"| {c['top1']} | {c['true_rank']} | {c['true_vs_best_other_margin']:+.4f} | {x['direction']} |")
        w.text("SUMMARY.md", "\n".join(lines) + "\n")
        w.json("MANIFEST.json", w.file_hashes())
        return result
    except BaseException as exc:
        w.json("FAILURE.json", {
            "status": "TECHNICAL_FAILURE_NO_PERFORMANCE_JUDGEMENT", "stage": stage,
            "error": f"{type(exc).__name__}: {exc}", "counters": counters, "task_counts": dict(task_counts(instrs)),
            "gates": gates, "lifecycle": [{"tag": i.tag, "tasks": [t.record() for t in i.tasks], "errors": i.errors}
                                          for i in instrs], "config_sha256": cfg_sha})
        w.json("MANIFEST.json", w.file_hashes())
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="store_true", help="execute; default is hash-only preflight")
    ap.add_argument("--test-receipt")
    a = ap.parse_args(argv)
    raw = Path(a.config).read_bytes()
    cfg, cfg_sha = json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()
    if not a.run:
        print(json.dumps({**preflight(cfg), "config_sha256": cfg_sha, "wrapper_sha256": sha256_file(SRC)}, indent=2))
        return 0
    print(json.dumps(run(cfg, cfg_sha, a.test_receipt)["decision"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
