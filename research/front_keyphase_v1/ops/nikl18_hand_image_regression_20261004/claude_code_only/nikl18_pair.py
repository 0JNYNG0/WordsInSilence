"""NIKL18 hand VIDEO (B) vs hand IMAGE (C) frozen regression adapter.

Thin wiring only: pinned cadence module (decode/extract/model/forward/rank/writer) and pinned
hand_image_pair module (mode factory/instrumentation/gates/compare/timing) do the work.
New code here: config/pin preflight, plain-key NIKL cache gate, official active_span preprocessing,
B-then-C orchestration over 18 clips, decision and outcome.

Default = preflight (stdlib only: config schema + hashes, no MediaPipe/project import/model/decode).
--run requires --test-receipt and a fresh output_dir.
"""
import argparse
import dataclasses
import hashlib
import inspect
import io
import json
import math
import os
import re
import sys
import traceback
from pathlib import Path

CONFIG_SCHEMA = "wis_nikl18_hand_image_regression_config_v1"
UNAVAILABLE = "UNAVAILABLE_NO_ACTIVE_HANDS"
N_CLIPS, N_FRAMES = 18, 1337
ARMS = (("B_video", "VIDEO"), ("C_hand_image", "IMAGE"))
NON_EXTRACTION_BUDGETS = ("forward_calls_max", "model_loads", "independent_forward_calls")
# Exact keyword-only parameters of cadence.extract_arm (ACTUAL_API_EXCERPTS); factory travels inside deps.
EXTRACT_KW = ("path", "clip_id", "arm", "n_frames", "n_selected", "deps", "counters", "registry")
CACHE_SCHEMA = "wis_observation_2d_v1"

BUDGETS = {"decoder_opened": 36, "decoded_frames": 2674, "selected_frames_per_arm": 1337,
           "landmarker_process_calls": 2674, "pose_detect_for_video": 2674, "hand_detect_for_video": 1337,
           "hand_detect_image": 1337, "forward_calls_max": 36, "model_loads": 1, "landmarker_constructed": 36,
           "pose_tasks_constructed": 36, "hand_video_tasks_constructed": 36,
           "hand_video_tasks_discarded_without_inference": 18, "hand_image_tasks_constructed": 18,
           "training": 0, "camera": 0}
FIELDS7 = ["xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score"]
FROZEN = {f"budgets.{k}": v for k, v in BUDGETS.items()}
FROZEN.update({
    "schema": CONFIG_SCHEMA, "is_schema_example": False,
    "extraction.arms": ["B_video", "C_hand_image"], "extraction.call_order": ["B_video", "C_hand_image"],
    "extraction.pose_running_mode": "VIDEO",
    "extraction.hand_running_mode_by_arm": {"B_video": "VIDEO", "C_hand_image": "IMAGE"},
    "extraction.frame_selection": "ALL_ORIGINAL_FRAMES", "extraction.cadence_helper_arm": "A_full_then_even",
    "extraction.use_stored_obs_full": True, "extraction.reject_missing_pts": True,
    "extraction.reject_nonincreasing_pts": True, "extraction.first_timestamp_us": 0,
    "extraction.rgb_format": "rgb24", "extraction.resize": False, "extraction.mirror": False,
    "extraction.interpolation": False,
    "preprocessing.mapping": "pilot.web_input.to_native_like", "preprocessing.span": "pilot.web_input.active_span",
    "preprocessing.span_policy": "PER_ARM_FIRST_LAST_USABLE_HAND_INCLUSIVE",
    "preprocessing.channels": "pilot.data.build_channels", "preprocessing.collate_dummy_label": 0,
    "preprocessing.service_additional_gates": False, "preprocessing.no_active_hand_status": UNAVAILABLE,
    "preprocessing.other_preprocessing_errors": "TECHNICAL_FAILURE",
    "scoring.model_class": "pilot.model_masked_v1.MaskedSignTCN", "scoring.n_classes": 93,
    "scoring.cpu_threads": 2, "scoring.mode": "eval_inference_mode", "scoring.scored_clips": N_CLIPS,
    "scoring.scope": "full93", "scoring.historical_expected_correct": N_CLIPS,
    "scoring.historical_score_bitexact_required": False, "scoring.historical_top1_match_required": True,
    "scoring.baseline_input_bitexact_required": True, "scoring.forward_calls_max": 36,
    "gates.baseline_fields": FIELDS7, "gates.baseline_comparison": "DTYPE_SHAPE_BYTES_EXACT",
    "gates.bc_pose_fields": ["xy_px", "observed", "usable", "conf"], "gates.bc_pose_slice": [0, 33],
    "gates.all_raw_pose_input_gates_before_model": True, "gates.b_correct_required": N_CLIPS,
    "gates.c_regressions_allowed": 0, "gates.c_unavailable_allowed_for_pass": 0, "gates.auto_retry": False,
    "diagnostics.reuse_instrumentation": True, "diagnostics.raw_hand": True, "diagnostics.assignment": True,
    "diagnostics.per_frame_timing": True, "diagnostics.rgb_persistence": False,
    "diagnostics.images_persistence": False, "diagnostics.new_runner_hash_separate": True,
    "decision.operational_promotion": False,
    **{f"isolation.{k}": False for k in ("camera", "device_enumeration", "network", "downloads", "training",
                                          "user_data", "operational_change", "source_mutation",
                                          "output_overwrite", "execute_default")},
    "isolation.explicit_run_flag_required": True,
})
SCHEMA_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
                   "minProperties", "const", "pattern", "$schema"}


class TechnicalFailure(Exception):
    pass


def _req(cond, msg):
    if not cond:
        raise TechnicalFailure(msg)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(p) -> str:
    return os.path.normcase(os.path.realpath(str(p)))


def _abs(cfg, p) -> Path:
    q = Path(p)
    return q if q.is_absolute() else Path(cfg["workspace_root"]) / q


def _get(obj, dotted):
    for part in dotted.split("."):
        obj = obj.get(part) if isinstance(obj, dict) else None
    return obj


def _walk(obj):
    yield obj
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)


def _safe(obj):
    """JSON-safe copy for FAILURE.json: non-finite -> None, arrays -> lists."""
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if hasattr(obj, "tolist"):
        return _safe(obj.tolist())
    return str(obj)


def _write_x(path: Path, obj) -> None:
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


# ------------------------------------------------------------------ config / schema / pins
def validate_schema(obj, sch, path="$"):
    """Subset of JSON Schema used by SCHEMA_EXAMPLE.json; unknown keywords are rejected, not ignored."""
    _req(set(sch) <= SCHEMA_KEYWORDS, f"{path}: unsupported schema keywords {sorted(set(sch) - SCHEMA_KEYWORDS)}")
    t = sch["type"]
    ok = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
          "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool),
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool)}[t](obj)
    _req(ok, f"{path}: expected {t}, got {type(obj).__name__}")
    if "const" in sch:
        _req(obj == sch["const"], f"{path}: must equal {sch['const']!r}")
    if "pattern" in sch:
        _req(re.fullmatch(sch["pattern"], obj) is not None, f"{path}: does not match {sch['pattern']}")
    if t == "object":
        props = sch.get("properties", {})
        missing = [k for k in sch.get("required", []) if k not in obj]
        _req(not missing, f"{path}: missing keys {missing}")
        _req(len(obj) >= sch.get("minProperties", 0), f"{path}: too few properties")
        extra = [k for k in obj if k not in props]
        addl = sch.get("additionalProperties", True)
        _req(addl is not False or not extra, f"{path}: unexpected keys {extra}")
        for k, v in obj.items():
            if k in props:
                validate_schema(v, props[k], f"{path}.{k}")
            elif isinstance(addl, dict):
                validate_schema(v, addl, f"{path}.{k}")
    if t == "array":
        _req(len(obj) >= sch.get("minItems", 0) and len(obj) <= sch.get("maxItems", len(obj)),
             f"{path}: item count {len(obj)}")
        for i, v in enumerate(obj):
            if "items" in sch:
                validate_schema(v, sch["items"], f"{path}[{i}]")


def check_frozen_values(cfg) -> list:
    errs = [f"{k}={_get(cfg, k)!r} != frozen {v!r}" for k, v in FROZEN.items() if _get(cfg, k) != v]
    clips = cfg["clips"]
    ids = [c["clip_id"] for c in clips]
    if not (len(ids) == N_CLIPS and len(set(ids)) == N_CLIPS and ids == sorted(ids)):
        errs.append("clips must be 18 unique clip_ids in clip_id order")
    if sum(c["expected_frames"] for c in clips) != N_FRAMES:
        errs.append(f"sum expected_frames != {N_FRAMES}")
    if any((c["expected_width"], c["expected_height"]) != (736, 648) for c in clips):
        errs.append("clip size != 736x648")
    ready = [c["readiness"] for c in clips]
    if (ready.count("READY"), ready.count("NEEDS_REVIEW")) != (12, 6) or len(ready) != N_CLIPS:
        errs.append("readiness must be READY12 + NEEDS_REVIEW6")
    if cfg["isolation"]["allowed_video_paths"] != [c["raw_path"] for c in clips]:
        errs.append("isolation.allowed_video_paths != clip raw_path list")
    return errs


def load_config(cfg_path, schema_path) -> dict:
    def no_const(c):
        raise TechnicalFailure(f"non-finite constant {c} in config")
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"), parse_constant=no_const)
    _req(isinstance(cfg, dict) and cfg.get("is_schema_example") is False,
         "is_schema_example must be false: schema example rejected before file access")
    validate_schema(cfg, json.loads(Path(schema_path).read_text(encoding="utf-8"))["json_schema"])
    errs = check_frozen_values(cfg)
    _req(not errs, "frozen value mismatch: " + "; ".join(errs))
    strings = [v for v in _walk(cfg) if isinstance(v, str)]
    _req(not any(s == "0" * 64 or "<LOCAL_PATH>" in s for s in strings), "placeholder path/zero hash present")
    return cfg


def manifest_readiness(doc) -> dict:
    """ASSUMPTION: manifest rows are dicts carrying clip_id + readiness anywhere in the document."""
    out = {}
    for d in _walk(doc):
        if isinstance(d, dict) and isinstance(d.get("clip_id"), str) and d.get("readiness") in ("READY", "NEEDS_REVIEW"):
            _req(out.get(d["clip_id"], d["readiness"]) == d["readiness"], f"manifest conflict {d['clip_id']}")
            out[d["clip_id"]] = d["readiness"]
    return out


def _read_verified(path, want) -> bytes:
    data = Path(path).read_bytes()
    _req(want is not None and hashlib.sha256(data).hexdigest() == want, f"{path}: unpinned or hash mismatch")
    return data


def verify_pins(cfg) -> dict:
    """Hash-only checks. Returns normalized path -> sha map."""
    pins = {_key(_abs(cfg, p)): s for p, s in cfg["pins"].items()}
    _req(len(pins) == len(cfg["pins"]), "duplicate pin paths after normalization")
    for p, s in cfg["pins"].items():
        _req(sha256_file(_abs(cfg, p)) == s, f"pin mismatch {p}")
    for e in cfg["pin_inventory"]:
        p = _abs(cfg, e["path"])
        _req(p.stat().st_size == e["bytes"] and pins.get(_key(p)) == e["sha256"], f"pin_inventory mismatch {p}")
    ex, pr = cfg["extraction"], cfg["preprocessing"]
    for p in (*cfg["reuse"].values(), ex["contract_path"], ex["hand_model_path"], ex["pose_model_path"],
              pr["reference_source_path"]):
        _req(_key(_abs(cfg, p)) in pins, f"required input not pinned: {p}")
    own = [(cfg["source_manifest_path"], cfg["source_manifest_sha256"]),
           (cfg["historical_evaluation_path"], cfg["historical_evaluation_sha256"]),
           (cfg["scoring"]["checkpoint_path"], cfg["scoring"]["checkpoint_sha256"])]
    for c in cfg["clips"]:
        own += [(c["raw_path"], c["raw_sha256"]), (c["baseline_npz_path"], c["baseline_npz_sha256"]),
                (c["baseline_sidecar_path"], c["baseline_sidecar_sha256"])]
    for p, s in own:
        q = _abs(cfg, p)
        _req(sha256_file(q) == s and pins.get(_key(q), s) == s, f"hash mismatch {p}")
    man = manifest_readiness(json.loads(_read_verified(_abs(cfg, cfg["source_manifest_path"]),
                                                       cfg["source_manifest_sha256"])))
    _req(man == {c["clip_id"]: c["readiness"] for c in cfg["clips"]}, "clips != source_manifest READY+NEEDS_REVIEW set")
    _req(not _abs(cfg, cfg["output_dir"]).exists(), "output_dir already exists (fresh directory required)")
    return pins


def preflight(cfg_path, schema_path) -> dict:
    cfg = load_config(cfg_path, schema_path)
    pins = verify_pins(cfg)
    return {"status": "PREFLIGHT_PASS", "config_sha256": sha256_file(cfg_path),
            "schema_doc_sha256": sha256_file(schema_path), "clips": N_CLIPS,
            "frames": sum(c["expected_frames"] for c in cfg["clips"]), "pins_checked": len(pins),
            "runtime_imports": 0, "decodes": 0, "model_loads": 0}


def validate_receipt(path, runner_sha, test_sha) -> dict:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    _req(isinstance(rec, dict), "receipt must be an object")
    _req(rec.get("status") == "PASS", "receipt status != PASS")
    _req(rec.get("runner_sha256") == runner_sha, "receipt runner_sha256 != current runner")
    _req(rec.get("test_sha256") == test_sha, "receipt test_sha256 != current test file")
    n = rec.get("tests_run")
    _req(isinstance(n, int) and not isinstance(n, bool) and n > 0, "receipt tests_run must be positive int")
    return rec


def snapshot(cfg, tracked: dict) -> dict:
    paths = {k: Path(v) for k, v in tracked.items()}
    for p in cfg["pins"]:
        paths[f"pin:{p}"] = _abs(cfg, p)
    for k in ("source_manifest_path", "historical_evaluation_path"):
        paths[k] = _abs(cfg, cfg[k])
    paths["checkpoint"] = _abs(cfg, cfg["scoring"]["checkpoint_path"])
    for c in cfg["clips"]:
        for f in ("raw_path", "baseline_npz_path", "baseline_sidecar_path"):
            paths[f"{c['clip_id']}:{f}"] = _abs(cfg, c[f])
    return {k: sha256_file(p) for k, p in sorted(paths.items())}


# ------------------------------------------------------------------ pure gates / preprocessing
def load_pinned_module(path, pins):
    """Hash-verify bytes, register in sys.modules, execute exactly the verified bytes."""
    import importlib.util
    data = _read_verified(path, pins.get(_key(path)))
    name = Path(path).stem
    _req(name not in sys.modules, f"module {name} already imported")
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None, origin=str(path)))
    mod.__file__ = str(path)
    sys.modules[name] = mod
    try:
        exec(compile(data, str(path), "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def full_view(r: dict) -> dict:
    """Shallow full-cadence view: selected_* -> all stored rows. Never mutates r."""
    v = dict(r)
    v["selected_obs"], v["selected_meta"] = r["stored_obs"], r["decoded_meta"]
    return v


def _is_exact(res) -> bool:
    return bool(res["equal"] if isinstance(res, dict) else res)


def check_cache_arrays(arrays: dict, fields, n: int) -> dict:
    import numpy as np
    _req(set(arrays) == {"schema", *fields}, f"cache keys {sorted(arrays)} != plain schema+{list(fields)}")
    sch = np.asarray(arrays["schema"])
    _req(sch.size == 1 and sch.reshape(()).item() == CACHE_SCHEMA, f"cache schema {sch!r} != {CACHE_SCHEMA}")
    fid, ts = arrays["frame_id"], arrays["timestamp_us"]
    _req(fid.shape == (n,) and np.array_equal(fid, np.arange(n)), "cache frame_id != 0..N-1")
    _req(ts.shape == (n,) and ts.dtype.kind in "iu" and int(ts[0]) == 0 and bool(np.all(np.diff(ts) > 0)),
         "cache timestamp_us must be integer, start 0, strictly increasing")
    for f in fields:
        _req(arrays[f].shape[:1] == (n,), f"cache {f} rows != {n}")
    return arrays


def check_decoded(r: dict, clip: dict) -> list:
    """Every decoded RGB frame is (expected_height, expected_width, 3) (no resize), all processed (decoded_meta),
    and the arm-level string r['time_base'] == clip time_base."""
    import numpy as np
    dm, n = r["decoded_meta"], clip["expected_frames"]
    missing = [k for k in ("rgb_shape", "processed") if k not in dm]
    if missing:
        return [f"decoded_meta lacks {missing}; actual keys {sorted(dm)}"]
    errs = []
    shp = np.asarray(dm["rgb_shape"])
    want = (clip["expected_height"], clip["expected_width"], 3)
    if not (shp.shape == (n, 3) and bool(np.all(shp == np.asarray(want)))):
        errs.append(f"decoded RGB shapes != {n} x {want}")
    tb = r.get("time_base")
    if not (isinstance(tb, str) and tb == clip["time_base"]):
        errs.append(f"time_base {tb!r} != {clip['time_base']!r}")
    pr = np.asarray(dm["processed"])
    if not (pr.shape == (n,) and pr.dtype == bool and bool(pr.all())):
        errs.append("processed flags not all True for every decoded frame")
    return errs


def cache_gate(stored: dict, cache: dict, fields, bitexact) -> list:
    return [f"{f}: not dtype/shape/bytes exact vs cache" for f in fields
            if f not in stored or not _is_exact(bitexact(stored[f], cache[f]))]


def official_prep(full: dict, fns: dict):
    """to_native_like -> active_span inclusive -> crop -> build_channels -> collate(label 0). None if no span."""
    import numpy as np
    xy = np.asarray(full["xy_px"], dtype=np.float32)
    nat, present = fns["to_native_like"](xy, np.asarray(full["usable"], dtype=bool))
    span = fns["active_span"](present)
    if span is None:
        return None
    lo, hi = int(span[0]), int(span[1])
    _req(0 <= lo <= hi < len(xy), f"span out of range {span}")
    nat, present = nat[lo:hi + 1], present[lo:hi + 1]
    ts = np.asarray(full["timestamp_us"], dtype=np.int64)[lo:hi + 1]
    ch = fns["build_channels"](nat, present, ts)
    x, pad, _ = fns["collate"]([dict(ch, label=0)])
    x, pad = np.asarray(x), np.asarray(pad)
    t = hi - lo + 1
    _req(x.dtype == np.float32 and x.shape == (1, t, fns["n_channels"]), f"unexpected x {x.dtype} {x.shape}")
    _req(bool(np.isfinite(x).all()), "non-finite model input")
    _req(pad.shape == (1, t) and bool(np.all(pad == 1)), "padding mask must be all real frames")
    return {"x": x, "pad": pad, "span": [lo, hi],
            "frame_id": np.asarray(full["frame_id"])[lo:hi + 1].tolist(), "timestamp_us": ts.tolist(),
            "present_shape": list(np.shape(present)), "present_count": int(np.count_nonzero(present)),
            "body_scale": scale_json(ch.get("body_scale")), "hand_scales": scale_json(ch.get("hand_scales"))}


def prep_arm(full: dict, fns: dict, arm: str):
    try:
        p = official_prep(full, fns)
    except TechnicalFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - any non-span preprocessing error is technical, reraised
        raise TechnicalFailure(f"{arm} preprocessing {type(exc).__name__}: {exc}") from exc
    if p is None:
        _req(arm == "C_hand_image", f"{arm}: active_span None = baseline reproduction failure")
    return p


def scale_json(v):
    import numpy as np
    if isinstance(v, dict):
        return {str(k): scale_json(x) for k, x in v.items()}
    if v is None:
        return {"value": None, "status": "absent"}
    a = np.asarray(v, dtype=np.float64)
    vals = [x if math.isfinite(x) else None for x in a.ravel().tolist()]
    return {"value": vals[0] if a.ndim == 0 else vals, "shape": list(a.shape),
            "nonfinite_count": vals.count(None),
            "status": "finite" if None not in vals else "nonfinite_serialized_null_missing"}


def input_record(p: dict, array_sha256) -> dict:
    out = {k: v for k, v in p.items() if k not in ("x", "pad")}
    out.update(x_sha256=array_sha256(p["x"]), x_shape=list(p["x"].shape), x_dtype=str(p["x"].dtype),
               pad_sha256=array_sha256(p["pad"]))
    return out


def decide(b_rows: list, c_rows: list) -> dict:
    """Fixed denominator 18. C unavailable stays in the denominator and fails the frozen gate."""
    unavailable = [r["clip_id"] for r in c_rows if r["status"] == UNAVAILABLE]
    b_ok = sum(r["correct"] is True for r in b_rows)
    c_ok = sum(r.get("correct") is True for r in c_rows)
    regressions = [c["clip_id"] for b, c in zip(b_rows, c_rows) if b["correct"] and c.get("correct") is False]
    passed = len(b_rows) == len(c_rows) == N_CLIPS and b_ok == N_CLIPS and c_ok == N_CLIPS and not unavailable
    return {"denominator": N_CLIPS, "B_correct": b_ok, "C_correct": c_ok, "C_unavailable": len(unavailable),
            "C_unavailable_clips": unavailable, "C_regressions": regressions,
            "expected_forwards": len(b_rows) + len(c_rows) - len(unavailable),
            "code": "PASS_FROZEN_MODE_REGRESSION" if passed else "FAIL_FROZEN_MODE_REGRESSION"}


def expected_counts(budgets: dict) -> dict:
    """Extraction view of the frozen budgets in hip.actual_counts key format (per-arm selected frames)."""
    out = {k: v for k, v in budgets.items() if k not in NON_EXTRACTION_BUDGETS and k != "selected_frames_per_arm"}
    out.update({f"selected_frames_per_arm:{tag}": budgets["selected_frames_per_arm"] for tag, _ in ARMS})
    return out


def check_extraction_counts(actual: dict, expected: dict, counters: dict) -> list:
    errs = [f"{k}: actual {actual.get(k, 'MISSING')} != {v}" for k, v in expected.items() if actual.get(k) != v]
    errs += [f"counter {k}={counters.get(k)} != 0" for k in ("tcn_forwards", "model_loads", "camera_calls",
                                                              "training_runs", "downloads") if counters.get(k) != 0]
    if counters.get("landmarker_closed") != counters.get("landmarker_constructed"):
        errs.append("landmarker_closed != landmarker_constructed")
    return errs


def historical_top1(doc, clip_ids) -> dict:
    """Object with a 'rows' list (other top-level keys allowed); reference = group 'public' + condition 'intact'
    + id in clips -> models.control.full93.pred."""
    _req(isinstance(doc, dict), "historical evaluation must be an object")
    rows = doc.get("rows")
    _req(isinstance(rows, list) and all(isinstance(r, dict) for r in rows),
         "historical evaluation 'rows' must be a list of row objects")
    out = {}
    for row in rows:
        if row.get("group") == "public" and row.get("condition") == "intact" and row.get("id") in clip_ids:
            _req(row["id"] not in out, f"historical: more than one public/intact row for {row['id']}")
            pred = _get(row, "models.control.full93.pred")
            _req(isinstance(pred, str), f"historical: {row['id']} models.control.full93.pred is not a string")
            out[row["id"]] = pred
    _req(set(out) == set(clip_ids), f"historical public/intact rows missing {sorted(set(clip_ids) - set(out))}")
    return out


# ------------------------------------------------------------------ run
def _project_modules():
    return sorted(m for m in sys.modules if m.split(".")[0] in ("wis_lab", "pilot"))


def check_extract_signature(fn) -> None:
    params = inspect.signature(fn).parameters
    _req(set(params) == set(EXTRACT_KW) and all(p.kind is p.KEYWORD_ONLY for p in params.values()),
         f"extract_arm parameters {list(params)} != keyword-only {list(EXTRACT_KW)}")


def hand_slices(lm_mod) -> tuple:
    s = (("left", lm_mod.LEFT_HAND_SLICE), ("right", lm_mod.RIGHT_HAND_SLICE))
    _req(all(isinstance(sl, slice) for _, sl in s), "landmarker hand slices must be slice objects")
    return s


def extract_one(cad, hip, deps, lm_mod, clip, raw, tag, mode, counters, registry, instrs):
    """One fresh instrumented Landmarker per arm; factory tag is exactly the arm tag, passed inside deps."""
    n, before = clip["expected_frames"], len(instrs)
    d = dataclasses.replace(deps, landmarker_factory=hip.make_factory(deps.landmarker_factory, lm_mod, tag, mode, instrs))
    r = cad.extract_arm(path=str(raw), clip_id=clip["clip_id"], arm=cad.ARM_A, n_frames=n,
                        n_selected=(n + 1) // 2, deps=d, counters=counters, registry=registry)
    _req(len(instrs) == before + 1, f"{clip['clip_id']} {tag}: expected exactly one instrumented landmarker")
    _req(instrs[-1].tag == tag, f"{clip['clip_id']}: instr tag {instrs[-1].tag!r} != {tag!r}")
    return r, instrs[-1]


def _instr_record(ins) -> dict:
    return {"tag": ins.tag, "mode": ins.mode, "tasks": [t.record() for t in ins.tasks], "frames": ins.frames,
            "errors": ins.errors, "provenance": ins.provenance, "construction": ins.construction}


def _execute(cfg, pins, out: Path, st: dict) -> dict:
    import numpy as np
    clips, budgets = cfg["clips"], cfg["budgets"]
    ids = [c["clip_id"] for c in clips]
    expected = expected_counts(budgets)
    st["stage"] = "historical"  # hash-pinned in verify_pins; validated before any import/decode
    hist = historical_top1(json.loads(_read_verified(_abs(cfg, cfg["historical_evaluation_path"]),
                                                     cfg["historical_evaluation_sha256"])), ids)
    _req(sum(hist[c["clip_id"]] == c["label"] for c in clips) == cfg["scoring"]["historical_expected_correct"],
         "historical evaluation is not 18/18")
    st["stage"] = "import_reuse"
    cad = load_pinned_module(_abs(cfg, cfg["reuse"]["cadence_module_path"]), pins)
    hip = load_pinned_module(_abs(cfg, cfg["reuse"]["hand_image_module_path"]), pins)
    _req(tuple(map(tuple, hip.ARMS)) == ARMS, f"helper ARMS {hip.ARMS} != {ARMS}")
    _req(cad.ARM_A == cfg["extraction"]["cadence_helper_arm"], f"cadence ARM_A {cad.ARM_A!r} != config")
    check_extract_signature(cad.extract_arm)
    inspect.signature(hip.compare_obs).bind(*([None] * 6))
    covered = set(hip.actual_counts(cad.new_counters(), []))
    missing = [k for k in expected if k not in covered]
    _req(not missing, f"helper actual_counts lacks budget keys {missing}")

    st["stage"] = "runtime_deps"
    oldcfg = json.loads(_read_verified(_abs(cfg, cfg["reuse"]["cadence_config_path"]),
                                       pins.get(_key(_abs(cfg, cfg["reuse"]["cadence_config_path"])))))
    oldplan = cad.validate_config(oldcfg, cad.DEFAULT_ROOTS)  # normalization only; old run/deadline never called
    _req(not _project_modules(), f"project modules preimported: {_project_modules()}")
    deps = cad.load_runtime_deps(oldplan, cad.DEFAULT_ROOTS, cad.sha256_file)
    rp = oldplan["role_paths"]
    for role, p in (("observation_extraction_contract", cfg["extraction"]["contract_path"]),
                    ("mediapipe_hand_model", cfg["extraction"]["hand_model_path"]),
                    ("mediapipe_pose_model", cfg["extraction"]["pose_model_path"]),
                    ("frozen_tcn_checkpoint", cfg["scoring"]["checkpoint_path"])):
        _req(_key(rp[role]) == _key(_abs(cfg, p)), f"{role}: pinned Deps path != config path")
    _req(cad.TCN_SHA256 == cfg["scoring"]["checkpoint_sha256"], "cadence TCN_SHA256 != config checkpoint")
    _req(cad.CPU_THREADS == cfg["scoring"]["cpu_threads"], "cadence CPU_THREADS != config")
    wi = sys.modules["pilot.web_input"]
    lm_mod = sys.modules["wis_lab.capture.landmarker"]
    _req(deps.to_native_like is wi.to_native_like, "to_native_like is not pinned pilot.web_input")
    fns = {"to_native_like": deps.to_native_like, "active_span": wi.active_span,
           "build_channels": deps.build_channels, "collate": deps.collate, "n_channels": deps.n_channels}
    st["provenance"] = {"versions": deps.versions, "import_audit": deps.import_audit}
    slices = hand_slices(lm_mod)

    writer = cad.ResultWriter(out)
    counters, registry, instrs, persisted_all = cad.new_counters(), [], [], {}
    st.update(counters=counters, instrs=instrs, persisted=persisted_all)
    fields, pose_fields = cfg["gates"]["baseline_fields"], cfg["gates"]["bc_pose_fields"]
    rows, inputs, frames_done = [], {}, 0
    st["rows"] = rows
    for seq, clip in enumerate(clips, 1):
        cid, n = clip["clip_id"], clip["expected_frames"]
        st["stage"] = f"extract:{cid}"
        cache = check_cache_arrays(dict(np.load(io.BytesIO(_read_verified(
            _abs(cfg, clip["baseline_npz_path"]), clip["baseline_npz_sha256"])), allow_pickle=False)), fields, n)
        raw = _abs(cfg, clip["raw_path"])
        _req(sha256_file(raw) == clip["raw_sha256"], f"{cid}: raw changed before decode")
        runs, persisted = {}, persisted_all.setdefault(cid, {})
        for arm, mode in ARMS:
            r, ins = extract_one(cad, hip, deps, lm_mod, clip, raw, arm, mode, counters, registry, instrs)
            # persist closed arm + instrumentation BEFORE any gate so a failing gate leaves inspectable evidence
            persisted[arm] = cad.persist_arm(writer, seq, arm, r)
            writer.json(f"{seq:02d}_{cid}_{arm}.instr.json", _instr_record(ins))
            runs[arm] = (r, ins)
            errs = check_decoded(r, clip) + list(hip.gate_diag(ins, full_view(r), n)) + list(ins.errors)
            _req(not errs, f"{cid} {arm} instrumentation/decode: {errs}")
            _req(np.array_equal(r["decoded_meta"]["frame_id"], np.arange(n)), f"{cid} {arm}: decoded ids != 0..N-1")
        (rb, ib), (rc, ic) = runs["B_video"], runs["C_hand_image"]
        errs = cache_gate(rb["stored_obs"], cache, fields, hip.bitexact)
        errs += ["C " + e for e in cache_gate(rc["stored_obs"], cache, ["frame_id", "timestamp_us"], hip.bitexact)]
        _req(set(rb["decoded_meta"]) == set(rc["decoded_meta"]), f"{cid}: B/C decoded_meta keys differ")
        errs += list(hip.gate_bc(full_view(rb), full_view(rc), pose_fields, tuple(rb["decoded_meta"])))
        _req(not errs, f"{cid} extraction gates: {errs}")
        frames_done += n
        for k, v in (("decoded_frames", 2 * frames_done), ("landmarker_process_calls", 2 * frames_done),
                     ("decoder_opened", 2 * seq), ("landmarker_constructed", 2 * seq),
                     ("landmarker_closed", 2 * seq)):
            _req(counters.get(k) == v, f"after {cid}: counter {k}={counters.get(k)} != {v}")

        st["stage"] = f"prep:{cid}"
        p_cache = prep_arm(cache, fns, "B_video")
        p_b = prep_arm(rb["stored_obs"], fns, "B_video")
        p_c = prep_arm(rc["stored_obs"], fns, "C_hand_image")
        _req(p_cache["span"] == p_b["span"] and _is_exact(hip.bitexact(p_cache["x"], p_b["x"]))
             and _is_exact(hip.bitexact(p_cache["pad"], p_b["pad"]))
             and (p_cache["body_scale"], p_cache["hand_scales"]) == (p_b["body_scale"], p_b["hand_scales"]),
             f"{cid}: B input from cache != B input from new extraction")
        inputs[cid] = {"B": p_b, "C": p_c}
        bs = p_b["span"]
        rows.append({
            "seq": seq, "clip_id": cid, "word_id": clip["word_id"], "label": clip["label"],
            "readiness": clip["readiness"], "n_frames": n, "raw_sha256": clip["raw_sha256"],
            "baseline_npz_sha256": clip["baseline_npz_sha256"], "cache_schema": CACHE_SCHEMA,
            "evaluation_selection": "all_stored_obs", "persisted": persisted,
            "B": {"input": input_record(p_b, cad.array_sha256), "cache_input_x_sha256": cad.array_sha256(p_cache["x"])},
            "C": ({"status": UNAVAILABLE, "forward": False, "reason": "active_span(C stored_obs) returned None"}
                  if p_c is None else {"status": "AVAILABLE", "input": input_record(p_c, cad.array_sha256)}),
            "span_change": ({"B_span": bs, "C_span": None} if p_c is None else
                            {"B_span": bs, "C_span": p_c["span"], "d_lo": p_c["span"][0] - bs[0],
                             "d_hi": p_c["span"][1] - bs[1], "changed": p_c["span"] != bs}),
            "observation_diff": hip.compare_obs(rb["stored_obs"], rc["stored_obs"], np.arange(n), slices,
                                                ib.frames, ic.frames)})
    st["stage"] = "extraction_counts"
    actual = hip.actual_counts(counters, instrs)
    errs = check_extraction_counts(actual, expected, counters)
    _req(not errs, f"extraction counts: {errs}")

    st["stage"] = "model_load"
    model, classes, model_info = deps.load_model()
    counters["model_loads"] += 1
    _req(len(classes) == cfg["scoring"]["n_classes"] == len(set(classes)), "checkpoint classes != 93 unique")
    s0 = deps.state_hash(model)
    st["model_state"] = {"payload_state_sha256": model_info["payload_state_sha256"], "before": s0}
    _req(s0 == model_info["payload_state_sha256"], "in-memory model state != checkpoint payload_state_sha256")

    def score(row, p):
        logits = np.asarray(deps.forward(model, p["x"], p["pad"]))
        counters["tcn_forwards"] += 1
        _req(logits.shape == (len(classes),) and bool(np.isfinite(logits).all()), f"{row['clip_id']}: bad logits")
        return {"logits": logits.tolist(), "score": cad.rank_against_label(logits, classes, row["label"])}

    for row in rows:  # all B first
        st["stage"] = f"score_B:{row['clip_id']}"
        row["B"].update(score(row, inputs[row["clip_id"]]["B"]))
        row["B"]["historical_top1"] = hist[row["clip_id"]]
        row["B"]["historical_top1_match"] = row["B"]["score"]["top1"] == hist[row["clip_id"]]
    bad = [r["clip_id"] for r in rows if not (r["B"]["score"]["correct"] and r["B"]["historical_top1_match"])]
    _req(not bad, f"BASELINE_REPRODUCTION_FAIL B not 18/18 or top1 != historical: {bad}")
    for row in rows:
        if inputs[row["clip_id"]]["C"] is not None:
            st["stage"] = f"score_C:{row['clip_id']}"
            row["C"].update(score(row, inputs[row["clip_id"]]["C"]))
    s1 = deps.state_hash(model)
    st["model_state"].update(after=s1, unchanged=s1 == s0)
    _req(s1 == s0, "in-memory model state changed during forwards")

    st["stage"] = "final_counts"
    b_rows = [{"clip_id": r["clip_id"], "correct": r["B"]["score"]["correct"]} for r in rows]
    c_rows = [{"clip_id": r["clip_id"], "status": r["C"]["status"],
               "correct": r["C"]["score"]["correct"] if "score" in r["C"] else None} for r in rows]
    decision = decide(b_rows, c_rows)
    _req(counters["tcn_forwards"] == decision["expected_forwards"] <= cfg["scoring"]["forward_calls_max"]
         and decision["expected_forwards"] <= budgets["forward_calls_max"],
         f"forwards {counters['tcn_forwards']} != 18 + available_C {decision['expected_forwards']}")
    _req(counters["model_loads"] == budgets["model_loads"] == 1, "model_loads != 1")
    final_actual = hip.actual_counts(counters, instrs)
    errs = check_extraction_counts(final_actual, expected, dict(counters, tcn_forwards=0, model_loads=0))
    _req(not errs, f"final extraction counts: {errs}")
    changes = [r["span_change"] for r in rows if r["span_change"]["C_span"] is not None]
    return {"classes": classes, "model": model_info, "model_state": st["model_state"], "rows": rows,
            "decision": decision, "counters": counters, "actual_counts": final_actual,
            "task_counts": hip.task_counts(instrs), "timing": hip.timing(instrs), "cache_schema": CACHE_SCHEMA,
            "span_change_summary": {"available": len(changes), "changed": sum(c["changed"] for c in changes),
                                    "sum_abs_d_lo": sum(abs(c["d_lo"]) for c in changes),
                                    "sum_abs_d_hi": sum(abs(c["d_hi"]) for c in changes),
                                    "unavailable": decision["C_unavailable"]},
            "writer": writer}


def run(cfg_path, schema_path, receipt_path) -> int:
    runner = Path(__file__).resolve()
    test = runner.with_name("test_nikl18.py")
    runner_sha, test_sha = sha256_file(runner), sha256_file(test)
    receipt = validate_receipt(receipt_path, runner_sha, test_sha)
    cfg = load_config(cfg_path, schema_path)
    pins = verify_pins(cfg)
    tracked = {"config": cfg_path, "runner": runner, "test": test, "receipt": receipt_path, "schema_doc": schema_path,
               "cadence_module": _abs(cfg, cfg["reuse"]["cadence_module_path"]),
               "hand_image_module": _abs(cfg, cfg["reuse"]["hand_image_module_path"])}
    before = snapshot(cfg, tracked)
    out = _abs(cfg, cfg["output_dir"])
    out.mkdir(exist_ok=False)
    st = {"stage": "start", "rows": [], "counters": None, "instrs": [], "provenance": None, "persisted": {},
          "model_state": None}
    prov = {"runner_sha256": runner_sha, "test_sha256": test_sha, "receipt": receipt,
            "receipt_sha256": before["receipt"], "config_sha256": before["config"],
            "cadence_module_sha256": before["cadence_module"], "hand_image_module_sha256": before["hand_image_module"]}
    try:
        res = _execute(cfg, pins, out, st)
        st["stage"] = "after_hashes"
        after = snapshot(cfg, tracked)
        _req(after == before, f"inputs changed during run: {sorted(k for k in before if after.get(k) != before[k])}")
        writer = res.pop("writer")
        res.update(status="COMPLETED_PENDING_INDEPENDENT_VERIFICATION", schema=CONFIG_SCHEMA,
                   evaluation_selection="all_stored_obs",
                   selection_note="helper selected_obs is internal even subset; all gates/scoring use stored_obs",
                   provenance=dict(prov, **(st["provenance"] or {})), hashes_before=before, hashes_after=after,
                   independent_forward_calls="separate receipt; not counted here")
        writer.json("RESULT.json", res)
        writer.json("MANIFEST.json", {"files": writer.file_hashes()})
        print(json.dumps({"status": res["status"], "decision": res["decision"]}, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        fail = {"status": "TECHNICAL_FAILURE", "stage": st["stage"], "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(), "provenance": prov,
                "runtime_provenance": st["provenance"], "counters": st["counters"], "partial_rows": st["rows"],
                "persisted_arms": st["persisted"], "model_state": st["model_state"], "hashes_before": before}
        hip_mod = lambda: sys.modules[Path(cfg["reuse"]["hand_image_module_path"]).stem]  # noqa: E731
        for name, fn in (("actual_counts", lambda: hip_mod().actual_counts(st["counters"], st["instrs"])),
                         ("task_counts", lambda: dict(hip_mod().task_counts(st["instrs"]))),
                         ("instr_tasks", lambda: [{"tag": i.tag, "mode": i.mode, "frames": len(i.frames),
                                                   "errors": i.errors, "tasks": [t.record() for t in i.tasks]}
                                                  for i in st["instrs"]]),
                         ("hashes_after", lambda: snapshot(cfg, tracked))):
            try:
                fail[name] = fn()
            except Exception as e2:  # noqa: BLE001 - diagnostic only; original failure preserved
                fail[name] = f"unavailable: {type(e2).__name__}: {e2}"
        _write_x(out / "FAILURE.json", fail)
        print(f"TECHNICAL_FAILURE at {st['stage']}: {exc}", file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt):
            raise
        return 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--schema", type=Path, default=Path(__file__).resolve().with_name("SCHEMA_EXAMPLE.json"))
    ap.add_argument("--run", action="store_true", help="explicit run (default: preflight only)")
    ap.add_argument("--test-receipt", type=Path)
    a = ap.parse_args(argv)
    if not a.run:
        try:
            print(json.dumps(preflight(a.config, a.schema), indent=2))
            return 0
        except TechnicalFailure as exc:
            print(f"PREFLIGHT_FAIL: {exc}", file=sys.stderr)
            return 1
    if a.test_receipt is None:
        ap.error("--run requires --test-receipt")
    return run(a.config, a.schema, a.test_receipt)


if __name__ == "__main__":
    sys.exit(main())
