"""Unsupported-frame factorial diagnostic on the frozen masked TCN (see mask_factorial.py).

Same frozen weights, exact same restored 'linear' input x / time grid as the prior irregular-time
run, four arms: grid_baseline (= prior linear logits), pool_only, blocks_only, blocks_and_pool
(primary). Diagnostic only: no training, no thresholds, no method search, no deployment gate, no
camera, no network. Prior algorithms are imported unmodified from pinned files:
  run_audit (R), timegrid (TG)            from cfg.helper_dir
  irregular_audit (I), corruptions (C)    from cfg.irregular_dir

Run: python run_factorial.py --config <config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import mask_factorial as M

GROUPS = ("native", "mediapipe")
CASES = ("isolated_drop", "burst_drop", "clock_noise")
PRIOR_ARMS = ("original", "corrupted", "previous", "linear")
REPORT_ARMS = ("prior_original", "prior_corrupted") + M.ARMS
NEW_ARMS = M.ARMS[1:]
PRIMARY_CASE = "burst_drop"
HELPER_FILES = ("run_audit.py", "timegrid.py")
IRREGULAR_FILES = ("irregular_audit.py", "corruptions.py")
API_MODULES = ("tools_prep_union93_candidate", "pilot.data", "pilot.web_input", "framerate_impl")
MODEL_API_MODULE = "model_masked_v1"
CAMERA_MODULES = ("cv2", "mediapipe")
REPRO_INDICES = {"native": [0, 349, 698], "mediapipe": [0, 56, 112]}
COMPARISONS = tuple((s, t) for s in ("grid_baseline", "prior_corrupted", "prior_original") for t in NEW_ARMS)
LIMITS = [
    "Mechanism diagnostic on the same frozen model and the exact prior restored inputs; no training, "
    "no thresholds, no method search, no deployment candidate and no model promotion regardless of result.",
    "Using the padding mask for internal unobserved frames is a new semantic use (padding originally meant "
    "batch membership) that the model was never trained with; a result does not prove a padding_mask bug "
    "nor that unsupported frames are the exclusive cause of the prior regression.",
    "Cells whose support is all ones are algebraically identical in all four arms and keep their prior "
    "predictions; prior isolated_drop/clock_noise failures remain regardless of this result.",
    "native = reused development rows, mediapipe = training-exposed rows; groups are never pooled and neither "
    "is an independent test or webcam feedback. Top-1/score movements are not real-world accuracy guarantees.",
]


class FactorialError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def norm(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def under(path, root) -> bool:
    try:
        return os.path.commonpath([norm(path), norm(root)]) == norm(root)
    except ValueError:
        return False


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json_exclusive(path, value, strict=True):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=not strict,
                  default=None if strict else _plain)
        fh.write("\n")


def jsonable(v):
    return json.loads(json.dumps(v, default=_plain))


def code_hashes(cfg) -> dict:
    here = os.path.dirname(os.path.abspath(__file__))
    out = {f: sha256_file(os.path.join(here, f)) for f in ("run_factorial.py", "mask_factorial.py")}
    out.update({f"helper/{f}": sha256_file(os.path.join(cfg["helper_dir"], f)) for f in HELPER_FILES})
    out.update({f"irregular/{f}": sha256_file(os.path.join(cfg["irregular_dir"], f)) for f in IRREGULAR_FILES})
    return out


def precheck(cfg):
    if not isinstance(cfg, dict):
        raise FactorialError("config must be a JSON object")
    for k in ("helper_dir", "irregular_dir", "output_dir", "prior_result", "plan_file", "lab_root"):
        if not isinstance(cfg.get(k), str) or not os.path.isabs(cfg[k]):
            raise FactorialError(f"config {k} must be an absolute string")
    ps = cfg.get("prior_scores")
    if not isinstance(ps, dict) or set(ps) != set(GROUPS) or any(
            not isinstance(v, str) or not os.path.isabs(v) for v in ps.values()):
        raise FactorialError("prior_scores must be {native: abs path, mediapipe: abs path}")
    pins = cfg.get("pins")
    if not isinstance(pins, dict) or not pins or any(not os.path.isabs(p) for p in pins):
        raise FactorialError("pins must be a non-empty object of absolute paths")
    pinned = {norm(p) for p in pins}
    need = [os.path.join(cfg["helper_dir"], f) for f in HELPER_FILES] + \
           [os.path.join(cfg["irregular_dir"], f) for f in IRREGULAR_FILES] + \
           [cfg["prior_result"], cfg["plan_file"], *ps.values()]
    miss = [p for p in need if norm(p) not in pinned]
    if miss:
        raise FactorialError("required files not pinned", {"missing": miss})
    if cfg.get("repro_indices") != REPRO_INDICES:
        raise FactorialError(f"repro_indices must be exactly {REPRO_INDICES}")
    out = cfg["output_dir"]
    roots = list(cfg.get("native_roots") or []) + list(cfg.get("mp_roots") or [])
    forbidden = roots + [cfg["helper_dir"], cfg["irregular_dir"], cfg["lab_root"],
                         os.path.dirname(cfg["prior_result"]), *(os.path.dirname(v) for v in ps.values())]
    if any(isinstance(r, str) and os.path.isabs(r) and under(out, r) for r in forbidden):
        raise FactorialError("output_dir lies inside a source/helper/lab/prior-output directory")


def main(argv=None):
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    cfg_bytes = Path(cfg_path).read_bytes()
    cfg = json.loads(cfg_bytes.decode("utf-8-sig"))
    precheck(cfg)
    out_dir = Path(cfg["output_dir"])
    if os.path.lexists(out_dir):
        raise FactorialError(f"output_dir {out_dir} exists; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=False)
    code = code_hashes(cfg)
    state = {"stage": "start", "group": None, "id": None, "case": None, "prepared_cells": {g: 0 for g in GROUPS},
             "forwards": {"repro_actual": 0, "repro_manual": 0, "identity_actual": 0, "identity_manual": 0,
                          "new_diagnostic": 0},
             "reused_cells": 0, "details": None, "pins_before": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "group": state["group"], "id": state["id"],
                   "case": state["case"], "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "counts": {"cells_prepared": state["prepared_cells"], "forwards": state["forwards"],
                              "reused_cells": state["reused_cells"], "training": 0, "camera": 0,
                              "user_data": 0, "downloads": 0},
                   "pins_before": state["pins_before"], "code_sha256": code,
                   "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                   "seconds": round(time.time() - started, 2),
                   "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            write_json_exclusive(out_dir / "FAILURE.json", failure, strict=False)
        except Exception as wexc:
            print(f"could not write FAILURE.json: {type(wexc).__name__}: {wexc}", file=sys.stderr)
        raise


def _import_pinned(name, directory, pin_of):
    if name in sys.modules:
        raise FactorialError(f"{name} already imported from elsewhere")
    sys.path.insert(0, directory)
    try:
        mod = __import__(name)
    finally:
        sys.path.remove(directory)
    p = os.path.join(directory, f"{name}.py")
    if norm(mod.__file__) != norm(p) or sha256_file(p) != pin_of.get(norm(p)):
        raise FactorialError(f"{name} is not the pinned file in {directory} ({mod.__file__})")
    return mod


def _camera_check():
    loaded = [m for m in CAMERA_MODULES if m in sys.modules]
    if loaded:
        raise FactorialError(f"camera modules imported: {loaded}")


def _run(cfg, cfg_path, cfg_bytes, out_dir, code, state, started):
    # 1. pins first
    state["stage"] = "pins"
    pins_before = {p: sha256_file(p) for p in cfg["pins"]}
    state["pins_before"] = pins_before
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise FactorialError("pin mismatch", {"mismatched": bad})
    pin_of = {norm(p): d for p, d in cfg["pins"].items()}

    # 2. prior helpers, unmodified (corruptions must be importable before irregular_audit)
    state["stage"] = "helper_import"
    pre = [m for m in ("run_audit", "timegrid", "irregular_audit", "corruptions") if m in sys.modules]
    if pre:
        raise FactorialError(f"prior modules already imported from elsewhere: {pre}")
    R = _import_pinned("run_audit", cfg["helper_dir"], pin_of)     # imports timegrid from helper_dir
    TG = sys.modules.get("timegrid")
    if TG is None or R.TG is not TG or norm(TG.__file__) != norm(os.path.join(cfg["helper_dir"], "timegrid.py")) \
            or sha256_file(TG.__file__) != pin_of.get(norm(TG.__file__)):
        raise FactorialError("timegrid is not the pinned helper_dir file used by run_audit")
    C = _import_pinned("corruptions", cfg["irregular_dir"], pin_of)
    I = _import_pinned("irregular_audit", cfg["irregular_dir"], pin_of)
    if I.C is not C:
        raise FactorialError("irregular_audit uses a different corruptions module")

    def frozen():
        return {"R.N_CLASSES": R.N_CLASSES, "R.NATIVE_N": R.NATIVE_N, "R.MP_N": R.MP_N,
                "R.FIXED_THREADS": R.FIXED_THREADS, "R.FIXED_TOLERANCE": R.FIXED_TOLERANCE,
                "TG.NOMINAL_STEP_US": TG.NOMINAL_STEP_US, "TG.EXACT_TOL_US": TG.EXACT_TOL_US,
                "TG.CLOCK_TOL_US": TG.CLOCK_TOL_US, "TG.N_JOINTS": TG.N_JOINTS, "C.N_JOINTS": C.N_JOINTS,
                "C.CASES": list(C.CASES), "C.DOMAIN": C.DOMAIN, "C.MAX_SOURCE_GAP": dict(C.MAX_SOURCE_GAP),
                "C.MAX_OFFSET_US": C.MAX_OFFSET_US, "I.ARMS": list(I.ARMS), "I.CASES": list(I.CASES),
                "I.REPRO_INDICES": I.REPRO_INDICES}
    frozen_before = frozen()
    expect = {"R.N_CLASSES": 93, "R.NATIVE_N": 699, "R.MP_N": 113, "R.FIXED_THREADS": 2, "R.FIXED_TOLERANCE": 1e-5,
              "TG.NOMINAL_STEP_US": 1e6 / 30.0, "TG.EXACT_TOL_US": 1, "TG.CLOCK_TOL_US": 2, "TG.N_JOINTS": 67,
              "C.N_JOINTS": 67, "C.CASES": list(CASES), "I.ARMS": list(PRIOR_ARMS), "I.CASES": list(CASES),
              "I.REPRO_INDICES": REPRO_INDICES}
    off = {k: frozen_before[k] for k, v in expect.items() if frozen_before[k] != v}
    if off:
        raise FactorialError("prior constants differ from the frozen method", {"constants": off})

    state["stage"] = "config"
    R.validate_config(cfg)
    tol = float(cfg["tolerance"])
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}
    lab = cfg["lab_root"]
    sys.path[1:1] = [lab, os.path.join(lab, "src"), cfg["pilot_src"], cfg["loader_dir"]]

    # 3. rows + listed source hashes before any array/model load
    state["stage"] = "metadata"
    rows = {"native": R.load_native_rows(cfg["native_metadata"]), "mediapipe": R.load_mp_rows(cfg["mp_manifest"])}
    state["stage"] = "source_hashes_before"
    sources_before = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    for g in GROUPS:
        R.check_listed(rows[g], sources_before[g], g)

    state["stage"] = "saved_material"
    classes, a_native, _ = R.load_saved_native(cfg["saved_native_logits"], rows["native"])
    a_mp, mp_arms = R.load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    saved_a = {"native": a_native, "mediapipe": a_mp}

    # 4. prior irregular-time scores + result (arms addressed by name only)
    state["stage"] = "prior_material"
    prior_lg, prior_rows = {}, {}
    pres = json.loads(Path(cfg["prior_result"]).read_text(encoding="utf-8"))
    if list(pres.get("arm_order", [])) != list(PRIOR_ARMS) or list(pres.get("cases", [])) != list(CASES):
        raise FactorialError("prior result arm/case order differs")
    for g in GROUPS:
        p = cfg["prior_scores"][g]
        with np.load(p, allow_pickle=False) as z:
            if any(k not in z.files for k in ("logits", "cases", "arms", "ids", "classes", "true_indices")):
                raise FactorialError(f"prior scores {g} lack keys; has {z.files}")
            lg = np.asarray(z["logits"], np.float64)
            pc, pa = [str(v) for v in z["cases"].tolist()], [str(v) for v in z["arms"].tolist()]
            pid, pcl = [str(v) for v in z["ids"].tolist()], [str(v) for v in z["classes"].tolist()]
            pti = np.asarray(z["true_indices"]).astype(np.int64).tolist()
        n = len(rows[g])
        if pc != list(CASES) or pa != list(PRIOR_ARMS) or pcl != classes:
            raise FactorialError(f"prior scores {g}: cases/arms/classes differ")
        if pid != [r["id"] for r in rows[g]] or pti != [classes.index(r["label"]) for r in rows[g]]:
            raise FactorialError(f"prior scores {g}: ids or true labels differ from manifest")
        if lg.shape != (n, len(CASES), len(PRIOR_ARMS), R.N_CLASSES) or not np.isfinite(lg).all():
            raise FactorialError(f"prior scores {g}: logits shape {lg.shape} or non-finite")
        oi = PRIOR_ARMS.index("original")
        if not all(np.array_equal(lg[:, ci, oi], saved_a[g]) for ci in range(len(CASES))):
            raise FactorialError(f"prior scores {g}: original logits differ from saved scores")
        prior_sha = (pres.get("hashes") or {}).get("scores", {}).get(g)
        if prior_sha is not None and prior_sha != pin_of[norm(p)]:
            raise FactorialError(f"prior scores {g}: sha differs from prior RESULT hashes.scores")
        cases_out = pres["groups"][g]["cases"]
        pr_rows = {}
        for c in CASES:
            rs = cases_out[c]["rows"]
            if len(rs) != n or [r["id"] for r in rs] != [r["id"] for r in rows[g]] \
                    or [r["index"] for r in rs] != list(range(n)):
                raise FactorialError(f"prior result {g}/{c}: row ids/order differ from manifest")
            pr_rows[c] = rs
        prior_lg[g], prior_rows[g] = lg, pr_rows
    AI = {a: PRIOR_ARMS.index(a) for a in PRIOR_ARMS}

    # 5. local APIs, pinned
    state["stage"] = "imports"
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels, collate
    from pilot.web_input import to_native_like
    import framerate_impl as F
    module_files = {n: os.path.abspath(sys.modules[n].__file__) for n in API_MODULES}
    modules_before = {k: sha256_file(p) for k, p in module_files.items()}
    unpinned = {k: p for k, p in module_files.items() if pin_of.get(norm(p)) != modules_before[k]}
    if unpinned:
        raise FactorialError("API modules not pinned or differ from pins", {"modules": unpinned})
    _camera_check()
    api = {"F": F, "build_channels": build_channels, "to_native_like": to_native_like,
           "native_roots": roots["native"], "mp_roots": roots["mediapipe"]}

    def prepare_base(g, i):
        r = rows[g][i]
        base = I.prepare_base(R, api, g, r, tol)
        pb = prior_rows[g][CASES[0]][i]["base"]
        for k in ("source_raw_sha256", "original_input_sha256"):
            if base["meta"][k] != pb[k]:
                raise FactorialError(f"{r['id']}: base {k} differs from prior result")
        return base

    def prepare_cell(g, i, base, c):
        """Prior contract checks via I.prepare_case, then the linear arm rebuilt for its provenance."""
        sid, prior = rows[g][i]["id"], prior_rows[g][c][i]["case"]
        _, meta = I.prepare_case(R, TG, api, base, sid, c)
        xy, pr, ts = base["xy"], base["pr"], base["ts"]
        src_hash = I.raw_hash(R, xy, pr, ts)
        cr = C.corrupt(xy, pr, ts, sid, c)
        cxy, cpr, cts = cr["xy"], cr["present"], cr["ts"]
        c_hash = I.raw_hash(R, cxy, cpr, cts)
        lin = TG.restore(cxy, cpr, cts, "linear")                     # corrupted arrays only
        snap = {k: R.sha256_array(lin[k]) for k in ("xy", "present", "ts", "observed", "estimated",
                                                     "exact_source_index")}
        x = build_channels(lin["xy"], lin["present"], lin["ts"])["x"]
        support = M.support_mask(lin)
        hx, l_hash = R.sha256_array(x), I.raw_hash(R, lin["xy"], lin["present"], lin["ts"])
        checks = {
            "source_unmodified": I.raw_hash(R, xy, pr, ts) == src_hash == meta["raw_sha256"]["source"]
                                 == prior["raw_sha256"]["source"],
            "corrupted_unmodified": I.raw_hash(R, cxy, cpr, cts) == c_hash == meta["raw_sha256"]["corrupted"]
                                    == prior["raw_sha256"]["corrupted"],
            "restored_raw_matches_prior": l_hash == meta["raw_sha256"]["linear"] == prior["raw_sha256"]["linear"],
            "restored_unmodified": snap == {k: R.sha256_array(lin[k]) for k in snap},
            "input_matches_prior_linear": hx == meta["input_sha256"]["linear"] == prior["input_sha256"]["linear"],
            "input_shape": x.dtype == np.float32 and x.shape == (len(lin["ts"]), 280) and np.isfinite(x).all(),
            "support_shape": support.shape == (x.shape[0],) and support[0] == 1 and support[-1] == 1,
        }
        if not all(checks.values()):
            raise FactorialError(f"{sid}/{c}: cell invariants failed", {"checks": checks})
        excl = M.excluded_indices(support)
        src = np.asarray(lin["exact_source_index"])
        if excl and ((src[excl] >= 0).any() or np.asarray(lin["present"])[excl].astype(bool).any()):
            raise FactorialError(f"{sid}/{c}: an excluded frame is exact-source or present")
        n_ex = M.check_excluded_input(x, support)
        cell = {"frames": int(x.shape[0]), "support_frames": int(support.sum()), "excluded_n": n_ex,
                "excluded_indices": excl, "restored_identity": bool(lin["identity"]),
                "generated_frames": int((src < 0).sum()), "input_sha256": hx,
                "support_sha256": R.sha256_array(support), "raw_sha256": meta["raw_sha256"],
                "prior_case_meta_identical": jsonable(meta) == prior}
        return x, support, cell

    # 6. preflight ALL cells before model loading; any failure stops everything, nothing filtered
    state["stage"] = "preflight"
    cells, failures = {g: [] for g in GROUPS}, []
    for g in GROUPS:
        state["group"] = g
        for i, r in enumerate(rows[g]):
            state["id"], state["case"] = r["id"], None
            try:
                base = prepare_base(g, i)
            except Exception as exc:
                failures.append({"group": g, "index": i, "id": r["id"], "case": None,
                                 "error": f"{type(exc).__name__}: {exc}", "details": getattr(exc, "details", None)})
                cells[g].append(None)
                continue
            row_cells = {}
            for c in CASES:
                state["case"] = c
                try:
                    row_cells[c] = prepare_cell(g, i, base, c)[2]
                    state["prepared_cells"][g] += 1
                except Exception as exc:
                    failures.append({"group": g, "index": i, "id": r["id"], "case": c,
                                     "error": f"{type(exc).__name__}: {exc}", "details": getattr(exc, "details", None)})
                    row_cells[c] = None
            cells[g].append(row_cells)
    state.update(group=None, id=None, case=None, stage="preflight_gate")
    if failures:
        counts = {}
        for f in failures:
            k = f"{f['group']}/{f['case'] or 'base'}"
            counts[k] = counts.get(k, 0) + 1
        raise FactorialError("preflight failed before model load; no cells filtered",
                             {"failures_n": counts, "failures": failures})
    _camera_check()
    excl_cells = {g: {c: [i for i in range(len(rows[g])) if cells[g][i][c]["excluded_n"] > 0] for c in CASES}
                  for g in GROUPS}
    n_excl_cells = sum(len(v) for g in GROUPS for v in excl_cells[g].values())
    n_cells = sum(len(rows[g]) for g in GROUPS) * len(CASES)
    expected_new = len(NEW_ARMS) * n_excl_cells

    # 7. model (frozen, eval, CPU, 2 threads)
    state["stage"] = "model"
    torch.set_num_threads(R.FIXED_THREADS)
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    M.require_eval(model)
    if [str(c) for c in model_classes] != classes:
        raise FactorialError("model class order differs from saved class order")
    model_mods = sorted({type(m).__module__ for m in model.modules() if not type(m).__module__.startswith("torch.")})
    if not any(n.split(".")[-1] == MODEL_API_MODULE for n in model_mods):
        raise FactorialError(f"model classes not from {MODEL_API_MODULE}", {"modules": model_mods})
    for n in model_mods:
        module_files[n] = os.path.abspath(sys.modules[n].__file__)
        modules_before[n] = sha256_file(module_files[n])
        if pin_of.get(norm(module_files[n])) != modules_before[n]:
            raise FactorialError(f"model API module {n} not pinned or differs", {"file": module_files[n]})
    for attr in ("proj", "blocks", "head"):
        if not hasattr(model, attr):
            raise FactorialError(f"model lacks {attr}")
    _camera_check()
    state_before = M.state_digest(model)
    n_cls = R.N_CLASSES

    def tensors(x, support):
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])
        return torch.from_numpy(bx), torch.from_numpy(pad), torch.from_numpy(support[None].astype(np.float32))

    def _out(lg):
        lg = lg[0].numpy().astype(np.float64)
        if lg.shape != (n_cls,) or not np.isfinite(lg).all():
            raise FactorialError(f"model output shape {lg.shape} or non-finite")
        return lg

    def actual(x, support, use_support, kind):
        bx, ones, sp = tensors(x, support)
        with torch.inference_mode():
            lg = _out(model(bx, sp if use_support else ones))
        state["forwards"][kind] += 1
        return lg

    def manual(x, support, arm, kind):
        bx, _, sp = tensors(x, support)
        lg = _out(M.diagnostic_forward(model, bx, sp, arm))
        state["forwards"][kind] += 1
        return lg

    def top1(v):
        return classes[int(np.argmax(v))]

    def reprepare(g, i, c, base):
        x, sup, cell = prepare_cell(g, i, base, c)
        if cell != cells[g][i][c]:
            raise FactorialError(f"{rows[g][i]['id']}/{c}: cell differs from preflight")
        return x, sup

    # 8. reproduction on fixed rows x all cases, then identity probes; BEFORE any new arm
    state["stage"] = "reproduction"
    repro, identity = [], []

    def identity_probe(g, i, c, x, sup, kind_a, kind_m):
        ref = prior_lg[g][i, CASES.index(c), AI["linear"]]
        a = actual(x, sup, False, kind_a)
        arms = {arm: manual(x, sup, arm, kind_m) for arm in M.ARMS}
        bit = all(np.array_equal(arms[arm], arms["grid_baseline"]) for arm in M.ARMS)
        d = float(np.max(np.abs(a - ref)))
        return {"group": g, "index": i, "id": rows[g][i]["id"], "case": c, "all_arms_bit_identical": bit,
                "manual_equals_actual_bitwise": bool(np.array_equal(arms["grid_baseline"], a)),
                "actual_vs_prior_linear_max_abs_diff": d,
                "ok": bool(bit and d <= tol and top1(a) == top1(ref))}

    for g in GROUPS:
        for i in cfg["repro_indices"][g]:
            state["group"], state["id"] = g, rows[g][i]["id"]
            base = prepare_base(g, i)
            for c in CASES:
                state["case"] = c
                x, sup = reprepare(g, i, c, base)
                ref = prior_lg[g][i, CASES.index(c), AI["linear"]]
                a = actual(x, sup, False, "repro_actual")
                m = manual(x, sup, "grid_baseline", "repro_manual")
                da, dm = float(np.max(np.abs(a - ref))), float(np.max(np.abs(m - ref)))
                rec = {"group": g, "index": i, "id": rows[g][i]["id"], "case": c,
                       "excluded_n": cells[g][i][c]["excluded_n"], "actual_vs_prior_linear_max_abs_diff": da,
                       "manual_grid_vs_prior_linear_max_abs_diff": dm, "top1_prior": top1(ref),
                       "top1_actual": top1(a), "top1_manual_grid": top1(m)}
                ok = da <= tol and dm <= tol and top1(a) == top1(ref) == top1(m)
                if cells[g][i][c]["excluded_n"] == 0:
                    p = identity_probe(g, i, c, x, sup, "identity_actual", "identity_manual")
                    identity.append({**p, "source": "reproduction_row"})
                    rec["identity_ok"] = p["ok"]
                    ok = ok and p["ok"]
                else:
                    both = manual(x, sup, "blocks_and_pool", "repro_manual")
                    am = actual(x, sup, True, "repro_actual")
                    db = float(np.max(np.abs(both - am)))
                    rec.update({"manual_both_vs_actual_masked_max_abs_diff": db,
                                "manual_both_equals_actual_masked_bitwise": bool(np.array_equal(both, am))})
                    ok = ok and db <= tol and top1(both) == top1(am)
                rec["ok"] = bool(ok)
                repro.append(rec)
    if len(repro) != 6 * len(CASES) or not all(r["ok"] for r in repro):
        raise FactorialError("reproduction failed; no new arm computed", {"reproduction": repro})

    # identity probe per group/case: first all-one cell by row index (mask only, never score)
    state["stage"] = "identity_probes"
    for g in GROUPS:
        for c in CASES:
            if any(p["group"] == g and p["case"] == c for p in identity):
                continue
            allone = [i for i in range(len(rows[g])) if cells[g][i][c]["excluded_n"] == 0]
            if not allone:
                continue
            i = allone[0]
            state["group"], state["id"], state["case"] = g, rows[g][i]["id"], c
            x, sup = reprepare(g, i, c, prepare_base(g, i))
            identity.append({**identity_probe(g, i, c, x, sup, "identity_actual", "identity_manual"),
                             "source": "first_all_one_cell_by_index"})
    if not all(p["ok"] for p in identity):
        raise FactorialError("all-one identity not proven on checkpoint; no reuse", {"identity": identity})
    proven = {(p["group"], p["case"]) for p in identity}

    # 9. new arms only on cells with exclusions; all-one cells reuse prior linear logits
    state["stage"] = "inference"
    logits, results = {}, {}
    for g in GROUPS:
        n = len(rows[g])
        lg = np.full((n, len(CASES), len(M.ARMS), n_cls), np.nan, dtype=np.float64)
        per_case = {c: [] for c in CASES}
        for i, r in enumerate(rows[g]):
            state["group"], state["id"], state["case"] = g, r["id"], None
            base = prepare_base(g, i)
            for ci, c in enumerate(CASES):
                state["case"] = c
                x, sup = reprepare(g, i, c, base)
                cell = cells[g][i][c]
                grid = prior_lg[g][i, ci, AI["linear"]]
                lg[i, ci, 0] = grid
                prior_in = prior_rows[g][c][i]["case"]["input_sha256"]["linear"]
                if cell["excluded_n"] == 0:
                    if not (M.reuse_allowed(sup, cell["input_sha256"], prior_in) and (g, c) in proven):
                        raise FactorialError(f"{r['id']}/{c}: reuse not permitted")
                    lg[i, ci, 1:] = grid
                    state["reused_cells"] += 1
                    source = "prior_linear_logits_reused_all_one_support"
                else:
                    for ai, arm in enumerate(M.ARMS[1:], start=1):
                        lg[i, ci, ai] = manual(x, sup, arm, "new_diagnostic")
                    source = "grid_baseline_prior_linear; new arms diagnostic_forward"
                arms = {"prior_original": R.score_row(prior_lg[g][i, ci, AI["original"]], classes, r["label"]),
                        "prior_corrupted": R.score_row(prior_lg[g][i, ci, AI["corrupted"]], classes, r["label"]),
                        **{a: R.score_row(lg[i, ci, k], classes, r["label"]) for k, a in enumerate(M.ARMS)}}
                per_case[c].append({"index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
                                    "true_index": classes.index(r["label"]), "recording_id": R.recording_id(r["id"]),
                                    "logits_source": source, "arms": arms, "cell": cell})
        if not np.isfinite(lg).all():
            raise FactorialError(f"{g}: non-finite logits")
        logits[g], results[g] = lg, per_case
    state.update(group=None, id=None, case=None)
    fw = state["forwards"]
    if fw["new_diagnostic"] != expected_new or state["reused_cells"] != n_cells - n_excl_cells:
        raise FactorialError("forward/reuse count mismatch", {"forwards": fw, "expected_new": expected_new,
                                                              "reused": state["reused_cells"]})

    # 10. model state unchanged, still eval
    state["stage"] = "model_state_after"
    state_after = M.state_digest(model)
    M.require_eval(model)
    if state_after != state_before:
        raise FactorialError("model state_dict changed", {
            "changed": [k for k in state_before["tensors"] if state_before["tensors"][k] != state_after["tensors"].get(k)]})
    _camera_check()

    # 11. scores
    state["stage"] = "write_scores"
    scores_sha = {}
    for g in GROUPS:
        p = out_dir / f"SCORES_{g}.npz"
        with open(p, "xb") as fh:
            np.savez_compressed(fh, logits=logits[g], cases=np.array(CASES), arms=np.array(M.ARMS),
                                ids=np.array([r["id"] for r in rows[g]]), classes=np.array(classes),
                                true_indices=np.array([classes.index(r["label"]) for r in rows[g]], dtype=np.int64))
        scores_sha[g] = sha256_file(p)

    # 12. hashes after
    state["stage"] = "hashes_after"
    pins_after = {p: sha256_file(p) for p in cfg["pins"]}
    sources_after = {g: R.hash_sources(rows[g], roots[g]) for g in GROUPS}
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = code_hashes(cfg)
    config_sha, config_after = hashlib.sha256(cfg_bytes).hexdigest(), sha256_file(cfg_path)
    frozen_after = frozen()
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": {g: [k for k in sources_before[g] if sources_before[g][k] != sources_after[g][k]]
                           for g in GROUPS},
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code, "config": config_after != config_sha,
               "constants": frozen_after != frozen_before}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code, "after": code_after}, "model_state": {"before": state_before, "after": state_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after}, "changed": changed},
        strict=False)
    if changed["pins"] or any(changed["sources"].values()) or changed["modules"] or changed["code"] \
            or changed["config"] or changed["constants"]:
        raise FactorialError("hashes changed during run", {"changed": changed})

    # 13. result
    state["stage"] = "write_result"
    groups_out, effects = {}, {}
    for g in GROUPS:
        cases_out = {}
        for c in CASES:
            rs = results[g][c]
            ex = [r for r in rs if r["cell"]["excluded_n"] > 0]
            cases_out[c] = {
                "n": len(rs),
                "per_arm_correct": {a: sum(int(r["arms"][a]["correct"]) for r in rs) for a in REPORT_ARMS},
                "per_signer": M.breakdown(rs, "signer_id", REPORT_ARMS),
                "per_class": M.breakdown(rs, "true", REPORT_ARMS),
                "comparisons": {f"{s}->{t}": R.compare_arms(rs, s, t) for s, t in COMPARISONS},
                "prior_corrupted_correct_grid_wrong_recoveries": M.recoveries(rs),
                "primary_effect_vs_grid_baseline": M.primary_effect(rs),
                "support": {"cells_with_exclusions": len(ex), "cells_all_one_support": len(rs) - len(ex),
                            "excluded_frames_total": sum(r["cell"]["excluded_n"] for r in rs),
                            "frames_total": sum(r["cell"]["frames"] for r in rs),
                            "generated_frames_total": sum(r["cell"]["generated_frames"] for r in rs),
                            "max_excluded_in_cell": max((r["cell"]["excluded_n"] for r in rs), default=0),
                            "prior_case_meta_identical_cells": sum(r["cell"]["prior_case_meta_identical"] for r in rs)},
                "calls": {"new_diagnostic_forwards": len(NEW_ARMS) * len(ex), "reused_cells": len(rs) - len(ex)},
                "rows": rs}
        effects[g] = cases_out[PRIMARY_CASE]["primary_effect_vs_grid_baseline"]
        groups_out[g] = {"description": ("native provided-coordinate development rows (annotation_dev, validation)"
                                         if g == "native" else "cached MediaPipe training-exposed diagnostic rows"),
                         "n_rows": len(rows[g]), "source_recordings": R.recording_counts(results[g][CASES[0]]),
                         "cases": cases_out}
    contrib = M.contribution(effects)
    outcome = {
        "primary_arm": M.PRIMARY, "case": PRIMARY_CASE, **contrib,
        "per_group": {g: {k: effects[g][k] for k in ("net", "base_correct", "arm_correct", "newly_correct",
                                                     "newly_wrong", "remaining_errors", "per_signer")} for g in GROUPS},
        "statement": ("primary net correct > 0 vs grid_baseline on burst_drop in every group"
                      if contrib["diagnostic_contribution"] else
                      "no diagnostic contribution: primary neutral or harmful vs grid_baseline in at least one group; "
                      "reported unchanged"),
        "not_claimed": ["padding_mask bug proven", "unsupported frames are the exclusive cause",
                        "real-world accuracy", "deployment or promotion"],
        "unaffected": "all-one support cells keep their prior predictions; prior isolated_drop/clock_noise "
                      "failures remain regardless of this result",
        "deployment_candidate": False, "model_promoted": False}
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "unsupported-frame factorial mechanism diagnostic on a frozen model; no training, no search",
        "cases": list(CASES), "arm_order": list(M.ARMS), "report_arms": list(REPORT_ARMS), "primary": M.PRIMARY,
        "arm_masks": {a: {"block_mask": "support" if a in M.BLOCK_SUPPORT else "ones",
                          "pool_mask": "support" if a in M.POOL_SUPPORT else "ones"} for a in M.ARMS},
        "support_rule": "1 unless exact_source_index < 0 and no joint present (observed or estimated)",
        "model": {"checkpoint": cfg["checkpoint"], "sha256": pin_of[norm(cfg["checkpoint"])], "classes": len(classes),
                  "api_modules": model_mods, "state_dict_combined_sha256": state_before["combined"],
                  "state_dict_unchanged": True},
        "saved_mp_arms": mp_arms, "reproduction": {"atol": tol, "indices": cfg["repro_indices"], "rows": repro},
        "identity_proof": identity, "outcome": outcome, "groups": groups_out, "interpretation_limits": LIMITS,
        "execution_counts": {"cells_total": n_cells, "cells_with_exclusions": n_excl_cells,
                             "cells_reused": state["reused_cells"], "expected_new_diagnostic": expected_new,
                             "forwards": fw, "forwards_total": sum(fw.values()),
                             "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "camera_modules_in_sys_modules": [m for m in CAMERA_MODULES if m in sys.modules],
        "hashes": {"code": code, "config": config_sha, "plan_file": pin_of[norm(cfg["plan_file"])],
                   "prior_result": pin_of[norm(cfg["prior_result"])],
                   "prior_scores": {g: pin_of[norm(cfg["prior_scores"][g])] for g in GROUPS},
                   "scores": scores_sha, "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", result)
    print(json.dumps({"diagnostic_contribution": contrib["diagnostic_contribution"], "net": contrib["net_by_group"],
                      "forwards": fw, "reused_cells": state["reused_cells"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:                 # FAILURE.json already written when output_dir existed
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
