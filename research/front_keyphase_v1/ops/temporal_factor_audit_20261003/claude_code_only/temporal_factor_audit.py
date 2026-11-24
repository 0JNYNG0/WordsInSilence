"""Temporal factor audit: decompose the completed stride-2 frame-decimation diagnostic.

Diagnostic only. No training, no repair proposal, no camera, no user data, no network.

Arms (all over the same development rows, same frozen model):
  A original             stored logits (not recomputed except repro_indices)
  B raw_decimated        stored logits: raw xy/present/ts[::2] rebuilt with build_channels
                         (scales re-estimated on the sparse clip)
  C fixed_scale_real_dt  NEW: build_channels(original)["x"][::2], dt column replaced with the
                         real decimated intervals using build_channels' exact float32 operations
  D fixed_scale_old_dt   NEW: build_channels(original)["x"][::2] with its OLD per-frame dt.
                         A deliberately inconsistent-clock diagnostic, NEVER a deployable path.

C vs B: normalization re-estimation (incl. derived validity differences, reported per row).
C vs D: dt channel effect conditional on the same sparse sequence/scales/masks.
D vs A: retained-frame sequence with old dt; still loses motion observations and changes
        convolution/pooling support.
The arms are not additive: no causal percentages, no uniquely identified architecture defect.

Run: python temporal_factor_audit.py --config <config.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# --- channel layout (data_api contract: 280 channels) -------------------------
N_CH = 280
COORDS = slice(0, 186)              # body xy 102 + hand xy 84
MASKS = slice(186, 279)             # body_valid 51 + hand_valid 42
BODY_VALID = slice(186, 237)
LEFT_HAND_VALID = slice(237, 258)
RIGHT_HAND_VALID = slice(258, 279)
DT_COL = 279
STRIDE, OFFSET = 2, 0

ARMS = ("original", "raw_decimated", "fixed_scale_real_dt", "fixed_scale_old_dt")
ARM_DESCRIPTIONS = {
    "original": "stored A: build_channels(original native arrays)",
    "raw_decimated": "stored B: build_channels(xy[::2], present[::2], ts[::2]); scales re-estimated",
    "fixed_scale_real_dt": "C: build_channels(original)['x'][::2].copy(), dt column = "
                           "[0, diff(ts[::2]).astype(float32)/1e6]",
    "fixed_scale_old_dt": "D: build_channels(original)['x'][::2].copy() incl. old dt; "
                          "inconsistent-clock diagnostic, NEVER deployable",
}
COMPARISONS = (("raw_decimated", "fixed_scale_real_dt"),
               ("fixed_scale_real_dt", "fixed_scale_old_dt"),
               ("original", "fixed_scale_old_dt"))
CONTEXT_COMPARISONS = (("original", "raw_decimated"), ("original", "fixed_scale_real_dt"))
INTERPRETATION_LIMITS = [
    "B->C isolates per-clip normalization re-estimation (body/hand scales and any hand-validity "
    "differences derived from scale finiteness) on the same retained frames with the same real dt.",
    "C->D isolates the dt channel value conditional on identical retained frames, coordinates, "
    "scales and masks.",
    "A->D: retained-frame sequence with the old per-frame dt; D still loses motion observations and "
    "changes convolution/pooling support. D is an inconsistent clock and never a deployable path.",
    "Arms are not additive: no linear/additive causal percentages are claimed, and no uniquely "
    "identified architecture defect is claimed.",
    "Development rows only; no training and no production repair are proposed.",
]
REQUIRED_PIN_KEYS = ("metadata", "checkpoint", "saved_logits", "saved_result", "plan_file")
PATH_KEYS = ("lab_root", "pilot_src", "loader_dir") + REQUIRED_PIN_KEYS + ("output_dir",)


class AuditError(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


# --- small helpers -------------------------------------------------------------
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def norm_path(p) -> str:
    return os.path.normcase(os.path.abspath(str(p)))


def under(path, root) -> bool:
    try:
        return os.path.commonpath([str(path), str(root)]) == str(root)
    except ValueError:
        return False


def num_or_null(v):
    """Finite float, else None (NaN/inf scales -> JSON null)."""
    if v is None:
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def write_json_exclusive(path, value):
    with open(path, "x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")


# --- config --------------------------------------------------------------------
def validate_config(cfg: dict) -> dict:
    missing = [k for k in PATH_KEYS + ("pins", "expected", "repro_indices", "repro_atol", "threads")
               if k not in cfg]
    if missing:
        raise AuditError(f"config lacks keys {missing}")
    for k in PATH_KEYS:
        if not isinstance(cfg[k], str) or not os.path.isabs(cfg[k]):
            raise AuditError(f"config path {k} must be an absolute string")
    pins = cfg["pins"]
    if not isinstance(pins, dict) or not pins:
        raise AuditError("config pins must be a non-empty object")
    for p, d in pins.items():
        if not os.path.isabs(p):
            raise AuditError(f"pin path not absolute: {p}")
        if not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d):
            raise AuditError(f"pin digest invalid for {p}")
    pinned = {norm_path(p) for p in pins}
    for k in REQUIRED_PIN_KEYS:
        if norm_path(cfg[k]) not in pinned:
            raise AuditError(f"required pin missing for {k}")
    e = cfg["expected"]
    for k in ("rows", "classes", "roles", "signers", "allowed_roots"):
        if k not in e:
            raise AuditError(f"config expected lacks {k}")
    if type(e["rows"]) is not int or e["rows"] < 1 or type(e["classes"]) is not int or e["classes"] < 2:
        raise AuditError("expected rows/classes must be positive ints")
    if not e["roles"] or not e["signers"] or not e["allowed_roots"]:
        raise AuditError("expected roles/signers/allowed_roots must be non-empty")
    for r in e["allowed_roots"]:
        if not isinstance(r, str) or not os.path.isabs(r):
            raise AuditError(f"allowed root not absolute: {r!r}")
    ri = cfg["repro_indices"]
    if not ri or any(type(i) is not int or not 0 <= i < e["rows"] for i in ri) or len(set(ri)) != len(ri):
        raise AuditError(f"repro_indices invalid: {ri!r}")
    atol = cfg["repro_atol"]
    if not isinstance(atol, (int, float)) or not math.isfinite(atol) or atol < 0:
        raise AuditError(f"repro_atol invalid: {atol!r}")
    if type(cfg["threads"]) is not int or cfg["threads"] < 1:
        raise AuditError(f"threads invalid: {cfg['threads']!r}")
    out = norm_path(cfg["output_dir"])
    if any(under(out, norm_path(r)) for r in e["allowed_roots"]):
        raise AuditError("output_dir lies inside an allowed source root")
    return cfg


def refuse_existing_output(output_dir):
    if os.path.lexists(output_dir):
        raise AuditError(f"output_dir {output_dir} exists; outputs are never overwritten")


# --- input construction (numpy only) ---------------------------------------------
def check_input(x, frames=None, name="x"):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != N_CH or x.shape[0] < 1:
        raise AuditError(f"{name}: expected [T>=1,{N_CH}] array, got {getattr(x, 'shape', None)}")
    if x.dtype != np.float32:
        raise AuditError(f"{name}: dtype {x.dtype} != float32")
    if frames is not None and x.shape[0] != frames:
        raise AuditError(f"{name}: {x.shape[0]} frames != expected {frames}")
    if not np.isfinite(x).all():
        raise AuditError(f"{name}: non-finite values")
    return x


def dt_column(ts) -> np.ndarray:
    """[T,1] float32 dt, the exact operations of build_channels."""
    ts = np.asarray(ts)
    dt = np.zeros((len(ts), 1), dtype=np.float32)
    if len(ts) > 1:
        dt[1:, 0] = np.diff(ts).astype(np.float32) / 1e6
    return dt


def make_c_d(x_orig, ts_orig):
    """-> (C, D, ts_dec). x_orig is never modified; C and D are independent copies."""
    ts_orig = np.asarray(ts_orig)
    if ts_orig.ndim != 1:
        raise AuditError("timestamps must be 1-D")
    check_input(x_orig, len(ts_orig), "original x")
    ts_dec = ts_orig[OFFSET::STRIDE].copy()
    d = x_orig[OFFSET::STRIDE].copy()
    c = d.copy()
    c[:, DT_COL] = dt_column(ts_dec)[:, 0]
    return c, d, ts_dec


def check_cd(c, d, x_orig, x_b=None):
    """C/D invariants; B (if given) must share frames and the exact real dt with C."""
    check_input(c, None, "C")
    check_input(d, len(c), "D")
    if not np.array_equal(c[:, :DT_COL], d[:, :DT_COL]):
        raise AuditError("C and D differ outside the dt channel")
    if not np.array_equal(d, x_orig[OFFSET::STRIDE]):
        raise AuditError("D is not original x[::2]")
    if x_b is not None:
        check_input(x_b, len(c), "B")
        if not np.array_equal(c[:, DT_COL], x_b[:, DT_COL]):
            raise AuditError("C dt differs from build_channels dt on the decimated timestamps")


def mask_diff(x_b, x_c) -> dict:
    """B vs C input differences; validity columns 186:279 broken down by block."""
    if x_b.shape != x_c.shape:
        raise AuditError(f"B/C shape mismatch {x_b.shape} vs {x_c.shape}")
    out = {}
    for name, sl in (("all_valid_186_279", MASKS), ("body_valid", BODY_VALID),
                     ("left_hand_valid", LEFT_HAND_VALID), ("right_hand_valid", RIGHT_HAND_VALID)):
        mb, mc = x_b[:, sl], x_c[:, sl]
        ne = mb != mc
        out[name] = {"entries_differ": int(ne.sum()), "frames_differ": int(ne.any(axis=1).sum()),
                     "b1_c0": int(((mb == 1) & (mc == 0)).sum()),
                     "b0_c1": int(((mb == 0) & (mc == 1)).sum())}
    diff = np.abs(x_b.astype(np.float64) - x_c.astype(np.float64))
    out["max_abs_diff_all"] = float(diff.max())
    out["max_abs_diff_coords"] = float(diff[:, COORDS].max())
    out["max_abs_diff_dt"] = float(diff[:, DT_COL].max())
    return out


def scales_of(built) -> dict:
    hs = built["hand_scales"]
    return {"body": num_or_null(built["body_scale"]), "left_hand": num_or_null(hs.get("left")),
            "right_hand": num_or_null(hs.get("right"))}


# --- scoring -------------------------------------------------------------------
def score_row(logits, classes, true_label) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(classes),) or not np.isfinite(logits).all():
        raise AuditError(f"logits shape {logits.shape} or non-finite")
    yi = classes.index(true_label)
    order = np.argsort(-logits, kind="stable")
    rival = next(int(j) for j in order if int(j) != yi)
    return {"pred": classes[int(order[0])], "correct": bool(int(order[0]) == yi),
            "rank": int(np.where(order == yi)[0][0]) + 1,
            "margin": float(logits[yi] - logits[rival])}


def compare_arms(rows, src, dst) -> dict:
    """Paired transition table src -> dst; arithmetic is self-checked."""
    gains, losses, w2w_changed = [], [], []
    c2c = w2w_same = 0
    for r in rows:
        a, b = r["arms"][src], r["arms"][dst]
        item = {"id": r["id"], "true": r["true"], "src_pred": a["pred"], "dst_pred": b["pred"]}
        if a["correct"] and b["correct"]:
            c2c += 1
        elif b["correct"]:
            gains.append(item)
        elif a["correct"]:
            losses.append(item)
        elif a["pred"] != b["pred"]:
            w2w_changed.append(item)
        else:
            w2w_same += 1
    n = len(rows)
    src_ok = sum(r["arms"][src]["correct"] for r in rows)
    dst_ok = sum(r["arms"][dst]["correct"] for r in rows)
    changed = sum(r["arms"][src]["pred"] != r["arms"][dst]["pred"] for r in rows)
    if dst_ok - src_ok != len(gains) - len(losses):
        raise AuditError(f"{src}->{dst}: net arithmetic mismatch")
    if changed != len(gains) + len(losses) + len(w2w_changed):
        raise AuditError(f"{src}->{dst}: top1-change arithmetic mismatch")
    if c2c + len(gains) + len(losses) + len(w2w_changed) + w2w_same != n:
        raise AuditError(f"{src}->{dst}: partition arithmetic mismatch")
    return {"src": src, "dst": dst, "n": n, "src_correct": src_ok, "dst_correct": dst_ok,
            "net": dst_ok - src_ok, "top1_changed": changed, "correct_to_correct": c2c,
            "gains_n": len(gains), "losses_n": len(losses),
            "wrong_to_wrong_changed_n": len(w2w_changed), "wrong_to_wrong_same": w2w_same,
            "gains": gains, "losses": losses, "wrong_to_wrong_changed": w2w_changed}


def a_correct_b_wrong(rows) -> dict:
    sub = [r for r in rows if r["arms"]["original"]["correct"] and not r["arms"]["raw_decimated"]["correct"]]
    items = [{"id": r["id"], "true": r["true"],
              "raw_decimated_pred": r["arms"]["raw_decimated"]["pred"],
              "fixed_scale_real_dt": {k: r["arms"]["fixed_scale_real_dt"][k] for k in ("pred", "correct", "rank")},
              "fixed_scale_old_dt": {k: r["arms"]["fixed_scale_old_dt"][k] for k in ("pred", "correct", "rank")}}
             for r in sub]
    c = [i["fixed_scale_real_dt"]["correct"] for i in items]
    d = [i["fixed_scale_old_dt"]["correct"] for i in items]
    return {"n": len(items),
            "correct_in_c": sum(c), "correct_in_d": sum(d),
            "correct_in_both": sum(x and y for x, y in zip(c, d)),
            "correct_in_c_only": sum(x and not y for x, y in zip(c, d)),
            "correct_in_d_only": sum(y and not x for x, y in zip(c, d)),
            "correct_in_neither": sum(not x and not y for x, y in zip(c, d)),
            "rows": items}


def per_signer(rows) -> dict:
    out = {}
    for r in rows:
        g = out.setdefault(str(r["signer_id"]), {"n": 0, **{a: 0 for a in ARMS}})
        g["n"] += 1
        for a in ARMS:
            g[a] += r["arms"][a]["correct"]
    return dict(sorted(out.items()))


# --- stored material validation --------------------------------------------------
def validate_saved(saved_ids, saved_classes, orig, dec, meta_rows, result_rows, n_rows, n_classes):
    """Exact id order, labels, signers, shapes, finiteness, class order. -> classes list."""
    classes = [str(c) for c in saved_classes]
    if len(classes) != n_classes or len(set(classes)) != n_classes:
        raise AuditError(f"saved classes: {len(classes)} entries, expected {n_classes} unique")
    meta_ids = [r["id"] for r in meta_rows]
    if len(meta_ids) != n_rows or len(set(meta_ids)) != n_rows:
        raise AuditError(f"metadata rows {len(meta_ids)} (unique {len(set(meta_ids))}) != {n_rows}")
    if [str(i) for i in saved_ids] != meta_ids:
        raise AuditError("saved logits id order differs from metadata order")
    for name, arr in (("original_logits", orig), ("decimated_logits", dec)):
        arr = np.asarray(arr)
        if arr.shape != (n_rows, n_classes):
            raise AuditError(f"{name} shape {arr.shape} != {(n_rows, n_classes)}")
        if not np.issubdtype(arr.dtype, np.floating) or not np.isfinite(arr).all():
            raise AuditError(f"{name} has non-finite or non-float values")
    if [r.get("id") for r in result_rows] != meta_ids:
        raise AuditError("saved result row id order differs from metadata order")
    for m, s in zip(meta_rows, result_rows):
        if m["label"] not in classes:
            raise AuditError(f"{m['id']}: label {m['label']!r} not in classes")
        if s.get("true") != m["label"]:
            raise AuditError(f"{m['id']}: saved true {s.get('true')!r} != metadata label {m['label']!r}")
        if s.get("signer_id") != m["signer_id"]:
            raise AuditError(f"{m['id']}: saved signer {s.get('signer_id')!r} != metadata {m['signer_id']!r}")
    for i, s in enumerate(result_rows):          # optional stored preds must agree with stored logits
        for key, arr in (("original", orig), ("decimated", dec)):
            pred = s.get(key, {}).get("pred") if isinstance(s.get(key), dict) else None
            if pred is not None and pred != classes[int(np.argsort(-np.asarray(arr[i]), kind="stable")[0])]:
                raise AuditError(f"{s['id']}: stored {key} pred disagrees with stored logits")
    return classes


def check_rows_allowed(meta_rows, roles, signers):
    for r in meta_rows:
        for k in ("id", "label", "path", "sha256", "role", "signer_id"):
            if k not in r:
                raise AuditError(f"metadata row lacks {k}: {r.get('id')!r}")
        if r["role"] not in roles:
            raise AuditError(f"{r['id']}: role {r['role']!r} outside config")
        if r["signer_id"] not in signers:
            raise AuditError(f"{r['id']}: signer {r['signer_id']!r} outside config")


def hash_sources(meta_rows, roots) -> dict:
    roots = [norm_path(Path(r).resolve()) for r in roots]
    out = {}
    for r in meta_rows:
        p = norm_path(Path(r["path"]).resolve())
        if not any(under(p, root) for root in roots):
            raise AuditError(f"{r['id']}: source outside allowed roots")
        out[r["id"]] = {"path": str(r["path"]), "sha256": sha256_file(p)}
    return out


def hash_pins(pins) -> dict:
    return {p: sha256_file(p) for p in pins}


# --- main ------------------------------------------------------------------------
def main(argv=None):
    started = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg_path = os.path.abspath(args.config)
    cfg_bytes = Path(cfg_path).read_bytes()
    cfg = validate_config(json.loads(cfg_bytes.decode("utf-8-sig")))
    out_dir = Path(cfg["output_dir"])
    refuse_existing_output(out_dir)                          # before any other read
    out_dir.mkdir(parents=True, exist_ok=False)

    code_sha = sha256_file(__file__)
    state = {"stage": "start", "row": None, "forwards_repro": 0, "forwards_new": 0, "details": None}
    try:
        return _run(cfg, cfg_path, cfg_bytes, out_dir, code_sha, state, started)
    except BaseException as exc:
        failure = {"status": "FAILED", "stage": state["stage"], "row": state["row"],
                   "error_class": type(exc).__name__, "error": str(exc),
                   "details": getattr(exc, "details", None) or state["details"],
                   "execution_counts": {"forwards_repro": state["forwards_repro"],
                                        "forwards_new": state["forwards_new"],
                                        "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
                   "code_sha256": code_sha, "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
                   "at_utc": datetime.now(timezone.utc).isoformat()}
        try:
            write_json_exclusive(out_dir / "FAILURE.json", failure)
        except FileExistsError:
            pass
        raise


def _run(cfg, cfg_path, cfg_bytes, out_dir, code_sha, state, started):
    e = cfg["expected"]
    n_rows, n_classes = e["rows"], e["classes"]
    roles, signers, roots = list(e["roles"]), list(e["signers"]), list(e["allowed_roots"])

    # 1. every pin, before any import or model load
    state["stage"] = "pins"
    pins_before = hash_pins(cfg["pins"])
    bad = {p: got for p, got in pins_before.items() if got != cfg["pins"][p]}
    if bad:
        raise AuditError("pin mismatch", {"mismatched": bad})
    pin_of = {norm_path(p): d for p, d in cfg["pins"].items()}

    # 2. metadata rows, role/signer gate, source hashes before loading
    state["stage"] = "metadata"
    meta_rows = json.loads(Path(cfg["metadata"]).read_text(encoding="utf-8-sig"))["annotation_dev"]
    if len(meta_rows) != n_rows:
        raise AuditError(f"metadata has {len(meta_rows)} rows, expected {n_rows}")
    check_rows_allowed(meta_rows, roles, signers)
    state["stage"] = "source_hashes_before"
    sources_before = hash_sources(meta_rows, roots)
    bad = [r["id"] for r in meta_rows if sources_before[r["id"]]["sha256"] != r["sha256"]]
    if bad:
        raise AuditError("source sha mismatch vs metadata", {"ids": bad})

    # 3. stored A/B material
    state["stage"] = "saved_material"
    with np.load(cfg["saved_logits"], allow_pickle=False) as z:
        need = ("ids", "classes", "original_logits", "decimated_logits")
        if any(k not in z.files for k in need):
            raise AuditError(f"saved logits lacks keys; has {z.files}")
        saved_ids = z["ids"].tolist()
        saved_classes = z["classes"].tolist()
        a_logits = np.asarray(z["original_logits"], dtype=np.float64)
        b_logits = np.asarray(z["decimated_logits"], dtype=np.float64)
    saved_result = json.loads(Path(cfg["saved_result"]).read_text(encoding="utf-8-sig"))
    classes = validate_saved(saved_ids, saved_classes, a_logits, b_logits, meta_rows,
                             saved_result["rows"], n_rows, n_classes)
    ck_sha = pin_of[norm_path(cfg["checkpoint"])]
    if saved_result.get("model", {}).get("sha256") != ck_sha:
        raise AuditError("saved result model.sha256 differs from pinned checkpoint")

    # 4. local APIs and model
    state["stage"] = "imports"
    sys.path[:0] = [cfg["lab_root"], cfg["pilot_src"], cfg["loader_dir"]]
    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import build_channels, collate
    import framerate_impl as F
    module_files = {name: os.path.abspath(m.__file__) for name, m in
                    (("tools_prep_union93_candidate", u), ("pilot.data", sys.modules["pilot.data"]),
                     ("framerate_impl", F))}
    modules_before = {k: sha256_file(p) for k, p in module_files.items()}

    state["stage"] = "model"
    torch.set_num_threads(cfg["threads"])
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, model_classes = u.rebuild_from_payload(payload)
    model.eval()
    if [str(c) for c in model_classes] != classes:
        raise AuditError("model class order differs from saved class order")

    def forward(x):
        check_input(x, None, "forward input")
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])
        with torch.inference_mode():
            lg = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)
        if lg.shape != (n_classes,) or not np.isfinite(lg).all():
            raise AuditError(f"model output shape {lg.shape} or non-finite")
        return lg

    def load(r):
        nat = F.load_native(r, roles, signers, roots)
        if nat["sha256"] != r["sha256"]:
            raise AuditError(f"{r['id']}: loader sha differs from metadata")
        return nat

    def build_ab(nat):
        built_a = build_channels(nat["xy"], nat["present"], nat["ts"])
        xy2, pr2, ts2 = F.decimate(nat["xy"], nat["present"], nat["ts"], stride=STRIDE, offset=OFFSET)
        if not F.kept_intervals_are_sums(nat["ts"], ts2, STRIDE, OFFSET):
            raise AuditError("decimated timestamps are not ts[::2]")
        built_b = build_channels(xy2, pr2, ts2)
        return built_a, built_b, ts2

    # 5. reproduce stored A and B on repro_indices; stop on any failure
    state["stage"] = "reproduction"
    atol = float(cfg["repro_atol"])
    repro = []
    for i in cfg["repro_indices"]:
        r = meta_rows[i]
        state["row"] = r["id"]
        built_a, built_b, _ = build_ab(load(r))
        for arm, x, saved in (("original", built_a["x"], a_logits[i]), ("raw_decimated", built_b["x"], b_logits[i])):
            lg = forward(x)
            state["forwards_repro"] += 1
            diff = float(np.max(np.abs(lg - saved)))
            top_now, top_saved = int(np.argmax(lg)), int(np.argmax(saved))
            repro.append({"index": i, "id": r["id"], "arm": arm, "max_abs_diff": diff,
                          "top1_now": classes[top_now], "top1_saved": classes[top_saved],
                          "ok": bool(diff <= atol and top_now == top_saved)})
    reproduction = {"indices": cfg["repro_indices"], "atol": atol, "rtol": 0, "rows": repro}
    if not all(x["ok"] for x in repro):
        state["details"] = {"reproduction": reproduction}
        raise AuditError("stored A/B reproduction failed; no new-arm inference performed",
                         {"reproduction": reproduction})

    # 6. inputs for all rows (no inference): original, B, C, D
    state["stage"] = "build_inputs"
    state["row"] = None
    inputs, rows = [], []
    for i, r in enumerate(meta_rows):
        state["row"] = r["id"]
        nat = load(r)
        built_a, built_b, ts2 = build_ab(nat)
        xa, xb = built_a["x"], built_b["x"]
        check_input(xa, len(nat["ts"]), "A")
        xa_copy = xa.copy()
        c, d, ts_dec = make_c_d(xa, nat["ts"])
        if not np.array_equal(ts_dec, ts2):
            raise AuditError("C/D timestamps differ from loader decimation")
        check_cd(c, d, xa, xb)
        if not np.array_equal(xa, xa_copy):
            raise AuditError("original input modified during C/D construction")
        sa, sb = scales_of(built_a), scales_of(built_b)
        inputs.append((c, d))
        rows.append({
            "index": i, "id": r["id"], "signer_id": r["signer_id"], "true": r["label"],
            "true_index": classes.index(r["label"]), "source_sha256": nat["sha256"], "status": "OK",
            "frames_original": int(len(nat["ts"])), "frames_decimated": int(len(ts2)),
            "scales": {"original": sa, "decimated": sb},
            "hand_scale_finiteness_changed": {h: (sa[h] is None) != (sb[h] is None)
                                              for h in ("left_hand", "right_hand")},
            "b_vs_c_input": mask_diff(xb, c),
            "c_vs_d_input": {"first279_identical": True,
                             "max_abs_diff_dt": float(np.max(np.abs(c[:, DT_COL].astype(np.float64)
                                                                    - d[:, DT_COL])))},
        })

    # 7. new-arm inference: exactly 2 forwards per row
    state["stage"] = "inference"
    logits = np.full((n_rows, len(ARMS), n_classes), np.nan, dtype=np.float64)
    logits[:, 0], logits[:, 1] = a_logits, b_logits
    for i, (row, (c, d)) in enumerate(zip(rows, inputs)):
        state["row"] = row["id"]
        logits[i, 2] = forward(c)
        logits[i, 3] = forward(d)
        state["forwards_new"] += 2
        row["arms"] = {arm: score_row(logits[i, k], classes, row["true"]) for k, arm in enumerate(ARMS)}
    state["row"] = None
    if not np.isfinite(logits).all():
        raise AuditError("non-finite logits in score tensor")
    if state["forwards_new"] != 2 * n_rows or state["forwards_repro"] != 2 * len(cfg["repro_indices"]):
        raise AuditError("forward count mismatch")

    # 8. scores, then post-run hash verification, then result
    state["stage"] = "write_scores"
    scores_path = out_dir / "SCORES.npz"
    with open(scores_path, "xb") as fh:
        np.savez_compressed(fh, logits=logits, arms=np.array(ARMS), ids=np.array([r["id"] for r in rows]),
                            classes=np.array(classes), true_indices=np.array([r["true_index"] for r in rows],
                                                                              dtype=np.int64))

    state["stage"] = "hashes_after"
    pins_after = hash_pins(cfg["pins"])
    sources_after = hash_sources(meta_rows, roots)
    modules_after = {k: sha256_file(p) for k, p in module_files.items()}
    code_after = sha256_file(__file__)
    config_after = sha256_file(cfg_path)
    config_sha = hashlib.sha256(cfg_bytes).hexdigest()
    changed = {"pins": [p for p in pins_before if pins_before[p] != pins_after[p]],
               "sources": [k for k in sources_before if sources_before[k] != sources_after[k]],
               "modules": [k for k in modules_before if modules_before[k] != modules_after[k]],
               "code": code_after != code_sha, "config": config_after != config_sha}
    write_json_exclusive(out_dir / "SOURCE_HASHES.json", {
        "pins": {"before": pins_before, "after": pins_after},
        "sources": {"before": sources_before, "after": sources_after},
        "imported_modules": {"files": module_files, "before": modules_before, "after": modules_after},
        "code": {"before": code_sha, "after": code_after},
        "config": {"path": cfg_path, "before": config_sha, "after": config_after},
        "changed": changed})
    if any(changed[k] for k in changed):
        raise AuditError("hashes changed during run", {"changed": changed})

    state["stage"] = "write_result"
    totals = {a: sum(r["arms"][a]["correct"] for r in rows) for a in ARMS}
    result = {
        "status": "COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "nature": "diagnostic decomposition of a completed stride-2 decimation run; no training, "
                  "no production repair",
        "arms": ARM_DESCRIPTIONS, "arm_order": list(ARMS),
        "transform": {"stride": STRIDE, "offset": OFFSET},
        "model": {"checkpoint": cfg["checkpoint"], "sha256": ck_sha, "classes": len(classes)},
        "n_rows": n_rows, "row_status": {"OK": sum(r["status"] == "OK" for r in rows)},
        "reproduction": reproduction,
        "per_arm_correct": totals,
        "per_signer": per_signer(rows),
        "comparisons": {f"{s}->{t}": compare_arms(rows, s, t) for s, t in COMPARISONS},
        "context_comparisons": {f"{s}->{t}": compare_arms(rows, s, t) for s, t in CONTEXT_COMPARISONS},
        "original_correct_raw_decimated_wrong": a_correct_b_wrong(rows),
        "input_diagnostics_summary": {
            "rows_with_b_c_valid_diff": sum(r["b_vs_c_input"]["all_valid_186_279"]["entries_differ"] > 0
                                            for r in rows),
            "b_c_valid_entries_differ": sum(r["b_vs_c_input"]["all_valid_186_279"]["entries_differ"]
                                            for r in rows),
            "rows_hand_scale_finiteness_changed": {
                h: sum(r["hand_scale_finiteness_changed"][h] for r in rows) for h in ("left_hand", "right_hand")},
            "max_abs_diff_b_c_all": max(r["b_vs_c_input"]["max_abs_diff_all"] for r in rows),
            "max_abs_diff_b_c_coords": max(r["b_vs_c_input"]["max_abs_diff_coords"] for r in rows),
            "max_abs_diff_b_c_dt": max(r["b_vs_c_input"]["max_abs_diff_dt"] for r in rows),
            "c_d_first279_identical_all_rows": all(r["c_vs_d_input"]["first279_identical"] for r in rows),
        },
        "interpretation_limits": INTERPRETATION_LIMITS,
        "rows": rows,
        "execution_counts": {"rows": n_rows, "forwards_repro": state["forwards_repro"],
                             "forwards_new": state["forwards_new"],
                             "forwards_total": state["forwards_repro"] + state["forwards_new"],
                             "stored_a_b_recomputed_for_all_rows": 0,
                             "training": 0, "camera": 0, "user_data": 0, "downloads": 0},
        "hashes": {"code": code_sha, "config": config_sha,
                   "plan_file": pin_of[norm_path(cfg["plan_file"])],
                   "scores_npz": sha256_file(scores_path), "imported_modules": modules_before},
        "seconds": round(time.time() - started, 2),
    }
    write_json_exclusive(out_dir / "RESULT.json", result)
    print(json.dumps({"per_arm_correct": totals, "n": n_rows}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AuditError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
