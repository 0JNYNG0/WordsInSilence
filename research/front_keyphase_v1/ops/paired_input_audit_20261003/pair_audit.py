"""Read-only paired coordinate-path audit.

Diagnostic only: existing source coordinates are NOT gold truth, and the data are
training-exposed, so nothing here is an independent accuracy benchmark. No tuning,
crops, imputation, coordinate correction, or score-based selection.
Usage: python pair_audit.py --config CONFIG_LOCAL_ONLY.json
"""
import argparse
import hashlib
import json
import os
import sys
import traceback

import numpy as np

ARMS = ("native", "time_control", "availability_control", "mediapipe")
PAIRS = (("native", "time_control"), ("time_control", "availability_control"),
         ("time_control", "mediapipe"), ("availability_control", "mediapipe"))
ROW_KEYS = ("sample_id", "clip_id", "label", "signer_id", "native_source", "input_path",
            "input_sha256", "observation_path", "observation_sha256")
# Assumed index layout (see HANDOFF.md): common51 = 9 body, 21 left hand, 21 right hand;
# hand_valid[T,42] = 21 left then 21 right (same order as hand_scales).
COMMON_GROUPS = (("body", slice(0, 9)), ("left_hand", slice(9, 30)), ("right_hand", slice(30, 51)))
HAND_GROUPS = (("left", slice(0, 21)), ("right", slice(21, 42)))


class AuditError(Exception):
    pass


def fail(msg):
    raise AuditError(msg)


def norm(p):
    return os.path.normcase(os.path.abspath(p))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_files(paths):
    return {p: sha256_file(p) for p in sorted({norm(p) for p in paths})}


def check_immutable(before, after):
    """Return paths whose hash changed, appeared or disappeared."""
    return sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))


def as_bool(a, name="mask"):
    """Exact binary mask -> bool. Accepts bool, or integer/float arrays (e.g. float32 masks from
    build_channels / prepared NPZ) holding only finite exact 0/1. Fractional or non-finite values fail."""
    a = np.asarray(a)
    if a.dtype == bool:
        return a
    if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating)):
        fail(f"{name} must be a bool/integer/float binary mask (dtype {a.dtype})")
    if not np.all(np.isfinite(a)) or not np.all((a == 0) | (a == 1)):
        fail(f"{name} must contain only finite exact 0/1 values (dtype {a.dtype})")
    return a == 1


def int_ids(a, name):
    a = np.asarray(a)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        fail(f"{name} must be a 1-D integer array (got {a.dtype}{a.shape})")
    return a.astype(np.int64)


def strictly_increasing(a, name):
    a = int_ids(a, name)
    if a.size == 0 or np.any(np.diff(a) <= 0):
        fail(f"{name} must be non-empty and strictly increasing")
    return a


def check_order(native_ids, prepared_ids):
    n, p = int_ids(native_ids, "native.frame_index"), int_ids(prepared_ids, "prepared.frame_id")
    if n.shape != p.shape or not np.array_equal(n, p):
        fail("native.frame_index != prepared.frame_id (exact ordered match required)")
    return strictly_increasing(n, "frame IDs")


def select_exact(obs_ids, wanted):
    """Indices of observation rows whose frame_id EXACTLY equals each wanted ID (no nearest)."""
    o, w = int_ids(obs_ids, "observation.frame_id"), int_ids(wanted, "wanted frame IDs")
    vals, counts = np.unique(o, return_counts=True)
    if np.any(counts > 1):
        fail(f"duplicate observation frame IDs: {vals[counts > 1][:5].tolist()}")
    pos = {int(v): i for i, v in enumerate(o)}
    missing = [int(x) for x in w if int(x) not in pos]
    if missing:
        fail(f"observation lacks exact frame IDs: {missing[:5]} ({len(missing)} total)")
    return np.array([pos[int(x)] for x in w], dtype=np.int64)


def check_shape(a, shape, name):
    a = np.asarray(a)
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        fail(f"{name} shape {a.shape} != expected {shape}")
    return a


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in keys if k not in z.files]
        if missing:
            fail(f"{path} missing keys {missing}")
        return {k: np.array(z[k]) for k in keys}


def rank_metrics(logits, true_idx):
    """Descending score; ties broken by lower class index first. Rank is 1-based."""
    s = np.asarray(logits, dtype=np.float64)
    if s.ndim != 1 or s.size < 2 or not np.all(np.isfinite(s)):
        fail("logits must be finite, 1-D, with >= 2 classes")
    order = np.lexsort((np.arange(s.size), -s))
    return {"top1": int(order[0]), "correct": bool(order[0] == true_idx),
            "true_rank": int(np.nonzero(order == true_idx)[0][0]) + 1,
            "true_margin": float(s[true_idx] - np.delete(s, true_idx).max()),
            "top2_gap": float(s[order[0]] - s[order[1]])}


def pair_counts(c1, c2, p1, p2):
    c1, c2 = np.asarray(c1, bool), np.asarray(c2, bool)
    return {"n": int(c1.size), "both_correct": int(np.sum(c1 & c2)),
            "first_only_correct": int(np.sum(c1 & ~c2)), "second_only_correct": int(np.sum(~c1 & c2)),
            "both_wrong": int(np.sum(~c1 & ~c2)),
            "prediction_same": int(np.sum(np.asarray(p1) == np.asarray(p2)))}


def overlap_counts(a, b, groups):
    """Availability counts native(a) vs mediapipe(b); groups = ((name, slice), ...) over axis 1."""
    out = {}
    for g, sl in (("all", slice(None)),) + tuple(groups):
        x, y = a[:, sl], b[:, sl]
        out[g] = {"slots": int(x.size), "native": int(x.sum()), "mediapipe": int(y.sum()),
                  "native_only": int((x & ~y).sum()), "mp_only": int((~x & y).sum()),
                  "both": int((x & y).sum())}
    return out


def l2_values(a, b, ma, mb):
    """Per-joint L2 differences at MUTUALLY valid coordinates only."""
    m = np.asarray(ma, bool) & np.asarray(mb, bool)
    d = np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1)[m]
    if not np.all(np.isfinite(d)):
        fail("non-finite normalized coordinate at a mutually valid joint")
    return d


def l2_stats(v):
    v = np.asarray(v, np.float64)
    if v.size == 0:
        return {"count": 0, "meanL2": None, "medianL2": None, "p95L2": None}
    return {"count": int(v.size), "meanL2": float(v.mean()), "medianL2": float(np.median(v)),
            "p95L2": float(np.percentile(v, 95))}


def sum_counts(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = sum_counts([out.get(k, {}), v]) if isinstance(v, dict) else out.get(k, 0) + v
    return out


def verify_rebuild(clip, prep, tol):
    """Rebuilt MediaPipe channels must equal the saved prepared input."""
    x = np.asarray(clip["x"])
    if int(clip["frames"]) != prep["x"].shape[0] or x.shape != prep["x"].shape:
        fail(f"rebuilt x shape {x.shape} / frames {clip['frames']} != prepared {prep['x'].shape}")
    if not np.allclose(x, prep["x"], rtol=0, atol=tol, equal_nan=True):
        fail(f"rebuilt x differs from prepared x (max abs {np.nanmax(np.abs(x - prep['x']))})")
    for k in ("body_valid", "hand_valid"):
        if not np.array_equal(as_bool(clip[k], k), as_bool(prep[k], k)):
            fail(f"rebuilt {k} differs from prepared {k}")
    got = np.array([clip["body_scale"], clip["hand_scales"]["left"], clip["hand_scales"]["right"]], np.float64)
    exp = np.concatenate([np.reshape(prep["body_scale"], -1), np.reshape(prep["hand_scales"], -1)]).astype(np.float64)
    if got.shape != exp.shape or not np.allclose(got, exp, rtol=0, atol=tol, equal_nan=True):
        fail(f"rebuilt scales {got.tolist()} != prepared {exp.tolist()}")


def score(api, model, clip):
    import torch
    xx, pad, _ = api["collate"]([dict(clip, label=0)])
    with torch.inference_mode():
        out = model(torch.from_numpy(np.asarray(xx)), torch.from_numpy(np.asarray(pad)))[0]
    return out.cpu().numpy().astype(np.float64)


def process_row(row, api, model, n_classes, tol):
    nat = load_npz(row["native_source"], ("xy_native", "coordinate_present", "frame_index", "timestamp_nominal_us"))
    prep = load_npz(row["input_path"], ("x", "body_valid", "hand_valid", "timestamp_us", "frame_id",
                                        "body_scale", "hand_scales"))
    obs = load_npz(row["observation_path"], ("xy_px", "usable", "frame_id", "timestamp_us"))
    T = check_shape(nat["frame_index"], (None,), "native.frame_index").shape[0]
    for d, k, shp in ((nat, "xy_native", (T, 67, 2)), (nat, "coordinate_present", (T, 67)),
                      (nat, "timestamp_nominal_us", (T,)), (prep, "x", (T, 280)), (prep, "body_valid", (T, 51)),
                      (prep, "hand_valid", (T, 42)), (prep, "timestamp_us", (T,)), (prep, "frame_id", (T,)),
                      (prep, "hand_scales", (2,)), (obs, "xy_px", (None, 75, 2)), (obs, "usable", (None, 75)),
                      (obs, "frame_id", (None,)), (obs, "timestamp_us", (None,))):
        check_shape(d[k], shp, k)
    if np.size(prep["body_scale"]) != 1 or obs["usable"].shape[0] != obs["xy_px"].shape[0] \
            or obs["frame_id"].shape[0] != obs["xy_px"].shape[0] or obs["timestamp_us"].shape[0] != obs["xy_px"].shape[0]:
        fail("prepared body_scale must be scalar and observation arrays must share N")
    ids = check_order(nat["frame_index"], prep["frame_id"])
    t_nom = strictly_increasing(nat["timestamp_nominal_us"], "native.timestamp_nominal_us")
    t_mp = strictly_increasing(prep["timestamp_us"], "prepared.timestamp_us")
    idx = select_exact(obs["frame_id"], ids)
    if not np.array_equal(int_ids(obs["timestamp_us"], "observation.timestamp_us")[idx], t_mp):
        fail("selected observation timestamps != prepared timestamps")
    mp_xy, mp_pres = api["to_native_like"](obs["xy_px"][idx], as_bool(obs["usable"][idx], "usable"))
    mp_xy, mp_pres = check_shape(np.asarray(mp_xy, np.float32), (T, 67, 2), "mp xy"), as_bool(mp_pres, "mp present")
    nxy, npres = nat["xy_native"].astype(np.float32), as_bool(nat["coordinate_present"], "coordinate_present")
    bc = api["build_channels"]
    clips = {"native": bc(nxy, npres, t_nom), "time_control": bc(nxy, npres, t_mp),
             "availability_control": bc(nxy, npres & mp_pres, t_mp), "mediapipe": bc(mp_xy, mp_pres, t_mp)}
    verify_rebuild(clips["mediapipe"], prep, tol)
    logits = np.stack([score(api, model, clips[a]) for a in ARMS])
    if logits.shape != (len(ARMS), n_classes) or not np.all(np.isfinite(logits)):
        fail(f"logits shape {logits.shape} != ({len(ARMS)}, {n_classes}) or non-finite")
    nc = check_shape(as_bool(api["to_common"](nxy, npres)[1]), (T, 51), "native common present")
    mc = check_shape(as_bool(api["to_common"](mp_xy, mp_pres)[1]), (T, 51), "mp common present")
    n, m = clips["native"], clips["mediapipe"]
    nbv, mbv, nhv, mhv = as_bool(n["body_valid"]), as_bool(m["body_valid"]), as_bool(n["hand_valid"]), as_bool(m["hand_valid"])
    xn, xm = np.asarray(n["x"], np.float64), np.asarray(m["x"], np.float64)
    body_d = l2_values(xn[:, 0:102].reshape(T, 51, 2), xm[:, 0:102].reshape(T, 51, 2), nbv, mbv)
    hand_d = l2_values(xn[:, 102:186].reshape(T, 42, 2), xm[:, 102:186].reshape(T, 42, 2), nhv, mhv)
    rec = {"frames": int(T),
           "raw_common51_availability": overlap_counts(nc, mc, COMMON_GROUPS),
           "effective_body_validity": overlap_counts(nbv, mbv, ()),
           "effective_hand_validity": overlap_counts(nhv, mhv, HAND_GROUPS),
           "source_comparison_not_accuracy": {"body51": l2_stats(body_d), "hand42": l2_stats(hand_d)},
           "time_only_max_abs_logit_change": float(np.max(np.abs(logits[1] - logits[0])))}
    return logits, rec, body_d, hand_d


def check_manifest_identity(rows, expected):
    """Exact rows/clips/labels counts AND exact signer-ID set vs expected["signers"] (a list of IDs).
    IDs are compared as str(...); a matching count with different IDs fails."""
    missing = [k for k in ("rows", "clips", "labels", "signers") if k not in expected]
    if missing:
        fail(f"config.expected missing {missing}")
    got = {"rows": len(rows), "clips": len({str(r["clip_id"]) for r in rows}),
           "labels": len({str(r["label"]) for r in rows})}
    exp = {k: int(expected[k]) for k in ("rows", "clips", "labels")}
    if got != exp:
        fail(f"manifest counts {got} != expected {exp}")
    want = expected["signers"]
    if not isinstance(want, list) or not want:
        fail(f"expected.signers must be a non-empty list of signer IDs (got {want!r})")
    want = [str(s) for s in want]
    if len(set(want)) != len(want):
        fail(f"expected.signers has duplicate IDs: {want}")
    have = sorted({str(r["signer_id"]) for r in rows})
    if have != sorted(want):
        fail(f"manifest signer IDs {have} != expected {sorted(want)}")
    return dict(got, signers=have)


def run(cfg_path, state):
    cfg_sha = sha256_file(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for k in ("manifest", "checkpoint", "output_dir", "python_paths", "pins", "plan", "plan_sha256", "expected", "tolerance"):
        if k not in cfg:
            fail(f"config missing {k}")
    out = cfg["output_dir"]
    if not all(os.path.isabs(cfg[k]) for k in ("manifest", "checkpoint", "output_dir", "plan")):
        fail("manifest, checkpoint, output_dir and plan must be absolute paths")
    tol = float(cfg["tolerance"])
    if not np.isfinite(tol) or tol < 0:
        fail("tolerance must be finite and >= 0")
    state["stage"] = "verify_pins"
    if sha256_file(cfg["plan"]) != str(cfg["plan_sha256"]).lower():
        fail("plan sha256 mismatch")
    pins = {norm(p): str(h).lower() for p, h in cfg["pins"].items()}
    for p in (cfg["manifest"], cfg["checkpoint"]):
        if norm(p) not in pins:
            fail(f"{p} must be pinned in config.pins")
    bad = [p for p, h in pins.items() if sha256_file(p) != h]
    if bad:
        fail(f"pin sha256 mismatch: {bad}")
    state["stage"] = "manifest"
    with open(cfg["manifest"], "r", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    for r in rows:
        missing = [k for k in ROW_KEYS if k not in r]
        if missing:
            fail(f"manifest row missing {missing}")
        if not all(os.path.isabs(r[k]) for k in ("native_source", "input_path", "observation_path")):
            fail(f"row {r['sample_id']}: source paths must be absolute")
    sids = [str(r["sample_id"]) for r in rows]
    if len(set(sids)) != len(sids):
        fail("duplicate sample_id in manifest")
    check_manifest_identity(rows, cfg["expected"])
    tracked = [cfg_path, cfg["plan"], cfg["manifest"], cfg["checkpoint"]] + list(cfg["pins"])
    tracked += [r[k] for r in rows for k in ("native_source", "input_path", "observation_path")]
    before = hash_files(tracked)
    for r in rows:
        for pk, hk in (("input_path", "input_sha256"), ("observation_path", "observation_sha256")):
            if before[norm(r[pk])] != str(r[hk]).lower():
                fail(f"row {r['sample_id']}: {hk} mismatch")
    state["stage"] = "import_and_load"
    for p in reversed(cfg["python_paths"]):
        sys.path.insert(0, p)
    import torch
    from pilot.data import build_channels, collate, to_common
    from pilot.web_input import to_native_like
    import tools_prep_union93_candidate as u
    api = {"build_channels": build_channels, "collate": collate, "to_common": to_common, "to_native_like": to_native_like}
    ck = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    model, classes = u.rebuild_from_payload(ck)
    model.eval()
    torch.set_num_threads(2)
    classes = [str(c) for c in classes]
    true_idx = []
    for r in rows:
        if classes.count(str(r["label"])) != 1:
            fail(f"label {r['label']!r} must occur exactly once in model classes")
        true_idx.append(classes.index(str(r["label"])))
    all_logits, recs, bodies, hands = [], [], [], []
    for r, ti in zip(rows, true_idx):
        state["stage"] = f"row {r['sample_id']}"
        logits, rec, bd, hd = process_row(r, api, model, len(classes), tol)
        arms = {a: rank_metrics(logits[i], ti) for i, a in enumerate(ARMS)}
        recs.append(dict({k: r[k] for k in ("sample_id", "clip_id", "label", "signer_id")}, true_index=ti, arms=arms, **rec))
        all_logits.append(logits)
        bodies.append(bd)
        hands.append(hd)
    state["stage"] = "immutability"
    after = hash_files(tracked)
    changed = check_immutable(before, after)
    if changed:
        fail(f"source files changed during audit: {changed}")
    state["stage"] = "summary"
    comps = {}
    for a1, a2 in PAIRS:
        def cnt(sel):
            rs = [x for x in recs if sel(x)]
            return pair_counts([x["arms"][a1]["correct"] for x in rs], [x["arms"][a2]["correct"] for x in rs],
                               [x["arms"][a1]["top1"] for x in rs], [x["arms"][a2]["top1"] for x in rs])
        comps[f"{a1}->{a2}"] = {
            "overall": cnt(lambda x: True),
            "per_label": {lb: cnt(lambda x, lb=lb: str(x["label"]) == lb) for lb in sorted({str(x["label"]) for x in recs})},
            "per_signer": {s: cnt(lambda x, s=s: str(x["signer_id"]) == s) for s in sorted({str(x["signer_id"]) for x in recs})}}
    summary = {
        "per_arm_correct": {a: int(sum(x["arms"][a]["correct"] for x in recs)) for a in ARMS},
        "paired_comparisons": comps,
        "frames_total": int(sum(x["frames"] for x in recs)),
        "availability_totals": sum_counts([{k: x[k] for k in ("raw_common51_availability", "effective_body_validity",
                                                              "effective_hand_validity")} for x in recs]),
        "pooled_source_comparison_not_accuracy": {"body51": l2_stats(np.concatenate(bodies) if bodies else []),
                                                  "hand42": l2_stats(np.concatenate(hands) if hands else [])},
        "time_only_max_abs_logit_change": max((x["time_only_max_abs_logit_change"] for x in recs), default=None)}
    result = {
        "status": "COMPLETE", "config_sha256": cfg_sha, "plan_sha256": str(cfg["plan_sha256"]).lower(),
        "checkpoint_sha256": before[norm(cfg["checkpoint"])], "arms": list(ARMS), "classes": classes,
        "forward_evaluations": {"rows": len(recs), "arms_per_row": len(ARMS), "total": len(recs) * len(ARMS)},
        "training_performed": False, "extraction_performed": False, "user_files_written": False,
        "caveats": ["Source coordinates are not gold truth; L2 deltas are source-comparison only, not accuracy.",
                    "Data are training-exposed diagnostics, not an independent accuracy benchmark.",
                    "Descriptive paired counts only; no p-values or causal claims about domain quality.",
                    "Index layout assumption: common51 body 0:9, left 9:30, right 30:51; hand42 left 0:21, right 21:42."],
        "summary": summary, "rows": recs}
    state["stage"] = "write"
    with open(os.path.join(out, "SCORES.npz"), "xb") as f:
        np.savez(f, logits=np.stack(all_logits), sample_ids=np.array(sids, dtype=str), arms=np.array(ARMS, dtype=str),
                 classes=np.array(classes, dtype=str), true_indices=np.array(true_idx, dtype=np.int64))
    write_json(os.path.join(out, "SOURCE_HASHES.json"), {"config_sha256": cfg_sha, "before": before, "after": after,
                                                         "immutable": True})
    write_json(os.path.join(out, "RESULT.json"), result)
    print(f"COMPLETE rows={len(recs)} forward_evals={len(recs) * len(ARMS)} "
          f"correct={summary['per_arm_correct']} out={out}")


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as f:
        out = json.load(f).get("output_dir")
    if not out or not os.path.isabs(out) or os.path.exists(out):
        print(f"FAILED: output_dir must be a new absolute path (got {out!r})", file=sys.stderr)
        return 2
    os.mkdir(out)
    state = {"stage": "config"}
    try:
        run(args.config, state)
        return 0
    except Exception as e:  # fail closed: receipt only, no results
        write_json(os.path.join(out, "FAILURE_RECEIPT.json"), {
            "status": "FAILED", "stage": state["stage"], "error_type": type(e).__name__, "error": str(e),
            "traceback": traceback.format_exc(), "note": "No results produced; rows are never skipped or reselected."})
        print(f"FAILED at {state['stage']}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
