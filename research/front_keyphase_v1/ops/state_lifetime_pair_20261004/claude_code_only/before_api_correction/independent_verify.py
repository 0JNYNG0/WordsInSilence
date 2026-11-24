"""Independent verifier for state_lifetime_pair results.

Separate result adapter: does NOT import the runner, MediaPipe, PyAV or camera code and
never re-extracts. Default audit = pins + artifact manifest + structure/clock/lifecycle +
raw per-frame check (pinned independent_raw_reference.verify_frame). --score additionally
loads the model once and repeats the 8 forwards (744 scores, max abs diff <= 1e-5, top1 and
label rank exact).

  python independent_verify.py --config CFG --result RESULT_DIR --out REPORT.json [--score]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

ARMS = ("A_fresh", "B_session")
ARM_SHORT = {"A_fresh": "A", "B_session": "B"}
GAP_US = 1_000_000
TOL = 1e-5
N_CLASSES = 93
CPU_THREADS = 2
FORBIDDEN_MODULES = ("mediapipe", "av", "cv2", "state_lifetime_pair")
NPZ_KEYS = ("decoded_frame_id", "timestamp_us", "absolute_us", "pts", "rgb_sha256",
            "rgb_shape", "hand_task_ms", "pose_task_ms", "offset_us")

# ---------------------------------------------------------------------------
# Pinned-interface assumptions (UNVERIFIED by author; check against pinned sources).
# ---------------------------------------------------------------------------
RAW_VERIFY_FN = "verify_frame"        # rr.verify_frame(f, z, row, 'VIDEO', rule)
CONTRACT_BASENAME = "contract.json"   # pinned contract holding the raw assignment rule
CONTRACT_RAW_RULE_KEY = "raw_rule"    # key of that rule inside the contract
ISR_REBUILD_FN = "rebuild_window_input"  # (z, start_us, end_us_exclusive) -> (x, pad)
ISR_LOAD_FN = "load_model"               # (checkpoint_path) -> (model, classes)
ISR_FORWARD_FN = "forward"               # (model, x, pad) -> logits(93)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Checks:
    def __init__(self):
        self.items, self.failed, self.total = [], 0, 0

    def __call__(self, name, cond, detail=None):
        ok = bool(cond)
        self.total += 1
        self.failed += not ok
        if not ok or detail is not None:
            self.items.append({"check": name, "ok": ok, "detail": detail})
        return ok


def raw_ok(ret):
    """ASSUMED verify_frame contract: None/True pass, False fails, a collection is a list
    of problems (empty = pass); exceptions fail."""
    if ret is None or ret is True:
        return True
    if ret is False:
        return False
    if isinstance(ret, (list, tuple, dict, set)):
        return len(ret) == 0
    return bool(ret)


def hash_pins(cfg):
    actual = {}
    for p in cfg["pins"]:
        try:
            actual[p] = sha256_file(p)
        except OSError as e:
            actual[p] = f"ERROR:{type(e).__name__}"
    return {"actual": actual, "mismatches": [p for p, h in cfg["pins"].items() if actual[p] != h]}


def artifact_hashes(result_dir):
    files = {}
    for root, _, names in os.walk(result_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            files[os.path.relpath(p, result_dir).replace("\\", "/")] = sha256_file(p)
    return files


def check_manifest(result_dir, ck):
    man = load_json(os.path.join(result_dir, "RESULT_MANIFEST.json"))["files"]
    actual = artifact_hashes(result_dir)
    actual.pop("RESULT_MANIFEST.json", None)
    ck("manifest_file_set", set(man) == set(actual),
       {"missing": sorted(set(man) - set(actual)), "extra": sorted(set(actual) - set(man))})
    ck("manifest_hashes", all(actual.get(k) == v for k, v in man.items()))
    return actual


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def half_open_ids(z, start, end):
    ts = z["timestamp_us"]
    sel = (ts >= start) & (ts < end)
    return [int(v) for v in z["decoded_frame_id"][sel]], [int(v) for v in ts[sel]]


def import_pinned(name, path, pins):
    if pins.get(path) != sha256_file(path):
        raise RuntimeError(f"pin mismatch before import: {path}")
    if name in sys.modules:
        raise RuntimeError(f"module name already loaded: {name}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# structure / clock / lifecycle (pure; no pinned helpers)
# ---------------------------------------------------------------------------
def verify_structure(cfg, result_dir, ck):
    clips = {c["clip_id"]: c for c in cfg["clips"]}
    budgets = cfg["budgets"]
    res = load_json(os.path.join(result_dir, "RESULT.json"))
    ck("result_status_pass", res.get("status") == "PASS", res.get("status"))
    fields = res["obs_fields"]
    ck("obs_fields_7", len(fields) == 7 and {"timestamp_us", "frame_id"} <= set(fields), fields)
    ctx = {"z": {}, "diag": {}, "fields": fields, "pos": {}}
    total = 0
    for gi, group in enumerate(cfg["groups"]):
        prev_last = None
        for pos, cid in enumerate(group):
            ctx["pos"][cid] = (gi, pos)
            for arm in ARMS:
                stem = f"{cid}__{arm}"
                ctx["z"][(cid, arm)] = load_npz(os.path.join(result_dir, "clips", stem + ".npz"))
                ctx["diag"][(cid, arm)] = load_json(os.path.join(result_dir, "diag", stem + ".json"))
            zA, zB = ctx["z"][(cid, "A_fresh")], ctx["z"][(cid, "B_session")]
            keys_needed = set(NPZ_KEYS) | {f"obs_{f}" for f in fields}
            for arm, z in (("A_fresh", zA), ("B_session", zB)):
                ck(f"{cid}/{arm}/npz_keys", keys_needed <= set(z), sorted(keys_needed - set(z)))
            n = len(zA["decoded_frame_id"])
            total += n
            ck(f"{cid}/frames_expected", n == int(clips[cid]["expected_frames_from_prior"]), n)
            for k in ("decoded_frame_id", "timestamp_us", "absolute_us", "pts", "rgb_sha256",
                      "rgb_shape", "offset_us"):
                ck(f"{cid}/AB_same_{k}", np.array_equal(zA[k], zB[k]))
            ids, ts, ab = zA["decoded_frame_id"], zA["timestamp_us"], zA["absolute_us"]
            off = int(zA["offset_us"])
            exp_off = 0 if pos == 0 else prev_last + GAP_US
            ck(f"{cid}/offset", off == exp_off, {"offset": off, "expected": exp_off})
            if pos == 0:
                ck(f"{cid}/group_first_relative0", int(ts[0]) == 0, int(ts[0]))
            else:
                ck(f"{cid}/gap_exact", int(ab[0]) - prev_last == GAP_US + int(ts[0]))
            ck(f"{cid}/ids_strict", bool(np.all(np.diff(ids) > 0)))
            ck(f"{cid}/pts_strict", bool(np.all(np.diff(ts) > 0)))
            ck(f"{cid}/relative_is_pts", np.array_equal(ts, zA["pts"]))
            ck(f"{cid}/absolute_eq_offset_plus_relative", np.array_equal(ab, ts + off))
            ck(f"{cid}/rgb_shape_3d", zA["rgb_shape"].ndim == 2 and zA["rgb_shape"].shape[1] == 3)
            for arm, z in (("A_fresh", zA), ("B_session", zB)):
                ck(f"{cid}/{arm}/obs_ts_relative", np.array_equal(z["obs_timestamp_us"], ts))
                ck(f"{cid}/{arm}/obs_frame_id", np.array_equal(z["obs_frame_id"], ids))
                for t in ("hand_task_ms", "pose_task_ms"):
                    ck(f"{cid}/{arm}/{t}_no_clamp", np.array_equal(z[t], ab // 1000))
                d = ctx["diag"][(cid, arm)]
                rows = d["rows"]
                ck(f"{cid}/{arm}/diag_rows", len(rows) == n and d["offset_us"] == off
                   and d["instr_slice"][1] - d["instr_slice"][0] == n)
                ck(f"{cid}/{arm}/diag_meta", all(
                    r["clip_id"] == cid and r["arm"] == arm and r["local_row"] == k
                    and r["frame_id"] == int(ids[k]) and r["relative_us"] == int(ts[k])
                    and r["absolute_us"] == int(ab[k]) and r["offset_us"] == off
                    for k, r in enumerate(rows)))
            if pos == 0:
                bad = [f for f in fields if zA[f"obs_{f}"].dtype != zB[f"obs_{f}"].dtype
                       or zA[f"obs_{f}"].tobytes() != zB[f"obs_{f}"].tobytes()]
                ck(f"{cid}/first_clip_bitexact", not bad, bad or None)
            prev_last = int(ab[-1])
    ck("total_decoded", total == budgets["decoded_frames"], total)
    ck("total_process", 2 * total == budgets["process_calls"], 2 * total)
    verify_registry(cfg, load_json(os.path.join(result_dir, "LM_REGISTRY.json")), ck)
    cnt = res["counters"]
    for k in ("decoded_frames", "process_calls", "landmarkers", "internal_tasks",
              "model_loads", "main_forwards"):
        ck(f"counter_{k}", cnt.get(k) == budgets[k], {"counter": cnt.get(k), "budget": budgets[k]})
    scores = load_json(os.path.join(result_dir, "SCORES.json"))
    ctx["scores"] = scores
    scored = [c for c in cfg["clips"] if c.get("score")]
    ck("scores_windows", [w["clip_id"] for w in scores["windows"]] == [c["clip_id"] for c in scored])
    ck("scores_classes93", len(scores["scoring"]["classes"]) == N_CLASSES)
    for w in scores["windows"]:
        c = clips[w["clip_id"]]["score"]
        ck(f"{w['clip_id']}/window_cfg", (w["label"], w["start_us"], w["end_us_exclusive"])
           == (c["label"], c["start_us"], c["end_us_exclusive"]))
        wid, wts = half_open_ids(ctx["z"][(w["clip_id"], "A_fresh")], w["start_us"], w["end_us_exclusive"])
        ck(f"{w['clip_id']}/window_half_open_all_frames",
           wid == w["window_frame_ids"] and wts == w["window_timestamp_us"])
        ck(f"{w['clip_id']}/history_flag", w["history_applies"] == (ctx["pos"][w["clip_id"]][1] > 0))
    return ctx


def verify_registry(cfg, reg, ck):
    lms = reg["landmarkers"]
    ev = reg["events"]
    budgets = cfg["budgets"]
    ck("lm_count", len(lms) == budgets["landmarkers"], len(lms))
    ck("task_count", sum(len(h["task_obj_ids"]) for h in lms) == budgets["internal_tasks"])
    ck("lm_ids_unique", len({h["lm_obj_id"] for h in lms}) == len(lms))
    tids = [t for h in lms for t in h["task_obj_ids"]]
    ck("task_ids_unique", len(set(tids)) == len(tids))
    ck("all_closed_ok", all(h["close_ok"] and len(h["close_records"]) == 2
                            and all(r["ok"] for r in h["close_records"]) for h in lms),
       [h["key"] for h in lms if not h["close_ok"]] or None)

    def seq(key, kind, cid=None):
        s = [e["seq"] for e in ev if e["key"] == key and e["event"] == kind
             and (cid is None or e["clip_id"] == cid)]
        return s

    for gi, group in enumerate(cfg["groups"]):
        bs = [h for h in lms if h["arm"] == "B_session" and h["group"] == gi]
        if not ck(f"group{gi}/one_B", len(bs) == 1):
            continue
        b = bs[0]
        ck(f"group{gi}/B_membership", b["clips"] == group, b["clips"])
        ck(f"group{gi}/B_closed_once", len(seq(b["key"], "close")) == 1)
        starts = [seq(b["key"], "clip_start", c) for c in group]
        ends = [seq(b["key"], "clip_end", c) for c in group]
        ck(f"group{gi}/B_events", all(len(s) == 1 for s in starts + ends))
        if all(len(s) == 1 for s in starts + ends):
            ck(f"group{gi}/B_lifecycle", b["created_seq"] < starts[0][0]
               and ends[-1][0] < b["closed_seq"])
            last = {e["seq"]: e["last_ms_seen"] for e in ev if e["key"] == b["key"]}
            for p in range(1, len(group)):
                ck(f"group{gi}/B_last_ms_unchanged_in_gap_{p}",
                   last[ends[p - 1][0]] == last[starts[p][0]])
        prev_close = None
        for cid in group:
            a = [h for h in lms if h["arm"] == "A_fresh" and h["clips"] == [cid]]
            if not ck(f"{cid}/one_fresh_A", len(a) == 1):
                continue
            a = a[0]
            s, e = seq(a["key"], "clip_start", cid), seq(a["key"], "clip_end", cid)
            ok = len(s) == 1 and len(e) == 1 and a["created_seq"] < s[0] < e[0] < a["closed_seq"]
            ck(f"{cid}/A_lifecycle", ok and (prev_close is None or prev_close < a["created_seq"]))
            ck(f"{cid}/A_process_calls", a["process_calls"] == int(
                next(c for c in cfg["clips"] if c["clip_id"] == cid)["expected_frames_from_prior"]))
            prev_close = a["closed_seq"]
    a_count = sum(h["arm"] == "A_fresh" for h in lms)
    ck("A_count", a_count == len(cfg["clips"]), a_count)


# ---------------------------------------------------------------------------
# raw per-frame check through the pinned raw reference
# ---------------------------------------------------------------------------
def verify_raw(ctx, rr, rule, ck):
    fn = getattr(rr, RAW_VERIFY_FN)
    n_ok = n_all = 0
    first_err = None
    for (cid, arm), z in ctx["z"].items():
        rows = ctx["diag"][(cid, arm)]["rows"]
        if not np.array_equal(z["timestamp_us"] + z["offset_us"], z["absolute_us"]):
            ck(f"{cid}/{arm}/raw_alias_precondition", False)
            continue
        zc = dict(z)  # copy; file arrays never mutated
        zc["obs_timestamp_us"] = z["absolute_us"]
        for f, r in enumerate(rows):
            n_all += 1
            try:
                ret = fn(f, zc, r["row"], "VIDEO", rule)
                ok = raw_ok(ret)
            except Exception as e:
                ok = False
                ret = f"{type(e).__name__}: {e}"
            n_ok += ok
            if not ok and first_err is None:
                first_err = {"clip_id": cid, "arm": arm, "f": f, "detail": repr(ret)}
    return n_ok, n_all, first_err


# ---------------------------------------------------------------------------
# score reproduction (--score only)
# ---------------------------------------------------------------------------
def verify_scores(cfg, ctx, isr, ck, result_dir):
    import torch
    torch.set_num_threads(CPU_THREADS)
    ckpt = cfg["scoring"]["checkpoint_path"]
    ck("checkpoint_pin", sha256_file(ckpt) == cfg["scoring"]["checkpoint_sha256"])
    model, classes = getattr(isr, ISR_LOAD_FN)(ckpt)
    model.eval()
    classes = list(classes)
    scores = ctx["scores"]
    ck("classes_match", classes == scores["scoring"]["classes"])
    inputs = load_npz(os.path.join(result_dir, "SCORE_INPUTS.npz"))
    rebuild, forward = getattr(isr, ISR_REBUILD_FN), getattr(isr, ISR_FORWARD_FN)
    n_fwd = n_cmp = 0
    worst = 0.0
    for w in scores["windows"]:
        cid, lab = w["clip_id"], w["label"]
        li = classes.index(lab)
        for arm in ARMS:
            s = ARM_SHORT[arm]
            x, pad = rebuild(ctx["z"][(cid, arm)], w["start_us"], w["end_us_exclusive"])
            x, pad = np.asarray(x), np.asarray(pad)
            ck(f"{cid}/{s}/x_equal", x.shape == inputs[f"{cid}__{s}__x"].shape
               and np.array_equal(x, inputs[f"{cid}__{s}__x"], equal_nan=True))
            ck(f"{cid}/{s}/pad_equal", np.array_equal(pad, inputs[f"{cid}__{s}__pad"]))
            with torch.no_grad():
                t = forward(model, x, pad)
            lg = np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t,
                            dtype=np.float64).reshape(-1)
            n_fwd += 1
            ref = np.asarray(w[s]["logits93"], dtype=np.float64)
            ok_shape = ck(f"{cid}/{s}/logits93", lg.shape == (N_CLASSES,) == ref.shape)
            if not ok_shape:
                continue
            d = float(np.max(np.abs(lg - ref)))
            worst = max(worst, d)
            n_cmp += lg.size
            ck(f"{cid}/{s}/max_abs<=1e-5", d <= TOL, d)
            ck(f"{cid}/{s}/top1", classes[int(np.argmax(lg))] == w[s]["top1"])
            ck(f"{cid}/{s}/label_rank", 1 + int((lg > lg[li]).sum()) == w[s]["label_rank"])
    ck("independent_forwards", n_fwd == cfg["budgets"]["independent_forwards"], n_fwd)
    ck("scores_compared_744", n_cmp == N_CLASSES * cfg["budgets"]["independent_forwards"], n_cmp)
    return {"forwards": n_fwd, "scores_compared": n_cmp, "max_abs_diff": worst, "model_loads": 1}


def forbidden_loaded():
    return [m for m in FORBIDDEN_MODULES if m in sys.modules]


def main(argv=None):
    ap = argparse.ArgumentParser(description="independent verify for state_lifetime_pair")
    ap.add_argument("--config", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score", action="store_true", help="load model and repeat 8 forwards")
    a = ap.parse_args(argv)
    result_dir, out = os.path.abspath(a.result), os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite {out}")
    if os.path.commonpath([result_dir, out]) == result_dir:
        raise SystemExit("report must be written outside the result dir")
    cfg = load_json(a.config)
    ck = Checks()
    report = {"schema": "wis_state_lifetime_pair_independent_v1", "config_sha256": sha256_file(a.config),
              "result_dir": result_dir, "score_mode": a.score}
    pre_pins = hash_pins(cfg)
    ck("pins_pre", not pre_pins["mismatches"], pre_pins["mismatches"] or None)
    pre_art = check_manifest(result_dir, ck)
    ck("no_forbidden_modules_start", not forbidden_loaded(), forbidden_loaded() or None)
    try:
        ctx = verify_structure(cfg, result_dir, ck)
        rr = import_pinned("_wis_slp_raw_ref", cfg["independent_raw_reference"], cfg["pins"])
        contract = [p for p in cfg["pins"] if os.path.basename(p) == CONTRACT_BASENAME]
        ck("contract_pinned_once", len(contract) == 1, contract)
        rule = load_json(contract[0])[CONTRACT_RAW_RULE_KEY]
        n_ok, n_all, first_err = verify_raw(ctx, rr, rule, ck)
        ck("raw_frames_all", n_all == cfg["budgets"]["process_calls"] and n_ok == n_all,
           {"ok": n_ok, "all": n_all, "first_error": first_err})
        report["raw"] = {"verified": n_ok, "total": n_all}
        if a.score:
            isr = import_pinned("_wis_slp_score_ref", cfg["independent_score_reference"], cfg["pins"])
            report["scores"] = verify_scores(cfg, ctx, isr, ck, result_dir)
    except Exception as e:
        ck("verifier_exception", False, f"{type(e).__name__}: {e}")
    ck("no_forbidden_modules_end", not forbidden_loaded(), forbidden_loaded() or None)
    post_pins = hash_pins(cfg)
    ck("pins_post_equal_pre", post_pins == pre_pins)
    post_art = artifact_hashes(result_dir)
    post_art.pop("RESULT_MANIFEST.json", None)
    ck("artifacts_unchanged", post_art == pre_art)
    report.update({"status": "PASS" if ck.failed == 0 else "FAIL", "n_failed": ck.failed,
                   "n_checks": ck.total, "checks": ck.items, "pins_post": post_pins})
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"status": report["status"], "n_failed": ck.failed, "out": out}))
    return 0 if ck.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
