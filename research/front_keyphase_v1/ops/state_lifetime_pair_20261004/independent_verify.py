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
from pathlib import Path
from fractions import Fraction

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
OBS_FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")

# ---------------------------------------------------------------------------
# Exact existing independent helper API; candidate runner is never imported.
# ---------------------------------------------------------------------------
RAW_VERIFY_FN = "verify_frame"        # rr.verify_frame(frame_dict, arrays, local_row, 'VIDEO', rule)
CONTRACT_BASENAME = "contract.json"
CONTRACT_RAW_RULE_KEY = "usable_rule"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as fh:
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


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def array_hash(a):
    a = np.ascontiguousarray(a)
    header = a.dtype.str + "|" + "x".join(map(str, a.shape)) + "|"
    return hashlib.sha256(header.encode("ascii") + a.tobytes()).hexdigest()


def require_clean(ck, stage):
    if ck.failed:
        raise AssertionError(f"{stage}: preceding audit checks failed ({ck.failed})")


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
def verify_structure(cfg, result_dir, ck, ref):
    clips = {c["clip_id"]: c for c in cfg["clips"]}
    budgets = cfg["budgets"]
    res = load_json(os.path.join(result_dir, "RESULT.json"))
    ck("result_status_pass", res.get("status") == "PASS", res.get("status"))
    fields = res["obs_fields"]
    ck("obs_fields_7", tuple(fields) == OBS_FIELDS, fields)
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
                ck(f"{cid}/AB_same_{k}", exact(zA[k], zB[k]))
            ids, ts, ab = zA["decoded_frame_id"], zA["timestamp_us"], zA["absolute_us"]
            off = int(zA["offset_us"])
            exp_off = 0 if pos == 0 else prev_last + GAP_US
            ck(f"{cid}/offset", off == exp_off, {"offset": off, "expected": exp_off})
            ck(f"{cid}/relative0", int(ts[0]) == 0, int(ts[0]))
            if pos > 0:
                ck(f"{cid}/gap_exact", int(ab[0]) - prev_last == GAP_US + int(ts[0]))
            ck(f"{cid}/ids_full", exact(ids, np.arange(n, dtype=np.int64)))
            ck(f"{cid}/pts_strict", bool(np.all(np.diff(ts) > 0)))
            for arm, z in (("A_fresh", zA), ("B_session", zB)):
                d = ctx["diag"][(cid, arm)]
                tb_value = z["time_base"].item() if "time_base" in z else d["time_base"]
                tb = Fraction(str(tb_value))
                ck(f"{cid}/{arm}/time_base_metadata", str(tb_value) == d["time_base"])
                pts = z["pts"]
                ck(f"{cid}/{arm}/raw_pts", pts.dtype == np.dtype("int64") and pts.shape == (n,)
                   and bool(np.all(np.diff(pts) > 0)) and tb > 0)
                expected_ts = np.asarray([int(round(float((int(p) - int(pts[0])) * tb) * 1_000_000))
                                          for p in pts], dtype=np.int64)
                ck(f"{cid}/{arm}/relative_from_source_pts", exact(expected_ts, ts))
                obs = {f: z["obs_" + f] for f in OBS_FIELDS}
                ck(f"{cid}/{arm}/valid_observation", ref.validate_observation(obs) == n)
            ck(f"{cid}/absolute_eq_offset_plus_relative", np.array_equal(ab, ts + off))
            ck(f"{cid}/rgb_shape_3d", zA["rgb_shape"].shape == (n, 3)
               and bool(np.all(zA["rgb_shape"] > 0)) and bool(np.all(zA["rgb_shape"][:, 2] == 3)))
            ck(f"{cid}/rgb_hashes", zA["rgb_sha256"].shape == (n,) and all(
                isinstance(h, str) and len(h) == 64 and set(h) <= set("0123456789abcdef")
                for h in zA["rgb_sha256"].tolist()))
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
                bad = [f for f in fields if not exact(zA[f"obs_{f}"], zB[f"obs_{f}"])]
                ck(f"{cid}/first_clip_bitexact", not bad, bad or None)
            prev_last = int(ab[-1])
    ck("total_decoded", total == budgets["decoded_frames"], total)
    ck("total_process", 2 * total == budgets["process_calls"], 2 * total)
    registry = load_json(os.path.join(result_dir, "LM_REGISTRY.json"))
    verify_registry(cfg, registry, ck)
    registry_by_key = {h["key"]: h for h in registry["landmarkers"]}
    for gi, group in enumerate(cfg["groups"]):
        prior_frames = 0
        for pos, cid in enumerate(group):
            n = int(clips[cid]["expected_frames_from_prior"])
            for arm in ARMS:
                d = ctx["diag"][(cid, arm)]
                h = registry_by_key[d["lm_key"]]
                start = 0 if arm == "A_fresh" else prior_frames
                ck(f"{cid}/{arm}/registry_link", h["arm"] == arm and h["group"] == gi
                   and cid in h["clips"] and d["group"] == gi and d["position"] == pos
                   and res["clips"][cid]["lm_keys"][arm] == h["key"])
                ck(f"{cid}/{arm}/instrument_slice", d["instr_slice"] == [start, start + n])
            prior_frames += n
    cnt = res["counters"]
    for k in ("decoded_frames", "process_calls", "landmarkers", "internal_tasks",
              "model_loads", "main_forwards"):
        ck(f"counter_{k}", cnt.get(k) == budgets[k], {"counter": cnt.get(k), "budget": budgets[k]})
    scores = load_json(os.path.join(result_dir, "SCORES.json"))
    ctx["scores"] = scores
    scored = [c for c in cfg["clips"] if c.get("score")]
    ck("scores_windows", [w["clip_id"] for w in scores["windows"]] == [c["clip_id"] for c in scored])
    ck("scores_classes93", len(scores["scoring"]["classes"]) == N_CLASSES
       and len(set(scores["scoring"]["classes"])) == N_CLASSES)
    sc = scores["scoring"]
    ck("scores_model_contract", sc["scope"] == "full93"
       and sc["checkpoint_sha256"] == cfg["scoring"]["checkpoint_sha256"]
       and sc["cpu_threads"] == CPU_THREADS and sc["tolerance"] == TOL
       and sc["frame_selection"] == cfg["scoring"]["frame_selection"])
    ck("main_model_state_unchanged", isinstance(sc["model_state_before"], str)
       and len(sc["model_state_before"]) == 64
       and sc["model_state_before"] == sc["model_state_after"])
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
    for h in lms:
        tasks = [r["task_record"] for r in h["close_records"]]
        ck(h["key"] + "/task_modes_and_calls", {t["task"] for t in tasks}
           == {"hand_VIDEO", "pose_VIDEO"}
           and all(t["calls"] == h["process_calls"] and t["close"]["attempted"]
                   and t["close"]["ok"] for t in tasks))
        ck(h["key"] + "/close_task_ids", [r["task_obj_id"] for r in h["close_records"]]
           == h["task_obj_ids"])
        p = h["provenance"]
        ck(h["key"] + "/instrument_clean", h["instrument_errors"] == [])
        ck(h["key"] + "/same_extractor", p == lms[0]["provenance"]
           and p["hand_running_mode"] == p["pose_running_mode"] == "VIDEO"
           and p["hand_api"] == "detect_for_video"
           and p["hand_options"] == {"num_hands": 2, "min_hand_detection_confidence": 0.5,
                                      "min_hand_presence_confidence": 0.5, "min_tracking_confidence": 0.5}
           and p["pose_options"] == {"running_mode": "VIDEO", "num_poses": 1})
        for part in ("hand_task", "pose_task"):
            ck(h["key"] + "/" + part + "_pin", cfg["pins"].get(p[part]["path"]) == p[part]["sha256"])
    ck("registry_process_total", sum(h["process_calls"] for h in lms) == budgets["process_calls"])
    ck("event_sequence", [e["seq"] for e in ev] == list(range(1, len(ev) + 1)))

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
        ck(f"group{gi}/B_process_calls", b["process_calls"] == sum(
            c["expected_frames_from_prior"] for c in cfg["clips"] if c["clip_id"] in group))
        ck(f"group{gi}/B_closed_once", len(seq(b["key"], "close")) == 1)
        starts = [seq(b["key"], "clip_start", c) for c in group]
        ends = [seq(b["key"], "clip_end", c) for c in group]
        ck(f"group{gi}/B_events", all(len(s) == 1 for s in starts + ends))
        if all(len(s) == 1 for s in starts + ends):
            ck(f"group{gi}/B_lifecycle", b["created_seq"] < starts[0][0]
               and ends[-1][0] < b["closed_seq"]
               and all(s[0] < e[0] for s, e in zip(starts, ends))
               and all(ends[p-1][0] < starts[p][0] for p in range(1, len(group))))
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
            ck(f"{cid}/A_closed_once", len(seq(a["key"], "close")) == 1)
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
        zc["decoded_rgb_shape"] = z["rgb_shape"]
        for local_row, r in enumerate(rows):
            n_all += 1
            try:
                frame = dict(r["row"])
                if frame["timestamp_us"] != r["absolute_us"]:
                    raise AssertionError("inner raw diagnostic absolute timestamp")
                if frame["frame_id"] != r["frame_id"]:
                    raise AssertionError("inner raw diagnostic original frame ID")
                if frame["row"] != local_row:
                    raise AssertionError("inner raw diagnostic local row")
                ret = fn(frame, zc, local_row, "VIDEO", rule)
                ok = ret is None  # exact helper returns None or raises AssertionError
            except Exception as e:
                ok = False
                ret = f"{type(e).__name__}: {e}"
            n_ok += ok
            if not ok and first_err is None:
                first_err = {"clip_id": cid, "arm": arm, "row": local_row, "detail": repr(ret)}
    return n_ok, n_all, first_err


# ---------------------------------------------------------------------------
# score reproduction (--score only)
# ---------------------------------------------------------------------------
def check_official_module(module, expected_path, cfg):
    path = os.path.normcase(os.path.abspath(module.__file__))
    expected = os.path.normcase(os.path.abspath(expected_path))
    if path != expected or cfg["pins"].get(str(expected_path)) != sha256_file(expected_path):
        raise AssertionError(f"Official source identity mismatch: {module.__name__}")


def ranking(logits, classes, label):
    lg = np.asarray(logits, dtype=np.float64)
    if lg.shape != (N_CLASSES,) or not np.isfinite(lg).all():
        raise AssertionError("Expected 93 finite scores")
    li = classes.index(label)
    order = np.lexsort((np.arange(N_CLASSES), -lg))
    best, rival = int(order[0]), int(order[order != li][0])
    return {"scope": "full93", "true_label": label, "top1": classes[best],
            "top1_index": best, "correct": best == li,
            "true_rank": int(np.flatnonzero(order == li)[0]) + 1,
            "best_other": classes[rival], "true_vs_best_other_margin": float(lg[li] - lg[rival])}


def verify_inputs(cfg, ctx, ref, ck, result_dir):
    """Eight official inputs, no model import or execution, no active_span."""
    pilot = Path(cfg["workspace_root"]) / "research/sign_model_pilot_v1/src"
    sys.path.insert(0, str(pilot))
    import pilot.data as data
    import pilot.web_input as web_input
    check_official_module(data, pilot / "pilot/data.py", cfg)
    check_official_module(web_input, pilot / "pilot/web_input.py", cfg)
    inputs = load_npz(os.path.join(result_dir, "SCORE_INPUTS.npz"))
    classes = ctx["scores"]["scoring"]["classes"]
    rebuilt, expected_keys, summaries = {}, set(), []
    for w in ctx["scores"]["windows"]:
        cid, lab = w["clip_id"], w["label"]
        li = classes.index(lab)
        for arm in ARMS:
            s = ARM_SHORT[arm]
            z = ctx["z"][(cid, arm)]
            obs = {k: z["obs_" + k] for k in OBS_FIELDS}
            ix = ref.window_indices(obs, w["start_us"], w["end_us_exclusive"])
            ck(f"{cid}/{s}/selected_ids", obs["frame_id"][ix].tolist() == w["window_frame_ids"])
            ck(f"{cid}/{s}/selected_pts", obs["timestamp_us"][ix].tolist() == w["window_timestamp_us"])
            xy, present = web_input.to_native_like(obs["xy_px"][ix], obs["usable"][ix])
            feature = data.build_channels(xy, present, obs["timestamp_us"][ix])
            x, pad, labels = data.collate([dict(feature, label=0)])
            ck(f"{cid}/{s}/dummy_label_and_real_mask", np.all(labels == 0) and np.all(pad == 1))
            xkey, pkey = f"{cid}__{s}__x", f"{cid}__{s}__pad"
            expected_keys.update((xkey, pkey))
            ck(f"{cid}/{s}/x_equal", exact(x, inputs[xkey]))
            ck(f"{cid}/{s}/pad_equal", exact(pad, inputs[pkey]))
            saved = w[s]["input"]
            for name, value in (("x", x), ("pad", pad)):
                ck(f"{cid}/{s}/{name}_metadata", saved[name + "_shape"] == list(value.shape)
                   and saved[name + "_dtype"] == value.dtype.str
                   and saved[name + "_sha256"] == array_hash(value))
            ck(f"{cid}/{s}/body_scale", feature["body_scale"] == saved["body_scale"])
            for side, scale in feature["hand_scales"].items():
                ck(f"{cid}/{s}/{side}_scale", (scale if np.isfinite(scale) else None)
                   == saved["hand_scales"][side])
            sr = ranking(w[s]["logits93"], classes, lab)
            ck(f"{cid}/{s}/stored_score", sr == w[s]["score"])
            ck(f"{cid}/{s}/stored_summary", sr["top1"] == w[s]["top1"]
               and sr["true_rank"] == w[s]["label_rank"]
               and w[s]["label_logit"] == w[s]["logits93"][li])
            rebuilt[(cid, arm)] = (x, pad)
            summaries.append({"clip_id": cid, "arm": arm, "x_exact": True,
                              "frames": len(ix), "score": sr})
    ck("score_input_file_set", set(inputs) == expected_keys)
    ck("rebuilt_inputs_8", len(rebuilt) == cfg["budgets"]["independent_forwards"] == 8)
    require_clean(ck, "input reproduction")
    ctx["rebuilt"] = rebuilt
    return {"count": len(rebuilt), "all_exact": True, "windows": summaries}


def state_digest(state):
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(f"|{t.dtype}|{tuple(t.shape)}|".encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_scores(cfg, ctx, ck):
    """One strict fixed-model load and exactly eight forwards after audit gates."""
    require_clean(ck, "before model load")
    import torch
    import pilot.model_masked_v1 as model_source
    path = Path(cfg["workspace_root"]) / "research/sign_model_pilot_v1/src/pilot/model_masked_v1.py"
    check_official_module(model_source, path, cfg)
    torch.set_num_threads(CPU_THREADS)
    ckpt = cfg["scoring"]["checkpoint_path"]
    ck("checkpoint_pin", sha256_file(ckpt) == cfg["scoring"]["checkpoint_sha256"])
    require_clean(ck, "checkpoint")
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    classes = list(payload["classes"])
    ck("classes_match", classes == ctx["scores"]["scoring"]["classes"]
       and payload["n_classes"] == len(classes) == len(set(classes)) == N_CLASSES)
    model = model_source.MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
        kernel_size=payload["kernel_size"], dropout=payload["dropout"], n_classes=payload["n_classes"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = state_digest(model.state_dict())
    ck("loaded_and_main_model_match", before == state_digest(payload["state_dict"])
       == ctx["scores"]["scoring"]["model_state_before"])
    require_clean(ck, "model identity")
    n_fwd = n_cmp = 0
    worst, windows = 0.0, []
    for w in ctx["scores"]["windows"]:
        cid = w["clip_id"]
        for arm in ARMS:
            s = ARM_SHORT[arm]
            x, pad = ctx["rebuilt"][(cid, arm)]
            with torch.inference_mode():
                lg = model(torch.from_numpy(x), torch.from_numpy(pad))[0].cpu().numpy().astype(np.float64)
            n_fwd += 1
            reference = np.asarray(w[s]["logits93"], dtype=np.float64)
            sr = ranking(lg, classes, w["label"])
            d = float(np.max(np.abs(lg - reference)))
            worst, n_cmp = max(worst, d), n_cmp + N_CLASSES
            ck(f"{cid}/{s}/max_abs<=1e-5", d <= TOL, d)
            ck(f"{cid}/{s}/top1_rank", sr["top1"] == w[s]["top1"]
               and sr["true_rank"] == w[s]["label_rank"])
            windows.append({"clip_id": cid, "arm": arm, "max_abs_diff": d, "score": sr})
    ck("independent_forwards", n_fwd == cfg["budgets"]["independent_forwards"] == 8, n_fwd)
    ck("scores_compared_744", n_cmp == 744, n_cmp)
    after = state_digest(model.state_dict())
    ck("independent_model_unchanged", after == before)
    return {"forwards": n_fwd, "scores_compared": n_cmp, "max_abs_diff": worst,
            "model_loads": 1, "state_before": before, "state_after": after, "windows": windows}


def check_run_provenance(cfg, config_path, result_dir, ck):
    run = load_json(os.path.join(result_dir, "RUN_MANIFEST.json"))
    result = load_json(os.path.join(result_dir, "RESULT.json"))
    before, after = run["preflight"], result["post_hashes"]
    cfg_sha = sha256_file(config_path)
    ck("run_config_identity", before["config_sha256"] == after["config_sha256"] == cfg_sha)
    ck("run_own_sources_unchanged", before["own_code_sha256"] == after["own_code_sha256"])
    unit = Path(cfg["unit_dir"])
    for name, digest in before["own_code_sha256"].items():
        if Path(name).name != name:
            raise AssertionError("non-leaf own-source path")
        ck("run_source:" + name, sha256_file(unit / name) == digest)
    ck("run_pins_unchanged", before["pins"] == after["pins"] and before["pins"]["ok"])
    return {"config_sha256": cfg_sha, "own_source_sha256": before["own_code_sha256"]}


def verify_comparison(ctx, result_dir, ck):
    """Recount all/history-only decisions from the already checked 93-score vectors."""
    saved = load_json(os.path.join(result_dir, "COMPARISON.json"))
    rows = []
    transitions = ("both_correct", "both_wrong", "B_new_wrong_regression_vs_fresh",
                   "B_new_correct_improvement_vs_fresh")
    classes = ctx["scores"]["scoring"]["classes"]
    ck("comparison_window_ids", [r["clip_id"] for r in saved["windows"]]
       == [w["clip_id"] for w in ctx["scores"]["windows"]])
    for w, row in zip(ctx["scores"]["windows"], saved["windows"]):
        a = ranking(w["A"]["logits93"], classes, w["label"])
        b = ranking(w["B"]["logits93"], classes, w["label"])
        ac, bc = a["correct"], b["correct"]
        transition = transitions[0] if ac and bc else transitions[1] if not ac and not bc else transitions[2] if ac else transitions[3]
        ck(w["clip_id"] + "/comparison_truth", row["A_correct"] == ac and row["B_correct"] == bc
           and row["A_top1"] == a["top1"] and row["B_top1"] == b["top1"]
           and row["A_label_rank"] == a["true_rank"] and row["B_label_rank"] == b["true_rank"]
           and row["transition"] == transition and row["history_applies"] == w["history_applies"])
        ax, ap = ctx["rebuilt"][(w["clip_id"], "A_fresh")]
        bx, bp = ctx["rebuilt"][(w["clip_id"], "B_session")]
        ck(w["clip_id"] + "/comparison_inputs", row["x_identical"] == exact(ax, bx)
           and row["pad_identical"] == exact(ap, bp))
        ck(w["clip_id"] + "/comparison_logit_diff", row["max_abs_logit_diff"] == float(np.max(np.abs(
            np.asarray(w["A"]["logits93"]) - np.asarray(w["B"]["logits93"])))))
        rows.append({"transition": transition, "A_correct": ac, "B_correct": bc,
                     "history_applies": w["history_applies"]})
    summaries = {}
    for key, selected in (("summary_all", rows), ("summary_history_only", [r for r in rows if r["history_applies"]])):
        summary = {k: sum(r["transition"] == k for r in selected) for k in transitions}
        summary.update(denominator=len(selected), A_correct=sum(r["A_correct"] for r in selected),
                       B_correct=sum(r["B_correct"] for r in selected))
        ck("comparison_" + key, saved[key] == summary)
        summaries[key] = summary
    ck("history_denominators", summaries["summary_all"]["denominator"] == 4
       and summaries["summary_history_only"]["denominator"] == 3)
    return summaries


def forbidden_loaded():
    return [m for m in sys.modules if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)]


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
    own_sha = sha256_file(__file__)
    pre_pins = hash_pins(cfg)
    ck("pins_pre", not pre_pins["mismatches"], pre_pins["mismatches"] or None)
    pre_art = check_manifest(result_dir, ck)
    ck("no_forbidden_modules_start", not forbidden_loaded(), forbidden_loaded() or None)
    try:
        require_clean(ck, "provenance")
        report["main_provenance"] = check_run_provenance(cfg, a.config, result_dir, ck)
        require_clean(ck, "main source identity")
        ck("frozen_clock_and_score_contract", cfg["gap_us"] == GAP_US
           and cfg["scoring"]["cpu_threads"] == CPU_THREADS
           and cfg["scoring"]["max_abs_logit_tolerance"] == TOL
           and cfg["arms"] == list(ARMS))
        isr = import_pinned("_wis_slp_score_ref", cfg["independent_score_reference"], cfg["pins"])
        ctx = verify_structure(cfg, result_dir, ck, isr)
        require_clean(ck, "structure")
        rr = import_pinned("_wis_slp_raw_ref", cfg["independent_raw_reference"], cfg["pins"])
        contract = [p for p in cfg["pins"] if os.path.basename(p) == CONTRACT_BASENAME]
        ck("contract_pinned_once", len(contract) == 1, contract)
        require_clean(ck, "contract")
        rule = load_json(contract[0])[CONTRACT_RAW_RULE_KEY]
        n_ok, n_all, first_err = verify_raw(ctx, rr, rule, ck)
        ck("raw_frames_all", n_all == cfg["budgets"]["process_calls"] and n_ok == n_all,
           {"ok": n_ok, "all": n_all, "first_error": first_err})
        report["raw"] = {"verified": n_ok, "total": n_all}
        require_clean(ck, "raw diagnostics")
        report["inputs"] = verify_inputs(cfg, ctx, isr, ck, result_dir)
        report["comparison"] = verify_comparison(ctx, result_dir, ck)
        if a.score:
            report["scores"] = verify_scores(cfg, ctx, ck)
        else:
            report["scores"] = {"status": "NOT_RUN", "forwards": 0, "model_loads": 0}
    except Exception as e:
        ck("verifier_exception", False, f"{type(e).__name__}: {e}")
    ck("no_forbidden_modules_end", not forbidden_loaded(), forbidden_loaded() or None)
    if not a.score:
        ck("default_has_no_torch", "torch" not in sys.modules)
    post_pins = hash_pins(cfg)
    ck("pins_post_equal_pre", post_pins == pre_pins)
    post_art = artifact_hashes(result_dir)
    post_art.pop("RESULT_MANIFEST.json", None)
    ck("artifacts_unchanged", post_art == pre_art)
    ck("config_unchanged", sha256_file(a.config) == report["config_sha256"])
    ck("verifier_unchanged", sha256_file(__file__) == own_sha)
    report.update({"status": "PASS" if ck.failed == 0 else "FAIL", "n_failed": ck.failed,
                   "n_checks": ck.total, "checks": ck.items, "pins_post": post_pins,
                   "result_sha256": sha256_file(os.path.join(result_dir, "RESULT.json")),
                   "verifier_sha256": sha256_file(__file__), "mediapipe_calls": 0,
                   "camera_calls": 0, "video_decodes": 0, "training_calls": 0})
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps({"status": report["status"], "n_failed": ck.failed, "out": out}))
    return 0 if ck.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
