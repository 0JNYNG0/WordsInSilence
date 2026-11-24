"""Hand observation x analysis span 2x2 diagnostic on saved NPZ observations (FROZEN_PLAN.md).

Default: stdlib-only preflight (config + 15 pins + source hashes). No data/model/runtime imports.
--run --receipt R: prepare BB,CC,BC,CB with the pinned official_prep, reuse stored diagonal logits
after exact input_record match, then exactly 2 cross forwards (BC then CB). Technical result only;
not candidate promotion, prior IMAGE rejection stands. Never retries.
"""
import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import sys
from pathlib import Path

RUNNER = Path(__file__).resolve()
HERE = RUNNER.parent
TEST = HERE / "test_span_factorial.py"
FROZEN = HERE / "FROZEN_PLAN.md"

SCHEMA = "wis_hand_mode_span_factorial_v1"
KEYS = {"schema", "is_example", "workspace_root", "pilot_src", "output_dir", "clip_id", "label",
        "previous_result", "previous_independent", "previous_config", "prep_module", "checkpoint",
        "checkpoint_sha256", "observations", "diagonals", "cross_conditions", "full_frames", "classes",
        "cpu_threads", "forwards", "independent_forwards", "new_extractions", "pins"}
FIXED = {"full_frames": 78, "classes": 93, "cpu_threads": 2, "forwards": 2, "independent_forwards": 2,
         "new_extractions": 0}
DIAGONALS = [{"id": "BB", "observation": "B", "span": [12, 74], "previous_arm": "B"},
             {"id": "CC", "observation": "C", "span": [12, 69], "previous_arm": "C"}]
CROSS = [{"id": "BC", "observation": "B", "span": [12, 69]},
         {"id": "CB", "observation": "C", "span": [12, 74]}]
PINNED_FIELDS = ("previous_result", "previous_independent", "previous_config", "prep_module", "checkpoint")
OBS_KEYS = ("obs_xy_px", "obs_usable", "obs_timestamp_us", "obs_frame_id", "obs_observed", "obs_conf",
            "obs_hand_score")
PERSISTED = {"B": "B_video", "C": "C_hand_image"}
PAYLOAD_KEYS = {"state_dict", "width", "dilations", "kernel_size", "dropout", "n_classes", "classes"}
# ponytail: MaskedSignTCN module path is not in API_REFERENCE; import must resolve under pilot_src or it fails.
MODEL_MODULE, MODEL_CLASS = "pilot.model", "MaskedSignTCN"
PATTERNS = {
    (True, False): "observation_bundle: B observation correct, C observation wrong at both spans",
    (False, True): "span_renormalization_bundle: long span correct, short span wrong for both observations",
    (True, True): "interaction: failure only in C-observation x C-short-span combination",
    (False, False): "B_long_only: only B-observation x B-long-span is correct",
}
CAVEAT = ("Conditional observation of one source clip only; not new-person generalization, not physical "
          "hand-coordinate truth, not proof of a tail-5-frame-only effect; contrasts are not independent "
          "causal values. true_margin uses each cell's own best rival, which may change between cells.")

CALLS = {"npz_load": 0, "official_prep": 0, "input_record": 0, "torch_load": 0, "forward": 0}
STAGE = {"name": "init"}


class TechnicalFailure(Exception):
    pass


def req(cond, msg):
    if not cond:
        raise TechnicalFailure(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(a):
    import numpy as np
    a = np.asarray(a)
    head = f"{a.dtype.str}|{'x'.join(str(d) for d in a.shape)}|".encode()
    return hashlib.sha256(head + np.ascontiguousarray(a).tobytes()).hexdigest()


def resolve(cfg, p):
    p = Path(p)
    return p if p.is_absolute() else Path(cfg["workspace_root"]) / p


def _is_sha(v):
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v) and set(v) != {"0"}


def validate_config(cfg):
    req(isinstance(cfg, dict) and set(cfg) == KEYS, "config keys differ from frozen schema")
    req(cfg["schema"] == SCHEMA and cfg["is_example"] is False, "example/wrong-schema config rejected")
    for k, v in FIXED.items():
        req(cfg[k] == v, f"config {k} != {v}")
    req(cfg["diagonals"] == DIAGONALS and cfg["cross_conditions"] == CROSS, "conditions differ from frozen plan")
    req(cfg["clip_id"] not in ("", "PUBLIC_CLIP") and cfg["label"] not in ("", "WORD"), "placeholder clip/label")
    paths = [cfg[k] for k in ("workspace_root", "pilot_src", "output_dir") + PINNED_FIELDS]
    paths += [cfg["observations"].get(k) for k in ("B", "C")]
    req(set(cfg["observations"]) == {"B", "C"}, "observations must be exactly B,C")
    pins = cfg["pins"]
    req(isinstance(pins, dict) and len(pins) == 15, "exactly 15 pins required")
    for p in paths + list(pins):
        req(isinstance(p, str) and p and "<" not in p and ">" not in p, f"placeholder path {p!r}")
    req(all(_is_sha(v) for v in pins.values()), "pin hashes must be real lowercase sha256")
    for p in [cfg[k] for k in PINNED_FIELDS] + [cfg["observations"]["B"], cfg["observations"]["C"]]:
        req(p in pins, f"{p} not pinned")
    req(_is_sha(cfg["checkpoint_sha256"]) and pins[cfg["checkpoint"]] == cfg["checkpoint_sha256"],
        "checkpoint_sha256 must equal its pin")


def check_receipt(path):
    r = json.loads(Path(path).read_text("utf-8"))
    req(r.get("status") == "PASS", "receipt status not PASS")
    req(r.get("runner_sha256") == sha256_file(RUNNER), "receipt runner_sha256 mismatch")
    req(r.get("test_sha256") == sha256_file(TEST), "receipt test_sha256 mismatch")
    t = r.get("tests_run")
    req(isinstance(t, int) and not isinstance(t, bool) and t > 0, "receipt tests_run must be > 0")


def hash_all(cfg, cfg_path, receipt):
    out = {f"pin:{p}": sha256_file(resolve(cfg, p)) for p in cfg["pins"]}
    out.update(config=sha256_file(cfg_path), runner=sha256_file(RUNNER), test=sha256_file(TEST),
               frozen=sha256_file(FROZEN))
    if receipt:
        out["receipt"] = sha256_file(receipt)
    return out


def preflight(cfg_path, receipt=None):
    cfg = json.loads(Path(cfg_path).read_text("utf-8"))
    validate_config(cfg)
    if receipt:
        check_receipt(receipt)
    hashes = hash_all(cfg, cfg_path, receipt)
    for p, want in cfg["pins"].items():
        req(hashes[f"pin:{p}"] == want, f"pin mismatch {p}")
    return cfg, hashes


def _has_value(obj, v):
    if isinstance(obj, dict):
        return any(_has_value(x, v) for x in obj.values())
    if isinstance(obj, list):
        return any(_has_value(x, v) for x in obj)
    return isinstance(obj, str) and obj.lower() == v


def check_previous(cfg, prev, indep, prev_cfg):
    rows = [r for r in prev["rows"] if r.get("clip_id") == cfg["clip_id"]]
    req(len(rows) == 1, f"expected exactly one previous row for clip_id, got {len(rows)}")
    row = rows[0]
    req(row["label"] == cfg["label"], "previous row label != config label")
    req(row["C"]["status"] == "AVAILABLE", "previous C status not AVAILABLE")
    classes = prev["classes"]
    req(isinstance(classes, list) and len(classes) == len(set(classes)) == 93, "previous classes not 93 unique")
    req(cfg["label"] in classes, "label not in classes")
    for arm in ("B", "C"):
        a = row[arm]
        req(isinstance(a["input"], dict) and isinstance(a["score"], dict) and "top1" in a["score"], f"{arm} layout")
        req(len(a["logits"]) == 93 and all(math.isfinite(float(v)) for v in a["logits"]), f"{arm} logits not 93 finite")
    d = prev["decision"]
    req(d["denominator"] == 18 and d["B"] == 18 and d["C"] == 17, "previous decision not 18/B18/C17")
    req(indep["status"] == "PASS" and indep["max_abs_logit_difference"] == 0.0, "previous independent not exact PASS")
    for diag in DIAGONALS:
        req(row[diag["previous_arm"]]["input"].get("span") == diag["span"], f"previous {diag['id']} span differs")
    for o, key in PERSISTED.items():
        ent, npz = row["persisted"][key], resolve(cfg, cfg["observations"][o])
        req(ent["npz"] == npz.name, f"persisted {key} npz filename mismatch")
        req(any(Path(p).name == ent["meta"] and resolve(cfg, p).parent == npz.parent for p in cfg["pins"]),
            f"persisted {key} meta not pinned beside npz")
    req(_has_value(prev_cfg, cfg["checkpoint_sha256"]) or _has_value(prev, cfg["checkpoint_sha256"]),
        "previous context does not reference this checkpoint sha256")
    return row


def load_prep(path):
    spec = importlib.util.spec_from_file_location("wis_span_factorial_prep", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    req(Path(mod.__file__).resolve() == Path(path).resolve(), "prep module path mismatch")
    req(callable(getattr(mod, "official_prep", None)) and callable(getattr(mod, "input_record", None)),
        "prep module lacks official_prep/input_record")
    return mod


def check_forbidden():
    bad = [m for m in sys.modules if m.split(".")[0] in ("av", "mediapipe") or "camera" in m.lower()]
    req(not bad, f"forbidden modules imported: {bad[:5]}")


def import_from(src, name):
    mod = importlib.import_module(name)
    req(Path(mod.__file__).resolve().is_relative_to(src), f"{name} resolved outside pilot_src: {mod.__file__}")
    check_forbidden()
    return mod


def load_obs(path):
    import numpy as np
    CALLS["npz_load"] += 1
    with np.load(path, allow_pickle=False) as z:
        req(all(k in z.files for k in OBS_KEYS), f"{path} missing obs_ keys")
        obs = {k: z[k] for k in OBS_KEYS}  # NpzFile is lazy: selected_frame_id/decoded_* never read
    for a in obs.values():
        a.setflags(write=False)
    return obs


def obs_hashes(obs):
    return {k: array_sha256(v) for k, v in obs.items()}


def check_pair(b, c, full_frames):
    import numpy as np
    for o in (b, c):
        req(all(len(a) == full_frames for a in o.values()), f"observation rows != {full_frames}")
    req(np.array_equal(b["obs_frame_id"], c["obs_frame_id"]), "B/C frame_id differ")
    req(np.array_equal(b["obs_timestamp_us"], c["obs_timestamp_us"]), "B/C timestamp_us differ")
    req(np.array_equal(b["obs_xy_px"][:, :33], c["obs_xy_px"][:, :33], equal_nan=True), "B/C pose first33 differ")


def prepare(prep, obs, span, pilot_fns, full_frames):
    lo, hi = int(span[0]), int(span[1])
    req(0 <= lo <= hi < full_frames, f"span out of bounds {span}")
    full = {"xy_px": obs["obs_xy_px"], "usable": obs["obs_usable"], "timestamp_us": obs["obs_timestamp_us"],
            "frame_id": obs["obs_frame_id"]}
    fns = dict(pilot_fns, active_span=lambda present: (lo, hi))  # explicit diagnostic crop, no retrim
    CALLS["official_prep"] += 1
    p = prep.official_prep(full, fns)
    req(p is not None and p["span"] == [lo, hi], "official_prep did not apply explicit span")
    req(p["x"].shape[1] == hi - lo + 1, "x rows != inclusive span length")
    req(p["frame_id"] == obs["obs_frame_id"][lo:hi + 1].tolist(), "frame_id not exact span rows")
    req(p["timestamp_us"] == obs["obs_timestamp_us"][lo:hi + 1].tolist(), "timestamps not exact span rows")
    return p


def record(prep, p):
    CALLS["input_record"] += 1
    return json.loads(json.dumps(prep.input_record(p, array_sha256), allow_nan=False))


def top1(logits, classes):
    req(len(logits) == len(classes) == 93 and all(math.isfinite(float(v)) for v in logits), "need 93 finite logits")
    return classes[max(range(93), key=lambda i: float(logits[i]))]


def score(logits, classes, truth, rival):
    l = [float(v) for v in logits]
    t, r = classes.index(truth), classes.index(rival)
    top = classes.index(top1(l, classes))
    others = [i for i in range(93) if i != t]
    best = max(others, key=l.__getitem__)
    return {"top1": classes[top], "correct": top == t, "true_rank": 1 + sum(v > l[t] for v in l),
            "true_logit": l[t], "best_rival": classes[best], "true_margin": l[t] - l[best],
            "fixed_rival": rival, "fixed_rival_gap": l[t] - l[r]}


def collect_logits(order, stored, forward_fn):
    reused = {cid: cid in stored for cid in order}
    req([c for c in order if not reused[c]] == ["BC", "CB"], "only BC,CB may be forwarded")
    logits = {cid: stored[cid] if reused[cid] else forward_fn(cid) for cid in order}
    return logits, reused


def interpret(bc_correct, cb_correct):
    return PATTERNS[(bool(bc_correct), bool(cb_correct))]


def contrasts(s):
    out = {}
    for m in ("true_margin", "fixed_rival_gap"):
        g = {k: s[k][m] for k in ("BB", "CC", "BC", "CB")}
        out[m] = {"span_on_B_obs_BC_minus_BB": g["BC"] - g["BB"], "span_on_C_obs_CC_minus_CB": g["CC"] - g["CB"],
                  "obs_on_long_span_CB_minus_BB": g["CB"] - g["BB"],
                  "obs_on_short_span_CC_minus_BC": g["CC"] - g["BC"],
                  "interaction_CC_minus_BC_minus_CB_plus_BB": g["CC"] - g["BC"] - g["CB"] + g["BB"]}
    return out


def state_sha(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode() + b"\0" + array_sha256(sd[k].detach().cpu().numpy()).encode())
    return h.hexdigest()


def load_model(cfg, src, classes):
    import torch
    torch.set_num_threads(cfg["cpu_threads"])
    req(torch.get_num_threads() == cfg["cpu_threads"], "cpu threads not applied")
    CALLS["torch_load"] += 1
    payload = torch.load(str(resolve(cfg, cfg["checkpoint"])), map_location="cpu", weights_only=True)
    req(PAYLOAD_KEYS <= set(payload), "checkpoint payload keys missing")
    ck_classes = list(payload["classes"])
    req(len(ck_classes) == len(set(ck_classes)) == payload["n_classes"] == 93, "Class contract")
    req(ck_classes == classes, "checkpoint classes != previous result classes")
    cls = getattr(import_from(src, MODEL_MODULE), MODEL_CLASS)
    model = cls(width=payload["width"], dilations=tuple(payload["dilations"]), kernel_size=payload["kernel_size"],
                dropout=payload["dropout"], n_classes=93)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    req(not model.training, "model not in eval")
    info = {"payload_state_sha": state_sha(payload["state_dict"]), "state_before": state_sha(model.state_dict()),
            "extra_payload_keys": sorted(set(payload) - PAYLOAD_KEYS)}
    req(info["state_before"] == info["payload_state_sha"], "model state != payload state")
    return torch, model, info


def forward(torch, model, p):
    import numpy as np
    CALLS["forward"] += 1
    with torch.inference_mode():
        out = model(torch.from_numpy(p["x"]), torch.from_numpy(p["pad"]))[0].detach().cpu().numpy().astype(np.float64)
    req(out.shape == (93,) and bool(np.isfinite(out).all()), "forward output not 93 finite")
    return out.tolist()


def run(cfg):
    import numpy as np
    r = lambda p: resolve(cfg, p)
    STAGE["name"] = "previous"
    prev = json.loads(r(cfg["previous_result"]).read_text("utf-8"))
    indep = json.loads(r(cfg["previous_independent"]).read_text("utf-8"))
    prev_cfg = json.loads(r(cfg["previous_config"]).read_text("utf-8"))
    row = check_previous(cfg, prev, indep, prev_cfg)
    classes = list(prev["classes"])

    STAGE["name"] = "imports"
    prep = load_prep(r(cfg["prep_module"]))
    src = r(cfg["pilot_src"]).resolve()
    req(not any(m == "pilot" or m.startswith("pilot.") for m in sys.modules), "pilot imported before path check")
    sys.path.insert(0, str(src))
    web_input, data = import_from(src, "pilot.web_input"), import_from(src, "pilot.data")
    pilot_fns = {"to_native_like": web_input.to_native_like, "build_channels": data.build_channels,
                 "collate": data.collate, "n_channels": data.TOTAL_CHANNELS}

    STAGE["name"] = "observations"
    obs = {k: load_obs(r(cfg["observations"][k])) for k in ("B", "C")}
    obs_before = {k: obs_hashes(v) for k, v in obs.items()}
    check_pair(obs["B"], obs["C"], cfg["full_frames"])

    STAGE["name"] = "prepare"
    conds = cfg["diagonals"] + cfg["cross_conditions"]
    preps = {c["id"]: prepare(prep, obs[c["observation"]], c["span"], pilot_fns, cfg["full_frames"]) for c in conds}
    recs = {cid: record(prep, p) for cid, p in preps.items()}

    STAGE["name"] = "diagonal_compare"  # all before any model import
    stored = {}
    for d in cfg["diagonals"]:
        arm = row[d["previous_arm"]]
        req(recs[d["id"]] == arm["input"], f"{d['id']} input_record != previous {d['previous_arm']} input")
        req(top1(arm["logits"], classes) == arm["score"]["top1"], f"{d['id']} stored logits top1 != saved score")
        stored[d["id"]] = [float(v) for v in arm["logits"]]
    req({k: obs_hashes(v) for k, v in obs.items()} == obs_before, "raw arrays changed during prepare")

    STAGE["name"] = "model"
    torch, model, model_info = load_model(cfg, src, classes)
    logits, reused = collect_logits([c["id"] for c in conds], stored, lambda cid: forward(torch, model, preps[cid]))
    model_info["state_after"] = state_sha(model.state_dict())
    req(model_info["state_after"] == model_info["state_before"], "model state changed")
    check_forbidden()

    STAGE["name"] = "score"  # truth used only from here on
    truth, rival = cfg["label"], row["C"]["score"]["top1"]
    req(rival != truth and rival in classes, "fixed rival must differ from truth")
    scores = {cid: score(logits[cid], classes, truth, rival) for cid in logits}
    compared = {}
    for d in cfg["diagonals"]:
        ps = row[d["previous_arm"]]["score"]
        common = sorted(set(ps) & set(scores[d["id"]]))
        req(all(ps[k] == scores[d["id"]][k] for k in common), f"{d['id']} saved score differs on {common}")
        compared[d["id"]] = common
    req(scores["BB"]["correct"] and not scores["CC"]["correct"], "BB correct / CC wrong not reproduced")
    req({k: obs_hashes(v) for k, v in obs.items()} == obs_before, "raw arrays changed")

    bb_hi, cc_hi = DIAGONALS[0]["span"][1], DIAGONALS[1]["span"][1]
    tail = list(range(cc_hi + 1, bb_hi + 1))
    c = obs["C"]
    longtail = {"rows": tail, "frame_id": c["obs_frame_id"][tail].tolist(),
                "timestamp_us": c["obs_timestamp_us"][tail].tolist(),
                "usable_hand_true_count": [int(np.count_nonzero(c["obs_usable"][i][33:])) for i in tail],
                "included_in_CB": set(c["obs_frame_id"][tail].tolist()) <= set(recs["CB"]["frame_id"]),
                "raw_unchanged": True}
    cells = [{"id": cd["id"], "observation": cd["observation"], "span": cd["span"], "input": recs[cd["id"]],
              "pad_all_one": bool(np.all(preps[cd["id"]]["pad"] == 1)), "logits93": logits[cd["id"]],
              "score": scores[cd["id"]], "reused": reused[cd["id"]]} for cd in conds]
    return {"classes": classes, "truth": truth, "fixed_rival": rival, "cells": cells,
            "pattern": interpret(scores["BC"]["correct"], scores["CB"]["correct"]),
            "contrasts": contrasts(scores), "caveat": CAVEAT, "c_longtail": longtail,
            "previous_score_keys_compared": compared, "obs_hashes": obs_before, "model": model_info,
            "import_files": {m.__name__: {"path": str(Path(m.__file__).resolve()), "sha256": sha256_file(m.__file__),
                                          "pinned": any(r(p).resolve() == Path(m.__file__).resolve() for p in cfg["pins"])}
                             for m in (prep, web_input, data, sys.modules[MODEL_MODULE])}}


def write_json(path, obj):
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--receipt")
    a = ap.parse_args(argv)
    try:
        req(not a.run or a.receipt, "--run requires --receipt")
        cfg, before = preflight(a.config, a.receipt if a.run else None)
        out = resolve(cfg, cfg["output_dir"])
        req(not out.exists() or not any(out.iterdir()), "output_dir must be fresh/empty")
    except Exception as e:  # noqa: BLE001 - preflight writes nothing
        print(json.dumps({"status": "PREFLIGHT_FAIL", "error": f"{type(e).__name__}: {e}"}))
        return 2
    if not a.run:
        print(json.dumps({"status": "PREFLIGHT_PASS", "hashes": before}, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    base = {"schema": SCHEMA, "config_path": str(Path(a.config).resolve()), "hashes_before": before}
    try:
        res = run(cfg)
        STAGE["name"] = "after_hash"
        after = hash_all(cfg, a.config, a.receipt)
        req(after == before, "pinned/config/source/receipt hashes changed")
        write_json(out / "RESULT.json", dict(base, status="TECHNICAL_PASS_DIAGNOSTIC_ONLY",
                                             note="diagnostic reproducibility only; no candidate promotion; "
                                                  "prior IMAGE rejection unchanged",
                                             hashes_after=after, calls=CALLS, **res))
        return 0
    except Exception as e:  # noqa: BLE001 - any failure is technical; preserved, never retried
        write_json(out / "FAILURE.json", dict(base, status="TECHNICAL_FAIL", stage=STAGE["name"],
                                              error=f"{type(e).__name__}: {e}", calls=CALLS))
        return 1


if __name__ == "__main__":
    sys.exit(main())
