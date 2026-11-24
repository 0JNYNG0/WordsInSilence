"""Read-only independent input/reference audit; no torch/model/candidate import."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from verify_reference import HANDS, array_hash, four_arms, raw_hash

ARMS = ("original", "original_repair", "masked", "masked_repair")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm(path):
    return str(Path(path).resolve()).casefold()


def load_environment(cfg):
    pins = {norm(p): v for p, v in cfg["pins"].items()}
    for p, value in cfg["pins"].items():
        if sha(p) != value:
            raise ValueError(f"pin mismatch: {p}")
    sys.path[:0] = [cfg["helper_dir"], cfg["loader_dir"], cfg["pilot_src"]]
    R = importlib.import_module("run_audit")
    F = importlib.import_module("framerate_impl")
    data = importlib.import_module("pilot.data")
    web = importlib.import_module("pilot.web_input")
    for module in (R, R.TG, F, data, web):
        path = Path(module.__file__).resolve()
        if pins.get(norm(path)) != sha(path):
            raise ValueError(f"import path/hash not pinned: {path}")
    api = {"F": F, "build_channels": data.build_channels, "to_native_like": web.to_native_like,
           "native_roots": cfg["native_roots"], "mp_roots": cfg["mp_roots"]}
    rows = {"native": R.load_native_rows(cfg["native_metadata"]), "mediapipe": R.load_mp_rows(cfg["mp_manifest"])}
    return R, api, rows


def prepare_row(R, api, group, row, tolerance):
    if group == "native":
        n = api["F"].load_native(row["meta"], {R.NATIVE_ROLE}, set(R.NATIVE_SIGNERS), api["native_roots"])
        xy, pr, ts = n["xy"], n["present"], n["ts"]
        diff = None
    else:
        xy, pr, ts, saved, diff = R.load_mp(api, row, tolerance)
    clock = R.clock_report(ts)
    if not (clock["strictly_increasing"] and clock["nominal_30hz"]):
        raise ValueError("original nominal clock contract failed")
    before = raw_hash(xy, pr, ts)
    arms, injected = four_arms(xy, pr, ts, row["id"])
    features = {name: api["build_channels"](a["xy"], a["present"], a["ts"]) for name, a in arms.items()}
    for built in features.values():
        x = built["x"]
        assert x.dtype == np.float32 and x.shape == (len(ts), 280) and np.isfinite(x).all()
    assert raw_hash(xy, pr, ts) == before
    return {"arms": arms, "features": features, "injected": injected, "clock": clock, "mp_diff": diff}


def iter_prepared(cfg):
    """Root independent forward verification can reuse these reference inputs."""
    R, api, rows = load_environment(cfg)
    for group in ("native", "mediapipe"):
        for index, row in enumerate(rows[group]):
            yield group, index, row, prepare_row(R, api, group, row, float(cfg["tolerance"]))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "median": None, "p95": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.quantile(a, 0.95))}


def prepare_all(cfg):
    R, api, rows = load_environment(cfg)
    roots = {"native": cfg["native_roots"], "mediapipe": cfg["mp_roots"]}
    hashes_before = {g: R.hash_sources(rows[g], roots[g]) for g in rows}
    for g in rows:
        R.check_listed(rows[g], hashes_before[g], g)
    classes, native_scores, _ = R.load_saved_native(cfg["saved_native_logits"], rows["native"])
    mp_scores, _ = R.load_saved_mp(cfg["saved_mp_logits"], rows["mediapipe"], classes)
    results = {}
    subsets = {"all": list(range(21)), "wrist": [0], "palm": [1, 5, 9, 13, 17],
               "fingers": [j for j in range(21) if j not in (0, 1, 5, 9, 13, 17)]}
    for group in ("native", "mediapipe"):
        rr, pools = [], {method: {s: [] for s in subsets} for method in ("linear", "previous")}
        clipmeans = {method: [] for method in pools}
        for index, row in enumerate(rows[group]):
            prepared = prepare_row(R, api, group, row, float(cfg["tolerance"]))
            arms, features, injected = prepared["arms"], prepared["features"], prepared["injected"]
            base, rec = arms["original"], arms["masked_repair"]
            scale = float(features["original"]["body_scale"])
            assert np.isfinite(scale) and scale > 0
            events = []
            for i in range(len(base["ts"])):
                for side, hand in enumerate(HANDS):
                    if not injected[i, hand].all():
                        continue
                    truth = base["xy"][i, hand].astype(np.float64)
                    lin = np.linalg.norm(rec["xy"][i, hand].astype(np.float64) - truth, axis=1) / scale
                    previous = np.linalg.norm(base["xy"][i - 1, hand].astype(np.float64) - truth, axis=1) / scale
                    events.append({"frame": i, "side": side, "linear": lin.tolist(), "previous": previous.tolist()})
            for method in pools:
                if events:
                    a = np.array([e[method] for e in events])
                    clipmeans[method].append(float(a.mean()))
                    for s, indices in subsets.items():
                        pools[method][s].extend(a[:, indices].ravel().tolist())
            arm_meta = {}
            for name, arm in arms.items():
                estimated = arm.get("estimated", np.zeros_like(injected))
                artificial = estimated & injected
                natural = estimated & ~base["present"].astype(bool)
                assert not np.any(artificial & natural)
                locations = lambda mask: [[i, side] for i in range(len(mask)) for side, hand in enumerate(HANDS) if mask[i, hand].all()]
                arm_meta[name] = {"raw_hashes": raw_hash(arm["xy"], arm["present"], arm["ts"]),
                                  "feature_hash": array_hash(features[name]["x"]), "frames": len(arm["ts"]),
                                  "body_scale": float(features[name]["body_scale"]),
                                  "artificial_filled_locations": locations(artificial), "natural_filled_locations": locations(natural)}
            rr.append({"id": row["id"], "index": index, "label": row["label"], "signer_id": row["signer_id"],
                       "arms": arm_meta, "injected_locations": [[e["frame"], e["side"]] for e in events],
                       "coordinate_events": events, "original_clock": prepared["clock"], "mp_prepared_diff": prepared["mp_diff"]})
        results[group] = {"rows": rr, "row_count": len(rr), "injected_hand_events": sum(len(r["coordinate_events"]) for r in rr),
                          "coordinate": {method: {"points": {s: stats(v) for s, v in pool.items()}, "clip_means": stats(clipmeans[method])} for method, pool in pools.items()}}
    hashes_after = {g: R.hash_sources(rows[g], roots[g]) for g in rows}
    assert hashes_before == hashes_after
    for path, expected in cfg["pins"].items():
        assert sha(path) == expected
    return {"status": "INDEPENDENT_INPUT_REFERENCE_COMPLETE", "groups": results,
            "source_hashes": hashes_after, "pins_verified": cfg["pins"], "arms": list(ARMS), "classes": classes,
            "inference": 0, "model_imported": False, "candidate_imported": False, "input_mutations": 0,
            "reference_files": {str(Path(__file__).resolve()): sha(__file__), str(Path(__file__).with_name("verify_reference.py")): sha(Path(__file__).with_name("verify_reference.py"))},
            "limits": ["Endpoint channel identity is assumed, not verified.", "Coordinate targets are original detector estimates, not anatomical truth.", "Original body scale is used only by evaluator, not passed to restore."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfgpath = Path(args.config).resolve()
    cfgbytes = cfgpath.read_bytes()
    cfg = json.loads(cfgbytes.decode("utf-8-sig"))
    result = prepare_all(cfg)
    assert cfgpath.read_bytes() == cfgbytes
    result["config_sha256"] = hashlib.sha256(cfgbytes).hexdigest()
    output = Path(__file__).with_name("INDEPENDENT_INPUTS.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "rows": {g: v["row_count"] for g, v in result["groups"].items()}, "inference": 0}))


if __name__ == "__main__":
    main()
