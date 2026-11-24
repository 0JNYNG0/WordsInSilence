"""Independent stored-result/metric audit. No model execution or candidate imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from verify_reference import HANDS, array_hash

ARMS = ["original", "original_repair", "masked", "masked_repair"]
PAIRS = [("original", "original_repair"), ("masked", "masked_repair"), ("original", "masked"), ("original", "masked_repair")]
SIDES = ["left", "right"]
SUBSETS = {"all": list(range(21)), "wrist": [0], "palm": [1, 5, 9, 13, 17], "fingers": [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combined(h):
    return hashlib.sha256("|".join(h[k] for k in ("xy", "present", "ts")).encode()).hexdigest()


def locations(events):
    return sorted((int(e["frame"]), SIDES.index(e["side"]) if isinstance(e["side"], str) else int(e["side"])) for e in events)


def numerical(actual, expected, name, atol=1e-9):
    if not np.allclose(actual, expected, rtol=1e-10, atol=atol, equal_nan=False):
        raise AssertionError(f"numeric mismatch {name}: {actual!r} vs {expected!r}")


def coord_summary(rows):
    out = {"available": any(r["coordinate_events"] for r in rows),
           "events": sum(len(r["coordinate_events"]) for r in rows),
           "clips_with_events": sum(bool(r["coordinate_events"]) for r in rows)}
    for method in ("linear", "previous"):
        out[method] = {}
        for name, idx in SUBSETS.items():
            perclip = [np.asarray([e[method] for e in r["coordinate_events"]], dtype=np.float64)[:, idx].reshape(-1) for r in rows if r["coordinate_events"]]
            if not perclip:
                out[method][name] = None
                continue
            pts = np.concatenate(perclip)
            means = np.array([x.mean() for x in perclip])
            out[method][name] = {"points": len(pts), "point_mean": float(pts.mean()), "point_median": float(np.median(pts)),
                                 "point_p95": float(np.quantile(pts, 0.95)), "clips": len(means),
                                 "clip_mean_mean": float(means.mean()), "clip_mean_median": float(np.median(means))}
    return out


def compare_numeric_tree(actual, expected, name):
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare_numeric_tree(actual[key], value, name + "/" + key)
    elif expected is None or isinstance(expected, bool):
        assert actual == expected, name
    else:
        numerical(actual, expected, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfgpath = Path(args.config).resolve()
    cfg = read(cfgpath)
    unit = Path(__file__).resolve().parent
    output = Path(cfg["output_dir"])
    result = read(output / "RESULT.json")
    preflight = read(output / "PREFLIGHT.json")
    independent = read(unit / "INDEPENDENT_INPUTS.json")
    assert result["arm_order"] == ARMS and independent["arms"] == ARMS
    classes = independent["classes"]
    checked_rows = checked_scores = 0
    max_event_error_diff = 0.0
    fallback_reports = []
    env = None

    def alternate_reference(group, index, reference, candidate_input):
        nonlocal env
        # The first independent reference uses a+w*(b-a). This second algebraic
        # form allows a transparent float-rounding audit, never a rule change.
        from verify_inputs import load_environment, prepare_row
        if env is None:
            env = load_environment(cfg)
        R, api, metadata = env
        row = metadata[group][index]
        prepared = prepare_row(R, api, group, row, float(cfg["tolerance"]))
        raw = prepared["arms"]
        report = {"group": group, "id": row["id"], "max_coordinate_difference": 0.0, "max_feature_difference": 0.0, "arms": []}
        for name in ARMS:
            a = raw[name]
            xy, pr, ts = a["xy"].copy(), a["present"].copy(), a["ts"].copy()
            if name in ("original_repair", "masked_repair"):
                received = raw["original" if name == "original_repair" else "masked"]
                for frame, side in reference["arms"][name]["artificial_filled_locations"] + reference["arms"][name]["natural_filled_locations"]:
                    sl = HANDS[side]
                    left, right = received["xy"][frame - 1, sl].astype(np.float64), received["xy"][frame + 1, sl].astype(np.float64)
                    t = received["ts"]
                    weight = (int(t[frame]) - int(t[frame - 1])) / (int(t[frame + 1]) - int(t[frame - 1]))
                    xy[frame, sl] = ((1.0 - weight) * left + weight * right).astype(xy.dtype)
            built = api["build_channels"](xy, pr, ts)["x"]
            if group == "mediapipe" and name == "original":
                with np.load(row["files"]["input"][0], allow_pickle=False) as z:
                    built = z["x"].astype(np.float32)
            rh = hashlib.sha256("|".join(array_hash(v) for v in (xy, pr, ts)).encode()).hexdigest()
            assert rh == candidate_input["raw_sha256"][name], (row["id"], name, "alternate raw")
            assert array_hash(built) == candidate_input["input_sha256"][name], (row["id"], name, "alternate features")
            valid = np.isfinite(xy) & np.isfinite(a["xy"])
            delta = float(np.max(np.abs(xy[valid].astype(np.float64) - a["xy"][valid].astype(np.float64)), initial=0.0))
            fdelta = float(np.max(np.abs(built.astype(np.float64) - prepared["features"][name]["x"].astype(np.float64)), initial=0.0))
            report["max_coordinate_difference"] = max(report["max_coordinate_difference"], delta)
            report["max_feature_difference"] = max(report["max_feature_difference"], fdelta)
            report["arms"].append({"arm": name, "raw_hash_matched": True, "feature_hash_matched": True})
        return report

    groups_out = {}
    for group in ("native", "mediapipe"):
        actual = result["groups"][group]
        references = independent["groups"][group]["rows"]
        rr = actual["rows"]
        pp = preflight["rows"][group]
        n = len(references)
        assert n == len(rr) == len(pp)
        with np.load(output / f"SCORES_{group}.npz", allow_pickle=False) as z:
            assert z["arms"].tolist() == ARMS and z["classes"].tolist() == classes
            assert z["ids"].tolist() == [r["id"] for r in references]
            truth = np.array([classes.index(r["label"]) for r in references])
            assert np.array_equal(z["true_indices"], truth)
            logits = z["logits"].copy()
        assert logits.shape == (n, 4, 93) and np.isfinite(logits).all()
        if group == "native":
            with np.load(cfg["saved_native_logits"], allow_pickle=False) as z:
                assert z["ids"].tolist() == [r["id"] for r in references]
                assert z["classes"].tolist() == classes
                saved = z["original_logits"]
        else:
            with np.load(cfg["saved_mp_logits"], allow_pickle=False) as z:
                assert z["sample_ids"].tolist() == [r["id"] for r in references]
                assert z["classes"].tolist() == classes
                saved = z["logits"][:, z["arms"].tolist().index("mediapipe")]
        assert np.array_equal(logits[:, 0], saved)
        preds = np.argmax(logits, axis=2)
        correct = preds == truth[:, None]
        for i, (r, ref, pf) in enumerate(zip(rr, references, pp)):
            assert r["id"] == ref["id"] == pf["id"] and r["index"] == i
            assert r["true"] == ref["label"] and r["signer_id"] == ref["signer_id"]
            inp = r["input"]
            for key in inp:
                assert inp[key] == pf[key], (r["id"], key, "preflight differs")
            expected_locations = sorted(tuple(x) for x in ref["injected_locations"])
            assert locations(inp["mask"]["events"]) == expected_locations
            assert locations(r["coordinate_events"]) == expected_locations
            assert inp["mask"]["events_n"] == len(expected_locations)
            natural = ref["arms"]["original_repair"]["natural_filled_locations"]
            assert locations(inp["repairs"]["original_repair"]["fills"]) == sorted(tuple(x) for x in natural)
            assert inp["repairs"]["masked_repair"]["artificial_events"] == len(expected_locations)
            assert inp["repairs"]["masked_repair"]["natural_events"] == len(natural)
            assert inp["repairs"]["masked_repair"]["artificial_slots"] == len(expected_locations) * 21
            assert inp["repairs"]["masked_repair"]["natural_slots"] == len(natural) * 21
            mismatch = any(inp["raw_sha256"][a] != combined(ref["arms"][a]["raw_hashes"]) or inp["input_sha256"][a] != ref["arms"][a]["feature_hash"] for a in ARMS)
            if mismatch:
                fallback_reports.append(alternate_reference(group, i, ref, inp))
            ev_ref = {(e["frame"], e["side"]): e for e in ref["coordinate_events"]}
            for event in r["coordinate_events"]:
                e = ev_ref[(event["frame"], SIDES.index(event["side"]))]
                for method in ("linear", "previous"):
                    delta = float(np.max(np.abs(np.asarray(event[method]) - np.asarray(e[method]))))
                    max_event_error_diff = max(max_event_error_diff, delta)
                    numerical(event[method], e[method], r["id"] + method, atol=1e-6)
            for ai, arm in enumerate(ARMS):
                order = np.argsort(-logits[i, ai], kind="stable")
                yi = int(truth[i])
                rival = next(int(k) for k in order if k != yi)
                observed = r["arms"][arm]
                assert observed["pred"] == classes[int(preds[i, ai])]
                assert observed["correct"] == bool(correct[i, ai])
                assert observed["rank"] == int(np.flatnonzero(order == yi)[0]) + 1
                numerical(observed["margin"], logits[i, ai, yi] - logits[i, ai, rival], "margin")
                checked_scores += 93
            checked_rows += 1
        totals = {a: int(correct[:, ai].sum()) for ai, a in enumerate(ARMS)}
        assert actual["totals"] == totals
        per_signer, per_class = {}, {}
        for key, by, field in (("per_signer", per_signer, "signer_id"), ("per_class", per_class, "label")):
            for value in sorted({str(r[field]) for r in references}):
                indices = [i for i, r in enumerate(references) if str(r[field]) == value]
                by[value] = {"n": len(indices), **{a: int(correct[indices, ai].sum()) for ai, a in enumerate(ARMS)}}
            assert actual[key] == by
        paired = {}
        for source, target in PAIRS:
            si, ti = ARMS.index(source), ARMS.index(target)
            gains = [r["id"] for i, r in enumerate(references) if not correct[i, si] and correct[i, ti]]
            losses = [r["id"] for i, r in enumerate(references) if correct[i, si] and not correct[i, ti]]
            key = source + "->" + target
            a = actual["comparisons"][key]
            assert a["gain_ids"] == gains and a["loss_ids"] == losses
            assert a["gains_n"] == len(gains) and a["losses_n"] == len(losses)
            assert [x["id"] for x in a["gains"]] == gains and [x["id"] for x in a["losses"]] == losses
            assert a["net"] == len(gains) - len(losses) == totals[target] - totals[source]
            assert a["top1_changed"] == int((preds[:, si] != preds[:, ti]).sum())
            paired[key] = {"gains": gains, "losses": losses, "net": len(gains) - len(losses)}
        coord = coord_summary(rr)
        compare_numeric_tree(actual["coordinate"], coord, group + "/coordinate")
        masked = paired["masked->masked_repair"]
        clean = paired["original->original_repair"]
        gates = {"coordinate_available": coord["available"],
                 "coordinate_linear_mean_lt_previous": coord["available"] and coord["linear"]["all"]["point_mean"] < coord["previous"]["all"]["point_mean"],
                 "recognition_masked_repair_vs_masked": masked["net"] > 0 if group == "native" else masked["net"] >= 0,
                 "every_signer_nondecrease": all(x["masked_repair"] >= x["masked"] for x in per_signer.values()),
                 "clean_original_repair_no_newly_wrong": not clean["losses"],
                 "strict_no_newly_wrong_original_and_masked_pairs": not clean["losses"] and not masked["losses"]}
        for key, value in gates.items():
            assert result["decision"]["gates"][group][key] == value
        groups_out[group] = {"totals": totals, "per_signer": per_signer, "comparisons": paired, "gates": gates, "coordinate": coord}
    diag_keys = ["coordinate_linear_mean_lt_previous", "recognition_masked_repair_vs_masked", "every_signer_nondecrease", "clean_original_repair_no_newly_wrong"]
    effect = all(g["gates"][key] for g in groups_out.values() for key in diag_keys)
    strict = all(g["gates"]["strict_no_newly_wrong_original_and_masked_pairs"] for g in groups_out.values())
    assert result["decision"]["diagnostic_gate_pass"] == (effect and result["decision"]["contracts_all_valid"])
    assert result["decision"]["strict_no_newwrong_all_pairs"] == strict
    for path, expected in cfg["pins"].items():
        assert sha(path) == expected
    verification = {"status": "PASS_STORED_INPUT_AND_RESULT_CHECK_NO_NEW_INFERENCE", "rows_checked": checked_rows,
                    "scalar_scores_checked": checked_scores, "groups": groups_out, "net_effect_gates": effect, "strict_no_newwrong": strict,
                    "hash_differences_explained_by_float_algebra_or_prepared_original": fallback_reports,
                    "max_coordinate_event_reference_difference": max_event_error_diff,
                    "candidate_imports": 0, "model_forwards": 0,
                    "limits": ["Model forward is verified separately by root.", "Reported model state/contract checks require separate execution evidence; this script independently checks inputs, stored scores and scientific gates.", "Coordinate event errors allow 1e-6 normalized numerical difference, with exact alternate-form hashes required for input mismatches."],
                    "hashes": {"RESULT": sha(output / "RESULT.json"), "PREFLIGHT": sha(output / "PREFLIGHT.json"), "INDEPENDENT_INPUTS": sha(unit / "INDEPENDENT_INPUTS.json"), "verifier": sha(__file__)}}
    with (unit / "INDEPENDENT_RESULT_CHECK.json").open("x", encoding="utf-8") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": verification["status"], "rows": checked_rows, "effect": effect, "strict": strict, "rounding_rows": len(fallback_reports)}))


if __name__ == "__main__":
    main()
