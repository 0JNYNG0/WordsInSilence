"""Stage 1 evaluation of BOTH frozen arms after training.json is frozen. No result-driven change.

Groups: annotation_dev original 699 and decimated 699 (stride 2 offset 0, stage0 routing), whole_dev
74, public 18/5/9, MP 7/7 (prior local_io.evaluation_inputs routing). No KSL77, user, reserved or
holdout material. Control, candidate and 83212868 share every non-head tensor with the parent
(checked bitwise), so pooled features are computed once through the frozen parent prefix ->
blocks.3 -> masked mean, and each head is applied as a float64 affine map (spot-checked against a
per-model predict_prefix). Before any comparison is reported, the recomputed 83212868 top-1 must
equal the stored evaluation.json control rows on every original row and the stage0 decimated
logits on all 699 decimated rows; otherwise EVALUATION_STOPPED_BASELINE_NOT_REPRODUCED.json.

The project reference (current832) rows are the STORED ones; the recomputed 832 is verification only.
Gates are applied independently vs the matched control and vs current832.

Run: py -3.13 ops/framerate_pair_20260927/evaluate_stage1.py
"""
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("fr_prepare_stage1", _here / "prepare_stage1.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["fr_prepare_stage1"] = S
_spec.loader.exec_module(S)

R = S.RUN
TRAIN_STATUS = "TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION"
OUT_JSON, OUT_ARR = R / "evaluation.json", R / "evaluation_arrays.npz"
FAILED, STOPPED = R / "EVALUATION_FAILED.json", R / "EVALUATION_STOPPED_BASELINE_NOT_REPRODUCED.json"
MODELS = ("current832", "control", "candidate")
DEC_GROUP, DEC_COND = "annotation_dev_decimated", "decimated_stride2_offset0"
APOLOGY = "NIA_SL_SEN0356_"
SIGNERS = ("11", "12", "13")
SCOPES = ("full93", "project18")


def score_all(score, yi, target, classes):
    """Per-scope top-1 like the prior evaluator; score None = input failed (counted wrong)."""
    out = {}
    for scope, inds in (("full93", list(range(len(classes)))), ("project18", target)):
        if yi not in inds:
            continue
        if score is None:
            out[scope] = {"pred": None, "correct": False, "rank": None, "margin": None, "input_failed": True}
            continue
        order = sorted(inds, key=lambda j: -float(score[j]))
        rival = next(j for j in order if j != yi)
        out[scope] = {"pred": classes[order[0]], "correct": bool(order[0] == yi),
                      "rank": order.index(yi) + 1, "margin": float(score[yi] - score[rival])}
    return out


def summarize(rows, m, scope):
    rs = [r for r in rows if scope in r["models"][m]]
    labels = sorted({r["true"] for r in rs})
    detail, recalls, f1s = {}, [], []
    for label in labels:
        support = sum(r["true"] == label for r in rs)
        tp = sum(r["true"] == label and r["models"][m][scope]["pred"] == label for r in rs)
        predicted = sum(r["models"][m][scope]["pred"] == label for r in rs)
        recall, f1 = tp / support, 2 * tp / (support + predicted)
        detail[label] = {"n": support, "correct": tp, "predicted": predicted, "recall": recall, "f1": f1}
        recalls.append(recall)
        f1s.append(f1)
    signers = sorted({r["signer_id"] for r in rs if "signer_id" in r})
    people = {str(s): {"n": sum(r.get("signer_id") == s for r in rs),
                       "correct": sum(r.get("signer_id") == s and r["models"][m][scope]["correct"] for r in rs)}
              for s in signers}
    return {"n": len(rs), "correct": sum(r["models"][m][scope]["correct"] for r in rs),
            "input_failed": sum(bool(r["models"][m][scope].get("input_failed")) for r in rs),
            "ground_truth_classes": len(labels),
            "macro_recall": float(np.mean(recalls)) if recalls else None,
            "macro_f1": float(np.mean(f1s)) if f1s else None, "per_class": detail, "per_signer": people}


def flip_lists(rs, new, old):
    out = {}
    for sc in SCOPES:
        both = [r for r in rs if sc in r["models"][new] and sc in r["models"][old]]
        key = lambda r: {"id": r["id"], "true": r["true"], "signer_id": r.get("signer_id"),
                         old: r["models"][old][sc]["pred"], new: r["models"][new][sc]["pred"]}
        gains = [key(r) for r in both if r["models"][new][sc]["correct"] and not r["models"][old][sc]["correct"]]
        regs = [key(r) for r in both if not r["models"][new][sc]["correct"] and r["models"][old][sc]["correct"]]
        out[sc] = {"gains": len(gains), "regressions": len(regs),
                   "prediction_flips": sum(r["models"][new][sc]["pred"] != r["models"][old][sc]["pred"] for r in both),
                   "gain_rows": gains, "regression_rows": regs}
    return out


def evaluate(env, train_rec, started):
    p, pp, lio, F, G, torch = env.p, env.pp, env.lio, env.F, env.G, env.torch
    from pilot.data import PilotDataError, build_channels
    from wis_lab.service.neural import PROJECT_18

    ck, parent, classes = lio.parent_model(pp)
    if len(classes) != 93:
        raise RuntimeError("class order is not 93")
    ref = p["references"]["project_reference"]
    paths = {"control": Path(train_rec["models"]["control"]["path"]),
             "candidate": Path(train_rec["models"]["candidate"]["path"]),
             "current832": S.resolve(ref["path"])}
    expect = {"control": train_rec["models"]["control"]["sha256"],
              "candidate": train_rec["models"]["candidate"]["sha256"], "current832": ref["sha256"]}
    parent_state = parent.state_dict()
    heads, models, model_hashes = {}, {}, {}
    for name, path in paths.items():
        if S.sha(path) != expect[name]:
            raise RuntimeError(f"{name} checkpoint hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        mdl, cl = lio.u.rebuild_from_payload(payload)
        mdl.eval()
        if cl != classes:
            raise RuntimeError(f"{name} class order differs")
        sd = mdl.state_dict()
        shared = set(sd) == set(parent_state) and all(
            torch.equal(sd[k], parent_state[k]) for k in sd if not k.startswith("head."))
        if not shared:
            raise RuntimeError(f"{name} does not share the frozen encoder/buffers; shared features unsafe")
        heads[name] = (sd["head.weight"].numpy().astype(np.float64), sd["head.bias"].numpy().astype(np.float64))
        models[name] = mdl
        model_hashes[name] = {"path": str(path), "sha256": expect[name]}
    target = [i for i, c in enumerate(classes) if c in set(PROJECT_18.values())]

    groups, meta = lio.evaluation_inputs(pp)
    if set(groups) != {"annotation_dev", "whole_dev", "public", "mp"}:
        raise RuntimeError(f"unexpected evaluation groups {sorted(groups)}")
    mat = p["evaluation"]["decimated_material"]
    roots = [S.resolve(r) for r in mat["source_roots"]]
    dec_xs, dec_err = [], {}
    for r in meta["annotation_dev"]:
        nat = F.load_native(r, mat["roles"], mat["signers"], roots)
        xy, pr, ts = F.decimate(nat["xy"], nat["present"], nat["ts"], stride=2, offset=0)
        if not F.kept_intervals_are_sums(nat["ts"], ts, 2, 0):
            raise RuntimeError(f"{r['id']}: kept timestamps are not ts[0::2]")
        try:
            dec_xs.append(build_channels(xy, pr, ts)["x"])
        except PilotDataError as exc:                        # counted wrong for every model, listed
            dec_xs.append(None)
            dec_err[r["id"]] = {"error_class": type(exc).__name__, "error": str(exc)}
    plan = [("annotation_dev", "annotation_dev", groups["annotation_dev"], None),
            (DEC_GROUP, "annotation_dev", dec_xs, DEC_COND),
            ("whole_dev", "whole_dev", groups["whole_dev"], None),
            ("public", "public", groups["public"], None),
            ("mp", "mp", groups["mp"], None)]

    old = S.read(S.resolve("runs/partial_encoder_pair_v1/evaluation.json"))
    oldmap = {(r["group"], r["id"], r["condition"]): r for r in old["rows"]}
    s0_rows = {r["id"]: r for r in S.read(S.resolve("runs/framerate_stage0_v1/stage0_result.json"))["rows"]}
    with np.load(S.resolve("runs/framerate_stage0_v1/stage0_logits.npz"), allow_pickle=False) as z:
        if z["classes"].tolist() != classes:
            raise RuntimeError("stage0 class order differs")
        s0_index = {k: i for i, k in enumerate(z["ids"].tolist())}
        s0_dec = z["decimated_logits"].copy()
    old_arr = np.load(S.resolve("runs/partial_encoder_pair_v1/evaluation_arrays.npz"), allow_pickle=False)
    old_feat = np.load(pp["paths"]["old_features"], allow_pickle=False)

    arrays = {"classes": np.array(classes), "project18_indices": np.array(target, dtype=np.int64)}
    rows, mismatches, status_mismatches = [], [], []
    verification = {"parent_pooled_rebuild_error": {}, "max_abs_832_logit_diff_vs_stored": {},
                    "max_spot_check_affine_error": 0.0}
    try:
        for gkey, mgroup, xs, cond_override in plan:
            specs = meta[mgroup]
            if len(specs) != len(xs):
                raise RuntimeError(f"{gkey}: metadata/input length differ")
            ok = [i for i, x in enumerate(xs) if x is not None]
            if cond_override is None and len(ok) != len(xs):
                raise RuntimeError(f"{gkey}: original input failed")
            prefixes = G.extract_prefix(parent, [xs[i] for i in ok])
            pooled_ok = G.predict_prefix(parent, prefixes)["pooled"]
            pooled = np.full((len(xs), pooled_ok.shape[1]), np.nan, dtype=np.float32)
            pooled[ok] = pooled_ok
            if cond_override is None:                        # validates the split path, not accuracy
                prior = old_feat[gkey + "_x"]
                err = float(np.max(np.abs(pooled - prior)))
                if not np.allclose(pooled, prior, atol=1e-4, rtol=1e-5):
                    raise RuntimeError(f"{gkey}: parent pooled rebuild error {err}")
                verification["parent_pooled_rebuild_error"][gkey] = err
            scores = {}
            for name, (w, b) in heads.items():
                sc = pooled.astype(np.float64) @ w.T + b      # failed rows stay NaN
                k = min(32, len(ok))
                if k:
                    direct = G.predict_prefix(models[name], prefixes[:k])["logits"].astype(np.float64)
                    e = float(np.max(np.abs(direct - sc[ok[:k]])))
                    if e >= 1e-4 or not np.array_equal(direct.argmax(1), sc[ok[:k]].argmax(1)):
                        raise RuntimeError(f"{gkey}/{name}: shared-feature affine differs from predict_prefix ({e})")
                    verification["max_spot_check_affine_error"] = max(verification["max_spot_check_affine_error"], e)
                scores[name] = sc
            if cond_override is None:
                key = f"{gkey}__control__scores"
                if key in old_arr.files and old_arr[key].shape == scores["current832"].shape:
                    verification["max_abs_832_logit_diff_vs_stored"][gkey] = float(np.max(np.abs(old_arr[key] - scores["current832"])))
            else:
                s0 = s0_dec[[s0_index[s["id"]] for s in specs]]
                verification["max_abs_832_logit_diff_vs_stored"][gkey] = float(np.max(np.abs(s0[ok] - scores["current832"][ok]))) if ok else None

            arrays[f"{gkey}__ids"] = np.array([s["id"] for s in specs])
            arrays[f"{gkey}__input_ok"] = np.array([x is not None for x in xs])
            arrays[f"{gkey}__pooled"] = pooled
            for name in ("control", "candidate"):
                arrays[f"{gkey}__{name}__scores"] = scores[name]
            arrays[f"{gkey}__current832_recomputed__scores"] = scores["current832"]
            conds, trues = [], []
            for i, spec in enumerate(specs):
                label = spec.get("label", spec.get("true", spec.get("reference_label")))
                yi = classes.index(label)
                cond = cond_override or spec.get("condition", "intact")
                okrow = xs[i] is not None
                row = {"group": gkey, "id": spec["id"], "condition": cond, "true": label,
                       "input_status": "OK" if okrow else "INPUT_FAILED", "models": {}}
                if "signer_id" in spec:
                    row["signer_id"] = spec["signer_id"]
                if not okrow:
                    row["input_error"] = dec_err[spec["id"]]
                for name in ("control", "candidate"):
                    row["models"][name] = score_all(scores[name][i] if okrow else None, yi, target, classes)
                recomputed = score_all(scores["current832"][i] if okrow else None, yi, target, classes)
                if cond_override is None:
                    old_row = oldmap[(gkey, spec["id"], cond)]
                    if old_row["true"] != label:
                        raise RuntimeError(f"{spec['id']}: stored truth differs")
                    row["models"]["current832"] = {sc: dict(v) for sc, v in old_row["models"]["control"].items() if sc in SCOPES}
                else:
                    s0ok = s0_rows[spec["id"]]["decimated"]["status"] == "OK"
                    if s0ok != okrow:
                        status_mismatches.append({"id": spec["id"], "stage0_ok": s0ok, "now_ok": okrow})
                    row["models"]["current832"] = score_all(s0_dec[s0_index[spec["id"]]] if s0ok else None,
                                                            yi, target, classes)
                for sc in SCOPES:
                    a = recomputed.get(sc, {}).get("pred")
                    b = row["models"]["current832"].get(sc, {}).get("pred")
                    if a != b or (sc in recomputed) != (sc in row["models"]["current832"]):
                        mismatches.append({"group": gkey, "id": spec["id"], "condition": cond, "scope": sc,
                                           "stored": b, "recomputed": a})
                conds.append(cond)
                trues.append(label)
                rows.append(row)
            arrays[f"{gkey}__conditions"] = np.array(conds)
            arrays[f"{gkey}__true"] = np.array(trues)
            print(json.dumps({"evaluated_group": gkey, "rows": len(xs), "input_failed": len(xs) - len(ok)}), flush=True)
    finally:
        old_arr.close()
        old_feat.close()

    verification.update({"top1_mismatches_832_recomputed_vs_stored": mismatches,
                         "decimated_input_status_mismatches_vs_stage0": status_mismatches,
                         "rows_checked": len(rows)})
    if mismatches or status_mismatches:
        S.write_json(STOPPED, {"status": "STOPPED_BASELINE_NOT_REPRODUCED", "at_kst": S.now(),
                               "verification": verification, "comparisons_reported": 0,
                               "policy_sha256": S.sha(S.POLICY), "training_record_sha256": S.sha(R / "training.json")})
        raise SystemExit("83212868 not reproduced; no comparison is attributed")

    # ---- summaries (project reference and method reference kept distinct)
    order = [g for g, *_ in plan]
    summary = {}
    for g in order:
        rs_g = [r for r in rows if r["group"] == g]
        summary[g] = {c: {m: {sc: summarize([r for r in rs_g if r["condition"] == c], m, sc) for sc in SCOPES}
                          for m in MODELS}
                      for c in sorted({r["condition"] for r in rs_g})}

    def s(g, c, m, sc="full93"):
        return summary[g][c][m][sc]

    ao = lambda m: s("annotation_dev", "intact", m)
    ad = lambda m: s(DEC_GROUP, DEC_COND, m)
    t33_rows = [r for r in rows if r["group"] == "annotation_dev" and classes.index(r["true"]) in target]
    ap = [r for r in rows if r["group"] == "annotation_dev" and r["id"].startswith(APOLOGY)]
    if len(t33_rows) != 33 or len(ap) != 3 or ao("candidate")["ground_truth_classes"] != 81:
        raise RuntimeError("annotation_dev structure differs (target33 / apology3 / 81 classes)")
    t33 = {m: sum(r["models"][m]["full93"]["correct"] for r in t33_rows) for m in MODELS}

    def signer_ge(a, b):
        return all(k in a["per_signer"] and a["per_signer"][k]["correct"] >= b["per_signer"][k]["correct"] for k in SIGNERS)

    def all_correct(g, c, n):
        f, q = s(g, c, "candidate"), s(g, c, "candidate", "project18")
        return f["n"] == n and f["correct"] == n and q["correct"] == q["n"]

    absolute = {
        "apology_dev_3of3": all(r["models"]["candidate"]["full93"]["correct"] for r in ap)
                            and all(v["correct"] for r in ap for sc, v in r["models"]["candidate"].items() if sc == "project18"),
        "whole_dev_all_correct": all_correct("whole_dev", "intact", 74),
        "public_intact_all_correct": all_correct("public", "intact", 18),
        "public_reviewed_masked_all_correct": all_correct("public", "reviewed_masked", 5),
        "public_additional_train_all_correct": all_correct("public", "additional_train", 9),
        "mp_whole_clip_all_correct": all_correct("mp", "whole_clip", 7),
        "mp_provided_annotation_all_correct": all_correct("mp", "provided_annotation", 7),
        "original_input_failures_zero": all(r["input_status"] == "OK" for r in rows if r["group"] != DEC_GROUP),
    }

    def relative(ref_name):
        return {
            "decimated_total_plus3": ad("candidate")["correct"] - ad(ref_name)["correct"] >= 3,
            "decimated_each_signer_nondecrease": signer_ge(ad("candidate"), ad(ref_name)),
            "original_total_nondecrease": ao("candidate")["correct"] >= ao(ref_name)["correct"],
            "original_each_signer_nondecrease": signer_ge(ao("candidate"), ao(ref_name)),
            "original_macro_recall_nondecrease": ao("candidate")["macro_recall"] >= ao(ref_name)["macro_recall"] - 1e-12,
            "original_target33_full93_nondecrease": t33["candidate"] >= t33[ref_name],
        }

    gates = {"vs_matched_control": {**relative("control"), **absolute},
             "vs_current832": {**relative("current832"), **absolute}}
    pass_c, pass_p = all(gates["vs_matched_control"].values()), all(gates["vs_current832"].values())
    decision = ("ELIGIBLE_PENDING_INDEPENDENT_AUDIT" if pass_c and pass_p else
                "METHOD_EFFECT_ONLY_NOT_PROJECT_IMPROVEMENT" if pass_c else "REJECTED_BY_PREDECLARED_GATES")
    key_numbers = {m: {"original699_full93": ao(m)["correct"], "original_per_signer": ao(m)["per_signer"],
                       "original_macro_recall_81": ao(m)["macro_recall"], "original_macro_f1_81": ao(m)["macro_f1"],
                       "decimated699_full93": ad(m)["correct"], "decimated_per_signer": ad(m)["per_signer"],
                       "decimated_input_failed": ad(m)["input_failed"], "target33_full93": t33[m],
                       "annotation_project18_restricted_reported_only": s("annotation_dev", "intact", m, "project18")["correct"],
                       "decimated_project18_restricted_reported_only": s(DEC_GROUP, DEC_COND, m, "project18")["correct"]}
                   for m in MODELS}
    changes = {}
    for g in order:
        for c in summary[g]:
            rs = [r for r in rows if r["group"] == g and r["condition"] == c]
            changes.setdefault(g, {})[c] = {"candidate_vs_control": flip_lists(rs, "candidate", "control"),
                                            "candidate_vs_current832": flip_lists(rs, "candidate", "current832"),
                                            "control_vs_current832": flip_lists(rs, "control", "current832")}

    with OUT_ARR.open("xb") as fh:
        np.savez_compressed(fh, **arrays)
    result = {
        "status": "PUBLIC_EVALUATION_COMPLETE_PENDING_INDEPENDENT_AUDIT", "at_kst": S.now(),
        "decision": decision, "gates": gates,
        "gate_rule": "project improvement only if vs_matched_control AND vs_current832 all pass AND independent audit passes",
        "key_numbers": key_numbers,
        "reference_roles": {"current832": "project reference, stored evaluation.json rows + stage0 decimated logits",
                            "control": "method reference, head-only reproduction in this run",
                            "candidate": "decimation-replacement arm"},
        "summary": summary, "changes": changes, "rows": rows, "class_order": classes,
        "project18_indices": target, "models": model_hashes, "verification": verification,
        "decimated_input_failures": dec_err,
        "policy_sha256": S.sha(S.POLICY), "training_record_sha256": S.sha(R / "training.json"),
        "arrays_sha256": S.sha(OUT_ARR),
        "arrays_layout": "classes, project18_indices, <group>__{ids,conditions,true,input_ok,pooled}, "
                         "<group>__{control,candidate,current832_recomputed}__scores (float64 = pooled @ head.weight.T + head.bias)",
        "counts": {"rows": len(rows), "shared_prefix_clips": sum(sum(x is not None for x in xs) for _g, _m, xs, _c in plan),
                   "ksl77_read": 0, "user_arrays_read": 0, "reserved_arrays_read": 0, "training": 0},
        "default_promoted": False, "registered": False,
        "seconds": round(time.time() - started, 2),
        "limits": p["limits"] + [
            "decimated 699 shares recordings with original 699; conditions are not independent cases",
            "actual project target truths in native development are few; target33 is a count, not a class-balanced claim",
            "public reference includes training examples and approved development stress cases"],
    }
    S.write_json(OUT_JSON, result)
    print(json.dumps({"decision": decision, "gates": gates,
                      "key": {m: {k: key_numbers[m][k] for k in ("original699_full93", "decimated699_full93", "target33_full93")}
                              for m in MODELS}}, ensure_ascii=False), flush=True)


def main():
    started = time.time()
    env = S.load_env()
    train_rec = S.read(R / "training.json")
    if train_rec.get("status") != TRAIN_STATUS:
        raise SystemExit("training record is not frozen/complete")
    for path in (OUT_JSON, OUT_ARR, FAILED, STOPPED):
        if path.exists():
            raise SystemExit(f"{path.name} exists; evaluation is never repeated or overwritten")
    if train_rec["policy_sha256"] != S.sha(S.POLICY):
        raise SystemExit("policy changed after training")
    S.verify_hashes(train_rec["frozen_inputs"])
    S.verify_hashes(train_rec["frozen_outputs"])
    try:
        evaluate(env, train_rec, started)
    except BaseException as exc:
        if not STOPPED.exists():
            S.write_json(FAILED, {"status": "EVALUATION_FAILED_PRESERVED", "at_kst": S.now(),
                                  "error_class": type(exc).__name__, "error": str(exc),
                                  "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
