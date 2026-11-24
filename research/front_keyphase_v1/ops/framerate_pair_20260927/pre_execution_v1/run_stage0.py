"""Stage 0: the frozen current model on annotation_dev 699, original and decimated (stride 2).

Order: policy and pins -> rebuild 83212868 -> reproduce the stored 699 control predictions
(691 correct, every top-1 equal) -> only then the decimated forward. Writes only
runs/framerate_stage0_v1/ and refuses to run if it exists. No training, no KSL77, no user data,
no camera. Raw source arrays are never written; logits go to a local npz.

Run: py -3.13 ops/framerate_pair_20260927/run_stage0.py
"""
import hashlib
import json
import pathlib
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

H = pathlib.Path(__file__).resolve().parent
L = H.parents[1]
OUT = L / "runs" / "framerate_stage0_v1"
sys.path[:0] = [str(H), str(L), str(L / "src"), str(L.parent / "sign_model_pilot_v1" / "src")]
import framerate_impl as F                                    # noqa: E402

sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(pathlib.Path(p).read_text(encoding="utf-8-sig"))


def write(name, value):
    with (OUT / name).open("x", encoding="utf-8", newline="\n") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")


def checked_policy():
    p = read(H / "comparison_policy.json")
    if p.get("status") != "FROZEN":
        raise SystemExit(f"policy status is {p.get('status')!r}, not FROZEN")
    for rel, digest in p["pins"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise SystemExit(f"pin not set: {rel}")
        got = sha((L / rel).resolve())
        if got != digest:
            raise SystemExit(f"pin mismatch {rel}: {got}")
    return p


def score_row(logits, classes, true_label):
    yi = classes.index(true_label)
    order = np.argsort(-logits, kind="stable")
    rival = next(int(j) for j in order if int(j) != yi)
    return {"pred": classes[int(order[0])], "correct": bool(int(order[0]) == yi),
            "rank": int(np.where(order == yi)[0][0]) + 1,
            "margin": float(logits[yi] - logits[rival])}


def tally(rows, key):
    out = defaultdict(lambda: {"n": 0, "original_correct": 0, "decimated_correct": 0,
                               "decimated_input_failed": 0})
    for r in rows:
        g = out[str(r[key])]
        g["n"] += 1
        g["original_correct"] += r["original"]["correct"]
        g["decimated_correct"] += bool(r["decimated"].get("correct"))
        g["decimated_input_failed"] += r["decimated"]["status"] != "OK"
    return dict(sorted(out.items()))


def main():
    started = time.time()
    p = checked_policy()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; a completed run is never overwritten")

    import torch
    import tools_prep_union93_candidate as u
    from pilot.data import PilotDataError, build_channels, collate
    from wis_lab.service.neural import MIN_FRAMES, MIN_OBSERVED_HAND_FRAMES

    torch.set_num_threads(int(p["execution"]["threads"]))
    base = p["current_baseline"]
    ck = torch.load(L / base["path"], map_location="cpu", weights_only=True)
    model, classes = u.rebuild_from_payload(ck)
    model.eval()
    assert len(classes) == 93

    mat = p["material"]
    rows_meta = read(L / mat["metadata"])[mat["group"]]
    assert len(rows_meta) == mat["rows"] == 699
    stored = {r["id"]: r for r in read(L / "runs/partial_encoder_pair_v1/evaluation.json")["rows"]
              if r["group"] == "annotation_dev" and r["condition"] == "intact"}
    assert len(stored) == 699
    stride, offset = p["transform"]["stride"], p["transform"]["offset"]
    assert (stride, offset) == (2, 0)
    root = (L / mat["allowed_source_root"]).resolve()

    def forward(x):
        bx, pad, _ = collate([{"x": x, "frames": int(len(x)), "label": 0}])
        with torch.inference_mode():
            return model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy().astype(np.float64)

    OUT.mkdir(parents=True)
    ids, natives, orig_logits, rows = [], [], [], []

    # ---- 1. reproduction of the stored control predictions, before anything else
    mismatches = []
    for r in rows_meta:
        nat = F.load_native(r, mat["allowed_roles"], mat["allowed_signers"], root)
        x = build_channels(nat["xy"], nat["present"], nat["ts"])["x"]
        lg = forward(x)
        s = score_row(lg, classes, r["label"])
        old = stored[r["id"]]
        assert old["true"] == r["label"], r["id"]
        if s["pred"] != old["models"]["control"]["full93"]["pred"]:
            mismatches.append({"id": r["id"], "stored": old["models"]["control"]["full93"]["pred"],
                               "now": s["pred"]})
        ids.append(r["id"])
        natives.append(nat)
        orig_logits.append(lg)
        rows.append({"id": r["id"], "recording_id": F.recording_id(r["id"]),
                     "signer_id": r["signer_id"], "true": r["label"], "source_sha256": nat["sha256"],
                     "frames_original": int(len(nat["ts"])), "original": s})
    orig_correct = sum(r["original"]["correct"] for r in rows)
    score_diff = None
    arr_path = L / "runs/partial_encoder_pair_v1/evaluation_arrays.npz"
    if arr_path.exists():
        with np.load(arr_path, allow_pickle=False) as z:
            key = "annotation_dev__control__scores"
            if key in z.files and z[key].shape == (699, 93):
                score_diff = float(np.max(np.abs(np.asarray(orig_logits) - z[key])))
    reproduction = {"rows": 699, "top1_mismatches": mismatches, "original_correct": orig_correct,
                    "required_correct": base["stored_annotation_dev_correct"],
                    "max_abs_logit_diff_vs_stored_scores": score_diff,
                    "stored_scores_sha256": sha(arr_path) if arr_path.exists() else None}
    if mismatches or orig_correct != base["stored_annotation_dev_correct"]:
        write("STAGE0_REPRODUCTION_FAILED.json",
              {"status": "STOPPED_REPRODUCTION_FAILED", "reproduction": reproduction,
               "decimated_forwards": 0, "policy_sha256": sha(H / "comparison_policy.json")})
        raise SystemExit("reproduction failed; decimated condition not evaluated")

    # ---- 2. decimated condition, same rows, same model
    dec_logits = np.full((699, 93), np.nan)
    clock_ok = 0
    for i, (row, nat) in enumerate(zip(rows, natives)):
        xy, pr, ts = F.decimate(nat["xy"], nat["present"], nat["ts"], stride=stride, offset=offset)
        assert F.kept_intervals_are_sums(nat["ts"], ts, stride, offset), row["id"]
        nominal = F.nominal_dt_doubles(nat["ts"], ts)
        clock_ok += nominal
        row["frames_decimated"] = int(len(ts))
        row["nominal_dt_doubled"] = nominal
        row["web_gate_information"] = F.web_gate_information(pr, MIN_FRAMES, MIN_OBSERVED_HAND_FRAMES)
        try:
            x = build_channels(xy, pr, ts)["x"]
        except PilotDataError as exc:
            row["decimated"] = {"status": "INPUT_FAILED", "error_class": type(exc).__name__,
                                "error": str(exc), "correct": False}
        else:
            lg = forward(x)
            dec_logits[i] = lg
            row["decimated"] = dict(status="OK", **score_row(lg, classes, row["true"]))
        d = row["decimated"]
        row["flip"] = d["status"] == "OK" and d["pred"] != row["original"]["pred"]
        row["correctness_changed"] = bool(d["correct"]) != row["original"]["correct"]

    # ---- 3. summaries and the stage-1 rule on paired rows only
    paired = [r for r in rows if r["decimated"]["status"] == "OK"]
    p_orig = sum(r["original"]["correct"] for r in paired)
    p_dec = sum(r["decimated"]["correct"] for r in paired)
    net_loss = p_orig - p_dec
    rule = p["stage1_decision_rule"]
    losses = [{"id": r["id"], "true": r["true"], "decimated_pred": r["decimated"]["pred"]}
              for r in paired if r["original"]["correct"] and not r["decimated"]["correct"]]
    gains = [{"id": r["id"], "true": r["true"], "original_pred": r["original"]["pred"]}
             for r in paired if r["decimated"]["correct"] and not r["original"]["correct"]]
    failed = [{"id": r["id"], "error_class": r["decimated"]["error_class"],
               "frames_decimated": r["frames_decimated"]} for r in rows if r["decimated"]["status"] != "OK"]
    with (OUT / "stage0_logits.npz").open("xb") as fh:
        np.savez_compressed(fh, ids=np.array(ids), classes=np.array(classes),
                            original_logits=np.asarray(orig_logits), decimated_logits=dec_logits)
    result = {
        "status": "STAGE0_COMPLETE_PENDING_CODEX_VERIFICATION",
        "at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "question": p["question"], "is_not": p["is_not"],
        "model": {"path": base["path"], "sha256": sha(L / base["path"])},
        "transform": {"stride": stride, "offset": offset},
        "reproduction": reproduction,
        "full_denominator": {"n": 699, "original_correct": orig_correct,
                             "decimated_correct": sum(bool(r["decimated"]["correct"]) for r in rows),
                             "decimated_input_failed": len(failed)},
        "paired": {"n": len(paired), "original_correct": p_orig, "decimated_correct": p_dec,
                   "net_loss": net_loss, "losses": losses, "gains": gains,
                   "flips": sum(r["flip"] for r in paired)},
        "stage1_rule": rule["proceed_to_stage1_design_if"],
        "stage1_rule_met": net_loss >= 7,
        "stage1_rule_nature": rule["threshold_nature"],
        "input_failures": failed,
        "web_gate_information_count": sum(r["web_gate_information"]["under_web_minimums"] for r in rows),
        "web_gate_information_meaning": "information only; never used to accept, reject or decide",
        "nominal_clock_doubled_rows": clock_ok,
        "per_signer": tally(rows, "signer_id"),
        "per_class": tally(rows, "true"),
        "per_recording": tally(rows, "recording_id"),
        "distinct_recordings": len({r["recording_id"] for r in rows}),
        "rows": rows,
        "hashes": {"policy": sha(H / "comparison_policy.json"), "framerate_impl": sha(H / "framerate_impl.py"),
                   "run_stage0": sha(pathlib.Path(__file__)), "logits_npz": sha(OUT / "stage0_logits.npz")},
        "counts": {"rows_read": 699, "model_forwards": 699 + len(paired), "training": 0, "ksl77_read": 0,
                   "user_data_read": 0, "reserved_read": 0, "downloads": 0, "camera": 0},
        "seconds": round(time.time() - started, 2),
        "limits": p["limits"],
    }
    write("stage0_result.json", result)
    print(json.dumps({k: result[k] for k in ("full_denominator", "stage1_rule_met")} |
                     {"paired": {k: result["paired"][k] for k in ("n", "original_correct",
                                                                   "decimated_correct", "net_loss")}},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
