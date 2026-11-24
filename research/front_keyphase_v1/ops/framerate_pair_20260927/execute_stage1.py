"""Stage 1: exactly one pair (control, candidate), 660 head-only updates each, from the same parent
and schedule. Every checkpoint, the training record and all code/cache/policy hashes are frozen in
runs/framerate_pair_v1/ BEFORE any development evaluation. Refuses an existing run dir; a failure
after the dir is created is written to TRAINING_FAILED.json and preserved. No model registration,
default or UI change; nothing is written outside runs/framerate_pair_v1/.

Run: py -3.13 ops/framerate_pair_20260927/execute_stage1.py
"""
import copy
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


def load_inputs(env):
    p, pp, lio, T = env.p, env.pp, env.lio, env.T
    ck, parent, classes = lio.parent_model(pp)
    with np.load(S.PRIOR / "train_prefixes.npz", allow_pickle=False) as z:
        if z["classes"].tolist() != classes:
            raise SystemExit("prior cache class order differs")
        base = lio.unpack(z["base_prefix"], z["base_offsets"])
        ctx = lio.unpack(z["pair_context_prefix"], z["pair_context_offsets"])
        by, cy = z["base_y"].copy(), z["pair_context_y"].copy()
        teacher, bi, ci = z["teacher_base_logits"].copy(), z["base_indices"].copy(), z["context_indices"].copy()
    meta = S.read(pp["paths"]["metadata"])
    eligible = S.eligibility(p, meta)
    with np.load(S.CACHE, allow_pickle=False) as z:
        ok = (z["classes"].tolist() == classes
              and z["base_ids"].tolist() == [r["id"] for r in meta["base"]]
              and np.array_equal(z["eligible_mask"], eligible)
              and np.array_equal(z["eligible_rows"], np.flatnonzero(eligible))
              and z["transform_stride_offset"].tolist() == [2, 0]
              and int(z["coin_seed"]) == p["coins"]["seed"])
        if not ok:
            raise SystemExit("decimated cache metadata differs from the metadata/policy")
        coins = z["coins"].copy()
        rows = z["eligible_rows"].copy()
        values, offsets = z["decimated_prefix"], z["decimated_offsets"]
        if len(offsets) != len(rows) + 1 or int(offsets[-1]) != len(values) or \
                not np.array_equal(np.diff(offsets), z["channel_frames_decimated"]):
            raise SystemExit("decimated cache offsets are inconsistent")
        dec = [None] * len(base)
        for k, i in enumerate(rows.tolist()):
            dec[i] = values[offsets[k]:offsets[k + 1]].copy()
    if not np.array_equal(coins, T.make_coins(int(p["coins"]["seed"]), tuple(p["coins"]["shape"]))):
        raise SystemExit("stored coins differ from the predeclared generator")
    if not (bi.shape == ci.shape == coins.shape == (660, 16) and len(base) == 2122 and len(ctx) == 224):
        raise SystemExit("schedule/coin/cache shapes differ from the contract")
    return ck, parent, classes, (base, by, ctx, cy, teacher, bi, ci, dec, eligible, coins)


def train(env, ck, parent, data, inputs_frozen, started):
    p, lio, T, torch = env.p, env.lio, env.T, env.torch
    base, by, ctx, cy, teacher, bi, ci, dec, eligible, coins = data
    S.write_json(R / "preflight.json", {
        "status": "READY_FOR_ONE_PAIR", "at_kst": S.now(), "frozen_inputs": inputs_frozen,
        "parent": p["parent"], "arms": p["arms"], "settings": p["settings"],
        "train_clips": [len(base), len(ctx)], "eligible_rows": int(eligible.sum()), "coins_true": int(coins.sum()),
        "train_signers": list(range(1, 10)), "development_arrays_read": 0, "user_training_records": 0})

    def progress(arm, done, total):
        print(json.dumps({"arm": arm, "steps": done, "of": total, "seconds": round(time.time() - started, 1)}), flush=True)

    kwargs = {k: p["settings"][k] for k in ("learning_rate", "weight_decay", "anchor_coefficient", "output_coefficient")}
    result = T.run_framerate_pair(parent, base, by, ctx, cy, teacher, bi, ci, dec, eligible, coins,
                                  progress=progress, **kwargs)

    head = {n for n, _ in parent.named_parameters() if n.startswith("head.")}
    expected = T.replacement_mask(bi, coins, eligible)
    if not (np.array_equal(result["schedule"]["base_indices"], bi) and np.array_equal(result["schedule"]["context_indices"], ci)
            and np.array_equal(result["schedule"]["coins"], coins)):
        raise RuntimeError("schedule/coins changed during training")
    if result["control"]["replacements"] != 0 or not np.array_equal(result["candidate"]["replaced"], expected):
        raise RuntimeError("actual replacements differ from the predeclared rule")
    ref_payload = torch.load(S.resolve(p["references"]["project_reference"]["path"]), map_location="cpu", weights_only=True)
    ref_state = ref_payload["state_dict"]
    specs, arrays = {}, {"base_indices": bi, "context_indices": ci, "coins": coins, "eligible_mask": eligible}
    for arm in ("control", "candidate"):
        val = result[arm]
        state = val["state_dict"]
        if set(val["trainable_names"]) != head or any(n.startswith("blocks.") for n in val["trainable_names"]):
            raise RuntimeError(f"{arm}: trainable set is not exactly head.*")
        altered = [k for k in state if not torch.equal(state[k], ck["state_dict"][k])]
        if not altered or not set(altered) <= head:
            raise RuntimeError(f"{arm}: changed tensors outside head.* or none changed: {altered}")
        if not (val["frozen_parameters_unchanged"] and val["buffers_unchanged"] and val["updates"] == 660
                and np.isfinite(val["losses"]).all()
                and all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())):
            raise RuntimeError(f"{arm}: frozen/buffer/finite/update check failed")
        out = copy.deepcopy(ck)
        out["state_dict"] = state
        out["arm"] = arm
        out["step"] = 660
        out["produced_by"] = "Claude generic framerate_train.run_framerate_pair with local matched comparison"
        out["parent_checkpoint"] = str(S.resolve(p["parent"]["path"]))
        out["parent_checkpoint_sha256"] = p["parent"]["sha256"]
        out["settings"] = S.plain(result["settings"])
        out["framerate_pair"] = {"policy_sha256": S.sha(S.POLICY), "trainable_names": list(val["trainable_names"]),
                                 "replacements": int(val["replacements"]), "coin_seed": int(p["coins"]["seed"]),
                                 "transform": "decimate stride2 offset0 (candidate only)",
                                 "all_buffers_fixed": True, "encoder_fixed": True, "registered_as_default": False}
        path = R / f"checkpoint_{arm}_final.pt"
        if path.exists():
            raise RuntimeError(f"{path} exists")
        torch.save(out, path)
        specs[arm] = {"path": str(path), "sha256": S.sha(path), "changed_tensors": altered,
                      "trainable_names": val["trainable_names"], "frozen_parameter_count": len(val["frozen_names"]),
                      "frozen_parameters_unchanged": val["frozen_parameters_unchanged"],
                      "buffers_unchanged": val["buffers_unchanged"], "updates": val["updates"],
                      "replacements": val["replacements"], "replaced_distinct_rows": val["replaced_distinct_rows"],
                      "loss_first": val["losses"][0].tolist(), "loss_last": val["losses"][-1].tolist()}
        arrays[f"{arm}_losses"] = val["losses"]
        arrays[f"{arm}_replaced"] = val["replaced"]
        if arm == "control":
            same = set(ref_state) == set(state) and all(torch.equal(state[k], ref_state[k]) for k in state)
            diff = max(float((state[k].double() - ref_state[k].double()).abs().max()) for k in head)
            specs[arm]["versus_83212868"] = {"state_bitwise_equal": bool(same), "max_abs_head_difference": diff,
                                             "meaning": "information only; 83212868 stays the project reference either way"}
    with (R / "training_arrays.npz").open("xb") as fh:
        np.savez_compressed(fh, **arrays)
    S.verify_hashes(inputs_frozen)
    if S.sha(S.resolve(p["parent"]["path"])) != p["parent"]["sha256"]:
        raise RuntimeError("parent checkpoint changed")
    outputs = {f"runs/framerate_pair_v1/{n}": S.sha(R / n) for n in
               ("preflight.json", "checkpoint_control_final.pt", "checkpoint_candidate_final.pt", "training_arrays.npz")}
    record = {
        "status": "TRAINING_COMPLETE_BOTH_CHECKPOINTS_FROZEN_BEFORE_DEVELOPMENT_EVALUATION",
        "at_kst": S.now(), "models": specs, "parent": p["parent"], "settings": result["settings"],
        "frozen_inputs": inputs_frozen, "frozen_outputs": outputs, "policy_sha256": S.sha(S.POLICY),
        "expected_candidate_replacements": int(expected.sum()),
        "both_arms": {"initial_parent_sha256": p["parent"]["sha256"],
                      "schedule_and_coins": {k: S.array_checksum(v) for k, v in
                                             (("base_indices", bi), ("context_indices", ci), ("coins", coins))},
                      "trainable": "head.* only"},
        "seconds": round(time.time() - started, 2), "actual_training_runs": 2, "optimizer_updates": 1320,
        "retry_for_accuracy": 0, "development_arrays_read": 0, "user_arrays_read": 0, "reserved_arrays_read": 0,
        "ksl77_read": 0, "new_data": 0, "registered_or_default_changed": False}
    S.write_json(R / "training.json", record)
    print(json.dumps({"status": record["status"], "seconds": record["seconds"],
                      "replacements": {a: specs[a]["replacements"] for a in specs},
                      "control_equals_83212868": specs["control"]["versus_83212868"]["state_bitwise_equal"]}), flush=True)


def main():
    started = time.time()
    env = S.load_env()
    S.require_tests()
    S.require_ready()
    S.require_audit()
    if R.exists():
        raise SystemExit(f"{R} exists; a run is never repeated or overwritten")
    ck, parent, _classes, data = load_inputs(env)
    inputs_frozen = S.frozen_input_hashes(env.p)
    R.mkdir(parents=False, exist_ok=False)
    try:
        train(env, ck, parent, data, inputs_frozen, started)
    except BaseException as exc:
        S.write_json(R / "TRAINING_FAILED.json", {"status": "TRAINING_FAILED_PRESERVED", "at_kst": S.now(),
                                                   "error_class": type(exc).__name__, "error": str(exc),
                                                   "traceback": traceback.format_exc(), "retry_allowed": False})
        raise


if __name__ == "__main__":
    main()
