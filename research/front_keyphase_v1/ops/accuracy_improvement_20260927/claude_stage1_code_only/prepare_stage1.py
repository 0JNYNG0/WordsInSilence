"""Stage 1 training prep: decimated prefixes of the 2095 eligible TRAIN base rows, plus the shared
environment helpers (policy/pin checks, explicit-path module loading) used by execute/evaluate.

Reads only: the frozen stage1 policy and its pins, the prior partial-encoder policy/pins/cache
provenance, the parent checkpoint, the pinned metadata, teacher and schedule files, and the native
npz of eligible train rows (role train, signers 1..9, sha256 exact, under expand/npz). No dev,
whole, public, MP, KSL77, user or reserved source is opened. Source coordinates and channel arrays
are never serialized; only decimated prefixes (model features) and metadata go to the cache.

Run (after tests_stage1.py --write-evidence and the policy freeze):
    py -3.13 ops/framerate_pair_20260927/prepare_stage1.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

H = Path(__file__).resolve().parent
L = H.parents[1]
PRIOR = L / "ops" / "partial_encoder_pair_20260922"
RUN = L / "runs" / "framerate_pair_v1"
POLICY = H / "stage1_policy.json"
CACHE = H / "decimated_prefixes.npz"
READY = H / "STAGE1_PREFIX_READY.json"
PREP_FAILED = H / "STAGE1_PREP_FAILED.json"
TESTS_EVIDENCE = H / "STAGE1_SYNTHETIC_TESTS.json"
AUDIT = H / "STAGE1_INDEPENDENT_AUDIT.json"
HREL = "ops/framerate_pair_20260927"
CODE = ("framerate_train.py", "tests_stage1.py", "prepare_stage1.py", "execute_stage1.py",
        "evaluate_stage1.py", "framerate_impl.py")
STAGE1_ARTIFACTS = ("stage1_policy.json", "STAGE1_SYNTHETIC_TESTS.json", "STAGE1_PREFIX_READY.json",
                    "STAGE1_INDEPENDENT_AUDIT.json", "decimated_prefixes.npz")
REQUIRED_PINS = (
    "runs/public_anchor_0p01_pair_v1/checkpoint_treatment_final.pt",
    "runs/partial_encoder_pair_v1/checkpoint_control_final.pt",
    "runs/partial_encoder_pair_v1/evaluation.json",
    "runs/partial_encoder_pair_v1/evaluation_arrays.npz",
    "runs/partial_encoder_pair_v1/training.json",
    "runs/framerate_stage0_v1/stage0_result.json",
    "runs/framerate_stage0_v1/stage0_logits.npz",
    "ops/partial_encoder_pair_20260922/comparison_policy.json",
    "ops/partial_encoder_pair_20260922/generated_v1.py",
    "ops/partial_encoder_pair_20260922/local_io.py",
    "ops/partial_encoder_pair_20260922/IMPLEMENTATION_AUDIT.json",
    "ops/partial_encoder_pair_20260922/TRAIN_PREFIX_READY.json",
    "ops/partial_encoder_pair_20260922/train_prefixes.npz",
    "ops/context_training_pair_20260921/feature_metadata.json",
    "tools_prep_union93_candidate.py",
    "../sign_model_pilot_v1/src/pilot/data.py",
    "../sign_model_pilot_v1/src/pilot/model_masked_v1.py",
    "ops/framerate_pair_20260927/comparison_policy.json",
) + tuple(f"{HREL}/{n}" for n in CODE)
KST = ZoneInfo("Asia/Seoul")
N_SAMPLES = 8


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def now() -> str:
    return datetime.now(KST).isoformat()


def plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [plain(x) for x in v]
    return v


def write_json(path, value) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as fh:     # never overwrite
        json.dump(plain(value), fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")


def array_checksum(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def load_module(name: str, path):
    """Load a file under a private module name; never through sys.path lookup."""
    path = Path(path).resolve()
    mod = sys.modules.get(name)
    if mod is not None:
        if Path(mod.__file__).resolve() != path:
            raise SystemExit(f"module name {name} already bound to {mod.__file__}")
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def resolve(rel: str) -> Path:
    return (L / rel).resolve()


def checked_policy() -> dict:
    if H != (L / "ops" / "framerate_pair_20260927").resolve():
        raise SystemExit(f"stage1 code must run from {HREL}, not {H}")
    p = read(POLICY)
    if p.get("status") != "FROZEN":
        raise SystemExit(f"stage1 policy status is {p.get('status')!r}, not FROZEN")
    missing = [r for r in REQUIRED_PINS if r not in p["pins"]]
    if missing:
        raise SystemExit(f"required pins absent: {missing}")
    for rel, digest in p["pins"].items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise SystemExit(f"pin not set: {rel}")
        got = sha(resolve(rel))
        if got != digest:
            raise SystemExit(f"pin mismatch {rel}: {got}")
    ref = p["references"]["project_reference"]
    if p["pins"][p["parent"]["path"]] != p["parent"]["sha256"] or p["pins"][ref["path"]] != ref["sha256"]:
        raise SystemExit("parent / project reference pin disagrees with the policy body")
    if (p["transform"]["stride"], p["transform"]["offset"]) != (2, 0):
        raise SystemExit("transform must be stride 2 offset 0")
    if (p["coins"]["seed"], p["coins"]["shape"]) != (20260927, [660, 16]):
        raise SystemExit("coin spec differs from the predeclared one")
    return p


def load_env() -> SimpleNamespace:
    """Frozen stage1 policy + prior pins/provenance, then the pinned modules by explicit path."""
    p = checked_policy()
    lio = load_module("fr_prior_local_io", PRIOR / "local_io.py")   # prepends its own sys.path entries
    pp = lio.policy()                                               # re-verifies every prior pin + parent
    F = load_module("fr_framerate_impl", H / "framerate_impl.py")
    T = load_module("fr_framerate_train", H / "framerate_train.py")
    if Path(T.G.__file__).resolve() != (PRIOR / "generated_v1.py").resolve():
        raise SystemExit(f"framerate_train bound the wrong generated_v1: {T.G.__file__}")
    if pp["parent"]["sha256"] != p["parent"]["sha256"] or \
            Path(pp["parent"]["path"]).resolve() != resolve(p["parent"]["path"]):
        raise SystemExit("stage1 parent differs from the prior parent")
    for k, v in p["settings"].items():
        if pp["settings"].get(k) != v:
            raise SystemExit(f"setting {k} differs from the prior control: {pp['settings'].get(k)!r} vs {v!r}")
    ia, tr = read(PRIOR / "IMPLEMENTATION_AUDIT.json"), read(PRIOR / "TRAIN_PREFIX_READY.json")
    ok = (ia["status"] == tr["status"] == "PASS"
          and ia["module_sha256"] == tr["module_sha256"] == sha(PRIOR / "generated_v1.py")
          and tr["cache_sha256"] == sha(PRIOR / "train_prefixes.npz")
          and tr["policy_sha256"] == sha(PRIOR / "comparison_policy.json"))
    if not ok:
        raise SystemExit("prior train-cache provenance does not verify")
    rec = read(L / "runs/partial_encoder_pair_v1/training.json")
    s, st = rec["settings"], p["settings"]
    ok = (rec["models"]["control"]["sha256"] == p["references"]["project_reference"]["sha256"]
          and rec["policy_sha256"] == sha(PRIOR / "comparison_policy.json")
          and s["steps"] == st["steps"] and s["half_batch"] == st["half_batch"]
          and s["optimizer"]["learning_rate"] == st["learning_rate"]
          and s["optimizer"]["weight_decay"] == st["weight_decay"]
          and list(s["optimizer"]["betas"]) == st["adamw_betas"] and s["optimizer"]["eps"] == st["adamw_eps"]
          and s["anchor_coefficient"] == st["anchor_coefficient"]
          and s["output_coefficient"] == st["output_coefficient"]
          and s["trainable_scopes"]["control"] == ["head."])
    if not ok:
        raise SystemExit("prior control training record does not match the stage1 routine")
    import torch
    torch.set_num_threads(int(st["cpu_threads"]))
    return SimpleNamespace(p=p, pp=pp, lio=lio, F=F, T=T, G=T.G, torch=torch)


def code_hashes() -> dict:
    return {n: sha(H / n) for n in CODE}


def require_tests() -> dict:
    ev = read(TESTS_EVIDENCE)
    ok = (ev.get("status") == "PASS" and ev.get("failed") == []
          and ev.get("framerate_train_sha256") == sha(H / "framerate_train.py")
          and ev.get("tests_sha256") == sha(H / "tests_stage1.py")
          and ev.get("prior_generated_v1_sha256") == sha(PRIOR / "generated_v1.py"))
    if not ok:
        raise SystemExit("STAGE1_SYNTHETIC_TESTS.json missing, failed or stale")
    return ev


def require_ready() -> dict:
    r = read(READY)
    ok = (r.get("status") == "PASS" and r.get("policy_sha256") == sha(POLICY)
          and r.get("cache_sha256") == sha(CACHE) and r.get("code_sha256") == code_hashes()
          and r.get("synthetic_tests_sha256") == sha(TESTS_EVIDENCE)
          and r.get("prior_train_cache_sha256") == sha(PRIOR / "train_prefixes.npz"))
    if not ok:
        raise SystemExit("STAGE1_PREFIX_READY.json missing, failed or stale")
    return r


def require_audit() -> dict:
    a = read(AUDIT)
    ok = (a.get("status") == "PASS" and a.get("policy_sha256") == sha(POLICY)
          and a.get("code_sha256") == code_hashes()
          and a.get("prefix_ready_sha256") == sha(READY)
          and a.get("decimated_cache_sha256") == sha(CACHE)
          and a.get("synthetic_tests_sha256") == sha(TESTS_EVIDENCE))
    if not ok:
        raise SystemExit("STAGE1_INDEPENDENT_AUDIT.json missing, not PASS or stale")
    return a


def frozen_input_hashes(p: dict) -> dict:
    d = dict(p["pins"])
    for n in STAGE1_ARTIFACTS:
        d[f"{HREL}/{n}"] = sha(H / n)
    return d


def verify_hashes(d: dict) -> None:
    for rel, digest in d.items():
        got = sha(resolve(rel))
        if got != digest:
            raise SystemExit(f"frozen hash changed {rel}: {got}")


def eligibility(p: dict, meta: dict) -> np.ndarray:
    """Eligible = base rows with method == native_annotation_forward; count checks per policy."""
    el, base = p["eligibility"], meta["base"]
    if len(base) != 2122 or len(meta["pair_context"]) != 224:
        raise SystemExit("unexpected base/context sizes")
    eligible = np.array([r.get("method") == el["method"] for r in base], dtype=bool)
    non = [r for r, e in zip(base, eligible) if not e]
    nikl = [r for r in non if str(r["id"]).startswith("NIKL_")]
    if int(eligible.sum()) != el["rows"] or len(nikl) != 18 or len(non) - len(nikl) != 9:
        raise SystemExit(f"eligibility counts differ: {int(eligible.sum())}, NIKL {len(nikl)}, other {len(non) - len(nikl)}")
    for r, e in zip(base, eligible):
        if e and (str(r["id"]).startswith("NIKL_") or r.get("role") != "train"
                  or r.get("signer_id") not in el["signers"]):
            raise SystemExit(f"eligible row outside train roles/signers: {r['id']}")
    return eligible


# --------------------------------------------------------------------------- #
# prep
# --------------------------------------------------------------------------- #
def main() -> None:
    started = time.time()
    env = load_env()
    p, pp, lio, F, T, G = env.p, env.pp, env.lio, env.F, env.T, env.G
    tests = require_tests()
    for path in (CACHE, READY, PREP_FAILED):
        if path.exists():
            raise SystemExit(f"{path.name} exists; prep is never repeated or overwritten")
    from pilot.data import build_channels                            # path set by prior local_io

    ck, parent, classes = lio.parent_model(pp)
    torch = env.torch
    torch.set_num_threads(int(p["settings"]["cpu_threads"]))
    meta = read(pp["paths"]["metadata"])
    base = meta["base"]
    eligible = eligibility(p, meta)
    rows_idx = np.flatnonzero(eligible).astype(np.int64)
    el = p["eligibility"]
    if el["source_roots"] != ["../sign_model_pilot_v1/expand/npz"] or el["roles"] != ["train"] \
            or el["signers"] != list(range(1, 10)):
        raise SystemExit("eligibility loader scope differs from the predeclared one")
    roots = [resolve(r) for r in el["source_roots"]]
    coins = T.make_coins(int(p["coins"]["seed"]), tuple(p["coins"]["shape"]))

    with np.load(PRIOR / "train_prefixes.npz", allow_pickle=False) as z:
        if z["classes"].tolist() != classes:
            raise SystemExit("prior cache class order differs")
        if z["base_y"].tolist() != [classes.index(r["label"]) for r in base]:
            raise SystemExit("prior cache base order differs from metadata")
        boff = z["base_offsets"].copy()
        teacher, bidx, cidx = z["teacher_base_logits"].copy(), z["base_indices"].copy(), z["context_indices"].copy()
        sample_pos = sorted({int(v) for v in np.linspace(0, len(rows_idx) - 1, N_SAMPLES).round()})
        sample_rows = [int(rows_idx[k]) for k in sample_pos]
        bp = z["base_prefix"]
        cached = {i: bp[boff[i]:boff[i + 1]].copy() for i in sample_rows}
        del bp
    orig_len = np.diff(boff)
    if len(orig_len) != len(base) or bidx.shape != cidx.shape or bidx.shape != coins.shape:
        raise SystemExit("prior cache shapes differ from the stage1 contract")
    with np.load(pp["paths"]["teacher_scores"], allow_pickle=False) as z:
        if not (np.array_equal(z["parent"], teacher) and z["ids"].tolist() == [r["id"] for r in base]):
            raise SystemExit("prior cache teacher differs from the pinned fixed-parent teacher")
    with np.load(pp["paths"]["schedule"], allow_pickle=False) as z:
        if not (np.array_equal(z["base_indices"], bidx) and np.array_equal(z["pair_indices"], cidx)):
            raise SystemExit("prior cache schedule differs from the pinned schedule")

    failures, prefixes, chunk, sample_x = [], [], [], {}
    shas, n_orig, n_dec, ch_dec, nominal = [], [], [], [], 0

    def flush():
        if chunk and not failures:
            prefixes.extend(G.extract_prefix(parent, chunk))
        chunk.clear()

    for k, i in enumerate(rows_idx.tolist()):
        row = base[i]
        try:
            nat = F.load_native(row, el["roles"], el["signers"], roots)
            xo = build_channels(nat["xy"], nat["present"], nat["ts"])["x"]
            if xo.shape[0] != int(orig_len[i]):
                raise ValueError(f"original channel frames {xo.shape[0]} != cached prefix length {int(orig_len[i])}")
            xy, pr, ts = F.decimate(nat["xy"], nat["present"], nat["ts"], stride=2, offset=0)
            if not F.kept_intervals_are_sums(nat["ts"], ts, 2, 0):
                raise ValueError("kept timestamps are not ts[0::2]")
            xd = build_channels(xy, pr, ts)["x"]
            if not (isinstance(xd, np.ndarray) and xd.dtype == np.float32 and xd.ndim == 2
                    and xd.shape[0] >= 1 and xd.shape[1] == 280 and np.isfinite(xd).all()):
                raise ValueError(f"decimated channels violate the input contract: {getattr(xd, 'shape', None)}")
        except Exception as exc:                                    # every failure halts below; none skipped
            failures.append({"row": i, "id": row.get("id"), "error_class": type(exc).__name__, "error": str(exc)})
            continue
        nominal += bool(F.nominal_dt_doubles(nat["ts"], ts))
        shas.append(nat["sha256"])
        n_orig.append(len(nat["ts"]))
        n_dec.append(len(ts))
        ch_dec.append(int(xd.shape[0]))
        if i in cached:
            sample_x[i] = xo
        chunk.append(xd)
        if len(chunk) == 128:
            flush()
            print(json.dumps({"decimated_rows": k + 1, "of": len(rows_idx), "failures": len(failures)}), flush=True)
    flush()

    def stop(status, **extra):
        write_json(PREP_FAILED, {"status": status, "at_kst": now(), "policy_sha256": sha(POLICY),
                                 "training": 0, "cache_written": False, **extra})
        raise SystemExit(status)

    if failures:
        stop("STOPPED_TRAINING_INPUT_FAILURE", failures=failures, eligible_rows=len(rows_idx))

    samples = []
    for i in sample_rows:
        xo, ref = sample_x[i], cached[i]
        xp = lio.native_input(base[i])                               # prior routing (sha checked)
        same_x = xo.shape == xp.shape and bool(np.array_equal(xo, xp))
        pref = G.extract_prefix(parent, [xo])[0]
        same_shape = pref.shape == ref.shape
        err = float(np.max(np.abs(pref - ref))) if same_shape else None
        ok = same_x and same_shape and bool(np.allclose(pref, ref, atol=1e-5, rtol=1e-5))
        samples.append({"row": i, "id": base[i]["id"], "channels_equal_prior_routing": same_x,
                        "prefix_shape_equal": same_shape, "max_abs_prefix_error": err, "ok": ok})
    if not all(s["ok"] for s in samples):
        stop("STOPPED_SAMPLED_ORIGINAL_EQUALITY_FAILED", samples=samples)

    width = int(parent.proj.out_channels)
    if len(prefixes) != len(rows_idx) or any(q.shape != (c, width) for q, c in zip(prefixes, ch_dec)):
        stop("STOPPED_PREFIX_SHAPE_MISMATCH", prefixes=len(prefixes))
    n_orig_a, n_dec_a = np.asarray(n_orig, np.int64), np.asarray(n_dec, np.int64)
    if not np.array_equal(n_dec_a, (n_orig_a + 1) // 2):
        stop("STOPPED_DECIMATED_FRAME_COUNT_RULE")
    packed, offsets = lio.pack(prefixes)
    expected = T.replacement_mask(bidx, coins, eligible)
    arrays = {
        "base_ids": np.array([r["id"] for r in base]),
        "eligible_mask": eligible,
        "eligible_rows": rows_idx,
        "source_sha256": np.array(shas),
        "decimated_prefix": packed.astype(np.float32, copy=False),
        "decimated_offsets": offsets.astype(np.int64, copy=False),
        "native_frames_original": n_orig_a,
        "native_frames_decimated": n_dec_a,
        "channel_frames_decimated": np.asarray(ch_dec, np.int64),
        "coins": coins,
        "coin_seed": np.array(p["coins"]["seed"], dtype=np.int64),
        "transform_stride_offset": np.array([2, 0], dtype=np.int64),
        "classes": np.array(classes),
    }
    with CACHE.open("xb") as fh:
        np.savez_compressed(fh, **arrays)
    result = {
        "status": "PASS",
        "at_kst": now(),
        "policy_sha256": sha(POLICY),
        "code_sha256": code_hashes(),
        "synthetic_tests_sha256": sha(TESTS_EVIDENCE),
        "synthetic_tests_at": tests.get("at_kst"),
        "prior_train_cache_sha256": sha(PRIOR / "train_prefixes.npz"),
        "prior_generated_v1_sha256": sha(PRIOR / "generated_v1.py"),
        "prior_local_io_sha256": sha(PRIOR / "local_io.py"),
        "parent": p["parent"],
        "metadata_sha256": sha(pp["paths"]["metadata"]),
        "cache": str(CACHE),
        "cache_sha256": sha(CACHE),
        "arrays": {k: array_checksum(v) for k, v in arrays.items()},
        "counts": {"base": len(base), "eligible": int(eligible.sum()),
                   "not_eligible_unchanged": int((~eligible).sum()), "pair_context_unchanged": len(meta["pair_context"]),
                   "input_failures": 0, "decimated_prefix_frames": int(len(packed)),
                   "channel_frames_equal_kept_native_frames": int(np.sum(np.asarray(ch_dec) == n_dec_a)),
                   "min_decimated_channel_frames": int(min(ch_dec)),
                   "nominal_dt_doubled_rows_information": nominal},
        "method_counts": dict(Counter(str(r.get("method")) for r in base)),
        "not_eligible_ids": [r["id"] for r, e in zip(base, eligible) if not e],
        "coins": {"generator": p["coins"]["generator"], "true": int(coins.sum()), "shape": list(coins.shape),
                  "expected_candidate_replacements": int(expected.sum()),
                  "expected_distinct_rows_replaced": int(np.unique(bidx[expected]).size)},
        "sampled_original_equality": samples,
        "reads": {"eligible_train_npz": len(rows_idx), "sampled_rows_reread_by_prior_routing": len(samples),
                  "development_or_public_or_mp": 0, "ksl77": 0, "user": 0, "reserved": 0},
        "model_forwards": {"decimated_prefix_extractions": len(prefixes), "sampled_original_prefix_extractions": len(samples)},
        "training": 0,
        "seconds": round(time.time() - started, 2),
    }
    write_json(READY, result)
    print(json.dumps({k: result[k] for k in ("status", "counts", "coins")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
