"""wis_lab entry points: doctor / manifest / extract / replay / serve / report.

Run from research/front_keyphase_v1:
    py -3.13 -m wis_lab.cli doctor          (needs src on sys.path; use run.py)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))
CONFIG_DIR = os.path.join(ROOT, "config")
CACHE_DIR = os.path.join(ROOT, "cache")
EVIDENCE_DIR = os.path.join(ROOT, "evidence")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "Tools", "videos")
LEGACY_JSON_DIR = os.path.join(
    os.path.expanduser("~"), "AppData", "LocalLow", "DefaultCompany", "WordsInSilence", "Recordings")


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> str:
    from .core.observation import sha256_file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return sha256_file(path)


# ---------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    import importlib
    report = {"checks": [], "ok": True}

    def check(name, fn):
        try:
            detail = fn()
            report["checks"].append({"name": name, "status": "OK", "detail": detail})
        except Exception as exc:  # noqa: BLE001 - doctor reports, never raises
            report["checks"].append({"name": name, "status": "FAIL",
                                     "detail": f"{type(exc).__name__}: {exc}"})
            report["ok"] = False

    check("python", lambda: platform.python_version())
    for mod in ("numpy", "cv2", "mediapipe", "fastapi", "uvicorn", "av"):
        check(f"import:{mod}",
              lambda m=mod: getattr(importlib.import_module(m), "__version__", "?"))

    from .capture.landmarker import DEFAULT_HAND_MODEL, DEFAULT_POSE_MODEL
    from .core.observation import sha256_file

    for label, path in (("hand", DEFAULT_HAND_MODEL), ("pose", DEFAULT_POSE_MODEL)):
        check(f"model:{label}", lambda p=path: {
            "path": p.replace("\\", "/"), "bytes": os.path.getsize(p),
            "sha256": sha256_file(p)})

    check("contract", lambda: load_json(os.path.join(CONFIG_DIR, "contract.json"))["contract_id"])
    check("words", lambda: len(load_json(os.path.join(CONFIG_DIR, "words.json"))["words"]))
    check("video_dir", lambda: {
        "path": VIDEO_DIR.replace("\\", "/"),
        "mp4_count": len([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")])})
    check("legacy_json_dir", lambda: {
        "path": LEGACY_JSON_DIR.replace("\\", "/"),
        "json_count": len([f for f in os.listdir(LEGACY_JSON_DIR) if f.endswith(".json")])})

    if not args.quiet:
        for c in report["checks"]:
            print(f"{c['status']:4} {c['name']}: {c['detail']}")
        print("doctor:", "OK" if report["ok"] else "FAIL")
    if args.out:
        write_json(args.out, report)
    return 0 if report["ok"] else 1


# -------------------------------------------------------------- manifest
def cmd_manifest(args) -> int:
    from .capture.video_source import probe
    from .core.observation import sha256_file

    words = load_json(os.path.join(CONFIG_DIR, "words.json"))
    entries = []
    for w in words["words"]:
        name = w["display_name"]
        path = os.path.join(VIDEO_DIR, name + ".mp4")
        legacy = os.path.join(LEGACY_JSON_DIR, name + ".json")
        legacy_slow = os.path.join(LEGACY_JSON_DIR, name + "_slow.json")
        meta = probe(path)
        entries.append({
            "clip_id": f"NIKL_{w['word_id']}_01",
            "word_id": w["word_id"],
            "display_name": name,
            "aliases": w["aliases"],
            "cue_flags": w["cue_flags"],
            "source_kind": "nikl_interpreter_word_clip",
            "source_verification": "user_report",
            "source_url": None,
            "absolute_path": path.replace("\\", "/"),
            "sha256": sha256_file(path),
            "bytes": os.path.getsize(path),
            "frame_count": meta["frame_count"],
            "width": meta["width"],
            "height": meta["height"],
            "time_base": meta["time_base"],
            "avg_rate": meta["avg_rate"],
            "codec": meta["codec"],
            "pix_fmt": meta["pix_fmt"],
            "duration_ms": meta["duration_ms"],
            "extractor_id": None,
            "legacy_json_path": legacy.replace("\\", "/") if os.path.exists(legacy) else None,
            "legacy_json_slow_path": legacy_slow.replace("\\", "/") if os.path.exists(legacy_slow) else None,
            "legacy_contract_match": False,
            "legacy_contract_mismatch_reason":
                "schemaVersion=2 stores normalized xy only (no pixel xy, no frame_id, no observed/usable "
                "masks, no per-hand score) and TimestampMs sits on a 33 ms grid rather than container PTS; "
                "hand side assignment is unrecorded. Not usable as a wis_observation_2d_v1 cache.",
            "readiness": "PENDING_EXTRACTION",
            "readiness_reason": "quality not measured yet",
            "observation_npz": None,
            "observation_sha256": None,
        })

    manifest = {
        "manifest_id": "nikl_front_19_v1",
        "created_date_kst": time.strftime("%Y-%m-%d"),
        "contract_id": load_json(os.path.join(CONFIG_DIR, "contract.json"))["contract_id"],
        "word_set_id": words["word_set_id"],
        "clip_count": len(entries),
        "source_dir": VIDEO_DIR.replace("\\", "/"),
        "source_note": "originals are read in place and never copied or modified",
        "legacy_json_dir": LEGACY_JSON_DIR.replace("\\", "/"),
        "legacy_json_role": "preserved cross-check only; not a cache for this contract",
        "excluded_from_this_baseline": [
            "data/raw/sessions/S001_P001 three-view own recordings",
            "past personal recordings and F59 game data",
            "AIHub continuous-sentence cuts",
        ],
        "clips": entries,
    }
    out = args.out or os.path.join(CONFIG_DIR, "source_manifest.json")
    digest = write_json(out, manifest)
    print(f"manifest: {len(entries)} clips -> {out} sha256={digest}")
    return 0


# --------------------------------------------------------------- extract
def cmd_extract(args) -> int:
    from .capture.landmarker import Landmarker
    from .capture.video_source import extract

    contract = load_json(os.path.join(CONFIG_DIR, "contract.json"))
    manifest_path = os.path.join(CONFIG_DIR, "source_manifest.json")
    manifest = load_json(manifest_path)
    words = load_json(os.path.join(CONFIG_DIR, "words.json"))
    gate = words["readiness_rule"]["quality_gate"]

    wanted = set(args.clip or [])
    results = []
    for clip in manifest["clips"]:
        if wanted and clip["clip_id"] not in wanted and clip["word_id"] not in wanted:
            continue
        # one landmarker per clip: VIDEO mode keeps monotonic state per stream
        with Landmarker(contract) as lm:
            extractor_id = lm.extractor_id()
            npz = os.path.join(CACHE_DIR, extractor_id, clip["clip_id"] + ".npz")
            if os.path.exists(npz) and not args.force:
                from .core.observation import Observation
                obs = Observation.load(npz)
                status = "cached"
            else:
                t0 = time.perf_counter()
                obs = extract(clip["absolute_path"], lm, clip["clip_id"])
                obs.sidecar["extract_seconds"] = round(time.perf_counter() - t0, 3)
                obs.save(npz)
                status = "extracted"

        q = obs.quality_summary()  # recomputed, so a cached NPZ still gets the current metrics
        readiness, reason = _readiness(clip["cue_flags"], q, gate)
        clip["extractor_id"] = extractor_id
        clip["observation_npz"] = os.path.relpath(npz, ROOT).replace("\\", "/")
        clip["observation_sha256"] = obs.sidecar["observation_sha256"]
        clip["quality"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in q.items()}
        clip["readiness"] = readiness
        clip["readiness_reason"] = reason
        results.append((clip["display_name"], status, readiness, q["any_hand_frame_rate"]))
        print(f"{clip['display_name']:>8} {status:9} {readiness:17} "
              f"hand={q['any_hand_frame_rate']:.3f} pose={q['pose_frame_rate']:.3f} "
              f"frames={q['frames']}")

    manifest["readiness_counts"] = {
        k: sum(1 for c in manifest["clips"] if c["readiness"] == k)
        for k in ("READY", "NEEDS_REVIEW", "UNSUPPORTED_CUE", "PENDING_EXTRACTION")
    }
    digest = write_json(manifest_path, manifest)
    print("manifest updated:", manifest["readiness_counts"], "sha256=" + digest)
    return 0


def _readiness(cue_flags, q, gate):
    if "head_motion" in cue_flags:
        return ("UNSUPPORTED_CUE",
                "primary cue is head movement; the R1 baseline scores hands and arms only")
    if q["any_hand_frame_rate"] <= 0.0:
        return "UNSUPPORTED_CUE", "no usable hand observation in the clip"
    bad = []
    if q["hand_rate_in_active_span"] < gate["hand_rate_in_active_span_min"]:
        bad.append(f"hand_rate_in_active_span {q['hand_rate_in_active_span']:.3f} "
                   f"< {gate['hand_rate_in_active_span_min']}")
    if q["shoulder_frame_rate"] < gate["shoulder_frame_rate_min"]:
        bad.append(f"shoulder_frame_rate {q['shoulder_frame_rate']:.3f} < {gate['shoulder_frame_rate_min']}")
    soft = [f for f in cue_flags if f in ("non_manual_possible", "low_discriminability_risk")]
    if bad or soft:
        return "NEEDS_REVIEW", "; ".join(bad + [f"cue flag: {f}" for f in soft])
    return "READY", "quality gate passed and all cue flags are inside the contract"


# ------------------------------------------------------ references / replay
CANDIDATE_READINESS = ("READY", "NEEDS_REVIEW")


def load_reference_set(set_id: str = "nikl19_uniform32_v1", *, contract: dict | None = None):
    """Build the candidate set from the observation cache. Returns (refs, lineage)."""
    from .core.observation import Observation
    from .core.recognize import build_reference

    if contract is None:
        contract = load_json(os.path.join(CONFIG_DIR, "contract.json"))
    manifest = load_json(os.path.join(CONFIG_DIR, "source_manifest.json"))
    budget = int(contract["sampling_A"]["budget"])

    refs, excluded = [], []
    for clip in manifest["clips"]:
        if clip["readiness"] not in CANDIDATE_READINESS:
            excluded.append({"word_id": clip["word_id"], "readiness": clip["readiness"],
                             "reason": clip["readiness_reason"]})
            continue
        obs = Observation.load(os.path.join(ROOT, clip["observation_npz"]))
        refs.append(build_reference(obs, clip["clip_id"], clip["word_id"], budget))
    lineage = {
        "set_id": set_id,
        "contract_id": contract["contract_id"],
        "extractor_id": manifest["clips"][0]["extractor_id"],
        "selector": f"uniform_time:{budget}",
        "candidate_count": len(refs),
        "candidate_word_ids": [r["word_id"] for r in refs],
        "excluded": excluded,
        "note": "one exemplar per word. A template set, not a trained 19-word model.",
    }
    return refs, lineage


def cmd_references(args) -> int:
    refs, lineage = load_reference_set()
    out = os.path.join(ROOT, "references", lineage["set_id"], "set.json")
    digest = write_json(out, lineage)
    print(f"reference set {lineage['set_id']}: {lineage['candidate_count']} candidates "
          f"(excluded {len(lineage['excluded'])}) sha256={digest}")
    for e in lineage["excluded"]:
        print(f"  excluded {e['word_id']}: {e['readiness']} - {e['reason']}")
    return 0


def cmd_replay(args) -> int:
    """File in, ranking out. The same core the HTTP layer will call."""
    from .core.observation import Observation
    from .core.recognize import predict, task_match

    contract = load_json(os.path.join(CONFIG_DIR, "contract.json"))
    manifest = load_json(os.path.join(CONFIG_DIR, "source_manifest.json"))
    refs, lineage = load_reference_set()

    npz = args.observation
    if not os.path.exists(npz):
        match = [c for c in manifest["clips"]
                 if args.observation in (c["clip_id"], c["word_id"], c["display_name"])]
        if not match:
            print(f"no registered observation for {args.observation!r}", file=sys.stderr)
            return 2
        npz = os.path.join(ROOT, match[0]["observation_npz"])

    from .core.observation import sha256_file
    obs = Observation.load(npz)
    result = predict(obs, refs, contract)
    result["observation_sha256"] = sha256_file(npz)
    result["reference_set_id"] = lineage["set_id"]
    result["contract_id"] = contract["contract_id"]
    result["extractor_id"] = lineage["extractor_id"]
    result["origin"] = "replay"
    if args.requested_word:
        result["task"] = task_match(result, args.requested_word)

    if args.json:
        write_json(args.json, result)
    print(f"status={result['observation_status']} recognition={result['recognition']} "
          f"execution_assessment={result['execution_assessment']}")
    if result["observation_status"] != "OK":
        print("  reason:", result["insufficient_reason"])
        return 0
    print(f"  top1={result['top1_word_id']} d={result['top1_distance']} "
          f"margin_to_second={result['margin_to_second']}")
    for r in result["ranking"][:args.top]:
        print(f"   {r.get('rank', '-'):>2}. {r['word_id']:<14} {r['distance']}")
    hs = result.get("handshape_check") or {}
    if hs:
        print(f"  handshape: {hs['status']}"
              + (f"  d={hs['distance']} (시범과의 관측 차이, 판정 아님)"
                 if hs.get("distance") is not None else f"  ({hs.get('reason','')})"))
    if "task" in result:
        print("  task:", result["task"]["task_match"], "rank", result["task"]["requested_rank"])
    return 0


# ----------------------------------------------------------------- report
def cmd_report(args) -> int:
    """Cross-distance matrix and a self-replay summary over the candidate set."""
    import csv

    from .core.dtw import dtw
    from .core.observation import Observation
    from .core.recognize import predict

    contract = load_json(os.path.join(CONFIG_DIR, "contract.json"))
    manifest = load_json(os.path.join(CONFIG_DIR, "source_manifest.json"))
    refs, lineage = load_reference_set()
    out_dir = os.path.join(ROOT, "reports", args.report_id)
    os.makedirs(out_dir, exist_ok=True)
    words = [r["word_id"] for r in refs]

    cost = {k: v for k, v in contract.get("cost", {}).items()
            if k in ("one_sided_penalty", "one_sided_mode", "group_weights", "hand_participation")}
    matrix = [[dtw(a, b, **cost)["distance"] for b in refs] for a in refs]
    with open(os.path.join(out_dir, "reference_distance_matrix.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["word_id"] + words)
        for name, row in zip(words, matrix):
            w.writerow([name] + [("" if v is None else f"{v:.5f}") for v in row])

    rows = []
    for clip in manifest["clips"]:
        if clip["readiness"] == "UNSUPPORTED_CUE":
            continue
        obs = Observation.load(os.path.join(ROOT, clip["observation_npz"]))
        res = predict(obs, refs, contract)
        nearest = [r for r in res["ranking"] if r["word_id"] != clip["word_id"]
                   and r["distance"] is not None]
        rows.append({
            "word_id": clip["word_id"],
            "readiness": clip["readiness"],
            "self_distance": res.get("top1_distance"),
            "top1_word_id": res.get("top1_word_id"),
            "nearest_other_word": nearest[0]["word_id"] if nearest else "",
            "nearest_other_distance": f"{nearest[0]['distance']:.5f}" if nearest else "",
            "sampled_frames": res["quality_summary"].get("sampled_frames"),
            "hand_rate_in_active_span": round(
                res["quality_summary"]["hand_rate_in_active_span"], 4),
        })
    with open(os.path.join(out_dir, "self_replay_summary.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    off = [matrix[i][j] for i in range(len(refs)) for j in range(len(refs))
           if i != j and matrix[i][j] is not None]
    summary = {
        "report_id": args.report_id,
        "reference_set": lineage,
        "candidate_count": len(refs),
        "self_distance_all_zero": all(r["self_distance"] == 0.0 for r in rows),
        "nearest_other_distance_min": round(min(off), 5),
        "nearest_other_distance_median": round(float(sorted(off)[len(off) // 2]), 5),
        "closest_pair": min(
            ((matrix[i][j], words[i], words[j]) for i in range(len(refs))
             for j in range(len(refs)) if i != j and matrix[i][j] is not None),
            key=lambda x: x[0]),
        "interpretation": (
            "separation between the stored exemplars only. It says nothing about how a human "
            "attempt scores, and there is no pass line: execution_assessment is UNCONFIGURED."),
        "performance_status": "NOT_MEASURED_NO_HUMAN_ATTEMPTS",
    }
    digest = write_json(os.path.join(out_dir, "summary.json"), summary)
    print(f"report {args.report_id}: {len(refs)} candidates, closest pair "
          f"{summary['closest_pair'][1]}~{summary['closest_pair'][2]} "
          f"d={summary['closest_pair'][0]:.5f}, median other {summary['nearest_other_distance_median']}")
    print("  ->", os.path.relpath(out_dir, ROOT).replace("\\", "/"), "sha256(summary)=" + digest)
    return 0



# ------------------------------------------------------------------ serve
def cmd_serve(args) -> int:
    """Start the local API + static test page. Nothing else listens on this port."""
    import uvicorn
    from .service import app as service
    if getattr(args, "contract", None):
        service.store = service.Store(contract_path=args.contract)

    print(f"open http://{args.host}:{args.port}/  (Ctrl+C to stop; the camera opens only when "
          f"you start a session on the page)")
    uvicorn.run(service.app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


# ------------------------------------------------------------------ bands
def cmd_bands(args) -> int:
    """Derive per-word flexion-angle bands from a session of CORRECT attempts.

    Stage 2 of the two-stage design. Nothing is trained: this records two numbers per word and
    the hashes of the recordings they came from.
    """
    import numpy as np

    from .core import features as F
    from .core.dtw import group_residuals
    from .core.recognize import AXIS_GROUPS
    from .core.observation import Observation, sha256_file

    contract = load_json(os.path.join(CONFIG_DIR, "contract.json"))
    cost = {k: v for k, v in contract.get("cost", {}).items()
            if k in ("one_sided_penalty", "one_sided_mode", "group_weights", "hand_participation")}
    refs, lineage = load_reference_set()
    by_word = {r["word_id"]: r for r in refs}
    budget = int(contract["sampling_A"]["budget"])

    run_dir = os.path.join(ROOT, "runs", args.session)
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "decisions.jsonl"), encoding="utf-8")
            if l.strip()]

    collected = {}
    for r in rows:
        w = r["requested_word_id"]
        if w not in by_word:
            continue
        npz = os.path.join(run_dir, "trials", r["trial_id"] + ".npz")
        obs = Observation.load(npz)
        try:
            q = F.uniform_time_sample(F.sequence_features(obs), budget)
        except F.FeatureError:
            continue
        g = group_residuals(q, by_word[w], **cost)["groups"]
        row = {"trial_id": r["trial_id"], "sha256": sha256_file(npz)[:16]}
        for axis, names in AXIS_GROUPS.items():
            vals = [g[k]["mean_distance"] for k in names
                    if g.get(k) and g[k]["mean_distance"] is not None]
            row[axis] = float(np.mean(vals)) if vals else None
        if all(row[a] is None for a in AXIS_GROUPS):
            continue
        collected.setdefault(w, []).append(row)

    axes = {a: {"groups": list(g), "rule": "upper bound only", "bands": {}}
            for a, g in AXIS_GROUPS.items()}
    for w, xs in collected.items():
        if len(xs) < args.min_samples:
            print(f"  skip {w}: only {len(xs)} attempts, minimum is {args.min_samples}")
            continue
        line = f"  {w:<14} n={len(xs):>3}"
        for a in AXIS_GROUPS:
            v = [x[a] for x in xs if x[a] is not None]
            if len(v) < args.min_samples:
                continue
            axes[a]["bands"][w] = {
                "n": len(v), "lo": round(min(v), 5), "hi": round(max(v), 5),
                "median": round(float(np.median(v)), 5), "std": round(float(np.std(v)), 5),
                "trial_sha256": [x["sha256"] for x in xs]}
            line += f"  {a} {min(v):.4f}~{max(v):.4f}"
        print(line)
    bands = axes["handshape"]["bands"]

    out = {
        "bands_id": args.bands_id,
        "created_date_kst": time.strftime("%Y-%m-%d"),
        "contract_id": contract["contract_id"],
        "reference_set_id": lineage["set_id"],
        "extractor_id": lineage["extractor_id"],
        "source_session": args.session,
        "participant": rows[0].get("participant_id") if rows else None,
        "purpose": rows[0].get("purpose") if rows else None,
        "axes": axes,
        "feature": "per axis, the mean group distance to that word's reference",
        "band_rule": "upper bound = max over the correct attempts in this session",
        "covered_words": sorted(bands),
        "uncovered_words": sorted(set(by_word) - set(bands)),
        "nothing_was_trained": "the recogniser, features, weights and DTW are unchanged; these are "
                               "two numbers per word",
        "one_participant": "these bands describe this participant only and are not transferable",
        "no_frozen_retest": "derived from the same recordings they were measured on; a frozen "
                            "re-test with new recordings has not been run",
        "bands": bands,
    }
    path = os.path.join(CONFIG_DIR, "handshape_bands.json")
    digest = write_json(path, out)
    print(f"bands: {len(bands)} words covered, {len(out['uncovered_words'])} uncovered "
          f"-> config/handshape_bands.json sha256={digest}")
    return 0


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wis_lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="environment, models and source presence")
    d.add_argument("--out")
    d.add_argument("--quiet", action="store_true")
    d.set_defaults(func=cmd_doctor)

    m = sub.add_parser("manifest", help="probe the source clips and write source_manifest.json")
    m.add_argument("--out")
    m.set_defaults(func=cmd_manifest)

    e = sub.add_parser("extract", help="build the observation cache and finalise readiness")
    e.add_argument("--clip", action="append", help="clip_id or word_id; repeatable; default all")
    e.add_argument("--force", action="store_true", help="re-extract even when cached")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("references", help="build and describe the candidate set")
    r.set_defaults(func=cmd_references)

    p = sub.add_parser("replay", help="score one stored observation against the candidate set")
    p.add_argument("observation", help="clip_id, word_id, display name, or a path to an npz")
    p.add_argument("--requested-word", help="exercise word id; scored separately from predict()")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--json", help="write the full result to this path")
    p.set_defaults(func=cmd_replay)

    rp = sub.add_parser("report", help="cross-distance matrix and self-replay summary")
    rp.add_argument("--report-id", default="r1a_baseline")
    rp.set_defaults(func=cmd_report)

    bd = sub.add_parser("bands", help="derive per-word flexion-angle bands from correct attempts")
    bd.add_argument("--session", required=True)
    bd.add_argument("--bands-id", default="handshape_v1")
    bd.add_argument("--min-samples", type=int, default=10)
    bd.set_defaults(func=cmd_bands)

    sv = sub.add_parser("serve", help="run the local API and the static test page")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--log-level", default="info")
    sv.add_argument("--contract", help="explicit candidate contract path; default contract.json is left unchanged")
    sv.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
