#!/usr/bin/env python3
"""Offline replay evaluation of exactly three frozen models over one CLOSED session.

Study labels that apply to EVERY output of this tool:
    VISIBLE_FEEDBACK_SINGLE_PERSON_DEVELOPMENT
    NOT_BLINDED
    NOT_EXECUTION_CORRECTNESS

The source model plan was frozen earlier. This session cohort is fixed after
capture and before offline inference -- it was NOT frozen before capture.
No performance numbers are produced from an OPEN session.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time

SCHEMA = "existing_ui_three_model_v1"
ROLES = ("parent", "control", "treatment")
EXPECTED_STEPS = {"parent": 220, "control": 660, "treatment": 660}
STUDY_LABELS = [
    "VISIBLE_FEEDBACK_SINGLE_PERSON_DEVELOPMENT",
    "NOT_BLINDED",
    "NOT_EXECUTION_CORRECTNESS",
]
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
NO_PRED = "__NO_PREDICTION__"
UNCONFIRMED_TOKENS = (None, "", "UNKNOWN")
FAILURE_OUTCOMES = ("CANCELLED", "ERROR", "TIMEOUT")
ARTIFACT_SUFFIXES = (".npz", ".sidecar.json", ".result.json")
EXPECTED_CLASS_COUNT = 93
MANIFEST_NAME = "manifest_LOCAL_ONLY.json"
STARTED_NAME = "execution_started.json"
RESULT_NAME = "result_LOCAL_ONLY.json"
CALLS_NAME = "inference_calls.jsonl"
ERROR_NAME = "fatal_error.json"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def _confined(path, root):
    p, r = _norm(path), _norm(root)
    return p == r or p.startswith(r + os.sep)


def _read_json(path, what):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        raise ValueError("unreadable/malformed JSON for %s at %s: %s" % (what, path, exc))


def _read_jsonl(path, what):
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                rows.append((lineno, json.loads(line)))
    except Exception as exc:
        raise ValueError("unreadable/malformed %s at %s: %s" % (what, path, exc))
    for lineno, obj in rows:
        if not isinstance(obj, dict):
            raise ValueError("%s line %d is not an object" % (what, lineno))
    return rows


def _write_json_exclusive(path, payload):
    blob = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(blob + "\n")


def _safe_id(value, what):
    if not isinstance(value, str) or not SAFE_ID.match(value):
        raise ValueError("unsafe or non-string %s: %r" % (what, value))
    return value


def _finite(value, what):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("%s must be a finite number, got %r" % (what, value))
    return float(value)


def _load_config(config_path):
    cfg = _read_json(config_path, "config")
    required = ("web_root", "pilot_root", "session_id", "participant_id", "minimum_created_epoch",
                "reference_set_id", "words", "models", "protected_sha256", "data_role")
    for key in required:
        if key not in cfg:
            raise ValueError("config missing required key: %s" % key)
    for key in ("web_root", "pilot_root"):
        if not os.path.isabs(cfg[key]) or not os.path.isdir(cfg[key]):
            raise ValueError("config.%s must be an existing absolute local directory" % key)
    _safe_id(cfg["session_id"], "session_id")
    _safe_id(cfg["participant_id"], "participant_id")
    _finite(cfg["minimum_created_epoch"], "config.minimum_created_epoch")
    words = cfg["words"]
    if not isinstance(words, dict) or len(words) != 18:
        raise ValueError("config.words must map exactly 18 word_ids")
    for wid, spec in words.items():
        _safe_id(wid, "word_id")
        if not isinstance(spec, dict) or "label" not in spec or "clip_id" not in spec:
            raise ValueError("config.words[%s] needs label and clip_id" % wid)
        if not isinstance(spec["label"], str) or not spec["label"]:
            raise ValueError("config.words[%s].label must be a non-empty string" % wid)
    labels = [words[w]["label"] for w in words]
    if len(set(labels)) != 18:
        raise ValueError("config.words labels must be unique")
    models = cfg["models"]
    if not isinstance(models, dict) or set(models) != set(ROLES):
        raise ValueError("config.models must map exactly parent/control/treatment")
    for role in ROLES:
        spec = models[role]
        if not isinstance(spec, dict) or not {"path", "sha256", "step"} <= set(spec):
            raise ValueError("config.models[%s] needs path, sha256, step" % role)
        if spec["step"] != EXPECTED_STEPS[role]:
            raise ValueError("config.models[%s].step must be %d" % (role, EXPECTED_STEPS[role]))
        if not os.path.isabs(spec["path"]):
            raise ValueError("config.models[%s].path must be absolute local" % role)
    if not isinstance(cfg["protected_sha256"], dict):
        raise ValueError("config.protected_sha256 must be an object")
    return cfg


def _verify_pins(pins, what):
    for path, digest in sorted(pins.items()):
        if not os.path.isfile(path):
            raise RuntimeError("%s pin vanished: %s" % (what, path))
        actual = _sha256_file(path)
        if actual != digest:
            raise RuntimeError("%s pin hash mismatch: %s (%s != %s)" % (what, path, actual, digest))


def _verify_absent(paths):
    for path in paths:
        if os.path.exists(path):
            raise RuntimeError("path recorded absent at prepare has appeared: %s" % path)


def _collect_reviews(path, session_id, trial_ids, words):
    reviews = {}
    for lineno, row in _read_jsonl(path, "trial_reviews.jsonl"):
        if row.get("session_id") != session_id:
            raise ValueError("trial_reviews line %d: unknown session_id" % lineno)
        tid = row.get("trial_id")
        if tid not in trial_ids:
            raise ValueError("trial_reviews line %d: unknown trial_id %r" % (lineno, tid))
        if row.get("source") != "participant_self_report":
            raise ValueError("trial_reviews line %d: source must be participant_self_report" % lineno)
        rev = row.get("revision")
        if isinstance(rev, bool) or not isinstance(rev, int) or rev <= 0:
            raise ValueError("trial_reviews line %d: revision must be a positive int" % lineno)
        actual = row.get("actual_word_id")
        if actual in UNCONFIRMED_TOKENS:
            actual = None
        elif actual not in words:
            raise ValueError("trial_reviews line %d: actual_word_id %r not a target word" % (lineno, actual))
        entry = {"revision": rev, "actual_word_id": actual,
                 "followed_guidance": row.get("followed_guidance"),
                 "followed_demonstration_form": row.get("followed_demonstration_form"),
                 "note": row.get("note")}
        seen = reviews.setdefault(None, {})
        prev = seen.setdefault(tid, {}).get(rev)
        if prev is None:
            seen[tid][rev] = entry
        elif prev != entry:
            raise ValueError("trial_reviews: conflicting rows at revision %d for %s" % (rev, tid))
    seen = reviews.pop(None, {})
    for tid, by_rev in seen.items():
        reviews[tid] = by_rev[max(by_rev)]
    return reviews


def _collect_failures(path, session_id, trial_ids):
    failures, metadata_rows = {}, 0
    for lineno, row in _read_jsonl(path, "timing_events.jsonl"):
        if row.get("event") != "trial_closed" and row.get("type") != "trial_closed":
            metadata_rows += 1
            continue
        if row.get("session_id") != session_id:
            raise ValueError("timing_events line %d: unknown session_id" % lineno)
        tid = row.get("trial_id")
        if tid not in trial_ids:
            raise ValueError("timing_events line %d: unknown trial_id %r" % (lineno, tid))
        outcome = row.get("outcome", row.get("state"))
        if outcome not in FAILURE_OUTCOMES:
            raise ValueError("timing_events line %d: outcome %r not in %s" % (lineno, outcome, FAILURE_OUTCOMES))
        if tid in failures and failures[tid] != outcome:
            raise ValueError("timing_events: conflicting outcomes for trial %s" % tid)
        failures[tid] = outcome
    return failures, metadata_rows


def _validate_result(doc, tid, cfg, path):
    if not isinstance(doc, dict):
        raise ValueError("result %s is not an object" % path)
    if doc.get("trial_id") != tid or doc.get("session_id") != cfg["session_id"]:
        raise ValueError("result %s identity mismatch" % path)
    if doc.get("origin") != "camera":
        raise ValueError("result %s origin must be 'camera'" % path)
    if doc.get("reference_set_id") != cfg["reference_set_id"]:
        raise ValueError("result %s reference_set_id mismatch" % path)
    obs_sha = doc.get("observation_sha256")
    if not isinstance(obs_sha, str) or not re.match(r"^[0-9a-f]{64}$", obs_sha):
        raise ValueError("result %s observation_sha256 malformed" % path)
    binding = doc.get("reference_binding")
    if not isinstance(binding, dict) or binding.get("status") != "BOUND":
        raise ValueError("result %s reference_binding must be BOUND" % path)
    if binding.get("reference_set_id") != cfg["reference_set_id"]:
        raise ValueError("result %s binding reference_set_id mismatch" % path)
    req = binding.get("requested_word_id")
    if req not in cfg["words"]:
        raise ValueError("result %s requested_word_id %r not a target word" % (path, req))
    if binding.get("demonstration_word_id") != req:
        raise ValueError("result %s requested != demonstration_word_id" % path)
    if binding.get("demonstration_clip_id") != cfg["words"][req]["clip_id"]:
        raise ValueError("result %s demonstration_clip_id mismatch" % path)
    return req, obs_sha, doc.get("attempt_label"), doc.get("timing")


def prepare(config_path, out_dir):
    """Pin one CLOSED session cohort and write the local-only manifest."""
    config_path = os.path.abspath(config_path)
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        raise ValueError("out_dir must not already exist: %s" % out_dir)
    cfg = _load_config(config_path)
    web_root, sid = cfg["web_root"], cfg["session_id"]
    runs_root = os.path.join(web_root, "runs")
    session_dir = os.path.join(runs_root, sid)
    trials_dir = os.path.join(session_dir, "trials")
    if not _confined(session_dir, runs_root) or not os.path.isdir(session_dir):
        raise ValueError("session directory missing or escapes web_root/runs: %s" % session_dir)

    pins = {config_path: _sha256_file(config_path), _norm(__file__): _sha256_file(__file__)}
    pins[os.path.abspath(__file__)] = _sha256_file(__file__)
    for path, digest in cfg["protected_sha256"].items():
        if not os.path.isabs(path):
            raise ValueError("protected path must be absolute: %s" % path)
        if not os.path.isfile(path):
            raise ValueError("protected path missing: %s" % path)
        if _sha256_file(path) != digest:
            raise ValueError("protected path hash mismatch: %s" % path)
        pins[os.path.abspath(path)] = digest
    for role in ROLES:
        mpath = os.path.abspath(cfg["models"][role]["path"])
        if not _confined(mpath, runs_root):
            raise ValueError("model %s must live below web_root/runs: %s" % (role, mpath))
        if not os.path.isfile(mpath):
            raise ValueError("model %s checkpoint missing: %s" % (role, mpath))
        if _sha256_file(mpath) != cfg["models"][role]["sha256"]:
            raise ValueError("model %s checkpoint hash mismatch" % role)
        pins[mpath] = cfg["models"][role]["sha256"]

    session_path = os.path.join(session_dir, "session.json")
    session = _read_json(session_path, "session.json")
    pins[session_path] = _sha256_file(session_path)
    if session.get("session_id") != sid or session.get("participant_id") != cfg["participant_id"]:
        raise ValueError("session identity does not match config")
    if session.get("purpose") not in ("development", "frozen_retest"):
        raise ValueError("session purpose must be development or frozen_retest")
    if session.get("state") != "CLOSED":
        raise ValueError("session state must be CLOSED; no performance outputs from an OPEN session")
    if session.get("camera") is not False or session.get("store_video") is not False:
        raise ValueError("session must record camera=false and store_video=false")
    created = _finite(session.get("created_epoch"), "session.created_epoch")
    if created <= float(cfg["minimum_created_epoch"]):
        raise ValueError("session created_epoch is not newer than minimum_created_epoch")
    trial_ids = session.get("trial_ids")
    if not isinstance(trial_ids, list):
        raise ValueError("session trial_ids must be a list")
    for tid in trial_ids:
        _safe_id(tid, "trial_id")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("session trial_ids contains duplicates")

    absent = []
    reviews_path = os.path.join(session_dir, "trial_reviews.jsonl")
    if os.path.isfile(reviews_path):
        reviews = _collect_reviews(reviews_path, sid, set(trial_ids), cfg["words"])
        pins[reviews_path] = _sha256_file(reviews_path)
    else:
        reviews, _ = {}, absent.append(reviews_path)
    timing_path = os.path.join(session_dir, "timing_events.jsonl")
    if os.path.isfile(timing_path):
        failures, timing_metadata_rows = _collect_failures(timing_path, sid, set(trial_ids))
        pins[timing_path] = _sha256_file(timing_path)
    else:
        failures, timing_metadata_rows = {}, 0
        absent.append(timing_path)

    accounted = set()
    for tid in trial_ids:
        for suffix in ARTIFACT_SUFFIXES:
            accounted.add(_norm(os.path.join(trials_dir, tid + suffix)))
    if os.path.isdir(trials_dir):
        for name in sorted(os.listdir(trials_dir)):
            full = os.path.join(trials_dir, name)
            if not os.path.isfile(full) or not name.endswith(ARTIFACT_SUFFIXES):
                continue
            if _norm(full) not in accounted:
                raise ValueError("unaccounted artifact for unknown trial id: %s" % full)

    rows = []
    for index, tid in enumerate(trial_ids):
        rpath = os.path.join(trials_dir, tid + ".result.json")
        spath = os.path.join(trials_dir, tid + ".sidecar.json")
        npath = os.path.join(trials_dir, tid + ".npz")
        for p in (rpath, spath, npath):
            if not _confined(p, session_dir):
                raise ValueError("observation path escapes the session directory: %s" % p)
        requested = attempt_label = server_timing = None
        result_sha = sidecar_sha = npz_sha = None
        if os.path.isfile(rpath):
            requested, result_sha, attempt_label, server_timing = _validate_result(
                _read_json(rpath, "result " + tid), tid, cfg, rpath)
            pins[rpath] = _sha256_file(rpath)
        else:
            absent.append(rpath)
        if os.path.isfile(spath):
            side = _read_json(spath, "sidecar " + tid)
            if not isinstance(side, dict):
                raise ValueError("sidecar %s is not an object" % spath)
            sidecar_sha = side.get("observation_sha256")
            if not isinstance(sidecar_sha, str) or not re.match(r"^[0-9a-f]{64}$", sidecar_sha):
                raise ValueError("sidecar %s observation_sha256 malformed" % spath)
            pins[spath] = _sha256_file(spath)
        else:
            absent.append(spath)
        if os.path.isfile(npath):
            npz_sha = _sha256_file(npath)  # hash only; never parsed here
            pins[npath] = npz_sha
        else:
            absent.append(npath)

        have = (result_sha is not None, sidecar_sha is not None, npz_sha is not None)
        if not any(have):
            artifact_status = "MISSING_ALL"
        elif result_sha is None:
            artifact_status = "MISSING_RESULT"
        elif sidecar_sha is None:
            artifact_status = "MISSING_SIDECAR"
        elif npz_sha is None:
            artifact_status = "MISSING_NPZ"
        elif not (result_sha == sidecar_sha == npz_sha):
            artifact_status = "OBSERVATION_SHA_MISMATCH"
        else:
            artifact_status = "COMPLETE"
        eligible = artifact_status == "COMPLETE"

        if result_sha is not None and tid in failures:
            raise ValueError("trial %s has both a result and a failure event" % tid)
        if result_sha is not None:
            terminal = "FEEDBACK"
        elif tid in failures:
            terminal = failures[tid]
        else:
            terminal = "UNVERIFIED_TERMINAL"

        review = reviews.get(tid)
        actual = review["actual_word_id"] if review else None
        rows.append({
            "trial_id": tid, "order_index": index,
            "artifact_status": artifact_status, "eligible": eligible, "terminal": terminal,
            "requested_word_id": requested,
            "actual_word_id": actual,
            "actual_label": cfg["words"][actual]["label"] if actual else None,
            "truth_known": actual is not None,
            "actual_differs_from_requested": (
                None if (actual is None or requested is None) else actual != requested),
            "review_revision": review["revision"] if review else None,
            "followed_guidance": review["followed_guidance"] if review else None,
            "followed_demonstration_form": review["followed_demonstration_form"] if review else None,
            "review_note_present": bool(review and review.get("note")),
            "attempt_label": attempt_label,
            "server_timing_metadata": server_timing,
            "observation_sha256": result_sha if eligible else None,
            "npz_path": npath if npz_sha else None,
        })

    manifest = {
        "schema": SCHEMA,
        "study_labels": STUDY_LABELS,
        "cohort_fixed": "AFTER_CAPTURE_BEFORE_OFFLINE_INFERENCE",
        "model_plan_frozen": "EARLIER_SOURCE_PLAN_ONLY",
        "data_role": cfg["data_role"],
        "config_path": config_path,
        "config": cfg,
        "session_purpose": session["purpose"],
        "session_created_epoch": created,
        "roots": {"web_root": web_root, "pilot_root": cfg["pilot_root"],
                  "runs_root": runs_root, "session_dir": session_dir, "out_dir": out_dir},
        "constraints": {"observations_confined_to": session_dir, "models_below": runs_root,
                        "output_dir": out_dir, "network": "FORBIDDEN", "npz_parsed_at_prepare": False},
        "input_hashes": {p: d for p, d in sorted(pins.items())},
        "absent_paths": sorted(set(absent)),
        "timing_metadata_rows": timing_metadata_rows,
        "rows": rows,
        "counts": {
            "all_attempts": len(rows),
            "eligible_n": sum(1 for r in rows if r["eligible"]),
            "confirmed_n": sum(1 for r in rows if r["truth_known"]),
            "unconfirmed_n": sum(1 for r in rows if not r["truth_known"]),
            "target_words": len(cfg["words"]),
        },
    }
    os.makedirs(out_dir)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    _write_json_exclusive(manifest_path, manifest)
    return manifest_path


def _register_arms(neural, cfg, manifest_sha):
    keys = {}
    for role in ROLES:
        spec = cfg["models"][role]
        key = "offline_replay_%s_%s" % (role, manifest_sha[:12])
        if key in neural.MODELS:
            raise RuntimeError("temporary registry key already present: %s" % key)
        neural.MODELS[key] = {
            "path": spec["path"], "sha256": spec["sha256"], "step": spec["step"],
            "project_words": dict(neural.PROJECT_18), "model_id": key,
            "role": "GENERIC_OFFLINE_ARM", "status_label": "GENERIC_OFFLINE_ARM",
        }
        keys[role] = key
    return keys


def _normalize_raw(raw, role, spec, allowed18, allowed93, outer_ms):
    if not isinstance(raw, dict):
        raise RuntimeError("recognise(%s) did not return a dict" % role)
    if raw.get("checkpoint_sha256") != spec["sha256"]:
        raise RuntimeError("recognise(%s) checkpoint_sha256 mismatch" % role)
    if raw.get("accepted") is not None:
        raise RuntimeError("recognise(%s) accepted must be None" % role)
    if raw.get("execution_assessment") != "UNCONFIGURED":
        raise RuntimeError("recognise(%s) execution_assessment must be UNCONFIGURED" % role)
    status = raw.get("model_status")
    out = {"status": status, "project18": None, "full93": None,
           "elapsed_ms": outer_ms, "model_reported_ms": None, "error": None}
    reported = raw.get("elapsed_ms")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool) and math.isfinite(float(reported)):
        out["model_reported_ms"] = float(reported)
    if status != "READY":
        out["status"] = "ERROR"
        out["error"] = "MODEL_STATUS_%s" % status
        return out
    scopes = raw.get("scopes")
    if not isinstance(scopes, dict) or not isinstance(scopes.get("project18"), dict) \
            or not isinstance(scopes.get("full"), dict):
        raise RuntimeError("recognise(%s) scopes malformed" % role)
    p18 = scopes["project18"].get("top1")
    p93 = scopes["full"].get("top1")
    if p18 is not None and p18 not in allowed18:
        raise RuntimeError("recognise(%s) project18 top1 %r outside target words" % (role, p18))
    if p93 is not None and p93 not in allowed93:
        raise RuntimeError("recognise(%s) full top1 %r outside class list" % (role, p93))
    out["project18"], out["full93"] = p18, p93
    return out


def execute(manifest_path):
    """Re-verify pins, run the three frozen arms once per eligible observation, write results."""
    manifest_path = os.path.abspath(manifest_path)
    manifest = _read_json(manifest_path, "manifest")
    if manifest.get("schema") != SCHEMA:
        raise ValueError("manifest schema must be %s" % SCHEMA)
    out_dir = os.path.dirname(manifest_path)
    if _norm(manifest["roots"]["out_dir"]) != _norm(out_dir):
        raise ValueError("manifest must live in its own recorded out_dir")
    started_path = os.path.join(out_dir, STARTED_NAME)
    result_path = os.path.join(out_dir, RESULT_NAME)
    for path in (started_path, result_path):
        if os.path.exists(path):
            raise RuntimeError("refusing to re-run; already present: %s" % path)

    cfg = manifest["config"]
    manifest_sha = _sha256_file(manifest_path)
    try:
        _verify_pins(manifest["input_hashes"], "input")
        _verify_absent(manifest["absent_paths"])
        rows = manifest["rows"]
        eligible = [r for r in rows if r["eligible"]]
        if not eligible:
            raise RuntimeError("0 eligible observations; refusing to load any model")

        _write_json_exclusive(started_path, {
            "schema": SCHEMA, "manifest_sha256": manifest_sha,
            "started_epoch": time.time(), "study_labels": STUDY_LABELS,
            "eligible_n": len(eligible), "models_loaded": False,
        })

        try:
            import torch
            for setter, value in (("set_num_threads", 2), ("set_num_interop_threads", 2)):
                fn = getattr(torch, setter, None)
                if callable(fn):
                    try:
                        fn(value)
                    except Exception:
                        pass
        except Exception:
            pass

        for root in (os.path.join(cfg["web_root"], "src"), os.path.join(cfg["pilot_root"], "src")):
            if root not in sys.path:
                sys.path.insert(0, root)
        from wis_lab.service import neural
        from wis_lab.core.observation import Observation

        if dict(neural.PROJECT_18) != {w: s["label"] for w, s in cfg["words"].items()}:
            raise RuntimeError("adapter PROJECT_18 does not equal config.words mapping")
        registry_snapshot = copy.deepcopy(neural.MODELS)
        default_snapshot = copy.deepcopy(neural.DEFAULT_MODEL)
        allowed18 = set(cfg["words"])
        labels18 = set(s["label"] for s in cfg["words"].values())
        calls_path = os.path.join(out_dir, CALLS_NAME)
        predictions, load_failures, calls_made = {}, [], 0
        classes_ref = None
        offline_start = time.perf_counter()
        try:
            keys = _register_arms(neural, cfg, manifest_sha)
            for role in ROLES:
                info = neural._load(keys[role])
                if not isinstance(info, dict) or info.get("status") != "READY":
                    raise RuntimeError("arm %s failed to load READY" % role)
                if info.get("checkpoint_sha256") != cfg["models"][role]["sha256"]:
                    raise RuntimeError("arm %s loaded checkpoint hash mismatch" % role)
                if info.get("checkpoint_step") != EXPECTED_STEPS[role]:
                    raise RuntimeError("arm %s loaded checkpoint step mismatch" % role)
                classes = info.get("classes")
                if not isinstance(classes, list) or len(classes) != EXPECTED_CLASS_COUNT \
                        or len(set(classes)) != EXPECTED_CLASS_COUNT:
                    raise RuntimeError("arm %s classes must be %d unique labels" % (role, EXPECTED_CLASS_COUNT))
                if not labels18 <= set(classes):
                    raise RuntimeError("arm %s classes missing target labels" % role)
                if classes_ref is None:
                    classes_ref = list(classes)
                elif list(classes) != classes_ref:
                    raise RuntimeError("arm %s class order differs across arms" % role)
            allowed93 = set(classes_ref)

            with open(calls_path, "a", encoding="utf-8") as calls_fh:
                for row in eligible:
                    tid = row["trial_id"]
                    try:
                        obs = Observation.load(row["npz_path"])
                    except Exception as exc:
                        load_failures.append({"trial_id": tid, "error": repr(exc)})
                        predictions[tid] = {r: {"status": "ERROR", "project18": None, "full93": None,
                                                "elapsed_ms": 0.0, "model_reported_ms": None,
                                                "error": "OBSERVATION_LOAD_FAILED"} for r in ROLES}
                        calls_fh.write(json.dumps({"event": "observation_load_error", "trial_id": tid,
                                                   "recognise_calls": 0}, allow_nan=False) + "\n")
                        calls_fh.flush()
                        continue
                    arm_out = {}
                    for role in ROLES:
                        t0 = time.perf_counter()
                        try:
                            raw = neural.recognise(obs, keys[role])
                        except Exception as exc:
                            arm_out[role] = {"status": "ERROR", "project18": None, "full93": None,
                                             "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                                             "model_reported_ms": None,
                                             "error": "RECOGNISER_RUNTIME_ERROR: %r" % (exc,)}
                        else:
                            arm_out[role] = _normalize_raw(
                                raw, role, cfg["models"][role], allowed18, allowed93,
                                (time.perf_counter() - t0) * 1000.0)
                        calls_made += 1
                        calls_fh.write(json.dumps({
                            "event": "recognise", "trial_id": tid, "role": role,
                            "status": arm_out[role]["status"],
                            "project18_top1": arm_out[role]["project18"],
                            "full93_top1": arm_out[role]["full93"],
                            "elapsed_ms": arm_out[role]["elapsed_ms"],
                        }, allow_nan=False) + "\n")
                        calls_fh.flush()
                    predictions[tid] = arm_out
        finally:
            neural.MODELS.clear()
            neural.MODELS.update(registry_snapshot)
            neural.DEFAULT_MODEL = default_snapshot

        total_ms = (time.perf_counter() - offline_start) * 1000.0
        _verify_pins(manifest["input_hashes"], "post-execution input")
        _verify_absent(manifest["absent_paths"])
        summary = summarize(rows, predictions, {w: s["label"] for w, s in cfg["words"].items()})
        _write_json_exclusive(result_path, {
            "schema": SCHEMA,
            "status": "EXECUTED_PENDING_INDEPENDENT_RECOUNT",
            "study_labels": STUDY_LABELS,
            "cohort_fixed": manifest["cohort_fixed"],
            "model_plan_frozen": manifest["model_plan_frozen"],
            "manifest_sha256": manifest_sha,
            "data_role": cfg["data_role"],
            "models": {r: {"path": cfg["models"][r]["path"], "sha256": cfg["models"][r]["sha256"],
                           "step": cfg["models"][r]["step"]} for r in ROLES},
            "class_space_size": EXPECTED_CLASS_COUNT,
            "rows": rows, "predictions": predictions, "summary": summary,
            "inference_calls": calls_made,
            "load_failures": load_failures,
            "total_offline_milliseconds": total_ms,
            "training_steps": 0, "model_promoted": False,
            "protected_unchanged": True,
            "timing_note": "offline wallclock only; stored server timing is metadata",
        })
        return result_path
    except BaseException as exc:
        try:
            with open(os.path.join(out_dir, ERROR_NAME), "w", encoding="utf-8") as fh:
                json.dump({"schema": SCHEMA, "manifest_sha256": manifest_sha,
                           "error_type": type(exc).__name__, "error": str(exc),
                           "epoch": time.time()}, fh, indent=2, allow_nan=False)
        except Exception:
            pass
        raise


def summarize(rows, predictions, label_map):
    """Independently recount cohort composition and per-arm accuracy. No tuning, no promotion."""
    if not isinstance(rows, list) or not isinstance(predictions, dict) or not isinstance(label_map, dict):
        raise ValueError("summarize expects rows list, predictions dict, label_map dict")
    if len(label_map) != 18:
        raise ValueError("label_map must contain exactly 18 target words")
    by_id = {}
    for row in rows:
        tid = row["trial_id"]
        if tid in by_id:
            raise ValueError("duplicate trial_id in rows: %s" % tid)
        by_id[tid] = row
    eligible_ids = set(t for t, r in by_id.items() if r["eligible"])
    for tid, arms in predictions.items():
        if tid not in by_id:
            raise ValueError("prediction for unknown trial: %s" % tid)
        if tid not in eligible_ids:
            raise ValueError("prediction supplied for ineligible trial: %s" % tid)
        if not isinstance(arms, dict) or set(arms) != set(ROLES):
            raise ValueError("prediction for %s must supply exactly the three roles" % tid)
        for role, arm in arms.items():
            if not isinstance(arm, dict) or "status" not in arm:
                raise ValueError("prediction %s/%s malformed" % (tid, role))
            ms = _finite(arm.get("elapsed_ms"), "elapsed_ms for %s/%s" % (tid, role))
            if ms < 0:
                raise ValueError("negative elapsed_ms for %s/%s" % (tid, role))
            if arm["status"] != "READY" and (arm.get("project18") or arm.get("full93")):
                raise ValueError("non-READY arm %s/%s cannot supply a prediction" % (tid, role))

    labels = dict(label_map)
    confirmed = [r for r in rows if r["truth_known"]]
    scopes = {}
    for scope, key in (("project18", "project18"), ("full93", "full93")):
        per_role = {}
        for role in ROLES:
            correct, failures, confusion = 0, 0, {}
            per_class = {w: {"support": 0, "correct": 0, "recall": None} for w in labels}
            tp = {w: 0 for w in labels}
            fp = {w: 0 for w in labels}
            fn = {w: 0 for w in labels}
            for row in confirmed:
                actual = row["actual_word_id"]
                truth = actual if scope == "project18" else labels[actual]
                arm = predictions.get(row["trial_id"], {}).get(role)
                pred = arm.get(key) if arm and arm.get("status") == "READY" else None
                if pred is None:
                    failures += 1
                per_class[actual]["support"] += 1
                bucket = confusion.setdefault(truth, {})
                token = NO_PRED if pred is None else pred
                bucket[token] = bucket.get(token, 0) + 1
                if pred is not None and pred == truth:
                    correct += 1
                    per_class[actual]["correct"] += 1
                    tp[actual] += 1
                else:
                    fn[actual] += 1
                    if pred is not None:
                        inv = {v: k for k, v in labels.items()}
                        pw = pred if scope == "project18" else inv.get(pred)
                        if pw in fp:
                            fp[pw] += 1
            for w, stats in per_class.items():
                if stats["support"]:
                    stats["recall"] = stats["correct"] / stats["support"]
            supported = [w for w in labels if per_class[w]["support"] > 0]
            f1s = []
            for w in supported:
                prec_d, rec_d = tp[w] + fp[w], tp[w] + fn[w]
                prec = tp[w] / prec_d if prec_d else 0.0
                rec = tp[w] / rec_d if rec_d else 0.0
                f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
            per_role[role] = {
                "correct": correct, "confirmed_n": len(confirmed),
                "accuracy": (correct / len(confirmed)) if confirmed else None,
                "inference_failures_on_confirmed": failures,
                "confusion": confusion, "per_class": per_class,
                "macro_f1": (sum(f1s) / len(f1s)) if f1s else None,
                "macro_f1_class_count": len(f1s),
                "macro_f1_classes": supported,
                "out_of_target18_predictions_counted_as_errors": scope == "full93",
            }
        paired = {}
        for name, (a, b) in (("treatment_vs_control", ("treatment", "control")),
                             ("treatment_vs_parent", ("treatment", "parent")),
                             ("control_vs_parent", ("control", "parent"))):
            gains, regress, changes, both = [], [], [], 0
            for row in confirmed:
                arms = predictions.get(row["trial_id"])
                if not arms:
                    continue
                truth = row["actual_word_id"] if scope == "project18" else labels[row["actual_word_id"]]
                pa = arms[a].get(key) if arms[a].get("status") == "READY" else None
                pb = arms[b].get(key) if arms[b].get("status") == "READY" else None
                if pa is not None and pb is not None:
                    both += 1
                if pa != pb:
                    changes.append(row["trial_id"])
                if pa == truth and pb != truth:
                    gains.append(row["trial_id"])
                elif pb == truth and pa != truth:
                    regress.append(row["trial_id"])
            paired[name] = {"gains": gains, "regressions": regress,
                            "prediction_changes": changes, "both_answered": both}
        scopes[scope] = {"per_role": per_role, "paired": paired}

    def _tally(field):
        out = {}
        for row in rows:
            out[str(row.get(field))] = out.get(str(row.get(field)), 0) + 1
        return out

    return {
        "study_labels": STUDY_LABELS,
        "all_attempts": len(rows),
        "confirmed_n": len(confirmed),
        "unconfirmed_n": len(rows) - len(confirmed),
        "eligible_n": len(eligible_ids),
        "terminal_counts": _tally("terminal"),
        "artifact_status_counts": _tally("artifact_status"),
        "coverage_by_actual": {w: sum(1 for r in confirmed if r["actual_word_id"] == w) for w in labels},
        "coverage_by_requested": {w: sum(1 for r in rows if r["requested_word_id"] == w) for w in labels},
        "unknown_truth_trials": [r["trial_id"] for r in rows if not r["truth_known"]],
        "missing_prediction_trials": sorted(eligible_ids - set(predictions)),
        "ineligible_trials": [r["trial_id"] for r in rows if not r["eligible"]],
        "attempt_label_counts": _tally("attempt_label"),
        "followed_guidance_counts": _tally("followed_guidance"),
        "followed_demonstration_form_counts": _tally("followed_demonstration_form"),
        "descriptive_metadata_note": "form/guidance/attempt labels are descriptive only; no quality verdict",
        "scopes": scopes,
        "zero_confirmed_metrics_are_null": not confirmed,
        "tuning_or_promotion": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline three-model replay of one CLOSED session.")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", required=True)
    prep.add_argument("--out-dir", required=True)
    ex = sub.add_parser("execute")
    ex.add_argument("--manifest", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            path = prepare(args.config, args.out_dir)
            manifest = _read_json(path, "manifest")
            counts = manifest["counts"]
            print("PREPARED %s" % SCHEMA)
            print("labels: %s" % ", ".join(STUDY_LABELS))
            print("attempts=%d eligible=%d confirmed=%d unconfirmed=%d target_words=%d" % (
                counts["all_attempts"], counts["eligible_n"], counts["confirmed_n"],
                counts["unconfirmed_n"], counts["target_words"]))
        else:
            path = execute(args.manifest)
            result = _read_json(path, "result")
            s = result["summary"]
            print("EXECUTED_PENDING_INDEPENDENT_RECOUNT")
            print("labels: %s" % ", ".join(STUDY_LABELS))
            print("attempts=%d eligible=%d confirmed=%d calls=%d load_failures=%d" % (
                s["all_attempts"], s["eligible_n"], s["confirmed_n"],
                result["inference_calls"], len(result["load_failures"])))
    except Exception as exc:
        sys.stderr.write("%s: %s\n" % (type(exc).__name__, exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
