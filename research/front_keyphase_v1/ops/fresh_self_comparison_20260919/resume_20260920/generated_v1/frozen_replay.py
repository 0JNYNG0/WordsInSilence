"""frozen_replay.py

Local-only, offline frozen replay driver for a WIS fresh self-capture retest.

prepare : validate + pin every input, build the cohort, emit a manifest.
execute : re-verify the manifest, then run frozen inference exactly once per
          (eligible trial, model role) through the unchanged local adapter.

No downloads, no camera, no subprocesses, no training, no model mutation,
no registry file edits, no reading of other runs.
"""

import sys

sys.dont_write_bytecode = True

import argparse
import copy
import hashlib
import json
import os
import re
import time
import traceback
from datetime import datetime, timedelta, timezone

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
COHORT_PATH = os.path.join(SCRIPT_DIR, "frozen_cohort.py")

ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
ROLES = ("parent", "control", "treatment")
EXPECTED_STEPS = {"parent": 220, "control": 660, "treatment": 660}
ROUNDS = 3
WORD_COUNT = 18
SLOT_COUNT = 54
FULL_CLASS_COUNT = 93
PROTOCOL_SCHEMA = "wis_fresh_self_capture_plan_v1"
JOURNAL_SCHEMA = "wis_frozen_capture_journal_v1"
KST = timezone(timedelta(hours=9))

MANIFEST_NAME = "manifest_LOCAL_ONLY.json"
STARTED_NAME = "execution_started.json"
RESULT_NAME = "result_LOCAL_ONLY.json"
ERROR_NAME = "execution_error.json"
CALLS_NAME = "inference_calls_LOCAL_ONLY.jsonl"


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def fail(msg):
    raise ValueError(msg)


def rp(path):
    return os.path.realpath(os.path.abspath(str(path)))


def safe_id(value, what):
    if not isinstance(value, str) or not ID_RE.match(value):
        fail("unsafe identifier for %s: %r" % (what, value))
    return value


def under(root, path):
    """True when realpath(path) is root itself or strictly inside root."""
    root = rp(root)
    path = rp(path)
    return path == root or path.startswith(root + os.sep)


def require_under(root, path, what):
    if not under(root, path):
        fail("%s escapes its permitted root: %s" % (what, path))
    return rp(path)


def join_id(root, *parts):
    """Join validated id components and re-check containment (symlink safe)."""
    cur = rp(root)
    target = os.path.join(cur, *parts)
    return require_under(cur, target, "path %s" % target)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path, what):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError as exc:
        fail("cannot read %s (%s): %s" % (what, path, exc))
    except json.JSONDecodeError as exc:
        fail("invalid JSON in %s (%s): %s" % (what, path, exc))
    if not isinstance(data, dict):
        fail("%s must be a JSON object: %s" % (what, path))
    return data


def read_jsonl(path, what):
    """Strict JSONL. Missing file -> (False, []). Bad content -> error."""
    if not os.path.exists(path):
        return False, []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError as exc:
        fail("cannot read %s (%s): %s" % (what, path, exc))
    rows = []
    for n, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            fail("invalid JSON at %s:%d: %s" % (path, n, exc))
        if not isinstance(obj, dict):
            fail("non-object record at %s:%d" % (path, n))
        rows.append(obj)
    return True, rows


def write_exclusive_json(path, payload):
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def parse_kst_epoch(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        fail("prepared_at_kst has unsupported type")
    text = value.strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        fail("prepared_at_kst is not an ISO-8601 timestamp: %r" % value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=KST)
    return dt.timestamp()


def load_cohort_module():
    if not os.path.isfile(COHORT_PATH):
        fail("sibling frozen_cohort.py not found next to this script")
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import frozen_cohort  # noqa: E402

    for name in ("build_cohort", "summarize"):
        if not callable(getattr(frozen_cohort, name, None)):
            fail("frozen_cohort.%s is missing or not callable" % name)
    return frozen_cohort


# --------------------------------------------------------------------------
# validation of the pinned inputs
# --------------------------------------------------------------------------
def validate_protocol(protocol):
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        fail("protocol schema mismatch")
    safe_id(protocol.get("protocol_id"), "protocol_id")
    safe_id(protocol.get("reference_set_id"), "reference_set_id")
    if protocol.get("participant_id") != "self":
        fail("only participant_id=self is accepted")
    if protocol.get("purpose") != "frozen_retest":
        fail("only purpose=frozen_retest is accepted")
    if protocol.get("rounds") != ROUNDS:
        fail("protocol rounds must be %d" % ROUNDS)

    words = protocol.get("words")
    if not isinstance(words, list) or len(words) != WORD_COUNT:
        fail("protocol must contain exactly %d words" % WORD_COUNT)
    word_ids = []
    for w in words:
        if not isinstance(w, dict):
            fail("word entries must be objects")
        word_ids.append(safe_id(w.get("word_id"), "word_id"))
        safe_id(w.get("clip_id"), "clip_id")
    if len(set(word_ids)) != WORD_COUNT:
        fail("duplicate word_id in protocol")

    slots = protocol.get("slots")
    if not isinstance(slots, list) or len(slots) != SLOT_COUNT:
        fail("protocol must contain exactly %d slots" % SLOT_COUNT)
    seen = set()
    for s in slots:
        if not isinstance(s, dict):
            fail("slot entries must be objects")
        if s.get("word_id") not in word_ids:
            fail("slot references an unplanned word_id")
        if s.get("round") not in range(1, ROUNDS + 1):
            fail("slot has an out-of-range round")
        seen.add(s.get("slot"))
    if len(seen) != SLOT_COUNT:
        fail("slot numbers are not unique")

    models = protocol.get("models")
    if not isinstance(models, dict) or set(models) != set(ROLES):
        fail("protocol must specify exactly the 3 model roles %s" % (ROLES,))
    for role in ROLES:
        spec = models[role]
        if not isinstance(spec, dict) or not spec.get("path") or not spec.get("sha256"):
            fail("model role %s needs path and sha256" % role)

    protected = protocol.get("protected_sha256")
    if not isinstance(protected, dict):
        fail("protocol protected_sha256 must be an object")
    return word_ids


def validate_journal(journal, protocol):
    if journal.get("schema") != JOURNAL_SCHEMA:
        fail("journal schema mismatch")
    if journal.get("protocol_id") != protocol["protocol_id"]:
        fail("journal/protocol protocol_id mismatch")
    if journal.get("status") != "CLOSED":
        fail("journal status must be CLOSED")
    session_id = safe_id(journal.get("session_id"), "session_id")
    safe_id(journal.get("client_id"), "client_id")
    slots = journal.get("slots")
    if not isinstance(slots, list) or len(slots) != SLOT_COUNT:
        fail("journal must contain exactly %d slots" % SLOT_COUNT)
    for entry in slots:
        if not isinstance(entry, dict):
            fail("journal slot entries must be objects")
        safe_id(entry.get("requested_word_id"), "requested_word_id")
        safe_id(entry.get("request_id"), "request_id")
        tid = entry.get("trial_id")
        if tid is not None:
            safe_id(tid, "trial_id")
    return session_id


def validate_session(session, journal, session_id):
    if session.get("session_id") != session_id:
        fail("session.json session_id does not match the journal")
    if session.get("client_id") != journal.get("client_id"):
        fail("session/journal client_id mismatch")
    if session.get("participant_id") != "self":
        fail("session participant_id must be self")
    if session.get("purpose") != "frozen_retest":
        fail("session purpose must be frozen_retest")
    if session.get("state") != "CLOSED":
        fail("session state must be CLOSED")
    if session.get("camera") is not False:
        fail("session camera flag must be false")
    if session.get("store_video") is not False:
        fail("session store_video flag must be false")
    trial_ids = session.get("trial_ids")
    if not isinstance(trial_ids, list):
        fail("session trial_ids must be a list")
    for tid in trial_ids:
        safe_id(tid, "session trial_id")
    created = session.get("created_epoch")
    if not isinstance(created, (int, float)) or isinstance(created, bool):
        fail("session created_epoch must be numeric")
    return trial_ids, float(created)


def validate_result(result, trial_id, session_id, reference_set_id, journal_by_trial):
    if not isinstance(result, dict):
        fail("result for %s is not an object" % trial_id)
    if result.get("trial_id") != trial_id:
        fail("result trial_id mismatch for %s" % trial_id)
    if result.get("session_id") != session_id:
        fail("result session_id mismatch for %s" % trial_id)
    if result.get("origin") != "camera":
        fail("result origin must be camera for %s" % trial_id)
    if result.get("reference_set_id") != reference_set_id:
        fail("result reference_set_id mismatch for %s" % trial_id)
    binding = result.get("reference_binding")
    if not isinstance(binding, dict):
        fail("result %s has no reference_binding" % trial_id)
    if binding.get("status") != "BOUND":
        fail("reference_binding status must be BOUND for %s" % trial_id)
    if binding.get("reference_set_id") != reference_set_id:
        fail("reference_binding reference_set_id mismatch for %s" % trial_id)
    requested = binding.get("requested_word_id")
    safe_id(requested, "binding requested_word_id")
    if binding.get("demonstration_word_id") != requested:
        fail("demonstration_word_id != requested_word_id for %s" % trial_id)
    journal_word = journal_by_trial.get(trial_id)
    if journal_word is not None and journal_word != requested:
        fail("journal/result requested_word_id mismatch for %s" % trial_id)
    clip_id = safe_id(binding.get("demonstration_clip_id"), "demonstration_clip_id")

    normalized = copy.deepcopy(result)
    normalized["reference_binding"]["clip_id"] = clip_id
    return normalized


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------
def cmd_prepare(args):
    cohort_mod = load_cohort_module()

    protocol_path = rp(args.protocol)
    journal_path = rp(args.journal)
    session_dir = rp(args.session_dir)
    web_root = rp(args.web_root)
    pilot_root = rp(args.pilot_root)
    out_dir = rp(args.out_dir)

    for root, name in ((web_root, "web_root"), (pilot_root, "pilot_root")):
        if not os.path.isdir(root):
            fail("%s is not a directory: %s" % (name, root))
    if os.path.exists(out_dir):
        fail("out-dir already exists; refusing to reuse it: %s" % out_dir)
    if under(out_dir, protocol_path) or under(out_dir, journal_path) or under(out_dir, session_dir):
        fail("inputs must not live inside the output directory")

    protocol = read_json(protocol_path, "protocol")
    plan_word_ids = validate_protocol(protocol)
    journal = read_json(journal_path, "journal")
    session_id = validate_journal(journal, protocol)

    runs_root = join_id(web_root, "runs")
    expected_session_dir = join_id(runs_root, session_id)
    if session_dir != expected_session_dir:
        fail("session dir must be exactly web_root/runs/<session_id>")

    session = read_json(join_id(session_dir, "session.json"), "session.json")
    trial_ids, created_epoch = validate_session(session, journal, session_id)

    prepared_epoch = parse_kst_epoch(protocol.get("prepared_at_kst"))
    if prepared_epoch > created_epoch:
        fail("protocol prepared_at_kst is after session created_epoch")

    reviews_path = join_id(session_dir, "trial_reviews.jsonl")
    events_path = join_id(session_dir, "timing_events.jsonl")
    reviews_exists, reviews = read_jsonl(reviews_path, "trial_reviews.jsonl")
    events_exists, events = read_jsonl(events_path, "timing_events.jsonl")

    journal_by_trial = {}
    for entry in journal["slots"]:
        if entry.get("trial_id"):
            journal_by_trial[entry["trial_id"]] = entry.get("requested_word_id")

    reference_set_id = protocol["reference_set_id"]
    trials_root = join_id(session_dir, "trials")
    results, artifacts, hashes, missing = {}, {}, {}, []

    for tid in trial_ids:
        safe_id(tid, "trial_id")
        base = join_id(trials_root, tid + ".result.json")
        sidecar = join_id(trials_root, tid + ".sidecar.json")
        npz = join_id(trials_root, tid + ".npz")
        desc = {
            "npz_exists": os.path.isfile(npz),
            "result_exists": os.path.isfile(base),
            "sidecar_exists": os.path.isfile(sidecar),
            "npz_sha256": None,
            "sidecar_observation_sha256": None,
            "paths": {"result": base, "sidecar": sidecar, "npz": npz},
        }
        if desc["result_exists"]:
            raw = read_json(base, "result for %s" % tid)
            results[tid] = validate_result(raw, tid, session_id, reference_set_id, journal_by_trial)
            hashes[base] = sha256_file(base)
        else:
            missing.append(base)
        if desc["sidecar_exists"]:
            sc = read_json(sidecar, "sidecar for %s" % tid)
            obs_sha = sc.get("observation_sha256")
            if obs_sha is not None and not isinstance(obs_sha, str):
                fail("sidecar observation_sha256 must be a string for %s" % tid)
            desc["sidecar_observation_sha256"] = obs_sha
            hashes[sidecar] = sha256_file(sidecar)
        else:
            missing.append(sidecar)
        if desc["npz_exists"]:
            npz_sha = sha256_file(npz)
            desc["npz_sha256"] = npz_sha
            hashes[npz] = npz_sha
        else:
            missing.append(npz)
        artifacts[tid] = desc

    cohort = cohort_mod.build_cohort(
        copy.deepcopy(protocol), copy.deepcopy(journal), copy.deepcopy(session),
        copy.deepcopy(reviews), copy.deepcopy(events),
        copy.deepcopy(results), copy.deepcopy(artifacts),
    )
    if not isinstance(cohort, dict) or not isinstance(cohort.get("rows"), list):
        fail("build_cohort returned an unusable cohort")
    if len(cohort["rows"]) != SLOT_COUNT:
        fail("cohort must retain all %d rows" % SLOT_COUNT)

    models = {}
    for role in ROLES:
        spec = protocol["models"][role]
        mpath = require_under(runs_root, rp(spec["path"]), "model %s" % role)
        if not os.path.isfile(mpath):
            fail("model file for role %s is missing: %s" % (role, mpath))
        digest = sha256_file(mpath)
        if digest != spec["sha256"]:
            fail("model digest mismatch for role %s" % role)
        hashes[mpath] = digest
        models[role] = {
            "path": mpath,
            "sha256": digest,
            "expected_checkpoint_step": EXPECTED_STEPS[role],
        }

    protected = {}
    for raw_path, expected in protocol["protected_sha256"].items():
        path = rp(raw_path)
        if not os.path.isfile(path):
            missing.append(path)
            protected[path] = {"expected": expected, "actual": None, "present": False}
            continue
        actual = sha256_file(path)
        if actual != expected:
            fail("protected file digest mismatch: %s" % path)
        protected[path] = {"expected": expected, "actual": actual, "present": True}

    for path, what in ((protocol_path, "protocol"), (journal_path, "journal"),
                       (SCRIPT_PATH, "script"), (COHORT_PATH, "frozen_cohort")):
        hashes[path] = sha256_file(path)
    hashes[join_id(session_dir, "session.json")] = sha256_file(join_id(session_dir, "session.json"))
    if reviews_exists:
        hashes[reviews_path] = sha256_file(reviews_path)
    else:
        missing.append(reviews_path)
    if events_exists:
        hashes[events_path] = sha256_file(events_path)
    else:
        missing.append(events_path)

    os.makedirs(out_dir)  # only after every input check passed
    manifest = {
        "schema": "wis_frozen_replay_manifest_v1",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {"web_root": web_root, "pilot_root": pilot_root,
                  "runs_root": runs_root, "session_dir": session_dir,
                  "trials_root": trials_root, "out_dir": out_dir},
        "paths": {"protocol": protocol_path, "journal": journal_path,
                  "session": join_id(session_dir, "session.json"),
                  "reviews": reviews_path, "events": events_path,
                  "script": SCRIPT_PATH, "cohort_module": COHORT_PATH},
        "reviews_present": reviews_exists,
        "events_present": events_exists,
        "protocol_id": protocol["protocol_id"],
        "session_id": session_id,
        "reference_set_id": reference_set_id,
        "plan": {"rounds": ROUNDS, "words": protocol["words"], "slots": protocol["slots"],
                 "word_ids": plan_word_ids},
        "expected_checkpoint_steps": dict(EXPECTED_STEPS),
        "models": models,
        "cohort": cohort,
        "artifacts": artifacts,
        "results_snapshot": results,
        "hashes": hashes,
        "protected_sha256": protected,
        "absent_paths": sorted(set(missing)),
        "inference_calls": 0,
        "training_steps": 0,
    }
    manifest_path = write_exclusive_json(os.path.join(out_dir, MANIFEST_NAME), manifest)

    eligible = [r for r in cohort["rows"] if r.get("inference_eligible")]
    print(json.dumps({
        "status": "PREPARED",
        "manifest": manifest_path,
        "planned_slots": SLOT_COUNT,
        "rows": len(cohort["rows"]),
        "eligible_trials": len(eligible),
        "hashed_files": len(hashes),
        "absent_paths": len(manifest["absent_paths"]),
        "inference_calls": 0,
    }, ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------
def reverify(manifest):
    for path, digest in manifest["hashes"].items():
        if not os.path.isfile(path):
            fail("pinned file disappeared: %s" % path)
        if sha256_file(path) != digest:
            fail("pinned file changed since prepare: %s" % path)
    for path in manifest["absent_paths"]:
        if os.path.exists(path):
            fail("previously absent file appeared: %s" % path)
    for entry in manifest["protected_sha256"].values():
        pass
    for path, info in manifest["protected_sha256"].items():
        if info["present"]:
            if not os.path.isfile(path) or sha256_file(path) != info["expected"]:
                fail("protected file changed: %s" % path)
        elif os.path.exists(path):
            fail("protected file reappeared: %s" % path)
    roots = manifest["roots"]
    if roots["session_dir"] != os.path.join(roots["runs_root"], manifest["session_id"]):
        fail("session dir boundary broken")
    for role, spec in manifest["models"].items():
        require_under(roots["runs_root"], spec["path"], "model %s" % role)


def register_models(neural, manifest, session_id):
    registry = neural.MODELS
    if not isinstance(registry, dict):
        fail("adapter MODELS is not a dict")
    keys = {}
    for role in ROLES:
        spec = manifest["models"][role]
        n = 0
        while True:
            key = "frozen_offline_%s_%s_%d" % (role, session_id, n)
            if key not in registry:
                break
            n += 1
        registry[key] = {
            "model_id": key,
            "path": spec["path"],
            "sha256": spec["sha256"],
            "project_words": dict(neural.PROJECT_18),
            "role": "offline frozen comparison",
            "status_label": "offline evaluation",
        }
        keys[role] = key
    return keys


def normalize(role, raw, expected_sha):
    if not isinstance(raw, dict):
        fail("recognise() for %s did not return an object" % role)
    if raw.get("checkpoint_sha256") != expected_sha:
        fail("checkpoint sha mismatch reported by recognise() for %s" % role)
    if raw.get("accepted") is not None:
        fail("recognise() accepted must be None for offline replay (%s)" % role)
    if raw.get("execution_assessment") != "UNCONFIGURED":
        fail("recognise() execution_assessment must be UNCONFIGURED (%s)" % role)
    scopes = raw.get("scopes") or {}
    p18 = (scopes.get("project18") or {}).get("top1")
    full = (scopes.get("full") or {}).get("top1")
    ready = raw.get("model_status") == "READY"
    return {
        "status": "READY" if ready else "ERROR",
        "project18": p18 if ready else None,
        "full93": full if ready else None,
        "elapsed_ms": raw.get("elapsed_ms"),
        "error": None if ready else "model_status=%r" % raw.get("model_status"),
    }


def cmd_execute(args):
    manifest_path = rp(args.manifest)
    manifest = read_json(manifest_path, "manifest")
    out_dir = rp(manifest["roots"]["out_dir"])
    if not os.path.isdir(out_dir):
        fail("manifest out_dir is missing: %s" % out_dir)
    result_path = os.path.join(out_dir, RESULT_NAME)
    started_path = os.path.join(out_dir, STARTED_NAME)
    error_path = os.path.join(out_dir, ERROR_NAME)
    calls_path = os.path.join(out_dir, CALLS_NAME)
    for path in (result_path, started_path):
        if os.path.exists(path):
            fail("refusing to rerun; %s already exists" % path)

    try:
        cohort_mod = load_cohort_module()
        reverify(manifest)
        manifest_sha = sha256_file(manifest_path)
        session_id = manifest["session_id"]
        write_exclusive_json(started_path, {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_sha, "session_id": session_id,
        })

        for root in (manifest["roots"]["web_root"], manifest["roots"]["pilot_root"]):
            src = os.path.join(root, "src")
            if os.path.isdir(src) and src not in sys.path:
                sys.path.insert(0, src)

        import torch
        from wis_lab.core.observation import Observation
        from wis_lab.service import neural

        for setter in ("set_num_threads", "set_num_interop_threads"):
            fn = getattr(torch, setter, None)
            if callable(fn):
                try:
                    fn(2)
                except (RuntimeError, ValueError):
                    pass

        registry = neural.MODELS
        registry_snapshot = dict(registry)
        default_snapshot = neural.DEFAULT_MODEL
        project18 = dict(neural.PROJECT_18)
        if set(project18) != set(manifest["plan"]["word_ids"]):
            fail("adapter PROJECT_18 does not match the protocol word ids")

        keys, loads, predictions, calls = {}, {}, {}, 0
        try:
            keys = register_models(neural, manifest, session_id)
            classes_ref = None
            for role in ROLES:
                info = neural._load(keys[role])
                if not isinstance(info, dict):
                    fail("_load() for %s returned no info" % role)
                if info.get("status") != "READY":
                    fail("model %s is not READY (fail closed)" % role)
                if info.get("checkpoint_sha256") != manifest["models"][role]["sha256"]:
                    fail("loaded checkpoint sha mismatch for %s" % role)
                if info.get("checkpoint_step") != EXPECTED_STEPS[role]:
                    fail("checkpoint_step mismatch for %s" % role)
                classes = list(info.get("classes") or [])
                if len(classes) != FULL_CLASS_COUNT:
                    fail("model %s does not expose %d classes" % (role, FULL_CLASS_COUNT))
                if classes_ref is None:
                    classes_ref = classes
                elif classes != classes_ref:
                    fail("class order differs between models")
                loads[role] = {"checkpoint_sha256": info["checkpoint_sha256"],
                               "checkpoint_step": info["checkpoint_step"],
                               "classes": len(classes)}

            rows = manifest["cohort"]["rows"]
            offline_elapsed_ms = 0.0
            with open(calls_path, "x", encoding="utf-8") as log:
                for row in rows:
                    if not row.get("inference_eligible"):
                        continue
                    tid = safe_id(row.get("trial_id"), "trial_id")
                    desc = manifest["artifacts"].get(tid) or {}
                    npz = desc.get("paths", {}).get("npz")
                    if not desc.get("npz_exists") or not npz:
                        continue
                    require_under(manifest["roots"]["trials_root"], npz, "npz for %s" % tid)
                    observation = Observation.load(npz)  # read once, shared by all roles
                    per_role = {}
                    for role in ROLES:
                        t0 = time.perf_counter()
                        try:
                            raw = neural.recognise(observation, keys[role])
                            norm = normalize(role, raw, manifest["models"][role]["sha256"])
                        except ValueError:
                            raise
                        except Exception as exc:  # runtime failure of one model only
                            norm = {"status": "ERROR", "project18": None, "full93": None,
                                    "elapsed_ms": None, "error": type(exc).__name__}
                        dt = (time.perf_counter() - t0) * 1000.0
                        offline_elapsed_ms += dt
                        calls += 1
                        per_role[role] = norm
                        log.write(json.dumps({
                            "trial_id": tid, "slot": row.get("slot"), "role": role,
                            "offline_call_ms": round(dt, 3), "prediction": norm,
                        }, ensure_ascii=False) + "\n")
                        log.flush()
                    predictions[tid] = per_role
        finally:
            if keys:
                for key in keys.values():
                    registry.pop(key, None)
            for key in list(registry):
                if key not in registry_snapshot:
                    registry.pop(key, None)
            registry.update(registry_snapshot)
            neural.DEFAULT_MODEL = default_snapshot

        if dict(neural.MODELS) != registry_snapshot or neural.DEFAULT_MODEL != default_snapshot:
            fail("adapter registry or default model was not restored")
        reverify(manifest)

        summary = cohort_mod.summarize(copy.deepcopy(manifest["cohort"]),
                                       copy.deepcopy(predictions), dict(project18))
        payload = {
            "status": "EXECUTED_PENDING_INDEPENDENT_RECOUNT",
            "manifest_sha256": manifest_sha,
            "cohort": manifest["cohort"],
            "models": {r: dict(manifest["models"][r], **loads[r]) for r in ROLES},
            "summary": summary,
            "predictions": predictions,
            "inference_calls": calls,
            "inference_calls_log": calls_path,
            "offline_elapsed_ms_total": round(offline_elapsed_ms, 3),
            "training_steps": 0,
            "model_promoted": False,
            "protected_files_unchanged": True,
            "label_map": project18,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_exclusive_json(result_path, payload)
        print(json.dumps({"status": payload["status"], "result": result_path,
                          "inference_calls": calls,
                          "trials_scored": len(predictions)},
                         ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        try:
            write_exclusive_json(error_path, {
                "status": "EXECUTION_FAILED",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            })
        except OSError:
            pass
        sys.stderr.write("execution failed: %s: %s\n" % (type(exc).__name__, exc))
        sys.stderr.write(traceback.format_exc())
        return 1


# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="local-only frozen replay driver")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for flag in ("--protocol", "--journal", "--session-dir", "--web-root",
                 "--pilot-root", "--out-dir"):
        p.add_argument(flag, required=True)
    e = sub.add_parser("execute")
    e.add_argument("--manifest", required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        try:
            return cmd_prepare(args)
        except (ValueError, RuntimeError, OSError) as exc:
            sys.stderr.write("prepare failed: %s: %s\n" % (type(exc).__name__, exc))
            return 1
    return cmd_execute(args)


if __name__ == "__main__":
    sys.exit(main())
