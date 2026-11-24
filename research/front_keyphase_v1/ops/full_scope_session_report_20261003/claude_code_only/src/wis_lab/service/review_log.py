"""What the participant says they actually signed, kept apart from what was asked and predicted.

requested_word_id is the prompt and attempt_label is the intention chosen BEFORE the attempt.
Neither says what the hands did. This module stores a separate, after-the-fact confirmation and
joins it with the results that were already written.

Rules the storage keeps:
  append only          nothing already written is edited; a correction is a new revision
  latest revision wins for aggregation, and only a revision that actually reached the disk counts
  read failure is a STATE, never an empty list. A store that cannot be read must not look like a
                       store with no confirmations, and must not let submit() write revision 1 on
                       top of revisions it could not see.
  no arbitrary paths   a review is addressed by session and trial id, never by a file path
  never an input       the confirmation is never fed to DTW or to the neural adapter

Counting rules, after independent review reproduced three ways of losing rows:
  * aggregation starts from the CONFIRMATIONS. Whether a result file exists, whether the
    observation was sufficient and whether a model answered are attached afterwards, so a
    confirmed attempt never disappears from its own denominator.
  * each method is tallied on its own over every supported confirmation. DTW being right must not
    be erased because the neural model was unavailable, and the reverse must not happen either.
  * the paired figure - both methods answered the same attempt - is reported separately and is
    never mixed with the per-method one.

What a confirmation is NOT: it is the participant's own report, not an expert judgement of sign
correctness. Execution quality, rejection of wrong performances, and any combined gate remain
NOT_CONFIGURED.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

REVIEW_FILE = "trial_reviews.jsonl"
REVIEW_SCHEMA = "wis_trial_review_v1"
# the words the neural checkpoint and the project share; the only fair common ground
COMMON_5 = ("ALDA", "GAMSAHAMNIDA", "NA", "JOTA", "MOREUDA")
UNKNOWN = "UNKNOWN"
FOLLOWED = ("yes", "no", "unsure")
MAX_NOTE = 300
METHODS = ("dtw", "neural")


def _path(run_dir: str) -> str:
    return os.path.join(run_dir, REVIEW_FILE)


def read_all(run_dir: str):
    """-> (rows, error). error is a string when the store exists but could not be read.

    A line that will not parse used to be skipped in silence. An append-only log can end in a torn
    write, and a reader that shrugs at it turns a lost confirmation into a confident smaller
    number. A malformed line is now a STATE of the whole store: no score is produced from it and
    submit() refuses to append, because a revision it could not see must not be overwritten.
    """
    p = _path(run_dir)
    if not os.path.exists(p):
        return [], None
    out = []
    try:
        with open(p, encoding="utf-8") as fh:
            for n_line, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    return [], f"STORE_MALFORMED:line {n_line}: {exc.msg}"
                if not isinstance(row, dict) or "trial_id" not in row or "revision" not in row:
                    return [], f"STORE_MALFORMED:line {n_line}: not a review row"
                out.append(row)
    except OSError as exc:
        # an unreadable store is NOT an empty store. Returning [] here made a report read as
        # "no confirmations" and would have let submit() start again at revision 1.
        return [], f"STORE_UNREADABLE:{type(exc).__name__}"
    return out, None


def latest_by_trial(run_dir: str):
    """-> (mapping of trial_id to its newest revision, error)."""
    rows, err = read_all(run_dir)
    if err:
        return {}, err
    best = {}
    for r in rows:
        cur = best.get(r["trial_id"])
        if cur is None or r["revision"] > cur["revision"]:
            best[r["trial_id"]] = r
    return best, None


# Whether the person followed the GUIDANCE ("try 알다") and whether they followed the specific
# FORM of the clip they were shown are different questions. followed_guidance already existed and
# keeps its meaning; this is a separate field and an old "yes" is never read as a new one.
FORM_FOLLOWED = (None, "yes", "no", "other_form", "unsure")


def _same_content(a: dict, b: dict) -> bool:
    return all(a.get(k) == b.get(k)
               for k in ("actual_word_id", "followed_guidance", "followed_demonstration_form",
                         "note"))


def submit(run_dir: str, session_id: str, trial_id: str, actual_word_id, followed_guidance,
           note=None, known_words=(), followed_demonstration_form=None) -> dict:
    """Validate, deduplicate, append. A failed read or write is never reported as saved."""
    if followed_guidance not in FOLLOWED:
        return {"status": "REJECTED", "reason": "followed_guidance must be yes, no or unsure"}
    if followed_demonstration_form not in FORM_FOLLOWED:
        return {"status": "REJECTED",
                "reason": "followed_demonstration_form must be null, yes, no, other_form "
                          "or unsure"}
    actual = actual_word_id or UNKNOWN
    if actual != UNKNOWN and known_words and actual not in known_words:
        return {"status": "REJECTED", "reason": "unknown word id"}
    note = (note or "").strip()[:MAX_NOTE] or None

    existing, err = read_all(run_dir)
    if err:
        # writing now could duplicate or overwrite a revision that is already on disk. A torn or
        # corrupt line counts here too: the newest revision may be exactly the one that will not
        # parse, and appending on top of it would bury a correction.
        return {"status": "NOT_STORED", "stored": False, "trial_id": trial_id,
                "reason": err, "revision": None,
                "meaning": "the existing store could not be read in full, so no revision was "
                           "written. A malformed line is treated the same way as an unreadable "
                           "file: nothing is appended and no score is produced."}
    rows = [r for r in existing if r["trial_id"] == trial_id]
    payload = {"actual_word_id": actual, "followed_guidance": followed_guidance,
               "followed_demonstration_form": followed_demonstration_form, "note": note}
    if rows:
        newest = max(rows, key=lambda r: r["revision"])
        if _same_content(newest, payload):
            return {"status": "DUPLICATE_IGNORED", "revision": newest["revision"],
                    "stored": True, "trial_id": trial_id}
        revision = newest["revision"] + 1
    else:
        revision = 1

    row = {"schema": REVIEW_SCHEMA, "event": "trial_review", "session_id": session_id,
           "trial_id": trial_id, "revision": revision, **payload,
           "source": "participant_self_report",
           "is_expert_verification": False,
           "form_field_meaning": "followed_demonstration_form is the participant's own answer "
                                 "about the clip they were shown. It is not an accuracy figure "
                                 "and not a correctness verdict.",
           "recorded_at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        with open(_path(run_dir), "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:
        return {"status": "NOT_STORED", "stored": False, "trial_id": trial_id,
                "reason": f"WRITE_FAILED:{type(exc).__name__}", "revision": None}
    return {"status": "STORED", "stored": True, "trial_id": trial_id, "revision": revision,
            "note_kept": bool(note)}


def _dtw_pick(result: dict, allowed, label="COMMON5"):
    rows = [r for r in (result.get("ranking") or [])
            if r.get("word_id") in allowed and r.get("status") == "OK"
            and r.get("distance") is not None]
    if not rows:
        return None, f"NO_{label}_CANDIDATE"
    return min(rows, key=lambda r: r["distance"])["word_id"], None


def _neural_pick(result: dict):
    lr = result.get("learned_recognition") or {}
    status = lr.get("model_status")
    if status != "READY" or not lr.get("ranking"):
        return None, "MODEL_" + (status or "ABSENT")
    top1 = (lr["ranking"].get("restricted_to_project_words") or {}).get("top1")
    if top1 is None:
        return None, "NO_RESTRICTED_CANDIDATE"
    return top1, None


def _empty_score(total=0):
    return {"correct": 0, "total": total, "value": None, "display": "기록 없음"}


def _score(hits, total):
    if total == 0:
        return _empty_score()
    return {"correct": hits, "total": total, "value": round(hits / total, 4),
            "display": f"{hits}/{total}"}


# --------------------------------------------------------------- opt-in candidate cohorts
COMMON_5_SET_ID = "wis_common5_v1"
COHORT_SCOPES = {"common5": ("dtw", "base_neural", "candidate"),
                 "project18": ("dtw", "candidate")}


def _creation(row: dict):
    """The opt-in evidence fixed when the trial was created, and where it came from.

    Three sources, in this order, because each is a weaker copy of the one before:

    1. the live creation metadata the caller passed,
    2. the trial's OWN research_model, saved beside the result - including an explicit
       opted_in false,
    3. the identity copy nested inside the candidate's answer.

    Reading the result's research_model only through (1) made two stored records read wrongly once
    the session was gone: an explicit refusal with a stray candidate block beside it got scored,
    and a trial that opted in but whose candidate crashed before writing a block disappeared
    instead of standing as unanswered.

    None of the three is the live registry. A cohort that let today's registry fill in yesterday's
    identity would be holding nothing fixed at all.
    """
    meta = row.get("creation_research_model")
    if meta:
        return meta, "creation_metadata"
    res = row.get("result") or {}
    saved = res.get("research_model")
    if saved:
        return saved, "result_research_model"
    rc = res.get("research_comparison")
    if rc is None:
        return None, None                       # no opt-in evidence of any kind
    # A candidate answered, but the identity this trial fixed is gone. That is not the same as
    # never having opted in, and it must not be dropped in silence: with no model_id and no
    # checkpoint it is reported unassignable below, never filled in from the live registry.
    return rc.get("identity_fixed_at_trial_creation") or {}, "result_identity_copy"


def _fixed_members(scope: str, creation: dict):
    """-> (members, set_id, reason). The task, taken from fixed evidence - never from a ranking.

    A stored ranking is ordered by score and can be short, so it says what the model answered, not
    what it was asked. common5 is this module's versioned five-word subset; project18 is the
    project word list the chosen model declared at trial creation.
    """
    if scope == "common5":
        return tuple(sorted(COMMON_5)), COMMON_5_SET_ID, None
    declared = creation.get("project_words_in_model")
    if not declared:
        return None, None, "NO_CANDIDATE_SET_EVIDENCE"
    words = [w for w in declared if w]
    if len(words) != len(declared) or len(set(words)) != len(words):
        return None, None, "CANDIDATE_SET_MALFORMED"
    members = tuple(sorted(words))
    return members, "declared_project_words:" + _set_sha(members)[:12], None


def _set_sha(members) -> str:
    """Canonical membership, never the score-ranked order - the same set must hash the same."""
    return hashlib.sha256("|".join(sorted(members)).encode("utf-8")).hexdigest()


def _scope_block_ok(block: dict, members):
    """The stored answer must be over EXACTLY the fixed set, by membership and by declared count."""
    if not block:
        return "CANDIDATE_SCOPE_ABSENT"
    order = block.get("order")
    if order is None:
        return "CANDIDATE_SET_ABSENT"
    ids = [r.get("word_id") for r in order]
    if any(not w for w in ids) or len(set(ids)) != len(ids):
        return "CANDIDATE_SET_MALFORMED"
    if tuple(sorted(ids)) != tuple(members):
        return "CANDIDATE_SET_INCOMPLETE"
    declared = block.get("candidate_count")
    if declared is not None and declared != len(members):
        return "CANDIDATE_SET_COUNT_MISMATCH"
    return None


def _answered_identity(rc: dict):
    """(model_id, checkpoint) of the model that actually wrote this answer block."""
    fixed = rc.get("identity_fixed_at_trial_creation") or {}
    return (rc.get("model_id") or fixed.get("model_id"),
            rc.get("checkpoint_sha256") or fixed.get("checkpoint_sha256"))


def _candidate_pick(scope: str, result: dict, creation: dict, members):
    """The candidate's own stored answer, or the reason it does not count. Nothing is recomputed."""
    rc = (result or {}).get("research_comparison")
    if rc is None:
        return None, "CANDIDATE_ABSENT"
    if _answered_identity(rc) != (creation.get("model_id"), creation.get("checkpoint_sha256")):
        # a different checkpoint answered than the one this trial fixed. Scoring it inside this
        # cohort would credit one model with another model's answer. The attempt itself stays.
        return None, "CANDIDATE_IDENTITY_MISMATCH"
    if rc.get("model_status") != "READY" or not rc.get("scopes"):
        return None, "CANDIDATE_" + str(rc.get("model_status") or "ABSENT")
    block = (rc.get("scopes") or {}).get(scope) or {}
    bad = _scope_block_ok(block, members)
    if bad:
        return None, bad
    top1 = block.get("top1")
    if not top1:
        return None, "CANDIDATE_NO_PICK"
    if top1 not in members:
        return None, "CANDIDATE_PICK_OUTSIDE_SET"
    return top1, None


def _picks(scope: str, row: dict, creation: dict, members) -> dict:
    """One pick per method in this cohort, each with its own reason when it did not answer."""
    result, missing = row.get("result"), row.get("result_reason")
    out = {}
    for m in COHORT_SCOPES[scope]:
        if result is None:
            out[m] = (None, missing or "NO_RESULT_FILE")
        elif m == "dtw":
            out[m] = (_dtw_pick(result, members, scope.upper())
                      if result.get("observation_status") == "OK"
                      else (None, "OBSERVATION_" + str(result.get("observation_status"))))
        elif m == "base_neural":
            out[m] = _neural_pick(result)
        else:
            out[m] = _candidate_pick(scope, result, creation, members)
    return out


def cohorts(confirmed: list) -> dict:
    """Group opt-in confirmed attempts by a fixed candidate identity, and score each group.

    Each entry carries the latest confirmed word, the stored result (or None with its reason) and
    the creation-time metadata for the trial.

    Three rules do most of the work. An explicit opt-out wins over any candidate block that may be
    lying beside it, and a record from before the candidate existed never gains one. The task is
    fixed by creation evidence rather than read off a ranking, so a short or unexpected ranking
    makes that one model answer unanswered instead of quietly turning the task into an easier one.
    And an attempt is never deleted for a failure: a missing result, an insufficient observation,
    a failed model or a checkpoint that does not match what the trial fixed all stay in the
    denominator, with their reason, and never remove another method's answer.
    """
    groups, unassignable = {}, []

    def cannot(row, scope, reason):
        unassignable.append({"trial_id": row["trial_id"], "scope": scope, "reason": reason,
                             "actual_word_id": row["actual_word_id"]})

    for row in confirmed:
        creation, source = _creation(row)
        if creation is None:
            continue                                # legacy: no opt-in evidence of any kind
        if creation.get("opted_in") is False:
            continue                                # explicit opt-out beats any stray block
        ok_identity = bool(creation.get("model_id") and creation.get("checkpoint_sha256"))
        for scope in COHORT_SCOPES:
            if not ok_identity:
                cannot(row, scope, "NO_STORED_MODEL_IDENTITY")
                continue
            members, set_id, bad = _fixed_members(scope, creation)
            if bad:
                cannot(row, scope, bad)
                continue
            key = (scope, creation["model_id"], creation["checkpoint_sha256"],
                   creation.get("feature_schema"))
            g = groups.setdefault(key, {"scope": scope, "members": members, "set_id": set_id,
                                        "creation": creation, "sources": set(), "rows": []})
            if g["members"] != members:
                # the same checkpoint declaring two different tasks. Neither is silently preferred.
                cannot(row, scope, "CANDIDATE_SET_MISMATCH")
                continue
            g["sources"].add(source)
            g["rows"].append((row, creation))

    out = {}
    for key, g in groups.items():
        scope, members = g["scope"], g["members"]
        methods = COHORT_SCOPES[scope]
        rows, tally = [], {m: {"answered": 0, "hits": 0, "unanswered": {}} for m in methods}
        supported = []          # the source rows this cohort actually scored, and nothing else
        for row, creation in g["rows"]:
            actual = row["actual_word_id"]
            if actual == UNKNOWN or actual not in members:
                rows.append({"trial_id": row["trial_id"], "actual_word_id": actual,
                             "in_candidate_set": False,
                             "excluded_reason": "ACTUAL_WORD_OUTSIDE_THIS_CANDIDATE_SET"})
                continue
            supported.append(row)
            out_row = {"trial_id": row["trial_id"], "actual_word_id": actual,
                       "in_candidate_set": True}
            for m, (pick, why) in _picks(scope, row, creation, members).items():
                out_row[m] = {"pick": pick, "reason": why,
                              "hit": (pick == actual) if pick else None}
                t = tally[m]
                if pick is None:
                    t["unanswered"][why] = t["unanswered"].get(why, 0) + 1
                else:
                    t["answered"] += 1
                    t["hits"] += int(pick == actual)
            rows.append(out_row)

        scored = [r for r in rows if r["in_candidate_set"]]
        n = len(scored)
        all_answered = [r for r in scored if all(r[m]["pick"] for m in methods)]
        confusion = {m: {} for m in methods}
        for m in methods:
            for r in scored:
                col = confusion[m].setdefault(r["actual_word_id"], {})
                pred = r[m]["pick"] or UNKNOWN
                col[pred] = col.get(pred, 0) + 1
        ident = {"model_id": key[1], "checkpoint_sha256": key[2], "feature_schema": key[3],
                 "candidate_set_id": g["set_id"], "candidate_set_sha256": _set_sha(members),
                 "candidate_count": len(members),
                 "identity_source": sorted(g["sources"]),
                 "fixed_by": "trial creation evidence, not the live registry"}
        out["|".join([scope, key[1], key[2], str(key[3]), _set_sha(members)])] = {
            "scope": scope,
            "identity": ident,
            "candidate_set": list(members),
            "methods": list(methods),
            "n_supported": n,
            "denominator_meaning": "every opt-in confirmed attempt whose confirmed word is in "
                                   "this fixed candidate set. A missing result, an insufficient "
                                   "observation, a failed model or a checkpoint that does not "
                                   "match what the trial fixed does not remove the attempt - it "
                                   "is counted as unanswered, with its reason.",
            "per_method": {m: {"hits": tally[m]["hits"], "answered": tally[m]["answered"],
                               "n": n, "unanswered": tally[m]["unanswered"],
                               "match_over_all_supported": _score(tally[m]["hits"], n),
                               "match_over_answered_only": _score(tally[m]["hits"],
                                                                  tally[m]["answered"])}
                           for m in methods},
            "all_answered": dict({m: _score(sum(1 for r in all_answered if r[m]["hit"]),
                                            len(all_answered)) for m in methods},
                                 n=len(all_answered),
                                 meaning="attempts where EVERY method in this cohort answered"),
            "per_word": {w: dict(n=sum(1 for r in scored if r["actual_word_id"] == w),
                                 **{m: sum(1 for r in scored
                                           if r["actual_word_id"] == w and r[m]["hit"])
                                    for m in methods})
                         for w in members},
            "confusion": confusion,
            "confusion_note": "rows are the confirmed word, columns are the pick. UNKNOWN is the "
                              "column for an attempt that this method did not answer.",
            "rows": rows,
            # this cohort's OWN attempts. It used to summarise every row in the group, so the
            # five-word cohort's timing included attempts whose word was outside its five.
            "timing": timing_summary(supported),
            "demonstration_form_followed": _form_tally(
                supported,
                "the confirmations THIS cohort scored: opt-in, confirmed, and inside this "
                "candidate set"),
            "is_not": "a correctness or quality figure. It is agreement with the word the "
                      "participant said they signed, over one fixed candidate set.",
        }
    return {"groups": out, "unassignable": unassignable,
            "unassignable_meaning": "an opt-in attempt whose fixed evidence does not say which "
                                    "model answered, or over which candidate set. It is counted "
                                    "here and never assigned to a cohort from the live registry.",
            "never_retrospective": "opt-out and pre-candidate records are not given candidate "
                                   "outputs",
            "no_mixing": "a different candidate count is a different task; cohorts with different "
                         "identities are never pooled"}


# ------------------------------------------------------- full trained-class scope (research)
# The candidate's answer over EVERY class it was trained on. Its block stores model LABELS, not
# project word ids, so it can only be scored through the class list and the project-id -> label
# mapping that the trial froze when it was created. Nothing here reads the live registry, and a
# trial that froze no such evidence is reported unavailable for this scope - never filled in.
FULL_CLASS_SET_SCHEMA = "wis_trained_class_set_v1"


def full_class_set_sha256(labels) -> str:
    """Canonical membership of a trained class set: the sorted labels as compact JSON.

    JSON rather than a joined string, so a label containing the separator cannot collide with two
    others. The checkpoint's own index order is kept beside this, in the evidence's labels list.
    """
    return hashlib.sha256(json.dumps(sorted(labels), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _mapping_sha(mapping: dict) -> str:
    return hashlib.sha256(json.dumps(sorted(mapping.items()), ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _full_evidence(creation: dict):
    """-> (evidence, reason). The class set and mapping this trial froze, or why it cannot be used.

    Every check fails closed: a malformed, duplicated, miscounted or mis-hashed class list, or a
    mapping that is malformed, points outside the list or names one label for two words, makes the
    whole scope unavailable for that trial instead of being repaired here.
    """
    ev = creation.get("trained_class_set")
    if not ev:
        return None, "FULL_CLASS_SET_NOT_RECORDED"
    if not isinstance(ev, dict):
        return None, "FULL_CLASS_SET_MALFORMED"
    if ev.get("status") != "AVAILABLE":
        return None, "FULL_CLASS_SET_UNAVAILABLE_AT_CREATION:" + str(ev.get("reason")
                                                                    or ev.get("status"))
    if (ev.get("model_id"), ev.get("checkpoint_sha256")) != (creation.get("model_id"),
                                                             creation.get("checkpoint_sha256")):
        return None, "FULL_CLASS_SET_IDENTITY_MISMATCH"
    labels = ev.get("labels")
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(x, str) or not x for x in labels)):
        return None, "FULL_CLASS_SET_MALFORMED"
    if len(set(labels)) != len(labels):
        return None, "FULL_CLASS_SET_DUPLICATE_LABEL"
    declared = creation.get("trained_class_count")
    if ev.get("count") != len(labels) or (declared is not None and declared != len(labels)):
        return None, "FULL_CLASS_COUNT_MISMATCH"
    sha = full_class_set_sha256(labels)
    if ev.get("membership_sha256") != sha:
        return None, "FULL_CLASS_SET_HASH_MISMATCH"
    mapping = ev.get("project_word_to_model_label")
    if not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in mapping.items()):
        return None, "FULL_MAPPING_MALFORMED"
    label_set = frozenset(labels)
    if any(v not in label_set for v in mapping.values()):
        return None, "FULL_MAPPING_LABEL_OUTSIDE_CLASS_SET"
    if len(set(mapping.values())) != len(mapping):
        return None, "FULL_MAPPING_AMBIGUOUS"
    return {"labels": list(labels), "label_set": label_set, "sha": sha, "count": len(labels),
            "mapping": dict(mapping), "mapping_sha": _mapping_sha(mapping)}, None


def _full_pick(row: dict, creation: dict, ev: dict):
    """The candidate's stored full-scope top1 LABEL, or why it does not count. Nothing is rerun."""
    result = row.get("result")
    if result is None:
        return None, row.get("result_reason") or "NO_RESULT_FILE"
    rc = result.get("research_comparison")
    if rc is None:
        return None, "CANDIDATE_ABSENT"
    if _answered_identity(rc) != (creation.get("model_id"), creation.get("checkpoint_sha256")):
        return None, "CANDIDATE_IDENTITY_MISMATCH"
    if rc.get("model_status") != "READY" or not rc.get("scopes"):
        return None, "CANDIDATE_" + str(rc.get("model_status") or "ABSENT")
    block = (rc.get("scopes") or {}).get("full")
    if not isinstance(block, dict) or not block:
        return None, "FULL_SCOPE_ABSENT"
    if block.get("candidate_count") != ev["count"]:
        return None, "FULL_ANSWER_COUNT_MISMATCH"
    answered_sha = block.get("class_set_sha256")
    if answered_sha is None:
        return None, "FULL_ANSWER_CLASS_SET_NOT_RECORDED"
    if answered_sha != ev["sha"]:
        return None, "FULL_ANSWER_CLASS_SET_MISMATCH"
    top1 = block.get("top1")
    if not top1:
        return None, "CANDIDATE_NO_PICK"
    if top1 not in ev["label_set"]:
        return None, "CANDIDATE_PICK_OUTSIDE_SET"
    return top1, None


def full_scope(rows: list) -> dict:
    """The candidate alone, over every class it was trained on, scored per frozen identity.

    `rows` is every trial of the session - confirmed or not - with the latest confirmed word
    (None when unconfirmed), the stored result and the creation metadata. Only the candidate method
    exists here: DTW and the base model answer over different candidate sets and are never placed
    in this denominator. The confirmed project word is translated into the model's label through the
    trial's own frozen mapping; the requested word is never read.
    """
    groups, unavailable = {}, []
    counts = {"trials_seen": len(rows), "opt_in": 0, "opted_out": 0, "no_opt_in_evidence": 0}
    for row in rows:
        creation, source = _creation(row)
        if creation is None:
            counts["no_opt_in_evidence"] += 1
            continue
        if creation.get("opted_in") is False:
            counts["opted_out"] += 1               # an explicit refusal beats any stray block
            continue
        counts["opt_in"] += 1

        def cannot(reason):
            unavailable.append({"trial_id": row["trial_id"], "reason": reason,
                                "confirmed": bool(row.get("confirmed")),
                                "actual_word_id": row.get("actual_word_id")})

        if not (creation.get("model_id") and creation.get("checkpoint_sha256")):
            cannot("NO_STORED_MODEL_IDENTITY")
            continue
        ev, bad = _full_evidence(creation)
        if bad:
            cannot(bad)
            continue
        key = (creation["model_id"], creation["checkpoint_sha256"],
               str(creation.get("feature_schema")), ev["sha"], ev["mapping_sha"])
        g = groups.setdefault(key, {"ev": ev, "sources": set(), "rows": []})
        g["sources"].add(source)
        g["rows"].append((row, creation))

    out = {}
    for key, g in groups.items():
        ev = g["ev"]
        scored, excluded = [], []
        tally = {"answered": 0, "hits": 0, "unanswered": {}}
        for row, creation in g["rows"]:
            tid, actual = row["trial_id"], row.get("actual_word_id")
            if not row.get("confirmed"):
                excluded.append({"trial_id": tid, "reason": "NOT_CONFIRMED"})
                continue
            if actual == UNKNOWN:
                excluded.append({"trial_id": tid, "reason": "CONFIRMED_UNKNOWN"})
                continue
            if actual not in ev["mapping"]:
                excluded.append({"trial_id": tid, "reason": "ACTUAL_WORD_NOT_IN_MODEL_MAPPING",
                                 "actual_word_id": actual})
                continue
            target = ev["mapping"][actual]
            pick, why = _full_pick(row, creation, ev)
            if pick is None:
                tally["unanswered"][why] = tally["unanswered"].get(why, 0) + 1
            else:
                tally["answered"] += 1
                tally["hits"] += int(pick == target)
            scored.append({"trial_id": tid, "actual_word_id": actual,
                           "actual_model_label": target, "pick_label": pick,
                           "hit": (pick == target) if pick else None, "reason": why})
        n = len(scored)
        excluded_counts = {}
        for e in excluded:
            excluded_counts[e["reason"]] = excluded_counts.get(e["reason"], 0) + 1
        out["|".join(("full",) + key)] = {
            "scope": "full",
            "identity": {"model_id": key[0], "checkpoint_sha256": key[1],
                         "feature_schema": _creation_schema(g),
                         "class_set_sha256": ev["sha"], "class_count": ev["count"],
                         "mapping_sha256": ev["mapping_sha"],
                         "identity_source": sorted(g["sources"]),
                         "fixed_by": "trial creation evidence, not the live registry"},
            "class_labels": ev["labels"],
            "project_word_to_model_label": ev["mapping"],
            "methods": ["candidate"],
            "n_supported": n,
            "unique_trials": len({r["trial_id"] for r, _ in g["rows"]}),
            "per_method": {"candidate": {
                "hits": tally["hits"], "answered": tally["answered"], "n": n,
                "unanswered": tally["unanswered"],
                "match_over_all_supported": _score(tally["hits"], n),
                "match_over_answered_only": _score(tally["hits"], tally["answered"])}},
            "rows": scored,
            "excluded": excluded,
            "excluded_counts": excluded_counts,
            "denominator_meaning": "every opt-in attempt the participant confirmed as a project "
                                   "word that this model's frozen mapping names. A missing result, "
                                   "a failed model or an answer from another model or class set "
                                   "stays here as unanswered, with its reason.",
            "not_compared_with": "DTW and the base model answer over restricted candidate sets "
                                 "and another class count; they are not scored in this scope.",
            "is_not": "a correctness or generalisation figure. It is agreement between the "
                      "candidate's stored top1 label and the label of the word the participant "
                      "said they signed.",
        }
    return {"status": "AVAILABLE" if out else "UNAVAILABLE",
            "groups": out,
            "unavailable": unavailable,
            "unavailable_unique_trials": len({u["trial_id"] for u in unavailable}),
            "counts": counts,
            "unavailable_meaning": "an opt-in trial whose creation evidence does not carry a "
                                   "verified class list and mapping for this scope. Records from "
                                   "before that evidence existed land here; nothing is inferred "
                                   "from a stored top3 or from the model loaded now.",
            "no_mixing": "different models, checkpoints, feature schemas, class sets or mappings "
                         "are different groups and are never pooled"}


def _creation_schema(g: dict):
    """The feature schema as the group's first creation record states it (None stays None)."""
    return g["rows"][0][1].get("feature_schema") if g["rows"] else None


def _form_tally(rows: list, denominator: str) -> dict:
    """The participant's own answer about the clip's form, over a stated population.

    The saved field is forwarded as it is. A confirmation recorded before this question existed
    carries null and stays not_answered: it is never read as a yes.
    """
    answered = [r for r in rows if r.get("followed_demonstration_form") is not None]
    out = {v: sum(1 for r in answered if r["followed_demonstration_form"] == v)
           for v in ("yes", "no", "other_form", "unsure")}
    return dict(out, answered=len(answered), not_answered=len(rows) - len(answered),
                denominator=denominator,
                denominator_n=len(rows),
                separate_from_followed_guidance="followed_guidance asks whether the instruction "
                                                "was followed; this asks whether the shown clip's "
                                                "form was followed. They are never merged.",
                is_not="an accuracy figure, a correctness verdict or an expert judgement")


def timing_summary(confirmed: list) -> dict:
    """Stored timing only, with its scope said out loud. Never a motion-end latency.

    `confirmed` must already be the population this figure belongs to. The caller decides that;
    this function does not filter, because a summary that quietly widened its own population is
    exactly the defect being fixed here.
    """
    vals, scopes, missing = [], {}, 0
    for r in confirmed:
        t = ((r.get("result") or {}).get("timing") or {})
        ms = t.get("server_end_candidate_to_result_ms")
        if ms is None:
            missing += 1
            continue
        vals.append(float(ms))
        sid = t.get("timing_scope_id") or "unlabelled_legacy_scope"
        scopes[sid] = scopes.get(sid, 0) + 1
    vals.sort()
    n = len(vals)
    # vals[n // 2] was the UPPER middle value on an even count: 100/200/400/800 answered 400,
    # not 300. An even count has no single middle, so the two middles are averaged.
    median = None if not n else (vals[n // 2] if n % 2
                                 else (vals[n // 2 - 1] + vals[n // 2]) / 2)
    return {"n": n,
            "median_ms": median,
            "mean_ms": round(sum(vals) / n, 2) if n else None,
            "min_ms": vals[0] if vals else None, "max_ms": vals[-1] if vals else None,
            "missing_timing": missing,
            "population_note": "this n is the number of attempts that stored a timing. It is a "
                               "DIFFERENT number from the recognition denominator: an attempt "
                               "with no result file or no stored timing stays in that denominator "
                               "and is counted here under missing_timing instead.",
            "scopes_seen": scopes,
            "scope_meaning": {
                "server_end_candidate_to_assembled_result_v2": "total pipeline: the server's end "
                                                               "candidate to the assembled "
                                                               "result, including model inference",
                "unlabelled_legacy_scope": "recorded before the scope was labelled; it excluded "
                                           "the neural inference"},
            "is_not": "the latency from the real end of the movement. Nothing here measures when "
                      "the hands actually stopped, or when a person saw anything."}


def report(run_dir: str, trials=None, trial_states=None, trial_meta=None) -> dict:
    """Join confirmations with the results already on disk. Same five candidates for both models.

    `trials` is every trial the session asked for, in order - including ones that were cancelled,
    errored or never produced a result. `trial_states` optionally maps trial id to its state so
    those can be counted apart. Nothing on disk is modified.
    """
    tdir = os.path.join(run_dir, "trials")
    if trials is None:
        trials = (sorted(f[: -len(".result.json")] for f in os.listdir(tdir)
                         if f.endswith(".result.json")) if os.path.isdir(tdir) else [])
    trials = list(trials)
    states = dict(trial_states or {})
    meta = dict(trial_meta or {})
    reviews, store_error = latest_by_trial(run_dir)

    cache = {}

    def _result(tid):
        """-> (result or None, reason or None). Read once; the cohorts need the same bytes."""
        if tid not in cache:
            rp = os.path.join(tdir, tid + ".result.json")
            if not os.path.exists(rp):
                cache[tid] = (None, "NO_RESULT_FILE")
            else:
                try:
                    with open(rp, encoding="utf-8") as fh:
                        cache[tid] = (json.load(fh), None)
                except (OSError, json.JSONDecodeError) as exc:
                    cache[tid] = (None, "RESULT_UNREADABLE:" + type(exc).__name__)
        return cache[tid]

    head = {
        "schema": "wis_trial_review_report_v2", "run_dir": run_dir,
        "generated_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "store_status": ("MALFORMED" if (store_error or "").startswith("STORE_MALFORMED")
                         else "UNREADABLE" if store_error else "OK"),
        "store_error": store_error,
        "comparison": {
            "candidates": list(COMMON_5),
            "meaning": "both models choose among the SAME five words, the only ones the neural "
                       "checkpoint and this project share. A restricted-candidate comparison.",
            "label": "참가자 확인 표현과의 일치",
            "is_not": ["새 사용자 일반화", "수행 정답률", "전문가 검증"],
        },
        "confirmation_source": "participant self-report, not expert verification",
        "not_configured": {
            "execution_quality": "NOT_CONFIGURED",
            "wrong_performance_rejection": "NOT_CONFIGURED",
            "combined_gate_pass_rate": "NOT_CONFIGURED",
            "meaning": "nothing here judges whether a sign was performed well",
        },
    }
    if store_error:
        # a read failure is not "no records". No score is produced at all.
        head.update(
            counts={"requested_total": len(trials), "trial_states": states},
            per_method=None, paired=None, per_word=None, confusion_matrix=None,
            rows=[], excluded=[],
            demonstration_form_followed=None, cohorts=None, full_scope=None,
            note="the confirmation store could not be read in full, so nothing was aggregated. "
                 "This is different from having no confirmations, and a malformed line is not a "
                 "smaller store - it is an unknown one.")
        return head

    counts = {"requested_total": len(trials), "confirmed_total": 0, "unconfirmed": 0,
              "confirmed_unknown": 0, "confirmed_outside_common5": 0,
              "confirmed_supported": 0, "no_result_file": 0, "result_unreadable": 0,
              "observation_insufficient": 0,
              "paired_valid": 0}
    by_state = {}
    for tid in trials:
        st = states.get(tid)
        if st:
            by_state[st] = by_state.get(st, 0) + 1

    rows, excluded = [], []
    method_tally = {m: {"answered": 0, "matched": 0, "unanswered": {}} for m in METHODS}

    for tid in trials:
        rev = reviews.get(tid)
        if rev is None:
            counts["unconfirmed"] += 1
            excluded.append({"trial_id": tid, "reason": "NOT_CONFIRMED",
                             "state": states.get(tid)})
            continue
        counts["confirmed_total"] += 1
        actual = rev["actual_word_id"]
        if actual == UNKNOWN:
            counts["confirmed_unknown"] += 1
            excluded.append({"trial_id": tid, "reason": "CONFIRMED_UNKNOWN"})
            continue
        if actual not in COMMON_5:
            counts["confirmed_outside_common5"] += 1
            excluded.append({"trial_id": tid, "reason": "OUTSIDE_COMMON_5",
                             "actual_word_id": actual})
            continue

        # a supported confirmation. It stays in every denominator from here, whatever happened
        # to the result, the observation or either model.
        counts["confirmed_supported"] += 1
        res, shared = _result(tid)
        if shared == "NO_RESULT_FILE":
            counts["no_result_file"] += 1
        elif shared:
            counts["result_unreadable"] += 1

        # ONLY a missing or unreadable result is a shared deficit. The top-level
        # observation_status belongs to the DTW predict path, so an observation it found
        # insufficient must not delete a neural answer that was produced from the same clip - the
        # adapter reports its own observation state under learned_recognition.model_status.
        dtw_block = None
        if shared is None and res.get("observation_status") != "OK":
            counts["observation_insufficient"] += 1
            dtw_block = "OBSERVATION_" + str(res.get("observation_status"))

        picks, reasons = {}, {}
        for m, fn in (("dtw", lambda r: _dtw_pick(r, COMMON_5)), ("neural", _neural_pick)):
            block = shared or (dtw_block if m == "dtw" else None)
            if block is not None:
                picks[m], reasons[m] = None, block
            else:
                picks[m], reasons[m] = fn(res)
            t = method_tally[m]
            if picks[m] is None:
                t["unanswered"][reasons[m]] = t["unanswered"].get(reasons[m], 0) + 1
            else:
                t["answered"] += 1
                t["matched"] += int(picks[m] == actual)

        row = {"trial_id": tid, "actual_word_id": actual,
               "followed_guidance": rev["followed_guidance"],
               "followed_demonstration_form": rev.get("followed_demonstration_form"),
               "requested_word_id": (res or {}).get("requested_word_id")
               or ((res or {}).get("task") or {}).get("requested_word_id"),
               "dtw_pick": picks["dtw"], "neural_pick": picks["neural"],
               "dtw_reason": reasons["dtw"], "neural_reason": reasons["neural"],
               "dtw_match": picks["dtw"] == actual if picks["dtw"] else None,
               "neural_match": picks["neural"] == actual if picks["neural"] else None,
               "paired": picks["dtw"] is not None and picks["neural"] is not None}
        rows.append(row)
        if row["paired"]:
            counts["paired_valid"] += 1
        else:
            excluded.append({"trial_id": tid, "reason": "NOT_PAIRED",
                             "dtw_reason": reasons["dtw"], "neural_reason": reasons["neural"]})

    # The five-word block above is unchanged and stays comparable with every earlier report.
    # The cohorts below are the part that was missing: they see EVERY confirmed attempt, so the
    # project words outside the five no longer vanish, and they read the candidate's own stored
    # answer instead of asking any model anything now.
    confirmed_rows = []
    for tid in trials:
        rev = reviews.get(tid)
        if rev is None:
            continue
        res, why = _result(tid)
        confirmed_rows.append({
            "trial_id": tid, "actual_word_id": rev["actual_word_id"],
            "followed_demonstration_form": rev.get("followed_demonstration_form"),
            "followed_guidance": rev.get("followed_guidance"),
            "result": res, "result_reason": why,
            "creation_research_model": (meta.get(tid) or {}).get("research_model")})
    cohort_block = cohorts(confirmed_rows)

    # The full trained-class scope sees every trial, so an opt-in attempt nobody confirmed is
    # counted as unconfirmed instead of vanishing. It reads the same cached result bytes.
    full_rows = []
    for tid in trials:
        rev = reviews.get(tid)
        res, why = _result(tid)
        full_rows.append({
            "trial_id": tid, "confirmed": rev is not None,
            "actual_word_id": rev["actual_word_id"] if rev else None,
            "result": res, "result_reason": why,
            "creation_research_model": (meta.get(tid) or {}).get("research_model")})
    full_block = full_scope(full_rows)

    paired_rows = [r for r in rows if r["paired"]]
    n = len(paired_rows)
    agree = {"both_match": sum(1 for r in paired_rows if r["dtw_match"] and r["neural_match"]),
             "dtw_only": sum(1 for r in paired_rows if r["dtw_match"] and not r["neural_match"]),
             "neural_only": sum(1 for r in paired_rows if r["neural_match"] and not r["dtw_match"]),
             "neither": sum(1 for r in paired_rows
                            if not r["dtw_match"] and not r["neural_match"])}

    def matrix(key):
        m = {a: {b: 0 for b in COMMON_5} for a in COMMON_5}
        for r in paired_rows:
            m[r["actual_word_id"]][r[key]] += 1
        return m

    supported = counts["confirmed_supported"]
    per_word = {}
    for w in COMMON_5:
        sub = [r for r in rows if r["actual_word_id"] == w]
        pair = [r for r in sub if r["paired"]]
        per_word[w] = {
            "confirmed_supported_n": len(sub), "paired_n": len(pair),
            "dtw_answered": sum(1 for r in sub if r["dtw_pick"] is not None),
            "dtw_correct": sum(1 for r in sub if r["dtw_match"]),
            "neural_answered": sum(1 for r in sub if r["neural_pick"] is not None),
            "neural_correct": sum(1 for r in sub if r["neural_match"]),
            "paired_dtw_correct": sum(1 for r in pair if r["dtw_match"]),
            "paired_neural_correct": sum(1 for r in pair if r["neural_match"]),
        }

    # a third axis with its own denominator: only the confirmations that answered it
    form_rows = [r for r in rows if r.get("followed_demonstration_form") is not None]
    form_tally = {"answered": len(form_rows), "not_answered": len(rows) - len(form_rows)}
    for v in ("yes", "no", "other_form", "unsure"):
        form_tally[v] = sum(1 for r in form_rows if r["followed_demonstration_form"] == v)

    head.update(
        demonstration_form_followed=dict(
            form_tally,
            scope="common5",
            scope_meaning="these counts come from the five-word block above, so the population is "
                          "confirmations whose word is one of the five. It is NOT every "
                          "confirmation in the session, and it is not any cohort's population - "
                          "each cohort reports its own under cohorts.groups[...].",
            denominator="only confirmations that answered this field; an older confirmation that "
                        "predates it is not_answered and is NEVER read as a yes",
            separate_from_followed_guidance="followed_guidance asks whether the instruction was "
                                            "followed; this asks whether the specific clip's form "
                                            "was followed. They are never merged.",
            is_not="an accuracy figure, a correctness verdict or an expert judgement"),
        counts=dict(counts, trial_states=by_state),
        per_method={
            "denominator": supported,
            "denominator_meaning": "every confirmed attempt whose word is one of the five. A "
                                   "missing result, an insufficient observation or one model "
                                   "being unavailable does NOT remove the attempt, and does not "
                                   "remove the other method's answer.",
            **{m: {"answered": method_tally[m]["answered"],
                   "matched": method_tally[m]["matched"],
                   "unanswered": method_tally[m]["unanswered"],
                   "match_over_all_supported": _score(method_tally[m]["matched"], supported),
                   "match_over_answered_only": _score(method_tally[m]["matched"],
                                                      method_tally[m]["answered"])}
               for m in METHODS}},
        paired={"n": n,
                "meaning": "attempts where BOTH methods answered. Never mixed with per_method.",
                "dtw": _score(agree["both_match"] + agree["dtw_only"], n),
                "neural": _score(agree["both_match"] + agree["neural_only"], n),
                "agreement": agree},
        per_word=per_word,
        confusion_matrix={"dtw": matrix("dtw_pick"), "neural": matrix("neural_pick"),
                          "rows": "confirmed actual word", "cols": "model pick",
                          "built_from": "paired rows only"},
        rows=rows, excluded=excluded,
        full_scope=dict(full_block,
                        relation_to_other_blocks="the candidate's own answer over every class it "
                                                 "was trained on. The five-word block and the "
                                                 "cohorts are restricted, easier tasks and are "
                                                 "never added to or compared with this one."),
        cohorts=dict(cohort_block,
                     relation_to_legacy_block="the five-word block above is unchanged. A cohort "
                                              "is a separate task with its own candidate set, and "
                                              "the two are never added together."))
    return head
