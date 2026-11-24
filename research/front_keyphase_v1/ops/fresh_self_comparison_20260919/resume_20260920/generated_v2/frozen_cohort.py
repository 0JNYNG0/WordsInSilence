"""frozen_cohort.py

Offline evaluation support for a frozen re-test cohort.

Standard library only.  No I/O is performed here: a trusted local audit driver
reads the original server-side artifacts (plan, capture journal, session record,
append-only review rows, timing events, saved results, artifact inventory) and
hands them over as plain JSON objects.

Two entry points:

    build_cohort(plan, journal, session, reviews, events, results, artifacts)
        -> a frozen, fully enumerated cohort (one row per planned slot).

    summarize(cohort, predictions, label_map)
        -> a report over caller-supplied, already-normalized predictions.

Design rules enforced throughout:

  * Truth is only ever the participant's latest self-reported ``actual_word_id``.
    Requested words, client claims and model predictions are never treated as
    truth, and a missing label is never back-filled from the requested word.
  * Structural mismatch raises ``ValueError``.  Validation uses explicit checks
    (not ``assert``) so that ``python -O`` cannot disable it.
  * Nothing is silently dropped: missing, cancelled, unverified and unconfirmed
    attempts survive as rows with explicit status fields.
  * No training, no model selection, no correctness assessment of the
    participant, no promotion decisions.
"""

import math
import re
from datetime import datetime

ROUNDS = 3
WORDS_PER_ROUND = 18
PLANNED_SLOTS = ROUNDS * WORDS_PER_ROUND  # 54

PLAN_SCHEMA = "wis_fresh_self_capture_plan_v1"
JOURNAL_SCHEMA = "wis_frozen_capture_journal_v1"
REVIEW_SOURCE = "participant_self_report"

ARMS = ("parent", "control", "treatment")
SCOPES = ("project18", "full93")
PAIRS = (
    ("treatment_vs_control", "control", "treatment"),
    ("treatment_vs_parent", "parent", "treatment"),
    ("control_vs_parent", "parent", "control"),
)

NO_PREDICTION = "__NO_PREDICTION__"
UNKNOWN_LABEL = "UNKNOWN"
FAILURE_OUTCOMES = ("CANCELLED", "ERROR", "TIMEOUT")

LIMITS = [
    "single participant who was exposed to the development set",
    "labels are participant self-report of the word actually signed, "
    "not an expert judgement of articulation correctness",
    "no promotion or release decision is supported by this report",
    "no training, tuning or model selection was performed",
    "stored server timings are processing timings and are reported "
    "separately from offline inference elapsed_ms",
]

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


# --------------------------------------------------------------------------
# validation primitives
# --------------------------------------------------------------------------

def _fail(message):
    raise ValueError(message)


def _obj(value, where):
    if not isinstance(value, dict):
        _fail("%s must be a JSON object" % where)
    return value


def _seq(value, where):
    if not isinstance(value, list):
        _fail("%s must be a JSON array" % where)
    return value


def _text(value, where):
    if not isinstance(value, str) or value == "":
        _fail("%s must be a non-empty string" % where)
    return value


def _safe_id(value, where):
    _text(value, where)
    if not _SAFE_ID.match(value):
        _fail("%s is not a filename-safe identifier: %r" % (where, value))
    return value


def _exact(value, expected, where):
    if value != expected:
        _fail("%s must be %r, got %r" % (where, expected, value))
    return value


def _index(value, where, low, high):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("%s must be an integer" % where)
    if value < low or value > high:
        _fail("%s must be in [%d, %d], got %d" % (where, low, high, value))
    return value


def _flag(value, where):
    if not isinstance(value, bool):
        _fail("%s must be a boolean" % where)
    return value


def _finite(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("%s must be a number" % where)
    number = float(value)
    if not math.isfinite(number):
        _fail("%s must be finite, got %r" % (where, value))
    return number


def _optional_text(value, where):
    if value is None:
        return None
    return _text(value, where)


def _epoch_of(stamp, where):
    _text(stamp, where)
    text = stamp[:-1] + "+00:00" if stamp.endswith("Z") else stamp
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        _fail("%s is not an ISO-8601 timestamp: %r" % (where, stamp))
    if moment.tzinfo is None:
        _fail("%s must carry an explicit UTC offset: %r" % (where, stamp))
    return moment.timestamp()


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------

def _check_plan(plan):
    _obj(plan, "plan")
    _exact(plan.get("schema"), PLAN_SCHEMA, "plan.schema")
    protocol_id = _safe_id(plan.get("protocol_id"), "plan.protocol_id")
    _exact(plan.get("participant_id"), "self", "plan.participant_id")
    _exact(plan.get("purpose"), "frozen_retest", "plan.purpose")
    _text(plan.get("reference_set_id"), "plan.reference_set_id")
    _exact(plan.get("rounds"), ROUNDS, "plan.rounds")
    prepared_epoch = _epoch_of(plan.get("prepared_at_kst"), "plan.prepared_at_kst")

    clip_of = {}
    for position, word in enumerate(_seq(plan.get("words"), "plan.words")):
        _obj(word, "plan.words[%d]" % position)
        word_id = _safe_id(word.get("word_id"), "plan.words[%d].word_id" % position)
        if word_id in clip_of:
            _fail("plan.words contains duplicate word_id %r" % word_id)
        clip_of[word_id] = _safe_id(
            word.get("clip_id"), "plan.words[%d].clip_id" % position)
        _text(word.get("display_name"), "plan.words[%d].display_name" % position)
    if len(clip_of) != WORDS_PER_ROUND:
        _fail("plan.words must hold exactly %d words, got %d"
              % (WORDS_PER_ROUND, len(clip_of)))

    slots = {}
    per_round = {index: set() for index in range(1, ROUNDS + 1)}
    for position, slot in enumerate(_seq(plan.get("slots"), "plan.slots")):
        where = "plan.slots[%d]" % position
        _obj(slot, where)
        number = _index(slot.get("slot"), where + ".slot", 1, PLANNED_SLOTS)
        if number != position + 1:
            _fail("%s.slot must be %d (ascending order), got %d"
                  % (where, position + 1, number))
        round_no = _index(slot.get("round"), where + ".round", 1, ROUNDS)
        if round_no != ((number - 1) // WORDS_PER_ROUND) + 1:
            _fail("%s.round must be %d for slot %d"
                  % (where, ((number - 1) // WORDS_PER_ROUND) + 1, number))
        word_id = _text(slot.get("word_id"), where + ".word_id")
        if number in slots:
            _fail("plan.slots repeats slot %d" % number)
        if word_id not in clip_of:
            _fail("%s.word_id %r is not in plan.words" % (where, word_id))
        if word_id in per_round[round_no]:
            _fail("round %d repeats word %r" % (round_no, word_id))
        per_round[round_no].add(word_id)
        slots[number] = {"slot": number, "round": round_no, "word_id": word_id}
    if len(slots) != PLANNED_SLOTS:
        _fail("plan.slots must hold exactly %d slots, got %d"
              % (PLANNED_SLOTS, len(slots)))
    for round_no in range(1, ROUNDS + 1):
        if len(per_round[round_no]) != WORDS_PER_ROUND:
            _fail("round %d does not cover all %d words" % (round_no, WORDS_PER_ROUND))

    return protocol_id, clip_of, slots, prepared_epoch


def _check_journal(journal, protocol_id, plan_slots):
    _obj(journal, "journal")
    _exact(journal.get("schema"), JOURNAL_SCHEMA, "journal.schema")
    _exact(journal.get("protocol_id"), protocol_id, "journal.protocol_id")
    client_id = _safe_id(journal.get("client_id"), "journal.client_id")
    session_id = _safe_id(journal.get("session_id"), "journal.session_id")
    _text(journal.get("status"), "journal.status")
    _index(journal.get("current_slot"), "journal.current_slot", 1, PLANNED_SLOTS)

    rows = {}
    seen_trials = {}
    for position, entry in enumerate(_seq(journal.get("slots"), "journal.slots")):
        where = "journal.slots[%d]" % position
        _obj(entry, where)
        number = _index(entry.get("slot"), where + ".slot", 1, PLANNED_SLOTS)
        if number != position + 1:
            _fail("%s.slot must be %d (ascending order), got %d"
                  % (where, position + 1, number))
        if number in rows:
            _fail("journal.slots repeats slot %d" % number)
        planned = plan_slots.get(number)
        if planned is None:
            _fail("%s refers to unplanned slot %d" % (where, number))
        if entry.get("round") != planned["round"]:
            _fail("%s.round disagrees with the plan for slot %d" % (where, number))
        requested = _text(entry.get("requested_word_id"), where + ".requested_word_id")
        if requested != planned["word_id"]:
            _fail("%s.requested_word_id disagrees with the plan for slot %d"
                  % (where, number))

        trial_id = entry.get("trial_id")
        request_id = entry.get("request_id")
        if trial_id is not None:
            _safe_id(trial_id, where + ".trial_id")
            if trial_id in seen_trials:
                _fail("trial_id %r appears in slots %d and %d"
                      % (trial_id, seen_trials[trial_id], number))
            seen_trials[trial_id] = number
            expected = "%s:%s:%d:%s" % (protocol_id, client_id, number, requested)
            if request_id != expected:
                _fail("%s.request_id must be %r, got %r" % (where, expected, request_id))
        elif request_id is not None:
            _text(request_id, where + ".request_id")

        rows[number] = {
            "slot": number,
            "round": planned["round"],
            "requested_word_id": requested,
            "trial_id": trial_id,
            "client_terminal_state": _optional_text(
                entry.get("terminal_state"), where + ".terminal_state"),
            "review_saved": bool(entry.get("review_saved")),
            "explicit_skip": bool(entry.get("explicit_skip")),
            "client_review_revision": entry.get("review_revision"),
            "transport_error": bool(entry.get("transport_error")),
        }
    if len(rows) != PLANNED_SLOTS:
        _fail("journal.slots must contain every planned slot (%d), got %d"
              % (PLANNED_SLOTS, len(rows)))
    ordered_trials = [rows[n]["trial_id"] for n in sorted(rows)
                      if rows[n]["trial_id"] is not None]
    return client_id, session_id, rows, ordered_trials


def _check_session(session, plan, client_id, session_id, ordered_trials, prepared_epoch):
    _obj(session, "session")
    _exact(session.get("session_id"), session_id, "session.session_id")
    _exact(session.get("client_id"), client_id, "session.client_id")
    _exact(session.get("participant_id"), plan["participant_id"], "session.participant_id")
    _exact(session.get("purpose"), plan["purpose"], "session.purpose")
    _exact(session.get("state"), "CLOSED", "session.state")
    _exact(session.get("store_video"), False, "session.store_video")
    _exact(session.get("camera"), False, "session.camera")
    created = _finite(session.get("created_epoch"), "session.created_epoch")
    if created < prepared_epoch:
        _fail("session was created before the plan was prepared; this is an older "
              "development session, not the frozen re-test session")
    listed = _seq(session.get("trial_ids"), "session.trial_ids")
    for position, trial_id in enumerate(listed):
        _safe_id(trial_id, "session.trial_ids[%d]" % position)
    if listed != ordered_trials:
        _fail("session.trial_ids do not match the journal trial ids in slot order; "
              "unmatched server attempts must be resolved, never dropped")
    return created


def _check_reviews(reviews, session_id, protocol_id, trial_slots, slot_rows, words):
    latest = {}
    seen = {}
    for position, row in enumerate(_seq(reviews, "reviews")):
        where = "reviews[%d]" % position
        _obj(row, where)
        _exact(row.get("session_id"), session_id, where + ".session_id")
        trial_id = _text(row.get("trial_id"), where + ".trial_id")
        if trial_id not in trial_slots:
            _fail("%s refers to trial %r which is not in this session" % (where, trial_id))
        _exact(row.get("source"), REVIEW_SOURCE, where + ".source")
        revision = row.get("revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            _fail("%s.revision must be a positive integer" % where)
        key = (trial_id, revision)
        if key in seen and seen[key] != row:
            _fail("conflicting duplicate revision %d for trial %r" % (revision, trial_id))
        seen[key] = row
        if trial_id not in latest or revision > latest[trial_id]["revision"]:
            latest[trial_id] = {"revision": revision, "row": row, "where": where}

    confirmed = {}
    for trial_id, keep in latest.items():
        row, where = keep["row"], keep["where"]
        slot = trial_slots[trial_id]
        prefix = "protocol=%s;slot=%d;" % (protocol_id, slot)
        note = row.get("note")
        if not isinstance(note, str) or not note.startswith(prefix):
            _fail("%s.note must start with %r" % (where, prefix))
        actual = row.get("actual_word_id")
        if actual is not None and not isinstance(actual, str):
            _fail("%s.actual_word_id must be a string or null" % where)
        if actual is not None and actual != "" and actual != UNKNOWN_LABEL:
            if actual not in words:
                _fail("%s.actual_word_id %r is outside the %d planned words"
                      % (where, actual, WORDS_PER_ROUND))
        else:
            actual = None
        confirmed[trial_id] = {
            "actual_word_id": actual,
            "review_revision": keep["revision"],
            "followed_guidance": row.get("followed_guidance"),
            "followed_demonstration_form": row.get("followed_demonstration_form"),
        }
    return confirmed


def _check_events(events, session_id, trial_slots):
    closed = {}
    for position, event in enumerate(_seq(events, "events")):
        where = "events[%d]" % position
        _obj(event, where)
        _exact(event.get("session_id"), session_id, where + ".session_id")
        trial_id = _text(event.get("trial_id"), where + ".trial_id")
        if trial_id not in trial_slots:
            _fail("%s refers to trial %r which is not in this session" % (where, trial_id))
        if event.get("event") != "trial_closed":
            continue
        outcome = event.get("outcome")
        if outcome is None:
            outcome = event.get("state")
        _text(outcome, where + ".outcome")
        if outcome not in FAILURE_OUTCOMES:
            _fail("%s.outcome %r is not a recognised trial_closed outcome"
                  % (where, outcome))
        if trial_id in closed and closed[trial_id] != outcome:
            _fail("conflicting trial_closed outcomes %r and %r for trial %r"
                  % (closed[trial_id], outcome, trial_id))
        closed[trial_id] = outcome
    return closed


def _check_results(results, trial_slots, slot_rows, clip_of):
    checked = {}
    for trial_id, result in _obj(results, "results").items():
        if trial_id not in trial_slots:
            _fail("results contains unknown trial %r" % trial_id)
        where = "results[%r]" % trial_id
        _obj(result, where)
        _exact(result.get("origin"), "camera", where + ".origin")
        digest = _text(result.get("observation_sha256"), where + ".observation_sha256")
        binding = _obj(result.get("reference_binding"), where + ".reference_binding")
        requested = slot_rows[trial_slots[trial_id]]["requested_word_id"]
        _exact(binding.get("requested_word_id"), requested,
               where + ".reference_binding.requested_word_id")
        if "clip_id" not in binding:
            _fail("%s.reference_binding is missing clip_id" % where)
        _exact(binding.get("clip_id"), clip_of[requested],
               where + ".reference_binding.clip_id")
        timing = result.get("timing")
        if timing is not None:
            _obj(timing, where + ".timing")
        checked[trial_id] = {"observation_sha256": digest, "server_timing": timing}
    return checked


def _artifact_status(trial_id, artifacts, result, where_map):
    if trial_id not in artifacts:
        return "ARTIFACTS_UNKNOWN", False
    entry = _obj(artifacts[trial_id], "artifacts[%r]" % trial_id)
    where = "artifacts[%r]" % trial_id
    npz_exists = _flag(entry.get("npz_exists"), where + ".npz_exists")
    result_exists = _flag(entry.get("result_exists"), where + ".result_exists")
    sidecar_exists = _flag(entry.get("sidecar_exists"), where + ".sidecar_exists")
    npz_digest = entry.get("npz_sha256")
    sidecar_digest = entry.get("sidecar_observation_sha256")
    if result is None or not result_exists:
        return "NO_RESULT", False
    if not npz_exists:
        return "MISSING_NPZ", False
    if not sidecar_exists:
        return "MISSING_SIDECAR", False
    if not isinstance(npz_digest, str) or npz_digest == "":
        return "EMPTY_NPZ_DIGEST", False
    if npz_digest != result["observation_sha256"] or npz_digest != sidecar_digest:
        return "HASH_MISMATCH", False
    return "OK", True


# --------------------------------------------------------------------------
# build_cohort
# --------------------------------------------------------------------------

def build_cohort(plan, journal, session, reviews, events, results, artifacts):
    """Freeze one row per planned slot from original server-side evidence."""
    protocol_id, clip_of, plan_slots, prepared_epoch = _check_plan(plan)
    client_id, session_id, slot_rows, ordered_trials = _check_journal(
        journal, protocol_id, plan_slots)
    _check_session(session, plan, client_id, session_id, ordered_trials, prepared_epoch)

    trial_slots = {row["trial_id"]: row["slot"] for row in slot_rows.values()
                   if row["trial_id"] is not None}
    review_by_trial = _check_reviews(
        reviews, session_id, protocol_id, trial_slots, slot_rows, clip_of)
    closed_outcomes = _check_events(events, session_id, trial_slots)
    result_by_trial = _check_results(results, trial_slots, slot_rows, clip_of)
    _obj(artifacts, "artifacts")
    for trial_id in artifacts:
        if trial_id not in trial_slots:
            _fail("artifacts contains unknown trial %r" % trial_id)

    rows = []
    counts = {
        "total": PLANNED_SLOTS, "captured": 0, "not_captured": 0,
        "confirmed": 0, "unconfirmed": 0, "eligible": 0,
        "actual_differs_from_requested": 0,
        "terminal_states": {}, "artifact_states": {},
    }

    for number in sorted(slot_rows):
        base = slot_rows[number]
        trial_id = base["trial_id"]
        requested = base["requested_word_id"]
        review = review_by_trial.get(trial_id) if trial_id else None
        result = result_by_trial.get(trial_id) if trial_id else None

        if trial_id is None:
            terminal_state = "NOT_CAPTURED"
            artifact_status, eligible = "NO_TRIAL", False
        else:
            outcome = closed_outcomes.get(trial_id)
            if outcome is not None and outcome in FAILURE_OUTCOMES:
                if result is not None:
                    _fail("trial %r has a saved result but the server recorded a "
                          "closed %s outcome" % (trial_id, outcome))
                terminal_state = outcome
            elif result is not None:
                terminal_state = "FEEDBACK"
            elif outcome is not None:
                terminal_state = outcome
            else:
                terminal_state = "UNVERIFIED_TERMINAL"
            artifact_status, eligible = _artifact_status(
                trial_id, artifacts, result, slot_rows)

        actual = review["actual_word_id"] if review else None
        row = {
            "slot": number,
            "round": base["round"],
            "requested_word_id": requested,
            "trial_id": trial_id,
            "terminal_state": terminal_state,
            "client_terminal_state": base["client_terminal_state"],
            "actual_word_id": actual,
            "review_revision": review["review_revision"] if review else None,
            "followed_guidance": review["followed_guidance"] if review else None,
            "followed_demonstration_form": (
                review["followed_demonstration_form"] if review else None),
            "actual_differs_from_requested": (
                actual is not None and actual != requested),
            "artifact_status": artifact_status,
            "inference_eligible": eligible,
            "observation_sha256": result["observation_sha256"] if result else None,
            "server_timing": result["server_timing"] if result else None,
            "explicit_skip": base["explicit_skip"],
            "transport_error": base["transport_error"],
        }
        rows.append(row)

        if trial_id is None:
            counts["not_captured"] += 1
        else:
            counts["captured"] += 1
        if actual is None:
            counts["unconfirmed"] += 1
        else:
            counts["confirmed"] += 1
        if row["actual_differs_from_requested"]:
            counts["actual_differs_from_requested"] += 1
        if eligible:
            counts["eligible"] += 1
        counts["terminal_states"][terminal_state] = (
            counts["terminal_states"].get(terminal_state, 0) + 1)
        counts["artifact_states"][artifact_status] = (
            counts["artifact_states"].get(artifact_status, 0) + 1)

    return {
        "protocol_id": protocol_id,
        "session_id": session_id,
        "client_id": client_id,
        "journal_status": journal["status"],
        "planned_slots": PLANNED_SLOTS,
        "rows": rows,
        "counts": counts,
    }


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def _check_label_map(label_map, planned_words):
    _obj(label_map, "label_map")
    if set(label_map) != planned_words:
        _fail("label_map must cover exactly the %d planned word ids"
              % WORDS_PER_ROUND)
    labels = {}
    for word_id in sorted(label_map):
        label = _text(label_map[word_id], "label_map[%r]" % word_id)
        if label in labels:
            _fail("label_map maps %r and %r to the same class label %r"
                  % (labels[label], word_id, label))
        labels[label] = word_id
    return labels


def _check_predictions(predictions, eligible):
    _obj(predictions, "predictions")
    normalized = {}
    elapsed = {arm: 0.0 for arm in ARMS}
    for trial_id in predictions:
        if trial_id not in eligible:
            _fail("predictions contains trial %r which is not inference-eligible"
                  % trial_id)
    for trial_id in sorted(predictions):
        entry = _obj(predictions[trial_id], "predictions[%r]" % trial_id)
        if set(entry) != set(ARMS):
            _fail("predictions[%r] must contain exactly the arms %s"
                  % (trial_id, ", ".join(ARMS)))
        per_arm = {}
        for arm in ARMS:
            where = "predictions[%r][%r]" % (trial_id, arm)
            arm_entry = _obj(entry[arm], where)
            status = entry[arm].get("status")
            if status not in ("READY", "ERROR"):
                _fail("%s.status must be READY or ERROR, got %r" % (where, status))
            ms = _finite(arm_entry.get("elapsed_ms"), where + ".elapsed_ms")
            if ms < 0:
                _fail("%s.elapsed_ms must be non-negative" % where)
            elapsed[arm] += ms
            error = arm_entry.get("error")
            if error is not None and not isinstance(error, str):
                _fail("%s.error must be a string or null" % where)
            scoped = {}
            for scope in SCOPES:
                value = arm_entry.get(scope)
                if value is not None and not isinstance(value, str):
                    _fail("%s.%s must be a string or null" % (where, scope))
                # an ERROR arm never contributes a prediction, whatever it carries
                scoped[scope] = value if status == "READY" and value != "" else None
            per_arm[arm] = {"status": status, "elapsed_ms": ms,
                            "error": error, "predictions": scoped}
        normalized[trial_id] = per_arm
    return normalized, elapsed


def _scope_truth(row, scope, labels_by_word):
    word_id = row["actual_word_id"]
    if word_id is None:
        return None
    return word_id if scope == "project18" else labels_by_word[word_id]


def _arm_report(rows, confirmed, scope, arm, normalized, classes):
    predicted_count_all = 0
    correct = 0
    failures = 0
    support = {cls: 0 for cls in classes}
    hits = {cls: 0 for cls in classes}
    false_positives = {cls: 0 for cls in classes}
    confusion = {}
    prediction_of = {}

    for row in rows:
        trial_id = row["trial_id"]
        entry = normalized.get(trial_id) if trial_id else None
        predicted = entry[arm]["predictions"][scope] if entry else None
        prediction_of[row["slot"]] = predicted
        if predicted is not None:
            predicted_count_all += 1

    for row in confirmed:
        truth = row["_truth"][scope]
        predicted = prediction_of[row["slot"]]
        support[truth] += 1
        bucket = confusion.setdefault(truth, {})
        key = NO_PREDICTION if predicted is None else predicted
        bucket[key] = bucket.get(key, 0) + 1
        if predicted is None:
            failures += 1
            continue
        if predicted == truth:
            correct += 1
            hits[truth] += 1
        elif predicted in false_positives:
            false_positives[predicted] += 1

    supported = [cls for cls in classes if support[cls] > 0]
    f1_total = 0.0
    for cls in supported:
        tp, fn, fp = hits[cls], support[cls] - hits[cls], false_positives[cls]
        denominator = 2 * tp + fp + fn
        f1_total += (2.0 * tp / denominator) if denominator else 0.0
    macro_f1 = (f1_total / len(supported)) if supported else None

    confirmed_n = len(confirmed)
    per_word = {}
    for cls in classes:
        recall = (hits[cls] / support[cls]) if support[cls] else None
        per_word[cls] = {"support": support[cls], "correct": hits[cls],
                         "recall": recall}

    return {
        "planned": PLANNED_SLOTS,
        "confirmed_n": confirmed_n,
        "correct": correct,
        "correct_over_planned": correct,
        "correct_over_planned_rate": correct / float(PLANNED_SLOTS),
        "correct_over_planned_note": (
            "coverage measure: slots without a confirmed self-reported label are "
            "not counted as wrong, only as not covered"),
        "accuracy_over_confirmed": (correct / float(confirmed_n)
                                    if confirmed_n else None),
        "inference_failures_on_confirmed": failures,
        "predicted_count_all": predicted_count_all,
        "per_class": per_word,
        "confusion": confusion,
        "macro_f1": macro_f1,
        "macro_f1_actual_class_count": len(supported),
        "_prediction_of": prediction_of,
    }, prediction_of


def summarize(cohort, predictions, label_map):
    """Report frozen-cohort outcomes for caller-normalized predictions."""
    _obj(cohort, "cohort")
    rows = _seq(cohort.get("rows"), "cohort.rows")
    if len(rows) != PLANNED_SLOTS:
        _fail("cohort must contain exactly %d rows" % PLANNED_SLOTS)

    planned_words = {row["requested_word_id"] for row in rows}
    labels_by_word = {}
    label_to_word = _check_label_map(label_map, planned_words)
    for label, word_id in label_to_word.items():
        labels_by_word[word_id] = label

    eligible = {row["trial_id"] for row in rows
                if row["inference_eligible"] and row["trial_id"]}
    normalized, elapsed = _check_predictions(predictions, eligible)

    confirmed = []
    for row in rows:
        if row["actual_word_id"] is not None:
            row = dict(row)
            row["_truth"] = {scope: _scope_truth(row, scope, labels_by_word)
                             for scope in SCOPES}
            confirmed.append(row)

    classes_for = {
        "project18": sorted(planned_words),
        "full93": sorted(labels_by_word[w] for w in planned_words),
    }

    report_scopes = {}
    for scope in SCOPES:
        arm_reports = {}
        arm_predictions = {}
        for arm in ARMS:
            arm_reports[arm], arm_predictions[arm] = _arm_report(
                rows, confirmed, scope, arm, normalized, classes_for[scope])
            arm_reports[arm].pop("_prediction_of", None)

        pair_reports = {}
        for name, base_arm, other_arm in PAIRS:
            gains, regressions, changes = [], [], []
            both_answered = 0
            for row in confirmed:
                truth = row["_truth"][scope]
                base = arm_predictions[base_arm][row["slot"]]
                other = arm_predictions[other_arm][row["slot"]]
                if base is not None and other is not None:
                    both_answered += 1
                if base != other:
                    changes.append(row["slot"])
                if other == truth and base != truth:
                    gains.append(row["slot"])
                elif base == truth and other != truth:
                    regressions.append(row["slot"])
            pair_reports[name] = {
                "baseline_arm": base_arm,
                "candidate_arm": other_arm,
                "confirmed_n": len(confirmed),
                "gains": gains,
                "regressions": regressions,
                "prediction_changes": changes,
                "both_answered": both_answered,
            }
        report_scopes[scope] = {"arms": arm_reports, "pairs": pair_reports}

    return {
        "protocol_id": cohort.get("protocol_id"),
        "session_id": cohort.get("session_id"),
        "planned_slots": PLANNED_SLOTS,
        "confirmed_n": len(confirmed),
        "eligible_n": len(eligible),
        "scopes": report_scopes,
        "offline_elapsed_ms_by_arm": {arm: elapsed[arm] for arm in ARMS},
        "offline_elapsed_note": (
            "offline inference timing only; stored server processing timings live "
            "on each cohort row as server_timing and are never mixed in"),
        "actual_requested_mismatches": [
            {"slot": row["slot"], "requested_word_id": row["requested_word_id"],
             "actual_word_id": row["actual_word_id"]}
            for row in rows if row["actual_differs_from_requested"]],
        "unconfirmed_slots": [row["slot"] for row in rows
                              if row["actual_word_id"] is None
                              and row["trial_id"] is not None],
        "not_captured_slots": [row["slot"] for row in rows
                               if row["trial_id"] is None],
        "limits": list(LIMITS),
    }
