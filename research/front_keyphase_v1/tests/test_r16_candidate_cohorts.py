"""functional_19: opt-in candidate cohorts, and a confirmation store that will not be misread.

Synthetic fixtures only. No model is loaded, no NPZ is opened, no real store is touched.

Run: py -3.13 tests/test_r16_candidate_cohorts.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from wis_lab.service import review_log  # noqa: E402

COMMON = ["ALDA", "GAMSAHAMNIDA", "NA", "JOTA", "MOREUDA"]
PROJECT = sorted(COMMON + ["GAM", "GONGBU", "NEO", "MAEKJU", "MEOKDA", "MUL", "PPANG", "SARANG",
                           "SILTA", "UNDONG", "INSA", "JADA", "JOESONGHADA"])
H1, H2, BASE_CK = "a" * 64, "c" * 64, "b" * 64

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def identity(h=H1):
    return {"opted_in": True, "fixed_at": "trial_creation", "model_key": "union93_candidate_v1",
            "model_id": "union93_candidate_v1", "checkpoint_sha256": h,
            "trained_class_count": 93, "feature_schema": "pilot.data.build_channels",
            "project_words_in_model": list(PROJECT), "model_status": "READY"}


def scope(words, pred, name):
    order = sorted(({"word_id": w, "score": 10.0 if w == pred else -float(i + 1)}
                    for i, w in enumerate(words)), key=lambda r: -r["score"])
    return {"scope": name, "candidate_count": len(words), "top1": order[0]["word_id"],
            "order": order}


def result(tid, dtw, base, cand, h=H1):
    ident = identity(h)
    return {"trial_id": tid, "observation_status": "OK", "requested_word_id": "UNDONG",
            "research_model": ident, "top1_word_id": dtw,
            "ranking": [{"word_id": w, "status": "OK",
                         "distance": 0.0 if w == dtw else 1.0 + i / 100}
                        for i, w in enumerate(PROJECT)],
            "timing": {"server_end_candidate_to_result_ms": 250.0,
                       "timing_scope_id": "server_end_candidate_to_assembled_result_v2"},
            "learned_recognition": {
                "model_id": "base", "checkpoint_sha256": BASE_CK, "model_status": "READY",
                "ranking": {"restricted_to_project_words": scope(COMMON, base, "common5")}},
            "research_comparison": {
                "model_id": "union93_candidate_v1", "checkpoint_sha256": h,
                "model_status": "READY", "is_under_validation": True,
                "identity_fixed_at_trial_creation": copy.deepcopy(ident),
                "scopes": {"common5": scope(COMMON, cand, "common5"),
                           "project18": scope(PROJECT, cand, "project18")}}}


def build(run_dir, specs, extra_reviews=()):
    """specs: (tid, actual, dtw, base, cand, hash, mutate). mutate may return WRITE_NO_RESULT."""
    os.makedirs(os.path.join(run_dir, "trials"))
    ids, states, meta, reviews = [], {}, {}, []
    for tid, actual, dp, bp, cp, h, mutate in specs:
        ids.append(tid)
        states[tid] = "FEEDBACK"
        data = result(tid, dp, bp, cp, h)
        write = True
        if mutate:
            write = mutate(data) is not False
        if data.get("research_model"):
            meta[tid] = {"research_model": copy.deepcopy(data["research_model"])}
        if write:
            with open(os.path.join(run_dir, "trials", tid + ".result.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
        if actual is not None:
            reviews.append({"schema": review_log.REVIEW_SCHEMA, "event": "trial_review",
                            "session_id": "synthetic19", "trial_id": tid, "revision": 1,
                            "actual_word_id": actual, "followed_guidance": "yes",
                            "followed_demonstration_form": "yes", "note": None,
                            "source": "participant_self_report", "is_expert_verification": False})
    reviews.extend(extra_reviews)
    with open(os.path.join(run_dir, review_log.REVIEW_FILE), "w", encoding="utf-8",
              newline="\n") as fh:
        for r in reviews:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return ids, states, meta


def drop_result(data):
    return False


def opt_out(data):
    data["research_model"] = {"opted_in": False, "model_key": None}
    # a stray candidate block is left lying beside the refusal on purpose
    return True


def legacy(data):
    data.pop("research_model")
    data.pop("research_comparison")
    return True


def failed_model(data):
    data["observation_status"] = "INSUFFICIENT_OBSERVATION"
    data["research_comparison"]["model_status"] = "INFERENCE_FAILED"
    data["research_comparison"]["scopes"] = None
    return True


def short_18(data):
    o = data["research_comparison"]["scopes"]["project18"]["order"]
    data["research_comparison"]["scopes"]["project18"]["order"] = [x for x in o
                                                                   if x["word_id"] != "NA"]
    return True


def answered_by_other(data):
    data["research_comparison"]["checkpoint_sha256"] = H2
    data["research_comparison"]["identity_fixed_at_trial_creation"]["checkpoint_sha256"] = H2
    return True


def no_identity(data):
    data.pop("research_model")
    data["research_comparison"].pop("identity_fixed_at_trial_creation")
    data["research_comparison"].pop("model_id")
    data["research_comparison"].pop("checkpoint_sha256")
    return True


def tagged(groups, scope_name, h):
    for key, g in groups.items():
        if g["scope"] == scope_name and h in key:
            return g
    return None


# ------------------------------------------------------------------ 1. the store, misread before
def test_store():
    tmp = tempfile.mkdtemp(prefix="f19a_")
    try:
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        p = os.path.join(run, review_log.REVIEW_FILE)
        good = {"schema": review_log.REVIEW_SCHEMA, "trial_id": "t1", "revision": 1,
                "actual_word_id": "NA", "followed_guidance": "yes",
                "followed_demonstration_form": None, "note": None}
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(good) + "\n")
            fh.write('{"schema": "wis_trial_review_v1", "trial_id": "t2", "revi')  # torn write

        rows, err = review_log.read_all(run)
        ok(err is not None and err.startswith("STORE_MALFORMED"), "torn line reports MALFORMED")
        ok(rows == [], "a malformed store yields no rows at all")

        rep = review_log.report(run, ["t1", "t2"], {"t1": "FEEDBACK"})
        ok(rep["store_status"] == "MALFORMED", "report says MALFORMED, not OK and not UNREADABLE")
        ok(rep["per_method"] is None and rep["paired"] is None and rep["cohorts"] is None,
           "no score of any kind is produced from a malformed store")
        ok(rep["rows"] == [], "no rows are published from a malformed store")

        before = open(p, encoding="utf-8").read()
        out = review_log.submit(run, "s", "t3", "NA", "yes", known_words=COMMON)
        ok(out["status"] == "NOT_STORED" and out["reason"].startswith("STORE_MALFORMED"),
           "submit refuses to append to a malformed store")
        ok(open(p, encoding="utf-8").read() == before, "the store is left exactly as it was")

        # a line that parses but is not a review row is also a malformed store
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(good) + "\n")
            fh.write("[1, 2, 3]\n")
        _, err2 = review_log.read_all(run)
        ok(err2 is not None and err2.startswith("STORE_MALFORMED"),
           "a parseable non-row is malformed too")

        # and a clean store still reads, so the guard did not break the normal path
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(good) + "\n\n")
        rows3, err3 = review_log.read_all(run)
        ok(err3 is None and len(rows3) == 1, "a clean store (blank line included) still reads")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 2. cohorts
def test_cohorts():
    tmp = tempfile.mkdtemp(prefix="f19b_")
    try:
        run = os.path.join(tmp, "run")
        specs = [
            ("c1", "NA", "NA", "GAMSAHAMNIDA", "NA", H1, None),           # base misses
            ("c2", "INSA", "INSA", "ALDA", "INSA", H1, None),             # outside common5
            ("c3", "ALDA", "NA", "ALDA", "NA", H1, failed_model),         # one method failed
            ("c4", "JOTA", "JOTA", "JOTA", "JOTA", H1, drop_result),      # missing result
            ("c5", None, "NA", "NA", "NA", H1, None),                     # unconfirmed
            ("c6", "MOREUDA", "MOREUDA", "MOREUDA", "MOREUDA", H1, opt_out),
            ("c7", "NA", "NA", "NA", "NA", H1, short_18),                 # incomplete 18 only
            ("c8", "NA", "NA", "NA", "NA", H1, answered_by_other),        # mixed hash
            ("c9", "UNKNOWN", "NA", "NA", "NA", H1, None),
            ("c10", "NA", "NA", "NA", "NA", H1, no_identity),
            ("c11", "NA", "NA", "NA", "NA", H2, None),                    # its own cohort
            ("legacy", "NA", "NA", "NA", "NA", H1, legacy),
        ]
        corrected = [{"schema": review_log.REVIEW_SCHEMA, "trial_id": "c1", "revision": 2,
                      "actual_word_id": "JOTA", "followed_guidance": "yes",
                      "followed_demonstration_form": "no", "note": None}]
        ids, states, meta = build(run, specs, corrected)
        rep = review_log.report(run, ids, states, trial_meta=meta)
        groups = rep["cohorts"]["groups"]
        una = rep["cohorts"]["unassignable"]

        ok(len(groups) == 4, "four cohorts: two scopes x two checkpoints")
        g1c, g1p = tagged(groups, "common5", H1), tagged(groups, "project18", H1)
        g2c, g2p = tagged(groups, "common5", H2), tagged(groups, "project18", H2)
        ok(all(g is not None for g in (g1c, g1p, g2c, g2p)), "every cohort is present")

        # the corrected confirmation is the truth, and it moves c1 from NA to JOTA
        r = next(x for x in g1c["rows"] if x["trial_id"] == "c1")
        ok(r["actual_word_id"] == "JOTA", "the latest revision is the confirmed word")
        ok(r["candidate"]["hit"] is False and r["candidate"]["pick"] == "NA",
           "a corrected confirmation turns a former match into a miss")

        # denominators: nothing is deleted for failing
        ok(sorted(x["trial_id"] for x in g1c["rows"] if x["in_candidate_set"]) ==
           ["c1", "c3", "c4", "c7", "c8"],
           "common5 scores exactly the five supported attempts, failures included")
        ok(sorted(x["trial_id"] for x in g1p["rows"] if x["in_candidate_set"]) ==
           ["c1", "c2", "c3", "c4", "c7", "c8"],
           "project18 scores the same attempts plus the word outside the five")
        ok(g1c["n_supported"] == 5 and g1p["n_supported"] == 6,
           "a failed, missing or mismatched attempt is never removed from the denominator")

        # each defect leaves its own reason, on its own method
        def why(g, tid, m):
            return next(x for x in g["rows"] if x["trial_id"] == tid)[m]["reason"]

        ok(why(g1c, "c3", "dtw") == "OBSERVATION_INSUFFICIENT_OBSERVATION"
           and why(g1c, "c3", "candidate") == "CANDIDATE_INFERENCE_FAILED"
           and why(g1c, "c3", "base_neural") is None,
           "a failed observation and a failed candidate do not remove the base answer")
        ok(all(why(g1c, "c4", m) == "NO_RESULT_FILE" for m in g1c["methods"]),
           "a missing result is unanswered everywhere, and still counted")
        ok(why(g1p, "c7", "candidate") == "CANDIDATE_SET_INCOMPLETE"
           and why(g1p, "c7", "dtw") is None,
           "an incomplete 18-word set silences the candidate, not the attempt")
        ok(why(g1c, "c7", "candidate") is None,
           "the same trial's intact five-word answer is unaffected")
        ok(why(g1c, "c8", "candidate") == "CANDIDATE_IDENTITY_MISMATCH"
           and why(g1c, "c8", "dtw") is None and why(g1c, "c8", "base_neural") is None,
           "a different checkpoint's answer is blocked; the other methods stay")

        # counted by hand for common5: c1 says NA against a corrected JOTA (miss), c3 failed,
        # c4 has no result, c7 hits, c8 is blocked by the identity mismatch.
        ok(g1c["per_method"]["candidate"]["hits"] == 1
           and g1c["per_method"]["candidate"]["answered"] == 2,
           "common5 candidate: 1 hit, 2 answered, 5 supported")
        ok(g1c["per_method"]["dtw"]["hits"] == 2 and g1c["per_method"]["base_neural"]["hits"] == 3,
           "and the other two methods are counted on the same five attempts")
        ok(g1c["per_method"]["candidate"]["unanswered"] ==
           {"CANDIDATE_INFERENCE_FAILED": 1, "NO_RESULT_FILE": 1, "CANDIDATE_IDENTITY_MISMATCH": 1},
           "the unanswered reasons are itemised, not pooled")
        ok(g1c["per_method"]["dtw"]["n"] == 5 and g1c["per_method"]["candidate"]["n"] == 5,
           "every method shares the all-supported denominator")

        # never retrospective, never invented
        every_tid = {x["trial_id"] for g in groups.values() for x in g["rows"]}
        ok("c6" not in every_tid, "an explicit opt-out never gains a candidate reading")
        ok("legacy" not in every_tid, "a pre-candidate record never gains one either")
        ok("c5" not in every_tid, "an unconfirmed attempt is not in any cohort")
        ok(not any(x["trial_id"] in ("c6", "legacy", "c5") for x in una),
           "and none of those is reported as unassignable; they simply do not apply")
        ok(sorted({x["trial_id"] for x in una}) == ["c10"],
           "only the opt-in attempt with no fixed identity is unassignable")
        ok({x["reason"] for x in una} == {"NO_STORED_MODEL_IDENTITY"},
           "its reason is said, and no identity is borrowed from the registry")

        # an attempt whose word is not in this set is kept visible but never scored
        c2rows = [x for x in g1c["rows"] if x["trial_id"] == "c2"]
        ok(len(c2rows) == 1 and c2rows[0]["in_candidate_set"] is False
           and c2rows[0]["excluded_reason"] == "ACTUAL_WORD_OUTSIDE_THIS_CANDIDATE_SET",
           "INSA is outside the five and is shown as such, not silently dropped")
        ok(any(x["trial_id"] == "c2" and x["in_candidate_set"] for x in g1p["rows"]),
           "the same INSA attempt IS scored in the eighteen")

        # identities are separate tasks
        ok(g1c["identity"]["checkpoint_sha256"] != g2c["identity"]["checkpoint_sha256"],
           "two checkpoints are two cohorts")
        ok(g2c["n_supported"] == 1 and g1c["n_supported"] == 5,
           "and their counts are never added together")
        ok(g1c["identity"]["candidate_count"] == 5 and g1p["identity"]["candidate_count"] == 18,
           "each cohort carries its own candidate count")
        ok(g1c["identity"]["candidate_set_sha256"] == g2c["identity"]["candidate_set_sha256"],
           "the same five words hash the same whatever the model")
        ok(g1c["identity"]["candidate_set_sha256"] != g1p["identity"]["candidate_set_sha256"],
           "five and eighteen do not")
        ok(g1p["candidate_set"] == sorted(g1p["candidate_set"]) and len(g1p["candidate_set"]) == 18,
           "the eighteen come from the declared project words, in canonical order")

        # the legacy five-word block is untouched
        ok(rep["per_method"]["denominator"] == rep["counts"]["confirmed_supported"],
           "the original five-word block still reports its own denominator")
        ok(rep["counts"]["confirmed_outside_common5"] >= 1 and
           any(e["reason"] == "OUTSIDE_COMMON_5" for e in rep["excluded"]),
           "and still excludes words outside the five, exactly as before")
        ok(set(rep["per_word"]) == set(COMMON), "the legacy per-word block is still the five")

        # timing, with its scope said out loud
        t = g1c["timing"]
        ok(t["scopes_seen"].get("server_end_candidate_to_assembled_result_v2") == t["n"]
           and t["n"] > 0, "stored timing is attached with its scope id")
        ok("motion" in t["is_not"] or "hands actually stopped" in t["is_not"],
           "and states it is not a motion-end latency")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------- 3. read back later, with no live session
def stray_block(data):
    """An explicit refusal, with a candidate block left lying beside it in the stored result."""
    data["research_model"] = {"opted_in": False, "model_key": None}
    return True


def candidate_crashed(data):
    """Opted in, and the candidate wrote nothing at all - the identity is still on disk."""
    data.pop("research_comparison")
    return True


def test_without_live_metadata():
    """trial_meta is empty here on purpose: an old session's report has no in-memory trial left."""
    tmp = tempfile.mkdtemp(prefix="f19c_")
    try:
        run = os.path.join(tmp, "run")
        specs = [("p1", "MOREUDA", "MOREUDA", "MOREUDA", "MOREUDA", H1, stray_block),
                 ("p2", "NA", "NA", "NA", "NA", H1, candidate_crashed)]
        ids, states, _ = build(run, specs)
        rep = review_log.report(run, ids, states)          # no trial_meta at all
        groups = rep["cohorts"]["groups"]
        seen = {r["trial_id"] for g in groups.values() for r in g["rows"]}

        ok("p1" not in seen, "a stored explicit opt-out is still an opt-out with no live metadata")
        ok(not any(u["trial_id"] == "p1" for u in rep["cohorts"]["unassignable"]),
           "and it is a refusal, not something that failed to be assigned")

        ok(len(groups) == 2 and all(g["n_supported"] == 1 for g in groups.values()),
           "a stored opt-in identity still forms its two cohorts without live metadata")
        ok(all(g["per_method"]["candidate"]["answered"] == 0 for g in groups.values()),
           "its crashed candidate answered nothing")
        ok(all(g["per_method"]["candidate"]["unanswered"] == {"CANDIDATE_ABSENT": 1}
               for g in groups.values()),
           "and stands as unanswered with its reason, instead of vanishing")
        ok(all(g["per_method"]["dtw"]["hits"] == 1 for g in groups.values()),
           "the same attempt's DTW answer is still counted")
        ok({g["identity"]["identity_source"][0] for g in groups.values()} ==
           {"result_research_model"},
           "the evidence used is the trial's own saved choice, not the live registry")
        ok(sorted(g["identity"]["candidate_count"] for g in groups.values()) == [5, 18],
           "both fixed candidate sets are still recovered from the saved identity")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 4. the hash is the membership
def test_hash_is_membership():
    a = scope(COMMON, "NA", "common5")
    b = scope(COMMON, "JOTA", "common5")
    ok([r["word_id"] for r in a["order"]] != [r["word_id"] for r in b["order"]],
       "two answers rank the same five words differently")
    ok(review_log._set_sha([r["word_id"] for r in a["order"]]) ==
       review_log._set_sha([r["word_id"] for r in b["order"]]),
       "but the membership hash is the same; it is not the score order")
    ok(review_log._set_sha(COMMON) != review_log._set_sha(PROJECT),
       "a different membership is a different hash")


if __name__ == "__main__":
    test_store()
    test_cohorts()
    test_without_live_metadata()
    test_hash_is_membership()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
