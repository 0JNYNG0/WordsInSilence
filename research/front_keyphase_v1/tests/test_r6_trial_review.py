"""R6: the participant's own confirmation, stored append-only and joined into a fair comparison.

Every number in this file comes from a SYNTHETIC fixture written by the test. Nothing here is a
recognition result, an accuracy figure, or a judgement about any real performance. No personal
recording is read and no camera is opened.

Run: py -3.13 tests/test_r6_trial_review.py
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.service import review_log as R  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic, written by this test; not a recognition result"}
NL = chr(10)


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make_result(run_dir, tid, requested, dtw_best, neural_top1, observation="OK",
                neural_status="READY"):
    """A minimal stored result: DTW ranking over the common five, plus a learned candidate."""
    order = [dtw_best] + [w for w in R.COMMON_5 if w != dtw_best]
    ranking = [{"word_id": w, "status": "OK", "distance": round(0.1 + 0.1 * i, 4), "rank": i + 1}
               for i, w in enumerate(order)]
    lr = {"model_status": neural_status, "is_a_verdict": False}
    if neural_status == "READY":
        lr["ranking"] = {"restricted_to_project_words": {"top1": neural_top1},
                         "over_all_trained_classes": {"top1": neural_top1}}
    else:
        lr["ranking"] = None
    res = {"result_id": "res_" + tid, "observation_status": observation,
           "requested_word_id": requested, "ranking": ranking, "learned_recognition": lr,
           "accepted": None, "execution_assessment": "UNCONFIGURED"}
    d = os.path.join(run_dir, "trials")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, tid + ".result.json"), "w", encoding="utf-8", newline=NL) as fh:
        json.dump(res, fh, ensure_ascii=False)
    return tid


@check("a confirmation is stored, a resend is ignored, a correction becomes a new revision")
def _store_dedupe_revise():
    tmp = tempfile.mkdtemp()
    try:
        a = R.submit(tmp, "s1", "t1", "ALDA", "yes", "메모")
        assert a["status"] == "STORED" and a["revision"] == 1, a
        b = R.submit(tmp, "s1", "t1", "ALDA", "yes", "메모")
        assert b["status"] == "DUPLICATE_IGNORED" and b["revision"] == 1, b
        c = R.submit(tmp, "s1", "t1", "NA", "no", None)
        assert c["status"] == "STORED" and c["revision"] == 2, c
        rows, err = R.read_all(tmp)
        assert err is None
        assert len(rows) == 2, "a duplicate reached the file"
        latest = R.latest_by_trial(tmp)[0]["t1"]
        assert latest["actual_word_id"] == "NA" and latest["revision"] == 2
        assert latest["is_expert_verification"] is False
        MEASURED["store"] = {"rows_on_disk": len(rows), "latest_revision": latest["revision"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("an invalid answer is rejected and nothing is written")
def _rejects():
    tmp = tempfile.mkdtemp()
    try:
        bad = R.submit(tmp, "s1", "t1", "ALDA", "maybe")
        assert bad["status"] == "REJECTED", bad
        unknown_word = R.submit(tmp, "s1", "t1", "NOT_A_WORD", "yes", known_words={"ALDA"})
        assert unknown_word["status"] == "REJECTED", unknown_word
        assert R.read_all(tmp) == ([], None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a write that fails is reported as not stored")
def _write_failure():
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, R.REVIEW_FILE), exist_ok=True)   # a directory blocks it
        out = R.submit(tmp, "s1", "t1", "ALDA", "yes")
        assert out["status"] == "NOT_STORED" and out["stored"] is False, out
        assert out["reason"], "no reason given"
        assert R.latest_by_trial(tmp)[0] == {}
        MEASURED["write_failure"] = {"status": out["status"], "reason_class": out["reason"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("before any confirmation the report has no score, not a zero")
def _no_data_is_not_zero():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA")
        rep = R.report(tmp)
        assert rep["counts"]["requested_total"] == 1
        assert rep["counts"]["unconfirmed"] == 1
        assert rep["paired"]["n"] == 0
        assert rep["paired"]["dtw"]["value"] is None
        assert rep["paired"]["dtw"]["display"] == "기록 없음"
        assert rep["not_configured"]["execution_quality"] == "NOT_CONFIGURED"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("the requested word and the confirmed word can differ, and the confirmed one is used")
def _requested_vs_actual():
    tmp = tempfile.mkdtemp()
    try:
        # asked for ALDA, actually signed NA; DTW says NA, the model says ALDA
        make_result(tmp, "t1", "ALDA", "NA", "ALDA")
        R.submit(tmp, "s1", "t1", "NA", "no")
        rep = R.report(tmp)
        row = rep["rows"][0]
        assert row["requested_word_id"] == "ALDA" and row["actual_word_id"] == "NA"
        assert row["dtw_match"] is True and row["neural_match"] is False
        assert rep["paired"]["agreement"] == {"both_match": 0, "dtw_only": 1,
                                          "neural_only": 0, "neither": 0}
        MEASURED["requested_vs_actual"] = row
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("hand-checked denominators, confusion cells and paired counts on a fixed fixture")
def _aggregate_arithmetic():
    tmp = tempfile.mkdtemp()
    try:
        # id, requested, dtw pick, neural pick, confirmed actual
        plan = [("t1", "ALDA", "ALDA", "ALDA", "ALDA"),       # both right
                ("t2", "NA", "NA", "ALDA", "NA"),            # dtw only
                ("t3", "JOTA", "ALDA", "JOTA", "JOTA"),      # neural only
                ("t4", "NA", "ALDA", "ALDA", "NA"),          # neither
                ("t5", "ALDA", "ALDA", "ALDA", "ALDA")]      # both right
        for tid, req, dtw, nn, actual in plan:
            make_result(tmp, tid, req, dtw, nn)
            R.submit(tmp, "s1", tid, actual, "yes")
        make_result(tmp, "t6", "ALDA", "ALDA", "ALDA")        # never confirmed
        make_result(tmp, "t7", "ALDA", "ALDA", "ALDA", observation="INSUFFICIENT_OBSERVATION")
        R.submit(tmp, "s1", "t7", "ALDA", "unsure")
        make_result(tmp, "t8", "ALDA", "ALDA", None, neural_status="CHECKPOINT_MISSING")
        R.submit(tmp, "s1", "t8", "ALDA", "yes")
        make_result(tmp, "t9", "INSA", "ALDA", "ALDA")
        R.submit(tmp, "s1", "t9", "INSA", "yes", known_words=None)   # outside the common five
        make_result(tmp, "t10", "ALDA", "ALDA", "ALDA")
        R.submit(tmp, "s1", "t10", R.UNKNOWN, "unsure")

        rep = R.report(tmp)
        c = rep["counts"]
        assert c["requested_total"] == 10, c
        assert c["paired_valid"] == 5, c
        assert c["confirmed_total"] == 9, c
        assert c["confirmed_supported"] == 7, c
        assert c["unconfirmed"] == 1 and c["confirmed_unknown"] == 1, c
        assert c["confirmed_outside_common5"] == 1, c
        assert c["observation_insufficient"] == 1, c
        # the unavailable model is now a per-method reason, not a shared exclusion
        assert rep["per_method"]["neural"]["unanswered"].get("MODEL_CHECKPOINT_MISSING") == 1,             rep["per_method"]["neural"]

        assert rep["paired"]["agreement"] == {"both_match": 2, "dtw_only": 1,
                                              "neural_only": 1, "neither": 1}, rep["paired"]
        assert rep["paired"]["dtw"]["display"] == "3/5"
        assert rep["paired"]["neural"]["display"] == "3/5"
        # per-method denominator: every supported confirmation, 5 paired + the
        # insufficient observation + the missing model = 7
        assert rep["per_method"]["denominator"] == 7, rep["per_method"]["denominator"]
        assert rep["per_method"]["dtw"]["match_over_all_supported"]["display"] == "4/7", \
            rep["per_method"]["dtw"]
        # t7 has an insufficient observation for DTW but the neural adapter still answered, so
        # the neural tally keeps it
        assert rep["per_method"]["neural"]["match_over_all_supported"]["display"] == "4/7", \
            rep["per_method"]["neural"]

        a = rep["per_word"]["ALDA"]
        assert (a["confirmed_supported_n"], a["paired_n"]) == (4, 2), a
        assert (a["dtw_correct"], a["neural_correct"]) == (3, 3), a
        na = rep["per_word"]["NA"]
        assert (na["confirmed_supported_n"], na["paired_n"]) == (2, 2), na
        assert (na["dtw_correct"], na["neural_correct"]) == (1, 0), na
        jo = rep["per_word"]["JOTA"]
        assert (jo["confirmed_supported_n"], jo["paired_n"]) == (1, 1), jo
        assert (jo["dtw_correct"], jo["neural_correct"]) == (0, 1), jo

        dm = rep["confusion_matrix"]["dtw"]
        assert dm["NA"]["NA"] == 1 and dm["NA"]["ALDA"] == 1, dm["NA"]
        assert dm["JOTA"]["ALDA"] == 1 and dm["JOTA"]["JOTA"] == 0, dm["JOTA"]
        nm = rep["confusion_matrix"]["neural"]
        assert nm["NA"]["ALDA"] == 2, nm["NA"]
        assert nm["JOTA"]["JOTA"] == 1, nm["JOTA"]

        reasons = sorted({e["reason"] for e in rep["excluded"]})
        assert reasons == ["CONFIRMED_UNKNOWN", "NOT_CONFIRMED", "NOT_PAIRED",
                           "OUTSIDE_COMMON_5"], reasons
        MEASURED["aggregate"] = {
            "counts": c, "agreement": rep["paired"]["agreement"],
            "paired_dtw": rep["paired"]["dtw"]["display"],
            "per_method_dtw": rep["per_method"]["dtw"]["match_over_all_supported"]["display"],
            "per_method_neural":
                rep["per_method"]["neural"]["match_over_all_supported"]["display"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a correction changes the aggregate; the earlier revision stays on disk")
def _revision_wins():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA")
        R.submit(tmp, "s1", "t1", "NA", "no")
        assert R.report(tmp)["paired"]["dtw"]["display"] == "0/1"
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        assert R.report(tmp)["paired"]["dtw"]["display"] == "1/1"
        assert len(R.read_all(tmp)[0]) == 2, "the earlier revision was overwritten"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("the confirmation never reaches a recogniser and is labelled self-report")
def _never_an_input():
    for mod in ("neural.py", "recognize.py", "dtw.py"):
        path = (os.path.join(ROOT, "src", "wis_lab", "service", mod) if mod == "neural.py"
                else os.path.join(ROOT, "src", "wis_lab", "core", mod))
        src = open(path, encoding="utf-8").read()
        assert "actual_word_id" not in src, f"{mod} reads the confirmation"
        assert "trial_review" not in src, f"{mod} reads the confirmation"
    tmp = tempfile.mkdtemp()
    try:
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        row = R.read_all(tmp)[0][0]
        assert row["source"] == "participant_self_report"
        assert row["is_expert_verification"] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a confirmed attempt with no result file stays in its own denominator")
def _confirmed_without_result():
    tmp = tempfile.mkdtemp()
    try:
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        rep = R.report(tmp, ["t1"])
        assert rep["per_method"]["denominator"] == 1, rep["per_method"]["denominator"]
        assert rep["counts"]["confirmed_supported"] == 1
        assert rep["counts"]["no_result_file"] == 1
        assert rep["per_method"]["dtw"]["unanswered"] == {"NO_RESULT_FILE": 1}
        assert rep["per_method"]["dtw"]["match_over_all_supported"]["display"] == "0/1"
        assert rep["paired"]["n"] == 0
        MEASURED["confirmed_without_result"] = rep["per_method"]["dtw"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("one method being unavailable does not erase the other method's answer")
def _one_method_down():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", None, neural_status="CHECKPOINT_MISSING")
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        rep = R.report(tmp, ["t1"])
        pm = rep["per_method"]
        assert pm["dtw"]["match_over_all_supported"]["display"] == "1/1", pm["dtw"]
        assert pm["neural"]["match_over_all_supported"]["display"] == "0/1", pm["neural"]
        assert pm["neural"]["unanswered"] == {"MODEL_CHECKPOINT_MISSING": 1}
        assert rep["paired"]["n"] == 0, "an unpaired row must not enter the paired figure"
        MEASURED["dtw_ok_neural_down"] = {"dtw": pm["dtw"]["match_over_all_supported"]["display"],
                                          "neural": pm["neural"]["match_over_all_supported"][
                                              "display"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("the reverse direction too: the neural answer survives a DTW gap")
def _other_method_down():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA")
        path = os.path.join(tmp, "trials", "t1.result.json")
        d = json.load(open(path, encoding="utf-8"))
        d["ranking"] = [{"word_id": "INSA", "status": "OK", "distance": 0.1, "rank": 1}]
        json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        pm = R.report(tmp, ["t1"])["per_method"]
        assert pm["neural"]["match_over_all_supported"]["display"] == "1/1", pm["neural"]
        assert pm["dtw"]["match_over_all_supported"]["display"] == "0/1", pm["dtw"]
        assert pm["dtw"]["unanswered"] == {"NO_COMMON5_CANDIDATE": 1}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("an unreadable store is a state, not an empty record, and blocks a new revision")
def _store_unreadable():
    tmp = tempfile.mkdtemp()
    try:
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        os.rename(os.path.join(tmp, R.REVIEW_FILE), os.path.join(tmp, "moved.jsonl"))
        os.makedirs(os.path.join(tmp, R.REVIEW_FILE), exist_ok=True)
        rows, err = R.read_all(tmp)
        assert rows == [] and err and err.startswith("STORE_UNREADABLE"), (rows, err)
        rep = R.report(tmp, ["t1"])
        assert rep["store_status"] == "UNREADABLE", rep["store_status"]
        assert rep["store_error"], "no error reported"
        assert rep["per_method"] is None and rep["paired"] is None, "a score was produced anyway"
        assert rep["counts"].get("unconfirmed") is None, "it looked like no confirmations"
        out = R.submit(tmp, "s1", "t1", "NA", "no")
        assert out["status"] == "NOT_STORED" and out["revision"] is None, out
        assert out["reason"].startswith("STORE_UNREADABLE"), out["reason"]
        MEASURED["store_unreadable"] = {"report": rep["store_status"], "submit": out["status"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("per-word counts separate confirmations from paired rows")
def _per_word_names():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA")
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        R.submit(tmp, "s1", "t2", "ALDA", "yes")          # confirmed, never resulted
        pw = R.report(tmp, ["t1", "t2"])["per_word"]["ALDA"]
        assert pw["confirmed_supported_n"] == 2, pw
        assert pw["paired_n"] == 1, pw
        assert pw["dtw_answered"] == 1 and pw["dtw_correct"] == 1, pw
        assert "confirmed_n" not in pw, "the ambiguous name is still there"
        MEASURED["per_word_names"] = pw
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("the report counts every trial the session asked for, not only the ones with a result")
def _all_trials_counted():
    tmp = tempfile.mkdtemp()
    try:
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA")
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        rep = R.report(tmp, ["t1", "t2", "t3"],
                       {"t1": "FEEDBACK", "t2": "CANCELLED", "t3": "ERROR"})
        assert rep["counts"]["requested_total"] == 3, rep["counts"]
        assert rep["counts"]["trial_states"] == {"FEEDBACK": 1, "CANCELLED": 1, "ERROR": 1}
        assert rep["counts"]["unconfirmed"] == 2, rep["counts"]
        MEASURED["all_trials"] = rep["counts"]["trial_states"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("an observation the DTW path called insufficient does not delete the neural answer")
def _observation_is_dtw_specific():
    tmp = tempfile.mkdtemp()
    try:
        # the top-level observation_status belongs to DTW predict; the adapter answered anyway
        make_result(tmp, "t1", "ALDA", "ALDA", "ALDA",
                    observation="INSUFFICIENT_OBSERVATION", neural_status="READY")
        R.submit(tmp, "s1", "t1", "ALDA", "yes")
        rep = R.report(tmp, ["t1"])
        pm = rep["per_method"]
        assert pm["denominator"] == 1, pm["denominator"]
        assert pm["dtw"]["answered"] == 0, pm["dtw"]
        assert pm["dtw"]["unanswered"] == {"OBSERVATION_INSUFFICIENT_OBSERVATION": 1}, pm["dtw"]
        assert pm["neural"]["answered"] == 1, pm["neural"]
        assert pm["neural"]["match_over_all_supported"]["display"] == "1/1", pm["neural"]
        assert rep["paired"]["n"] == 0, "an unpaired row entered the paired figure"
        assert rep["counts"]["observation_insufficient"] == 1
        assert rep["counts"]["no_result_file"] == 0
        MEASURED["observation_is_dtw_specific"] = {
            "dtw": pm["dtw"]["unanswered"],
            "neural": pm["neural"]["match_over_all_supported"]["display"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("a missing or unreadable result blocks both methods, as a shared deficit")
def _shared_deficit():
    tmp = tempfile.mkdtemp()
    try:
        R.submit(tmp, "s1", "t1", "ALDA", "yes")                     # no result file at all
        os.makedirs(os.path.join(tmp, "trials"), exist_ok=True)
        with open(os.path.join(tmp, "trials", "t2.result.json"), "w", encoding="utf-8") as fh:
            fh.write("{ not json")
        R.submit(tmp, "s1", "t2", "NA", "yes")
        rep = R.report(tmp, ["t1", "t2"])
        pm = rep["per_method"]
        assert pm["denominator"] == 2, pm["denominator"]
        for m in ("dtw", "neural"):
            assert pm[m]["answered"] == 0, pm[m]
            assert set(pm[m]["unanswered"]) == {"NO_RESULT_FILE",
                                                "RESULT_UNREADABLE:JSONDecodeError"}, pm[m]
        assert rep["counts"]["no_result_file"] == 1
        assert rep["counts"]["result_unreadable"] == 1
        MEASURED["shared_deficit"] = pm["dtw"]["unanswered"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
