"""functional_20: the report's own arithmetic, and which population each number belongs to.

Synthetic fixtures only. No model, no camera, no capture run, no real store.

Run: py -3.13 tests/test_r17_report_statistics.py
"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wis_lab.service import review_log  # noqa: E402
import test_r16_candidate_cohorts as F  # noqa: E402

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def timed(ms, scope_id="server_end_candidate_to_assembled_result_v2"):
    """One row shaped as timing_summary reads it. ms None means the attempt stored no timing."""
    if ms is None:
        return {"result": {"timing": {}}}
    return {"result": {"timing": {"server_end_candidate_to_result_ms": ms,
                                  "timing_scope_id": scope_id}}}


# ------------------------------------------------------------------ 1. median and mean
def test_median_and_mean():
    empty = review_log.timing_summary([])
    ok(empty["n"] == 0 and empty["median_ms"] is None and empty["mean_ms"] is None,
       "no timing at all is None, never zero")

    odd = review_log.timing_summary([timed(100), timed(800), timed(400)])
    ok(odd["median_ms"] == 400, "an odd count takes the single middle value (100/400/800 -> 400)")
    ok(odd["mean_ms"] == round(1300 / 3, 2), "and its mean is the arithmetic mean")

    even = review_log.timing_summary([timed(100), timed(200), timed(400), timed(800)])
    ok(even["median_ms"] == 300,
       "an even count averages the two middles: 100/200/400/800 -> 300, not 400")
    ok(even["mean_ms"] == 375, "and the mean of the same four is 375")
    ok(even["min_ms"] == 100 and even["max_ms"] == 800, "min and max are unchanged")

    two = review_log.timing_summary([timed(100), timed(200)])
    ok(two["median_ms"] == 150, "two values average to the midpoint")

    # order must not matter, and it used to matter less visibly than it should have
    shuffled = review_log.timing_summary([timed(800), timed(100), timed(400), timed(200)])
    ok(shuffled["median_ms"] == 300 and shuffled["mean_ms"] == 375,
       "the input order does not change either figure")

    mixed = review_log.timing_summary([timed(100), timed(None), timed(300), timed(None)])
    ok(mixed["n"] == 2 and mixed["missing_timing"] == 2,
       "an attempt that stored no timing is counted apart, not as a zero")
    ok(mixed["median_ms"] == 200 and mixed["mean_ms"] == 200,
       "and it does not enter the median or the mean")
    ok("DIFFERENT number from the recognition denominator" in mixed["population_note"],
       "the summary states that its n is not the recognition denominator")


# ------------------------------------------------------------------ 2. whose attempts are timed
def test_timing_population():
    tmp = tempfile.mkdtemp(prefix="f20a_")
    try:
        run = os.path.join(tmp, "run")

        def slow(data):
            """INSA is outside the five, and used to drag the five-word timing with it."""
            data["timing"]["server_end_candidate_to_result_ms"] = 9000.0
            return True

        def no_timing(data):
            data.pop("timing")
            return True

        specs = [("t1", "NA", "NA", "NA", "NA", F.H1, None),              # 250 ms, in both
                 ("t2", "JOTA", "JOTA", "JOTA", "JOTA", F.H1, None),      # 250 ms, in both
                 ("t3", "INSA", "INSA", "ALDA", "INSA", F.H1, slow),      # 9000 ms, 18 only
                 ("t4", "ALDA", "ALDA", "ALDA", "ALDA", F.H1, no_timing)]  # no timing, in both
        ids, states, meta = F.build(run, specs)
        rep = review_log.report(run, ids, states, trial_meta=meta)
        g = rep["cohorts"]["groups"]
        c5 = next(x for x in g.values() if x["scope"] == "common5")
        p18 = next(x for x in g.values() if x["scope"] == "project18")

        ok(c5["n_supported"] == 3 and p18["n_supported"] == 4,
           "the five-word cohort scores three attempts, the eighteen-word cohort four")
        ok(c5["timing"]["n"] == 2 and c5["timing"]["max_ms"] == 250.0,
           "the five-word timing counts only its own attempts; the 9000 ms INSA is not among them")
        ok(p18["timing"]["n"] == 3 and p18["timing"]["max_ms"] == 9000.0,
           "the eighteen-word timing does include it, because that cohort scored it")
        ok(c5["timing"]["median_ms"] == 250.0 and p18["timing"]["median_ms"] == 250.0,
           "each cohort's median is over its own population")
        ok(c5["timing"]["missing_timing"] == 1 and p18["timing"]["missing_timing"] == 1,
           "the attempt with no stored timing is counted as missing in both")
        ok(c5["timing"]["n"] != c5["n_supported"] and p18["timing"]["n"] != p18["n_supported"],
           "the timing n and the recognition denominator are different numbers, and both are kept")
        ok(any(r["trial_id"] == "t4" and r["in_candidate_set"] for r in c5["rows"]),
           "a missing timing never removes the attempt from the recognition denominator")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 3. the form question's scope
def test_form_scope():
    tmp = tempfile.mkdtemp(prefix="f20b_")
    try:
        run = os.path.join(tmp, "run")
        specs = [("f1", "NA", "NA", "NA", "NA", F.H1, None),
                 ("f2", "JOTA", "JOTA", "JOTA", "JOTA", F.H1, None),
                 ("f3", "INSA", "INSA", "ALDA", "INSA", F.H1, None),
                 ("f4", "ALDA", "ALDA", "ALDA", "ALDA", F.H1, None)]
        ids, states, meta = F.build(run, specs)
        # rewrite the confirmations: one of each answer, and one from before the field existed
        answers = {"f1": "yes", "f2": "other_form", "f3": "no", "f4": None}
        path = os.path.join(run, review_log.REVIEW_FILE)
        import json
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        for r in rows:
            r["followed_demonstration_form"] = answers[r["trial_id"]]
            if r["trial_id"] == "f4":
                del r["followed_demonstration_form"]      # an older record has no field at all
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        rep = review_log.report(run, ids, states, trial_meta=meta)
        top = rep["demonstration_form_followed"]
        ok(top["scope"] == "common5" and "five" in top["scope_meaning"],
           "the top-level block now says it counts the five-word population")
        ok(top["answered"] == 2 and top["yes"] == 1 and top["other_form"] == 1
           and top["not_answered"] == 1,
           "its legacy values are preserved: f3 is outside the five, f4 answered nothing")

        g = rep["cohorts"]["groups"]
        c5 = next(x for x in g.values() if x["scope"] == "common5")
        p18 = next(x for x in g.values() if x["scope"] == "project18")
        ok(c5["demonstration_form_followed"]["denominator_n"] == 3
           and p18["demonstration_form_followed"]["denominator_n"] == 4,
           "each cohort states its own denominator, and they differ")
        ok(c5["demonstration_form_followed"]["no"] == 0
           and p18["demonstration_form_followed"]["no"] == 1,
           "the 'no' belongs to the eighteen-word cohort only, because only it scored that word")
        ok(c5["demonstration_form_followed"]["not_answered"] == 1
           and c5["demonstration_form_followed"]["yes"] == 1,
           "the record from before the question existed stays not_answered, and is not a yes")
        ok(p18["demonstration_form_followed"]["answered"] == 3,
           "the eighteen-word cohort counts the three that answered")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_median_and_mean()
    test_timing_population()
    test_form_scope()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
