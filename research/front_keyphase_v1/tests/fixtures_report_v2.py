"""Build real wis_trial_review_report_v2 payloads for the browser-client check.

The fixtures come from the actual aggregator on a throwaway directory, not from a hand written
shape, so the screen is tested against what the server really returns. No session, no camera and
nothing under runs/ is touched.

Run: py -3.13 tests/fixtures_report_v2.py   (prints {populated, empty, unreadable} as JSON)
"""
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from wis_lab.service import review_log  # noqa: E402

COMMON = review_log.COMMON_5


def _result(word, nn_word, observation="OK", nn_status="READY"):
    return {
        "observation_status": observation,
        "ranking": [{"word_id": w, "status": "OK", "distance": 0.1 if w == word else 0.9}
                    for w in COMMON],
        "learned_recognition": {
            "model_status": nn_status,
            "ranking": None if nn_status != "READY" else {
                "restricted_to_project_words": {"top1": nn_word, "candidate_count": 5},
                "over_all_trained_classes": {"top1": nn_word, "candidate_count": 80,
                                             "top3": [{"label": nn_word}]}}},
    }


def build(run_dir):
    tdir = os.path.join(run_dir, "trials")
    os.makedirs(tdir, exist_ok=True)
    known = set(COMMON) | {"INSA"}
    plan = [
        # trial, confirmed word, dtw pick, neural pick, observation, neural status
        ("t1", "ALDA", "ALDA", "ALDA", "OK", "READY"),
        ("t2", "NA", "ALDA", "NA", "OK", "READY"),
        # DTW could not answer, the neural adapter could: the attempt stays in both denominators
        ("t3", "JOTA", "JOTA", "JOTA", "INSUFFICIENT_HAND_FRAMES", "READY"),
        # the model was unavailable, DTW answered
        ("t4", "MOREUDA", "MOREUDA", None, "OK", "CHECKPOINT_MISSING"),
    ]
    for tid, actual, dtw, nn, obs, nns in plan:
        with open(os.path.join(tdir, tid + ".result.json"), "w", encoding="utf-8") as fh:
            json.dump(_result(dtw, nn, obs, nns), fh)
        review_log.submit(run_dir, "ses_fixture", tid, actual, "yes", None, known_words=known)
    # confirmed but never produced a result file at all
    review_log.submit(run_dir, "ses_fixture", "t5", "GAMSAHAMNIDA", "no", None, known_words=known)
    # confirmed a word outside the shared five, and one the participant could not recall
    review_log.submit(run_dir, "ses_fixture", "t6", "INSA", "yes", None, known_words=known)
    review_log.submit(run_dir, "ses_fixture", "t7", review_log.UNKNOWN, "unsure", None,
                      known_words=known)
    trials = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"]   # t8 was never confirmed
    return review_log.report(run_dir, trials=trials, trial_states={"t8": "CANCELLED"})


def main():
    tmp = tempfile.mkdtemp(prefix="wis_report_fixture_")
    try:
        populated = build(os.path.join(tmp, "run"))
        empty = review_log.report(os.path.join(tmp, "empty"), trials=[])
        # an unreadable store: the confirmation log is a directory, so reading it raises
        bad = os.path.join(tmp, "bad")
        os.makedirs(os.path.join(bad, review_log.REVIEW_FILE), exist_ok=True)
        unreadable = review_log.report(bad, trials=["t1"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump({"populated": populated, "empty": empty, "unreadable": unreadable},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
