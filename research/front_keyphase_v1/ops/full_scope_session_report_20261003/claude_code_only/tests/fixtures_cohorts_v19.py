"""Emit real report payloads for the screen check. Built by the real aggregator, never by hand."""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_r16_candidate_cohorts as F  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "src"))
from wis_lab.service import review_log  # noqa: E402


def _with_cohorts():
    tmp = tempfile.mkdtemp(prefix="f19ui_")
    try:
        run = os.path.join(tmp, "run")
        specs = [("c1", "NA", "NA", "GAMSAHAMNIDA", "NA", F.H1, None),
                 ("c2", "INSA", "INSA", "ALDA", "INSA", F.H1, None),
                 ("c3", "ALDA", "NA", "ALDA", "NA", F.H1, F.failed_model),
                 ("c4", "JOTA", "JOTA", "JOTA", "JOTA", F.H1, F.drop_result),
                 ("c7", "NA", "NA", "NA", "NA", F.H1, F.short_18),
                 ("c8", "NA", "NA", "NA", "NA", F.H1, F.answered_by_other),
                 ("c10", "NA", "NA", "NA", "NA", F.H1, F.no_identity),
                 ("c11", "NA", "NA", "NA", "NA", F.H2, None)]
        ids, states, meta = F.build(run, specs)
        return review_log.report(run, ids, states, trial_meta=meta)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _malformed():
    tmp = tempfile.mkdtemp(prefix="f19ui2_")
    try:
        run = os.path.join(tmp, "run")
        os.makedirs(run)
        with open(os.path.join(run, review_log.REVIEW_FILE), "w", encoding="utf-8",
                  newline="\n") as fh:
            fh.write('{"trial_id": "t1", "revision": 1, "actual_word_id": "NA"}\n')
            fh.write('{"trial_id": "t2", "revi')
        return review_log.report(run, ["t1", "t2"], {"t1": "FEEDBACK"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _empty():
    tmp = tempfile.mkdtemp(prefix="f19ui3_")
    try:
        run = os.path.join(tmp, "run")
        os.makedirs(os.path.join(run, "trials"))
        return review_log.report(run, [], {})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    json.dump({"cohorts": _with_cohorts(), "malformed": _malformed(), "empty": _empty()},
              sys.stdout, ensure_ascii=False)
