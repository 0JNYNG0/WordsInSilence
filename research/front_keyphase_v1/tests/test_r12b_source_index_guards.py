"""R12b: three guards on the source-video index tool.

Each check first reproduces the defect against the OLD rule, then requires the new behaviour:

  1. --out pointing at an input file would have overwritten the original, and an existing report
     was replaced without asking;
  2. AI Hub's multi-part names (.zip.part1, .zip.part01) were not seen as incomplete because the
     test was endswith('.part');
  3. with a corrupt or unopened input, a target read as "missing", which cannot be told apart from
     "confirmed absent in everything given".

Small synthetic ZIPs only. No model, no inference, no repeat of another suite.

Run: py -3.13 tests/test_r12b_source_index_guards.py
"""
import json
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import tools_source_video_index as T  # noqa: E402

CHECKS = []
MEASURED = {"fixture": "synthetic ZIPs in a temporary directory"}
NL = chr(10)
A = T.TARGETS[0]
TMP = ""


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for m in members:
            z.writestr(m, b"directory entry only")


_SEQ = [0]


def run(argv, out=None):
    # a fresh destination per call: the overwrite guard is under test, not in the way of it
    if out is None:
        _SEQ[0] += 1
        out = os.path.join(TMP, f"report_{_SEQ[0]}.json")
    sys.argv = ["tool"] + argv + ["--out", out]
    code = T.main()
    if code != 0 or not os.path.exists(out):
        return code, None                       # a refusal writes nothing; never parse the input
    with open(out, encoding="utf-8") as fh:
        return code, json.load(fh)


@check("--out is refused when it resolves to an input archive, and the input survives")
def _out_collides_with_zip():
    p = os.path.join(TMP, "bundle_guard.zip")
    make_zip(p, [f"REAL/{A}"])
    before = open(p, "rb").read()
    code, _ = run(["--zip", p], out=p)
    after = open(p, "rb").read()
    assert after == before, "the input archive was overwritten by the report"
    assert code != 0, "a colliding --out should not report success"
    with zipfile.ZipFile(p) as z:                     # still a valid archive
        assert any(n.endswith(A) for n in z.namelist())
    MEASURED["out_collision_zip"] = {"input_bytes_unchanged": True, "exit_code": code}


@check("--out is refused when it resolves to the list file or an explicit video")
def _out_collides_with_list():
    d = os.path.join(TMP, "vids2")
    os.makedirs(d, exist_ok=True)
    vid = os.path.join(d, A)
    with open(vid, "wb") as fh:
        fh.write(b"0123456789")
    lst = os.path.join(TMP, "files2.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write(vid + NL)
    for target in (lst, vid):
        before = open(target, "rb").read()
        code, _ = run(["--list", lst], out=target)
        assert open(target, "rb").read() == before, f"{target} was overwritten"
        assert code != 0
    # the same path written differently must still collide
    code, _ = run(["--list", lst], out=os.path.join(TMP, "sub", "..", "files2.txt"))
    assert code != 0, "a non-normalised spelling of the same path slipped through"


@check("an existing report is not replaced without --force")
def _no_silent_overwrite():
    p = os.path.join(TMP, "bundle_guard.zip")
    out = os.path.join(TMP, "existing_report.json")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write('{"keep":"me"}')
    code, _ = run(["--zip", p], out=out)
    assert code != 0, "an existing report was replaced silently"
    assert json.load(open(out, encoding="utf-8")) == {"keep": "me"}
    code, rep = run(["--zip", p, "--force"], out=out)
    assert code == 0 and rep is not None and rep["task"] == "source_video_index"
    MEASURED["overwrite_guard"] = {"refused_without_force": True, "allowed_with_force": True}


@check("AI Hub multi-part names count as incomplete, not as readable archives")
def _multipart_incomplete():
    names = ["m.zip.part", "m.zip.part1", "m.zip.part01", "m.zip.part12", "m.zip.crdownload"]
    paths = []
    for n in names:
        p = os.path.join(TMP, n)
        make_zip(p, [f"REAL/{A}"])      # a real zip under a partial name stays untouched
        paths.append(p)
    argv = []
    for p in paths:
        argv += ["--zip", p]
    _, rep = run(argv)
    got = {os.path.basename(a["path"]): a["status"] for a in rep["archives"]}
    assert all(v == "incomplete" for v in got.values()), got
    assert all("members_total" not in a for a in rep["archives"])
    MEASURED["incomplete_names"] = got


@check("a target is unknown, not missing, when an input could not be read")
def _unknown_vs_missing():
    good = os.path.join(TMP, "good.zip")
    make_zip(good, ["REAL/other.mp4"])
    bad = os.path.join(TMP, "corrupt.zip")
    with open(bad, "wb") as fh:
        fh.write(b"definitely not a zip")
    # everything readable and the target genuinely absent -> missing
    _, rep = run(["--zip", good])
    assert rep["per_target"][A]["status"] == "missing", rep["per_target"][A]
    assert rep["inputs_unread"] == []
    assert rep["per_target"][A]["completeness"] == "all_inputs_read"
    # one unreadable input -> the same absence can no longer be called missing
    _, rep2 = run(["--zip", good, "--zip", bad])
    t = rep2["per_target"][A]
    assert t["status"] == "unknown", t
    assert t["completeness"] == "some_inputs_unread"
    assert bad in [u["path"] for u in rep2["inputs_unread"]]
    assert rep2["status"] == "INCOMPLETE_INPUTS"
    # an incomplete download counts the same way
    part = os.path.join(TMP, "x.zip.part1")
    make_zip(part, ["REAL/other.mp4"])
    _, rep3 = run(["--zip", good, "--zip", part])
    assert rep3["per_target"][A]["status"] == "unknown"
    MEASURED["unknown_vs_missing"] = {"all_read": "missing", "one_unreadable": "unknown",
                                      "one_incomplete": "unknown"}


@check("a found target stays found even when another input was unreadable")
def _found_survives_unread():
    good = os.path.join(TMP, "has_target.zip")
    make_zip(good, [f"REAL/{A}"])
    bad = os.path.join(TMP, "corrupt.zip")
    _, rep = run(["--zip", good, "--zip", bad])
    t = rep["per_target"][A]
    assert t["status"] == "found", t
    assert t["completeness"] == "some_inputs_unread"


def main() -> int:
    global TMP
    TMP = tempfile.mkdtemp(prefix="wis_r12b_")
    failed = 0
    try:
        for name, fn in CHECKS:
            try:
                fn()
                print(f"PASS  {name}")
            except (AssertionError, Exception) as exc:      # noqa: B014 - a raise is a failure
                failed += 1
                print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
