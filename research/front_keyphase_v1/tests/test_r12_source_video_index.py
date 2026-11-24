"""R12: the source-video index tool, against small synthetic ZIPs.

Only the things that could mislead a reader are checked: exact matching, the case and path
normalisation policy, a duplicate basename, a target that is absent, a corrupt archive, an
incomplete download marker, and the refusal to search anything when no path is given.

No training, no inference, no repeat of an earlier suite. Nothing is extracted.

Run: py -3.13 tests/test_r12_source_video_index.py
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
MEASURED = {"fixture": "synthetic ZIPs built in a temporary directory"}
NL = chr(10)
A, B, C = T.TARGETS


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def make_zip(path, members):
    with zipfile.ZipFile(path, "w") as z:
        for m in members:
            z.writestr(m, b"not a real video, only a directory entry")


@check("an exact basename in a nested path is found, with its size and CRC")
def _exact(tmp):
    p = os.path.join(tmp, "bundle_a.zip")
    make_zip(p, [f"REAL/SEN/{A}", "REAL/SEN/other.mp4"])
    rep = run(["--zip", p])
    t = rep["per_target"][A]
    assert t["status"] == "found", t
    hit = t["candidates"][0]
    assert hit["member"] == f"REAL/SEN/{A}"
    assert hit["uncompressed_bytes"] > 0 and len(hit["crc32"]) == 8
    assert hit["case_mismatch"] is False
    assert rep["per_target"][B]["status"] == "missing"
    MEASURED["exact"] = {"member": hit["member"], "crc32": hit["crc32"]}


@check("a case-only difference is still found and is flagged, not silently equated")
def _case(tmp):
    p = os.path.join(tmp, "bundle_case.zip")
    make_zip(p, ["real/sen/" + A.lower()])
    t = run(["--zip", p])["per_target"][A]
    assert t["status"] == "found"
    assert t["candidates"][0]["case_mismatch"] is True, t
    MEASURED["case"] = t["candidates"][0]["member"]


@check("two members with the same basename are ambiguous and neither is chosen")
def _ambiguous(tmp):
    p = os.path.join(tmp, "bundle_dup.zip")
    make_zip(p, [f"one/{A}", f"two/{A}"])
    t = run(["--zip", p])["per_target"][A]
    assert t["status"] == "ambiguous", t
    assert len(t["candidates"]) == 2
    assert "none was chosen" in t["note"]
    MEASURED["ambiguous"] = [c["member"] for c in t["candidates"]]


@check("the same basename in two different archives is ambiguous too")
def _ambiguous_across(tmp):
    p1 = os.path.join(tmp, "b1.zip")
    p2 = os.path.join(tmp, "b2.zip")
    make_zip(p1, [A])
    make_zip(p2, [f"nested/{A}"])
    t = run(["--zip", p1, "--zip", p2])["per_target"][A]
    assert t["status"] == "ambiguous", t
    assert {c["zip_path"] for c in t["candidates"]} == {p1, p2}


@check("an archive numbered 01 is not assumed to hold REAL01")
def _no_number_assumption(tmp):
    p = os.path.join(tmp, "01_real_sen_video.zip")
    make_zip(p, ["REAL/SEN/NIA_SL_SEN0042_REAL07_F.mp4"])
    rep = run(["--zip", p])
    assert all(rep["per_target"][t]["status"] == "missing" for t in T.TARGETS)
    assert rep["archives"][0]["status"] == "read"
    assert rep["status"] == "NOT_ALL_FOUND"


@check("a corrupt archive is reported unreadable, not treated as empty")
def _corrupt(tmp):
    p = os.path.join(tmp, "broken.zip")
    with open(p, "wb") as fh:
        fh.write(b"PK\\x03\\x04 this is not a zip central directory")
    a = run(["--zip", p])["archives"][0]
    assert a["status"] == "unreadable", a
    assert "reason" in a


@check("an incomplete download is named incomplete and is not opened")
def _incomplete(tmp):
    p = os.path.join(tmp, "bundle.zip.part")
    make_zip(p, [A])                       # even a valid zip under a .part name stays untouched
    a = run(["--zip", p])["archives"][0]
    assert a["status"] == "incomplete", a
    assert "members_total" not in a
    assert run(["--zip", p])["per_target"][A]["status"] == "missing"


@check("with no input path nothing is searched")
def _no_input():
    rep = run([])
    assert rep["status"] == "NO_INPUT_PATH_GIVEN"
    assert rep["archives"] == [] and rep["explicit_files"] == []
    assert "never scans a disk" in rep["note"]


@check("an explicitly listed video file is accepted without any archive")
def _explicit_file(tmp):
    d = os.path.join(tmp, "vids")
    os.makedirs(d, exist_ok=True)
    f = os.path.join(d, C)
    with open(f, "wb") as fh:
        fh.write(b"0123456789")
    lst = os.path.join(tmp, "files.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write("# explicit paths only\n" + f + "\n")
    t = run(["--list", lst])["per_target"][C]
    assert t["status"] == "found" and t["candidates"][0]["zip_path"] is None
    assert t["candidates"][0]["uncompressed_bytes"] == 10


def run(argv):
    out = os.path.join(RUN_TMP, "report.json")
    sys.argv = ["tool"] + argv + ["--out", out]
    T.main()
    with open(out, encoding="utf-8") as fh:
        return json.load(fh)


RUN_TMP = ""


def main() -> int:
    global RUN_TMP
    RUN_TMP = tempfile.mkdtemp(prefix="wis_r12_")
    failed = 0
    try:
        for name, fn in CHECKS:
            try:
                fn(RUN_TMP) if fn.__code__.co_argcount else fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    finally:
        shutil.rmtree(RUN_TMP, ignore_errors=True)
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
