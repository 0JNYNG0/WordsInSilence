"""R1b self-check: HTTP and the CLI must reach the same decision on the same stored input.

Uses fastapi.testclient, so no port is bound and no camera is opened.
Run: py -3.13 tests/test_r1b.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from fastapi.testclient import TestClient  # noqa: E402

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402
from wis_lab.core.recognize import predict  # noqa: E402
from wis_lab.service.app import API, app  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
REFS, LINEAGE = load_reference_set()
DISTANCE_TOLERANCE = 1e-9   # same process, same core: the numbers must be identical

client = TestClient(app)
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def cli_result(word_id):
    clip = next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)
    obs = Observation.load(os.path.join(ROOT, clip["observation_npz"]))
    return predict(obs, REFS, CONTRACT)


@check("health reports the honest states")
def _health():
    h = client.get(API + "/health").json()
    assert h["status"] == "OK"
    assert h["execution_assessment"] == "UNCONFIGURED"
    assert h["ml_hybrid"] == "NOT_READY"
    assert h["camera"] is None, "health must not open a camera"


@check("references lists the prepared count and names every exclusion")
def _references():
    r = client.get(API + "/references").json()
    assert r["prepared_count"] == 18
    assert r["total_clip_count"] == 19
    assert len(r["items"]) == 18
    assert r["excluded"] and r["excluded"][0]["word_id"] == "KKEUDEOGIDA"
    for it in r["items"]:
        assert it["readiness"] in ("READY", "NEEDS_REVIEW")
        assert it["readiness_reason"]


@check("HTTP replay matches the CLI decision and distances on every prepared word")
def _parity():
    for clip in MANIFEST["clips"]:
        if clip["readiness"] == "UNSUPPORTED_CUE":
            continue
        w = clip["word_id"]
        http = client.post(API + "/replays", json={"observation_id": w}).json()
        cli = cli_result(w)
        assert http["observation_status"] == cli["observation_status"], w
        assert http["top1_word_id"] == cli["top1_word_id"], w
        assert http["recognition"] == cli["recognition"], w
        assert http["execution_assessment"] == cli["execution_assessment"], w
        assert http["accepted"] == cli["accepted"] is None, w
        hh = [(r["word_id"], r["distance"]) for r in http["ranking"]]
        cc = [(r["word_id"], r["distance"]) for r in cli["ranking"]]
        assert [a for a, _ in hh] == [a for a, _ in cc], (w, "ranking order differs")
        for (wa, da), (wb, db) in zip(hh, cc):
            assert wa == wb
            if da is None or db is None:
                assert da == db, (w, wa)
            else:
                assert abs(da - db) <= DISTANCE_TOLERANCE, (w, wa, da, db)


@check("replay takes registered ids only, never a filesystem path")
def _no_paths():
    bad = client.post(API + "/replays", json={
        "observation_id": "cache/2683eadbc5f1c588/NIKL_INSA_01.npz"})
    assert bad.status_code == 404, bad.status_code
    assert bad.json()["error"] == "NO_SUCH_OBSERVATION_ID"
    abs_path = client.post(API + "/replays", json={
        "observation_id": os.path.join(ROOT, "config", "contract.json")})
    assert abs_path.status_code == 404


@check("replay never counts as a human attempt and never opens the camera")
def _replay_not_human():
    r = client.post(API + "/replays", json={"observation_id": "INSA"}).json()
    assert r["counts_as_human_attempt"] is False
    assert r["origin"] == "replay"
    assert client.get(API + "/health").json()["camera"] is None


@check("a session without a camera still answers, and reports the camera failure honestly")
def _session_no_camera():
    r = client.post(API + "/sessions", json={"participant_id": "test", "open_camera": False}).json()
    sid = r["session_id"]
    assert r["camera"] is False
    got = client.get(API + f"/sessions/{sid}").json()
    assert got["state"] == "OPEN" and got["camera"] is False
    # a trial needs the camera; asking for one must be a clean 409, not a fake success
    t = client.post(API + f"/sessions/{sid}/trials",
                    json={"request_id": "r1", "requested_word_id": "INSA"})
    assert t.status_code == 409, t.status_code
    assert t.json()["error"] == "CAMERA_NOT_OWNED_BY_SESSION"
    assert client.post(API + f"/sessions/{sid}/close").json()["state"] == "CLOSED"


@check("closed session refuses new work instead of silently reopening")
def _closed_session():
    sid = client.post(API + "/sessions", json={"open_camera": False}).json()["session_id"]
    client.post(API + f"/sessions/{sid}/close")
    hb = client.post(API + f"/sessions/{sid}/heartbeat", json={})
    assert hb.status_code == 409 and hb.json()["error"] == "SESSION_CLOSED"
    t = client.post(API + f"/sessions/{sid}/trials", json={"request_id": "x"})
    assert t.status_code == 409


@check("preview refuses a session that does not own the camera")
def _preview_guard():
    sid = client.post(API + "/sessions", json={"open_camera": False}).json()["session_id"]
    p = client.get(API + f"/sessions/{sid}/preview.jpg")
    assert p.status_code == 409 and p.json()["error"] == "CAMERA_NOT_OWNED_BY_SESSION"
    client.post(API + f"/sessions/{sid}/close")


@check("unknown session and trial ids are 404, not a crash")
def _unknown():
    assert client.get(API + "/sessions/ses_nope").status_code == 404
    assert client.get(API + "/trials/tri_nope").status_code == 404
    assert client.post(API + "/trials/tri_nope/stop").status_code == 404
    assert client.post(API + "/trials/tri_nope/cancel").status_code == 404


@check("an unprepared word is not offered and cannot be replayed")
def _unsupported_word():
    items = client.get(API + "/references").json()["items"]
    assert "KKEUDEOGIDA" not in [i["word_id"] for i in items]
    r = client.post(API + "/replays", json={"observation_id": "KKEUDEOGIDA"}).json()
    # it still has an observation, but it must not be in the candidate set
    assert "KKEUDEOGIDA" not in [x["word_id"] for x in r["ranking"]]


@check("the static test page is served from the same origin")
def _web():
    page = client.get("/")
    assert page.status_code == 200, page.status_code
    js = client.get("/app.js")
    assert js.status_code == 200, js.status_code
    # the browser must never open a camera itself; match the call, not the word in a comment
    for text in (page.text, js.text):
        assert "navigator.mediaDevices" not in text, "the browser must not open a camera"
        assert "getUserMedia(" not in text, "the browser must not open a camera"
    assert client.get("/style.css").status_code == 200


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
