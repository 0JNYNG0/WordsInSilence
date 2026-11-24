"""R1c camera-path check. OPENS THE REAL WEBCAM briefly, then releases it - ONLY WHEN ALLOWED.

BLOCKED BY DEFAULT (2026-09-26): it touches a real device (opening the camera AND probing device
indices via /devices) only when the environment variable WIS_ALLOW_REAL_CAMERA is exactly "1".
Without it, cv2.VideoCapture is replaced before anything runs and main() returns without a check.

It records a short take through the live path and asserts the plumbing, not the user's signing:
the observation reaches the core in the right shape, a preview exists, the camera survives two
trials without a restart, cancel produces no result, and close releases the device.

Run (real camera, explicit): set WIS_ALLOW_REAL_CAMERA=1 && py -3.13 tests/test_r1c_camera.py
Default run: prints BLOCKED and touches no device.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

REAL_CAMERA_ALLOWED = os.environ.get("WIS_ALLOW_REAL_CAMERA") == "1"
DEVICE_CALLS_BLOCKED = []


class _NoRealDevice:
    """Installed unless explicitly allowed: any attempt to open or probe a device fails loudly."""
    def __init__(self, *args, **kwargs):
        DEVICE_CALLS_BLOCKED.append(args)
        raise RuntimeError("real camera access blocked: set WIS_ALLOW_REAL_CAMERA=1 to allow")


if not REAL_CAMERA_ALLOWED:
    import cv2 as _cv2
    _cv2.VideoCapture = _NoRealDevice          # covers CameraWorker._run and list_devices

from fastapi.testclient import TestClient  # noqa: E402

from wis_lab.service.app import API, app  # noqa: E402

client = TestClient(app)
CHECKS = []
RECORD_S = 2.0


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


STATE = {}


@check("a session opens the camera and the worker reports real frames")
def _open():
    r = client.post(API + "/sessions", json={"participant_id": "camera_check",
                                             "purpose": "development",
                                             "open_camera": True, "device_index": 0}).json()
    assert r["camera"] is True, r.get("camera_error")
    STATE["sid"] = r["session_id"]
    deadline = time.time() + 5
    while time.time() < deadline:
        st = client.get(API + f"/sessions/{STATE['sid']}").json()["camera_status"]
        if st and st["frame_id"] >= 0 and st["fps_estimate"] > 0:
            STATE["res"] = st["resolution"]
            return
        time.sleep(0.2)
    raise AssertionError("camera produced no frames within 5 s")


@check("a preview jpeg is available and carries its frame id")
def _preview():
    deadline = time.time() + 5
    while time.time() < deadline:
        p = client.get(API + f"/sessions/{STATE['sid']}/preview.jpg")
        if p.status_code == 200:
            assert p.headers["content-type"] == "image/jpeg"
            assert p.content[:2] == b"\xff\xd8", "not a jpeg"
            assert int(p.headers["X-Frame-Id"]) >= 0
            return
        time.sleep(0.2)
    raise AssertionError("no preview within 5 s")


@check("a second session cannot steal the camera")
def _exclusive():
    other = client.post(API + "/sessions", json={"open_camera": True, "device_index": 0})
    assert other.status_code == 409, other.status_code
    assert other.json()["error"] == "CAMERA_OWNED_BY_ANOTHER_SESSION"


@check("a trial records through the live path and returns a scored result")
def _trial():
    t = client.post(API + f"/sessions/{STATE['sid']}/trials",
                    json={"request_id": "cam_1", "requested_word_id": "INSA"}).json()
    assert t["state"] == "RECORDING", t
    STATE["tid"] = t["trial_id"]
    STATE["frame_id_before"] = client.get(
        API + f"/sessions/{STATE['sid']}").json()["camera_status"]["frame_id"]
    time.sleep(RECORD_S)
    stopped = client.post(API + f"/trials/{STATE['tid']}/stop").json()
    assert stopped["state"] in ("FEEDBACK", "ERROR"), stopped
    res = stopped["result"]
    assert res is not None, stopped.get("error")
    STATE["result"] = res
    assert res["origin"] == "camera"
    assert res["observation_status"] in ("OK", "INSUFFICIENT")
    assert res["execution_assessment"] == "UNCONFIGURED"
    assert res["accepted"] is None
    assert res["observation_sha256"]
    assert res["timing"]["server_end_candidate_to_result_ms"] >= 0


@check("the same request_id does not start a second recording")
def _idempotent():
    again = client.post(API + f"/sessions/{STATE['sid']}/trials",
                        json={"request_id": "cam_1", "requested_word_id": "INSA"}).json()
    assert again["trial_id"] == STATE["tid"], again
    assert again["deduplicated"] is True


@check("the recorded observation is on disk in the contract shape")
def _observation_file():
    from wis_lab.core.observation import Observation, sha256_file
    path = os.path.join(ROOT, "runs", STATE["sid"], "trials", STATE["tid"] + ".npz")
    assert os.path.exists(path), path
    assert sha256_file(path) == STATE["result"]["observation_sha256"]
    obs = Observation.load(path)          # .validate() runs inside
    assert obs.n_frames > 0
    side = obs.sidecar
    assert side["origin"] == "webcam"
    assert side["mirrored"] is False, "analysis input must be un-mirrored"
    assert "NOT the sensor exposure time" in side["clock"]
    assert obs.timestamp_us[0] == 0 and (obs.timestamp_us[1:] > obs.timestamp_us[:-1]).all()


@check("a second trial reuses the same camera without a restart")
def _no_restart():
    before = client.get(API + f"/sessions/{STATE['sid']}").json()["camera_status"]["frame_id"]
    t = client.post(API + f"/sessions/{STATE['sid']}/trials",
                    json={"request_id": "cam_2", "requested_word_id": "SARANG"}).json()
    assert t["state"] == "RECORDING"
    time.sleep(1.0)
    client.post(API + f"/trials/{t['trial_id']}/stop")
    after = client.get(API + f"/sessions/{STATE['sid']}").json()["camera_status"]["frame_id"]
    # a restart would reset the worker's frame counter
    assert after > before, (before, after)
    assert t["clock_id"] == STATE["result"]["clock_id"], "clock id changed: the camera restarted"


@check("cancel yields no result and is not counted as a success")
def _cancel():
    t = client.post(API + f"/sessions/{STATE['sid']}/trials",
                    json={"request_id": "cam_3", "requested_word_id": "INSA"}).json()
    time.sleep(0.5)
    c = client.post(API + f"/trials/{t['trial_id']}/cancel").json()
    assert c["state"] == "CANCELLED"
    assert c["result"] is None
    got = client.get(API + f"/trials/{t['trial_id']}").json()
    assert got["state"] == "CANCELLED" and got["result"] is None


@check("display-ack is measured on the server clock only")
def _ack():
    a = client.post(API + f"/trials/{STATE['tid']}/display-ack", json={
        "result_id": STATE["result"]["result_id"],
        "client_event": {"clock": "browser_performance_now",
                         "client_receive_to_render_event_ms": 12.3}}).json()
    assert a["server_end_candidate_to_display_ack_ms"] >= 0
    assert "not evidence that a pixel was seen" in a["note"]
    bad = client.post(API + f"/trials/{STATE['tid']}/display-ack",
                      json={"result_id": "res_wrong"})
    assert bad.status_code == 409


@check("the attempt log records the trial without raw arrays")
def _decisions():
    path = os.path.join(ROOT, "runs", STATE["sid"], "decisions.jsonl")
    assert os.path.exists(path)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    assert any(r["trial_id"] == STATE["tid"] for r in rows)
    for r in rows:
        assert r["execution_assessment"] == "UNCONFIGURED"
        assert r["accepted"] is None
        assert "xy_px" not in r and "features" not in r


@check("close releases the camera")
def _close():
    c = client.post(API + f"/sessions/{STATE['sid']}/close").json()
    assert c["camera_released"] is True
    h = client.get(API + "/health").json()
    assert h["camera"] is None and h["camera_session_id"] is None
    d = client.get(API + "/devices").json()
    assert d["devices"], "the device is not reusable after close"


def main() -> int:
    if not REAL_CAMERA_ALLOWED:
        print("BLOCKED  real-camera checks not run (set WIS_ALLOW_REAL_CAMERA=1 to allow). "
              "No device was opened or probed. This is NOT a pass.")
        return 0
    failed = 0
    try:
        for name, fn in CHECKS:
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failed += 1
                print(f"FAIL  {name}: {exc}")
    finally:
        sid = STATE.get("sid")
        if sid:
            client.post(API + f"/sessions/{sid}/close")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
