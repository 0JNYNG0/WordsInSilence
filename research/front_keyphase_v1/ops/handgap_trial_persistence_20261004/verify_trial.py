"""Codex independent verification only. Never starts a camera thread or MediaPipe.

Public stored arrays enter the real worker's recording buffer; real take/API/model/save
paths run. Outputs live only beneath this unit and are NOT participant recordings.
"""
import builtins
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

UNIT = Path(__file__).resolve().parent
LAB = UNIT.parents[1]
PHASE = sys.argv[1] if len(sys.argv) > 1 else "before"
OUT = UNIT / ("isolated_" + PHASE)
assert not OUT.exists(), "Preserve earlier verification outputs"
OUT.mkdir()
(OUT / "NOT_PARTICIPANT_DATA.txt").write_text(
    "Mock camera trial lifecycle with public NIKL reference arrays. Not human evaluation.", encoding="utf-8")
sys.path.insert(0, str(LAB / "src"))
import cv2

attempts = []
def forbidden(*a, **k):
    attempts.append("device access")
    raise AssertionError("REAL CAMERA / DEVICE ACCESS FORBIDDEN")
cv2.VideoCapture = forbidden
from wis_lab.service import camera, app as A
from wis_lab.core.observation import Observation
camera.list_devices = forbidden
from fastapi.testclient import TestClient
import torch
torch.set_num_threads(2)

st = A.Store()
plan = json.loads((UNIT / "FROZEN_PLAN.json").read_text(encoding="utf-8"))
expected = json.loads((LAB / "ops/partial_hand_web_optin_20261003/API_AFTER.json").read_text())
source = {}
pins = {}
fields = ["xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score"]
for cid in {c["clip"] for c in plan["cases"]}:
    clip = next(c for c in st.manifest["clips"] if c["clip_id"] == cid)
    p = LAB / clip["observation_npz"]
    source[cid] = Observation.load(str(p))
    pins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()

def mock_start(self, *a, **k):
    self.extractor_id = st.lineage["extractor_id"]
    assert self._thread is None

def feed(obs):
    cam = st.camera
    assert cam._thread is None and cam._recording
    cam._actual = (obs.sidecar["width"], obs.sidecar["height"])
    with cam._lock:
        cam._rec_rows = [{k: getattr(obs, k)[i].copy() for k in fields}
                         for i in range(obs.n_frames)]

def clean(value):
    # Only timestamps of execution, not observation timestamps or transform provenance.
    if isinstance(value, dict):
        return {k: clean(v) for k,v in value.items()
                if k not in {"elapsed_ms", "elapsed_ms_both_arms", "result_event_persisted"}}
    if isinstance(value, list): return [clean(v) for v in value]
    return value

report = {"status": "RUNNING", "phase": PHASE, "hardware": "MOCK_ONLY", "cases": []}
with patch.object(A, "store", st), patch.object(A, "ROOT", str(OUT)), \
     patch.object(camera.CameraWorker, "start", mock_start):
    client = TestClient(A.app, raise_server_exceptions=False)
    def post(path, data=None):
        r = client.post(A.API + path, json=data)
        assert r.status_code == 200, (path, r.status_code, r.text[:500])
        return r.json()
    sess = post("/sessions", {"participant_id": "MOCK_PUBLIC_NOT_PERSON",
        "purpose": "diagnostic", "client_id": "mock_only", "store_video": False,
        "shadow_assignment": False, "open_camera": True})
    sid = sess["session_id"]
    for i,c in enumerate(plan["cases"]):
        demo = next(r["clip_id"] for r in st.refs if r["word_id"] == c["requested"])
        req = {"request_id": "fixed_"+str(i), "requested_word_id": c["requested"],
            "client_id": "mock_only", "research_model_key": c["model"],
            "demonstration_clip_id": demo, "attempt_label": "normal"}
        created = post(f"/sessions/{sid}/trials", req); tid = created["trial_id"]
        dedup = post(f"/sessions/{sid}/trials", dict(req,research_model_key="base"))
        assert dedup["trial_id"] == tid and dedup["deduplicated"]
        assert st.trials[tid]["research_model"]["model_key"] == c["model"]
        obs = source[c["clip"]]; feed(obs)
        if c["end"] == "auto":
            st.camera.request_stop("hand_down")
            response = client.get(A.API+f"/trials/{tid}").json()
        else: response = post(f"/trials/{tid}/stop")
        assert response["state"] == "FEEDBACK", response
        result = response["result"]
        run = Path(st.sessions[sid]["run_dir"])
        saved = json.loads((run/"trials"/(tid+".result.json")).read_text(encoding="utf-8"))
        assert clean(saved) == clean(result), "Saved result differs from API response"
        recorded = Observation.load(str(run/"trials"/(tid+".npz")))
        for k in fields:
            assert np.array_equal(getattr(recorded,k),getattr(obs,k),equal_nan=True),(i,k)
        rc = saved["research_comparison"]
        assert rc["model_id"] == c["model"]
        assert rc["identity_fixed_at_trial_creation"]["model_key"] == c["model"]
        assert saved["accepted"] is None and saved["execution_assessment"] == "UNCONFIGURED"
        if c["model"].endswith("_handgap"):
            want=expected["repaired"][c["clip"]]["research_comparison"]
            assert rc["hand_gap_comparison"]["logit_audit"] == want["hand_gap_comparison"]["logit_audit"]
            assert rc["input_transform"]["provenance"] == want["input_transform"]["provenance"]
            assert rc["identity_fixed_at_trial_creation"]["input_transform"]["transform_id"] == "wis_hand_gap_restore_v1"
        else:
            assert rc["scopes"] == expected["raw"][c["clip"]]["research_comparison"]["scopes"]
        # Confirmation never becomes a new inference input or rewrites the result file.
        checksum = hashlib.sha256((run/"trials"/(tid+".result.json")).read_bytes()).hexdigest()
        actual = next(r["word_id"] for r in st.refs if r["clip_id"] == c["clip"])
        review = post(f"/trials/{tid}/review", {"session_id":sid,"actual_word_id":actual,
            "followed_guidance":"yes" if actual==c["requested"] else "no"})
        assert review["stored"] and review["is_expert_verification"] is False
        assert hashlib.sha256((run/"trials"/(tid+".result.json")).read_bytes()).hexdigest()==checksum
        report["cases"].append({"index":i,"trial_id":tid,"source":c["clip"],"requested":c["requested"],
            "model":c["model"],"end":c["end"],"full_top1":rc["scopes"]["full"]["top1"],
            "stored_result_equal":True,"raw_arrays_exact":True,"confirmation_saved":True})
    rep=client.get(A.API+f"/sessions/{sid}/review-report").json()
    assert rep["counts"]["requested_total"]==4 and rep["counts"]["confirmed_total"]==4
    ids={g["identity"]["model_id"] for g in rep["cohorts"]["groups"].values()}
    assert ids=={"tcn_control_83212868","tcn_control_83212868_handgap"},ids
    (OUT/"SESSION_REPORT.json").write_text(json.dumps(rep,ensure_ascii=False,indent=2),encoding="utf-8")
    report["cohorts_separate"] = True
    report["report_limitation"] = "Session report scores outer repaired only, not nested original; paired evidence in result JSON"

    # Inject failure only at the result JSON write, using the real finalization path.
    tid=post(f"/sessions/{sid}/trials", {"request_id":"disk_failure","client_id":"mock_only",
        "requested_word_id":"GAMSAHAMNIDA","research_model_key":"tcn_control_83212868_handgap"})["trial_id"]
    feed(source["NIKL_GAMSAHAMNIDA_01"])
    real_open=builtins.open
    def faulty_open(file,mode="r",*a,**k):
        if tid+".result.json" in str(file) and ("w" in mode or "a" in mode or "x" in mode):
            raise OSError("Injected result disk failure for verification")
        return real_open(file,mode,*a,**k)
    with patch("builtins.open",faulty_open):
        failure=client.post(A.API+f"/trials/{tid}/stop")
    after=client.get(A.API+f"/trials/{tid}").json()
    retry=client.post(A.API+f"/trials/{tid}/stop")
    review=client.post(A.API+f"/trials/{tid}/review",json={"session_id":sid,
        "actual_word_id":"GAMSAHAMNIDA","followed_guidance":"yes"})
    report["write_failure"]={"stop_http":failure.status_code,"state_after":after["state"],
        "result_present_in_memory":after.get("result") is not None,
        "result_file_exists":(run/"trials"/(tid+".result.json")).exists(),
        "retry_http":retry.status_code,"retry":retry.json(),"review_http":review.status_code,
        "review":review.json()}
    report["write_failure_safe"]=after["state"]!="FEEDBACK" and after.get("result") is None and review.status_code!=200
    post(f"/sessions/{sid}/close")
    assert st.camera is None and st.camera_session is None
assert not attempts
for p,h in pins.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
report.update(status="PASS" if report["write_failure_safe"] else "PERSISTENCE_FAILURE_CONFIRMED",
    normal_cases_passed=len(report["cases"]),device_attempts=0,source_hashes_unchanged=True)
(UNIT/("VERIFICATION_"+PHASE.upper()+".json")).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(report,ensure_ascii=False,indent=2))
