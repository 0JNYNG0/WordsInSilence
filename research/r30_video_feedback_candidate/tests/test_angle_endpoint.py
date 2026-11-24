"""Synthetic tests for the on-demand angle-comparison endpoint.

A hand-built store is injected, so no contract, reference set, neural model, camera or recording is
loaded. The route functions are called directly: what is under test is the ownership, binding,
no-fallback, hash, cache and non-mutation behaviour, not HTTP transport.
"""
from __future__ import annotations

import json
import os

import pytest

import synthetic as S

app_module = pytest.importorskip(
    "wis_lab.service.app",
    reason="the local service layer needs fastapi, pydantic, cv2 and mediapipe installed")

from fastapi import HTTPException                                    # noqa: E402
from wis_lab.cli import ROOT                                         # noqa: E402
from wis_lab.core import video_review as V                           # noqa: E402

SESSION = "ses_synthetic"
TRIAL = "tri_synthetic"
RGB_ROWS = [7, 11, 16, 20, 25, 29, 34, 38, 43, 47, 52]
CLIP, WORD = "SYN_word_01", "synthetic_word"


class FakeStore:
    """Exactly the attributes the endpoint reads. No camera, no models, no disk scan."""

    def __init__(self, refs, trials):
        import threading
        self.lock = threading.RLock()
        self.contract = S.CONTRACT
        self.contract_sha256 = "c" * 64
        self.refs = refs
        self.lineage = {"set_id": "synthetic_set_v1", "extractor_id": "synthetic_extractor"}
        self.trials = trials
        self.sessions = {SESSION: {"session_id": SESSION, "state": "OPEN"}}
        self.camera = None
        self.camera_session = None

    def acquire_camera(self, *a, **k):                # pragma: no cover - must never be called
        raise AssertionError("the diagnostic endpoint must not touch the camera")


def _relative(path) -> str:
    """The stored observation_path is relative to ROOT, exactly as _finish writes it."""
    try:
        return os.path.relpath(str(path), ROOT).replace("\\", "/")
    except ValueError:                       # pragma: no cover - different Windows drive
        pytest.skip("pytest tmp_path is on another drive than the candidate root")


def build(tmp_path, *, refs=None, binding_status="BOUND", following_status="AVAILABLE",
          following_clip=CLIP, following_word=WORD, observation_path=None):
    obs = S.make_observation()
    npz = tmp_path / "trial.npz"
    sha = obs.save(str(npz))                         # the real hash of the real file
    manifest = S.make_manifest(obs, RGB_ROWS, session_id=SESSION, trial_id=TRIAL)
    capture = tmp_path / "trial.capture.zip"
    S.write_capture_zip(capture, manifest)
    result = {
        "result_id": "res_synthetic", "trial_id": TRIAL, "session_id": SESSION,
        "observation_sha256": sha, "top1_word_id": "another_word",
        "ranking": [{"word_id": "another_word", "distance": 0.1, "rank": 1}],
        "demonstration_following": {"status": following_status,
                                    "reference_clip_id": following_clip,
                                    "reference_word_id": following_word},
        "diagnostic_capture": {"status": "AVAILABLE", "opted_in": True},
    }
    trial = {
        "trial_id": TRIAL, "session_id": SESSION, "state": "FEEDBACK", "result": result,
        "observation_sha256": sha,
        "observation_path": observation_path or _relative(npz),
        "capture_path": str(capture),
        "reference_binding": {"status": binding_status, "demonstration_clip_id": CLIP,
                              "demonstration_word_id": WORD},
    }
    if refs is None:
        refs = [S.make_reference(word_id=WORD, clip_id=CLIP)]
    return FakeStore(refs, {TRIAL: trial}), trial, obs


@pytest.fixture
def store(tmp_path, monkeypatch):
    st, trial, obs = build(tmp_path)
    monkeypatch.setattr(app_module, "store", st)
    return st


def call(trial_id=TRIAL, session_id=SESSION):
    return app_module.get_capture_angle_comparison(trial_id, session_id=session_id)


def payload_of(response):
    return json.loads(response.body)          # strict parse: a NaN here would raise


# ----------------------------------------------------------------- happy path
def test_route_is_registered_under_the_existing_diagnostic_path():
    paths = {getattr(r, "path", None) for r in app_module.app.routes}
    assert "/api/v1/trials/{trial_id}/diagnostic/angle-comparison" in paths
    assert "/api/v1/trials/{trial_id}/diagnostic" in paths          # unchanged
    assert "/api/v1/trials/{trial_id}/diagnostic/frames/{index}" in paths


def test_returns_a_comparison_for_the_bound_reference(store):
    body = payload_of(call())
    assert body["schema"] == V.SCHEMA and body["status"] == "OK"
    assert body["reference"]["clip_id"] == CLIP and body["reference"]["word_id"] == WORD
    assert body["counts"]["saved_frames"] == len(RGB_ROWS)
    assert body["timing"]["served_from_cache"] is False
    assert body["is_a_verdict"] is False


def test_the_endpoint_does_not_touch_the_camera_or_the_result(store):
    before = json.dumps(store.trials[TRIAL]["result"], sort_keys=True)
    call()
    assert store.camera is None and store.camera_session is None
    assert json.dumps(store.trials[TRIAL]["result"], sort_keys=True) == before


def test_response_is_strict_json_and_carries_a_measured_timing(store):
    response = call()
    assert response.headers["cache-control"] == "no-store"
    body = payload_of(response)
    assert isinstance(body["timing"]["server_compute_ms"], (int, float))
    assert body["timing"]["clock"] == app_module.SERVER_CLOCK
    assert "NaN" not in response.body.decode("utf-8")


# ---------------------------------------------------------------------- cache
def test_a_repeat_call_is_served_from_cache_without_recomputing(store, monkeypatch):
    calls = []
    real = V.frame_angle_comparison

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(app_module.video_review, "frame_angle_comparison", counted)
    first = payload_of(call())
    second = payload_of(call())
    third = payload_of(call())
    assert len(calls) == 1, "slider or toggle traffic must not re-run the DTW"
    assert first["timing"]["served_from_cache"] is False
    assert second["timing"]["served_from_cache"] is True
    assert third["timing"]["served_from_cache"] is True
    assert second["timing"]["server_compute_ms"] == first["timing"]["server_compute_ms"]
    assert second["frames"] == first["frames"]
    assert app_module.ANGLE_CACHE in store.trials[TRIAL]


def test_cache_is_keyed_to_the_exact_inputs(store, monkeypatch):
    calls = []
    real = V.frame_angle_comparison
    monkeypatch.setattr(app_module.video_review, "frame_angle_comparison",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    call()
    store.contract_sha256 = "d" * 64          # a restarted server with a different contract
    call()
    assert len(calls) == 2, "a different frozen config must not be answered from the old cache"


# -------------------------------------------------------------- no fallback
def test_an_unbound_trial_is_refused_and_top1_is_not_substituted(tmp_path, monkeypatch):
    other = S.make_reference(word_id="another_word", clip_id="SYN_other_01")
    st, _, _ = build(tmp_path, binding_status="UNDETERMINED_NO_DEMONSTRATION_SPECIFIED",
                     refs=[other])
    monkeypatch.setattr(app_module, "store", st)
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.status_code == 409
    assert exc.value.detail["reason"] == "REFERENCE_NOT_BOUND"
    assert "no_fallback" in exc.value.detail


def test_an_unavailable_demonstration_is_refused(tmp_path, monkeypatch):
    st, _, _ = build(tmp_path, following_status="UNDETERMINED_OBSERVATION_INSUFFICIENT")
    monkeypatch.setattr(app_module, "store", st)
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["reason"] == "DEMONSTRATION_FOLLOWING_NOT_AVAILABLE"
    assert exc.value.detail["demonstration_following_status"] == \
        "UNDETERMINED_OBSERVATION_INSUFFICIENT"


def test_a_demonstration_that_disagrees_with_the_binding_is_refused(tmp_path, monkeypatch):
    st, _, _ = build(tmp_path, following_clip="SYN_other_01")
    monkeypatch.setattr(app_module, "store", st)
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["reason"] == "DEMONSTRATION_IDENTITY_DISAGREES_WITH_BINDING"


def test_a_bound_word_that_is_not_prepared_never_borrows_another_template(tmp_path, monkeypatch):
    other = S.make_reference(word_id="another_word", clip_id="SYN_other_01")
    st, _, _ = build(tmp_path, refs=[other])
    monkeypatch.setattr(app_module, "store", st)
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["reason"] == "BOUND_REFERENCE_IS_NOT_A_PREPARED_CANDIDATE"
    assert exc.value.detail["reference_word_id"] == WORD


# --------------------------------------------------------------- ownership
def test_another_session_cannot_read_this_capture(store):
    with pytest.raises(HTTPException) as exc:
        call(session_id="ses_someone_else")
    assert exc.value.status_code == 404


def test_an_unknown_trial_is_not_found(store):
    with pytest.raises(HTTPException) as exc:
        call(trial_id="tri_missing")
    assert exc.value.status_code == 404


def test_a_trial_without_a_saved_capture_is_not_found(store):
    store.trials[TRIAL].pop("capture_path")
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["error"] == "NO_SAVED_RGB"


# ------------------------------------------------------------------- hashes
def test_a_capture_for_another_observation_is_invalid(store):
    store.trials[TRIAL]["observation_sha256"] = "e" * 64
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["error"] == "CAPTURE_INVALID"


def test_an_observation_file_that_is_not_the_recorded_one_is_refused(tmp_path, monkeypatch):
    decoy = S.make_observation(n_frames=30, hand_span=(3, 27))
    decoy_path = tmp_path / "decoy.npz"
    decoy.save(str(decoy_path))
    st, _, _ = build(tmp_path, observation_path=_relative(decoy_path))
    monkeypatch.setattr(app_module, "store", st)
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.detail["reason"] == "OBSERVATION_HASH_MISMATCH"


def test_a_missing_observation_file_fails_alone_with_a_reason(store):
    store.trials[TRIAL]["observation_path"] = "runs/does/not/exist.npz"
    with pytest.raises(HTTPException) as exc:
        call()
    assert exc.value.status_code == 409
    assert exc.value.detail["error"] in ("ANGLE_COMPARISON_FAILED",
                                         "ANGLE_COMPARISON_UNAVAILABLE")
    assert store.trials[TRIAL]["result"]["ranking"], "the result survives the failure"
