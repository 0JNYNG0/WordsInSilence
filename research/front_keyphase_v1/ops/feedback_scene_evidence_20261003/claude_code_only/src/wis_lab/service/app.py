"""Local API. A thin transport over wis_lab.core - no score is computed in this file.

One camera-owning session per server, one active trial per session, leases instead of
connection liveness. Server monotonic time is never subtracted from a browser clock.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..cli import CONFIG_DIR, ROOT, load_json, load_reference_set, write_json
from ..core.observation import Observation, sha256_file, sha256_json
from ..core import features
from ..core import video_review
from ..core.joints import joint_deviation
from ..core.trajectory import location_path, source_observation
from ..core.recognize import (QueryPreparation, axis_deviation, predict, prepared_sample,
                              task_match)
from . import neural, review_log, diagnostic_capture, scene_evidence
from .camera import CameraError, CameraWorker

API = "/api/v1"
LEASE_TTL_S = 20.0
HEARTBEAT_HINT_S = 5.0
TRIAL_STATES = ("IDLE", "PREPARING", "RECORDING", "END_PENDING", "ANALYZING", "FEEDBACK",
                "CANCELLED", "TIMEOUT", "ERROR")
NON_SUCCESS_STATES = ("CANCELLED", "TIMEOUT", "ERROR")


class Store:
    """All mutable server state. One lock; the camera has its own thread."""

    def __init__(self, contract_path: str | None = None):
        self.lock = threading.RLock()
        self.contract_path = os.path.abspath(contract_path or os.path.join(CONFIG_DIR, "contract.json"))
        self.contract = load_json(self.contract_path)
        self.contract_sha256 = sha256_file(self.contract_path)
        self.refs, self.lineage = load_reference_set(contract=self.contract)
        self.manifest = load_json(os.path.join(CONFIG_DIR, "source_manifest.json"))
        self.sessions: dict[str, dict] = {}
        self.trials: dict[str, dict] = {}
        self.camera: CameraWorker | None = None
        self.camera_session: str | None = None
        self.started_at = time.time()
        self.telemetry_write_failures = 0

    # -- camera ownership: one session server-wide ------------------------
    def acquire_camera(self, session_id: str, device_index: int, shadow_assignment: bool = False) -> None:
        with self.lock:
            if self.camera_session and self.camera_session != session_id:
                if self._lease_alive(self.camera_session):
                    raise HTTPException(409, {
                        "error": "CAMERA_OWNED_BY_ANOTHER_SESSION",
                        "owner_session_id": self.camera_session})
                self._release_camera_locked("previous owner lease expired")
            if self.camera is not None and self.camera.shadow_assignment != bool(shadow_assignment):
                # a camera left from another session with the other assignment mode is not reused
                self._release_camera_locked("assignment mode differs")
            if self.camera is None:
                cam = CameraWorker(self.contract, device_index=device_index,
                                   shadow_assignment=shadow_assignment)
                cam.start()
                self.camera = cam
            self.camera_session = session_id

    def _release_camera_locked(self, reason: str) -> None:
        if self.camera is not None:
            self.camera.stop()
        self.camera = None
        self.camera_session = None
        return reason

    def release_camera(self, reason: str) -> None:
        with self.lock:
            self._release_camera_locked(reason)

    def _lease_alive(self, session_id: str) -> bool:
        s = self.sessions.get(session_id)
        return bool(s and s["state"] != "CLOSED"
                    and (time.monotonic() - s["lease_touched_monotonic"]) < LEASE_TTL_S)

    def reap_expired(self) -> list[str]:
        """Drop sessions whose lease expired. A dead browser tab must not hold the camera."""
        with self.lock:
            dead = [sid for sid, s in self.sessions.items()
                    if s["state"] != "CLOSED"
                    and (time.monotonic() - s["lease_touched_monotonic"]) >= LEASE_TTL_S]
            for sid in dead:
                self.sessions[sid]["state"] = "CLOSED"
                self.sessions[sid]["closed_reason"] = "LEASE_EXPIRED"
                for tid in self.sessions[sid]["trial_ids"]:
                    t = self.trials[tid]
                    if t["state"] not in ("FEEDBACK",) + NON_SUCCESS_STATES:
                        t["state"] = "CANCELLED"
                        t["error"] = "SESSION_LEASE_EXPIRED"
                        _closed_event(self.sessions[sid], t, "CANCELLED",
                                      "SESSION_LEASE_EXPIRED")
                if self.camera_session == sid:
                    self._release_camera_locked("lease expired")
                s = self.sessions[sid]
                s['camera'] = False
                write_json(
                    os.path.join(s['run_dir'], 'session.json'),
                    {k: v for k, v in s.items() if k != 'request_ids'},
                )
            return dead


store: Store | None = None
app = FastAPI(title="wis_lab local recognition API", version="1.0")


def S() -> Store:
    global store
    if store is None:
        store = Store()
    return store


def session_or_404(session_id: str) -> dict:
    s = S().sessions.get(session_id)
    if s is None:
        raise HTTPException(404, {"error": "NO_SUCH_SESSION"})
    return s


def touch(s: dict, client_id: str | None) -> None:
    if client_id and s["client_id"] and client_id != s["client_id"]:
        raise HTTPException(409, {"error": "CLIENT_MISMATCH", "owner_client_id": s["client_id"]})
    s["lease_touched_monotonic"] = time.monotonic()


# ----------------------------------------------------------------- models
class SessionCreate(BaseModel):
    participant_id: str = "self"
    purpose: str = "development"          # development | frozen_retest
    client_id: str | None = None
    store_video: bool = False
    # OPT-IN paired test only. False = the operating path exactly as before.
    shadow_assignment: bool = False
    device_index: int = 0
    open_camera: bool = True


class TrialCreate(BaseModel):
    request_id: str
    requested_word_id: str | None = None
    client_id: str | None = None
    # the clip the UI actually showed. Fixed into the trial now so the feedback can never drift
    # onto whatever a recogniser names later.
    demonstration_clip_id: str | None = None
    # OPT-IN only. Naming a research model adds a second, clearly separate reading; naming
    # nothing leaves this trial exactly as it was before functional_18.
    research_model_key: str | None = None
    # A handshape sample is not an attempt at a sign. It is recorded the same way but is never
    # counted as a sign attempt, so it cannot leak into performance statistics.
    sample_label: str | None = None
    # What the participant MEANT to do. Recorded at capture time so error labels never depend on
    # anyone remembering the order afterwards - which is how one session became unscoreable.
    attempt_label: str | None = None


class ReplayCreate(BaseModel):
    observation_id: str                   # a registered id, never a path
    session_id: str | None = None
    research_model_key: str | None = None   # opt-in, same rule as a trial
    # the reference to DRAW against. It is never given to the recogniser: predict() stays blind
    # and this comparison is computed afterwards, from the same stored clip.
    compare_to_word_id: str | None = None


class TrialReview(BaseModel):
    """What the participant says they actually signed. Never an input to any recogniser."""
    session_id: str                            # the UI's current session, checked against the trial
    actual_word_id: str | None = None          # a word id, or None / UNKNOWN
    followed_guidance: str                     # yes | no | unsure
    # a DIFFERENT question from followed_guidance: did they follow the form of the clip shown
    followed_demonstration_form: str | None = None    # null | yes | no | other_form | unsure
    note: str | None = None


class Ack(BaseModel):
    result_id: str
    client_event: dict | None = None


# ----------------------------------------------------------------- routes
@app.get(API + "/health")
def health():
    st = S()
    st.reap_expired()
    cam = st.camera.status() if st.camera else None
    return {
        "status": "OK",
        "telemetry_write_failures": st.telemetry_write_failures,
        "learned_recognition": neural.capability(),
        "contract_id": st.contract["contract_id"],
        "contract_sha256": st.contract_sha256,
        "feedback_policy": st.contract.get("feedback") or {},
        "extractor_id": st.lineage["extractor_id"],
        "reference_set_id": st.lineage["set_id"],
        "models_ready": True,
        "camera": cam,
        "camera_session_id": st.camera_session,
        "execution_assessment": "UNCONFIGURED",
        "ml_hybrid": "NOT_READY",
        "lease_ttl_s": LEASE_TTL_S,
        "heartbeat_hint_s": HEARTBEAT_HINT_S,
        "uptime_s": round(time.time() - st.started_at, 1),
    }


@app.get(API + "/references")
def references():
    st = S()
    by_word = {c["word_id"]: c for c in st.manifest["clips"]}
    return {
        "set_id": st.lineage["set_id"],
        "prepared_count": st.lineage["candidate_count"],
        "total_clip_count": len(st.manifest["clips"]),
        "excluded": st.lineage["excluded"],
        "items": [{
            "word_id": r["word_id"],
            "clip_id": r["clip_id"],
            "display_name": by_word[r["word_id"]]["display_name"],
            "aliases": by_word[r["word_id"]]["aliases"],
            "readiness": by_word[r["word_id"]]["readiness"],
            "readiness_reason": by_word[r["word_id"]]["readiness_reason"],
            "cue_flags": by_word[r["word_id"]]["cue_flags"],
            "duration_ms": by_word[r["word_id"]]["duration_ms"],
        } for r in st.refs],
        "note": "one exemplar per word: a template set, not a trained model",
    }


@app.get(API + "/references/{word_id}/clip.mp4")
def reference_clip(word_id: str):
    clip = next((c for c in S().manifest["clips"] if c["word_id"] == word_id), None)
    if clip is None:
        raise HTTPException(404, {"error": "NO_SUCH_WORD"})
    return FileResponse(clip["absolute_path"], media_type="video/mp4")


@app.get(API + "/evidence/references/{clip_id}/frames/{frame_id}")
def reference_evidence_frame(clip_id: str, frame_id: int, source_sha256: str,
                             observation_sha256: str):
    """One registered source frame by exact identity. Transport only; see scene_evidence."""
    clips = list(S().manifest["clips"])
    try:
        out = scene_evidence.reference_frame(clips, ROOT, clip_id, frame_id, source_sha256,
                                             observation_sha256)
    except scene_evidence.EvidenceError as exc:
        raise HTTPException(exc.status, {"error": "EVIDENCE_UNAVAILABLE", "reason": exc.reason})
    return Response(out["jpeg"], media_type="image/jpeg", headers=out["headers"])


@app.post(API + "/sessions")
def create_session(body: SessionCreate):
    st = S()
    st.reap_expired()
    sid = "ses_" + uuid.uuid4().hex[:12]
    with st.lock:
        s = {
            "session_id": sid,
            "participant_id": body.participant_id,
            "purpose": body.purpose,
            "client_id": body.client_id or ("cli_" + uuid.uuid4().hex[:8]),
            "store_video": body.store_video,
            "state": "OPEN",
            "created_epoch": time.time(),
            "lease_touched_monotonic": time.monotonic(),
            "trial_ids": [],
            "request_ids": {},
            "camera": False,
            "run_dir": os.path.join(ROOT, "runs", sid),
        }
        if body.shadow_assignment:        # key only when opted in: the default session is unchanged
            s["shadow_assignment"] = True
        st.sessions[sid] = s
        os.makedirs(os.path.join(s["run_dir"], "trials"), exist_ok=True)
    camera_error = None
    if body.open_camera:
        try:
            st.acquire_camera(sid, body.device_index, shadow_assignment=body.shadow_assignment)
            s["camera"] = True
        except HTTPException:
            raise
        except CameraError as exc:
            camera_error = str(exc)
    write_json(os.path.join(s["run_dir"], "session.json"), {
        k: v for k, v in s.items() if k not in ("request_ids",)})
    return {"session_id": sid, "client_id": s["client_id"], "camera": s["camera"],
            "store_video": s["store_video"],
            "camera_error": camera_error, "lease_ttl_s": LEASE_TTL_S,
            "heartbeat_hint_s": HEARTBEAT_HINT_S,
            **({"shadow_assignment": True} if s.get("shadow_assignment") else {})}


@app.get(API + "/sessions/{session_id}")
def get_session(session_id: str):
    st = S()
    st.reap_expired()
    s = session_or_404(session_id)
    active = [t for t in (st.trials[i] for i in s["trial_ids"])
              if t["state"] in ("PREPARING", "RECORDING", "END_PENDING", "ANALYZING")]
    return {
        "session_id": session_id, "state": s["state"], "purpose": s["purpose"],
        "participant_id": s["participant_id"], "client_id": s["client_id"],
        "store_video": s["store_video"],
        "camera": s["camera"] and st.camera_session == session_id,
        "camera_status": st.camera.status() if (st.camera and st.camera_session == session_id) else None,
        "trial_count": len(s["trial_ids"]),
        "active_trial_id": active[0]["trial_id"] if active else None,
        "closed_reason": s.get("closed_reason"),
        "lease_age_s": round(time.monotonic() - s["lease_touched_monotonic"], 2),
    }


@app.post(API + "/sessions/{session_id}/heartbeat")
def heartbeat(session_id: str, body: dict | None = None):
    s = session_or_404(session_id)
    if s["state"] == "CLOSED":
        raise HTTPException(409, {"error": "SESSION_CLOSED", "reason": s.get("closed_reason")})
    touch(s, (body or {}).get("client_id"))
    return {"session_id": session_id, "lease_ttl_s": LEASE_TTL_S, "state": s["state"]}


@app.post(API + "/sessions/{session_id}/trials")
def create_trial(session_id: str, body: TrialCreate):
    st = S()
    st.reap_expired()
    s = session_or_404(session_id)
    if s["state"] == "CLOSED":
        raise HTTPException(409, {"error": "SESSION_CLOSED", "reason": s.get("closed_reason")})
    touch(s, body.client_id)

    with st.lock:
        # same request_id -> same trial, never a second recording
        if body.request_id in s["request_ids"]:
            tid = s["request_ids"][body.request_id]
            return {"trial_id": tid, "deduplicated": True, "state": st.trials[tid]["state"]}
        active = [st.trials[i] for i in s["trial_ids"]
                  if st.trials[i]["state"] in ("PREPARING", "RECORDING", "END_PENDING", "ANALYZING")]
        if active:
            raise HTTPException(409, {"error": "TRIAL_ALREADY_ACTIVE",
                                      "trial_id": active[0]["trial_id"],
                                      "state": active[0]["state"]})
        if st.camera is None or st.camera_session != session_id:
            raise HTTPException(409, {"error": "CAMERA_NOT_OWNED_BY_SESSION"})
        if not body.requested_word_id and not body.sample_label:
            raise HTTPException(400, {"error": "NEED_A_WORD_OR_A_SAMPLE_LABEL"})

        tid = "tri_" + uuid.uuid4().hex[:12]
        t = {
            "trial_id": tid, "session_id": session_id, "state": "PREPARING",
            "requested_word_id": body.requested_word_id, "request_id": body.request_id,
            "sample_label": body.sample_label,
            "attempt_label": body.attempt_label,
            "clock_id": st.camera.clock_id, "created_monotonic_ns": time.perf_counter_ns(),
            "result": None, "error": None, "timing": {}, "origin": "camera",
            # fixed now, from the clip the UI says it is showing. Never inferred afterwards.
            "reference_binding": _reference_binding(st, body.demonstration_clip_id,
                                                    body.requested_word_id),
            # fixed now as well: which research model, if any, this trial opted into. It cannot
            # be changed while the trial runs, and a trial that named none never gets one.
            "research_model": _research_choice(body.research_model_key),
        }
        st.trials[tid] = t
        s["trial_ids"].append(tid)
        s["request_ids"][body.request_id] = tid
        try:
            if s.get("store_video", False):
                st.camera.begin_recording(store_video=True)
            else:
                st.camera.begin_recording()
            t["state"] = "RECORDING"
        except CameraError as exc:
            t["state"] = "ERROR"
            t["error"] = str(exc)
            raise HTTPException(409, {"error": "CAMERA_BUSY", "detail": str(exc)})
    return {"trial_id": tid, "state": t["state"], "clock_id": t["clock_id"],
            "max_record_s": 10.0, "deduplicated": False}


_FORM_REVIEW_PATH = os.path.join(ROOT, "evidence", "external", "five_word_visual_review.json")


def _form_review_states() -> dict:
    """The external visual observation, by word. An AI-assistant review, not an expert's."""
    try:
        with open(_FORM_REVIEW_PATH, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    rv = doc.get("reviewer") or {}
    return {r["word_id"]: {"state": r["visual_form_status"],
                           "reviewer_kind": rv.get("kind"),
                           "human_expert_validation": rv.get("human_expert_validation")}
            for r in doc.get("rows", [])}


def _reference_binding(st: Store, clip_id, requested_word_id) -> dict:
    """Fix WHICH demonstration this trial is about, at the moment the trial is created.

    An unspecified, unknown or unsupported demonstration is said so. It is never quietly replaced
    by whatever the recogniser later names: that would make the feedback describe a guess.
    """
    base = {"reference_set_id": st.lineage["set_id"], "requested_word_id": requested_word_id,
            "bound_at": "trial_creation",
            "note": "the clip this trial's demonstration feedback is about. Binding a clip does "
                    "not claim it is the only valid form of that word."}
    if not clip_id:
        return dict(base, status="UNDETERMINED_NO_DEMONSTRATION_SPECIFIED",
                    demonstration_clip_id=None, demonstration_word_id=None)
    clip = next((c for c in st.manifest["clips"] if c["clip_id"] == clip_id), None)
    if clip is None:
        return dict(base, status="UNDETERMINED_UNKNOWN_DEMONSTRATION",
                    demonstration_clip_id=clip_id, demonstration_word_id=None)
    prepared = next((r for r in st.refs if r["clip_id"] == clip_id), None)
    states = _form_review_states()
    out = dict(base, demonstration_clip_id=clip_id,
               demonstration_word_id=clip["word_id"],
               observation_npz=clip["observation_npz"],
               observation_sha256=clip.get("observation_sha256"),
               source_sha256=clip.get("sha256"),
               readiness=clip.get("readiness"),
               form_review_state=states.get(clip["word_id"], {"state": "NOT_REVIEWED"}),
               prepared_as_candidate=prepared is not None)
    if prepared is None:
        out["status"] = "UNDETERMINED_DEMONSTRATION_NOT_A_PREPARED_CANDIDATE"
    elif requested_word_id and clip["word_id"] != requested_word_id:
        out["status"] = "UNDETERMINED_DEMONSTRATION_WORD_DIFFERS_FROM_REQUEST"
    else:
        out["status"] = "BOUND"
    return out


def _research_choice(key) -> dict:
    """Fix the opt-in at trial creation, with the identity of whatever was named.

    Opting in is explicit. No default is substituted for a key that fails to load, and a key that
    is not in the registry is said so rather than quietly ignored.
    """
    if not key:
        return {"opted_in": False, "model_key": None,
                "meaning": "no research candidate was requested for this trial; the default "
                           "recogniser answers alone, exactly as before"}
    ident = neural.model_identity(key)
    return {"opted_in": True, "fixed_at": "trial_creation", "no_fallback_to_default": True,
            **ident}


def _research_comparison(obs, t: dict) -> dict | None:
    """The opt-in second reading. Never replaces the default, never borrows its answer."""
    choice = t.get("research_model") or {}
    if not choice.get("opted_in"):
        return None
    key = choice.get("model_key")
    try:
        out = neural.recognise(obs, model_key=key)
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        out = {"schema": neural.ADAPTER_SCHEMA, "model_key": key,
               "model_status": "ADAPTER_ERROR", "detail": type(exc).__name__, "ranking": None,
               "is_a_verdict": False}
    out["requested_word_coverage"] = neural.word_coverage(t.get("requested_word_id"),
                                                          model_key=key)
    # project_words_in_model is part of the identity, not decoration: it is the declared task a
    # later report groups by. Without it the stored result cannot say what set the answer was over.
    out["identity_fixed_at_trial_creation"] = {k: choice.get(k) for k in
                                               ("model_key", "model_id", "checkpoint_sha256",
                                                "feature_schema", "trained_class_count",
                                                "project_words_in_model")}
    out["is_under_validation"] = True
    out["promotion"] = "none: the default recogniser, the pass line and every verdict field are "\
                       "unchanged"
    out["reading_rule"] = ("naming a word is not a statement that the sign was performed "
                           "correctly, and the scopes have different candidate counts so their "
                           "numbers are not comparable with each other")
    return out


def _split_result(result: dict, t: dict) -> None:
    """Two named blocks over what was already computed. No distance is recalculated here."""
    binding = t.get("reference_binding") or {"status": "UNDETERMINED_NO_DEMONSTRATION_SPECIFIED"}
    result["reference_binding"] = binding
    lr = result.get("learned_recognition") or {}
    rank = lr.get("ranking") or {}
    research = result.get("research_comparison")
    result["lexical_identification"] = {
        "question": "which word is this?",
        "research_comparison": ({"model_key": research.get("model_key"),
                                 "model_status": research.get("model_status"),
                                 "scopes": research.get("scopes"),
                                 "is_under_validation": True,
                                 "separate_from_the_default": True}
                                if research else
                                {"opted_in": False,
                                 "meaning": "no research candidate was requested"}),
        "dtw": {"candidate_set": "prepared reference set",
                "candidate_count": len([r for r in (result.get("ranking") or [])
                                        if r.get("distance") is not None]),
                "top1": result.get("top1_word_id"), "ranking": result.get("ranking")},
        "neural": {"model_status": lr.get("model_status"),
                   "over_all_trained_classes": rank.get("over_all_trained_classes"),
                   "restricted_to_project_words": rank.get("restricted_to_project_words")},
        "requested_word_given_to_recognisers": False,
    }
    task = result.get("task") or {}
    if binding.get("status") != "BOUND":
        result["demonstration_following"] = {
            "question": "did this follow the demonstration that was shown?",
            "status": binding.get("status"),
            "reference_clip_id": binding.get("demonstration_clip_id"),
            "axis_deviation": None, "axis_checks": None, "joint_deviation": None,
            "location_path": None,
            "why": "no demonstration is bound for this trial, so there is nothing to compare "
                   "against. The identification top1 is NOT used as a stand-in.",
        }
        return
    # AVAILABLE has to be earned, not inferred from a task block merely existing. The analysis
    # must be ABOUT the bound word, and it must rest on an observation the predict path accepted;
    # otherwise a different reference's numbers, or numbers from a clip the path called
    # insufficient, would be shown as this demonstration's feedback.
    out = {
        "question": "did this follow the demonstration that was shown?",
        "reference_clip_id": binding.get("demonstration_clip_id"),
        "reference_word_id": binding.get("demonstration_word_id"),
        "form_review_state": binding.get("form_review_state"),
        "observation_status": result.get("observation_status"),
        "insufficient_reason": result.get("insufficient_reason"),
        "axis_deviation": None, "axis_checks": None, "joint_deviation": None,
        "location_path": None, "is_a_verdict": False,
    }
    if not task or not task.get("axis_deviation_vs_requested"):
        out["status"] = "UNDETERMINED_NO_TASK_ANALYSIS"
    elif task.get("requested_word_id") != binding.get("demonstration_word_id"):
        out["status"] = "UNDETERMINED_ANALYSIS_IS_FOR_A_DIFFERENT_WORD"
        out["analysis_word_id"] = task.get("requested_word_id")
    elif result.get("observation_status") != "OK":
        out["status"] = "UNDETERMINED_OBSERVATION_INSUFFICIENT"
    else:
        # the same analysis task_match already produced against that word; nothing is recomputed
        out.update(status="AVAILABLE",
                   axis_deviation=task.get("axis_deviation_vs_requested"),
                   axis_checks=task.get("axis_checks_vs_requested"),
                   joint_deviation=task.get("joint_deviation_vs_requested"),
                   location_path=task.get("location_path_vs_requested"))
    result["demonstration_following"] = out


def _finish(t: dict, st: Store, s: dict) -> dict:
    """Pull the take off the camera, score it through the same core the CLI uses."""
    t["state"] = "ANALYZING"
    obs, meta = st.camera.take()
    t["timing"]["server_end_candidate_ns"] = meta["end_candidate_ns"]
    t["end_reason"] = meta["end_reason"]
    if obs is None or obs.n_frames == 0:
        t["state"] = "ERROR"
        t["error"] = "NO_FRAMES_RECORDED"
        _closed_event(s, t, "ERROR", "NO_FRAMES_RECORDED")
        return t

    npz = os.path.join(s["run_dir"], "trials", t["trial_id"] + ".npz")
    obs.save(npz)
    t["observation_path"] = os.path.relpath(npz, ROOT).replace("\\", "/")
    t["observation_sha256"] = obs.sidecar["observation_sha256"]
    shadow = None
    sobs = meta.pop("_shadow_obs", None)
    if s.get("shadow_assignment", False):
        # saved for OFFLINE paired comparison only: never scored live, never shown, never a verdict
        try:
            if sobs is None:
                shadow = {"status": "MISSING", "opted_in": True}
            else:
                spath = os.path.join(s["run_dir"], "trials", t["trial_id"] + ".shadow_start_provisional.npz")
                sobs.save(spath)
                shadow = {"status": "SAVED", "opted_in": True,
                          "candidate": "start_provisional_candidate_20260926",
                          "path": os.path.relpath(spath, ROOT).replace("\\", "/"),
                          "observation_sha256": sobs.sidecar["observation_sha256"],
                          "same_frames_as_primary": True, "scored_live": False}
        except Exception as exc:  # a shadow failure must never touch the operating result
            shadow = {"status": "ERROR", "opted_in": True, "reason": type(exc).__name__}
    if shadow is not None:
        t["shadow_assignment"] = shadow

    capture_started = time.perf_counter_ns()
    capture_path = os.path.join(s["run_dir"], "trials", t["trial_id"] + ".capture.zip")
    capture = {"status": "DISABLED", "opted_in": False}
    if s.get("store_video", False):
        try:
            capture = diagnostic_capture.persist_capture(
                meta.pop("_capture_buffer", None), obs, capture_path,
                session_id=s["session_id"], trial_id=t["trial_id"])
            if capture["status"] == "DISABLED":
                capture = {"status": "UNAVAILABLE", "opted_in": True,
                           "reason": "CAPTURE_BUFFER_MISSING"}
            if capture["status"] in ("AVAILABLE", "PARTIAL"):
                t["capture_path"] = capture_path
        except Exception as exc:  # a video storage failure must not discard recognition
            capture = {"status": "ERROR", "opted_in": True, "reason": type(exc).__name__}
    meta.pop("_capture_buffer", None)
    capture["persist_ms"] = round((time.perf_counter_ns() - capture_started) / 1e6, 3)

    # this take's query preparation, computed at most once and shared only by predict and
    # task_match below; it ends with this call. neural.recognise still gets the raw obs.
    prep = QueryPreparation(obs)
    result = predict(prep, st.refs, st.contract)
    result["result_id"] = "res_" + uuid.uuid4().hex[:12]
    result["trial_id"] = t["trial_id"]
    result["session_id"] = t["session_id"]
    result["observation_sha256"] = t["observation_sha256"]
    result["reference_set_id"] = st.lineage["set_id"]
    result["contract_id"] = st.contract["contract_id"]
    result["extractor_id"] = st.lineage["extractor_id"]
    result["origin"] = "camera"
    result["diagnostic_capture"] = capture
    if "shadow_assignment" in t:
        result["shadow_assignment"] = t["shadow_assignment"]
    result["end_reason"] = meta["end_reason"]
    if t["requested_word_id"] and not t.get("sample_label"):
        # the axes are checked against the REQUESTED word here, not against the prediction
        result["task"] = task_match(result, t["requested_word_id"], observation=prep,
                                    reference_set=st.refs, frozen_config=st.contract)
    if t.get("sample_label"):
        result["sample_label"] = t["sample_label"]
        result["is_sign_attempt"] = False
        result["note"] = ("handshape sample, not an attempt at a sign. Never counted as "
                          "performance.")

    # the same observation, read by the research candidate. Blind: the requested word is not
    # passed in. A failure here is reported in the field and never ends the trial.
    try:
        result["learned_recognition"] = neural.recognise(obs)
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        result["learned_recognition"] = {"schema": neural.ADAPTER_SCHEMA,
                                         "model_status": "ADAPTER_ERROR",
                                         "detail": type(exc).__name__, "ranking": None,
                                         "is_a_verdict": False}
    result["learned_recognition"]["requested_word_coverage"] = neural.word_coverage(
        t.get("requested_word_id"))
    result["research_model"] = t.get("research_model")
    research = _research_comparison(obs, t)
    if research is not None:
        result["research_comparison"] = research
    _split_result(result, t)
    # Stamped HERE, after the whole result is assembled. It used to be taken before
    # neural.recognise, so the neural inference time fell outside the figure entirely. Earlier
    # recordings were measured under the old scope and are NOT reinterpreted.
    t["timing"]["server_result_ns"] = time.perf_counter_ns()
    t["timing"]["server_end_candidate_to_result_ms"] = round(
        (t["timing"]["server_result_ns"] - meta["end_candidate_ns"]) / 1e6, 2)
    t["timing"]["timing_scope"] = ("end candidate to the assembled result, INCLUDING the neural "
                                   "adapter, whether it succeeded or failed")
    t["timing"]["timing_scope_id"] = "server_end_candidate_to_assembled_result_v2"
    result["timing"] = dict(t["timing"])
    result["clock_id"] = t["clock_id"]
    t["result"] = result
    t["state"] = "FEEDBACK"

    with open(os.path.join(s["run_dir"], "trials", t["trial_id"] + ".result.json"),
              "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    _append_decision(s, t, result)
    ok, err = _timing_event(s, {
        "event": "trial_result",
        "session_id": s["session_id"], "trial_id": t["trial_id"],
        "result_id": result["result_id"], "clock_id": t["clock_id"],
        "server_clock": SERVER_CLOCK,
        "server_end_candidate_ns": t["timing"].get("server_end_candidate_ns"),
        "server_result_ns": t["timing"].get("server_result_ns"),
        "server_end_candidate_to_result_ms":
            t["timing"].get("server_end_candidate_to_result_ms"),
        "observation_status": result["observation_status"],
        "end_reason": result.get("end_reason"),
        "origin": result["origin"],
    })
    t["timing"]["result_event_persisted"] = ok
    if err:
        t["timing"]["result_event_error"] = err
    return t


SERVER_CLOCK = "time.perf_counter_ns, monotonic, this process only"
TIMING_EVENTS = "timing_events.jsonl"


def _timing_event(s: dict, payload: dict):
    """Append one telemetry row and report honestly whether it reached the disk.

    An extension record: nothing already written is rewritten. Returns (ok, error); callers must
    not describe a failed write as a saved measurement.
    """
    row = {**payload, "recorded_at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        with open(os.path.join(s["run_dir"], TIMING_EVENTS), "a",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True, None
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _closed_event(s: dict, t: dict, outcome: str, detail=None):
    """A trial that ends without a result: cancelled, or a technical error.

    Without this row such a trial leaves no trace in timing_events.jsonl, and an aggregate built
    from that file would quietly drop it from its own denominator instead of counting it.
    """
    prior = t.get("closed_event")
    if prior is not None:
        # a trial can be closed twice - lease expiry then an explicit close - but it is one trial
        return prior["persisted"], prior["error"]
    ok, err = _timing_event(s, {
        "event": "trial_closed", "session_id": s["session_id"], "trial_id": t["trial_id"],
        "clock_id": t.get("clock_id"), "outcome": outcome, "state": t.get("state"),
        "end_reason": t.get("end_reason"), "detail": detail,
        "has_result": t.get("result") is not None,
        "note": "no result was produced, so this trial can never carry a display-ack latency",
    })
    t["closed_event"] = {"outcome": outcome, "detail": detail, "persisted": ok, "error": err}
    if not ok:
        S().telemetry_write_failures += 1
    return ok, err


def _append_decision(s: dict, t: dict, result: dict) -> None:
    row = {
        "trial_id": t["trial_id"], "session_id": s["session_id"],
        "purpose": s["purpose"], "participant_id": s["participant_id"],
        "is_sign_attempt": not t.get("sample_label"),
        "sample_label": t.get("sample_label"),
        "attempt_label": t.get("attempt_label"),
        "requested_word_id": t["requested_word_id"],
        "observation_status": result["observation_status"],
        "top1_word_id": result.get("top1_word_id"),
        "top1_distance": result.get("top1_distance"),
        "margin_to_second": result.get("margin_to_second"),
        "task_match": result.get("task", {}).get("task_match"),
        "execution_assessment": result["execution_assessment"],
        "accepted": result["accepted"],
        "end_reason": result.get("end_reason"),
        "origin": result["origin"],
        "observation_sha256": result.get("observation_sha256"),
    }
    with open(os.path.join(s["run_dir"], "decisions.jsonl"), "a",
              encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _capture_for(trial_id: str, session_id: str):
    t = S().trials.get(trial_id)
    if t is None or t["session_id"] != session_id:
        raise HTTPException(404, {"error": "NO_SUCH_CAPTURE"})
    path = t.get("capture_path")
    if not path:
        raise HTTPException(404, {"error": "NO_SAVED_RGB"})
    try:
        manifest = diagnostic_capture.read_manifest(path)
        if (manifest["trial_id"] != trial_id or manifest["session_id"] != session_id
                or manifest["observation_sha256"] != t["observation_sha256"]):
            raise ValueError("capture identity mismatch")
    except Exception:
        raise HTTPException(409, {"error": "CAPTURE_INVALID"})
    return path, manifest


@app.get(API + "/trials/{trial_id}/diagnostic")
def get_capture(trial_id: str, session_id: str):
    _, manifest = _capture_for(trial_id, session_id)
    return JSONResponse(manifest, headers={"Cache-Control": "no-store"})


@app.get(API + "/trials/{trial_id}/diagnostic/frames/{index}")
def get_capture_frame(trial_id: str, index: int, session_id: str):
    path, _ = _capture_for(trial_id, session_id)
    try:
        data = diagnostic_capture.read_frame(path, index)
    except IndexError:
        raise HTTPException(404, {"error": "NO_SUCH_CAPTURE_FRAME"})
    except Exception:
        raise HTTPException(409, {"error": "CAPTURE_FRAME_INVALID"})
    return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ------------------------------------------- video-review angle comparison
# On demand only, and only for a trial that already has its result. It draws on the frames the
# person can actually see; it never changes a score, a ranking, a threshold or the camera.
ANGLE_CACHE = "angle_comparison_cache"


def _angle_identity(t: dict, manifest: dict, ref: dict, st: Store) -> str:
    """Everything the comparison depends on. A cache entry can never answer for another input."""
    return sha256_json({
        "module": video_review.SCHEMA,
        "trial_id": t["trial_id"], "session_id": t["session_id"],
        "observation_sha256": t.get("observation_sha256"),
        "capture_observation_sha256": manifest.get("observation_sha256"),
        # the exact saved frames, by identity AND by image digest
        "frames": [[f.get("frame_id"), f.get("timestamp_us"), f.get("observation_row"),
                    f.get("sha256")] for f in (manifest.get("frames") or [])],
        "reference_clip_id": ref.get("clip_id"), "reference_word_id": ref.get("word_id"),
        "reference_set_id": st.lineage["set_id"],
        "contract_sha256": st.contract_sha256,
        "cost": video_review.cost_config(st.contract),
        "band_ratio": video_review.band_ratio(st.contract),
    })


def _bound_reference(t: dict, st: Store) -> dict:
    """The reference this trial is ALREADY bound to, or an explicit refusal.

    There is deliberately no fallback. Not the prediction's top1, not the requested word, not the
    nearest prepared clip: a demonstration nobody bound is a demonstration nobody can be compared
    against, and inventing one would make the colours describe a guess.
    """
    result = t.get("result") or {}
    df = result.get("demonstration_following") or {}
    binding = t.get("reference_binding") or {}
    if binding.get("status") != "BOUND":
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "REFERENCE_NOT_BOUND",
                                  "binding_status": binding.get("status"),
                                  "no_fallback": "the predicted word is not used as a stand-in"})
    if df.get("status") != "AVAILABLE":
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "DEMONSTRATION_FOLLOWING_NOT_AVAILABLE",
                                  "demonstration_following_status": df.get("status"),
                                  "no_fallback": "the predicted word is not used as a stand-in"})
    clip_id, word_id = df.get("reference_clip_id"), df.get("reference_word_id")
    if (clip_id != binding.get("demonstration_clip_id")
            or word_id != binding.get("demonstration_word_id")):
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "DEMONSTRATION_IDENTITY_DISAGREES_WITH_BINDING"})
    ref = next((r for r in st.refs
                if r["clip_id"] == clip_id and r["word_id"] == word_id), None)
    if ref is None:
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "BOUND_REFERENCE_IS_NOT_A_PREPARED_CANDIDATE",
                                  "reference_clip_id": clip_id, "reference_word_id": word_id,
                                  "no_fallback": "another template is not substituted"})
    return ref


@app.get(API + "/trials/{trial_id}/diagnostic/angle-comparison")
def get_capture_angle_comparison(trial_id: str, session_id: str):
    """Finger-angle differences on the SAVED RGB frames of one trial. Observation, not a verdict."""
    st = S()
    _, manifest = _capture_for(trial_id, session_id)      # same ownership + identity checks
    with st.lock:
        t = st.trials.get(trial_id)
        if t is None or t["session_id"] != session_id:
            raise HTTPException(404, {"error": "NO_SUCH_CAPTURE"})
        ref = _bound_reference(t, st)
        npz_rel, obs_sha = t.get("observation_path"), t.get("observation_sha256")
        contract = st.contract
        identity = _angle_identity(t, manifest, ref, st)
        cached = t.get(ANGLE_CACHE)
    if not npz_rel or not obs_sha:
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "NO_SAVED_OBSERVATION_FOR_THIS_TRIAL"})
    if cached and cached["identity"] == identity:
        return _angle_response(cached, served_from_cache=True)

    # Computed with NO lock held: this is a DTW, and the camera lifecycle must not wait on it.
    started = time.perf_counter_ns()
    try:
        npz = os.path.join(ROOT, npz_rel)
        on_disk = sha256_file(npz)
        if on_disk != obs_sha or manifest.get("observation_sha256") != obs_sha:
            raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                      "reason": "OBSERVATION_HASH_MISMATCH"})
        obs = Observation.load(npz)
        if (obs.sidecar or {}).get("observation_sha256") != obs_sha:
            raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                      "reason": "OBSERVATION_SIDECAR_HASH_MISMATCH"})
        payload = video_review.frame_angle_comparison(obs, manifest, ref, contract)
        # strict JSON, proved here rather than promised: NaN/Infinity would raise
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except HTTPException:
        raise
    except video_review.ReviewError as exc:
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_UNAVAILABLE",
                                  "reason": "INPUTS_DO_NOT_BELONG_TOGETHER",
                                  "detail": str(exc)})
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        # Fails alone. The result, its ranking and the saved-video viewer are untouched.
        raise HTTPException(409, {"error": "ANGLE_COMPARISON_FAILED",
                                  "reason": type(exc).__name__})
    entry = {"identity": identity, "payload": payload,
             "compute_ms": round((time.perf_counter_ns() - started) / 1e6, 2)}
    served = False
    with st.lock:
        t = st.trials.get(trial_id)
        if t is not None and t.get("observation_sha256") == obs_sha:
            prior = t.get(ANGLE_CACHE)
            if prior and prior["identity"] == identity:
                # a concurrent request for the SAME identity won the race: return its measurement,
                # not a second one, so the timing field still describes the entry being served
                entry, served = prior, True
            else:
                t[ANGLE_CACHE] = entry
    return _angle_response(entry, served_from_cache=served)


def _angle_response(entry: dict, *, served_from_cache: bool) -> Response:
    timing = {
        "server_compute_ms": entry["compute_ms"],
        "served_from_cache": served_from_cache,
        "clock": SERVER_CLOCK,
        "scope": "observation load, hash check, one reference DTW and payload build, measured on "
                 "this machine for this trial only. It is not a benchmark and not a promise.",
    }
    # The cached PAYLOAD is reused; only the timing envelope is new, so a repeated call (slider,
    # bone toggle, re-opened panel) re-serializes but never re-runs the DTW. allow_nan=False is
    # re-applied on every response rather than trusted from the first one.
    body = json.dumps({**entry["payload"], "timing": timing}, ensure_ascii=False, allow_nan=False)
    return Response(body, media_type="application/json",
                    headers={"Cache-Control": "no-store"})


@app.post(API + "/trials/{trial_id}/stop")
def stop_trial(trial_id: str):
    st = S()
    t = st.trials.get(trial_id)
    if t is None:
        raise HTTPException(404, {"error": "NO_SUCH_TRIAL"})
    s = session_or_404(t["session_id"])
    touch(s, None)
    with st.lock:
        if t["state"] in ("FEEDBACK",) + NON_SUCCESS_STATES:
            return {"trial_id": trial_id, "state": t["state"], "already_final": True}
        if st.camera is None:
            t["state"] = "ERROR"
            t["error"] = "CAMERA_GONE"
            _closed_event(s, t, "ERROR", "CAMERA_GONE")
            raise HTTPException(409, {"error": "CAMERA_GONE"})
        try:
            st.camera.request_stop("manual")
        except CameraError:
            pass
        t["state"] = "END_PENDING"
        _finish(t, st, s)
    return {"trial_id": trial_id, "state": t["state"], "result": t["result"], "error": t["error"]}


@app.post(API + "/trials/{trial_id}/cancel")
def cancel_trial(trial_id: str):
    """Cancelled is not a success and produces no result."""
    st = S()
    t = st.trials.get(trial_id)
    if t is None:
        raise HTTPException(404, {"error": "NO_SUCH_TRIAL"})
    s = session_or_404(t["session_id"])
    touch(s, None)
    with st.lock:
        if t["state"] in ("FEEDBACK",) + NON_SUCCESS_STATES:
            return {"trial_id": trial_id, "state": t["state"], "already_final": True}
        if st.camera is not None:
            try:
                st.camera.request_stop("cancel")
                st.camera.take()
            except CameraError:
                pass
        t["state"] = "CANCELLED"
        t["result"] = None
        ok, err = _closed_event(s, t, "CANCELLED")
    return {"trial_id": trial_id, "state": "CANCELLED", "result": None,
            "telemetry_persisted": ok, "telemetry_error": err}


@app.get(API + "/trials/{trial_id}")
def get_trial(trial_id: str):
    st = S()
    t = st.trials.get(trial_id)
    if t is None:
        raise HTTPException(404, {"error": "NO_SUCH_TRIAL"})
    s = st.sessions.get(t["session_id"])
    # the camera may have reached its own end candidate while the page was polling
    with st.lock:
        if (t["state"] == "RECORDING" and st.camera is not None
                and st.camera_session == t["session_id"]
                and st.camera.end_candidate() is not None):
            t["state"] = "END_PENDING"
            _finish(t, st, s)
    return {"trial_id": trial_id, "state": t["state"], "result": t["result"],
            "error": t["error"], "requested_word_id": t["requested_word_id"],
            "end_reason": t.get("end_reason")}


@app.post(API + "/trials/{trial_id}/display-ack")
def display_ack(trial_id: str, body: Ack):
    """Record the display acknowledgement once, on disk, without overwriting the first one.

    The measurement is server end-candidate to ack received, return trip included. It is a PROXY:
    not motion end to feedback seen, and an ack is not evidence that a pixel was displayed or that
    anyone looked. Browser numbers stay on the browser clock and are never subtracted from ours.
    """
    st = S()
    with st.lock:
        t = st.trials.get(trial_id)
        if t is None or t["result"] is None:
            raise HTTPException(404, {"error": "NO_SUCH_RESULT"})
        if body.result_id != t["result"]["result_id"]:
            raise HTTPException(409, {"error": "RESULT_ID_MISMATCH"})
        prior = t.get("display_ack")
        if prior is not None:
            # a repeat, including a concurrent one, returns the FIRST measurement unchanged
            return {**prior["response"], "duplicate": True}
        now = time.perf_counter_ns()
        s = session_or_404(t["session_id"])
        client = dict(body.client_event or {})
        ms = round((now - t["timing"]["server_end_candidate_ns"]) / 1e6, 2)
        record = {
            "event": "display_ack",
            "session_id": s["session_id"], "trial_id": t["trial_id"],
            "result_id": body.result_id, "clock_id": t["clock_id"],
            "server_clock": SERVER_CLOCK,
            "server_end_candidate_ns": t["timing"].get("server_end_candidate_ns"),
            "server_result_ns": t["timing"].get("server_result_ns"),
            "server_ack_received_ns": now,
            "server_end_candidate_to_result_ms":
                t["timing"].get("server_end_candidate_to_result_ms"),
            "server_result_to_ack_ms": (
                round((now - t["timing"]["server_result_ns"]) / 1e6, 2)
                if t["timing"].get("server_result_ns") else None),
            "server_end_candidate_to_display_ack_ms": ms,
            "metric_definition": "server end-candidate to display ack received, return trip "
                                 "included; a proxy, not motion end to feedback seen",
            "motion_end_to_feedback_verified": False,
            "observation_status": t["result"]["observation_status"],
            "end_reason": t["result"].get("end_reason"),
            "document_visibility": client.get("document_visibility"),
            "client_clock": client.get("clock"),
            "client_reported": client,
            "client_note": "browser numbers, never differenced against the server clock; a "
                           "requestAnimationFrame callback is a render event, not a pixel",
        }
        ok, err = _timing_event(s, record)
        response = {
            "trial_id": trial_id, "result_id": body.result_id,
            "metric": "server_end_candidate_to_display_ack_ms",
            "server_end_candidate_to_display_ack_ms": ms,
            "clock": SERVER_CLOCK,
            "persisted": ok, "persist_error": err, "duplicate": False,
            "motion_end_to_feedback_verified": False,
            "note": "proxy latency on the server clock; an ack is not evidence that a pixel was "
                    "seen. Browser timings are kept separately under client_reported. A failed "
                    "write is reported as persisted=false, never as a saved measurement.",
        }
        t["display_ack"] = {"record": record, "response": response}
        t["timing"]["server_ack_received_ns"] = now
        t["timing"]["server_end_candidate_to_display_ack_ms"] = ms
        t["timing"]["display_ack_persisted"] = ok
        t["timing"]["client_reported"] = client
        t["result"]["timing"] = dict(t["timing"])
        return response


@app.post(API + "/trials/{trial_id}/review")
def trial_review(trial_id: str, body: TrialReview):
    """Record the participant's own confirmation for a FINISHED trial of their own session.

    Append only. A resend of the same content is ignored, a correction becomes a new revision,
    and a write that did not reach the disk is never reported as stored.
    """
    st = S()
    t = st.trials.get(trial_id)
    if t is None:
        raise HTTPException(404, {"error": "NO_SUCH_TRIAL"})
    if t.get("result") is None or t["state"] != "FEEDBACK":
        raise HTTPException(409, {"error": "TRIAL_NOT_FINISHED", "state": t["state"]})
    # the body carries the session the UI thinks it is in. Without it a confirmation could be
    # filed against another session's trial and nothing would notice. This guards against a mixed
    # up local UI; it is not an authentication scheme.
    if body.session_id != t["session_id"]:
        raise HTTPException(409, {"error": "TRIAL_NOT_IN_SESSION",
                                  "reason": "this trial belongs to a different session"})
    s = session_or_404(t["session_id"])
    known = {r["word_id"] for r in st.refs}
    with st.lock:
        out = review_log.submit(s["run_dir"], s["session_id"], trial_id,
                                body.actual_word_id, body.followed_guidance, body.note,
                                known_words=known,
                                followed_demonstration_form=body.followed_demonstration_form)
    if out["status"] == "REJECTED":
        raise HTTPException(400, {"error": "INVALID_REVIEW", "reason": out["reason"]})
    out["source"] = "participant_self_report"
    out["is_expert_verification"] = False
    out["note_to_display"] = "참가자 본인 확인이며 수어 정확성의 전문가 검증이 아닙니다."
    return out


@app.get(API + "/sessions/{session_id}/review-report")
def review_report(session_id: str):
    """Join the confirmations with the results already written for this session."""
    s = session_or_404(session_id)
    st = S()
    # EVERY trial the session asked for, including cancelled, errored and never-resulted ones.
    # Filtering to trials that produced a result made those vanish from the report entirely, and
    # falling back to a directory scan would have hidden the difference.
    order = list(s["trial_ids"])
    states = {tid: st.trials.get(tid, {}).get("state", "UNKNOWN") for tid in order}
    # The opt-in identity fixed at creation travels with the request, so a trial whose result
    # never landed is still countable and is never reconstructed from whatever is loaded now.
    meta = {tid: {"research_model": st.trials.get(tid, {}).get("research_model")}
            for tid in order if st.trials.get(tid, {}).get("research_model")}
    rep = review_log.report(s["run_dir"], order, states, trial_meta=meta)
    rep["session_id"] = session_id
    rep["latency"] = {"source": "GET /api/v1/... latency is reported by "
                                "wis_lab.service.latency_report over timing_events.jsonl",
                      "metric": "server end-candidate to display ack received",
                      "is_not": "motion end to feedback seen"}
    return rep


@app.post(API + "/replays")
def replay(body: ReplayCreate):
    """Stored input, no camera. Accepts a registered id only, never a filesystem path."""
    st = S()
    clip = next((c for c in st.manifest["clips"]
                 if body.observation_id in (c["clip_id"], c["word_id"])), None)
    if clip is None:
        raise HTTPException(404, {"error": "NO_SUCH_OBSERVATION_ID",
                                  "hint": "use a clip_id or word_id from /references"})
    npz = os.path.join(ROOT, clip["observation_npz"])
    obs = Observation.load(npz)
    t0 = time.perf_counter_ns()
    # request-local: shared by predict and the explicit comparison below, dropped on return
    prep = QueryPreparation(obs)
    result = predict(prep, st.refs, st.contract)
    result["result_id"] = "res_" + uuid.uuid4().hex[:12]
    result["observation_sha256"] = sha256_file(npz)
    result["reference_set_id"] = st.lineage["set_id"]
    result["contract_id"] = st.contract["contract_id"]
    result["extractor_id"] = st.lineage["extractor_id"]
    result["origin"] = "replay"
    # identity metadata of the clip actually read, for the scene-evidence photos; not a score
    result["replay_input_source"] = scene_evidence.source_identity(clip)
    # same observation, same adapter as the live path
    try:
        result["learned_recognition"] = neural.recognise(obs)
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        result["learned_recognition"] = {"schema": neural.ADAPTER_SCHEMA,
                                         "model_status": "ADAPTER_ERROR",
                                         "detail": type(exc).__name__, "ranking": None,
                                         "is_a_verdict": False}
    result["learned_recognition"]["requested_word_coverage"] = neural.word_coverage(
        clip.get("word_id"))
    result["timing"] = {"server_predict_ms": round((time.perf_counter_ns() - t0) / 1e6, 2),
                        "includes_learned_recognition": True}
    result["counts_as_human_attempt"] = False
    fake_trial = {"research_model": _research_choice(body.research_model_key),
                  "requested_word_id": None}
    result["research_model"] = fake_trial["research_model"]
    research = _research_comparison(obs, fake_trial)
    if research is not None:
        result["research_comparison"] = research
    if body.compare_to_word_id:
        result["explicit_reference_comparison"] = _explicit_comparison(
            prep, body.compare_to_word_id, st)
    # a stored replay has no requested word, so the demonstration is whatever was NAMED on screen.
    # The same binding rule applies: unnamed or unprepared is undetermined, never substituted.
    ref_clip = next((r["clip_id"] for r in st.refs
                     if r["word_id"] == body.compare_to_word_id), None)         if body.compare_to_word_id else None
    binding = _reference_binding(st, ref_clip, body.compare_to_word_id)
    binding["bound_at"] = "replay_request"
    binding["chosen_by"] = "explicit_ui_choice_not_the_recogniser"
    erc = result.get("explicit_reference_comparison") or {}
    _split_result(result, {"reference_binding": binding})
    _replay_following(result, binding, erc)
    return result


def _replay_following(result: dict, binding: dict, erc: dict) -> None:
    """Apply the explicit comparison to demonstration_following, under the camera path's rule.

    Two conditions, not one. The comparison must be ABOUT the bound word, and it must rest on an
    observation the predict path accepted. _explicit_comparison only needs sequence_features to
    succeed, so it still returns numbers for a take predict() called INSUFFICIENT - a clip whose
    every reference match fell below the path evidence minimum, for instance. Promoting that to
    AVAILABLE published a comparison the recognition path had already refused, which is exactly
    what _split_result refuses to do on the camera path.
    """
    following = result.get("demonstration_following")
    if following is None:
        return
    if not (binding.get("status") == "BOUND" and erc.get("status") == "OK"
            and erc.get("reference_word_id") == binding.get("demonstration_word_id")):
        return
    if result.get("observation_status") != "OK":
        following.update(
            status="UNDETERMINED_OBSERVATION_INSUFFICIENT",
            observation_status=result.get("observation_status"),
            insufficient_reason=result.get("insufficient_reason"),
            why="the explicit comparison is about the bound word, but the predict path did not "
                "accept this observation. Its numbers are not shown as this demonstration's "
                "feedback; nothing is substituted for them.")
        return
    following.update(
        status="AVAILABLE",
        axis_deviation=erc.get("axis_deviation"),
        joint_deviation=erc.get("joint_deviation"),
        location_path=erc.get("location_path"),
        axis_checks=None,
        axis_checks_note="no band check is computed for a replay reference")


def _explicit_comparison(obs, word_id: str, st) -> dict:
    """Compare a stored clip with a NAMED reference, outside the recogniser.

    This is what the comparison view draws when there is no requested word, as on a replay. The
    word comes from the person looking at the screen, never from the prediction, and predict() has
    already run without it. No participant record is created and nothing counts as an attempt.

    obs may be the request's QueryPreparation, so the query is not prepared a second time.
    """
    ref = next((r for r in st.refs if r["word_id"] == word_id), None)
    if ref is None:
        return {"status": "UNAVAILABLE", "reason": "NO_SUCH_REFERENCE_WORD",
                "reference_word_id": word_id}
    try:
        seq, q = prepared_sample(obs, int(st.contract["sampling_A"]["budget"]))
    except features.FeatureError as exc:
        return {"status": "UNAVAILABLE", "reason": type(exc).__name__,
                "reference_word_id": word_id}
    return {"status": "OK", "reference_word_id": word_id,
            "chosen_by": "explicit_ui_choice_not_the_recogniser",
            "joint_deviation": joint_deviation(q, ref, st.contract),
            "location_path": location_path(q, ref, st.contract,
                                           query_source=source_observation(seq)),
            "axis_deviation": axis_deviation(q, word_id, st.refs, st.contract),
            "counts_as_human_attempt": False,
            "is_a_verdict": False,
            "note": "a difference report against the named reference. It says nothing about "
                    "whether the clip was performed correctly."}


@app.get(API + "/sessions/{session_id}/preview.jpg")
def preview(session_id: str):
    st = S()
    s = session_or_404(session_id)
    if st.camera is None or st.camera_session != session_id:
        raise HTTPException(409, {"error": "CAMERA_NOT_OWNED_BY_SESSION"})
    touch(s, None)
    jpg, frame_id = st.camera.preview_jpeg()
    if jpg is None:
        raise HTTPException(503, {"error": "NO_PREVIEW_YET"})
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"X-Frame-Id": str(frame_id), "Cache-Control": "no-store"})


@app.post(API + "/sessions/{session_id}/close")
def close_session(session_id: str):
    """Releases the camera immediately rather than waiting for the lease to lapse."""
    st = S()
    s = session_or_404(session_id)
    with st.lock:
        closed = []
        for tid in s["trial_ids"]:
            t = st.trials[tid]
            if t["state"] not in ("FEEDBACK",) + NON_SUCCESS_STATES:
                t["state"] = "CANCELLED"
                t["error"] = "SESSION_CLOSED"
                ok, err = _closed_event(s, t, "CANCELLED", "SESSION_CLOSED")
                closed.append({"trial_id": tid, "telemetry_persisted": ok,
                               "telemetry_error": err})
        if st.camera_session == session_id:
            st.release_camera("session closed")
        s["state"] = "CLOSED"
        s["closed_reason"] = s.get("closed_reason") or "CLOSED_BY_CLIENT"
        s["camera"] = False
    write_json(os.path.join(s["run_dir"], "session.json"),
               {k: v for k, v in s.items() if k not in ("request_ids",)})
    return {"session_id": session_id, "state": "CLOSED", "camera_released": True,
            "cancelled_trials": closed,
            "telemetry_unpersisted": sum(1 for c in closed if not c["telemetry_persisted"])}


@app.get(API + "/devices")
def devices():
    from .camera import list_devices
    if S().camera is not None:
        raise HTTPException(409, {"error": "CAMERA_IN_USE",
                                  "hint": "close the session before probing devices"})
    return {"devices": list_devices()}


@app.exception_handler(HTTPException)
def http_exc(_request, exc: HTTPException):
    detail = exc.detail if isinstance(exc.detail, dict) else {"error": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content=detail)


WEB_DIR = os.path.join(ROOT, "web")
if os.path.isdir(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
