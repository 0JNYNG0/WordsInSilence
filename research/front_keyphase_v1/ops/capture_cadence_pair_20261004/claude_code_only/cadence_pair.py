"""Capture cadence pair: A full-PTS extraction then even IDs vs B even-RGB-only extraction.

Fixed public-video input diagnostic (FROZEN_PLAN.md, 2026-10-04 KST). Not a learner, not a production
feature, not a camera/web reproduction. The only difference between the arms is whether MediaPipe saw
the odd frames in between:

  A  every decoded frame goes through Landmarker.process; rows with even frame_id are kept.
  B  every frame is decoded, only even frame_id RGB goes through Landmarker.process.

Both arms keep the original frame_id and the real PTS microseconds (video_source.py semantics). Each
clip/arm gets a new Landmarker. Selected frame_id / timestamp_us / RGB hash must match across arms.

Default invocation (no --run) only validates the config and re-hashes the pins. It never constructs a
decoder, landmarker or model and never imports av / mediapipe / torch. --run additionally requires the
synthetic TEST_RECEIPT.json for the current code, a COMPLETE teacher-unit CODEX_CLOSEOUT.json, a
time before 08:00 KST and a fresh result directory. Nothing happens at import.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable

import numpy as np

UNIT = Path(__file__).resolve().parent
FRONT = Path(__file__).resolve().parents[2]
WORKSPACE = FRONT.parents[1]

CONFIG_SCHEMA = "wis_capture_cadence_pair_config_v1"
RESULT_SCHEMA = "wis_capture_cadence_pair_result_v1"
CONFIG_NAME = "PREPARED_CONFIG.json"
RECEIPT_NAME = "TEST_RECEIPT.json"
RECEIPT_SCHEMA = "wis_capture_cadence_pair_test_receipt_v1"
CLOSEOUT_NAME = "CODEX_CLOSEOUT.json"
OWN_CODE_FILES = ("cadence_pair.py", "test_cadence.py")

ARM_A = "A_full_then_even"
ARM_B = "B_even_rgb_only"
STRIDE, PHASE = 2, 0
TCN_SHA256 = "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91"
N_CLASSES = 93
N_CHANNELS = 280
CPU_THREADS = 2
KST = timezone(timedelta(hours=9))
DEADLINE_KST = datetime(2026, 10, 4, 8, 0, 0, tzinfo=KST)

# Frozen plan table: (clip_id, path relative to FRONT, prior frame count, (label, start_us, end_us_exclusive) | None)
FROZEN_CLIPS = (
    ("NIA_SL_SEN0408_REAL01_F", "ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL01_F.mp4",
     169, ("나", 3_334_000, 4_585_000)),
    ("NIA_SL_SEN0408_REAL02_F", "ops/known_signer_unseen_clip_pair_20260925/raw/NIA_SL_SEN0408_REAL02_F.mp4",
     166, ("나", 3_887_000, 4_840_000)),
    ("NIA_SL_SEN0356_REAL01_F", "runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL01_F.mp4",
     131, ("죄송하다", 1_588_000, 2_940_000)),
    ("NIA_SL_SEN0356_REAL02_F", "runs/apology_source_review_v1/raw/NIA_SL_SEN0356_REAL02_F.mp4",
     118, ("죄송하다", 1_821_000, 3_001_000)),
    ("NIA_SL_SEN0254_REAL01_F", "runs/greeting_source_review_v1/raw/NIA_SL_SEN0254_REAL01_F.mp4",
     138, None),
    ("NIA_SL_SEN0354_REAL01_F", "runs/greeting_source_review_v1/raw/NIA_SL_SEN0354_REAL01_F.mp4",
     131, None),
)
TOTAL_FRAMES = 853
TOTAL_SELECTED = 428
TOTAL_PROCESS_CALLS = 1450          # A full 853 + B selected 428 + first A repeat 169
TOTAL_DECODED = 1875                # A 853 + B 853 (B decodes every frame) + A repeat 169
TOTAL_LANDMARKERS = 13              # 6 A + 6 B + 1 A repeat
TOTAL_FORWARDS = 8

OBS_FIELDS = ("xy_px", "timestamp_us", "frame_id", "observed", "usable", "conf", "hand_score")
DECODE_FIELDS = ("frame_id", "timestamp_us", "pts", "rgb_sha256", "rgb_shape")
GROUPS = (("pose", 0, 33), ("left_hand", 33, 54), ("right_hand", 54, 75))
HANDS = GROUPS[1:]
CODE_ROLES = ("landmarker_source", "observation_source", "input_preprocessing_source",
              "point_mapping_source", "tcn_model_source")
ALLOWED_PROJECT_MODULES = {
    "wis_lab", "wis_lab.core", "wis_lab.core.observation",
    "wis_lab.capture", "wis_lab.capture.landmarker", "wis_lab.capture.video_source",
    "pilot", "pilot.data", "pilot.web_input", "pilot.model_masked_v1",
}

LIMITS = (
    "6 public known-signer clips (AIHub front video, signers seen in training)",
    "4 scored windows, 2 words (나, 죄송하다)",
    "no anatomical coordinate ground truth",
    "not a webcam / resource / latency / camera-speed reproduction",
    "not an independent new-person evaluation",
    "repeatedly used development material",
)
CAVEATS = (
    "Mask differences are directional counts between arms, not counts of truly lost normal hands or "
    "left/right errors.",
    "Coordinate distances compare the same slot index; the same slot is not guaranteed to be the same "
    "physical hand in both arms.",
    "Scores include the interaction of raw coordinate changes with per-window normalisation "
    "(body_scale / hand_scales recomputed per arm).",
    "Selected cadence is reported from measured PTS gaps; it is not called 15 fps unless measured so.",
)


# ---------------------------------------------------------------- errors
class CadenceError(Exception):
    kind = "TECHNICAL_FAILURE"


class ConfigError(CadenceError):
    kind = "CONFIG"


class PinError(CadenceError):
    kind = "PIN_MISMATCH"


class GateError(CadenceError):
    kind = "RUN_GATE"


class DecodeContractError(CadenceError):
    kind = "DECODE_CONTRACT"


class SelectionMismatch(CadenceError):
    kind = "SELECTION_MISMATCH"


class NondeterminismError(CadenceError):
    kind = "NONDETERMINISTIC"


class LandmarkerNotFresh(CadenceError):
    kind = "LANDMARKER_NOT_FRESH"


class PreprocessingError(CadenceError):
    kind = "PREPROCESSING"


class ScoringError(CadenceError):
    kind = "SCORING"


class ModelStateChanged(CadenceError):
    kind = "MODEL_STATE_CHANGED"


class CountMismatch(CadenceError):
    kind = "COUNT_MISMATCH"


# ---------------------------------------------------------------- small helpers
@dataclass(frozen=True)
class Roots:
    workspace: Path
    front: Path
    unit: Path

    @property
    def front_src(self) -> Path:
        return self.front / "src"

    @property
    def pilot_src(self) -> Path:
        return self.workspace / "research" / "sign_model_pilot_v1" / "src"

    @property
    def result(self) -> Path:
        return self.unit / "result"


DEFAULT_ROOTS = Roots(WORKSPACE, FRONT, UNIT)


def role_paths(roots: Roots) -> dict:
    """Where each pinned role must live. The Landmarker defaults resolve to the same model files."""
    pilot = roots.pilot_src / "pilot"
    return {
        "mediapipe_hand_model": roots.workspace / "Tools" / "models" / "hand_landmarker.task",
        "mediapipe_pose_model": roots.workspace / "Tools" / "models" / "pose_landmarker_lite.task",
        "frozen_tcn_checkpoint": roots.front / "runs" / "partial_encoder_pair_v1" / "checkpoint_control_final.pt",
        "observation_extraction_contract": roots.front / "config" / "contract.json",
        "landmarker_source": roots.front_src / "wis_lab" / "capture" / "landmarker.py",
        "observation_source": roots.front_src / "wis_lab" / "core" / "observation.py",
        "input_preprocessing_source": pilot / "data.py",
        "point_mapping_source": pilot / "web_input.py",
        "tcn_model_source": pilot / "model_masked_v1.py",
        "this_frozen_plan": roots.unit / "FROZEN_PLAN.md",
    }


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def own_code_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {name: sha256_file(here / name) for name in OWN_CODE_FILES}


def _norm(p) -> str:
    return os.path.normcase(os.path.normpath(os.fspath(p)))


def _same(a, b) -> bool:
    return a is not None and b is not None and _norm(a) == _norm(b)


def _is_under(p, root) -> bool:
    pn, rn = _norm(p), _norm(root).rstrip("\\/")
    return pn == rn or pn.startswith(rn + os.sep)


def _is_sha(s) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _req(cond, msg: str, exc=ConfigError) -> None:
    if not cond:
        raise exc(msg)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, Fraction):
        return f"{o.numerator}/{o.denominator}"
    return o


def array_sha256(a: np.ndarray) -> str:
    """Hash of dtype, shape and C-order bytes."""
    c = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{c.dtype.str}|{'x'.join(str(s) for s in c.shape)}|".encode("ascii"))
    h.update(c.tobytes())
    return h.hexdigest()


def rgb_digest(rgb) -> str:
    """Selected-RGB identity: uint8 HxWx3 only; dtype and shape are part of the hash."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DecodeContractError(
            f"decoded frame is not uint8 HxWx3 RGB: {type(rgb).__name__} "
            f"{getattr(rgb, 'dtype', None)} {getattr(rgb, 'shape', None)}")
    return array_sha256(rgb)


# ---------------------------------------------------------------- paths / config / pins
def check_local_path(p, root: Path, suffix: str | None = None) -> Path:
    """Absolute local regular file under root. No URL, device index, UNC/device namespace, '..' or redirect."""
    _req(isinstance(p, str) and p and p == p.strip(), f"path must be a non-empty string: {p!r}")
    low = p.lower()
    _req("://" not in p, f"URL refused: {p}")
    _req(not low.startswith(("http:", "https:", "rtsp:", "rtmp:", "udp:", "tcp:", "file:", "video=", "/dev/")),
         f"URL/device refused: {p}")
    _req(not p.startswith(("\\\\", "//")), f"UNC/device namespace refused: {p}")
    _req(not p.isdigit(), f"device index refused: {p}")
    path = Path(p)
    _req(path.is_absolute(), f"path must be absolute: {p}")
    _req(".." not in path.parts, f"'..' refused: {p}")
    if suffix is not None:
        _req(path.suffix == suffix, f"expected {suffix} file: {p}")
    _req(_is_under(path, root), f"path outside {root}: {p}")
    _req(path.is_file(), f"not an existing regular file: {p}")
    _req(_same(path.resolve(strict=True), path), f"path redirects elsewhere (link/alias): {p}")
    return path


def validate_config(cfg, roots: Roots) -> dict:
    """Exact schema + frozen plan check. Returns the normalised plan. Reads no media."""
    _req(isinstance(cfg, dict), "config is not an object")
    _req(cfg.get("schema") == CONFIG_SCHEMA, f"schema must be {CONFIG_SCHEMA}")
    status = str(cfg.get("status", "")).upper()
    _req("EXAMPLE" not in status and "NOT_REAL" not in status, "schema example config refused")
    _req("<WORKSPACE>" not in json.dumps(cfg, ensure_ascii=False), "placeholder paths refused")
    _req(_same(cfg.get("workspace_root"), roots.workspace), "workspace_root mismatch")
    _req(_same(cfg.get("front_root"), roots.front), "front_root mismatch")
    _req(_same(cfg.get("unit_dir"), roots.unit), "unit_dir mismatch")
    _req(_same(cfg.get("output_dir"), roots.result), "output_dir must be exactly <unit>/result")

    clips_cfg = cfg.get("clips")
    _req(isinstance(clips_cfg, list) and len(clips_cfg) == len(FROZEN_CLIPS), "exactly 6 clips required")
    clips = []
    for (cid, rel, n, win), c in zip(FROZEN_CLIPS, clips_cfg):
        _req(isinstance(c, dict) and c.get("clip_id") == cid, f"clip order/id mismatch at {cid}")
        path = check_local_path(c.get("path"), roots.workspace, ".mp4")
        _req(_same(path, roots.front / rel), f"{cid}: path differs from frozen plan")
        _req(path.stem == cid, f"{cid}: file stem mismatch")
        _req(_is_sha(c.get("sha256")), f"{cid}: sha256 missing")
        _req(_is_int(c.get("expected_frames_from_prior")) and c["expected_frames_from_prior"] == n,
             f"{cid}: expected_frames_from_prior != {n}")
        _req(_is_int(c.get("expected_selected_even_frames"))
             and c["expected_selected_even_frames"] == (n + 1) // 2,
             f"{cid}: expected_selected_even_frames != {(n + 1) // 2}")
        score = c.get("score")
        if win is None:
            _req(score is None, f"{cid}: must not be scored")
        else:
            _req(isinstance(score, dict) and score.get("label") == win[0]
                 and _is_int(score.get("start_us")) and score["start_us"] == win[1]
                 and _is_int(score.get("end_us_exclusive")) and score["end_us_exclusive"] == win[2],
                 f"{cid}: score window/label differs from frozen plan")
            score = {"label": win[0], "start_us": win[1], "end_us_exclusive": win[2]}
        clips.append({"clip_id": cid, "path": path, "sha256": c["sha256"], "n_frames": n,
                      "n_selected": (n + 1) // 2, "score": score})
    _req(sum(c["n_frames"] for c in clips) == TOTAL_FRAMES, "total frames != 853")
    _req(sum(c["n_selected"] for c in clips) == TOTAL_SELECTED, "total selected != 428")
    _req(sum(c["score"] is not None for c in clips) == 4, "exactly 4 scored clips required")

    ext = cfg.get("extraction") or {}
    _req(ext.get("arms") == [ARM_A, ARM_B], "extraction.arms mismatch")
    _req(ext.get("stride") == STRIDE and ext.get("phase") == PHASE, "stride/phase must be 2/0")
    _req(ext.get("running_mode") == "VIDEO", "running_mode must be VIDEO")
    for k in ("reject_missing_pts", "reject_nonincreasing_pts", "expected_counts_required"):
        _req(ext.get(k) is True, f"extraction.{k} must be true")
    _req(ext.get("estimated_total_process_calls") == TOTAL_PROCESS_CALLS, "estimated process calls != 1450")
    det = cfg.get("determinism") or {}
    _req(det.get("clip_id") == FROZEN_CLIPS[0][0] and det.get("repeat_arm") == ARM_A
         and det.get("extra_runs") == 1 and det.get("arrays") == list(OBS_FIELDS),
         "determinism block differs from frozen plan")
    sc = cfg.get("scoring") or {}
    _req(sc.get("checkpoint_sha256") == TCN_SHA256, "checkpoint sha differs from frozen TCN")
    _req(sc.get("model_class") == "pilot.model_masked_v1.MaskedSignTCN", "model_class mismatch")
    _req(sc.get("state_dict_load") == "strict", "state_dict_load must be strict")
    _req(sc.get("n_classes") == N_CLASSES and sc.get("forward_calls") == TOTAL_FORWARDS
         and sc.get("scored_clips") == 4 and sc.get("cpu_threads") == CPU_THREADS, "scoring counts mismatch")
    for k in ("active_span", "handgap", "timegrid", "interpolation"):
        _req(sc.get(k) is False, f"scoring.{k} must be false")
    iso = cfg.get("isolation") or {}
    allowed = iso.get("allowed_video_paths")
    _req(isinstance(allowed, list) and len(allowed) == 6
         and {_norm(a) for a in allowed} == {_norm(c["path"]) for c in clips},
         "isolation.allowed_video_paths must be exactly the 6 clips")
    for k in ("camera", "network", "downloads", "training", "device_enumeration", "user_data",
              "output_overwrite", "source_mutation", "execute_default"):
        _req(iso.get(k) is False, f"isolation.{k} must be false")
    _req(iso.get("explicit_run_flag_required") is True, "explicit_run_flag_required must be true")

    pins = cfg.get("pins")
    inventory = cfg.get("pin_inventory", cfg.get("input_files"))
    _req(isinstance(pins, dict) and pins, "pins missing")
    _req(isinstance(inventory, list) and len(inventory) == len(pins), "pin_inventory must list every pin once")
    pin_map = {}
    for p, sha in pins.items():
        path = check_local_path(p, roots.workspace)
        _req(_is_sha(sha), f"bad pin sha for {p}")
        key = _norm(path)
        _req(key not in pin_map, f"duplicate pin {p}")
        pin_map[key] = {"path": path, "sha256": sha, "role": None}
    for item in inventory:
        _req(isinstance(item, dict) and isinstance(item.get("role"), str), "bad pin_inventory entry")
        key = _norm(check_local_path(item.get("path"), roots.workspace))
        _req(key in pin_map and pin_map[key]["sha256"] == item.get("sha256"),
             f"pin_inventory/pins disagree for {item.get('path')}")
        _req(pin_map[key]["role"] is None, f"pin listed twice in inventory: {item.get('path')}")
        pin_map[key]["role"] = item["role"]
    by_role: dict = {}
    for ent in pin_map.values():
        by_role.setdefault(ent["role"], []).append(ent)
    _req(len(by_role.get("public_source_video", [])) == 6, "exactly 6 public_source_video pins required")
    rp = role_paths(roots)
    for role, expected in rp.items():
        ents = by_role.get(role, [])
        _req(len(ents) == 1 and _same(ents[0]["path"], expected), f"role {role} must be pinned once at {expected}")
    for c in clips:
        ent = pin_map.get(_norm(c["path"]))
        _req(ent is not None and ent["role"] == "public_source_video" and ent["sha256"] == c["sha256"],
             f"{c['clip_id']}: clip sha / pin disagree")
    _req(_same(ext.get("hand_model_path"), rp["mediapipe_hand_model"]), "hand_model_path != pinned role")
    _req(_same(ext.get("pose_model_path"), rp["mediapipe_pose_model"]), "pose_model_path != pinned role")
    _req(_same(ext.get("contract_path"), rp["observation_extraction_contract"]), "contract_path != pinned role")
    _req(_same(sc.get("checkpoint_path"), rp["frozen_tcn_checkpoint"]), "checkpoint_path != pinned role")
    _req(pin_map[_norm(rp["frozen_tcn_checkpoint"])]["sha256"] == TCN_SHA256, "checkpoint pin != frozen TCN")

    limits = list((cfg.get("decision") or {}).get("limits") or [])
    return {"clips": clips, "pins": pin_map, "role_paths": rp, "config_limits": limits}


def hash_pins(plan: dict, hash_file) -> tuple:
    actual, mismatches = {}, []
    for ent in plan["pins"].values():
        try:
            got = hash_file(ent["path"])
        except OSError as exc:
            got = f"ERROR {type(exc).__name__}: {exc}"
        actual[str(ent["path"])] = got
        if got != ent["sha256"]:
            mismatches.append({"path": str(ent["path"]), "role": ent["role"], "pinned": ent["sha256"], "actual": got})
    return actual, mismatches


def verify_pins(plan: dict, hash_file) -> dict:
    actual, mismatches = hash_pins(plan, hash_file)
    if mismatches:
        raise PinError(json.dumps(mismatches, ensure_ascii=False))
    return actual


# ---------------------------------------------------------------- run gates
def check_receipt(roots: Roots) -> dict:
    p = roots.unit / RECEIPT_NAME
    _req(p.is_file(), f"{RECEIPT_NAME} missing; run test_cadence.py first", GateError)
    data = json.loads(p.read_text(encoding="utf-8"))
    _req(data.get("schema") == RECEIPT_SCHEMA and data.get("status") == "PASS", "test receipt is not PASS", GateError)
    _req(_is_int(data.get("num_tests")) and data["num_tests"] > 0
         and data.get("failures") == 0 and data.get("errors") == 0, "test receipt counts invalid", GateError)
    current = own_code_hashes()
    _req(data.get("code_sha256") == current, "test receipt was produced for different code", GateError)
    return {"path": str(p), "sha256": sha256_file(p), "num_tests": data["num_tests"], "code_sha256": current}


def check_deadline(now: Callable[[], datetime]) -> dict:
    t = now()
    _req(t.tzinfo is not None, "clock must be timezone-aware", GateError)
    tk = t.astimezone(KST)
    _req(tk < DEADLINE_KST, f"no new experiment at/after {DEADLINE_KST.isoformat()} (now {tk.isoformat()})", GateError)
    return {"now_kst": tk.isoformat(), "deadline_kst": DEADLINE_KST.isoformat()}


CLOSED_STATUSES = ("COMPLETE", "COMPLETED")
UNFINISHED_TOKENS = ("PENDING", "RUNNING", "WAITING")
CLOSEOUT_DETAIL_KEYS = ("verdict", "decision", "result", "error", "errors", "reason", "stop_reason", "message")


def closeout_status_closed(status) -> bool:
    """COMPLETE/COMPLETED exactly or as a COMPLETE_/COMPLETED_ token prefix with no unfinished token.

    Closed means the unit finished, not that its method succeeded (e.g. a technical preflight stop).
    INCOMPLETE / NOT_COMPLETE / COMPLETE_PENDING are refused.
    """
    if not isinstance(status, str):
        return False
    s = status.strip().upper()
    if s in CLOSED_STATUSES:
        return True
    if not s.startswith(tuple(c + "_" for c in CLOSED_STATUSES)):
        return False
    return "IN_PROGRESS" not in s and not any(t in UNFINISHED_TOKENS for t in s.split("_"))


def check_closeout(path_str, roots: Roots, hash_file) -> dict:
    _req(isinstance(path_str, str) and path_str, "--teacher-closeout is required for --run", GateError)
    p = Path(path_str)
    _req(p.is_absolute() and ".." not in p.parts and p.name == CLOSEOUT_NAME,
         f"teacher closeout must be an absolute path to {CLOSEOUT_NAME}", GateError)
    _req(_is_under(p, roots.front) and not _is_under(p, roots.unit),
         "teacher closeout must be another unit under FRONT", GateError)
    _req(p.is_file(), f"teacher closeout missing: {p}", GateError)
    data = json.loads(p.read_text(encoding="utf-8"))
    status = data.get("status") if isinstance(data, dict) else None
    _req(closeout_status_closed(status), f"teacher unit not closed: status={status!r}", GateError)
    return {"path": str(p), "sha256": hash_file(p), "status": status,
            "verdict": data.get("verdict", data.get("decision")),
            "details": {k: data[k] for k in CLOSEOUT_DETAIL_KEYS if k in data},
            "closed_means": "teacher unit finished; not a performance or method PASS"}


# ---------------------------------------------------------------- decoding
def strict_frames(raw, time_base):
    """(pts, rgb) stream -> (frame_id, pts, timestamp_us, rgb).

    Same clock as video_source.extract: int(round(float((pts - origin_pts) * time_base) * 1e6)), first PTS
    is the origin, first frame_id is 0. Unlike video_source a missing or non-increasing PTS is not skipped:
    it aborts, so the decoded ordinal is the frame_id.
    """
    tb = Fraction(time_base)
    if tb <= 0:
        raise DecodeContractError(f"invalid time_base {time_base}")
    origin = last_pts = last_us = None
    for fid, (pts, rgb) in enumerate(raw):
        if pts is None:
            raise DecodeContractError(f"missing PTS at decoded ordinal {fid}")
        if isinstance(pts, bool) or not isinstance(pts, (int, np.integer)):
            raise DecodeContractError(f"non-integer PTS {pts!r} at ordinal {fid}")
        pts = int(pts)
        if origin is None:
            origin = pts
        ts_us = int(round(float((pts - origin) * tb) * 1_000_000))
        if last_pts is not None and (pts <= last_pts or ts_us <= last_us):
            raise DecodeContractError(
                f"non-increasing PTS at ordinal {fid}: pts {last_pts}->{pts}, us {last_us}->{ts_us}")
        last_pts, last_us = pts, ts_us
        yield fid, pts, ts_us, rgb


class PyAVDecoder:
    """One local file, one frame at a time. Nothing cached; RGB leaves with each yielded frame."""

    def __init__(self, av_module, path: str):
        self._av, self.path = av_module, path
        self._container = self._stream = None
        self.time_base = None
        self.meta: dict = {}

    def __enter__(self):
        self._container = self._av.open(self.path, mode="r")
        try:
            videos = list(self._container.streams.video)
            if not videos:
                raise DecodeContractError(f"no video stream in {self.path}")
            stream = videos[0]
            if stream.time_base is None:
                raise DecodeContractError("stream time_base missing")
            self._stream = stream
            self.time_base = Fraction(stream.time_base.numerator, stream.time_base.denominator)
            rate = stream.average_rate
            cc = stream.codec_context
            self.meta = {
                "n_video_streams": len(videos),
                "time_base": f"{self.time_base.numerator}/{self.time_base.denominator}",
                "average_rate": f"{rate.numerator}/{rate.denominator}" if rate else None,
                "average_rate_hz": float(rate) if rate else None,
                "codec": cc.name, "pix_fmt": cc.pix_fmt,
                "width": int(cc.width), "height": int(cc.height),
                "declared_frames": int(stream.frames or 0),
            }
        except BaseException:
            self._container.close()
            self._container = None
            raise
        return self

    def raw(self):
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                yield frame.pts, frame.to_ndarray(format="rgb24")

    def __exit__(self, *exc):
        if self._container is not None:
            self._container.close()
            self._container = None
        return False


# ---------------------------------------------------------------- runtime dependencies
@dataclass
class Deps:
    open_decoder: Callable[[str], Any]
    landmarker_factory: Callable[[], Any]
    empty_frames: Callable[[int], dict]
    observation_cls: Callable[..., Any]
    to_native_like: Callable
    build_channels: Callable
    collate: Callable
    load_model: Callable[[], tuple]
    forward: Callable
    state_hash: Callable[[Any], str]
    n_channels: int
    versions: dict
    import_audit: dict


def _torch_state_sha(state: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(f"|{t.dtype}|{tuple(t.shape)}|".encode("ascii"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def validate_classes(classes, n_classes) -> None:
    _req(isinstance(classes, list) and all(isinstance(c, str) for c in classes), "classes must be strings",
         ScoringError)
    _req(len(classes) == N_CLASSES and n_classes == N_CLASSES and len(set(classes)) == N_CLASSES,
         f"classes must be {N_CLASSES} unique and match n_classes", ScoringError)


def load_runtime_deps(plan: dict, roots: Roots, hash_file) -> Deps:
    """Run mode only: pin-check sources, then import av / mediapipe / torch and the pinned project code."""
    rp = plan["role_paths"]
    for role in CODE_ROLES + ("observation_extraction_contract", "mediapipe_hand_model",
                              "mediapipe_pose_model", "frozen_tcn_checkpoint"):
        pinned = plan["pins"][_norm(rp[role])]["sha256"]
        _req(hash_file(rp[role]) == pinned, f"{role} changed before import", PinError)
    preloaded = sorted(m for m in sys.modules if m.split(".")[0] in ("wis_lab", "pilot"))
    _req(not preloaded, f"project modules already imported from elsewhere: {preloaded}", GateError)
    added = []
    for p in (str(roots.pilot_src), str(roots.front_src)):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)

    import av                                               # noqa: E402  run mode only
    import mediapipe                                        # noqa: E402
    import torch                                            # noqa: E402
    from wis_lab.core import observation as obs_mod          # noqa: E402
    from wis_lab.capture import landmarker as lm_mod         # noqa: E402
    from pilot import data as data_mod                       # noqa: E402
    from pilot import web_input as wi_mod                    # noqa: E402
    from pilot import model_masked_v1 as model_mod           # noqa: E402

    files = {}
    for mod, role in ((lm_mod, "landmarker_source"), (obs_mod, "observation_source"),
                      (data_mod, "input_preprocessing_source"), (wi_mod, "point_mapping_source"),
                      (model_mod, "tcn_model_source")):
        _req(_same(Path(mod.__file__).resolve(), rp[role]), f"{mod.__name__} imported from {mod.__file__}",
             GateError)
        files[mod.__name__] = mod.__file__
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("wis_lab", "pilot"))
    _req(not [m for m in loaded if "camera" in m.lower()], f"camera module imported: {loaded}", GateError)
    _req(data_mod.TOTAL_CHANNELS == N_CHANNELS, "pilot.data.TOTAL_CHANNELS != 280", GateError)

    contract_bytes = Path(rp["observation_extraction_contract"]).read_bytes()
    _req(hashlib.sha256(contract_bytes).hexdigest()
         == plan["pins"][_norm(rp["observation_extraction_contract"])]["sha256"], "contract changed", PinError)
    contract = json.loads(contract_bytes.decode("utf-8"))
    hand, pose = str(rp["mediapipe_hand_model"]), str(rp["mediapipe_pose_model"])

    def landmarker_factory():
        lm = lm_mod.Landmarker(contract, hand_model=hand, pose_model=pose)
        if not (_same(lm.hand_model_path, hand) and _same(lm.pose_model_path, pose)):
            lm.close()
            raise GateError("Landmarker resolved model paths differ from pinned task files")
        return lm

    ckpt = Path(rp["frozen_tcn_checkpoint"])

    def load_model():
        blob = ckpt.read_bytes()
        _req(hashlib.sha256(blob).hexdigest() == TCN_SHA256, "checkpoint bytes != frozen TCN", PinError)
        torch.set_num_threads(CPU_THREADS)
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        for k in ("width", "dilations", "kernel_size", "dropout", "n_classes", "classes", "state_dict"):
            _req(k in payload, f"payload missing {k}", ScoringError)
        classes = list(payload["classes"])
        validate_classes(classes, payload["n_classes"])
        # same path as tools_prep_union93_candidate.rebuild_from_payload, strict made explicit
        model = model_mod.MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                                        kernel_size=payload["kernel_size"], dropout=payload["dropout"],
                                        n_classes=payload["n_classes"])
        res = model.load_state_dict(payload["state_dict"], strict=True)
        _req(not list(res.missing_keys) and not list(res.unexpected_keys), "state_dict not exact", ScoringError)
        model.eval()
        payload_sha = _torch_state_sha(payload["state_dict"])
        _req(_torch_state_sha(model.state_dict()) == payload_sha, "loaded state != payload state", ScoringError)
        info = {"checkpoint_path": str(ckpt), "checkpoint_sha256": TCN_SHA256,
                "class": "pilot.model_masked_v1.MaskedSignTCN",
                "constructor": {k: payload[k] for k in ("width", "dilations", "kernel_size", "dropout", "n_classes")},
                "parameter_count": int(model.parameter_count()), "payload_state_sha256": payload_sha,
                "torch_threads": int(torch.get_num_threads())}
        return model, classes, info

    def forward(model, x, pad):
        _req(not model.training, "model must be in eval mode", ScoringError)
        with torch.inference_mode():
            out = model(torch.from_numpy(x), torch.from_numpy(pad))
        return out[0].detach().cpu().numpy().astype(np.float64)

    return Deps(
        open_decoder=lambda path: PyAVDecoder(av, path),
        landmarker_factory=landmarker_factory,
        empty_frames=obs_mod.empty_frames,
        observation_cls=obs_mod.Observation,
        to_native_like=wi_mod.to_native_like,
        build_channels=data_mod.build_channels,
        collate=data_mod.collate,
        load_model=load_model,
        forward=forward,
        state_hash=lambda model: _torch_state_sha(model.state_dict()),
        n_channels=N_CHANNELS,
        versions={"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                  "av": getattr(av, "__version__", None), "mediapipe": getattr(mediapipe, "__version__", None),
                  "torch": torch.__version__},
        import_audit={"sys_path_added": added, "module_files": files, "project_modules_loaded": loaded,
                      "extractor_code_version": getattr(lm_mod, "EXTRACTOR_CODE_VERSION", None),
                      "landmarker_default_hand_equals_pin": _same(getattr(lm_mod, "DEFAULT_HAND_MODEL", None), hand),
                      "landmarker_default_pose_equals_pin": _same(getattr(lm_mod, "DEFAULT_POSE_MODEL", None), pose),
                      "contract_sha256": hashlib.sha256(contract_bytes).hexdigest()},
    )


# ---------------------------------------------------------------- extraction
def _check_fresh(lm, registry: list) -> None:
    if any(lm is x for x in registry):
        raise LandmarkerNotFresh("landmarker instance reused across clip/arm")
    if getattr(lm, "_last_ms", None) != -1:
        raise LandmarkerNotFresh("landmarker has processed frames before this clip/arm")
    for k in ("handedness_disagreements", "handedness_comparisons", "unassigned_hands"):
        if getattr(lm, k, 0) != 0:
            raise LandmarkerNotFresh(f"landmarker counter {k} not zero")
    registry.append(lm)


def interval_stats(ts: np.ndarray):
    if ts.size < 2:
        return None
    d = np.diff(ts.astype(np.int64))
    med = float(np.median(d))
    uniq = np.unique(d)
    return {"gaps": int(d.size), "min_us": int(d.min()), "median_us": med, "max_us": int(d.max()),
            "n_unique_gaps": int(uniq.size), "unique_gaps_us_first10": uniq[:10].tolist(),
            "median_rate_hz": (1e6 / med) if med > 0 else None}


def extract_arm(*, path: str, clip_id: str, arm: str, n_frames: int, n_selected: int, deps: Deps,
                counters: dict, registry: list) -> dict:
    """Decode every frame of one clip; A processes all, B only even frame_id. Fresh Landmarker."""
    if arm not in (ARM_A, ARM_B):
        raise ValueError(arm)
    n_rows = n_frames if arm == ARM_A else n_selected
    out = deps.empty_frames(n_rows)
    dec_fid = np.full(n_frames, -1, dtype=np.int64)
    dec_ts = np.full(n_frames, -1, dtype=np.int64)
    dec_pts = np.zeros(n_frames, dtype=np.int64)
    dec_sha = np.zeros(n_frames, dtype="<U64")
    dec_shape = np.zeros((n_frames, 3), dtype=np.int64)
    processed = np.zeros(n_frames, dtype=np.bool_)
    decoded = process_calls = ms_clamps = 0
    last_ms = -1
    lm = deps.landmarker_factory()
    counters["landmarker_constructed"] += 1
    try:
        _check_fresh(lm, registry)
        with deps.open_decoder(path) as dec:
            counters["decoder_opened"] += 1
            meta = dict(dec.meta)
            time_base = Fraction(dec.time_base)
            for fid, pts, ts_us, rgb in strict_frames(dec.raw(), time_base):
                if fid >= n_frames:
                    raise DecodeContractError(f"{clip_id}: more than the expected {n_frames} frames")
                digest = rgb_digest(rgb)
                decoded += 1
                counters["decoded_frames"] += 1
                dec_fid[fid], dec_ts[fid], dec_pts[fid] = fid, ts_us, pts
                dec_sha[fid], dec_shape[fid] = digest, rgb.shape
                if arm == ARM_A:
                    row = fid
                elif fid % STRIDE == PHASE:
                    row = fid // STRIDE
                else:
                    continue                                      # B: decoded, never shown to MediaPipe
                out["timestamp_us"][row] = ts_us
                out["frame_id"][row] = fid
                ms = ts_us // 1000                                # mirrors Landmarker.next_timestamp_ms
                if ms <= last_ms:
                    ms_clamps += 1
                    ms = last_ms + 1
                last_ms = ms
                counters["landmarker_process_calls"] += 1
                process_calls += 1
                lm.process(rgb, ts_us, out, row)
                processed[fid] = True
        if decoded != n_frames:
            raise DecodeContractError(f"{clip_id}: decoded {decoded} frames, expected {n_frames}")
        lm_diag = {"extractor_id": lm.extractor_id(),
                   "handedness_label_disagreements": getattr(lm, "handedness_disagreements", None),
                   "handedness_label_comparisons": getattr(lm, "handedness_comparisons", None),
                   "unassigned_hand_detections": getattr(lm, "unassigned_hands", None),
                   "mediapipe_ms_clamps": ms_clamps}
    finally:
        try:
            lm.close()
        finally:
            counters["landmarker_closed"] += 1

    if not np.array_equal(dec_fid, np.arange(n_frames, dtype=np.int64)):
        raise DecodeContractError(f"{clip_id}: decoded frame ids are not 0..N-1")
    even = np.arange(PHASE, n_frames, STRIDE, dtype=np.int64)
    expected_processed = np.ones(n_frames, bool) if arm == ARM_A else np.isin(np.arange(n_frames), even)
    if not np.array_equal(processed, expected_processed):
        raise DecodeContractError(f"{clip_id}/{arm}: processed frames differ from arm rule")
    stored = {k: out[k] for k in OBS_FIELDS}
    selected = ({k: v[even].copy() for k, v in stored.items()} if arm == ARM_A else stored)
    if not (np.array_equal(selected["frame_id"], even) and np.array_equal(selected["timestamp_us"], dec_ts[even])):
        raise SelectionMismatch(f"{clip_id}/{arm}: selected rows do not carry original ids/PTS")
    deps.observation_cls(**stored).validate()
    if arm == ARM_A:
        deps.observation_cls(**selected).validate()
    decoded_meta = {"frame_id": dec_fid, "timestamp_us": dec_ts, "pts": dec_pts, "rgb_sha256": dec_sha,
                    "rgb_shape": dec_shape, "processed": processed}
    return {
        "clip_id": clip_id, "arm": arm, "stored_obs": stored, "selected_obs": selected,
        "decoded_meta": decoded_meta,
        "selected_meta": {k: decoded_meta[k][even] for k in DECODE_FIELDS},
        "decoder_meta": meta, "time_base": f"{time_base.numerator}/{time_base.denominator}",
        "counts": {"decoded_frames": decoded, "process_calls": process_calls, "stored_rows": n_rows,
                   "selected_rows": int(even.size)},
        "landmarker": lm_diag,
        "intervals": {"decoded": interval_stats(dec_ts), "selected": interval_stats(dec_ts[even])},
    }


def persist_arm(writer, seq: int, tag: str, r: dict) -> dict:
    base = f"{seq:02d}_{r['clip_id']}_{tag}"
    arrays = {f"obs_{k}": v for k, v in r["stored_obs"].items()}
    arrays.update({f"decoded_{k}": v for k, v in r["decoded_meta"].items()})
    arrays["selected_frame_id"] = r["selected_meta"]["frame_id"]
    npz = writer.npz(base + ".npz", arrays)
    meta = {k: r[k] for k in ("clip_id", "arm", "decoder_meta", "time_base", "counts", "landmarker", "intervals")}
    meta.update({"tag": tag, "seq": seq, "npz": npz.name, "npz_sha256": sha256_file(npz),
                 "stored_rows": "all decoded rows" if r["arm"] == ARM_A else "even frame_id rows only",
                 "rgb_content_stored": False})
    writer.json(base + ".meta.json", meta)
    return {"npz": npz.name, "meta": base + ".meta.json"}


def exact_equal(x: np.ndarray, y: np.ndarray) -> dict:
    """dtype/shape exact, NaN positions identical, every non-NaN value equal (atol 0, rtol 0)."""
    info = {"dtype": [str(x.dtype), str(y.dtype)], "shape": [list(x.shape), list(y.shape)]}
    if x.dtype != y.dtype or x.shape != y.shape:
        info.update(equal=False, reason="dtype_or_shape")
        return info
    if x.dtype.kind == "f":
        nx, ny = np.isnan(x), np.isnan(y)
        both = ~nx & ~ny
        nan_diff = int(np.count_nonzero(nx != ny))
        val_diff = int(np.count_nonzero(x[both] != y[both]))
        info.update(nan_pattern_equal=nan_diff == 0, nan_position_differences=nan_diff,
                    value_differences=val_diff, equal=nan_diff == 0 and val_diff == 0)
    else:
        diff = int(np.count_nonzero(x != y))
        info.update(value_differences=diff, equal=diff == 0)
    return info


def determinism_report(a: dict, r: dict) -> dict:
    fields = {k: exact_equal(a["stored_obs"][k], r["stored_obs"][k]) for k in OBS_FIELDS}
    decode = {k: exact_equal(a["decoded_meta"][k], r["decoded_meta"][k])
              for k in DECODE_FIELDS + ("processed",)}
    ok = all(v["equal"] for v in fields.values()) and all(v["equal"] for v in decode.values())
    return {"clip_id": a["clip_id"], "arm": ARM_A, "rows_compared": "all decoded rows",
            "rule": "dtype/shape exact; NaN positions identical; finite floats atol0 rtol0; bool/int exact",
            "observation_fields": fields, "decode_meta": decode, "all_equal": ok}


def check_selection(a: dict, b: dict) -> dict:
    n = a["decoded_meta"]["frame_id"].size
    rep = {
        "selected": {k: exact_equal(a["selected_meta"][k], b["selected_meta"][k]) for k in DECODE_FIELDS},
        "all_decoded": {k: exact_equal(a["decoded_meta"][k], b["decoded_meta"][k]) for k in DECODE_FIELDS},
        "observation_ids": {k: exact_equal(a["selected_obs"][k], b["selected_obs"][k])
                            for k in ("frame_id", "timestamp_us")},
    }
    even = np.arange(PHASE, n, STRIDE, dtype=np.int64)
    rep["selected_ids_are_even_from_0_no_last_append"] = bool(
        np.array_equal(a["selected_meta"]["frame_id"], even))
    rep["selected_frames"] = int(even.size)
    ok = rep["selected_ids_are_even_from_0_no_last_append"] and all(
        v["equal"] for grp in ("selected", "all_decoded", "observation_ids") for v in rep[grp].values())
    rep["match"] = ok
    if not ok:
        raise SelectionMismatch(f"{a['clip_id']}: " + json.dumps(jsonable(rep), ensure_ascii=False))
    return rep


def run_extractions(plan: dict, deps: Deps, counters: dict, writer, log: list) -> tuple:
    """Fixed order: first A, first A repeat, determinism verdict, first B, then A/B for the other five."""
    registry: list = []
    results: dict = {}
    files: dict = {}
    seq = 0

    def do(clip, arm, tag):
        nonlocal seq
        r = extract_arm(path=str(clip["path"]), clip_id=clip["clip_id"], arm=arm, n_frames=clip["n_frames"],
                        n_selected=clip["n_selected"], deps=deps, counters=counters, registry=registry)
        files.setdefault(clip["clip_id"], {})[tag] = persist_arm(writer, seq, tag, r)
        log.append({"seq": seq, "clip_id": clip["clip_id"], "tag": tag, "arm": arm,
                    "decoded_frames": r["counts"]["decoded_frames"], "process_calls": r["counts"]["process_calls"]})
        seq += 1
        return r

    clips = plan["clips"]
    first = clips[0]
    a0 = do(first, ARM_A, "A")
    a0r = do(first, ARM_A, "A_repeat")
    det = determinism_report(a0, a0r)
    writer.json("DETERMINISM.json", det)
    if not det["all_equal"]:
        raise NondeterminismError(f"{first['clip_id']}: A repeat differs; stopping before B / other clips")
    del a0r
    b0 = do(first, ARM_B, "B")
    results[first["clip_id"]] = {"A": a0, "B": b0, "selection": check_selection(a0, b0)}
    for clip in clips[1:]:
        a = do(clip, ARM_A, "A")
        b = do(clip, ARM_B, "B")
        results[clip["clip_id"]] = {"A": a, "B": b, "selection": check_selection(a, b)}
    return results, det, files


# ---------------------------------------------------------------- metrics (selected rows only)
def mask_diff(ma: np.ndarray, mb: np.ndarray) -> dict:
    out = {}
    for name, s0, s1 in GROUPS + (("all", 0, 75),):
        x, y = ma[:, s0:s1], mb[:, s0:s1]
        out[name] = {"entries": int(x.size), "A_true": int(x.sum()), "B_true": int(y.sum()),
                     "A_true_B_false": int(np.count_nonzero(x & ~y)),
                     "A_false_B_true": int(np.count_nonzero(~x & y)),
                     "differing": int(np.count_nonzero(x != y))}
    return out


def _runs(mask: np.ndarray) -> list:
    runs, n = [], 0
    for v in mask:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)
    return runs


def hand_rates(usable: np.ndarray) -> dict:
    t = usable.shape[0]
    out = {}
    for name, s0, s1 in HANDS:
        u = usable[:, s0:s1]
        absent = ~u.any(axis=1)
        runs = _runs(absent)
        out[name] = {"frames": int(t),
                     "any_usable_fraction": float(u.any(axis=1).mean()) if t else None,
                     "all21_usable_fraction": float(u.all(axis=1).mean()) if t else None,
                     "all21_absent_frames": int(absent.sum()),
                     "all21_absent_runs": runs, "all21_absent_max_run": max(runs) if runs else 0}
    return out


def dist_stats(d) -> dict:
    d = np.asarray(d, dtype=np.float64)
    if d.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(d.size), "mean": float(d.mean()), "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max())}


def coord_distances(a: dict, b: dict) -> dict:
    xa, xb = a["xy_px"].astype(np.float64), b["xy_px"].astype(np.float64)
    if xa.shape != xb.shape:
        raise SelectionMismatch("coordinate arrays differ in shape")
    common = a["usable"] & b["usable"] & np.isfinite(xa).all(-1) & np.isfinite(xb).all(-1)
    d = np.full(common.shape, np.nan)
    d[common] = np.linalg.norm(xa[common] - xb[common], axis=-1)
    out = {name: dist_stats(d[:, s0:s1][common[:, s0:s1]]) for name, s0, s1 in GROUPS}
    out["all"] = dist_stats(d[common])
    return out


def compute_metrics(a_sel: dict, b_sel: dict) -> dict:
    return {"rows": int(a_sel["frame_id"].size),
            "observed": mask_diff(a_sel["observed"], b_sel["observed"]),
            "usable": mask_diff(a_sel["usable"], b_sel["usable"]),
            "hand_rates": {"A": hand_rates(a_sel["usable"]), "B": hand_rates(b_sel["usable"])},
            "coordinate_distance_px": coord_distances(a_sel, b_sel),
            "coordinate_note": "common usable finite same-slot points only; None when no common point"}


# ---------------------------------------------------------------- scoring
def window_rows(ts: np.ndarray, start_us: int, end_us_exclusive: int) -> np.ndarray:
    return np.flatnonzero((ts >= start_us) & (ts < end_us_exclusive))


def prepare_window(sel: dict, idx: np.ndarray, deps: Deps) -> dict:
    """Unchanged to_native_like + build_channels + collate(label 0). No active span, no fill-in."""
    try:
        nat, present = deps.to_native_like(sel["xy_px"][idx], sel["usable"][idx])
        ch = deps.build_channels(nat, present, sel["timestamp_us"][idx])
        x, pad, _y = deps.collate([dict(ch, label=0)])
    except CadenceError:
        raise
    except Exception as exc:                                  # noqa: BLE001 - technical failure, recorded
        raise PreprocessingError(f"{type(exc).__name__}: {exc}") from exc
    x, pad = np.asarray(x), np.asarray(pad)
    _req(x.dtype == np.float32 and x.shape == (1, idx.size, deps.n_channels), f"unexpected x {x.dtype} {x.shape}",
         PreprocessingError)
    _req(pad.shape == (1, idx.size) and bool(np.all(pad == 1)), "padding mask must be all real frames",
         PreprocessingError)
    return {"x": x, "pad": pad, "info": {
        "x_sha256": array_sha256(x), "x_shape": list(x.shape), "x_dtype": str(x.dtype),
        "frames": int(ch["frames"]), "body_scale": ch["body_scale"], "hand_scales": ch["hand_scales"],
        "body_valid_sum": float(np.sum(ch["body_valid"])) if "body_valid" in ch else None,
        "hand_valid_sum": float(np.sum(ch["hand_valid"])) if "hand_valid" in ch else None}}


def rank_against_label(logits, classes: list, label: str) -> dict:
    """Score helper used only after the forward. Ranks all classes; ties break by class index."""
    lg = np.asarray(logits, dtype=np.float64)
    _req(lg.shape == (len(classes),) and bool(np.isfinite(lg).all()), "logits must be finite, one per class",
         ScoringError)
    _req(label in classes, f"label {label!r} not in checkpoint classes", ScoringError)
    yi = classes.index(label)
    order = sorted(range(len(classes)), key=lambda j: (-lg[j], j))
    rival = next(j for j in order if j != yi)
    return {"scope": "full93", "true_label": label, "top1": classes[order[0]], "top1_index": int(order[0]),
            "correct": order[0] == yi, "true_rank": order.index(yi) + 1, "best_other": classes[rival],
            "true_vs_best_other_margin": float(lg[yi] - lg[rival])}


def direction(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "both_correct"
    if not a_correct and not b_correct:
        return "both_wrong"
    return "B_regression_vs_A" if a_correct else "B_improvement_vs_A"


def run_scoring(plan: dict, results: dict, deps: Deps, counters: dict, log: list) -> dict:
    model, classes, info = deps.load_model()
    counters["model_loads"] += 1
    validate_classes(classes, N_CLASSES)
    state_before = deps.state_hash(model)
    windows = []
    for clip in plan["clips"]:
        s = clip["score"]
        if s is None:
            continue
        r = results[clip["clip_id"]]
        a_sel, b_sel = r["A"]["selected_obs"], r["B"]["selected_obs"]
        ia = window_rows(a_sel["timestamp_us"], s["start_us"], s["end_us_exclusive"])
        ib = window_rows(b_sel["timestamp_us"], s["start_us"], s["end_us_exclusive"])
        _req(np.array_equal(ia, ib) and np.array_equal(a_sel["frame_id"][ia], b_sel["frame_id"][ib]),
             f"{clip['clip_id']}: window rows differ between arms", SelectionMismatch)
        _req(ia.size > 0, f"{clip['clip_id']}: empty scoring window", PreprocessingError)
        arms = {}
        for tag, sel in (("A", a_sel), ("B", b_sel)):
            prep = prepare_window(sel, ia, deps)
            counters["tcn_forwards"] += 1
            logits = np.asarray(deps.forward(model, prep["x"], prep["pad"]), dtype=np.float64)
            log.append({"forward": counters["tcn_forwards"], "clip_id": clip["clip_id"], "arm": tag})
            arms[tag] = {"input": prep["info"], "logits": logits}
        for tag in arms:                                     # label enters only here, after both forwards
            arms[tag]["score"] = rank_against_label(arms[tag]["logits"], classes, s["label"])
        ra, rb = arms["A"]["score"], arms["B"]["score"]
        windows.append({
            "clip_id": clip["clip_id"], "label": s["label"], "start_us": s["start_us"],
            "end_us_exclusive": s["end_us_exclusive"], "window_rows": int(ia.size),
            "window_frame_ids": a_sel["frame_id"][ia], "window_timestamp_us": a_sel["timestamp_us"][ia],
            "A": {"input": arms["A"]["input"], "score": ra, "logits93": arms["A"]["logits"]},
            "B": {"input": arms["B"]["input"], "score": rb, "logits93": arms["B"]["logits"]},
            "same_top1": ra["top1"] == rb["top1"], "same_x": arms["A"]["input"]["x_sha256"] == arms["B"]["input"]["x_sha256"],
            "max_abs_logit_difference": float(np.max(np.abs(arms["A"]["logits"] - arms["B"]["logits"]))),
            "direction": direction(ra["correct"], rb["correct"])})
    state_after = deps.state_hash(model)
    if state_after != state_before:
        raise ModelStateChanged(f"state {state_before} -> {state_after}")
    return {"model": info, "classes": classes, "state_sha256_before": state_before,
            "state_sha256_after": state_after, "state_unchanged": True, "windows": windows}


def interpret(windows: list) -> dict:
    n_reg = sum(w["direction"] == "B_regression_vs_A" for w in windows)
    n_imp = sum(w["direction"] == "B_improvement_vs_A" for w in windows)
    counts = {d: sum(w["direction"] == d for w in windows)
              for d in ("both_correct", "both_wrong", "B_regression_vs_A", "B_improvement_vs_A")}
    worse = (f"B is wrong where A is correct in {n_reg} window(s): limited evidence that extraction input "
             "history (MediaPipe tracking of intermediate frames) affects prediction at the same selected "
             "timestamps. A separate cause review may be proposed; no automatic code change.")
    better = (f"B is correct where A is wrong in {n_imp} window(s): this does not show lower cadence is "
              "beneficial or never harmful; do not adopt B or lower the frame rate.")
    if n_reg and n_imp:
        code, text = "MIXED_DIRECTIONS", worse + " " + better
    elif n_reg:
        code, text = "B_WORSE_IN_SOME_WINDOW", worse
    elif n_imp:
        code, text = "B_BETTER_IN_SOME_WINDOW", better
    else:
        code = "SAME_CORRECTNESS_ALL_WINDOWS"
        text = ("A and B have the same correctness in every scored window. No additional recognition effect "
                "was observed in this tiny set (4 windows, 2 words, 2 known signers); this is not proof that "
                "capture cadence / fps has no impact. Coordinate, mask and score differences are reported as "
                "observations only; no improvement, bottleneck or need for a fix is claimed. The unit closes "
                "without phase, frame or corpus search.")
    return {"code": code, "text": text, "direction_counts": counts,
            "all_top1_identical": all(w["same_top1"] for w in windows), "windows": len(windows),
            "scope": "4 fixed windows, 2 words (나, 죄송하다), known signers, full93 ranking only"}


# ---------------------------------------------------------------- output
class ResultWriter:
    """Exclusive-create writes inside the single result directory only."""

    def __init__(self, out_dir: Path):
        self.dir = out_dir
        self.files: list = []

    def _path(self, name: str) -> Path:
        _req(name and "/" not in name and "\\" not in name and not name.startswith(".") and ".." not in name,
             f"bad output name {name!r}", GateError)
        return self.dir / name

    def json(self, name: str, obj) -> Path:
        p = self._path(name)
        text = json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with open(p, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        self.files.append(name)
        return p

    def text(self, name: str, s: str) -> Path:
        p = self._path(name)
        with open(p, "x", encoding="utf-8", newline="\n") as fh:
            fh.write(s)
        self.files.append(name)
        return p

    def npz(self, name: str, arrays: dict) -> Path:
        p = self._path(name)
        with open(p, "xb") as fh:
            np.savez_compressed(fh, **arrays)
        self.files.append(name)
        return p

    def file_hashes(self) -> dict:
        return {n: sha256_file(self.dir / n) for n in self.files}


def new_counters() -> dict:
    return {"decoded_frames": 0, "landmarker_process_calls": 0, "landmarker_constructed": 0,
            "landmarker_closed": 0, "decoder_opened": 0, "model_loads": 0, "tcn_forwards": 0,
            "camera_calls": 0, "training_runs": 0, "downloads": 0}


EXPECTED_COUNTS = {"decoded_frames": TOTAL_DECODED, "landmarker_process_calls": TOTAL_PROCESS_CALLS,
                   "landmarker_constructed": TOTAL_LANDMARKERS, "landmarker_closed": TOTAL_LANDMARKERS,
                   "decoder_opened": TOTAL_LANDMARKERS, "model_loads": 1, "tcn_forwards": TOTAL_FORWARDS,
                   "camera_calls": 0, "training_runs": 0, "downloads": 0}


def _fmt(v, spec="+.4f"):
    return "N/A" if v is None else format(v, spec)


def render_summary(res: dict) -> str:
    d = res["decision"]
    lines = ["# Capture cadence pair - result summary", "",
             f"- status: {res['status']}", f"- decision: {d['code']}", f"- scope: {d['scope']}", "",
             d["text"], "",
             "| clip | label | rows | A top1 | A rank | A margin | B top1 | B rank | B margin | direction |",
             "|---|---|---:|---|---:|---:|---|---:|---:|---|"]
    for w in res["scoring"]["windows"]:
        a, b = w["A"]["score"], w["B"]["score"]
        lines.append(f"| {w['clip_id']} | {w['label']} | {w['window_rows']} | {a['top1']} | {a['true_rank']} | "
                     f"{_fmt(a['true_vs_best_other_margin'])} | {b['top1']} | {b['true_rank']} | "
                     f"{_fmt(b['true_vs_best_other_margin'])} | {w['direction']} |")
    lines += ["", "## Selected-row observation differences (A vs B, not ground truth)", "",
              "| clip | rows | usable A_true_B_false pose/L/R | usable A_false_B_true pose/L/R | dist px mean / p95 / max (n) |",
              "|---|---:|---|---|---|"]
    for c in res["clips"]:
        m = c["metrics"]
        u = m["usable"]
        dd = m["coordinate_distance_px"]["all"]
        lines.append(f"| {c['clip_id']} | {m['rows']} | "
                     f"{u['pose']['A_true_B_false']}/{u['left_hand']['A_true_B_false']}/{u['right_hand']['A_true_B_false']} | "
                     f"{u['pose']['A_false_B_true']}/{u['left_hand']['A_false_B_true']}/{u['right_hand']['A_false_B_true']} | "
                     f"{_fmt(dd['mean'], '.3f')} / {_fmt(dd['p95'], '.3f')} / {_fmt(dd['max'], '.3f')} ({dd['count']}) |")
    ec = res["execution_counts"]
    lines += ["", "## Execution counts (expected / actual)", ""]
    lines += [f"- {k}: {ec['expected'][k]} / {ec['actual'].get(k)}" for k in ec["expected"]]
    lines += ["", "## Limits", ""] + [f"- {x}" for x in res["limits"]]
    lines += ["", "## Caveats", ""] + [f"- {x}" for x in res["caveats"]]
    return "\n".join(lines) + "\n"


def _clip_entry(clip, r, metrics, files) -> dict:
    a, b = r["A"], r["B"]
    return {"clip_id": clip["clip_id"], "source_sha256": clip["sha256"], "scored": clip["score"] is not None,
            "score": clip["score"], "decoder_meta": a["decoder_meta"], "time_base": a["time_base"],
            "frames": {"decoded": a["counts"]["decoded_frames"], "selected": a["counts"]["selected_rows"],
                       "expected_decoded": clip["n_frames"], "expected_selected": clip["n_selected"]},
            "intervals": a["intervals"],
            "counts": {"A": a["counts"], "B": b["counts"]},
            "landmarker": {"A": a["landmarker"], "B": b["landmarker"]},
            "selection_check": r["selection"], "metrics": metrics, "files": files.get(clip["clip_id"], {})}


def run(*, cfg_path: Path, cfg_sha: str, plan: dict, roots: Roots, deps_loader, hash_file, now,
        closeout_arg, out) -> int:
    try:
        gates = {"receipt": check_receipt(roots), "deadline": check_deadline(now),
                 "teacher_closeout": check_closeout(closeout_arg, roots, hash_file)}
        pins_before = verify_pins(plan, hash_file)
        _req(not roots.result.exists(), f"output directory already exists: {roots.result}", GateError)
        roots.result.mkdir(parents=False, exist_ok=False)
    except (CadenceError, OSError, ValueError) as exc:
        out(json.dumps({"status": "RUN_REFUSED", "kind": getattr(exc, "kind", type(exc).__name__),
                        "error": str(exc)}, ensure_ascii=True, indent=2))
        return 2

    writer = ResultWriter(roots.result)
    counters, log = new_counters(), []
    stage = "load_runtime"
    started = now().astimezone(KST).isoformat()
    code_before = own_code_hashes()
    try:
        writer.json("RUN_STARTED.json", {
            "schema": RESULT_SCHEMA, "started_at_kst": started, "config_path": str(cfg_path),
            "config_sha256": cfg_sha, "own_code_sha256": code_before, "gates": gates,
            "pins_before": pins_before, "expected_counts": EXPECTED_COUNTS})
        deps = deps_loader(plan, roots, hash_file)
        writer.json("RUNTIME.json", {"versions": deps.versions, "import_audit": deps.import_audit})
        stage = "extraction"
        results, det, files = run_extractions(plan, deps, counters, writer, log)
        stage = "metrics"
        metrics = {cid: compute_metrics(r["A"]["selected_obs"], r["B"]["selected_obs"]) for cid, r in results.items()}
        stage = "scoring"
        scoring = run_scoring(plan, results, deps, counters, log)
        stage = "verify_after"
        pins_after = verify_pins(plan, hash_file)
        cfg_after = sha256_file(cfg_path)
        _req(cfg_after == cfg_sha, "config changed during run", PinError)
        _req(own_code_hashes() == code_before, "own code changed during run", PinError)
        _req(counters == EXPECTED_COUNTS, f"execution counts {counters} != {EXPECTED_COUNTS}", CountMismatch)
        result = {
            "schema": RESULT_SCHEMA, "status": "COMPLETE",
            "question": ("With identical selected even-frame RGB and real PTS, does MediaPipe also tracking the "
                         "frames in between change coordinates, observation masks or word predictions?"),
            "arms": {ARM_A: "all decoded frames processed, even frame_id rows kept",
                     ARM_B: "all frames decoded, only even frame_id RGB processed"},
            "started_at_kst": started, "finished_at_kst": now().astimezone(KST).isoformat(),
            "config": {"path": str(cfg_path), "sha256_before": cfg_sha, "sha256_after": cfg_after},
            "own_code_sha256": code_before, "gates": gates,
            "pins_before": pins_before, "pins_after": pins_after, "pins_unchanged": pins_before == pins_after,
            "runtime": {"versions": deps.versions, "import_audit": deps.import_audit},
            "determinism": {"clip_id": det["clip_id"], "all_equal": det["all_equal"], "file": "DETERMINISM.json"},
            "clips": [_clip_entry(c, results[c["clip_id"]], metrics[c["clip_id"]], files) for c in plan["clips"]],
            "scoring": scoring, "decision": interpret(scoring["windows"]),
            "execution_counts": {"expected": EXPECTED_COUNTS, "actual": counters, "log": log},
            "limits": list(LIMITS) + [x for x in plan["config_limits"] if x not in LIMITS],
            "caveats": list(CAVEATS),
            "claims_not_made": ["coordinate correctness", "normal-hand loss counts", "web/real-time speed cause",
                                "new-person generalisation", "sign execution accuracy", "operational change"],
        }
        writer.json("RESULT.json", result)
        writer.text("SUMMARY.md", render_summary(jsonable(result)))
        writer.json("FILE_HASHES.json", writer.file_hashes())
        out(json.dumps({"status": "COMPLETE", "decision": result["decision"]["code"],
                        "result_dir": str(roots.result)}, ensure_ascii=True, indent=2))
        return 0
    except BaseException as exc:                                   # noqa: BLE001 - preserved, then reported
        failure = {"schema": RESULT_SCHEMA, "status": "TECHNICAL_FAILURE",
                   "kind": getattr(exc, "kind", "UNEXPECTED_EXCEPTION"), "stage": stage,
                   "error_type": type(exc).__name__, "message": str(exc),
                   "traceback": traceback.format_exc(), "started_at_kst": started,
                   "no_accuracy_verdict": True, "counters": dict(counters),
                   "expected_counts": EXPECTED_COUNTS, "execution_log": log,
                   "completed_files": list(writer.files)}
        try:
            failure["pins_after"], failure["pin_mismatches_after"] = hash_pins(plan, hash_file)
        except BaseException as pin_exc:                            # noqa: BLE001
            failure["pins_after_error"] = repr(pin_exc)
        try:
            failure["failed_at_kst"] = now().astimezone(KST).isoformat()
            writer.json("FAILURE.json", failure)
            writer.json("FILE_HASHES.json", writer.file_hashes())
        except BaseException as write_exc:                          # noqa: BLE001
            out(f"could not write FAILURE.json: {write_exc!r}")
        out(json.dumps({"status": "TECHNICAL_FAILURE", "kind": failure["kind"], "stage": stage,
                        "error": str(exc)[:2000], "result_dir": str(roots.result)}, ensure_ascii=True, indent=2))
        if not isinstance(exc, Exception):
            raise
        return 3


# ---------------------------------------------------------------- CLI
def _load_config(cfg_path: Path, roots: Roots) -> tuple:
    _req(cfg_path.is_absolute() and cfg_path.name == CONFIG_NAME and _same(cfg_path.parent, roots.unit),
         f"config must be <unit>/{CONFIG_NAME}")
    _req(cfg_path.is_file(), f"config missing: {cfg_path}")
    raw = cfg_path.read_bytes()
    return json.loads(raw.decode("utf-8")), hashlib.sha256(raw).hexdigest()


def main(argv=None, *, roots: Roots = DEFAULT_ROOTS, deps_loader=load_runtime_deps, hash_file=sha256_file,
         now: Callable[[], datetime] | None = None, out: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help=f"absolute path to <unit>/{CONFIG_NAME}")
    ap.add_argument("--run", action="store_true", help="actually decode, extract and score (gated)")
    ap.add_argument("--teacher-closeout", help=f"absolute path to the teacher unit {CLOSEOUT_NAME}")
    args = ap.parse_args(argv)
    now = now or (lambda: datetime.now(timezone.utc))

    if args.run and not args.config:
        out(json.dumps({"status": "RUN_REFUSED", "error": "--run requires --config"}, ensure_ascii=True))
        return 2
    cfg_path = Path(args.config) if args.config else roots.unit / CONFIG_NAME
    try:
        cfg, cfg_sha = _load_config(cfg_path, roots)
        plan = validate_config(cfg, roots)
        pins = verify_pins(plan, hash_file)
    except (CadenceError, OSError, ValueError) as exc:
        out(json.dumps({"status": "PREFLIGHT_FAIL", "kind": getattr(exc, "kind", type(exc).__name__),
                        "error": str(exc)}, ensure_ascii=True, indent=2))
        return 2

    if args.run:
        return run(cfg_path=cfg_path, cfg_sha=cfg_sha, plan=plan, roots=roots, deps_loader=deps_loader,
                   hash_file=hash_file, now=now, closeout_arg=args.teacher_closeout, out=out)

    readiness = {}
    for name, fn in (("receipt", lambda: check_receipt(roots)), ("deadline", lambda: check_deadline(now)),
                     ("teacher_closeout", lambda: check_closeout(args.teacher_closeout, roots, hash_file))):
        try:
            readiness[name] = {"ok": True, **fn()}
        except (CadenceError, OSError, ValueError) as exc:
            readiness[name] = {"ok": False, "error": str(exc)}
    readiness["output_dir_fresh"] = {"ok": not roots.result.exists(), "path": str(roots.result)}
    out(json.dumps(jsonable({
        "status": "PREFLIGHT_OK", "config_path": str(cfg_path), "config_sha256": cfg_sha,
        "pins_verified": len(pins), "clips": [c["clip_id"] for c in plan["clips"]],
        "expected_counts": EXPECTED_COUNTS, "readiness_for_run": readiness,
        "ready_for_run": all(v["ok"] for v in readiness.values()),
        "constructed": {"decoder": 0, "landmarker": 0, "model": 0}}), ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
