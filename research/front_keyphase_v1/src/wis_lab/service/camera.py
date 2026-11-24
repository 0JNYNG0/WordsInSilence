"""The one thread that owns the camera and the MediaPipe models.

Opened once for a test session and kept open across trials: nothing here restarts the camera
 per attempt. Preview encoding is throttled and follows landmark extraction.
"""
from __future__ import annotations

import threading
import time
import uuid

import cv2
import numpy as np

from ..capture.landmarker import Landmarker
from ..core.observation import N_POINTS, Observation, empty_frames
from .diagnostic_capture import CaptureBuffer

PREVIEW_INTERVAL_S = 1 / 15.0
PREVIEW_JPEG_QUALITY = 70
MAX_RECORD_S = 10.0
END_QUIET_S = 0.75          # hands out of view this long ends the take
MAX_RECORD_FRAMES = 1200


class CameraError(Exception):
    pass


class CameraWorker:
    """Owns one cv2.VideoCapture plus one Landmarker for the life of a test session."""

    def __init__(self, contract: dict, device_index: int = 0, width: int = 1280, height: int = 720,
                 shadow_assignment: bool = False):
        self.contract = contract
        # OPT-IN paired test only: the operating assignment stays the recorded/scored output and the
        # frozen start-provisional candidate is saved beside it (service/shadow_assignment.py)
        self.shadow_assignment = bool(shadow_assignment)
        self._shadow_gen_seen: int | None = None
        self.device_index = device_index
        self.want = (width, height)
        self.clock_id = "server_monotonic_" + uuid.uuid4().hex[:8]

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: str | None = None
        self._thread: threading.Thread | None = None

        self._preview: bytes | None = None
        self._preview_frame_id = -1
        self._preview_captured_ns = 0
        self._frame_id = -1
        self._actual = (0, 0)
        self._fps_estimate = 0.0

        # recording state, all guarded by _lock
        self._recording = False
        self._rec_rows: list[dict] = []
        self._rec_started_ns = 0
        self._rec_origin_ns = 0
        self._last_hand_ns = 0
        self._end_candidate_ns: int | None = None
        self._end_reason: str | None = None
        self.extractor_id: str | None = None
        self._rec_generation = 0
        self._capture_buffer = None
        self._candidate_c = False
        self._candidate_lm = None
        self._candidate_requested = threading.Event()
        self._candidate_ready = threading.Event()
        self._candidate_error = None
        self._candidate_extractor_id = None

    # ---- lifecycle ------------------------------------------------------
    def start(self, timeout_s: float = 20.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="wis-camera", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            self.stop()
            raise CameraError(self._error or f"camera did not become ready within {timeout_s}s")
        if self._error:
            raise CameraError(self._error)

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._recording = False
            self._rec_generation += 1
            self._rec_rows = []
            self._capture_buffer = None
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None

    def status(self) -> dict:
        with self._lock:
            return {
                "clock_id": self.clock_id,
                "device_index": self.device_index,
                "resolution": list(self._actual),
                "fps_estimate": round(self._fps_estimate, 2),
                "frame_id": self._frame_id,
                "preview_frame_id": self._preview_frame_id,
                "preview_age_ms": (round((time.perf_counter_ns() - self._preview_captured_ns) / 1e6, 1)
                                   if self._preview_captured_ns else None),
                "recording": self._recording,
                "recorded_frames": len(self._rec_rows),
                "error": self._error,
            }

    def preview_jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._preview, self._preview_frame_id

    # ---- trial control --------------------------------------------------
    def begin_recording(self, *, store_video: bool = False, candidate_c: bool = False) -> int:
        # Initialise on the camera owner thread BEFORE starting the recording clock.
        # Each C take has fresh detector/association state, matching the frozen experiment.
        if candidate_c:
            with self._lock:
                if self._recording:
                    raise CameraError("already recording")
            self._candidate_error = None
            self._candidate_ready.clear()
            self._candidate_requested.set()
            if not self._candidate_ready.wait(20.0):
                raise CameraError("C_DETECTOR_PREPARE_TIMEOUT")
            if self._candidate_error:
                raise CameraError(self._candidate_error)
        with self._lock:
            if self._recording:
                raise CameraError("already recording")
            self._recording = True
            self._rec_generation += 1
            self._capture_buffer = CaptureBuffer() if store_video else None
            self._candidate_c = bool(candidate_c)
            self._rec_rows = []
            self._rec_started_ns = time.perf_counter_ns()
            self._rec_origin_ns = 0
            self._last_hand_ns = 0
            self._end_candidate_ns = None
            self._end_reason = None
            return self._rec_started_ns

    def request_stop(self, reason: str = "manual") -> int:
        """Mark the end candidate now; the worker flushes on its next tick."""
        with self._lock:
            if not self._recording:
                raise CameraError("not recording")
            if self._end_candidate_ns is None:
                self._end_candidate_ns = time.perf_counter_ns()
                self._end_reason = reason
            return self._end_candidate_ns

    def take(self) -> tuple[Observation | None, dict]:
        """Close the current take and hand back its observation."""
        with self._lock:
            rows = self._rec_rows
            self._recording = False
            self._rec_generation += 1
            capture = self._capture_buffer
            self._capture_buffer = None
            self._rec_rows = []
            end_ns = self._end_candidate_ns or time.perf_counter_ns()
            reason = self._end_reason or "manual"
            started = self._rec_started_ns
            w, h = self._actual
            candidate_c = self._candidate_c
            candidate_error = self._candidate_error
        meta = {
            "_capture_buffer": capture,
            "clock_id": self.clock_id,
            "end_candidate_ns": end_ns,
            "end_reason": reason,
            "recorded_frames": len(rows),
            "record_duration_ms": round((end_ns - started) / 1e6, 1),
            "candidate_c_error": candidate_error if candidate_c else None,
        }
        if not rows:
            return None, meta

        out = empty_frames(len(rows))
        for i, r in enumerate(rows):
            out["xy_px"][i] = r["xy_px"]
            out["timestamp_us"][i] = r["timestamp_us"]
            out["frame_id"][i] = r["frame_id"]
            out["observed"][i] = r["observed"]
            out["usable"][i] = r["usable"]
            out["conf"][i] = r["conf"]
            out["hand_score"][i] = r["hand_score"]
        obs = Observation(**out)
        obs.sidecar = {
            "schema": "wis_observation_2d_v1",
            "origin": "webcam",
            "clock_id": self.clock_id,
            "clock": "backend monotonic read time of the frame (perf_counter_ns); "
                     "NOT the sensor exposure time",
            "width": w, "height": h,
            "mirrored": False,
            "rotation_deg": 0,
            "device_index": self.device_index,
            "extractor_id": self.extractor_id,
            "hand_assignment_policy": self.contract["hand_assignment"]["policy"],
            "end_reason": reason,
            "decoded_frames": len(rows),
        }
        obs.sidecar["quality"] = obs.quality_summary()
        if candidate_c and not candidate_error:
            try:
                from .word100_candidate import finalize_observation
                co = empty_frames(len(rows))
                for i, r in enumerate(rows):
                    co["timestamp_us"][i] = out["timestamp_us"][i]
                    co["frame_id"][i] = out["frame_id"][i]
                    for k, v in r["candidate_c"].items():
                        co[k][i] = v
                cobs = Observation(**co)
                cobs.sidecar = dict(obs.sidecar, hand_assignment_policy="frozen_joint_v3",
                                    extractor_id=self._candidate_extractor_id,
                                    origin="webcam_candidate_c", detector_reset="each_take")
                cobs.sidecar.pop("quality", None)
                meta["_candidate_c_obs"] = finalize_observation(cobs)
            except Exception as exc:
                meta['candidate_c_error'] = f'C_OBSERVATION_FINALIZE_ERROR: {type(exc).__name__}'
        if rows and "shadow" in rows[0]:
            sh = empty_frames(len(rows))
            for i, r in enumerate(rows):
                sh["timestamp_us"][i] = out["timestamp_us"][i]
                sh["frame_id"][i] = out["frame_id"][i]
                for k, v in r["shadow"].items():
                    sh[k][i] = v
            sobs = Observation(**sh)
            sobs.sidecar = dict(obs.sidecar, hand_assignment_policy="SHADOW_start_provisional_candidate_20260926",
                                shadow_of_primary=True)
            sobs.sidecar.pop("quality", None)
            sobs.sidecar["quality"] = sobs.quality_summary()
            meta["_shadow_obs"] = sobs
        return obs, meta

    def end_candidate(self) -> int | None:
        with self._lock:
            return self._end_candidate_ns

    # ---- worker ---------------------------------------------------------
    def _run(self) -> None:
        cap = None
        lm = None
        try:
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise CameraError(f"cannot open camera index {self.device_index}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.want[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.want[1])
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("camera opened but returned no frame")
            with self._lock:
                self._actual = (int(frame.shape[1]), int(frame.shape[0]))

            if self.shadow_assignment:
                from .shadow_assignment import ShadowLandmarker
                lm = ShadowLandmarker(self.contract)
            else:
                lm = Landmarker(self.contract)
            self.extractor_id = lm.extractor_id()
            self._ready.set()

            last_preview = 0.0
            last_ns = time.perf_counter_ns()
            while not self._stop.is_set():
                if self._candidate_requested.is_set():
                    self._candidate_requested.clear()
                    try:
                        if self._candidate_lm is not None:
                            self._candidate_lm.close()
                            self._candidate_lm = None
                        from .word100_candidate import CandidateLandmarker
                        self._candidate_lm = CandidateLandmarker(self.contract)
                        self._candidate_extractor_id = self._candidate_lm.extractor_id()
                        self._candidate_error = None
                    except Exception as exc:
                        self._candidate_error = f"C_DETECTOR_PREPARE_ERROR: {type(exc).__name__}"
                    finally:
                        self._candidate_ready.set()
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                now_ns = time.perf_counter_ns()
                with self._lock:
                    self._frame_id += 1
                    frame_id = self._frame_id
                    dt = (now_ns - last_ns) / 1e9
                    if dt > 0:
                        self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * (1 / dt)
                    last_ns = now_ns
                    recording = self._recording

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if recording:
                    self._record_frame(rgb, now_ns, frame_id, lm)

                # preview is throttled and deliberately after the analysis path
                if time.perf_counter() - last_preview >= PREVIEW_INTERVAL_S:
                    last_preview = time.perf_counter()
                    ok_enc, buf = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY])
                    if ok_enc:
                        with self._lock:
                            self._preview = buf.tobytes()
                            self._preview_frame_id = frame_id
                            self._preview_captured_ns = now_ns
        except Exception as exc:  # noqa: BLE001 - surfaced through status()/start()
            self._error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            if self._candidate_lm is not None:
                self._candidate_lm.close()
                self._candidate_lm = None
            if lm is not None:
                lm.close()
            if cap is not None:
                cap.release()

    def _record_frame(self, rgb, now_ns: int, frame_id: int, lm: Landmarker) -> None:
        with self._lock:
            if not self._recording:
                return
            generation = self._rec_generation
            if self._rec_origin_ns == 0:
                self._rec_origin_ns = now_ns
            origin = self._rec_origin_ns
            started = self._rec_started_ns
            already = len(self._rec_rows)
            end_pending = self._end_candidate_ns is not None
            candidate_c = self._candidate_c
        if already >= MAX_RECORD_FRAMES:
            return

        ts_us = (now_ns - origin) // 1000
        row = empty_frames(1)
        # MediaPipe VIDEO mode gets a SESSION-continuous clock, not this recording's clock. ts_us
        # restarts at 0 every recording while the Landmarker lives for the whole session, so the
        # monotonic clamp in next_timestamp_ms collapsed every frame of every recording after the
        # first to 1 ms apart. The stored timestamp_us below is per-recording and unchanged.
        srow = None
        if self.shadow_assignment:
            # take boundary: the candidate's tracks start empty for every take, as in the offline
            # evaluation. Worker thread only, so it cannot race a frame in flight. MediaPipe and its
            # session clock are NOT reset.
            if generation != self._shadow_gen_seen:
                lm.reset_assignment_state()
                self._shadow_gen_seen = generation
            srow = empty_frames(1)
            lm.process(rgb, now_ns // 1000, row, 0, shadow_out=srow)
        else:
            lm.process(rgb, now_ns // 1000, row, 0)
        crow = None
        if candidate_c:
            crow = empty_frames(1)
            if not self._candidate_error:
                try:
                    self._candidate_lm.process(rgb, ts_us, crow, 0)
                except Exception as exc:
                    with self._lock:
                        if self._recording and generation == self._rec_generation:
                            self._candidate_error = f"C_DETECTOR_FRAME_ERROR: {type(exc).__name__}"
        hands_visible = bool(row["usable"][0, 33:75].any())

        with self._lock:
            if not self._recording or generation != self._rec_generation:
                return
            if self._rec_rows and ts_us <= self._rec_rows[-1]["timestamp_us"]:
                ts_us = self._rec_rows[-1]["timestamp_us"] + 1
            self._rec_rows.append({
                "xy_px": row["xy_px"][0], "timestamp_us": ts_us, "frame_id": frame_id,
                "observed": row["observed"][0], "usable": row["usable"][0],
                "conf": row["conf"][0], "hand_score": row["hand_score"][0],
                **({"shadow": {k: srow[k][0] for k in ("xy_px", "observed", "usable", "conf",
                                                       "hand_score")}} if srow is not None else {}),
                **({"candidate_c": {k: crow[k][0] for k in ("xy_px", "observed", "usable", "conf",
                                                            "hand_score")}} if crow is not None else {}),
            })
            # The same RGB input, frame ID and final timestamp as the appended row.
            # The lock prevents take/cancel from handing out a partially changing buffer.
            if self._capture_buffer is not None:
                self._capture_buffer.add(rgb, frame_id, ts_us)
            if hands_visible:
                self._last_hand_ns = now_ns
            if end_pending:
                return
            elapsed = (now_ns - started) / 1e9
            if elapsed >= MAX_RECORD_S:
                self._end_candidate_ns = now_ns
                self._end_reason = "timeout_max_duration"
            elif (not candidate_c and self._last_hand_ns and not hands_visible
                  and (now_ns - self._last_hand_ns) / 1e9 >= END_QUIET_S):
                self._end_candidate_ns = now_ns
                self._end_reason = "auto_hands_lowered"


def list_devices(max_index: int = 5) -> list[dict]:
    """Probe camera indices without leaving anything open."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                ok, frame = cap.read()
                found.append({"index": i, "reads_frame": bool(ok and frame is not None),
                              "resolution": [int(frame.shape[1]), int(frame.shape[0])]
                              if ok and frame is not None else None})
        finally:
            cap.release()
    return found


assert N_POINTS == 75  # the recorder writes whole 75-point rows
