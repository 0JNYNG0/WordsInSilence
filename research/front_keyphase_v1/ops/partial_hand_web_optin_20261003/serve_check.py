"""Serve the REAL lab app on 127.0.0.1:8767 for a browser replay check, with device access blocked.

Copy this file into the real unit directory (lab/ops/<unit>/serve_check.py) and run it from there:

    py -3.13 serve_check.py

What it does, in order, BEFORE the app module is imported:
  1. cv2.VideoCapture is wrapped: a call whose first argument is not an existing file path (an int
     device index, a digit string, a URL, a pipeline) is recorded and raises. Reading a registered
     reference MP4 by its file path is still allowed, so replay evidence frames keep working.
  2. wis_lab.service.camera.CameraWorker.start and wis_lab.service.camera.list_devices are replaced
     by functions that record the attempt and raise.
  3. A small HTTP guard refuses POST /api/v1/sessions, GET /api/v1/devices and the session preview
     with 403, so no session (and no runs/<session> directory) is created by a stray click.

It does not change the app implementation, open a session, create fixtures or fall back to any
synthetic data. Stored public references are replayed through the existing POST /api/v1/replays.
Every blocked attempt is printed and appended to serve_check_device_access.jsonl beside this file.

Not blockable from here without guessing an API (see HANDOFF.md): any device access inside
CameraWorker.__init__ that does not go through cv2.VideoCapture, and any other camera backend the
camera module may use internally.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

UNIT = pathlib.Path(__file__).resolve().parent
LAB = pathlib.Path(__file__).resolve().parents[2]
HOST, PORT = "127.0.0.1", 8767
LOG = UNIT / "serve_check_device_access.jsonl"
BLOCKED_ROUTES = (("POST", "/api/v1/sessions"), ("GET", "/api/v1/devices"))
ATTEMPTS = []


class DeviceAccessBlocked(RuntimeError):
    pass


def _record(what: str, detail) -> None:
    row = {"blocked": what, "detail": repr(detail)[:200],
           "at_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ATTEMPTS.append(row)
    print("[serve_check] BLOCKED", json.dumps(row, ensure_ascii=False), flush=True)
    try:
        with open(LOG, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[serve_check] could not write {LOG}: {exc}", flush=True)


def _block_cv2() -> str:
    try:
        import cv2
    except Exception as exc:                              # noqa: BLE001
        return f"cv2 not importable ({type(exc).__name__}); nothing to wrap"
    real = cv2.VideoCapture

    def guarded(*args, **kwargs):
        src = args[0] if args else kwargs.get("filename", kwargs.get("index"))
        if isinstance(src, (str, os.PathLike)) and os.path.isfile(os.fspath(src)):
            return real(*args, **kwargs)                  # a file on disk, not a device
        _record("cv2.VideoCapture", src)
        raise DeviceAccessBlocked(f"serve_check: cv2.VideoCapture({src!r}) is blocked")

    cv2.VideoCapture = guarded
    return "cv2.VideoCapture wrapped (files allowed, devices blocked)"


def _block_camera_module() -> list:
    from wis_lab.service import camera
    done = []
    err = getattr(camera, "CameraError", DeviceAccessBlocked)

    def start(self, *a, **k):
        _record("CameraWorker.start", a)
        raise err("serve_check: CameraWorker.start is blocked")

    def list_devices(*a, **k):
        _record("camera.list_devices", a)
        raise DeviceAccessBlocked("serve_check: list_devices is blocked")

    if hasattr(camera, "CameraWorker") and hasattr(camera.CameraWorker, "start"):
        camera.CameraWorker.start = start
        done.append("CameraWorker.start")
    if hasattr(camera, "list_devices"):
        camera.list_devices = list_devices
        done.append("camera.list_devices")
    return done


def main() -> int:
    if not (LAB / "src" / "wis_lab").is_dir():
        print(f"[serve_check] {LAB} is not the lab root (no src/wis_lab). Copy this file to "
              f"lab/ops/<unit>/ and run it there.", flush=True)
        return 2
    sys.path.insert(0, str(LAB / "src"))
    print("[serve_check]", _block_cv2(), flush=True)
    print("[serve_check] blocked in camera module:", _block_camera_module(), flush=True)

    from wis_lab.service import app as app_module          # imported only after the blocks
    from fastapi.responses import JSONResponse

    @app_module.app.middleware("http")
    async def _guard(request, call_next):
        p, m = request.url.path, request.method.upper()
        if (m, p) in BLOCKED_ROUTES or (p.startswith("/api/v1/sessions/")
                                        and p.endswith("/preview.jpg")):
            _record(f"HTTP {m} {p}", None)
            return JSONResponse(status_code=403, content={
                "error": "BLOCKED_BY_SERVE_CHECK",
                "meaning": "camera sessions and device listing are disabled in this check server; "
                           "use stored replay"})
        return await call_next(request)

    try:
        import uvicorn
    except Exception as exc:                              # noqa: BLE001
        print(f"[serve_check] uvicorn is not importable ({type(exc).__name__}); not serving.",
              flush=True)
        return 2
    print(f"[serve_check] lab={LAB}", flush=True)
    print(f"[serve_check] open http://{HOST}:{PORT}/ - replay a registered reference only; "
          f"do not open a session", flush=True)
    uvicorn.run(app_module.app, host=HOST, port=PORT, log_level="info")
    print(f"[serve_check] stopped. blocked attempts: {len(ATTEMPTS)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
