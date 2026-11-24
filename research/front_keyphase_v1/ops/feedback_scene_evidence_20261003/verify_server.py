"""Codex verification harness only: replay UI, no camera/MediaPipe construction."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import cv2

blocked = []
requests = []

def no_device(*args, **kwargs):
    blocked.append("forbidden device operation")
    raise RuntimeError("device access forbidden in evidence verification")

cv2.VideoCapture = no_device
from wis_lab.service import app as service, camera
from wis_lab.capture import landmarker
camera.list_devices = no_device
camera.CameraWorker.start = no_device
service.Store.acquire_camera = no_device
landmarker.Landmarker.__init__ = no_device

from fastapi.responses import JSONResponse

@service.app.middleware("http")
async def verification_access(request, call_next):
    path = request.url.path
    if path == "/__evidence_check":
        st = service.store
        return JSONResponse({"device_attempts": len(blocked), "requests": requests,
                             "sessions": len(st.sessions) if st else 0,
                             "trials": len(st.trials) if st else 0,
                             "camera": "closed" if not st or st.camera is None else "unexpected"})
    requests.append({"method": request.method, "path": path})
    if path == "/api/v1/devices" or path.startswith("/api/v1/sessions"):
        blocked.append("forbidden camera/session route")
        return JSONResponse({"error": "CAMERA_BLOCKED_FOR_VERIFICATION"}, status_code=403)
    return await call_next(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(service.app, host="127.0.0.1", port=8766, log_level="warning")
