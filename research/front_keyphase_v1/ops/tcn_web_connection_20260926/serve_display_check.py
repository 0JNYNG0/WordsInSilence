"""Check-only server for the browser display check. NOT a production entry point.
- cv2.VideoCapture is replaced by a blocker BEFORE the app is imported: no camera open, no device probe.
- one in-memory registry entry with a wrong hash (_check_unavailable) to see the 'unavailable' display;
  registry file on disk is not edited. Port 8766 so the normal 8765 server is never disturbed."""
import os, sys
import cv2
BLOCKED = []
class _NoDevice:
    def __init__(self, *a, **k):
        BLOCKED.append(a); raise RuntimeError("camera access blocked in the display-check server")
cv2.VideoCapture = _NoDevice
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
from wis_lab.service import app as service, neural
neural.MODELS["_check_unavailable"] = dict(neural.MODELS["tcn_control_83212868"],
                                           model_id="_check_unavailable", sha256="0" * 64)
import uvicorn
print("display-check server on http://127.0.0.1:8766/ (camera blocked)", flush=True)
uvicorn.run(service.app, host="127.0.0.1", port=8766, log_level="warning")
