"""Codex verification harness: production page/report renderer, synthetic records only."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2

blocked = []
requests = []


def no_device(*args, **kwargs):
    blocked.append("device operation")
    raise RuntimeError("Camera forbidden in report verification")


cv2.VideoCapture = no_device
from wis_lab.service import app as service, camera
from wis_lab.capture import landmarker

camera.list_devices = no_device
camera.CameraWorker.start = no_device
service.Store.acquire_camera = no_device
landmarker.Landmarker.__init__ = no_device

from fastapi.responses import HTMLResponse, JSONResponse


@service.app.middleware("http")
async def verification_access(request, call_next):
    path = request.url.path
    if path == "/__report_check":
        return JSONResponse({"device_attempts": len(blocked), "requests": requests,
                             "sessions": len(service.store.sessions) if service.store else 0,
                             "camera": "closed" if not service.store or service.store.camera is None
                             else "unexpected"})
    requests.append({"method": request.method, "path": path})
    if path == "/api/v1/devices" or path.startswith("/api/v1/sessions"):
        blocked.append("device/session route")
        return JSONResponse({"error": "CAMERA_BLOCKED_FOR_VERIFICATION"}, status_code=403)
    if path == "/__report_fixture":
        name = "LEGACY_REPORT.json" if request.query_params.get("legacy") == "1" else "SYNTHETIC_REPORT.json"
        report = json.loads((HERE / name).read_text(encoding="utf-8"))
        html = (ROOT / "web/index.html").read_text(encoding="utf-8")
        script = """<script>
          document.title = '합성 자료 보고서 검증';
          document.getElementById('report').innerHTML =
            '<p style="background:#fff3bf;padding:12px"><b>합성 자료로 확인하는 화면입니다. 실제 사용자 성능이 아닙니다.</b></p>'
            + renderReport(REPORT_FIXTURE);
          document.getElementById('report').scrollIntoView();
        </script>""".replace("REPORT_FIXTURE", json.dumps(report, ensure_ascii=False).replace("<", "\\u003c"))
        return HTMLResponse(html.replace("</body>", script + "</body>"))
    return await call_next(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(service.app, host="127.0.0.1", port=8766, log_level="warning")
