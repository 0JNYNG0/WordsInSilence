"""Bounded, unauthenticated retrieval of the official sample link observed in the page DOM."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import requests

OUT = Path(__file__).resolve().parent
URL = "https://aihub.or.kr/file/down.do?fileSn=15287&smpleFileSn=15287&dataSetSn=636"
LIMIT = 128 * 1024 * 1024
report = {"url": URL, "at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
          "credentials_used": False, "maximum_bytes": LIMIT,
          "prior_attempt": "workspace-sandbox connection denied WinError10013; no response/payload received"}
try:
    with requests.get(URL, headers={"Referer": "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=636"},
                      stream=True, timeout=(10, 30)) as response:
        report.update(http_status=response.status_code,
                      content_type=response.headers.get("Content-Type"),
                      content_length=response.headers.get("Content-Length"),
                      content_disposition=response.headers.get("Content-Disposition"))
        if response.status_code != 200 or "text/html" in response.headers.get("Content-Type", ""):
            prefix = next(response.iter_content(1024), b"")
            report.update(status="NO_SAMPLE_PAYLOAD", body_prefix=prefix.decode("utf-8", errors="replace")[:700])
        elif int(response.headers.get("Content-Length", 0)) > LIMIT:
            report.update(status="DECLINED_OVERSIZE_SAMPLE", payload_bytes_read=0)
        else:
            size, chunks = 0, []
            for chunk in response.iter_content(256 * 1024):
                size += len(chunk)
                if size > LIMIT:
                    raise ValueError("sample exceeded the128MiB diagnostic bound")
                chunks.append(chunk)
            blob = b"".join(chunks)
            with (OUT / "sample_response.bin").open("xb") as handle:
                handle.write(blob)
            report.update(status="PAYLOAD_RECEIVED_TYPE_NOT_YET_VERIFIED", payload_bytes=len(blob),
                          sha256=hashlib.sha256(blob).hexdigest(), first_bytes_hex=blob[:16].hex())
except Exception as error:
    report.update(status="FETCH_FAILED", error_type=type(error).__name__, error=str(error)[:700])
finally:
    (OUT / "sample_fetch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
