"""Inspect only the small label artifact linked by the KSL dataset authors; no pickle execution."""
from pathlib import Path
import hashlib
import json
import pickletools
import requests

out = Path(__file__).resolve().parent
url = "https://drive.google.com/uc?export=download&id=1nFF_VsilnJv1jyshtdUKtoX_kNbwCk3w"
report = {"repository": "https://github.com/Yangseung/KSL",
          "published_label_link": "https://drive.google.com/file/d/1nFF_VsilnJv1jyshtdUKtoX_kNbwCk3w/view?usp=sharing",
          "request_url": url, "credentials_used": False, "limit_bytes": 2 * 1024 * 1024,
          "pickle_load_executed": False, "training_approved": False}
try:
    with requests.get(url, stream=True, timeout=(10, 25)) as response:
        report.update(http_status=response.status_code, content_type=response.headers.get("Content-Type"),
                      content_length=response.headers.get("Content-Length"),
                      content_disposition=response.headers.get("Content-Disposition"))
        if response.status_code != 200 or "text/html" in response.headers.get("Content-Type", ""):
            report["status"] = "LABEL_PAYLOAD_UNAVAILABLE"
        elif int(response.headers.get("Content-Length", 0)) > report["limit_bytes"]:
            report["status"] = "OVERSIZE_NOT_READ"
        else:
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > report["limit_bytes"]:
                    raise ValueError("label payload exceeded2MiB")
                chunks.append(chunk)
            blob = b"".join(chunks)
            with (out / "ksl77_class_label.p").open("xb") as handle:
                handle.write(blob)
            strings = []
            opnames = set()
            for opcode, value, position in pickletools.genops(blob):
                opnames.add(opcode.name)
                if isinstance(value, str):
                    strings.append(value)
            report.update(status="PICKLE_STRINGS_INSPECTED_WITHOUT_EXECUTION", payload_bytes=len(blob),
                          sha256=hashlib.sha256(blob).hexdigest(), string_values=strings,
                          opcode_names=sorted(opnames))
except Exception as error:
    report.update(status="FETCH_OR_PARSE_FAILED", error_type=type(error).__name__, error=str(error)[:500])
(out / "ksl77_label_fetch.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
