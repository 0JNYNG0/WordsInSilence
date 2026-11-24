"""Bounded official-source acquisition; never imports downloaded code."""
import argparse
import hashlib
import json
import pathlib
import urllib.request
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parent
PROVENANCE = ROOT / "TINYNEXT_PROVENANCE.json"
API = "https://api.github.com/repos/yuffeenn/TinyNeXt/commits/main"
SOURCE_FILES = ["LICENSE", "README.md", "classification/README.md",
                "classification/requirements.txt", "classification/models/tinynext.py",
                "classification/models/modules.py", "classification/config.py",
                "classification/util/datasets.py"]


def read_public(url, cap):
    request = urllib.request.Request(url, headers={"User-Agent": "WordsInSilence-public-source-review"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise RuntimeError("Response exceeded predeclared byte cap")
    return data


def save_record(path, url, payload):
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing acquisition differs; refusing overwrite: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return {"path": str(path), "url": url, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if PROVENANCE.exists():
        doc = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    else:
        doc = {"reviewer": "Codex ksl_source_audit", "repository": "https://github.com/yuffeenn/TinyNeXt",
               "first_network_error": {"operation": "GET official GitHub commit metadata", "sandbox": "default",
                                       "type": "URLError", "message": "[Errno 11001] getaddrinfo failed",
                                       "preserved_from_exec": "fe0636", "no_partial_files": True},
               "files": [], "events": [], "private_uploads": 0, "training_runs": 0}
    try:
        if "commit" not in doc:
            raw = read_public(API, 600000)
            commit = json.loads(raw)
            doc["commit"] = commit["sha"]
            doc["commit_date"] = commit["commit"]["committer"]["date"]
            doc["metadata"] = save_record(ROOT / "upstream" / "COMMIT.json", API, raw)
        base = "https://raw.githubusercontent.com/yuffeenn/TinyNeXt/" + doc["commit"] + "/"
        if args.weights:
            if not doc.get("source_review_passed"):
                raise RuntimeError("Record source review before downloading checkpoint")
            source = "classification/logs/tinynext_t/tinynext_t.pth"
            target = ROOT / "tinynext_t.pth"
            payload = read_public(base + source, 20_000_000)
            doc["checkpoint"] = save_record(target, base + source, payload)
        else:
            known = {x["url"] for x in doc["files"]}
            for source in SOURCE_FILES:
                url = base + source
                if url in known:
                    continue
                payload = read_public(url, 200000)
                doc["files"].append(save_record(ROOT / "upstream" / source, url, payload))
        doc["events"].append({"at": datetime.now(timezone.utc).isoformat(), "stage": "weights" if args.weights else "sources", "status": "ok"})
    except Exception as error:
        doc["events"].append({"at": datetime.now(timezone.utc).isoformat(), "stage": "weights" if args.weights else "sources", "status": "error", "type": type(error).__name__, "message": str(error)})
        raise
    finally:
        PROVENANCE.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"commit": doc["commit"], "source_files": len(doc["files"]), "checkpoint": doc.get("checkpoint")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
