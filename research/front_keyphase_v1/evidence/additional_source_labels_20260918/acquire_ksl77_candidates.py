"""Acquire all available study/eat originals via two bounded public ZIP ranges.

Source prefixes are unverified identity candidates. Only prefix 00 is cleared
for visual source triage; no training/dev/test split is assigned here.
"""
from pathlib import Path
import hashlib
import json
import re
import struct
import zlib

import requests

OUT = Path(__file__).resolve().parent
RAW = OUT.parents[1] / "runs" / "ksl77_source_review_v1" / "raw"
URL = "https://www.dropbox.com/scl/fi/daqmxyxycgr4r39lzn7gf/video.zip?rlkey=uphg8vs7n4mrdgl7bzi62y0on&dl=1"
index = json.loads((OUT / "ksl77_zip_index.json").read_text(encoding="utf-8"))
labels = json.loads((OUT / "ksl77_label_mapping.json").read_text(encoding="utf-8"))["class_labels"]
report = {"source": "KSL77_AUTHOR_VIDEO_ARCHIVE", "url": URL, "credentials_used": False,
          "archive_bytes": index["archive_bytes"], "archive_directory_sha256": index["central_directory_sha256"],
          "training_approved": False, "signer_identity_verified": False,
          "split_status": "UNASSIGNED_SEPARATE_SOURCE_NAMESPACE",
          "visual_triage_allowed_prefixes": ["00"], "other_prefixes": "METADATA_AND_INTEGRITY_ONLY",
          "requests": [], "files": []}


def fetch(start, end):
    expected = end - start + 1
    if not 0 < expected <= 128 * 1024 * 1024:
        raise ValueError("Class range exceeds 128MiB cap")
    with requests.get(URL, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                      stream=True, timeout=(15, 45)) as response:
        rec = {"start": start, "end": end, "http_status": response.status_code,
               "content_range": response.headers.get("Content-Range"), "received_bytes": 0}
        report["requests"].append(rec)
        if response.status_code != 206:
            raise ValueError("Range not honored; body not read")
        if rec["content_range"] != f"bytes {start}-{end}/{index['archive_bytes']}":
            raise ValueError("Range/total size changed")
        chunks = []
        for chunk in response.iter_content(1024 * 1024):
            rec["received_bytes"] += len(chunk)
            if rec["received_bytes"] > expected:
                raise ValueError("Response exceeded range budget")
            chunks.append(chunk)
        blob = b"".join(chunks)
        if len(blob) != expected:
            raise ValueError("Incomplete range response")
        return blob


try:
    for folder, target in (("16", "MEOKDA"), ("48", "GONGBU")):
        candidates = [m for m in index["members"] if re.fullmatch(folder + r"/\d{2}_" + folder + r"\.MP4", m["name"])]
        assert len(candidates) == {"16": 17, "48": 19}[folder]
        start = min(m["local_header_offset"] for m in candidates)
        last_offset = max(m["local_header_offset"] for m in candidates)
        end = min(m["local_header_offset"] for m in index["members"] if m["local_header_offset"] > last_offset) - 1
        block = fetch(start, end)
        for member in candidates:
            position = member["local_header_offset"] - start
            signature, needed, flags, method, mtime, mdate, crc, csize, usize, nlen, xlen = struct.unpack_from("<4s5H3I2H", block, position)
            if signature != b"PK\x03\x04" or flags != 0 or method != 8:
                raise ValueError("Unsupported local header")
            if (f"{crc:08x}", csize, usize) != (member["crc32"], member["compressed_size"], member["uncompressed_size"]):
                raise ValueError("Local/central integrity fields differ")
            filename = block[position + 30:position + 30 + nlen].decode("ascii")
            if filename != member["name"] or not 0 < usize <= 16 * 1024 * 1024:
                raise ValueError("Name or uncompressed byte budget mismatch")
            compressed_start = position + 30 + nlen + xlen
            compressed = block[compressed_start:compressed_start + csize]
            if len(compressed) != csize:
                raise ValueError("Compressed payload truncated")
            decompressor = zlib.decompressobj(-15)
            data = decompressor.decompress(compressed, usize + 1)
            if len(data) != usize or not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
                raise ValueError("Deflate length/termination mismatch")
            if f"{zlib.crc32(data) & 0xffffffff:08x}" != member["crc32"]:
                raise ValueError("CRC32 mismatch")
            destination = RAW / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(data)
            prefix = Path(filename).name[:2]
            report["files"].append({"member": filename, "target_candidate": target,
                                    "source_label_id": int(folder), "source_label": labels[str(int(folder))],
                                    "source_identity_candidate": "KSL77_PREFIX_" + prefix,
                                    "identity_verified": False, "split": "UNASSIGNED",
                                    "triage_allowed": prefix == "00", "media_inspected": False,
                                    "path": destination.as_posix(), "bytes": usize,
                                    "crc32": member["crc32"], "sha256": hashlib.sha256(data).hexdigest()})
        print(json.dumps({"class": folder, "label": labels[str(int(folder))], "files_acquired": len(candidates),
                          "http_body_bytes": len(block)}, ensure_ascii=False), flush=True)
    report["status"] = "ALL_36_AVAILABLE_CANDIDATES_ACQUIRED_CRC_VERIFIED"
except Exception as error:
    report.update(status="ACQUISITION_STOPPED", error_type=type(error).__name__, error=str(error)[:600])
report["acquired_bytes"] = sum(f["bytes"] for f in report["files"])
report["network_body_bytes"] = sum(r["received_bytes"] for r in report["requests"])
(OUT / "ksl77_acquisition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k not in ("files", "requests")}, ensure_ascii=False))
