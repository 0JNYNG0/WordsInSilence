"""Read the public author-linked ZIP directory with bounded anonymous HTTP ranges.

Never fall back to downloading the full archive when a server ignores Range.
This is source acquisition evidence, not training or model code.
"""
from pathlib import Path
import hashlib
import json
import re
import struct

import requests

OUT = Path(__file__).resolve().parent
URL = "https://www.dropbox.com/scl/fi/daqmxyxycgr4r39lzn7gf/video.zip?rlkey=uphg8vs7n4mrdgl7bzi62y0on&dl=1"
report = {"probe_version": 2, "repository": "https://github.com/Yangseung/KSL", "request_url": URL,
          "credentials_used": False, "full_archive_download_allowed": False,
          "training_approved": False, "requests": []}


def bounded_range(range_value, maximum):
    with requests.get(URL, headers={"Range": range_value, "Accept-Encoding": "identity"},
                      stream=True, timeout=(15, 40)) as response:
        record = {"range": range_value, "status": response.status_code,
                  "content_range": response.headers.get("Content-Range"),
                  "content_length": response.headers.get("Content-Length"),
                  "content_type": response.headers.get("Content-Type"), "body_bytes": 0}
        report["requests"].append(record)
        if response.status_code != 206:
            raise ValueError("Range not honored; body not read, no full-download fallback")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", record["content_range"] or "")
        if not match:
            raise ValueError("Missing/invalid Content-Range")
        start, end, total = map(int, match.groups())
        expected = end - start + 1
        if expected > maximum or expected <= 0:
            raise ValueError("Oversize or invalid returned range")
        if range_value.startswith("bytes=-"):
            if end != total - 1 or start != max(0, total - int(range_value[7:])):
                raise ValueError("Incorrect suffix range returned")
        elif range_value != f"bytes={start}-{end}":
            raise ValueError("Requested range differs from returned range")
        parts = []
        for part in response.iter_content(65536):
            record["body_bytes"] += len(part)
            if record["body_bytes"] > maximum:
                raise ValueError("Body exceeds byte budget")
            parts.append(part)
        data = b"".join(parts)
        if len(data) != expected:
            raise ValueError("Truncated response")
        return data, start, total


try:
    tail, tail_start, total = bounded_range("bytes=-65557", 65557)
    (OUT / "ksl77_zip_tail.bin").write_bytes(tail)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("ZIP end-of-directory record missing")
    signature, disk, cd_disk, disk_count, count, size, offset, comment_length = struct.unpack_from("<4s4H2IH", tail, eocd)
    if disk != 0 or cd_disk != 0 or count != disk_count:
        raise ValueError("Multi-volume ZIP is not supported by this bounded reader")
    if eocd + 22 + comment_length != len(tail):
        raise ValueError("Invalid ZIP end record")
    if count == 65535 or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        locator_sig, locator_disk, record_offset, disk_total = struct.unpack_from("<4sLQL", tail, eocd - 20)
        if locator_sig != b"PK\x06\x07" or locator_disk != 0 or disk_total != 1:
            raise ValueError("Invalid or multi-volume ZIP64 locator")
        if not tail_start <= record_offset <= tail_start + eocd - 76:
            raise ValueError("ZIP64 end record is outside bounded tail")
        record = struct.unpack_from("<4sQ2H2L4Q", tail, record_offset - tail_start)
        sig64, length64, made64, needed64, disk64, cd_disk64, disk_count64, count, size, offset = record
        if sig64 != b"PK\x06\x06" or disk64 != 0 or cd_disk64 != 0 or disk_count64 != count:
            raise ValueError("Invalid or split ZIP64 end record")
        if record_offset + 12 + length64 != tail_start + eocd - 20:
            raise ValueError("ZIP64 record length does not reach locator")
        report["zip64"] = True
    if not 0 < size <= 16 * 1024 * 1024 or offset + size > tail_start + eocd:
        raise ValueError("Invalid or oversized ZIP central directory")
    directory, _, total_again = bounded_range(f"bytes={offset}-{offset + size - 1}", size)
    if total_again != total:
        raise ValueError("Archive length changed between requests")
    members, position = [], 0
    for _ in range(count):
        values = struct.unpack_from("<4s6H3I5H2I", directory, position)
        if values[0] != b"PK\x01\x02":
            raise ValueError("Invalid central-directory signature")
        _, made_by, needed, flags, method, mtime, mdate, crc, csize, usize, name_len, extra_len, comment_len, member_disk, internal, external, local_offset = values
        name_bytes = directory[position + 46:position + 46 + name_len]
        name = name_bytes.decode("utf-8" if flags & 0x800 else "cp437")
        extra = directory[position + 46 + name_len:position + 46 + name_len + extra_len]
        extra_pos, extra64 = 0, None
        while extra_pos < len(extra):
            kind, extra_size = struct.unpack_from("<HH", extra, extra_pos)
            payload = extra[extra_pos + 4:extra_pos + 4 + extra_size]
            if len(payload) != extra_size:
                raise ValueError("Truncated extra field")
            if kind == 1:
                extra64 = payload
            extra_pos += 4 + extra_size
        if 0xFFFFFFFF in (usize, csize, local_offset) or member_disk == 0xFFFF:
            if extra64 is None:
                raise ValueError("ZIP64 member extra field missing")
            extra64_pos = 0
            expanded = []
            for value, sentinel, fmt in ((usize, 0xFFFFFFFF, "Q"), (csize, 0xFFFFFFFF, "Q"),
                                         (local_offset, 0xFFFFFFFF, "Q"), (member_disk, 0xFFFF, "L")):
                if value == sentinel:
                    value, = struct.unpack_from("<" + fmt, extra64, extra64_pos)
                    extra64_pos += struct.calcsize("<" + fmt)
                expanded.append(value)
            usize, csize, local_offset, member_disk = expanded
        if member_disk != 0:
            raise ValueError("Split member is unsupported")
        members.append({"name": name, "compressed_size": csize, "uncompressed_size": usize,
                        "crc32": f"{crc:08x}", "compression_method": method,
                        "flags": flags, "local_header_offset": local_offset})
        position += 46 + name_len + extra_len + comment_len
    if position != len(directory):
        raise ValueError("Unparsed central-directory bytes")
    (OUT / "ksl77_zip_directory.bin").write_bytes(directory)
    index = {"archive_bytes": total, "central_directory_offset": offset,
             "central_directory_bytes": size, "central_directory_sha256": hashlib.sha256(directory).hexdigest(),
             "members": members}
    (OUT / "ksl77_zip_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report.update(status="ZIP_DIRECTORY_VERIFIED", archive_bytes=total, member_count=len(members),
                  directory_bytes=size, sample_names=[m["name"] for m in members[:20]])
except Exception as error:
    report.update(status="BOUNDED_ZIP_INSPECTION_STOPPED", error_type=type(error).__name__, error=str(error)[:600])
(OUT / "ksl77_zip_probe_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
