"""Extract one exact, train-side local sentence member; no new download/train."""
from pathlib import Path
import hashlib
import json
import zipfile
import zlib

import av
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
rows = json.loads((OUT / "candidate_rows.json").read_text(encoding="utf-8"))["rows"]
selected = [r for r in rows if r["target_candidate_not_approved"] == "JOESONGHADA" and r["signer_id"] == 1]
assert len(selected) == 1
source = selected[0]
assert source["split_role"] == "train_candidate" and source["training_approved"] is False
catalog = json.loads((OUT / "catalog.json").read_text(encoding="utf-8"))
member = source["local_video_archive_match"]
destination = OUT.parents[1] / "runs/apology_source_review_v1/raw" / source["source_video_basename"]
destination.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(catalog["video_archive"]) as archive:
    info = archive.getinfo(member["member"])
    assert info.file_size == member["bytes"] and info.file_size < 16 * 1024 * 1024
    data = archive.read(info)
assert len(data) == member["bytes"] and f"{zlib.crc32(data) & 0xffffffff:08x}" == member["crc32"]
with destination.open("xb") as handle:
    handle.write(data)
checksum = hashlib.sha256(data).hexdigest()
assert hashlib.sha256(destination.read_bytes()).hexdigest() == checksum
report = {"status": "PASS_EXACT_MEMBER_AND_SAVED_BYTES_NOT_SEMANTIC_APPROVAL", "source": source,
          "source_archive": catalog["video_archive"], "path": destination.as_posix(),
          "bytes": len(data), "crc32": member["crc32"], "sha256": checksum,
          "new_downloads": 0, "training_approved": False, "other_signers_media_read": 0, "clips": []}
sheet = Image.new("RGB", (2240, 570), "#eeeeee")
draw = ImageDraw.Draw(sheet)
reference = ROOT / "Tools/videos/죄송하다.mp4"
assert reference.exists()
for row, (path, clip_id, window) in enumerate(((reference, "NIKL_JOESONGHADA", None),
                                             (destination, source["clip_id"], (source["annotation_start_s"], source["annotation_end_s"])))):
    frames, times = [], []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_image()
            image.thumbnail((280, 235))
            frames.append(image)
            times.append(float(frame.pts * frame.time_base))
    assert times and all(b > a for a, b in zip(times, times[1:]))
    if window is None:
        wanted = [times[0] + (times[-1] - times[0]) * (0.1 + 0.8 * i / 7) for i in range(8)]
    else:
        wanted = [window[0] + (window[1] - window[0]) * i / 7 for i in range(8)]
    indices = [min(range(len(times)), key=lambda n: abs(times[n] - t)) for t in wanted]
    draw.text((8, row * 285 + 3), clip_id + " - static source triage only", fill="black")
    for col, i in enumerate(indices):
        thumb = frames[i]
        sheet.paste(thumb, (col * 280 + (280 - thumb.width) // 2, row * 285 + 23))
        draw.text((col * 280 + 6, row * 285 + 267), f"frame {i} {times[i]:.3f}s", fill="black")
    report["clips"].append({"id": clip_id, "path": path.as_posix(), "frames": len(times),
                            "first_last_pts_s": [times[0], times[-1]], "monotonic_pts": True,
                            "display_indices": indices, "display_times_s": [times[i] for i in indices],
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
sheet.save(OUT / "apology_comparison.jpg", quality=93)
(OUT / "apology_acquisition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in report.items() if k not in ("source",)}, ensure_ascii=False))
