"""Decode only predeclared prefix 00 candidates and existing word references.

Generate contact sheets for source triage, not lexical approval or evaluation.
All other KSL source prefixes remain media-uninspected.
"""
from pathlib import Path
import hashlib
import json

import av
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RAW = OUT.parents[1] / "runs" / "ksl77_source_review_v1" / "raw"
acq = json.loads((OUT / "ksl77_acquisition.json").read_text(encoding="utf-8"))
allowed = {f["member"]: f for f in acq["files"] if f["triage_allowed"]}
assert set(allowed) == {"16/00_16.MP4", "48/00_48.MP4"}
report = {"status": "DECODE_AND_CONTACT_SHEETS_ONLY", "reviewed_prefixes": ["00"],
          "other_prefixes_media_decoded": 0, "training_approved": False, "clips": []}


def decode(path, clip_id):
    thumbs, times, shapes = [], [], set()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        reported_frames, reported_fps = stream.frames, str(stream.average_rate)
        for frame in container.decode(stream):
            shapes.add((frame.width, frame.height))
            times.append(float(frame.pts * frame.time_base) if frame.pts is not None else None)
            thumb = frame.to_image()
            thumb.thumbnail((240, 220))
            thumbs.append(thumb)
    assert thumbs and all(t is not None for t in times)
    assert all(b > a for a, b in zip(times, times[1:]))
    indices = [round((len(thumbs) - 1) * f) for f in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)]
    report["clips"].append({"id": clip_id, "path": path.as_posix(), "frames": len(thumbs),
                            "reported_frames": reported_frames, "reported_fps": reported_fps,
                            "dimensions": sorted(shapes), "first_last_pts_s": [times[0], times[-1]],
                            "monotonic_pts": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "displayed_frame_indices": indices, "displayed_pts_s": [times[i] for i in indices]})
    return [(thumbs[i], times[i]) for i in indices]


for folder, korean, target in (("16", "먹다", "MEOKDA"), ("48", "공부", "GONGBU")):
    sheet = Image.new("RGB", (2400, 520), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row, (path, clip_id) in enumerate(((ROOT / "Tools" / "videos" / (korean + ".mp4"), "NIKL_" + target),
                                          (RAW / folder / ("00_" + folder + ".MP4"), "KSL77_00_" + folder))):
        draw.text((8, row * 260 + 4), clip_id + " - source triage only", fill="black")
        samples = decode(path, clip_id)
        for column, (thumb, timestamp) in enumerate(samples):
            x = column * 240
            sheet.paste(thumb, (x + (240 - thumb.width) // 2, row * 260 + 25))
            draw.text((x + 6, row * 260 + 244), f"{timestamp:.3f}s", fill="black")
    sheet.save(OUT / ("comparison_" + target.lower() + ".jpg"), quality=92)
(OUT / "ksl77_prefix00_decode.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
