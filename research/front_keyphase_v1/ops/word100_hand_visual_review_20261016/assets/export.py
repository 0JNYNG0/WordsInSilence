"""Export saved hand evidence and decoded video frames; never run a detector.

Run from any directory with Python -B. All mutations are confined to this unit.
Hand coordinates remain in the exact native/crop detector-input pixel space.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import av
import numpy as np
from PIL import Image

UNIT = Path(__file__).resolve().parents[1]
OPS = UNIT.parent
JOINT = OPS / "word100_hand_assignment_joint_20261016"
PUBLIC = UNIT / "public"
CASES = [
    ("sobok", "소복", "NIA_SL_WORD0651_REAL03_F", "crop", 57),
    ("believe-native", "믿다", "NIA_SL_WORD1139_REAL01_F", "native", 71),
    ("believe-crop", "믿다", "NIA_SL_WORD1139_REAL01_F", "crop", 56),
    ("believe-entry", "믿다", "NIA_SL_WORD1139_REAL01_F", "crop", 46),
    ("coldcoffee", "냉커피", "NIA_SL_WORD0106_REAL01_F", "crop", 52),
    ("reception", "응접실", "NIA_SL_WORD0766_REAL03_F", "crop", 46),
    ("eighty-nine", "팔십구", "NIA_SL_WORD2721_REAL02_F", "crop", 43),
    ("seongnam", "성남시수어통역센터", "NIA_SL_WORD1675_REAL03_F", "crop", 42),
    ("joy", "즐겁다", "NIA_SL_WORD1140_REAL01_F", "crop", 94),
    ("seasons", "사계절", "NIA_SL_WORD2572_REAL01_F", "crop", 34),
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value):
    """JSON-safe scalar conversion without rounding finite evidence values."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def main():
    evidence_path = UNIT / "EVIDENCE_EXPORT.json"
    data_path = PUBLIC / "data.js"
    if evidence_path.exists() or data_path.exists():
        raise RuntimeError("Completed or partial export already exists; do not overwrite evidence")
    pins = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Frozen input hash mismatch: {path}")
        if str(path) in pins and pins[str(path)] != actual:
            raise RuntimeError(f"Input changed during export: {path}")
        pins[str(path)] = actual
        return path

    def read_json(path):
        path = pin(path)
        return json.loads(path.read_text(encoding="utf-8-sig"))

    pin(Path(__file__))
    result = read_json(JOINT / "REPLAY_RESULT.json")
    rows = {(x["clip_id"], x["mode"]): x for x in result["rows"]}
    input_cases = []
    all_sources = {}
    for cid, label, clip, mode, target in CASES:
        row = rows[clip, mode]
        raw_dir = Path(row["raw_dir"])
        sidecar = read_json(raw_dir / "raw.sidecar.json")
        raw_path = pin(raw_dir / "raw.npz", sidecar["raw_sha256"])
        with np.load(raw_path, allow_pickle=False) as source:
            raw = {k: source[k].copy() for k in source.files}
        source_path = pin(sidecar["source_path"], sidecar["source_sha256"])
        replay = JOINT / "replay" / clip / mode
        assignments = {}
        for filename, key in [("assignment.npz", "current"), ("previous_assignment.npz", "previous")]:
            path = replay / filename
            expected = next(v for k, v in row["outputs"].items() if Path(k).resolve() == path.resolve())
            pin(path, expected)
            with np.load(path, allow_pickle=False) as source:
                assignments[key] = source["sides"].copy()
        trace_path = replay / "trace.json"
        pin(trace_path, next(v for k, v in row["outputs"].items() if Path(k).resolve() == trace_path.resolve()))
        traces = read_json(trace_path)
        count = len(raw["raw_count"])
        assert len(traces) == count
        assert assignments["current"].shape == assignments["previous"].shape == raw["baseline_side"].shape
        assert 0 <= target < count
        start, end = max(0, target - 12), min(count - 1, target + 12)
        assert np.all(np.diff(raw["timestamp_us"]) > 0)
        frameids = raw["frame_id"][start:end + 1]
        assert len(set(map(int, frameids))) == len(frameids)
        width, height = int(sidecar["width"]), int(sidecar["height"])
        assert np.all(raw["width"] == width) and np.all(raw["height"] == height)
        box = sidecar["box"] if mode == "crop" else None
        if box is not None:
            assert box[2] - box[0] == width and box[3] - box[1] == height
        packed = dict(id=cid, label=label, clip=clip, mode=mode, target=target,
                      raw=raw, assignments=assignments, traces=traces, sidecar=sidecar,
                      start=start, end=end, width=width, height=height, source=source_path,
                      box=box, frames=[])
        input_cases.append(packed)
        all_sources.setdefault(source_path, []).append(packed)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "frames").mkdir(exist_ok=True)
    image_evidence = []
    decode_evidence = []

    # Same PTS and frame-id rules as the recorded video_source.extract, copied
    # here to avoid importing its MediaPipe-dependent Landmarker module.
    for source_path, cases in all_sources.items():
        requests = {}
        for case in cases:
            for i in range(case["start"], case["end"] + 1):
                requests.setdefault(int(case["raw"]["frame_id"][i]), []).append((case, i))
        delivered = set()
        with av.open(str(source_path)) as container:
            stream = container.streams.video[0]
            tb = stream.time_base or Fraction(1, 1000)
            origin_pts, last_us, row, dropped = None, -1, 0, 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is None:
                        continue
                    if origin_pts is None:
                        origin_pts = frame.pts
                    timestamp_us = int(round(float((frame.pts - origin_pts) * tb) * 1_000_000))
                    if timestamp_us <= last_us:
                        dropped += 1
                        continue
                    last_us = timestamp_us
                    frame_id = row + dropped
                    row += 1
                    if frame_id not in requests:
                        continue
                    rgb = frame.to_ndarray(format="rgb24")
                    for case, i in requests[frame_id]:
                        raw = case["raw"]
                        assert timestamp_us == int(raw["timestamp_us"][i])
                        assert i == row - 1, "Saved row and decoded row differ"
                        pixels = rgb
                        if case["box"] is not None:
                            x0, y0, x1, y1 = case["box"]
                            pixels = pixels[y0:y1, x0:x1]
                        assert pixels.shape[:2] == (case["height"], case["width"])
                        photo = Image.fromarray(pixels)
                        image_width = min(960, photo.width)
                        image_height = round(photo.height * image_width / photo.width)
                        if image_width != photo.width:
                            photo = photo.resize((image_width, image_height), Image.Resampling.LANCZOS)
                        filename = f"{case['id']}_{frame_id:05d}.jpg"
                        output = PUBLIC / "frames" / filename
                        if output.exists():
                            raise RuntimeError(f"Refusing to overwrite image: {output}")
                        photo.save(output, format="JPEG", quality=90, subsampling=0, optimize=True)
                        case["frames"].append(dict(row=i, index=frame_id, timeSeconds=timestamp_us / 1_000_000,
                                                   timestampUs=timestamp_us, image=f"frames/{filename}",
                                                   imageWidth=image_width, imageHeight=image_height))
                        image_evidence.append(dict(path=str(output.relative_to(UNIT)), sha256=sha(output),
                                                   caseId=case["id"], frameId=frame_id, timestampUs=timestamp_us,
                                                   inputSize=[case["width"], case["height"]], jpegSize=[image_width, image_height],
                                                   cropBox=case["box"]))
                    delivered.add(frame_id)
        assert delivered == set(requests)
        decode_evidence.append(dict(source=str(source_path), framesDecoded=row, nonmonotonicDropped=dropped,
                                    selectedDistinctFrames=len(delivered), timestampAndFrameIdExact=True))
        print(json.dumps({"source": source_path.name, "exported_case_frames": sum(len(c["frames"]) for c in cases)}, ensure_ascii=False), flush=True)

    exported = []
    for case in input_cases:
        raw = case["raw"]
        expected_rows = list(range(case["start"], case["end"] + 1))
        assert [f["row"] for f in case["frames"]] == expected_rows
        zoom_points = []
        for f in case["frames"]:
            i = f["row"]
            trace = case["traces"][i]
            assert trace["frame"] == i and trace["timestamp_us"] == f["timestampUs"]
            detections = []
            for j in range(int(raw["raw_count"][i])):
                points = raw["hand_xy_px"][i, j]
                detections.append(dict(index=j, label=str(raw["raw_label_name"][i, j]),
                                       labelCode=int(raw["raw_label"][i, j]), score=float(raw["raw_score"][i, j]),
                                       points=clean(points), baselineSide=int(raw["baseline_side"][i, j]),
                                       previousSide=int(case["assignments"]["previous"][i, j]),
                                       currentSide=int(case["assignments"]["current"][i, j]),
                                       reason=trace["reasons"][j]))
                zoom_points.extend(points[np.isfinite(points).all(axis=1)].tolist())
            wrists = []
            for side, point_index in [(0, 15), (1, 16)]:
                point = raw["pose_xy_px"][i, point_index]
                usable = bool(raw["pose_usable"][i, point_index])
                wrists.append(dict(side=side, point=clean(point), usable=usable,
                                   observed=bool(raw["pose_observed"][i, point_index]),
                                   confidence=float(raw["pose_conf"][i, point_index])))
                if usable and np.isfinite(point).all():
                    zoom_points.append(point.tolist())
            f["raw"] = detections
            f["poseWrists"] = wrists
            f["trace"] = clean(trace)
        width, height = case["width"], case["height"]
        if zoom_points:
            arr = np.clip(np.asarray(zoom_points), [0, 0], [width, height])
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            center = (lo + hi) / 2
            size = np.maximum(hi - lo + [width * .12, height * .12], [width * .4, height * .3])
            size = np.minimum(size, [width, height])
            origin = np.clip(center - size / 2, [0, 0], np.array([width, height]) - size)
            zoom = np.concatenate((origin, size)).tolist()
        else:
            zoom = [0, 0, width, height]
        exported.append(dict(id=case["id"], label=case["label"], clipId=case["clip"], mode=case["mode"],
                             targetFrame=int(raw["frame_id"][case["target"]]), targetRow=case["target"],
                             width=width, height=height, totalClipFrames=len(raw["raw_count"]),
                             sourceCropBox=case["box"], frames=case["frames"], zoomBox=zoom))
    dataset = dict(schema="wis_hand_visual_review_v1", createdUtc=datetime.now(timezone.utc).isoformat(),
                   coordinateSpace="Original native/crop detector-input pixels, before JPEG resize; crop origin already applied in raw cache",
                   sideMapping={"-1": "unassigned", "0": "anatomical left", "1": "anatomical right"},
                   scoreMeaning="MediaPipe handedness label score; not joint accuracy or word recognition probability",
                   originalFrameIndexing="zero-based stored frame_id; original PTS microseconds retained",
                   stages={"baseline": "production", "previous": "repair_v2", "current": "joint_v3"}, cases=exported)
    text = "window.HAND_REVIEW = " + json.dumps(clean(dataset), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n"
    with data_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    after = {path: sha(Path(path)) for path in pins}
    assert after == pins, "A source or saved result changed during export"
    evidence = dict(status="EXPORTED_EXISTING_EVIDENCE_NO_NEW_DETECTION", createdUtc=datetime.now(timezone.utc).isoformat(),
                    exporter="Codex raw_capture", cases=len(exported), caseFrames=len(image_evidence),
                    uniqueSourceVideos=len(all_sources), sourcePins=pins, inputHashesUnchanged=True,
                    decode=decode_evidence, images=image_evidence, dataJsSha256=sha(data_path),
                    rawCoordinatesExact=True, timestampAndFrameIdExact=True,
                    jpegTransform="Exact prior crop if crop mode, then width<=960 proportional resize and JPEG encoding only; no generated or edited content",
                    prohibitedActionsPerformed=dict(detector=0, training=0, classifierInference=0, downloads=0,
                                                     productionWrites=0, priorUnitWrites=0, sourceWrites=0))
    with evidence_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"status": evidence["status"], "cases": len(exported), "frames": len(image_evidence), "inputPins": len(pins)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        failure = UNIT / "EXPORT_FAILURE.json"
        if not failure.exists():
            failure.write_text(json.dumps({"firstError": repr(error), "atUtc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        raise
