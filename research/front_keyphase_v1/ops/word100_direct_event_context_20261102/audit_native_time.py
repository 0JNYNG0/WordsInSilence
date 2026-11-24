"""Read media presentation timestamps; no image exports, features or scores."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import av
import numpy as np

U = Path(__file__).resolve().parent
O = U.parent
AE = O / "word100_guide_feature_supervision_20261102"
D = O / "word100_action_episode_supervision_20261102"
Q = O / "word100_predicate_supervision_20261102"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dst = U / "NATIVE_TIME_AUDIT.json"
    assert not dst.exists()
    queue = read(AE / "CONTEXT_REVIEW_QUEUE.json")
    media = {r["clip_id"]: r for r in read(D / "REVIEW_MANIFEST.json")["rows"]}
    sources = {r["clip_id"]: r for r in read(Q / "SUPERVISION.json")["sources"]}
    rows = []
    for packet in queue["rows"]:
        cid = packet["clip_id"]
        m = media[cid]
        assert sha(m["source"]) == m["source_sha256"]
        source = sources[cid]
        with np.load(source["input"], allow_pickle=False) as a:
            frames = a["frame_id"].copy()
            times = a["timestamp_us"].copy()
        assert np.array_equal(frames, np.arange(len(frames)))
        pts = []
        with av.open(m["source"]) as container:
            stream = container.streams.video[0]
            average_rate = str(stream.average_rate)
            base_rate = str(stream.base_rate)
            time_base = str(stream.time_base)
            for frame in container.decode(video=0):
                assert frame.pts is not None
                pts.append(round(Fraction(frame.pts) * frame.time_base * 1_000_000))
        assert len(pts) == len(frames)
        difference = np.asarray(pts, np.int64) - times
        shown = {r["frame"]: r for r in m["frames"]}
        witness = sorted({f for c in packet["claims"] for f in c["witness_frames"]})
        query_times = {}
        for c in packet["claims"]:
            for f, tt in zip(c["witness_frames"], c["witness_timestamps_us"]):
                if f in query_times:
                    assert query_times[f] == tt
                query_times[f] = tt
        details = [dict(frame=f, queue_timestamp_us=query_times[f], cached_timestamp_us=int(times[f]),
            decoded_media_pts_us=pts[f], old_sheet_seconds=shown[f]["seconds"],
            old_sheet_minus_media_us=round(shown[f]["seconds"] * 1e6) - pts[f]) for f in witness]
        rows.append(dict(clip_id=cid, source=m["source"], source_sha256=m["source_sha256"],
            native_input=source["input"], native_input_sha256=sha(source["input"]), frames=len(frames),
            average_rate=average_rate, base_rate=base_rate, time_base=time_base,
            max_abs_cache_media_difference_us=int(np.max(np.abs(difference))),
            cached_axis_matches_media=bool(np.array_equal(times, pts)),
            queue_axis_matches_cache=all(r["queue_timestamp_us"] == r["cached_timestamp_us"] for r in details),
            max_abs_sheet_display_difference_us=max(abs(r["old_sheet_minus_media_us"]) for r in details),
            witnesses=details))
    good = all(r["cached_axis_matches_media"] and r["queue_axis_matches_cache"] for r in rows)
    result = dict(status="PASS_NATIVE_AXIS_DISPLAY_FPS_DIFFERENCE_DIAGNOSED" if good else "NATIVE_AXIS_MISMATCH_REQUIRES_REPAIR_BEFORE_TRAINING",
        clips=len(rows), rows=rows, av_version=av.__version__,
        cause="Old prepare_review.py printed frame / OpenCV average FPS. Decoded presentation timestamps are the authoritative playback axis; average rate can differ due to stream duration accounting.",
        remedy="Keep immutable old sheets/results. Use native timestamp_us[frame]/1e6 for new text links/captions, after this PTS verification. Do not retime or relabel the original points.",
        new_exported_images=0, new_features=0, new_model_inference=0, new_training=0,
        original_files_modified=False, whole_goal_complete=False,
        pins={str(p): sha(p) for p in [AE / "CONTEXT_REVIEW_QUEUE.json", D / "REVIEW_MANIFEST.json", D / "prepare_review.py", Q / "SUPERVISION.json", Path(__file__)]})
    dst.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(status=result["status"], clips=len(rows),
        affected_display_clips=sum(r["max_abs_sheet_display_difference_us"] > 1 for r in rows),
        max_sheet_error_us=max(r["max_abs_sheet_display_difference_us"] for r in rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()
