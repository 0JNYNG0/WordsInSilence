"""Validated integration entry point for a new analysis result.

The frozen v1 saved comparison remains unchanged. This wrapper checks input
identity and source uniqueness before calling it, without doing model inference.
"""
from numbers import Integral
import re
from guide_comparison import GuideComparator


class RuntimeGuideComparator:
    def __init__(self):
        self.comparator = GuideComparator()

    def preflight(self, result, frame_ids, timestamps_us, *, expected_word, expected_source_sha256):
        if expected_word not in {c["word_id"] for c in self.comparator.constraints}:
            raise ValueError("Target expression is not supported by this source contract")
        if not isinstance(expected_source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
            raise ValueError("A verified source video SHA256 is required")
        if result.get("word_id") != expected_word or result.get("source_sha256") != expected_source_sha256:
            raise ValueError("The analysis does not belong to the requested word/video")
        frames = list(frame_ids)
        times = list(timestamps_us)
        if not frames or len(frames) != len(times) or any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 for v in frames + times):
            raise ValueError("Invalid native integer axis")
        if any(b <= a for a, b in zip(frames, frames[1:])) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Nonmonotone native axis")
        if result.get("frames") != len(frames):
            raise ValueError("Native axis and analysis frame count differ")
        ids = set()
        for component in result.get("source_components", []):
            cid = component["component_id"]
            if cid in ids:
                raise ValueError("Duplicated observation component")
            ids.add(cid)
            clauses = component["source_clauses"]
            names = [c["id"] for c in clauses]
            if len(set(names)) != len(names):
                raise ValueError("Duplicated source clause within a component")
        return dict(word_id=expected_word, source_sha256=expected_source_sha256,
            native_frames=len(frames), source_components=len(ids), identity_checked=True)

    def attach(self, result, frame_ids, timestamps_us, *, expected_word, expected_source_sha256):
        checked = self.preflight(result, frame_ids, timestamps_us,
            expected_word=expected_word, expected_source_sha256=expected_source_sha256)
        out = self.comparator.attach(result, frame_ids, timestamps_us)
        out["guide_comparison_input_identity"] = checked
        return out
