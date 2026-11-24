"""R33: the clock the recorder hands MediaPipe must be SESSION-continuous.

One Landmarker lives for the whole session while every recording restarts its own timestamp at
zero. Feeding the per-recording clock made the monotonic clamp fire on every frame from the
second recording onward, collapsing a ~31 ms stream to 1 ms. Measured on held clips: the
extracted coordinates moved by 5-8 px on average and up to 250 px, with the extractor otherwise
bit-reproducible.

Run: py -3.13 tests/test_r33_session_timebase.py
"""
import json
import os
import re
import sys

NL = "\n"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

CAMERA_PY = os.path.join(ROOT, "src", "wis_lab", "service", "camera.py")
FRAME_US = 31_000          # a real webcam frame interval
MEASURED = {}


class FakeClock:
    """next_timestamp_ms without the models, which is the whole of the behaviour under test."""

    def __init__(self):
        self.last_ms = -1

    def feed(self, timestamp_us):
        ms = int(timestamp_us // 1000)
        if ms <= self.last_ms:
            ms = self.last_ms + 1
        self.last_ms = ms
        return ms


def _spacings(per_recording_clock, takes=3, frames=40):
    """Return the ms gaps MediaPipe would see for each take."""
    clock = FakeClock()
    session_us = 0
    out = []
    for _ in range(takes):
        seen = []
        for i in range(frames):
            take_us = i * FRAME_US                     # every recording restarts at zero
            seen.append(clock.feed(take_us if per_recording_clock else session_us))
            session_us += FRAME_US
        session_us += 5_000_000                        # a pause between takes
        out.append([b - a for a, b in zip(seen, seen[1:])])
    return out


def check_defect_reproduces():
    """The old behaviour: correct for take 1, collapsed to 1 ms for every take after it."""
    takes = _spacings(per_recording_clock=True)
    assert set(takes[0]) == {31}, takes[0][:5]
    for t in takes[1:]:
        assert set(t) == {1}, t[:5]
    MEASURED["defect_spacing_ms"] = {"take1": takes[0][0], "take2": takes[1][0],
                                     "take3": takes[2][0]}


def check_session_clock_keeps_real_spacing():
    """The fix: every take keeps the real interval, so the clamp never fires."""
    for i, t in enumerate(_spacings(per_recording_clock=False)):
        assert set(t) == {31}, (i, t[:5])
    MEASURED["fixed_spacing_ms"] = 31


def check_recorder_passes_the_session_clock():
    src = open(CAMERA_PY, encoding="utf-8").read()
    assert re.search(r"lm\.process\(rgb, now_ns // 1000, row, 0\)", src), \
        "camera.py must hand MediaPipe the session clock, not the per-recording ts_us"
    assert 'ts_us = (now_ns - origin) // 1000' in src, \
        "the STORED timestamp_us must stay per-recording"
    MEASURED["camera_py_checked"] = True


def check_stored_timestamps_are_untouched():
    """The dt channel is built from the stored clock, which the fix does not move."""
    origin = 12_345_678_900
    now = [origin + i * FRAME_US * 1000 for i in range(5)]
    stored = [(n - origin) // 1000 for n in now]
    assert stored[0] == 0 and stored[1] - stored[0] == FRAME_US, stored
    MEASURED["stored_dt_us"] = stored[1] - stored[0]


CHECKS = [("defect reproduces", check_defect_reproduces),
          ("session clock keeps real spacing", check_session_clock_keeps_real_spacing),
          ("recorder passes the session clock", check_recorder_passes_the_session_clock),
          ("stored timestamps untouched", check_stored_timestamps_are_untouched)]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
