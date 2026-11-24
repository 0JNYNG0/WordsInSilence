"""Decide the handedness fallback direction from data, not from doc folklore.

Counts only the unambiguous case: both pose wrists usable AND two hands detected.
Run: py -3.13 tests/diag_handedness.py
"""
import json
import os
import sys

import av
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.capture.landmarker import Landmarker  # noqa: E402
from wis_lab.core.observation import POSE_L_WRIST, POSE_R_WRIST  # noqa: E402

CLIPS = ["감사합니다", "사랑", "인사", "공부"]
VIDEO_DIR = os.path.join(ROOT, "..", "..", "Tools", "videos")


def main() -> int:
    contract = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
    agree_raw = agree_swapped = frames = 0
    for name in CLIPS:
        path = os.path.abspath(os.path.join(VIDEO_DIR, name + ".mp4"))
        with Landmarker(contract) as lm, av.open(path) as container:
            stream = container.streams.video[0]
            h, w = stream.codec_context.height, stream.codec_context.width
            ms = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    ms += 33
                    rgb = frame.to_ndarray(format="rgb24")
                    from mediapipe import Image, ImageFormat
                    img = Image(image_format=ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                    pose = lm.pose.detect_for_video(img, ms)
                    hands = lm.hand.detect_for_video(img, ms)
                    if not pose.pose_landmarks or len(hands.hand_landmarks) != 2:
                        continue
                    pl = pose.pose_landmarks[0]
                    if min(pl[POSE_L_WRIST].visibility, pl[POSE_R_WRIST].visibility) < 0.5:
                        continue
                    ref = {0: np.array([pl[POSE_L_WRIST].x * w, pl[POSE_L_WRIST].y * h]),
                           1: np.array([pl[POSE_R_WRIST].x * w, pl[POSE_R_WRIST].y * h])}
                    wr = [np.array([hands.hand_landmarks[i][0].x * w,
                                    hands.hand_landmarks[i][0].y * h]) for i in range(2)]
                    # unambiguous pairing: the assignment with the smaller total distance
                    straight = np.linalg.norm(wr[0] - ref[0]) + np.linalg.norm(wr[1] - ref[1])
                    crossed = np.linalg.norm(wr[0] - ref[1]) + np.linalg.norm(wr[1] - ref[0])
                    side = [0, 1] if straight <= crossed else [1, 0]
                    labels = [hands.handedness[i][0].category_name for i in range(2)]
                    raw = [0 if lab == "Left" else 1 for lab in labels]
                    swapped = [0 if lab == "Right" else 1 for lab in labels]
                    frames += 2
                    agree_raw += sum(int(raw[i] == side[i]) for i in range(2))
                    agree_swapped += sum(int(swapped[i] == side[i]) for i in range(2))
        print(f"{name}: running totals frames={frames} raw={agree_raw} swapped={agree_swapped}")

    assert frames > 0, "no two-hand frames found; cannot decide"
    print(f"\ntwo-hand joint decisions: {frames}")
    print(f"  raw mediapipe label agrees with pose anatomy : {agree_raw} ({agree_raw / frames:.3f})")
    print(f"  swapped label agrees with pose anatomy       : {agree_swapped} ({agree_swapped / frames:.3f})")
    winner = "raw" if agree_raw > agree_swapped else "swapped"
    print(f"  => fallback should use the {winner.upper()} MediaPipe handedness label")
    assert max(agree_raw, agree_swapped) / frames > 0.8, "neither direction is clearly better"
    return 0


if __name__ == "__main__":
    sys.exit(main())
