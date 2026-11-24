"""R0 self-check. Plain asserts, no framework.

Run: py -3.13 tests/test_r0.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, N_POINTS, POSE_SLICE, RIGHT_HAND_SLICE, Observation, sha256_file,
)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


@check("layout: 33 pose + 21 left + 21 right = 75")
def _layout():
    assert POSE_SLICE.stop - POSE_SLICE.start == 33
    assert LEFT_HAND_SLICE.stop - LEFT_HAND_SLICE.start == 21
    assert RIGHT_HAND_SLICE.stop - RIGHT_HAND_SLICE.start == 21
    assert RIGHT_HAND_SLICE.stop == N_POINTS


@check("manifest: 19 clips, unique ids, every readiness assigned")
def _manifest():
    m = load("config/source_manifest.json")
    assert m["clip_count"] == 19 and len(m["clips"]) == 19
    assert len({c["clip_id"] for c in m["clips"]}) == 19
    assert len({c["word_id"] for c in m["clips"]}) == 19
    allowed = {"READY", "NEEDS_REVIEW", "UNSUPPORTED_CUE"}
    for c in m["clips"]:
        assert c["readiness"] in allowed, (c["clip_id"], c["readiness"])
        assert c["readiness_reason"], c["clip_id"]


@check("sources unchanged: on-disk sha256 still matches the manifest")
def _sources():
    for c in load("config/source_manifest.json")["clips"]:
        assert os.path.exists(c["absolute_path"]), c["absolute_path"]
        assert sha256_file(c["absolute_path"]) == c["sha256"], c["clip_id"]


@check("legacy json declared incompatible, not used as a cache")
def _legacy():
    m = load("config/source_manifest.json")
    for c in m["clips"]:
        assert c["legacy_contract_match"] is False
        assert c["legacy_json_path"], c["clip_id"]
        assert c["observation_npz"].startswith("cache/"), c["clip_id"]


@check("every observation loads, validates and matches its recorded sha256")
def _observations():
    for c in load("config/source_manifest.json")["clips"]:
        npz = os.path.join(ROOT, c["observation_npz"])
        assert sha256_file(npz) == c["observation_sha256"], c["clip_id"]
        obs = Observation.load(npz)  # .validate() runs inside
        assert obs.n_frames == c["frame_count"], (c["clip_id"], obs.n_frames, c["frame_count"])


@check("missing points are NaN, never a valid-looking (0,0)")
def _missing():
    for c in load("config/source_manifest.json")["clips"][:5]:
        obs = Observation.load(os.path.join(ROOT, c["observation_npz"]))
        missing = ~obs.observed
        assert np.isnan(obs.xy_px[missing]).all(), c["clip_id"]
        assert not (obs.xy_px[missing] == 0).any(), c["clip_id"]


@check("timestamps come from PTS: strictly increasing and off the 33 ms grid")
def _timestamps():
    off_grid = 0
    for c in load("config/source_manifest.json")["clips"]:
        obs = Observation.load(os.path.join(ROOT, c["observation_npz"]))
        assert (np.diff(obs.timestamp_us) > 0).all(), c["clip_id"]
        assert obs.timestamp_us[0] == 0, c["clip_id"]
        if (obs.timestamp_us % 1000 != 0).any():
            off_grid += 1
    assert off_grid > 0, "every timestamp landed on whole ms; PTS may not be preserved"


@check("hand points only appear on a side that also has a hand score")
def _hand_score_consistency():
    for c in load("config/source_manifest.json")["clips"]:
        obs = Observation.load(os.path.join(ROOT, c["observation_npz"]))
        for side, sl in ((0, LEFT_HAND_SLICE), (1, RIGHT_HAND_SLICE)):
            has_pts = obs.observed[:, sl].any(axis=1)
            has_score = np.isfinite(obs.hand_score[:, side])
            assert (has_pts == has_score).all(), (c["clip_id"], side)


@check("subject is not mirrored: pose left shoulder sits right of pose right shoulder")
def _not_mirrored():
    total = flipped = 0
    for c in load("config/source_manifest.json")["clips"]:
        obs = Observation.load(os.path.join(ROOT, c["observation_npz"]))
        ok = obs.usable[:, 11] & obs.usable[:, 12]
        dx = obs.xy_px[ok, 11, 0] - obs.xy_px[ok, 12, 0]
        total += int(ok.sum())
        flipped += int((dx <= 0).sum())
    assert total > 0
    assert flipped == 0, f"{flipped}/{total} frames contradict the un-mirrored assumption"


@check("readiness is evidence-backed, not a blanket READY")
def _readiness_honest():
    m = load("config/source_manifest.json")
    counts = m["readiness_counts"]
    assert counts["PENDING_EXTRACTION"] == 0
    assert sum(counts.values()) == 19
    # every NEEDS_REVIEW / UNSUPPORTED_CUE must name a measured number or a declared cue flag
    for c in m["clips"]:
        if c["readiness"] == "READY":
            continue
        reason = c["readiness_reason"]
        assert ("cue" in reason or "<" in reason or "head" in reason), (c["clip_id"], reason)


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
