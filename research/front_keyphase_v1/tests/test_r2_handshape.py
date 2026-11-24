"""Stage 2 self-check: the per-axis check is separate from the ranking and honest.

2026-09-26 (observed_difference_only_v1): this suite used to require IN_BAND / OUT_OF_BAND
against ranges from one participant's attempts. Those ranges are no longer read, so the checks
that asserted the in/out answer now assert what replaced it: the observed distance itself, its
support, independence from the ranking and from every personal-range key in the contract, and
the same treatment for every word. The historical band file is still checked for its lineage,
because it is kept as a record.

Run: py -3.13 tests/test_r2_handshape.py
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from wis_lab.cli import load_reference_set  # noqa: E402
from wis_lab.core.observation import (  # noqa: E402
    LEFT_HAND_SLICE, RIGHT_HAND_SLICE, Observation,
)
from wis_lab.core import features as F  # noqa: E402
from wis_lab.core.recognize import axis_check, predict  # noqa: E402

CONTRACT = json.load(open(os.path.join(ROOT, "config", "contract.json"), encoding="utf-8"))
MANIFEST = json.load(open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8"))
BANDS = json.load(open(os.path.join(ROOT, "config", "handshape_bands.json"), encoding="utf-8"))
REFS, _ = load_reference_set()
CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def obs_for(word_id):
    clip = next(c for c in MANIFEST["clips"] if c["word_id"] == word_id)
    return Observation.load(os.path.join(ROOT, clip["observation_npz"]))


def clone(o):
    return Observation(xy_px=o.xy_px.copy(), timestamp_us=o.timestamp_us.copy(),
                       frame_id=o.frame_id.copy(), observed=o.observed.copy(),
                       usable=o.usable.copy(), conf=o.conf.copy(),
                       hand_score=o.hand_score.copy(), sidecar=copy.deepcopy(o.sidecar))


COVERED = BANDS["covered_words"]
AXES = BANDS["axes"]


BAND_KEYS = ("band_max", "band_n", "band_observed", "headroom", "band_verdict")


@check("a reference against itself is observed at distance 0, with no band attached")
def _identity():
    for w in COVERED:
        hs = predict(obs_for(w), REFS, CONTRACT)["handshape_check"]
        # status may add _PARTIAL_SUPPORT under runtime v2, which is correct for a one-handed
        # sign - the resting hand's group is never compared
        assert hs["status"] in ("OBSERVED", "OBSERVED_PARTIAL_SUPPORT"), (w, hs["status"])
        assert hs["distance"] == 0.0, (w, hs["distance"])
        assert hs["is_a_verdict"] is False
        assert not any(k in hs for k in BAND_KEYS), (w, sorted(hs))


@check("a substituted hand shape moves the observed handshape distance away from 0")
def _wrong_shape():
    donor = obs_for("SARANG")
    flagged = 0
    for w in ("GAMSAHAMNIDA", "INSA", "JADA"):
        o = clone(obs_for(w))
        n = min(o.n_frames, donor.n_frames)
        for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
            base = sl.start
            for t in range(n):
                if not (o.observed[t, base] and donor.observed[t, base]):
                    continue
                local = donor.xy_px[t, base + 1:base + 21] - donor.xy_px[t, base]
                o.xy_px[t, base + 1:base + 21] = o.xy_px[t, base] + local
                o.observed[t, base + 1:base + 21] = donor.observed[t, base + 1:base + 21]
                o.usable[t, base + 1:base + 21] = donor.usable[t, base + 1:base + 21]
        o.xy_px[~o.observed] = np.nan
        # The old check counted OUT_OF_BAND against a personal limit. No limit exists now, so
        # the measurable part is that the substitution shows up on the axis against the clip's
        # own word, where the untouched clip is exactly 0 (see _identity). This says nothing
        # about how large a difference would matter.
        hs = predict(o, REFS, CONTRACT)["handshape_check"]
        q = F.uniform_time_sample(F.sequence_features(o), int(CONTRACT["sampling_A"]["budget"]))
        own = axis_check(q, w, REFS, CONTRACT, "handshape")
        if own.get("distance") is not None and own["distance"] > 0.0:
            flagged += 1
        assert not any(k in hs for k in BAND_KEYS), sorted(hs)
    assert flagged >= 2, f"only {flagged}/3 substituted-shape inputs moved the handshape axis"


@check("the per-axis check ignores every personal-range key and never changes the ranking")
def _ranking_untouched():
    no_stage2 = dict(CONTRACT)
    no_stage2["handshape_check"] = {k: v for k, v in CONTRACT["handshape_check"].items()
                                    if k != "axes"}
    flag_off = copy.deepcopy(CONTRACT)
    flag_off["handshape_check"]["enabled"] = False
    for w in COVERED:
        o = obs_for(w)
        a = predict(o, REFS, CONTRACT)
        for other in (no_stage2, flag_off):
            b = predict(o, REFS, other)
            assert [r["word_id"] for r in a["ranking"]] == [r["word_id"] for r in b["ranking"]], w
            assert a["top1_distance"] == b["top1_distance"], w
            # same output with or without the personal ranges: they are not read at all
            assert a["axis_checks"] == b["axis_checks"], w


@check("a word that never had a personal range gets the same observed check as any other")
def _no_band():
    uncovered = BANDS["uncovered_words"]
    assert uncovered, "expected some words to have had no band"
    w = uncovered[0]
    hs = predict(obs_for(w), REFS, CONTRACT)["handshape_check"]
    assert hs["status"] in ("OBSERVED", "OBSERVED_PARTIAL_SUPPORT"), (w, hs)
    assert hs["distance"] is not None and hs["support"]["measured_groups"], (w, hs)
    assert not any(k in hs for k in BAND_KEYS), (w, sorted(hs))


@check("stage 2 does not turn into a pass line")
def _not_a_pass_line():
    res = predict(obs_for(COVERED[0]), REFS, CONTRACT)
    assert res["execution_assessment"] == "UNCONFIGURED"
    assert res["accepted"] is None
    assert res["recognition"] == "UNCONFIGURED"
    assert "neither on that axis nor on the performance as a whole" in res[
        "execution_assessment_note"]


@check("the band file carries its lineage and its caveats")
def _lineage():
    for key in ("source_session", "contract_id", "reference_set_id", "extractor_id",
                "nothing_was_trained", "one_participant", "no_frozen_retest"):
        assert BANDS.get(key), key
    for w, b in BANDS["bands"].items():
        assert b["n"] >= 10, (w, b["n"])
        assert len(b["trial_sha256"]) == b["n"], w
        assert b["lo"] < b["hi"], w


@check("an observation with no usable hands never reaches a handshape verdict")
def _no_hands():
    o = clone(obs_for(COVERED[0]))
    for sl in (LEFT_HAND_SLICE, RIGHT_HAND_SLICE):
        o.observed[:, sl] = False
        o.usable[:, sl] = False
        o.xy_px[:, sl] = np.nan
    o.hand_score[:] = np.nan
    res = predict(o, REFS, CONTRACT)
    assert res["observation_status"] == "INSUFFICIENT"
    assert "handshape_check" not in res or res.get("handshape_check") is None


@check("every configured axis has a band for every covered word")
def _axes_present():
    from wis_lab.core.recognize import AXIS_GROUPS
    assert set(AXES) == set(AXIS_GROUPS), (set(AXES), set(AXIS_GROUPS))
    for axis, spec in AXES.items():
        assert spec["rule"] == "upper bound only", axis
        for w in COVERED:
            assert w in spec["bands"], (axis, w)
            assert spec["bands"][w]["n"] >= 10


@check("task_match runs the axes against the REQUESTED word, not the prediction")
def _vs_requested():
    from wis_lab.core.recognize import task_match
    w, other = COVERED[0], COVERED[1]
    o = obs_for(w)
    res = predict(o, REFS, CONTRACT)
    t = task_match(res, other, observation=o, reference_set=REFS, frozen_config=CONTRACT)
    ax = t["axis_checks_vs_requested"]
    # the checks must be about `other`, the word that was asked for
    for a, v in ax.items():
        assert v["word_id"] == other, (a, v["word_id"])
    # and predict must still be blind: its own checks are about its own top-1
    assert res["axis_checks"]["handshape"]["word_id"] == res["top1_word_id"]


@check("task_match without the observation stays backwards compatible")
def _optional_args():
    from wis_lab.core.recognize import task_match
    res = predict(obs_for(COVERED[0]), REFS, CONTRACT)
    t = task_match(res, COVERED[0])
    assert "axis_checks_vs_requested" not in t
    assert t["task_match"] is True


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
