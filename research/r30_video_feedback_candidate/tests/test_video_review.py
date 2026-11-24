"""Synthetic tests for wis_lab.core.video_review.

They check the BINDING and the BOOKKEEPING - which saved frame is compared against what, and what
is said when it cannot be - plus the arithmetic of the degree conversion. They do not check, and
cannot check, whether a colour helps anybody learn a sign.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

import synthetic as S
from wis_lab.core import features as F
from wis_lab.core import video_review as V

RGB_ROWS = [2, 7, 11, 16, 20, 25, 29, 34, 38, 43, 47, 52, 57]   # 2 and 57 are outside the span
ELIGIBLE_ROWS = [r for r in RGB_ROWS if 5 <= r <= 54]


@pytest.fixture(scope="module")
def obs():
    return S.make_observation()


@pytest.fixture(scope="module")
def reference():
    return S.make_reference()


def run(obs, reference, rows=RGB_ROWS, **kw):
    return V.frame_angle_comparison(obs, S.make_manifest(obs, rows, **kw), reference, S.CONTRACT)


# ------------------------------------------------------------- angle mapping
def test_angle_map_is_explicit_and_one_to_one():
    assert len(V.ANGLE_MAP) == V.INTERIOR_ANGLE_COUNT == 15
    vertices = [a["vertex_landmark"] for a in V.ANGLE_MAP]
    segments = [tuple(a["colored_segment"]) for a in V.ANGLE_MAP]
    assert len(set(vertices)) == 15, "each angle must own its vertex"
    assert len(set(segments)) == 15, "no bone may have to choose between two angle values"
    assert 0 not in vertices, "the wrist is not a finger angle"
    assert all(0 not in seg for seg in segments), "the wrist bone is never coloured"
    assert [a["finger"] for a in V.ANGLE_MAP].count("thumb") == 3
    # the vertex really is the middle landmark of the triple bend_angles measures
    k = 0
    for chain in F.FINGER_CHAINS:
        for i in range(1, 4):
            assert V.ANGLE_MAP[k]["vertex_landmark"] == chain[i]
            assert V.ANGLE_MAP[k]["segments"] == [[chain[i - 1], chain[i]], [chain[i], chain[i + 1]]]
            k += 1
    assert V.UNMAPPED_GROUP_INDICES["indices"] == [15, 16, 17, 18]


# --------------------------------------------------- every saved frame is reported
def test_every_saved_frame_is_reported_with_identity(obs, reference):
    out = run(obs, reference)
    assert out["status"] == "OK"
    assert len(out["frames"]) == len(RGB_ROWS) == out["counts"]["saved_frames"]
    for row, saved in zip(out["frames"], RGB_ROWS):
        assert row["frame_id"] == int(obs.frame_id[saved])
        assert row["timestamp_us"] == int(obs.timestamp_us[saved])
        assert row["state"] in V.FRAME_STATES
        assert "comparison_available" in row
    assert out["reference"]["clip_id"] == reference["clip_id"]
    assert out["reference"]["word_id"] == reference["word_id"]


def test_outside_span_is_outside_not_nearest_neighbour(obs, reference):
    out = run(obs, reference)
    by_row = {r["index"]: r for r in out["frames"]}
    for position, saved in enumerate(RGB_ROWS):
        state = by_row[position]["state"]
        if saved < 5 or saved > 54:
            assert state == "OUTSIDE_SPAN"
            assert by_row[position]["hands"] == {}
            assert by_row[position]["comparison_available"] is False
        else:
            assert state != "OUTSIDE_SPAN"


def test_all_eligible_saved_frames_are_accounted_for(obs, reference):
    out = run(obs, reference)
    accounted = [r for r in out["frames"]
                 if r["state"] in ("COMPARED", "NO_ANGLE_OBSERVED", "UNALIGNED")]
    assert len(accounted) == len(ELIGIBLE_ROWS), "no eligible frame may be silently dropped"
    assert out["counts"]["COMPARED"] >= 2
    assert {r["observation_row"] for r in accounted} == set(ELIGIBLE_ROWS)


def test_the_rgb_list_is_not_the_classifier_sampling_list(obs, reference):
    """The whole point: the two selections differ, so a nearby sample may not be reused."""
    uniform = set(S.uniform_query_rows(obs))
    assert set(ELIGIBLE_ROWS) != uniform
    assert [r for r in ELIGIBLE_ROWS if r not in uniform], \
        "the fixture must contain a saved frame the classifier never sampled"
    out = run(obs, reference)
    compared = {r["observation_row"] for r in out["frames"] if r["state"] == "COMPARED"}
    assert compared - uniform, "frames outside the uniform sample must still be compared"


# ------------------------------------------------------------ identity rejection
@pytest.mark.parametrize("override,reason", [
    ({"timestamp_us": 1}, "TIMESTAMP_DISAGREES_WITH_OBSERVATION"),
    ({"observation_row": 0}, "OBSERVATION_ROW_DISAGREES_WITH_FRAME_ID"),
    ({"frame_id": 999_999}, "FRAME_ID_NOT_IN_OBSERVATION"),
    ({"frame_id": None}, "MANIFEST_FIELD_MISSING_OR_NOT_AN_INTEGER"),
])
def test_inconsistent_identity_is_rejected_not_matched(obs, reference, override, reason):
    out = run(obs, reference, overrides={3: override})
    bad = out["frames"][3]
    assert bad["state"] == "IDENTITY_REJECTED"
    assert bad["reason"] == reason
    assert bad["observation_row"] is None and bad["hands"] == {}
    assert bad["comparison_available"] is False
    # the other frames are unaffected: one bad row does not void the panel
    assert out["counts"]["COMPARED"] >= 2


def test_duplicate_frame_id_in_the_manifest_rejects_both_copies(obs, reference):
    rows = list(RGB_ROWS)
    out = run(obs, reference, overrides={4: {"frame_id": int(obs.frame_id[RGB_ROWS[5]]),
                                             "timestamp_us": int(obs.timestamp_us[RGB_ROWS[5]]),
                                             "observation_row": RGB_ROWS[5]}}, rows=rows)
    states = [out["frames"][i]["state"] for i in (4, 5)]
    assert states == ["IDENTITY_REJECTED", "IDENTITY_REJECTED"]
    assert out["frames"][4]["reason"] == "DUPLICATE_FRAME_ID_IN_MANIFEST"


def test_manifest_from_another_observation_is_refused(obs, reference):
    other = S.make_observation(sha256="f" * 64)
    manifest = S.make_manifest(other, RGB_ROWS)
    with pytest.raises(V.ReviewError):
        V.frame_angle_comparison(obs, manifest, reference, S.CONTRACT)


@pytest.mark.parametrize("patch", [{"schema": "something_else"}, {"status": "UNAVAILABLE"},
                                   {"frames": []}])
def test_unusable_manifests_raise_rather_than_return_empty_colours(obs, reference, patch):
    manifest = {**S.make_manifest(obs, RGB_ROWS), **patch}
    with pytest.raises(V.ReviewError):
        V.frame_angle_comparison(obs, manifest, reference, S.CONTRACT)


# -------------------------------------------------------- degrees and aggregation
def test_degrees_come_from_the_existing_bend_features(obs, reference):
    out = run(obs, reference)
    seq = F.sequence_features(obs, trim=True)
    lo = int(seq["span"][0])
    start = V._SLICES["bend_L"].start
    row = next(r for r in out["frames"] if r["state"] == "COMPARED"
               and r["hands"]["L"]["status"] == "COMPARED")
    idx = row["observation_row"] - lo
    expected = [round(float(seq["features"][idx, start + k]) * 180.0, 2) for k in range(15)]
    got = [a["query_deg"] for a in row["hands"]["L"]["angles"]]
    assert got == expected
    assert all(0.0 <= a["mean_abs_diff_deg"] <= 180.0 for a in row["hands"]["L"]["angles"])


def test_repeated_reference_matches_are_aggregated_explicitly(obs, reference):
    out = run(obs, reference)
    multi = [r for r in out["frames"]
             if r["state"] == "COMPARED" and len(r["reference_rows"]) > 1]
    assert multi, "the fixture must produce at least one query frame with several matches"
    hand = next(h for r in multi for h in r["hands"].values()
                if h["status"] == "COMPARED" and h["compared_pairs"] > 1)
    for angle in hand["angles"]:
        pairs = angle["per_pair_abs_diff_deg"]
        assert angle["support"] == len(pairs) == hand["compared_pairs"]
        assert angle["mean_abs_diff_deg"] == pytest.approx(sum(pairs) / len(pairs), abs=0.01)
        assert angle["min_abs_diff_deg"] == pytest.approx(min(pairs))
        assert angle["max_abs_diff_deg"] == pytest.approx(max(pairs))
        # not the first, not the smallest: the reported value is the mean of them all
        assert len(hand["reference_rows_compared"]) == len(pairs)


def test_single_match_reports_support_one_without_a_fake_spread(obs, reference):
    out = run(obs, reference)
    singles = [h for r in out["frames"] if r["state"] == "COMPARED"
               for h in r["hands"].values()
               if h["status"] == "COMPARED" and h["compared_pairs"] == 1]
    for hand in singles:
        for angle in hand["angles"]:
            assert angle["support"] == 1
            assert "per_pair_abs_diff_deg" not in angle


# ----------------------------------------------------------- missingness semantics
def test_a_hand_absent_from_the_reference_is_missing_not_zero(obs):
    reference = S.make_reference(right_missing=True)
    out = run(obs, reference)
    row = next(r for r in out["frames"] if r["state"] == "COMPARED")
    right = row["hands"]["R"]
    assert right["status"] == "NOT_COMPARABLE"
    assert right["reason"] == "REFERENCE_GROUP_NOT_OBSERVED"
    assert right["angles"] == []
    assert right["pair_status"]["reference_group_missing"] == right["matched_pairs"]
    assert row["hands"]["L"]["status"] == "COMPARED"


def test_a_hand_absent_from_the_query_frame_is_missing_not_zero(reference):
    obs = S.make_observation(left_missing_rows=ELIGIBLE_ROWS[2:4])
    out = run(obs, reference)
    hit = [r for r in out["frames"] if r["observation_row"] in ELIGIBLE_ROWS[2:4]]
    assert hit
    for row in hit:
        if row["state"] != "COMPARED":
            continue
        assert row["hands"]["L"]["status"] == "NOT_COMPARABLE"
        assert row["hands"]["L"]["reason"] == "QUERY_GROUP_NOT_OBSERVED"


def test_too_few_saved_frames_inside_the_span_is_unaligned_not_invented(obs, reference):
    out = run(obs, reference, rows=[2, 11, 57])          # one eligible frame only
    states = [r["state"] for r in out["frames"]]
    assert states == ["OUTSIDE_SPAN", "UNALIGNED", "OUTSIDE_SPAN"]
    assert out["frames"][1]["reason"] == "FEWER_THAN_TWO_SAVED_FRAMES_INSIDE_THE_SPAN"
    assert out["alignment"]["path_length"] == 0


def test_an_observation_without_hands_reports_unavailable(reference):
    obs = S.make_observation(n_frames=12, hand_span=(1, 0), right_missing=True)
    manifest = S.make_manifest(obs, [1, 2, 3])
    out = V.frame_angle_comparison(obs, manifest, reference, S.CONTRACT)
    assert out["status"] == "UNAVAILABLE"
    assert out["reason"] == "FeatureError"
    assert out["frames"] == []


# ------------------------------------------------------------- output contract
def test_output_is_strict_json_without_nan(obs, reference):
    out = run(obs, reference)
    text = json.dumps(out, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["schema"] == V.SCHEMA
    assert out["is_a_verdict"] is False
    assert out["unit"] == "degrees"


def test_inputs_are_not_mutated(obs, reference):
    """The saved observation and the prepared reference are read, never written to."""
    def snapshot(value):
        # bytes, so a NaN compares equal to itself and a bool array needs no special case
        return (value.dtype, value.shape, value.tobytes()) if isinstance(value, np.ndarray) \
            else repr(copy.deepcopy(value))

    ref_before = {k: snapshot(v) for k, v in reference.items()}
    keys_before = sorted(reference)
    xy_before = obs.xy_px.tobytes()
    usable_before = obs.usable.tobytes()
    run(obs, reference)
    assert obs.xy_px.tobytes() == xy_before
    assert obs.usable.tobytes() == usable_before
    assert sorted(reference) == keys_before, "no key was added to the prepared reference"
    for key, value in ref_before.items():
        assert snapshot(reference[key]) == value, key


def test_the_extra_dtw_is_declared_as_one_reference_only(obs, reference):
    out = run(obs, reference)
    assert out["alignment"]["query_selector"] == "saved_rgb_frames"
    assert out["alignment"]["band_ratio"] == V.band_ratio(S.CONTRACT)
    assert out["alignment"]["cost_config"] == V.cost_config(S.CONTRACT)
    assert out["reference"]["sampled_rows"] == len(reference["features"])
    # nothing that belongs to recognition appears in this payload
    for forbidden in ("ranking", "top1_word_id", "distance", "accepted", "execution_assessment",
                      "probability", "score"):
        assert forbidden not in out
