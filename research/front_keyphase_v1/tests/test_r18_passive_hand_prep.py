"""functional_21: the passive-hand preparation helper, on synthetic fixtures only.

No training, no inference, no model load, no camera, no server, no participant data. The public
manifest and the frozen training list are read as lists; no clip is decoded and no cache is opened
unless a check needs its name and hash.

Run: py -3.13 tests/test_r18_passive_hand_prep.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wis_lab.core import passive_hand as P  # noqa: E402
from wis_lab.core.observation import (LEFT_HAND_SLICE, N_POINTS, Observation,  # noqa: E402
                                      POSE_L_WRIST, POSE_R_WRIST, POSE_SLICE, RIGHT_HAND_SLICE)

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ synthetic clip and sources
CLIP, WORD, SHA = "NIKL_TEST_01", "TESTWORD", "e" * 64
NPZ = "cache/2683eadbc5f1c588/NIKL_TEST_01.npz"
VIDEO_SHA = "d" * 64
ANNOTATIONS_20260914 = os.path.join(
    "C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff",
    "WordsInSilence_기능완성_20260913",
    "public_passive_hand_annotations_20260914.json")


def make_obs(t=12, left_frames=None, right_frames=None) -> Observation:
    """A small clip: pose everywhere, both hands present on the frames given."""
    rng = np.arange(t, dtype=np.float32)
    xy = np.full((t, N_POINTS, 2), np.nan, dtype=np.float32)
    observed = np.zeros((t, N_POINTS), dtype=bool)
    usable = np.zeros((t, N_POINTS), dtype=bool)
    conf = np.full((t, N_POINTS), np.nan, dtype=np.float32)
    hand_score = np.full((t, 2), np.nan, dtype=np.float32)

    xy[:, POSE_SLICE, 0] = (100.0 + rng)[:, None]
    xy[:, POSE_SLICE, 1] = 200.0
    xy[:, 11] = np.stack([80.0 + rng, np.full(t, 150.0)], axis=1)     # shoulders apart
    xy[:, 12] = np.stack([180.0 + rng, np.full(t, 150.0)], axis=1)
    xy[:, POSE_L_WRIST] = np.stack([90.0 + rng, np.full(t, 300.0)], axis=1)
    xy[:, POSE_R_WRIST] = np.stack([170.0 + rng, np.full(t, 300.0)], axis=1)
    observed[:, POSE_SLICE] = True
    usable[:, POSE_SLICE] = True
    conf[:, POSE_SLICE] = 0.9

    for sl, frames, col in ((LEFT_HAND_SLICE, left_frames, 0), (RIGHT_HAND_SLICE, right_frames, 1)):
        frames = range(t) if frames is None else frames
        for f in frames:
            pts = np.stack([np.arange(21, dtype=np.float32) + 10.0 * col + f,
                            np.arange(21, dtype=np.float32) + 400.0], axis=1)
            xy[f, sl] = pts
            observed[f, sl] = True
            usable[f, sl] = True
            hand_score[f, col] = 0.8

    obs = Observation(xy_px=xy, timestamp_us=(np.arange(t, dtype=np.int64) + 1) * 33_000,
                      frame_id=np.arange(t, dtype=np.int64), observed=observed, usable=usable,
                      conf=conf, hand_score=hand_score,
                      sidecar={"observation_sha256": "f" * 64})
    obs.validate()
    return obs


def sources(split_role="training_original_and_every_augmentation", npz=NPZ, sha=SHA,
            manifest_sha=None, source_kind="nikl_interpreter_word_clip"):
    tmp = tempfile.mkdtemp(prefix="f21src_")
    man = {"manifest_id": "synthetic", "clips": [
        {"clip_id": CLIP, "word_id": WORD, "sha256": VIDEO_SHA, "source_kind": source_kind,
         "observation_npz": npz,
         "observation_sha256": sha if manifest_sha is None else manifest_sha}]}
    inputs = {"nikl": [{"id": CLIP, "word_id": WORD, "role": split_role, "npz": npz,
                        "npz_sha256": sha}]}
    mp = os.path.join(tmp, "source_manifest.json")
    ip = os.path.join(tmp, "inputs.json")
    for path, doc in ((mp, man), (ip, inputs)):
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, ensure_ascii=False)
    try:
        return P.load_sources(mp, ip)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def annotation(**over):
    ann = {"schema": P.ANNOTATION_SCHEMA, "clip_id": CLIP, "word_id": WORD,
           "cache_sha256": SHA,
           "hand_roles": {"left": "non_sign_hand", "right": "sign_hand"},
           "basis": P.REVIEW_BASIS, "reviewed_by": "root_public_footage_review",
           "reviewed_interval": {"start_frame": 0, "end_frame": 999},
           "covers_whole_sign_interval": True,
           "evidence": {"source_video_sha256": VIDEO_SHA, "every_decoded_frame_reviewed": True,
                        "frame_count": 12,
                        "visual_observation": "the right hand signs; the left arm stays lowered "
                                              "at the side, with no contact and no support"},
           "is_expert_verification": False}
    ann.update(over)
    return ann


# ------------------------------------------------------------------ 1. what may be removed
def test_eligibility():
    src = sources()
    ok(P.check_annotation(annotation(), src)["eligible"],
       "a reviewed single non-signing hand on a training clip is eligible")

    def why(ann, s=None):
        return P.check_annotation(ann, s or src)["reasons"]

    ok("UNKNOWN_CLIP_ID" in why(annotation(clip_id="NIKL_OTHER_01")),
       "an unknown clip is refused")
    ok("CACHE_SHA256_MISMATCH" in why(annotation(cache_sha256="0" * 64)),
       "a cache hash that does not match the frozen list is refused")
    ok("WORD_ID_MISMATCH" in why(annotation(word_id="SOMETHING_ELSE")),
       "a word that does not match the manifest is refused")
    ok("NOT_A_TRAINING_SPLIT_CLIP" in why(annotation(), sources(split_role="validation")),
       "a validation clip is refused")
    ok("BOTH_HANDS_MARKED_NON_SIGNING" in
       why(annotation(hand_roles={"left": "non_sign_hand", "right": "non_sign_hand"})),
       "removing both hands is refused")
    ok("NO_HAND_REVIEWED_AS_NON_SIGNING" in
       why(annotation(hand_roles={"left": "supporting_hand", "right": "sign_hand"})),
       "a supporting hand is not a non-signing hand")
    ok("HAND_ROLE_UNKNOWN_IS_NOT_ELIGIBLE" in
       why(annotation(hand_roles={"left": "non_sign_hand", "right": "unknown"})),
       "an unknown role anywhere in the record is refused")
    ok("NO_HAND_REVIEWED_AS_NON_SIGNING" in
       why(annotation(hand_roles={"left": "required_hand", "right": "sign_hand"})),
       "a required hand is never removed")
    for basis in ("low_motion", "hand_participation", "model_prediction", "word_name",
                  "user_did_not_move", "unknown"):
        r = why(annotation(basis=basis))
        ok("BASIS_IS_NOT_A_REVIEW" in r, f"{basis} is not a review and cannot grant eligibility")
    ok("NO_REVIEWER_RECORDED" in why(annotation(reviewed_by="")), "an unsigned annotation is "
       "refused")
    ok("REVIEW_DOES_NOT_COVER_THE_WHOLE_SIGN" in why(annotation(covers_whole_sign_interval=False)),
       "a review that does not cover the whole sign is refused")
    ok("MALFORMED_REVIEWED_INTERVAL" in
       why(annotation(reviewed_interval={"start_frame": 5, "end_frame": 1})),
       "an interval that runs backwards is refused")
    ok("MALFORMED_REVIEWED_INTERVAL" in why(annotation(reviewed_interval="whole clip")),
       "an interval that is not an interval is refused")
    ok("MALFORMED_HAND_ROLES" in why(annotation(hand_roles={"both": "non_sign_hand"})),
       "a side that is not an anatomical side is refused")
    ok("ANNOTATION_MAY_NOT_CHOOSE_A_PATH" in why(annotation(npz="C:/anything.npz")),
       "an annotation cannot name its own input file")
    ok("CACHE_PATH_NOT_IN_PUBLIC_CACHE" in
       why(annotation(), sources(npz="C:/elsewhere/clip.npz")),
       "a training entry pointing outside the public cache is refused")
    ok("WRONG_ANNOTATION_SCHEMA" in why(annotation(schema="something_v9")),
       "an annotation of another schema is refused")

    # nothing about a word name or a quiet hand can stand in for the review
    ok(not P.check_annotation({"schema": P.ANNOTATION_SCHEMA, "clip_id": CLIP, "word_id": WORD,
                               "cache_sha256": SHA, "hand_roles": {"left": "non_sign_hand"},
                               "basis": "low_motion", "reviewed_by": "auto",
                               "reviewed_interval": {"start_frame": 0, "end_frame": 9},
                               "covers_whole_sign_interval": True}, src)["eligible"],
       "a low-motion record is not made eligible by being otherwise complete")


# ------------------------------------------------------------------ 2. what the copy changes
def test_mask_is_exact():
    obs = make_obs()
    before = copy.deepcopy(obs)
    masked = P.mask_hand(obs, "left")

    ok(np.array_equal(obs.xy_px, before.xy_px, equal_nan=True)
       and np.array_equal(obs.observed, before.observed)
       and np.array_equal(obs.usable, before.usable)
       and np.array_equal(obs.hand_score, before.hand_score, equal_nan=True),
       "the original observation is not modified")

    ok(np.isnan(masked.xy_px[:, LEFT_HAND_SLICE]).all()
       and not masked.observed[:, LEFT_HAND_SLICE].any()
       and not masked.usable[:, LEFT_HAND_SLICE].any(),
       "the chosen hand is absent: NaN, not observed, not usable")
    ok(np.isnan(masked.hand_score[:, 0]).all() and np.array_equal(
        masked.hand_score[:, 1], obs.hand_score[:, 1], equal_nan=True),
       "its hand score is NaN and the other side's is untouched")
    ok(not (masked.xy_px[:, LEFT_HAND_SLICE] == 0).any(),
       "absent is never zero, which would read as a hand at the origin")

    keep = np.ones(N_POINTS, dtype=bool)
    keep[LEFT_HAND_SLICE] = False
    ok(np.array_equal(masked.xy_px[:, keep], obs.xy_px[:, keep], equal_nan=True),
       "every other point, the opposite hand included, is byte-identical")
    ok(np.array_equal(masked.xy_px[:, POSE_L_WRIST], obs.xy_px[:, POSE_L_WRIST])
       and np.array_equal(masked.xy_px[:, POSE_R_WRIST], obs.xy_px[:, POSE_R_WRIST]),
       "the pose wrists are body pose and stay, including the masked side's wrist")
    ok(np.array_equal(masked.timestamp_us, obs.timestamp_us)
       and np.array_equal(masked.frame_id, obs.frame_id),
       "timestamps and frame ids are unchanged")
    masked.validate()
    ok(True, "the masked copy satisfies Observation.validate")

    right = P.mask_hand(obs, "right")
    ok(np.isnan(right.xy_px[:, RIGHT_HAND_SLICE]).all()
       and right.usable[:, LEFT_HAND_SLICE].any(),
       "the right side masks the right hand only")
    try:
        P.mask_hand(obs, "both")
        ok(False, "a side that is not left or right is rejected")
    except ValueError:
        ok(True, "a side that is not left or right is rejected")


# ------------------------------------------------------------------ 3. one fixed span, 280 ch
def test_pair():
    src = sources()
    obs = make_obs()
    out = P._prepare_pair_from_arrays(annotation(), obs, src)
    ok(out["prepared"], "an eligible annotation prepares a pair")
    pair = out["pair"]

    a, b = pair["variants"]["original"], pair["variants"]["masked"]
    ok(a["x"].shape == b["x"].shape and a["x"].shape[1] == 280,
       "both variants carry the same 280 channels over the same number of frames")
    ok(a["x"].shape[0] == pair["span"]["frames"],
       "and exactly the frames of the ORIGINAL active span")
    ok(pair["span"]["taken_from"].startswith("the ORIGINAL"),
       "the span is taken from the original and held fixed for both")

    # 280 = body_rel 51x2 (0..101) + hand_rel 42x2 (102..185) + body_valid 51 (186..236)
    # + hand_valid 42 (237..278) + dt (279). body_rel covers the hands too: common points are
    # 9 body, then 21 left hand, then 21 right hand.
    body9 = slice(0, 18)                      # the nine body points, in body-relative space
    left_rel = slice(18, 60)                  # the masked hand, in body-relative space
    right_rel = slice(60, 102)                # the opposite hand, same space
    ok(np.allclose(a["x"][:, body9], b["x"][:, body9]),
       "the nine body points are identical between the two variants")
    ok(np.allclose(a["x"][:, right_rel], b["x"][:, right_rel]),
       "and so is the opposite hand, in body-relative space")
    ok(np.allclose(a["x"][:, 279], b["x"][:, 279]),
       "and so is the dt channel, because the time base did not move")
    ok(np.allclose(a["x"][:, 186:195], b["x"][:, 186:195]),
       "the nine body validity channels are unchanged")

    ok(not np.allclose(a["x"][:, left_rel], b["x"][:, left_rel])
       and np.allclose(b["x"][:, left_rel], 0.0),
       "the masked hand's body-relative channels are zeroed, and only after being marked invalid")
    ok(b["x"][:, 195:216].sum() == 0 and a["x"][:, 195:216].sum() > 0,
       "its body validity channels say absent, where the intact variant says present")
    hand_valid_left, hand_valid_right = slice(237, 258), slice(258, 279)
    ok(b["x"][:, hand_valid_left].sum() == 0 and a["x"][:, hand_valid_left].sum() > 0,
       "its hand validity channels say absent too")
    ok(np.allclose(a["x"][:, hand_valid_right], b["x"][:, hand_valid_right]),
       "while the opposite hand's validity is untouched")
    ok(np.allclose(b["x"][:, 102:144], 0.0)
       and np.allclose(a["x"][:, 144:186], b["x"][:, 144:186]),
       "the masked hand's own hand-relative channels are zero, the opposite hand's unchanged")
    ok(np.isfinite(b["x"]).all(), "no NaN reaches the channels; missingness is carried by masks")

    ok(pair["grouping"]["source_group"] == CLIP
       and pair["grouping"]["role"] == "training_original_and_every_augmentation"
       and pair["grouping"]["independent_samples_added"] == 0,
       "the variant keeps the source group and the training role, and adds no new sample")
    ok(pair["provenance"]["verified"] is False
       and pair["provenance"]["cache_sha256_declared_by_list"] == SHA,
       "arrays handed in by a caller are NEVER stamped as verified provenance")
    ok(pair["review_evidence"]["visual_observation"]
       and pair["review_evidence"]["source_video_sha256"] == VIDEO_SHA,
       "the reason the removal was allowed travels with the pair")
    ok(pair["annotation"]["side"] == "left" and pair["masked_side"] == "left",
       "and which hand was removed, by anatomical side")


# ------------------------------------------------------------------ 4. pairs that must be refused
def test_pair_refusals():
    src = sources()

    # the reviewed interval must cover the sign, not part of it
    obs = make_obs()
    short = P._prepare_pair_from_arrays(annotation(reviewed_interval={"start_frame": 4, "end_frame": 6}),
                           obs, src)
    ok(not short["prepared"]
       and short["reasons"] == ["REVIEWED_INTERVAL_DOES_NOT_COVER_THE_SIGN"],
       "a review of part of the sign cannot license removing the hand for all of it")

    # nothing is left to learn from once the only other hand is gone
    only_left = make_obs(left_frames=range(12), right_frames=[])
    none_left = P._prepare_pair_from_arrays(annotation(), only_left, src)
    ok(not none_left["prepared"]
       and none_left["reasons"] == ["NO_OBSERVED_OPPOSITE_HAND_FRAME"],
       "a pair with no observed opposite hand is refused, not built")

    # an ineligible annotation never reaches the arrays
    bad = P._prepare_pair_from_arrays(annotation(basis="low_motion"), make_obs(), src)
    ok(not bad["prepared"] and bad["pair"] is None and "BASIS_IS_NOT_A_REVIEW" in bad["reasons"],
       "an ineligible annotation produces no pair at all")

    # one hand visible only briefly still prepares, because the review - not the data - decides
    brief = make_obs(left_frames=[0, 1], right_frames=range(12))
    out = P._prepare_pair_from_arrays(annotation(), brief, src)
    ok(out["prepared"] and out["pair"]["opposite_hand_frames_in_span"] == 12,
       "a hand that is barely visible is still removed only because it was reviewed")


# --------------------------------------------- 5. the four provenance counterexamples
def public_root(tmp, *, clip="NIKL_PUB_01", word="PUBWORD", split="training_original_and_every_"
                "augmentation", tamper=False, manifest_sha=None, npz_rel=None,
                kind="nikl_interpreter_word_clip", obs=None):
    """A miniature repository: two public lists and a real cache file, hashed as they would be."""
    import hashlib
    root = os.path.join(tmp, clip)
    npz_rel = npz_rel or f"cache/2683eadbc5f1c588/{clip}.npz"
    os.makedirs(os.path.join(root, "config"))
    os.makedirs(os.path.join(root, "runs", "union93_candidate_v1"))
    os.makedirs(os.path.dirname(os.path.join(root, npz_rel)), exist_ok=True)
    (obs or make_obs()).save(os.path.join(root, npz_rel))
    if tamper:
        with open(os.path.join(root, npz_rel), "ab") as fh:
            fh.write(b"\0")                      # the file is no longer what the list says
    real = hashlib.sha256(open(os.path.join(root, npz_rel), "rb").read()).hexdigest()
    declared = real if not tamper else "9" * 64
    man = {"clips": [{"clip_id": clip, "word_id": word, "sha256": VIDEO_SHA,
                      "source_kind": kind, "observation_npz": npz_rel,
                      "observation_sha256": declared if manifest_sha is None else manifest_sha}]}
    inp = {"nikl": [{"id": clip, "word_id": word, "role": split, "npz": npz_rel,
                     "npz_sha256": declared}]}
    for rel, doc in (("config/source_manifest.json", man),
                     ("runs/union93_candidate_v1/inputs.json", inp)):
        with open(os.path.join(root, rel), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, ensure_ascii=False)
    return root, declared


def public_annotation(clip="NIKL_PUB_01", word="PUBWORD", cache_sha="", **over):
    ann = annotation(clip_id=clip, word_id=word, cache_sha256=cache_sha)
    ann.update(over)
    return ann


def test_counterexamples():
    tmp = tempfile.mkdtemp(prefix="f21pub_")
    try:
        # (1) one clip's annotation must not be able to arrive with another clip's arrays.
        # There is no longer an entry point that accepts arrays at all: the boundary loads them.
        a_root, a_sha = public_root(tmp, clip="NIKL_PUBA_01", word="WORDA",
                                    obs=make_obs(t=12))
        b_root, b_sha = public_root(tmp, clip="NIKL_PUBB_01", word="WORDB",
                                    obs=make_obs(t=20))
        shutil.copytree(os.path.join(b_root, "cache"), os.path.join(a_root, "cache"),
                        dirs_exist_ok=True)          # both caches sit side by side
        out_a = P.prepare_from_public_clip(
            public_annotation(clip="NIKL_PUBA_01", word="WORDA", cache_sha=a_sha), root=a_root)
        ok(out_a["prepared"], "the verified boundary prepares its own clip")
        ok(out_a["pair"]["provenance"]["cache_sha256_actual"] == a_sha
           and out_a["pair"]["provenance"]["verified"] is True,
           "and stamps the hash it actually computed from the file it read")
        ok(out_a["pair"]["provenance"]["observation_frames"] == 12,
           "the arrays are the ones on disk for THAT clip, not a caller's")
        ok(not hasattr(P, "prepare_pair"),
           "no public function accepts caller-supplied arrays any more")
        wrong = P.prepare_from_public_clip(
            public_annotation(clip="NIKL_PUBA_01", word="WORDA", cache_sha=b_sha), root=a_root)
        ok(not wrong["prepared"] and "CACHE_SHA256_MISMATCH" in wrong["reasons"],
           "an annotation asserting another clip's cache hash is refused")

        # (2) the review evidence must be there
        base = public_annotation(clip="NIKL_PUBA_01", word="WORDA", cache_sha=a_sha)
        gone = dict(base)
        del gone["evidence"]
        r = P.prepare_from_public_clip(gone, root=a_root)
        ok(not r["prepared"] and "REVIEW_EVIDENCE_MISSING" in r["reasons"],
           "an annotation with no evidence at all prepares nothing")
        empty = dict(base, evidence={})
        ok("REVIEW_EVIDENCE_MISSING" in P.prepare_from_public_clip(empty, root=a_root)["reasons"],
           "and neither does an empty evidence block")
        blank = dict(base, evidence=dict(base["evidence"], visual_observation="   "))
        ok("REVIEW_EVIDENCE_HAS_NO_VISUAL_OBSERVATION"
           in P.prepare_from_public_clip(blank, root=a_root)["reasons"],
           "evidence that says nothing about what was seen is not evidence")
        other = dict(base, evidence=dict(base["evidence"], source_video_sha256="7" * 64))
        ok("REVIEW_EVIDENCE_VIDEO_IS_NOT_THIS_CLIP"
           in P.prepare_from_public_clip(other, root=a_root)["reasons"],
           "a review of some other footage cannot license this clip")
        part = dict(base, evidence=dict(base["evidence"], every_decoded_frame_reviewed=False))
        ok("REVIEW_EVIDENCE_DOES_NOT_COVER_EVERY_FRAME"
           in P.prepare_from_public_clip(part, root=a_root)["reasons"],
           "a partial viewing is refused")

        # (3) the two public lists must agree about the cache hash
        d_root, d_sha = public_root(tmp, clip="NIKL_PUBD_01", word="WORDD",
                                    manifest_sha="3" * 64)
        r = P.prepare_from_public_clip(
            public_annotation(clip="NIKL_PUBD_01", word="WORDD", cache_sha=d_sha), root=d_root)
        ok(not r["prepared"] and "MANIFEST_AND_TRAINING_CACHE_HASH_DISAGREE" in r["reasons"],
           "a manifest hash that differs from the training hash prepares nothing")

        # (4) the evidence is kept in the output
        pair = out_a["pair"]
        ev = pair["review_evidence"]
        ok(ev["source_video_sha256"] == VIDEO_SHA and ev["visual_observation"],
           "the saved pair keeps the source video identity and what the reviewer saw")
        ok("frames 0-" in ev["reviewed_interval_description"]
           and "whole sign" in ev["reviewed_interval_description"],
           "and the interval description with its whole-sign rationale")
        ok(P.NOT_CERTIFICATION in (pair["annotation_is"], pair["annotation"]["annotation_is"])
           and "not a certification" in pair["annotation_is"].lower(),
           "and the statement that this is not an expert certification")
        ok(pair["annotation"]["is_expert_verification"] is False,
           "the pair never records the annotation as an expert verification")
        claim = dict(base, is_expert_verification=True)
        ok("ANNOTATION_MAY_NOT_CLAIM_EXPERT_VERIFICATION"
           in P.prepare_from_public_clip(claim, root=a_root)["reasons"],
           "and an annotation claiming to be one is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resolved_boundary():
    tmp = tempfile.mkdtemp(prefix="f21res_")
    try:
        root, sha = public_root(tmp, clip="NIKL_PUBE_01", word="WORDE")
        good = P.resolve_public_source("NIKL_PUBE_01", root=root)
        ok(good["ok"] and good["source"]["actual_cache_sha256"] == sha,
           "a resolvable clip reports the hash computed from the file itself")

        t_root, _ = public_root(tmp, clip="NIKL_PUBF_01", word="WORDF", tamper=True)
        bad = P.resolve_public_source("NIKL_PUBF_01", root=t_root)
        ok(not bad["ok"] and "CACHE_FILE_SHA256_MISMATCH" in bad["reasons"],
           "a file whose bytes do not match the declared hash is refused")

        v_root, v_sha = public_root(tmp, clip="NIKL_PUBG_01", word="WORDG", split="validation")
        ok("NOT_A_TRAINING_SPLIT_CLIP"
           in P.resolve_public_source("NIKL_PUBG_01", root=v_root)["reasons"],
           "a validation clip is refused at the boundary too")

        k_root, _ = public_root(tmp, clip="NIKL_PUBH_01", word="WORDH", kind="private_capture")
        ok("NOT_A_PUBLIC_SOURCE_KIND"
           in P.resolve_public_source("NIKL_PUBH_01", root=k_root)["reasons"],
           "a source kind that is not a public clip is refused")

        o_root, _ = public_root(tmp, clip="NIKL_PUBI_01", word="WORDI",
                                npz_rel="elsewhere/NIKL_PUBI_01.npz")
        ok("CACHE_PATH_NOT_IN_PUBLIC_CACHE"
           in P.resolve_public_source("NIKL_PUBI_01", root=o_root)["reasons"],
           "a cache path outside the public cache is refused")

        ok("UNKNOWN_CLIP_ID" in P.resolve_public_source("NIKL_NOT_HERE", root=root)["reasons"],
           "a clip neither list knows is refused")

        os.remove(os.path.join(root, "cache", "2683eadbc5f1c588", "NIKL_PUBE_01.npz"))
        ok("CACHE_FILE_MISSING"
           in P.resolve_public_source("NIKL_PUBE_01", root=root)["reasons"],
           "a declared file that is not there is refused, not assumed")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 6. writing is opt-in
def test_writes():
    src = sources()
    pair = P._prepare_pair_from_arrays(annotation(), make_obs(), src)["pair"]
    ok(pair["provenance"]["verified"] is False,
       "a pair built from caller arrays stays marked unverified even when written")
    tmp = tempfile.mkdtemp(prefix="f21out_")
    try:
        existing = os.path.join(tmp, "already")
        os.makedirs(existing)
        try:
            P.write_prepared(pair, existing)
            ok(False, "an existing output directory is refused")
        except FileExistsError:
            ok(True, "an existing output directory is refused")

        fresh = os.path.join(tmp, "new")
        written = P.write_prepared(pair, fresh)
        ok(all(os.path.exists(written[k]["path"]) for k in ("original", "masked", "meta")),
           "an explicit write produces both variants and a provenance file")
        ok(len({written[k]["sha256"] for k in ("original", "masked")}) == 2,
           "the two variants are different files")
        doc = json.load(open(written["meta"]["path"], encoding="utf-8"))
        ok(doc["provenance"]["cache_sha256_declared_by_list"] == SHA and "variants" not in doc,
           "the provenance file records the source and does not carry the arrays")
        ok(doc["review_evidence"]["visual_observation"],
           "and it carries the review that allowed the removal")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 6. against the real lists
def test_real_lists_are_readable():
    """The public lists only: no clip is decoded, no cache is opened, no model is loaded."""
    src = P.load_sources(os.path.join(ROOT, "config", "source_manifest.json"),
                         os.path.join(ROOT, "runs", "union93_candidate_v1", "inputs.json"))
    ok(len(src) == 18, "the frozen training list joins to the public manifest for 18 clips")
    ok(all(v["split_role"] in P.TRAINING_ROLES for v in src.values()),
       "every one of them carries a training split role")
    ok(all(v["word_id"] == v["manifest_word_id"] for v in src.values()),
       "and the two lists agree on the word")
    # no eligible record exists yet, and none may be invented here
    made_up = {"schema": P.ANNOTATION_SCHEMA, "clip_id": "NIKL_NA_01", "word_id": "NA",
               "cache_sha256": src["NIKL_NA_01"]["npz_sha256"],
               "hand_roles": {"left": "non_sign_hand", "right": "sign_hand"},
               "basis": "low_motion", "reviewed_by": "guessed_from_the_word",
               "reviewed_interval": {"start_frame": 0, "end_frame": 999},
               "covers_whole_sign_interval": True}
    ok(not P.check_annotation(made_up, src)["eligible"],
       "an annotation guessed from a word name is refused on the real list too")


# --------------------------------------------- 8. root's five reviewed public annotations
def test_root_public_annotations():
    """Read-only. The five annotations root authored, run through the real verified boundary.

    Their contents are not modified here. No model is loaded and no participant file is opened.
    """
    if not os.path.exists(ANNOTATIONS_20260914):
        print("SKIP  root's public annotation file is not present")
        return
    with open(ANNOTATIONS_20260914, encoding="utf-8") as fh:
        doc = json.load(fh)
    anns = doc["annotations"]
    ok(len(anns) == 5, "root authored five reviewed public annotations")

    prepared = {}
    for ann in anns:
        out = P.prepare_from_public_clip(ann)
        if not out["prepared"]:
            ok(False, f"{ann['clip_id']} prepares through the verified boundary: {out['reasons']}")
            continue
        prepared[ann["clip_id"]] = out["pair"]
    ok(len(prepared) == 5, "all five prepare through the verified boundary")

    for clip_id, pair in prepared.items():
        p_ = pair["provenance"]
        ok(p_["verified"] is True and p_["cache_sha256_actual"] == p_["cache_sha256_declared"]
           == p_["manifest_observation_sha256"],
           f"{clip_id}: the file's actual hash matches both public lists")
        ok(pair["review_evidence"]["visual_observation"]
           and pair["review_evidence"]["source_video_sha256"] == p_["source_video_sha256"],
           f"{clip_id}: the review is kept, and it is a review of THIS recording")
        ok(pair["masked_side"] == "left" and pair["grouping"]["independent_samples_added"] == 0,
           f"{clip_id}: one reviewed hand, and no new independent sample")
        ok(pair["variants"]["original"]["x"].shape == pair["variants"]["masked"]["x"].shape
           and pair["variants"]["original"]["x"].shape[1] == 280,
           f"{clip_id}: both variants share one span and the 280-channel contract")

    # an unreviewed clip stays out, however similar it looks
    excluded = doc["excluded_unreviewed_clips"][0]
    src = P.load_sources(os.path.join(ROOT, "config", "source_manifest.json"),
                         os.path.join(ROOT, "runs", "union93_candidate_v1", "inputs.json"))
    borrowed = dict(anns[0], clip_id=excluded, word_id=src[excluded]["word_id"],
                    cache_sha256=src[excluded]["npz_sha256"])
    out = P.prepare_from_public_clip(borrowed)
    ok(not out["prepared"] and "REVIEW_EVIDENCE_VIDEO_IS_NOT_THIS_CLIP" in out["reasons"],
       "one clip's review cannot be carried over to an unreviewed clip")


if __name__ == "__main__":
    test_eligibility()
    test_mask_is_exact()
    test_pair()
    test_pair_refusals()
    test_counterexamples()
    test_resolved_boundary()
    test_writes()
    test_real_lists_are_readable()
    test_root_public_annotations()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
