"""functional_22: the mechanics of the matched experiment, before anyone runs it.

Synthetic fixtures and a tiny model. This is NOT the experiment: the real run is 220 steps on the
real candidate and only root performs it, once, after review.

No participant capture, no camera, no server, no web registry. The public manifest, the frozen
training list and the five reviewed public caches are read.

Run: py -3.13 tests/test_r19_visibility_experiment.py
"""
import copy
import json
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

import tools_visibility_experiment as V  # noqa: E402
from wis_lab.core import passive_hand as PH  # noqa: E402
from wis_lab.core.observation import Observation  # noqa: E402

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ 1. the plan does nothing
def test_plan_has_no_side_effects():
    calls = []
    real = V.load_candidate
    V.load_candidate = lambda *a, **k: calls.append(1) or real(*a, **k)
    tmp = tempfile.mkdtemp(prefix="f22plan_")
    before = set(os.listdir(ROOT))
    try:
        doc = V.plan()
        ok(doc["would_train"] is False, "the plan says plainly that it would not train")
        ok(not calls, "the plan loads no model")
        ok(set(os.listdir(ROOT)) == before and not os.listdir(tmp),
           "the plan writes nothing, anywhere")
        ok(doc["starting_checkpoint"]["matches"] is True,
           "the plan checks the starting checkpoint's actual hash")
        ok(len(doc["eligible_clips"]) == 5 and not doc["refused_annotations"],
           "the five reviewed clips are eligible and nothing was refused")
        ok(all(v["verified_through_full_boundary"] for v in doc["eligible_clips"].values()),
           "each one went through the FULL approved boundary, not just the hash resolver")
        ok(all(v["reviewed_span"]["frames"] > 0 and v["opposite_hand_frames_in_span"] > 0
               for v in doc["eligible_clips"].values()),
           "so each carries the reviewed span and a surviving opposite hand")
        ok(all(v["side"] == "left" for v in doc["eligible_clips"].values()),
           "and the side comes from the review, not from this runner")
        ok(doc["setting"]["optimizer_steps"] == 220 and doc["setting"]["seed"] == 20260914
           and doc["setting"]["cpu_threads"] == 2
           and doc["setting"]["lr"] == {"encoder": 1e-5, "head": 1e-4},
           "the frozen setting is exactly the one root chose")
    finally:
        V.load_candidate = real
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 2. one starting point
def test_same_start():
    proof = V.starting_point_proof(V.SETTING)
    ok(proof["identical_state_dict"], "two rebuilds of the candidate carry identical weights")
    ok(proof["identical_logits_on_synthetic_input"],
       "and give identical logits on a synthetic input")
    ok(proof["class_count"] == 93, "all 93 class rows are preserved")
    _, classes = V.load_candidate()
    _, classes2 = V.load_candidate()
    ok(classes == classes2 and len(set(classes)) == 93,
       "the class ORDER is the candidate's own, and is stable")


# ------------------------------------------------------------------ 3. one schedule, both arms
def fake_pools(n_src=6, n_nikl=4, frames=10):
    src = [{"sample_id": f"S{i}", "class_index": i % 3,
            "x": np.random.RandomState(i).rand(frames, 280).astype(np.float32)}
           for i in range(n_src)]
    nikl = []
    for i in range(n_nikl):
        obs = make_obs(t=frames + 2)
        from pilot.web_input import active_span, to_native_like
        nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                      np.asarray(obs.usable, dtype=bool))
        lo, hi = active_span(present)
        nikl.append({"clip_id": f"NIKL_FAKE_{i}", "word_id": f"W{i}", "label": f"w{i}",
                     "class_index": i % 3, "nat": nat[lo:hi + 1], "present": present[lo:hi + 1],
                     "ts": np.asarray(obs.timestamp_us, dtype=np.int64)[lo:hi + 1]})
    return src, nikl


def make_obs(t=12, mask_left=False):
    from wis_lab.core.observation import (LEFT_HAND_SLICE, N_POINTS, POSE_L_WRIST, POSE_R_WRIST,
                                          POSE_SLICE, RIGHT_HAND_SLICE)
    rng = np.arange(t, dtype=np.float32)
    xy = np.full((t, N_POINTS, 2), np.nan, dtype=np.float32)
    observed = np.zeros((t, N_POINTS), dtype=bool)
    usable = np.zeros((t, N_POINTS), dtype=bool)
    conf = np.full((t, N_POINTS), np.nan, dtype=np.float32)
    hand_score = np.full((t, 2), np.nan, dtype=np.float32)
    xy[:, POSE_SLICE, 0] = (100.0 + rng)[:, None]
    xy[:, POSE_SLICE, 1] = 200.0
    xy[:, 11] = np.stack([80.0 + rng, np.full(t, 150.0)], axis=1)
    xy[:, 12] = np.stack([180.0 + rng, np.full(t, 150.0)], axis=1)
    xy[:, POSE_L_WRIST] = np.stack([90.0 + rng, np.full(t, 300.0)], axis=1)
    xy[:, POSE_R_WRIST] = np.stack([170.0 + rng, np.full(t, 300.0)], axis=1)
    observed[:, POSE_SLICE] = True
    usable[:, POSE_SLICE] = True
    conf[:, POSE_SLICE] = 0.9
    for sl, col in ((LEFT_HAND_SLICE, 0), (RIGHT_HAND_SLICE, 1)):
        if sl == LEFT_HAND_SLICE and mask_left:
            continue
        for f in range(t):
            xy[f, sl] = np.stack([np.arange(21, dtype=np.float32) + 10.0 * col + f,
                                  np.arange(21, dtype=np.float32) + 400.0], axis=1)
            observed[f, sl] = True
            usable[f, sl] = True
            hand_score[f, col] = 0.8
    obs = Observation(xy_px=xy, timestamp_us=(np.arange(t, dtype=np.int64) + 1) * 33_000,
                      frame_id=np.arange(t, dtype=np.int64), observed=observed, usable=usable,
                      conf=conf, hand_score=hand_score, sidecar={})
    obs.validate()
    return obs


def test_schedule_is_deterministic_and_shared():
    src, nikl = fake_pools()
    a = V.build_schedule(V.SETTING, src, nikl)
    b = V.build_schedule(V.SETTING, src, nikl)
    ok(a["digest"] == b["digest"], "the same seed and pools give the same schedule digest")
    ok(a["steps"] == b["steps"], "and literally the same draws")
    ok(len(a["steps"]) == 220, "220 steps are scheduled")

    draws = [d for st in a["steps"] for d in st["nikl"]]
    ok(len(draws) == 220 * 16, "every step draws 16 public clips")
    ok(all("mask_draw" in d for d in draws),
       "a masking coin is drawn for EVERY public draw, eligible or not")
    ok(all(len(st["source"]) == 16 for st in a["steps"]),
       "and 16 class-balanced source clips")
    coins = sum(1 for d in draws if d["mask_draw"])
    ok(1400 < coins < 2000, f"the coin is a fair 0.5 over 3520 draws (got {coins})")

    eligible = {"NIKL_FAKE_1": {"side": "left"}}
    counts = V.masked_draw_counts(a, eligible)
    ok(counts["coin_selected_masking"] == coins and counts["control_arm_masked"] == 0,
       "the control never masks, though it consumes the same coins")
    ok(counts["masked_in_augmentation_arm"]
       == sum(1 for d in draws if d["mask_draw"] and d["clip_id"] == "NIKL_FAKE_1"),
       "only the eligible clip's masking draws act, and only in the augmentation arm")
    ok(set(counts["per_clip"]) == {"NIKL_FAKE_1"},
       "an ineligible clip drawing the coin changes nothing")

    c = V.build_schedule(dict(V.SETTING, seed=1), src, nikl)
    ok(c["digest"] != a["digest"], "a different seed is a different schedule")


# ------------------------------------------------------------------ 4. what masking touches
def test_masking_touches_one_hand_only():
    src, nikl = fake_pools(n_nikl=1)
    item = nikl[0]
    before_nat = item["nat"].copy()
    before_pr = item["present"].copy()
    draw = {"tilt_deg": 3.5, "speed": 1.1, "mask_draw": True}

    intact = V.nikl_example(item, draw, None)
    masked = V.nikl_example(item, draw, "left")

    ok(np.array_equal(item["nat"], before_nat) and np.array_equal(item["present"], before_pr),
       "the source arrays are not mutated by a masked draw")
    ok(intact.shape == masked.shape,
       "the same tilt and speed give the same tensor shape in both arms")
    ok(np.allclose(intact[:, 0:18], masked[:, 0:18]),
       "the nine body points are identical, so the tilt was not re-estimated")
    ok(np.allclose(intact[:, 60:102], masked[:, 60:102]),
       "the opposite hand is identical in body-relative space")
    ok(np.allclose(intact[:, 144:186], masked[:, 144:186]),
       "and in its own hand-relative space")
    ok(np.allclose(intact[:, 279], masked[:, 279]), "the time base is identical")
    ok(masked[:, 237:258].sum() == 0 and intact[:, 237:258].sum() > 0,
       "only the masked hand's validity changed")
    ok(np.isfinite(masked).all(), "no NaN reaches the channels")

    other = V.nikl_example(item, {"tilt_deg": -2.0, "speed": 0.9, "mask_draw": True}, "left")
    ok(other.shape != masked.shape or not np.allclose(other, masked),
       "a different tilt and speed draw really does produce a different example")


def test_adapter_agrees_with_the_reviewed_helper():
    from pilot.web_input import active_span, to_native_like
    obs = make_obs(t=14)
    nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                 np.asarray(obs.usable, dtype=bool))
    lo, hi = active_span(present)
    a_nat, a_pr = V.mask_native(nat[lo:hi + 1], present[lo:hi + 1], "left")
    r_nat, r_pr = V.masked_like_reviewed(obs, "left")
    ok(np.array_equal(a_pr, r_pr[lo:hi + 1]),
       "on synthetic arrays the adapter marks exactly the points the reviewed helper does")
    ok(np.array_equal(a_nat[a_pr], r_nat[lo:hi + 1][r_pr[lo:hi + 1]]),
       "and leaves every surviving coordinate identical")

    # and on the five real public pairs
    anns = json.load(open(V.DEFAULT_ANNOTATIONS, encoding="utf-8"))["annotations"]
    found = V.eligible_clips(anns, verify_bytes=True)
    src = V.F17.load_json(os.path.join(ROOT, "config", "source_manifest.json"))
    by_clip = {c["clip_id"]: c for c in src["clips"]}
    agree = []
    for clip_id, e in found["eligible"].items():
        nat, present, ts = V.F17._native_from_nikl(
            os.path.join(ROOT, by_clip[clip_id]["observation_npz"]),
            by_clip[clip_id]["observation_sha256"])
        real_obs = Observation.load(os.path.join(ROOT, by_clip[clip_id]["observation_npz"]))
        full_pr = V._present_of(real_obs)
        lo, hi = active_span(full_pr)
        a_nat, a_pr = V.mask_native(nat, present, e["side"])
        r_nat, r_pr = V.masked_like_reviewed(real_obs, e["side"])
        agree.append(np.array_equal(a_pr, r_pr[lo:hi + 1])
                     and np.array_equal(a_nat[a_pr], r_nat[lo:hi + 1][r_pr[lo:hi + 1]]))
    ok(len(agree) == 5 and all(agree),
       "the two definitions of absent agree on all five reviewed public clips")


# ------------------------------------------------------------------ 5. the loop, in miniature
def tiny_candidate(classes):
    """Deterministic, so two calls really are the same starting point, as the runner demands."""
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN
    torch.manual_seed(0)
    m = MaskedSignTCN(width=8, dilations=(1, 2), kernel_size=3, dropout=0.1,
                      n_classes=len(classes))
    return m, list(classes)


def test_loop_and_outputs():
    import torch
    classes = ["w0", "w1", "w2"]
    src, nikl = fake_pools()
    setting = dict(V.SETTING, optimizer_steps=2)
    schedule = V.build_schedule(setting, src, nikl)
    eligible = {"NIKL_FAKE_1": {"side": "left"}}

    real = V.load_candidate
    V.load_candidate = lambda *a, **k: tiny_candidate(classes)
    try:
        a = V.run_arm(V.ARM_CONTROL, schedule, src, nikl, classes, eligible, setting, log=lambda *_: None)
        b = V.run_arm(V.ARM_MASKED, schedule, src, nikl, classes, eligible, setting,
                      log=lambda *_: None)
        ok(a["masked_examples_used"] == 0, "the control masks nothing")
        ok(b["masked_examples_used"] > 0, "the augmentation arm masks the eligible clip's draws")
        ok(a["batch_shape_digest"] == b["batch_shape_digest"],
           "both arms saw the same tensor shapes, step for step")
        ok(a["steps"] == b["steps"] == 2, "both arms ran the scheduled number of steps")
        ok(a["intermediate_checkpoints_kept"] == 0, "no intermediate checkpoint is kept")
        ok(a["loss_first"] != b["loss_first"] or b["masked_examples_used"] == 0,
           "the arms diverge exactly where a reviewed hand was masked, and nowhere else")
        ok(not torch.equal(a["model"].head.weight, b["model"].head.weight)
           if hasattr(a["model"], "head") else True,
           "so they end up in different places, which is the point of the comparison")

        # with nothing eligible the two arms must be indistinguishable: same weights, same draws,
        # same coins consumed, nothing masked. Any difference would mean something else moved.
        c = V.run_arm(V.ARM_CONTROL, schedule, src, nikl, classes, {}, setting,
                      log=lambda *_: None)
        d = V.run_arm(V.ARM_MASKED, schedule, src, nikl, classes, {}, setting,
                      log=lambda *_: None)
        ok(c["loss_first"] == d["loss_first"] and c["loss_last"] == d["loss_last"],
           "with no eligible clip the two arms are identical step for step")
        ok(d["masked_examples_used"] == 0,
           "because the coin is still consumed but nothing is eligible to mask")
    finally:
        V.load_candidate = real


def test_output_directory_must_be_new():
    tmp = tempfile.mkdtemp(prefix="f22out_")
    try:
        existing = os.path.join(tmp, "already")
        os.makedirs(existing)
        trained = []
        real = V.run_arm
        V.run_arm = lambda *a, **k: trained.append(1)
        try:
            V.run(existing)
            ok(False, "an existing output directory is refused")
        except FileExistsError:
            ok(True, "an existing output directory is refused")
        finally:
            V.run_arm = real
        ok(not trained, "and it is refused BEFORE any training starts")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------- 6. the preflight corrections, end to end
def real_na_item(label):
    """The real public NIKL_NA_01 arrays, through the same loader the run uses."""
    src = V.F17.load_json(os.path.join(ROOT, "config", "source_manifest.json"))
    c = next(x for x in src["clips"] if x["clip_id"] == "NIKL_NA_01")
    nat, present, ts = V.F17._native_from_nikl(os.path.join(ROOT, c["observation_npz"]),
                                               c["observation_sha256"])
    return {"clip_id": "NIKL_NA_01", "word_id": "NA", "label": label, "class_index": 0,
            "nat": nat, "present": present, "ts": ts}


def na_annotation_file(tmp, **over):
    doc = json.load(open(V.DEFAULT_ANNOTATIONS, encoding="utf-8"))
    ann = dict(next(a for a in doc["annotations"] if a["clip_id"] == "NIKL_NA_01"), **over)
    path = os.path.join(tmp, "one_annotation.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"annotations": [ann]}, fh, ensure_ascii=False)
    return path


def test_run_end_to_end_synthetically():
    """A miniature of the real command: two steps, a tiny model, one real public clip.

    This is NOT the experiment. It exists so the wiring - the label lookup, the resolved input
    manifest, the payload provenance and the two evaluation scopes - is exercised rather than
    asserted about.
    """
    import torch
    tmp = tempfile.mkdtemp(prefix="f22run_")
    label = "\ub098"                       # NIKL_NA_01's display label, as the class order has it
    classes = [label, "w1", "w2"]
    seen = {}

    def fake_source_split(split, mapping):
        seen[split] = dict(mapping)
        n = 4 if split == "train" else 3
        return [{"sample_id": f"{split}_{i}", "class_index": i % len(classes),
                 "label": classes[i % len(classes)],
                 "x": np.random.RandomState(i).rand(9, 280).astype(np.float32)}
                for i in range(n)]

    def fake_load_nikl(mapping):
        seen["nikl_mapping"] = dict(mapping)
        return [real_na_item(label)]

    real_split, real_nikl, real_cand = (V.F17.load_source_split, V.F17.load_nikl,
                                        V.load_candidate)
    V.F17.load_source_split = fake_source_split
    V.F17.load_nikl = fake_load_nikl
    V.load_candidate = lambda *a, **k: tiny_candidate(classes)
    try:
        out = os.path.join(tmp, "run1")
        doc = V.run(out, annotations_path=na_annotation_file(tmp),
                    setting=dict(V.SETTING, optimizer_steps=2),
                    expected_counts={"source_train": 4, "source_validation": 3, "nikl": 1,
                                     "reserved_test": 700},
                    log=lambda *_: None)

        # (1) the label lookup is the runner's own, and it really was used
        ok(seen.get("train") == {label: 0, "w1": 1, "w2": 2},
           "the run built the label->index map itself and passed it to the source loader")
        ok(seen.get("nikl_mapping", {}).get("NA") == 0,
           "and the public clip's word maps to its own class index")

        # (3) the actual ids are recorded, not withheld
        inputs = json.load(open(os.path.join(out, "inputs.json"), encoding="utf-8"))
        ok([r["id"] for r in inputs["source_train"]] == [f"train_{i}" for i in range(4)],
           "every training id is written out, by id")
        ok([r["id"] for r in inputs["source_validation"]] == [f"validation_{i}" for i in range(3)],
           "and every validation id")
        ok(inputs["nikl"][0]["reviewed_non_sign_hand"] == "left"
           and inputs["nikl"][0]["masked_variant_stays_with_this_source"] is True,
           "the public clip records its reviewed hand and stays with its own source")
        ok(inputs["parent_checkpoint_sha256"] == V.CANDIDATE_SHA256,
           "the input manifest names THIS run's parent checkpoint")
        ok(inputs["reserved_test"]["read"] is False
           and inputs["participant_captures"]["read"] is False,
           "and states that the reserved split and participant captures were not read")
        ok(doc["inputs"]["sha256"] and os.path.exists(os.path.join(out, "schedule.json")),
           "the manifest and the actual schedule are both written and hashed")
        ok("training_ids" not in json.dumps(doc)
           and doc["inputs"]["counts"] == inputs["counts"],
           "the old placeholder key is gone; the result points at the manifest and its counts")

        # (2) the payload describes this run, not functional_17's
        pay = torch.load(os.path.join(out, f"{V.ARM_MASKED}.final.pt"), map_location="cpu",
                         weights_only=False)
        ok(pay["parent_checkpoint_sha256"] == V.CANDIDATE_SHA256,
           "the saved payload's parent is the union-93 candidate")
        ok(pay["parent_checkpoint_sha256"] != V.F17.CHECKPOINT_SHA256,
           "not functional_17's base checkpoint")
        ok(pay["fixed_setting"]["optimizer_steps"] == 2
           and pay["fixed_setting"]["lr"] == {"encoder": 1e-5, "head": 1e-4},
           "the saved setting is the one this run actually used")
        ok(pay["fixed_setting"]["optimizer_steps"] != 660
           and pay["augmentation"] is not V.F17.AUGMENT,
           "and neither the 660-step setting nor functional_17's augmentation object is embedded")
        ok(pay["arm"] == V.ARM_MASKED and pay["produced_by"] == "functional_22",
           "the payload says which arm produced it")
        ok(pay["classes"] == classes and pay["n_classes"] == len(classes),
           "the class order travels with the weights")
        ok(pay["annotations_sha256"] and pay["code_hashes"],
           "so do the annotation and code hashes")

        # (5) both public scopes, from one pass, tagged and never mixed
        rob = doc["arms"][V.ARM_MASKED]["evaluation"]["training_reference_robustness"]
        ok(rob["scopes_from_one_forward_pass"] is True, "both scopes come from one forward pass")
        ok(rob["full93"]["candidate_count"] == len(classes)
           and rob["project18"]["candidate_count"] == 1,
           "and each carries its own candidate count")
        ok("PROJECT18" in rob["scope_warning"] and "3/5" in rob["scope_warning"],
           "the earlier 3/5 diagnostic is named as PROJECT18, with no full93 counterpart")
        ok(set(doc["comparison"]["training_reference_robustness"]) == {"full93", "project18"},
           "the comparison is reported per scope")
        ok(doc["arms"][V.ARM_MASKED]["evaluation"]["source_validation_scope"].startswith("FULL93"),
           "the source validation stays in the full 93 scope only")

        # and the run refuses to write twice into the same place
        try:
            V.run(out, annotations_path=na_annotation_file(tmp), log=lambda *_: None)
            ok(False, "a second run into the same directory is refused")
        except FileExistsError:
            ok(True, "a second run into the same directory is refused")
    finally:
        V.F17.load_source_split, V.F17.load_nikl, V.load_candidate = (real_split, real_nikl,
                                                                      real_cand)
        shutil.rmtree(tmp, ignore_errors=True)


def test_full_boundary_is_not_bypassed():
    """A clip whose hash resolves but whose review does not cover the sign must be refused."""
    tmp = tempfile.mkdtemp(prefix="f22bnd_")
    try:
        short = na_annotation_file(tmp, reviewed_interval={"start_frame": 30, "end_frame": 31})
        anns = json.load(open(short, encoding="utf-8"))["annotations"]
        resolver = PH.resolve_public_source("NIKL_NA_01")
        ok(resolver["ok"], "the hash resolver alone is satisfied by this clip")
        found = V.eligible_clips(anns, verify_bytes=True)
        ok(not found["eligible"] and found["refused"],
           "but the full boundary refuses it, so it never becomes eligible")
        ok("REVIEWED_INTERVAL_DOES_NOT_COVER_THE_SIGN" in found["refused"][0]["reasons"],
           "and says the review does not cover the sign")
        loose = V.eligible_clips(anns, verify_bytes=False)
        ok(loose["eligible"] and not loose["eligible"]["NIKL_NA_01"][
               "verified_through_full_boundary"],
           "the plan's cheap pass admits it but marks it unverified, and a run never uses that")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_augmentation_policy_is_this_runs_own():
    ok(V.AUGMENT["forbidden"] is not V.F17.AUGMENT["forbidden"]
       and V.AUGMENT["speed_range"] is not V.F17.AUGMENT["speed_range"],
       "no nested object is shared with functional_17, so it cannot be mutated through this one")
    before = json.dumps(V.F17.AUGMENT, ensure_ascii=False, sort_keys=True)
    V.AUGMENT["forbidden"].append("scratch")
    V.AUGMENT["speed_range"].append(9.9)
    after = json.dumps(V.F17.AUGMENT, ensure_ascii=False, sort_keys=True)
    V.AUGMENT["forbidden"].pop()
    V.AUGMENT["speed_range"].pop()
    ok(before == after, "writing through this runner's copy leaves functional_17's untouched")
    ex = V.AUGMENT["reviewed_non_sign_hand_missingness"]
    ok(ex["probability"] == 0.5 and "augmentation arm only" in ex["applies_to"],
       "the reviewed-missingness exception is described, not inherited as 'occlusion forbidden'")
    ok("occlusion" in V.AUGMENT["forbidden"]
       and any("not reviewed" in f for f in V.AUGMENT["forbidden"]),
       "occlusion stays forbidden, and so does masking an unreviewed hand")
    ok(V.AUGMENT["tilt_degrees"] == V.F17.AUGMENT["tilt_degrees"]
       and V.AUGMENT["speed_range"] == list(V.F17.AUGMENT["speed_range"]),
       "the tilt and speed ranges are still functional_17's values")


def test_counts_must_match_the_public_lists():
    try:
        V.resolve_inputs([], [], [], ["a"], {}, {"path": "x", "sha256": "y"},
                         {"source_train": 1, "source_validation": 1, "nikl": 1,
                          "reserved_test": 1})
        ok(False, "a resolved count that does not match the expectation stops the run")
    except RuntimeError:
        ok(True, "a resolved count that does not match the expectation stops the run")


# ------------------------------------------------------------------ 7. what must never happen
def test_role_separation():
    try:
        V.assert_roles_disjoint({"a", "b"}, {"b"})
        ok(False, "a validation sample in the training pool stops the run")
    except RuntimeError:
        ok(True, "a validation sample in the training pool stops the run")
    V.assert_roles_disjoint({"a"}, {"b"})
    ok(True, "disjoint pools pass")

    frozen = json.load(open(os.path.join(ROOT, "runs", "union93_candidate_v1", "inputs.json"),
                            encoding="utf-8"))
    tr = {r["id"] if isinstance(r, dict) and "id" in r else r for r in frozen["source_train"]}
    va = {r["id"] if isinstance(r, dict) and "id" in r else r for r in frozen["source_validation"]}
    ok(tr and va and not (tr & va),
       "the frozen source lists this experiment reuses are already disjoint")


def test_no_forbidden_access():
    """Check the CODE, not the prose: the docstring says 'camera' precisely to disclaim one."""
    import tokenize
    path = os.path.join(ROOT, "tools_visibility_experiment.py")
    with open(path, encoding="utf-8") as fh:
        toks = list(tokenize.generate_tokens(fh.readline))
    code = " ".join(t.string for t in toks
                    if t.type not in (tokenize.COMMENT, tokenize.STRING))
    literals = [t.string for t in toks if t.type == tokenize.STRING]

    machinery = ["multiprocessing", "threading", "subprocess", "Popen", "DataLoader", "fork",
                 "ProcessPool", "ThreadPool", "webcam", "VideoCapture"]
    hits = [w for w in machinery if w in code]
    ok(not hits, f"no worker pool, background process or capture device is used in code {hits}")

    participant = ["Recordings", "trial_reviews", ".result.json", "persistentDataPath",
                   "LocalLow", "sessions/"]
    bad = [lit for lit in literals if any(f in lit for f in participant)]
    ok(not bad, f"no string in the runner points at a participant path {bad[:3]}")
    ok(any("source_manifest" in lit for lit in literals)
       and any("union93_candidate_v1" in lit for lit in literals),
       "the only paths it names are the public manifest and the frozen candidate run")
    doc = V.plan()
    ok(doc["evaluation"]["participant_captures"].startswith("not read"),
       "the plan states that no participant capture is read or evaluated")
    ok(doc["evaluation"]["reserved_700"] == "not read", "and that the reserved 700 stay untouched")


if __name__ == "__main__":
    test_plan_has_no_side_effects()
    test_same_start()
    test_schedule_is_deterministic_and_shared()
    test_masking_touches_one_hand_only()
    test_adapter_agrees_with_the_reviewed_helper()
    test_loop_and_outputs()
    test_output_directory_must_be_new()
    test_run_end_to_end_synthetically()
    test_full_boundary_is_not_bypassed()
    test_augmentation_policy_is_this_runs_own()
    test_counts_must_match_the_public_lists()
    test_role_separation()
    test_no_forbidden_access()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
