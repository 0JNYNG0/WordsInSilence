"""functional_22: one matched continuation experiment, frozen BEFORE it is run.

The question is narrow: does extra training PLUS reviewed passive-hand missingness do anything
that extra training alone does not? Two arms, same starting weights, same 220 steps, same data,
same draws. The only difference is that in one arm a NIKL draw whose coin came up masking, on a
clip root actually reviewed, has that reviewed non-signing hand marked absent.

This is a targeted development and robustness experiment. It is NOT a generalisation measurement,
NOT an execution-quality evaluation, and it promotes nothing.

Running it is deliberately awkward: the default invocation only prints and validates the plan.
Training happens only with --run and a NEW output directory, after root has reviewed this file.

  py -3.13 tools_visibility_experiment.py                      # plan only, no training
  py -3.13 tools_visibility_experiment.py --run --out runs/visibility_v1

No server, camera, web registry, Unity, backup, git or new dependency is touched. No participant
capture is read, here or in the evaluation. The frozen functional_17 code, the pinned base
checkpoint and the union-93 candidate are read only and never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import tools_prep_union93_candidate as F17  # noqa: E402  (read-only reuse, never modified)
from wis_lab.core import passive_hand as PH  # noqa: E402

TASK_ID = "functional_22"
CANDIDATE_REL = os.path.join("runs", "union93_candidate_v1", "union93_final.pt")
CANDIDATE_SHA256 = "e7c9c31a9c747e2c12a9a56695d2e4e9dbec9c372c8e2a3f209d00fa4c25dced"
REFERENCE_REL = os.path.join("runs", "union93_candidate_v1", "result.json")
DEFAULT_ANNOTATIONS = os.path.join(
    "C:/Users/local-user/.codex/.chatgpt-projects/g-p-6a12a39d590c8191960741b826fa75ff",
    "WordsInSilence_\uae30\ub2a5\uc644\uc131_20260913",
    "public_passive_hand_annotations_20260914.json")

ARM_CONTROL = "control_extra_training_only"
ARM_MASKED = "extra_training_plus_reviewed_missingness"
ARMS = (ARM_CONTROL, ARM_MASKED)

# Fixed before any training, and not searched over.
SETTING = {
    "seed": 20260914,
    "optimizer_steps": 220,
    "batch": {"total": 32, "source": 16, "nikl": 16},
    "source_sampling": "class balanced over the union classes",
    "nikl_sampling": "uniform over the 18 public training clips",
    "lr": {"encoder": 1e-5, "head": 1e-4},
    "optimizer": "AdamW", "weight_decay": 1e-3,
    "cpu_threads": 2, "data_loading_workers": 0,
    "mask_probability": 0.5,
    "mask_draw_rule": "a Bernoulli(0.5) is drawn for EVERY NIKL draw in BOTH arms and consumed "
                      "whether or not the clip is eligible, so the two arms share one random "
                      "stream. The control ignores the outcome; the masked arm acts on it only "
                      "when the clip is one root reviewed.",
    "report_step": "the FINAL step only; no best-epoch selection and no step chosen after seeing "
                   "results",
}
# The same tilt and speed ranges as functional_17, DEEP copied so no nested list is shared with
# that tool, and with this run's own policy statement. Inheriting functional_17's blanket
# "occlusion forbidden" would have described the opposite of what one arm here deliberately does.
AUGMENT = {
    "tilt_degrees": float(F17.AUGMENT["tilt_degrees"]),
    "speed_range": [float(v) for v in F17.AUGMENT["speed_range"]],
    "reviewed_non_sign_hand_missingness": {
        "applies_to": "the augmentation arm only, and only to a clip whose non-signing hand root "
                      "reviewed over the whole sign",
        "what": "that one detailed hand is marked absent before normalisation; body pose, the "
                "opposite hand and the timestamps are untouched",
        "why_it_is_not_occlusion": "nothing is hidden behind anything. A hand a reviewer found "
                                   "does not participate is reported as unobserved, which is a "
                                   "state the contract already has a name for.",
        "probability": 0.5,
        "never": "an unreviewed hand, both hands, a hand chosen by word name, by low motion or by "
                 "a model's output",
    },
    "forbidden": ["occlusion", "noise", "mirror", "handshape change", "time reversal",
                  "arbitrary cropping", "per-word augmentation",
                  "masking any hand that was not reviewed as non-signing"],
    "inherited_from": "functional_17 tilt and speed only; its forbidden list is restated here "
                      "rather than borrowed, and F17.AUGMENT is never mutated",
}
EXPECTED_COUNTS = {"source_train": 2095, "source_validation": 696, "nikl": 18,
                   "reserved_test": 700}
# native 67-slot layout: BODY_25 0-24, left hand 25-45, right hand 46-66
NATIVE_HAND = {"left": slice(25, 46), "right": slice(46, 67)}
NATIVE_SHOULDERS = (5, 2)            # BODY_25 left/right shoulder slots, as tilt_clip uses them

SOURCE_VALIDATION_REFERENCE = {"correct": 683, "total": 696,
                               "accuracy": 683 / 696,
                               "from": "runs/union93_candidate_v1/result.json "
                                       "source_validation_full_93"}
MAX_ACCURACY_REGRESSION_PP = 2.0
REFERENCE_DOMAIN = "TRAINING_REFERENCE_ROBUSTNESS"


def sha256_file(path: str) -> str:
    return F17.sha256_file(path)


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- the reviewed annotations
def load_annotations(path: str) -> dict:
    """Root's reviewed public annotations. Read only; their contents are never edited here."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return {"path": path, "sha256": sha256_file(path), "doc": doc,
            "annotations": list(doc.get("annotations", []))}


def eligible_clips(annotations: list, verify_bytes: bool, root: str = ROOT) -> dict:
    """-> {clip_id: {side, ...}}. Eligibility comes from functional_21, never from this file.

    With verify_bytes the FULL approved boundary runs - prepare_from_public_clip - which resolves
    the clip through both public lists, computes the cache file's actual sha256, loads the arrays
    itself, and then checks the things a hash cannot: that the reviewed interval covers the active
    span, and that an observed opposite hand survives the removal. Checking only the resolver
    would have admitted a pair the review does not actually cover.

    Without verify_bytes only the list-level review check runs. That is the plan's cheap pass, and
    the plan says so; a run always takes the full boundary.
    """
    sources = PH.load_sources(os.path.join(root, PH.MANIFEST_REL),
                              os.path.join(root, PH.TRAINING_INPUTS_REL))
    out, refused = {}, []
    for ann in annotations:
        checked = PH.check_annotation(ann, sources)
        if not checked["eligible"]:
            refused.append({"clip_id": ann.get("clip_id"), "reasons": checked["reasons"]})
            continue
        rec = checked["record"]
        entry = {"clip_id": rec["clip_id"], "word_id": rec["word_id"], "side": rec["side"],
                 "cache_sha256": rec["cache_sha256"], "split_role": rec["split_role"],
                 "review_evidence": rec["review_evidence"],
                 "annotation_is": rec["annotation_is"],
                 "annotation": ann,           # the decision stays bound to what root wrote
                 "verified_through_full_boundary": False}
        if verify_bytes:
            pair = PH.prepare_from_public_clip(ann, root=root)
            if not pair["prepared"]:
                refused.append({"clip_id": rec["clip_id"], "reasons": pair["reasons"]})
                continue
            p = pair["pair"]
            entry.update(
                verified_through_full_boundary=True,
                cache_sha256_actual=p["provenance"]["cache_sha256_actual"],
                source_video_sha256=p["provenance"]["source_video_sha256"],
                reviewed_span={"first_frame": p["span"]["first_frame"],
                               "last_frame": p["span"]["last_frame"],
                               "frames": p["span"]["frames"]},
                opposite_hand_frames_in_span=p["opposite_hand_frames_in_span"],
                boundary="wis_lab.core.passive_hand.prepare_from_public_clip")
            if p["masked_side"] != rec["side"]:
                raise RuntimeError(f"{rec['clip_id']}: the boundary masked a different side")
        out[rec["clip_id"]] = entry
    return {"eligible": out, "refused": refused}



# --------------------------------------------------------------------- the masking adapter
def mask_native(nat: np.ndarray, present: np.ndarray, side: str):
    """Mark one detailed hand absent in the NATIVE arrays, before any normalisation.

    A copy is returned; the caller's arrays are never touched. Absent in this layout means
    present=False with the coordinate zeroed, exactly as to_native_like writes an absent slot -
    the zero is unreachable because the mask says so, and build_channels reads the mask, not the
    number.

    This adapter exists because the training clips are already span-trimmed native arrays. It is
    checked against the reviewed Observation helper on synthetic arrays AND on all five public
    pairs, so it cannot drift into a second, quieter definition of "absent".
    """
    if side not in NATIVE_HAND:
        raise ValueError(f"side must be left or right, got {side!r}")
    nat2 = np.array(nat, dtype=np.float32, copy=True)
    pr2 = np.array(present, dtype=bool, copy=True)
    sl = NATIVE_HAND[side]
    nat2[:, sl, :] = 0.0
    pr2[:, sl] = False
    return nat2, pr2


def masked_like_reviewed(obs, side: str):
    """The reviewed path, for comparison only: mask the Observation, then convert."""
    from pilot.web_input import to_native_like
    m = PH.mask_hand(obs, side)
    return to_native_like(np.asarray(m.xy_px, dtype=np.float32), np.asarray(m.usable, dtype=bool))


# --------------------------------------------------------------------- the shared schedule
def build_schedule(setting: dict, source_train: list, nikl: list) -> dict:
    """Every draw both arms will make, decided once, before either arm starts.

    Source picks, NIKL picks, tilt, speed and the masking coin all come from this one stream, so
    the arms cannot diverge in what they saw - only in what was done to it. The coin is drawn for
    every NIKL pick regardless of eligibility, because a stream that skipped ineligible clips
    would no longer line up between the arms.
    """
    rng = np.random.default_rng(setting["seed"])
    by_class = {}
    for c in source_train:
        by_class.setdefault(c["class_index"], []).append(c["sample_id"])
    class_keys = sorted(by_class)
    nikl_ids = [it["clip_id"] for it in nikl]
    lo, hi = AUGMENT["speed_range"]
    steps = []
    for _ in range(int(setting["optimizer_steps"])):
        src = []
        for _ in range(setting["batch"]["source"]):
            k = class_keys[int(rng.integers(len(class_keys)))]
            j = int(rng.integers(len(by_class[k])))
            src.append({"class_index": k, "within": j, "sample_id": by_class[k][j]})
        nk = []
        for _ in range(setting["batch"]["nikl"]):
            i = int(rng.integers(len(nikl_ids)))
            deg = float(rng.uniform(-AUGMENT["tilt_degrees"], AUGMENT["tilt_degrees"]))
            speed = float(rng.uniform(lo, hi))
            coin = bool(rng.random() < setting["mask_probability"])
            nk.append({"index": i, "clip_id": nikl_ids[i], "tilt_deg": round(deg, 6),
                       "speed": round(speed, 6), "mask_draw": coin})
        steps.append({"source": src, "nikl": nk})
    return {"steps": steps, "digest": _sha256_obj(steps),
            "meaning": "one schedule, consumed identically by both arms"}


def masked_draw_counts(schedule: dict, eligible: dict) -> dict:
    """How often the coin came up masking, and how often that actually masked something."""
    coins = acted = 0
    per_clip = {}
    for st in schedule["steps"]:
        for d in st["nikl"]:
            coins += int(d["mask_draw"])
            if d["mask_draw"] and d["clip_id"] in eligible:
                acted += 1
                per_clip[d["clip_id"]] = per_clip.get(d["clip_id"], 0) + 1
    total = len(schedule["steps"]) * len(schedule["steps"][0]["nikl"]) if schedule["steps"] else 0
    return {"nikl_draws": total, "coin_selected_masking": coins,
            "masked_in_augmentation_arm": acted, "per_clip": per_clip,
            "control_arm_masked": 0,
            "meaning": "the coin is drawn and consumed in both arms; only the augmentation arm "
                       "acts on it, and only for a clip root reviewed"}


# --------------------------------------------------------------------- one batch item
def nikl_example(item: dict, draw: dict, mask_side: str | None):
    """One NIKL draw -> channels. Masking happens first, in native space, before normalisation.

    The span was fixed when the clip was loaded, from the INTACT clip. It is not recomputed here:
    asking the masked arrays where the sign starts would move the window as well as the hand, and
    the two arms would no longer differ in one thing.
    """
    from pilot.data import build_channels
    nat, present = item["nat"], item["present"]
    if mask_side:
        nat, present = mask_native(nat, present, mask_side)
    nat, present = F17.tilt_clip(nat, present, draw["tilt_deg"])
    nat, present, ts = F17.time_resample(nat, present, item["ts"], draw["speed"])
    return build_channels(nat, present, ts)["x"]


def assert_roles_disjoint(train_ids, val_ids) -> None:
    """No validation sample may be trained on. Checked before a single step, never after."""
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise RuntimeError(f"{len(overlap)} validation samples appear in the training pool: "
                           f"{overlap[:5]}")


def _shoulders_survive(item: dict) -> bool:
    """tilt_clip centres on the shoulders and falls back to the mean of whatever is present.

    If a clip ever lost both shoulders, that fallback would be computed from a different point set
    in the masked arm, and the tilt would silently stop being the same tilt. Shoulders are body
    pose and masking never touches them, so this only has to be true once - but it is checked
    rather than assumed.
    """
    ok = item["present"][:, NATIVE_SHOULDERS[0]] & item["present"][:, NATIVE_SHOULDERS[1]]
    return bool(ok.any())


# --------------------------------------------------------------------- one arm
def run_arm(arm: str, schedule: dict, source_train: list, nikl: list, classes: list,
            eligible: dict, setting: dict, log=print) -> dict:
    """A fresh model, a fresh optimiser and reset seeds, then the shared schedule."""
    import torch
    from pilot.data import collate

    F17.set_seeds(setting["seed"])                  # identical dropout stream in both arms
    model, classes_in = load_candidate()
    if list(classes_in) != list(classes):
        raise RuntimeError("the candidate's class order changed between arms")
    head = [p for n, p in model.named_parameters() if n.startswith("head.")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": setting["lr"]["encoder"]},
                             {"params": head, "lr": setting["lr"]["head"]}],
                            weight_decay=setting["weight_decay"])
    by_id = {c["sample_id"]: c for c in source_train}
    by_clip = {it["clip_id"]: it for it in nikl}

    losses, shapes, masked_here = [], [], 0
    model.train()
    for n_step, st in enumerate(schedule["steps"]):
        batch = []
        for pick in st["source"]:
            c = by_id[pick["sample_id"]]
            batch.append({"x": c["x"], "frames": int(len(c["x"])), "label": c["class_index"]})
        for draw in st["nikl"]:
            it = by_clip[draw["clip_id"]]
            side = None
            if arm == ARM_MASKED and draw["mask_draw"] and draw["clip_id"] in eligible:
                side = eligible[draw["clip_id"]]["side"]
                masked_here += 1
            x = nikl_example(it, draw, side)
            batch.append({"x": x, "frames": int(len(x)), "label": it["class_index"]})
        x, pad, y = collate(batch)
        shapes.append(list(x.shape))
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))
        loss = torch.nn.functional.cross_entropy(logit, torch.from_numpy(y))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {n_step}: {loss.item()!r}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if n_step % 40 == 0 or n_step == len(schedule["steps"]) - 1:
            log(f"[{arm}] step {n_step + 1}/{len(schedule['steps'])} loss {losses[-1]:.4f}")
    model.eval()
    return {"arm": arm, "model": model, "steps": len(losses),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_mean_last_20": float(np.mean(losses[-20:])) if losses else None,
            "masked_examples_used": masked_here,
            "batch_shape_digest": _sha256_obj(shapes),
            "intermediate_checkpoints_kept": 0}


def load_candidate(path: str | None = None):
    """Rebuild the frozen union-93 candidate from its own payload, hash-checked first."""
    import torch
    p = path or os.path.join(ROOT, CANDIDATE_REL)
    got = sha256_file(p)
    if got != CANDIDATE_SHA256:
        raise ValueError(f"the candidate checkpoint is not the pinned one: {got}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    model, classes = F17.rebuild_from_payload(payload)
    return model, classes


def starting_point_proof(setting: dict) -> dict:
    """Both arms really do start from the same weights. Checked on a synthetic input only."""
    import torch
    a, classes = load_candidate()
    b, _ = load_candidate()
    a.eval()
    b.eval()          # stated here rather than trusted: dropout would make the probe meaningless
    same = all(torch.equal(pa, pb) for pa, pb in zip(a.state_dict().values(),
                                                     b.state_dict().values()))
    probe = np.zeros((1, 8, 280), dtype=np.float32)
    probe[0, :, 279] = 0.033
    pad = np.ones((1, 8), dtype=bool)
    with torch.inference_mode():
        la = a(torch.from_numpy(probe), torch.from_numpy(pad))[0].numpy()
        lb = b(torch.from_numpy(probe), torch.from_numpy(pad))[0].numpy()
    return {"identical_state_dict": bool(same),
            "identical_logits_on_synthetic_input": bool(np.array_equal(la, lb)),
            "probe": "a zero-filled synthetic clip with a constant dt; no private data is used "
                     "for this check",
            "class_count": len(classes), "class_order_sha256": _sha256_obj(list(classes))}


# --------------------------------------------------------------------- evaluation
def reference_rows(model, clips: list, classes: list, project18: list) -> list:
    """One forward per clip, read in BOTH public scopes. No second model evaluation.

    A prediction over 93 candidates and a prediction over 18 are different tasks and are reported
    as such, side by side, from the same logits. They are never mixed, and a number from one scope
    is never set against a number from the other.
    """
    import torch
    from pilot.data import collate
    idx18 = [classes.index(c) for c in project18]
    out = []
    with torch.inference_mode():
        for c in clips:
            x, pad, _ = collate([{"x": c["x"], "frames": int(len(c["x"])), "label": 0}])
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
            true_idx = classes.index(c["label"])
            row = {"id": c.get("clip_id") or c.get("sample_id"), "true": c["label"],
                   "condition": c.get("condition", "intact")}
            for scope, idx in (("full93", list(range(len(classes)))), ("project18", idx18)):
                order = [idx[i] for i in np.argsort(-logit[idx])]
                pred = classes[order[0]]
                rank = (order.index(true_idx) + 1) if true_idx in order else None
                row[scope] = {"pred": pred, "true_label_rank": rank,
                              "candidate_count": len(idx),
                              "correct": bool(pred == c["label"])}
            out.append(row)
    return out


def _scope_block(rows: list, scope: str) -> dict:
    intact = [r for r in rows if r["condition"] == "intact"]
    masked = [r for r in rows if r["condition"] == "reviewed_masked"]
    paired = []
    for m in masked:
        base = m["id"].split("::")[0]
        i = next((r for r in intact if r["id"] == base), None)
        paired.append({"clip_id": base,
                       "intact_correct": bool(i and i[scope]["correct"]),
                       "masked_correct": m[scope]["correct"],
                       "intact_rank": i[scope]["true_label_rank"] if i else None,
                       "masked_rank": m[scope]["true_label_rank"],
                       "masked_pred": m[scope]["pred"]})
    return {
        "scope": scope,
        "candidate_count": (rows[0][scope]["candidate_count"] if rows else None),
        "intact_n": len(intact),
        "intact_correct": sum(1 for r in intact if r[scope]["correct"]),
        "masked_n": len(masked),
        "masked_correct": sum(1 for r in masked if r[scope]["correct"]),
        "paired": paired,
        "intact_identification_lost": [r["id"] for r in intact if not r[scope]["correct"]],
    }


def reference_clips(nikl: list, eligible: dict) -> list:
    """The 18 intact public training references, plus exactly the five reviewed masked ones."""
    from pilot.data import build_channels
    out = []
    for it in nikl:
        x = build_channels(it["nat"], it["present"], it["ts"])["x"]
        out.append({"clip_id": it["clip_id"], "label": it["label"], "x": x,
                    "condition": "intact"})
    for it in nikl:
        e = eligible.get(it["clip_id"])
        if not e:
            continue
        nat, present = mask_native(it["nat"], it["present"], e["side"])
        x = build_channels(nat, present, it["ts"])["x"]
        out.append({"clip_id": it["clip_id"] + "::masked_" + e["side"], "label": it["label"],
                    "x": x, "condition": "reviewed_masked"})
    return out



def evaluate_arm(model, source_val: list, nikl: list, classes: list, eligible: dict) -> dict:
    """Two domains, named apart, and neither of them is a fresh-accuracy figure."""
    val = F17.evaluate(model, source_val, classes)
    acc = val["accuracy"]
    drop_pp = (SOURCE_VALIDATION_REFERENCE["accuracy"] - acc) * 100.0
    project18 = sorted({it["label"] for it in nikl})
    rows = reference_rows(model, reference_clips(nikl, eligible), classes, project18)
    return {
        "source_validation_full_93": {k: v for k, v in val.items() if k != "rows"},
        "source_validation_rows": val["rows"],
        "source_validation_scope": "FULL93 only. The source task is never read in the 18-word "
                                   "scope.",
        "source_validation_reference": SOURCE_VALIDATION_REFERENCE,
        "accuracy_regression_pp": round(drop_pp, 4),
        "continuity_gate": {
            "max_allowed_regression_pp": MAX_ACCURACY_REGRESSION_PP,
            "passed": bool(drop_pp <= MAX_ACCURACY_REGRESSION_PP),
            "meaning": "a continuity check that the source task did not fall apart. It is NOT a "
                       "claim of superiority over the existing candidate.",
        },
        "training_reference_robustness": {
            "domain": REFERENCE_DOMAIN,
            "is_not": "fresh accuracy, generalisation, or an execution-quality figure. Every clip "
                      "here was in this model's own training material.",
            "scopes_from_one_forward_pass": True,
            "full93": _scope_block(rows, "full93"),
            "project18": _scope_block(rows, "project18"),
            "project18_membership": project18,
            "scope_warning": "93 candidates and 18 candidates are different tasks. A figure from "
                             "one scope is never compared with a figure from the other - the "
                             "earlier 3/5 masked diagnostic was PROJECT18 and has no full93 "
                             "counterpart here.",
            "rows": rows,
            "span": "the original active span, used as the diagnostic control for both "
                    "conditions",
        },
        "participant_capture_evaluated": False,
    }



# --------------------------------------------------------------------- the plan
def plan(annotations_path: str = DEFAULT_ANNOTATIONS, verify_bytes: bool = True,
         root: str = ROOT) -> dict:
    """What would run, and whether it could. No model, no arrays, no training, no writes."""
    ann = load_annotations(annotations_path)
    found = eligible_clips(ann["annotations"], verify_bytes=verify_bytes, root=root)
    cand = os.path.join(root, CANDIDATE_REL)
    cand_sha = sha256_file(cand) if os.path.exists(cand) else None
    return {
        "task_id": TASK_ID,
        "would_train": False,
        "arms": {ARM_CONTROL: "extra training only; every NIKL draw stays intact",
                 ARM_MASKED: "the same extra training; a NIKL draw whose coin selected masking, "
                             "on a clip root reviewed, has that reviewed non-signing hand marked "
                             "absent"},
        "difference_between_arms": "one thing only: the reviewed hand's observation. Same start, "
                                   "same steps, same data, same draws, same shapes.",
        "setting": SETTING,
        "augmentation": AUGMENT,
        "starting_checkpoint": {"path": CANDIDATE_REL, "expected_sha256": CANDIDATE_SHA256,
                                "actual_sha256": cand_sha,
                                "matches": cand_sha == CANDIDATE_SHA256},
        "annotations": {"path": ann["path"], "sha256": ann["sha256"],
                        "count": len(ann["annotations"]),
                        "read_only": "their contents are never edited by this runner"},
        "eligible_clips": {k: {"word_id": v["word_id"], "side": v["side"],
                               "verified_through_full_boundary":
                                   v["verified_through_full_boundary"],
                               "reviewed_span": v.get("reviewed_span"),
                               "opposite_hand_frames_in_span":
                                   v.get("opposite_hand_frames_in_span")}
                           for k, v in found["eligible"].items()},
        "refused_annotations": found["refused"],
        "eligibility_rule": "functional_21 decides it. This runner adds no rule of its own, no "
                            "label branch and no inference about an unreviewed hand.",
        "evaluation": {
            "source_validation": "the existing 696 validation clips in the full 93-label scope, "
                                 f"against {SOURCE_VALIDATION_REFERENCE['correct']}/"
                                 f"{SOURCE_VALIDATION_REFERENCE['total']}, continuity gate "
                                 f"{MAX_ACCURACY_REGRESSION_PP} pp",
            "training_references": "all 18 intact public training clips plus exactly the five "
                                   "reviewed masked conditions, reported as "
                                   f"{REFERENCE_DOMAIN}",
            "reserved_700": "not read",
            "participant_captures": "not read and not evaluated in this command",
        },
        "outputs": "final weights per arm and one JSON. No intermediate checkpoint, no overwrite.",
        "how_to_run": "add --run and --out <NEW directory>, after root has reviewed this file",
        "this_is_not": "a promotion, an improvement claim, a threshold change or a quality pass",
    }


# --------------------------------------------------------------------- the run
def _label_index_map(classes) -> dict:
    """AIHub word label -> its index in the candidate's own class order.

    Kept here rather than borrowing tools_prep_union93_candidate's private helper: it is one line,
    and this run must not depend on another tool's private name to decide what a label means.
    """
    return {name: i for i, name in enumerate(classes)}


def resolve_inputs(src_train: list, src_val: list, nikl: list, classes: list,
                   eligible: dict, annotations: dict, expected_counts: dict,
                   root: str = ROOT) -> dict:
    """Exactly what this run reads, by id, with the real hashes - resolved before any step.

    Written here rather than reusing functional_17's version because that one stamps functional_17's
    parent checkpoint and its own fixed setting. The ids and hashes are the same public facts; the
    provenance around them belongs to THIS run.
    """
    manifest = F17.load_json(os.path.join(F17.PILOT, "expand", "manifest.json"))
    by_id = {m["sample_id"]: m for m in manifest}
    clips = {c["clip_id"]: c for c in F17.load_json(
        os.path.join(root, "config", "source_manifest.json"))["clips"]}

    def source_rows(items, role):
        out = []
        for c in items:
            m = by_id.get(c["sample_id"], {})
            out.append({"id": c["sample_id"], "role": role, "label": m.get("word_label"),
                        "class_index": c["class_index"], "npz": m.get("npz"),
                        "npz_sha256": m.get("npz_sha256")})
        return out

    train_rows = source_rows(src_train, "training")
    val_rows = source_rows(src_val, "selection_and_regression_only")
    nikl_rows = []
    for it in nikl:
        c = clips.get(it["clip_id"], {})
        e = eligible.get(it["clip_id"])
        nikl_rows.append({"id": it["clip_id"], "role": "training_original_and_every_augmentation",
                          "word_id": it["word_id"], "label": it["label"],
                          "class_index": it["class_index"],
                          "npz": c.get("observation_npz"),
                          "npz_sha256": c.get("observation_sha256"),
                          "reviewed_non_sign_hand": e["side"] if e else None,
                          "masked_variant_stays_with_this_source": True})
    reserved = sorted(m["sample_id"] for m in manifest if m["split_role"] == "reserved_test")

    ids = {"training": {r["id"] for r in train_rows} | {r["id"] for r in nikl_rows},
           "selection_and_regression_only": {r["id"] for r in val_rows},
           "reserved_test": set(reserved)}
    for a, b in (("training", "selection_and_regression_only"),
                 ("training", "reserved_test"),
                 ("selection_and_regression_only", "reserved_test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise RuntimeError(f"{a} and {b} overlap on {sorted(overlap)[:5]}")

    counts = {"source_train": len(train_rows), "source_validation": len(val_rows),
              "nikl": len(nikl_rows), "reserved_test_not_read": len(reserved)}
    want = {"source_train": expected_counts["source_train"],
            "source_validation": expected_counts["source_validation"],
            "nikl": expected_counts["nikl"],
            "reserved_test_not_read": expected_counts["reserved_test"]}
    if counts != want:
        raise RuntimeError(f"resolved counts {counts} do not match the expected {want}")

    code = {rel: sha256_file(os.path.join(root, rel)) for rel in (
        "tools_visibility_experiment.py",
        "design/functional_22_visibility_experiment.md",
        "tests/test_r19_visibility_experiment.py",
        "src/wis_lab/core/passive_hand.py",
        "tools_prep_union93_candidate.py",
        "config/source_manifest.json") if os.path.exists(os.path.join(root, rel))}
    code["expand/manifest.json"] = sha256_file(os.path.join(F17.PILOT, "expand", "manifest.json"))
    code["union93_candidate_v1/inputs.json"] = sha256_file(
        os.path.join(root, "runs", "union93_candidate_v1", "inputs.json"))

    return {
        "schema": "wis_visibility_resolved_inputs_v1",
        "parent_checkpoint_sha256": CANDIDATE_SHA256,
        "parent_checkpoint": CANDIDATE_REL,
        "class_order_sha256": _sha256_obj(list(classes)),
        "counts": counts, "roles_are_disjoint": True,
        "annotations": {"path": annotations["path"], "sha256": annotations["sha256"],
                        "eligible": sorted(eligible)},
        "hashes": code,
        "source_train": train_rows, "source_validation": val_rows, "nikl": nikl_rows,
        "reserved_test": {"read": False, "ids": reserved,
                          "note": "listed by id only so the separation can be checked; no "
                                  "reserved NPZ is opened by this run"},
        "participant_captures": {"read": False,
                                 "note": "no participant session is opened by this run"},
    }


def checkpoint_payload(model, classes: list, step: int, arm: str, setting: dict,
                       annotations_sha256: str, code_hashes: dict) -> dict:
    """A payload that says what THIS run produced, not what an earlier one did.

    functional_17's saver embeds its own base checkpoint hash, its 660-step setting and its own
    augmentation statement. Reusing it here would have labelled these weights with another run's
    parent and another run's hyperparameters. The architecture, the class order and the weights
    are read from the pinned candidate and from this model; everything around them is this run's.
    """
    import torch

    from pilot.data import TOTAL_CHANNELS
    from pilot.model_masked_v1 import MODEL_TYPE, MODEL_VERSION
    parent = torch.load(os.path.join(ROOT, CANDIDATE_REL), map_location="cpu",
                        weights_only=False)
    return {
        "kind": parent.get("kind"), "class_name": parent.get("class_name"),
        "model_type": MODEL_TYPE, "model_version": MODEL_VERSION,
        "width": parent["width"], "dilations": list(parent["dilations"]),
        "kernel_size": parent["kernel_size"], "dropout": parent["dropout"],
        "in_channels": parent.get("in_channels", TOTAL_CHANNELS),
        "classes": list(classes), "n_classes": len(classes),
        "feature_schema": parent.get("feature_schema"),
        "step": int(step),
        "produced_by": TASK_ID,
        "arm": arm,
        "parent_checkpoint_sha256": CANDIDATE_SHA256,
        "parent_checkpoint": CANDIDATE_REL,
        "parent_step": parent.get("step"),
        "grandparent_base_checkpoint_sha256": parent.get("base_checkpoint_sha256"),
        "fixed_setting": setting,
        "augmentation": AUGMENT,
        "annotations_sha256": annotations_sha256,
        "code_hashes": code_hashes,
        "is_not": "a promoted model. Nothing in the web registry or the default recogniser points "
                  "at this file.",
        "state_dict": model.state_dict(),
    }



def run(out_dir: str, annotations_path: str = DEFAULT_ANNOTATIONS, root: str = ROOT,
        setting: dict | None = None, expected_counts: dict | None = None, log=print) -> dict:
    """Both arms, then one evaluation each. Refuses an existing output directory."""
    import torch
    setting = setting or SETTING
    expected_counts = expected_counts or EXPECTED_COUNTS
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists; the output directory must be new")
    torch.set_num_threads(int(setting["cpu_threads"]))
    started = time.time()

    ann = load_annotations(annotations_path)
    found = eligible_clips(ann["annotations"], verify_bytes=True, root=root)
    eligible = found["eligible"]
    if not eligible:
        raise RuntimeError("no reviewed annotation survived the full boundary; nothing to compare")
    if not all(e["verified_through_full_boundary"] for e in eligible.values()):
        raise RuntimeError("an eligible clip was admitted without the full boundary")

    proof = starting_point_proof(setting)
    if not (proof["identical_state_dict"] and proof["identical_logits_on_synthetic_input"]):
        raise RuntimeError("the two arms would not start from the same weights")

    _, classes = load_candidate()
    src_train = F17.load_source_split("train", _label_index_map(classes))
    src_val = F17.load_source_split("validation", _label_index_map(classes))
    nikl = F17.load_nikl(_word_index_map(root, classes))
    assert_roles_disjoint({c["sample_id"] for c in src_train},
                          {c["sample_id"] for c in src_val})
    for it in nikl:
        if not _shoulders_survive(it):
            raise RuntimeError(f"{it['clip_id']} has no frame with both shoulders; the tilt "
                               "centre would differ between arms")
    for clip_id in eligible:
        if clip_id not in {it["clip_id"] for it in nikl}:
            raise RuntimeError(f"{clip_id} is reviewed but not in the training list")

    resolved = resolve_inputs(src_train, src_val, nikl, classes, eligible, ann,
                              expected_counts, root=root)
    schedule = build_schedule(setting, src_train, nikl)
    counts = masked_draw_counts(schedule, eligible)
    adapter = adapter_agreement(nikl, eligible, root=root)
    if not adapter["agrees_everywhere"]:
        raise RuntimeError("the native masking adapter disagrees with the reviewed helper")

    os.makedirs(out_dir)
    written = {}
    for name, doc in (("inputs.json", resolved), ("schedule.json", schedule)):
        p = os.path.join(out_dir, name)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        written[name] = sha256_file(p)

    results, saved = {}, {}
    for arm in ARMS:
        stats = run_arm(arm, schedule, src_train, nikl, classes, eligible, setting, log=log)
        model = stats.pop("model")
        ev = evaluate_arm(model, src_val, nikl, classes, eligible)
        path = os.path.join(out_dir, f"{arm}.final.pt")
        torch.save(checkpoint_payload(model, classes, stats["steps"], arm, setting,
                                      ann["sha256"], resolved["hashes"]), path)
        saved[arm] = {"path": os.path.basename(path), "sha256": sha256_file(path)}
        results[arm] = {"training": stats, "evaluation": ev}
        log(f"[{arm}] done: validation {ev['source_validation_full_93']['correct']}/"
            f"{ev['source_validation_full_93']['n']}")

    if results[ARM_CONTROL]["training"]["batch_shape_digest"] != \
            results[ARM_MASKED]["training"]["batch_shape_digest"]:
        raise RuntimeError("the arms did not see the same tensor shapes")

    doc = _result_document(setting, ann, found, proof, schedule, counts, adapter, classes,
                           resolved, written, results, saved, started, root)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
    return doc


def _word_index_map(root: str, classes: list) -> dict:
    """clip word_id -> its index in the candidate's own class order. The order is never rebuilt."""
    src = F17.load_json(os.path.join(root, "config", "source_manifest.json"))
    return {c["word_id"]: classes.index(c["display_name"])
            for c in src["clips"] if c["display_name"] in classes}


def adapter_agreement(nikl: list, eligible: dict, root: str = ROOT) -> dict:
    """The native adapter must agree with the reviewed Observation helper, clip by clip."""
    from pilot.web_input import active_span
    from wis_lab.core.observation import Observation
    src = F17.load_json(os.path.join(root, "config", "source_manifest.json"))
    by_clip = {c["clip_id"]: c for c in src["clips"]}
    rows, all_ok = [], True
    for it in nikl:
        e = eligible.get(it["clip_id"])
        if not e:
            continue
        obs = Observation.load(os.path.join(root, by_clip[it["clip_id"]]["observation_npz"]))
        nat_r, pr_r = masked_like_reviewed(obs, e["side"])
        lo, hi = active_span(_present_of(obs))          # the INTACT span, exactly as loaded
        a_nat, a_pr = mask_native(it["nat"], it["present"], e["side"])
        same = (np.array_equal(a_pr, pr_r[lo:hi + 1])
                and np.array_equal(a_nat[a_pr], nat_r[lo:hi + 1][pr_r[lo:hi + 1]]))
        all_ok = all_ok and bool(same)
        rows.append({"clip_id": it["clip_id"], "side": e["side"], "agrees": bool(same),
                     "span": [int(lo), int(hi)]})
    return {"agrees_everywhere": all_ok, "per_clip": rows,
            "meaning": "the fast native adapter and the reviewed Observation helper mark the same "
                       "points absent, on the same span",
            "is_not": "evidence that the role review covers the span - that is checked by the "
                      "full boundary, not by array agreement"}


def _present_of(obs):
    from pilot.web_input import to_native_like
    _, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                np.asarray(obs.usable, dtype=bool))
    return present


def _result_document(setting, ann, found, proof, schedule, counts, adapter, classes,
                     resolved, written, results, saved, started, root) -> dict:
    c = results[ARM_CONTROL]["evaluation"]["training_reference_robustness"]
    m = results[ARM_MASKED]["evaluation"]["training_reference_robustness"]
    per_scope = {}
    for scope in ("full93", "project18"):
        cs, ms = c[scope], m[scope]
        per_scope[scope] = {
            "candidate_count": cs["candidate_count"],
            "reviewed_masked_condition": {
                ARM_CONTROL: f"{cs['masked_correct']}/{cs['masked_n']}",
                ARM_MASKED: f"{ms['masked_correct']}/{ms['masked_n']}"},
            "intact_public_references": {
                ARM_CONTROL: f"{cs['intact_correct']}/{cs['intact_n']}",
                ARM_MASKED: f"{ms['intact_correct']}/{ms['intact_n']}"},
            "incremental_effect": (
                "the augmentation arm identified more of the reviewed masked conditions than the "
                "matched control, in THIS scope" if ms["masked_correct"] > cs["masked_correct"]
                else "the incremental effect is NOT demonstrated in this scope: the augmentation "
                     "arm did not improve the targeted condition over the matched control"),
            "readiness_failure": sorted(set(cs["intact_identification_lost"]
                                            + ms["intact_identification_lost"])),
        }
    return {
        "task_id": TASK_ID,
        "schema": "wis_visibility_experiment_v1",
        "purpose": "compare extra training alone with extra training plus reviewed passive-hand "
                   "missingness",
        "this_is": "a targeted development and robustness experiment on training references",
        "this_is_not": ["a generalisation measurement", "an execution-quality evaluation",
                        "a promotion", "an automatic improvement claim"],
        "setting": setting, "augmentation": AUGMENT,
        "starting_point": {"checkpoint": CANDIDATE_REL, "sha256": CANDIDATE_SHA256, **proof},
        "annotations": {"path": ann["path"], "sha256": ann["sha256"],
                        "eligible": sorted(found["eligible"]), "refused": found["refused"],
                        "verified_through": "wis_lab.core.passive_hand.prepare_from_public_clip",
                        "per_clip": {k: {"side": v["side"],
                                         "reviewed_span": v.get("reviewed_span"),
                                         "opposite_hand_frames_in_span":
                                             v.get("opposite_hand_frames_in_span")}
                                     for k, v in found["eligible"].items()}},
        "code_hashes": resolved["hashes"],
        "inputs": {"file": "inputs.json", "sha256": written["inputs.json"],
                   "counts": resolved["counts"],
                   "roles_are_disjoint": True,
                   "class_order_sha256": resolved["class_order_sha256"],
                   "reserved_test_read": False, "participant_captures_read": False,
                   "note": "every training and validation id this run used is listed there, by "
                           "id and hash. Nothing is withheld."},
        "schedule": {"file": "schedule.json", "sha256": written["schedule.json"],
                     "digest": schedule["digest"], "steps": len(schedule["steps"]),
                     "shared_by_both_arms": True},
        "masked_draws": counts,
        "adapter_agreement": adapter,
        "class_order": {"n": len(classes), "sha256": _sha256_obj(list(classes)),
                        "unchanged_from_candidate": True},
        "arms": {arm: {"training": results[arm]["training"],
                       "evaluation": results[arm]["evaluation"],
                       "final_checkpoint": saved[arm]} for arm in ARMS},
        "comparison": {
            "source_validation_full93": {
                arm: (f"{results[arm]['evaluation']['source_validation_full_93']['correct']}/"
                      f"{results[arm]['evaluation']['source_validation_full_93']['n']}")
                for arm in ARMS},
            "continuity_gate_passed": {
                arm: results[arm]["evaluation"]["continuity_gate"]["passed"] for arm in ARMS},
            "training_reference_robustness": per_scope,
            "scope_rule": "each scope is compared only with itself. The earlier 3/5 masked "
                          "diagnostic was PROJECT18 and is not comparable with any full93 number.",
            "no_retuning": "nothing is adjusted after seeing these numbers",
        },
        "promotion": "none. No default model, threshold or pass line changed.",
        "elapsed_s": round(time.time() - started, 1),
        "finished_at_local": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }



def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="store_true",
                    help="actually train. Without it, only the plan is printed.")
    ap.add_argument("--out", help="a NEW output directory; required with --run")
    ap.add_argument("--annotations", default=DEFAULT_ANNOTATIONS)
    ap.add_argument("--no-verify-bytes", action="store_true",
                    help="plan only: skip hashing the five reviewed cache files")
    args = ap.parse_args(argv)

    if not args.run:
        doc = plan(args.annotations, verify_bytes=not args.no_verify_bytes)
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        print("\nplan only. Nothing was trained, loaded or written. Add --run --out <NEW dir> "
              "once root has reviewed this runner.")
        return 0
    if not args.out:
        print("--run requires --out <NEW directory>", file=sys.stderr)
        return 2
    doc = run(args.out, args.annotations)
    print(json.dumps(doc["comparison"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
