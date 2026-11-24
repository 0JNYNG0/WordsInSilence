"""functional_09: is the pinned checkpoint sensitive to losing the hip observation?

Contract: design/functional_09_hip_mask.md - written before this ran.

ONE new condition. The 280-channel input is built exactly as today, then six channels and only six
are set to zero: the 4 body-relative coordinates of left_hip and right_hip, and their 2 validity
channels. Every index is derived from pilot.data's own constants. Everything else is asserted
identical before a single forward pass is made.

Evaluated once over the fixed 45 train clips (five shared words x nine signers, first sample each)
and the five common NIKL clips. The ten web attempts are NOT re-inferred: their hips are already
absent, so the transform must be a no-op there, and that is checked by comparing arrays.

The 45 train clips are IN-SAMPLE. Numbers over them are a memorisation reading, never a
generalisation or accuracy figure.

Nothing is trained, tuned, searched or deployed. No camera, no server restart, no raw pose output.

Run: py -3.13 tools_diag_hip_mask.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

from wis_lab.core.observation import Observation  # noqa: E402
from wis_lab.service import neural  # noqa: E402

PILOT = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1")
PREP = os.path.join(PILOT, "expand")
SESSION = os.path.join(ROOT, "runs", "ses_5778156c6ae4")
SIGNERS = tuple(range(1, 10))            # the train split's nine signers


def hip_channels() -> dict:
    """Derived from pilot.data's constants, never typed as literals."""
    from pilot.data import CH_BODY_XY, CH_HAND_XY, COMMON_BODY_ORDER, TOTAL_CHANNELS

    hips = [COMMON_BODY_ORDER.index("left_hip"), COMMON_BODY_ORDER.index("right_hip")]
    xy = [c for p in hips for c in (2 * p, 2 * p + 1)]
    valid_base = CH_BODY_XY + CH_HAND_XY
    valid = [valid_base + p for p in hips]
    return {"hip_point_indices": hips, "body_rel_channels": xy, "body_valid_channels": valid,
            "all": xy + valid, "total_channels": TOTAL_CHANNELS,
            "hand_xy_channels": [CH_BODY_XY, CH_BODY_XY + CH_HAND_XY - 1],
            "dt_channel": TOTAL_CHANNELS - 1}


def zero_hips(x: np.ndarray, ch: dict) -> np.ndarray:
    out = x.copy()
    out[:, ch["all"]] = 0.0
    return out


def assert_only_hips_changed(a: np.ndarray, b: np.ndarray, ch: dict) -> dict:
    """Every channel outside the six must be bit-identical, and the six must be zero in b."""
    from pilot.data import CH_BODY_XY, CH_HAND_XY, TOTAL_CHANNELS

    others = np.array([c for c in range(TOTAL_CHANNELS) if c not in set(ch["all"])])
    assert np.array_equal(a[:, others], b[:, others]), "a channel outside the six moved"
    assert np.all(b[:, ch["all"]] == 0.0), "a hip channel was not zeroed"
    hand_xy = slice(CH_BODY_XY, CH_BODY_XY + CH_HAND_XY)
    hand_valid = slice(CH_BODY_XY + CH_HAND_XY + 51, TOTAL_CHANNELS - 1)
    body_valid_others = [c for c in range(CH_BODY_XY + CH_HAND_XY,
                                          CH_BODY_XY + CH_HAND_XY + 51)
                         if c not in set(ch["body_valid_channels"])]
    return {
        "hand_xy_identical": bool(np.array_equal(a[:, hand_xy], b[:, hand_xy])),
        "hand_valid_identical": bool(np.array_equal(a[:, hand_valid], b[:, hand_valid])),
        "other_body_valid_identical": bool(np.array_equal(a[:, body_valid_others],
                                                          b[:, body_valid_others])),
        "dt_identical": bool(np.array_equal(a[:, ch["dt_channel"]], b[:, ch["dt_channel"]])),
        "channels_changed": int((a != b).any(axis=0).sum()),
    }


def infer(x: np.ndarray, model, classes) -> dict:
    import torch
    from pilot.data import collate

    clip = {"x": x, "frames": int(len(x))}
    bx, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(bx), torch.from_numpy(pad))[0].numpy()
    idx = {classes[i]: i for i in range(len(classes))}
    restricted = sorted(((w, lab, float(logit[idx[lab]]))
                         for w, lab in neural.PROJECT_OVERLAP.items() if lab in idx),
                        key=lambda r: -r[2])
    return {"top1_all": classes[int(np.argmax(logit))],
            "top1_project": restricted[0][0] if restricted else None}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def train_selection(manifest, train_ids) -> list:
    """Five shared words x nine signers, first sample of each pair. Fixed, not searched."""
    want = {lab: wid for wid, lab in neural.PROJECT_OVERLAP.items()}
    picked = {}
    for m in sorted(manifest, key=lambda r: r["sample_id"]):
        lab = m["word_label"]
        if lab not in want or m.get("split_role") != "train" or m["sample_id"] not in train_ids:
            continue
        key = (want[lab], int(m["signer_id"]))
        if key[1] in SIGNERS and key not in picked:
            picked[key] = m
    return [picked[k] for k in sorted(picked)]


def main() -> int:
    started = time.time()
    c = neural._load()                                  # noqa: SLF001 - the same pinned checkpoint
    if c["status"] != "READY":
        print("model not ready:", c.get("detail"))
        return 1
    model, classes = c["model"], c["classes"]
    ch = hip_channels()

    from pilot.data import build_channels
    from pilot.web_input import active_span, to_native_like

    with open(os.path.join(PREP, "manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(os.path.join(PILOT, "runs", "masked_bn_pair_v1", "inputs.json"),
              encoding="utf-8") as fh:
        train_ids = {it["sample_id"] for it in json.load(fh)["train"]}

    rows, invariance = [], []
    for m in train_selection(manifest, train_ids):
        z = np.load(os.path.join(PREP, "npz", os.path.basename(m["npz"])))
        clip = build_channels(z["xy_native"], z["coordinate_present"], z["timestamp_nominal_us"])
        base = clip["x"]
        masked = zero_hips(base, ch)
        invariance.append(assert_only_hips_changed(base, masked, ch))
        word = {lab: wid for wid, lab in neural.PROJECT_OVERLAP.items()}[m["word_label"]]
        rows.append({"kind": "train_in_sample", "id": m["sample_id"], "word_id": word,
                     "signer_id": int(m["signer_id"]), "frames": int(clip["frames"]),
                     "npz_sha256": m["npz_sha256"],
                     "baseline": infer(base, model, classes),
                     "hips_zeroed": infer(masked, model, classes)})

    with open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8") as fh:
        clips = json.load(fh)["clips"]
    for clip_meta in clips:
        if clip_meta["word_id"] not in neural.PROJECT_OVERLAP:
            continue
        path = os.path.join(ROOT, clip_meta["observation_npz"])
        obs = Observation.load(path)
        nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                      np.asarray(obs.usable, dtype=bool))
        span = active_span(present)
        lo, hi = span
        clip = build_channels(nat[lo:hi + 1], present[lo:hi + 1],
                              np.asarray(obs.timestamp_us, dtype=np.int64)[lo:hi + 1])
        base = clip["x"]
        masked = zero_hips(base, ch)
        invariance.append(assert_only_hips_changed(base, masked, ch))
        rows.append({"kind": "nikl_reference", "id": clip_meta["clip_id"],
                     "word_id": clip_meta["word_id"], "signer_id": None,
                     "frames": int(clip["frames"]), "npz_sha256": sha256_file(path),
                     "baseline": infer(base, model, classes),
                     "hips_zeroed": infer(masked, model, classes)})

    # the ten web attempts already have no hips: the transform must be a no-op, so nothing is run
    web = []
    tdir = os.path.join(SESSION, "trials")
    for f in sorted(os.listdir(tdir)):
        if not f.endswith(".npz"):
            continue
        obs = Observation.load(os.path.join(tdir, f))
        nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                      np.asarray(obs.usable, dtype=bool))
        span = active_span(present)
        if span is None:
            continue
        lo, hi = span
        clip = build_channels(nat[lo:hi + 1], present[lo:hi + 1],
                              np.asarray(obs.timestamp_us, dtype=np.int64)[lo:hi + 1])
        base = clip["x"]
        web.append({"trial_id": f[:-4],
                    "hip_channels_already_zero": bool(np.all(base[:, ch["all"]] == 0.0)),
                    "transform_is_identity": bool(np.array_equal(base, zero_hips(base, ch)))})

    def tally(kind, cond):
        sub = [r for r in rows if r["kind"] == kind]
        return {"n": len(sub),
                "top1_project_equals_word": sum(1 for r in sub
                                                if r[cond]["top1_project"] == r["word_id"]),
                "top1_all_equals_word_label": sum(
                    1 for r in sub
                    if r[cond]["top1_all"] == neural.PROJECT_OVERLAP[r["word_id"]])}

    changed = [{"kind": r["kind"], "id": r["id"], "word_id": r["word_id"],
                "signer_id": r["signer_id"],
                "project_from": r["baseline"]["top1_project"],
                "project_to": r["hips_zeroed"]["top1_project"],
                "all_changed": r["baseline"]["top1_all"] != r["hips_zeroed"]["top1_all"]}
               for r in rows
               if r["baseline"]["top1_project"] != r["hips_zeroed"]["top1_project"]
               or r["baseline"]["top1_all"] != r["hips_zeroed"]["top1_all"]]

    out = {
        "task_id": "functional_09_hip_mask",
        "design": "design/functional_09_hip_mask.md",
        "checkpoint_sha256": neural.CHECKPOINT_SHA256,
        "condition_count": 1,
        "hip_channels": ch,
        "channel_invariance": {
            "checked_inputs": len(invariance),
            "all_hand_xy_identical": all(i["hand_xy_identical"] for i in invariance),
            "all_hand_valid_identical": all(i["hand_valid_identical"] for i in invariance),
            "all_other_body_valid_identical": all(i["other_body_valid_identical"]
                                                  for i in invariance),
            "all_dt_identical": all(i["dt_identical"] for i in invariance),
            "max_channels_changed": max(i["channels_changed"] for i in invariance),
        },
        "in_sample_warning": "the 45 train clips were fitted on by this checkpoint. Their numbers "
                             "are a memorisation reading, NEVER a generalisation or accuracy "
                             "figure.",
        "tallies": {k: {cond: tally(k, cond) for cond in ("baseline", "hips_zeroed")}
                    for k in ("train_in_sample", "nikl_reference")},
        "label_changes": changed,
        "web_attempts_not_re_inferred": {
            "n": len(web),
            "all_hip_channels_already_zero": all(w["hip_channels_already_zero"] for w in web),
            "transform_is_identity_on_all": all(w["transform_is_identity"] for w in web),
            "meaning": "the ten attempts already carry no hip observation, so this condition "
                       "cannot change them. No inference was repeated on them.",
            "per_trial": web,
        },
        "rows": rows,
        "elapsed_s": round(time.time() - started, 1),
        "no_training": True, "no_threshold_change": True, "no_grid_search": True,
        "no_other_joint_condition": True, "no_deployment": True,
        "camera_opened": False, "server_touched": False, "raw_pose_emitted": False,
    }
    path = os.path.join(ROOT, "evidence", "functional_09_hip_mask.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"hip_channels": ch["all"],
                      "channel_invariance": out["channel_invariance"],
                      "tallies": out["tallies"],
                      "label_changes": len(changed),
                      "web_transform_identity":
                          out["web_attempts_not_re_inferred"]["transform_is_identity_on_all"],
                      "written": os.path.relpath(path, ROOT)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
