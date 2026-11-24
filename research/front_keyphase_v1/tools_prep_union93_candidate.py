"""functional_17: prepare ONE union-93 fine-tuning candidate. The main training is NOT run here.

Contract fixed first: design/functional_17_union93_candidate.md. This is a FIRST LIMITED
experiment and the settings are not claimed to be optimal.

What this module provides, all as real functions a reviewer can call:
  * the frozen data lists - AIHub train 2095 and the 18 NIKL demonstrations for training, the 696
    source validation clips for selection and regression only, the reserved 700 untouched;
  * the union-93 class order: the existing 80 in their existing order, then the 13 NIKL words the
    model does not know. KKEUDEOGIDA is held back explicitly - its cue is unsupported by the
    observation contract, so it is absent by rule, not by accident;
  * build_candidate(): encoder weights and head rows 0..79 copied from the pinned checkpoint, 13
    fresh rows appended, so the first 80 logits start identical;
  * the two augmentations - one whole-clip tilt of +/-5 degrees, and a 0.8-1.2x time resample that
    copies exact original frames, interpolates only between frames that BOTH observed a point,
    restores nothing and deletes no exact frame;
  * train_candidate(): the single fixed setting, gated behind an explicit --train flag.

Never here: a threshold search, a per-word rule, a model or web change, the camera, the server,
Unity, a backup, or any use of the private recordings, the user's ten attempts, the three-view
material, Validation 17-18 or the reserved 700.

Run (prepare and check only):  py -3.13 tools_prep_union93_candidate.py
Run (root, after review):      py -3.13 tools_prep_union93_candidate.py --train
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(PILOT, "src"))

from wis_lab.service import neural  # noqa: E402

CHECKPOINT_SHA256 = "56a17a4d0a4795a2463add5572acc0d985c3cb7a1cbaee2853698f106d5b34b5"
CACHE_OUT = os.path.join(ROOT, "runs", "cache_functional_17")
HELD_BACK = {"KKEUDEOGIDA": "UNSUPPORTED_CUE - the observation contract does not carry this "
                            "sign's distinguishing cue, so it is excluded by rule, not by "
                            "absence of material"}

# the single fixed setting. Not a search, and not claimed to be optimal.
SETTING = {
    "seed": 20260913,
    "optimizer_steps": 660,
    "batch": {"total": 32, "source": 16, "nikl": 16},
    "source_sampling": "class balanced",
    "nikl_sampling": "uniform over the 18",
    "lr": {"encoder": 1e-4, "head": 1e-3},
    "optimizer": "AdamW", "weight_decay": 1e-3,
    "cpu_threads": 4,
    "report_step": "the FINAL step only; no step is chosen after seeing results",
}
AUGMENT = {"tilt_degrees": 5.0, "speed_range": [0.8, 1.2],
           "forbidden": ["occlusion", "noise", "mirror", "handshape change", "time reversal",
                         "arbitrary cropping", "per-word augmentation"]}
EXCLUDED_FROM_TRAINING = ["private recordings", "the user's ten attempts", "three-view material",
                          "Validation 17-18", "the reserved 700 test clips",
                          "AIHub source validation (selection and regression only)"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ data lists
def data_lists() -> dict:
    """Every list this candidate may touch, and what each one is for."""
    manifest = load_json(os.path.join(PILOT, "expand", "manifest.json"))
    train_ids = {it["sample_id"] for it in load_json(
        os.path.join(PILOT, "runs", "masked_bn_pair_v1", "inputs.json"))["train"]}
    src_manifest = load_json(os.path.join(ROOT, "config", "source_manifest.json"))

    source_train = sorted(m["sample_id"] for m in manifest
                          if m["split_role"] == "train" and m["sample_id"] in train_ids)
    source_validation = sorted(m["sample_id"] for m in manifest
                               if m["split_role"] == "validation")
    reserved = sorted(m["sample_id"] for m in manifest if m["split_role"] == "reserved_test")
    ready = {"READY", "NEEDS_REVIEW"}
    nikl = sorted((c["clip_id"], c["word_id"], c["display_name"])
                  for c in src_manifest["clips"] if c["readiness"] in ready)
    held = sorted((c["clip_id"], c["word_id"], c["readiness"])
                  for c in src_manifest["clips"] if c["readiness"] not in ready)
    return {
        "source_train": source_train,
        "source_validation": source_validation,
        "reserved_test": reserved,
        "nikl_training": nikl,
        "nikl_held_back": held,
        "roles": {
            "source_train": "training",
            "nikl_training": "training, originals AND every augmentation",
            "source_validation": "selection and regression check only; never trained on",
            "reserved_test": "untouched by this candidate",
        },
        "excluded_from_training": EXCLUDED_FROM_TRAINING,
    }


def union_classes() -> tuple:
    """The existing 80 in their existing order, then the 13 the model does not know."""
    import torch

    got = sha256_file(neural.CHECKPOINT)
    if got != CHECKPOINT_SHA256:
        raise SystemExit(f"checkpoint hash mismatch: {got}")
    ck = torch.load(neural.CHECKPOINT, map_location="cpu", weights_only=True)
    base = list(ck["classes"])
    lists = data_lists()
    new = sorted({(w, name) for _, w, name in lists["nikl_training"] if name not in base})
    classes = base + [name for _, name in new]
    mapping = {w: base.index(name) if name in base else len(base) + i
               for i, (w, name) in enumerate(new)}
    for _, w, name in lists["nikl_training"]:
        if name in base:
            mapping[w] = base.index(name)
    return classes, mapping, [w for w, _ in new], ck


def build_candidate():
    """Encoder and the first 80 head rows copied; 13 fresh rows appended."""
    import torch
    from pilot.model_masked_v1 import MaskedSignTCN

    classes, mapping, new_words, ck = union_classes()
    cfg = ck.get("config", {})

    def need(k):
        return ck[k] if k in ck else cfg[k]

    model = MaskedSignTCN(width=need("width"), dilations=tuple(need("dilations")),
                          kernel_size=need("kernel_size"), dropout=need("dropout"),
                          n_classes=len(classes))
    state = dict(ck["state_dict"])
    old_w = state.pop("head.weight")
    old_b = state.pop("head.bias")
    missing = model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.head.weight[:old_w.shape[0]] = old_w        # the existing rows, byte for byte
        model.head.bias[:old_b.shape[0]] = old_b
    model.eval()
    return model, classes, mapping, new_words, {
        "copied_head_rows": int(old_w.shape[0]),
        "new_head_rows": len(classes) - int(old_w.shape[0]),
        "unexpected_keys": list(missing.unexpected_keys),
        "missing_keys": [k for k in missing.missing_keys if not k.startswith("head.")],
    }


# ------------------------------------------------------------------ augmentation
def tilt_clip(nat: np.ndarray, present: np.ndarray, degrees: float):
    """ONE angle for the whole clip, as a camera tilt is. Masks are untouched.

    The rotation centre is immaterial: build_channels subtracts the body origin and divides by the
    shoulder width afterwards, so any fixed centre yields the same channels. The shoulder midpoint
    is used because it exists in every frame this project accepts.
    """
    a = np.radians(float(degrees))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)
    out = nat.astype(np.float64).copy()
    ok = present[:, 5] & present[:, 2]                 # BODY_25 left/right shoulder slots
    centre = (out[ok][:, [5, 2]].mean(axis=(0, 1)) if ok.any()
              else out[present].mean(axis=0) if present.any() else np.zeros(2))
    moved = (out - centre) @ rot.T + centre
    out[present] = moved[present]                      # absent points keep their zero, untouched
    return out.astype(np.float32), present.copy()


def time_resample(nat: np.ndarray, present: np.ndarray, ts: np.ndarray, speed: float,
                  tol_us: int = 1):
    """0.8-1.2x, order preserving, coordinates and mask and time together.

    Both original endpoints are kept: the target grid spans t0..t1 exactly, so the last original
    instant cannot be lost to a floor(). The emitted timestamps span the scaled duration, so the
    clip really is faster or slower. A target that lands on an original timestamp copies that
    frame VERBATIM, so an exact frame is never dropped because a neighbour is missing; only a
    strictly-between time is interpolated, per point, and only where BOTH bracketing frames
    observed it - an interior gap is never filled in.

    A non-monotonic clock or an unusable speed is a FAILURE, not a silent copy.
    """
    ts = np.asarray(ts, dtype=np.int64)
    n = len(ts)
    if n < 2:
        raise ValueError("time_resample needs at least two frames")
    if not np.all(np.diff(ts) > 0):
        raise ValueError("time_resample refuses a non-monotonic clock")
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError(f"time_resample refuses speed {speed!r}")
    t0, t1 = int(ts[0]), int(ts[-1])
    if t1 <= t0:
        raise ValueError("time_resample refuses a clip with no duration")
    step = float(np.median(np.diff(ts)))
    span_in = float(t1 - t0)
    span_out = span_in / float(speed)
    n_out = max(2, int(round(span_out / step)) + 1)
    if span_out < n_out - 1:
        raise ValueError("the scaled duration is too short to hold a monotonic clock")

    xy = np.zeros((n_out, nat.shape[1], 2), dtype=np.float32)
    pr = np.zeros((n_out, nat.shape[1]), dtype=bool)
    out_ts = np.zeros(n_out, dtype=np.int64)
    for k in range(n_out):
        frac = k / (n_out - 1)                       # 0 .. 1 inclusive: both ends are kept
        src = t0 + span_in * frac                    # where in the ORIGINAL this sample sits
        out_ts[k] = int(round(t0 + span_out * frac))
        near = int(np.searchsorted(ts, int(round(src)), side="left"))
        exact = next((i for i in (near - 1, near, near + 1)
                      if 0 <= i < n and abs(int(ts[i]) - src) <= tol_us), None)
        if exact is not None:
            xy[k] = nat[exact]
            pr[k] = present[exact]
            continue
        hi = int(np.searchsorted(ts, src, side="right"))
        lo = hi - 1
        if lo < 0 or hi >= n:
            continue
        w = (src - float(ts[lo])) / float(ts[hi] - ts[lo])
        both = present[lo] & present[hi]
        pr[k] = both
        xy[k][both] = ((1.0 - w) * nat[lo][both] + w * nat[hi][both]).astype(np.float32)
    if not np.all(np.diff(out_ts) > 0):
        raise ValueError("the resampled clock is not strictly increasing")
    return xy, pr, out_ts


# ------------------------------------------------------------------ loaders (hash verified)
def _channels_from_source(npz_path: str, expect_sha: str) -> dict:
    """An AIHub clip, verified against the manifest hash, through the SAME feature builder."""
    from pilot.data import build_channels

    got = sha256_file(npz_path)
    if got != expect_sha:
        raise ValueError(f"npz hash mismatch for {os.path.basename(npz_path)}: {got}")
    z = np.load(npz_path)
    return build_channels(z["xy_native"], z["coordinate_present"], z["timestamp_nominal_us"])


def _native_from_nikl(npz_path: str, expect_sha: str):
    """A NIKL cache, verified by hash, through to_native_like + active_span - the same rule."""
    from pilot.web_input import active_span, to_native_like

    from wis_lab.core.observation import Observation
    got = sha256_file(npz_path)
    if got != expect_sha:
        raise ValueError(f"observation hash mismatch for {os.path.basename(npz_path)}: {got}")
    obs = Observation.load(npz_path)
    nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                  np.asarray(obs.usable, dtype=bool))
    span = active_span(present)
    if span is None:
        raise ValueError(f"no usable hand frame in {os.path.basename(npz_path)}")
    lo, hi = span
    ts = np.asarray(obs.timestamp_us, dtype=np.int64)[lo:hi + 1]
    return nat[lo:hi + 1], present[lo:hi + 1], ts


def load_source_split(split: str, mapping_by_label: dict) -> list:
    """Every clip of one AIHub split, hash-checked. The reserved split is never requested."""
    if split not in ("train", "validation"):
        raise ValueError(f"refusing to load split {split!r}")
    manifest = load_json(os.path.join(PILOT, "expand", "manifest.json"))
    train_ids = {it["sample_id"] for it in load_json(
        os.path.join(PILOT, "runs", "masked_bn_pair_v1", "inputs.json"))["train"]}
    out = []
    for m in sorted(manifest, key=lambda r: r["sample_id"]):
        if m["split_role"] != split:
            continue
        if split == "train" and m["sample_id"] not in train_ids:
            continue
        label = m["word_label"]
        if label not in mapping_by_label:
            raise ValueError(f"{m['sample_id']} carries a label outside the union: {label}")
        c = _channels_from_source(os.path.join(PILOT, "expand", os.path.basename(
            os.path.dirname(m["npz"])), os.path.basename(m["npz"])), m["npz_sha256"])
        out.append({"sample_id": m["sample_id"], "label": label,
                    "class_index": mapping_by_label[label], "x": c["x"],
                    "frames": int(c["frames"]), "source": "aihub_" + split})
    return out


def load_nikl(mapping: dict) -> list:
    """The 18 prepared demonstrations, hash-checked, kept as NATIVE arrays so they can be
    augmented before build_channels sees them."""
    src = load_json(os.path.join(ROOT, "config", "source_manifest.json"))
    ready = {"READY", "NEEDS_REVIEW"}
    out = []
    for c in sorted(src["clips"], key=lambda r: r["clip_id"]):
        if c["readiness"] not in ready:
            continue
        nat, present, ts = _native_from_nikl(os.path.join(ROOT, c["observation_npz"]),
                                             c["observation_sha256"])
        out.append({"clip_id": c["clip_id"], "word_id": c["word_id"], "label": c["display_name"],
                    "class_index": mapping[c["word_id"]], "nat": nat, "present": present,
                    "ts": ts, "source": "nikl_demonstration"})
    return out


def augment_nikl(item: dict, rng) -> np.ndarray:
    """One whole-clip tilt and one speed, then the SAME build_channels everything else uses."""
    from pilot.data import build_channels

    deg = float(rng.uniform(-AUGMENT["tilt_degrees"], AUGMENT["tilt_degrees"]))
    speed = float(rng.uniform(*AUGMENT["speed_range"]))
    nat, present = tilt_clip(item["nat"], item["present"], deg)
    nat, present, ts = time_resample(nat, present, item["ts"], speed)
    return build_channels(nat, present, ts)["x"], {"tilt_deg": round(deg, 3),
                                                   "speed": round(speed, 3)}


def set_seeds(seed: int) -> None:
    """Fixed BEFORE the model is built, so the new head rows are reproducible too."""
    import random

    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------ evaluation
def evaluate(model, clips, classes, restrict_to=None) -> dict:
    """Blind forward over a clip list. restrict_to limits the logits to a class subset."""
    import torch
    from pilot.data import collate

    idx = list(range(len(classes))) if restrict_to is None else \
        [classes.index(c) for c in restrict_to]
    rows, correct = [], 0
    with torch.inference_mode():
        for c in clips:
            x, pad, _ = collate([{"x": c["x"], "frames": int(len(c["x"])), "label": 0}])
            logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
            sub = logit[idx]
            pred = classes[idx[int(np.argmax(sub))]]
            correct += int(pred == c["label"])
            rows.append({"id": c.get("sample_id") or c.get("clip_id"),
                         "true": c["label"], "pred": pred})
    labels = sorted({r["true"] for r in rows})
    f1s = []
    for lab in labels:
        tp = sum(1 for r in rows if r["true"] == lab and r["pred"] == lab)
        fp = sum(1 for r in rows if r["true"] != lab and r["pred"] == lab)
        fn = sum(1 for r in rows if r["true"] == lab and r["pred"] != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {
        "n": len(rows), "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": float(np.mean(f1s)) if f1s else None,
        "macro_f1_denominator": len(labels),
        "macro_f1_scope": "classes with ground truth in this set ONLY; the denominator is never "
                          "inflated to the full candidate count",
        "candidate_count": len(idx),
        "ground_truth_counts": {lab: sum(1 for r in rows if r["true"] == lab) for lab in labels},
        "prediction_counts": {p2: sum(1 for r in rows if r["pred"] == p2)
                              for p2 in sorted({r["pred"] for r in rows})},
        "rows": rows,
    }


def run_training(model, source_train, nikl, classes, setting, rng, log=print,
                 steps=None) -> dict:
    """The loop itself, with everything it needs passed in so a check can drive it small.

    The head and the encoder carry different learning rates, no intermediate checkpoint is kept,
    and a non-finite loss stops the run at once rather than being averaged away.
    """
    import torch
    from pilot.data import collate

    steps = int(steps if steps is not None else setting["optimizer_steps"])
    n_src = setting["batch"]["source"]
    n_nikl = setting["batch"]["nikl"]
    head_params = [p for n2, p in model.named_parameters() if n2.startswith("head.")]
    enc_params = [p for n2, p in model.named_parameters() if not n2.startswith("head.")]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": setting["lr"]["encoder"]},
         {"params": head_params, "lr": setting["lr"]["head"]}],
        weight_decay=setting["weight_decay"])
    by_class = {}
    for c in source_train:
        by_class.setdefault(c["class_index"], []).append(c)
    class_keys = sorted(by_class)
    losses, used_source, used_nikl, aug_log = [], set(), set(), []
    model.train()
    for step in range(steps):
        batch = []
        for _ in range(n_src):                       # class balanced: pick a class, then a clip
            k = class_keys[int(rng.integers(len(class_keys)))]
            c = by_class[k][int(rng.integers(len(by_class[k])))]
            batch.append({"x": c["x"], "frames": int(len(c["x"])), "label": c["class_index"]})
            used_source.add(c["sample_id"])
        for _ in range(n_nikl):                      # uniform over the 18, always augmented
            it = nikl[int(rng.integers(len(nikl)))]
            x, how = augment_nikl(it, rng)
            batch.append({"x": x, "frames": int(len(x)), "label": it["class_index"]})
            used_nikl.add(it["clip_id"])
            if step < 3:
                aug_log.append(dict(how, clip_id=it["clip_id"], step=step))
        x, pad, y = collate(batch)
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))
        loss = torch.nn.functional.cross_entropy(logit, torch.from_numpy(y))
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()!r}")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 60 == 0 or step == steps - 1:
            log(f"step {step + 1}/{steps} loss {losses[-1]:.4f}")
    model.eval()
    return {"steps": steps, "loss_first": losses[0], "loss_last": losses[-1],
            "loss_mean_last_20": float(np.mean(losses[-20:])),
            "losses": [round(v, 6) for v in losses],
            "distinct_source_clips_seen": len(used_source),
            "distinct_nikl_clips_seen": len(used_nikl),
            "augmentation_examples": aug_log,
            "intermediate_checkpoints_kept": 0,
            "learning_rates": {"encoder": setting["lr"]["encoder"],
                               "head": setting["lr"]["head"]}}


def base_source_reference() -> dict:
    """The pinned run's own validation figure, verified before it is used as the base."""
    res = load_json(os.path.join(PILOT, "runs", "masked_bn_pair_v1", "result.json"))
    block = res["per_model"]["MaskedSignTCN"]
    ck = block["checkpoints"]["best"]
    if ck["sha256"] != CHECKPOINT_SHA256:
        raise ValueError("the base run's best checkpoint is not the pinned one")
    best = block["validation"]["best"]
    samples = sorted(r["sample_id"] for r in best["per_sample"])
    return {"checkpoint_sha256": ck["sha256"], "step": ck["step"],
            "correct": best["correct"], "total": best["total"],
            "accuracy": best["correct"] / best["total"],
            "macro_f1": best["macro_f1_unrounded"],
            "sample_ids": samples}


def checkpoint_payload(model, classes, step: int) -> dict:
    """Everything a SEPARATE process needs to rebuild this model without guessing.

    The earlier save wrote width=None and no dilations, kernel size or dropout, so nothing else
    could reconstruct it. Architecture, class order, feature schema, step, the base it started
    from and the fixed settings all travel with the weights now.
    """
    import torch

    from pilot.data import TOTAL_CHANNELS
    from pilot.model_masked_v1 import MODEL_TYPE, MODEL_VERSION
    ck = torch.load(neural.CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = ck.get("config", {})

    def need(k):
        return ck[k] if k in ck else cfg[k]

    return {
        "kind": ck.get("kind"), "class_name": ck.get("class_name"),
        "model_type": MODEL_TYPE, "model_version": MODEL_VERSION,
        "width": need("width"), "dilations": list(need("dilations")),
        "kernel_size": need("kernel_size"), "dropout": need("dropout"),
        "in_channels": ck.get("in_channels", TOTAL_CHANNELS),
        "classes": list(classes), "n_classes": len(classes),
        "feature_schema": {"builder": "pilot.data.build_channels",
                           "total_channels": TOTAL_CHANNELS,
                           "layout": "body_rel 102 + hand_rel 84 + body_valid 51 + "
                                     "hand_valid 42 + dt 1",
                           "normalisation": "unchanged from the pinned model"},
        "step": int(step),
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "base_step": ck.get("step"),
        "fixed_setting": SETTING, "augmentation": AUGMENT,
        "state_dict": model.state_dict(),
    }


def rebuild_from_payload(payload):
    """Rebuild a model from a saved payload alone. Used to prove the save is sufficient."""
    from pilot.model_masked_v1 import MaskedSignTCN

    model = MaskedSignTCN(width=payload["width"], dilations=tuple(payload["dilations"]),
                          kernel_size=payload["kernel_size"], dropout=payload["dropout"],
                          n_classes=payload["n_classes"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, list(payload["classes"])


def resolve_inputs(src_train, src_val, nikl, classes) -> dict:
    """Exactly what this run will read, with the real hashes, resolved BEFORE any step.

    The counts and the role separation are asserted here rather than assumed. The reserved split
    is named only to say it is not read; no reserved NPZ is opened.
    """
    manifest = load_json(os.path.join(PILOT, "expand", "manifest.json"))
    by_id = {m["sample_id"]: m for m in manifest}
    src_manifest = load_json(os.path.join(ROOT, "config", "source_manifest.json"))
    by_clip = {c["clip_id"]: c for c in src_manifest["clips"]}

    def source_rows(clips, role):
        out = []
        for c in clips:
            m = by_id[c["sample_id"]]
            out.append({"id": c["sample_id"], "role": role, "label": m["word_label"],
                        "class_index": c["class_index"], "npz": m["npz"],
                        "npz_sha256": m["npz_sha256"]})
        return out

    train_rows = source_rows(src_train, "training")
    val_rows = source_rows(src_val, "selection_and_regression_only")
    nikl_rows = [{"id": it["clip_id"], "role": "training_original_and_every_augmentation",
                  "word_id": it["word_id"], "label": it["label"],
                  "class_index": it["class_index"],
                  "npz": by_clip[it["clip_id"]]["observation_npz"],
                  "npz_sha256": by_clip[it["clip_id"]]["observation_sha256"]}
                 for it in nikl]

    ids = {"training": {r["id"] for r in train_rows} | {r["id"] for r in nikl_rows},
           "selection_and_regression_only": {r["id"] for r in val_rows},
           "reserved_test": {m["sample_id"] for m in manifest
                             if m["split_role"] == "reserved_test"}}
    if len(train_rows) != 2095:
        raise SystemExit(f"expected 2095 source training clips, resolved {len(train_rows)}")
    if len(val_rows) != 696:
        raise SystemExit(f"expected 696 validation clips, resolved {len(val_rows)}")
    if len(nikl_rows) != 18:
        raise SystemExit(f"expected 18 demonstrations, resolved {len(nikl_rows)}")
    if len(ids["reserved_test"]) != 700:
        raise SystemExit("the reserved split is not the expected 700")
    for a, b in (("training", "selection_and_regression_only"),
                 ("training", "reserved_test"),
                 ("selection_and_regression_only", "reserved_test")):
        overlap = ids[a] & ids[b]
        if overlap:
            raise SystemExit(f"{a} and {b} overlap on {sorted(overlap)[:5]}")

    code = {}
    for rel in ("tools_prep_union93_candidate.py",
                "design/functional_17_union93_candidate.md",
                "tests/test_r14_union93_prep.py",
                "config/source_manifest.json"):
        code[rel] = sha256_file(os.path.join(ROOT, rel))
    code["expand/manifest.json"] = sha256_file(os.path.join(PILOT, "expand", "manifest.json"))
    code["masked_bn_pair_v1/inputs.json"] = sha256_file(
        os.path.join(PILOT, "runs", "masked_bn_pair_v1", "inputs.json"))

    return {
        "schema": "wis_union93_resolved_inputs_v1",
        "counts": {"source_train": len(train_rows), "source_validation": len(val_rows),
                   "nikl": len(nikl_rows), "reserved_test_not_read": len(ids["reserved_test"])},
        "roles_are_disjoint": True,
        "reserved_test": {"read": False,
                          "note": "listed by id only so the separation can be checked; no "
                                  "reserved NPZ is opened by this run"},
        "class_order_sha256": hashlib.sha256("|".join(classes).encode("utf-8")).hexdigest(),
        "base_checkpoint_sha256": CHECKPOINT_SHA256,
        "hashes": code,
        "source_train": train_rows, "source_validation": val_rows, "nikl": nikl_rows,
        "reserved_test_ids": sorted(ids["reserved_test"]),
    }


def train_candidate(run_dir: str | None = None) -> int:
    """The single fixed setting, end to end. Writes only into a NEW run directory."""
    import time

    import torch

    started = time.time()
    torch.set_num_threads(int(SETTING["cpu_threads"]))
    run_dir = run_dir or os.path.join(ROOT, "runs", "union93_candidate_v1")
    if os.path.exists(run_dir):
        raise SystemExit(f"{run_dir} exists; refusing to overwrite a previous run")
    os.makedirs(run_dir)

    set_seeds(SETTING["seed"])                      # BEFORE the model is created
    model, classes, mapping, new_words, init = build_candidate()
    initial = os.path.join(run_dir, "union93_initial.pt")
    torch.save(checkpoint_payload(model, classes, 0), initial)

    base = base_source_reference()
    src_train = load_source_split("train", _source_label_map(classes))
    src_val = load_source_split("validation", _source_label_map(classes))
    if sorted(c["sample_id"] for c in src_val) != base["sample_ids"]:
        raise SystemExit("the validation sample set does not match the base run's")
    nikl = load_nikl(mapping)

    # resolved BEFORE the first step: exact ids, roles, real hashes, counts and disjointness
    resolved = resolve_inputs(src_train, src_val, nikl, classes)
    with open(os.path.join(run_dir, "inputs.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(resolved, fh, ensure_ascii=False, indent=1)

    rng = np.random.default_rng(SETTING["seed"])
    log_path = os.path.join(run_dir, "train.log")
    with open(log_path, "a", encoding="utf-8", newline="\n") as fh:
        def log(msg):
            fh.write(msg + "\n")
            fh.flush()
            print(msg)
        stats = run_training(model, src_train, nikl, classes, SETTING, rng, log=log)

    final = os.path.join(run_dir, "union93_final.pt")
    payload = checkpoint_payload(model, classes, stats["steps"])
    torch.save(payload, final)
    # a separate process must be able to rebuild this: prove it here, on a tiny synthetic input
    reloaded = torch.load(final, map_location="cpu", weights_only=False)
    model2, classes2 = rebuild_from_payload(reloaded)
    if classes2 != list(classes):
        raise SystemExit("the reloaded class order differs")
    probe = np.zeros((1, 8, payload["in_channels"]), dtype=np.float32)
    probe[0, :, :4] = 0.1
    pad = np.ones((1, 8), dtype=np.float32)
    with torch.inference_mode():
        a = model(torch.from_numpy(probe), torch.from_numpy(pad))[0].numpy()
        b = model2(torch.from_numpy(probe), torch.from_numpy(pad))[0].numpy()
    if not np.array_equal(a, b):
        raise SystemExit(f"save/load changed the logits by {float(np.abs(a - b).max())}")

    base80 = classes[:80]
    val_restricted = evaluate(model, src_val, classes, restrict_to=base80)
    val_full = evaluate(model, src_val, classes)
    nikl_clips = []
    for it in nikl:
        from pilot.data import build_channels
        nikl_clips.append({"clip_id": it["clip_id"], "label": it["label"],
                           "x": build_channels(it["nat"], it["present"], it["ts"])["x"]})
    nikl_eval = evaluate(model, nikl_clips, classes)

    drop_f1 = base["macro_f1"] - val_restricted["macro_f1"]
    drop_acc = base["accuracy"] - val_full["accuracy"]
    verdict = "HOLD" if (drop_f1 > 0.02 or drop_acc > 0.02) else "WITHIN_THE_PRE_REGISTERED_BAND"

    report = {
        "task_id": "functional_17_union93_training",
        "design": "design/functional_17_union93_candidate.md",
        "run_dir": os.path.relpath(run_dir, ROOT),
        "seed_fixed_before_model_creation": True,
        "setting": SETTING, "augmentation": AUGMENT,
        "classes": {"union": len(classes), "base": 80, "new": len(new_words),
                    "new_word_ids": new_words},
        "model_hashes": {"base_checkpoint": CHECKPOINT_SHA256,
                         "initial": sha256_file(initial), "final": sha256_file(final)},
        "data": {"source_train": len(src_train), "source_validation": len(src_val),
                 "nikl": len(nikl), "reserved_test": "not loaded",
                 "validation_matches_base_sample_set": True,
                 "resolved_inputs": "inputs.json in this run directory",
                 "resolved_counts": resolved["counts"],
                 "roles_disjoint_asserted_before_training": True},
        "checkpoint_reload_identical_logits": True,
        "training": stats,
        "elapsed_s": round(time.time() - started, 1),
        "base_source_reference": {k: v for k, v in base.items() if k != "sample_ids"},
        "source_validation_restricted_80": {k: v for k, v in val_restricted.items()
                                            if k != "rows"},
        "source_validation_full_93": {k: v for k, v in val_full.items() if k != "rows"},
        "nikl18_training_re_recognition": {k: v for k, v in nikl_eval.items() if k != "rows"},
        "per_sample": {"source_validation_restricted_80": val_restricted["rows"],
                       "source_validation_full_93": val_full["rows"],
                       "nikl18": nikl_eval["rows"]},
        "decision": {"macro_f1_drop_restricted_80": round(drop_f1, 6),
                     "accuracy_drop_full_93": round(drop_acc, 6),
                     "rule": "HOLD if either falls more than 2 percentage points below the base",
                     "verdict": verdict},
        "scope": {"this_is": "a source regression check on material the base model was fitted on",
                  "this_is_not": ["a new-person performance figure",
                                  "a generalisation result",
                                  "a web replacement decision"],
                  "nikl18": "training-data re-recognition only",
                  "full_93_macro_denominator": "the extra 13 classes have no ground truth in the "
                                               "696, so the macro denominator stays at the "
                                               "classes that DO have ground truth"},
    }
    out = os.path.join(run_dir, "result.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("model_hashes", "data", "decision")},
                     ensure_ascii=False, indent=1))
    print("written", os.path.relpath(out, ROOT))
    return 0


def _source_label_map(classes) -> dict:
    """AIHub label -> class index, over the union order."""
    return {name: i for i, name in enumerate(classes)}


def main(argv) -> int:
    if "--train" in argv:
        return train_candidate()
    os.makedirs(CACHE_OUT, exist_ok=True)
    lists = data_lists()
    model, classes, mapping, new_words, init = build_candidate()
    report = {
        "task_id": "functional_17_union93_candidate",
        "design": "design/functional_17_union93_candidate.md",
        "status": "PREPARED_NOT_TRAINED",
        "first_limited_experiment": True,
        "settings_are_not_claimed_optimal": True,
        "model": {
            "base_checkpoint_sha256": CHECKPOINT_SHA256,
            "base_classes": len(classes) - len(new_words),
            "new_classes": len(new_words), "union_classes": len(classes),
            "new_word_ids": new_words,
            "class_order": "the existing 80 keep their existing order; the new 13 are appended",
            "head_rows_copied": init["copied_head_rows"],
            "head_rows_new": init["new_head_rows"],
            "encoder": "copied from the pinned checkpoint",
            "answer_key_given_to_the_model": False,
        },
        "held_back": HELD_BACK,
        "data": {k: (len(v) if isinstance(v, list) else v) for k, v in lists.items()},
        "data_lists": lists,
        "setting": SETTING,
        "augmentation": AUGMENT,
        "decision_rule": {
            "report_step": "the 660th step only; no intermediate checkpoint is written or chosen",
            "source_validation_readings": "two separate readings - macro-F1 restricted to the "
                                          "original 80 logits, and accuracy over all 93 "
                                          "candidates",
            "gates": {
                "restricted_80_macro_f1": {"base": 0.9897937504961497,
                                           "hold_if": "a drop of more than 2 percentage points"},
                "full_93_accuracy": {"base": 689 / 696,
                                     "hold_if": "a drop of more than 2 percentage points"}},
            "hold_rule": "BOTH gates are checked and EITHER one failing means HOLD: the "
                         "restricted-80 macro-F1 and the full-93 accuracy must each stay within 2 "
                         "percentage points of the base",
            "macro_denominator": "the extra 13 classes have no ground truth in the 696, so the "
                                 "macro denominator is never inflated to 93",
            "nikl18_recognition": "a training check only, never a new-person performance figure",
            "user_ten_attempts": "not read here; root may look once after the training is frozen",
        },
        "not_done": ["the main training", "threshold search", "per-word rule", "web change",
                     "pass line", "core DTW change", "server", "backup", "Unity"],
        "trainer": {"implemented": True, "executed_here": False,
                    "entry": "py -3.13 tools_prep_union93_candidate.py --train",
                    "writes_to": "runs/union93_candidate_v1 (refuses to overwrite)",
                    "loaders": "every AIHub npz and every NIKL cache is hash-verified before it "
                               "is read; the reserved split is refused by name",
                    "seeds": "random, numpy and torch fixed BEFORE the model is created",
                    "loss_guard": "a non-finite loss raises immediately",
                    "intermediate_checkpoints": 0},
    }
    path = os.path.join(ROOT, "evidence", "functional_17_union93_prep.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("status", "model", "held_back", "data")},
                     ensure_ascii=False, indent=1))
    print("written", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
