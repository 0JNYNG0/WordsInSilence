"""functional_08: one diagnostic over the neural input's time condition.

Contract and hypothesis: design/functional_08_time_conditions.md - written before this ran.

Three conditions, same checkpoint, same feature builder, same 75->67 assignment, same active-span
rule as the web adapter:

  A  current fixed input      real frames, real timestamps
  B  dt channel only          identical coordinates and masks, timestamps replaced by a uniform
                              30 Hz grid so dt becomes the training constant
  C  resampled to 30 Hz       a target landing on an original timestamp copies that frame's
                              coordinates and mask verbatim; only strictly-between times are
                              interpolated, per point, and only where both bracketing frames saw
                              that point. No group is erased because another is absent, no target
                              is made past the original end, and no gap is filled.
                              (v1 of this condition was withdrawn: see evidence.)

Nothing is trained, tuned, searched or deployed. No camera is opened and the web server is not
touched. Raw joint arrays and video never leave this process: the report carries aggregates,
hashes, interval statistics and predicted labels only.

Run: py -3.13 tools_diag_time_conditions.py
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

SESSION = os.path.join(ROOT, "runs", "ses_5778156c6ae4")
PREP = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "expand")
COMMON_5 = ("ALDA", "GAMSAHAMNIDA", "NA", "JOTA", "MOREUDA")
TRAIN_STEP_US = 1_000_000.0 / 30.0          # the training timebase: frame_index / 30
HAND_SLOTS = slice(25, 67)                  # the native slots that carry a hand


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# ---------------------------------------------------------------- the three time conditions
def condition_a(nat, present, ts):
    """What the web does today."""
    return nat, present, ts


def condition_b(nat, present, ts):
    """Same frames, same masks; only the dt channel's value changes."""
    n = len(ts)
    grid = (np.arange(n, dtype=np.float64) * TRAIN_STEP_US).astype(np.int64)
    return nat, present, grid


EXACT_TOL_US = 1        # integer-microsecond rounding tolerance, part of the contract below


def condition_c(nat, present, ts):
    """Resample to 30 Hz. Exact samples are COPIED; only strictly-between times interpolate.

    Contract (v2, after the independent counterexample):

    * A target that lands on an original timestamp within EXACT_TOL_US = 1 us copies that frame's
      coordinates and mask VERBATIM. The v1 code interpolated there anyway and so ANDed the mask
      with the neighbour, which dropped observed boundary samples and made a 30 Hz input not an
      identity.
    * Only strictly between two original frames is a point interpolated, and only where BOTH
      bracketing frames observe THAT point. Presence is preserved per point: a missing hand never
      deletes the body, and no group is erased because another group is absent. The v1 code gated
      the whole resampling on runs of hand observation, which erased body observation wherever the
      hands were not seen.
    * No target is generated beyond the original last timestamp. The count is computed with the
      same 1 us tolerance, and a nominal target that overshoots the original end by at most that
      tolerance is SNAPPED to the original end and copied - otherwise a sub-microsecond rounding
      difference in the source timestamps silently dropped the last original sample (6 frames
      became 5, 32 became 31).
    * The output timestamps are the actual target times minus the start, not a recomputed nominal
      grid, so the end boundary the payload reports is the one that was used.
    * An isolated exact observation survives, because it is copied rather than interpolated.
    dt therefore carries the training constant to within that same tolerance.
    """
    tsf = np.asarray(ts, dtype=np.int64)
    n_in = len(tsf)
    if n_in < 2 or tsf[-1] <= tsf[0]:
        return nat, present, ts
    t0, t1 = int(tsf[0]), int(tsf[-1])
    # the tolerance belongs in the COUNT too: without it a source whose integer timestamps were
    # truncated rather than rounded loses its last sample to a sub-microsecond shortfall
    n_out = int(np.floor((t1 - t0 + EXACT_TOL_US) / TRAIN_STEP_US)) + 1
    targets = t0 + np.rint(np.arange(n_out, dtype=np.float64) * TRAIN_STEP_US).astype(np.int64)
    # a nominal target that overshoots the end by at most the tolerance is snapped onto it; any
    # target beyond that is dropped, so no time past the original end is ever emitted
    over = (targets > t1) & (targets <= t1 + EXACT_TOL_US)
    targets[over] = t1
    targets = np.unique(targets[targets <= t1])
    xy = np.zeros((len(targets), nat.shape[1], 2), dtype=np.float32)
    pr = np.zeros((len(targets), nat.shape[1]), dtype=bool)
    for k, tau in enumerate(targets):
        near = int(np.searchsorted(tsf, tau, side="left"))
        exact = None
        for i in (near - 1, near, near + 1):
            if 0 <= i < n_in and abs(int(tsf[i]) - int(tau)) <= EXACT_TOL_US:
                exact = i
                break
        if exact is not None:
            xy[k] = nat[exact]
            pr[k] = present[exact]
            continue
        hi = int(np.searchsorted(tsf, tau, side="right"))
        lo = hi - 1
        if lo < 0 or hi >= n_in:
            continue                                       # outside the original: nothing invented
        span = float(tsf[hi] - tsf[lo])
        w = 0.0 if span <= 0 else (float(tau) - float(tsf[lo])) / span
        both = present[lo] & present[hi]                   # per point, per group
        pr[k] = both
        xy[k][both] = (1.0 - w) * nat[lo][both] + w * nat[hi][both]
    return xy, pr, (targets - t0).astype(np.int64)      # the times actually used, from the start


CONDITIONS = (("A_current", condition_a), ("B_dt_only", condition_b), ("C_resampled", condition_c))


# ---------------------------------------------------------------- inference
def predict(nat, present, ts, model, classes):
    """The adapter's own preparation, then one forward pass. No tuning of any kind."""
    import torch
    from pilot.data import build_channels, collate
    from pilot.web_input import active_span

    span = active_span(present)
    if span is None:
        return {"status": "NO_USABLE_HAND_FRAME"}
    lo, hi = span
    observed = int(present[lo:hi + 1, HAND_SLOTS].any(axis=1).sum())
    if (hi - lo + 1) < neural.MIN_FRAMES or observed < neural.MIN_OBSERVED_HAND_FRAMES:
        return {"status": "INSUFFICIENT_OBSERVATION", "frames": int(hi - lo + 1),
                "observed_hand_frames": observed}
    if not np.all(np.diff(ts[lo:hi + 1]) > 0):
        return {"status": "NON_MONOTONIC_TIMESTAMPS"}
    clip = build_channels(nat[lo:hi + 1], present[lo:hi + 1], ts[lo:hi + 1])
    x, pad, _ = collate([dict(clip, label=0)])
    with torch.inference_mode():
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    idx = {classes[i]: i for i in range(len(classes))}
    restricted = sorted(((w, lab, float(logit[idx[lab]]))
                         for w, lab in neural.PROJECT_OVERLAP.items() if lab in idx),
                        key=lambda r: -r[2])
    return {"status": "OK",
            "frames": int(clip["frames"]),
            "observed_hand_frames": observed,
            "top1_all": classes[int(np.argmax(logit))],
            "top1_project": restricted[0][0] if restricted else None,
            "median_dt_ms": round(float(np.median(np.diff(ts[lo:hi + 1]))) / 1000.0, 3)}


def _train_ids() -> set:
    """The sample ids the pinned checkpoint's own run recorded as its train split."""
    run = os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "runs",
                       "masked_bn_pair_v1", "inputs.json")
    with open(run, encoding="utf-8") as fh:
        return {it["sample_id"] for it in json.load(fh)["train"]}


def _length_stats(items, fps):
    fc = np.array([m["frame_count"] for m in items], dtype=float)
    return {"n": int(len(fc)), "min": int(fc.min()), "p50": int(np.median(fc)),
            "p90": int(np.percentile(fc, 90)), "max": int(fc.max()),
            "seconds_p50": round(float(np.median(fc)) / fps, 3),
            "seconds_p90": round(float(np.percentile(fc, 90)) / fps, 3)}


def training_time_policy() -> dict:
    """Small existing metadata only: the interval, the length spread and the augmentation policy."""
    with open(os.path.join(PREP, "build_report.json"), encoding="utf-8") as fh:
        rep = json.load(fh)
    with open(os.path.join(PREP, "manifest.json"), encoding="utf-8") as fh:
        man = json.load(fh)
    fc = np.array([m["frame_count"] for m in man], dtype=float)
    bases = sorted({m["timebase"] for m in man})
    fps = sorted({m["nominal_fps"] for m in man})
    return {
        "nominal_fps": fps,
        "timebase": bases,
        "dt_per_frame_s": round(1.0 / fps[0], 6) if len(fps) == 1 else None,
        "dt_is_constant_in_training": len(fps) == 1 and len(bases) == 1,
        "clip_frame_count": {"n": int(len(fc)), "min": int(fc.min()), "p50": int(np.median(fc)),
                             "p90": int(np.percentile(fc, 90)), "max": int(fc.max())},
        "clip_seconds_at_nominal_fps": {"p50": round(float(np.median(fc)) / fps[0], 3),
                                        "p90": round(float(np.percentile(fc, 90)) / fps[0], 3)},
        # the line above is EVERY manifest entry. What the model actually saw is the train split
        # among the pilot-eligible clips, which is what load_split feeds, so both are reported.
        "scope_note": "clip_frame_count and clip_seconds_at_nominal_fps cover EVERY manifest "
                      "entry, validation and reserved_test included, so they are not what the "
                      "model saw. checkpoint_train_set is the exact train sample list recorded by "
                      "the pinned checkpoint's own run; manifest_split_role_train is the "
                      "manifest's train role for comparison.",
        "all_manifest_entries": _length_stats(man, fps[0]),
        # what the pinned checkpoint actually saw: the exact train sample ids its run recorded
        "checkpoint_train_set": _length_stats(
            [m for m in man if m["sample_id"] in _train_ids()], fps[0]),
        "manifest_split_role_train": _length_stats(
            [m for m in man if m.get("split_role") == "train"], fps[0]),
        "time_augmentation": "NONE FOUND: pilot/data.py builds dt from the timestamps and no "
                             "resampling, time warp, speed jitter or frame dropping exists in "
                             "src/pilot",
        "source": {"build_report": rep["task_id"], "written_samples": rep["written_samples"],
                   "min_frames_setting": rep["settings"]["min_frames"]},
    }


def main(only=None) -> int:
    started = time.time()
    conditions = [(n2, f) for n2, f in CONDITIONS if only is None or n2 == only]
    c = neural._load()                                   # noqa: SLF001 - the same cache the web uses
    if c["status"] != "READY":
        print("model not ready:", c.get("detail"))
        return 1
    model, classes = c["model"], c["classes"]

    reviews = {}
    with open(os.path.join(SESSION, "trial_reviews.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                reviews[r["trial_id"]] = r["actual_word_id"]

    from pilot.web_input import to_native_like

    inputs = []
    tdir = os.path.join(SESSION, "trials")
    for tid in sorted(reviews):
        npz = os.path.join(tdir, tid + ".npz")
        stored = None
        rp = os.path.join(tdir, tid + ".result.json")
        if os.path.exists(rp):
            with open(rp, encoding="utf-8") as fh:
                res = json.load(fh)
            lr = res.get("learned_recognition") or {}
            rank = lr.get("ranking") or {}
            stored = {
                "model_status": lr.get("model_status"),
                "top1_all": (rank.get("over_all_trained_classes") or {}).get("top1"),
                "top1_project": (rank.get("restricted_to_project_words") or {}).get("top1"),
                "dtw_top1": res.get("top1_word_id"),
                "observation_status": res.get("observation_status"),
            }
        inputs.append({"kind": "session_attempt", "id": tid, "npz": npz,
                       "confirmed": reviews[tid], "stored": stored})

    with open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8") as fh:
        clips = json.load(fh)["clips"]
    for clip in clips:
        if clip["word_id"] in COMMON_5:
            inputs.append({"kind": "nikl_reference", "id": clip["clip_id"],
                           "npz": os.path.join(os.path.dirname(ROOT), "front_keyphase_v1",
                                               clip["observation_npz"]),
                           "confirmed": clip["word_id"], "stored": None})

    rows = []
    for item in inputs:
        obs = Observation.load(item["npz"])
        nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                      np.asarray(obs.usable, dtype=bool))
        ts = np.asarray(obs.timestamp_us, dtype=np.int64)
        row = {"kind": item["kind"], "id": item["id"], "confirmed_word_id": item["confirmed"],
               "npz_sha256": sha256_file(item["npz"]), "stored": item["stored"],
               "native_frames": int(len(ts)),
               "median_dt_ms": round(float(np.median(np.diff(ts))) / 1000.0, 3)
               if len(ts) > 1 else None,
               "conditions": {}}
        for name, fn in conditions:
            a, p, t = fn(nat.copy(), present.copy(), ts.copy())
            row["conditions"][name] = predict(a, p, t, model, classes)
        rows.append(row)

    if only is not None:
        # merge into the existing record: the other conditions are not re-run
        path = os.path.join(ROOT, "evidence", "functional_08_time_conditions.json")
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
        by_id = {r["id"]: r for r in rows}
        for r in prev["rows"]:
            if r["id"] in by_id:
                r["conditions"][only] = by_id[r["id"]]["conditions"][only]
        print(json.dumps({"re_evaluated": only, "inputs": len(rows)}, ensure_ascii=False))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(prev, fh, ensure_ascii=False, indent=1)
        return 0

    def tally(kind, cond):  # noqa: D401
        sub = [r for r in rows if r["kind"] == kind]
        ok = [r for r in sub if r["conditions"][cond].get("status") == "OK"]
        hit = sum(1 for r in ok
                  if r["conditions"][cond]["top1_project"] == r["confirmed_word_id"])
        return {"n": len(sub), "answered": len(ok), "matched": hit,
                "unanswered": [r["conditions"][cond].get("status") for r in sub
                               if r["conditions"][cond].get("status") != "OK"]}

    attempts = [r for r in rows if r["kind"] == "session_attempt"]
    baseline_ok = all(r["stored"] and r["stored"]["top1_project"]
                      == r["conditions"]["A_current"].get("top1_project") for r in attempts
                      if r["stored"] and r["stored"].get("model_status") == "READY")
    changed = {}
    for name, _ in CONDITIONS[1:]:
        changed[name] = [{"id": r["id"], "kind": r["kind"],
                          "from": r["conditions"]["A_current"].get("top1_project"),
                          "to": r["conditions"][name].get("top1_project")}
                         for r in rows
                         if r["conditions"]["A_current"].get("top1_project")
                         != r["conditions"][name].get("top1_project")]

    out = {
        "task_id": "functional_08_time_conditions",
        "design": "design/functional_08_time_conditions.md",
        "session": os.path.basename(SESSION),
        "is_a_cause_claim": False,
        "reading_rule": "a label change demonstrates sensitivity to the time condition. It does "
                        "NOT establish that the time condition caused the misses, and selecting "
                        "anything after seeing these ten attempts is development diagnosis, never "
                        "an independent performance test.",
        "checkpoint_sha256": neural.CHECKPOINT_SHA256,
        "training_time_policy": training_time_policy(),
        "baseline_reproduces_stored_labels": bool(baseline_ok),
        "tallies": {k: {name: tally(k, name) for name, _ in CONDITIONS}
                    for k in ("session_attempt", "nikl_reference")},
        "label_changes_vs_A": changed,
        "rows": rows,
        "elapsed_s": round(time.time() - started, 1),
        "no_training": True, "no_threshold_change": True, "no_model_search": True,
        "no_deployment": True, "camera_opened": False, "server_touched": False,
    }
    path = os.path.join(ROOT, "evidence", "functional_08_time_conditions.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"tallies": out["tallies"],
                      "baseline_reproduces_stored_labels": out["baseline_reproduces_stored_labels"],
                      "training_dt_s": out["training_time_policy"]["dt_per_frame_s"],
                      "time_augmentation": out["training_time_policy"]["time_augmentation"][:40],
                      "changes": {k: len(v) for k, v in changed.items()},
                      "written": os.path.relpath(path, ROOT)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(only=arg))
