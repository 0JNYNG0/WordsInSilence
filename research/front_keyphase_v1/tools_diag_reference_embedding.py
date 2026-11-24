"""functional_11: nearest NIKL reference in the pinned checkpoint's own embedding space.

Contract: design/functional_11_reference_embedding.md - written before this ran.

The fixed MaskedSignTCN is used as it is. No file under neural.py, no model source and no
checkpoint is modified: the 64-wide pooled vector entering self.head is captured with a
forward_pre_hook, and head(captured) is asserted equal to the logits the same forward returned.

Each embedding is L2-normalised; the predicted word is the NIKL reference with the highest cosine
similarity. The 18 DTW candidates are the references, one clip each, all equal - no weighting, no
threshold, no search, no fusion, no personalisation. Preprocessing and the time condition are
today's (condition A); time condition C is not used. The ten attempts are evaluation only and are
never support samples.

Hashes, scalars and labels only: no raw coordinate, video or embedding array is printed.

Run: py -3.13 tools_diag_reference_embedding.py
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
HEAD_GAP_TOL = 1e-4          # the run stops if head(embedding) and the logits differ by more
CACHE = os.path.join(ROOT, "runs", "cache_functional_11")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class HeadTap:
    """Capture the pooled vector entering the head, without touching the model's source."""

    def __init__(self, model):
        self.value = None
        self._handle = model.head.register_forward_pre_hook(self._grab)

    def _grab(self, _module, args):
        self.value = args[0].detach().clone()

    def close(self):
        self._handle.remove()


def embed(clip_x: np.ndarray, model, tap) -> tuple:
    """One forward. Returns (pooled[64], logits) and proves the two are consistent."""
    import torch
    from pilot.data import collate

    x, pad, _ = collate([{"x": clip_x, "frames": int(len(clip_x)), "label": 0}])
    with torch.inference_mode():
        logit = model(torch.from_numpy(x), torch.from_numpy(pad))[0]
        pooled = tap.value[0]
        # the captured vector must reproduce the logits this very forward returned
        again = model.head(pooled.unsqueeze(0))[0]
        gap = float((again - logit).abs().max())
    # an explicit check, not a flag written afterwards: if the captured vector does not reproduce
    # this forward's logits, every number downstream is meaningless, so the run stops here
    if not (gap < HEAD_GAP_TOL):
        raise AssertionError(f"head(captured embedding) != logits of the same forward: {gap}")
    return pooled.numpy().astype(np.float64), logit.numpy(), gap


def web_clip(path: str):
    from pilot.data import build_channels
    from pilot.web_input import active_span, to_native_like

    obs = Observation.load(path)
    nat, present = to_native_like(np.asarray(obs.xy_px, dtype=np.float32),
                                  np.asarray(obs.usable, dtype=bool))
    span = active_span(present)
    if span is None:
        return None
    lo, hi = span
    ts = np.asarray(obs.timestamp_us, dtype=np.int64)[lo:hi + 1]
    if not np.all(np.diff(ts) > 0):
        return None
    return build_channels(nat[lo:hi + 1], present[lo:hi + 1], ts)


def main() -> int:
    started = time.time()
    c = neural._load()                                # noqa: SLF001 - the same pinned checkpoint
    if c["status"] != "READY":
        print("model not ready:", c.get("detail"))
        return 1
    model, classes = c["model"], c["classes"]
    tap = HeadTap(model)
    os.makedirs(CACHE, exist_ok=True)

    with open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    # the references the DTW baseline already uses: the prepared candidates, one clip each
    ready = {"READY", "NEEDS_REVIEW"}
    refs, ref_failed, max_gap = [], [], 0.0
    for clip_meta in manifest["clips"]:
        if clip_meta["readiness"] not in ready:
            ref_failed.append({"word_id": clip_meta["word_id"], "clip_id": clip_meta["clip_id"],
                               "reason": "READINESS_" + clip_meta["readiness"]})
            continue
        path = os.path.join(ROOT, clip_meta["observation_npz"])
        try:
            clip = web_clip(path)
            if clip is None:
                raise ValueError("NO_USABLE_SPAN")
            vec, logit, gap = embed(clip["x"], model, tap)
        except Exception as exc:                      # noqa: BLE001 - reported, never hidden
            ref_failed.append({"word_id": clip_meta["word_id"], "clip_id": clip_meta["clip_id"],
                               "reason": f"{type(exc).__name__}: {exc}"})
            continue
        max_gap = max(max_gap, gap)
        refs.append({"word_id": clip_meta["word_id"], "clip_id": clip_meta["clip_id"],
                     "npz_sha256": sha256_file(path), "frames": int(clip["frames"]),
                     "head_top1": classes[int(np.argmax(logit))], "vec": vec})

    reviews = {}
    with open(os.path.join(SESSION, "trial_reviews.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                reviews[r["trial_id"]] = r["actual_word_id"]

    def nearest(vec, exclude_clip=None):
        v = vec / max(float(np.linalg.norm(vec)), 1e-12)
        scored = []
        for r in refs:
            if exclude_clip is not None and r["clip_id"] == exclude_clip:
                continue
            u = r["vec"] / max(float(np.linalg.norm(r["vec"])), 1e-12)
            scored.append((float(np.dot(v, u)), r["word_id"], r["clip_id"]))
        scored.sort(key=lambda t: -t[0])
        return scored

    rows = []
    tdir = os.path.join(SESSION, "trials")
    for tid in sorted(reviews):
        npz = os.path.join(tdir, tid + ".npz")
        clip = web_clip(npz)
        if clip is None:
            rows.append({"trial_id": tid, "confirmed_word_id": reviews[tid],
                         "status": "NO_USABLE_SPAN"})
            continue
        vec, logit, gap = embed(clip["x"], model, tap)
        max_gap = max(max_gap, gap)
        scored = nearest(vec)
        with open(os.path.join(tdir, tid + ".result.json"), encoding="utf-8") as fh:
            stored = json.load(fh)
        rows.append({
            "trial_id": tid, "confirmed_word_id": reviews[tid], "status": "OK",
            "npz_sha256": sha256_file(npz), "frames": int(clip["frames"]),
            "embedding_top1_18": scored[0][1],
            "embedding_top1_common5": next(
                (w for _, w, _ in scored if w in neural.PROJECT_OVERLAP), None),
            "embedding_top1_similarity": round(scored[0][0], 4),
            "embedding_margin_to_second": round(scored[0][0] - scored[1][0], 4)
            if len(scored) > 1 else None,
            "head_top1_all80": classes[int(np.argmax(logit))],
            "dtw_top1": stored.get("top1_word_id"),
            "dtw_observation_status": stored.get("observation_status"),
        })
        np.save(os.path.join(CACHE, tid + ".npy"), vec)   # local cache only, never printed

    for r in refs:
        np.save(os.path.join(CACHE, r["clip_id"] + ".npy"), r["vec"])

    ok = [r for r in rows if r["status"] == "OK"]
    # the denominator is every confirmed attempt, not the ones that happened to be evaluable:
    # scoring over survivors alone would quietly flatter any condition that drops hard inputs
    total = len(rows)
    not_evaluated = [{"trial_id": r["trial_id"], "status": r["status"]}
                     for r in rows if r["status"] != "OK"]
    emb18 = sum(1 for r in ok if r["embedding_top1_18"] == r["confirmed_word_id"])
    emb5 = sum(1 for r in ok if r["embedding_top1_common5"] == r["confirmed_word_id"])
    dtw18 = sum(1 for r in ok if r["dtw_top1"] == r["confirmed_word_id"])
    # NIKL self-match: an implementation check, excluded from every performance figure.
    # Leave-one-out is NOT computed: with exactly one reference clip per word, removing it deletes
    # the only support for the correct answer, so the score is structurally forced to zero and
    # would read as a performance number while carrying no information.
    self_match = sum(1 for r in refs if nearest(r["vec"])[0][2] == r["clip_id"])

    confusion = {}
    for r in ok:
        confusion.setdefault(r["confirmed_word_id"], {}).setdefault(r["embedding_top1_18"], 0)
        confusion[r["confirmed_word_id"]][r["embedding_top1_18"]] += 1
    cross = {}
    for r in ok:
        key = f"emb={r['embedding_top1_18']}|dtw={r['dtw_top1']}"
        cross[key] = cross.get(key, 0) + 1

    rule_18 = emb18 > 4
    rule_5 = emb5 >= 8
    out = {
        "task_id": "functional_11_reference_embedding",
        "design": "design/functional_11_reference_embedding.md",
        "checkpoint_sha256": neural.CHECKPOINT_SHA256,
        "method": "pooled 64-d vector entering head, captured by forward_pre_hook; L2-normalised; "
                  "cosine nearest among the DTW reference clips, one each, all equal",
        "no_model_file_modified": True,
        "head_consistency": {"max_abs_logit_gap": round(max_gap, 9),
                             "tolerance": HEAD_GAP_TOL,
                             "rule": "checked inside embed(): the run raises and stops on the "
                                     "first input where head(captured embedding) differs from the "
                                     "logits of that same forward. Reaching this line means every "
                                     "input passed.",
                             "enforced_by_assertion": True},
        "time_condition": "A (today's preprocessing). Condition C was NOT used.",
        "references": {"n": len(refs), "word_ids": sorted(r["word_id"] for r in refs),
                       "failed": ref_failed,
                       "embedding_dim": int(len(refs[0]["vec"])) if refs else 0},
        "implementation_check_not_performance": {
            "nikl_self_match": f"{self_match}/{len(refs)}",
            "leave_one_out": "NOT COMPUTED - with one reference clip per word it removes the only "
                             "support for the correct answer, so the value is structurally forced "
                             "and is not a performance reading",
            "meaning": "self-match only shows the cache and the metric work. It is excluded from "
                       "every performance figure below."},
        "attempts": {
            "n": total, "evaluated": len(ok), "not_evaluated": not_evaluated,
            "denominator": "every confirmed attempt (n), not the evaluable subset",
            "used_as_support_samples": False,
            "embedding_over_18_candidates": f"{emb18}/{total}",
            "embedding_over_common_5": f"{emb5}/{total}",
            "dtw_over_18_candidates": f"{dtw18}/{total}",
            "dtw_over_common_5_recorded_by_the_user": "8/10",
            "head_over_all_80_recorded": "2/10"},
        "confusion_embedding_18": confusion,
        "cross_table_embedding_vs_dtw": cross,
        "pre_registered_rule": {
            "over_18_must_beat_dtw_4_of_10": {"required": "> 4", "observed": emb18,
                                              "met": bool(rule_18)},
            "over_common_5_must_not_be_below_dtw_8_of_10": {"required": ">= 8", "observed": emb5,
                                                            "met": bool(rule_5)},
            "verdict": "FOLLOW_UP_CANDIDATE" if (rule_18 and rule_5) else "NOT_PROMOTED"},
        "not_an_independent_performance_result": "these ten attempts are development material "
                                                 "already examined; no figure here is an "
                                                 "independent or new-person performance result",
        "web_unchanged": True, "no_training": True, "no_threshold": True, "no_fusion": True,
        "no_personalisation": True, "no_candidate_search": True,
        "embedding_cache_dir": os.path.relpath(CACHE, ROOT),
        "raw_arrays_emitted": False,
        "rows": [{k: v for k, v in r.items()} for r in rows],
        "elapsed_s": round(time.time() - started, 1),
    }
    tap.close()
    path = os.path.join(ROOT, "evidence", "functional_11_reference_embedding.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("head_consistency", "references",
                                          "implementation_check_not_performance", "attempts",
                                          "cross_table_embedding_vs_dtw", "pre_registered_rule")},
                     ensure_ascii=False, indent=1))
    print("written", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
