"""Run the research neural candidate on the SAME observation the DTW path already uses.

This is a research surface, not a verdict. Three things it never does:

  * it never sees requested_word_id or attempt_label. The ranking is produced blind.
  * a neural candidate is not a statement that the sign was performed correctly. accepted and the
    execution assessment stay exactly where they were: null and UNCONFIGURED.
  * it never takes the DTW app down. torch missing, checkpoint missing, hash mismatch or an
    inference error all come back as a model_status string while recognition continues.

Scope, stated plainly because it is small. The DEFAULT checkpoint knows 80 words from the AIHub
sentence corpus, and only five of them overlap the project's 19: 알다, 감사합니다, 나, 좋다, 모르다.
A registry entry carries its OWN coverage - the opt-in union-93 candidate covers 18 of the 19 -
so every result states the coverage of the model that answered rather than repeating one model's
number. Whatever a model was not trained on is reported NOT_IN_MODEL and is never scored. Independent review replayed the five reference clips through this checkpoint and got the
right name 2 of 5 over all 80 candidates, and 4 of 5 when the choice is restricted to the five
project words. That weakness is reported, not hidden, and the checkpoint was NOT re-picked to
improve it - it was chosen because its padding-statistics fix had already been reviewed and because
it was selected on source validation, never on any web score.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
import time

import numpy as np

from ..cli import ROOT

PILOT = os.path.normpath(os.path.join(ROOT, "..", "sign_model_pilot_v1"))
CHECKPOINT = os.path.join(PILOT, "runs", "masked_bn_pair_v1", "MaskedSignTCN_best.pt")
CHECKPOINT_SHA256 = "56a17a4d0a4795a2463add5572acc0d985c3cb7a1cbaee2853698f106d5b34b5"
MODEL_ID = "masked_sign_tcn_bn_pair_best"
ADAPTER_SCHEMA = "wis_learned_recognition_v1"
# the project words this checkpoint was trained on, by the id the web uses
PROJECT_OVERLAP = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다", "NA": "나",
                   "JOTA": "좋다", "MOREUDA": "모르다"}
MIN_FRAMES = 4                 # minimum frames in the active span
MIN_OBSERVED_HAND_FRAMES = 4   # minimum frames that actually show a hand
_LOCK = threading.Lock()

# ---------------------------------------------------------------- the registry
# The default stays exactly what it was. A research candidate is an ENTRY here, never a
# replacement: a caller has to name it, and a caller that names nothing gets the default.
DEFAULT_MODEL = "base"
CANDIDATE_DIR = os.path.join(ROOT, "runs", "union93_candidate_v1")
VISIBILITY_DIR = os.path.join(ROOT, "runs", "visibility_candidate_v1")
# the 18 project words the union-93 candidate covers. KKEUDEOGIDA is absent by rule - its cue is
# not carried by the observation contract - and is reported NOT_IN_MODEL like any other.
PROJECT_18 = {
    "ALDA": "알다", "GAMSAHAMNIDA": "감사합니다", "NA": "나", "JOTA": "좋다",
    "MOREUDA": "모르다", "GAM": "감", "GONGBU": "공부", "NEO": "너", "MAEKJU": "맥주",
    "MEOKDA": "먹다", "MUL": "물", "PPANG": "빵", "SARANG": "사랑", "SILTA": "싫다",
    "UNDONG": "운동", "INSA": "인사", "JADA": "자다", "JOESONGHADA": "죄송하다",
}
MODELS = {
    "base": {
        "model_id": MODEL_ID,
        "path": CHECKPOINT,
        "sha256": CHECKPOINT_SHA256,
        "project_words": PROJECT_OVERLAP,
        "role": "the operating default; unchanged by functional_18",
        "status_label": "기본",
    },
    "union93_candidate_v1": {
        "model_id": "union93_candidate_v1",
        "path": os.path.join(CANDIDATE_DIR, "union93_final.pt"),
        "sha256": "e7c9c31a9c747e2c12a9a56695d2e4e9dbec9c372c8e2a3f209d00fa4c25dced",
        "project_words": PROJECT_18,
        "role": "an opt-in research comparison. NOT promoted, NOT a verdict, and not evidence "
                "about a new person",
        "status_label": "검증 중",
    },
    # functional_23: the augmentation arm of the functional_22 experiment, continued 220 steps
    # from the entry above with the reviewed non-signing hand sometimes marked unobserved. It is
    # another ENTRY, not a replacement: the default and the entry above are untouched, and a
    # trial that names nothing still gets the default.
    "union93_visibility_v1": {
        "model_id": "union93_visibility_v1",
        "path": os.path.join(VISIBILITY_DIR, "visibility_final.pt"),
        "sha256": "258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8",
        "project_words": PROJECT_18,
        "role": "an opt-in research comparison, trained with reviewed passive-hand missingness. "
                "NOT promoted, NOT a verdict, and not evidence about a new person",
        "status_label": "검증 중",
    },
}
# the five both models know: the only scope where a name-for-name comparison is even defined
COMMON_5 = {k: v for k, v in PROJECT_OVERLAP.items()}

_CACHES = {}
_CACHE = {"model": None, "classes": None, "status": None, "detail": None}
_CACHES["base"] = _CACHE


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _cache_for(key: str) -> dict:
    if key not in _CACHES:
        _CACHES[key] = {"model": None, "classes": None, "status": None, "detail": None}
    return _CACHES[key]


def _load(model_key: str = DEFAULT_MODEL):
    """Load once per model and cache. Every failure is a status, never an exception.

    The hash read sits inside the guard too: an OSError there used to propagate out of
    capability() and out of the replay handler, which would have taken the DTW path down with the
    model. A lock keeps two first callers from loading the same checkpoint twice.

    A named model that cannot be loaded reports its own failure. It is NEVER quietly replaced by
    the default - a caller asking for the research candidate must not be handed the base model
    and told nothing.
    """
    if model_key not in MODELS:
        return {"model": None, "classes": None, "status": "UNKNOWN_MODEL_KEY",
                "detail": f"{model_key!r} is not in the registry"}
    c = _cache_for(model_key)
    if c["status"] is not None:
        return c
    with _LOCK:
        if c["status"] is not None:
            return c
        return _load_locked(model_key)


def _load_locked(model_key: str = DEFAULT_MODEL):
    spec = MODELS[model_key]
    path, pinned = spec["path"], spec["sha256"]
    c = _cache_for(model_key)
    try:
        if not os.path.exists(path):
            c.update(status="CHECKPOINT_MISSING",
                     detail="the pinned checkpoint is not present")
            return c
        digest = _sha256(path)
    except OSError as exc:                                # noqa: BLE001 - reported, not raised
        c.update(status="CHECKPOINT_UNREADABLE", detail=type(exc).__name__)
        return c
    if digest != pinned:
        c.update(status="CHECKPOINT_HASH_MISMATCH",
                 detail="the checkpoint on disk is not the pinned one")
        return c
    try:
        import torch  # noqa: F401
    except Exception:                                     # noqa: BLE001 - reported, not raised
        c.update(status="TORCH_UNAVAILABLE", detail="torch is not installed in this runtime")
        return c
    src = os.path.join(PILOT, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import torch
        from pilot.model_masked_v1 import MaskedSignTCN
        ck = torch.load(path, weights_only=True, map_location="cpu")
        classes = ck["classes"]
        cfg = ck.get("config", {})

        def need(k):
            if k in ck:
                return ck[k]
            if k in cfg:
                return cfg[k]
            raise KeyError(k)

        model = MaskedSignTCN(width=need("width"), dilations=tuple(need("dilations")),
                              kernel_size=need("kernel_size"), dropout=need("dropout"),
                              n_classes=len(classes))
        model.load_state_dict(ck["state_dict"])
        model.eval()
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        c.update(status="LOAD_FAILED", detail=f"{type(exc).__name__}")
        return c
    c.update(model=model, classes=classes, status="READY", detail=None,
             checkpoint_step=ck.get("step"), model_version=ck.get("model_version"),
             checkpoint_sha256=digest,
             feature_schema=(ck.get("feature_schema") or {}).get("builder",
                                                                 "pilot.data.build_channels"),
             in_channels=ck.get("in_channels"))
    return c


def model_identity(model_key: str = DEFAULT_MODEL) -> dict:
    """Exactly which model answered, fixed per trial and written beside the result."""
    spec = MODELS.get(model_key)
    if spec is None:
        return {"model_key": model_key, "status": "UNKNOWN_MODEL_KEY"}
    c = _load(model_key)
    return {
        "model_key": model_key, "model_id": spec["model_id"],
        "checkpoint_sha256": spec["sha256"],
        "checkpoint_present": os.path.exists(spec["path"]),
        "model_status": c["status"], "model_detail": c.get("detail"),
        "trained_class_count": len(c.get("classes") or []),
        "feature_schema": c.get("feature_schema", "pilot.data.build_channels"),
        "in_channels": c.get("in_channels"),
        "role": spec["role"], "status_label": spec["status_label"],
        "project_words_in_model": sorted(spec["project_words"]),
        "is_a_verdict": False,
    }


def registry_capability() -> dict:
    """Every registered model and its identity, so a client can offer the opt-in honestly."""
    return {"default_model_key": DEFAULT_MODEL,
            "models": {k: model_identity(k) for k in MODELS},
            "scopes": ["common5", "project18", "full"],
            "scope_warning": "the scopes have different candidate counts and are different "
                             "tasks; they are never compared as equals",
            "opt_in": "a research candidate answers only when a trial names it. Nothing changes "
                      "for a trial that does not.",
            "identities_are_never_pooled": "each entry carries its own model id and checkpoint "
                                           "hash. Two candidates are two different tasks' worth "
                                           "of evidence and are never added into one figure.",
            "promotion": "none. The default model, the pass line and every verdict field are "
                         "unchanged."}


def capability() -> dict:
    """What the neural surface can do, answerable before a session starts."""
    c = _load()
    classes = c.get("classes") or []
    return {
        "schema": ADAPTER_SCHEMA, "model_id": MODEL_ID,
        "model_status": c["status"], "model_detail": c.get("detail"),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_present": os.path.exists(CHECKPOINT),
        "trained_class_count": len(classes),
        "project_words_in_model": sorted(PROJECT_OVERLAP),
        "project_words_not_in_model": "every other word of the 19 for THIS model; they are "
                                      "reported NOT_IN_MODEL and are never scored. Another "
                                      "registry entry has its own coverage.",
        "is_a_verdict": False,
        "note": "a research candidate over the words this checkpoint was trained on. It is not a "
                "correctness judgement and does not change accepted or the execution assessment.",
        "known_weakness": "independent replay of the five reference clips gave the right name 2 "
                          "of 5 over all 80 candidates, 4 of 5 restricted to the five project "
                          "words",
    }


def _channels(obs) -> tuple:
    """Observation -> the pilot's 280-channel contract, via the same 75->67 assignment."""
    from pilot.data import build_channels
    from pilot.web_input import active_span, to_native_like
    xy = np.asarray(obs.xy_px, dtype=np.float32)
    usable = np.asarray(obs.usable, dtype=bool)
    ts = np.asarray(obs.timestamp_us, dtype=np.int64)
    nat, present = to_native_like(xy, usable)
    span = active_span(present)
    if span is None:
        return None, "NO_USABLE_HAND_FRAME", {"span_frames": 0, "observed_hand_frames": 0,
                                              "observed_hand_fraction": 0.0}
    lo, hi = span
    span_frames = int(hi - lo + 1)
    # the span is only the distance between the FIRST and LAST frame that showed a hand. Counting
    # it as observation let two visible frames at the ends of a long clip pass as a full take, and
    # the model then named a word from almost nothing. Count the frames that carry a hand instead.
    observed = int(present[lo:hi + 1, 25:67].any(axis=1).sum())
    stats = {"span_frames": span_frames, "observed_hand_frames": observed,
             "observed_hand_fraction": round(observed / span_frames, 4) if span_frames else 0.0}
    if span_frames < MIN_FRAMES:
        return None, "TOO_FEW_USABLE_FRAMES", stats
    if observed < MIN_OBSERVED_HAND_FRAMES:
        return None, "TOO_FEW_OBSERVED_HAND_FRAMES", stats
    if not np.all(np.diff(ts[lo:hi + 1]) > 0):
        return None, "NON_MONOTONIC_TIMESTAMPS", stats
    return build_channels(nat[lo:hi + 1], present[lo:hi + 1], ts[lo:hi + 1]), None, stats


def _scoped(logit, classes, words: dict, name: str, meaning: str) -> dict:
    """One ranking scope, carrying its own candidate count so scopes cannot be read as one task."""
    idx = {classes[i]: i for i in range(len(classes))}
    rows = sorted(({"word_id": w, "label": lab, "score": round(float(logit[idx[lab]]), 5)}
                   for w, lab in words.items() if lab in idx), key=lambda r: -r["score"])
    return {"scope": name, "candidate_count": len(rows),
            "top1": rows[0]["word_id"] if rows else None, "order": rows,
            "meaning": meaning,
            "not_comparable_across_scopes": "a different candidate count is a different task; "
                                            "these numbers are never placed side by side as if "
                                            "they measured the same thing"}


def recognise(obs, model_key: str = DEFAULT_MODEL) -> dict:
    """Blind ranking over every trained class. No requested word reaches this function.

    model_key names WHICH registry entry answers. A key that cannot be loaded reports its own
    failure; it is never silently answered by the default model.
    """
    started = time.perf_counter_ns()
    spec = MODELS.get(model_key)
    if spec is None:
        return {"schema": ADAPTER_SCHEMA, "model_key": model_key,
                "model_status": "UNKNOWN_MODEL_KEY", "ranking": None, "is_a_verdict": False,
                "accepted": None, "execution_assessment": "UNCONFIGURED",
                "detail": "no such model in the registry; the default was NOT substituted"}
    c = _load(model_key)
    base = {"schema": ADAPTER_SCHEMA, "model_id": spec["model_id"], "model_key": model_key,
            "checkpoint_sha256": spec["sha256"], "model_status": c["status"],
            "status_label": spec["status_label"], "role": spec["role"],
            "is_a_verdict": False, "accepted": None, "execution_assessment": "UNCONFIGURED"}
    if c["status"] != "READY":
        base.update(detail=c.get("detail"), ranking=None,
                    elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
        return base
    try:
        clip, why, stats = _channels(obs)
        if clip is None:
            base.update(model_status="INSUFFICIENT_OBSERVATION", detail=why, ranking=None,
                        support=dict(stats, meaning="the observation was not sufficient to run "
                                                    "the model; a capture fact, not a judgement "
                                                    "about the sign"),
                        elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
            return base
        import torch
        from pilot.data import collate
        x, pad, _ = collate([dict(clip, label=0)])
        with torch.inference_mode():
            logit = c["model"](torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        base.update(model_status="INFERENCE_FAILED", detail=f"{type(exc).__name__}", ranking=None,
                    elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
        return base

    classes = c["classes"]
    order = np.argsort(-logit)
    full = [{"label": classes[int(i)], "score": round(float(logit[int(i)]), 5)}
            for i in order[:3]]
    idx = {classes[i]: i for i in range(len(classes))}
    project_words = spec["project_words"]
    restricted = sorted(
        ({"word_id": w, "label": lab, "score": round(float(logit[idx[lab]]), 5)}
         for w, lab in project_words.items() if lab in idx),
        key=lambda r: -r["score"])
    scopes = {
        "common5": _scoped(logit, classes, COMMON_5, "common5",
                           "the five words every registered model knows; the only scope where "
                           "two models answer the same question"),
        "project18": _scoped(logit, classes, project_words, "project18",
                             "the project words THIS model was trained on - five for the base "
                             "model, eighteen for the candidate"),
        "full": {"scope": "full", "candidate_count": len(classes),
                 "top1": full[0]["label"], "top3": full,
                 "meaning": "every word this model was trained on: 80 for the base model, 93 for "
                            "the candidate",
                 "not_comparable_across_scopes": "80 candidates and 93 candidates are different "
                                                 "tasks and are never compared as equals"},
    }
    base.update(
        elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2),
        checkpoint_step=c.get("checkpoint_step"), model_version=c.get("model_version"),
        scopes=scopes,
        ranking={
            "over_all_trained_classes": {
                "candidate_count": len(classes), "top1": full[0]["label"], "top3": full,
                "meaning": "the model's own answer over every word it was trained on; it may name "
                           "a word outside this project's 19"},
            "restricted_to_project_words": {
                "candidate_count": len(restricted), "top1": restricted[0]["word_id"]
                if restricted else None, "order": restricted,
                "meaning": "the same logits read over only the project words this checkpoint "
                           "knows. A restricted-candidate reading, not a separate model, and an "
                           "easier task than the full ranking"},
        },
        support=dict(stats, frames_used=int(clip["frames"]),
                     trained_class_count=len(classes),
                     project_words_in_model=sorted(project_words),
                     meaning="span_frames is the distance between the first and last frame that "
                             "showed a hand; observed_hand_frames is how many of those actually "
                             "carried one"),
        # the old note hard-coded "the 14 project words this checkpoint never saw", which is a
        # fact about the BASE model only. Coverage is a property of whichever model answered.
        note="a research candidate. Not a correctness verdict, not a score, and not evidence "
             f"about the project words this model was never trained on "
             f"({len(project_words)} of the project's 19 are covered here; the rest are reported "
             f"NOT_IN_MODEL and are never scored).")
    return base


def word_coverage(word_id: str | None, model_key: str = DEFAULT_MODEL) -> dict:
    """Whether a requested word is one THIS model could name at all.

    Coverage is per model. A word the named model never learned is NOT_IN_MODEL, and no other
    model's coverage is borrowed to make it look covered.
    """
    if not word_id:
        return {"status": "NO_REQUESTED_WORD", "model_key": model_key}
    spec = MODELS.get(model_key)
    if spec is None:
        return {"status": "UNKNOWN_MODEL_KEY", "model_key": model_key, "word_id": word_id}
    words = spec["project_words"]
    if word_id in words:
        return {"status": "IN_MODEL", "word_id": word_id, "label": words[word_id],
                "model_key": model_key}
    return {"status": "NOT_IN_MODEL", "word_id": word_id, "model_key": model_key,
            "meaning": "this model was never trained on this word, so its ranking says nothing "
                       "about it, and no other model's coverage is used in its place"}
