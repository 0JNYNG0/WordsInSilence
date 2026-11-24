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
import json
import os
import sys
import threading
import time

import numpy as np

from ..cli import ROOT
from .review_log import FULL_CLASS_SET_SCHEMA, full_class_set_sha256

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
    # 2026-09-26: the reference TCN every recent post-hoc evaluation used, offered as an opt-in
    # TEST entry only. Loaded through the same generic loader and _channels path as the entries
    # above; pinned by hash, and a missing/mismatched file is reported, never replaced.
    "tcn_control_83212868": {
        "model_id": "tcn_control_83212868",
        "path": os.path.join(ROOT, "runs", "partial_encoder_pair_v1", "checkpoint_control_final.pt"),
        "sha256": "83212868e83796a26d8b8d99eb441b39b4720bbde3f1aa06afdea8c166fe2e91",
        "project_words": PROJECT_18,
        "role": "an opt-in TEST connection of the current reference TCN (93 classes). NOT the "
                "operating default, NOT promoted, NOT a verdict, and not evidence about a new person",
        "status_label": "시험용",
    },
}
# 2026-10-03: the SAME frozen832 checkpoint as tcn_control_83212868, read twice on the same
# observation - once on the original coordinates and once after hand_gap.restore filled isolated
# one-frame whole-hand gaps inside the original active span. Not a new model and not a claim of
# better accuracy: an input-processing comparison under a model_id of its own, so a report never
# pools the transformed answers with the raw entry's.
HANDGAP_KEY = "tcn_control_83212868_handgap"
HANDGAP_TRANSFORM_ID = "wis_hand_gap_restore_v1"
HANDGAP_MODULE = __package__ + ".hand_gap"
# the frozen candidate source (FROZEN_PLAN.candidate_source_sha256); a different file is refused
HANDGAP_SOURCE_SHA256 = "6ae29cea6ae030d8567a9c79749cf8717738a1b2c2b73eaa93d4b38fd539ac07"
HANDGAP_SCHEMA = "wis_hand_gap_comparison_v1"
MODELS[HANDGAP_KEY] = dict(
    MODELS["tcn_control_83212868"],
    model_id=HANDGAP_KEY,
    role="an opt-in TEST comparison: the same tcn_control_83212868 weights read on the original "
         "coordinates and on a copy where an isolated one-frame whole-hand gap was estimated from "
         "its neighbours. NOT a new trained model, NOT promoted, NOT a verdict, and not evidence "
         "of better accuracy",
    status_label="시험용",
    input_transform=HANDGAP_TRANSFORM_ID,
)
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


def _class_list(c: dict):
    """The loaded checkpoint's classes in index order, or None if they are not a clean label list."""
    classes = c.get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        return None
    labels = list(classes)
    if any(not isinstance(x, str) or not x for x in labels) or len(set(labels)) != len(labels):
        return None
    return labels


def trained_class_set(model_key: str = DEFAULT_MODEL) -> dict:
    """The full class list of the VERIFIED loaded checkpoint, plus the project-id -> label mapping.

    Frozen into a trial at creation so a later report can score the full-scope answer (which is a
    model label) against the participant's confirmed project word. Only a model that loaded and
    passed its hash check yields a list; anything else is UNAVAILABLE and carries no labels, so no
    coverage is ever implied for it.
    """
    spec = MODELS.get(model_key)
    base = {"schema": FULL_CLASS_SET_SCHEMA, "model_key": model_key,
            "model_id": spec["model_id"] if spec else None,
            "checkpoint_sha256": spec["sha256"] if spec else None,
            "labels": None, "count": None, "membership_sha256": None,
            "project_word_to_model_label": None}
    if spec is None:
        return dict(base, status="UNAVAILABLE", reason="UNKNOWN_MODEL_KEY")
    c = _load(model_key)
    if c["status"] != "READY":
        return dict(base, status="UNAVAILABLE", reason="MODEL_" + str(c["status"]))
    labels = _class_list(c)
    if labels is None:
        return dict(base, status="UNAVAILABLE", reason="CLASS_LIST_MALFORMED")
    present = set(labels)
    return dict(
        base, status="AVAILABLE", reason=None,
        labels=labels, label_order="checkpoint_class_index",
        count=len(labels), membership_sha256=full_class_set_sha256(labels),
        membership_hash_rule="sha256 of the sorted labels as compact JSON",
        project_word_to_model_label={w: lab for w, lab in sorted(spec["project_words"].items())
                                     if lab in present},
        project_words_without_class=sorted(w for w, lab in spec["project_words"].items()
                                           if lab not in present),
        source="the classes stored in the hash-verified checkpoint that was loaded")


def model_identity(model_key: str = DEFAULT_MODEL) -> dict:
    """Exactly which model answered, fixed per trial and written beside the result."""
    spec = MODELS.get(model_key)
    if spec is None:
        return {"model_key": model_key, "status": "UNKNOWN_MODEL_KEY"}
    c = _load(model_key)
    return {
        "trained_class_set": trained_class_set(model_key),
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
        # only an entry that transforms its input carries this key; every other identity is as before
        **({"input_transform": handgap_identity()} if spec.get("input_transform") else {}),
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


def _channels(obs, transform=None) -> tuple:
    """Observation -> the pilot's 280-channel contract, via the same 75->67 assignment.

    transform, when given, sees COPIES of the native arrays of the span the original gates already
    fixed and accepted, and returns replacement (xy, present) for that span only. The gates below
    never see its output, and the span is not recomputed from it.
    """
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
    if transform is not None:
        xy_s, pr_s = transform(nat[lo:hi + 1].copy(), present[lo:hi + 1].copy(),
                               ts[lo:hi + 1].copy(), lo, hi)
        return build_channels(xy_s, pr_s, ts[lo:hi + 1]), None, stats
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
    failure; it is never silently answered by the default model. An entry with an input_transform
    answers twice on the same observation; see _recognise_handgap.
    """
    spec = MODELS.get(model_key)
    if spec is not None and spec.get("input_transform") == HANDGAP_TRANSFORM_ID:
        return _recognise_handgap(obs, model_key)
    return _infer(obs, model_key)[0]


class _TransformError(Exception):
    """An input transform that could not run. Reported as its own status, never as the original."""

    def __init__(self, status: str, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _infer(obs, model_key: str, transform=None):
    """-> (result, logits or None, class labels or None). The body recognise always had."""
    started = time.perf_counter_ns()
    spec = MODELS.get(model_key)
    if spec is None:
        return {"schema": ADAPTER_SCHEMA, "model_key": model_key,
                "model_status": "UNKNOWN_MODEL_KEY", "ranking": None, "is_a_verdict": False,
                "accepted": None, "execution_assessment": "UNCONFIGURED",
                "detail": "no such model in the registry; the default was NOT substituted"}, \
            None, None
    c = _load(model_key)
    base = {"schema": ADAPTER_SCHEMA, "model_id": spec["model_id"], "model_key": model_key,
            "checkpoint_sha256": spec["sha256"], "model_status": c["status"],
            "status_label": spec["status_label"], "role": spec["role"],
            "is_a_verdict": False, "accepted": None, "execution_assessment": "UNCONFIGURED"}
    if c["status"] != "READY":
        base.update(detail=c.get("detail"), ranking=None,
                    elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
        return base, None, None
    try:
        clip, why, stats = _channels(obs) if transform is None else _channels(obs, transform)
        if clip is None:
            base.update(model_status="INSUFFICIENT_OBSERVATION", detail=why, ranking=None,
                        support=dict(stats, meaning="the observation was not sufficient to run "
                                                    "the model; a capture fact, not a judgement "
                                                    "about the sign"),
                        elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
            return base, None, None
        import torch
        from pilot.data import collate
        x, pad, _ = collate([dict(clip, label=0)])
        with torch.inference_mode():
            logit = c["model"](torch.from_numpy(x), torch.from_numpy(pad))[0].numpy()
    except _TransformError as exc:                        # the transform's own failure, by name
        base.update(model_status=exc.status, detail=exc.detail, ranking=None,
                    elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
        return base, None, None
    except Exception as exc:                              # noqa: BLE001 - reported, not raised
        base.update(model_status="INFERENCE_FAILED", detail=f"{type(exc).__name__}", ranking=None,
                    elapsed_ms=round((time.perf_counter_ns() - started) / 1e6, 2))
        return base, None, None

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
                 # which class set this answer was over, so a report can check it against the
                 # set frozen at trial creation. An identity record only; the ranking is unchanged.
                 "class_set_sha256": (full_class_set_sha256(_class_list(c))
                                      if _class_list(c) else None),
                 "labels_are": "model class labels, not project word ids",
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
    return base, logit, classes


# ---------------------------------------------------------- short whole-hand gap comparison
_HANDGAP = {"module": None, "status": None, "detail": None, "source_sha256": None}
_HANDGAP_LOCK = threading.Lock()


def _handgap_module() -> dict:
    """Import hand_gap once, on first use, and pin it by source hash. Failure is a status.

    Nothing here falls back: an import error, a missing restore or a different source file leaves
    the repaired arm unavailable with its reason, and the original arm stays separately available.
    """
    if _HANDGAP["status"] is not None:
        return _HANDGAP
    with _HANDGAP_LOCK:
        if _HANDGAP["status"] is not None:
            return _HANDGAP
        try:
            import importlib
            mod = importlib.import_module(HANDGAP_MODULE)
            digest = _sha256(mod.__file__)
        except Exception as exc:                          # noqa: BLE001 - reported, not raised
            _HANDGAP.update(module=None, status="TRANSFORM_IMPORT_FAILED",
                            detail=f"{type(exc).__name__}: {exc}", source_sha256=None)
            return _HANDGAP
        if digest != HANDGAP_SOURCE_SHA256:
            _HANDGAP.update(module=None, status="TRANSFORM_SOURCE_HASH_MISMATCH",
                            detail="the hand_gap source on disk is not the frozen candidate",
                            source_sha256=digest)
        elif not callable(getattr(mod, "restore", None)):
            _HANDGAP.update(module=None, status="TRANSFORM_RESTORE_MISSING",
                            detail="hand_gap has no callable restore", source_sha256=digest)
        else:
            _HANDGAP.update(module=mod, status="READY", detail=None, source_sha256=digest)
        return _HANDGAP


def handgap_identity() -> dict:
    """The transform identity frozen into a trial beside the model identity."""
    m = _handgap_module()
    return {"transform_id": HANDGAP_TRANSFORM_ID, "module": HANDGAP_MODULE, "function": "restore",
            "pinned_source_sha256": HANDGAP_SOURCE_SHA256,
            "loaded_source_sha256": m["source_sha256"],
            "status": m["status"], "detail": m["detail"],
            "rule": "per hand: all 21 joints absent at one interior frame, all 21 present at both "
                    "neighbours, neighbour span <= 66668 us -> time-weighted linear estimate",
            "applied_to": "a copy of the native 67-point arrays inside the original active span, "
                          "after the original support gates, before the unchanged "
                          "pilot.data.build_channels",
            "not_applied_to": "the observation, DTW, finger/position/movement feedback, "
                              "the support gates",
            "is_a_trained_model": False}


def _source_frame_ids(obs, t: int):
    """-> (ids or None, why). Frame ids are used only when they are a clean increasing integer row."""
    fid = getattr(obs, "frame_id", None)
    if fid is None:
        return None, "FRAME_ID_ABSENT"
    fid = np.asarray(fid)
    if fid.shape != (t,) or fid.dtype == np.bool_ or not np.issubdtype(fid.dtype, np.integer):
        return None, "FRAME_ID_MALFORMED"
    if t > 1 and not np.all(np.diff(fid) > 0):
        return None, "FRAME_ID_NOT_INCREASING"
    return fid, None


def _handgap_transform(obs, mod, holder: dict):
    """The callable _channels hands the span copies to. Writes provenance into holder only."""
    def transform(xy_s, pr_s, ts_s, lo, hi):
        try:
            out = mod.restore(xy_s, pr_s, ts_s)
        except Exception as exc:                          # noqa: BLE001 - reported, not raised
            raise _TransformError("INPUT_TRANSFORM_FAILED", f"{type(exc).__name__}: {exc}")
        try:
            return _provenance(out, lo, hi)
        except Exception as exc:                          # noqa: BLE001 - reported, not raised
            raise _TransformError("INPUT_TRANSFORM_FAILED",
                                  f"provenance {type(exc).__name__}: {exc}")

    def _provenance(out, lo, hi):
        est, obsd = np.asarray(out["estimated"], bool), np.asarray(out["observed"], bool)
        ts_src = np.asarray(obs.timestamp_us, dtype=np.int64)
        fids, fid_why = _source_frame_ids(obs, int(ts_src.shape[0]))
        sides = getattr(mod, "HANDS", {"left": slice(25, 46), "right": slice(46, 67)})
        fills, by_side, est_pts, obs_pts = [], {}, {}, {}
        for side, sl in sides.items():
            rows = [int(i) for i in np.flatnonzero(est[:, sl].any(axis=1))]
            by_side[side] = len(rows)
            est_pts[side] = int(est[:, sl].sum())
            obs_pts[side] = int(obsd[:, sl].sum())
            for i in rows:
                src = lo + i
                fills.append({
                    "side": side, "source_row": src, "span_row": i,
                    "source_frame_id": int(fids[src]) if fids is not None else None,
                    "timestamp_us": int(ts_src[src]),
                    "neighbour_source_rows": [src - 1, src + 1],
                    "neighbour_timestamps_us": [int(ts_src[src - 1]), int(ts_src[src + 1])],
                    "estimated_points": int(est[i, sl].sum()),
                    "kind": "estimated_not_observed"})
        fills.sort(key=lambda f: (f["source_row"], f["side"]))
        ident = bool(out.get("identity"))
        span = {"source_row_first": int(lo), "source_row_last": int(hi),
                "span_frames": int(hi - lo + 1),
                "timestamp_us_first": int(ts_src[lo]), "timestamp_us_last": int(ts_src[hi]),
                "source_frame_id_first": int(fids[lo]) if fids is not None else None,
                "source_frame_id_last": int(fids[hi]) if fids is not None else None}
        record = {"transform_id": HANDGAP_TRANSFORM_ID,
                  "source_sha256": _HANDGAP["source_sha256"], "span": span, "fills": fills}
        holder.update(
            status="IDENTITY_NO_FILL" if ident else "FILLED", identity=ident,
            source_active_span=span,
            source_index_kind="frame_id" if fids is not None else "source_row",
            source_frame_id_unused_reason=fid_why,
            filled_hand_frames=dict(by_side, total=sum(by_side.values())),
            hand_points={"observed": obs_pts, "estimated": est_pts,
                         "observed_total": sum(obs_pts.values()),
                         "estimated_total": sum(est_pts.values())},
            fills=fills,
            provenance_sha256=hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            coordinates_included=False)
        return out["xy"], out["present"]
    return transform


def _logit_list(logit):
    if logit is None:
        return None
    vals = [float(v) for v in np.asarray(logit).reshape(-1)]
    return vals if all(np.isfinite(vals)) else None


def _recognise_handgap(obs, model_key: str) -> dict:
    """The same weights, the same observation, two inputs: original and short-gap restored.

    The repaired answer is the outer block (its model_id is this entry's own); the original sits
    in hand_gap_comparison.original. Both pass the same original gates - _channels runs them on
    the untouched presence before any transform sees a copy - so an estimate can never turn an
    insufficient observation into an accepted one. Neither arm sees a requested word.
    """
    started = time.perf_counter_ns()
    original, raw_logit, raw_classes = _infer(obs, model_key)
    original["arm"] = "original_coordinates"
    original["input_transform"] = {"transform_id": "none"}

    holder = {}
    m = _handgap_module()
    if m["status"] == "READY":
        transform = _handgap_transform(obs, m["module"], holder)
    else:
        def transform(*_a):
            raise _TransformError("INPUT_TRANSFORM_UNAVAILABLE", f"{m['status']}: {m['detail']}")
    repaired, rep_logit, rep_classes = _infer(obs, model_key, transform)
    repaired["arm"] = "short_gap_restored"
    if holder:
        prov = holder
    elif repaired["model_status"] in ("INPUT_TRANSFORM_UNAVAILABLE", "INPUT_TRANSFORM_FAILED"):
        prov = {"status": "UNAVAILABLE", "reason": repaired["model_status"],
                "detail": repaired.get("detail"), "coordinates_included": False}
    else:
        # the shared gates (or the model) stopped before any transform ran
        prov = {"status": "NOT_ATTEMPTED", "reason": repaired["model_status"],
                "detail": repaired.get("detail"), "coordinates_included": False}
    repaired["input_transform"] = dict(handgap_identity(), provenance=prov)

    same_labels = raw_classes is not None and list(raw_classes) == list(rep_classes or [])
    raw_l, rep_l = _logit_list(raw_logit), _logit_list(rep_logit)
    repaired["hand_gap_comparison"] = {
        "schema": HANDGAP_SCHEMA,
        "original": original,
        "repaired": "the enclosing block: this research_comparison itself",
        "same_weights": {"model_key": model_key, "checkpoint_sha256": MODELS[model_key]["sha256"]},
        "logit_audit": {
            "label_order": list(raw_classes) if raw_classes is not None
            else (list(rep_classes) if rep_classes is not None else None),
            "label_order_is": "checkpoint_class_index",
            "same_label_order_both_arms": same_labels if rep_classes is not None else None,
            "original_coordinates": raw_l,
            "short_gap_restored": rep_l,
            "max_abs_difference": (max(abs(a - b) for a, b in zip(raw_l, rep_l))
                                   if raw_l and rep_l and same_labels else None),
            "precision": "unrounded float32 logits as float; the scopes are rounded to 5 places",
            "meaning": "for audit only; a non-finite or absent arm is null"},
        "elapsed_ms_both_arms": round((time.perf_counter_ns() - started) / 1e6, 2),
        "meaning": "one checkpoint, one observation, two input treatments. A comparison of input "
                   "processing, not a new trained model, not a correctness grading and not "
                   "evidence of better accuracy. Feedback and DTW use the original observation.",
    }
    return repaired


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
