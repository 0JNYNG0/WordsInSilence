"""functional_23: a second opt-in candidate in the lab, and the identity that travels with it.

No training, no participant replay, no camera, no server, no threshold change. The registry, the
stored-result contract and the report are exercised with the real registry and synthetic trials.

Run: py -3.13 tests/test_r20_visibility_candidate.py
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from wis_lab.service import neural as N  # noqa: E402
from wis_lab.service import review_log  # noqa: E402

NEW = "union93_visibility_v1"
OLD = "union93_candidate_v1"
NEW_SHA = "258f23e1ad515a25b50e6affac8ac4e284ba8f6ae6fac3372c1e7a31ea6848a8"
OLD_SHA = "e7c9c31a9c747e2c12a9a56695d2e4e9dbec9c372c8e2a3f209d00fa4c25dced"

FAILS = []


def ok(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ 1. the copied checkpoint
def test_copied_checkpoint():
    import hashlib
    d = os.path.join(ROOT, "runs", "visibility_candidate_v1")
    ck = os.path.join(d, "visibility_final.pt")
    ok(os.path.isfile(ck), "the reviewed augmentation checkpoint sits in a new run directory")
    got = hashlib.sha256(open(ck, "rb").read()).hexdigest()
    ok(got == NEW_SHA, "and it is byte-identical to the experiment's own file")

    prov = json.load(open(os.path.join(d, "provenance.json"), encoding="utf-8"))
    ok(prov["checkpoint"]["sha256"] == NEW_SHA
       and prov["source_experiment"]["sha256"] == NEW_SHA,
       "the provenance names the exact source file and hash")
    ok(prov["source_experiment"]["arm"] == "extra_training_plus_reviewed_missingness"
       and prov["source_experiment"]["matched_control_sha256"].startswith("9f2eb842"),
       "and which arm it is, beside its matched control")
    ok(prov["parent"]["sha256"] == OLD_SHA,
       "it also names the parent checkpoint it was continued from")
    ok(prov["promotion"].startswith("none"), "and states that nothing was promoted")
    rep = prov["development_replay_summary_only"]
    ok(rep["is_not"].startswith("a new test"),
       "the reused development replay is recorded under what it is NOT")
    ok(rep["project18"]["augmented"] == "17/18" and rep["full93"]["augmented"] == "17/18"
       and rep["project18"] is not rep["full93"],
       "both scopes are recorded, and separately, even where they agree")
    ok(rep["still_wrong"] and rep["no_lost_previously_correct_words"] is True,
       "what is still wrong is recorded beside what improved")
    ok("no participant session" in rep["not_opened_here"],
       "and it states that no participant session was opened here")

    # older models are preserved
    for rel in ("runs/union93_candidate_v1/union93_final.pt",
                "runs/union93_candidate_v1/union93_initial.pt"):
        ok(os.path.isfile(os.path.join(ROOT, rel)), f"{rel} is still there")


# ------------------------------------------------------------------ 2. the registry
def test_registry():
    # 2026-09-26: one opt-in TEST entry (the current reference TCN) was added deliberately
    ok(set(N.MODELS) == {"base", OLD, NEW, "tcn_control_83212868"},
       "four entries: the default, two opt-in candidates and one opt-in test connection")
    ok(N.DEFAULT_MODEL == "base", "the default is still base")
    ok(N.MODELS["base"]["sha256"] == N.CHECKPOINT_SHA256
       and N.MODELS[OLD]["sha256"] == OLD_SHA,
       "neither the default nor the old candidate moved")

    new = N.model_identity(NEW)
    old = N.model_identity(OLD)
    ok(new["checkpoint_sha256"] == NEW_SHA and new["model_status"] == "READY",
       "the new entry loads, at its own hash")
    ok(new["trained_class_count"] == old["trained_class_count"] == 93,
       "both candidates carry 93 trained classes")
    ok(sorted(new["project_words_in_model"]) == sorted(old["project_words_in_model"])
       and len(new["project_words_in_model"]) == 18,
       "the same 18 project words are covered")
    ok(new["feature_schema"] == old["feature_schema"] and new["in_channels"] == old["in_channels"],
       "and the same neural input adapter")
    ok(new["model_id"] != old["model_id"] and new["checkpoint_sha256"] != old["checkpoint_sha256"],
       "but they are distinct identities")

    cap = N.registry_capability()
    ok(cap["default_model_key"] == "base" and set(cap["models"]) == {"base", OLD, NEW, "tcn_control_83212868"},
       "the capability lists all four and keeps base as the default")
    ok("never" in cap["identities_are_never_pooled"],
       "and says two candidates are never added into one figure")


def test_no_silent_fallback():
    r = N.recognise(_obs(), model_key="union93_visibility_v9")
    ok(r["model_status"] == "UNKNOWN_MODEL_KEY" and r["ranking"] is None,
       "an unknown key reports itself and answers nothing")
    ok("NOT substituted" in r["detail"], "and says the default was not substituted")

    saved = dict(N.MODELS[NEW])
    N.MODELS[NEW] = dict(saved, sha256="0" * 64)
    N._CACHES.pop(NEW, None)
    try:
        out = N.recognise(_obs(), model_key=NEW)
        ok(out["model_status"] == "CHECKPOINT_HASH_MISMATCH", "a hash mismatch is a status")
        ok(out["ranking"] is None and out.get("accepted") is None,
           "it answers nothing and never becomes a verdict")
        ok(out["model_id"] == NEW, "and it is still reported as THAT model, not as the default")
    finally:
        N.MODELS[NEW] = saved
        N._CACHES.pop(NEW, None)


def _obs():
    """A small synthetic observation, enough for the adapter to have something to refuse."""
    import numpy as np

    from wis_lab.core.observation import (LEFT_HAND_SLICE, N_POINTS, Observation, POSE_SLICE,
                                          RIGHT_HAND_SLICE)
    t = 12
    xy = np.full((t, N_POINTS, 2), np.nan, dtype=np.float32)
    observed = np.zeros((t, N_POINTS), dtype=bool)
    usable = np.zeros((t, N_POINTS), dtype=bool)
    rng = np.arange(t, dtype=np.float32)
    xy[:, POSE_SLICE, 0] = (100.0 + rng)[:, None]
    xy[:, POSE_SLICE, 1] = 200.0
    xy[:, 11] = np.stack([80.0 + rng, np.full(t, 150.0)], axis=1)
    xy[:, 12] = np.stack([180.0 + rng, np.full(t, 150.0)], axis=1)
    observed[:, POSE_SLICE] = True
    usable[:, POSE_SLICE] = True
    for sl, col in ((LEFT_HAND_SLICE, 0), (RIGHT_HAND_SLICE, 1)):
        for f in range(t):
            xy[f, sl] = np.stack([np.arange(21, dtype=np.float32) + 10.0 * col + f,
                                  np.arange(21, dtype=np.float32) + 400.0], axis=1)
            observed[f, sl] = True
            usable[f, sl] = True
    return Observation(xy_px=xy, timestamp_us=(np.arange(t, dtype=np.int64) + 1) * 33_000,
                       frame_id=np.arange(t, dtype=np.int64), observed=observed, usable=usable,
                       conf=np.full((t, N_POINTS), 0.9, dtype=np.float32),
                       hand_score=np.full((t, 2), 0.8, dtype=np.float32), sidecar={})


# ------------------------------------------------------------------ 3. off by default
def test_off_by_default():
    app = open(os.path.join(ROOT, "src", "wis_lab", "service", "app.py"), encoding="utf-8").read()
    ok("if not key:" in app and '"opted_in": False' in app,
       "a trial that names no key is recorded as not opted in")
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    ok("if (!box || !box.checked) return null;" in js,
       "and the lab sends no key at all while the box is unchecked")
    html = open(os.path.join(ROOT, "web", "index.html"), encoding="utf-8").read()
    box = html.split('id="opt-research"')[1].split(">")[0]
    ok("checked" not in box, "the opt-in checkbox is not checked by default")
    ok('id="opt-research-model"' in html, "there is an explicit model selector beside it")
    for key in (OLD, NEW):
        ok(f'value="{key}"' in html, f"{key} can be chosen explicitly")
    ok("비사용 손 관측 보완 후보" in html,
       "the new candidate is named for what it is, not as automatic hand detection")
    for wrong in ("자동", "자동 감지", "통과", "합격", "정확도 향상"):
        ok(wrong not in html.split('id="opt-research-model"')[1].split("</select>")[0],
           f"the selector does not claim '{wrong}'")


def test_ui_sends_the_chosen_key():
    js = open(os.path.join(ROOT, "web", "app.js"), encoding="utf-8").read()
    ok("RESEARCH_MODEL_KEY" not in js, "the single hard-coded key is gone")
    ok(js.count("research_model_key: researchModelKey()") == 2,
       "both the trial and the replay send whichever key was chosen")
    ok('const sel = $("opt-research-model");' in js and "sel.value" in js,
       "and the key comes from the selector, not from a constant")
    ok("if (sel) sel.disabled = !!locked;" in js,
       "the selector is locked for the length of a take, like the checkbox")
    ok('RESEARCH_LABEL[id.model_id] || id.model_id' in js,
       "a cohort is labelled by the identity it actually carries")
    ok("한 분모로 합치지 않습니다" in js,
       "and the screen says different candidates are not merged into one denominator")


# ------------------------------------------------------------------ 4. identity travels
def test_identity_is_pinned_and_shown():
    app = open(os.path.join(ROOT, "src", "wis_lab", "service", "app.py"), encoding="utf-8").read()
    ok('"fixed_at": "trial_creation"' in app and '"no_fallback_to_default": True' in app,
       "the chosen model is fixed when the trial is created")
    ok("identity_fixed_at_trial_creation" in app,
       "and that identity is written beside the result")

    choice = _research_choice(NEW)
    ok(choice["opted_in"] and choice["model_key"] == NEW
       and choice["checkpoint_sha256"] == NEW_SHA,
       "choosing the new candidate pins its key AND its hash")
    ok(_research_choice(OLD)["checkpoint_sha256"] == OLD_SHA,
       "choosing the old one pins the old hash")
    ok(_research_choice(None)["opted_in"] is False,
       "choosing nothing opts in to nothing")


def _research_choice(key):
    from wis_lab.service.app import _research_choice as f
    return f(key)


def test_report_keeps_identities_apart():
    """Two candidates in one session must never land in one denominator."""
    import copy

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_r16_candidate_cohorts as C

    tmp = tempfile.mkdtemp(prefix="f23rep_")
    try:
        run = os.path.join(tmp, "run")
        specs = [("v1", "NA", "NA", "NA", "NA", OLD_SHA, None),
                 ("v2", "NA", "NA", "NA", "NA", NEW_SHA, None)]
        ids, states, meta = C.build(run, specs)
        for tid, sha in (("v1", OLD_SHA), ("v2", NEW_SHA)):
            meta[tid]["research_model"]["model_id"] = (
                OLD if sha == OLD_SHA else NEW)
            meta[tid]["research_model"]["checkpoint_sha256"] = sha
            p = os.path.join(run, "trials", tid + ".result.json")
            data = json.load(open(p, encoding="utf-8"))
            data["research_model"] = copy.deepcopy(meta[tid]["research_model"])
            data["research_comparison"]["model_id"] = meta[tid]["research_model"]["model_id"]
            data["research_comparison"]["checkpoint_sha256"] = sha
            data["research_comparison"]["identity_fixed_at_trial_creation"] = copy.deepcopy(
                meta[tid]["research_model"])
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(data, fh, ensure_ascii=False)

        rep = review_log.report(run, ids, states, trial_meta=meta)
        groups = rep["cohorts"]["groups"]
        ok(len(groups) == 4, "two candidates x two scopes are four separate cohorts")
        by_model = {}
        for g in groups.values():
            by_model.setdefault(g["identity"]["model_id"], []).append(g)
        ok(set(by_model) == {OLD, NEW}, "each cohort carries its own model id")
        ok(all(g["n_supported"] == 1 for g in groups.values()),
           "and its own denominator of one attempt; nothing is pooled into two")
        hashes = {g["identity"]["checkpoint_sha256"] for g in groups.values()}
        ok(hashes == {OLD_SHA, NEW_SHA}, "both checkpoint hashes are reported, apart")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 5. still not a verdict
def test_never_a_verdict():
    r = N.recognise(_obs(), model_key=NEW)
    ok(r["is_a_verdict"] is False and r["accepted"] is None
       and r["execution_assessment"] == "UNCONFIGURED",
       "the new candidate is not a verdict: accepted null, execution UNCONFIGURED")
    ok(r["schema"] == N.ADAPTER_SCHEMA and r["model_id"] == NEW,
       "the result carries the adapter schema and the model that answered")
    for name, sc in (r.get("scopes") or {}).items():
        ok("candidate_count" in sc and "not_comparable_across_scopes" in sc,
           f"the {name} scope states its candidate count and that scopes are different tasks")

    import inspect
    ok(list(inspect.signature(N.recognise).parameters) == ["obs", "model_key"],
       "recognise takes an observation and a model key - a requested word cannot be passed in")
    ok(not any("mask" in k for spec in N.MODELS.values() for k in spec),
       "no registry entry carries any live-masking behaviour")
    ok(all(spec.get("status_label") == ("시험용" if k == "tcn_control_83212868" else "검증 중")
           for k, spec in N.MODELS.items() if k != "base"),
       "candidates are labelled under validation and the test entry as test-only, never as passed")


if __name__ == "__main__":
    test_copied_checkpoint()
    test_registry()
    test_no_silent_fallback()
    test_off_by_default()
    test_ui_sends_the_chosen_key()
    test_identity_is_pinned_and_shown()
    test_report_keeps_identities_apart()
    test_never_a_verdict()
    print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAIL: " + ", ".join(FAILS)))
    sys.exit(1 if FAILS else 0)
