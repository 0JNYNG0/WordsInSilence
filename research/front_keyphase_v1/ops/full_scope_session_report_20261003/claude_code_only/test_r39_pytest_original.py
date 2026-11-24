"""R39: the full trained-class research scope in the session report.

Synthetic fixtures only. Every case goes through review_log.report(), so the confirmation store,
the stored result files and the creation metadata are joined exactly as the session report does.
No camera, no checkpoint and no torch are needed.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from wis_lab.service import review_log  # noqa: E402

PROJECT = {"ALDA": "알다", "GAMSAHAMNIDA": "감사합니다", "NA": "나", "JOTA": "좋다",
           "MOREUDA": "모르다", "GAM": "감", "GONGBU": "공부"}
# deliberately not 93: the scope must be generic over whatever the model was trained on
LABELS_A = list(PROJECT.values()) + [f"기타{i}" for i in range(5)]           # 12 classes
LABELS_B = LABELS_A + ["추가1", "추가2"]                                       # 14 classes
CK_A, CK_B = "a" * 64, "b" * 64


def make_creation(model_id="cand_a", ck=CK_A, labels=None, mapping=None, schema="fs_v1",
                  class_count=None, **ev_over):
    labels = list(LABELS_A if labels is None else labels)
    present = set(labels)
    if mapping is None:
        mapping = {w: lab for w, lab in PROJECT.items() if lab in present}
    ev = {"schema": review_log.FULL_CLASS_SET_SCHEMA, "status": "AVAILABLE", "reason": None,
          "model_key": model_id, "model_id": model_id, "checkpoint_sha256": ck,
          "labels": labels, "label_order": "checkpoint_class_index", "count": len(labels),
          "membership_sha256": review_log.full_class_set_sha256(labels),
          "project_word_to_model_label": mapping}
    ev.update(ev_over)
    return {"opted_in": True, "fixed_at": "trial_creation", "model_key": model_id,
            "model_id": model_id, "checkpoint_sha256": ck, "model_status": "READY",
            "feature_schema": schema,
            "trained_class_count": len(labels) if class_count is None else class_count,
            "project_words_in_model": sorted(PROJECT), "trained_class_set": ev}


def legacy_creation(model_id="cand_a", ck=CK_A):
    c = make_creation(model_id, ck)
    del c["trained_class_set"]
    return c


def make_result(creation, full_top1, p18_top1=None, requested="ALDA", status="READY",
                answered_model_id=None, full_count=None, full_sha="auto",
                saved_research_model=True, identity_copy=True):
    ev = creation.get("trained_class_set") or {}
    labels = ev.get("labels") or LABELS_A
    words = sorted(PROJECT)
    p18_top1 = p18_top1 or words[0]
    p18 = [{"word_id": w, "label": PROJECT[w], "score": 1.0 if w == p18_top1 else 0.0}
           for w in words]
    c5 = [r for r in p18 if r["word_id"] in review_log.COMMON_5]
    rc = {"model_id": answered_model_id or creation["model_id"],
          "checkpoint_sha256": creation["checkpoint_sha256"], "model_status": status,
          "requested_word_coverage": {"status": "IN_MODEL", "word_id": requested}}
    if identity_copy:
        rc["identity_fixed_at_trial_creation"] = {
            k: creation.get(k) for k in ("model_key", "model_id", "checkpoint_sha256",
                                         "feature_schema", "trained_class_count",
                                         "project_words_in_model", "trained_class_set")}
    if status == "READY":
        rc["scopes"] = {
            "common5": {"scope": "common5", "candidate_count": len(c5),
                        "top1": max(c5, key=lambda r: r["score"])["word_id"], "order": c5},
            "project18": {"scope": "project18", "candidate_count": len(p18),
                          "top1": p18_top1, "order": p18},
            "full": {"scope": "full",
                     "candidate_count": len(labels) if full_count is None else full_count,
                     "top1": full_top1, "top3": [{"label": full_top1, "score": 1.0}],
                     "class_set_sha256": (review_log.full_class_set_sha256(labels)
                                          if full_sha == "auto" else full_sha)}}
    res = {"observation_status": "OK", "requested_word_id": requested,
           "task": {"requested_word_id": requested},
           "ranking": [{"word_id": w, "status": "OK", "distance": 0.1 if w == p18_top1 else 1.0}
                       for w in words],
           "learned_recognition": {"model_status": "READY",
                                   "ranking": {"restricted_to_project_words": {"top1": "ALDA"}}},
           "research_comparison": rc,
           "timing": {"server_end_candidate_to_result_ms": 10.0,
                      "timing_scope_id": "server_end_candidate_to_assembled_result_v2"}}
    if saved_research_model:
        res["research_model"] = creation
    return res


def run(tmp_path, trials, with_meta=True):
    """trials: dicts with tid, creation, result ('none' = no file, 'torn' = unreadable), actual."""
    run_dir = str(tmp_path)
    os.makedirs(os.path.join(run_dir, "trials"), exist_ok=True)
    meta = {}
    for t in trials:
        path = os.path.join(run_dir, "trials", t["tid"] + ".result.json")
        if t.get("result") == "torn":
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{")
        elif t.get("result") not in (None, "none"):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(t["result"], fh, ensure_ascii=False)
        if t.get("actual") is not None:
            out = review_log.submit(run_dir, "ses_t", t["tid"], t["actual"], "yes")
            assert out["stored"] is True
        if with_meta and t.get("creation") is not None:
            meta[t["tid"]] = {"research_model": t["creation"]}
    order = [t["tid"] for t in trials]
    return review_log.report(run_dir, order, {tid: "FEEDBACK" for tid in order},
                             trial_meta=meta)


def full_groups(rep):
    return list(rep["full_scope"]["groups"].values())


def only_group(rep):
    gs = full_groups(rep)
    assert len(gs) == 1, gs
    return gs[0]


def cohort(rep, scope):
    gs = [g for g in rep["cohorts"]["groups"].values() if g["scope"] == scope]
    assert len(gs) == 1, gs
    return gs[0]


# ------------------------------------------------------------------------------------- scoring
def test_full_wrong_while_restricted18_right(tmp_path):
    c = make_creation()
    rep = run(tmp_path, [{"tid": "t1", "creation": c, "actual": "GAM",
                          "result": make_result(c, full_top1="공부", p18_top1="GAM")}])
    g = only_group(rep)
    cand = g["per_method"]["candidate"]
    assert (cand["hits"], cand["answered"], cand["n"]) == (0, 1, 1)
    assert g["rows"][0]["actual_model_label"] == "감"
    assert g["rows"][0]["pick_label"] == "공부" and g["rows"][0]["hit"] is False
    assert g["methods"] == ["candidate"]                 # no DTW / base model in this scope
    # the restricted task is untouched and still separate
    p18 = cohort(rep, "project18")
    assert p18["per_method"]["candidate"]["hits"] == 1
    assert rep["counts"]["confirmed_outside_common5"] == 1


def test_full_correct_and_non_project_label_is_a_miss(tmp_path):
    c = make_creation()
    rep = run(tmp_path, [
        {"tid": "t1", "creation": c, "actual": "ALDA", "result": make_result(c, "알다")},
        {"tid": "t2", "creation": c, "actual": "NA", "result": make_result(c, "기타3")},
    ])
    cand = only_group(rep)["per_method"]["candidate"]
    assert (cand["hits"], cand["answered"], cand["n"]) == (1, 2, 2)
    assert only_group(rep)["identity"]["class_count"] == len(LABELS_A)


def test_word_id_in_place_of_label_is_not_scored_as_a_label(tmp_path):
    c = make_creation()
    rep = run(tmp_path, [{"tid": "t1", "creation": c, "actual": "GAM",
                          "result": make_result(c, full_top1="GAM")}])
    cand = only_group(rep)["per_method"]["candidate"]
    assert cand["n"] == 1 and cand["answered"] == 0
    assert cand["unanswered"] == {"CANDIDATE_PICK_OUTSIDE_SET": 1}


def test_failures_stay_in_the_denominator(tmp_path):
    c = make_creation()
    rep = run(tmp_path, [
        {"tid": "t1", "creation": c, "actual": "ALDA", "result": "none"},
        {"tid": "t2", "creation": c, "actual": "ALDA", "result": "torn"},
        {"tid": "t3", "creation": c, "actual": "ALDA",
         "result": make_result(c, "알다", status="INFERENCE_FAILED")},
        {"tid": "t4", "creation": c, "actual": "ALDA",
         "result": make_result(c, "알다", answered_model_id="someone_else")},
        {"tid": "t5", "creation": c, "actual": "ALDA",
         "result": make_result(c, "알다", full_count=len(LABELS_A) - 1)},
        {"tid": "t6", "creation": c, "actual": "ALDA",
         "result": make_result(c, "알다", full_sha="0" * 64)},
        {"tid": "t7", "creation": c, "actual": "ALDA", "result": make_result(c, "알다")},
    ])
    cand = only_group(rep)["per_method"]["candidate"]
    assert cand["n"] == 7 and cand["answered"] == 1 and cand["hits"] == 1
    assert cand["match_over_all_supported"]["display"] == "1/7"
    assert cand["unanswered"] == {
        "NO_RESULT_FILE": 1, "RESULT_UNREADABLE:JSONDecodeError": 1,
        "CANDIDATE_INFERENCE_FAILED": 1, "CANDIDATE_IDENTITY_MISMATCH": 1,
        "FULL_ANSWER_COUNT_MISMATCH": 1, "FULL_ANSWER_CLASS_SET_MISMATCH": 1}


def test_unknown_unconfirmed_and_unsupported_are_not_scored(tmp_path):
    c = make_creation()
    rep = run(tmp_path, [
        {"tid": "t1", "creation": c, "actual": "UNKNOWN", "result": make_result(c, "알다")},
        {"tid": "t2", "creation": c, "actual": None, "result": make_result(c, "알다")},
        {"tid": "t3", "creation": c, "actual": "KKEUDEOGIDA", "result": make_result(c, "알다")},
        {"tid": "t4", "creation": c, "actual": "ALDA", "result": make_result(c, "나")},
    ])
    g = only_group(rep)
    assert g["n_supported"] == 1
    assert g["per_method"]["candidate"]["match_over_all_supported"]["display"] == "0/1"
    assert g["excluded_counts"] == {"CONFIRMED_UNKNOWN": 1, "NOT_CONFIRMED": 1,
                                    "ACTUAL_WORD_NOT_IN_MODEL_MAPPING": 1}
    assert g["unique_trials"] == 4


def test_requested_word_change_leaves_the_tally(tmp_path):
    c = make_creation()
    trials = [{"tid": "t1", "creation": c, "actual": "GAM",
               "result": make_result(c, "감", requested="GAM")},
              {"tid": "t2", "creation": c, "actual": "ALDA",
               "result": make_result(c, "감", requested="ALDA")}]
    before = only_group(run(tmp_path, trials))
    # rewrite only what was asked for; the confirmations and answers stay as they were
    for t in trials:
        p = os.path.join(str(tmp_path), "trials", t["tid"] + ".result.json")
        with open(p, encoding="utf-8") as fh:
            res = json.load(fh)
        res["requested_word_id"] = "MOREUDA"
        res["task"]["requested_word_id"] = "MOREUDA"
        res["research_comparison"]["requested_word_coverage"]["word_id"] = "MOREUDA"
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False)
    order = [t["tid"] for t in trials]
    after = only_group(review_log.report(
        str(tmp_path), order, trial_meta={t["tid"]: {"research_model": c} for t in trials}))
    assert after["per_method"] == before["per_method"]
    assert after["rows"] == before["rows"]


def test_opt_out_is_never_scored_even_with_a_stray_block(tmp_path):
    c = make_creation()
    out = {"opted_in": False, "model_key": None}
    res = make_result(c, "알다")
    res["research_model"] = out
    rep = run(tmp_path, [{"tid": "t1", "creation": out, "actual": "ALDA", "result": res}])
    fs = rep["full_scope"]
    assert fs["groups"] == {} and fs["unavailable"] == []
    assert fs["counts"]["opted_out"] == 1 and fs["status"] == "UNAVAILABLE"


# ------------------------------------------------------------------------ evidence fails closed
# (arguments for make_creation, then fields overwritten inside the frozen evidence, reason)
BAD_EVIDENCE = [
    ({}, dict(membership_sha256="f" * 64), "FULL_CLASS_SET_HASH_MISMATCH"),
    ({}, dict(count=len(LABELS_A) + 1), "FULL_CLASS_COUNT_MISMATCH"),
    (dict(class_count=len(LABELS_A) + 1), {}, "FULL_CLASS_COUNT_MISMATCH"),
    (dict(labels=LABELS_A + [LABELS_A[0]]), {}, "FULL_CLASS_SET_DUPLICATE_LABEL"),
    (dict(labels=LABELS_A + [""]), {}, "FULL_CLASS_SET_MALFORMED"),
    (dict(mapping={"ALDA": "없는클래스"}), {}, "FULL_MAPPING_LABEL_OUTSIDE_CLASS_SET"),
    (dict(mapping={"ALDA": "알다", "NA": "알다"}), {}, "FULL_MAPPING_AMBIGUOUS"),
    ({}, dict(project_word_to_model_label=["ALDA"]), "FULL_MAPPING_MALFORMED"),
    ({}, dict(model_id="another_model"), "FULL_CLASS_SET_IDENTITY_MISMATCH"),
    ({}, dict(checkpoint_sha256=CK_B), "FULL_CLASS_SET_IDENTITY_MISMATCH"),
    ({}, dict(status="UNAVAILABLE", reason="MODEL_CHECKPOINT_MISSING", labels=None),
     "FULL_CLASS_SET_UNAVAILABLE_AT_CREATION:MODEL_CHECKPOINT_MISSING"),
]


@pytest.mark.parametrize("kw,ev_over,reason", BAD_EVIDENCE)
def test_bad_creation_evidence_is_unavailable_not_repaired(tmp_path, kw, ev_over, reason):
    c = make_creation(**kw)
    c["trained_class_set"].update(ev_over)
    res = make_result(make_creation(), "알다")     # a perfectly good answer is not enough
    rep = run(tmp_path, [{"tid": "t1", "creation": c, "actual": "ALDA", "result": res}])
    fs = rep["full_scope"]
    assert fs["groups"] == {}
    assert [u["reason"] for u in fs["unavailable"]] == [reason]
    assert fs["unavailable"][0]["confirmed"] is True
    # the restricted reports still read this trial as before
    assert rep["per_method"]["denominator"] == 1


def test_separate_models_and_class_sets_are_never_pooled(tmp_path):
    a = make_creation()
    b = make_creation("cand_b", CK_B, labels=LABELS_B)
    a2 = make_creation(schema="fs_v2")                       # same checkpoint, other features
    rep = run(tmp_path, [
        {"tid": "t1", "creation": a, "actual": "ALDA", "result": make_result(a, "알다")},
        {"tid": "t2", "creation": b, "actual": "ALDA", "result": make_result(b, "나")},
        {"tid": "t3", "creation": a2, "actual": "ALDA", "result": make_result(a2, "알다")},
    ])
    gs = full_groups(rep)
    assert len(gs) == 3
    assert sorted(g["identity"]["class_count"] for g in gs) == sorted(
        [len(LABELS_A), len(LABELS_B), len(LABELS_A)])
    assert all(g["n_supported"] == 1 for g in gs)
    by_model = {(g["identity"]["model_id"], g["identity"]["feature_schema"]): g for g in gs}
    assert by_model[("cand_b", "fs_v1")]["per_method"]["candidate"]["hits"] == 0
    assert by_model[("cand_a", "fs_v1")]["per_method"]["candidate"]["hits"] == 1


def test_legacy_record_keeps_old_reports_and_full_is_explicitly_unavailable(tmp_path):
    c = legacy_creation()
    res = make_result(c, "알다", p18_top1="ALDA")
    # an old result: full block with no class-set hash, and a top3 that must not become a set
    del res["research_comparison"]["scopes"]["full"]["class_set_sha256"]
    rep = run(tmp_path, [{"tid": "t1", "creation": c, "actual": "ALDA", "result": res}])
    before = sorted(os.listdir(os.path.join(str(tmp_path), "trials")))
    raw_path = os.path.join(str(tmp_path), "trials", "t1.result.json")
    with open(raw_path, "rb") as fh:
        raw = fh.read()
    fs = rep["full_scope"]
    assert fs["status"] == "UNAVAILABLE" and fs["groups"] == {}
    assert [u["reason"] for u in fs["unavailable"]] == ["FULL_CLASS_SET_NOT_RECORDED"]
    assert cohort(rep, "project18")["per_method"]["candidate"]["hits"] == 1
    assert cohort(rep, "common5")["n_supported"] == 1
    assert rep["per_method"]["dtw"]["matched"] == 1
    # reading is not writing: nothing is backfilled
    review_log.report(str(tmp_path), ["t1"], trial_meta={"t1": {"research_model": c}})
    assert sorted(os.listdir(os.path.join(str(tmp_path), "trials"))) == before
    with open(raw_path, "rb") as fh:
        assert fh.read() == raw


def test_no_retroactive_inference_without_session_metadata(tmp_path):
    c = legacy_creation()
    stored_only_copy = make_result(c, "알다", saved_research_model=False)
    no_opt_in = make_result(c, "알다", saved_research_model=False)
    del no_opt_in["research_comparison"]
    rep = run(tmp_path, [
        {"tid": "t1", "creation": None, "actual": "ALDA", "result": stored_only_copy},
        {"tid": "t2", "creation": None, "actual": "ALDA", "result": no_opt_in},
    ], with_meta=False)
    fs = rep["full_scope"]
    assert fs["groups"] == {}
    assert [(u["trial_id"], u["reason"]) for u in fs["unavailable"]] == [
        ("t1", "FULL_CLASS_SET_NOT_RECORDED")]
    assert fs["counts"]["no_opt_in_evidence"] == 1


def test_saved_result_fallbacks_carry_the_evidence(tmp_path):
    c = make_creation()
    via_saved = make_result(c, "감")
    via_copy = make_result(c, "감", saved_research_model=False)
    rep = run(tmp_path, [
        {"tid": "t1", "creation": None, "actual": "GAM", "result": via_saved},
        {"tid": "t2", "creation": None, "actual": "GAM", "result": via_copy},
    ], with_meta=False)
    g = only_group(rep)
    assert g["per_method"]["candidate"]["match_over_all_supported"]["display"] == "2/2"
    assert g["identity"]["identity_source"] == ["result_identity_copy", "result_research_model"]


def test_store_failure_produces_no_full_scope(tmp_path):
    with open(os.path.join(str(tmp_path), review_log.REVIEW_FILE), "w", encoding="utf-8") as fh:
        fh.write("{not json\n")
    rep = review_log.report(str(tmp_path), ["t1"])
    assert rep["store_status"] == "MALFORMED" and rep["full_scope"] is None


# ------------------------------------------------------- metadata propagation (no camera/torch)
LABELS_7 = ["알다", "감", "공부", "기타0", "기타1", "기타2", "기타3"]


def _neural(monkeypatch, status="READY"):
    pytest.importorskip("numpy")
    try:
        from wis_lab.service import neural
    except ImportError as exc:                       # pragma: no cover - environment dependent
        pytest.skip(f"neural not importable here: {exc}")
    key = "union93_candidate_v1"
    monkeypatch.setitem(neural._CACHES, key, {
        "model": None, "classes": list(LABELS_7) if status == "READY" else None,
        "status": status, "detail": None, "feature_schema": "fs_test"})
    return neural, key


def test_identity_freezes_the_verified_class_list(monkeypatch):
    neural, key = _neural(monkeypatch)
    ev = neural.model_identity(key)["trained_class_set"]
    assert ev["status"] == "AVAILABLE"
    assert ev["labels"] == LABELS_7                      # checkpoint order kept
    assert ev["count"] == 7
    assert ev["membership_sha256"] == review_log.full_class_set_sha256(LABELS_7)
    assert ev["project_word_to_model_label"] == {"ALDA": "알다", "GAM": "감", "GONGBU": "공부"}
    assert "JOTA" in ev["project_words_without_class"]
    assert ev["model_id"] == neural.MODELS[key]["model_id"]


def test_unloadable_model_has_no_coverage(monkeypatch):
    neural, key = _neural(monkeypatch, status="CHECKPOINT_MISSING")
    ev = neural.model_identity(key)["trained_class_set"]
    assert ev["status"] == "UNAVAILABLE" and ev["reason"] == "MODEL_CHECKPOINT_MISSING"
    assert ev["labels"] is None and ev["project_word_to_model_label"] is None


def test_creation_metadata_reaches_the_report(monkeypatch, tmp_path):
    neural, key = _neural(monkeypatch)
    pytest.importorskip("fastapi")
    try:
        from wis_lab.service import app
    except ImportError as exc:                       # pragma: no cover - environment dependent
        pytest.skip(f"app not importable here: {exc}")
    choice = app._research_choice(key)
    sha = choice["trained_class_set"]["membership_sha256"]

    def fake_recognise(obs, model_key=None):
        assert obs is None and model_key == key
        return {"schema": neural.ADAPTER_SCHEMA, "model_key": key,
                "model_id": choice["model_id"], "checkpoint_sha256": choice["checkpoint_sha256"],
                "model_status": "READY",
                "scopes": {"full": {"scope": "full", "candidate_count": 7, "top1": "감",
                                    "top3": [{"label": "감", "score": 1.0}],
                                    "class_set_sha256": sha}}}

    monkeypatch.setattr(neural, "recognise", fake_recognise)
    out = app._research_comparison(None, {"research_model": choice, "requested_word_id": "ALDA"})
    assert out["identity_fixed_at_trial_creation"]["trained_class_set"] == \
        choice["trained_class_set"]
    os.makedirs(os.path.join(str(tmp_path), "trials"))
    with open(os.path.join(str(tmp_path), "trials", "t1.result.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"observation_status": "OK", "ranking": [], "research_model": choice,
                   "research_comparison": out}, fh, ensure_ascii=False)
    review_log.submit(str(tmp_path), "ses_t", "t1", "GAM", "yes")
    g = only_group(review_log.report(str(tmp_path), ["t1"]))
    assert g["identity"]["class_count"] == 7
    assert g["per_method"]["candidate"]["match_over_all_supported"]["display"] == "1/1"
