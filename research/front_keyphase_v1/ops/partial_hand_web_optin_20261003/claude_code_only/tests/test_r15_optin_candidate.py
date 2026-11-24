"""R15: the opt-in research candidate. Synthetic and public inputs only.

What must hold: a trial that opts into nothing behaves exactly as before; a named model answers
under its own identity and never borrows the default's; the scopes carry their own candidate
counts and are never presented as one task; a word the named model never learned is NOT_IN_MODEL;
and the requested word never reaches any model input.

No camera, no training, no threshold, no server restart, no user-ten replay.

Run: py -3.13 tests/test_r15_optin_candidate.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "sign_model_pilot_v1", "src"))

from wis_lab.service import neural as N  # noqa: E402

CHECKS = []
MEASURED = {"inputs": "the public NIKL reference clips and synthetic arrays; no private material"}
NL = chr(10)
CANDIDATE = "union93_candidate_v1"


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def client():
    from fastapi.testclient import TestClient

    from wis_lab.service.app import app
    return TestClient(app)


@check("the registry keeps the default untouched and marks the candidate as under validation")
def _registry():
    cap = N.registry_capability()
    assert cap["default_model_key"] == "base", cap["default_model_key"]
    base, cand = cap["models"]["base"], cap["models"][CANDIDATE]
    assert base["checkpoint_sha256"] == N.CHECKPOINT_SHA256
    assert base["trained_class_count"] == 80 and len(base["project_words_in_model"]) == 5
    assert cand["checkpoint_sha256"] == \
        "e7c9c31a9c747e2c12a9a56695d2e4e9dbec9c372c8e2a3f209d00fa4c25dced"
    assert cand["trained_class_count"] == 93 and len(cand["project_words_in_model"]) == 18
    assert cand["status_label"] == "검증 중"
    assert "none" in cap["promotion"]
    assert "never compared as equals" in cap["scope_warning"]
    # the held-back word is not quietly included
    assert "KKEUDEOGIDA" not in cand["project_words_in_model"]
    MEASURED["registry"] = {"base_classes": base["trained_class_count"],
                            "candidate_classes": cand["trained_class_count"],
                            "candidate_project_words": len(cand["project_words_in_model"])}


@check("opting into nothing leaves the result exactly as it was")
def _opt_out():
    c = client()
    r = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01"})
    assert r.status_code == 200
    d = r.json()
    assert "research_comparison" not in d, "a candidate answered without being asked"
    assert d["research_model"]["opted_in"] is False
    # the default reading is still there and still the default model
    assert d["learned_recognition"]["checkpoint_sha256"] == N.CHECKPOINT_SHA256
    assert d["accepted"] is None and d["execution_assessment"] == "UNCONFIGURED"
    assert d["lexical_identification"]["research_comparison"]["opted_in"] is False
    MEASURED["opt_out"] = {"research_block": False,
                           "accepted": d["accepted"],
                           "execution_assessment": d["execution_assessment"]}


@check("opting in adds a separate reading without touching the default one")
def _opt_in():
    c = client()
    a = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01"}).json()
    b = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01",
                                        "research_model_key": CANDIDATE}).json()
    # the default answer is byte-for-byte the same with and without the opt-in
    assert a["learned_recognition"]["ranking"] == b["learned_recognition"]["ranking"]
    assert a["ranking"] == b["ranking"], "the DTW ranking moved"
    assert a["accepted"] == b["accepted"] and \
        a["execution_assessment"] == b["execution_assessment"]
    rc = b["research_comparison"]
    assert rc["model_key"] == CANDIDATE and rc["model_status"] == "READY"
    assert rc["checkpoint_sha256"] != N.CHECKPOINT_SHA256, "the candidate reused the base hash"
    assert rc["is_under_validation"] is True and rc["is_a_verdict"] is False
    assert rc["accepted"] is None and rc["execution_assessment"] == "UNCONFIGURED"
    assert "none" in rc["promotion"]
    MEASURED["opt_in"] = {"default_unchanged": True,
                          "candidate_status": rc["model_status"]}


@check("the three scopes carry their own candidate counts and refuse to be read as one task")
def _scopes():
    c = client()
    d = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01",
                                        "research_model_key": CANDIDATE}).json()
    sc = d["research_comparison"]["scopes"]
    assert set(sc) == {"common5", "project18", "full"}, sorted(sc)
    assert sc["common5"]["candidate_count"] == 5
    assert sc["project18"]["candidate_count"] == 18
    assert sc["full"]["candidate_count"] == 93
    for v in sc.values():
        assert "different task" in v["not_comparable_across_scopes"]
    # the base model's own scopes are its own sizes, not the candidate's
    e = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01",
                                        "research_model_key": "base"}).json()
    b = e["research_comparison"]["scopes"]
    assert b["project18"]["candidate_count"] == 5 and b["full"]["candidate_count"] == 80
    MEASURED["scopes"] = {"candidate": {k: sc[k]["candidate_count"] for k in sc},
                          "base": {k: b[k]["candidate_count"] for k in b}}


@check("a word the named model never learned is NOT_IN_MODEL, per model")
def _coverage():
    assert N.word_coverage("INSA")["status"] == "NOT_IN_MODEL"
    assert N.word_coverage("INSA", CANDIDATE)["status"] == "IN_MODEL"
    held = N.word_coverage("KKEUDEOGIDA", CANDIDATE)
    assert held["status"] == "NOT_IN_MODEL", held
    assert "no other model" in held["meaning"]
    assert N.word_coverage("NA", "not_a_model")["status"] == "UNKNOWN_MODEL_KEY"
    MEASURED["coverage"] = {"INSA_base": "NOT_IN_MODEL", "INSA_candidate": "IN_MODEL",
                            "KKEUDEOGIDA_candidate": "NOT_IN_MODEL"}


@check("an unknown or unloadable model reports itself and is never answered by the default")
def _no_fallback():
    c = client()
    d = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01",
                                        "research_model_key": "not_a_model"}).json()
    rc = d["research_comparison"]
    assert rc["model_status"] == "UNKNOWN_MODEL_KEY", rc
    assert rc["ranking"] is None and "NOT substituted" in rc["detail"]
    assert "scopes" not in rc, "a failed model still produced a ranking"
    # a pinned hash that does not match must refuse too, without borrowing the base
    real = N.MODELS[CANDIDATE]["sha256"]
    N.MODELS[CANDIDATE]["sha256"] = "0" * 64
    N._CACHES.pop(CANDIDATE, None)                        # noqa: SLF001
    try:
        out = N.recognise(_public_observation(), model_key=CANDIDATE)
        assert out["model_status"] == "CHECKPOINT_HASH_MISMATCH", out["model_status"]
        assert out["ranking"] is None
    finally:
        N.MODELS[CANDIDATE]["sha256"] = real
        N._CACHES.pop(CANDIDATE, None)                    # noqa: SLF001
    MEASURED["no_fallback"] = ["UNKNOWN_MODEL_KEY", "CHECKPOINT_HASH_MISMATCH"]


def _public_observation():
    from wis_lab.core.observation import Observation
    with open(os.path.join(ROOT, "config", "source_manifest.json"), encoding="utf-8") as fh:
        clip = next(c for c in json.load(fh)["clips"] if c["word_id"] == "JOTA")
    return Observation.load(os.path.join(ROOT, clip["observation_npz"]))


@check("the requested word never reaches a model input, on either model")
def _blind():
    import inspect
    # the whole blind path: dispatch, the shared inference body, the two-arm entry, the channel
    # builder and the transform it may call. Each function's own docstring is dropped (prose).
    path = (N.recognise, N._infer, N._recognise_handgap,            # noqa: SLF001
            N._channels, N._handgap_transform)                      # noqa: SLF001
    parts = {}
    for fn in path:
        head, _doc, rest = inspect.getsource(fn).split('"""', 2)
        parts[fn.__name__] = head + rest
    body = "\n".join(parts.values())
    # identifiers, not prose: the word "task" legitimately appears in the scope explanations
    for banned in ("requested_word_id", "attempt_label", "task_match", "reference_binding"):
        assert banned not in body, banned
    # recognise only dispatches the observation and the key
    assert "return _recognise_handgap(obs, model_key)" in parts["recognise"]
    assert "return _infer(obs, model_key)[0]" in parts["recognise"]
    # both hand-gap arms go through the same inference body with nothing else added
    assert "_infer(obs, model_key)" in parts["_recognise_handgap"]
    assert "_infer(obs, model_key, transform)" in parts["_recognise_handgap"]
    # the only thing handed to the network is the clip built from the observation
    infer = parts["_infer"]
    assert 'c["model"](torch.from_numpy(x), torch.from_numpy(pad))' in infer
    assert "collate([dict(clip, label=0)])" in infer, "the label slot must be a constant zero"
    assert infer.count('c["model"](') == 1 and infer.count("collate(") == 1
    # the transform sees only copies of the observation's own span arrays
    assert "transform(nat[lo:hi + 1].copy(), present[lo:hi + 1].copy()," in parts["_channels"]
    # the adapter only ever receives the observation and a key
    sig = list(inspect.signature(N.recognise).parameters)
    assert sig == ["obs", "model_key"], sig
    app_src = open(os.path.join(ROOT, "src", "wis_lab", "service", "app.py"),
                   encoding="utf-8").read()
    body = app_src[app_src.index("def _research_comparison("):
                   app_src.index("def _split_result(")]
    assert "neural.recognise(obs, model_key=key)" in body
    assert "requested_word_id" in body and "word_coverage" in body, \
        "coverage is reported, which is the only place the requested word is used"
    assert body.index("word_coverage") > body.index("neural.recognise"), \
        "the requested word must only be used AFTER the blind ranking"
    MEASURED["blind"] = {"recognise_parameters": sig}


@check("the coverage note states THIS model's coverage, not a constant from another")
def _coverage_note():
    c = client()
    base = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01"}).json()
    cand = c.post("/api/v1/replays", json={"observation_id": "NIKL_JOTA_01",
                                           "research_model_key": CANDIDATE}).json()
    bn = base["learned_recognition"]["note"]
    cn = cand["research_comparison"]["note"]
    assert "14 project words" not in bn and "14 project words" not in cn, "a stale constant"
    assert "5 of the project's 19" in bn, bn
    assert "18 of the project's 19" in cn, cn
    assert "NOT_IN_MODEL" in bn and "NOT_IN_MODEL" in cn
    MEASURED["coverage_note"] = {"base": "5 of 19", "candidate": "18 of 19"}


@check("the model choice is fixed at trial creation and carries its identity")
def _fixed_choice():
    from wis_lab.service.app import _research_choice

    none = _research_choice(None)
    assert none["opted_in"] is False and none["model_key"] is None
    one = _research_choice(CANDIDATE)
    assert one["opted_in"] is True and one["fixed_at"] == "trial_creation"
    assert one["no_fallback_to_default"] is True
    for key in ("model_key", "model_id", "checkpoint_sha256", "trained_class_count",
                "feature_schema", "status_label"):
        assert key in one, key
    assert one["trained_class_count"] == 93 and one["status_label"] == "검증 중"
    bad = _research_choice("not_a_model")
    assert bad["opted_in"] is True and bad["status"] == "UNKNOWN_MODEL_KEY"
    MEASURED["fixed_choice"] = {"identity_keys": sorted(k for k in one)}


@check("capture timing, DTW, the demonstration binding and the verdict fields are untouched")
def _unchanged_surfaces():
    c = client()
    d = c.post("/api/v1/replays", json={"observation_id": "NIKL_NA_01",
                                        "compare_to_word_id": "NA",
                                        "research_model_key": CANDIDATE}).json()
    assert d["reference_binding"]["status"] == "BOUND"
    assert d["demonstration_following"]["status"] == "AVAILABLE"
    assert d["accepted"] is None and d["execution_assessment"] == "UNCONFIGURED"
    assert d["counts_as_human_attempt"] is False
    assert d["observation_status"] == "OK" and d["ranking"]
    app_src = open(os.path.join(ROOT, "src", "wis_lab", "service", "app.py"),
                   encoding="utf-8").read()
    finish = app_src[app_src.index("def _finish("):app_src.index("def _append_decision(")]
    # the research reading happens after the result is built and before the single time stamp
    assert finish.index("_research_comparison(obs, t)") < finish.index(
        't["timing"]["server_result_ns"]')
    MEASURED["unchanged"] = {"accepted": d["accepted"],
                             "execution_assessment": d["execution_assessment"],
                             "binding": d["reference_binding"]["status"]}


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:                     # noqa: BLE001 - any raise is a failure
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"{NL}{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    print(json.dumps(MEASURED, ensure_ascii=False, indent=1, default=str))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
