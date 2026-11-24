"""Independent checks of new metadata gates; no real arrays or checkpoints loaded."""
import copy
import functools
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

OPS = Path(__file__).resolve().parent
WEB = OPS.parents[1]
spec = importlib.util.spec_from_file_location("pair_v4", WEB / "tools_native_data_pair.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
plan0 = json.loads((OPS / "plan.json").read_text(encoding="utf-8-sig"))
base0 = json.loads(Path(plan0["base_inputs"]).read_text(encoding="utf-8-sig"))
cand0 = json.loads(Path(plan0["candidate_report"]).read_text(encoding="utf-8-sig"))
OUT = str(WEB / "evidence" / "native_pair_new_code_audit" / "not_created")
checks = []


class ReservedTripwire:
    def __iter__(self):
        raise AssertionError("reserved metadata iterated")
    def __len__(self):
        raise AssertionError("reserved metadata inspected")
    def __getitem__(self, key):
        raise AssertionError("reserved metadata indexed")


def attempt(name, change=None, expect_ready=False, pending=False):
    plan, base, cand = copy.deepcopy(plan0), copy.deepcopy(base0), copy.deepcopy(cand0)
    base["reserved_test"] = ReservedTripwire()
    if change:
        change(plan, base, cand)
    reads = []

    def read_json(path):
        reads.append(str(path))
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(plan0["base_inputs"])):
            return base
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(plan0["candidate_report"])):
            return cand
        raise AssertionError("unexpected metadata read")

    def forbid(*args, **kwargs):
        raise AssertionError("preflight attempted content hashing")

    try:
        with patch.object(m, "_read_json", side_effect=read_json), patch.object(m, "_sha256", side_effect=forbid):
            report, prepared = m.preflight(plan, OUT)
        valid_counts = report.get("counts", {}).get("control") == 2113 and report.get("counts", {}).get("treatment") == 2122 and report.get("counts", {}).get("eval") == 699
        passed = report["status"] == "ready" if expect_ready else report["status"] in {"gated", "invalid"}
        if expect_ready:
            passed = passed and prepared is not None and valid_counts and len(reads) == 2
            passed = passed and prepared["index_order"] == list(range(93))
        if pending:
            passed = passed and prepared is not None and valid_counts and len(reads) == 2
        passed = passed and report["arrays_loaded"] == 0 and report["trained_models"] == 0
        checks.append({"name": name, "passed": bool(passed), "status": report["status"], "blocks": report["blocks"][:5]})
    except Exception as exc:
        checks.append({"name": name, "passed": False, "exception": type(exc).__name__, "detail": str(exc)})


def main():
    # Cache only stat results within this audit; no old data contents are opened.
    isfile = functools.lru_cache(maxsize=None)(m.os.path.isfile)
    with patch.object(m.os.path, "isfile", side_effect=isfile):
        attempt("approved real metadata accepted with frozen class order", expect_ready=True)
        attempt("pending semantic review retains metadata readiness", lambda p,b,c: p.update(semantic_review={"status":"pending","reviewer":None,"evidence":None}), pending=True)
        attempt("wrong candidate rows schema rejected", lambda p,b,c: c.update(rows=12))
        attempt("heldout signer excluded", lambda p,b,c: c["rows"][0].update(signer_id=14,clip_id="NIA_SL_SEN0356_REAL14_F"))
        attempt("REAL011 token rejected", lambda p,b,c: b["source_train"][0].update(id=b["source_train"][0]["id"].replace("REAL01", "REAL011")))
        attempt("role mismatch rejected", lambda p,b,c: b["source_validation"][0].update(role="training"))
        attempt("development class label mismatch rejected", lambda p,b,c: b["source_validation"][0].update(label="synthetic_bad_label"))
        attempt("cross-split content duplicate rejected", lambda p,b,c: b["source_validation"][0].update(npz_sha256=b["source_train"][0]["npz_sha256"]))
        attempt("candidate signer mismatch rejected", lambda p,b,c: c["rows"][0].update(signer_id=2))
        attempt("candidate nonhex digest rejected", lambda p,b,c: c["rows"][0].update(npz_sha256="z"*64))
        attempt("candidate path escape rejected", lambda p,b,c: c["rows"][0].update(npz=str(WEB.parent / "outside.npz")))
        attempt("development view cannot masquerade as another signer", lambda p,b,c: c["rows"][-1].update(clip_id="NIA_SL_SEN0356_REAL131_F"))

    # Canonical path safety without creating any junction or deleting any directory.
    actual_realpath = m.os.path.realpath
    escaped = str(WEB / "runs" / "fixture_link" / "x.npz")
    def resolve_link(path, *args, **kwargs):
        if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.abspath(escaped)):
            return str(WEB.parent / "outside.npz")
        return actual_realpath(path, *args, **kwargs)
    with patch.object(m.os.path, "realpath", side_effect=resolve_link):
        checks.append({"name": "canonical link escape rejected", "passed": not m._under(str(WEB), escaped)})

    class EvalRecorder:
        def evaluate(self, model, clips, classes, restrict_to=None):
            return {"ids":[c["id"] for c in clips],"classes":classes,"restrict_to":restrict_to}
    ev = m._target_eval(EvalRecorder(), None, [{"id":1,"label":"A"},{"id":2,"label":"B"}], ["A","B","C"], ["A"])
    checks.append({"name":"target helper filters GT while retaining full model class order", "passed":ev=={"ids":[1],"classes":["A","B","C"],"restrict_to":["A"]}})
    for value in ("XREAL01", "REAL01suffix", "REAL011", "REAL01_REAL01", "REAL01_REAL02", "REAL01_REALbad"):
        errors = []
        token = m._strict_signer_token(value, {"REAL01"}, "fixture", errors)
        checks.append({"name":"strict malformed token rejected: " + value, "passed":token is None and bool(errors)})
    checks.append({"name":"default path imports no model or array libraries", "passed":"torch" not in sys.modules and "numpy" not in sys.modules})
    checks.append({"name":"preflight creates no output directory", "passed":not Path(OUT).exists()})
    report={"status":"PASS" if all(c["passed"] for c in checks) else "FAIL", "passed":sum(c["passed"] for c in checks),"total":len(checks),"checks":checks,"arrays_loaded":0,"checkpoints_loaded":0,"training_runs":0,"source_manifests_changed":False}
    with (OPS / "metadata_contract_v6.json").open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:report[k] for k in ("status","passed","total","training_runs")}))
    if report["status"]!="PASS":
        print(json.dumps([c for c in checks if not c["passed"]],ensure_ascii=True))
    return int(report["status"]!="PASS")


if __name__=="__main__":
    raise SystemExit(main())
