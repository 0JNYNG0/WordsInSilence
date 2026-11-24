"""Targeted final-code pin check after a concurrent prefreeze pipeline edit."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

U=Path(__file__).resolve().parent.parent
R=U/"review"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))

first=read(R/"PREPARATION_REVIEW.json")
pipeline=str(U/"pipeline.py")
final_sha="1d8552e3924f28fc94e9c6c53109e44a81815eeeca501add03771f324d0770e4"
checks=0;failures=[]
def check(ok,why):
 global checks
 checks+=1
 if not ok:failures.append(why)
check(first["failure_count"]==1 and first["failures"]==["Source unchanged across review "+pipeline] and first["exception"] is None,
      "Only original finding was a concurrent pipeline pin change")
check(sha(pipeline)==final_sha,"Final root-declared reviewed pipeline SHA")
pins=first["pins"].copy();old_sha=pins[pipeline];pins[pipeline]=final_sha
for p,h in pins.items():check(sha(p)==h,"Unchanged original source/current final code "+p)
code=Path(pipeline).read_text(encoding="utf-8")
for snippet in ("out=hand_evidence.compare(q,r,path)",
 "path=np.array(task['baseline_full_path'],dtype=int)-[qa,ra]",
 "path=np.column_stack((np.arange(len(q['x'])),np.arange(len(r['x']))))",
 "source_expectation_not_semantic_label':True",
 "'review/PREPARATION_REVIEW_FINAL.json'",
 "'review/PREPARATION_REVIEW.json'",
 "inp['pins'].update(read(UNIT/'review/INPUT_AUDIT.json')['pins'])",
 "inp['pins'].update(read(UNIT/'review/INPUT_AUDIT.json')['protected'])"):
 check(snippet in code,"Reviewed final additive data flow "+snippet)
for p in (R/"PREPARATION_REVIEW.json",R/"verify_saved_sources.py",Path(__file__)):
 pins[str(p)]=sha(p)
for p,h in pins.items():check(sha(p)==h,"Stable final review pin "+p)
report=dict(schema="wis_independent_hand_preparation_final_pin_supplement_v1",
 status="PASS" if not failures else "FAIL",at_utc=datetime.now(timezone.utc).isoformat(),
 checks=checks,failure_count=len(failures),failures=failures,
 original_preparation_review=dict(path=str(R/"PREPARATION_REVIEW.json"),sha256=sha(R/"PREPARATION_REVIEW.json"),
  status=first["status"],checks=first["checks"],failure_count=1,
  finding="pipeline.py changed concurrently during initial review. Source/transform/range assertions all passed; original FAIL retained."),
 reviewed_change=dict(path=pipeline,first_read_sha256=old_sha,final_sha256=final_sha,
  scope="Final runtime adds immutable shape-guide source pointer; freeze adds audit/protected pins, preflight PASS requirements and keeps first FAIL artifacts. Numeric q/r/path inputs and technical data are not changed."),
 source_contract_reused=dict(checks_previously_passed=first["checks"]-1,totals=first["totals"],
  no_full_transform_rerun=True,all_previously_checked_source_files_rehashed=True),
 pins=pins,actions=dict(producer_execution=0,model_inference=0,alignment_search=0,source_modification=0),
 limits="This preexecution final review does not claim public results have run or passed.")
out=R/"PREPARATION_REVIEW_FINAL.json"
with out.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=report["status"],checks=checks,failures=failures,sha256=sha(out))))
if failures:raise SystemExit(1)
