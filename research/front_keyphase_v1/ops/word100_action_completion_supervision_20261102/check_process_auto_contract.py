"""Pure synthetic qualification and saved binding checks; no model imports."""
from pathlib import Path
import json,hashlib,ast,numpy as np
from process_auto_proposal import canonical_anchor,make_proposal
from process_qualification import propose_motion_intervals,qualify_process,known_seam_diagnostic
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
checks=[]
def check(name,value):
 assert value,name
 checks.append(name)
def q(point,av,p=.9,scores=None):
 spec=propose_motion_intervals(point,av)
 if scores is None:scores=[.9]*len(spec["candidates"])
 return qualify_process(p,point,av,scores)
point=np.array([[.9,.1],[.1,.1],[.1,.9]])
r=q(point,[1,1,1]);check("one-frame predicates qualify with learned interval",r["qualified_display"]==1)
check("endpoints without learned motion do not qualify",q(point,[1,1,1],scores=[.1])["qualified_display"] is None)
check("highmotion lowprocess discrepancy",q(point,[1,1,1],p=.1)["status"]=="PROCESS_AND_LOCAL_EVIDENCE_DISAGREE")
check("blackout breaks endpoint bridge",q(point,[1,0,1])["qualified_display"] is None)
pp=np.concatenate([point,[[0,0]],point],axis=0)
check("later intact occurrence survives other gap",q(pp,[1,0,1,0,1,1,1])["qualified_display"]==1)
check("allblank remains unavailable",q(point,[0,0,0])["status"]=="INSUFFICIENT_OBSERVATION")
check("singleframe no invented interval",q([[.9,.9]],[1])["qualified_display"] is None)
check("two-real-frame short chain allowed",q([[.9,.1],[.1,.9]],[1,1])["qualified_display"]==1)
oracle=known_seam_diagnostic(r,[1])
check("oracle seam separates same cached score",r["qualified_display"]==1 and oracle["qualified_display"] is None and oracle["same_runtime_candidates_and_scores_reused"])
rr=q(np.tile(point,(6,1)),[1]*18)
check("finite interval cap",rr["candidate_count"]==8 and rr["candidate_cap_reached"])
check("no absence or word correctness certificate",rr["observed_process_absence_certified"] is False and rr["whole_word_correctness"] is None)
guides=[dict(word_id="W",frames=[15,20],canonical_stage_id="s",boundary_provenance="canonical01_saved_reviewed_span")]
anchor=canonical_anchor(guides,"W",np.arange(40),40)
p=make_proposal(np.column_stack([np.arange(40),np.arange(40)]),np.arange(40),np.arange(40)*33333,anchor)
check("fixed single context hull",p["full_native_index_extent"]==[3,32] and p["max_proposals_per_query"]==1)
try:canonical_anchor(guides,"W",np.arange(40)+1,40)
except AssertionError:checks.append("reject nonidentity canonical source/index axis")
else:raise AssertionError("nonidentity axis accepted")
try:make_proposal([[0,0],[39,39]],np.arange(40),np.arange(40)*33333,anchor)
except AssertionError:checks.append("reject invalid saved path without fallback")
else:raise AssertionError("bad path accepted")
manifest=read(B/"PROCESS_AUTO_NATIVE_MANIFEST.json")
check("fixed 45FIT15DEV48source inventory",manifest["counts"]==dict(fit_queries=45,development_queries=15,fit_clips=36,development_clips=12,total_queries=60))
for p,h in manifest["pins"].items():assert sha(p)==h,p
check("all saved input/result/proposal dependency hashes",True)
for cid in {r["clip_id"] for r in manifest["rows"] if r["word_id"]=="WORD1567"}:
 rr=[r["proposal"] for r in manifest["rows"] if r["clip_id"]==cid];assert len(rr)==2 and rr[0]==rr[1]
check("two1567queries share exact automatic proposal",True)
files=["process_auto_proposal.py","process_qualification.py","prepare_process_auto_native.py","evaluate_process_auto.py"]
for name in files:ast.parse((B/name).read_text(encoding="utf-8-sig"))
check("four new code files parse",True)
out=dict(status="PASS_PURE_CONTRACT_AND_SAVED_BINDINGS",checks=checks,count=len(checks),failures=0,
 pins={str(B/n):sha(B/n) for n in files+["PROCESS_AUTO_PROPOSAL_PLAN.json","PROCESS_AUTO_NATIVE_MANIFEST.json"]},
 actual_model_forward=0,training=0,new_DTW=0,old_result_regrading=0,
 scope="Synthetic functions plus integrity/schema of prepared saved arrays; no semantic accuracy or candidate-score evaluation.")
dest=B/"PROCESS_AUTO_CONTRACT_VERIFIED.json";assert not dest.exists()
dest.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(status=out["status"],checks=len(checks),failures=0)))

