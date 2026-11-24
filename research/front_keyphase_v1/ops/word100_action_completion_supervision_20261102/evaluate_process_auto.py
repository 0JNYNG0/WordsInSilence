"""One new process-head qualification/automatic-context evaluation, never old heads.
Reuses297 saved raw process/point outputs. Scores <=8 learned-selected intervals
in one batch per view, then60 fixed full-native query proposals once each.
"""
from pathlib import Path
from datetime import datetime,timezone
import json,time,traceback
import numpy as np
import torch
from process_head import ProcessEvidenceNet,QUERIES,make_input,batch
from process_qualification import propose_motion_intervals,qualify_process,known_seam_diagnostic
from evaluate_process_head import read,sha,load_case,check_pins
U=Path(__file__).resolve().parent
def write(p,x):
 with Path(p).open("x",encoding="utf-8") as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
def qualify(model,seq,query,p,point,available,seams):
 spec=propose_motion_intervals(point,available);cs=spec["candidates"];scores=[]
 if cs:
  x,l=batch([seq[c["output_indices"]] for c in cs])
  qi=torch.full((len(cs),),QUERIES.index(query),dtype=torch.long)
  scores=model.interval_logits(x,l,qi).sigmoid().tolist()
 runtime=qualify_process(p,point,available,scores)
 return runtime,known_seam_diagnostic(runtime,seams)
def summarize(rows,field="qualification"):
 known_positive=[r for r in rows if r["truth"]==1];known_zero=[r for r in rows if r["truth"]==0]
 return dict(total=len(rows),known_positive=len(known_positive),known_zero=len(known_zero),
  unknown=sum(r["truth"] is None for r in rows),
  confirmed_positive_correct=sum(r[field]["qualified_display"]==1 for r in known_positive),
  confirmed_positive_wrong=sum(r[field]["qualified_display"]==1 for r in known_zero),
  unknown_confirmed_positive=sum(r["truth"] is None and r[field]["qualified_display"]==1 for r in rows),
  known_positive_recall=sum(r[field]["qualified_display"]==1 for r in known_positive)/len(known_positive) if known_positive else None,
  raw_low_on_known_positive=sum(r[field]["raw_display"]==0 for r in known_positive),
  raw_low_on_known_zero=sum(r[field]["raw_display"]==0 for r in known_zero),
  all_absence_claims=False,qualified_negative_is_not_defined=True)
def native_truth(target,frame_ids):
 yes=[s for s in target["sequence_links"] if s["visible_chain"]=="yes"]
 source_truth=1 if yes else None
 frames=set(int(x) for x in frame_ids)
 supported=[]
 for s in yes:
  required=set(s["setup_frames"]+s["terminal_frames"])
  if s.get("motion_interval"):
   lo,hi=s["motion_interval"];required.update(range(lo,hi+1))
  if required and required<=frames:supported.append(s)
 return source_truth,(1 if supported else None),len(supported)
def main():
 started=U/"PROCESS_AUTO_EVALUATION_STARTED.json";out=U/"PROCESS_AUTO_EVALUATION.json"
 assert not started.exists() and not out.exists(),"No silent repeated/interrupted candidate evaluation"
 contract=read(U/"PROCESS_COMPARISON_FROZEN.json");check_pins(contract)
 train=read(U/"PROCESS_TRAIN_RESULT.json");raw=read(U/"PROCESS_EVALUATION.json")
 verified_raw=read(U/"PROCESS_EVALUATION_SAVED_VERIFIED.json")
 assert verified_raw["status"]=="PASS_SAVED_VALUE_ARITHMETIC_AND_LINEAGE_NO_INFERENCE"
 assert verified_raw["evaluation_sha256"]==sha(U/"PROCESS_EVALUATION.json")
 assert raw["checkpoint_sha256"]==train["checkpoint_sha256"]==verified_raw["checkpoint_sha256"]
 assert train["contract_sha256"]==raw["contract_sha256"]==sha(U/"PROCESS_COMPARISON_FROZEN.json")
 assert len(raw["rows"])==297 and sha(train["checkpoint"])==train["checkpoint_sha256"] and sha(train["normalization"])==train["normalization_sha256"]
 manifest=read(U/"PROCESS_AUTO_NATIVE_MANIFEST.json")
 for p,h in manifest["pins"].items():assert sha(p)==h,p
 stats=read(train["normalization"]);torch.set_num_threads(4)
 model=ProcessEvidenceNet();model.load_state_dict(torch.load(train["checkpoint"],map_location="cpu",weights_only=True));model.eval()
 views={};input_index={}
 for role,tname,iname in [("fit","PROCESS_VIEW_TARGETS.json","PROCESS_INPUT_MANIFEST.json"),("development","DEV_PROCESS_VIEW_TARGETS.json","DEV_PROCESS_INPUT_MANIFEST.json")]:
  for r in read(U/tname)["rows"]:views[r["view_id"]]=r
  for r in read(U/iname)["rows"]:input_index[(role,r["pixel_view_key"])]=r
 native_targets={r["id"]:r for name in ("PROCESS_TARGETS.json","DEV_PROCESS_TARGETS.json") for r in read(U/name)["rows"]}
 write(started,dict(at=datetime.now(timezone.utc).isoformat(),saved_raw_queries=297,full_native_queries=60,
  max_interval_candidates_per_query=8,contract_sha256=sha(U/"PROCESS_COMPARISON_FROZEN.json")))
 transformed=[];automatic=[];t0=time.perf_counter()
 with torch.inference_mode():
  for saved in raw["rows"]:
   r=views[saved["view_id"]];assert saved["truth"]==r["observed_process_evidence"]
   for key in ("query_id","clip_id","role","physical_query","pixel_view_key"):
    assert saved[key]==r[key],key
   assert saved["person"]==r["recorded_person_id"] and saved["kind"]==r["construction_kind_audit_only"]
   a=load_case(r,input_index[(r["role"],r["pixel_view_key"])]);seq=make_input(a,stats)
   assert len(saved["point_probabilities"])==len(seq)
   assert saved["image_available"]==a["image_available"].tolist()
   assert saved["source_frames"]==[f["source_native_frame"] for f in r["ordered_frame_map"]]
   assert saved["source_timestamps_us"]==[f["source_native_timestamp_us"] for f in r["ordered_frame_map"]]
   assert saved["view_timestamps_us"]==a["timestamp_us"].tolist()
   start=time.perf_counter()
   q,oracle=qualify(model,seq,r["physical_query"],saved["probability"],saved["point_probabilities"],a["image_available"],r["seam_output_indices"])
   transformed.append(dict(view_id=r["view_id"],query_id=r["query_id"],clip_id=r["clip_id"],person=r["recorded_person_id"],
    role=r["role"],physical_query=r["physical_query"],kind=r["construction_kind_audit_only"],truth=r["observed_process_evidence"],
    qualification=q,oracle_known_seam_diagnostic=oracle,new_interval_seconds=time.perf_counter()-start,
    raw_process_point_forward_repeated=False,annotated_interval_truth_used_to_select_candidates=False))
  for r in manifest["rows"]:
   assert sha(r["input"])==r["input_sha256"] and sha(r["result"])==r["result_sha256"]
   with np.load(r["input"],allow_pickle=False) as z:a={k:z[k].copy() for k in ("frame_id","timestamp_us","x","mask","rgb","image_available")}
   assert a["frame_id"].tolist()==r["frame_ids"] and a["timestamp_us"].tolist()==r["timestamp_us"]
   ii=r["proposal"]["output_indices"];seq=make_input(a,stats)[ii];available=a["image_available"][ii]
   start=time.perf_counter();x,l=batch([seq]);qi=torch.tensor([QUERIES.index(r["physical_query"])])
   prediction=model(x,l,qi);p=float(prediction["process_logits"].sigmoid()[0]);point=prediction["point_logits"].sigmoid()[0].tolist()
   q,_=qualify(model,seq,r["physical_query"],p,point,available,[])
   # Bind display evidence back to the full native source/time axis.
   if q["witnessed_chain"]:
    c=q["witnessed_chain"];full=[ii[j] for j in c["output_indices"]]
    c.update(full_native_indices=full,source_frame_ids=a["frame_id"][full].tolist(),source_timestamps_us=a["timestamp_us"][full].tolist())
   target=native_targets[r["query_id"]];assert target["source_sha256"]==r["source_sha256"]
   truth,scope_truth,supported=native_truth(target,a["frame_id"][ii])
   automatic.append(dict(query_id=r["query_id"],clip_id=r["clip_id"],person=r["person"],role=r["role"],physical_query=r["physical_query"],
    truth=truth,proposal_scope_truth=scope_truth,proposal_contains_reviewed_positive_chain=supported>0,
    qualification=q,probability=p,point_probabilities=point,proposal=r["proposal"],
    source_frame_ids=a["frame_id"][ii].tolist(),source_timestamps_us=a["timestamp_us"][ii].tolist(),image_available=available.tolist(),
    recovered=truth==1 and scope_truth==1 and q["qualified_display"]==1,
    new_process_seconds=time.perf_counter()-start,additional_DTW=0,
    learned_selected_chain_exact_time_truth=None,exact_boundary_accuracy_claim=False,
    limitation="Proposal witness containment is label-support bookkeeping, never a model completion score. A selected learned interval outside directly labelled scope has unknown local truth."))
 summary={}
 for role in ("fit","development"):
  vv=[r for r in transformed if r["role"]==role];aa=[r for r in automatic if r["role"]==role]
  denom=sum(r["truth"]==1 for r in aa);rec=sum(r["recovered"] for r in aa)
  per_query={}
  for query in QUERIES:
   ar=[r for r in aa if r["physical_query"]==query];d=sum(r["truth"]==1 for r in ar)
   per_query[query]=dict(total=len(ar),known_positive=d,recovered=sum(r["recovered"] for r in ar),
    proposal_without_reviewed_chain=sum(r["truth"]==1 and r["proposal_scope_truth"] is None for r in ar))
  summary[role]=dict(real_runtime_qualification=summarize(vv),
   oracle_only_known_edit_seam_diagnostic=summarize(vv,"oracle_known_seam_diagnostic"),
   per_query={q:summarize([r for r in vv if r["physical_query"]==q]) for q in QUERIES},
   per_kind={k:summarize([r for r in vv if r["kind"]==k]) for k in sorted({r["kind"] for r in vv})},
   per_person={str(person):summarize([r for r in vv if r["person"]==person]) for person in sorted({r["person"] for r in vv})},
   automatic=dict(total=len(aa),known_positive_denominator=denom,recovered=rec,recall=rec/denom if denom else None,
    per_query=per_query,unknown=sum(r["truth"] is None for r in aa),
    unknown_confirmed_positive=sum(r["truth"] is None and r["qualification"]["qualified_display"]==1 for r in aa),
    per_person={str(person):dict(known_positive=sum(r["truth"]==1 for r in aa if r["person"]==person),recovered=sum(r["recovered"] for r in aa if r["person"]==person)) for person in sorted({r["person"] for r in aa})},
    proposal_without_reviewed_positive_chain=sum(r["truth"]==1 and r["proposal_scope_truth"] is None for r in aa),
    raw_low_on_known_positive=sum(r["truth"]==1 and r["qualification"]["raw_display"]==0 for r in aa)))
 check_pins(contract)
 write(out,dict(status="SINGLE_CANDIDATE_QUALIFICATION_AND_AUTOMATIC_CONTEXT_EVALUATED_NOT_ADOPTION",view_rows=transformed,
  automatic_rows=automatic,summaries=summary,seconds=time.perf_counter()-t0,
  checkpoint_sha256=train["checkpoint_sha256"],contract_sha256=sha(U/"PROCESS_COMPARISON_FROZEN.json"),
  saved_raw_evaluation_sha256=sha(U/"PROCESS_EVALUATION.json"),old_models_executed=0,new_DTW=0,
  candidate_forward_budget=dict(repeated_raw_process_point=0,automatic_process_point=60,interval_batch_per_query_at_most=1,max_intervals_per_batch=8),
  baseline_full_process_score_available=False,whole_goal_complete=False))
 print(json.dumps(summary["development"],ensure_ascii=False))
if __name__=="__main__":
 try:main()
 except BaseException:
  p=U/"PROCESS_AUTO_EVALUATION_FIRST_FAILURE.txt"
  if not p.exists():p.write_text(traceback.format_exc(),encoding="utf-8")
  raise

