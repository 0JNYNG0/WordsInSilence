"""One saved-value decomposition at frozen thresholds; no inference/threshold search."""
from pathlib import Path
from collections import Counter,defaultdict
from statistics import mean,median
import json,hashlib
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw=read(U/"PROCESS_EVALUATION.json");auto=read(U/"PROCESS_AUTO_EVALUATION.json")
rr={r["view_id"]:r for r in raw["rows"]}
targets={r["view_id"]:r for fn in ("PROCESS_VIEW_TARGETS.json","DEV_PROCESS_VIEW_TARGETS.json") for r in read(U/fn)["rows"]}
def decomp(r,points):
 q=r["qualification"];cs=q["interval_candidates"]
 A=q["raw_probability"]<.8;B=not cs;C=bool(cs) and max(c["motion_probability"] for c in cs)<.8
 flags=[k for k,v in (("raw_below_0.8",A),("no_ordered_candidate",B),("selected_interval_below_0.8",C)) if v]
 av=r.get("image_available",rr[r["view_id"]]["image_available"] if "view_id" in r else [])
 ss=any(p[0]>=.8 and bool(a) for p,a in zip(points,av));tt=any(p[1]>=.8 and bool(a) for p,a in zip(points,av))
 return dict(id=r.get("view_id",r.get("query_id")),role=r["role"],person=r["person"],query=r["physical_query"],
  kind=r.get("kind","automatic"),truth=r["truth"],raw_probability=q["raw_probability"],
  candidate_count=len(cs),candidate_cap_reached=q["candidate_cap_reached"],
  max_selected_interval_probability=max((c["motion_probability"] for c in cs),default=None),
  any_setup_high=ss,any_terminal_high=tt,
  no_candidate_detail=("both_predicates_below" if not ss and not tt else "setup_below" if not ss else "terminal_below" if not tt else "no_same_observed_fragment_order") if B else None,
  flags=flags,exclusive_cause="+".join(flags) or "all_conditions_met",
  qualified=q["qualified_display"]==1,status=q["status"])
views=[decomp(r,rr[r["view_id"]]["point_probabilities"]) for r in auto["view_rows"]]
autos=[decomp(r,r["point_probabilities"]) for r in auto["automatic_rows"]]
def reduce_rows(rows):
 pos=[r for r in rows if r["truth"]==1]
 return dict(total=len(rows),positive=len(pos),negative=sum(r["truth"]==0 for r in rows),unknown=sum(r["truth"] is None for r in rows),
  positive_raw_high=sum(r["raw_probability"]>=.8 for r in pos),positive_qualified=sum(r["qualified"] for r in pos),
  positive_exclusive_causes=dict(Counter(r["exclusive_cause"] for r in pos)),
  positive_marginal_failure_counts={k:sum(k in r["flags"] for r in pos) for k in ("raw_below_0.8","no_ordered_candidate","selected_interval_below_0.8")},
  positive_candidate_absence_detail=dict(Counter(r["no_candidate_detail"] for r in pos if r["no_candidate_detail"])),
  positive_cap_reached=sum(r["candidate_cap_reached"] for r in pos),
  positive_more_than_one_failure=sum(len(r["flags"])>1 for r in pos),
  negative_qualified=sum(r["qualified"] for r in rows if r["truth"]==0),
  unknown_qualified=sum(r["qualified"] for r in rows if r["truth"] is None),
  dependent_source_queries=len({r["id"].split("__")[0] for r in rows}))
groups={}
for role in ("fit","development"):
 vs=[r for r in views if r["role"]==role];aa=[r for r in autos if r["role"]==role]
 groups[role]=dict(manual_all=reduce_rows(vs),manual_native=reduce_rows([r for r in vs if r["kind"]=="native"]),automatic_native=reduce_rows(aa),
  manual_by_query={q:reduce_rows([r for r in vs if r["query"]==q]) for q in sorted({r["query"] for r in vs})},
  automatic_by_query={q:reduce_rows([r for r in aa if r["query"]==q]) for q in sorted({r["query"] for r in aa})})
def metrics(records):
 pos=[p for y,p in records if y==1];neg=[p for y,p in records if y==0]
 return dict(known=len(pos)+len(neg),positive=len(pos),negative=len(neg),
  positive_raw_ge05=sum(p>=.5 for p in pos),positive_high_ge08=sum(p>=.8 for p in pos),
  negative_raw_ge05=sum(p>=.5 for p in neg),negative_falsehigh_ge08=sum(p>=.8 for p in neg),
  negative_low_le02=sum(p<=.2 for p in neg),
  positive_mean=mean(pos) if pos else None,negative_mean=mean(neg) if neg else None)
components={}
for role in ("fit","development"):
 rows=[r for r in raw["rows"] if r["role"]==role];per={}
 for field in ("setup","terminal"):
  records=[(v[field+"_truth"],v[field+"_probability"]) for r in rows for v in r["sparse_point_scores"] if v[field+"_truth"] is not None]
  native=[(v[field+"_truth"],v[field+"_probability"]) for r in rows if r["kind"]=="native" for v in r["sparse_point_scores"] if v[field+"_truth"] is not None]
  per[field]=dict(all=metrics(records),native_only=metrics(native),
   by_query={q:metrics([(v[field+"_truth"],v[field+"_probability"]) for r in rows if r["physical_query"]==q for v in r["sparse_point_scores"] if v[field+"_truth"] is not None]) for q in sorted({r["physical_query"] for r in rows})})
 per["annotated_motion"]=dict(all=metrics([(v["truth"],v["probability"]) for r in rows for v in r["interval_scores"] if v["truth"] is not None]),
  native_only=metrics([(v["truth"],v["probability"]) for r in rows if r["kind"]=="native" for v in r["interval_scores"] if v["truth"] is not None]),
  by_query={q:metrics([(v["truth"],v["probability"]) for r in rows if r["physical_query"]==q for v in r["interval_scores"] if v["truth"] is not None]) for q in sorted({r["physical_query"] for r in rows})})
 components[role]=per
manual_native={r["query_id"]:r for r in raw["rows"] if r["kind"]=="native"}
manual_decomp={r["id"].split("__")[0]:r for r in views if r["kind"]=="native"}
paired=[]
for a,d in zip(auto["automatic_rows"],autos):
 n=manual_native[a["query_id"]];nd=manual_decomp[a["query_id"]]
 paired.append(dict(query_id=a["query_id"],role=a["role"],person=a["person"],physical_query=a["physical_query"],truth=a["truth"],
  manual_frames=n["frames"],auto_frames=len(a["source_frame_ids"]),manual_p=n["probability"],auto_p=a["probability"],
  manual_qualified=nd["qualified"],auto_qualified=d["qualified"],manual_candidates=nd["candidate_count"],auto_candidates=d["candidate_count"],
  manual_cause=nd["exclusive_cause"],auto_cause=d["exclusive_cause"],proposal_contains_reviewed_chain=a["proposal_contains_reviewed_positive_chain"]))
paired_summary={}
for role in ("fit","development"):
 pp=[p for p in paired if p["role"]==role and p["truth"]==1]
 paired_summary[role]=dict(known_positive=len(pp),manual_high=sum(p["manual_p"]>=.8 for p in pp),auto_high=sum(p["auto_p"]>=.8 for p in pp),
  high_to_not_high=sum(p["manual_p"]>=.8 and p["auto_p"]<.8 for p in pp),
  not_high_to_high=sum(p["manual_p"]<.8 and p["auto_p"]>=.8 for p in pp),
  qualified_transitions=dict(Counter(str(p["manual_qualified"])+"->"+str(p["auto_qualified"]) for p in pp)),
  manual_frame_median=median(p["manual_frames"] for p in pp),auto_frame_median=median(p["auto_frames"] for p in pp),
  raw_p_delta_median=median(p["auto_p"]-p["manual_p"] for p in pp),proposal_misses=sum(not p["proposal_contains_reviewed_chain"] for p in pp))
false=[]
for r in auto["view_rows"]:
 if r["truth"]==0 and r["qualification"]["qualified_display"]==1:
  saved=rr[r["view_id"]];t=targets[r["view_id"]];c=r["qualification"]["witnessed_chain"]
  false.append(dict(view_id=r["view_id"],query=r["physical_query"],p=saved["probability"],qualification=r["qualification"],oracle=r["oracle_known_seam_diagnostic"],
   source_frames=saved["source_frames"],source_times_us=saved["source_timestamps_us"],view_times_us=saved["view_timestamps_us"],
   selected_chain_source_frames=[saved["source_frames"][i] for i in c["output_indices"]],seam_output_indices=t["seam_output_indices"],
   truth_reason=t["evidence_reason"],annotated_interval_diagnostics=saved["interval_scores"],raw_point_scores=saved["sparse_point_scores"]))
retime=[]
for r in raw["rows"]:
 if r["kind"].startswith("speed_"):
  n=manual_native[r["query_id"]]
  assert r["source_frames"]==n["source_frames"]
  retime.append(dict(id=r["view_id"],role=r["role"],person=r["person"],p=r["probability"],native_p=n["probability"],
    p_delta=r["probability"]-n["probability"],point_max_absolute_delta=max(abs(a-b) for ap,bp in zip(r["point_probabilities"],n["point_probabilities"]) for a,b in zip(ap,bp))))
retime_summary={role:dict(views=sum(r["role"]==role for r in retime),source_queries=len({r["id"].split("__")[0] for r in retime if r["role"]==role}),
  max_p_absolute_delta=max(abs(r["p_delta"]) for r in retime if r["role"]==role),
  max_point_absolute_delta=max(r["point_max_absolute_delta"] for r in retime if r["role"]==role)) for role in ("fit","development")}
special=[dict(id=r["view_id"],truth=r["truth"],p=rr[r["view_id"]]["probability"],qualified=r["qualification"]["qualified_display"],cause=next(d["exclusive_cause"] for d in views if d["id"]==r["view_id"])) for r in auto["view_rows"] if r["role"]=="fit" and r["truth"]==1 and r["kind"] in ("prefix","suffix","middle_splice","missing_rgb")]
out=dict(status="SAVED_VALUE_FIXED_THRESHOLD_DECOMPOSITION",groups=groups,component_diagnostics=components,paired_native_summary=paired_summary,
 paired_native_rows=paired,positive_manual_rows=[r for r in views if r["truth"]==1],positive_auto_rows=[r for r in autos if r["truth"]==1],
 false_confirmations=false,retime_summary=retime_summary,retime_rows=retime,alternative_surviving_process_positives=special,
 source_pins={str(U/n):sha(U/n) for n in ("PROCESS_EVALUATION.json","PROCESS_AUTO_EVALUATION.json","PROCESS_VIEW_TARGETS.json","DEV_PROCESS_VIEW_TARGETS.json","PROCESS_COMPARISON_FROZEN.json","PROCESS_EVALUATION_SAVED_VERIFIED.json","PROCESS_AUTO_EVALUATION_VERIFIED.json")},
 new_model_forwards=0,new_training=0,threshold_sweep=False,old_model_results_regraded=False,
 scope="Arithmetic decomposition of this saved candidate at predeclared.8; no alternative decision rule, model, or prediction generation.")
dest=U/"PROCESS_CAUSE_COUNTS.json";assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(groups={k:{g:v[g] for g in ("manual_all","manual_native","automatic_native")} for k,v in groups.items()},
 component_overall={k:{j:v[j]["all"] for j in v} for k,v in components.items()},paired=paired_summary,retime=retime_summary,alternate=special),ensure_ascii=False))

