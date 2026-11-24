"""Saved within-view annotated-vs-selected interval scope comparison; no new scores."""
from pathlib import Path
import json,hashlib
from collections import Counter
from statistics import median
U=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding="utf-8-sig"))
raw=read(U/"PROCESS_EVALUATION.json");auto=read(U/"PROCESS_AUTO_EVALUATION.json");counts=read(U/"PROCESS_CAUSE_COUNTS.json")
index={r["view_id"]:r for r in raw["rows"]}
rows=[]
for r in auto["view_rows"]:
 if r["truth"]!=1:continue
 saved=index[r["view_id"]];cs=r["qualification"]["interval_candidates"];positive=[v for v in saved["interval_scores"] if v["truth"]==1]
 if not cs or not positive:continue
 overlaps=[]
 for c in cs:
  ci=c["output_indices"]
  for a in positive:
   ai=a["output_indices"]
   if ci[0]<=ai[0] and ci[-1]>=ai[-1]:
    overlaps.append(dict(selected_indices=[ci[0],ci[-1]],annotated_indices=[ai[0],ai[-1]],
     selected_source_extent=[saved["source_frames"][ci[0]],saved["source_frames"][ci[-1]]],
     annotated_source_extent=[saved["source_frames"][ai[0]],saved["source_frames"][ai[-1]]],
     selected_length=len(ci),annotated_length=len(ai),extra_prefix_frames=ai[0]-ci[0],extra_suffix_frames=ci[-1]-ai[-1],
     selected_probability=c["motion_probability"],annotated_probability=a["probability"],same_exact_interval=ci==ai,
     selected_source_contiguous=all(saved["source_frames"][i+1]==saved["source_frames"][i]+1 for i in range(ci[0],ci[-1]))))
 rows.append(dict(id=r["view_id"],role=r["role"],query=r["physical_query"],kind=r["kind"],
  max_annotated=max(a["probability"] for a in positive),max_selected=max(c["motion_probability"] for c in cs),
  selected_lengths=[len(c["output_indices"]) for c in cs],annotated_positive_lengths=[len(a["output_indices"]) for a in positive],
  containing_pairs=overlaps))
summary={}
for role in ("fit","development"):
 rr=[r for r in rows if r["role"]==role];pp=[p for r in rr for p in r["containing_pairs"]]
 strict=[p for p in pp if not p["same_exact_interval"] and p["selected_source_contiguous"]]
 summary[role]=dict(positive_views_with_both_scopes=len(rr),
  high_annotated_but_all_selected_low=sum(r["max_annotated"]>=.8 and r["max_selected"]<.8 for r in rr),
  all_annotated_and_selected_high=sum(r["max_annotated"]>=.8 and r["max_selected"]>=.8 for r in rr),
  selected_interval_length_median=median([x for r in rr for x in r["selected_lengths"]]) if rr else None,
  annotated_positive_interval_length_median=median([x for r in rr for x in r["annotated_positive_lengths"]]) if rr else None,
  containing_pairs=len(pp),strict_contiguous_superset_pairs=len(strict),
  strict_superset_highannotation_lowselected=sum(p["annotated_probability"]>=.8 and p["selected_probability"]<.8 for p in strict),
  strict_superset_prefix_median=median(p["extra_prefix_frames"] for p in strict) if strict else None,
  strict_superset_length_median=median(p["selected_length"] for p in strict) if strict else None,
  same_exact_pairs=sum(p["same_exact_interval"] for p in pp),
  same_exact_max_probability_difference=max([abs(p["annotated_probability"]-p["selected_probability"]) for p in pp if p["same_exact_interval"]],default=None))
special=[]
target_index={r["view_id"]:r for r in read(U/"PROCESS_VIEW_TARGETS.json")["rows"]}
for p in counts["alternative_surviving_process_positives"]:
 r=target_index[p["id"]];saved=index[p["id"]]
 witnesses={side:r.get("bound_"+side+"_witnesses",[]) for side in ("setup","terminal")}
 sparse={v["output_index"]:v for v in r["sparse_relation_targets"]}
 extra={}
 for side,ws in witnesses.items():
  extra[side]=[]
  for w in ws:
   i=w["output_index"];t=sparse.get(i)
   extra[side].append(dict(output_index=i,source_frame=w.get("source_frame",w.get("source_native_frame")),
     source_direct_witness=w,point_supervision=t.get(side+"_relation") if t else None,
     predicted_probability=saved["point_probabilities"][i][0 if side=="setup" else 1]))
 special.append(dict(id=p["id"],direct_witnesses_vs_sparse_point_training=extra))
out=dict(status="SAVED_SCOPE_DIAGNOSTIC_ONLY",summary=summary,rows=rows,alternative_direct_phase_witness_audit=special,
 inference=0,new_scores=0,threshold_search=False)
p=U/"PROCESS_SCOPE_COUNTS.json";assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
false=counts["false_confirmations"][0]
print(json.dumps(dict(summary=summary,false={k:false[k] for k in ("view_id","p","selected_chain_source_frames","seam_output_indices","truth_reason")},
 false_qualification=false["qualification"],native_components={role:{name:obj["native_only"] for name,obj in data.items()} for role,data in counts["component_diagnostics"].items()},
 alternative_phase=special,
 illustrative_supersets=[dict(id=r["id"],pairs=[p for p in r["containing_pairs"] if p["annotated_probability"]>=.8 and p["selected_probability"]<.8 and p["selected_source_contiguous"]]) for r in rows if any(p["annotated_probability"]>=.8 and p["selected_probability"]<.8 and p["selected_source_contiguous"] for p in r["containing_pairs"])][:6]),ensure_ascii=False))

