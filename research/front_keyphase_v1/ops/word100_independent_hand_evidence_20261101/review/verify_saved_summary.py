"""Verify published summary from saved per-link/per-axis arrays and source pointers."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,gzip,json,traceback
import numpy as np

R=Path(__file__).resolve().parent;U=R.parent
pins={};checks=0;failures=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 p=Path(p);b=p.read_bytes()
 return json.loads(gzip.decompress(b) if p.suffix==".gz" else b.decode("utf-8-sig"))
def check(ok,label):
 global checks
 checks+=1
 if not ok:failures.append(label)
def pin(p,expected=None):
 p=str(Path(p).resolve());h=sha(p)
 check(expected is None or h==expected,"SHA "+p);pins[p]=h
def same(a,b,label):
 if isinstance(b,dict):
  check(isinstance(a,dict) and set(a)==set(b),label+" keys")
  if isinstance(a,dict):
   for k in set(a)&set(b):same(a[k],b[k],label+"/"+str(k))
 elif isinstance(b,list):
  check(isinstance(a,list) and len(a)==len(b),label+" length")
  if isinstance(a,list):
   for i,(x,y) in enumerate(zip(a,b)):same(x,y,label+"/"+str(i))
 elif isinstance(b,float):check(a is not None and np.isclose(a,b,rtol=1e-12,atol=1e-10),label)
 else:check(a==b,label)

def main():
 for name in ("RESULT_MANIFEST.json","EVALUATION.json","INPUTS.json","FROZEN.json","review/SHAPE_GUIDE_LINKS.json"):
  pin(U/name)
 evaluation=read(U/"EVALUATION.json");manifest=read(U/"RESULT_MANIFEST.json");inputs=read(U/"INPUTS.json")
 tasks={t["id"]:t for t in inputs["natural"]+inputs["technical"]}
 guide_path=R/"SHAPE_GUIDE_LINKS.json";guide=read(guide_path)
 tally=defaultdict(Counter);conditions=defaultdict(Counter);natural=Counter()
 summaries=[];links=Counter();axes=Counter();expected_tol=1e-8
 check(len(manifest["results"])==len(tasks)==927,"927 full output scope")
 for item in manifest["results"]:
  pin(item["path"],item["sha256"]);d=read(item["path"]);t=tasks[item["id"]];e=d["evidence"];path=e["path"]
  if d["kind"]=="natural":
   ptr=d["shape_guide_link"]
   check(ptr["path"]==str(guide_path) and ptr["sha256"]==sha(guide_path) and
         ptr["source_expectation_not_semantic_label"] is True,d["id"]+": guide exact source")
   parts=ptr["json_pointer"].split("/")
   check(parts[:2]==["","records"] and len(parts)==3,d["id"]+": guide JSON pointer shape")
   row=guide["records"][int(parts[2])]
   check((row["query"],row["reference"],row["parent_id"])==(d["query"],d["reference"],d["parent_id"]),
         d["id"]+": exact parent guide pointer")
   check(row["query_range"]==d["query_range"] and row["reference_range"]==d["reference_range"],d["id"]+": guide range")
   check([s["readout"] for s in row["seed_links"]]==d["shape_sources"],d["id"]+": guide351 seed links")
   check([{k:p[k] for k in ("id","path","sha256")} for p in d["position_sources"]]==row["continuous_position_records"],
         d["id"]+": guide585 immutable position links")
   links["guide_parent_pointers"]+=1;links["shape_seed_links"]+=len(row["seed_links"]);links["positions"]+=len(d["position_sources"])
  else:check(d["shape_guide_link"] is None,d["id"]+": no synthetic model/guide claim")
  for name,g in e["groups"].items():
   for side in ("query","reference"):
    z=g[side];own=np.asarray(z["own_observed"],bool);matched=np.asarray(z["matched_support"],bool);partial=np.asarray(z["mixed_counterpart_missing"],bool)
    counts=dict(possible=int(own.size),own_observed=int(own.sum()),own_unobserved=int((~own).sum()),
     matched_support=int(matched.sum()),direct_complete=int((matched&~partial).sum()),
     partial_shared=int(partial.sum()),not_comparable=int((~matched).sum()),
     observed_without_shared_counterpart=int((own&~matched).sum()))
    same(z["counts"],counts,d["id"]+":"+side+name+" raw boolean counts")
    tally[(d["kind"],name,side)].update(counts);axes[d["kind"]+"_"+side]+=own.size
   if d["kind"]=="natural":
    v=[v for row in g["links"]["absolute_difference_deg"] for v in row if v is not None]
    natural[name+"_supported_records"]+=bool(v);natural[name+"_positive_records"]+=bool(v and max(v)>expected_tol)
  if d["kind"]=="technical":
   c=t["edit"]["condition"];hand=t["edit"]["hand"]
   name=hand+("_finger_angles14" if c in ("finger_change","one_finger_missing") else "_hand_axis2d")
   dim=2 if name.endswith("finger_angles14") else 0
   vals=[]
   for k,(qi,ri) in enumerate(path):
    if c=="short_axis_change" and t["edit"]["axis_delta_degrees"][qi]==0:continue
    v=e["groups"][name]["links"]["absolute_difference_deg"][k][dim]
    if v is not None:vals.append(v)
   conditions[c]["opportunities"]+=1
   conditions[c]["has_shared_target_evidence"]+=bool(vals)
   conditions[c]["no_shared_target_evidence"]+=not bool(vals)
   conditions[c]["has_measured_target_difference"]+=bool(vals and max(vals)>expected_tol)
   summaries.append(dict(id=d["id"],condition=c,hand=hand,shared_target_cells=len(vals),
                         target_peak_degrees=max(vals) if vals else None,real_motion_error=None))
 expected_tally={k:dict(v) for k,v in tally.items()}
 actual_tally={(r["kind"],r["group"],r["axis"]):{k:v for k,v in r.items() if k not in ("kind","group","axis")} for r in evaluation["axis_counts"]}
 same(actual_tally,expected_tally,"Published all-axis counts from saved masks")
 same(evaluation["technical_conditions"],{k:dict(v) for k,v in conditions.items()},"All9 conditions from saved links")
 same(evaluation["technical_record_summary"],summaries,"All810 technical source-link summaries")
 same(evaluation["natural_positive_readout_records"],dict(natural),"All117 natural positive counts from saved links")
 same(links,dict(guide_parent_pointers=117,shape_seed_links=351,positions=585),"Exact immutable source joins")
 check(all(v["opportunities"]==90 for v in conditions.values()) and len(conditions)==9,"810 opportunity denominator including no-evidence cases")
 expected_axes=Counter()
 for t in inputs["natural"]+inputs["technical"]:
  for side in ("query","reference"):
   span=t[side+"_range"];expected_axes[t["kind"]+"_"+side]+=(span[1]-span[0]+1)*32
 same(dict(axes),dict(expected_axes),"Full32-dimensional both-axis denominators")
 for k,v in dict(natural_records=117,technical_records=810,prior_position_results_reused=585,prior_shape_seed_links=351,
                 new_inference=0,new_training=0,new_dtw=0,no_automatic_correction=True,
                 educational_accuracy=None,normal_rotation_crossing_accuracy=None,failures=[]).items():
  same(evaluation[k],v,"Published scope/limits "+k)
 same(evaluation["timing"],{k:manifest[k] for k in ("compute_seconds","wall_seconds","end_to_end_latency")},"Measured timing copy")
 for p,h in pins.items():check(sha(p)==h,"Source stable through summary audit "+p)
 return dict(source_links=dict(links),all_axis_cells=dict(axes),conditions={k:dict(v) for k,v in conditions.items()},
             natural_positive_readout_records=dict(natural))

if __name__=="__main__":
 output=R/"RESULT_SUMMARY_VERIFICATION.json"
 if output.exists():raise SystemExit("Preserve first summary audit; refuse overwrite")
 error=None;result=None
 try:result=main()
 except Exception:error=traceback.format_exc();failures.append("Verifier exception")
 report=dict(status="PASS" if not failures else "FAIL",checks=checks,failure_count=len(failures),failures=failures[:50],
   exception=error,results=result,at_utc=datetime.now(timezone.utc).isoformat(),pins=pins,script_sha256=sha(__file__),
   method="Independent saved explicit boolean arrays and per-link residual reduction; no readout/feature/model/path producer execution.",
   limits=["Positive natural differences are normal-reference performance differences, not educational errors.",
           "No-evidence technical targets remain in denominator and are not a successful detection.",
           "Summary validates evidence bookkeeping, not correctness of old alignment or3D hand interpretation."],
   actions=dict(model_inference=0,feature_extraction=0,path_search=0,position_recalculation=0,original_writes=0))
 with output.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
 print(json.dumps(dict(status=report["status"],checks=checks,failures=len(failures),sha256=sha(output))))
 if failures:raise SystemExit(1)
