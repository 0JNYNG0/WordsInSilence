"""Saved new-geometry bridge audit. Reuses independent arithmetic, not producers."""
from pathlib import Path
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import sys
import traceback
import numpy as np
sys.dont_write_bytecode=True
U=Path(__file__).resolve().parents[1]
O=U.parent
CASE=O/"word100_action_feedback_casebook_20261101"
HAND=O/"word100_independent_hand_evidence_20261101"


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);out=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out);return out


ARITHMETIC=HAND/"review/independent_arithmetic.py"
SUMMARY=CASE/"review/verify_casebook.py"
arithmetic=module(ARITHMETIC,"bridge_independent_arithmetic")
summary_checks=module(SUMMARY,"bridge_saved_summary_audit")


def read(p):
    p=Path(p)
    return json.loads(gzip.decompress(p.read_bytes())) if p.suffix==".gz" else json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in ("x","mask","frame_id","timestamp_us")}


def run():
    a=summary_checks.Audit()
    manifest=read(U/"BRIDGE_MANIFEST.json");frozen=read(U/"BRIDGE_FROZEN.json")
    tasks=read(CASE/"INPUTS.json");recovery=read(U/"RECOVERY_RESULT_MANIFEST.json")
    original_results={r["id"]:r for r in read(CASE/"RESULT_MANIFEST.json")["results"] if r["kind"]=="natural"}
    outputs={r["id"]:r for r in manifest["records"]}
    a.equal(set(outputs),{t["id"] for t in tasks["natural"]},"exact39 parent identities")
    a.equal(len(manifest["records"]),39,"all39 parent rows")
    a.equal(len({(r["query"],r["reference"]) for r in outputs.values()}),12,"all12 pairs")
    base={alias:load(row["input"]) for alias,row in tasks["rows"].items()}
    candidate={alias:{k:v.copy() for k,v in src.items()} for alias,src in base.items()}
    for rec in recovery["records"]:
        alias=rec["alias"];z=load(rec["output"]);b=base[alias];d=candidate[alias]
        a.equal(sha(rec["output"]),rec["output_sha256"],alias+":recovery input hash")
        positions={int(fid):i for i,fid in enumerate(b["frame_id"])}
        for j,fid in enumerate(z["frame_id"]):
            i=positions[int(fid)]
            a.equal(int(z["timestamp_us"][j]),int(b["timestamp_us"][i]),alias+":overlay native time")
            a.check(np.array_equal(z["x"][j,b["mask"][i]],b["x"][i,b["mask"][i]]),alias+":old supported values locked")
            a.check(z["mask"][j,b["mask"][i]].all(),alias+":old support locked")
            d["x"][i]=z["x"][j];d["mask"][i]=z["mask"][j]
    old_counts,new_counts=Counter(),Counter()
    def tally(dst,e):
        for g in e["groups"].values():
            for axis in ("query","reference"):dst.update(g[axis]["counts"])
    changed=0
    for t in tasks["natural"]:
        ident=t["id"];record=outputs[ident];old_geometry=read(t["hand_record"]["path"])
        old_case=read(original_results[ident]["path"]);eold=old_geometry["evidence"]
        qa,qb=t["query_range"];ra,rb=t["reference_range"]
        ranges=[(t["query"],qa,qb),(t["reference"],ra,rb)]
        affected=any(not np.array_equal(candidate[alias][key][lo:hi+1],base[alias][key][lo:hi+1])
                     for alias,lo,hi in ranges for key in ("x","mask"))
        tally(old_counts,eold)
        if not affected:
            a.equal(record["mode"],"REUSED_UNCHANGED_BASELINE",ident+":unchanged parent not recomputed")
            a.equal(record["geometry"],t["hand_record"],ident+":exact reused geometry pointer")
            a.equal(record["result"],original_results[ident],ident+":exact reused summary pointer")
            a.equal(sha(record["result"]["path"]),record["result"]["sha256"],ident+":reused summary hash")
            tally(new_counts,eold);continue
        changed+=1
        a.equal(record["mode"],"NEW_RECOVERED_GEOMETRY_FIXED_OLD_PATH",ident+":new observation condition")
        geometry=a.artifact(record["geometry"],ident+":new geometry")
        result=a.artifact(record["result"],ident+":new case")
        enew=geometry["evidence"]
        a.equal(result["source"],record["geometry"],ident+":summary exact geometry source")
        a.equal(geometry["baseline_geometry"],t["hand_record"],ident+":baseline remains immutable source")
        a.equal(geometry["frozen_sha256"],sha(U/"BRIDGE_FROZEN.json"),ident+":new geometry freeze")
        a.equal(enew["path"],eold["path"],ident+":same original local path")
        q={k:v[qa:qb+1].copy() for k,v in candidate[t["query"]].items()}
        r={k:v[ra:rb+1].copy() for k,v in candidate[t["reference"]].items()}
        arithmetic.verify_evidence(q,r,eold["path"],enew,
            lambda name,ok,identity:a.check(bool(ok),identity+":"+name),ident)
        for name,oldgroup in eold["groups"].items():
            newgroup=enew["groups"][name]
            old_shared=np.asarray(oldgroup["links"]["shared_support"],bool)
            new_shared=np.asarray(newgroup["links"]["shared_support"],bool)
            a.check(new_shared[old_shared].all(),ident+":"+name+":old shared links retained")
            for field in ("signed_difference_deg","absolute_difference_deg"):
                oldvalues=np.asarray(oldgroup["links"][field],float)
                newvalues=np.asarray(newgroup["links"][field],float)
                a.check(np.array_equal(oldvalues[old_shared],newvalues[old_shared]),ident+":"+name+":original shared link value exact "+field)
            for axis in ("query","reference"):
                oldmask=np.asarray(oldgroup[axis]["own_observed"],bool)
                newmask=np.asarray(newgroup[axis]["own_observed"],bool)
                a.check(newmask[oldmask].all(),ident+":"+name+":"+axis+":original own support retained")
                a.equal(newgroup[axis]["possible_shape"],oldgroup[axis]["possible_shape"],ident+":"+name+":"+axis+":fixed frame feature denominator")
        summary_checks.verify_summary(a,geometry,result["summary"],ident+":summary")
        a.equal(result["summary"]["description_support"]["provided"],old_case["summary"]["description_support"]["provided"],ident+":same reference description")
        a.equal(result["summary"]["correspondence_support"]["provided"],old_case["summary"]["correspondence_support"]["provided"],ident+":same correspondence caveat")
        a.equal(result["summary"]["visible_difference_review"]["status"],"NOT_REVIEWED",ident+":old RGB review not new geometry truth")
        a.equal(result["summary"]["visible_difference_review"]["provided"],None,ident+":old RGB not promoted")
        for k in ("old_learned_shape_outputs_are_baseline_only","old_position_results_are_baseline_only"):
            a.equal(geometry[k],True,ident+":"+k)
        for k in ("shape_sources","position_sources","shape_guide_link"):
            a.equal(geometry[k],old_geometry[k],ident+":original auxiliary source "+k)
        a.equal(result["legacy_shape_and_position_outputs_role"],"BASELINE_ONLY_NOT_REINFERRED_FROM_RECOVERED_INPUT",ident+":old outputs kept separate")
        a.equal(result["educational_verdict"],None,ident+":no educational verdict")
        tally(new_counts,enew)
    a.equal(changed,manifest["changed_parents"],"changed count independently reconstructed")
    a.equal(39-changed,manifest["unchanged_reused"],"unchanged reuse count")
    a.equal(changed,manifest["new_coordinate_comparisons"],"only changed inputs compared")
    a.equal(dict(old_counts),manifest["baseline_counts"],"baseline accounting")
    a.equal(dict(new_counts),manifest["candidate_counts"],"new accounting")
    a.equal(old_counts["possible"],30560,"original fixed both-axis denominator")
    a.equal(new_counts["possible"],30560,"new fixed both-axis denominator")
    for k in ("old_comparisons_rerun","new_dtw","new_training","new_classifier_inference"):
        a.equal(manifest[k],0,"zero "+k)
    for q in ("P036","P048","P066","P072"):
        a.equal(len({r["reference"] for r in outputs.values() if r["query"]==q}),3,q+":all references retained")
    for p,h in frozen["pins"].items():a.equal(sha(p),h,"frozen source "+Path(p).name)
    a.counts.update({"new_geometry_checked":changed,"unchanged_baseline_reused":39-changed,
                     "fixed_parents":39,"fixed_both_axis_cells":30560})
    return a


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=U/"review/BRIDGE_VERIFICATION.json");args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    try:a=run()
    except Exception:
        a=summary_checks.Audit();a.check(False,"verifier exception",traceback.format_exc())
    result={"status":"PASS" if not a.failures else "FAIL","checks":a.checks,"failures":a.failures,"counts":dict(a.counts),
            "verifier_sha256":sha(__file__),"independent_library_pins":{str(p):sha(p) for p in (ARITHMETIC,SUMMARY)},
            "method":"Only new changed-parent arithmetic independently checked; unchanged baseline referenced by hash. Producer compare/summarize/DTW/detector not imported or called.",
            "new_inference":0,"new_DTW":0,"old_comparison_reruns":0,"educational_accuracy":None}
    with args.output.open("x",encoding="utf-8") as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
    print(json.dumps({"status":result["status"],"checks":a.checks,"failures":len(a.failures),"output":str(args.output)}))
    raise SystemExit(0 if not a.failures else 1)
