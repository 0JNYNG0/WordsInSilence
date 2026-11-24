"""Independent saved-source/transform/scope audit. No producer or solver imports."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import traceback
import numpy as np

HERE=Path(__file__).resolve().parent
U=HERE.parent
OPS=U.parent
HOLD=OPS/"word100_hold_invariant_trajectory_20261101"
CONT=OPS/"word100_continuous_trajectory_20261101"
checks=0
failures=[]
pins={}
totals=Counter()

def read(p):
    p=Path(p);b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix==".gz" else b.decode("utf-8-sig"))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def check(ok, label):
    global checks
    checks+=1
    if not ok:failures.append(label)

def require(ok,label):
    check(ok,label)
    if not ok:raise AssertionError(label)

def pin(p,h=None):
    p=str(Path(p).resolve());actual=sha(p)
    check(h is None or actual==h,"SHA "+p)
    if p in pins:check(pins[p]==actual,"Unchanged concurrent source "+p)
    pins[p]=actual

def array_equal(a,b,label):
    check(np.array_equal(a,b,equal_nan=True),label)

def load(row,span=None):
    with np.load(row["input"],allow_pickle=False) as z:
        a={k:z[k].copy() for k in ("x","mask","frame_id","timestamp_us")}
    if span:
        a={k:v[span[0]:span[1]+1] for k,v in a.items()}
    a["x"]=a["x"].astype(np.float64)
    return a

def verify_transform(t,base):
    with np.load(t["input"]["path"],allow_pickle=False) as z:
        q={k:z[k].copy() for k in ("x","mask","frame_id","timestamp_us")}
    for k in ("frame_id","timestamp_us"):array_equal(q[k],base[k],t["id"]+": native "+k)
    off={"L":0,"R":28}[t["edit"]["hand"]];condition=t["edit"]["condition"]
    n=len(base["x"]);expected=base["x"].copy();mask=base["mask"].copy()
    theta=np.zeros(n)
    if condition in ("axis_rotation","common_arm_hand_rotation"):theta.fill(np.pi/6)
    if condition=="short_axis_change":theta[max(0,n//2-2):min(n,max(0,n//2-2)+5)]=np.pi/6
    if condition=="axis_jitter":theta=np.array([np.pi/360 if i%2==0 else -np.pi/360 for i in range(n)])
    cols=[]
    if np.any(theta):
        z=base["x"][:,off+25]+1j*base["x"][:,off+26]
        rotated=z*np.exp(1j*theta)
        expected[:,off+25]=rotated.real;expected[:,off+26]=rotated.imag
        cols=[off+25,off+26]
        if condition=="common_arm_hand_rotation":
            arm=(base["x"][:,off+2]-base["x"][:,off+4])+1j*(base["x"][:,off+3]-base["x"][:,off+5])
            arm*=np.exp(1j*theta)
            expected[:,off+4]=base["x"][:,off+2]-arm.real
            expected[:,off+5]=base["x"][:,off+3]-arm.imag
            cols += [off+4,off+5]
    if condition=="finger_change":
        j=off+13
        for i in range(n):expected[i,j]+=0.1 if base["x"][i,j]<=0.9 else -0.1
        cols=[j]
    if condition=="common_translation":
        cols=[j for o in (0,28) for j in range(o,o+10,2)]
        expected[:,cols]+=0.2
    if condition=="axis_missing":mask[:,off+25:off+27]=False;cols=[off+25,off+26]
    if condition=="one_finger_missing":mask[:,off+13]=False;cols=[off+13]
    check(t["edit"]["modified_columns"]==sorted(cols),t["id"]+": explicit edit columns")
    check(np.allclose(q["x"],expected,rtol=1e-13,atol=1e-13,equal_nan=True),t["id"]+": independent transformed69 values")
    array_equal(q["mask"],mask,t["id"]+": exact edit mask")
    untouched=sorted(set(range(69))-set(cols))
    array_equal(q["x"][:,untouched],base["x"][:,untouched],t["id"]+": all unedited columns exact")
    check(np.allclose(t["edit"]["axis_delta_degrees"],theta*180/np.pi,rtol=1e-13,atol=1e-13),t["id"]+": edit magnitude/short scope")
    check(t["edit"]["shape_probabilities_reused"] is False and
          t["edit"]["feature_space_only_not_physical_video"] is True and
          t["no_saved_control_model_outputs"] is True,t["id"]+": synthetic label/output limits")
    check(t["path_origin"]=="predeclared_identity_not_estimated_alignment",t["id"]+": identity technical path")
    totals["technical_"+condition]+=1
    return q

def prepare_audit():
    inp=read(U/"INPUTS.json");old=read(HOLD/"TASKS.json")
    plan=read(U/"PLAN.json")
    audit=read(HERE/"INPUT_AUDIT.json")
    for p in (U/"INPUTS.json",U/"PLAN.json",U/"pipeline.py",U/"hand_evidence.py",
              HERE/"INPUT_AUDIT.json",HERE/"SHAPE_GUIDE_LINKS.json",HERE/"RGB_VISIBILITY_REVIEW.json"):
        pin(p)
    for p,h in {**inp["pins"],**audit["pins"],**audit["protected"]}.items():pin(p,h)
    check(inp["rows"]==old["rows"],"All16 original source row descriptors")
    check(inp["denominators"]==dict(clips=16,pairs=36,natural_parents=117,position_records=585,technical_parents=45,technical_opportunities=810),
          "Full prespecified denominators")
    require(len(inp["natural"])==117 and len(inp["technical"])==810,"Natural/technical inventories")
    require(len({t["id"] for t in inp["natural"]+inp["technical"]})==927,"Unique927 tasks")
    old_by={(t["query"],t["reference"],t["parent"]["segment_id"]):t for t in old["natural"]}
    shape=set();positions=set();natural_pairs=set()
    original_arrays={a:load(row) for a,row in inp["rows"].items()}
    for a,ar in original_arrays.items():
        check(len(ar["x"])==inp["rows"][a]["frames"] and ar["x"].shape[1]==69,a+": original69 schema")
        check(ar["mask"].dtype==np.bool_ and ar["mask"].shape==ar["x"].shape,a+": originalmask schema")
        check(np.array_equal(ar["frame_id"],np.arange(len(ar["x"]))),a+": nativeordinal identity")
    for t in inp["natural"]:
        key=t["query"],t["reference"],t["parent"]["segment_id"]
        original=old_by[key];check(all(t[k]==v for k,v in original.items()),t["id"]+": immutable old natural descriptor")
        natural_pairs.add(key[:2])
        geom=read(t["source_geometry"]["path"])
        check(t["baseline_full_path"]==geom["baseline_full_path"],t["id"]+": saved geometry path")
        full=read(t["baseline"])["path"];lo,hi=t["reference_range"]
        crop=[p for p in full if lo<=p[1]<=hi]
        check(crop==t["baseline_full_path"],t["id"]+": exact contiguous old global crop")
        check(t["query_range"]==[min(p[0] for p in crop),max(p[0] for p in crop)],t["id"]+": query scope derives from old path")
        check({p[0] for p in crop}==set(range(t["query_range"][0],t["query_range"][1]+1)) and
              {p[1] for p in crop}==set(range(lo,hi+1)),t["id"]+": full bothaxes source coverage")
        check(inp["rows"][t["reference"]]["role"]=="fit",t["id"]+": FIT reference")
        check([r["component"] for r in t["position_sources"]]==old["component_names"],t["id"]+": five unchanged components")
        for link in t["position_sources"]:
            pin(link["path"],link["sha256"]);v=read(link["path"]);positions.add(link["path"])
            check((v["query"],v["reference"],v["parent_id"],v["component"])==key+(link["component"],),t["id"]+": position link identity")
            pin(v["source_record"],v["source_sha256"])
            src=read(v["source_record"])
            check(src["query_range"]==t["query_range"] and src["reference_range"]==t["reference_range"] and
                  src["baseline_full_path"]==crop,t["id"]+": unchanged position scope")
        for s,p in zip((42,43,44),t["source_readouts"]):
            v=read(p);shape.add(p)
            check(v["query"]==t["query"] and v["reference"]==t["reference"] and v["seed"]==s,
                  t["id"]+": seed-specific old shape evidence")
            check(v["path"]==full,t["id"]+": old shape/global path")
        totals["position_links"]+=5;totals["shape_parent_seed_links"]+=3
    check(len(positions)==585 and len(shape)==108 and len(natural_pairs)==36,"Natural unique source inventory")
    phase=read(OPS/"word100_core_stage_evidence_20261031/PHASE_DATA.json")
    parents={(c["alias"],p["segment_id"]):(c,p) for c in phase["clips"] for p in c["segments"] if p["role"]=="core_candidate"}
    expected={(a,p,h,c) for (a,p) in parents for h in ("L","R") for c in plan["technical"]["conditions"]}
    actual={(t["query"],t["parent"]["segment_id"],t["edit"]["hand"],t["edit"]["condition"]) for t in inp["technical"]}
    check(actual==expected and len(actual)==810,"45parents x2hands x9conditions exact technical Cartesian inventory")
    for t in inp["technical"]:
        c,p=parents[t["query"],t["parent"]["segment_id"]]
        check(t["parent"]==p and t["query_range"]==t["reference_range"]==p["core"] and t["role"]==c["role"],
              t["id"]+": technical original parent scope/role")
        verify_transform(t,load(inp["rows"][t["reference"]],t["reference_range"]))
    # Static data-flow check: model, probability, labels and guide are never supplied to new measurement API.
    driver=(U/"pipeline.py").read_text(encoding="utf-8")
    check("out=hand_evidence.compare(q,r,path)" in driver and "continuous.compare(" not in driver and ".align(" not in driver,
          "New channel API receives only saved numeric crops and supplied path")
    check("shape_sources':task.get('source_readouts',[])" in driver,"Old shape evidence only attached as immutable source links")
    totals["natural_parents"]=117;totals["technical_opportunities"]=810
    return inp

def verify_outputs(inp):
    require((U/"RESULT_MANIFEST.json").exists(),"Parent must complete public run first")
    fr=read(U/"FROZEN.json");fh=sha(U/"FROZEN.json");pin(U/"FROZEN.json")
    for p,h in {**fr["pins"],**fr["input_pins"]}.items():pin(p,h)
    manifest=read(U/"RESULT_MANIFEST.json");pin(U/"RESULT_MANIFEST.json")
    tasks={t["id"]:t for t in inp["natural"]+inp["technical"]}
    require(len(manifest["results"])==927 and {r["id"] for r in manifest["results"]}==set(tasks),"Full927 output inventory")
    check(manifest["new_training"]==manifest["new_model_inference"]==manifest["new_dtw"]==0,"No model/path execution")
    for i,item in enumerate(manifest["results"]):
        pin(item["path"],item["sha256"]);d=read(item["path"]);t=tasks[item["id"]];e=d["evidence"]
        check(d["id"]==t["id"] and d["kind"]==t["kind"] and item["kind"]==t["kind"],t["id"]+": result identity")
        for k in ("query","reference","query_range","reference_range"):check(d[k]==t[k],t["id"]+": output "+k)
        check(d["parent_id"]==t["parent"]["segment_id"] and d["frozen_sha256"]==fh,t["id"]+": parent/freeze")
        check(d["position_sources"]==t.get("position_sources",[]) and d["shape_sources"]==t.get("source_readouts",[]) and
              d["technical_edit"]==t.get("edit"),t["id"]+": unchanged position/shape/edit joins")
        check(d["educational_verdict"] is None and d["whole_sign_verdict"] is None and
              d["old_global_path_replaced"] is False and d["new_inference"] is False,t["id"]+": interpretation preservation")
        r=load(inp["rows"][t["reference"]],t["reference_range"])
        if t["kind"]=="natural":
            q=load(inp["rows"][t["query"]],t["query_range"])
            path=[[a-t["query_range"][0],b-t["reference_range"][0]] for a,b in t["baseline_full_path"]]
        else:
            with np.load(t["input"]["path"],allow_pickle=False) as z:q={k:z[k].copy() for k in ("x","mask","frame_id","timestamp_us")}
            path=[[j,j] for j in range(len(q["x"]))]
        check(e["path"]==path and e["path_full_scope"] is True,t["id"]+": exact supplied full path")
        check(e["query_frame_ids"]==q["frame_id"].tolist() and e["reference_frame_ids"]==r["frame_id"].tolist(),
              t["id"]+": exact native axes")
        for k in ("position_used_for_gating","learned_shape_used_for_gating","position_curve_recomputed",
                  "new_alignment_computed","zero_difference_is_correct_sign","missing_values_imputed",
                  "three_dimensional_rotation_validated","direct_wrist_flexion_validated"):
            check(e[k] is False,t["id"]+": no promotion "+k)
        for k in ("scalar_agreement","semantic_verdict","whole_sign_verdict","correction_advice"):
            check(e[k] is None,t["id"]+": unset verdict "+k)
        expected_names={f"{h}_{g}" for h in ("L","R") for g in ("finger_angles14","hand_axis2d","hand_forearm_relative2d")}
        check(set(e["groups"])==expected_names,t["id"]+": separate six groups")
        for name,g in e["groups"].items():
            off=0 if name[0]=="L" else 28
            cols=list(range(off+11,off+25)) if name.endswith("finger_angles14") else (
                [off+25,off+26] if name.endswith("hand_axis2d") else [off+25,off+26,off+2,off+3,off+4,off+5])
            dim=14 if name.endswith("finger_angles14") else 1
            check(g["source_columns"]==cols and g["units"]=="degrees",t["id"]+": source-column identity "+name)
            check(g["scalar_agreement"] is None and g["semantic_verdict"] is None,t["id"]+": independent channel limits")
            for side,a in (("query",q),("reference",r)):
                z=g[side];n=len(a["x"])
                check(z["original_source_mask"]==a["mask"][:,cols].tolist(),t["id"]+": exact rawmask "+side+name)
                check(z["frame_id"]==a["frame_id"].tolist() and z["timestamp_us"]==a["timestamp_us"].tolist(),
                      t["id"]+": original frame/PTS "+side+name)
                check(z["possible_shape"]==[n,dim] and z["counts"]["possible"]==n*dim and
                      z["counts"]["own_observed"]+z["counts"]["own_unobserved"]==n*dim and
                      z["counts"]["matched_support"]+z["counts"]["not_comparable"]==n*dim,
                      t["id"]+": all cells retained "+side+name)
                check(z["all_opportunities_retained"] is True and z["unaligned_frame_indices"]==[],
                      t["id"]+": no cropped-out opportunity "+side+name)
                expected_aligned=np.bincount([p[0 if side=="query" else 1] for p in path],minlength=n)
                check(z["aligned_link_count"]==expected_aligned.tolist(),t["id"]+": full multiplicity "+side+name)
                totals[t["kind"]+"_"+side+"_cells"]+=n*dim
        totals["verified_"+t["kind"]]+=1
        if (i+1)%200==0:print("verified source outputs",i+1,"/927",flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("phase",choices=("pre","post"));args=ap.parse_args()
    output=HERE/("PREPARATION_REVIEW.json" if args.phase=="pre" else "RESULT_VERIFICATION.json")
    if output.exists():raise SystemExit("Preserve first verifier output; refuse overwrite")
    exc=None
    try:
        inp=prepare_audit()
        if args.phase=="post":verify_outputs(inp)
        for p,h in pins.items():check(sha(p)==h,"Source unchanged across review "+p)
    except Exception:
        exc=traceback.format_exc();failures.append("Verifier exception")
    report=dict(schema="wis_independent_hand_source_verification_v1",phase=args.phase,
        status="PASS" if not failures else "FAIL",checks=checks,failure_count=len(failures),
        failures=failures[:100],exception=exc,at_utc=datetime.now(timezone.utc).isoformat(),
        totals=dict(totals),pins=pins,script_sha256=sha(__file__),
        scope="Source/hash/scope/left-right/technical edit/full-axis and immutable position/shape joining. Independent measurement and saved residual arithmetic belongs to semantic_model's separate verifier.",
        interpretation=["Technical input-space edits are not actual rotation/wrist-flexion or sign-error examples.",
            "Original native posterior is not rerun; synthetic controls carry no model posterior.",
            "RGB visibility is not saved landmark support; no source mask is filled from visible hands.",
            "No new path is estimated; original global correspondence is retained, not semantically certified.",
            "A zero position difference cannot suppress another channel or missing opportunity."],
        actions=dict(new_model_inference=0,new_training=0,alignment_search=0,position_recalculation=0,
                     source_extraction=0,producer_module_imports=0,source_writes=0,private_access=0))
    with output.open("x",encoding="utf-8") as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(status=report["status"],phase=args.phase,checks=checks,failure_count=len(failures),
                         output=str(output),sha256=sha(output))))
    if failures:raise SystemExit(1)

if __name__=="__main__":main()
