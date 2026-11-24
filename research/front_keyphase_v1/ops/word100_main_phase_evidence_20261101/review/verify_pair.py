"""Independent U/W saved-result verifier. Shared arithmetic helpers are copied from the prior independent verifier, not a producer.

No producer import, model, feature extraction, dynamic program or path search.
Run only after the root reports PAIR_MANIFEST.json and PAIR_SUMMARY.json complete.
The first verification record is exclusively created and never overwritten.
"""
from pathlib import Path
from collections import Counter
import ast
import datetime
import hashlib
import json
import math
import statistics
import numpy as np

U = Path(__file__).resolve().parents[1]
OUT = U / "review/VERIFICATION_PAIR.json"
NORMAL = {"identity", "speed_075", "speed_125", "hold15", "repeat2_same", "repeat3_same", "early_slow", "late_slow"}
KEYS = ("x", "mask", "frame_id", "timestamp_us")
checks = 0
failures = []
pins_checked = {}
cache = {}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def check(condition, label):
    global checks
    checks += 1
    if not bool(condition):
        failures.append({"check": label})

def same(actual, expected, label):
    if isinstance(expected, dict):
        check(isinstance(actual, dict), label + ":dict")
        if not isinstance(actual, dict):
            return
        check(set(actual) == set(expected), label + ":keys")
        for k, v in expected.items():
            if k in actual:
                same(actual[k], v, label + "/" + str(k))
    elif isinstance(expected, (list, tuple)):
        check(isinstance(actual, (list, tuple)), label + ":list")
        if not isinstance(actual, (list, tuple)):
            return
        check(len(actual) == len(expected), label + ":length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, label + "/" + str(i))
    elif isinstance(expected, float):
        check(isinstance(actual, (int, float)) and math.isclose(float(actual), expected, abs_tol=1e-10, rel_tol=1e-10), label)
    else:
        check(actual == expected, label)

def pincheck(path, expected):
    path = str(Path(path).resolve())
    if path not in pins_checked:
        pins_checked[path] = sha(path)
    check(pins_checked[path] == expected, "pin:" + path)

def load(path):
    path = str(path)
    if path not in cache:
        with np.load(path, allow_pickle=False) as z:
            cache[path] = {k: z[k].copy() for k in KEYS}
    return cache[path]

def digest(q, r):
    return {
        side: {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
               for k, v in a.items()}
        for side, a in (("query", q), ("reference", r))
    }

def path_audit(path, n, m, label):
    p = np.asarray(path)
    check(p.ndim == 2 and p.shape[1:] == (2,) and len(p) > 0, label + ":path_shape")
    check(np.issubdtype(p.dtype, np.integer), label + ":path_integer")
    p = p.astype(np.int64)
    check(p[0].tolist() == [0, 0] and p[-1].tolist() == [n-1, m-1], label + ":full_endpoints")
    check(all(0 <= i < n and 0 <= j < m for i, j in p), label + ":index_bounds")
    check(set(map(tuple, np.diff(p, axis=0))).issubset({(1,0),(0,1),(1,1)}), label + ":strict_monotone_moves")
    check(np.array_equal(np.unique(p[:,0]), np.arange(n)), label + ":every_query_frame")
    check(np.array_equal(np.unique(p[:,1]), np.arange(m)), label + ":every_reference_frame")
    return p

def independent_evidence(q, r, p):
    """Accumulate all link evidence using indexed reductions, not producer loops."""
    qi, ri = p[:,0], p[:,1]
    qx, rx = q["x"], r["x"]
    qm, rm = q["mask"], r["mask"]
    edge_known = qm[qi] & rm[ri] & np.isfinite(qx[qi]) & np.isfinite(rx[ri])
    edge_difference = np.where(edge_known, np.minimum(abs(qx[qi]-rx[ri]),1),0)
    supported, maxima = [], []
    for ids, source in ((qi,qx),(ri,rx)):
        support = np.zeros(source.shape, dtype=bool)
        peak = np.zeros_like(source)
        np.logical_or.at(support, ids, edge_known)
        np.maximum.at(peak, ids, edge_difference)
        supported.append(support)
        maxima.append(peak)
    axes = {}
    for k, (name, mask) in enumerate((("query",qm),("reference",rm))):
        support, peak = supported[k], maxima[k]
        axes[name] = {
            "total_slots": int(mask.size), "observed_slots": int(np.count_nonzero(mask)),
            "unobserved_slots": int(mask.size-np.count_nonzero(mask)),
            "compared_slots": int(np.count_nonzero(support)),
            "observed_without_comparison": int(np.count_nonzero(mask & ~support)),
            "not_compared_slots": int(support.size-np.count_nonzero(support)),
            "shape": {
                h: {"observed":int(np.count_nonzero(mask[:,cols])),
                    "compared":int(np.count_nonzero(support[:,cols])),
                    "peak_difference":float(np.max(peak[:,cols],initial=0)) if support[:,cols].any() else None}
                for h,cols in (("L",range(11,25)),("R",range(39,53)))
            }
        }
    relation_known = qm[qi][:,[1,29]].all(1) & rm[ri][:,[1,29]].all(1)
    opposite = ((qx[qi,1]-qx[qi,29])*(rx[ri,1]-rx[ri,29]) < 0) & relation_known
    axes.update(height_conflicts=p[opposite].tolist(),
                height_supported_path_cells=int(np.count_nonzero(relation_known)),
                height_relation_is_not_linguistic_truth=True, no_correctness_verdict=True)
    return axes, supported, maxima

def independent_truth(p, meta, kind, n, m):
    if "query_unrolled_index" in meta:
        qloc, rloc = meta["query_unrolled_index"], meta["reference_unrolled_index"]
        qocc, rocc = meta["query_occurrence"], meta["reference_occurrence"]
    elif "original_indices" in meta:
        qloc, rloc = meta["original_indices"], range(m)
        qocc, rocc = [0]*n, [0]*m
    elif n == m:
        qloc, rloc, qocc, rocc = range(n), range(m), [0]*n, [0]*m
    else:
        return None
    qerrors = [[] for _ in range(n)]
    rerrors = [[] for _ in range(m)]
    qwrong, rwrong, nwrong = set(), set(), 0
    qseam = sorted({j for i in range(1,n) if qocc[i]!=qocc[i-1] for j in (i-1,i)})
    rseam = sorted({j for i in range(1,m) if rocc[i]!=rocc[i-1] for j in (i-1,i)})
    seam_mismatch = 0
    for qi, ri in p:
        e = abs(qloc[qi]-rloc[ri])
        qerrors[qi].append(e)
        rerrors[ri].append(e)
        if qocc[qi] != rocc[ri]:
            qwrong.add(int(qi)); rwrong.add(int(ri)); nwrong += 1
            seam_mismatch += int(qi in qseam or ri in rseam)
    # Every original axis index contributes once after worst-link aggregation.
    qmax = [max(x) for x in qerrors]
    rmax = [max(x) for x in rerrors]
    qm, rm = sum(qmax)/n, sum(rmax)/m
    qinterior = [v for i,v in enumerate(qmax) if i not in qseam]
    rinterior = [v for i,v in enumerate(rmax) if i not in rseam]
    return {"query_mean_max_error_frames":float(qm),
            "reference_mean_max_error_frames":float(rm),
            "mean_two_axes":float((qm+rm)/2),
            "peak_error_frames":float(max(max(qmax),max(rmax))),
            "occurrence_mismatch_cells":nwrong,
            "query_wrong_occurrence_frames":len(qwrong),
            "reference_wrong_occurrence_frames":len(rwrong),
            "interior_mean_two_axes":float((sum(qinterior)/len(qinterior)+sum(rinterior)/len(rinterior))/2),
            "synthetic_seam":{
                "query_frames":qseam,"reference_frames":rseam,
                "query_mean_max_error":float(sum(qmax[i] for i in qseam)/len(qseam)) if qseam else None,
                "reference_mean_max_error":float(sum(rmax[i] for i in rseam)/len(rseam)) if rseam else None,
                "mismatched_path_cells_touching_seam":seam_mismatch
            },
            "source_truth_not_passed_to_candidate":True,
            "counterfactual_feature_edit":kind not in NORMAL}

def independent_shape(c, q, bsupport, csupport, bpeak, cpeak):
    changed = c["metadata"].get("changed_frames", [])
    if not changed:
        return None
    cols = list(range(11,25) if c["kind"]=="left_short_shape_change_5f" else range(39,53))
    selector = np.ix_(changed,cols)
    known = q["mask"][selector]
    before = np.max(bpeak[selector],axis=1)
    after = np.max(cpeak[selector],axis=1)
    return {"frames":changed,"before_max":before.tolist(),"after_max":after.tolist(),
            "nonzero_before":int(sum(v>1e-6 for v in before)),
            "nonzero_after":int(sum(v>1e-6 for v in after)),
            "actual_observed_changed_cells":c["metadata"].get("actual_observed_changed_cells",0),
            "known_edited_cells":int(known.sum()),
            "baseline_compared_edited_cells":int((known&bsupport[selector]).sum()),
            "candidate_compared_edited_cells":int((known&csupport[selector]).sum()),
            "lost_compared_edited_cells":int((known&bsupport[selector]&~csupport[selector]).sum())}


def partial(actual, expected, label):
    """Verify computed fields; descriptive producer prose is not a number."""
    if isinstance(expected,dict):
        check(isinstance(actual,dict),label+":object")
        if isinstance(actual,dict):
            for k,v in expected.items():
                check(k in actual,label+"/"+k+":present")
                if k in actual:partial(actual[k],v,label+"/"+k)
    elif isinstance(expected,list):
        check(isinstance(actual,list) and len(actual)==len(expected),label+":list_length")
        for i,(a,b) in enumerate(zip(actual,expected)):partial(a,b,label+"/"+str(i))
    else:same(actual,expected,label)


NAT = {}
TIME_KEYS = ["query_mean_max_error_frames","reference_mean_max_error_frames",
             "peak_error_frames","query_wrong_occurrence_frames",
             "reference_wrong_occurrence_frames","interior_mean_two_axes"]


def natural_sources():
    casepath=U.parent/"word100_action_feedback_casebook_20261101/INPUTS.json"
    physpath=U.parent/"word100_crossperson_stage_audit_20261101/PHYSICAL_CORRESPONDENCE.json"
    broadpath=U.parent/"word100_core_stage_evidence_20261031/PHASE_DATA.json"
    case,phys,broad=read(casepath),read(physpath),read(broadpath)
    NAT.update(parents=case["natural"],physical={c["alias"]:c for c in phys["clips"]},
               broad={c["alias"]:c for c in broad["clips"]},kinds=phys["kinds"],
               pins=[{"path":str(p.resolve()),"sha256":sha(p)} for p in (casepath,physpath,broadpath)])


def physical_at(alias,index):
    matches=[r for r in NAT["physical"][alias]["regions"] if r["outer"][0]<=index<=r["outer"][1]]
    check(len(matches)<=1,f"physical_draft_nonoverlap:{alias}:{index}")
    r=matches[0] if matches else None
    kind=r.get("kind") if r else None
    return {"region_id":r["id"] if r else None,
            "region_type":r["region_type"] if r else "outside_saved_reviewed_region",
            "kind":kind,"physical_family":NAT["kinds"].get(kind,{}).get("physical_kind",kind) if kind else None,
            "occurrence":r.get("occurrence") if r else None,
            "motion_subphase":r.get("motion_subphase") if r else None}


def broad_at(alias,index):
    members=[{"segment_id":s["segment_id"],"role":s["role"],"core":s["core"]}
             for s in NAT["broad"][alias]["segments"] if s["core"][0]<=index<=s["core"][1]]
    return {"memberships":members,"unknown_unannotated_gap":len(members)==0,"linguistic_role_confirmed":False}


def natural_counts(frames):
    unknown=sum(f["reference_physical_draft"]["kind"] is None for f in frames)
    flags={k:sum(bool(f["flags"][k]) for f in frames) for k in frames[0]["flags"]}
    patterns=Counter(
        f'same={int(f["flags"]["any_same_physical_family"])},different={int(f["flags"]["any_different_physical_family"])},unknown={int(f["flags"]["any_unknown_comparison"])}'
        for f in frames)
    certain=[f["flags"] for f in frames if f["reference_physical_draft"]["kind"] is not None]
    return {"reference_frames":len(frames),"reference_physical_certain":len(frames)-unknown,
            "reference_boundary_or_unknown":unknown,"flags_nonexclusive":flags,
            "family_flag_patterns_partition":dict(patterns),
            "reference_certain_any_different_family":sum(f["any_different_physical_family"] for f in certain),
            "reference_certain_any_query_unknown":sum(f["any_query_unknown"] for f in certain),
            "reference_certain_same_and_different_family":sum(f["any_same_physical_family"] and f["any_different_physical_family"] for f in certain),
            "reference_certain_same_and_unknown":sum(f["any_same_physical_family"] and f["any_unknown_comparison"] for f in certain),
            "path_link_count":sum(len(f["matched_query_indices"]) for f in frames),
            "reference_axis_weight":"one per original reference frame; linked query multiplicity does not change denominator"}


def natural_eval(qa,ra,p):
    parent_records=[]
    for parent in NAT["parents"]:
        if (parent["query"],parent["reference"])!=(qa,ra):continue
        frames=[]
        for ri in range(parent["reference_range"][0],parent["reference_range"][1]+1):
            qids=[int(qi) for qi,rj in p if rj==ri]
            r=physical_at(ra,ri)
            qs=[physical_at(qa,qi) for qi in qids]
            known=[q for q in qs if q["kind"] is not None]
            rknown=r["kind"] is not None
            qkind={q["kind"] for q in known}
            qfamily={q["physical_family"] for q in known}
            occ=[q for q in known if rknown and q["occurrence"] is not None and r["occurrence"] is not None]
            unknown=any(q["kind"] is None for q in qs)
            flags={
                "any_same_physical_family":bool(rknown and r["physical_family"] in qfamily),
                "any_different_physical_family":bool(rknown and qfamily-{r["physical_family"]}),
                "any_unknown_comparison":bool(not rknown or unknown),
                "any_query_unknown":unknown,"reference_kind_unknown":not rknown,
                "any_same_exact_kind":bool(rknown and r["kind"] in qkind),
                "any_different_exact_kind":bool(rknown and qkind-{r["kind"]}),
                "any_same_motion_subphase":bool(rknown and r["motion_subphase"] is not None and any(q["motion_subphase"]==r["motion_subphase"] for q in known)),
                "any_different_motion_subphase":bool(rknown and r["motion_subphase"] is not None and any(q["motion_subphase"] is not None and q["motion_subphase"]!=r["motion_subphase"] for q in known)),
                "any_same_recorded_occurrence":any(q["occurrence"]==r["occurrence"] for q in occ),
                "any_different_recorded_occurrence":any(q["occurrence"]!=r["occurrence"] for q in occ),
                "any_occurrence_unknown":bool(r["occurrence"] is None or any(q["occurrence"] is None for q in qs))}
            frames.append({"reference_index":ri,"reference_physical_draft":r,
                           "reference_broad_phase_draft":broad_at(ra,ri),"matched_query_indices":qids,
                           "matched_query_kind_set":sorted(qkind),"matched_query_physical_family_set":sorted(qfamily),
                           "matched_query_motion_subphase_set":sorted({q["motion_subphase"] for q in known if q["motion_subphase"] is not None}),
                           "matched_query_regions":[{"query_index":qi,"physical_draft":q,"broad_phase_draft":broad_at(qa,qi)} for qi,q in zip(qids,qs)],
                           "flags":flags,"family_conflict_flag":flags["any_different_physical_family"],
                           "unknown_is_retained_even_when_same_family_exists":True,
                           "unique_correct_query_frame":None,"educational_or_learner_error_verdict":None})
        parent_records.append({"id":parent["id"],"parent_id":parent["parent_id"],"reference_scope_inclusive":parent["reference_range"],
                               "fixed_parent_physical_description":parent["parent_physical_description"],"frames":frames,
                               "counts":natural_counts(frames),"same_kind_is_not_a_uniquely_correct_time":True})
    allframes=[f for parent in parent_records for f in parent["frames"]]
    return {"query":qa,"reference":ra,"parents":parent_records,"counts":natural_counts(allframes),
            "path_sha256":hashlib.sha256(json.dumps(p.tolist(),separators=(",",":")).encode("utf-8")).hexdigest(),
            "full_path_query_frames":NAT["physical"][qa]["native_frame_count"],
            "full_path_reference_frames":NAT["physical"][ra]["native_frame_count"],
            "full_path_links":len(p),"all_native_axes_retained":True,"source_pins":NAT["pins"],
            "evaluation_drafts_not_matcher_or_training_inputs":True,"educational_accuracy":None,"unique_true_time_accuracy":None,"new_alignment_or_inference":0}


def natural_aggregate(results):
    keys=[(r["query"],r["reference"]) for r in results]
    check(len(keys)==12 and len(set(keys))==12 and set(keys)=={(p["query"],p["reference"]) for p in NAT["parents"]},"natural12_exact_pair_inventory")
    parents=[p for r in results for p in r["parents"]]
    check(len(parents)==39 and len({p["id"] for p in parents})==39,"natural39_unique_parents")
    frames=[f for p in parents for f in p["frames"]]
    counts=natural_counts(frames)
    check((counts["reference_frames"],counts["reference_physical_certain"],counts["reference_boundary_or_unknown"])==(440,360,80),"natural_fixed440_certain360_unknown80")
    return {"unique_pairs":12,"parent_instances":39,"counts":counts,
            "pair_counts":[{"query":r["query"],"reference":r["reference"],"counts":r["counts"]} for r in results],
            "conflict_reference_keys":[{"query":r["query"],"reference":r["reference"],"parent_id":p["parent_id"],"reference_index":f["reference_index"]}
                                      for r in results for p in r["parents"] for f in p["frames"] if f["family_conflict_flag"]],
            "source_pins":NAT["pins"],"no_drop_of_unknown":True,"no_path_cell_weighting":True,
            "educational_accuracy":None,"unique_true_time_accuracy":None,"counts_are_evaluation_draft_diagnostics_only":True}


def sidecar_support(q,r):
    out={}
    for axis,a in (("query",q),("reference",r)):
        out[axis]={}
        for side,off in (("L",0),("R",28)):
            valid_count=zero_count=0
            for i in range(1,len(a["x"])):
                dt=(float(a["timestamp_us"][i])-float(a["timestamp_us"][i-1]))/1e6
                xy=a["x"][[i-1,i],off:off+2].astype(np.float64)
                if int(a["frame_id"][i])-int(a["frame_id"][i-1])!=1 or not (0<dt<=.15):continue
                if not a["mask"][[i-1,i],off:off+2].all() or not np.isfinite(xy).all():continue
                vx,vy=(xy[1]-xy[0])/dt
                if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(math.hypot(vx,vy))):continue
                valid_count+=1;zero_count+=int(vx==0 and vy==0)
            out[axis][side]={"total":len(a["x"]),"observed":valid_count,"zero_observed":zero_count}
    return out


def main():
    assert not OUT.exists(),"Preserve first result; fix only with a separately named verifier."
    assert (U/"PAIR_MANIFEST.json").exists() and (U/"PAIR_SUMMARY.json").exists(),"Root completion authorization required."
    inp,manifest,frozen,summary=[read(U/name) for name in ("PAIR_INPUTS.json","PAIR_MANIFEST.json","FROZEN_PAIR.json","PAIR_SUMMARY.json")]
    for p,h in {**frozen["pins"],**inp["pins"]}.items():pincheck(p,h)
    check(manifest["frozen_sha256"]==sha(U/"FROZEN_PAIR.json"),"frozen_pair_sha")
    natural_sources()
    controls={c["id"]:c for c in inp["controls"]}
    pairs={f'{p["query"]}_{p["reference"]}':p for p in inp["natural"]}
    prior=read(U.parent/"word100_phase_correspondence_20261101/INPUTS.json")
    old_controls={c["id"]:c for c in prior["controls"] if c["kind"]!="swap_adjacent_core_blocks"}
    check(len(inp["controls"])==len(controls)==192,"192_unique_prepared_control_ids")
    check(inp["natural"]==prior["natural"],"same12_natural_source_pairs")
    for alias,row in inp["inputs"].items():
        check(row["original"]==prior["inputs"][alias]["original"],alias+":same_original_input_no_recovery_panel")
    for cid,c in controls.items():
        q,r=load(c["input"]),load(c["reference"])
        for axis,a in (("query",q),("reference",r)):
            check(a["x"].shape==a["mask"].shape and a["x"].shape[1]==69 and a["mask"].dtype==np.bool_,cid+":"+axis+":original69_schema")
        if cid in old_controls:
            for k in ("id","alias","kind","input","reference","metadata"):
                same(c[k],old_controls[cid][k],cid+":unchanged_old_control/"+k)
        else:
            check(c["kind"] in {"early_slow","late_slow"},cid+":only32_prespecified_new_controls")
            n=len(r["x"]);mid=n//2
            ids=[i for i in range(n) for _ in range(2 if ((i<mid)==(c["kind"]=="early_slow")) else 1)]
            same(c["metadata"]["original_indices"],ids,cid+":independent_nonlinear_time_mapping")
            check(set(ids)==set(range(n)),cid+":all_source_samples_retained")
            for k in ("x","mask"):
                check(np.array_equal(q[k],r[k][ids],equal_nan=True),cid+":new_time_only_"+k)
            check(np.array_equal(q["frame_id"],np.arange(len(ids))),cid+":reset_frame_ids")
            check(np.array_equal(q["timestamp_us"],np.arange(len(ids))*33333),cid+":reset_pts")
    check(set(old_controls).issubset(controls) and len(set(controls)-set(old_controls))==32,"160_old_plus32_new_exact")
    check(len(manifest["controls"])==192 and {r["id"] for r in manifest["controls"]}==set(controls),"192_exact_control_results")
    check(len(manifest["natural"])==12 and {r["id"] for r in manifest["natural"]}==set(pairs),"12_exact_natural_results")
    check(len({r["file"] for r in manifest["controls"]+manifest["natural"]})==204,"204_unique_result_files")
    counts=Counter();cs=[];ns=[];shape_rows=[]
    for group in ("controls","natural"):
        for entry in manifest[group]:
            label=entry["id"];pincheck(entry["file"],entry["sha256"]);d=read(entry["file"])
            if group=="controls":
                c=controls[label];qfile,rfile=c["input"],c["reference"]
                for k,v in {"id":label,"kind":c["kind"],"normal_control":c["kind"] in NORMAL,"input":qfile,"reference_input":rfile}.items():same(d[k],v,label+"/"+k)
                check(c["normal_control"]==(c["kind"] in NORMAL),label+":normal_contract")
            else:
                c=pairs[label];qfile=inp["inputs"][c["query"]]["original"];rfile=inp["inputs"][c["reference"]]["original"]
                same(d["query"],c["query"],label+":qalias");same(d["reference"],c["reference"],label+":ralias")
            q,r=load(qfile),load(rfile);n,m=len(q["x"]),len(r["x"])
            same(d["input_digests_before_after"],digest(q,r),label+":input_digest")
            same(d["sidecar_support"],sidecar_support(q,r),label+":sidecar_support")
            temp={}
            for mode in ("unit","bounded"):
                v=d[mode];p=path_audit(v["path"],n,m,label+":"+mode)
                check(all(abs(i/max(n-1,1)-j/max(m-1,1))<=.25 for i,j in p),label+":"+mode+":fixed_band")
                check(v["objective_is_accuracy"] is False and v["physical_timing_verified"] is False,label+":"+mode+":no_truth_promotion")
                check(math.isfinite(v["objective_sum"]) and v["objective_sum"]>=0,label+":"+mode+":finite_objective")
                check(v["exact_optimal_path_count_capped"] in [1,2],label+":"+mode+":tie_count_range")
                e,s,pk=independent_evidence(q,r,p);same(v["evidence"],e,label+":"+mode+":evidence")
                temp[mode]=(p,e,s,pk)
                counts["paths"]+=1;counts[group+"_query_axis_slots"]+=n*69;counts[group+"_reference_axis_slots"]+=m*69
                counts[group+"_query_observed_slots"]+=int(q["mask"].sum());counts[group+"_reference_observed_slots"]+=int(r["mask"].sum())
                if group=="controls":
                    t=independent_truth(p,c["metadata"],c["kind"],n,m)
                    same(v["truth"],t,label+":"+mode+":truth")
                    d[mode]={**v,"evidence":e,"truth":t}
                else:
                    ev=natural_eval(c["query"],c["reference"],p)
                    partial(v["natural_evaluation"],ev,label+":"+mode+":natural_raw_draft")
                    d[mode]={**v,"evidence":e,"natural_evaluation":ev}
            if group=="controls" and c["kind"] in {"left_short_shape_change_5f","short_shape_change_5f"}:
                sv=independent_shape(c,q,temp["unit"][2][0],temp["bounded"][2][0],temp["unit"][3][0],temp["bounded"][3][0])
                sg={"frames":sv["frames"],"actual_observed_changed_cells":sv["actual_observed_changed_cells"],
                    "supported_cells_unit":sv["baseline_compared_edited_cells"],"supported_cells_bounded":sv["candidate_compared_edited_cells"],
                    "lost_supported_cells":sv["lost_compared_edited_cells"],
                    "query_frame_peaks_unit":sv["before_max"],"query_frame_peaks_bounded":sv["after_max"]}
                same(d["short_shape_guard"],sg,label+":short_guard")
                check(sv["known_edited_cells"]==sv["actual_observed_changed_cells"],label+":actual_known_edited_cells")
                d["short_shape_guard"]=sg;shape_rows.append({"id":label,**sv})
            (cs if group=="controls" else ns).append(d)
    check(counts["paths"]==408,"408_all_saved_paths_no_DP")

    improved=[];regressed=[];unchanged=[]
    for d in cs:
        if not d["normal_control"]:continue
        a,b=d["unit"]["truth"],d["bounded"]["truth"]
        bad=[k for k in TIME_KEYS if b[k]>a[k]+1e-9]
        good=[k for k in TIME_KEYS if b[k]<a[k]-1e-9]
        row={"id":d["id"],"kind":d["kind"],"worse":bad,"better":good,"unit":a,"bounded":b}
        if bad:regressed.append(row)
        elif good:improved.append(row)
        else:unchanged.append(d["id"])
    shorts=[d for d in cs if "short_shape_guard" in d]
    losses=[]
    for d in shorts:
        g=d["short_shape_guard"]
        lost=sum(b+1e-9<a for a,b in zip(g["query_frame_peaks_unit"],g["query_frame_peaks_bounded"]))
        if lost or g["lost_supported_cells"]:losses.append({"id":d["id"],"lost_peak_frames":lost,"lost_support":g["lost_supported_cells"]})
    denominator_fail=[]
    for d in cs+ns:
        for axis in ("query","reference"):
            a,b=d["unit"]["evidence"][axis],d["bounded"]["evidence"][axis]
            for k in ("total_slots","observed_slots","unobserved_slots"):
                if a[k]!=b[k]:denominator_fail.append({"id":d.get("id",d.get("query")),"axis":axis,"key":k})
            if b["compared_slots"]==0 and any(s["peak_difference"] is not None for s in b["shape"].values()):
                denominator_fail.append({"id":d.get("id"),"axis":axis,"key":"unknown called zero"})
    agg={mode:natural_aggregate([d[mode]["natural_evaluation"] for d in ns]) for mode in ("unit","bounded")}
    natural_changes=[];natural_bad=[];natural_good=[]
    for d in ns:
        for ap,bp in zip(d["unit"]["natural_evaluation"]["parents"],d["bounded"]["natural_evaluation"]["parents"]):
            check(ap["id"]==bp["id"],"natural_parent_order")
            for a,b in zip(ap["frames"],bp["frames"]):
                check(a["reference_index"]==b["reference_index"],"natural_reference_order")
                flags=["any_different_physical_family","any_unknown_comparison"]
                bad=[k for k in flags if b["flags"][k] and not a["flags"][k]]
                good=[k for k in flags if a["flags"][k] and not b["flags"][k]]
                if bad or good:
                    row={"parent":ap["id"],"reference_frame":a["reference_index"],"query":d["query"],"reference":d["reference"],
                         "unit_query":a["matched_query_indices"],"bounded_query":b["matched_query_indices"],"worse":bad,"better":good}
                    natural_changes.append(row)
                    (natural_bad if bad else natural_good).append(row)
    gates={
        "normal_speed_hold_repeat_no_regression":not regressed,
        "short_shape_support_and_peak_preserved":not losses,
        "full_observed_missing_denominators_preserved":not denominator_fail,
        "natural_no_new_family_or_unknown_flags":not natural_bad,
        "known_time_useful_improvement":bool(improved),
        "natural_draft_useful_improvement":bool(natural_good)}
    bykind={}
    for kind in sorted({d["kind"] for d in cs}):
        arr=[d for d in cs if d["kind"]==kind]
        bykind[kind]={"n":len(arr),"normal":arr[0]["normal_control"],
                      "changed_paths":sum(d["unit"]["path"]!=d["bounded"]["path"] for d in arr),
                      "mean_source_frame_error":{m:statistics.mean(d[m]["truth"]["mean_two_axes"] for d in arr) for m in ("unit","bounded")}}
    expected={
        "decision":"RESEARCH_CANDIDATE_PASSES_LIMITED_GATES" if all(gates.values()) else "NOT_ADOPTED",
        "controls":192,"normal_controls":sum(d["normal_control"] for d in cs),"natural_pairs":12,"gates":gates,
        "normal_improved":improved,"normal_regressed":regressed,"normal_unchanged":unchanged,"by_kind":bykind,
        "short_cases":len(shorts),"short_actually_changed_cases":sum(d["short_shape_guard"]["actual_observed_changed_cells"]>0 for d in shorts),
        "short_actual_changed_cells":sum(d["short_shape_guard"]["actual_observed_changed_cells"] for d in shorts),
        "short_losses":losses,"denominator_failures":denominator_fail,"natural_aggregate":agg,
        "natural_changed_paths":sum(d["unit"]["path"]!=d["bounded"]["path"] for d in ns),
        "natural_frame_changes":natural_changes,"natural_regressions":natural_bad,"natural_improvements":natural_good,
        "runtime_seconds":{m:sum(d[m]["seconds"] for d in cs+ns) for m in ("unit","bounded")},
        "runtime_not_full_video_or_end_user_latency":True,"no_new_training_or_detector_inference":True,
        "natural_kind_flags_are_partial_physical_drafts_not_linguistic_accuracy":True,"no_parameter_tuning_or_rerun_allowed":True}
    partial(summary,expected,"PAIR_SUMMARY")
    # Static routing checks do not call the matcher/evaluator.
    rt=ast.parse((U/"run_pair.py").read_text(encoding="utf-8"))
    costcalls=[n for n in ast.walk(rt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=="motion_cost.cost"]
    check(len(costcalls)==1 and [ast.unparse(a) for a in costcalls[0].args]==["q","r","mode"],"matcher_receives_no_metadata_or_GT")
    nevcalls=[n for n in ast.walk(rt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=="natural_evaluator.evaluate"]
    check(len(nevcalls)==1 and [ast.unparse(a) for a in nevcalls[0].args]==["a","b","d[mode]['path']"],"drafts_used_post_path_only")
    for p,h in {**frozen["pins"],**inp["pins"]}.items():check(sha(p)==h,"post_audit_unchanged:"+p)
    result={
        "schema":"main_phase_pair_independent_verification.v1","status":"PASS" if not failures else "FAIL",
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"checks":checks,"failures":failures,
        "paths_verified":408,"control_results":192,"natural_results":12,"axis_totals_summed_over_both_modes":dict(counts),
        "recomputed_decision":expected["decision"],"recomputed_gates":gates,
        "normal_improvement_count":len(improved),"normal_regression_count":len(regressed),"short_loss_cases":losses,
        "natural_counts":{m:a["counts"] for m,a in agg.items()},
        "natural_regression_count":len(natural_bad),"natural_improvement_count":len(natural_good),
        "short_shape_support":shape_rows,
        "source_pins":[{"path":str(p.resolve()),"sha256":sha(p)} for p in (U/"PAIR_INPUTS.json",U/"FROZEN_PAIR.json",U/"PAIR_MANIFEST.json",U/"PAIR_SUMMARY.json",U/"run_pair.py",U/"summarize_pair.py",U/"natural_evaluator.py",Path(__file__))],
        "independent_natural_sources":NAT["pins"],
        "actions":{"producer_or_matcher_or_evaluator_import":0,"DP_or_path_search":0,"new_inference":0,"new_RGB":0,"prior_result_edits":0},
        "limits":["No objective optimality rerun; objective scalars are checked for finite nonnegative values only.",
                  "Known source-time and occurrence construction metadata remain evaluator-only.",
                  "Reference440/certain360/unknown80 raw drafts are fixed diagnostics, not educational or unique-time accuracy.",
                  "Runtime mutation protection is supported by producer static guard plus independent pinned saved digest equality."]}
    with OUT.open("x",encoding="utf-8") as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({k:result[k] for k in ("status","checks","failures","recomputed_decision","recomputed_gates")},ensure_ascii=False))
    print(sha(OUT))


if __name__=="__main__":
    try:
        main()
    except Exception:
        import traceback
        error=U/"review/FIRST_PAIR_VERIFICATION_ERROR.txt"
        if not error.exists():error.write_text(traceback.format_exc(),encoding="utf-8")
        raise

